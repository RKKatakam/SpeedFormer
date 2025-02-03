import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass
import optax
from functools import partial
import nn  # your custom neural network module
import jax.tree_util as jtu

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


def to_dtype(tree, dtype):
  """Converts all leaves of a pytree to the specified dtype.

  Args:
    tree: The pytree to convert.  This can be a nested structure of
      lists, tuples, dicts, and JAX arrays.
    dtype: The desired dtype (e.g., jnp.float32, jnp.float64, jnp.bfloat16).

  Returns:
    A new pytree with the same structure as the input, but with all
    leaves converted to the specified dtype.
  """
  return jtu.tree_map(lambda leaf: leaf.astype(dtype) if isinstance(leaf, jnp.ndarray) else leaf, tree)

def xavier_uniform(key, shape, dtype=jnp.float32):
    fan_in, fan_out = shape[0], shape[-1]
    scale = jnp.sqrt(6. / (fan_in + fan_out))
    return jax.random.uniform(key, shape, dtype, -scale, scale)

# --------------------------
# Embeddings module
# --------------------------
class Embeddings(nn.Module):
    vocab_size: nn.Static[int]
    embedding_dim: nn.Static[int]
    embeddings: jnp.ndarray

    def __init__(self, vocab_size: int, embedding_dim: int, key: jax.random.PRNGKey):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Initialize an embedding matrix of shape (vocab_size, embedding_dim)
        self.embeddings = xavier_uniform(key, (vocab_size, embedding_dim))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Expects x of shape (batch, seq_len) containing integer token indices.
        Returns embedded tokens of shape (batch, seq_len, embedding_dim).
        """
        return self.embeddings[x]


# --------------------------
# Single-head attention (one head only)
# --------------------------
class SingleHeadAttention(nn.Module):
    embed_dim: nn.Static[int]    # input (model) dimension
    head_dim: nn.Static[int]    # per-head dimension (typically = embed_dim // num_heads)
    scale: nn.Static[float]
    query: nn.Linear
    key_layer: nn.Linear  # renamed to avoid clashing with the module argument "key"
    value: nn.Linear

    def __init__(self, embed_dim: int, head_dim: int, key: jax.random.PRNGKey):
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        # For a single head, project from embed_dim to head_dim.
        self.query = nn.Linear(embed_dim, head_dim, key)
        self.key_layer = nn.Linear(embed_dim, head_dim, key)
        self.value = nn.Linear(embed_dim, head_dim, key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, seq_len, embed_dim)
        Returns: (batch, seq_len, head_dim)
        """
        batch, seq_len, _ = x.shape
        q = self.query(x)       # shape: (batch, seq_len, head_dim)
        k = self.key_layer(x)   # shape: (batch, seq_len, head_dim)
        v = self.value(x)       # shape: (batch, seq_len, head_dim)

        # Compute scaled dot-product attention scores: (batch, seq_len, seq_len)
        attn_scores = jnp.einsum('bqd,bkd->bqk', q, k) * self.scale

        # Apply a causal mask so that each token attends only to itself and earlier tokens.
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        attn_scores = jnp.where(mask == 0, -1e9, attn_scores)

        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        # Compute weighted sum of values, resulting in shape (batch, seq_len, head_dim)
        out = jnp.einsum('bqk,bkd->bqd', attn_weights, v)
        return out


# --------------------------
# Multi-head attention (combines several heads)
# --------------------------
class MultiHeadAttention(nn.Module):
    embed_dim: nn.Static[int]    # overall model dimension (e.g. 128)
    num_heads: nn.Static[int]
    head_dim: nn.Static[int]      # per-head dimension (embed_dim // num_heads)
    attention_heads: list  # list of SingleHeadAttention modules
    output: nn.Linear      # final linear projection

    def __init__(self, embed_dim: int, num_heads: int, key: jax.random.PRNGKey):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Split the key into num_heads parts (each head gets its own key)
        keys = jax.random.split(key, num_heads)
        self.attention_heads = [
            SingleHeadAttention(embed_dim, self.head_dim, k) for k in keys
        ]
        # Final linear projection: maps concatenated heads back to embed_dim.
        self.output = nn.Linear(embed_dim, embed_dim, key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (batch, seq_len, embed_dim)
        Returns: (batch, seq_len, embed_dim)
        """
        # Compute each head’s attention output; each has shape (batch, seq_len, head_dim)
        head_outputs = [head(x) for head in self.attention_heads]
        # Concatenate outputs along the last axis: shape (batch, seq_len, num_heads * head_dim)
        concatenated = jnp.concatenate(head_outputs, axis=-1)
        # Apply final linear projection.
        return self.output(concatenated)


# --------------------------
# Transformer (using attention layers)
# --------------------------
class Transformer(nn.Module):
    vocab_size: nn.Static[int]
    embedding_dim: nn.Static[int]
    model_dim: nn.Static[int]
    num_heads: nn.Static[int]
    num_layers: nn.Static[int]
    embeddings: Embeddings
    decoder_layers: list  # list of MultiHeadAttention modules
    output: nn.Linear

    def __init__(self, vocab_size: int, embedding_dim: int, model_dim: int,
                 num_heads: int, num_layers: int,
                 key: jax.random.PRNGKey):
        # We need one key for embeddings, one for each decoder layer, and one for the output.
        keys = jax.random.split(key, num_layers + 2)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        # Initialize the embeddings using the first key.
        self.embeddings = Embeddings(vocab_size, embedding_dim, keys[0])
        # Create decoder layers (each a multi-head attention module).
        # (Here we assume model_dim == embedding_dim for simplicity.)
        self.decoder_layers = [
            MultiHeadAttention(model_dim, num_heads, k) for k in keys[1:-1]
        ]
        # Final output projection from model_dim to vocab_size.
        self.output = nn.Linear(model_dim, vocab_size, keys[-1])


    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Expects x of shape (batch, seq_len) containing token indices.
        Returns logits of shape (batch, seq_len, vocab_size).
        """
        x = self.embeddings(x)  # (batch, seq_len, embedding_dim)
        for layer in self.decoder_layers:
            x = layer(x)
        return self.output(x)


# --------------------------
# Data Loading and Batching
# --------------------------
def get_batch(tokens: np.ndarray, batch_size: int, seq_len: int):
    # Randomly sample starting indices for each batch.
    indices = np.random.randint(0, len(tokens) - seq_len - 1, size=batch_size)
    x_batch = np.stack([tokens[i:i+seq_len] for i in indices])
    # Target is x shifted by one token.
    y_batch = np.stack([tokens[i+1:i+seq_len+1] for i in indices])
    return jnp.array(x_batch), jnp.array(y_batch)

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    # Reshape logits to (batch * seq_len, vocab_size)
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)
    labels = labels.reshape(-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()

with open("data.txt", "rb") as f:
    raw_data = f.read()
# Convert raw bytes to a numpy array of uint8 tokens (0-255).
data = np.frombuffer(raw_data, dtype=np.uint8)

# --------------------------
# Loss Function
# --------------------------
def compute_loss(model, x, y):
    """
    Compute cross entropy loss.
      - x: input tokens, shape (batch, seq_len)
      - y: target tokens, shape (batch, seq_len)
    """
    # Forward pass: logits has shape (batch, seq_len, vocab_size)
    logits = model(x)
    # Flatten the logits and targets so that we have (batch * seq_len, vocab_size) and (batch * seq_len,)
    logits = logits.reshape(-1, logits.shape[-1])
    targets = y.reshape(-1)
    # Compute per-example cross-entropy loss using optax's helper.
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return loss.mean()


# --------------------------
# Training Step (one update)
# --------------------------
@jax.jit
def train_step(model, opt_state, x, y):
    """
    Performs a single training update:
      1. Computes loss and gradients.
      2. Computes the parameter updates from the optimizer.
      3. Returns the updated model and optimizer state.
    """
    # Compute loss and gradients with respect to the model (assuming model is a pytree of parameters).
    loss, grads = jax.value_and_grad(compute_loss)(model, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    # Apply updates to model parameters.
    new_model = optax.apply_updates(model, updates)
    return new_model, opt_state, loss


# --------------------------
# Training Hyperparameters
# --------------------------
learning_rate = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=1000)
batch_size = 32
seq_len = 64  # Length of each training sequence
num_epochs = 100
steps_per_epoch = 100  # Adjust as needed (or compute based on data size)

# --------------------------
# Initialize Model and Optimizer
# --------------------------
# Create a transformer instance (byte-level vocabulary size = 256)
model = Transformer(
    vocab_size=256,
    embedding_dim=1024,
    model_dim=1024,   # (Assuming model_dim == embedding_dim)
    num_heads=16,
    num_layers=6,
    key=jax.random.PRNGKey(0)
)


# Initialize the optimizer.
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate)
)

opt_state = optimizer.init(model)

# --------------------------
# Training Loop
# --------------------------

for epoch in range(num_epochs):
    total_loss = 0
    for step in range(steps_per_epoch):
        x, y = get_batch(data, batch_size, seq_len)
        model, opt_state, loss = train_step(model, opt_state, x, y)
        total_loss += loss

    total_loss_numeric = float(total_loss)
    steps_numeric = float(steps_per_epoch)
    print(f"Epoch {epoch}, avg loss: {total_loss_numeric / steps_numeric:.3f}")
    

