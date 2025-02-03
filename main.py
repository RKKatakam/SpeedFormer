import jax
import jax.numpy as jnp
import optax
import numpy as np
from test import Transformer, TransformerConfig, create_transformer

# -------------------------------------------------------------------------------
# Load raw data from data.txt (byte-level)
# -------------------------------------------------------------------------------
with open("data.txt", "rb") as f:
    raw_data = f.read()
# Convert raw bytes to a numpy array of uint8 tokens (0-255).
tokens = np.frombuffer(raw_data, dtype=np.uint8)

# -------------------------------------------------------------------------------
# Hyperparameters / Configurations
# -------------------------------------------------------------------------------
batch_size = 32
seq_len = 128  # Sequence length to use (adjust as needed)
num_steps = 1000  # Number of training steps

# Create a TransformerConfig instance
config = TransformerConfig(
    vocab_size=256,  # Byte-level vocab size
    embedding_dim=128,
    model_dim=128,
    num_heads=4,
    ff_dim=512,
    dropout_rate=0.1,
    num_layers=2,
    key=jax.random.PRNGKey(0)
)

# Instantiate the Transformer model.
model = create_transformer(config)

# Set up the optimizer.
lr = 1e-3
optimizer = optax.adam(lr)
opt_state = optimizer.init(model)

# -------------------------------------------------------------------------------
# Utility functions: batching and loss computation
# -------------------------------------------------------------------------------
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

# -------------------------------------------------------------------------------
# Training step (jit compiled)
# -------------------------------------------------------------------------------
@jax.jit
def train_step(model, opt_state, x, y):
    def loss_fn(m):
        logits = m(x)
        return cross_entropy_loss(logits, y)
    loss, grads = jax.value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = optax.apply_updates(model, updates)
    return model, opt_state, loss

# -------------------------------------------------------------------------------
# Training Loop
# -------------------------------------------------------------------------------
for step in range(num_steps):
    x_batch, y_batch = get_batch(tokens, batch_size, seq_len)
    model, opt_state, loss = train_step(model, opt_state, x_batch, y_batch)
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss:.5f}")