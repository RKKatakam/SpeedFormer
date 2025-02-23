import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import tiktoken
from transformer import Transformer
import json
# save the compilations cache
jax.config.update("jax_compilation_cache_dir", "jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

# Initialize tokenizer and load/clean data
enc = tiktoken.get_encoding("gpt2")

def clean_text(text):
    return ' '.join(text.split())

# Load and prepare data
with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    cleaned_text = clean_text(text)

# Tokenize
tokens = enc.encode(cleaned_text)
data = jnp.array(tokens)

# Training parameters
BATCH_SIZE = 32
SEQUENCE_LENGTH = 128
LEARNING_RATE = 3e-4
VOCAB_SIZE = enc.n_vocab  # Use tokenizer's vocabulary size
MAX_EPOCHS = 1

# Model parameters
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6
D_FF = 2048
MAX_SEQ_LEN = 1024
DROPOUT_RATE = 0.1

# Initialize model
model = Transformer(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    max_seq_len=MAX_SEQ_LEN,
    dropout_rate=DROPOUT_RATE,
    key=jax.random.PRNGKey(0)
)



def get_batch(data, batch_size, block_size, key):
    """Generate a small batch of data of inputs x and targets y"""
    ix = jax.random.randint(key, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@eqx.filter_value_and_grad
def compute_loss(model, x, y, key):
    """Compute cross entropy loss"""
    logits, _ = model(x, key=key)
    # Reshape logits and targets for cross entropy
    B, T, C = logits.shape
    logits = logits.reshape(-1, C)
    targets = y.reshape(-1)
    
    # Cross entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.mean(loss)

@eqx.filter_jit
def train_step(model, opt_state, x, y, key, optimizer):
    """Single training step"""
    loss, grads = compute_loss(model, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def save(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


# Training loop
def train():
    model_state = model
    key = jax.random.PRNGKey(0)
    # Create optimizer
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Calculate steps per epoch
    n_samples = len(data) - SEQUENCE_LENGTH
    steps_per_epoch = n_samples // BATCH_SIZE
    
    
    for epoch in range(MAX_EPOCHS):
        total_loss = 0.0
        key, epoch_key = jax.random.split(key)
        
        # Training steps within each epoch
        for step in range(steps_per_epoch):
            step_key = jax.random.fold_in(epoch_key, step)
            batch_key, train_key = jax.random.split(step_key)
            
            # Get batch
            x, y = get_batch(data, BATCH_SIZE, SEQUENCE_LENGTH, batch_key)
            
            # Training step
            model_state, opt_state, loss = train_step(
                model_state, opt_state, x, y, train_key, optimizer
            )
            total_loss += loss
            
            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{MAX_EPOCHS}, Step {step}/{steps_per_epoch}, Loss: {loss:.4f}")
                # save the model
                save("model.eqx", {"epoch": epoch, "step": step}, model_state)
        
        # Print epoch summary
        avg_loss = total_loss / steps_per_epoch
        print(f"Epoch {epoch+1}/{MAX_EPOCHS} completed. Average Loss: {avg_loss:.4f}")
    
    return model_state

if __name__ == "__main__":
    trained_model = train()
