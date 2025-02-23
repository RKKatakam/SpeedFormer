import jax
import equinox as eqx
from attention.latent_attention import MultiHeadLatentAttention
from model.transformer import Transformer
import jax.numpy as jnp

# Model parameters
batch_size = 1
seq_len = 1
d_model = 128
vocab_size = 10000
n_heads = 8
n_layers = 4
max_len = 128
dropout = 0.1
rope_theta = 10000.0

# Initialize model
key = jax.random.PRNGKey(0)
model = Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    max_len=max_len,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout=dropout,
    rope_theta=rope_theta,
    key=key
)
model = eqx.filter_jit(model)

kv_cache = None
past_length = 0
output = []
x = jax.random.randint(key, (1, 1), 0, vocab_size)
output.append(x[0, 0])  # Store the scalar token value

for i in range(10):
    logits, kv_cache = model(x, kv_cache=kv_cache, past_length=past_length, inference=True)
    next_token = jnp.argmax(logits, axis=-1)
    output.append(int(next_token[0]))  # Store the scalar token value
    past_length += 1
    x = next_token.reshape(1, 1)  # Reshape to (batch_size, seq_len)

print(output)