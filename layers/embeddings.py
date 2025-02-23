from jax import numpy as jnp
from equinox import Module
import jax
class Embeddings(Module):
    token_embeddings: jnp.ndarray
    position_embeddings: jnp.ndarray
    
    def __init__(self, vocab_size: int, d_model: int, max_len: int, key: jax.random.PRNGKey):
        key1, key2 = jax.random.split(key)
        self.token_embeddings = jax.random.normal(key1, (vocab_size, d_model))
        self.position_embeddings = jax.random.normal(key2, (max_len, d_model))
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Convert float inputs to integers for embedding lookup
        x = x.astype(jnp.int32)
        seq_len = x.shape[-1]
        return self.token_embeddings[x] + self.position_embeddings[jnp.arange(seq_len)]
