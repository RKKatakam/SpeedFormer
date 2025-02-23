import jax
from jax import numpy as jnp
import equinox as eqx
from equinox import Module
from equinox.nn import RMSNorm
from layers.dense import DenseGeneral
from layers.embeddings import Embeddings
from layers.mlp import MLP
from attention.latent_attention import MultiHeadLatentAttention
from utils.initializers import get_keys

class TransformerBlock(Module):
    attention: MultiHeadLatentAttention
    mlp: MLP
    ln1: RMSNorm
    ln2: RMSNorm
    dropout_key: jax.random.PRNGKey
    
    def __init__(self, d_model: int, n_heads: int, max_len: int = 1024, rope_theta: float = 10000.0, dropout: float = 0.1, *, key: jax.random.PRNGKey):
        keys = get_keys(key, 3)
        self.attention = MultiHeadLatentAttention(d_model, n_heads, max_len, rope_theta, key=keys[0])
        self.mlp = MLP(d_model, dropout, key=keys[1])
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.dropout_key = keys[2]
        
    def __call__(self, x: jax.Array, kv_cache: jax.Array = None, past_length: int = 0, inference: bool = False) -> jax.Array:
        residual = x
        x = jax.vmap(jax.vmap(self.ln1))(x)
        x, kv_cache = self.attention(x, kv_cache, past_length, inference)
        x = x + residual
        x = jax.vmap(jax.vmap(self.ln2))(x)
        x = self.mlp(x, key=self.dropout_key)
        if inference:
            return x, kv_cache
        else:
            return x
        
        
class Transformer(Module):
    embeddings: Embeddings
    blocks: list[TransformerBlock]
    ln_f: RMSNorm
    final_proj: DenseGeneral
    
    def __init__(self, vocab_size: int, d_model: int, max_len: int, n_heads: int, n_layers: int, dropout: float, rope_theta: float, *, key: jax.random.PRNGKey):
        self.embeddings = Embeddings(vocab_size, d_model, max_len, key=key)
        self.blocks = [TransformerBlock(d_model, n_heads, max_len, rope_theta, dropout, key=jax.random.PRNGKey(i)) for i in range(n_layers)]
        self.ln_f = RMSNorm(d_model)
        self.final_proj = DenseGeneral(d_model, vocab_size, key=key)
        
    def __call__(self, x: jax.Array, kv_cache: list[jax.Array] = None, past_length: int = 0, inference: bool = False) -> tuple[jax.Array, list[jax.Array]]:
        # x: (batch_size, seq_len)
        x = self.embeddings(x) # (batch_size, seq_len, d_model)
        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            layer_kv = None if kv_cache is None else kv_cache[i]
            if inference:
                x, new_layer_kv = block(x, layer_kv, past_length, inference)
                new_kv_cache.append(new_layer_kv)
            else:
                x = block(x, layer_kv, past_length, inference)
                
        x = jax.vmap(jax.vmap(self.ln_f))(x) # (batch_size, seq_len, d_model)
        logits = self.final_proj(x)  # (batch_size, seq_len, vocab_size)
        logits = jax.nn.softmax(logits, axis=-1) # (batch_size, seq_len, vocab_size)
        
        if inference:
            # For inference, we only want the last token's probabilities
            logits = logits[:, -1, :] # (batch_size, vocab_size)
            return logits, new_kv_cache
        else:
            # For training, flatten the logits
            logits = jnp.reshape(logits, (logits.shape[0], -1))
            return logits