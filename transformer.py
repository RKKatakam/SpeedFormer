import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional
from mla import MLA, LayerNorm

class FeedForward(eqx.Module):
    linear1: jnp.ndarray
    linear2: jnp.ndarray
    dropout_rate: float = eqx.static_field()

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float, *, key):
        keys = jax.random.split(key, 2)
        self.linear1 = 0.01 * jax.random.normal(keys[0], (d_model, d_ff))
        self.linear2 = 0.01 * jax.random.normal(keys[1], (d_ff, d_model))
        self.dropout_rate = dropout_rate

    def __call__(self, x, *, key):
        x = jnp.matmul(x, self.linear1)
        x = jax.nn.gelu(x)
        x = jax.random.bernoulli(key, 1 - self.dropout_rate, x.shape) * x / (1 - self.dropout_rate)
        x = jnp.matmul(x, self.linear2)
        return x

class TransformerBlock(eqx.Module):
    attention: MLA
    ff: FeedForward
    ln1: LayerNorm
    ln2: LayerNorm
    dropout_rate: float = eqx.static_field()

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float, max_seq_len: int, *, key):
        keys = jax.random.split(key, 2)
        self.attention = MLA(d_model, n_heads, max_len=max_seq_len, key=keys[0])
        self.ff = FeedForward(d_model, d_ff, dropout_rate, key=keys[1])
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.dropout_rate = dropout_rate

    def __call__(self, x, kv_cache=None, past_length=0, *, key):
        keys = jax.random.split(key, 2)
        
        # First sub-block: MLA attention
        residual = x
        x = self.ln1(x)
        x, new_kv_cache = self.attention(x, kv_cache, past_length)
        x = jax.random.bernoulli(keys[0], 1 - self.dropout_rate, x.shape) * x / (1 - self.dropout_rate)
        x = x + residual

        # Second sub-block: Feed-forward
        residual = x
        x = self.ln2(x)
        x = self.ff(x, key=keys[1])
        x = x + residual

        return x, new_kv_cache

class Transformer(eqx.Module):
    token_embedding: jnp.ndarray
    position_embedding: jnp.ndarray
    layers: list[TransformerBlock]
    ln_f: LayerNorm
    d_model: int = eqx.static_field()
    vocab_size: int = eqx.static_field()
    max_seq_len: int = eqx.static_field()
    dropout_rate: float = eqx.static_field()

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
        dropout_rate: float,
        *,
        key
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        # Split keys for all components
        keys = jax.random.split(key, n_layers + 3)
        
        # Token and position embeddings
        self.token_embedding = 0.01 * jax.random.normal(keys[0], (vocab_size, d_model))
        self.position_embedding = 0.01 * jax.random.normal(keys[1], (max_seq_len, d_model))
        
        # Transformer layers
        self.layers = [
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                max_seq_len=max_seq_len,
                key=keys[i+2]
            )
            for i in range(n_layers)
        ]
        
        # Final layer norm
        self.ln_f = LayerNorm(d_model)

    def __call__(
        self,
        input_ids: jnp.ndarray,
        kv_caches: Optional[list] = None,
        past_length: int = 0,
        *,
        key
    ):
        B, S = input_ids.shape
        
        # Get embeddings
        x = self.token_embedding[input_ids]
        positions = jnp.arange(past_length, past_length + S)
        x = x + self.position_embedding[positions]

        # Initialize dropout keys
        keys = jax.random.split(key, len(self.layers) + 1)
        
        # Initialize KV caches if not provided
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        
        # Store new KV caches
        new_kv_caches = []

        # Apply dropout to embeddings
        x = jax.random.bernoulli(keys[0], 1 - self.dropout_rate, x.shape) * x / (1 - self.dropout_rate)

        # Process through transformer layers
        for layer, cache, layer_key in zip(self.layers, kv_caches, keys[1:]):
            x, new_cache = layer(x, cache, past_length, key=layer_key)
            new_kv_caches.append(new_cache)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = jnp.matmul(x, self.token_embedding.T)

        return logits, new_kv_caches
    
