import jax
from jax import numpy as jnp
from equinox import Module
from equinox.nn import RMSNorm
from layers.dense import DenseGeneral
from utils.rope import apply_rope_x
from utils.initializers import get_keys
import equinox as eqx


class MultiHeadLatentAttention(Module):
    q_down: DenseGeneral
    q_up: DenseGeneral
    kv_down: DenseGeneral
    kv_up: DenseGeneral
    o_proj: DenseGeneral
    q_layernorm: RMSNorm
    kv_layernorm: RMSNorm
    cos_cached: jax.Array
    sin_cached: jax.Array
    
    d_model: int = eqx.static_field()
    n_heads: int = eqx.static_field()
    dh: int = eqx.static_field()
    q_proj_dim: int = eqx.static_field()
    kv_proj_dim: int = eqx.static_field()
    qk_nope_dim: int = eqx.static_field()
    qk_rope_dim: int = eqx.static_field()
    max_seq_len: int = eqx.static_field()
    rope_theta: float = eqx.static_field()

    def __init__(self, d_model: int, n_heads: int, max_len: int = 1024, rope_theta: float = 10000.0, *, key: jax.random.PRNGKey):
        keys = get_keys(key, 6)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = (2 * d_model) // 3
        self.qk_nope_dim = self.dh // 2
        self.qk_rope_dim = self.dh // 2
        self.max_seq_len = max_len
        self.rope_theta = rope_theta

        # Q projections
        self.q_down = DenseGeneral(d_model, self.q_proj_dim, keys[0], use_bias=False)
        self.q_up = DenseGeneral(self.q_proj_dim, d_model, keys[1], use_bias=False)
        self.q_layernorm = RMSNorm(self.q_proj_dim)

        # KV projections
        self.kv_down = DenseGeneral(d_model, self.kv_proj_dim + self.qk_rope_dim, keys[2], use_bias=False)
        self.kv_up = DenseGeneral(self.kv_proj_dim, d_model + (n_heads * self.qk_nope_dim), keys[3], use_bias=False)
        self.kv_layernorm = RMSNorm(self.kv_proj_dim)

        # Output projection
        self.o_proj = DenseGeneral(d_model, d_model, keys[4], use_bias=False)

        # RoPE embeddings
        freqs = 1.0 / (rope_theta ** (jnp.arange(0, self.dh, 2) / self.dh))
        emb = jnp.outer(jnp.arange(max_len), freqs)
        self.cos_cached = jnp.broadcast_to(jnp.cos(emb)[None, None, :, :], (1, 1, max_len, self.dh//2))
        self.sin_cached = jnp.broadcast_to(jnp.sin(emb)[None, None, :, :], (1, 1, max_len, self.dh//2))

    def __call__(self, x: jax.Array, kv_cache: jax.Array = None, past_length: int = 0, inference: bool = False) -> tuple[jax.Array, jax.Array]:
        B, S, D = x.shape

        # Q Projections
        compressed_q = self.q_down(x)
        compressed_q = jax.vmap(jax.vmap(self.q_layernorm))(compressed_q)
        Q = self.q_up(compressed_q)
        Q = jnp.reshape(Q, (B, -1, self.n_heads, self.dh))
        Q = jnp.transpose(Q, (0, 2, 1, 3))
        Q, Q_for_rope = jnp.split(Q, [self.qk_nope_dim], axis=-1)

        # Q Decoupled RoPE
        cos_q = jnp.repeat(self.cos_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2], 2, axis=-1)
        sin_q = jnp.repeat(self.sin_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2], 2, axis=-1)
        Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)

        # KV Projections
        if kv_cache is None:
            compressed_kv = self.kv_down(x)
            KV_for_lora, K_for_rope = jnp.split(compressed_kv, [self.kv_proj_dim], axis=-1)
            KV_for_lora = jax.vmap(jax.vmap(self.kv_layernorm))(KV_for_lora)
        else:
            if inference:
                # During inference, we only process the new token and use cached KV
                new_kv = self.kv_down(x[:, -1:, :])  # Only process last token
                compressed_kv = jnp.concatenate([kv_cache, new_kv], axis=1)
                new_kv, new_K_for_rope = jnp.split(new_kv, [self.kv_proj_dim], axis=-1)
                old_kv, old_K_for_rope = jnp.split(kv_cache, [self.kv_proj_dim], axis=-1)
                new_kv = jax.vmap(jax.vmap(self.kv_layernorm))(new_kv)
                old_kv = jax.vmap(jax.vmap(self.kv_layernorm))(old_kv)
                KV_for_lora = jnp.concatenate([old_kv, new_kv], axis=1)
                K_for_rope = jnp.concatenate([old_K_for_rope, new_K_for_rope], axis=1)
            else:
                new_kv = self.kv_down(x)
                compressed_kv = jnp.concatenate([kv_cache, new_kv], axis=1)
                new_kv, new_K_for_rope = jnp.split(new_kv, [self.kv_proj_dim], axis=-1)
                old_kv, old_K_for_rope = jnp.split(kv_cache, [self.kv_proj_dim], axis=-1)
                new_kv = jax.vmap(jax.vmap(self.kv_layernorm))(new_kv)
                old_kv = jax.vmap(jax.vmap(self.kv_layernorm))(old_kv)
                KV_for_lora = jnp.concatenate([old_kv, new_kv], axis=1)
                K_for_rope = jnp.concatenate([old_K_for_rope, new_K_for_rope], axis=1)

        KV = self.kv_up(KV_for_lora)
        KV = jnp.reshape(KV, (B, -1, self.n_heads, self.dh + self.qk_nope_dim))
        KV = jnp.transpose(KV, (0, 2, 1, 3))
        K, V = jnp.split(KV, [self.qk_nope_dim], axis=-1)
        S_full = K.shape[2]

        # K RoPE
        K_for_rope = jnp.reshape(K_for_rope, (B, -1, 1, self.qk_rope_dim))
        K_for_rope = jnp.transpose(K_for_rope, (0, 2, 1, 3))
        cos_k = jnp.repeat(self.cos_cached[:, :, :S_full, :self.qk_rope_dim//2], 2, axis=-1)
        sin_k = jnp.repeat(self.sin_cached[:, :, :S_full, :self.qk_rope_dim//2], 2, axis=-1)
        K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)

        # Combine heads
        K_for_rope = jnp.repeat(K_for_rope, self.n_heads, axis=1)
        q_heads = jnp.concatenate([Q, Q_for_rope], axis=-1)
        k_heads = jnp.concatenate([K, K_for_rope], axis=-1)
        v_heads = V

        # Create attention mask
        mask = jnp.tril(jnp.ones((S, S_full)))
        mask = jnp.where(mask == 1, 0.0, -1e9)
        mask = mask[None, None, :, :]

        # Compute attention
        scale = 1.0 / jnp.sqrt(q_heads.shape[-1])
        scores = (q_heads @ jnp.transpose(k_heads, (0, 1, 3, 2))) * scale
        scores = scores + mask
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = attn_weights @ v_heads

        # Reshape and project output
        x = jnp.transpose(attn_output, (0, 2, 1, 3))
        x = jnp.reshape(x, (B, S, D))
        x = self.o_proj(x)

        return x, compressed_kv 