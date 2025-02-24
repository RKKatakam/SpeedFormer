import jax
import jax.numpy as jnp
import equinox as eqx
from einops import rearrange, repeat

def apply_rope_x(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

class LayerNorm(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    eps: float = eqx.static_field()
    normalized_shape: tuple = eqx.static_field()
    axis: int = eqx.static_field()

    def __init__(self, normalized_shape, eps=1e-5, axis=-1):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape) if isinstance(normalized_shape, tuple) else (normalized_shape,)
        self.weight = jnp.ones(self.normalized_shape)
        self.bias = jnp.zeros(self.normalized_shape)
        self.eps = eps
        self.axis = axis
        
    
    def __call__(self, x):
        mean = jnp.mean(x, axis=self.axis, keepdims=True)
        var = jnp.var(x, axis=self.axis, keepdims=True)
        return (x - mean) / jnp.sqrt(var + self.eps)



class MLA(eqx.Module):
    # Model dimensions and parameters
    d_model: int = eqx.static_field()
    n_heads: int = eqx.static_field()
    dh: int = eqx.static_field()
    q_proj_dim: int 
    kv_proj_dim: int
    qk_nope_dim: int 
    qk_rope_dim: int

    # Parameters for Q projections
    W_dq: jnp.ndarray
    W_uq: jnp.ndarray
    q_layernorm: LayerNorm

    # Parameters for KV projections
    W_dkv: jnp.ndarray 
    W_ukv: jnp.ndarray 
    kv_layernorm: LayerNorm

    # Output projection
    W_o: jnp.ndarray

    # RoPE buffers (non-trainable)
    max_seq_len: int = eqx.static_field()
    rope_theta: float = eqx.static_field()
    cos_cached: jax.Array
    sin_cached: jax.Array 

    def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0, *, key):
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = (2 * d_model) // 3

        self.qk_nope_dim = self.dh // 2
        self.qk_rope_dim = self.dh // 2

        # Split key for reproducibility
        keys = jax.random.split(key, 6)
        self.W_dq = 0.01 * jax.random.normal(keys[0], (d_model, self.q_proj_dim), dtype=jnp.float32)
        self.W_uq = 0.01 * jax.random.normal(keys[1], (self.q_proj_dim, d_model), dtype=jnp.float32)
        self.q_layernorm = LayerNorm(self.q_proj_dim, eps=1e-5)

        self.W_dkv = 0.01 * jax.random.normal(keys[2], (d_model, self.kv_proj_dim + self.qk_rope_dim), dtype=jnp.float32)
        self.W_ukv = 0.01 * jax.random.normal(keys[3], (self.kv_proj_dim, d_model + (n_heads * self.qk_nope_dim)), dtype=jnp.float32)
        self.kv_layernorm = LayerNorm(self.kv_proj_dim, eps=1e-5)

        self.W_o = 0.01 * jax.random.normal(keys[4], (d_model, d_model), dtype=jnp.float32)

        self.max_seq_len = max_len
        self.rope_theta = rope_theta

       
        freq_seq = jnp.arange(0, self.dh, 2, dtype=jnp.float32)
        freqs = 1.0 / (rope_theta ** (freq_seq / self.dh))
        positions = jnp.arange(max_len, dtype=jnp.float32)
        emb = jnp.outer(positions, freqs) # Precompute RoPE cos and sin caches with explicit float32 dtype
       
        self.cos_cached = jnp.cos(emb)[None, None, :, :].astype(jnp.float32)
        self.sin_cached = jnp.sin(emb)[None, None, :, :].astype(jnp.float32)
        
        

    def __call__(self, x, kv_cache=None, past_length=0):
        # x shape: (B, S, D) with D == d_model
        B, S, D = x.shape

        # Q projections
        compressed_q = jnp.matmul(x, self.W_dq)  # (B, S, q_proj_dim)
        compressed_q = jax.vmap(jax.vmap(self.q_layernorm))(compressed_q)
        Q = jnp.matmul(compressed_q, self.W_uq)  # (B, S, d_model)
        # Reshape and transpose to (B, n_heads, S, dh)
        Q = Q.reshape(B, S, self.n_heads, self.dh).transpose(0, 2, 1, 3)
        # Split Q into two parts along the last dimension
        Q, Q_for_rope = jnp.split(Q, [self.qk_nope_dim], axis=-1)

        # Q Decoupled RoPE
        cos_q = jax.lax.stop_gradient(self.cos_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(2, axis=-1))
        sin_q = jax.lax.stop_gradient(self.sin_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(2, axis=-1))
        Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)

        # KV Projections
        if kv_cache is None:
            compressed_kv = jnp.matmul(x, self.W_dkv)  # (B, S, kv_proj_dim + qk_rope_dim)
            KV_for_lora, K_for_rope = jnp.split(compressed_kv, [self.kv_proj_dim], axis=-1)
            KV_for_lora = jax.vmap(jax.vmap(self.kv_layernorm))(KV_for_lora)
        else:
            new_kv = jnp.matmul(x, self.W_dkv)
            compressed_kv = jnp.concatenate([kv_cache, new_kv], axis=1)
            new_kv_val, new_K_for_rope = jnp.split(new_kv, [self.kv_proj_dim], axis=-1)
            old_kv, old_K_for_rope = jnp.split(kv_cache, [self.kv_proj_dim], axis=-1)
            new_kv_val = jax.vmap(jax.vmap(self.kv_layernorm))(new_kv_val)
            old_kv = jax.vmap(jax.vmap(self.kv_layernorm))(old_kv)
            KV_for_lora = jnp.concatenate([old_kv, new_kv_val], axis=1)
            K_for_rope = jnp.concatenate([old_K_for_rope, new_K_for_rope], axis=1)

        # Compute combined KV projection
        KV = jnp.matmul(KV_for_lora, self.W_ukv)
        # Reshape to (B, S_full, n_heads, (dh + qk_nope_dim)) then transpose to (B, n_heads, S_full, ...)
        KV = KV.reshape(B, -1, self.n_heads, self.dh + self.qk_nope_dim).transpose(0, 2, 1, 3)
        K, V = jnp.split(KV, [self.qk_nope_dim], axis=-1)
        S_full = K.shape[2]

        # K RoPE
        K_for_rope = K_for_rope.reshape(B, -1, 1, self.qk_rope_dim).transpose(0, 2, 1, 3)
        cos_k = jax.lax.stop_gradient(self.cos_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(2, axis=-1))
        sin_k = jax.lax.stop_gradient(self.sin_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(2, axis=-1))
        K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)
        # Repeat for each head
        K_for_rope = jnp.repeat(K_for_rope, self.n_heads, axis=1)

        # Concatenate the no-rotary and rotary parts for queries and keys
        q_heads = jnp.concatenate([Q, Q_for_rope], axis=-1)
        k_heads = jnp.concatenate([K, K_for_rope], axis=-1)
        v_heads = V  # Already in shape (B, n_heads, S_full, dh)

        # Create attention mask: (S, S_full) with lower triangular structure offset by past_length
        mask = jnp.tril(jnp.ones((S, S_full), dtype=bool), k=past_length)
        mask = mask[None, None, :, :]  # Expand to (B, n_heads, S, S_full)

        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(q_heads.shape[-1])
        attn_logits = jnp.matmul(q_heads, jnp.swapaxes(k_heads, -1, -2)) * scale
        # Mask out disallowed positions by setting to a very negative number
        attn_logits = jnp.where(mask, attn_logits, -1e9)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_output = jnp.matmul(attn_weights, v_heads)
        # Reshape back to (B, S, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, S, D)

        # Apply final output projection (note the transpose on W_o to match dimensions)
        out = jnp.matmul(attn_output, self.W_o.T)

        return out, compressed_kv



class CrossMLA(eqx.Module):
    # Model dimensions and parameters
    d_model: int = eqx.static_field()
    n_heads: int = eqx.static_field()
    dh: int = eqx.static_field()
    q_proj_dim: int 
    kv_proj_dim: int
    qk_nope_dim: int 
    qk_rope_dim: int

    # Parameters for Q projections
    W_dq: jnp.ndarray
    W_uq: jnp.ndarray
    q_layernorm: LayerNorm

    # Parameters for KV projections
    W_dkv: jnp.ndarray 
    W_ukv: jnp.ndarray 
    kv_layernorm: LayerNorm

    # Output projection
    W_o: jnp.ndarray

    # RoPE buffers (non-trainable)
    max_seq_len: int = eqx.static_field()
    rope_theta: float = eqx.static_field()
    cos_cached: jax.Array
    sin_cached: jax.Array 

    def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0, *, key):
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = (2 * d_model) // 3

        self.qk_nope_dim = self.dh // 2
        self.qk_rope_dim = self.dh // 2

        # Split key for reproducibility
        keys = jax.random.split(key, 6)
        self.W_dq = 0.01 * jax.random.normal(keys[0], (d_model, self.q_proj_dim), dtype=jnp.float32)
        self.W_uq = 0.01 * jax.random.normal(keys[1], (self.q_proj_dim, d_model), dtype=jnp.float32)
        self.q_layernorm = LayerNorm(self.q_proj_dim, eps=1e-5)

        self.W_dkv = 0.01 * jax.random.normal(keys[2], (d_model, self.kv_proj_dim + self.qk_rope_dim), dtype=jnp.float32)
        self.W_ukv = 0.01 * jax.random.normal(keys[3], (self.kv_proj_dim, d_model + (n_heads * self.qk_nope_dim)), dtype=jnp.float32)
        self.kv_layernorm = LayerNorm(self.kv_proj_dim, eps=1e-5)

        self.W_o = 0.01 * jax.random.normal(keys[4], (d_model, d_model), dtype=jnp.float32)

        self.max_seq_len = max_len
        self.rope_theta = rope_theta

       
        freq_seq = jnp.arange(0, self.dh, 2, dtype=jnp.float32)
        freqs = 1.0 / (rope_theta ** (freq_seq / self.dh))
        positions = jnp.arange(max_len, dtype=jnp.float32)
        emb = jnp.outer(positions, freqs) # Precompute RoPE cos and sin caches with explicit float32 dtype
       
        self.cos_cached = jnp.cos(emb)[None, None, :, :].astype(jnp.float32)
        self.sin_cached = jnp.sin(emb)[None, None, :, :].astype(jnp.float32)
        
        
    
    def __call__(self, q_input, kv_input, kv_cache=None, past_length=0):
        # q_input shape: (B, Sq, D)
        # kv_input shape: (B, Skv, D) where D == d_model
        B, Sq, D = q_input.shape
        _, Skv, _ = kv_input.shape

        # Q projections (from q_input)
        compressed_q = jnp.matmul(q_input, self.W_dq)  # (B, Sq, q_proj_dim)
        compressed_q = jax.vmap(jax.vmap(self.q_layernorm))(compressed_q)
        Q = jnp.matmul(compressed_q, self.W_uq)  # (B, Sq, d_model)
        Q = Q.reshape(B, Sq, self.n_heads, self.dh).transpose(0, 2, 1, 3)
        Q, Q_for_rope = jnp.split(Q, [self.qk_nope_dim], axis=-1)

        # Q Decoupled RoPE
        cos_q = jax.lax.stop_gradient(self.cos_cached[:, :, past_length:past_length+Sq, :self.qk_rope_dim//2].repeat(2, axis=-1))
        sin_q = jax.lax.stop_gradient(self.sin_cached[:, :, past_length:past_length+Sq, :self.qk_rope_dim//2].repeat(2, axis=-1))
        Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)

        # KV Projections (from kv_input)
        if kv_cache is None:
            compressed_kv = jnp.matmul(kv_input, self.W_dkv)  # (B, Skv, kv_proj_dim + qk_rope_dim)
            KV_for_lora, K_for_rope = jnp.split(compressed_kv, [self.kv_proj_dim], axis=-1)
            KV_for_lora = jax.vmap(jax.vmap(self.kv_layernorm))(KV_for_lora)
        else:
            new_kv = jnp.matmul(kv_input, self.W_dkv)
            compressed_kv = jnp.concatenate([kv_cache, new_kv], axis=1)
            new_kv_val, new_K_for_rope = jnp.split(new_kv, [self.kv_proj_dim], axis=-1)
            old_kv, old_K_for_rope = jnp.split(kv_cache, [self.kv_proj_dim], axis=-1)
            new_kv_val = jax.vmap(jax.vmap(self.kv_layernorm))(new_kv_val)
            old_kv = jax.vmap(jax.vmap(self.kv_layernorm))(old_kv)
            KV_for_lora = jnp.concatenate([old_kv, new_kv_val], axis=1)
            K_for_rope = jnp.concatenate([old_K_for_rope, new_K_for_rope], axis=1)

        # Rest of KV processing
        KV = jnp.matmul(KV_for_lora, self.W_ukv)
        KV = KV.reshape(B, -1, self.n_heads, self.dh + self.qk_nope_dim).transpose(0, 2, 1, 3)
        K, V = jnp.split(KV, [self.qk_nope_dim], axis=-1)
        S_full = K.shape[2]

        # K RoPE
        K_for_rope = K_for_rope.reshape(B, -1, 1, self.qk_rope_dim).transpose(0, 2, 1, 3)
        cos_k = jax.lax.stop_gradient(self.cos_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(2, axis=-1))
        sin_k = jax.lax.stop_gradient(self.sin_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(2, axis=-1))
        K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)
        K_for_rope = jnp.repeat(K_for_rope, self.n_heads, axis=1)

        # Concatenate parts
        q_heads = jnp.concatenate([Q, Q_for_rope], axis=-1)
        k_heads = jnp.concatenate([K, K_for_rope], axis=-1)
        v_heads = V

        # Cross attention doesn't need causal mask
        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(q_heads.shape[-1])
        attn_logits = jnp.matmul(q_heads, jnp.swapaxes(k_heads, -1, -2)) * scale
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_output = jnp.matmul(attn_weights, v_heads)
        
        # Reshape back to (B, Sq, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, Sq, D)

        # Final output projection
        out = jnp.matmul(attn_output, self.W_o.T)

        return out, compressed_kv



        
        
    
    