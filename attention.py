import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

def apply_rope_x(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class MultiLayerAttention(nn.Module):
    """Multi-Layer Attention with LoRA and RoPE support.
    
    Can be used for both self-attention and cross-attention.
    """
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.0,
        max_len: int = 1024, 
        rope_theta: float = 10000.0,
        causal: bool = True,
        q_proj_ratio: float = 0.5,     # Controls compressed q dimension (LoRA)
        kv_proj_ratio: float = 2/3,    # Controls compressed kv dimension (LoRA)
        bias: bool = False
    ):
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        # Core parameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.causal = causal
        self.dropout_p = dropout
        
        # LoRA projection dimensions
        self.q_proj_dim = int(d_model * q_proj_ratio)
        self.kv_proj_dim = int(d_model * kv_proj_ratio)
        
        # RoPE dimensions
        self.qk_nope_dim = self.dh // 2  # Non-RoPE dimension
        self.qk_rope_dim = self.dh // 2  # RoPE dimension
        
        # Query projections with LoRA
        self.W_dq = nn.Parameter(0.01 * torch.randn((d_model, self.q_proj_dim)))
        self.W_uq = nn.Parameter(0.01 * torch.randn((self.q_proj_dim, self.d_model)))
        self.q_layernorm = nn.LayerNorm(self.q_proj_dim)
        
        # Key-Value projections with LoRA
        self.W_dkv = nn.Parameter(0.01 * torch.randn((d_model, self.kv_proj_dim + self.qk_rope_dim)))
        self.W_ukv = nn.Parameter(0.01 * torch.randn((self.kv_proj_dim,
                                                      self.d_model + (self.n_heads * self.qk_nope_dim))))
        self.kv_layernorm = nn.LayerNorm(self.kv_proj_dim)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # RoPE parameters
        self.max_seq_len = max_len
        self.rope_theta = rope_theta
        
        # Initialize RoPE embeddings
        self._init_rope_embeddings()

    def _init_rope_embeddings(self):
        """Initialize and cache the RoPE embeddings."""
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
        emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]
        
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        kv_cache: Optional[torch.Tensor] = None,
        past_length: int = 0,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: Query embeddings (B, Sq, D)
            key: Key embeddings (B, Sk, D), if None uses query (self-attention)
            value: Value embeddings (B, Sv, D), if None uses key (self-attention)
            kv_cache: Previous KV projections for incremental decoding
            past_length: Length of kv_cache for positional calculations
            attn_mask: Attention mask (B, Sq, Sk) or (Sq, Sk) or (1, Sq, Sk)
            key_padding_mask: Mask for padded tokens in key
            need_weights: Whether to return attention weights
            
        Returns:
            output: Attention output (B, Sq, D)
            kv_cache: Updated KV cache for incremental decoding
        """
        # Handle self-attention case
        is_self_attn = key is None and value is None
        if is_self_attn:
            key = value = query
            
        B, Sq, D = query.size()
        Sk = key.size(1)
        
        # Process Query
        compressed_q = query @ self.W_dq
        compressed_q = self.q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq
        Q = Q.view(B, -1, self.n_heads, self.dh).transpose(1, 2)
        Q, Q_for_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        
        # Apply RoPE to Query
        cos_q = self.cos_cached[:, :, past_length:past_length+Sq, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        sin_q = self.sin_cached[:, :, past_length:past_length+Sq, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)
        
        # Process Key-Value
        if kv_cache is None:
            compressed_kv = key @ self.W_dkv
            KV_for_lora, K_for_rope = torch.split(compressed_kv,
                                                [self.kv_proj_dim, self.qk_rope_dim],
                                                dim=-1)
            KV_for_lora = self.kv_layernorm(KV_for_lora)
        else:
            new_kv = key @ self.W_dkv
            compressed_kv = torch.cat([kv_cache, new_kv], dim=1)
            new_kv, new_K_for_rope = torch.split(new_kv,
                                               [self.kv_proj_dim, self.qk_rope_dim],
                                               dim=-1)
            old_kv, old_K_for_rope = torch.split(kv_cache,
                                               [self.kv_proj_dim, self.qk_rope_dim],
                                               dim=-1)
            new_kv = self.kv_layernorm(new_kv)
            old_kv = self.kv_layernorm(old_kv)
            KV_for_lora = torch.cat([old_kv, new_kv], dim=1)
            K_for_rope = torch.cat([old_K_for_rope, new_K_for_rope], dim=1)
        
        # Form K and V
        KV = KV_for_lora @ self.W_ukv
        KV = KV.view(B, -1, self.n_heads, self.dh+self.qk_nope_dim).transpose(1, 2)
        K, V = torch.split(KV, [self.qk_nope_dim, self.dh], dim=-1)
        S_full = K.size(2)
        
        # Apply RoPE to Key
        K_for_rope = K_for_rope.view(B, -1, 1, self.qk_rope_dim).transpose(1, 2)
        cos_k = self.cos_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        sin_k = self.sin_cached[:, :, :S_full, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)
        K_for_rope = K_for_rope.repeat(1, self.n_heads, 1, 1)
        
        # Concatenate parts to form complete Q and K
        q_heads = torch.cat([Q, Q_for_rope], dim=-1)
        k_heads = torch.cat([K, K_for_rope], dim=-1)
        v_heads = V  # Already reshaped before the split
        
        # Handle masking
        if attn_mask is not None:
            # Use provided attention mask
            sq_mask = attn_mask.to(torch.bool)
        elif self.causal and is_self_attn:
            # Create causal mask for self-attention
            causal_mask = torch.ones((Sq, S_full), device=query.device)
            causal_mask = torch.tril(causal_mask, diagonal=past_length)
            sq_mask = (causal_mask == 1)[None, None, :, :]
        else:
            # No masking for non-causal or cross-attention without explicit mask
            sq_mask = None
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Expand key_padding_mask to match attention dimensions
            if sq_mask is None:
                sq_mask = ~key_padding_mask[:, None, None, :]
            else:
                sq_mask = sq_mask & (~key_padding_mask[:, None, None, :])
        
        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            q_heads, k_heads, v_heads,
            attn_mask=sq_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False  # We handle causality with our mask
        )
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).reshape(B, Sq, D)
        output = self.out_proj(attn_output)
        
        # Optional attention weights for return
        attn_weights = None
        if need_weights:
            # Manual attention weight calculation
            attn_weights = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(q_heads.size(-1))
            if sq_mask is not None:
                attn_weights = attn_weights.masked_fill(~sq_mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
        
        return output, compressed_kv, attn_weights if need_weights else None