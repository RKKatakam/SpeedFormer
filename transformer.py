import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from attention import MultiLayerAttention


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation"""
    def __init__(
        self, 
        d_model: int, 
        d_ff: int = None, 
        dropout: float = 0.0,
        act_fn: str = "swiglu"
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        # Adjust d_ff based on activation function
        self.act_fn = act_fn.lower()
        if self.act_fn == "swiglu":
            self.w1 = nn.Linear(d_model, d_ff * 2, bias=False)
            self.act = lambda x: F.silu(x[..., :d_ff]) * x[..., d_ff:]
        elif self.act_fn == "gelu":
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.act = F.gelu
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")
            
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_fn == "swiglu":
            x = self.w1(x)
            x = self.act(x)
        else:
            x = self.w1(x)
            x = self.act(x)
        
        x = self.dropout(x)
        x = self.w2(x)
        return x

class DecoderBlock(nn.Module):
    """Transformer decoder block with Multi-Layer Attention"""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        max_len: int = 1024,
        rope_theta: float = 10000.0,
        q_proj_ratio: float = 0.5,  
        kv_proj_ratio: float = 2/3
    ):
        super().__init__()
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Attention layer
        self.attn = MultiLayerAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_len=max_len,
            rope_theta=rope_theta,
            causal=True,
            q_proj_ratio=q_proj_ratio,
            kv_proj_ratio=kv_proj_ratio
        )
        
        # Feed-forward network
        self.ff = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        past_length: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        residual = x
        x = self.norm1(x)
        attn_out, new_kv_cache, _ = self.attn(
            query=x,
            kv_cache=kv_cache,
            past_length=past_length
        )
        x = residual + attn_out
        
        # Feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        
        return x, new_kv_cache

class Decoder(nn.Module):
    """Transformer decoder stack with Multi-Layer Attention"""
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        max_len: int = 1024,
        rope_theta: float = 10000.0
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_len=max_len,
                rope_theta=rope_theta
            ) for _ in range(n_layers)
        ])
        
        self.norm_out = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        x: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        past_length: int = 0
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Initialize kv_caches if not provided
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
            
        new_kv_caches = []
        
        # Process through each layer
        for i, layer in enumerate(self.layers):
            x, new_kv_cache = layer(x, kv_caches[i], past_length)
            new_kv_caches.append(new_kv_cache)
            
        # Apply final normalization
        x = self.norm_out(x)
        
        return x, new_kv_caches