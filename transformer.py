import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from attention import MLA

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        is_decoder: bool = True,
        dropout: float = 0.1,
        max_len: int = 1024,
        rope_theta: float = 10000.0
    ):
        super().__init__()
        self.is_decoder = is_decoder
        
        # Attention - causal for decoder, non-causal for encoder
        self.attn = MLA(
            d_model=d_model,
            n_heads=n_heads,
            max_len=max_len,
            rope_theta=rope_theta,
            causal=is_decoder  # Causal attention for decoder only
        )
        
        # FFN
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model)
        )
        
        
        # Layer norms and dropout
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, kv_cache=None, past_length=0):
        # Attention with residual connection
        residual = x
        x = self.attn_norm(x)
            
        x, new_kv = self.attn(x, kv_cache, past_length)
        x = self.dropout(x)
        x = residual + x
            
        # Feed forward with residual connection
        residual = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = residual + x
        
        return x, new_kv

class Encoder(nn.Module):
    """
    Encoder-only Transformer model (like BERT)
    """
    def __init__(
        self, 
        n_layers: int, 
        d_model: int, 
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 1024,
        rope_theta: float = 10000.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_mult=ff_mult,
                is_decoder=False,  # Always encoder mode
                dropout=dropout,
                max_len=max_len,
                rope_theta=rope_theta
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder-only model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Processed output tensor
        """
        for layer in self.layers:
            x, _ = layer(x)
        return self.norm(x)

class Decoder(nn.Module):

    def __init__(
        self, 
        n_layers: int, 
        d_model: int, 
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_len: int = 1024,
        rope_theta: float = 10000.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_mult=ff_mult,
                is_decoder=True,  # Always decoder mode
                dropout=dropout,
                max_len=max_len,
                rope_theta=rope_theta
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        kv_caches: Optional[List] = None, 
        past_length: int = 0
    ) -> Tuple[torch.Tensor, List]:
        """
        Forward pass of the decoder-only model with optional KV caching.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            kv_caches: Optional list of key-value caches for each layer
            past_length: Length of past tokens for position calculations
            
        Returns:
            Tuple of (output tensor, list of new key-value caches)
        """
        if kv_caches is None:
            kv_caches = [None] * len(self.layers)
        new_kv_caches = []
        
        for layer, kv_cache in zip(self.layers, kv_caches):
            x, new_kv = layer(x, kv_cache, past_length)
            new_kv_caches.append(new_kv)
        
        return self.norm(x), new_kv_caches
