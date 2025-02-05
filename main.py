import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass
from torch2jax import t2j
from torch import Tensor

torch.manual_seed(42)

# --- Updated model arguments ---
@dataclass
class Config:
    vocab_size: int
    seq_len: int
    d_model: int
    num_heads: int = None
    v_head_dim: int = None
    
    nope_head_dim: int = None
    rope_head_dim: int = None
    
    hidden_dim: int = None
    num_kv_heads: int = None
    num_layers: int = 4
    dropout: float = 0.0
    bias: bool = False
    weight_tying: bool = False
    activation: str = "silu"
    mlp: str = "GLU"
    kv_lora_rank: int = None
    q_lora_rank: int = None
    attn_type: str = "mla"
    
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # [: (dim // 2)] for odd number truncation
    # torch.arange(0, dim, 2) -> 2(i-1)//d while i= 1,2,..,(d//2)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # gives diffrent angle vector

    # e^it = cos(t) + i sin(t)
    freqs_cos = torch.cos(freqs)  # real
    freqs_sin = torch.sin(freqs)  # imaginary
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.dim()
    assert 1 < ndim
    assert freqs_cis.shape == (
        x.shape[1],
        x.shape[-1],
    ), f"{freqs_cis.shape=}, {(x.shape[1], x.shape[-1])=}"

    # keep 2nd (T) and last(freq) dim same else make dim 1 for freq_cis
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # print(shape)
    return freqs_cis.view(shape)


def apply_rope(q,k, cis):
    # Idea suppose vector v = [x,y,x1,y1,...] # v.shape = dim
    # convert vetor into complex num # ie two vec one real, one imagery
    # [x,y,x1,y1,...] -> x+iy, x1+iy1
    # Multiplying by complex num == roatate vector
    # => (x + iy) * (cos + isin) -> x'+iy'
    # restack
    # x'+iy' -> [x',y',x1',y1'...]
    # you roated vector in chunks of two lfg!!!
    _, seq_len, _, _ = q.shape

    freqs_cos, freqs_sin = cis
    freqs_cos, freqs_sin = freqs_cos[:seq_len], freqs_sin[:seq_len]

    #  rehsape a shape (...,n )-> (..., n//2,2)
    q_cis = q.float().reshape(
        q.shape[:-1] + (-1, 2)
    )  # (B,T,nhead,C) -> (B,T,nhead,Cc,2) # Cc = C//2
    k_cis = k.float().reshape(k.shape[:-1] + (-1, 2))  # (B,T,nhead,C) -> (B,T,nhead,Cc,2)

    xq_r, xq_i = q_cis.unbind(-1)  # (B,T,nhead,Cc,2) -> ((B,T,Cc), (B,T,Cc)) split into two tuple
    xk_r, xk_i = k_cis.unbind(-1)  # (B,T,nhead,Cc,2) -> ((B,T,Cc), (B,T,Cc))

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)  # freqs.shape = (1,T,1,Cc)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # e+if = (a+ib) * (c+di) = (ac-bd) + i (ad+bc)
    # a = xq_r , b = xq_i
    # c = fcos , d = fsin
    # ...
    # e = (ac-bd) = xq_r * freqs_cos - xq_i * freqs_sin
    # f = (c+di)  = xq_r * freqs_sin + xq_i * freqs_cos

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin  # (ac-bd)   # shape =  # (B,T,nhead,Cc)
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos  # (ad+bc) * i
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin  # (ac-bd)
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos  # (ad+bc) * i

    # now we stack r,i -> [r,i,r2,i2]
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1)  # (B,T,nhead,Cc,2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1)  # (B,T,nhead,Cc,2)

    # flatten last two dimensions
    xq_out = xq_out.flatten(3)  # (B,T,nhead,C)
    xk_out = xk_out.flatten(3)  # (B,T,nhead,C)

    return xq_out.type_as(q), xk_out.type_as(q)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        """
        Paper: https://arxiv.org/abs/1910.07467
        """
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


    
class MultiHeadLatentAttention(nn.Module):
    """
    Multi Head Latent Attention 
    paper: https://arxiv.org/pdf/2405.04434
    
    TLDR: 
    kv are low ranks, this verient of attention project q,k,v to low rank to save memory,
    replace linear with lora(ish) layers

    by joey00072 (https://github.com/joey00072)
    """
    def __init__(self, config: Config):
        super().__init__()
        
        assert config.v_head_dim is not None , f"v_head_dim is not defined {config.v_head_dim=}"
        assert config.q_lora_rank is not None , f"q_lora_rank is not defined {config.q_lora_rank=}"
        assert config.kv_lora_rank is not None , f"kv_lora_rank is not defined {config.kv_lora_rank=}"
        assert config.rope_head_dim is not None , f"rope_head_dim is not defined {config.rope_head_dim=}"
        
        self.config = config
        
        self.dim = config.d_model
        self.num_heads = config.num_heads
        self.v_head_dim = config.v_head_dim
        
        self.nope_head_dim = config.nope_head_dim
        self.rope_head_dim = config.rope_head_dim
        
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        
        self.dropout = config.dropout
        
        # note: head dim of query and key if different from head dim of value
        
        # (attention_dim == num_head*head_dim) > d_model in deepseekv2
        # this is dim between wV and wQ
        self.value_dim = self.num_heads * self.v_head_dim
        
        # this is dims between wQ and wK
        self.nope_dim = self.num_heads * self.nope_head_dim
        self.rope_dim = self.num_heads * self.rope_head_dim  
        
        # query compression
        self.compress_q_linear = nn.Linear(self.dim, self.q_lora_rank, bias=False)  # W_DQ
        self.decompress_q_nope = nn.Linear(self.q_lora_rank, self.nope_dim, bias=False)
        self.decompress_q_rope = nn.Linear(self.q_lora_rank, self.rope_dim, bias=False)
        self.q_norm = RMSNorm(dim=self.q_lora_rank)
        
        
        # key and value compression
        self.compress_kv_linear = nn.Linear(self.dim, self.kv_lora_rank, bias=False)  # W_DKV
        self.decompress_k_nope = nn.Linear(self.kv_lora_rank, self.nope_dim, bias=False)
        self.decompress_v_linear = nn.Linear(self.kv_lora_rank, self.value_dim, bias=False)
        self.kv_norm = RMSNorm(dim=self.kv_lora_rank)
        
        
        self.k_rope_linear = nn.Linear(self.dim, self.rope_head_dim  , bias=False)
        # self.rope_norm = RMSNorm(self.rope_dim) # not in deepseekv2

        self.proj = nn.Linear(self.value_dim , self.dim, bias=False)
        self.res_dropout = nn.Dropout(p=config.dropout)
        
        self.scale = 1/ (self.value_dim**0.5)
        
 
    def forward(self, x: Tensor,mask: torch.Tensor, freqs_cis: Tensor):
        batch_size, seq_len, _ = x.shape

        compressed_q = self.compress_q_linear(x)
        norm_q = self.q_norm(compressed_q)
        query_nope:Tensor = self.decompress_q_nope(norm_q)
        query_rope:Tensor = self.decompress_q_rope(norm_q)

        compressed_kv = self.compress_kv_linear(x)
        norm_kv = self.kv_norm(compressed_kv)
        key_nope: Tensor = self.decompress_k_nope(norm_kv)
        value: Tensor = self.decompress_v_linear(norm_kv)
        
        key_rope:Tensor = self.k_rope_linear(x)
        # norm_rope = self.rope_norm(key_rope)

        query_nope = query_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        query_rope = query_rope.view(batch_size, seq_len, self.num_heads, self.rope_head_dim).transpose(1,2)
        
        key_rope = key_rope.view(batch_size, seq_len, 1, self.rope_head_dim).transpose(1,2)
        key_nope = key_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        
        value = value.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1,2)

        # query_nope = query_nope * self.scale
        # key_nope = key_nope * self.scale
        value = value * self.scale
        
        q_rope,k_rope = apply_rope(query_rope,key_rope, cis=freqs_cis)
        
        q_recombined = torch.empty((batch_size,self.num_heads,seq_len, self.rope_head_dim + self.nope_head_dim), device=x.device)
        k_recombined = torch.empty((batch_size, self.num_heads, seq_len, self.rope_head_dim + self.nope_head_dim), device=x.device)
        
        q_recombined[:,:,:,:self.nope_head_dim] = query_nope
        q_recombined[:,:,:,self.nope_head_dim:] = q_rope
        
        # k_rope = torch.repeat_interleave(k_rope, self.num_heads, dim=1) # >> you dont need to do this <<
        # 👇 broadcasting will do replication krope to all heads automagically
        k_recombined[:,:,:,:self.nope_head_dim] = key_nope
        k_recombined[:,:,:,self.nope_head_dim:] = k_rope

        output = F.scaled_dot_product_attention(q_recombined, k_recombined, value, is_causal=True, dropout_p=self.dropout)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.v_head_dim)

        output = self.proj(output)
        output = self.res_dropout(output)
        return output




class MLA_Inference(MultiHeadLatentAttention):
    def __init__(self,config:Config):
        super().__init__(config)
        self.inference_merged = False
        
    def inference_merge(self):
        Wd_Qnope = self.decompress_q_nope.weight.detach()
        Wd_Knope = self.decompress_k_nope.weight.detach()
        Wd_V = self.decompress_v_linear.weight.detach()
        
        W_proj = self.proj.weight.detach()
        
        Wd_Qnope = Wd_Qnope.reshape(self.num_heads, Wd_Qnope.T.shape[0], -1)
        Wd_Knope = Wd_Knope.reshape(self.num_heads, Wd_Knope.T.shape[0], -1)
        
        # print(f"Wd_Qnope.shape: {Wd_Qnope.shape}, Wd_Knope.shape: {Wd_Knope.shape}")
        WdQK = Wd_Qnope @ Wd_Knope.transpose(-2, -1)
        # print(f"WdQK.shape: {WdQK.shape}")
        
        WdVO = Wd_V.T @ W_proj
        
        # print(f"WdQK.shape: {WdQK.shape}, WdVO.shape: {WdVO.shape}")
        
        self.register_buffer("WdQK", WdQK)
        
        self.inference_merged = True
        
    def forward(self,x:Tensor,freqs_cis:Tensor):
        assert self.inference_merged, "model is not merged run .inference_merge() first"


        batch_size, seq_len, _ = x.shape

        
        def _test_self_attention(x:Tensor):
            compressed_q = self.compress_q_linear(x)
            norm_q = self.q_norm(compressed_q)
            query_nope:Tensor = self.decompress_q_nope(norm_q)
            query_rope:Tensor = self.decompress_q_rope(norm_q)

            compressed_kv = self.compress_kv_linear(x)
            norm_kv = self.kv_norm(compressed_kv)
            key_nope: Tensor = self.decompress_k_nope(norm_kv)
            value: Tensor = self.decompress_v_linear(norm_kv)
            
            key_rope:Tensor = self.k_rope_linear(x)
            # norm_rope = self.rope_norm(key_rope)

            query_nope = query_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
            query_rope = query_rope.view(batch_size, seq_len, self.num_heads, self.rope_head_dim).transpose(1,2)
            
            key_rope = key_rope.view(batch_size, seq_len, 1, self.rope_head_dim).transpose(1,2)
            key_nope = key_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
            
            value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
            
            k_rope, q_rope = apply_rope(query_rope,key_rope, cis=freqs_cis)
            
            q_recombined = torch.empty((batch_size,self.num_heads,seq_len, self.head_dim), device=x.device)
            k_recombined = torch.empty((batch_size, self.num_heads, seq_len, self.head_dim), device=x.device)
            
            q_recombined[:,:,:,:self.nope_head_dim] = query_nope
            q_recombined[:,:,:,self.nope_head_dim:] = q_rope
            
            # k_rope = torch.repeat_interleave(k_rope, self.num_heads, dim=1) # >> you dont need to do this <<
            # 👇 broadcasting will do replication krope to all heads automagically
            k_recombined[:,:,:,:self.nope_head_dim] = key_nope
            k_recombined[:,:,:,self.nope_head_dim:] = k_rope

            output = F.scaled_dot_product_attention(q_recombined, k_recombined, value, is_causal=True)

            output = output.contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

            output = self.proj(output)

            return output



if __name__ == "__main__":
    
    d_model = 1024
    num_heads = 70
    
    v_head_dim = 32
    kv_lora_rank = 64
    q_lora_rank = 3 * kv_lora_rank
    
    rope_head_dim = 64
    nope_head_dim = 32
    
    config = Config(
        vocab_size=256,
        d_model=d_model,
        seq_len=2048,
        num_heads=num_heads,
        v_head_dim=v_head_dim,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
    )

    mla = MultiHeadLatentAttention(config)
    x = torch.randn(2, 3000, d_model)
    freqs_cis = precompute_freqs_cis(config.rope_head_dim, config.seq_len)
    # mla = torch.compile(mla)
    print(f"Model Size: {sum(p.numel() for p in mla.parameters())/1e6}M params, attn size {d_model*d_model*4/1e6}m")
    output = mla(x,None, freqs_cis)
    print(output.shape)
