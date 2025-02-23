import jax
from jax import numpy as jnp
import equinox as eqx
from equinox import nn

from .dense import DenseGeneral
from .activations import SwiGLU
from utils.initializers import get_keys

class MLP(eqx.Module):
    c_fc: DenseGeneral
    swiglu: SwiGLU
    c_proj: DenseGeneral
    dropout: nn.Dropout
    
    def __init__(self, d_model: int, dropout: float, key: jax.random.PRNGKey):
        keys = get_keys(key, 4)
        self.c_fc = DenseGeneral(d_model, 4 * d_model, keys[0])
        self.swiglu = SwiGLU(4 * d_model, 4 * d_model, keys[1])
        self.c_proj = DenseGeneral(4 * d_model, d_model, keys[2])
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        x = self.c_fc(x)
        x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x, key=key)
        return x 