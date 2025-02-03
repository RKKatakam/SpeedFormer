import jax
import jax.numpy as jnp
from .._module import Module, Static


class Gelu(Module):
    
    def __init__(self):
        super().__init__()
    
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        jax.nn.gelu(x)