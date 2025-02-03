import jax
import jax.numpy as jnp

from .._module import Module, Static


class Dropout(Module):
    rate: Static[float]
    
    def __init__(self, rate: float):
        self.rate = rate
    
    def __call__(self, x):
        # have to do it manually not using any jax functions
        return x * jax.random.bernoulli(self.rate, x.shape)