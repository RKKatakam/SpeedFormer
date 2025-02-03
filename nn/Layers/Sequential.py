import jax
import jax.numpy as jnp
from .._module import Module, Static


class Sequential(Module):
    layers: list[callable]
    
    def __init__(self, *layers: list[callable]):
        self.layers = layers
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    