import jax
import jax.numpy as jnp
from .._module import Module, Static


class LayerNorm(Module):
    # Mark configuration parameters as static using the Static type.
    features: Static[int]
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, features: int, key: jax.random.PRNGKey):
        self.features = features
        key_w, key_b = jax.random.split(key)
        # Initialize trainable parameters.
        self.weight = jax.random.normal(key_w, (features,))
        self.bias = jax.random.normal(key_b, (features,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.weight * (x - mean) / jnp.sqrt(var + 1e-5) + self.bias
    
