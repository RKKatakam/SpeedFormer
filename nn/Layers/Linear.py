import jax
import jax.numpy as jnp
from .._module import Module, Static


def xavier_uniform(key, shape, dtype=jnp.float32):
    fan_in, fan_out = shape
    scale = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)



class Linear(Module):
    # Mark configuration parameters as static using the Static type.
    in_features: Static[int]
    out_features: Static[int]
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_features: int, out_features: int, key: jax.random.PRNGKey):
        self.in_features = in_features
        self.out_features = out_features
        key_w, key_b = jax.random.split(key)
        # Initialize trainable parameters.
        self.weight = xavier_uniform(key_w, (in_features, out_features))
        self.bias = jax.random.normal(key_b, (out_features,))
        
        
    def __init(self, in_features: int, out_features: int):
        random = jax.random.PRNGKey(random.randint(0, 1000))
        self.in_features = in_features
        self.out_features = out_features
        key_w, key_b = jax.random.split(random)
        # Initialize trainable parameters.
        self.weight = xavier_uniform(key_w, (in_features, out_features))
        self.bias = jax.random.normal(key_b, (out_features,))
        

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, self.weight) + self.bias
    