import jax
from jax import numpy as jnp
import equinox as eqx
from equinox import Module

from utils.initializers import xavier_init, get_keys

class DenseGeneral(Module):
    weight: jax.Array
    bias: jax.Array
    in_features: int = eqx.static_field()
    out_features: int = eqx.static_field()
    use_bias: bool = eqx.static_field()

    def __init__(self, in_features: int, out_features: int, key: jax.random.PRNGKey, use_bias: bool = True):
        keys = get_keys(key, 2 if use_bias else 1)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weight = xavier_init(keys[0], (in_features, out_features))
        if use_bias:
            self.bias = xavier_init(keys[1], (out_features,))
        else:
            self.bias = jnp.zeros(out_features)
            
    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.einsum("...ij,...jk->...ik", x, self.weight) + self.bias 