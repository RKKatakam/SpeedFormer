import jax
from jax import numpy as jnp
import equinox as eqx

class SwiGLU(eqx.Module):
    W: jax.Array
    V: jax.Array
    b: jax.Array
    c: jax.Array

    def __init__(self, dim_in, dim_out, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.W = jax.random.normal(k1, (dim_in, dim_out))
        self.V = jax.random.normal(k2, (dim_in, dim_out))
        self.b = jax.random.normal(k3, (dim_out,))
        self.c = jax.random.normal(k4, (dim_out,))

    def __call__(self, x):
        return jax.nn.swish(jnp.dot(x, self.W) + self.b) * (jnp.dot(x, self.V) + self.c) 