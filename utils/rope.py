import jax
from jax import numpy as jnp

def rotate_half(x: jax.Array) -> jax.Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rope_x(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    return (x * cos) + (rotate_half(x) * sin) 