import jax
from jax import numpy as jnp

def xavier_init(key: jax.random.PRNGKey, shape: tuple[int, ...]) -> jax.Array:
    return jax.random.normal(key, shape) * jnp.sqrt(2 / sum(shape))

def get_keys(key: jax.random.PRNGKey, num_keys: int) -> tuple[jax.random.PRNGKey, ...]:
    return tuple(jax.random.split(key, num_keys+1))[:-1] 