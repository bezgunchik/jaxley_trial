import jax
import jax.numpy as jnp
from jax import random

print("Backend JAX is using:", jax.default_backend())

key = random.PRNGKey(0)
x = random.normal(key, (1000,))
print("Array:", x)
