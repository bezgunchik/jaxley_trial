import os

# Set the environment variable for JAX platform to CPU
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from jax import default_backend

print("Current JAX platform:", default_backend())


import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import jit

import jaxley as jx
from jaxley.channels import Na, K, Leak
from jaxley.synapses import IonotropicSynapse
from jaxley.connect import fully_connect