import jax
from jax import numpy as jnp

@jax.jit
def Gc_comp(r, z):

    return jnp.pi * r**2

@jax.jit
def G_comp(a, b, z):

    return jnp.pi * a * b

@jax.jit 
def Gc(r, z, y, x):

    return 0.5 * r * (
        r * (x - y) 
        + z * (jnp.cos(y) - jnp.cos(x))
    )

@jax.jit
def G(a, b, z, y, x):

    return 0.5 * a * (
        b * (x - y) 
        + z * (jnp.cos(y) - jnp.cos(x))
    )