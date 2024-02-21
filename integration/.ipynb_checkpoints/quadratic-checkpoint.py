import jax
from jax import numpy as jnp

@jax.jit
def Gc_comp(r, z):

    return jnp.pi * r**2

@jax.jit
def G_comp(a, b, z):

    return 0.25 * jnp.pi * a * b * (
        a * a + b * b + 4 * z * z
    )

@jax.jit 
def Gc(r, z, y, x):

    r2 = r * r
    z2 = z * z

    return 0.125 * r * (
        -2 * r * (r2 + 2 * z2) * (y - x)
        + z * (
            2 * (3 * r2 + z2) * jnp.cos(y)
            + r * z * jnp.sin(2 * y)
            - 2 * jnp.cos(x) * (
                3 * r2 + z2 
                + r * z * jnp.sin(x)
            )
        )
    )

@jax.jit
def G(a, b, z, y, x):

    a2 = a * a
    b2 = b * b
    z2 = z * z
    c2 = (a - b) * (a + b)

    return a * (
        -6 * b * (a2 + b2 + 4 * z2) * (y - x)
        + 3 * z * (a2 + 11 * b2 + 4 * z2) * jnp.cos(y)
        + c2 * z * jnp.cos(3 * y)
        - 2 * z * jnp.cos(x) * (
            a2 + 17 * b2 + 6 * z2 
            + c2 * jnp.cos(2 * x)
        )
        + 3 * b * (2 * z2) * (
            jnp.sin(2 * y) - jnp.sin(2 * x)
        )
    ) / 48
