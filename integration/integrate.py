import jax 
import jax.numpy as jnp
from integrals import G_circ, G, G_complete_circ, G_complete

one = 0.999999999999999

@jax.jit
def ellipse_arc(a, b, z, phi1, phi2):

    return jax.lax.cond(
        jnp.abs(phi2 - phi1 - 2 * jnp.pi) < 1e-6,
        lambda: 2 * G_complete(a, b, z),
        lambda: ellipse_arc_incomplete(a, b, z, phi1, phi2)
    )

@jax.jit
def circle_arc(r, z, phi1, phi2):

    return jax.lax.cond(
        jnp.abs(phi2 - phi1 - 2 * jnp.pi) < 1e-6,
        lambda: 2 * G_complete_circ(r, z),
        lambda: circle_arc_incomplete(r, z, phi1, phi2)
    )

# this function assumes that both limits are in (0, 2*pi) 
def ellipse_arc_incomplete(a, b, z, phi1, phi2):

    s1 = jnp.sin(phi1)
    s2 = jnp.sin(phi2)
    c1 = jnp.cos(phi1)
    c2 = jnp.cos(phi2)

    conditions = jnp.array([
        (c1 > 0) & (c2 > 0) & (s1 > s2), # C
        (c1 > 0) & (c2 > 0),             # A/F
        (c1 > 0) & (c2 < 0),             # B
        (c1 < 0) & (c2 < 0),             # D
        (c1 < 0) & (c2 > 0)              # E
    ])

    outcomes = [
        lambda: 2 * G_complete(a, b, z) - G(a, b, z, s2, s1),
        lambda: G(a, b, z, s1, s2),
        lambda: G(a, b, z, s1, one) + G(a, b, z, s2, one),
        lambda: G(a, b, z, s2, s1),
        lambda: G(a, b, z, -one, s1) + G(a, b, z, -one, s2)
    ]

    ind = jnp.argwhere(conditions, size=1).squeeze()
    return jax.lax.switch(ind, outcomes)

# this function assumes that both limits are in (0, 2*pi) 
def circle_arc_incomplete(r, z, phi1, phi2):

    s1 = jnp.sin(phi1)
    s2 = jnp.sin(phi2)
    c1 = jnp.cos(phi1)
    c2 = jnp.cos(phi2)

    conditions = jnp.array([
        (c1 > 0) & (c2 > 0) & (s1 > s2), # C
        (c1 > 0) & (c2 > 0),             # A/F
        (c1 > 0) & (c2 < 0),             # B
        (c1 < 0) & (c2 < 0),             # D
        (c1 < 0) & (c2 > 0)              # E
    ])

    outcomes = [
        lambda: 2 * G_complete_circ(r, z) - G_circ(r, z, s2, s1),
        lambda: G_circ(r, z, s1, s2),
        lambda: G_circ(r, z, s1, one) + G_circ(r, z, s2, one),
        lambda: G_circ(r, z, s2, s1),
        lambda: G_circ(r, z, -one, s1) + G_circ(r, z, -one, s2)
    ]

    ind = jnp.argwhere(conditions, size=1).squeeze()
    return jax.lax.switch(ind, outcomes)