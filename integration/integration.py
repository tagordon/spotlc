import jax 
import jax.numpy as jnp
import constant as c
import linear as l
import quadratic as q

# note that no provision is made for phi = +/- pi 
# shouldn't occur unless user inputs that number 
# directly. So don't do it, user. 

@jax.jit
def ellipse_arc(ld_params, a, b, z, lims):

    uc = 1 - ld_params[0] - 2 * ld_params[1]
    ul = ld_params[0] + 2 * ld_params[1]
    uq = ld_params[1]

    terms = ellipse(a, b, z, lims[0], lims[1])
    return terms[0] * uc + terms[1] * ul + terms[2] * uq

@jax.jit
def ellipse_full(ld_params, a, b, z):

    uc = 1 - ld_params[0] - 2 * ld_params[1]
    ul = ld_params[0] + 2 * ld_params[1]
    uq = ld_params[1]

    terms = c.G_comp(a, b, z), 2 * l.G_comp(a, b, z), q.G_comp(a, b, z)
    return terms[0] * uc + terms[1] * ul + terms[2] * uq

ellipse_arc_batch = jax.vmap(ellipse_arc, in_axes=(None, 0, 0, 0, 1))
ellipse_full_batch = jax.vmap(ellipse_full, in_axes=(None, 0, 0, 0))

@jax.jit
def circle_arc(ld_params, r, z, lims):

    uc = 1 - ld_params[0] - 2 * ld_params[1]
    ul = ld_params[0] + 2 * ld_params[1]
    uq = ld_params[1]

    terms = circle(r, z, lims[0], lims[1])
    return terms[0] * uc + terms[1] * ul + terms[2] * uq

@jax.jit
def circle_full(ld_params, r, z, lims):

    uc = 1 - ld_params[0] - 2 * ld_params[1]
    ul = ld_params[0] + 2 * ld_params[1]
    uq = ld_params[1]

    terms = c.Gc_comp(r, z), 2 * l.Gc_comp(r, z), q.Gc_comp(r, z)
    return terms[0] * uc + terms[1] * ul + terms[2] * uq

circle_arc_batch = jax.vmap(circle_arc, in_axes=(None, 0, 0, 0, 0))
circle_full_batch = jax.vmap(circle_full, in_axes=(None, 0, 0))

#def ellipse_terms(a, b, z, phi1, phi2):
#
#    return jax.lax.cond(
#        jnp.abs(phi2 - phi1 - 2 * jnp.pi) < 1e-6,
#        lambda: (c.G_comp(a, b, z), 2 * l.G_comp(a, b, z), q.G_comp(a, b, z)),
#        lambda: ellipse(a, b, z, phi1, phi2)
#    )

#def circle_terms(r, z, phi1, phi2):
#
#    return jax.lax.cond(
#        jnp.abs(phi2 - phi1 - 2 * jnp.pi) < 1e-6,
#        lambda: c.Gc_comp(r, z), 2 * l.Gc_comp(r, z), q.Gc_comp(r, z),
#        lambda: circle(r, z, phi1, phi2)
#    )

# this function assumes that both limits are in (0, 2*pi) 
def ellipse(a, b, z, phi1, phi2):

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
        lambda: (
            c.G(a, b, z, phi1, phi2),
            2 * l.G_comp(a, b, z) - l.G(a, b, z, s2, s1),
            q.G(a, b, z, phi1, phi2)
        ),
        lambda: (
            c.G(a, b, z, phi1, phi2),
            l.G(a, b, z, s1, s2),
            q.G(a, b, z, phi1, phi2)
        ),
        lambda: (
            c.G(a, b, z, phi1, phi2),
            l.G_x1(a, b, z, s1) + l.G_x1(a, b, z, s2),
            q.G(a, b, z, phi1, phi2),
        ),
        lambda: (
            c.G(a, b, z, phi1, phi2),
            l.G(a, b, z, s2, s1),
            q.G(a, b, z, phi1, phi2),
        ),
        lambda: (
            c.G(a, b, z, phi1, phi2),
            l.G_ym1(a, b, z, s1) + l.G_ym1(a, b, z, s2),
            q.G(a, b, z, phi1, phi2)
        )
    ]

    ind = jnp.argwhere(conditions, size=1).squeeze()
    return jax.lax.switch(ind, outcomes)

# this function assumes that both limits are in (0, 2*pi) 
def circle(r, z, phi1, phi2):

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
        lambda: (
            c.Gc(a, b, z, phi1, phi2),
            2 * l.Gc_comp(a, b, z) - l.Gc(a, b, z, s2, s1),
            q.Gc(a, b, z, phi1, phi2)
        ),
        lambda: (
            c.Gc(a, b, z, phi1, phi2),
            l.Gc(a, b, z, s1, s2),
            q.Gc(a, b, z, phi1, phi2)
        ),
        lambda: (
            c.Gc(a, b, z, phi1, phi2),
            l.Gc_x1(a, b, z, s1) + l.Gc_x1(a, b, z, s2),
            q.Gc(a, b, z, phi1, phi2),
        ),
        lambda: (
            c.Gc(a, b, z, phi1, phi2),
            l.Gc(a, b, z, s2, s1),
            q.Gc(a, b, z, phi1, phi2),
        ),
        lambda: (
            c.Gc(a, b, z, phi1, phi2),
            l.Gc_ym1(a, b, z, s1) + l.Gc_ym1(a, b, z, s2),
            q.Gc(a, b, z, phi1, phi12)
        )
    ]

    ind = jnp.argwhere(conditions, size=1).squeeze()
    return jax.lax.switch(ind, outcomes)