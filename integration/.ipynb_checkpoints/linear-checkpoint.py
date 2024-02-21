import jax
from jax import numpy as jnp
from integrals import * 
from integrals_halfcomp import *

trap_order = 10

# for the complete integral we just use the 
# trapezoidal rule, which converges exponentially 
# for periodic functions 
@jax.jit
def Gc_comp(r, z):

    def ig(t):

        s = jnp.sin(t)
        x = r**2 + z**2 + 2 * r * z * s
        return r * (r + s * z) * (
            1 - (1 - x) ** 1.5
        ) / (3 * x)

    ni = jnp.arange(trap_order)
    twopi_n = 2 * jnp.pi / trap_order
    
    return twopi_n * jnp.sum(ig(twopi_n * ni)) * 0.5

@jax.jit
def G_comp(a, b, z):

    def ig(t):

        s = jnp.sin(t)
        x = (a * jnp.cos(t))**2 + (z + b * s)**2
        return a * (b + s * z) * (
            1 - (1 - x) ** 1.5
        ) / (3 * x)
    
    ni = jnp.arange(trap_order)
    twopi_n = 2 * jnp.pi / trap_order
    
    return twopi_n * jnp.sum(ig(twopi_n * ni)) * 0.5

# full integral for linear term -- circular case 
@jax.jit 
def Gc(r, z, y, x):

    ig1 = g1_circ(r, z, y, x)
    ig2 = g2_circ(r, z, y, x)
    ig3 = g3_circ(r, z, y, x)
    ig4 = g4_circ(r, z, y, x)

    return ig1 + ig2 + ig3 + ig4

# full integral for linear term -- elliptical case  
@jax.jit
def G(a, b, z, y, x):

    c = jnp.sqrt(a * a - b * b)
    oc2 = 1 / (c * c)
    f = (1 - a * a - z * z) * oc2
    g = -2 * b * z * oc2
    sqcz = jnp.sqrt(c * c + z * z)
    dp = (b * z + a * sqcz) * oc2
    dm = (b * z - a * sqcz) * oc2

    ig1 = g1(a, b, c, z, y, x, dp, dm)
    ig2 = g2(a, b, c, z, y, x, f, g, dp, dm)
    ig3 = g3(a, b, c, z, y, x, f, g, dp, dm)
    ig4 = g4(a, b, c, z, y, x, f, g, dp, dm)
    ig5 = g5(a, b, c, z, y, x, f, g, dp, dm)
    ig6 = g6(a, b, c, z, y, x, f, g, dp, dm)

    return ig1 + ig2 + ig3 + ig4 + ig5 + ig6

def g1_circ(r, z, y, x):

    fac = r / (6 * (r + z) * (r - z))
    fx = jnp.arctan((x + z / r) / jnp.sqrt(1 - x * x))
    fy = jnp.arctan((y + z / r) / jnp.sqrt(1 - y * y))
    return (fx - fy) / 3

def g2_circ(r, z, y, x):

    fac = r * (1 + r + z) * (r + z - 1) * jnp.sqrt(r * z / 2) / (3 * (r + z))
    return fac * int23_circ(r, z, y, x, 1, -1)

def g3_circ(r, z, y, x):

    a4 = (r * r + z * z) / (2 * r * z)
    fac = 2 * r * r * z * z / jnp.sqrt(2 * r * z) / (3 * (r + z) * (r - z))
    return fac * int23_circ(r, z, y, x, a4, 1)

def g4_circ(r, z, y, x):

    fac = r * (r - z + 1) * (r - z - 1) / (6 * (r - z)) * jnp.sqrt(2 * r * z)
    return fac * int4_circ(r, z, y, x)


def g1(a, b, c, z, y, x, dp, dm):

    c2 = c * c
    fac = - a / (3 * c2 * (dm * dm - 1) * (dm - dp) * (dp * dp - 1))
    v = (dm - 1) * (dm - dp) * (dp - 1) * (b - z)

    sm = jnp.sign(dm - 1)
    sp = jnp.sign(dp - 1)
    
    u1 = 2 * sm * jnp.sqrt((dm - 1) * (dm + 1)) * (dp * dp - 1) * (b + dm * z)
    u2 = 2 * sp * jnp.sqrt((dp - 1) * (dp + 1)) * (dm * dm - 1) * (b + dp * z)

    def f(t):

        q = jnp.sqrt((1 + t) / (1 - t))
        return (
            -v * (jnp.arctan(q) + jnp.arctan(1 / q)) 
            - u1 * jnp.arctan(jnp.sqrt((dm + 1) * (1 - t) / ((dm - 1) * (1 + t))))
            + u2 * jnp.arctan(jnp.sqrt((dp + 1) * (1 - t) / ((dp - 1) * (1 + t))))
        )

    return fac * (f(y) - f(x))
    
def g2(a, b, c, z, y, x, f, g, dp, dm):

    fac = a * c * z / 6 + a * (b + z) / (6 * c * (dm - 1) * (dp - 1))
    return fac * int2(y, x, f, g)

def g3(a, b, c, z, y, x, f, g, dp, dm):

    fac = -a * c * z / 6 + a * (b - z) / (6 * c * (dm + 1) * (dp + 1))
    return fac * int3(y, x, f, g)

def g4(a, b, c, z, y, x, f, g, dp, dm):

    fac = a * b * c / 3
    return fac * int4(y, x, f, g)

def g5(a, b, c, z, y, x, f, g, dp, dm):

    fac = a * (b + z * dm) / (3 * c * (dm * dm - 1) * (dm - dp))
    return fac * int5(y, x, f, g, dm)

def g6(a, b, c, z, y, x, f, g, dp, dm):

    fac = a * (b + z * dp) / (3 * c * (dp * dp - 1) * (dm - dp))
    return fac * int6(y, x, f, g, dp)

# half-complete integrals

@jax.jit 
def Gc_ym1(r, z, x):

    ig1 = g1_circ_ym1(r, z, x)
    ig2 = g2_circ_ym1(r, z, x)
    ig3 = g3_circ_ym1(r, z, x)
    ig4 = g4_circ_ym1(r, z, x)

    return ig1 + ig2 + ig3 + ig4

@jax.jit 
def Gc_x1(r, z, y):

    ig1 = g1_circ_x1(r, z, y)
    ig2 = g2_circ_x1(r, z, y)
    ig3 = g3_circ_x1(r, z, y)
    ig4 = g4_circ_x1(r, z, y)

    return ig1 + ig2 + ig3 + ig4

@jax.jit
def G_ym1(a, b, z, x):

    c = jnp.sqrt(a * a - b * b)
    oc2 = 1 / (c * c)
    f = (1 - a * a - z * z) * oc2
    g = -2 * b * z * oc2
    sqcz = jnp.sqrt(c * c + z * z)
    dp = (b * z + a * sqcz) * oc2
    dm = (b * z - a * sqcz) * oc2

    ig1 = g1_ym1(a, b, c, z, x, dp, dm)
    ig2 = g2_ym1(a, b, c, z, x, f, g, dp, dm)
    ig3 = g3_ym1(a, b, c, z, x, f, g, dp, dm)
    ig4 = g4_ym1(a, b, c, z, x, f, g, dp, dm)
    ig5 = g5_ym1(a, b, c, z, x, f, g, dp, dm)
    ig6 = g6_ym1(a, b, c, z, x, f, g, dp, dm)

    return ig1 + ig2 + ig3 + ig4 + ig5 + ig6

@jax.jit
def G_x1(a, b, z, y):

    c = jnp.sqrt(a * a - b * b)
    oc2 = 1 / (c * c)
    f = (1 - a * a - z * z) * oc2
    g = -2 * b * z * oc2
    sqcz = jnp.sqrt(c * c + z * z)
    dp = (b * z + a * sqcz) * oc2
    dm = (b * z - a * sqcz) * oc2

    ig1 = g1_x1(a, b, c, z, y, dp, dm)
    ig2 = g2_x1(a, b, c, z, y, f, g, dp, dm)
    ig3 = g3_x1(a, b, c, z, y, f, g, dp, dm)
    ig4 = g4_x1(a, b, c, z, y, f, g, dp, dm)
    ig5 = g5_x1(a, b, c, z, y, f, g, dp, dm)
    ig6 = g6_x1(a, b, c, z, y, f, g, dp, dm)

    return ig1 + ig2 + ig3 + ig4 + ig5 + ig6

def g1_circ_x1(r, z, y):

    fac = r / (6 * (r + z) * (r - z))
    fx = jnp.pi / 2
    fy = jnp.arctan((y + z / r) / jnp.sqrt(1 - y * y))
    return (fx - fy) / 3

def g1_circ_ym1(r, z, x):

    fac = r / (6 * (r + z) * (r - z))
    fx = jnp.arctan((x + z / r) / jnp.sqrt(1 - x * x))
    fy = jnp.pi / 2
    return (fx - fy) / 3

def g2_circ_ym1(r, z, x):

    fac = r * (1 + r + z) * (r + z - 1) * jnp.sqrt(r * z / 2) / (3 * (r + z))
    return fac * int23_circ_ym1(r, z, -1.0, x, 1, -1)

def g2_circ_x1(r, z, y):

    fac = r * (1 + r + z) * (r + z - 1) * jnp.sqrt(r * z / 2) / (3 * (r + z))
    return fac * int2_circ_x1(r, z, y, 1.0)

def g3_circ_ym1(r, z, x):

    a4 = (r * r + z * z) / (2 * r * z)
    fac = 2 * r * r * z * z / jnp.sqrt(2 * r * z) / (3 * (r + z) * (r - z))
    return fac * int23_circ_ym1(r, z, -1.0, x, a4, 1)

def g3_circ_x1(r, z, y):

    a4 = (r * r + z * z) / (2 * r * z)
    fac = 2 * r * r * z * z / jnp.sqrt(2 * r * z) / (3 * (r + z) * (r - z))
    return fac * int3_circ_x1(r, z, y, 1.0, a4)

def g4_circ_xm1(r, z, y, x):

    fac = r * (r - z + 1) * (r - z - 1) / (6 * (r - z)) * jnp.sqrt(2 * r * z)
    return fac * int4_circ_halfcomp(r, z, y, 1.0)

def g4_circ_ym1(r, z, y, x):

    fac = r * (r - z + 1) * (r - z - 1) / (6 * (r - z)) * jnp.sqrt(2 * r * z)
    return fac * int4_circ_halfcomp(r, z, -1.0, x)

def g1_x1(a, b, c, z, y, dp, dm):

    c2 = c * c
    fac = - a / (3 * c2 * (dm * dm - 1) * (dm - dp) * (dp * dp - 1))
    v = (dm - 1) * (dm - dp) * (dp - 1) * (b - z)

    sm = jnp.sign(dm - 1)
    sp = jnp.sign(dp - 1)
    
    u1 = 2 * sm * jnp.sqrt((dm - 1) * (dm + 1)) * (dp * dp - 1) * (b + dm * z)
    u2 = 2 * sp * jnp.sqrt((dp - 1) * (dp + 1)) * (dm * dm - 1) * (b + dp * z)

    def f(t):

        q = jnp.sqrt((1 + t) / (1 - t))
        return (
            -v * (jnp.arctan(q) + jnp.arctan(1 / q)) 
            - u1 * jnp.arctan(jnp.sqrt((dm + 1) * (1 - t) / ((dm - 1) * (1 + t))))
            + u2 * jnp.arctan(jnp.sqrt((dp + 1) * (1 - t) / ((dp - 1) * (1 + t))))
        )

    fx = -v * jnp.pi / 2

    return fac * (f(y) - fx)

def g1_ym1(a, b, c, z, x, dp, dm):

    c2 = c * c
    fac = - a / (3 * c2 * (dm * dm - 1) * (dm - dp) * (dp * dp - 1))
    v = (dm - 1) * (dm - dp) * (dp - 1) * (b - z)

    sm = jnp.sign(dm - 1)
    sp = jnp.sign(dp - 1)
    
    u1 = 2 * sm * jnp.sqrt((dm - 1) * (dm + 1)) * (dp * dp - 1) * (b + dm * z)
    u2 = 2 * sp * jnp.sqrt((dp - 1) * (dp + 1)) * (dm * dm - 1) * (b + dp * z)

    def f(t):

        q = jnp.sqrt((1 + t) / (1 - t))
        return (
            -v * (jnp.arctan(q) + jnp.arctan(1 / q)) 
            - u1 * jnp.arctan(jnp.sqrt((dm + 1) * (1 - t) / ((dm - 1) * (1 + t))))
            + u2 * jnp.arctan(jnp.sqrt((dp + 1) * (1 - t) / ((dp - 1) * (1 + t))))
        )

    fy = -v * jnp.pi / 2 - u1 * jnp.pi / 2 + u2 * jnp.pi / 2

    return fac * (fy - f(x))
    
def g2_ym1(a, b, c, z, x, f, g, dp, dm):

    fac = a * c * z / 6 + a * (b + z) / (6 * c * (dm - 1) * (dp - 1))
    return fac * int2_ym1(x, f, g)

def g2_x1(a, b, c, z, y, f, g, dp, dm):

    fac = a * c * z / 6 + a * (b + z) / (6 * c * (dm - 1) * (dp - 1))
    return fac * int2_x1(y, f, g)

def g3_ym1(a, b, c, z, x, f, g, dp, dm):

    fac = -a * c * z / 6 + a * (b - z) / (6 * c * (dm + 1) * (dp + 1))
    return fac * int3_ym1(x, f, g)

def g3_x1(a, b, c, z, y, f, g, dp, dm):

    fac = -a * c * z / 6 + a * (b - z) / (6 * c * (dm + 1) * (dp + 1))
    return fac * int3_x1(y, f, g)

def g4_ym1(a, b, c, z, x, f, g, dp, dm):

    fac = a * b * c / 3
    return fac * int4_ym1(x, f, g)

def g4_x1(a, b, c, z, y, f, g, dp, dm):

    fac = a * b * c / 3
    return fac * int4_x1(y, f, g)

def g5_ym1(a, b, c, z, x, f, g, dp, dm):

    fac = a * (b + z * dm) / (3 * c * (dm * dm - 1) * (dm - dp))
    return fac * int5_ym1(x, f, g, dm)

def g5_x1(a, b, c, z, y, f, g, dp, dm):

    fac = a * (b + z * dm) / (3 * c * (dm * dm - 1) * (dm - dp))
    return fac * int5_x1(y, f, g, dm)

def g6_ym1(a, b, c, z, x, f, g, dp, dm):

    fac = a * (b + z * dp) / (3 * c * (dp * dp - 1) * (dm - dp))
    return fac * int6_ym1(x, f, g, dp)

def g6_x1(a, b, c, z, y, f, g, dp, dm):

    fac = a * (b + z * dp) / (3 * c * (dp * dp - 1) * (dm - dp))
    return fac * int6_x1(y, f, g, dp)