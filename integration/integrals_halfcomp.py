# these functions integrate the terms in the 
# partial fraction decomposition of the integrand 
# for the line integral along the arc of the ellipse 
# in the linear term of the limb-darkening profile. 

# these functions are for the case where one of the limits 
# of integration is the branch point. 

import jax
from jax import numpy as jnp
from carlson import rf, rj_posp, rc_posy, rd

def int23_circ_ym1(r, z, y, x, a4, b4):

    a3 = (1 - (r**2 + z**2)) / (2 * r * z)
    
    d12 = -2
    d13 = -1 - a3
    d14 = b4 - a4
    d24 = b4 + a4
    d34 = a3 * b4 + a4

    r24 = -d24 / b4
    r14 = d14 / b4
    r34 = - d34 / b4

    x1 = jnp.sqrt(1 + x)
    x2 = jnp.sqrt(1 - x)
    x3 = jnp.sqrt(a3 - x)
    x4 = jnp.sqrt(a4 + b4 * x)

    y1 = jnp.sqrt(1 + y)
    y2 = jnp.sqrt(1 - y)
    y3 = jnp.sqrt(a3 - y)
    y4 = jnp.sqrt(a4 + b4 * y)

    U1 = (x1 * y2 * y3 + y1 * x2 * x3) / (x - y)
    U2 = (x2 * y1 * y3 + y2 * x1 * x3) / (x - y)
    U3 = (x3 * y1 * y2 + y3 * x1 * x2) / (x - y)

    W2 = U1 * U1 - b4 * d12 * d13 / d14

    I1c = 2 * rf(U3 * U3, U2 * U2, U1 * U1)
    I2c = (2 / 3) * d12 * d13 * rd(U3 * U3, U2 * U2, U1 * U1) + 2 * x1 * y1 / U1
    I3c = -2 * d12 * d13 / (3 * d14) * rj_posp(U3 * U3, U2 * U2, U1 * U1, W2)
    A111 = x1 * x2 * x3 - y1 * y2 * y3
    J1c = d12 * d13 * I1c - 2 * A111

    return (
        - 3 * r24 * d34 * I3c 
        + (r14 + r24 + r34) * I2c 
        - J1c
    ) / (3 * b4)

def int3_circ_x1(r, z, y, x, a4):

    a3 = (1 - (r**2 + z**2)) / (2 * r * z)
    
    d12 = -2
    d13 = -1 - a3
    d14 = 1 - a4
    d24 = 1 + a4
    d34 = a3 * 1 + a4

    r24 = -d24
    r14 = d14
    r34 = - d34

    x1 = jnp.sqrt(1 + x)
    x2 = jnp.sqrt(1 - x)
    x3 = jnp.sqrt(a3 - x)
    x4 = jnp.sqrt(a4 +  x)

    y1 = jnp.sqrt(1 + y)
    y2 = jnp.sqrt(1 - y)
    y3 = jnp.sqrt(a3 - y)
    y4 = jnp.sqrt(a4 + y)

    U1 = (x1 * y2 * y3 + y1 * x2 * x3) / (x - y)
    U2 = (x2 * y1 * y3 + y2 * x1 * x3) / (x - y)
    U3 = (x3 * y1 * y2 + y3 * x1 * x2) / (x - y)

    W2 = U1 * U1 - d12 * d13 / d14
    Q2 = (x4 * y4 / (x1 * y1))**2 * W2
    P2 = Q2 + d24 * d34 / d14

    I1c = 2 * rf(U3 * U3, U2 * U2, U1 * U1)
    I2c = (2 / 3) * d12 * d13 * rd(U3 * U3, U2 * U2, U1 * U1) + 2 * x1 * y1 / U1
    I3c = -2 * d12 * d13 / (3 * d14) * rj_posp(U3 * U3, U2 * U2, U1 * U1, W2) + 2 * rc_posy(P2, Q2)

    A111 = x1 * x2 * x3 - y1 * y2 * y3
    J1c = d12 * d13 * I1c - 2 * A111

    return (
        - 3 * r24 * d34 * I3c 
        + (r14 + r24 + r34) * I2c 
        - J1c
    ) / 3

def int2_circ_x1(r, z, y, x):

    a3 = (1 - (r**2 + z**2)) / (2 * r * z)
    
    d12 = -2
    d13 = -1 - a3
    d14 = -2
    d34 = -a3 + 1

    r14 = -d14
    r34 = d34

    x1 = jnp.sqrt(1 + x)
    x2 = jnp.sqrt(1 - x)
    x3 = jnp.sqrt(a3 - x)
    x4 = jnp.sqrt(1 - x)

    y1 = jnp.sqrt(1 + y)
    y2 = jnp.sqrt(1 - y)
    y3 = jnp.sqrt(a3 - y)
    y4 = jnp.sqrt(1 - y)

    U1 = (x1 * y2 * y3 + y1 * x2 * x3) / (x - y)
    U2 = (x2 * y1 * y3 + y2 * x1 * x3) / (x - y)
    U3 = (x3 * y1 * y2 + y3 * x1 * x2) / (x - y)

    I1c = 2 * rf(U3 * U3, U2 * U2, U1 * U1)
    I2c = (2 / 3) * d12 * d13 * rd(U3 * U3, U2 * U2, U1 * U1) + 2 * x1 * y1 / U1
    
    A111 = x1 * x2 * x3 - y1 * y2 * y3
    J1c = d12 * d13 * I1c - 2 * A111

    return (J1c - (r14 + r34) * I2c) / 3

def int4_circ(r, z, y, x):

    a2 = (1 - (r**2 + z**2)) / (2 * r * z)
    
    d12 = a2 - 1
    d13 = 2
    d23 = a2 + 1

    x1 = jnp.sqrt(1 - x)
    x2 = jnp.sqrt(a2 - x)
    x3 = jnp.sqrt(1 + x)

    y1 = jnp.sqrt(1 - y)
    y2 = jnp.sqrt(a2 - y)
    y3 = jnp.sqrt(1 + y)

    U1 = (x1 * y2 * y3 + y1 * x2 * x3) / (x - y)
    U2 = (x2 * y1 * y3 + y2 * x1 * x3) / (x - y)
    U3 = (x3 * y1 * y2 + y3 * x1 * x2) / (x - y)

    I1c = 2 * rf(U3 * U3, U2 * U2, U1 * U1)
    I2c = (2 / 3) * d12 * d13 * rd(U3 * U3, U2 * U2, U1 * U1) + 2 * x1 * y1 / U1

    A111 = x1 * x2 * x3 - y1 * y2 * y3
    J1c = d12 * d13 * I1c + 2 * A111

    return ((d23 + d13) * I2c + J1c) / 3

def g1_circ_x1(r, z, y):

    fac = r / (6 * (r + z) * (r - z))
    fx = jnp.pi / 2
    fy = jnp.arctan((y + z / r) / jnp.sqrt(1 - y * y))
    return (fx - fy) / 3

# sign of fy (and fx above) should check if the value of G_circ comes out wrong 
def g1_circ_ym1(r, z, x):

    fac = r / (6 * (r + z) * (r - z))
    fx = jnp.arctan((x + z / r) / jnp.sqrt(1 - x * x))
    fy = jnp.pi / 2
    return (fx - fy) / 3

def g4_circ(r, z, y, x):

    fac = r * (r - z + 1) * (r - z - 1) / (6 * (r - z)) * jnp.sqrt(2 * r * z)
    return fac * int4_circ(r, z, y, x)

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

@jax.jit 
def G_circ_ym1(r, z, x):

    ig1 = g1_circ_ym1(r, z, x)
    ig2 = g2_circ_ym1(r, z, x)
    ig3 = g3_circ_ym1(r, z, x)
    ig4 = g4_circ(r, z, -1.0, x)

    return ig1 + ig2 + ig3 + ig4

@jax.jit 
def G_circ_x1(r, z, y):

    ig1 = g1_circ_x1(r, z, y)
    ig2 = g2_circ_x1(r, z, y)
    ig3 = g3_circ_x1(r, z, y)
    ig4 = g4_circ(r, z, y, 1.0)

    return ig1 + ig2 + ig3 + ig4

# this is the full integral along the arc
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

# this is the full integral along the arc
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

def int23_A(y, x, f, g, b1, b4):

    x1 = jnp.sqrt(1 + b1 * x)
    x4 = jnp.sqrt(1 + b4 * x)

    y1 = jnp.sqrt(1 + b1 * y)
    y4 = jnp.sqrt(1 + b4 * y)

    xi = jnp.sqrt(f + g * x + x * x)
    eta = jnp.sqrt(f + g * y + y * y)

    del2 = 4 * f - g * g
    bet4 = g * b4 - 2
    bet1 = g * b1 - 2

    c1sq = 2 * f - 2 * g * b1 + 2
    c4sq = 2 * f - 2 * g * b4 + 2
    c14sq = -2 * f + 2
    c1c4 = jnp.sqrt(c1sq * c4sq)

    d14 = b4 - b1
    r14 = -d14

    A111m1 = x1 * xi / x4 - y1 * eta / y4
    A1111 = x1 * xi * x4 - y1 * eta * y4

    c3 = (r14 * r14 - c1sq + del2 + c4sq) / 8
    s = (r14 + bet4 / b4) / 8
    c2 = c4sq / (2 * b4)
    c1 = c1sq / (16 * b1) * (1 / b1 + 1 / b4 - g)

    M2 = (x1 * y4 + y1 * x4)**2 * (
        2 * xi * eta + 2 * f + g * (x + y) + 2 * x * y
    ) / (x - y)**2
    
    Lp2 = M2 + c14sq + c1c4
    Lm2 = M2 + c14sq - c1c4
    U = (x1 * x4 * eta + y1 * y4 * xi) / (x - y)
    U2 = U * U
    W2 = U2 + 0.5 * c1sq
    Q2 = W2 / (x1 * y1)**2
    P2 = Q2 - 1
    rho = d14 * (bet1 - jnp.sqrt(2 * c1sq)) / b1
    erf = rf(M2, Lm2, Lp2)
    erj = rj_posp(M2, Lm2, Lp2, M2 + rho)
    erd = rd(M2, Lm2, Lp2)
    rcuw = rc_posy(U2, W2)
    rcpq = rc_posy(P2, Q2)

    I1 = 4 * erf
    I2 = (
        (2 / 3) * jnp.sqrt(c1sq / c4sq)
        * (4 * (c14sq + c1c4) * erd 
           - 6 * erf + 3 / U) 
    )
    I3p = (
        jnp.sqrt(2 * c1sq / 9) 
        * (4 * rho * erj 
           - 6 * erf + 3 * rcuw) 
        + 2 * rcpq
    )

    return (
        c3 * I3p 
        + s * (c2 * I2 - 2 * (x4 * xi / x1 - y4 * eta / y1)) 
        + c1 * I1 + A1111 / (2 * b4)
    )

def int23_B(y, x, f, g, b1, b4):

    x1 = jnp.sqrt(1 + b1 * x)
    x4 = jnp.sqrt(1 + b4 * x)

    y1 = jnp.sqrt(1 + b1 * y)
    y4 = jnp.sqrt(1 + b4 * y)

    xi = jnp.sqrt(f + g * x + x * x)
    eta = jnp.sqrt(f + g * y + y * y)

    del2 = 4 * f - g * g
    bet4 = g * b4 - 2
    bet1 = g * b1 - 2

    c1sq = 2 * f - 2 * g * b1 + 2
    c4sq = 2 * f - 2 * g * b4 + 2
    c14sq = -2 * f + 2
    c1c4 = jnp.sqrt(c1sq * c4sq)

    d14 = b4 - b1
    r14 = -d14

    A111m1 = x1 * xi / x4 - y1 * eta / y4
    A1111 = x1 * xi * x4 - y1 * eta * y4

    c3 = (r14 * r14 - c1sq + del2 + c4sq) / 8
    s = (r14 + bet4 / b4) / 8
    c2 = c4sq / (2 * b4)
    c1 = c1sq / (16 * b1) * (1 / b1 + 1 / b4 - g)

    M2 = (x1 * y4 + y1 * x4)**2 * (
        2 * xi * eta + 2 * f + g * (x + y) + 2 * x * y
    ) / (x - y)**2
    
    Lp2 = M2 + c14sq + c1c4
    Lm2 = M2 + c14sq - c1c4
    U = (x1 * x4 * eta + y1 * y4 * xi) / (x - y)
    U2 = U * U
    W2 = U2 + 0.5 * c1sq
    rho = d14 * (bet1 - jnp.sqrt(2 * c1sq)) / b1
    erf = rf(M2, Lm2, Lp2)
    erj = rj_posp(M2, Lm2, Lp2, M2 + rho)
    erd = rd(M2, Lm2, Lp2)
    rcuw = rc_posy(U2, W2)

    I1 = 4 * erf
    I2 = (
        (2 / 3) * jnp.sqrt(c1sq / c4sq)
        * (4 * (c14sq + c1c4) * erd 
           - 6 * erf + 3 / U) 
        + 2 * x1 * y1 / (x4 * y4 * U)
    )
    I3p = (
        jnp.sqrt(2 * c1sq / 9) 
        * (4 * rho * erj 
           - 6 * erf + 3 * rcuw) 
        #+ 2 * rcpq
    )

    return (
        c3 * I3p 
        + s * (c2 * I2 + 2 * A111m1) 
        + c1 * I1 + A1111 / (2 * b4)
    )
    
int2_ym1 = jax.jit(
    lambda x, f, g: 
    int23_B(-1.0, x, f, g, 1, -1)
)

int2_x1 = jax.jit(
    lambda y, f, g: 
    int23_A(y, 1.0, f, g, 1, -1)
)

int3_ym1 = jax.jit(
    lambda x, f, g: 
    int23_A(-1.0, x, f, g, -1, 1)
)

int3_x1 = jax.jit(
    lambda y, f, g: 
    int23_B(y, 1.0, f, g, -1, 1)
)

# y = -1
def int4_ym1(x, f, g):

    x1 = jnp.sqrt(1 + x)
    x4 = jnp.sqrt(1 - x)

    y1 = 0.0
    y4 = jnp.sqrt(2.0)

    xi = jnp.sqrt(f + g * x + x * x)
    eta = jnp.sqrt(f - g + 1)

    del2 = 4 * f - g * g
    bet4 = -g - 2
    bet1 = g - 2

    c1sq = 2 * f - 2 * g + 2
    c4sq = 2 * f + 2 * g + 2
    c14sq = -2 * f + 2
    c1c4 = jnp.sqrt(c1sq * c4sq)

    d14 = -2
    r14 = 2

    A111m1 = x1 * xi / x4 - y1 * eta / y4

    M2 = (x1 * y4 + y1 * x4)**2 * (
        2 * xi * eta + 2 * f + g * (x - 1.0) - 2 * x
    ) / (x + 1.0)**2
    
    Lp2 = M2 + c14sq + c1c4
    Lm2 = M2 + c14sq - c1c4
    U = (x1 * x4 * eta + y1 * y4 * xi) / (x + 1.0)
    U2 = U * U
    W2 = U2 + 0.5 * c1sq
    rho = d14 * (bet1 - jnp.sqrt(2 * c1sq))
    erf = rf(M2, Lm2, Lp2)
    erj = rj_posp(M2, Lm2, Lp2, M2 + rho)
    erd = rd(M2, Lm2, Lp2)
    rcuw = rc_posy(U2, W2)

    I1 = 4 * erf
    I2 = (
        (2 / 3) * jnp.sqrt(c1sq / c4sq)
        * (4 * (c14sq + c1c4) * erd - 6 * erf + 3 / U) 
        + 2 * x1 * y1 / (x4 * y4 * U)
    )
    I3p = (
        jnp.sqrt(2 * c1sq / 9) 
        * (4 * rho * erj - 6 * erf + 3 * rcuw) 
    )

    return -0.25 * (c4sq * I2 - c1sq * I1 - 2 * g * I3p) + A111m1

# x = 1
def int4_x1(y, f, g):

    x1 = jnp.sqrt(2.0)
    x4 = 0.0

    y1 = jnp.sqrt(1 + y)
    y4 = jnp.sqrt(1 - y)

    xi = jnp.sqrt(f + g + 1.0)
    eta = jnp.sqrt(f + g * y + y * y)

    del2 = 4 * f - g * g
    bet4 = -g - 2
    bet1 = g - 2

    c1sq = 2 * f - 2 * g + 2
    c4sq = 2 * f + 2 * g + 2
    c14sq = -2 * f + 2
    c1c4 = jnp.sqrt(c1sq * c4sq)

    d14 = -2
    r14 = 2

    A111m1 = x1 * xi / x4 - y1 * eta / y4

    M2 = (x1 * y4 + y1 * x4)**2 * (
        2 * xi * eta + 2 * f + g * (1.0 + y) + 2 * y
    ) / (1.0 - y)**2
    
    Lp2 = M2 + c14sq + c1c4
    Lm2 = M2 + c14sq - c1c4
    U = (x1 * x4 * eta + y1 * y4 * xi) / (1.0 - y)
    U2 = U * U
    W2 = U2 + 0.5 * c1sq
    Q2 = W2 / (x1 * y1)**2
    P2 = Q2 - 1
    rho = d14 * (bet1 - jnp.sqrt(2 * c1sq))
    erf = rf(M2, Lm2, Lp2)
    erj = rj_posp(M2, Lm2, Lp2, M2 + rho)
    erd = rd(M2, Lm2, Lp2)
    rcuw = rc_posy(U2, W2)
    rcpq = rc_posy(P2, Q2)

    I1 = 4 * erf
    I2 = (
        (2 / 3) * jnp.sqrt(c1sq / c4sq)
        * (4 * (c14sq + c1c4) * erd - 6 * erf + 3 / U) 
    )
    I3p = (
        jnp.sqrt(2 * c1sq / 9) 
        * (4 * rho * erj - 6 * erf + 3 * rcuw) 
        + 2 * rcpq
    )

    return (
        -0.25 * (c4sq * I2 - c1sq * I1 - 2 * g * I3p) 
        + y4 * eta / y1
    )

# x = 1 case 
def int56_x1(y, f, g, a5, b5):
    
    x1 = 0.0
    x4 = jnp.sqrt(2.0)
    x52 = a5 + b5
    
    y1 = jnp.sqrt(1 - y)
    y4 = jnp.sqrt(1 + y)
    y52 = a5 + b5 * y

    d14 = 2
    d15 = b5 + a5
    d45 = b5 - a5

    r14 = -d14
    r15 = -d15 / b5
    r45 = d45 / b5

    alpha1 = -2 * f - g
    alpha4 = 2 * f - g
    alpha5 = 2 * f * b5 - g * a5

    beta1 = -g - 2
    beta4 = g - 2
    beta5 = g * b5 - 2 * a5

    delta2 = 4 * f - g * g
    c1sq = 2 * f + 2 * g + 2
    c4sq = 2 * f - 2 * g + 2
    c5sq = 2 * f * b5 * b5 - 2 * g * a5 * b5 + 2 * a5 * a5
    c14sq = -2 * f + 2
    c15sq = -2 * f * b5 - g * (b5 - a5) + 2 * a5
    c45sq = 2 * f * b5 - g * (b5 + a5) + 2 * a5
    c1c4 = jnp.sqrt(c1sq * c4sq)
    c1c5 = jnp.sqrt(c1sq * c5sq)

    xi = jnp.sqrt(f + g + 1.0)
    eta = jnp.sqrt(f + g * y + y * y)

    A1111 = x1 * xi * x4 - y1 * eta * y4
    A111m1 = x1 * xi / x4 - y1 * eta / y4

    M2 = (x1 * y4 + y1 * x4)**2 * ((xi + eta)**2 - (1.0 - y)**2) / (1.0 - y)**2
    Lm2 = M2 + c14sq - c1c4
    Lp2 = M2 + c14sq + c1c4
    Wp2 = M2 + d14 * (c15sq + c1c5) / d15
    U = (x1 * x4 * eta + y1 * y4 * xi) / (1.0 - y)
    U2 = U * U
    W2 = U2 - c1sq * d45 / (2 * d15)

    W12 = U2 + c1sq * 0.5
    rho = -d14 * (beta1 - jnp.sqrt(2 * c1sq))

    erf = rf(M2, Lm2, Lp2)
    I1 = 4 * erf
    I2 = (2 / 3) * jnp.sqrt(c1sq / c4sq) * (
        4 * (c14sq + c1c4) * rd(M2, Lm2, Lp2) 
        - 6 * erf 
        + 3 / U
    )# + 2 * x1 * y1 / (x4 * y4 * U)
    I3 = (2 / 3) * jnp.sqrt(c1sq / c5sq) * (
        4 * (d14 / d15) * (c15sq + c1c5) * rj_posp(M2, Lm2, Lp2, Wp2) 
        - 6 * erf 
        + 3 * rc_posy(U2, W2)
    )
    Ip3 = (1 / 3) * jnp.sqrt(2 * c1sq) * (
        4 * rho * rj_posp(M2, Lm2, Lp2, M2 + rho) 
        - 6 * erf 
        + 3 * rc_posy(U2, W12)
    )

    sigma = r15 + r45 + beta5 / b5
    tau = r15**2 + r45**2 + c5sq / b5**2 - delta2
    c3 = c5sq * d45 / (2 * b5**3)
    cp3 = (sigma**2 - 2 * tau) / (8 * b5)
    c2 = c4sq / 2
    c1 = -c1sq / (16 * b5) * (-1 + a5 / b5 - g - 3 * r45)

    return (
        c3 * I3 
        + cp3 * Ip3 
        + (sigma / (8 * b5)) * (c2 * I2 + 2 * A111m1) 
        + c1 * I1 + A1111 * 0.5 / b5
    )

# y = -1 case 
def int56_ym1(x, f, g, a5, b5):
    
    x1 = jnp.sqrt(1 - x)
    x4 = jnp.sqrt(1 + x)
    x52 = a5 + b5 * x
    
    y1 = jnp.sqrt(2.0)
    y4 = 0.0
    y52 = a5 - b5

    d14 = 2
    d15 = b5 + a5
    d45 = b5 - a5

    r14 = -d14
    r15 = -d15 / b5
    r45 = d45 / b5

    alpha1 = -2 * f - g
    alpha4 = 2 * f - g
    alpha5 = 2 * f * b5 - g * a5

    beta1 = -g - 2
    beta4 = g - 2
    beta5 = g * b5 - 2 * a5

    delta2 = 4 * f - g * g
    c1sq = 2 * f + 2 * g + 2
    c4sq = 2 * f - 2 * g + 2
    c5sq = 2 * f * b5 * b5 - 2 * g * a5 * b5 + 2 * a5 * a5
    c14sq = -2 * f + 2
    c15sq = -2 * f * b5 - g * (b5 - a5) + 2 * a5
    c45sq = 2 * f * b5 - g * (b5 + a5) + 2 * a5
    c1c4 = jnp.sqrt(c1sq * c4sq)
    c1c5 = jnp.sqrt(c1sq * c5sq)

    xi = jnp.sqrt(f + g * x + x * x)
    eta = jnp.sqrt(f - g + 1)

    A1111 = x1 * xi * x4 - y1 * eta * y4
    A111m1 = x1 * xi / x4 - y1 * eta / y4

    M2 = (x1 * y4 + y1 * x4)**2 * ((xi + eta)**2 - (x + 1.0)**2) / (x + 1.0)**2
    Lm2 = M2 + c14sq - c1c4
    Lp2 = M2 + c14sq + c1c4
    Wp2 = M2 + d14 * (c15sq + c1c5) / d15
    U = (x1 * x4 * eta + y1 * y4 * xi) / (x + 1.0)
    U2 = U * U
    W2 = U2 - c1sq * d45 / (2 * d15)
    Q2 = x52 * y52 / (x1 * y1) ** 2 * W2
    P2 = Q2 + c5sq * d45 / (2 * d15)

    W12 = U2 + c1sq * 0.5
    Q12 = W12 / (x1 * y1)**2
    P12 = Q12 - 1
    rho = -d14 * (beta1 - jnp.sqrt(2 * c1sq))

    erf = rf(M2, Lm2, Lp2)
    I1 = 4 * erf
    I2 = (2 / 3) * jnp.sqrt(c1sq / c4sq) * (
        4 * (c14sq + c1c4) * rd(M2, Lm2, Lp2) 
        - 6 * erf 
        + 3 / U
    )# + 2 * x1 * y1 / (x4 * y4 * U)
    I3 = (2 / 3) * jnp.sqrt(c1sq / c5sq) * (
        4 * (d14 / d15) * (c15sq + c1c5) * rj_posp(M2, Lm2, Lp2, Wp2) 
        - 6 * erf 
        + 3 * rc_posy(U2, W2)
    ) + 2 * rc_posy(P2, Q2)
    Ip3 = (1 / 3) * jnp.sqrt(2 * c1sq) * (
        4 * rho * rj_posp(M2, Lm2, Lp2, M2 + rho) 
        - 6 * erf 
        + 3 * rc_posy(U2, W12)
    ) + 2 * rc_posy(P12, Q12)

    sigma = r15 + r45 + beta5 / b5
    tau = r15**2 + r45**2 + c5sq / b5**2 - delta2
    c3 = c5sq * d45 / (2 * b5**3)
    cp3 = (sigma**2 - 2 * tau) / (8 * b5)
    c2 = c4sq / 2
    c1 = -c1sq / (16 * b5) * (-1 + a5 / b5 - g - 3 * r45)

    return (
        c3 * I3 
        + cp3 * Ip3 
        + (sigma / (8 * b5)) * (c2 * I2 - 2 * (x4 * xi / x1 - y4 * eta / y1))
        + c1 * I1 + A1111 * 0.5 / b5
    )

int5_ym1 = (
    lambda x, f, g, dm: 
    -int56_ym1(x, f, g, -dm, 1)
)

int5_x1 = (
    lambda y, f, g, dm: 
    -int56_x1(y, f, g, -dm, 1)
)

int6_ym1 = (
    lambda x, f, g, dp: 
    -int56_ym1(x, f, g, dp, -1)
)

int6_x1 = (
    lambda y, f, g, dp: 
    -int56_x1(y, f, g, dp, -1)
)

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