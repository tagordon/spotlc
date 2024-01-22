# these functions integrate the terms in the 
# partial fraction decomposition of the integrand 
# for the line integral along the arc of the ellipse 
# in the linear term of the limb-darkening profile. 

import jax
from jax import numpy as jnp
#from carlson import rfv, rjpv, rcpv, rdv
from carlson import rf, rj_posp, rc_posy, rd

def int23_circ(r, z, y, x, a4, b4):

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
    y2 = jnp.sqrt(1 -  y)
    y3 = jnp.sqrt(a3 - y)
    y4 = jnp.sqrt(a4 + b4 * y)

    U1 = (x1 * y2 * y3 + y1 * x2 * x3) / (x - y)
    U2 = (x2 * y1 * y3 + y2 * x1 * x3) / (x - y)
    U3 = (x3 * y1 * y2 + y3 * x1 * x2) / (x - y)

    W2 = U1 * U1 - b4 * d12 * d13 / d14
    Q2 = (x4 * y4 / (x1 * y1))**2 * W2
    P2 = Q2 + b4 * d24 * d34 / d14

    I1c = 2 * rf(U3 * U3, U2 * U2, U1 * U1)
    I2c = (2 / 3) * d12 * d13 * rd(U3 * U3, U2 * U2, U1 * U1) + 2 * x1 * y1 / U1
    I3c = -2 * d12 * d13 / (3 * d14) * rj_posp(U3 * U3, U2 * U2, U1 * U1, W2) + 2 * rc_posy(P2, Q2)

    A111 = x1 * x2 * x3 - y1 * y2 * y3
    J1c = d12 * d13 * I1c - 2 * A111

    return (
        - 3 * r24 * d34 * I3c 
        + (r14 + r24 + r34) * I2c 
        - J1c
    ) / (3 * b4)

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

def g1_circ(r, z, y, x):

    fac = r / (6 * (r + z) * (r - z))
    fx = jnp.arctan((x + z / r) / jnp.sqrt(1 - x * x))
    fy = jnp.arctan((y + z / r) / jnp.sqrt(1 - y * y))
    return (fx - fy) / 3

def g4_circ(r, z, y, x):

    fac = r * (r - z + 1) * (r - z - 1) / (6 * (r - z)) * jnp.sqrt(2 * r * z)
    return fac * int4_circ(r, z, y, x)

def g2_circ(r, z, y, x):

    fac = r * (1 + r + z) * (r + z - 1) * jnp.sqrt(r * z / 2) / (3 * (r + z))
    return fac * int23_circ(r, z, y, x, 1, -1)

def g3_circ(r, z, y, x):

    a4 = (r * r + z * z) / (2 * r * z)
    fac = 2 * r * r * z * z / jnp.sqrt(2 * r * z) / (3 * (r + z) * (r - z))
    return fac * int23_circ(r, z, y, x, a4, 1)

@jax.jit 
def G_circ(r, z, y, x):

    ig1 = g1_circ(r, z, y, x)
    ig2 = g2_circ(r, z, y, x)
    ig3 = g3_circ(r, z, y, x)
    ig4 = g4_circ(r, z, y, x)

    return ig1 + ig2 + ig3 + ig4

# for the complete integral we just use the 
# trapezoidal rule, which converges exponentially 
# for periodic functions 
@jax.jit
def G_complete(a, b, z):

    def ig(t):

        s = jnp.sin(t)
        x = (a * jnp.cos(t))**2 + (z + b * s)**2
        return a * (b + s * z) * (
            1 - (1 - x) ** 1.5
        ) / (3 * x)
    
    ni = jnp.arange(12)
    twopi_n = jnp.pi / 6
    
    return twopi_n * jnp.sum(ig(twopi_n * ni)) * 0.5

@jax.jit
def G_complete_circ(r, z):

    def ig(t):

        s = jnp.sin(t)
        x = r**2 + z**2 + 2 * r * z * s
        return r * (r + s * z) * (
            1 - (1 - x) ** 1.5
        ) / (3 * x)

    ni = jnp.arange(12)
    twopi_n = jnp.pi / 6
    
    return twopi_n * jnp.sum(ig(twopi_n * ni)) * 0.5

# this is the full integral along the arc
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

def int23(y, x, f, g, b1, b4):

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
        + 2 * x1 * y1 / (x4 * y4 * U)
    )
    I3p = (
        jnp.sqrt(2 * c1sq / 9) 
        * (4 * rho * erj 
           - 6 * erf + 3 * rcuw) 
        + 2 * rcpq
    )

    return (
        c3 * I3p 
        + s * (c2 * I2 + 2 * A111m1) 
        + c1 * I1 + A1111 / (2 * b4)
    )

int2 = jax.jit(
    lambda y, x, f, g: 
    int23(y, x, f, g, 1, -1)
)
int3 = jax.jit(
    lambda y, x, f, g: 
    int23(y, x, f, g, -1, 1)
)

def int4(y, x, f, g):

    x1 = jnp.sqrt(1 + x)
    x4 = jnp.sqrt(1 - x)

    y1 = jnp.sqrt(1 + y)
    y4 = jnp.sqrt(1 - y)

    xi = jnp.sqrt(f + g * x + x * x)
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
        2 * xi * eta + 2 * f + g * (x + y) + 2 * x * y
    ) / (x - y)**2
    
    Lp2 = M2 + c14sq + c1c4
    Lm2 = M2 + c14sq - c1c4
    U = (x1 * x4 * eta + y1 * y4 * xi) / (x - y)
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
        + 2 * x1 * y1 / (x4 * y4 * U)
    )
    I3p = (
        jnp.sqrt(2 * c1sq / 9) 
        * (4 * rho * erj - 6 * erf + 3 * rcuw) 
        + 2 * rcpq
    )

    return -0.25 * (c4sq * I2 - c1sq * I1 - 2 * g * I3p) + A111m1
    
def int56(y, x, f, g, a5, b5):
    
    x1 = jnp.sqrt(1 - x)
    x4 = jnp.sqrt(1 + x)
    x52 = a5 + b5 * x
    
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

    xi = jnp.sqrt(f + g * x + x * x)
    eta = jnp.sqrt(f + g * y + y * y)

    A1111 = x1 * xi * x4 - y1 * eta * y4
    A111m1 = x1 * xi / x4 - y1 * eta / y4

    M2 = (x1 * y4 + y1 * x4)**2 * ((xi + eta)**2 - (x - y)**2) / (x - y)**2
    Lm2 = M2 + c14sq - c1c4
    Lp2 = M2 + c14sq + c1c4
    Wp2 = M2 + d14 * (c15sq + c1c5) / d15
    U = (x1 * x4 * eta + y1 * y4 * xi) / (x - y)
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
    ) + 2 * x1 * y1 / (x4 * y4 * U)
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
        + (sigma / (8 * b5)) * (c2 * I2 + 2 * A111m1) 
        + c1 * I1 + A1111 * 0.5 / b5
    )

int5 = (
    lambda y, x, f, g, dm: 
    -int56(y, x, f, g, -dm, 1)
)
int6 = (
    lambda y, x, f, g, dp: 
    -int56(y, x, f, g, dp, -1)
)

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