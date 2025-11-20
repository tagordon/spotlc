import jax
from jax import jit
import jax.numpy as jnp

from .matrix import *
#from matrix import *

from jax import config
config.update("jax_enable_x64", True)

root3o2 = 0.8660254037844386
o3 = 0.333333333333333333
nans = jnp.array([[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]])
sign = jax.jit(lambda x: -jnp.int64(jnp.signbit(x)) * 2 + 1)
 
def matrix_form(E):

    h, k = E['center']
    cos = jnp.cos(E['theta'])
    cos2 = cos**2
    sin = jnp.sin(E['theta'])
    sin2 = sin**2
    oa2 = 1 / E['a']**2
    ob2 = 1 / E['b']**2

    M = jnp.zeros((3, 3))

    M = M.at[0,0].set(cos2 * oa2 + sin2 * ob2)
    M = M.at[0,1].set(cos * sin * (oa2 - ob2))
    M = M.at[0,2].set(-h * cos2 * oa2 
                      + (ob2 - oa2) * k * cos * sin 
                      - h * sin2 * ob2
                     )
    M = M.at[1,1].set(cos2 * ob2 + sin2 * oa2)
    M = M.at[1,2].set(-k * cos2 * ob2 
                      + (ob2 - oa2) * h * cos * sin 
                      - k * sin2 * oa2
                     )
    M = M.at[2,2].set((0.5 * oa2 * ob2) * (
        (1/oa2 + 1/ob2) * (h * h + k * k) - 
        (1/oa2 - 1/ob2) * (
            (h * h - k * k) * (cos2 - sin2) + 
            4 * h * k * cos * sin
        )
    ) - 1)

    M = M.at[1,0].set(M[0,1])
    M = M.at[2,0].set(M[0,2])
    M = M.at[2,1].set(M[1,2])

    return M

def get_cubic_coeffs(A, B):

    BAA = det_col(B[0], A[1], A[2])
    AAB = det_col(A[0], A[1], B[2])
    ABA = det_col(A[0], B[1], A[2])

    ABB = det_col(A[0], B[1], B[2])
    BAB = det_col(B[0], A[1], B[2])
    BBA = det_col(B[0], B[1], A[2])

    a = det(A) 
    d = det(B)
    scale = 1.0 / (a * d)

    a = a * scale
    d = d * scale

    b = (AAB + ABA + BAA) * scale * o3
    c = (ABB + BAB + BBA) * scale * o3

    return a, b, c, d

def get_degenerate_conic(A, B):

    a, b, c, d = get_cubic_coeffs(A, B)
    d1, d2, d3, Delta = discriminant(a, b, c, d)

    x, w = jax.lax.cond(
        Delta > 0, 
        lambda: solve_cubic_three_real(
            a, b, c, d, 
            d1, d2, d3, Delta
        ),
        lambda: solve_cubic_one_real(
            a, b, c, d, 
            d1, d2, d3, Delta
        )
    )
    
    return x * A + w * B

def split_degenerate_conic(M):

    Q = M + skew_sym(M)

    i, j = jnp.unravel_index(jnp.argmax(jnp.abs(Q)), (3, 3))
    c1 = Q.T[j].flatten()
    c2 = Q[i].flatten()

    return c1, c2

def coords(B, disc, c, S):

    C = B + jnp.sqrt(disc) / c[2] * S

    i = jnp.argmax(jnp.abs(C))
    i, j = jnp.unravel_index(i, (3, 3))
    p = C[i]
    q = C.T[j]

    return jnp.array([p[:2] / p[2], q[:2] / q[2]])

def get_intersect(A, B):

    M = get_degenerate_conic(A, B)
    c1, c2 = split_degenerate_conic(M)

    Ml1 = skew_sym_vec(-c1)
    Ml2 = skew_sym_vec(c2)

    B1 = quad_form(B, Ml1)
    B2 = quad_form(B, Ml2)
    A1 = quad_form(A, Ml1)
    A2 = quad_form(A, Ml2)

    discB1 = B1[0, 1] * B1[1, 0] - B1[0, 0] * B1[1, 1]
    discB2 = B2[0, 1] * B2[1, 0] - B2[0, 0] * B2[1, 1]
    discA1 = A1[0, 1] * A1[1, 0] - A1[0, 0] * A1[1, 1]
    discA2 = A2[0, 1] * A2[1, 0] - A2[0, 0] * A2[1, 1]

    pA = coords(A1, discA1, c1, Ml1)
    qA = coords(A2, discA2, c2, Ml2)
    pB = coords(B1, discB1, c1, Ml1)
    qB = coords(B2, discB2, c2, Ml2)

    tol = 1e-6
    cond1 = jnp.all(jnp.abs(pA - pB) < tol)
    cond2 = jnp.all(jnp.abs(qA - qB) < tol)

    conditions = jnp.array([
        cond1 & cond2,
        cond1,
        cond2,
        True
    ])

    i = jnp.argwhere(conditions, size=1).squeeze()

    intersects = jax.lax.switch(
        i, 
        [
            lambda: jnp.concatenate(jnp.array([pA, qA])),
            lambda: jnp.concatenate(jnp.array([pA, nans])),
            lambda: jnp.concatenate(jnp.array([qA, nans])),
            lambda: jnp.concatenate(jnp.array([nans, nans]))
        ]
    )

    return intersects
    
def discriminant(a, b, c, d):

    d1 = a * c - b * b
    d2 = a * d - b * c
    d3 = b * d - c * c
    Delta = 4 * d1 * d3 - d2 * d2

    return d1, d2, d3, Delta

# using https://courses.cs.washington.edu/courses/cse590b/13au/lecture_notes/solvecubic_p5.pdf
# if Delta < 0 there is one real root. We only need this root for calculating the 
# single intersection between the two ellipses.
def solve_cubic_one_real(a, b, c, d, d1, d2, d3, Delta): 

    cond = b**3 * d >= a * c**3

    Atild, Cbar, Dbar = jax.lax.cond(
        cond,
        lambda: (a, d1, -2 * b * d1 + a * d2),
        lambda: (d, d3, -d * d2 + 2 * c * d3)
    )

    T0 = jnp.abs(Atild) * jnp.sqrt(-Delta)
    T1 = jnp.complex128(jnp.abs(Dbar) + T0)
    p = -sign(Dbar) * (T1 * 0.5) ** o3

    q = jax.lax.cond(
        T1 == T0,
        lambda: -p,
        lambda: -Cbar / p
    )
    
    x = jax.lax.cond(
        Cbar > 0,
        lambda: -Dbar / (p * p + q * q + Cbar),
        lambda: p + q
    )

    x = p + q
    x = jnp.real(x)

    x, w = jax.lax.cond(
        cond,
        lambda: (x - b, a),
        lambda: (-d, x + c)
    )
    
    return x, w

# using https://courses.cs.washington.edu/courses/cse590b/13au/lecture_notes/solvecubic_p5.pdf
# if Delta > 0 there are three real roots. This corresponds to two intersections 
# between the two ellipses or no intersections between the ellipses (which is annoying) 
def solve_cubic_three_real(a, b, c, d, d1, d2, d3, Delta):

    Da = -2 * b * d1 + a * d2
    sqCa = jnp.sqrt(-d1)

    theta_a = jnp.abs(jnp.arctan2(a * jnp.sqrt(Delta), -Da)) * o3

    x1a = 2 * sqCa * jnp.cos(theta_a)
    x3a = 2 * sqCa * (-0.5 * jnp.cos(theta_a) - root3o2 * jnp.sin(theta_a))

    xl = jax.lax.cond(
        x1a + x3a > 2 * b,
        lambda: x1a - b,
        lambda: x3a - b
    )

    return xl, a
    