from jax import jit
import jax.numpy as jnp

eq = jit(lambda A, B: jnp.all(A == B))

def adj3(M):
        
    adj = jnp.zeros_like(M)

    adj = adj.at[0,0].set(M[1,1]*M[2,2] - M[1,2]*M[2,1])
    adj = adj.at[1,0].set(M[1,2]*M[2,0] - M[1,0]*M[2,2])
    adj = adj.at[2,0].set(M[1,0]*M[2,1] - M[1,1]*M[2,0])
    adj = adj.at[1,1].set(M[0,0]*M[2,2] - M[0,2]*M[2,0])
    adj = adj.at[2,1].set(M[0,1]*M[2,0] - M[0,0]*M[2,1])
    adj = adj.at[2,2].set(M[0,0]*M[1,1] - M[0,1]*M[1,0])

    adj = adj.at[0,1].set(adj[1,0])
    adj = adj.at[0,2].set(adj[2,0])
    adj = adj.at[1,2].set(adj[2,1])
    
    return adj

def det(M):

    f1 = M[0,0] * (M[1,1] * M[2,2] - M[1,2] * M[2,1])
    f2 = - M[0,1] * (M[1,0] * M[2,2] - M[1,2] * M[2,0])
    f3 = M[0,2] * (M[1,0] * M[2,1] - M[1,1] * M[2,0])

    return f1 + f2 + f3

def det_col(A, B, C):

    f1 = A[0] * (B[1] * C[2] - C[1] * B[2])
    f2 = - B[0] * (A[1] * C[2] - C[1] * A[2])
    f3 = C[0] * (A[1] * B[2] - B[1] * A[2])

    return f1 + f2 + f3

# this is for computing S.T * A * S
# where S is a skew-antisymmetric matrix 
# representing the outer product with a vector 
# s = (m, t, g) and A is a symmetric matrix. 
def quad_form(A, S):

    R = jnp.zeros_like(A)

    a = A[0,0]
    b = A[0,1]
    c = A[0,2]

    d = A[1,1]
    e = A[1,2]

    f = A[2,2]

    m = S[0,1]
    t = S[0,2]
    g = S[1,2]

    r00 = -m * (-m * d - t * e) - t * (-m * e - t * f)
    r01 = -m * (m * b - g * e) - t * (m * c - g * f)
    r02 = -m * (t * b + g * d) - t * (t * c + g * e)

    r10 = m * (-m * b - t * c) - g * (-m * e - t * f)
    r11 = m * (m * a - g * c) - g * (m * c - g * f)
    r12 = m * (t * a + g * b) - g * (t * c + g * e)

    r20 = t * (-m * b - t * c) + g * (-m * d - t * e)
    r21 = t * (m * a - g * c) + g * (m * b - g * e)
    r22 = t * (t * a + g * b) + g * (t * b + g * d)

    R = R.at[0,0].set(r00)
    R = R.at[0,1].set(r01)
    R = R.at[0,2].set(r02)
    R = R.at[1,0].set(r10)
    R = R.at[1,1].set(r11)
    R = R.at[1,2].set(r12)
    R = R.at[2,0].set(r20)
    R = R.at[2,1].set(r21)
    R = R.at[2,2].set(r22)

    return R

def skew_sym_vec(p):

    return jnp.array([
        [0, p[2], -p[1]], 
        [-p[2], 0, p[0]], 
        [p[1], -p[0], 0]
    ])

# get the skew symmetric matrix
# for splitting the degenerate 
# conic represented by M
def skew_sym(M):

    B = -adj3(M)
    diag = jnp.abs(jnp.diag(B))
    i = jnp.argmax(diag)

    p = jnp.array(
        B[i] / jnp.sqrt(diag[i])
    ).flatten()

    return skew_sym_vec(p)

# calculates p^T.A.p for symmetric A 
# and p = (x, y, 1) 
def mult_vec_sym(A, p):

    return (
        A[2,2] 
        + 2 * A[0,2] * p[0] 
        + A[0,0] * p[0] * p[0] 
        + 2 * (A[1,2] + A[0,1] * p[0]) * p[1] 
        + A[1,1] * p[1] * p[1]
    )
    






