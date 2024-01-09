import jax
from jax import jit
import jax.numpy as jnp
from conics import *
from matrix import *

# definition for the stellar limb
star = {
        'a':1.0, 
        'b':1.0, 
        'center':(0.0, 0.0), 
        'theta':0.0
    }

star['matrix'] = matrix_form(star)

def add_matrix_form(spot):

    spot['matrix'] = matrix_form(spot)
    return spot

# returns the derivative of the field corresponding to 
# B along the boundary of A in the counter-clockwise direction.
# Used to determine the order of the arc boundaries. 
def direction(A, B, p):

    h, k = A['center']
    alpha = A['a']
    beta = A['b']
    theta = A['theta']
    
    cosp = jnp.cos(p)
    cost = jnp.cos(theta)
    sinp = jnp.sin(p)
    sint = jnp.sin(theta)
    
    beta2 = beta * beta
    alpha2 = alpha * alpha
    ab = alpha * beta

    gamma = jnp.sqrt(cosp * cosp * beta2 + sinp * sinp * alpha2)
    ogamma = 1 / gamma
    ogamma3 = ogamma**3
    
    p0 = h + ab * jnp.cos(theta + p) * ogamma
    p1 = k + ab * jnp.sin(theta + p) * ogamma

    dp0 = - ab * (beta2 * cosp * sint + alpha2 * cost * sinp) * ogamma3
    dp1 = ab * (beta2 * cost * cosp - alpha2 * sint * sinp) * ogamma3

    m = B['matrix']
    a = m[0,0]
    b = m[0,1]
    c = m[0,2]
    d = m[1,1]
    e = m[1,2]

    return (c + a * p0 + b * p1) * dp0 + (e + b * p0 + d * p1) * dp1

def field_value(A, B, p):

    h, k = A['center']
    alpha = A['a']
    beta = A['b']
    theta = A['theta']
    cosp = jnp.cos(p)
    costheta = jnp.cos(theta)
    sinp = jnp.sin(p)
    sintheta = jnp.sin(theta)

    x = h + alpha * cosp * costheta - beta * sinp * sintheta
    y = k + alpha * cosp * sintheta + beta * sinp * costheta 
    v = jnp.array([x, y])
    
    return mult_vec_sym(B['matrix'], v)
        
# A, B are spot dictionaries 
# this answers "is A inside B?".
def is_inside(A, B, xsecs):

    no_intersects = jnp.all(
        jnp.isnan(
            xsecs
        )
    )

    h, k = A['center']
    p = jnp.array([h, k, 1.0])
    center_inside = mult_vec_sym(B['matrix'], p) < 0.0
    return no_intersects & center_inside & ~is_same(A, B) & (A['a'] < B['a'])

def is_distant(A, B):

    ha, ka = A['center']
    hb, kb = B['center']
    d = jnp.sqrt((ha - hb)**2 + (ka - kb)**2)

    aa = A['a']
    ab = B['a']

    return aa + ab < d

def is_distant_star(A):

    h, k = A['center']
    r = jnp.sqrt(h**2 + k**2)
    return r + A['a'] < 1.0

def is_same(A, B):

    return eq(A['matrix'], B['matrix'])

# wraps get_intersection 
# so that it can be called with dictionaries 
intersect = lambda A, B: get_intersect(
    A['matrix'], 
    B['matrix']
)

def get_angle(point, ellipse):

    h, k = ellipse['center']
    x, y = point

    return jnp.arctan2((y - k), (x - h)) - ellipse['theta']

get_angles = jax.vmap(get_angle, in_axes=(0, None))

fix_angle = lambda p: p - 2 * jnp.pi * jnp.floor(p / (2 * jnp.pi))
fix_angles = jax.vmap(fix_angle)

def get_xy(angle, ellipse):

    h, k = ellipse['center']
    a, b = ellipse['a'], ellipse['b']
    sin, cos = jnp.sin(ellipse['theta']), jnp.cos(ellipse['theta'])
    sinp, cosp = jnp.sin(angle), jnp.cos(angle)

    r = a * b / jnp.sqrt((a * sinp)**2 + (b * cosp)**2)
    x = r * jnp.cos(angle)
    y = r * jnp.sin(angle)

    xp = x * cos - y * sin
    yp = x * sin + y * cos

    return h + xp, k + yp