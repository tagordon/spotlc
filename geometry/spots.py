import jax
import jax.numpy as jnp

from .conics import *
from .arc_utils import *
from .utils import *

def xtwo(arcset, spotj, xsecs):

    angles = get_angles(xsecs, arcset['spot'])
    p, q, r, s = jnp.sort(fix_angles(angles))
    
    dF = direction(arcset['spot'], spotj, p)

    p, q, r, s = jax.lax.cond(
        dF < 0,
        lambda: (p, q, r, s),
        lambda: (q, r, s, p)
    )

    arcset['pq'] = (p, q)
    arcset['i'] = 0
    arcset['k'] = arcset['newk']

    arcset = jax.lax.while_loop(i_less_k, split_arcs, arcset)

    arcset['pq'] = (r, s)
    arcset['i'] = 0
    arcset['k'] = arcset['newk']

    return jax.lax.while_loop(i_less_k, split_arcs, arcset)

def xone(arcset, spotj, xsecs):

    angles = get_angles(xsecs[:2], arcset['spot'])
    p, q = fix_angles(angles)

    dF = direction(arcset['spot'], spotj, q)

    p, q = jax.lax.cond(
        dF < 0,
        lambda: (q, p),
        lambda: (p, q)
    )

    arcset['pq'] = (p, q)
    arcset['i'] = 0
    arcset['k'] = arcset['newk']
    
    return jax.lax.while_loop(i_less_k, split_arcs, arcset)

def delete_arc(arcset, spotj, xsecs):
    arcset['arcs'] = jnp.zeros_like(arcset['arcs'])
    return arcset

def xspots(arcset, spoti, spotj):

    xsecs = intersect(spotj, spoti)

    conditions = jnp.array([
        is_inside(spoti, spotj, xsecs),
        jnp.isnan(xsecs[0][0]),
        jnp.isnan(xsecs[2][0]),
        True
    ])

    i = jnp.argwhere(conditions, size=1).squeeze()

    arcset = jax.lax.switch(
        i,
        [
            delete_arc,
            lambda a, b, c: a,
            xone,
            xtwo
        ],
        arcset, spotj, xsecs
    )

    return arcset

def add_spot(arcset, spotj):
    
    spoti = arcset['spot']

    conditions = jnp.array([
        is_same(spoti, spotj),
        is_distant(spoti, spotj),
        True
    ])

    i = jnp.argwhere(conditions, size=1).squeeze()

    arcset = jax.lax.switch(
        i, 
        [
            lambda a, b, c: a,
            lambda a, b, c: a,
            xspots
        ],
        arcset, spoti, spotj
    )
    
    return arcset, 0

def loop(spots, spoti):

    arcs = jnp.zeros((2*len(spots['a']), 2)) * jnp.nan
    arcs = arcs.at[0].set(jnp.array([0, 2*jnp.pi]))
    
    arcset = {
        'i':0, 
        'k':1, 
        'newk':1, 
        'spot':spoti, 
        'arcs':arcs, 
        'pq':(0.0, 0.0),
    }

    return spots, jax.lax.scan(add_spot, arcset, spots)

@jit 
def compute(spots):

    _, (res, _) = jax.lax.scan(loop, spots, spots)
    return res['arcs']