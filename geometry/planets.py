from conics import *
from arc_utils import *
from utils import *
import jax
import jax.numpy as jnp

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
    arcset['i'] = arcset['start']
    arcset['k'] = arcset['newk']

    arcset = jax.lax.while_loop(i_less_k, split_arcs, arcset)

    arcset['pq'] = (r, s)
    arcset['i'] = arcset['start']
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
    arcset['i'] = arcset['start']
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

def loop(planets, planeti):

    planet = planeti['planet']
    arcs = planeti['arc']

    k = jax.lax.cond(
        ~jnp.any(arcs > 0),
        lambda: len(arcs),
        lambda: jnp.argwhere(
            jnp.any(
                arcs[::-1] > 0, axis=1
            ), size=1
        ).squeeze()
    )
    k = len(arcs) - k

    i = jnp.argwhere(
        jnp.any(
            arcs > 0, axis=1
        ), size=1
    ).squeeze()
    
    arcset = {
        'start': i,
        'i': i,
        'k': k,
        'newk': k,
        'spot': planet, 
        'arcs': arcs, 
        'pq': (0.0, 0.0),
    }

    return planets, jax.lax.scan(add_spot, arcset, planets)

@jit 
def compute(planets, arcs):

    arcs = jax.vmap(invert_arcs)(arcs)
    arcdict = {'planet':planets, 'arc':arcs}

    _, (res, _) = jax.lax.scan(loop, planets, arcdict)
    return res['arcs']