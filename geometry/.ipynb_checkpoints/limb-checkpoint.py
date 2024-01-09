from conics import *
from arc_utils import *
from utils import *
import jax
import jax.numpy as jnp

def xstar(star_arcdict, spot, xsecs):

    angles = get_angles(xsecs, star)
    p, q = fix_angles(angles)

    dF = direction(star, spot, q)

    p, q = jax.lax.cond(
        dF < 0,
        lambda: (q, p),
        lambda: (p, q)
    )

    star_arcdict['pq'] = (p, q)
    star_arcdict['i'] = 0
    star_arcdict['k'] = star_arcdict['newk']

    star_arcdict = jax.lax.while_loop(
        i_less_k, 
        split_arcs,
        star_arcdict
    )

    return star_arcdict

def xspot(spot_arcdict_loop, xsecs):

    spot = spot_arcdict_loop['spot']
    angles = get_angles(xsecs, spot)
    p, q = fix_angles(angles)

    dF = direction(spot, star, q)

    #jax.debug.print("{}", spot_arcdict_loop)

    p, q = jax.lax.cond(
        dF > 0,
        lambda: (q, p),
        lambda: (p, q)
    )

    spot_arcdict_loop['pq'] = (p, q)
    spot_arcdict_loop['i'] = spot_arcdict_loop['start']
    
    return jax.lax.while_loop(
        i_less_k, 
        split_arcs,
        spot_arcdict_loop
    )

def xspot_star(star_arcdict, spot_arcdict):

    spot = spot_arcdict['spot']
    xsecs = intersect(star, spot)

    star_arcdict = jax.lax.cond(
        jnp.isnan(xsecs[0][0]), 
        lambda a, b, c: a, 
        xstar, 
        star_arcdict, spot, xsecs[:2]
    )

    k = jax.lax.cond(
        ~jnp.any(spot_arcdict['arc'] > 0),
        lambda: len(spot_arcdict['arc'][::-1]),
        lambda: jnp.argwhere(
            jnp.any(
                spot_arcdict['arc'][::-1] > 0, axis=1
            ), size=1
        ).squeeze()
    )
    k = len(spot_arcdict['arc']) - k

    i = jnp.argwhere(
        jnp.any(
            spot_arcdict['arc'] > 0, axis=1
        ), size=1
    ).squeeze()

    spot_arcdict_loop = {
        'arcs':spot_arcdict['arc'],
        'spot':spot,
        'start': i,
        'i': i,
        'k': k,
        'newk': k,
        'pq': (0.0, 0.0)
    }

    spot_arcdict_loop = jax.lax.cond(
        jnp.isnan(xsecs[0][0]),
        lambda a, b: a,
        xspot,
        spot_arcdict_loop, xsecs[:2]
    )

    spot_arcdict['arc'] = spot_arcdict_loop['arcs']
    
    return star_arcdict, spot_arcdict

def loop(star_arcdict, spot_arcdict):

    return jax.lax.cond(
        ~is_distant_star(spot_arcdict['spot']),
        xspot_star,
        lambda a, b: (a, b),
        star_arcdict, spot_arcdict
    ) 

@jit
def compute(arcs, spots):

    star_arcs = jnp.zeros((2*len(spots['a']), 2)) * jnp.nan
    star_arcs = star_arcs.at[0].set(jnp.array([0, 2*jnp.pi]))
    spot_arcdict = {'arc':arcs, 'spot':spots}

    star_arcdict = {
        'arcs':star_arcs,
        'i': 0,
        'k': 1,
        'newk': 1,
        'pq':(0.0, 0.0)
    }
    
    star_arcdict, spot_arcdict = jax.lax.scan(loop, star_arcdict, spot_arcdict)

    return star_arcdict['arcs'], spot_arcdict['arc']