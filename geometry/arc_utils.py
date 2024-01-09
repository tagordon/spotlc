import jax
from jax import jit
import jax.numpy as jnp

i_less_k = lambda arcset: arcset['i'] < arcset['k']
less = jax.jit(lambda a, b, c, d: jnp.less_equal(a, b) & jnp.less_equal(b, c) & jnp.less_equal(c, d))

@jit 
def invert_arcs(arcs):

    return jax.lax.cond(
        jnp.any(arcs > 0),
        lambda: invert_with_arcs(arcs),
        lambda: arcs.at[0, 1].set(2 * jnp.pi)
    )

def invert_with_arcs(arcs):

    arcs = jnp.sort(arcs, axis=0)
    p = arcs.T[1]
    q = arcs.T[0]

    idx = len(q) - jnp.argwhere(jnp.isfinite(q[::-1]), size=1)
    
    p = jnp.roll(p, 1)
    new_arcs = jnp.zeros_like(arcs)
    new_arcs = jnp.array([p, q]).T
    new_arcs = jnp.nan_to_num(new_arcs)
    new_arcs = jnp.mod(new_arcs, 2 * jnp.pi)

    new_arcs = jax.lax.cond(
        new_arcs.at[idx, 1] == 0.0,
        lambda: new_arcs.at[idx, 1].set(2 * jnp.pi),
        lambda: new_arcs
    )
    return new_arcs

@jit
def set_two(arcset, a, b):

    arcset['arcs'] = arcset['arcs'].at[arcset['i']].set(a)
    arcset['arcs'] = arcset['arcs'].at[arcset['newk']].set(b)
    arcset['newk'] += 1
    arcset['i'] += 1
    
    return arcset
    
@jit
def set_one(arcset, a):

    arcset['arcs'] = arcset['arcs'].at[arcset['i']].set(a)
    arcset['i'] += 1

    return arcset

@jit 
def set_three(arcset, a, b, c):

    arcset['arcs'] = arcset['arcs'].at[arcset['i']].set(a)
    arcset['arcs'] = arcset['arcs'].at[arcset['newk']].set(b)
    arcset['arcs'] = arcset['arcs'].at[arcset['newk'] + 1].set(c)
    arcset['i'] += 1
    arcset['newk'] += 2
    
    return arcset

def split_arcs(arcset):

    (p, q) = arcset['arcs'][arcset['i']]
    (pp, qp) = arcset['pq']

    conditions = jnp.array([
        (p == pp) & (q == qp),
        (p == q),
        
        less(p, q, pp, qp),
        less(p, q, qp, pp),
        less(p, pp, q, qp), 
        less(p, pp, qp, q),
        less(p, qp, pp, q),
        less(p, qp, q, pp),

        less(q, p, qp, pp), 
        less(q, p, pp, qp), 
        less(q, pp, p, qp), 
        less(q, pp, qp, p), 
        less(q, qp, p, pp), 
        less(q, qp, pp, p), 

        less(pp, q, p, qp), 
        less(pp, q, qp, p), 
        less(pp, p, q, qp), 
        less(pp, p, qp, q), 
        less(pp, qp, p, q), 
        less(pp, qp, q, p),

        less(qp, p, q, pp), 
        less(qp, pp, q, p), 
        less(qp, q, p, pp), 
        less(qp, p, pp, q), 
        less(qp, pp, p, q), 
        less(qp, q, pp, p),

        True
    ])

    outcomes = [      
        lambda: set_one(arcset, (0.0, 0.0)),
        lambda: set_one(arcset, (0.0, 0.0)),
        
        lambda: set_one(arcset, (p, q)), 
        lambda: set_one(arcset, (0.0, 0.0)),
        lambda: set_one(arcset, (p, pp)), 
        lambda: set_two(arcset, (p, pp), (qp, q)),
        lambda: set_one(arcset, (qp, pp)),
        lambda: set_one(arcset, (qp, q)),

        lambda: set_one(arcset, (qp, pp)), 
        lambda: set_three(arcset, (p, pp), (qp, 0.0), (0.0, q)),
        lambda: set_two(arcset, (qp, 0.0), (0.0, q)),
        lambda: set_two(arcset, (p, 0.0), (0.0, q)),
        lambda: set_one(arcset, (p, pp)), 
        lambda: set_one(arcset, (0.0, 0.0)),

        lambda: set_two(arcset, (qp, 0.0), (0.0, pp)), 
        lambda: set_two(arcset, (p, 0.0), (0.0, pp)),
        lambda: set_one(arcset, (0.0, 0.0)),
        lambda: set_one(arcset, (qp, q)),
        lambda: set_one(arcset, (p, q)),
        lambda: set_three(arcset, (p, 0.0), (0.0, pp), (qp, q)),

        lambda: set_one(arcset, (p, q)),
        lambda: set_one(arcset, (qp, pp)),
        lambda: set_two(arcset, (p, pp), (qp, q)),
        lambda: set_one(arcset, (p, pp)),
        lambda: set_one(arcset, (0.0, 0.0)),
        lambda: set_one(arcset, (qp, q)),

        lambda: set_one(arcset, (jnp.nan, jnp.nan))
    ]

    ind = jnp.argwhere(conditions, size=1).squeeze()
    
    return jax.lax.switch(ind, outcomes)

def split_limb(arcset):

    (p, q) = arcset['arcs'][arcset['i']]
    (ps, qs) = arcset['pq']

    conditions = jnp.array([
        less(p, ps, qs, q),
        less(p, ps, q, qs),
        less(ps, p, qs, q),
        less(qs, p, ps, q),
        less(p, qs, q, ps),
        less(p, qs, ps, q),
        less(p, ps, q, qs),
        True
    ])

    outcomes = [
        lambda: set_two(arcset, (p, ps), (qs, q)),
        lambda: set_one(arcset, (p, ps)),
        lambda: set_one(arcset, (qs, q)),
        lambda: set_one(arcset, (p, ps)),
        lambda: set_one(arcset, (qs, q)),
        lambda: set_one(arcset, (qs, ps)),
        lambda: set_one(arcset, (ps, q)),
        lambda: set_one(arcset, (p, q))
    ]

    ind = jnp.argwhere(conditions, size=1).squeeze()
    return jax.lax.switch(ind, outcomes)