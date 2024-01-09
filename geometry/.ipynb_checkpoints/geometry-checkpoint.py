import jax
from jax import jit
from jax.tree_util import tree_map
import numpy as np
import jax.numpy as jnp
#from conics import *
from utils import add_matrix_form
 
from spots import compute
from planets import compute as compute_planets
from arc_utils import invert_arcs
from limb import compute as compute_limb

@jit 
def build_geometry(spots, planets):

    nplanets = len(planets['a'])

    spots = jax.vmap(add_matrix_form)(spots)
    planets = jax.vmap(add_matrix_form)(planets)

    spots_and_planets = tree_map(
        lambda a, b: jnp.concatenate([a, b]), 
        spots, 
        planets
    )

    interior = compute(spots_and_planets)
    limb, interior = compute_limb(interior, spots_and_planets)
    plimb, _ = compute_limb(compute(planets), planets)
    plimb = invert_arcs(plimb)
    planet_spot = compute_planets(planets, interior[-nplanets:])
    _, planet_spot = compute_limb(planet_spot, planets)
    slimb = invert_arcs(jnp.concatenate([limb, plimb]))

    return interior, planet_spot, plimb, slimb