import jax
from jax import jit
from jax.tree_util import tree_map
import numpy as np
import jax.numpy as jnp

from .utils import add_matrix_form
from .spots import compute
from .planets import compute as compute_planets
from .arc_utils import invert_arcs
from .limb import compute as compute_limb

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

    # compute arcs for planets and spots without regard for the limb of the star
    interior = compute(spots_and_planets)
    # remote portions of arcs outside of the stellar disk and build 
    # arcs for limb of star 
    limb, interior = compute_limb(interior, spots_and_planets)
    # compute arcs along the stellar limb that are interior to the planet
    plimb, _ = compute_limb(compute(planets), planets)
    # and then remove those arcs from the set of arcs along the stellar limb 
    plimb = invert_arcs(plimb)
    # compute arcs along the edge of the planet that are interior to the spots 
    planet_spot = compute_planets(planets, interior[-nplanets:])
    _, planet_spot = compute_limb(planet_spot, planets)
    slimb = invert_arcs(jnp.concatenate([limb, plimb]))

    return interior, planet_spot, plimb, slimb