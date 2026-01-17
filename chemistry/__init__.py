"""Chemistry systems - primordial elements, reactions, and resonance."""

from .elements import (
    ElementType, ElementProperties, ElementSignature,
    ELEMENT_PROPS, ELEMENT_SIGNATURES, ELEMENT_TERRAIN_SOURCES,
    APEX_BANE_ELEMENT, APEX_BANE_DAMAGE_MULTIPLIER, APEX_BANE_FEAR_RADIUS,
    compute_herbie_spectral_signature, compute_element_resonance,
    compute_object_resonance, get_herbie_element_affinities
)
from .element_objects import ElementObject, ElementField, Construction
from .spawner import (
    ElementSpawner, integrate_chemistry_to_world_objects,
    check_element_interaction, on_element_dropped,
    save_chemistry_state, load_chemistry_state
)
