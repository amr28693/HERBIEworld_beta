"""
Species Definitions - Parameters for each creature type.

Each species has unique morphology, NLSE parameters, metabolism,
movement characteristics, and diet/predation behavior.
"""

from typing import List, Dict
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SpeciesParams:
    """
    Parameters defining a species' morphology, dynamics, and diet.
    
    All creatures share the same fundamental NLSE architecture but
    with different parameters that lead to emergent behavioral differences.
    """
    name: str
    color_base: str  # Primary color for visualization
    
    # Morphology
    num_limbs: int           # 0, 1, 2, 3, or 4
    body_scale: float        # Multiplier on body size (0.5 - 1.5)
    core_radius: float       # Core basin radius
    limb_length: float       # Length of limbs
    
    # NLSE parameters
    body_g_base: float       # Base nonlinearity for body field
    torus_g_base: float      # Base nonlinearity for torus
    limb_g_base: float       # Base nonlinearity for limbs
    
    # Energy/metabolism
    energy_efficiency: float      # How well food converts (0.5 - 1.5)
    metabolism_rate: float        # Hunger increase rate multiplier
    max_age_base: int             # Base lifespan
    
    # Movement
    speed_factor: float      # Movement speed multiplier
    
    # KdV neural parameters
    kdv_amplitude: float         # Signal strength multiplier
    afferent_sensitivity: float  # Sensory sensitivity
    efferent_strength: float     # Motor output strength
    
    # Fields with defaults must come last
    reproduction_refractory: int = 1000  # Recovery time between births (steps)
    diet: str = 'herbivore'           # 'herbivore', 'omnivore', 'carnivore'
    hunt_damage: float = 0.0          # Damage dealt per contact frame
    prey_species: List[str] = field(default_factory=list)  # Species this can eat
    
    def __post_init__(self):
        if self.prey_species is None:
            self.prey_species = []


# =============================================================================
# SPECIES DEFINITIONS
# =============================================================================

SPECIES_HERBIE = SpeciesParams(
    name="Herbie",
    color_base="cyan",
    num_limbs=3,
    body_scale=1.0,
    core_radius=2.2,
    limb_length=4.0,
    body_g_base=0.3,
    torus_g_base=1.2,
    limb_g_base=0.6,
    energy_efficiency=1.0,
    metabolism_rate=1.0,
    max_age_base=10000,
    speed_factor=1.0,
    kdv_amplitude=1.0,
    afferent_sensitivity=1.0,
    efferent_strength=1.0,
    reproduction_refractory=1200,
    diet='herbivore',
    hunt_damage=0.0,
    prey_species=[],
)

SPECIES_BLOB = SpeciesParams(
    name="Blob",
    color_base="magenta",
    num_limbs=0,
    body_scale=1.3,
    core_radius=3.0,
    limb_length=0.0,
    body_g_base=0.5,
    torus_g_base=0.8,
    limb_g_base=0.0,
    energy_efficiency=1.3,
    metabolism_rate=0.7,
    max_age_base=8000,
    speed_factor=0.6,
    kdv_amplitude=0.7,
    afferent_sensitivity=1.3,
    efferent_strength=0.5,
    reproduction_refractory=600,
    diet='herbivore',
    hunt_damage=0.0,
    prey_species=[],
)

SPECIES_BIPED = SpeciesParams(
    name="Biped",
    color_base="orange",
    num_limbs=2,
    body_scale=0.85,
    core_radius=1.8,
    limb_length=4.5,
    body_g_base=0.35,
    torus_g_base=1.4,
    limb_g_base=0.8,
    energy_efficiency=0.9,
    metabolism_rate=1.2,
    max_age_base=9000,
    speed_factor=1.3,
    kdv_amplitude=1.1,
    afferent_sensitivity=0.9,
    efferent_strength=1.3,
    reproduction_refractory=800,
    diet='herbivore',
    hunt_damage=0.0,
    prey_species=[],
)

SPECIES_MONO = SpeciesParams(
    name="Mono",
    color_base="lime",
    num_limbs=1,
    body_scale=0.7,
    core_radius=1.5,
    limb_length=5.0,
    body_g_base=0.25,
    torus_g_base=1.0,
    limb_g_base=0.9,
    energy_efficiency=1.1,
    metabolism_rate=0.9,
    max_age_base=11000,
    speed_factor=0.9,
    kdv_amplitude=0.9,
    afferent_sensitivity=1.1,
    efferent_strength=1.0,
    reproduction_refractory=700,
    diet='herbivore',
    hunt_damage=0.0,
    prey_species=[],
)

# Scavenger - opportunistic omnivore
SPECIES_SCAVENGER = SpeciesParams(
    name="Scavenger",
    color_base="brown",
    num_limbs=2,
    body_scale=0.95,
    core_radius=2.0,
    limb_length=3.5,
    body_g_base=0.4,
    torus_g_base=1.1,
    limb_g_base=0.7,
    energy_efficiency=1.2,
    metabolism_rate=1.0,
    max_age_base=9500,
    speed_factor=1.1,
    kdv_amplitude=1.0,
    afferent_sensitivity=1.2,
    efferent_strength=1.0,
    reproduction_refractory=900,
    diet='omnivore',
    hunt_damage=0.15,
    prey_species=['Blob', 'Mono'],
)

# Apex - rare apex predator
SPECIES_APEX = SpeciesParams(
    name="Apex",
    color_base="red",
    num_limbs=4,
    body_scale=1.4,
    core_radius=2.8,
    limb_length=5.5,
    body_g_base=0.45,
    torus_g_base=1.6,
    limb_g_base=0.85,
    energy_efficiency=0.85,
    metabolism_rate=1.2,
    max_age_base=7000,
    speed_factor=1.6,
    kdv_amplitude=1.3,
    afferent_sensitivity=1.4,
    efferent_strength=1.5,
    reproduction_refractory=1000,
    diet='carnivore',
    hunt_damage=0.5,
    prey_species=['Herbie', 'Blob', 'Biped', 'Mono', 'Scavenger'],
)

# Collections
ALL_SPECIES = [
    SPECIES_HERBIE, SPECIES_BLOB, SPECIES_BIPED, SPECIES_MONO,
    SPECIES_SCAVENGER, SPECIES_APEX
]

SPECIES_BY_NAME: Dict[str, SpeciesParams] = {s.name: s for s in ALL_SPECIES}

# Spawn weights (Apex is rare)
SPECIES_SPAWN_WEIGHTS = {
    'Herbie': 1.0,
    'Blob': 1.0,
    'Biped': 1.0,
    'Mono': 1.0,
    'Scavenger': 0.6,
    'Apex': 0.1,
}


# =============================================================================
# SPECIES-SPECIFIC BASIN/LIMB CONFIGURATION
# =============================================================================

def get_species_basins(species: SpeciesParams) -> dict:
    """
    Generate basin configuration for a species.
    
    Basins are potential wells that confine the body field,
    defining the creature's morphology.
    
    Args:
        species: Species parameters
        
    Returns:
        Dict of basin definitions
    """
    basins = {
        'core': {
            'pos': (0.0, 0.0),
            'radius': species.core_radius,
            'depth': 12.0 * species.body_scale,
            'torus_coupling': 0.7
        }
    }
    
    if species.num_limbs >= 1:
        basins['limb_0'] = {
            'pos': (0.0, species.core_radius + 0.6),
            'radius': 0.9 * species.body_scale,
            'depth': 5.0,
            'torus_coupling': 0.35
        }
    
    if species.num_limbs >= 2:
        angle = np.pi / 6
        r = species.core_radius + 0.6
        basins['limb_1'] = {
            'pos': (-r * np.sin(angle), -r * np.cos(angle)),
            'radius': 0.9 * species.body_scale,
            'depth': 5.0,
            'torus_coupling': 0.35
        }
    
    if species.num_limbs >= 3:
        angle = np.pi / 6
        r = species.core_radius + 0.6
        basins['limb_2'] = {
            'pos': (r * np.sin(angle), -r * np.cos(angle)),
            'radius': 0.9 * species.body_scale,
            'depth': 5.0,
            'torus_coupling': 0.35
        }
    
    # Apex has 4 limbs - rear grasping limb
    if species.num_limbs >= 4:
        basins['limb_3'] = {
            'pos': (0.0, -(species.core_radius + 0.6)),
            'radius': 0.9 * species.body_scale,
            'depth': 5.0,
            'torus_coupling': 0.35
        }
    
    return basins


def get_species_limb_defs(species: SpeciesParams) -> dict:
    """
    Generate limb angle definitions for a species.
    
    Args:
        species: Species parameters
        
    Returns:
        Dict mapping limb names to (basin_name, base_angle) tuples
    """
    defs = {}
    
    if species.num_limbs >= 1:
        defs['limb_0'] = ('limb_0', np.pi/2)
    
    if species.num_limbs >= 2:
        defs['limb_1'] = ('limb_1', np.pi + np.pi/6)
    
    if species.num_limbs >= 3:
        defs['limb_2'] = ('limb_2', -np.pi/6)
    
    if species.num_limbs >= 4:
        defs['limb_3'] = ('limb_3', -np.pi/2)
    
    return defs


def get_random_species(exclude: List[str] = None) -> SpeciesParams:
    """
    Get a random species weighted by spawn weights.
    
    Args:
        exclude: List of species names to exclude
        
    Returns:
        Randomly selected SpeciesParams
    """
    if exclude is None:
        exclude = []
    
    available = [s for s in ALL_SPECIES if s.name not in exclude]
    if not available:
        return SPECIES_HERBIE
    
    weights = [SPECIES_SPAWN_WEIGHTS.get(s.name, 1.0) for s in available]
    total = sum(weights)
    weights = [w / total for w in weights]
    
    return np.random.choice(available, p=weights)
