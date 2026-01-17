"""Ecology systems - disease, predation, leviathan, and emergent behaviors."""

from .disease import DiseaseStrain, DiseaseOutbreak, DiseaseSystem, DISEASE_STRAINS
from .leviathan import Leviathan, LeviathanSystem
from .ants import (
    ReactionDiffusionField, Ant, AntColony,
    herbie_sense_ants, herbie_can_raid_colony, herbie_raid_colony
)
from .emergent import (
    # Brain state tracking
    BrainSnapshot, BrainStateTracker,
    # Culture and territorial patterns
    CultureTracker,
    # Emergent homesteading
    HerbieNest, NestTracker,
    # Landscape manipulation
    Hole, DiggingSystem,
    # Pigment/art system
    SmearMark, SmearableObject, SmearSystem, spawn_pigment_object,
    # User favorites
    FavoriteHerbieTracker,
    # Ant interactions
    AntCreatureInteraction,
)
