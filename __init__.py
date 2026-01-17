"""
HERBIE World - Emergent Artificial Life Simulation

A multi-species ecosystem with NLSE-based neural dynamics,
KdV soliton signaling, and emergent behavior.

Usage:
    python -m herbie_world              # Run with visualization
    python -m herbie_world --overnight  # Headless mode

Package structure:
- core/: Constants, utilities, field operations
- brain/: TorusBrain, KdV channels, placement memory
- body/: BodyField, skeleton, morphology, limbs, metabolism
- creature/: Species definitions, creature class, behaviors
- world/: Terrain, objects, weather, seasons, day/night
- ecology/: Disease, predation, leviathan, meteors, ants
- chemistry/: Element system, constructions, sounds
- evolution/: History tracking, evolution tree
- persistence/: Save/load state management
- audio/: Input/output audio system
- visualization/: All rendering and display
- manager/: Creature orchestration
- events/: Event logging
- statistics/: Lifetime tracking, ecological balance
"""

__version__ = "17.26"  # D17Zv2B
__author__ = "Anderson"

from .main import main, main_visual, main_overnight

# Submodule imports
from . import (
    core, brain, body, creature, world, ecology, 
    chemistry, evolution, persistence, audio, 
    visualization, manager, events, statistics
)
