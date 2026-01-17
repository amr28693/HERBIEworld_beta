# HERBIE World - Emergent Artificial Life Simulation

Version 17.26 (D17Zv2B)

A multi-species artificial life ecosystem with NLSE-based neural dynamics, emergent behavior, and comprehensive world simulation.

## Overview

HERBIE World simulates an ecosystem of creatures with wave-equation-based neural systems (torus brains running NLSE dynamics), KdV soliton signaling channels, and emergent social behaviors. Creatures forage, reproduce, form families, build shelters, create art, and develop culture - all emerging from continuous field dynamics rather than rule-based AI.

## Installation

```bash
# Clone or extract the package
# Requires: Python 3.10+, numpy, matplotlib, scipy

# Optional dependencies for full features:
pip install sounddevice  # For audio I/O (optional)
```

## Usage

### Command Line

```bash
# Run with visualization (interactive mode)
python -m herbie_world

# Run headless for long simulations (no GUI)
python -m herbie_world --overnight

# Show help
python -m herbie_world --help
```

### From Python

```python
from herbie_world import main_visual, main_overnight

# Interactive mode with matplotlib visualization
main_visual()

# Headless mode - runs 50,000 steps, logs to console
main_overnight(steps=50000)
```

## Package Structure

```
herbie_world/                    17,725 lines
├── core/                        Constants, utilities
│   ├── constants.py             NLSE parameters, world size, time step
│   └── utils.py                 Statistical helpers, field operations
│
├── brain/                       Neural systems
│   ├── torus.py                 TorusBrain - NLSE on periodic domain
│   ├── kdv_channels.py          KdV soliton afferent/efferent signaling
│   └── placement_memory.py      Spatial memory field
│
├── body/                        Physical embodiment
│   ├── body_field.py            BodyField - NLSE with potential wells
│   ├── skeleton.py              PiezoSkeleton - mechanical feedback
│   ├── morphology.py            Morphology - limb coordination
│   ├── limbs.py                 LimbField - appendage dynamics
│   └── metabolism.py            Metabolism - energy, hunger, digestion
│
├── creature/                    Creature definitions
│   ├── species.py               SpeciesParams for Herbie, Blob, Apex, etc.
│   ├── traits.py                MutatedTraits for inheritance
│   ├── creature.py              Base Creature class
│   ├── herbie.py                HerbieWithHands - enhanced Herbie
│   ├── herbie_hands.py          GripperLimb, HerbieHands - tool use
│   └── herbie_social.py         Mating, naming, families
│
├── world/                       Environment
│   ├── terrain.py               Perlin noise terrain generation
│   ├── objects.py               WorldObject, MeatChunk, HerbieCorpse
│   ├── weather.py               Weather events (rain, drought, bloom)
│   ├── seasons.py               Four-season cycle
│   ├── day_night.py             Day/night cycle with lighting
│   ├── mycelia.py               Underground fungal network
│   ├── aquatic.py               Fish, kelp, coral ecosystems
│   └── multi_world.py           MultiWorld container
│
├── ecology/                     Ecological systems
│   ├── disease.py               Disease outbreaks and immunity
│   ├── leviathan.py             Apex predator / ecosystem balancer
│   ├── ants.py                  Ant colony simulation
│   └── emergent.py              Nests, digging, smears, culture, achievements
│
├── chemistry/                   Element system
│   ├── elements.py              ElementType enum, properties
│   ├── element_objects.py       ElementObject, ElementField, Construction
│   └── spawner.py               ElementSpawner, integration helpers
│
├── evolution/                   
│   └── tree.py                  EvolutionTree lineage tracking
│
├── audio/                       Sound system
│   ├── audio_system.py          AudioSystem - mic input, sonification output
│   └── soundscape.py            WorldSoundscape - ambient/creature sounds
│
├── visualization/               Display
│   ├── colors.py                Color mapping, sky colors
│   ├── renderers.py             Entity rendering functions
│   ├── panels.py                Status panels, charts
│   ├── chemistry_render.py      Element/construction rendering
│   ├── specialized_views.py     Evolution tree, art gallery, isometric
│   └── main_vis.py              HerbieVisualization main class
│
├── statistics/                  Analytics
│   └── __init__.py              LifetimeRecord, LifetimeTracker,
│                                EcologicalBalance, EvolutionHistory,
│                                create_lifetime_plot, create_evolution_summary_plot
│
├── persistence/                 Save/load
│   └── world_state.py           WorldPersistence serialization
│
├── manager/                     Orchestration
│   └── creature_manager.py      CreatureManager - main simulation loop
│
├── events/                      Logging
│   └── logger.py                EventLogger singleton
│
└── main.py                      Entry points
```

## Key Concepts

### NLSE-Based Neural Dynamics

Creatures use the Nonlinear Schrödinger Equation (NLSE) for both brain and body dynamics:

```
i∂ψ/∂t = -½∇²ψ + V(x)ψ + g|ψ|²ψ
```

This creates emergent attractor basins, solitons, and coherent wave packets that self-organize into meaningful neural patterns without explicit programming.

### KdV Soliton Signaling

Neural signals travel as KdV solitons through dedicated channels:
- **Afferent** (sensory): hunger, sound, touch, proximity, terrain, light
- **Efferent** (motor): limb activation, grip commands, vocalization

### Emergent Behaviors

All complex behaviors emerge from field dynamics:
- **Foraging**: Gradient following in placement memory
- **Bonding**: Torus phase resonance between creatures
- **Tool use**: Element manipulation through grip mechanics
- **Art**: Pigment smearing creates visual patterns
- **Shelter**: Construction building from elements
- **Culture**: Naming traditions and family bonds

## Species

| Species | Diet | Special Traits |
|---------|------|----------------|
| **Herbie** | Herbivore | Hands, social, tool use, families |
| **Blob** | Herbivore | Simple, efficient forager |
| **Biped** | Herbivore | Two-legged locomotion |
| **Mono** | Herbivore | Single-limbed rolling |
| **Scavenger** | Carnivore | Eats corpses |
| **Apex** | Carnivore | Active predator |

## Keyboard Controls (Visualization)

| Key | Action |
|-----|--------|
| ← → | Select creature |
| Space | Pause/resume |
| L | Toggle local/world view |
| H | Cycle Herbies only |
| A | Toggle ants |
| P | Toggle pheromones |
| M | Toggle mycelia |
| N | Toggle nests |
| S | Toggle smears |
| T | Evolution tree popup |
| G | Art gallery view |
| I | Isometric city view |

## Data Files

The simulation saves state to `data/`:
- `herbie_world_state.pkl` - Full world state
- `herbie_chemistry.json` - Element/construction state
- `herbie_evolution.json` - Lifetime records

## License

Research/educational use. Contact author for other uses.

## Author

Anderson (2024-2025)
