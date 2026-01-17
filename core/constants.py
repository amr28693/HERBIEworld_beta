"""
Core constants for HERBIE World simulation.

This module contains all grid parameters, field operators, and fundamental
constants used throughout the simulation. These are computed once at import
time for efficiency.
"""

import os
import numpy as np

# =============================================================================
# ERROR HANDLING
# =============================================================================
np.seterr(divide='ignore', invalid='ignore', over='ignore')

# =============================================================================
# PATHS
# =============================================================================
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels from core/ to get to the actual data directory
    # In practice, this should be set by the main script
    SCRIPT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
except:
    SCRIPT_DIR = os.getcwd()

# File paths - these may be overridden by main script
STATE_FILE = os.path.join(SCRIPT_DIR, "herbie_state.npz")
PREFS_FILE = os.path.join(SCRIPT_DIR, "herbie_learned.json")
EVOLUTION_FILE = os.path.join(SCRIPT_DIR, "herbie_evolution.json")
GENERATIONS_DIR = os.path.join(SCRIPT_DIR, "generations")
ANALYTICS_DIR = os.path.join(SCRIPT_DIR, "analytics")
EVENT_LOG_FILE = os.path.join(SCRIPT_DIR, "event_log.jsonl")
HERBIE_MULTI_STATE = os.path.join(SCRIPT_DIR, "herbie_multi_state.npz")
WORLD_STATE_FILE = os.path.join(SCRIPT_DIR, "world_state_D17k.npz")
CHEMISTRY_FILE = os.path.join(SCRIPT_DIR, "herbie_chemistry.json")


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    os.makedirs(GENERATIONS_DIR, exist_ok=True)
    os.makedirs(ANALYTICS_DIR, exist_ok=True)


# =============================================================================
# TIME STEPS
# =============================================================================
dt = 0.015          # Simulation timestep
VIZ_DT = 0.035      # Visualization timestep


# =============================================================================
# BODY FIELD (2D NLSE)
# =============================================================================
BODY_L = 14.0
Nx, Ny = 96, 96
dx = BODY_L / Nx

# Spatial grids
x = np.linspace(-BODY_L/2, BODY_L/2, Nx, endpoint=False)
y = np.linspace(-BODY_L/2, BODY_L/2, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)

# Spectral grids
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dx)
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

# Linear propagator for 2D NLSE
L_op_2d = np.exp(-1j * K2 * dt / 2)


# =============================================================================
# TORUS BRAIN (1D Ring NLSE)
# =============================================================================
N_torus = 64
theta_torus = np.linspace(0, 2*np.pi, N_torus, endpoint=False)
k_torus = np.fft.fftfreq(N_torus, d=(2*np.pi/N_torus)/(2*np.pi)) * 2*np.pi
L_op_torus = np.exp(-1j * k_torus**2 * dt / 2)


# =============================================================================
# LIMB FIELDS (1D NLSE)
# =============================================================================
N_limb = 48
limb_length = 4.0
dx_limb = limb_length / N_limb
x_limb = np.linspace(0, limb_length, N_limb, endpoint=False)
k_limb = np.fft.fftfreq(N_limb, d=dx_limb) * 2*np.pi
L_op_limb = np.exp(-1j * 0.5 * k_limb**2 * dt / 2)


# =============================================================================
# KdV NEURAL CHANNELS
# =============================================================================
N_kdv = 48
kdv_length = 6.0
dx_kdv = kdv_length / N_kdv
x_kdv = np.linspace(0, kdv_length, N_kdv, endpoint=False)
k_kdv = np.fft.fftfreq(N_kdv, d=dx_kdv) * 2*np.pi
kdv_damp = np.exp(-0.008 * k_kdv**2)
L_op_kdv = np.exp(-1j * k_kdv**3 * dt * 0.08) * kdv_damp


# =============================================================================
# PLACEMENT MEMORY FIELD (D17Zv2) - Emergent Caching Substrate
# =============================================================================
N_pmem = 48  # Same resolution as KdV
theta_pmem = np.linspace(0, 2*np.pi, N_pmem, endpoint=False)
k_pmem = np.fft.fftfreq(N_pmem, d=2*np.pi/N_pmem) * 2*np.pi


# =============================================================================
# WORLD PARAMETERS
# =============================================================================
WORLD_L = 100.0         # World size (updated for multi-creature)
WORLD_L_K = 100         # Terrain world size
TERRAIN_RESOLUTION = 100  # 100x100 terrain grid


# =============================================================================
# SMOOTHING / DYNAMICS
# =============================================================================
G_SMOOTH = 0.92


# =============================================================================
# AUDIO PARAMETERS
# =============================================================================
SAMPLE_RATE = 44100
BLOCK_IN = 2048
BLOCK_OUT = 1024


# =============================================================================
# MORPHOLOGY CONSTANTS
# =============================================================================
BASINS = {
    'core':   {'pos': (0.0, 0.0),   'radius': 2.2, 'depth': 12.0, 'torus_coupling': 0.7},
    'limb_0': {'pos': (0.0, 2.8),   'radius': 0.9, 'depth': 5.0,  'torus_coupling': 0.35},
    'limb_1': {'pos': (-2.4, -1.4), 'radius': 0.9, 'depth': 5.0,  'torus_coupling': 0.35},
    'limb_2': {'pos': (2.4, -1.4),  'radius': 0.9, 'depth': 5.0,  'torus_coupling': 0.35},
}

LIMB_DEFS = {
    'limb_0': ('limb_0', np.pi/2),         # Top limb
    'limb_1': ('limb_1', np.pi + np.pi/6), # Bottom-left
    'limb_2': ('limb_2', -np.pi/6),        # Bottom-right
}


# =============================================================================
# NEURAL CHANNEL DEFINITIONS
# =============================================================================

# Afferent channel definitions (sensory → brain)
AFFERENT_CHANNELS = {
    'touch_0':       {'color': 'cyan',    'desc': 'Limb 0 tactile'},
    'touch_1':       {'color': 'lime',    'desc': 'Limb 1 tactile'},
    'touch_2':       {'color': 'orange',  'desc': 'Limb 2 tactile'},
    'env_reward':    {'color': 'yellow',  'desc': 'Positive reinforcement'},
    'env_pain':      {'color': 'red',     'desc': 'Aversive signals'},
    'proprioception':{'color': 'magenta', 'desc': 'Body state awareness'},
}

# Efferent channel definitions (brain → motor)
EFFERENT_CHANNELS = {
    'motor_0': {'color': 'red',     'desc': 'Limb 0 motor', 'target': 'limb_0'},
    'motor_1': {'color': 'crimson', 'desc': 'Limb 1 motor', 'target': 'limb_1'},
    'motor_2': {'color': 'darkred', 'desc': 'Limb 2 motor', 'target': 'limb_2'},
}


# =============================================================================
# SEASON SYSTEM
# =============================================================================
SEASONS = {
    'spring': {'growth_rate': 1.5,  'temp_mod': 1.0, 'storm_prob': 0.15, 'day_length': 0.5},
    'summer': {'growth_rate': 2.0,  'temp_mod': 1.3, 'storm_prob': 0.08, 'day_length': 0.65},
    'autumn': {'growth_rate': 0.8,  'temp_mod': 0.9, 'storm_prob': 0.20, 'day_length': 0.45},
    'winter': {'growth_rate': 0.2,  'temp_mod': 0.5, 'storm_prob': 0.25, 'day_length': 0.35},
}
SEASON_ORDER = ['spring', 'summer', 'autumn', 'winter']
SEASON_LENGTH = 2000  # Steps per season


# =============================================================================
# MUTATION SYSTEM
# =============================================================================
MUTABLE_TRAITS = {
    'speed_factor':      (0.3, 2.5),
    'energy_efficiency': (0.5, 2.0),
    'sense_range':       (5.0, 40.0),
    'metabolism_rate':   (0.5, 2.0),
    'body_scale':        (0.5, 2.0),
}
MUTATION_RATE = 0.15      # 15% chance per trait
MUTATION_MAGNITUDE = 0.08  # ±8% drift


# =============================================================================
# DIGESTION SYSTEM
# =============================================================================
DIGESTION_DURATION = {
    'small': 80,
    'medium': 150,
    'large': 250,
}
DIGESTION_SPEED_PENALTY = 0.3  # Move at 30% speed while digesting


# =============================================================================
# LEVIATHAN SYSTEM
# =============================================================================
LEVIATHAN_MYTHIC_INTERVAL = 8000
LEVIATHAN_PREDATOR_THRESHOLD = 5
LEVIATHAN_PREDATOR_CHECK_INTERVAL = 500
LEVIATHAN_MIN_INTERVAL = 3000
LEVIATHAN_SIZE = 12.0
LEVIATHAN_SPEED = 0.15
LEVIATHAN_DURATION = 350
LEVIATHAN_HUNT_RANGE = 25.0
LEVIATHAN_KILL_RANGE = 8.0
LEVIATHAN_FERTILIZE_INTERVAL = 50
LEVIATHAN_FERTILIZE_RADIUS = 12.0
LEVIATHAN_COLOR_BASE = '#8B0000'
LEVIATHAN_COLOR_GLOW = '#FF4500'


# =============================================================================
# HERBIE-SPECIFIC PARAMETERS
# =============================================================================
HERBIE_AUDIO_MULTIPLIER = 1.8

HERBIE_SEX_COLORS = {
    'male':   '#4A90D9',   # Blue
    'female': '#D94A90',   # Pink
    'child':  '#90D94A',   # Green (juveniles)
}

# Courtship and mating
HERBIE_COURTSHIP_DISTANCE = 8.0
HERBIE_COURTSHIP_DURATION = 150
HERBIE_MATING_HUNGER_MAX = 0.4
HERBIE_MATING_MIN_AGE = 250

# Gestation
HERBIE_GESTATION_DURATION = 400
HERBIE_PREGNANCY_SPEED_PENALTY = 0.6
HERBIE_PREGNANCY_HUNGER_MULT = 1.5

# Juveniles
HERBIE_JUVENILE_DURATION = 600
HERBIE_JUVENILE_SCALE = 0.6
HERBIE_JUVENILE_FEED_RANGE = 10.0
HERBIE_PARENTAL_FEED_RATE = 0.3

# Family dynamics
HERBIE_FAMILY_COHESION_RANGE = 20.0
HERBIE_DEFENSE_BONUS_FAMILY = 0.3

# Resonance / bonding
HERBIE_RESONANCE_THRESHOLD = 0.5
HERBIE_RESONANCE_BOND_THRESHOLD = 0.7
HERBIE_RESONANCE_CHECK_DISTANCE = 15.0
HERBIE_RESONANCE_MEMORY = 5

# Names
HERBIE_NAMES_MALE = [
    # Globally diverse - see herbie_social.py for full categorized list
    "Atlas", "Chen", "Akira", "Boris", "Jabari", "Diego", "Arjun", "Omar", "Fionn",
    "Wei", "Hiro", "Dmitri", "Kofi", "Marco", "Dev", "Hassan", "Cian", "Moss",
]

HERBIE_NAMES_FEMALE = [
    # Globally diverse - see herbie_social.py for full categorized list  
    "Aurora", "Mei", "Hana", "Mila", "Zuri", "Sofia", "Priya", "Layla", "Niamh",
    "Lan", "Sakura", "Katya", "Amara", "Lucia", "Maya", "Yasmin", "Aoife", "Luna",
]


# =============================================================================
# ELEMENT / CHEMISTRY SYSTEM
# =============================================================================
APEX_BANE_DAMAGE_MULTIPLIER = 5.0
APEX_BANE_FEAR_RADIUS = 8.0


# =============================================================================
# VISUALIZATION TOGGLES
# =============================================================================
CHEMISTRY_SHOW_FIELD = False
CHEMISTRY_SHOW_CONCENTRATIONS = False
