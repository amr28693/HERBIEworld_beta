"""
Utility functions for HERBIE World simulation.

Statistical helpers, field operations, and color mapping.
"""

import numpy as np
import matplotlib.colors as mcolors


def complex_to_rgb(psi, amp_gamma=0.8):
    """
    Convert complex field to RGB using HSV color mapping.
    
    Phase → Hue, Amplitude → Value
    
    Args:
        psi: Complex numpy array
        amp_gamma: Gamma correction for amplitude (default 0.8)
    
    Returns:
        RGB array with shape (*psi.shape, 3)
    """
    amp = np.abs(psi)
    amp_n = np.clip(amp / (np.percentile(amp, 99) + 1e-12), 0, 1) ** amp_gamma
    hue = (np.angle(psi) + np.pi) / (2 * np.pi)
    hsv = np.zeros((*psi.shape, 3))
    hsv[..., 0] = hue
    hsv[..., 1] = 0.85
    hsv[..., 2] = amp_n
    return mcolors.hsv_to_rgb(hsv)


def safe_corrcoef(a, b):
    """
    Compute correlation coefficient with NaN protection.
    
    Returns 0.0 if:
    - Either array has fewer than 10 elements
    - Either array has near-zero variance
    - Computation fails for any reason
    
    Args:
        a, b: Array-like sequences
    
    Returns:
        float: Pearson correlation coefficient, or 0.0 on failure
    """
    if len(a) < 10 or len(b) < 10:
        return 0.0
    a, b = np.array(a), np.array(b)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    try:
        corr = np.corrcoef(a, b)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    except:
        return 0.0


def compute_entropy(distribution):
    """
    Compute Shannon entropy of a distribution.
    
    H = -Σ p_i * log(p_i)
    
    Args:
        distribution: Array-like of non-negative values
    
    Returns:
        float: Shannon entropy in nats
    """
    p = np.array(distribution, dtype=float)
    p = p / (np.sum(p) + 1e-12)
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))


def compute_field_entropy(psi):
    """
    Compute entropy of |ψ|² distribution (quantum probability density).
    
    Args:
        psi: Complex numpy array (wavefunction)
    
    Returns:
        float: Shannon entropy of probability density
    """
    I = np.abs(psi.flatten())**2
    I = I / (np.sum(I) + 1e-12)
    I = I[I > 1e-12]
    return float(-np.sum(I * np.log(I)))


def wrap_angle(angle):
    """
    Wrap angle to [-π, π].
    
    Args:
        angle: Angle in radians
    
    Returns:
        Wrapped angle in [-π, π]
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def circular_mean(angles, weights=None):
    """
    Compute circular (angular) mean.
    
    Args:
        angles: Array of angles in radians
        weights: Optional weights (default: uniform)
    
    Returns:
        tuple: (mean_angle, concentration)
            mean_angle: Circular mean in radians
            concentration: Resultant length (0-1), higher = more concentrated
    """
    if weights is None:
        weights = np.ones_like(angles)
    weights = np.array(weights) / (np.sum(weights) + 1e-12)
    
    x = np.sum(weights * np.cos(angles))
    y = np.sum(weights * np.sin(angles))
    
    mean_angle = np.arctan2(y, x)
    concentration = np.sqrt(x**2 + y**2)
    
    return float(mean_angle), float(concentration)


def smooth_step(x, edge0=0.0, edge1=1.0):
    """
    Hermite interpolation (smooth step function).
    
    Args:
        x: Input value
        edge0: Lower edge (returns 0 below this)
        edge1: Upper edge (returns 1 above this)
    
    Returns:
        Smoothly interpolated value in [0, 1]
    """
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


def periodic_distance(a, b, period=2*np.pi):
    """
    Compute distance on a periodic domain.
    
    Args:
        a, b: Positions
        period: Period of the domain
    
    Returns:
        Shortest distance accounting for periodicity
    """
    diff = np.abs(a - b)
    return np.minimum(diff, period - diff)


def sigmoid(x):
    """
    Numerically stable sigmoid function.
    
    Args:
        x: Input value or array
    
    Returns:
        Sigmoid of x
    """
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


def get_latitude_climate_modifier(y: float, world_size: float = 100.0) -> float:
    """
    Get climate modifier based on latitude (y position).
    
    Poles (y near ±world_size/2) are colder.
    Equator (y near 0) is warmer.
    
    Args:
        y: Y coordinate
        world_size: Size of the world
    
    Returns:
        Climate modifier (0 = cold, 1 = warm)
    """
    normalized_y = abs(y) / (world_size / 2)
    return 1.0 - 0.5 * normalized_y


def apply_terrain_movement(creature, terrain, force, dt: float):
    """
    Apply movement with terrain effects.
    
    - Movement cost slows creature
    - Impassable terrain blocks movement
    
    Args:
        creature: Creature with pos, vel attributes
        terrain: Terrain system
        force: Force vector to apply
        dt: Time step
    
    Returns:
        tuple: (new_pos, new_vel)
    """
    current_terrain = terrain.get_terrain_at(creature.pos)
    movement_cost = current_terrain.movement_cost
    
    # Scale force by terrain
    effective_force = force / movement_cost
    
    # Calculate new position
    new_vel = creature.vel + effective_force * dt * 20
    new_vel *= 0.96  # Damping
    new_pos = creature.pos + new_vel * dt
    
    # Check if new position is passable
    if terrain.is_passable(new_pos):
        return new_pos, new_vel
    else:
        # Slide along obstacle
        # Try x-only movement
        test_pos = np.array([new_pos[0], creature.pos[1]])
        if terrain.is_passable(test_pos):
            return test_pos, np.array([new_vel[0], 0])
        
        # Try y-only movement
        test_pos = np.array([creature.pos[0], new_pos[1]])
        if terrain.is_passable(test_pos):
            return test_pos, np.array([0, new_vel[1]])
        
        # Blocked completely
        return creature.pos, np.zeros(2)
