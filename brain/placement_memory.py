"""
Placement Memory Field - Emergent caching substrate.

KdV-based spatial memory field for placement locations (D17Zv2 feature).
This enables emergent food caching behavior without explicit coding.
"""

from typing import Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np

from ..core.constants import N_pmem, theta_pmem, k_pmem, dt


@dataclass
class PlacementMemoryParams:
    """
    Individual variation in placement memory dynamics.
    These are heritable traits that affect caching-like behavior emergence.
    
    Different parameter combinations lead to different "personalities":
    - High alpha + low gamma = strong, persistent memories
    - Low base_coupling = less influenced by cached locations
    - High satiation_gate = only caches when very well-fed
    """
    # KdV evolution parameters
    alpha: float = 1.8       # Nonlinearity strength (soliton formation)
    beta: float = 0.08       # Dispersion coefficient
    gamma: float = 0.002     # Dissipation rate (memory decay)
    
    # Coupling parameters
    nucleation_strength: float = 1.0    # How strongly placements imprint
    base_coupling: float = 0.25         # Base coupling to torus (hunger modulates this)
    
    # Behavioral thresholds (soft, not hard gates)
    satiation_gate: float = 0.3         # Below this, placements don't imprint well
    
    @classmethod
    def random(cls) -> 'PlacementMemoryParams':
        """Generate individual with random variation."""
        return cls(
            alpha=np.random.uniform(1.2, 2.4),
            beta=np.random.uniform(0.05, 0.12),
            gamma=np.random.uniform(0.001, 0.004),
            nucleation_strength=np.random.uniform(0.5, 1.5),
            base_coupling=np.random.uniform(0.15, 0.4),
            satiation_gate=np.random.uniform(0.2, 0.45),
        )
    
    @classmethod
    def inherit(cls, parent: 'PlacementMemoryParams', mutation_rate: float = 0.15) -> 'PlacementMemoryParams':
        """Inherit parameters from parent with possible mutations."""
        child_params = {}
        for field_name in ['alpha', 'beta', 'gamma', 'nucleation_strength', 'base_coupling', 'satiation_gate']:
            parent_val = getattr(parent, field_name)
            if np.random.random() < mutation_rate:
                # Mutate by ±12%
                child_params[field_name] = parent_val * np.random.uniform(0.88, 1.12)
            else:
                child_params[field_name] = parent_val
        return cls(**child_params)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'nucleation_strength': self.nucleation_strength,
            'base_coupling': self.base_coupling,
            'satiation_gate': self.satiation_gate,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PlacementMemoryParams':
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PlacementMemoryField:
    """
    KdV-based spatial memory field for placement locations.
    
    When a Herbie drops an object while satiated, a soliton is nucleated
    in this field at the angular direction of the drop. The soliton
    persists and evolves via KdV dynamics, creating a "memory" of where
    things were placed.
    
    The field couples to the torus brain with hunger-modulated gain:
    - Low hunger: weak coupling (no need to return, food is available)
    - High hunger: strong coupling (memories pull on attention/movement)
    
    Individual variation in parameters means some Herbies will develop
    strong "caching" behavior and others won't - this is emergent, not coded.
    """
    
    def __init__(self, params: PlacementMemoryParams = None):
        """
        Initialize placement memory field.
        
        Args:
            params: Individual variation parameters (random if not provided)
        """
        self.params = params or PlacementMemoryParams.random()
        
        # The field itself - angular representation of placement directions
        self.M = np.zeros(N_pmem)
        
        # Reference position (updated periodically to Herbie's "home" or center of range)
        self.reference_pos = np.zeros(2)
        
        # Statistics
        self.total_nucleations = 0
        self.total_retrievals = 0  # Times the field strongly influenced movement
        self.peak_amplitude_history = deque(maxlen=500)
    
    def nucleate(self, drop_direction: float, satiation: float, object_value: float = 1.0):
        """
        Nucleate a soliton at the angular direction of a placement.
        
        Args:
            drop_direction: Angle in radians (world coords relative to reference)
            satiation: Current satiation level - gates imprint strength
            object_value: Importance/energy of dropped object (optional weighting)
        """
        # Satiation gates nucleation strength (soft gate, not hard cutoff)
        sat_factor = np.clip(
            (satiation - self.params.satiation_gate) / (1 - self.params.satiation_gate), 
            0, 1
        )
        if sat_factor < 0.05:
            return  # Too hungry/stressed to form lasting memory
        
        # Amplitude scales with satiation and object value
        amplitude = sat_factor * object_value * self.params.nucleation_strength
        
        # Width of initial pulse (narrower = more coherent soliton, but amplitude-dependent)
        width = 0.4 / (amplitude + 0.3)
        
        # Wrap direction to [0, 2π]
        theta = drop_direction % (2 * np.pi)
        
        # Add gaussian pulse - handle wraparound
        dist = np.abs(theta_pmem - theta)
        dist = np.minimum(dist, 2*np.pi - dist)  # Periodic distance
        self.M += amplitude * np.exp(-(dist / width) ** 2)
        
        self.total_nucleations += 1
    
    def evolve(self):
        """
        KdV evolution step for the placement memory field.
        
        Solitons persist but slowly decay; they can interfere constructively
        if multiple placements happen in similar directions.
        """
        # Clip for stability
        self.M = np.clip(self.M, -3, 3)
        
        # Linear step (dispersion) - spectral method
        M_k = np.fft.fft(self.M)
        # KdV dispersion: exp(-i * beta * k^3 * dt)
        L_op = np.exp(-1j * self.params.beta * k_pmem**3 * dt)
        M_k = M_k * L_op
        self.M = np.fft.ifft(M_k).real
        
        # Nonlinear step: KdV nonlinearity u * du/dx
        dM_dtheta = np.fft.ifft(1j * k_pmem * np.fft.fft(self.M)).real
        self.M -= self.params.alpha * self.M * dM_dtheta * dt
        
        # Dissipation (memory decay)
        self.M *= (1 - self.params.gamma)
        
        # Track peak amplitude
        self.peak_amplitude_history.append(np.max(np.abs(self.M)))
        
        # NaN protection
        if np.any(np.isnan(self.M)):
            self.M = np.zeros(N_pmem)
    
    def get_direction_bias(self, hunger: float) -> Tuple[float, float]:
        """
        Get directional bias from memory field for torus coupling.
        
        This doesn't tell the Herbie to "go to cache" - it just biases
        the torus dynamics toward remembered placement directions when hungry.
        
        Args:
            hunger: Current hunger level (0-1)
            
        Returns:
            Tuple of (direction, strength) where:
            - direction: angle of strongest memory pull (radians)
            - strength: coupling strength (hunger-modulated)
        """
        # Hunger-modulated coupling: quadratic so it really kicks in when desperate
        coupling = self.params.base_coupling * (0.05 + 0.95 * hunger ** 2)
        
        # Find peak direction (weighted circular mean for smoothness)
        M_positive = np.maximum(self.M, 0)
        total_weight = np.sum(M_positive) + 1e-12
        
        if total_weight < 0.1:
            return 0.0, 0.0  # No significant memories
        
        # Circular mean direction
        x_component = np.sum(M_positive * np.cos(theta_pmem)) / total_weight
        y_component = np.sum(M_positive * np.sin(theta_pmem)) / total_weight
        direction = np.arctan2(y_component, x_component)
        
        # Strength based on peak amplitude and coupling
        strength = np.max(M_positive) * coupling
        
        if strength > 0.1:
            self.total_retrievals += 1
        
        return float(direction), float(strength)
    
    def update_reference(self, new_pos: np.ndarray, smoothing: float = 0.02):
        """
        Slowly update reference position (Herbie's 'home base' drifts over time).
        
        Args:
            new_pos: Current position
            smoothing: Exponential smoothing factor (lower = slower drift)
        """
        self.reference_pos = smoothing * new_pos + (1 - smoothing) * self.reference_pos
    
    def world_to_memory_angle(self, world_pos: np.ndarray, from_pos: np.ndarray) -> float:
        """
        Convert world position to angular direction relative to reference.
        
        Args:
            world_pos: Target position in world coordinates
            from_pos: Origin position
            
        Returns:
            Angle in radians
        """
        direction = world_pos - from_pos
        return np.arctan2(direction[1], direction[0])
    
    def get_activity(self) -> float:
        """Current total activity level."""
        return float(np.sum(np.abs(self.M)))
    
    def get_peak(self) -> float:
        """Current peak amplitude."""
        return float(np.max(np.abs(self.M)))
    
    def get_stats(self) -> dict:
        """Return field statistics."""
        return {
            'total_nucleations': self.total_nucleations,
            'total_retrievals': self.total_retrievals,
            'current_activity': self.get_activity(),
            'peak_amplitude': self.get_peak(),
            'mean_peak': float(np.mean(self.peak_amplitude_history)) if self.peak_amplitude_history else 0,
        }
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            'M': self.M.tolist(),
            'reference_pos': self.reference_pos.tolist(),
            'params': self.params.to_dict(),
            'total_nucleations': self.total_nucleations,
            'total_retrievals': self.total_retrievals,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PlacementMemoryField':
        """Deserialize from dictionary."""
        params = PlacementMemoryParams.from_dict(d.get('params', {}))
        field = cls(params)
        field.M = np.array(d.get('M', np.zeros(N_pmem)))
        field.reference_pos = np.array(d.get('reference_pos', [0, 0]))
        field.total_nucleations = d.get('total_nucleations', 0)
        field.total_retrievals = d.get('total_retrievals', 0)
        return field
