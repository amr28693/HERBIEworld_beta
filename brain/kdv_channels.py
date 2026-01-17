"""
KdV Neural Channels - Soliton-based signaling.

KdV (Korteweg-de Vries) equation dynamics create soliton waves that
propagate along neural channels, carrying sensory and motor signals.
"""

from typing import List
from collections import deque
import numpy as np

from ..core.constants import (
    N_kdv, kdv_length, x_kdv, k_kdv, L_op_kdv, dt,
    AFFERENT_CHANNELS, EFFERENT_CHANNELS
)


class KdVChannel:
    """
    Soliton-based neural signaling channel using KdV dynamics.
    
    AFFERENT (sensory → brain):
    - touch_0, touch_1, touch_2: Limb-specific tactile sensation
    - env_reward: Positive reinforcement signals  
    - env_pain: Aversive/negative signals
    - proprioception: Internal body state awareness
    
    EFFERENT (brain → motor):
    - motor_0, motor_1, motor_2: Limb-specific motor commands
    
    Solitons propagate along the channel and trigger events at the destination.
    """
    
    def __init__(self, name: str, direction: str = 'afferent', description: str = ""):
        """
        Initialize a KdV channel.
        
        Args:
            name: Channel identifier (e.g., 'touch_0', 'motor_1')
            direction: 'afferent' (sensory→brain) or 'efferent' (brain→motor)
            description: Human-readable description
        """
        self.name = name
        self.direction = direction
        self.description = description
        self.u = np.zeros(N_kdv)  # Field state
        
        # Arrival tracking
        self.arrivals: List[float] = []
        self.arrival_times: List[float] = []
        
        # Activity statistics
        self.total_nucleations = 0
        self.total_arrivals = 0
        self.recent_activity = deque(maxlen=100)
        
        # Get color from channel definitions
        if direction == 'afferent' and name in AFFERENT_CHANNELS:
            self.color = AFFERENT_CHANNELS[name]['color']
        elif direction == 'efferent' and name in EFFERENT_CHANNELS:
            self.color = EFFERENT_CHANNELS[name]['color']
        else:
            self.color = 'white'
        
    def nucleate(self, amplitude: float, position_frac: float = 0.0):
        """
        Inject a soliton pulse into the channel.
        
        Args:
            amplitude: Pulse strength (0-2.5)
            position_frac: Starting position (0=near source, 1=near destination)
        """
        amplitude = np.clip(amplitude, 0, 2.5)
        if amplitude < 0.05:
            return
            
        width = 0.35 / (amplitude + 0.25)
        
        if self.direction == 'afferent':
            # Start near sensory end (low indices)
            pos = position_frac * kdv_length * 0.25
        else:
            # Start near brain end (high indices) for efferent
            pos = kdv_length * (0.75 + position_frac * 0.2)
            
        self.u += amplitude * np.exp(-((x_kdv - pos) / width)**2)
        self.total_nucleations += 1
        self.recent_activity.append(amplitude)
        
    def evolve(self):
        """KdV evolution with soliton dynamics."""
        self.u = np.clip(self.u, -4, 4)
        
        # Linear step (dispersion)
        u_k = np.fft.fft(self.u) * L_op_kdv
        self.u = np.fft.ifft(u_k).real
        
        # KdV nonlinearity: u * du/dx term
        du_dx = np.fft.ifft(1j * k_kdv * np.fft.fft(self.u)).real
        self.u -= 1.8 * self.u * du_dx * dt
        
        # Boundary damping
        self.u[:4] *= np.linspace(0.85, 1.0, 4)
        self.u[-4:] *= np.linspace(1.0, 0.85, 4)
        
        # Dissipation
        self.u *= 0.997
        
        # Check for arrivals at destination end
        if self.direction == 'afferent':
            dest = self.u[-12:]  # Brain end
        else:
            dest = self.u[:12]   # Motor end
            
        peak = np.max(np.abs(dest))
        if peak > 0.12:
            self.arrivals.append(float(peak))
            self.total_arrivals += 1
            
        # NaN protection
        if np.any(np.isnan(self.u)):
            self.u = np.zeros(N_kdv)
            
    def get_arrivals(self) -> List[float]:
        """Get and clear pending arrivals."""
        arr = self.arrivals.copy()
        self.arrivals = []
        return arr
        
    def get_activity(self) -> float:
        """Current total activity level."""
        return float(np.sum(np.abs(self.u)))
    
    def get_mean_activity(self) -> float:
        """Average recent activity."""
        if not self.recent_activity:
            return 0.0
        return float(np.mean(self.recent_activity))
    
    def get_stats(self) -> dict:
        """Return channel statistics."""
        return {
            'name': self.name,
            'direction': self.direction,
            'total_nucleations': self.total_nucleations,
            'total_arrivals': self.total_arrivals,
            'current_activity': self.get_activity(),
            'mean_recent_activity': self.get_mean_activity(),
            'arrival_rate': self.total_arrivals / (self.total_nucleations + 1),
        }
    
    def to_dict(self) -> dict:
        """Serialize channel state for persistence."""
        return {
            'name': self.name,
            'direction': self.direction,
            'description': self.description,
            'u': self.u.tolist(),
            'total_nucleations': self.total_nucleations,
            'total_arrivals': self.total_arrivals,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'KdVChannel':
        """Deserialize channel state from persistence."""
        channel = cls(
            name=d['name'],
            direction=d['direction'],
            description=d.get('description', '')
        )
        channel.u = np.array(d.get('u', np.zeros(N_kdv)))
        channel.total_nucleations = d.get('total_nucleations', 0)
        channel.total_arrivals = d.get('total_arrivals', 0)
        return channel


def create_afferent_channels() -> dict:
    """
    Create a complete set of afferent (sensory) channels.
    
    Returns:
        Dict mapping channel names to KdVChannel objects
    """
    channels = {}
    for name, props in AFFERENT_CHANNELS.items():
        channels[name] = KdVChannel(
            name=name,
            direction='afferent',
            description=props['desc']
        )
    return channels


def create_efferent_channels() -> dict:
    """
    Create a complete set of efferent (motor) channels.
    
    Returns:
        Dict mapping channel names to KdVChannel objects
    """
    channels = {}
    for name, props in EFFERENT_CHANNELS.items():
        channels[name] = KdVChannel(
            name=name,
            direction='efferent',
            description=props['desc']
        )
    return channels
