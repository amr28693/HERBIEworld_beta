"""
LimbField - 1D NLSE field for each limb.

Each limb has its own wave dynamics that couple to the body field
and receive efferent motor commands from the brain.
"""

from typing import Tuple
import numpy as np

from ..core.constants import N_limb, L_op_limb, dt


class LimbField:
    """
    1D NLSE field for each limb.
    
    Limb fields:
    - Receive input from body field at the base
    - Receive efferent commands from brain
    - Generate momentum that influences limb angle
    - Track pulse position/amplitude for motor output
    """
    
    def __init__(self, name: str):
        """
        Initialize a limb field.
        
        Args:
            name: Limb identifier (e.g., 'limb_0')
        """
        self.name = name
        
        # Initialize with small random noise
        self.psi = 0.03 * (np.random.randn(N_limb) + 1j * np.random.randn(N_limb))
        
        self.g = 0.6  # Nonlinearity strength
        self.energy = 0.0
        self.pulse_position = 0.0  # 0-1, where energy is concentrated
        self.pulse_amplitude = 0.0
        
    def inject_from_body(self, amplitude: float, phase: float, arousal: float):
        """
        Inject energy from body field at limb base.
        
        Args:
            amplitude: Injection amplitude
            phase: Phase of body field at connection point
            arousal: Brain arousal level (modulates injection strength)
        """
        strength = 0.2 * (0.4 + arousal)
        inj = strength * amplitude * np.exp(1j * phase)
        for i in range(8):
            self.psi[i] += inj * np.exp(-i**2 / 6) * (1 + 0.3 * np.random.randn())
            
    def inject_efferent(self, amplitude: float):
        """
        Inject efferent motor command from brain.
        
        Args:
            amplitude: Command amplitude
        """
        inj = amplitude * 0.3 * np.exp(1j * np.random.uniform(0, 2*np.pi))
        center = N_limb // 3
        for i in range(-4, 5):
            idx = center + i
            if 0 <= idx < N_limb:
                self.psi[idx] += inj * np.exp(-i**2 / 4)
        
    def get_tip_state(self) -> Tuple[float, float]:
        """
        Get phase and amplitude at limb tip.
        
        Returns:
            (phase, amplitude) at tip region
        """
        tip_psi = self.psi[-10:]
        return float(np.angle(np.mean(tip_psi))), float(np.sqrt(np.mean(np.abs(tip_psi)**2)))
        
    def get_momentum(self) -> float:
        """
        Get net momentum (phase gradient weighted by intensity).
        
        Returns:
            Net momentum along limb
        """
        I = np.abs(self.psi)**2
        I_sum = np.sum(I) + 1e-12
        phase = np.angle(self.psi)
        dphase = np.gradient(phase)
        return float(np.sum(I * dphase) / I_sum)
        
    def evolve(self, arousal: float, hunger: float, dream_depth: float):
        """
        Evolve limb field dynamics.
        
        Args:
            arousal: Brain arousal level
            hunger: Hunger level
            dream_depth: Sleep/dream depth
        """
        # Update nonlinearity
        g_target = 0.4 + 0.6 * arousal - 0.2 * hunger - 0.3 * dream_depth
        self.g = 0.9 * self.g + 0.1 * g_target
        
        # Split-step NLSE
        psi_k = np.fft.fft(self.psi) * L_op_limb
        self.psi = np.fft.ifft(psi_k)
        self.psi *= np.exp(1j * self.g * dt * np.abs(self.psi)**2)
        
        # Boundary damping
        self.psi[:2] *= np.array([0.92, 0.96])
        self.psi[-4:] *= np.array([0.97, 0.94, 0.88, 0.8])
        
        # Global dissipation
        self.psi *= 0.996
        
        # Noise injection
        self.psi += 0.003 * (np.random.randn(N_limb) + 1j * np.random.randn(N_limb))
        
        # Energy regulation
        E = np.sum(np.abs(self.psi)**2)
        target_E = 5.0 * (0.5 + 0.8 * arousal)
        if E > 1e-12:
            self.psi *= np.clip((target_E / E) ** 0.05, 0.94, 1.06)
            
        # Update metrics
        I = np.abs(self.psi)**2
        self.energy = float(np.sum(I))
        
        if self.energy > 1e-10:
            self.pulse_position = float(np.sum(I * np.arange(N_limb)) / np.sum(I) / N_limb)
            self.pulse_amplitude = float(np.sqrt(np.max(I)))
        else:
            self.pulse_position = 0.0
            self.pulse_amplitude = 0.0
    
    def to_dict(self) -> dict:
        """Serialize limb field state."""
        return {
            'name': self.name,
            'psi_real': self.psi.real.tolist(),
            'psi_imag': self.psi.imag.tolist(),
            'g': self.g,
            'energy': self.energy,
            'pulse_position': self.pulse_position,
            'pulse_amplitude': self.pulse_amplitude,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'LimbField':
        """Deserialize limb field state."""
        limb = cls(d['name'])
        limb.psi = np.array(d['psi_real']) + 1j * np.array(d['psi_imag'])
        limb.g = d.get('g', 0.6)
        limb.energy = d.get('energy', 0.0)
        limb.pulse_position = d.get('pulse_position', 0.0)
        limb.pulse_amplitude = d.get('pulse_amplitude', 0.0)
        return limb
