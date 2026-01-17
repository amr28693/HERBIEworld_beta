"""
BodyField - 2D NLSE field representing the creature's body.

The body field is where all physical dynamics happen - it's the
"substance" of the creature that interacts with the world.
"""

from typing import Tuple, TYPE_CHECKING
import numpy as np

from ..core.constants import (
    BASINS, BODY_L, Nx, Ny, X, Y, dx,
    L_op_2d, G_SMOOTH, dt
)

if TYPE_CHECKING:
    from .morphology import Morphology


class BodyField:
    """
    2D NLSE field representing the creature's body.
    
    The body field:
    - Is confined by basin attractors and skin boundary
    - Responds to external potentials (food, obstacles)
    - Generates momentum for movement
    - Couples to torus brain via phase and arousal
    """
    
    def __init__(self, morph: 'Morphology'):
        """
        Initialize body field.
        
        Args:
            morph: Morphology defining body shape
        """
        self.morph = morph
        
        # Initialize with Gaussian at core
        core = BASINS['core']['pos']
        self.psi = np.exp(-((X - core[0])**2 + (Y - core[1])**2) / 3.5).astype(np.complex128)
        self.psi += 0.015 * (np.random.randn(Ny, Nx) + 1j * np.random.randn(Ny, Nx))
        
        self.g = 0.5  # Nonlinearity strength
        self.energy = float(np.sum(np.abs(self.psi)**2))
        self.E_target = self.energy
        self.containment = 1.0  # How much density is inside skin
        
    def get_region_state(self, region_name: str) -> Tuple[float, float]:
        """
        Get phase and amplitude in a named body region (basin).
        
        Args:
            region_name: Basin name (e.g., 'core', 'limb_0')
            
        Returns:
            (phase, amplitude) in that region
        """
        info = BASINS.get(region_name)
        if info is None:
            return 0.0, 0.0
        bx, by = info['pos']
        mask = ((X - bx)**2 + (Y - by)**2) < info['radius']**2
        if np.sum(mask) < 5:
            return 0.0, 0.0
        psi_r = self.psi[mask]
        return float(np.angle(np.mean(psi_r))), float(np.sqrt(np.mean(np.abs(psi_r)**2)))
        
    def inject_at_region(self, region_name: str, amplitude: float, phase: float):
        """
        Inject energy at a named body region.
        
        Args:
            region_name: Basin name
            amplitude: Injection amplitude
            phase: Injection phase
        """
        info = BASINS.get(region_name)
        if info is None:
            return
        bx, by = info['pos']
        self.psi += 0.04 * amplitude * np.exp(-((X-bx)**2 + (Y-by)**2) / info['radius']**2) * np.exp(1j * phase)
        
    def get_momentum(self) -> np.ndarray:
        """
        Get net momentum from phase gradients.
        
        Returns:
            2D momentum vector
        """
        I = np.abs(self.psi)**2
        I_sum = np.sum(I) + 1e-12
        phase = np.angle(self.psi)
        px = np.sum(I * np.gradient(phase, axis=1)) / I_sum
        py = np.sum(I * np.gradient(phase, axis=0)) / I_sum
        if np.isnan(px) or np.isnan(py):
            return np.array([0.0, 0.0])
        return np.array([float(px), float(py)])
    
    def inject_momentum(self, direction: np.ndarray, strength: float):
        """
        Inject directional momentum via phase tilt.
        
        Args:
            direction: 2D direction vector
            strength: Injection strength
        """
        mag = np.linalg.norm(direction)
        if mag < 1e-6:
            return
        dx_n, dy_n = direction[0]/mag, direction[1]/mag
        phase_tilt = strength * 0.3 * (dx_n * X + dy_n * Y) / BODY_L
        self.psi *= np.exp(1j * phase_tilt)
        
    def evolve(self, V_total: np.ndarray, torus_phase: float, torus_arousal: float, 
               metabolic_g: float, dream_depth: float, audio_amp: float):
        """
        Evolve body field dynamics for one timestep.
        
        Args:
            V_total: Combined potential (basins + objects + boundaries)
            torus_phase: Phase from torus brain
            torus_arousal: Arousal level from torus
            metabolic_g: Nonlinearity modifier from metabolism
            dream_depth: Sleep/dream depth
            audio_amp: Audio amplitude
        """
        # Update nonlinearity
        g_base = 0.3
        g_target = g_base + 0.5 * torus_arousal + metabolic_g + 0.4 * audio_amp - 0.4 * dream_depth
        self.g = G_SMOOTH * self.g + (1 - G_SMOOTH) * g_target
        
        # Scale potential
        V_scaled = V_total * 0.4
        
        # Split-step NLSE
        psi_k = np.fft.fft2(self.psi) * L_op_2d
        self.psi = np.fft.ifft2(psi_k)
        self.psi *= np.exp(1j * self.g * dt * np.abs(self.psi)**2)
        self.psi *= np.exp(-1j * V_scaled * dt)
        
        # Torus-body coupling
        if torus_arousal > 0.08:
            phase_kick = torus_phase + 0.3 * np.sin(2 * np.pi * X / BODY_L) 
            self.psi *= np.exp(1j * 0.015 * torus_arousal * phase_kick)
            
        # Audio response
        if audio_amp > 0.01:
            core = BASINS['core']['pos']
            r_core = np.sqrt((X - core[0])**2 + (Y - core[1])**2)
            self.psi *= (1 + 0.02 * audio_amp * np.exp(-r_core**2 / 4))
            if audio_amp > 0.02:
                ripple_phase = audio_amp * 5 * np.sin(r_core * 1.5 - self.energy * 0.01)
                self.psi *= np.exp(1j * 0.01 * ripple_phase)
                
        # Global dissipation
        self.psi *= 0.9985
        
        # Noise injection (confined to body interior)
        skin = self.morph.compute_skin()
        noise_amp = 0.0008 * (1 + 0.5 * torus_arousal + audio_amp)
        self.psi += noise_amp * (np.random.randn(Ny, Nx) + 1j * np.random.randn(Ny, Nx)) * np.exp(-skin / 4)
        
        # Energy regulation
        E = np.sum(np.abs(self.psi)**2)
        drive = 1.0 + 0.4 * torus_arousal + 0.3 * audio_amp - 0.15 * dream_depth
        self.E_target = 0.95 * self.E_target + 0.05 * (400 * drive)
        if E > 1e-9:
            self.psi *= np.clip((self.E_target / E) ** 0.04, 0.96, 1.04)
            
        # Update metrics
        self.energy = float(np.sum(np.abs(self.psi)**2))
        I = np.abs(self.psi)**2
        self.containment = float(np.sum(I[skin < 1.5]) / (np.sum(I) + 1e-12))
    
    def get_intensity(self) -> np.ndarray:
        """Get intensity field |ψ|²."""
        return np.abs(self.psi)**2
    
    def to_dict(self) -> dict:
        """Serialize body field state."""
        return {
            'psi_real': self.psi.real.tolist(),
            'psi_imag': self.psi.imag.tolist(),
            'g': self.g,
            'energy': self.energy,
            'E_target': self.E_target,
            'containment': self.containment,
        }
    
    @classmethod
    def from_dict(cls, d: dict, morph: 'Morphology') -> 'BodyField':
        """Deserialize body field state."""
        body = cls.__new__(cls)
        body.morph = morph
        body.psi = np.array(d['psi_real']) + 1j * np.array(d['psi_imag'])
        body.g = d.get('g', 0.5)
        body.energy = d.get('energy', float(np.sum(np.abs(body.psi)**2)))
        body.E_target = d.get('E_target', body.energy)
        body.containment = d.get('containment', 1.0)
        return body
