"""
PiezoSkeleton - Stress field from body gradients.

Internal proprioceptive feedback system that tracks body deformation
and provides feedback to maintain structural integrity.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from ..core.constants import Ny, Nx


class PiezoSkeleton:
    """
    Stress field from body gradients - internal proprioceptive feedback.
    
    The skeleton tracks changes in body density and produces a stress field
    that feeds back into body dynamics, helping maintain coherent body shape.
    """
    
    def __init__(self):
        """Initialize skeleton stress field."""
        self.S = np.zeros((Ny, Nx))
        self.S_prev = np.zeros((Ny, Nx))
        self.rms = 0.0
        
        # Parameters
        self.drive_gain = 0.018
        self.diffuse = 0.10
        self.leak = 0.993
        self.feedback_gain = 0.30
        
    def step(self, body_I: np.ndarray):
        """
        Update stress field based on body intensity changes.
        
        Args:
            body_I: Current body intensity field |ψ|²
        """
        # Drive from change in body density
        drive = gaussian_filter(body_I - self.S_prev, sigma=1.2)
        
        # Laplacian diffusion
        lap = (np.roll(self.S, 1, 0) + np.roll(self.S, -1, 0) +
               np.roll(self.S, 1, 1) + np.roll(self.S, -1, 1) - 4*self.S)
        
        # Update stress field
        self.S = self.leak * self.S + self.drive_gain * drive + self.diffuse * lap
        self.S = gaussian_filter(self.S, sigma=0.6)
        
        # Track RMS stress
        self.rms = float(np.sqrt(np.mean(self.S**2) + 1e-18))
        
        # Store current body state for next step
        self.S_prev = body_I.copy()
        
    def get_potential(self) -> np.ndarray:
        """
        Get feedback potential for body field.
        
        Returns:
            Potential field that pushes body toward lower stress configurations
        """
        S_norm = self.S / (np.percentile(np.abs(self.S), 99) + 1e-12)
        return self.feedback_gain * np.clip(S_norm, -2.5, 2.5)
    
    def to_dict(self) -> dict:
        """Serialize skeleton state."""
        return {
            'S': self.S.tolist(),
            'S_prev': self.S_prev.tolist(),
            'rms': self.rms,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PiezoSkeleton':
        """Deserialize skeleton state."""
        skel = cls()
        skel.S = np.array(d.get('S', np.zeros((Ny, Nx))))
        skel.S_prev = np.array(d.get('S_prev', np.zeros((Ny, Nx))))
        skel.rms = d.get('rms', 0.0)
        return skel
