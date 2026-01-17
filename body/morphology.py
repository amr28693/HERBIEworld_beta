"""
Morphology - Body shape with dynamic limb angles.

Defines the spatial structure of the creature's body, including
basin attractors for body regions and limb geometry.
"""

import time
from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter

from ..core.constants import (
    BASINS, LIMB_DEFS, limb_length, dt,
    Ny, Nx, X, Y
)


class Morphology:
    """
    Body shape with dynamic limb angles.
    
    The morphology defines:
    - Basin attractors that confine body field density
    - Limb angles that can change based on neural activity
    - Skin boundary that contains the body field
    """
    
    def __init__(self):
        """Initialize morphology with default limb positions."""
        # Limb angles (radians) - start at default positions
        self.limb_angles = {name: info[1] for name, info in LIMB_DEFS.items()}
        self.limb_velocities = {name: 0.0 for name in LIMB_DEFS}
        self.limb_torques = {name: 0.0 for name in LIMB_DEFS}
        
        # Skin parameters
        self.skin_stiffness = 7.0
        self.skin_thickness = 0.55
        
    def get_limb_tip(self, limb_name: str) -> Tuple[float, float]:
        """
        Get the world-space position of a limb tip.
        
        Args:
            limb_name: Name of the limb (e.g., 'limb_0')
            
        Returns:
            (x, y) position of limb tip
        """
        origin_name, _ = LIMB_DEFS[limb_name]
        origin = BASINS[origin_name]['pos']
        angle = self.limb_angles[limb_name]
        return (origin[0] + limb_length * np.cos(angle),
                origin[1] + limb_length * np.sin(angle))
                
    def apply_efferent_torque(self, limb_name: str, strength: float):
        """
        Apply motor torque to a limb.
        
        Args:
            limb_name: Name of the limb
            strength: Torque strength
        """
        self.limb_torques[limb_name] += strength * 0.4
        
    def update_limb_angles(self, limb_fields: dict, torus_bias: np.ndarray, 
                           skel_rms: float, hunger: float, audio_amp: float):
        """
        Update limb angles based on neural and environmental inputs.
        
        Args:
            limb_fields: Dict of LimbField objects
            torus_bias: Directional bias from torus brain
            skel_rms: Skeleton stress RMS
            hunger: Current hunger level
            audio_amp: Audio amplitude
        """
        for limb_name, limb in limb_fields.items():
            _, base_angle = LIMB_DEFS[limb_name]
            
            # Contributions to angular acceleration
            momentum = limb.get_momentum()
            extension_bias = (limb.pulse_position - 0.5) * limb.pulse_amplitude * 0.5
            eff_torque = self.limb_torques.get(limb_name, 0.0) * (0.5 + 0.5 * np.clip(limb.energy / 2.0, 0, 1))
            stress_kick = skel_rms * 0.3 * (np.random.random() - 0.5)
            audio_kick = audio_amp * 0.4 * np.sin(self.limb_angles[limb_name] * 3 + limb.energy)
            hunger_wiggle = hunger * 0.2 * np.sin(time.time() * 3 + hash(limb_name))
            
            # Torus brain influence on limb angle
            bias_angle = np.arctan2(torus_bias[1], torus_bias[0])
            angle_diff = bias_angle - self.limb_angles[limb_name]
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            torus_pull = angle_diff * np.linalg.norm(torus_bias) * 0.15
            
            # Sum all contributions
            angular_accel = (momentum * 0.8 + extension_bias + eff_torque * 1.2 + 
                           stress_kick + audio_kick + hunger_wiggle + torus_pull)
            
            # Update velocity with damping
            self.limb_velocities[limb_name] = 0.7 * self.limb_velocities[limb_name] + 0.3 * angular_accel
            
            # Return-to-neutral spring
            neutral_diff = base_angle - self.limb_angles[limb_name]
            neutral_diff = np.arctan2(np.sin(neutral_diff), np.cos(neutral_diff))
            self.limb_velocities[limb_name] += 0.01 * neutral_diff
            
            # Apply damping
            self.limb_velocities[limb_name] *= 0.95
            
            # Update angle
            self.limb_angles[limb_name] += self.limb_velocities[limb_name] * dt * 10
            
        # Decay applied torques
        for name in self.limb_torques:
            self.limb_torques[name] *= 0.75
            
    def compute_skin(self) -> np.ndarray:
        """
        Compute skin boundary distance field.
        
        Returns:
            Field where values represent distance from skin boundary,
            used to confine body field and detect contact.
        """
        min_dist = np.ones((Ny, Nx)) * 100.0
        
        # Distance from basin centers
        for name, info in BASINS.items():
            bx, by = info['pos']
            r = info['radius']
            dist = np.sqrt((X - bx)**2 + (Y - by)**2) - r
            min_dist = np.minimum(min_dist, dist)
            
        # Distance from limb segments
        for limb_name in LIMB_DEFS:
            origin_name, _ = LIMB_DEFS[limb_name]
            p1 = BASINS[origin_name]['pos']
            p2 = self.get_limb_tip(limb_name)
            
            x1, y1 = p1
            x2, y2 = p2
            dx_c, dy_c = x2 - x1, y2 - y1
            length = np.sqrt(dx_c**2 + dy_c**2)
            
            if length > 0.01:
                ux, uy = dx_c / length, dy_c / length
                proj = np.clip((X - x1)*ux + (Y - y1)*uy, 0, length)
                dist = np.sqrt((X - (x1 + proj*ux))**2 + (Y - (y1 + proj*uy))**2) - 0.45
                min_dist = np.minimum(min_dist, dist)
                
        return self.skin_stiffness * (1 - np.exp(-np.maximum(0, min_dist) / self.skin_thickness))
        
    def get_basin_potential(self) -> np.ndarray:
        """
        Compute basin potential that attracts body field to body regions.
        
        Returns:
            Potential field with wells at basin locations
        """
        V = np.ones((Ny, Nx)) * 9.0
        
        # Basin wells
        for name, info in BASINS.items():
            bx, by = info['pos']
            V -= info['depth'] * np.exp(-((X - bx)**2 + (Y - by)**2) / info['radius']**2)
            
        # Channels connecting core to limb bases
        for b1, b2 in [('core', 'limb_0'), ('core', 'limb_1'), ('core', 'limb_2')]:
            p1, p2 = BASINS[b1]['pos'], BASINS[b2]['pos']
            x1, y1 = p1
            x2, y2 = p2
            dx_c, dy_c = x2 - x1, y2 - y1
            length = np.sqrt(dx_c**2 + dy_c**2)
            
            if length > 0.01:
                ux, uy = dx_c / length, dy_c / length
                proj = np.clip((X - x1)*ux + (Y - y1)*uy, 0, length)
                dist = np.sqrt((X - (x1 + proj*ux))**2 + (Y - (y1 + proj*uy))**2)
                V -= 3.5 * np.exp(-dist**2 / 0.35**2)
                
        return gaussian_filter(V, sigma=0.6)
    
    def to_dict(self) -> dict:
        """Serialize morphology state."""
        return {
            'limb_angles': self.limb_angles.copy(),
            'limb_velocities': self.limb_velocities.copy(),
            'limb_torques': self.limb_torques.copy(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Morphology':
        """Deserialize morphology state."""
        morph = cls()
        morph.limb_angles = d.get('limb_angles', morph.limb_angles)
        morph.limb_velocities = d.get('limb_velocities', morph.limb_velocities)
        morph.limb_torques = d.get('limb_torques', morph.limb_torques)
        return morph
