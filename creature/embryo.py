"""
Embryonic Development System - Soliton-based morphogenesis.

Each embryo develops from a single soliton (fertilized egg) through symmetry
breaking and pattern formation. The final phenotype emerges from the
developmental trajectory, not just the genotype.

Key features:
- Single soliton → symmetry breaking → body plan
- Environmental coupling during gestation
- Path-dependent phenotype determination
- Developmental logging for analysis

CE Theory alignment:
- Each symmetry breaking = collapse event (information → structure)
- Development explores morphospace gradients
- Final form is attractor basin in developmental space
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class DevelopmentalStage(Enum):
    """Stages of embryonic development."""
    ZYGOTE = 0        # Single soliton
    CLEAVAGE = 1      # First divisions (2-4-8 cell)
    MORULA = 2        # Solid ball, high symmetry
    BLASTULA = 3      # Hollow sphere forming
    GASTRULA = 4      # Symmetry breaking begins
    NEURULA = 5       # Body plan established
    ORGANOGENESIS = 6 # Organ primordia form
    FETAL = 7         # Growth and refinement
    READY = 8         # Ready for birth


@dataclass
class DevelopmentalEvent:
    """A discrete event during development."""
    step: int
    stage: DevelopmentalStage
    event_type: str  # 'division', 'symmetry_break', 'basin_form', 'perturbation'
    details: Dict = field(default_factory=dict)


@dataclass
class EmbryoField:
    """
    2D NLSE field representing a developing embryo.
    
    The field evolves from a single soliton into complex patterns
    representing body plan and organ primordia.
    """
    # Field dimensions (small - embryo is tiny!)
    N: int = 32
    
    # NLSE parameters (inherited from parents)
    g: float = 0.5       # Nonlinearity (affects basin formation)
    dt: float = 0.05     # Time step
    
    # Field state
    psi: np.ndarray = field(default_factory=lambda: np.zeros((32, 32), dtype=complex))
    
    # Development tracking
    stage: DevelopmentalStage = DevelopmentalStage.ZYGOTE
    dev_step: int = 0
    total_steps: int = 200  # Steps of development
    
    # Symmetry metrics
    bilateral_symmetry: float = 1.0  # 1 = perfect symmetry, 0 = asymmetric
    radial_symmetry: float = 1.0     # Radial vs bilateral
    
    # Basin count (proto-organs)
    n_basins: int = 1
    basin_positions: List[Tuple[float, float]] = field(default_factory=list)
    
    # Environmental influences
    maternal_stress: float = 0.0     # Mother's hunger/fear
    environmental_noise: float = 0.05
    
    # Trait modifiers (emerge from development)
    trait_modifiers: Dict[str, float] = field(default_factory=dict)
    
    # Development log
    events: List[DevelopmentalEvent] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize with single soliton (zygote)."""
        self.psi = np.zeros((self.N, self.N), dtype=complex)
        self._initialize_zygote()
    
    def _initialize_zygote(self):
        """Create initial soliton representing fertilized egg."""
        x = np.linspace(-np.pi, np.pi, self.N)
        y = np.linspace(-np.pi, np.pi, self.N)
        X, Y = np.meshgrid(x, y)
        
        # Single Gaussian soliton at center
        r2 = X**2 + Y**2
        self.psi = 1.5 * np.exp(-r2 / 0.5) * np.exp(1j * 0)
        
        self.events.append(DevelopmentalEvent(
            step=0,
            stage=DevelopmentalStage.ZYGOTE,
            event_type='initialization',
            details={'initial_mass': float(np.sum(np.abs(self.psi)**2))}
        ))
    
    def set_parental_genetics(self, g_mother: float, g_father: float, 
                               parent_traits: Dict[str, float] = None):
        """Set embryo parameters from parental genetics."""
        # Nonlinearity inherited with some variation
        self.g = 0.5 * (g_mother + g_father) + np.random.uniform(-0.1, 0.1)
        self.g = np.clip(self.g, 0.2, 0.8)
        
        # Inherit trait tendencies (will be modified by development)
        if parent_traits:
            for trait, value in parent_traits.items():
                # Start with inherited value, will be modified
                self.trait_modifiers[trait] = value
    
    def set_maternal_environment(self, mother_hunger: float, mother_stress: float):
        """Update maternal environmental influence."""
        self.maternal_stress = 0.7 * self.maternal_stress + 0.3 * (mother_hunger + mother_stress * 0.5)
        self.environmental_noise = 0.05 + 0.1 * self.maternal_stress
    
    def step(self) -> Optional[DevelopmentalEvent]:
        """
        Advance development by one step.
        
        Returns developmental event if something significant happened.
        """
        self.dev_step += 1
        event = None
        
        # Determine current stage based on progress
        progress = self.dev_step / self.total_steps
        
        if progress < 0.1:
            new_stage = DevelopmentalStage.CLEAVAGE
        elif progress < 0.2:
            new_stage = DevelopmentalStage.MORULA
        elif progress < 0.3:
            new_stage = DevelopmentalStage.BLASTULA
        elif progress < 0.45:
            new_stage = DevelopmentalStage.GASTRULA
        elif progress < 0.6:
            new_stage = DevelopmentalStage.NEURULA
        elif progress < 0.8:
            new_stage = DevelopmentalStage.ORGANOGENESIS
        elif progress < 1.0:
            new_stage = DevelopmentalStage.FETAL
        else:
            new_stage = DevelopmentalStage.READY
        
        # Log stage transitions
        if new_stage != self.stage:
            event = DevelopmentalEvent(
                step=self.dev_step,
                stage=new_stage,
                event_type='stage_transition',
                details={'from': self.stage.name, 'to': new_stage.name}
            )
            self.events.append(event)
            self.stage = new_stage
        
        # Evolve the NLSE field
        self._evolve_field()
        
        # Stage-specific dynamics
        if self.stage == DevelopmentalStage.CLEAVAGE:
            self._do_cleavage()
        elif self.stage == DevelopmentalStage.GASTRULA:
            self._do_gastrulation()
        elif self.stage == DevelopmentalStage.NEURULA:
            self._do_neurulation()
        elif self.stage == DevelopmentalStage.ORGANOGENESIS:
            self._do_organogenesis()
        
        # Update symmetry metrics
        self._update_symmetry_metrics()
        
        # Apply environmental perturbations
        if np.random.random() < self.environmental_noise:
            self._apply_perturbation()
        
        return event
    
    def _evolve_field(self):
        """Evolve NLSE field using split-step FFT."""
        # Nonlinear step
        intensity = np.abs(self.psi)**2
        self.psi *= np.exp(-1j * self.g * intensity * self.dt)
        
        # Linear step (Laplacian in Fourier space)
        psi_k = np.fft.fft2(self.psi)
        kx = np.fft.fftfreq(self.N, 1/(2*np.pi)) 
        ky = np.fft.fftfreq(self.N, 1/(2*np.pi))
        KX, KY = np.meshgrid(kx, ky)
        k2 = KX**2 + KY**2
        psi_k *= np.exp(-1j * 0.5 * k2 * self.dt)
        self.psi = np.fft.ifft2(psi_k)
        
        # Add small noise to break perfect symmetry
        noise = np.random.randn(self.N, self.N) + 1j * np.random.randn(self.N, self.N)
        self.psi += 0.001 * self.environmental_noise * noise
    
    def _do_cleavage(self):
        """Simulate cell division during cleavage."""
        # Periodic perturbations cause soliton splitting
        if self.dev_step % 5 == 0:
            # Add radial perturbation
            x = np.linspace(-np.pi, np.pi, self.N)
            y = np.linspace(-np.pi, np.pi, self.N)
            X, Y = np.meshgrid(x, y)
            r = np.sqrt(X**2 + Y**2)
            
            # Ring perturbation causes division
            perturbation = 0.1 * np.exp(-((r - 0.5)**2) / 0.1)
            self.psi += perturbation * np.exp(1j * np.random.uniform(0, 2*np.pi))
    
    def _do_gastrulation(self):
        """Break radial symmetry → bilateral."""
        # Add asymmetric perturbation along one axis
        x = np.linspace(-np.pi, np.pi, self.N)
        y = np.linspace(-np.pi, np.pi, self.N)
        X, Y = np.meshgrid(x, y)
        
        # Anterior-posterior gradient
        ap_gradient = 0.05 * X * np.exp(-(X**2 + Y**2) / 2)
        self.psi += ap_gradient
        
        # This breaks radial symmetry
        self.radial_symmetry *= 0.95
    
    def _do_neurulation(self):
        """Establish neural tube / body axis."""
        x = np.linspace(-np.pi, np.pi, self.N)
        y = np.linspace(-np.pi, np.pi, self.N)
        X, Y = np.meshgrid(x, y)
        
        # Dorsal-ventral patterning
        dv_pattern = 0.03 * np.exp(-Y**2 / 0.5) * np.cos(2 * X)
        self.psi += dv_pattern
        
        # Count basins forming
        self._count_basins()
    
    def _do_organogenesis(self):
        """Form organ primordia as stable basins."""
        # Let basins stabilize through NLSE dynamics
        # The nonlinear term creates stable soliton-like structures
        
        # Slightly increase nonlinearity to lock in patterns
        effective_g = self.g * (1 + 0.01 * (self.dev_step - 0.6 * self.total_steps))
        intensity = np.abs(self.psi)**2
        self.psi *= np.exp(-1j * 0.01 * effective_g * intensity)
        
        # Update basin count
        self._count_basins()
        
        # Map basins to traits
        self._map_basins_to_traits()
    
    def _apply_perturbation(self):
        """Apply environmental perturbation."""
        # Maternal stress creates asymmetric perturbations
        x = np.linspace(-np.pi, np.pi, self.N)
        y = np.linspace(-np.pi, np.pi, self.N)
        X, Y = np.meshgrid(x, y)
        
        # Random localized perturbation
        px, py = np.random.uniform(-2, 2, 2)
        perturbation = self.maternal_stress * 0.05 * np.exp(-((X-px)**2 + (Y-py)**2) / 0.3)
        self.psi += perturbation
        
        if self.maternal_stress > 0.5:
            self.events.append(DevelopmentalEvent(
                step=self.dev_step,
                stage=self.stage,
                event_type='stress_perturbation',
                details={'stress_level': self.maternal_stress, 'position': (px, py)}
            ))
    
    def _update_symmetry_metrics(self):
        """Calculate symmetry metrics of current field."""
        intensity = np.abs(self.psi)**2
        
        # Bilateral symmetry (left-right)
        left = intensity[:, :self.N//2]
        right = np.flip(intensity[:, self.N//2:], axis=1)
        if np.sum(intensity) > 0.01:
            self.bilateral_symmetry = 1 - np.sum(np.abs(left - right)) / np.sum(intensity)
        
        # Radial symmetry (using polar coordinates)
        # Approximated by comparing rings
        center = self.N // 2
        x = np.arange(self.N) - center
        y = np.arange(self.N) - center
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        radial_var = 0
        for r in range(1, self.N//3):
            ring_mask = (R >= r-0.5) & (R < r+0.5)
            if np.sum(ring_mask) > 0:
                ring_values = intensity[ring_mask]
                if len(ring_values) > 1:
                    radial_var += np.std(ring_values) / (np.mean(ring_values) + 0.01)
        
        self.radial_symmetry = max(0, 1 - radial_var / (self.N//3))
    
    def _count_basins(self):
        """Count distinct basins (local maxima) in the field."""
        from scipy import ndimage
        
        intensity = np.abs(self.psi)**2
        
        # Find local maxima
        max_filtered = ndimage.maximum_filter(intensity, size=3)
        local_max = (intensity == max_filtered) & (intensity > 0.1 * np.max(intensity))
        
        # Label connected regions
        labeled, n_features = ndimage.label(local_max)
        self.n_basins = n_features
        
        # Get basin positions
        self.basin_positions = []
        for i in range(1, n_features + 1):
            positions = np.where(labeled == i)
            if len(positions[0]) > 0:
                cy = np.mean(positions[0]) / self.N * 2 * np.pi - np.pi
                cx = np.mean(positions[1]) / self.N * 2 * np.pi - np.pi
                self.basin_positions.append((cx, cy))
    
    def _map_basins_to_traits(self):
        """Map basin configuration to trait modifiers."""
        # Number of basins affects body complexity
        if self.n_basins >= 4:
            self.trait_modifiers['speed_factor'] = self.trait_modifiers.get('speed_factor', 1.0) * 1.05
        
        # Bilateral symmetry affects coordination
        if self.bilateral_symmetry > 0.8:
            self.trait_modifiers['efferent_strength'] = self.trait_modifiers.get('efferent_strength', 1.0) * 1.03
        
        # Low symmetry might indicate developmental issues
        if self.bilateral_symmetry < 0.5:
            self.trait_modifiers['energy_efficiency'] = self.trait_modifiers.get('energy_efficiency', 1.0) * 0.95
        
        # Maternal stress effects
        if self.maternal_stress > 0.6:
            self.trait_modifiers['afferent_sensitivity'] = self.trait_modifiers.get('afferent_sensitivity', 1.0) * 1.1
    
    def get_final_traits(self) -> Dict[str, float]:
        """Get final trait modifiers after development."""
        return self.trait_modifiers.copy()
    
    def get_development_summary(self) -> Dict:
        """Get summary of developmental process."""
        return {
            'total_steps': self.dev_step,
            'final_stage': self.stage.name,
            'n_basins': self.n_basins,
            'bilateral_symmetry': self.bilateral_symmetry,
            'radial_symmetry': self.radial_symmetry,
            'n_events': len(self.events),
            'n_perturbations': sum(1 for e in self.events if e.event_type == 'stress_perturbation'),
            'trait_modifiers': self.trait_modifiers,
        }
    
    def get_field_image(self) -> np.ndarray:
        """Get intensity field as image for visualization."""
        return np.abs(self.psi)**2
    
    def is_ready(self) -> bool:
        """Check if development is complete."""
        return self.stage == DevelopmentalStage.READY or self.dev_step >= self.total_steps


def create_embryo_from_parents(mother, father) -> EmbryoField:
    """Create an embryo from two parent creatures."""
    embryo = EmbryoField()
    
    # Get parental NLSE parameters
    g_mother = mother.body.g if hasattr(mother, 'body') else 0.5
    g_father = father.body.g if hasattr(father, 'body') else 0.5
    
    # Get parental traits
    parent_traits = {}
    if hasattr(mother, 'traits') and mother.traits:
        for trait, value in mother.traits.to_dict().items():
            parent_traits[trait] = value
    if hasattr(father, 'traits') and father.traits:
        for trait, value in father.traits.to_dict().items():
            # Average with mother's
            if trait in parent_traits:
                parent_traits[trait] = 0.5 * (parent_traits[trait] + value)
            else:
                parent_traits[trait] = value
    
    embryo.set_parental_genetics(g_mother, g_father, parent_traits)
    
    return embryo


def develop_embryo_during_gestation(embryo: EmbryoField, 
                                     mother_hunger: float,
                                     mother_stress: float,
                                     steps: int = 10) -> List[DevelopmentalEvent]:
    """
    Run embryo development for several steps.
    
    Called during each gestation step to advance development.
    """
    events = []
    
    # Update maternal environment
    embryo.set_maternal_environment(mother_hunger, mother_stress)
    
    # Run development steps
    for _ in range(steps):
        event = embryo.step()
        if event:
            events.append(event)
    
    return events
