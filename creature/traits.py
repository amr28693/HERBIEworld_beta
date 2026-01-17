"""
Mutated Traits - Individual variation from species baseline.

Traits can mutate during reproduction, creating heritable variation
that enables evolutionary adaptation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .species import SpeciesParams


# Mutation parameters
MUTATION_RATE = 0.15  # 15% chance per trait
MUTATION_MAGNITUDE = 0.12  # ±12% drift

# Trait bounds (trait_name -> (min, max))
MUTABLE_TRAITS = {
    'speed_factor': (0.4, 2.0),
    'energy_efficiency': (0.5, 1.8),
    'afferent_sensitivity': (0.5, 2.0),
    'efferent_strength': (0.5, 2.0),
    'metabolism_rate': (0.5, 1.8),
}


@dataclass
class MutatedTraits:
    """Tracks individual's mutations from species baseline."""
    speed_factor: Optional[float] = None
    energy_efficiency: Optional[float] = None
    afferent_sensitivity: Optional[float] = None
    efferent_strength: Optional[float] = None
    metabolism_rate: Optional[float] = None
    
    def get(self, name: str, default: float) -> float:
        """Get trait value, falling back to default if not mutated."""
        val = getattr(self, name, None)
        return val if val is not None else default
    
    def to_dict(self) -> dict:
        """Serialize to dict, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    @classmethod
    def from_dict(cls, d: dict) -> 'MutatedTraits':
        """Deserialize from dict."""
        valid_keys = {'speed_factor', 'energy_efficiency', 'afferent_sensitivity',
                      'efferent_strength', 'metabolism_rate'}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})
    
    @classmethod
    def mutate_from_parent(cls, parent_traits: Optional['MutatedTraits'], 
                           species: 'SpeciesParams',
                           mutation_rate_mult: float = 1.0) -> 'MutatedTraits':
        """
        Create child traits with possible mutations.
        
        Args:
            parent_traits: Parent's trait mutations (or None for first gen)
            species: Species baseline parameters
            mutation_rate_mult: Multiplier for mutation rate (from launcher)
            
        Returns:
            New MutatedTraits with possible mutations
        """
        child = cls()
        
        # Apply rate multiplier
        effective_rate = MUTATION_RATE * mutation_rate_mult
        effective_magnitude = MUTATION_MAGNITUDE * (0.5 + 0.5 * mutation_rate_mult)
        
        for trait_name, (min_val, max_val) in MUTABLE_TRAITS.items():
            # Get parent value or species baseline
            if parent_traits and getattr(parent_traits, trait_name) is not None:
                base_val = getattr(parent_traits, trait_name)
            else:
                base_val = getattr(species, trait_name, 1.0)
            
            # Maybe mutate
            if np.random.random() < effective_rate:
                drift = np.random.uniform(-effective_magnitude, effective_magnitude)
                new_val = np.clip(base_val * (1 + drift), min_val, max_val)
                setattr(child, trait_name, new_val)
            elif parent_traits and getattr(parent_traits, trait_name) is not None:
                # Inherit parent's mutated value without further mutation
                setattr(child, trait_name, base_val)
        
        return child
    
    def get_summary(self, species: 'SpeciesParams' = None) -> str:
        """Get human-readable summary of mutations."""
        parts = []
        for trait_name in MUTABLE_TRAITS:
            val = getattr(self, trait_name, None)
            if val is not None:
                baseline = getattr(species, trait_name, 1.0) if species else 1.0
                pct = (val / baseline - 1) * 100
                sign = '+' if pct >= 0 else ''
                parts.append(f"{trait_name}:{sign}{pct:.0f}%")
        return ', '.join(parts) if parts else 'baseline'


# Additional species capabilities (added dynamically)
HIBERNATING_SPECIES = {'Blob', 'Mono'}
DEFENDING_SPECIES = {'Herbie': 0.15, 'Biped': 0.1}

# Digestion parameters for predators
DIGESTION_DURATION = {
    'Apex': 400,
    'Scavenger': 200,
}
DIGESTION_SPEED_PENALTY = 0.3
