"""
Disease System - Density-dependent disease outbreaks.

Diseases create natural population crashes that prevent runaway explosions.
Virulence and lethality scale with local population density.
"""

from typing import List, Dict, Tuple, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

if TYPE_CHECKING:
    from .terrain import Terrain


@dataclass
class DiseaseStrain:
    """A specific disease strain with base characteristics."""
    name: str
    base_virulence: float      # Base infection chance (0-1)
    base_lethality: float      # Base death chance per step when infected (0-1)
    base_duration: int         # Base outbreak duration in steps
    incubation: int            # Steps before creature becomes contagious
    contagion_radius: float    # How far it spreads spatially
    species_targets: List[str] # Which species can catch it
    density_scaling: float     # How much density amplifies effects (1.0 = linear)
    terrain_affinity: str = None  # Optional: spreads faster in certain terrain


# Pre-defined disease strains
DISEASE_STRAINS = [
    DiseaseStrain(
        name="Spore Sickness",
        base_virulence=0.08,
        base_lethality=0.003,
        base_duration=250,
        incubation=50,
        contagion_radius=6.0,
        species_targets=['Blob', 'Herbie', 'Mono', 'Biped'],
        density_scaling=1.5,
    ),
    DiseaseStrain(
        name="Crimson Wasting",
        base_virulence=0.06,
        base_lethality=0.008,
        base_duration=400,
        incubation=80,
        contagion_radius=4.0,
        species_targets=['Blob', 'Mono'],
        density_scaling=2.5,
    ),
    DiseaseStrain(
        name="Grey Plague",
        base_virulence=0.04,
        base_lethality=0.015,
        base_duration=300,
        incubation=100,
        contagion_radius=10.0,
        species_targets=['Blob', 'Biped', 'Mono', 'Scavenger', 'Herbie'],
        density_scaling=3.0,
    ),
    DiseaseStrain(
        name="Swamp Rot",
        base_virulence=0.10,
        base_lethality=0.005,
        base_duration=350,
        incubation=30,
        contagion_radius=5.0,
        species_targets=['Blob', 'Mono', 'Biped'],
        density_scaling=2.0,
        terrain_affinity='water',
    ),
    DiseaseStrain(
        name="Neural Tremors",
        base_virulence=0.03,
        base_lethality=0.012,
        base_duration=200,
        incubation=60,
        contagion_radius=3.0,
        species_targets=['Herbie'],
        density_scaling=1.0,
    ),
]


@dataclass
class DiseaseOutbreak:
    """An active disease outbreak in the world."""
    disease_id: str
    strain: DiseaseStrain
    start_step: int
    duration_remaining: int
    infected_creatures: Set[str] = field(default_factory=set)
    symptomatic_creatures: Set[str] = field(default_factory=set)
    infection_times: Dict[str, int] = field(default_factory=dict)
    total_deaths: int = 0


class DiseaseSystem:
    """
    Density-dependent disease system.
    
    Key mechanics:
    - Outbreak probability scales with total population density
    - Virulence scales with LOCAL density (crowded areas spread faster)
    - Lethality scales with density AND creature weakness (hungry die faster)
    - Duration extends logarithmically with infected count
    - Targets dominant species preferentially
    """
    
    def __init__(self, world_size: float = 100.0):
        """Initialize disease system."""
        self.world_size = world_size
        self.active_outbreaks: List[DiseaseOutbreak] = []
        self.outbreak_history: List[dict] = []
        self.step_count = 0
        self.total_deaths_from_disease = 0
        self.outbreak_cooldown = 0
        
        # Tuning parameters
        self.min_population_for_outbreak = 40
        self.base_outbreak_probability = 0.0008
        self.max_concurrent_outbreaks = 2
        
    def get_local_density(self, pos: np.ndarray, creatures: List, radius: float = 12.0) -> float:
        """Calculate creature density around a position."""
        count = sum(1 for c in creatures if c.alive and 
                   np.linalg.norm(c.pos - pos) < radius)
        area = np.pi * radius ** 2
        return count / max(area, 1.0)
    
    def get_species_counts(self, creatures: List) -> Dict[str, int]:
        """Count living creatures by species."""
        counts = {}
        for c in creatures:
            if c.alive:
                counts[c.species.name] = counts.get(c.species.name, 0) + 1
        return counts
    
    def get_global_density(self, creatures: List) -> float:
        """Overall creature density."""
        alive = sum(1 for c in creatures if c.alive)
        return alive / (self.world_size ** 2)
    
    def calculate_effective_virulence(self, strain: DiseaseStrain, 
                                       local_density: float) -> float:
        """Virulence scales with local density (1x to 5x)."""
        density_factor = 1.0 + (local_density * 80) ** strain.density_scaling
        density_factor = min(density_factor, 5.0)
        return strain.base_virulence * density_factor
    
    def calculate_effective_lethality(self, strain: DiseaseStrain,
                                       creature_hunger: float,
                                       local_density: float) -> float:
        """Lethality scales with density and creature weakness."""
        hunger_factor = 0.5 + creature_hunger
        density_factor = 1.0 + (local_density * 40) ** (strain.density_scaling * 0.5)
        density_factor = min(density_factor, 3.0)
        return strain.base_lethality * hunger_factor * density_factor
    
    def get_outbreak_probability(self, population_density: float, species_counts: Dict) -> float:
        """Calculate probability of new outbreak."""
        if self.outbreak_cooldown > 0:
            return 0.0
            
        total_alive = sum(species_counts.values())
        if total_alive < self.min_population_for_outbreak:
            return 0.0
        
        prob = self.base_outbreak_probability * (1.0 + population_density * 150)
        
        # Dominance bonus
        if species_counts:
            max_species_count = max(species_counts.values())
            dominance_ratio = max_species_count / total_alive
            if dominance_ratio > 0.35:
                prob *= (1.0 + (dominance_ratio - 0.35) * 4.0)
        
        return min(0.01, prob)
    
    def update(self, creatures: List, world_area: float, terrain_map=None) -> Tuple[List[str], List]:
        """
        Update disease system.
        
        Returns:
            (messages, newly_dead_creatures)
        """
        self.step_count += 1
        messages = []
        dead_from_disease = []
        
        if self.outbreak_cooldown > 0:
            self.outbreak_cooldown -= 1
        
        species_counts = self.get_species_counts(creatures)
        alive_count = sum(species_counts.values())
        global_density = alive_count / max(world_area, 1.0)
        
        # Maybe spawn new outbreak
        if len(self.active_outbreaks) < self.max_concurrent_outbreaks:
            outbreak_prob = self.get_outbreak_probability(global_density, species_counts)
            if np.random.random() < outbreak_prob:
                outbreak = self._spawn_outbreak(species_counts)
                if outbreak:
                    self.active_outbreaks.append(outbreak)
                    initial_count = self._seed_initial_infections(outbreak, creatures)
                    messages.append(f"[DISEASE] 🦠 {outbreak.strain.name} outbreak! "
                                  f"Targeting: {', '.join(outbreak.strain.species_targets)}, "
                                  f"Initial infections: {initial_count}")
        
        # Process active outbreaks
        expired = []
        for outbreak in self.active_outbreaks:
            new_dead, new_messages = self._process_outbreak(outbreak, creatures, terrain_map)
            dead_from_disease.extend(new_dead)
            messages.extend(new_messages)
            
            outbreak.duration_remaining -= 1
            
            # Check end conditions
            alive_infected = sum(1 for cid in outbreak.infected_creatures 
                               if any(c.creature_id == cid and c.alive for c in creatures))
            
            if (outbreak.duration_remaining <= 0 or 
                alive_infected == 0 or
                outbreak.total_deaths > 60):
                expired.append(outbreak)
        
        for outbreak in expired:
            self.active_outbreaks.remove(outbreak)
            self.outbreak_history.append({
                'name': outbreak.strain.name,
                'total_deaths': outbreak.total_deaths,
                'duration': outbreak.strain.base_duration - outbreak.duration_remaining,
                'start': outbreak.start_step,
                'end': self.step_count
            })
            messages.append(f"[DISEASE] {outbreak.strain.name} subsided. Deaths: {outbreak.total_deaths}")
            self.outbreak_cooldown = 400 + np.random.randint(0, 400)
        
        self.total_deaths_from_disease += len(dead_from_disease)
        return messages, dead_from_disease
    
    def _spawn_outbreak(self, species_counts: Dict) -> Optional[DiseaseOutbreak]:
        """Spawn a new disease outbreak targeting dominant species."""
        if not species_counts:
            return None
        
        total_creatures = sum(species_counts.values())
        
        weights = []
        for strain in DISEASE_STRAINS:
            weight = 1.0
            for target in strain.species_targets:
                if target in species_counts:
                    species_ratio = species_counts[target] / total_creatures
                    weight += species_ratio * 3.0
            weights.append(weight)
        
        total_weight = sum(weights)
        r = np.random.random() * total_weight
        cumulative = 0
        selected_strain = DISEASE_STRAINS[0]
        for strain, weight in zip(DISEASE_STRAINS, weights):
            cumulative += weight
            if r <= cumulative:
                selected_strain = strain
                break
        
        duration_multiplier = 1.0 + np.log10(max(1, total_creatures / 30))
        duration = int(selected_strain.base_duration * duration_multiplier)
        
        return DiseaseOutbreak(
            disease_id=f"D{self.step_count}_{np.random.randint(10000)}",
            strain=selected_strain,
            start_step=self.step_count,
            duration_remaining=duration,
        )
    
    def _seed_initial_infections(self, outbreak: DiseaseOutbreak, creatures: List) -> int:
        """Infect initial creatures to start the outbreak."""
        susceptible = [c for c in creatures if c.alive and 
                      c.species.name in outbreak.strain.species_targets]
        
        if not susceptible:
            return 0
        
        densities = [self.get_local_density(c.pos, creatures) for c in susceptible]
        max_density = max(densities) if densities else 1.0
        weights = [(d / max_density) ** 2 + 0.1 for d in densities]
        weights = np.array(weights) / sum(weights)
        
        patient_zero = np.random.choice(susceptible, p=weights)
        outbreak.infected_creatures.add(patient_zero.creature_id)
        outbreak.infection_times[patient_zero.creature_id] = self.step_count
        
        for c in susceptible:
            if c.creature_id != patient_zero.creature_id:
                dist = np.linalg.norm(c.pos - patient_zero.pos)
                if dist < outbreak.strain.contagion_radius * 1.5:
                    if np.random.random() < 0.25:
                        outbreak.infected_creatures.add(c.creature_id)
                        outbreak.infection_times[c.creature_id] = self.step_count
        
        return len(outbreak.infected_creatures)
    
    def _process_outbreak(self, outbreak: DiseaseOutbreak, creatures: List, 
                          terrain_map=None) -> Tuple[List, List[str]]:
        """Process one outbreak step."""
        dead_creatures = []
        messages = []
        strain = outbreak.strain
        
        creature_by_id = {c.creature_id: c for c in creatures if c.alive}
        
        outbreak.infected_creatures = {cid for cid in outbreak.infected_creatures 
                                        if cid in creature_by_id}
        
        # Update symptomatic status
        for cid in outbreak.infected_creatures:
            if cid not in outbreak.symptomatic_creatures:
                infection_time = outbreak.infection_times.get(cid, self.step_count)
                if self.step_count - infection_time > strain.incubation:
                    outbreak.symptomatic_creatures.add(cid)
        
        # Spread from symptomatic creatures
        newly_infected = set()
        for cid in outbreak.symptomatic_creatures:
            if cid not in creature_by_id:
                continue
            infected = creature_by_id[cid]
            local_density = self.get_local_density(infected.pos, creatures)
            virulence = self.calculate_effective_virulence(strain, local_density)
            
            for other in creatures:
                if not other.alive or other.creature_id in outbreak.infected_creatures:
                    continue
                if other.species.name not in strain.species_targets:
                    continue
                
                dist = np.linalg.norm(other.pos - infected.pos)
                if dist < strain.contagion_radius:
                    proximity_factor = 1.0 - (dist / strain.contagion_radius)
                    if np.random.random() < virulence * proximity_factor:
                        newly_infected.add(other.creature_id)
                        outbreak.infection_times[other.creature_id] = self.step_count
        
        outbreak.infected_creatures.update(newly_infected)
        
        # Apply effects to symptomatic creatures
        for cid in list(outbreak.symptomatic_creatures):
            if cid not in creature_by_id:
                outbreak.symptomatic_creatures.discard(cid)
                continue
                
            creature = creature_by_id[cid]
            local_density = self.get_local_density(creature.pos, creatures)
            hunger = getattr(creature.metabolism, 'hunger', 0.5)
            lethality = self.calculate_effective_lethality(strain, hunger, local_density)
            
            if np.random.random() < lethality:
                creature.alive = False
                creature.metabolism.is_dead = True
                creature.metabolism.cause_of_death = f"disease:{strain.name}"
                dead_creatures.append(creature)
                outbreak.total_deaths += 1
                outbreak.infected_creatures.discard(cid)
                outbreak.symptomatic_creatures.discard(cid)
                messages.append(f"[DISEASE] {creature.species.name} died from {strain.name}!")
                continue
            
            # Sublethal effects
            damage = strain.base_lethality * 0.5
            creature.metabolism.hunger = min(1.0, creature.metabolism.hunger + damage)
            creature.body.energy *= (1.0 - damage * 0.3)
            
            if hasattr(creature, 'afferent') and 'env_pain' in creature.afferent:
                creature.afferent['env_pain'].nucleate(damage * 1.5, 0.0)
        
        # Recovery chance for mild strains
        if strain.base_lethality < 0.005:
            for cid in list(outbreak.infected_creatures):
                if np.random.random() < 0.008:
                    outbreak.infected_creatures.discard(cid)
                    outbreak.symptomatic_creatures.discard(cid)
        
        return dead_creatures, messages
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'step_count': self.step_count,
            'total_deaths_from_disease': self.total_deaths_from_disease,
            'outbreak_cooldown': self.outbreak_cooldown,
            'outbreak_history': self.outbreak_history[-20:],
        }
    
    @classmethod
    def from_dict(cls, d: dict, world_size: float) -> 'DiseaseSystem':
        """Deserialize from persistence."""
        system = cls(world_size)
        system.step_count = d.get('step_count', 0)
        system.total_deaths_from_disease = d.get('total_deaths_from_disease', 0)
        system.outbreak_cooldown = d.get('outbreak_cooldown', 0)
        system.outbreak_history = d.get('outbreak_history', [])
        return system
