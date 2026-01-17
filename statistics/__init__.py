"""
Statistics Tracking - Comprehensive creature lifetime and population analytics.

Contains:
- LifetimeRecord: Complete record of one creature's life
- LifetimeTracker: Collects statistics during creature lifetime
- EcologicalBalance: Sigmoid-based carrying capacity for population regulation
"""

from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional
import numpy as np


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))


@dataclass
class LifetimeRecord:
    """Complete record of one creature's life with comprehensive analytics."""
    # Identity
    creature_id: str = ""
    generation: int = 0
    parent_id: str = ""
    lineage_depth: int = 0
    species_name: str = ""
    
    # Temporal
    birth_step: int = 0
    death_step: int = 0
    steps_lived: int = 0
    birth_time: str = ""
    death_time: str = ""
    
    # Cause of death
    cause_of_death: str = ""
    
    # Consumption & Metabolism
    total_consumed: float = 0.0
    objects_eaten: int = 0
    total_defecated: float = 0.0
    peak_gut_contents: float = 0.0
    mean_hunger: float = 0.0
    mean_satiation: float = 0.0
    starvation_episodes: int = 0
    
    # Movement
    total_distance: float = 0.0
    mean_speed: float = 0.0
    peak_speed: float = 0.0
    movement_entropy: float = 0.0
    exploration_radius: float = 0.0
    
    # Energy
    peak_energy: float = 0.0
    mean_energy: float = 0.0
    energy_variance: float = 0.0
    
    # Brain metrics
    mean_coherence: float = 0.0
    peak_coherence: float = 0.0
    mean_circulation: float = 0.0
    brain_entropy: float = 0.0
    torus_energy_variance: float = 0.0
    
    # Body metrics
    body_entropy: float = 0.0
    mean_containment: float = 0.0
    body_g_variance: float = 0.0
    
    # Limb metrics
    limb_energy_balance: float = 0.0
    limb_coordination: float = 0.0
    mean_limb_activity: float = 0.0
    
    # Reproduction
    offspring_count: int = 0
    first_reproduction_step: int = 0
    
    # Defecation tracking
    total_defecations: int = 0
    total_food_consumed: float = 0.0
    
    # Learning
    objects_learned: int = 0
    learning_rate_avg: float = 0.0
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            else:
                result[k] = v
        return result
    
    @classmethod
    def from_dict(cls, d: dict) -> 'LifetimeRecord':
        """Deserialize from dictionary."""
        record = cls()
        for k, v in d.items():
            if hasattr(record, k):
                setattr(record, k, v)
        return record
    
    def fitness_score(self) -> float:
        """Composite fitness metric."""
        survival = self.steps_lived / 10000  # Normalize to ~1
        feeding = self.total_consumed / 100
        reproduction = self.offspring_count * 2
        efficiency = self.total_consumed / (self.total_distance + 1) * 10
        coherence = self.mean_coherence / 10
        return survival + feeding + reproduction + efficiency + coherence


class LifetimeTracker:
    """Collects comprehensive statistics during a creature's life."""
    
    def __init__(self, creature_id: str, generation: int, 
                 parent_id: str = "", lineage_depth: int = 0,
                 species_name: str = ""):
        self.record = LifetimeRecord(
            creature_id=creature_id, 
            generation=generation,
            parent_id=parent_id,
            lineage_depth=lineage_depth,
            species_name=species_name,
            birth_time=datetime.now().isoformat()
        )
        
        # Position tracking
        self.positions: List[np.ndarray] = []
        self.velocities: List[np.ndarray] = []
        
        # Time series for analysis
        self.energies: List[float] = []
        self.hungers: List[float] = []
        self.satiations: List[float] = []
        self.coherences: List[float] = []
        self.circulations: List[float] = []
        self.speeds: List[float] = []
        self.rewards: List[float] = []
        self.gut_contents: List[float] = []
        self.containments: List[float] = []
        
        # Limb tracking
        self.limb_energies: Dict[str, List[float]] = {}
        self.brain_entropies: deque = deque(maxlen=500)
        self.body_entropies: deque = deque(maxlen=500)
        
        # Event tracking
        self.starvation_starts: List[int] = []
        self.defecation_events: List[int] = []
        self.reproduction_steps: List[int] = []
        
        self.start_step = 0
        
    def update(self, creature, step_count: int):
        """Update tracker with current creature state."""
        if self.start_step == 0:
            self.start_step = step_count
            self.record.birth_step = step_count
        
        # Position & velocity
        if hasattr(creature, 'pos'):
            pos = creature.pos.copy()
            vel = creature.vel.copy() if hasattr(creature, 'vel') else np.zeros(2)
            
            if len(self.positions) > 0:
                self.record.total_distance += np.linalg.norm(pos - self.positions[-1])
            
            self.positions.append(pos)
            self.velocities.append(vel)
            
            speed = np.linalg.norm(vel)
            self.speeds.append(speed)
            self.record.peak_speed = max(self.record.peak_speed, speed)
        
        # Energy
        if hasattr(creature, 'body'):
            E = creature.body.energy
            self.energies.append(E)
            self.record.peak_energy = max(self.record.peak_energy, E)
        
        # Metabolism
        if hasattr(creature, 'metabolism'):
            meta = creature.metabolism
            
            if hasattr(meta, 'hunger'):
                self.hungers.append(meta.hunger)
            
            if hasattr(meta, 'satiation'):
                self.satiations.append(meta.satiation)
            
            if hasattr(meta, 'gut_contents'):
                self.gut_contents.append(meta.gut_contents)
                self.record.peak_gut_contents = max(
                    self.record.peak_gut_contents, meta.gut_contents
                )
        
        # Brain (torus)
        if hasattr(creature, 'torus'):
            torus = creature.torus
            
            if hasattr(torus, 'coherence'):
                coh = torus.coherence
                self.coherences.append(coh)
                self.record.peak_coherence = max(self.record.peak_coherence, coh)
            
            if hasattr(torus, 'circulation'):
                self.circulations.append(torus.circulation)
    
    def finalize(self, step_count: int, cause_of_death: str = "unknown"):
        """Finalize record when creature dies."""
        self.record.death_step = step_count
        self.record.death_step = step_count
        self.record.steps_lived = step_count - self.record.birth_step
        self.record.death_time = datetime.now().isoformat()
        self.record.cause_of_death = cause_of_death
        
        # Calculate means
        if self.energies:
            self.record.mean_energy = float(np.mean(self.energies))
            self.record.energy_variance = float(np.var(self.energies))
        
        if self.speeds:
            self.record.mean_speed = float(np.mean(self.speeds))
        
        if self.hungers:
            self.record.mean_hunger = float(np.mean(self.hungers))
        
        if self.satiations:
            self.record.mean_satiation = float(np.mean(self.satiations))
        
        if self.coherences:
            self.record.mean_coherence = float(np.mean(self.coherences))
        
        if self.circulations:
            self.record.mean_circulation = float(np.mean(self.circulations))
        
        # Calculate exploration radius
        if len(self.positions) > 1:
            center = np.mean(self.positions, axis=0)
            distances = [np.linalg.norm(p - center) for p in self.positions]
            self.record.exploration_radius = float(np.max(distances))
        
        return self.record
    
    def log_defecation(self, amount: float, step: int):
        """Log a defecation event."""
        self.defecation_events.append(step)
        self.record.total_defecations += 1
    
    def log_reproduction(self, step: int):
        """Log a reproduction event."""
        self.reproduction_steps.append(step)
        self.record.offspring_count += 1
    
    def log_consumption(self, amount: float):
        """Log food consumption."""
        self.record.total_food_consumed += amount
    
    def log_starvation_start(self, step: int):
        """Log when starvation state begins."""
        self.starvation_starts.append(step)


class EcologicalBalance:
    """
    Sigmoid-based carrying capacity for population regulation.
    Herbivore capacity = f(food availability)
    Predator capacity = f(prey population) - Lotka-Volterra style
    """
    
    # Base carrying capacities
    HERBIVORE_K = {
        'Herbie': 4,
        'Blob': 6,
        'Biped': 5,
        'Mono': 6,
        'Grazer': 8,
    }
    
    # Predator : prey ratio (1 predator per N prey)
    PREDATOR_PREY_RATIO = {
        'Scavenger': 8,   # 1 scavenger per 8 prey
        'Apex': 15,       # 1 apex per 15 prey (much rarer)
    }
    
    # Sigmoid steepness (higher = sharper cutoff at capacity)
    HERBIVORE_STEEPNESS = 0.8
    PREDATOR_STEEPNESS = 2.5
    
    @classmethod
    def get_spawn_probability(cls, species_name: str, diet: str, 
                               current_pop: int, total_prey: int = 0, 
                               food_count: int = 0) -> float:
        """
        Returns probability [0,1] of spawning this species.
        Uses sigmoid: P = 1 / (1 + exp(k * (N - K)))
        """
        if diet == 'herbivore':
            # Herbivore capacity scales with food
            base_K = cls.HERBIVORE_K.get(species_name, 3)
            food_factor = min(2.0, max(0.5, food_count / 20))
            K = base_K * food_factor
            k = cls.HERBIVORE_STEEPNESS
        else:
            # Predator capacity = prey / ratio
            ratio = cls.PREDATOR_PREY_RATIO.get(species_name, 8)
            K = max(0.5, total_prey / ratio)
            k = cls.PREDATOR_STEEPNESS
        
        # Sigmoid: high when pop << K, low when pop >> K
        # Invert so low pop = high spawn probability
        x = k * (current_pop - K)
        prob = 1.0 - sigmoid(x)
        
        return float(np.clip(prob, 0.0, 1.0))
    
    @classmethod
    def get_natural_death_probability(cls, age: int, max_age: int, 
                                       season_name: str = "") -> float:
        """
        Natural death from old age. 
        Probability increases sharply past 80% lifespan.
        """
        if max_age <= 0:
            return 0.0
        
        age_ratio = age / max_age
        
        if age_ratio < 0.8:
            return 0.0
        
        # Exponential ramp from 80% to 100% lifespan
        base_prob = 0.001 * np.exp(5 * (age_ratio - 0.8))
        
        # Harsher in winter
        if season_name == 'Winter':
            base_prob *= 1.5
        
        return min(0.05, base_prob)  # Cap at 5% per step
    
    @classmethod
    def get_population_pressure(cls, species_counts: Dict[str, int], 
                                food_count: int) -> Dict[str, float]:
        """
        Get population pressure for each species.
        High pressure = overcrowded, low pressure = room to grow.
        """
        total_prey = sum(v for k, v in species_counts.items() 
                        if k not in ['Apex', 'Scavenger'])
        
        pressures = {}
        for species, count in species_counts.items():
            if species in cls.HERBIVORE_K:
                base_K = cls.HERBIVORE_K[species]
                food_factor = min(2.0, max(0.5, food_count / 20))
                K = base_K * food_factor
            elif species in cls.PREDATOR_PREY_RATIO:
                ratio = cls.PREDATOR_PREY_RATIO[species]
                K = max(0.5, total_prey / ratio)
            else:
                K = 5.0  # Default
            
            pressures[species] = count / K if K > 0 else 0.0
        
        return pressures


# =============================================================================
# EVOLUTION HISTORY - Persistent storage and analysis
# =============================================================================

class EvolutionHistory:
    """Persistent storage and analysis of all lifetime records."""
    
    def __init__(self, filepath: str = None):
        self.records: List[LifetimeRecord] = []
        self.lineages: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        self.filepath = filepath
        
        if filepath:
            self._load()
    
    def _load(self):
        """Load records from file."""
        import json
        import os
        
        if not self.filepath or not os.path.exists(self.filepath):
            return
        
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            self.records = [LifetimeRecord.from_dict(r) for r in data.get('records', [])]
            self.lineages = data.get('lineages', {})
            print(f"[Evolution] Loaded {len(self.records)} lifetime records")
        except Exception as e:
            print(f"[Evolution] Load failed: {e}")
    
    def save(self):
        """Save records to file."""
        import json
        
        if not self.filepath:
            return
        
        try:
            with open(self.filepath, 'w') as f:
                json.dump({
                    'records': [r.to_dict() for r in self.records],
                    'lineages': self.lineages,
                    'last_updated': datetime.now().isoformat(),
                    'total_lifetimes': len(self.records),
                    'statistics': self.compute_global_stats()
                }, f, indent=2)
        except Exception as e:
            print(f"[Evolution] Save failed: {e}")
    
    def add(self, record: LifetimeRecord):
        """Add a lifetime record."""
        self.records.append(record)
        
        # Track lineage
        if record.parent_id:
            if record.parent_id not in self.lineages:
                self.lineages[record.parent_id] = []
            self.lineages[record.parent_id].append(record.creature_id)
        
        # Auto-save periodically
        if len(self.records) % 5 == 0:
            self.save()
            print(f"[Evolution] Auto-saved {len(self.records)} records")
    
    def compute_global_stats(self) -> dict:
        """Compute statistics across all records."""
        if not self.records:
            return {}
        
        return {
            'total_creatures': len(self.records),
            'total_generations': max(r.generation for r in self.records) + 1,
            'mean_lifespan': float(np.mean([r.steps_lived for r in self.records])),
            'max_lifespan': max(r.steps_lived for r in self.records),
            'total_offspring': sum(r.offspring_count for r in self.records),
            'mean_consumption': float(np.mean([r.total_consumed for r in self.records])),
            'mean_fitness': float(np.mean([r.fitness_score() for r in self.records])),
            'causes_of_death': self._count_causes(),
            'lineage_depths': [r.lineage_depth for r in self.records],
        }
    
    def _count_causes(self) -> dict:
        """Count causes of death."""
        causes = {}
        for r in self.records:
            causes[r.cause_of_death] = causes.get(r.cause_of_death, 0) + 1
        return causes
    
    def get_generation_stats(self) -> dict:
        """Statistics grouped by generation."""
        if not self.records:
            return {}
        
        gens = {}
        for r in self.records:
            g = r.generation
            if g not in gens:
                gens[g] = []
            gens[g].append(r)
        
        stats = {}
        for g, recs in gens.items():
            stats[g] = {
                'count': len(recs),
                'mean_lifespan': float(np.mean([r.steps_lived for r in recs])),
                'mean_consumed': float(np.mean([r.total_consumed for r in recs])),
                'mean_offspring': float(np.mean([r.offspring_count for r in recs])),
                'mean_fitness': float(np.mean([r.fitness_score() for r in recs])),
            }
        return stats
    
    def get_lineage_tree(self, root_id: str = None) -> dict:
        """Get family tree starting from root."""
        if root_id is None:
            roots = [r.creature_id for r in self.records if not r.parent_id]
            return {root: self.get_lineage_tree(root) for root in roots}
        
        children = self.lineages.get(root_id, [])
        return {child: self.get_lineage_tree(child) for child in children}


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def create_lifetime_plot(record: LifetimeRecord, tracker: LifetimeTracker, 
                         save_path: str, world_size: float = 100.0):
    """Generate comprehensive summary plot for one lifetime."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(16, 12), facecolor='black')
    fig.suptitle(
        f'Lifetime Analysis: Gen {record.generation} | {record.steps_lived} steps | {record.cause_of_death}',
        color='white', fontsize=14, fontweight='bold'
    )
    
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # Row 1: Key metrics bars
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.set_facecolor('black')
    metrics = ['Consumed', 'Distance/10', 'Offspring×10', 'Objects×5', 'Fitness']
    values = [record.total_consumed, record.total_distance/10, 
              record.offspring_count*10, record.objects_eaten*5, record.fitness_score()*10]
    colors = ['lime', 'cyan', 'yellow', 'orange', 'magenta']
    ax1.barh(metrics, values, color=colors, alpha=0.7)
    ax1.set_title('Cumulative Stats', color='white', fontsize=10)
    ax1.tick_params(colors='white', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color('gray')
    
    # Row 1: Brain-body metrics
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.set_facecolor('black')
    metrics2 = ['Brain H', 'Body H', 'Move H', 'Coordination']
    values2 = [record.brain_entropy, record.body_entropy, 
               record.movement_entropy, record.limb_coordination]
    colors2 = ['purple', 'blue', 'magenta', 'cyan']
    ax2.barh(metrics2, values2, color=colors2, alpha=0.7)
    ax2.axvline(0, color='white', alpha=0.3)
    ax2.set_title('Brain-Body Metrics', color='white', fontsize=10)
    ax2.tick_params(colors='white', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('gray')
    
    # Row 2: Energy over time
    if tracker.energies:
        ax3 = fig.add_subplot(gs[1, 0:2])
        ax3.set_facecolor('black')
        ax3.plot(tracker.energies, 'orange', lw=0.8, alpha=0.8, label='Energy')
        ax3.set_title('Energy over Time', color='orange', fontsize=10)
        ax3.tick_params(colors='white', labelsize=7)
        for spine in ax3.spines.values():
            spine.set_color('gray')
    
    # Row 2: Hunger & Satiation
    if tracker.hungers:
        ax4 = fig.add_subplot(gs[1, 2:4])
        ax4.set_facecolor('black')
        ax4.plot(tracker.hungers, 'red', lw=0.8, alpha=0.8, label='Hunger')
        if tracker.satiations:
            ax4.plot(tracker.satiations, 'lime', lw=0.8, alpha=0.6, label='Satiation')
        ax4.set_title('Hunger & Satiation', color='red', fontsize=10)
        ax4.legend(loc='upper right', fontsize=7, facecolor='black', labelcolor='white')
        ax4.tick_params(colors='white', labelsize=7)
        for spine in ax4.spines.values():
            spine.set_color('gray')
    
    # Row 3: Trajectory
    if tracker.positions:
        ax5 = fig.add_subplot(gs[2, 0:2])
        ax5.set_facecolor('black')
        path = np.array(tracker.positions)
        n = len(path)
        for i in range(n - 1):
            alpha = 0.1 + 0.9 * (i / n)
            ax5.plot(path[i:i+2, 0], path[i:i+2, 1], 'c-', lw=0.5, alpha=alpha)
        ax5.plot(path[0, 0], path[0, 1], 'go', ms=8, label='Birth')
        ax5.plot(path[-1, 0], path[-1, 1], 'ro', ms=8, label='Death')
        half = world_size / 2
        ax5.set_xlim(-half, half)
        ax5.set_ylim(-half, half)
        ax5.set_aspect('equal')
        ax5.set_title('Life Trajectory', color='cyan', fontsize=10)
        ax5.legend(loc='upper right', fontsize=7, facecolor='black', labelcolor='white')
        ax5.tick_params(colors='white', labelsize=7)
        for spine in ax5.spines.values():
            spine.set_color('gray')
    
    # Row 4: Text summary
    ax7 = fig.add_subplot(gs[3, 0:2])
    ax7.set_facecolor('black')
    ax7.axis('off')
    summary_text = f"""
    CREATURE: {record.creature_id[:16]}
    PARENT:   {record.parent_id[:16] if record.parent_id else 'None (Gen 0)'}
    LINEAGE:  Depth {record.lineage_depth}
    
    Lifespan:      {record.steps_lived:,} steps
    Peak Energy:   {record.peak_energy:.1f}
    Peak Speed:    {record.peak_speed:.3f}
    Mean Hunger:   {record.mean_hunger:.3f}
    """
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=9,
             color='white', family='monospace', verticalalignment='top')
    
    plt.savefig(save_path, facecolor='black', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Evolution] Saved lifetime plot: {save_path}")


def create_evolution_summary_plot(history: EvolutionHistory, save_path: str):
    """Generate cross-generational analysis plot."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    if len(history.records) < 3:
        print("[Evolution] Not enough records for summary plot")
        return
    
    fig = plt.figure(figsize=(16, 12), facecolor='black')
    fig.suptitle(f'Evolution Summary ({len(history.records)} lifetimes)', 
                 color='white', fontsize=14, fontweight='bold')
    
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    records = history.records
    gens = [r.generation for r in records]
    lifespans = [r.steps_lived for r in records]
    consumed = [r.total_consumed for r in records]
    fitness = [r.fitness_score() for r in records]
    brain_h = [r.brain_entropy for r in records]
    
    # Lifespan vs generation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('black')
    ax1.scatter(gens, lifespans, c='cyan', alpha=0.6, s=30)
    if len(set(gens)) > 1:
        z = np.polyfit(gens, lifespans, 1)
        x_fit = [min(gens), max(gens)]
        ax1.plot(x_fit, [z[0]*x + z[1] for x in x_fit], 'yellow', lw=2, 
                label=f'Trend: {z[0]:+.0f}/gen')
        ax1.legend(facecolor='black', labelcolor='white', fontsize=8)
    ax1.set_xlabel('Generation', color='white')
    ax1.set_ylabel('Lifespan', color='white')
    ax1.set_title('Lifespan Evolution', color='cyan')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('gray')
    
    # Fitness vs generation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('black')
    ax2.scatter(gens, fitness, c='magenta', alpha=0.6, s=30)
    if len(set(gens)) > 1:
        z = np.polyfit(gens, fitness, 1)
        x_fit = [min(gens), max(gens)]
        ax2.plot(x_fit, [z[0]*x + z[1] for x in x_fit], 'yellow', lw=2,
                label=f'Trend: {z[0]:+.2f}/gen')
        ax2.legend(facecolor='black', labelcolor='white', fontsize=8)
    ax2.set_xlabel('Generation', color='white')
    ax2.set_ylabel('Fitness Score', color='white')
    ax2.set_title('Fitness Evolution', color='magenta')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('gray')
    
    # Consumption vs generation
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('black')
    ax3.scatter(gens, consumed, c='lime', alpha=0.6, s=30)
    ax3.set_xlabel('Generation', color='white')
    ax3.set_ylabel('Total Consumed', color='white')
    ax3.set_title('Feeding Success', color='lime')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('gray')
    
    # Brain entropy vs lifespan
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor('black')
    valid = [(h, l) for h, l in zip(brain_h, lifespans) if h > 0]
    if valid:
        h_vals, l_vals = zip(*valid)
        ax4.scatter(h_vals, l_vals, c='purple', alpha=0.6, s=30)
    ax4.set_xlabel('Brain Entropy', color='white')
    ax4.set_ylabel('Lifespan', color='white')
    ax4.set_title('Brain Entropy vs Lifespan', color='purple')
    ax4.tick_params(colors='white')
    for spine in ax4.spines.values():
        spine.set_color('gray')
    
    # Causes of death pie chart
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor('black')
    causes = history._count_causes()
    if causes:
        labels = list(causes.keys())
        sizes = list(causes.values())
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        ax6.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%',
               textprops={'color': 'white', 'fontsize': 8})
        ax6.set_title('Causes of Death', color='white')
    
    # Generation statistics table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.set_facecolor('black')
    ax7.axis('off')
    
    gen_stats = history.get_generation_stats()
    if gen_stats:
        header = "Gen | Count | Lifespan | Consumed | Offspring | Fitness"
        rows = [header, "-" * 60]
        for g in sorted(gen_stats.keys()):
            s = gen_stats[g]
            rows.append(f" {g:2d} | {s['count']:5d} | {s['mean_lifespan']:8.0f} | "
                       f"{s['mean_consumed']:8.1f} | {s['mean_offspring']:9.2f} | "
                       f"{s['mean_fitness']:7.2f}")
        
        ax7.text(0.05, 0.95, '\n'.join(rows), transform=ax7.transAxes,
                fontsize=9, color='white', family='monospace', verticalalignment='top')
    
    plt.savefig(save_path, facecolor='black', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Evolution] Saved evolution summary: {save_path}")
