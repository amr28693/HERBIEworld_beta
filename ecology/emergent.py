"""
Emergent Behavior Systems - Observation and tracking of emergent phenomena.

These systems are PURELY OBSERVATIONAL - they document emergent patterns
without modifying creature behavior. The emergence is in the creatures;
these systems just make it visible.

Systems:
- CultureTracker: Lineage divergence, territorial patterns
- NestTracker: Emergent homesteading from defecation patterns
- BrainStateTracker: Entropy/KL metrics for proto-emotional states
- DiggingSystem: Mechanical hole creation (emergent burial/caching)
- SmearSystem: Pigment marks encoding brain state (emergent art)
- FavoriteHerbieTracker: User favorites and family trees
- AntCreatureInteraction: Ant-creature physics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING
import numpy as np

from ..core.constants import WORLD_L, BODY_L, Nx, Ny, dx, dt, X, Y

if TYPE_CHECKING:
    from ..creature.creature import Creature


# =============================================================================
# BRAIN SNAPSHOT - Data container for brain state metrics
# =============================================================================

@dataclass
class BrainSnapshot:
    """Snapshot of brain state metrics at a moment in time."""
    creature_id: str = ""
    step: int = 0
    
    # Torus (cognitive) metrics
    torus_entropy: float = 0.0
    torus_kl_from_uniform: float = 0.0
    torus_coherence: float = 0.0
    torus_circulation: float = 0.0
    torus_energy: float = 0.0
    
    # Body (somatic) metrics
    body_entropy: float = 0.0
    body_kl_from_uniform: float = 0.0
    body_energy: float = 0.0
    
    # Derived complexity (norm of entropy and KL)
    cognitive_complexity: float = 0.0
    somatic_complexity: float = 0.0


# =============================================================================
# CULTURE TRACKER - Observes lineage divergence and territorial patterns
# =============================================================================

class CultureTracker:
    """
    Observes and measures emergent cultural patterns.
    PURELY OBSERVATIONAL - does not modify any creature behavior.
    Just documents what's already happening.
    """
    
    def __init__(self):
        self.observations = []
        self.last_observation_step = 0
        self.observation_interval = 1000  # Log every 1000 steps
        
    def observe(self, creatures: List, step: int, leviathan_mgr=None) -> Optional[dict]:
        """
        Observe current state without modifying anything.
        Returns observation dict if interval reached, else None.
        """
        if step - self.last_observation_step < self.observation_interval:
            return None
        
        self.last_observation_step = step
        
        # Get Herbies only
        herbies = [c for c in creatures 
                   if c.species.name == "Herbie" and c.alive and hasattr(c, 'mating_state')]
        
        if len(herbies) < 2:
            return None
        
        observation = {
            'step': step,
            'population': len(herbies),
            'lineages': {},
            'spatial': {},
            'behavioral': {},
        }
        
        # === LINEAGE ANALYSIS ===
        lineages = {}
        for h in herbies:
            founder = self._find_founder(h, creatures)
            founder_name = founder.mating_state.name if founder and hasattr(founder, 'mating_state') else 'Unknown'
            
            if founder_name not in lineages:
                lineages[founder_name] = []
            lineages[founder_name].append(h)
        
        # === SPATIAL CLUSTERING ===
        for lineage_name, members in lineages.items():
            if not members:
                continue
                
            positions = np.array([m.pos for m in members])
            center = np.mean(positions, axis=0)
            
            if len(members) > 1:
                distances = [np.linalg.norm(m.pos - center) for m in members]
                cohesion = np.mean(distances)
            else:
                cohesion = 0.0
            
            # Terrain preference
            terrain_counts = {}
            for m in members:
                if hasattr(m, 'terrain') and m.terrain:
                    t = m.terrain.get_terrain_at(m.pos)
                    t_name = t.name if t else 'unknown'
                    terrain_counts[t_name] = terrain_counts.get(t_name, 0) + 1
            
            observation['lineages'][lineage_name] = {
                'count': len(members),
                'center': center.tolist(),
                'cohesion': cohesion,
                'terrain_pref': terrain_counts,
                'generations': [m.generation for m in members],
            }
        
        # === BEHAVIORAL DIVERGENCE ===
        if len(lineages) >= 2:
            lineage_potentials = {}
            for lineage_name, members in lineages.items():
                potentials = []
                for m in members:
                    if hasattr(m, 'learned_potential') and m.learned_potential:
                        pot_sum = sum(m.learned_potential.values()) if isinstance(m.learned_potential, dict) else 0
                        potentials.append(pot_sum)
                if potentials:
                    lineage_potentials[lineage_name] = np.mean(potentials)
            
            if len(lineage_potentials) >= 2:
                values = list(lineage_potentials.values())
                if np.mean(values) > 0:
                    divergence = np.std(values) / np.mean(values)
                else:
                    divergence = 0.0
                observation['behavioral']['divergence'] = divergence
                observation['behavioral']['lineage_potentials'] = lineage_potentials
        
        # === TERRITORY OVERLAP ===
        if len(lineages) >= 2:
            centers = [obs['center'] for obs in observation['lineages'].values()]
            if len(centers) >= 2:
                center_distances = []
                for i, c1 in enumerate(centers):
                    for c2 in centers[i+1:]:
                        center_distances.append(np.linalg.norm(np.array(c1) - np.array(c2)))
                observation['spatial']['avg_lineage_separation'] = np.mean(center_distances)
        
        # === LEVIATHAN LEGACY ===
        if leviathan_mgr and hasattr(leviathan_mgr, 'fertile_patches'):
            for lineage_name, members in lineages.items():
                near_fertile = 0
                for m in members:
                    for patch in leviathan_mgr.fertile_patches:
                        if np.linalg.norm(m.pos - patch.pos) < patch.radius * 1.5:
                            near_fertile += 1
                            break
                if lineage_name in observation['lineages']:
                    observation['lineages'][lineage_name]['near_leviathan_legacy'] = near_fertile
        
        self.observations.append(observation)
        self._log_observation(observation)
        
        return observation
    
    def _find_founder(self, herbie, all_creatures) -> Optional[Any]:
        """Trace back to generation 0 ancestor."""
        if herbie.generation == 0:
            return herbie
        
        if hasattr(herbie, 'mating_state') and herbie.mating_state.parent_ids:
            parent_ids = herbie.mating_state.parent_ids
            for c in all_creatures:
                if c.creature_id in parent_ids:
                    return self._find_founder(c, all_creatures)
        
        return herbie
    
    def _log_observation(self, obs: dict):
        """Print observation summary."""
        print(f"\n{'='*60}")
        print(f"[CULTURE OBSERVATION] Step {obs['step']} | Population: {obs['population']}")
        print(f"{'='*60}")
        
        for lineage_name, data in obs['lineages'].items():
            center = data['center']
            gens = data['generations']
            gen_range = f"G{min(gens)}-G{max(gens)}" if gens else "?"
            terrain = max(data['terrain_pref'].items(), key=lambda x: x[1])[0] if data['terrain_pref'] else '?'
            legacy = data.get('near_leviathan_legacy', 0)
            legacy_str = f" | near sacred ground: {legacy}" if legacy > 0 else ""
            
            print(f"  Lineage '{lineage_name}': {data['count']} members ({gen_range})")
            print(f"    territory: ({center[0]:+.0f}, {center[1]:+.0f}) | primary terrain: {terrain}{legacy_str}")
            print(f"    cohesion: {data['cohesion']:.1f} (lower = tighter group)")
        
        if 'divergence' in obs.get('behavioral', {}):
            div = obs['behavioral']['divergence']
            div_desc = "identical" if div < 0.1 else "similar" if div < 0.3 else "diverging" if div < 0.6 else "distinct"
            print(f"  Behavioral divergence: {div:.2f} ({div_desc})")
        
        if 'avg_lineage_separation' in obs.get('spatial', {}):
            sep = obs['spatial']['avg_lineage_separation']
            print(f"  Lineage separation: {sep:.1f} units (higher = more territorial)")
        
        print(f"{'='*60}\n")
    
    def get_summary(self) -> dict:
        """Get summary of all observations."""
        if not self.observations:
            return {}
        
        return {
            'total_observations': len(self.observations),
            'time_span': (self.observations[0]['step'], self.observations[-1]['step']),
            'max_lineages': max(len(o['lineages']) for o in self.observations),
            'observations': self.observations
        }


# =============================================================================
# HERBIE NEST - Data container for emergent homesteads
# =============================================================================

@dataclass
class HerbieNest:
    """An emergent nest formed by Herbie homesteading."""
    center: np.ndarray
    owner_id: str
    owner_name: str
    formation_step: int
    nutrient_count: int = 0
    time_occupied: int = 0
    last_occupied_step: int = 0
    visitors: List[str] = field(default_factory=list)
    is_abandoned: bool = False
    decay_progress: float = 0.0


# =============================================================================
# NEST TRACKER - Emergent homesteading from creature behavior
# =============================================================================

class NestTracker:
    """
    Tracks emergent nest formation from Herbie behavior.
    A nest forms when a Herbie stays in one area and defecates repeatedly.
    PURELY OBSERVATIONAL - nests are recognized, not created.
    
    Abandoned nests decay over time but can be reinhabited!
    """
    
    NEST_RADIUS = 8.0
    NEST_FORMATION_TIME = 400
    NEST_MIN_NUTRIENTS = 3
    NEST_DECAY_TIME = 3000
    
    def __init__(self):
        self.nests: Dict[str, HerbieNest] = {}
        self.herbie_positions: Dict[str, List[np.ndarray]] = {}
        self.position_history_length = 100
    
    def update(self, creatures: List, nutrients: List, step: int) -> List[HerbieNest]:
        """
        Observe creatures and detect nest formation.
        Returns list of newly formed nests.
        """
        new_nests = []
        
        herbies = [c for c in creatures 
                   if c.species.name == "Herbie" and c.alive and hasattr(c, 'mating_state')]
        
        for herbie in herbies:
            hid = herbie.creature_id
            
            # Track position history
            if hid not in self.herbie_positions:
                self.herbie_positions[hid] = []
            
            self.herbie_positions[hid].append(herbie.pos.copy())
            if len(self.herbie_positions[hid]) > self.position_history_length:
                self.herbie_positions[hid].pop(0)
            
            # Check if staying in one area
            if len(self.herbie_positions[hid]) >= self.position_history_length:
                positions = np.array(self.herbie_positions[hid])
                center = np.mean(positions, axis=0)
                spread = np.mean([np.linalg.norm(p - center) for p in positions])
                
                if spread < self.NEST_RADIUS:
                    nearby_nutrients = sum(1 for n in nutrients 
                                          if np.linalg.norm(n.pos - center) < self.NEST_RADIUS)
                    
                    if nearby_nutrients >= self.NEST_MIN_NUTRIENTS:
                        if hid not in self.nests:
                            name = herbie.mating_state.name if hasattr(herbie, 'mating_state') else hid[:8]
                            nest = HerbieNest(
                                center=center.copy(),
                                owner_id=hid,
                                owner_name=name,
                                formation_step=step,
                                nutrient_count=nearby_nutrients,
                                time_occupied=self.position_history_length,
                                last_occupied_step=step
                            )
                            self.nests[hid] = nest
                            new_nests.append(nest)
                            msg = f"[NEST] '{name}' established a homestead at ({center[0]:+.0f}, {center[1]:+.0f})!"
                            print(msg)
                            
                            # Log to event system
                            try:
                                from ..events.logger import event_log
                                from ..events.narrative_log import narrative_log
                                event_log().log('nest_established', step,
                                    owner=name, pos=(center[0], center[1]), nutrients=nearby_nutrients)
                                narrative_log().log(msg, step, force=True)
                            except ImportError:
                                pass
                        else:
                            self.nests[hid].nutrient_count = nearby_nutrients
                            self.nests[hid].time_occupied += 1
                            self.nests[hid].last_occupied_step = step
            
            # Check for nest visits
            for nest_owner, nest in self.nests.items():
                if nest_owner != hid:
                    if np.linalg.norm(herbie.pos - nest.center) < self.NEST_RADIUS:
                        if hid not in nest.visitors:
                            nest.visitors.append(hid)
                            visitor_name = herbie.mating_state.name if hasattr(herbie, 'mating_state') else hid[:8]
                            print(f"[NEST] '{visitor_name}' visited '{nest.owner_name}'s homestead")
        
        # Process abandoned nests
        for hid in list(self.nests.keys()):
            nest = self.nests[hid]
            owner_alive = any(c.creature_id == hid and c.alive for c in creatures)
            
            if not owner_alive:
                if not nest.is_abandoned and step - nest.last_occupied_step > 500:
                    nest.is_abandoned = True
                    print(f"[NEST] '{nest.owner_name}'s homestead lies abandoned...")
                
                if nest.is_abandoned:
                    nest.decay_progress += 1.0 / self.NEST_DECAY_TIME
                    
                    # Check for reinhabiting
                    for herbie in herbies:
                        if np.linalg.norm(herbie.pos - nest.center) < self.NEST_RADIUS:
                            new_name = herbie.mating_state.name if hasattr(herbie, 'mating_state') else herbie.creature_id[:8]
                            print(f"[NEST] '{new_name}' has claimed '{nest.owner_name}'s old homestead!")
                            self.nests[herbie.creature_id] = nest
                            del self.nests[hid]
                            nest.owner_id = herbie.creature_id
                            nest.owner_name = new_name
                            nest.is_abandoned = False
                            nest.decay_progress = 0.0
                            nest.last_occupied_step = step
                            break
                    
                    if nest.decay_progress >= 1.0:
                        print(f"[NEST] '{nest.owner_name}'s homestead has returned to the wild.")
                        del self.nests[hid]
        
        return new_nests
    
    def get_nests_for_display(self) -> List[Tuple[np.ndarray, str, int, bool, float]]:
        """Return (center, owner_name, nutrient_count, is_active, decay_progress)."""
        return [(n.center, n.owner_name, n.nutrient_count, not n.is_abandoned, n.decay_progress) 
                for n in self.nests.values()]
    
    def get_nest_at(self, pos: np.ndarray) -> Optional[HerbieNest]:
        """Check if position is within any nest."""
        for nest in self.nests.values():
            if np.linalg.norm(pos - nest.center) < self.NEST_RADIUS:
                return nest
        return None


# =============================================================================
# HOLE - Data container for dug holes
# =============================================================================

@dataclass
class Hole:
    """A hole dug by a Herbie - purely physical object."""
    pos: np.ndarray
    depth: float = 0.0
    digger_id: str = ""
    digger_name: str = ""
    creation_step: int = 0
    contents: List = field(default_factory=list)
    is_covered: bool = False
    cover_step: int = 0


# =============================================================================
# DIGGING SYSTEM - Landscape manipulation by Herbies
# =============================================================================

class DiggingSystem:
    """
    PURELY MECHANICAL hole system.
    
    Herbies can:
    1. Dig holes (when stationary + has energy + probabilistic)
    2. Objects (including corpses) can FALL INTO holes
    3. Cover holes (push dirt back, probabilistic)
    
    NO special logic for burial or caching - just physics.
    If a Herbie digs near where something died, and the corpse
    falls in, and then the Herbie covers it... that's burial.
    But the code doesn't know or care.
    """
    
    HOLE_RADIUS = 2.0
    MAX_DEPTH = 1.0
    DIG_RATE = 0.1
    DIG_ENERGY_COST = 2.0
    DIG_STATIONARY_THRESHOLD = 0.3
    HOLE_DECAY_TIME = 8000
    FALL_IN_DEPTH_THRESHOLD = 0.5
    
    def __init__(self):
        self.holes: Dict[str, Hole] = {}
        self.next_hole_id = 0
    
    def update(self, creatures: List, world_objects: List, step: int) -> List[str]:
        """Process digging and hole physics."""
        events = []
        
        herbies = [c for c in creatures 
                   if c.species.name == "Herbie" and c.alive and hasattr(c, 'hands')]
        
        for herbie in herbies:
            if self._should_dig(herbie):
                event = self._try_dig(herbie, step)
                if event:
                    events.append(event)
            
            if self._should_cover(herbie):
                event = self._try_cover(herbie, step)
                if event:
                    events.append(event)
        
        # Objects fall into holes
        for obj in world_objects:
            if hasattr(obj, 'pos'):
                event = self._check_fall_in(obj, step)
                if event:
                    events.append(event)
        
        # Corpses fall into holes
        for creature in creatures:
            if not creature.alive and hasattr(creature, 'pos'):
                event = self._check_corpse_fall(creature, step)
                if event:
                    events.append(event)
        
        self._decay_holes(step)
        
        return events
    
    def _should_dig(self, herbie) -> bool:
        """Probabilistic dig check."""
        if np.linalg.norm(herbie.vel) > self.DIG_STATIONARY_THRESHOLD:
            return False
        
        if not hasattr(herbie, 'body') or herbie.body.energy < self.DIG_ENERGY_COST * 2:
            return False
        
        dig_chance = 0.0003
        if hasattr(herbie, '_curiosity'):
            dig_chance *= (1 + herbie._curiosity * 2)
        
        return np.random.random() < dig_chance
    
    def _try_dig(self, herbie, step: int) -> Optional[str]:
        """Dig or deepen a hole."""
        pos = herbie.pos.copy()
        existing = self._get_hole_at(pos)
        
        if existing and not existing.is_covered:
            if existing.depth < self.MAX_DEPTH:
                existing.depth = min(self.MAX_DEPTH, existing.depth + self.DIG_RATE)
                herbie.body.energy -= self.DIG_ENERGY_COST
            return None
        
        hole_id = f"hole_{self.next_hole_id}"
        self.next_hole_id += 1
        
        name = herbie.mating_state.name if hasattr(herbie, 'mating_state') else herbie.creature_id[:8]
        
        hole = Hole(
            pos=pos.copy(),
            depth=self.DIG_RATE,
            digger_id=herbie.creature_id,
            digger_name=name,
            creation_step=step
        )
        self.holes[hole_id] = hole
        herbie.body.energy -= self.DIG_ENERGY_COST
        
        return f"[DIG] '{name}' dug at ({pos[0]:+.0f}, {pos[1]:+.0f})"
    
    def _should_cover(self, herbie) -> bool:
        """Probabilistic cover check."""
        hole = self._get_hole_at(herbie.pos)
        if not hole or hole.is_covered:
            return False
        return np.random.random() < 0.0005
    
    def _try_cover(self, herbie, step: int) -> Optional[str]:
        """Cover a hole."""
        hole = self._get_hole_at(herbie.pos)
        if not hole or hole.is_covered:
            return None
        
        hole.is_covered = True
        hole.cover_step = step
        name = herbie.mating_state.name if hasattr(herbie, 'mating_state') else herbie.creature_id[:8]
        
        contents_desc = f" (contains {len(hole.contents)} objects)" if hole.contents else ""
        return f"[COVER] '{name}' filled in hole{contents_desc}"
    
    def _check_fall_in(self, obj, step: int) -> Optional[str]:
        """Check if object falls into a hole."""
        for hole in self.holes.values():
            if hole.is_covered:
                continue
            if hole.depth < self.FALL_IN_DEPTH_THRESHOLD:
                continue
            if np.linalg.norm(obj.pos - hole.pos) < self.HOLE_RADIUS * 0.5:
                if obj not in hole.contents:
                    hole.contents.append(obj)
                    obj.pos = hole.pos.copy()
                    return f"[FALL] Object fell into hole at ({hole.pos[0]:+.0f}, {hole.pos[1]:+.0f})"
        return None
    
    def _check_corpse_fall(self, creature, step: int) -> Optional[str]:
        """Check if dead creature falls into hole."""
        for hole in self.holes.values():
            if hole.is_covered:
                continue
            if hole.depth < self.FALL_IN_DEPTH_THRESHOLD:
                continue
            if np.linalg.norm(creature.pos - hole.pos) < self.HOLE_RADIUS * 0.5:
                if creature not in hole.contents:
                    hole.contents.append(creature)
                    name = "unknown"
                    if hasattr(creature, 'mating_state') and creature.mating_state.name:
                        name = creature.mating_state.name
                    return f"[FALL] Remains of '{name}' fell into hole"
        return None
    
    def _get_hole_at(self, pos: np.ndarray) -> Optional[Hole]:
        """Get hole at position."""
        for hole in self.holes.values():
            if np.linalg.norm(pos - hole.pos) < self.HOLE_RADIUS:
                return hole
        return None
    
    def _decay_holes(self, step: int):
        """Uncovered empty holes slowly fill in."""
        to_remove = []
        for hole_id, hole in self.holes.items():
            if not hole.is_covered and not hole.contents:
                if step - hole.creation_step > self.HOLE_DECAY_TIME:
                    to_remove.append(hole_id)
        for hole_id in to_remove:
            del self.holes[hole_id]
    
    def get_holes_for_display(self) -> List[Tuple[np.ndarray, float, bool, int, str]]:
        """Return (pos, depth, is_covered, contents_count, digger_name)."""
        return [(h.pos, h.depth, h.is_covered, len(h.contents), h.digger_name)
                for h in self.holes.values()]


# =============================================================================
# SMEAR MARK - Data container for pigment marks
# =============================================================================

@dataclass  
class SmearMark:
    """A single smear mark on a surface."""
    pos: np.ndarray
    color: Tuple[float, float, float]
    intensity: float = 1.0
    creation_step: int = 0
    creator_id: str = ""
    torus_phase: float = 0.0
    body_entropy: float = 0.0


# =============================================================================
# SMEARABLE OBJECT - Pigment that can be dragged to make marks
# =============================================================================

class SmearableObject:
    """An object that can leave marks when dragged."""
    
    def __init__(self, pos: np.ndarray, color: Tuple[float, float, float], 
                 pigment_amount: float = 100.0):
        self.pos = pos.copy() if isinstance(pos, np.ndarray) else np.array(pos)
        self.color = color
        self.pigment_amount = pigment_amount
        self.is_smearable = True
        self.object_id = f"pigment_{np.random.randint(10000)}"
        
        # WorldObject compatibility
        self.alive = True
        self.compliance = 0.3
        self.size = 0.8
        self.mass = 0.2
        self.energy = 5.0
        self.contact = 0.0
        self.reward = 0.0
        self.vel = np.zeros(2)
        self.display_vel = np.zeros(2)
        self.initial_size = self.size
        self.max_energy = self.energy
    
    def compute_contact(self, creature_pos: np.ndarray, body_I: np.ndarray) -> float:
        """Compute contact with creature body field."""
        if not self.alive:
            self.contact = 0.0
            return 0.0
        rel = self.pos - creature_pos
        obj_i = int((rel[0] + BODY_L/2) / dx)
        obj_j = int((rel[1] + BODY_L/2) / dx)
        if 0 <= obj_i < Nx and 0 <= obj_j < Ny:
            i_min, i_max = max(0, obj_i - 5), min(Nx, obj_i + 6)
            j_min, j_max = max(0, obj_j - 5), min(Ny, obj_j + 6)
            region = body_I[j_min:j_max, i_min:i_max]
            self.contact = float(np.mean(region)) if region.size > 0 else 0.0
        else:
            self.contact = 0.0
        return self.contact
    
    def compute_reward(self) -> Tuple[float, float]:
        """Compute reward from eating this object."""
        if not self.alive or self.contact < 0.04:
            self.reward = 0.0
            return 0.0, 0.0
        self.reward = self.contact * (self.compliance - 0.35) * 0.5
        return self.reward, 0.0
    
    def apply_push(self, creature_pos: np.ndarray, creature_vel: np.ndarray,
                   body_momentum: np.ndarray, contact_strength: float):
        """Apply push from creature contact."""
        if not self.alive or contact_strength < 0.08:
            return
        direction = self.pos - creature_pos
        dist = np.linalg.norm(direction)
        if dist < 0.1:
            return
        direction = direction / dist
        push_strength = contact_strength * 0.3 / (self.mass + 0.1)
        self.vel += direction * push_strength
    
    def update(self):
        """Update position."""
        if not self.alive:
            return
        self.vel *= 0.94
        self.pos += self.vel * dt
        margin = self.size + 1
        self.pos = np.clip(self.pos, -WORLD_L/2 + margin, WORLD_L/2 - margin)
    
    def get_smear_color(self, intensity: float = 1.0) -> Tuple[float, float, float]:
        """Get color modulated by remaining pigment."""
        fade = min(1.0, self.pigment_amount / 50.0)
        return (self.color[0] * fade * intensity,
                self.color[1] * fade * intensity, 
                self.color[2] * fade * intensity)


def spawn_pigment_object(pos: np.ndarray, pigment_type: str = 'charcoal') -> SmearableObject:
    """
    Create a smearable pigment object.
    
    Types: charcoal, berry, mud, ochre, mineral_blue, mineral_green
    """
    colors = {
        'charcoal': (0.1, 0.1, 0.1),
        'berry': (0.6, 0.1, 0.3),
        'mud': (0.4, 0.3, 0.2),
        'ochre': (0.8, 0.5, 0.2),
        'mineral_blue': (0.2, 0.3, 0.7),
        'mineral_green': (0.2, 0.6, 0.3),
    }
    color = colors.get(pigment_type, (0.5, 0.5, 0.5))
    return SmearableObject(pos, color, pigment_amount=100.0)


# =============================================================================
# SMEAR SYSTEM - Pigment marks encoding brain state
# =============================================================================

class SmearSystem:
    """
    Tracks smear marks left on the world.
    
    PURELY MECHANICAL:
    - Smearable objects exist in world
    - When Herbie moves while holding smearable object, mark is left
    - Marks encode the Herbie's brain state at moment of creation
    - Marks fade over time
    
    NO interpretation - if patterns emerge, that's emergence.
    """
    
    SMEAR_FADE_RATE = 0.0001
    MIN_INTENSITY = 0.1
    SMEAR_SPACING = 0.5
    MAX_MARKS = 2000
    SMEARABLE_TERRAINS = {'plains', 'forest', 'hills', 'cave'}
    
    def __init__(self):
        self.marks: List[SmearMark] = []
        self.last_smear_pos: Dict[str, np.ndarray] = {}
    
    def update(self, creatures: List, terrain, step: int) -> List[str]:
        """Check for smearing and fade existing marks."""
        events = []
        
        herbies = [c for c in creatures 
                   if c.species.name == "Herbie" and c.alive and hasattr(c, 'hands')]
        
        for herbie in herbies:
            event = self._check_smear(herbie, terrain, step)
            if event:
                events.append(event)
        
        self._fade_marks(step)
        
        if len(self.marks) > self.MAX_MARKS:
            self.marks.sort(key=lambda m: m.intensity, reverse=True)
            self.marks = self.marks[:self.MAX_MARKS]
        
        return events
    
    def _check_smear(self, herbie, terrain, step: int) -> Optional[str]:
        """Check if Herbie is smearing."""
        smearable = None
        for hand in [herbie.hands.left, herbie.hands.right]:
            if hand.held_object and hasattr(hand.held_object, 'is_smearable'):
                if hand.held_object.is_smearable:
                    smearable = hand.held_object
                    break
        
        if not smearable:
            return None
        
        if np.linalg.norm(herbie.vel) < 0.1:
            return None
        
        terrain_type = terrain.get_terrain_at(herbie.pos)
        if terrain_type.name.lower() not in self.SMEARABLE_TERRAINS:
            return None
        
        last_pos = self.last_smear_pos.get(herbie.creature_id)
        if last_pos is not None:
            if np.linalg.norm(herbie.pos - last_pos) < self.SMEAR_SPACING:
                return None
        
        if smearable.pigment_amount <= 0:
            return None
        
        # Capture brain state
        torus_phase = 0.0
        body_entropy = 0.0
        if hasattr(herbie, 'torus'):
            torus_I = np.abs(herbie.torus.psi)**2
            peak_idx = np.argmax(torus_I)
            torus_phase = np.angle(herbie.torus.psi[peak_idx])
        if hasattr(herbie, 'body'):
            body_I = np.abs(herbie.body.psi)**2
            body_I_norm = body_I / (np.sum(body_I) + 1e-12)
            body_entropy = float(-np.sum(body_I_norm * np.log(body_I_norm + 1e-12)))
        
        # Modulate color by brain state
        base_color = smearable.get_smear_color()
        phase_mod = 0.5 + 0.5 * np.sin(torus_phase)
        entropy_mod = 0.7 + 0.3 * min(1.0, body_entropy / 5.0)
        
        final_color = (
            min(1.0, base_color[0] * entropy_mod),
            min(1.0, base_color[1] * phase_mod),
            min(1.0, base_color[2] * (1 - phase_mod * 0.3))
        )
        
        mark = SmearMark(
            pos=herbie.pos.copy(),
            color=final_color,
            intensity=1.0,
            creation_step=step,
            creator_id=herbie.creature_id,
            torus_phase=torus_phase,
            body_entropy=body_entropy
        )
        
        self.marks.append(mark)
        self.last_smear_pos[herbie.creature_id] = herbie.pos.copy()
        smearable.pigment_amount -= 1.0
        
        return None  # Silent
    
    def _fade_marks(self, step: int):
        """Fade and remove old marks."""
        surviving = []
        for mark in self.marks:
            mark.intensity -= self.SMEAR_FADE_RATE
            if mark.intensity >= self.MIN_INTENSITY:
                surviving.append(mark)
        self.marks = surviving
    
    def get_marks_for_display(self) -> List[Tuple[np.ndarray, Tuple[float, float, float], float]]:
        """Return (pos, color, intensity) for visualization."""
        return [(m.pos, m.color, m.intensity) for m in self.marks]
    
    def get_marks_by_creator(self, creator_id: str) -> List[SmearMark]:
        """Get all marks by a specific creator."""
        return [m for m in self.marks if m.creator_id == creator_id]
    
    def find_art_clusters(self, min_marks: int = 10, cluster_radius: float = 8.0) -> List[dict]:
        """Find clusters of marks that could be considered 'art'."""
        if len(self.marks) < min_marks:
            return []
        
        clusters = []
        used_marks = set()
        
        for mark in self.marks:
            if id(mark) in used_marks:
                continue
            
            nearby = [m for m in self.marks 
                     if np.linalg.norm(mark.pos - m.pos) < cluster_radius]
            
            if len(nearby) >= min_marks:
                positions = np.array([m.pos for m in nearby])
                center = np.mean(positions, axis=0)
                min_pos = np.min(positions, axis=0)
                max_pos = np.max(positions, axis=0)
                creator_ids = set(m.creator_id for m in nearby)
                
                clusters.append({
                    'center': center,
                    'marks': nearby,
                    'bounds': (min_pos, max_pos),
                    'creator_ids': creator_ids,
                    'mark_count': len(nearby)
                })
                
                for m in nearby:
                    used_marks.add(id(m))
        
        return clusters


# =============================================================================
# BRAIN STATE TRACKER - Entropy/KL for emergent emotion
# =============================================================================

class BrainStateTracker:
    """
    Tracks brain state metrics over time for emergent emotion analysis.
    PURELY OBSERVATIONAL - reads state, never modifies.
    
    Key metrics:
    - Entropy: How "spread out" the field distribution is
    - KL divergence from uniform: How far from "resting state"
    - Circulation: Net directional bias in torus
    - Coherence: How focused attention is
    
    The norm of (entropy, KL) gives a "complexity" measure that could
    correlate with emotional arousal or cognitive engagement.
    """
    
    def __init__(self):
        self.history: Dict[str, List[BrainSnapshot]] = {}
        self.last_observation_step = 0
        self.observation_interval = 50
        self.max_history_per_creature = 200
        
    def observe(self, creatures: List, step: int) -> Dict[str, BrainSnapshot]:
        """Observe brain states."""
        if step - self.last_observation_step < self.observation_interval:
            return {}
        
        self.last_observation_step = step
        snapshots = {}
        
        herbies = [c for c in creatures 
                   if c.species.name == "Herbie" and c.alive and hasattr(c, 'torus')]
        
        for herbie in herbies:
            snapshot = self._compute_snapshot(herbie, step)
            snapshots[herbie.creature_id] = snapshot
            
            if herbie.creature_id not in self.history:
                self.history[herbie.creature_id] = []
            self.history[herbie.creature_id].append(snapshot)
            
            if len(self.history[herbie.creature_id]) > self.max_history_per_creature:
                self.history[herbie.creature_id].pop(0)
        
        return snapshots
    
    def _compute_snapshot(self, herbie, step: int) -> BrainSnapshot:
        """Compute all brain metrics for one Herbie."""
        snapshot = BrainSnapshot(creature_id=herbie.creature_id, step=step)
        
        # Torus metrics
        torus_psi = herbie.torus.psi
        torus_I = np.abs(torus_psi)**2
        torus_I_norm = torus_I / (np.sum(torus_I) + 1e-12)
        
        snapshot.torus_entropy = self._compute_entropy(torus_I_norm)
        uniform = np.ones_like(torus_I_norm) / len(torus_I_norm)
        snapshot.torus_kl_from_uniform = self._compute_kl(torus_I_norm, uniform)
        snapshot.torus_coherence = herbie.torus.coherence
        snapshot.torus_circulation = herbie.torus.circulation
        snapshot.torus_energy = herbie.torus.energy
        
        # Body metrics
        body_psi = herbie.body.psi
        body_I = np.abs(body_psi)**2
        body_I_flat = body_I.flatten()
        body_I_norm = body_I_flat / (np.sum(body_I_flat) + 1e-12)
        
        snapshot.body_entropy = self._compute_entropy(body_I_norm)
        uniform_body = np.ones_like(body_I_norm) / len(body_I_norm)
        snapshot.body_kl_from_uniform = self._compute_kl(body_I_norm, uniform_body)
        snapshot.body_energy = float(np.sum(body_I))
        
        # Derived complexity
        snapshot.cognitive_complexity = np.sqrt(
            snapshot.torus_entropy**2 + snapshot.torus_kl_from_uniform**2
        )
        snapshot.somatic_complexity = np.sqrt(
            snapshot.body_entropy**2 + snapshot.body_kl_from_uniform**2
        )
        
        return snapshot
    
    def _compute_entropy(self, p: np.ndarray) -> float:
        """Compute Shannon entropy."""
        p_safe = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(p_safe * np.log(p_safe)))
    
    def _compute_kl(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence D_KL(P || Q)."""
        p_safe = np.clip(p, 1e-12, 1.0)
        q_safe = np.clip(q, 1e-12, 1.0)
        return float(np.sum(p_safe * np.log(p_safe / q_safe)))
    
    def get_creature_history(self, creature_id: str) -> List[BrainSnapshot]:
        """Get full history for a creature."""
        return self.history.get(creature_id, [])
    
    def get_emotional_trajectory(self, creature_id: str) -> Optional[dict]:
        """Analyze emotional trajectory over time."""
        history = self.get_creature_history(creature_id)
        if len(history) < 10:
            return None
        
        cognitive = [s.cognitive_complexity for s in history]
        somatic = [s.somatic_complexity for s in history]
        
        return {
            'cognitive_mean': np.mean(cognitive),
            'cognitive_std': np.std(cognitive),
            'cognitive_trend': np.polyfit(range(len(cognitive)), cognitive, 1)[0],
            'somatic_mean': np.mean(somatic),
            'somatic_std': np.std(somatic),
            'somatic_trend': np.polyfit(range(len(somatic)), somatic, 1)[0],
            'correlation': float(np.corrcoef(cognitive, somatic)[0, 1]) if len(cognitive) > 1 else 0.0
        }
    
    def log_current_states(self, snapshots: Dict[str, BrainSnapshot], creatures: List):
        """Log a summary of current brain states."""
        if not snapshots:
            return
        
        print(f"\n[BRAIN STATES] Step {list(snapshots.values())[0].step if snapshots else '?'}")
        
        for cid, snap in snapshots.items():
            name = cid[:8]
            for c in creatures:
                if c.creature_id == cid and hasattr(c, 'mating_state'):
                    name = c.mating_state.name
                    break
            
            print(f"  {name}: cognitive={snap.cognitive_complexity:.2f} "
                  f"somatic={snap.somatic_complexity:.2f} "
                  f"circ={snap.torus_circulation:+.2f}")


# =============================================================================
# FAVORITE HERBIE TRACKER
# =============================================================================

class FavoriteHerbieTracker:
    """Track favorited Herbies and their family trees."""
    
    def __init__(self):
        self.favorites = set()
        self.family_trees = {}
        
    def toggle_favorite(self, creature_id: str, herbie) -> bool:
        """Toggle favorite status. Returns new status."""
        if creature_id in self.favorites:
            self.favorites.discard(creature_id)
            return False
        else:
            self.favorites.add(creature_id)
            self._build_family_tree(creature_id, herbie)
            return True
    
    def is_favorite(self, creature_id: str) -> bool:
        return creature_id in self.favorites
    
    def _build_family_tree(self, creature_id: str, herbie):
        """Build family tree info for a Herbie."""
        if not hasattr(herbie, 'mating_state'):
            return
        
        from ..creature.herbie_social import HerbieSex
        
        ms = herbie.mating_state
        self.family_trees[creature_id] = {
            'name': ms.name,
            'sex': ms.sex.name,
            'parents': ms.parent_ids,
            'offspring': list(ms.offspring_ids),
            'mate': ms.mate_id if ms.sex == HerbieSex.CARRIER else list(ms.mate_ids),
            'generation': herbie.generation,
        }
    
    def get_favorite_names(self, creatures: List) -> List[str]:
        """Get names of all favorited Herbies."""
        names = []
        for c in creatures:
            if c.creature_id in self.favorites and hasattr(c, 'mating_state'):
                names.append(c.mating_state.name or c.creature_id[:8])
        return names


# =============================================================================
# ANT-CREATURE INTERACTION
# =============================================================================

class AntCreatureInteraction:
    """
    Manages interactions between ant colonies and creatures.
    
    Ants can:
    - Be a food source (creatures eat ants)
    - Swarm and damage creatures that disturb the nest
    - Create trails that creatures can follow/avoid
    """
    
    SWARM_DAMAGE = 0.002
    NEST_DEFENSE_RADIUS = 10.0
    ANT_FOOD_VALUE = 2.0
    
    @classmethod
    def process_interactions(cls, colony, creatures: List, world, step: int):
        """Process all ant-creature interactions."""
        for creature in creatures:
            if not creature.alive:
                continue
            
            dist_to_nest = np.linalg.norm(creature.pos - colony.nest_pos)
            
            if dist_to_nest < cls.NEST_DEFENSE_RADIUS:
                cls._swarm_attack(colony, creature, dist_to_nest)
            
            if creature.species.diet in ['carnivore', 'omnivore']:
                cls._try_eat_ants(colony, creature, world)
    
    @classmethod
    def _swarm_attack(cls, colony, creature, dist_to_nest: float):
        """Ants swarm and attack creature near nest."""
        attack_intensity = 1.0 - (dist_to_nest / cls.NEST_DEFENSE_RADIUS)
        
        nearby_ants = colony.get_ants_near(creature.pos, 5.0) if hasattr(colony, 'get_ants_near') else []
        n_attacking = len(nearby_ants)
        
        if n_attacking > 0:
            damage = cls.SWARM_DAMAGE * n_attacking * attack_intensity
            creature.metabolism.hunger = min(1.0, creature.metabolism.hunger + damage)
            
            if hasattr(creature, 'afferent') and 'env_pain' in creature.afferent:
                creature.afferent['env_pain'].nucleate(damage * 10, 0.0)
            
            if hasattr(colony, 'disturb'):
                colony.disturb(creature.pos, 3.0)
            
            if n_attacking > 5 and np.random.random() < 0.1:
                print(f"[Ants] Swarming {creature.species.name}! ({n_attacking} ants)")
    
    @classmethod
    def _try_eat_ants(cls, colony, creature, world):
        """Creature tries to eat ants."""
        if creature.metabolism.hunger < 0.3:
            return
        
        nearby_ants = colony.get_ants_near(creature.pos, 2.0) if hasattr(colony, 'get_ants_near') else []
        
        if nearby_ants:
            ant = nearby_ants[0]
            if ant in colony.ants:
                colony.ants.remove(ant)
            
            creature.metabolism.hunger = max(0, creature.metabolism.hunger - 0.05)
            creature.total_reward += cls.ANT_FOOD_VALUE
            
            if hasattr(colony, 'disturb'):
                colony.disturb(creature.pos, 4.0)
