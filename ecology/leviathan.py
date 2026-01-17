"""
Leviathan System - Mythic apex creature.

The Leviathan is a rare, massive creature that:
- Spawns when predators are too numerous OR on mythic timer
- Traverses the map slowly, hunting predators
- Leaves fertile burn patches that boost food growth
- Ignores/protects herbivores
"""

from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

if TYPE_CHECKING:
    from ..world.objects import NutrientPatch


# Leviathan constants
LEVIATHAN_MYTHIC_INTERVAL = 8000
LEVIATHAN_PREDATOR_THRESHOLD = 5
LEVIATHAN_PREDATOR_CHECK_INTERVAL = 500
LEVIATHAN_MIN_INTERVAL = 3000

LEVIATHAN_SIZE = 12.0
LEVIATHAN_SPEED = 0.15
LEVIATHAN_DURATION = 350

LEVIATHAN_HUNT_RANGE = 25.0
LEVIATHAN_KILL_RANGE = 8.0
LEVIATHAN_FERTILIZE_INTERVAL = 50
LEVIATHAN_FERTILIZE_RADIUS = 12.0

LEVIATHAN_COLOR_BASE = '#8B0000'
LEVIATHAN_COLOR_GLOW = '#FF4500'


@dataclass
class FertilePatch:
    """A patch of fertilized ground that boosts food growth and spawns nutrients."""
    pos: np.ndarray
    radius: float
    intensity: float = 1.0
    duration: int = 2000
    age: int = 0
    nutrients_spawned: int = 0
    
    def update(self) -> bool:
        """Update patch. Returns False when expired."""
        self.age += 1
        self.intensity = max(0.3, 1.0 - (self.age / self.duration) * 0.7)
        return self.age < self.duration
    
    def get_growth_bonus(self, pos: np.ndarray) -> float:
        """Get food growth bonus at position."""
        dist = np.linalg.norm(pos - self.pos)
        if dist > self.radius:
            return 0.0
        return self.intensity * (1.0 - dist / self.radius) * 2.0
    
    def maybe_spawn_nutrient(self, world) -> bool:
        """Actively spawn nutrients within the patch area."""
        from ..world.objects import NutrientPatch
        
        spawn_prob = 0.02 * self.intensity * (1.0 - self.age / self.duration)
        
        if np.random.random() < spawn_prob:
            angle = np.random.uniform(0, 2*np.pi)
            dist = np.random.uniform(0, self.radius)
            nutrient_pos = self.pos + np.array([np.cos(angle), np.sin(angle)]) * dist
            nutrients_amount = np.random.uniform(25, 45) * self.intensity
            world.nutrients.append(NutrientPatch(nutrient_pos, nutrients_amount))
            self.nutrients_spawned += 1
            return True
        return False


@dataclass 
class Leviathan:
    """The mythic Leviathan - hunts predators, fertilizes land."""
    pos: np.ndarray
    target_pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    size: float = LEVIATHAN_SIZE
    speed: float = LEVIATHAN_SPEED
    
    age: int = 0
    max_age: int = LEVIATHAN_DURATION
    
    trail: List[np.ndarray] = field(default_factory=list)
    
    predators_killed: int = 0
    patches_created: int = 0
    
    is_hunting: bool = False
    hunt_target_id: Optional[str] = None
    last_roar_step: int = 0
    
    is_genesis: bool = False  # Special creation event version
    
    def __post_init__(self):
        direction = self.target_pos - self.pos
        dist = np.linalg.norm(direction)
        if dist > 0:
            self.vel = (direction / dist) * self.speed
    
    @property
    def alive(self) -> bool:
        return self.age < self.max_age
    
    def update(self, predators: List, world_size: float) -> dict:
        """Update leviathan. Returns event dict."""
        events = {'kills': [], 'fertilize': None, 'roar': False, 'exit': False}
        
        self.age += 1
        self.trail.append(self.pos.copy())
        if len(self.trail) > 20:
            self.trail.pop(0)
        
        if self.age >= self.max_age or self._is_at_edge(world_size):
            events['exit'] = True
            return events
        
        events['kills'] = self._hunt_predators(predators)
        
        if self.age - self.last_roar_step > 100:
            events['roar'] = True
            self.last_roar_step = self.age
        
        fertilize_interval = 15 if self.is_genesis else LEVIATHAN_FERTILIZE_INTERVAL
        
        if self.age % fertilize_interval == 0:
            if self.is_genesis:
                events['fertilize'] = FertilePatch(
                    pos=self.pos.copy(),
                    radius=LEVIATHAN_FERTILIZE_RADIUS * 1.2,
                    intensity=2.0,
                    duration=2500
                )
            else:
                events['fertilize'] = FertilePatch(
                    pos=self.pos.copy(),
                    radius=LEVIATHAN_FERTILIZE_RADIUS,
                    intensity=1.0,
                    duration=2000
                )
            self.patches_created += 1
        
        self._move(predators, world_size)
        return events
    
    def _is_at_edge(self, world_size: float) -> bool:
        margin = world_size * 0.48
        return (abs(self.pos[0]) > margin or abs(self.pos[1]) > margin)
    
    def _hunt_predators(self, predators: List) -> List[str]:
        kills = []
        for pred in predators:
            if not pred.alive:
                continue
            dist = np.linalg.norm(pred.pos - self.pos)
            
            if dist < LEVIATHAN_KILL_RANGE:
                pred.alive = False
                pred.metabolism.is_dead = True
                pred.metabolism.cause_of_death = "leviathan"
                kills.append(pred.creature_id)
                self.predators_killed += 1
                self.is_hunting = False
                self.hunt_target_id = None
            elif dist < LEVIATHAN_HUNT_RANGE and not self.is_hunting:
                self.is_hunting = True
                self.hunt_target_id = pred.creature_id
        return kills
    
    def _move(self, predators: List, world_size: float):
        if self.is_hunting and self.hunt_target_id:
            target = None
            for pred in predators:
                if pred.creature_id == self.hunt_target_id and pred.alive:
                    target = pred
                    break
            if target:
                direction = target.pos - self.pos
                dist = np.linalg.norm(direction)
                if dist > 0.1:
                    self.vel = (direction / dist) * self.speed * 1.3
            else:
                self.is_hunting = False
                self.hunt_target_id = None
        
        if not self.is_hunting:
            direction = self.target_pos - self.pos
            dist = np.linalg.norm(direction)
            if dist > 1:
                self.vel = (direction / dist) * self.speed
        
        self.pos = self.pos + self.vel
        self.pos += np.random.randn(2) * 0.05


class LeviathanSystem:
    """Manages Leviathan spawning and world effects."""
    
    def __init__(self, world_size: float):
        """Initialize leviathan system."""
        self.world_size = world_size
        self.leviathan: Optional[Leviathan] = None
        self.fertile_patches: List[FertilePatch] = []
        
        self.step_count = 0
        self.last_spawn_step = -LEVIATHAN_MIN_INTERVAL
        self.total_spawns = 0
        self.total_kills = 0
        
        self.genesis_triggered = False
        self.genesis_step = 50
        
        self._next_mythic_spawn = np.random.randint(
            LEVIATHAN_MYTHIC_INTERVAL // 2,
            LEVIATHAN_MYTHIC_INTERVAL * 2
        )
    
    def is_active(self) -> bool:
        """Check if a leviathan is currently active."""
        return self.leviathan is not None and self.leviathan.alive
    
    def add_sacred_ground(self, pos: np.ndarray, radius: float = 15.0):
        """Add a sacred/fertile patch at position."""
        patch = FertilePatch(
            pos=pos.copy(),
            radius=radius,
            intensity=1.0,
            duration=2000
        )
        self.fertile_patches.append(patch)
    
    def spawn_genesis_leviathan(self, world_size: float = None):
        """
        Spawn the Genesis Leviathan without instant-killing predators.
        
        The Leviathan will hunt predators as it crosses the world,
        so survival depends on position and luck.
        """
        self.genesis_triggered = True
        
        ws = world_size or self.world_size
        
        # Spawn from random edge, cross to opposite
        edge = np.random.randint(4)
        half = ws / 2 - 5
        
        if edge == 0:
            start = np.array([np.random.uniform(-half, half), half])
            end = np.array([np.random.uniform(-half, half), -half])
        elif edge == 1:
            start = np.array([half, np.random.uniform(-half, half)])
            end = np.array([-half, np.random.uniform(-half, half)])
        elif edge == 2:
            start = np.array([np.random.uniform(-half, half), -half])
            end = np.array([np.random.uniform(-half, half), half])
        else:
            start = np.array([-half, np.random.uniform(-half, half)])
            end = np.array([half, np.random.uniform(-half, half)])
        
        self.leviathan = Leviathan(pos=start, target_pos=end)
        self.leviathan.speed = LEVIATHAN_SPEED * 2.0  # Fast but not instant
        self.leviathan.max_age = 250  # Longer journey
        self.leviathan.size = LEVIATHAN_SIZE * 1.3
        self.leviathan.is_genesis = True
        
        direction = end - start
        dist = np.linalg.norm(direction)
        if dist > 0:
            self.leviathan.vel = (direction / dist) * self.leviathan.speed
        
        self.last_spawn_step = self.step_count
        self.total_spawns += 1
    
    def trigger_genesis(self, creatures: List):
        """The Genesis Leviathan - a creation myth event (legacy, kills all)."""
        self.genesis_triggered = True
        
        # Kill all predators (done silently, manager will print)
        predators = [c for c in creatures 
                    if c.alive and c.species.diet in ['carnivore', 'omnivore']]
        for pred in predators:
            pred.alive = False
        
        # Spawn Genesis Leviathan
        edge = np.random.randint(4)
        half = self.world_size / 2 - 5
        
        if edge == 0:
            start = np.array([np.random.uniform(-half, half), half])
            end = np.array([np.random.uniform(-half, half), -half])
        elif edge == 1:
            start = np.array([half, np.random.uniform(-half, half)])
            end = np.array([-half, np.random.uniform(-half, half)])
        elif edge == 2:
            start = np.array([np.random.uniform(-half, half), -half])
            end = np.array([np.random.uniform(-half, half), half])
        else:
            start = np.array([-half, np.random.uniform(-half, half)])
            end = np.array([half, np.random.uniform(-half, half)])
        
        self.leviathan = Leviathan(pos=start, target_pos=end)
        self.leviathan.speed = LEVIATHAN_SPEED * 2.5
        self.leviathan.max_age = 150
        self.leviathan.size = LEVIATHAN_SIZE * 1.3
        self.leviathan.is_genesis = True
        
        direction = end - start
        dist = np.linalg.norm(direction)
        if dist > 0:
            self.leviathan.vel = (direction / dist) * self.leviathan.speed
        
        self.last_spawn_step = self.step_count
        self.total_spawns += 1
    
    def update(self, creatures: List, audio_system=None, world=None) -> dict:
        """Update leviathan system."""
        self.step_count += 1
        events = {
            'spawn': False, 'exit': False, 'kills': [],
            'roar': False, 'fertilize': None,
            'active': self.leviathan is not None,
            'genesis': False,
            'nutrients_spawned': 0
        }
        
        # Check for Genesis
        if not self.genesis_triggered and self.step_count == self.genesis_step:
            self.trigger_genesis(creatures)
            events['genesis'] = True
        
        # Update patches
        self.fertile_patches = [p for p in self.fertile_patches if p.update()]
        
        if world is not None:
            for patch in self.fertile_patches:
                if patch.maybe_spawn_nutrient(world):
                    events['nutrients_spawned'] += 1
        
        # Check spawn conditions
        if self.leviathan is None:
            should_spawn, reason = self._check_spawn_conditions(creatures)
            if should_spawn:
                self._spawn_leviathan(reason)
                events['spawn'] = True
        
        # Update active leviathan
        if self.leviathan:
            predators = [c for c in creatures
                        if c.alive and c.species.diet in ['carnivore', 'omnivore']]
            
            lev_events = self.leviathan.update(predators, self.world_size)
            
            events['kills'] = lev_events['kills']
            events['roar'] = lev_events['roar']
            events['fertilize'] = lev_events['fertilize']
            
            self.total_kills += len(lev_events['kills'])
            
            if lev_events['fertilize']:
                self.fertile_patches.append(lev_events['fertilize'])
            
            if lev_events['exit']:
                events['exit'] = True
                if self.leviathan.is_genesis:
                    print(f"\n[GENESIS] The First Leviathan vanishes beyond the edge of the world.")
                    print(f"[GENESIS] {self.leviathan.patches_created} scorched groves smolder in its wake.")
                    print(f"[GENESIS] From ash, new life will grow.\n")
                else:
                    print(f"[LEVIATHAN] 🐉 The shadow passes. Killed {self.leviathan.predators_killed}, "
                          f"fertilized {self.leviathan.patches_created} groves.")
                self.leviathan = None
        
        return events
    
    def _check_spawn_conditions(self, creatures: List) -> Tuple[bool, str]:
        if self.step_count - self.last_spawn_step < LEVIATHAN_MIN_INTERVAL:
            return False, ""
        
        predator_count = sum(1 for c in creatures
                            if c.alive and c.species.diet in ['carnivore', 'omnivore'])
        
        if predator_count >= LEVIATHAN_PREDATOR_THRESHOLD:
            if self.step_count % LEVIATHAN_PREDATOR_CHECK_INTERVAL == 0:
                return True, "predator_imbalance"
        
        if self.step_count >= self._next_mythic_spawn:
            self._next_mythic_spawn = self.step_count + np.random.randint(
                LEVIATHAN_MYTHIC_INTERVAL // 2,
                LEVIATHAN_MYTHIC_INTERVAL * 2
            )
            return True, "mythic_cycle"
        
        return False, ""
    
    def _spawn_leviathan(self, reason: str):
        edge = np.random.randint(4)
        half = self.world_size / 2 - 5
        
        if edge == 0:
            start = np.array([np.random.uniform(-half, half), half])
            end = np.array([np.random.uniform(-half, half), -half])
        elif edge == 1:
            start = np.array([half, np.random.uniform(-half, half)])
            end = np.array([-half, np.random.uniform(-half, half)])
        elif edge == 2:
            start = np.array([np.random.uniform(-half, half), -half])
            end = np.array([np.random.uniform(-half, half), half])
        else:
            start = np.array([-half, np.random.uniform(-half, half)])
            end = np.array([half, np.random.uniform(-half, half)])
        
        self.leviathan = Leviathan(pos=start, target_pos=end)
        self.last_spawn_step = self.step_count
        self.total_spawns += 1
        
        print(f"[LEVIATHAN] 🐉 A great shadow falls across the land!")
    
    def get_food_growth_bonus(self, pos: np.ndarray) -> float:
        """Get food growth bonus from fertile patches at position."""
        bonus = 0.0
        for patch in self.fertile_patches:
            bonus += patch.get_growth_bonus(pos)
        return bonus
    
    def get_visual_data(self) -> dict:
        """Get data for visualization."""
        data = {
            'active': self.leviathan is not None,
            'patches': [(p.pos.copy(), p.radius, p.intensity) for p in self.fertile_patches]
        }
        if self.leviathan:
            data['pos'] = self.leviathan.pos.copy()
            data['size'] = self.leviathan.size
            data['trail'] = [t.copy() for t in self.leviathan.trail]
            data['is_hunting'] = self.leviathan.is_hunting
            data['is_genesis'] = self.leviathan.is_genesis
        return data
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'step_count': self.step_count,
            'last_spawn_step': self.last_spawn_step,
            'total_spawns': self.total_spawns,
            'total_kills': self.total_kills,
            'genesis_triggered': self.genesis_triggered,
            '_next_mythic_spawn': self._next_mythic_spawn,
            'fertile_patches': [
                {'pos': p.pos.tolist(), 'radius': p.radius, 
                 'intensity': p.intensity, 'duration': p.duration, 'age': p.age}
                for p in self.fertile_patches
            ]
        }
    
    @classmethod
    def from_dict(cls, d: dict, world_size: float) -> 'LeviathanSystem':
        """Deserialize from persistence."""
        system = cls(world_size)
        system.step_count = d.get('step_count', 0)
        system.last_spawn_step = d.get('last_spawn_step', -LEVIATHAN_MIN_INTERVAL)
        system.total_spawns = d.get('total_spawns', 0)
        system.total_kills = d.get('total_kills', 0)
        system.genesis_triggered = d.get('genesis_triggered', False)
        system._next_mythic_spawn = d.get('_next_mythic_spawn', LEVIATHAN_MYTHIC_INTERVAL)
        
        for p_data in d.get('fertile_patches', []):
            patch = FertilePatch(
                pos=np.array(p_data['pos']),
                radius=p_data['radius'],
                intensity=p_data['intensity'],
                duration=p_data['duration'],
                age=p_data['age']
            )
            system.fertile_patches.append(patch)
        
        return system
