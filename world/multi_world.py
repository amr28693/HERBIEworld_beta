"""
MultiWorld - The game world container.

Manages:
- World objects (food, barriers)
- Nutrients (decomposing matter that spawns new food)
- Creature-object interactions
- Terrain-aware spawning and ecosystem updates
"""

from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from ..core.constants import WORLD_L, BODY_L, Nx, Ny, X, Y
from .objects import WorldObject, NutrientPatch
from .terrain import Terrain, TERRAIN_RESOLUTION

if TYPE_CHECKING:
    from ..creature.creature import Creature
    from .seasons import Season


class MultiWorld:
    """
    World supporting multiple creatures.
    
    Contains:
    - Food and barrier objects
    - Nutrient patches (become food over time)
    - Terrain (optional)
    """
    
    def __init__(self, terrain: Terrain = None, world_size: float = None, seed: int = None):
        """
        Initialize world.
        
        Args:
            terrain: Pre-existing terrain (creates new if None)
            world_size: World size (uses WORLD_L if None)
            seed: Random seed for terrain generation
        """
        self.world_size = world_size or WORLD_L
        
        # Terrain
        if terrain is not None:
            self.terrain = terrain
        else:
            self.terrain = Terrain(self.world_size, TERRAIN_RESOLUTION, seed)
        
        # Objects
        self.objects: List[WorldObject] = []
        self.nutrients: List[NutrientPatch] = []
        
        # Statistics
        self.total_consumed = 0.0
        self.objects_eaten = 0
        self.objects_sprouted = 0
        
        # Fertile patch bonus function (set by manager for Leviathan sacred ground)
        self._fertile_bonus_fn = None
        
        # Spawn initial objects
        self._spawn_objects_terrain_aware()
    
    def _spawn_objects_terrain_aware(self):
        """Spawn food objects based on terrain - more in plains/forest."""
        from ..creature.herbie_hands import add_grip_properties_to_objects
        
        n_food = 100  # Starting food count
        n_barriers = 10
        
        # === PRIMORDIAL GROVES ===
        n_groves = np.random.randint(2, 5)
        grove_types = ['fruit', 'root', 'berry', 'fungal']
        
        for _ in range(n_groves):
            grove_x = np.random.uniform(-self.world_size/2 + 15, self.world_size/2 - 15)
            grove_y = np.random.uniform(-self.world_size/2 + 15, self.world_size/2 - 15)
            grove_center = np.array([grove_x, grove_y])
            grove_type = np.random.choice(grove_types)
            
            self._spawn_grove(grove_center, grove_type)
        
        print(f"[World] Spawned {n_groves} primordial groves")
        
        # === SCATTERED FOOD ===
        attempts = 0
        while len([o for o in self.objects if o.compliance > 0.5]) < n_food and attempts < 400:
            attempts += 1
            
            x = np.random.uniform(-self.world_size/2 + 5, self.world_size/2 - 5)
            y = np.random.uniform(-self.world_size/2 + 5, self.world_size/2 - 5)
            pos = np.array([x, y])
            
            terrain_type = self.terrain.get_terrain_at(pos)
            
            if np.random.random() < terrain_type.food_growth_rate:
                obj = self._create_terrain_food(pos, terrain_type)
                if obj:
                    add_grip_properties_to_objects([obj])
                    self.objects.append(obj)
        
        # === BARRIERS ===
        for _ in range(n_barriers):
            for attempt in range(20):
                x = np.random.uniform(-self.world_size/2 + 5, self.world_size/2 - 5)
                y = np.random.uniform(-self.world_size/2 + 5, self.world_size/2 - 5)
                pos = np.array([x, y])
                
                terrain_type = self.terrain.get_terrain_at(pos)
                if terrain_type.name in ['hills', 'mountain', 'forest']:
                    size = np.random.uniform(1.5, 2.5)
                    obj = WorldObject((x, y), size, np.random.uniform(0.1, 0.25),
                                      np.random.uniform(2, 4), 'gray')
                    add_grip_properties_to_objects([obj])
                    self.objects.append(obj)
                    break
        
        print(f"[World] Spawned {len(self.objects)} objects")
    
    def _spawn_grove(self, center: np.ndarray, grove_type: str):
        """Spawn a grove of food at the given location."""
        from ..creature.herbie_hands import add_grip_properties_to_objects
        
        if grove_type == 'fruit':
            n_in_grove = np.random.randint(8, 13)
            spread = 6
            for _ in range(n_in_grove):
                pos = center + np.random.randn(2) * spread
                pos = np.clip(pos, -self.world_size/2 + 3, self.world_size/2 - 3)
                size = np.random.uniform(1.0, 1.6)
                energy = np.random.uniform(25, 40)
                color = np.random.choice(['orangered', 'gold', 'crimson', 'orange'])
                obj = WorldObject(pos, size, np.random.uniform(0.75, 0.9), energy, color)
                add_grip_properties_to_objects([obj])
                self.objects.append(obj)
                
        elif grove_type == 'root':
            n_in_grove = np.random.randint(6, 11)
            spread = 4
            for _ in range(n_in_grove):
                pos = center + np.random.randn(2) * spread
                pos = np.clip(pos, -self.world_size/2 + 3, self.world_size/2 - 3)
                size = np.random.uniform(0.8, 1.3)
                energy = np.random.uniform(15, 28)
                color = np.random.choice(['sienna', 'peru', 'burlywood', 'tan'])
                obj = WorldObject(pos, size, np.random.uniform(0.7, 0.85), energy, color)
                add_grip_properties_to_objects([obj])
                self.objects.append(obj)
                
        elif grove_type == 'berry':
            n_in_grove = np.random.randint(12, 18)
            spread = 8
            for _ in range(n_in_grove):
                pos = center + np.random.randn(2) * spread
                pos = np.clip(pos, -self.world_size/2 + 3, self.world_size/2 - 3)
                size = np.random.uniform(0.5, 0.9)
                energy = np.random.uniform(8, 18)
                color = np.random.choice(['purple', 'magenta', 'deeppink', 'mediumvioletred'])
                obj = WorldObject(pos, size, np.random.uniform(0.8, 0.95), energy, color)
                add_grip_properties_to_objects([obj])
                self.objects.append(obj)
                
        else:  # fungal
            n_in_grove = np.random.randint(5, 9)
            spread = 3
            for _ in range(n_in_grove):
                pos = center + np.random.randn(2) * spread
                pos = np.clip(pos, -self.world_size/2 + 3, self.world_size/2 - 3)
                size = np.random.uniform(0.6, 1.1)
                energy = np.random.uniform(12, 35)
                color = np.random.choice(['wheat', 'khaki', 'darkkhaki', 'palegoldenrod'])
                obj = WorldObject(pos, size, np.random.uniform(0.72, 0.88), energy, color)
                add_grip_properties_to_objects([obj])
                self.objects.append(obj)
    
    def _create_terrain_food(self, pos: np.ndarray, terrain_type) -> Optional[WorldObject]:
        """Create food appropriate for terrain type."""
        if terrain_type.name == "forest":
            size = np.random.uniform(1.2, 1.8)
            energy = np.random.uniform(15, 30)
            color = 'darkgreen'
        elif terrain_type.name == "plains":
            size = np.random.uniform(0.8, 1.4)
            energy = np.random.uniform(10, 25)
            color = np.random.choice(['lightgreen', 'yellowgreen', 'lime'])
        elif terrain_type.name == "shore":
            size = np.random.uniform(0.6, 1.0)
            energy = np.random.uniform(8, 15)
            color = 'palegreen'
        elif terrain_type.name == "hills":
            size = np.random.uniform(0.5, 0.9)
            energy = np.random.uniform(5, 12)
            color = 'olive'
        else:
            return None
        
        compliance = np.random.uniform(0.7, 0.9)
        return WorldObject(pos, size, compliance, energy, color)
    
    # =========================================================================
    # CREATURE INTERACTIONS
    # =========================================================================
    
    def get_object_potential_for_creature(self, creature: 'Creature') -> np.ndarray:
        """Get potential field from objects for creature's body field evolution."""
        V = np.zeros((Ny, Nx))
        
        for i, obj in enumerate(self.objects):
            if obj.alive:
                rel = obj.pos - creature.pos
                dist = np.sqrt((X - rel[0])**2 + (Y - rel[1])**2)
                
                learned = creature.learned_potential.get(i, 0.0)
                
                if obj.compliance > 0.5:
                    # Attractive - food
                    strength = obj.compliance * 0.5 * (1 + obj.energy / (obj.max_energy + 1e-6))
                    V -= strength * 2.5 * np.exp(-dist**2 / obj.size**2) * (1 + learned)
                else:
                    # Repulsive - barrier
                    barrier_dist = dist - obj.size
                    V += (1 - obj.compliance) * 5.0 * np.exp(-np.maximum(0, barrier_dist)**2 / 0.6)
        
        return V
    
    def get_boundary_potential(self, creature_pos: np.ndarray) -> np.ndarray:
        """Get potential from world boundaries."""
        wx = creature_pos[0] + X
        wy = creature_pos[1] + Y
        edge_dist = np.minimum(
            np.minimum(wx + self.world_size/2, self.world_size/2 - wx),
            np.minimum(wy + self.world_size/2, self.world_size/2 - wy)
        )
        return 5.0 * np.exp(-np.maximum(0, edge_dist) / 3.0)
    
    def process_creature_interactions(self, creature: 'Creature', body_I: np.ndarray,
                                       body_momentum: np.ndarray) -> Tuple[float, float]:
        """
        Process creature's interactions with world objects.
        
        Returns (total_reward, total_extracted).
        """
        total_reward = 0.0
        total_extracted = 0.0
        
        for i, obj in enumerate(self.objects):
            if not obj.alive:
                continue
            
            rel = obj.pos - creature.pos
            if np.linalg.norm(rel) > BODY_L:
                continue
            
            obj.compute_contact(creature.pos, body_I)
            reward, extracted = obj.compute_reward()
            
            extracted *= creature.species.energy_efficiency
            reward *= creature.species.energy_efficiency
            
            total_reward += reward
            total_extracted += extracted
            
            if extracted > 0:
                self.total_consumed += extracted
                current = creature.learned_potential.get(i, 0.0)
                lr = 0.025 / (1 + abs(current))
                creature.learned_potential[i] = np.clip(current + reward * lr, -0.9, 0.9)
            
            if not obj.alive and obj.compliance > 0.5:
                self.objects_eaten += 1
            
            obj.apply_push(creature.pos, creature.vel, body_momentum, obj.contact)
            obj.update()
        
        return total_reward, total_extracted
    
    def get_reward_source_for_creature(self, creature: 'Creature') -> Optional[int]:
        """Get index of object currently providing reward to creature."""
        for i, obj in enumerate(self.objects):
            if obj.alive and obj.contact > 0.1 and obj.reward > 0.01:
                return i
        return None
    
    def get_nearest_food(self, pos: np.ndarray) -> Optional[WorldObject]:
        """Find nearest food object to position."""
        nearest = None
        nearest_dist = float('inf')
        
        for obj in self.objects:
            if obj.alive and obj.compliance > 0.6:
                dist = np.linalg.norm(obj.pos - pos)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = obj
        
        return nearest
    
    def drop_nutrient(self, pos: np.ndarray, amount: float):
        """Drop a nutrient patch at position (from defecation)."""
        drop_pos = pos + np.random.randn(2) * 2.0
        drop_pos = np.clip(drop_pos, -self.world_size/2 + 4, self.world_size/2 - 4)
        self.nutrients.append(NutrientPatch(drop_pos, amount))
        
        # Notify visualization callback if registered
        if hasattr(self, 'on_poop_callback') and self.on_poop_callback:
            self.on_poop_callback(drop_pos, amount)
    
    # =========================================================================
    # ECOSYSTEM UPDATE
    # =========================================================================
    
    def update_ecosystem(self, season: 'Season' = None, fertile_bonus_fn=None):
        """
        Update ecosystem - nutrients become food, random spawning.
        
        Args:
            season: Current season (affects spawn rates)
            fertile_bonus_fn: Function(pos) -> bonus for fertile patches
        """
        from ..creature.herbie_hands import add_grip_properties_to_objects
        from .seasons import SEASONS
        
        if season is None:
            season = SEASONS['summer']
        
        if fertile_bonus_fn is None and self._fertile_bonus_fn is not None:
            fertile_bonus_fn = self._fertile_bonus_fn
        
        # Process nutrients
        for nutrient in self.nutrients[:]:
            if nutrient.update():
                terrain_type = self.terrain.get_terrain_at(nutrient.pos)
                spawn_prob = season.food_spawn_rate * terrain_type.food_growth_rate
                
                # Fertile patch bonus
                if fertile_bonus_fn:
                    bonus = fertile_bonus_fn(nutrient.pos)
                    if bonus > 0.1:
                        spawn_prob *= (1.0 + bonus)
                
                if np.random.random() < spawn_prob:
                    self._spawn_from_nutrient(nutrient, fertile_bonus_fn)
                
                self.nutrients.remove(nutrient)
        
        # Random spawning
        if np.random.random() < 0.003 * season.food_spawn_rate:
            self._spawn_random_food(season, fertile_bonus_fn)
        
        # Seasonal decay
        if season.food_decay_rate > 1.0:
            for obj in self.objects:
                if obj.alive and obj.compliance > 0.5:
                    obj.energy -= 0.01 * (season.food_decay_rate - 1.0)
                    if obj.energy < 1:
                        obj.alive = False
        
        # Clean dead objects
        self.objects = [obj for obj in self.objects if obj.alive or obj.compliance < 0.5]
    
    def _spawn_from_nutrient(self, nutrient: NutrientPatch, fertile_bonus_fn=None):
        """Spawn food from a nutrient patch."""
        from ..creature.herbie_hands import add_grip_properties_to_objects
        
        terrain_type = self.terrain.get_terrain_at(nutrient.pos)
        
        if terrain_type.name in ['water', 'mountain']:
            return
        
        size = 0.6 + 0.4 * min(nutrient.nutrients / 20, 1.0)
        size *= terrain_type.food_growth_rate
        energy = nutrient.nutrients * 1.5 * terrain_type.food_growth_rate
        
        if fertile_bonus_fn:
            bonus = fertile_bonus_fn(nutrient.pos)
            if bonus > 0:
                size *= (1.0 + bonus * 0.5)
                energy *= (1.0 + bonus)
        
        compliance = 0.7 + 0.15 * np.random.random()
        
        colors = {'plains': 'lightgreen', 'forest': 'darkgreen',
                  'shore': 'palegreen', 'hills': 'olive'}
        color = colors.get(terrain_type.name, 'lime')
        
        obj = WorldObject(nutrient.pos.copy(), size, compliance, energy, color)
        add_grip_properties_to_objects([obj])
        self.objects.append(obj)
        self.objects_sprouted += 1
    
    def _spawn_random_food(self, season: 'Season', fertile_bonus_fn=None):
        """Spawn random food somewhere in the world."""
        from ..creature.herbie_hands import add_grip_properties_to_objects
        
        for _ in range(10):
            x = np.random.uniform(-self.world_size/2 + 5, self.world_size/2 - 5)
            y = np.random.uniform(-self.world_size/2 + 5, self.world_size/2 - 5)
            pos = np.array([x, y])
            
            terrain_type = self.terrain.get_terrain_at(pos)
            if terrain_type.name not in ['water', 'mountain']:
                spawn_prob = terrain_type.food_growth_rate
                
                if fertile_bonus_fn:
                    spawn_prob *= (1.0 + fertile_bonus_fn(pos))
                
                if np.random.random() < spawn_prob:
                    size = np.random.uniform(0.8, 1.3) * terrain_type.food_growth_rate
                    energy = np.random.uniform(15, 30) * terrain_type.food_growth_rate
                    
                    if fertile_bonus_fn:
                        bonus = fertile_bonus_fn(pos)
                        if bonus > 0:
                            size *= (1.0 + bonus * 0.3)
                            energy *= (1.0 + bonus * 0.5)
                    
                    obj = WorldObject(pos, size, 0.75, energy, 'lime')
                    add_grip_properties_to_objects([obj])
                    self.objects.append(obj)
                    self.objects_sprouted += 1
                    return
    
    def count_food(self) -> int:
        """Count living food objects."""
        return sum(1 for o in self.objects if o.alive and o.compliance > 0.5)
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def to_dict(self) -> dict:
        """Serialize world state."""
        return {
            'world_size': self.world_size,
            'objects': [obj.to_dict() for obj in self.objects],
            'nutrients': [{'pos': n.pos.tolist(), 'nutrients': n.nutrients, 'age': n.age} 
                         for n in self.nutrients],
            'total_consumed': self.total_consumed,
            'objects_eaten': self.objects_eaten,
            'objects_sprouted': self.objects_sprouted,
        }
    
    @classmethod
    def from_dict(cls, data: dict, terrain: Terrain = None) -> 'MultiWorld':
        """Deserialize world state."""
        from ..creature.herbie_hands import add_grip_properties_to_objects
        
        world = cls.__new__(cls)
        world.world_size = data.get('world_size', WORLD_L)
        world.terrain = terrain or Terrain(world.world_size, TERRAIN_RESOLUTION)
        world.objects = []
        world.nutrients = []
        world._fertile_bonus_fn = None
        
        # Restore objects
        for obj_data in data.get('objects', []):
            obj = WorldObject.from_dict(obj_data)
            add_grip_properties_to_objects([obj])
            world.objects.append(obj)
        
        # Restore nutrients
        for nut_data in data.get('nutrients', []):
            nut = NutrientPatch(np.array(nut_data['pos']), nut_data['nutrients'])
            nut.age = nut_data.get('age', 0)
            world.nutrients.append(nut)
        
        world.total_consumed = data.get('total_consumed', 0.0)
        world.objects_eaten = data.get('objects_eaten', 0)
        world.objects_sprouted = data.get('objects_sprouted', 0)
        
        return world
