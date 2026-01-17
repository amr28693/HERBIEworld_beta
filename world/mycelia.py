"""
Mycorrhizal Network - Underground fungal nutrient highways.

A grid-based fungal network that:
- Connects food sources via nutrient highways
- Redistributes energy between plants
- Signals stress/damage through the network
- Spawns fruiting bodies (mushrooms) when healthy
- Transforms/transports chemistry elements
"""

from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from .terrain import Terrain
    from .objects import WorldObject


@dataclass
class FruitingBody:
    """A visible mushroom spawned by healthy network."""
    pos: np.ndarray
    age: int = 0
    max_age: int = 300
    size: float = 0.4
    spore_ready: bool = False
    
    def update(self) -> bool:
        """Returns False when mushroom dies."""
        self.age += 1
        # Grow for first third of life
        if self.age < self.max_age // 3:
            self.size = 0.4 + 0.4 * (self.age / (self.max_age // 3))
        # Spore release in final third
        if self.age > 2 * self.max_age // 3:
            self.spore_ready = True
        return self.age < self.max_age


class MyceliumNetwork:
    """
    Underground fungal network connecting food sources.
    
    The network grows toward food sources, creating connections
    that allow nutrient redistribution and chemical signaling.
    Prefers forest and cave terrain, avoids water.
    """
    
    def __init__(self, world_size: float, resolution: int = 50):
        """
        Initialize mycelium network.
        
        Args:
            world_size: Size of world in units
            resolution: Grid resolution (coarser than terrain for efficiency)
        """
        self.world_size = world_size
        self.resolution = resolution
        self.cell_size = world_size / resolution
        
        # Network density grid (0-1)
        self.density = np.zeros((resolution, resolution))
        
        # Nutrient flow field (for visualization)
        self.flow_x = np.zeros((resolution, resolution))
        self.flow_y = np.zeros((resolution, resolution))
        
        # Signal propagation (stress/damage signals)
        self.signal = np.zeros((resolution, resolution))
        
        # Fruiting bodies (visible mushrooms)
        self.fruiting_bodies: List[FruitingBody] = []
        
        # Statistics
        self.total_nutrient_transferred = 0.0
        self.total_fruiting_bodies_spawned = 0
        self.step_count = 0
        
        # Parameters
        self.growth_rate = 0.002
        self.decay_rate = 0.001
        self.spread_sigma = 1.2
        self.nutrient_transfer_rate = 0.05
        self.signal_decay = 0.1
        self.fruiting_threshold = 0.6  # Density needed to spawn mushroom
        self.fruiting_probability = 0.001
        
    def world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        x = int((pos[0] + self.world_size/2) / self.world_size * self.resolution)
        y = int((pos[1] + self.world_size/2) / self.world_size * self.resolution)
        x = np.clip(x, 0, self.resolution - 1)
        y = np.clip(y, 0, self.resolution - 1)
        return x, y
    
    def grid_to_world(self, i: int, j: int) -> np.ndarray:
        """Convert grid indices to world coordinates."""
        x = (i / self.resolution) * self.world_size - self.world_size/2
        y = (j / self.resolution) * self.world_size - self.world_size/2
        return np.array([x, y])
    
    def get_terrain_affinity(self, terrain_name: str) -> float:
        """
        Get growth affinity for terrain type.
        Fungi love forest/cave, tolerate plains, avoid water.
        """
        affinities = {
            'forest': 1.5,
            'cave': 1.3,
            'plains': 1.0,
            'hills': 0.8,
            'shore': 0.4,
            'water': 0.05,
            'mountain': 0.1,
        }
        return affinities.get(terrain_name, 0.5)
    
    def update(self, terrain: 'Terrain', food_objects: List['WorldObject'], 
               dt_mult: int = 1) -> List[str]:
        """
        Update mycelium network.
        
        Args:
            terrain: Terrain object for affinity lookup
            food_objects: List of food WorldObjects
            dt_mult: Steps to simulate (for efficiency, can skip frames)
            
        Returns:
            List of event messages
        """
        messages = []
        self.step_count += 1
        
        # Only update every few steps for efficiency
        if self.step_count % 5 != 0:
            return messages
        
        # === GROWTH TOWARD FOOD ===
        food_attraction = np.zeros((self.resolution, self.resolution))
        for obj in food_objects:
            if not obj.alive or obj.compliance < 0.5:
                continue
            gi, gj = self.world_to_grid(obj.pos)
            # Food creates attraction gradient
            for di in range(-5, 6):
                for dj in range(-5, 6):
                    ni, nj = gi + di, gj + dj
                    if 0 <= ni < self.resolution and 0 <= nj < self.resolution:
                        dist = np.sqrt(di**2 + dj**2) + 0.1
                        food_attraction[ni, nj] += obj.energy * 0.01 / dist
        
        # === TERRAIN AFFINITY ===
        terrain_mult = np.ones((self.resolution, self.resolution))
        for i in range(self.resolution):
            for j in range(self.resolution):
                world_pos = self.grid_to_world(i, j)
                t = terrain.get_terrain_at(world_pos)
                terrain_mult[i, j] = self.get_terrain_affinity(t.name)
        
        # === GROWTH DYNAMICS ===
        # Growth where there's food attraction and good terrain
        growth = self.growth_rate * food_attraction * terrain_mult
        
        # Decay everywhere
        decay = self.decay_rate * self.density
        
        # Update density
        self.density += growth - decay
        
        # Spread via diffusion
        self.density = gaussian_filter(self.density, sigma=self.spread_sigma)
        
        # Clamp
        self.density = np.clip(self.density, 0, 1)
        
        # === SIGNAL PROPAGATION ===
        # Signals decay and spread
        self.signal = gaussian_filter(self.signal, sigma=2.0)
        self.signal *= (1 - self.signal_decay)
        
        # === NUTRIENT REDISTRIBUTION ===
        # High-density areas share with low-density connected areas
        # This is a simple diffusion that moves nutrients along the network
        # (Actual implementation would modify food object energy)
        
        # === FRUITING BODIES ===
        # Spawn mushrooms in high-density areas
        for i in range(self.resolution):
            for j in range(self.resolution):
                if (self.density[i, j] > self.fruiting_threshold and 
                    np.random.random() < self.fruiting_probability):
                    # Check terrain is suitable
                    world_pos = self.grid_to_world(i, j)
                    t = terrain.get_terrain_at(world_pos)
                    if t.name in ['forest', 'cave', 'plains']:
                        # Add some randomness to exact position
                        pos = world_pos + np.random.randn(2) * self.cell_size * 0.3
                        self.fruiting_bodies.append(FruitingBody(pos=pos))
                        self.total_fruiting_bodies_spawned += 1
        
        # Update existing fruiting bodies
        surviving = []
        for mushroom in self.fruiting_bodies:
            if mushroom.update():
                surviving.append(mushroom)
                # Spore dispersal extends network
                if mushroom.spore_ready and np.random.random() < 0.01:
                    gi, gj = self.world_to_grid(mushroom.pos)
                    for di in range(-3, 4):
                        for dj in range(-3, 4):
                            ni, nj = gi + di, gj + dj
                            if 0 <= ni < self.resolution and 0 <= nj < self.resolution:
                                self.density[ni, nj] += 0.02
        self.fruiting_bodies = surviving
        
        return messages
    
    def inject_signal(self, pos: np.ndarray, strength: float = 1.0):
        """
        Inject a stress/damage signal at position.
        Signal will propagate through connected network.
        
        Args:
            pos: World position
            strength: Signal strength
        """
        gi, gj = self.world_to_grid(pos)
        # Only propagates where network exists
        if self.density[gi, gj] > 0.1:
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni, nj = gi + di, gj + dj
                    if 0 <= ni < self.resolution and 0 <= nj < self.resolution:
                        dist = np.sqrt(di**2 + dj**2) + 0.1
                        self.signal[ni, nj] += strength * self.density[ni, nj] / dist
    
    def get_signal_at(self, pos: np.ndarray) -> float:
        """Get signal strength at world position."""
        gi, gj = self.world_to_grid(pos)
        return float(self.signal[gi, gj])
    
    def get_density_at(self, pos: np.ndarray) -> float:
        """Get network density at world position."""
        gi, gj = self.world_to_grid(pos)
        return float(self.density[gi, gj])
    
    def get_total_biomass(self) -> float:
        """Get total network biomass."""
        return float(np.sum(self.density))
    
    def get_coverage(self) -> float:
        """Get fraction of world covered by network."""
        return float(np.mean(self.density > 0.1))
    
    def redistribute_nutrients(self, food_objects: List['WorldObject'], 
                                transfer_rate: float = None):
        """
        Redistribute energy between connected food sources.
        High-energy plants share with low-energy neighbors via network.
        
        Args:
            food_objects: List of food WorldObjects
            transfer_rate: Override transfer rate
        """
        if transfer_rate is None:
            transfer_rate = self.nutrient_transfer_rate
            
        # Build list of (object, grid_pos, network_density)
        connected = []
        for obj in food_objects:
            if not obj.alive or obj.compliance < 0.5:
                continue
            gi, gj = self.world_to_grid(obj.pos)
            density = self.density[gi, gj]
            if density > 0.1:  # Must be connected to network
                connected.append((obj, gi, gj, density))
        
        if len(connected) < 2:
            return
            
        # Calculate average energy
        total_energy = sum(obj.energy for obj, _, _, _ in connected)
        avg_energy = total_energy / len(connected)
        
        # Transfer from high to low
        for obj, gi, gj, density in connected:
            if obj.energy > avg_energy * 1.2:
                # Donor - give some energy
                transfer = (obj.energy - avg_energy) * transfer_rate * density
                obj.energy -= transfer
                self.total_nutrient_transferred += transfer
            elif obj.energy < avg_energy * 0.8:
                # Recipient - receive energy proportional to network connection
                # Energy comes from the network's "pool"
                receive = (avg_energy - obj.energy) * transfer_rate * density * 0.5
                obj.energy += receive
    
    def get_visual_data(self) -> dict:
        """Get data for visualization."""
        return {
            'density': self.density.copy(),
            'signal': self.signal.copy(),
            'fruiting_bodies': [(m.pos.copy(), m.size, m.age/m.max_age) 
                               for m in self.fruiting_bodies],
            'resolution': self.resolution,
            'world_size': self.world_size,
        }
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'density': self.density.tolist(),
            'signal': self.signal.tolist(),
            'fruiting_bodies': [
                {'pos': m.pos.tolist(), 'age': m.age, 'max_age': m.max_age, 'size': m.size}
                for m in self.fruiting_bodies
            ],
            'total_nutrient_transferred': self.total_nutrient_transferred,
            'total_fruiting_bodies_spawned': self.total_fruiting_bodies_spawned,
            'step_count': self.step_count,
        }
    
    @classmethod
    def from_dict(cls, d: dict, world_size: float) -> 'MyceliumNetwork':
        """Deserialize from persistence."""
        resolution = len(d['density'])
        network = cls(world_size, resolution)
        network.density = np.array(d['density'])
        network.signal = np.array(d.get('signal', np.zeros_like(network.density)))
        network.fruiting_bodies = [
            FruitingBody(
                pos=np.array(m['pos']),
                age=m['age'],
                max_age=m['max_age'],
                size=m['size']
            )
            for m in d.get('fruiting_bodies', [])
        ]
        network.total_nutrient_transferred = d.get('total_nutrient_transferred', 0)
        network.total_fruiting_bodies_spawned = d.get('total_fruiting_bodies_spawned', 0)
        network.step_count = d.get('step_count', 0)
        return network
