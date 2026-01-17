"""
Terrain System - 2D terrain with heightmap.

Terrain affects movement speed, food growth rates, and provides
shelter and strategic locations for creatures.
"""

from typing import Tuple
from dataclasses import dataclass
import numpy as np


# Default terrain resolution
TERRAIN_RESOLUTION = 100


# =============================================================================
# PERLIN NOISE GENERATOR
# =============================================================================

class PerlinNoise:
    """Simple 2D Perlin noise for terrain generation."""
    
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        # Permutation table
        self.perm = np.arange(256, dtype=np.int32)
        np.random.shuffle(self.perm)
        self.perm = np.tile(self.perm, 2)
        
        # Gradients
        self.gradients = np.array([
            [1, 1], [-1, 1], [1, -1], [-1, -1],
            [1, 0], [-1, 0], [0, 1], [0, -1]
        ], dtype=np.float32)
    
    def _fade(self, t):
        """Smoothstep fade function."""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _lerp(self, a, b, t):
        return a + t * (b - a)
    
    def _dot_grid_gradient(self, ix, iy, x, y):
        """Compute dot product of distance and gradient vectors."""
        idx = self.perm[self.perm[ix % 256] + iy % 256] % 8
        gradient = self.gradients[idx]
        dx, dy = x - ix, y - iy
        return dx * gradient[0] + dy * gradient[1]
    
    def noise(self, x, y):
        """Get noise value at (x, y)."""
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = x0 + 1, y0 + 1
        
        sx = self._fade(x - x0)
        sy = self._fade(y - y0)
        
        n0 = self._dot_grid_gradient(x0, y0, x, y)
        n1 = self._dot_grid_gradient(x1, y0, x, y)
        ix0 = self._lerp(n0, n1, sx)
        
        n0 = self._dot_grid_gradient(x0, y1, x, y)
        n1 = self._dot_grid_gradient(x1, y1, x, y)
        ix1 = self._lerp(n0, n1, sx)
        
        return self._lerp(ix0, ix1, sy)
    
    def octave_noise(self, x, y, octaves: int = 4, persistence: float = 0.5):
        """Multi-octave noise for more natural terrain."""
        total = 0.0
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            total += self.noise(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2.0
        
        return total / max_value


# =============================================================================
# TERRAIN TYPES
# =============================================================================

@dataclass
class TerrainType:
    """Definition of a terrain type and its properties."""
    name: str
    height_min: float
    height_max: float
    color: str
    movement_cost: float      # Multiplier on movement speed
    food_growth_rate: float   # Multiplier on food spawning
    passable: bool
    predator_avoid: float = 0.0   # Chance predators avoid this terrain (0-1)
    shelter_bonus: float = 0.0    # Reduces hunger rate when resting here


# Terrain type definitions
TERRAIN_WATER = TerrainType(
    name="water",
    height_min=-1.0,
    height_max=0.2,
    color="#1a5276",
    movement_cost=3.0,
    food_growth_rate=0.1,
    passable=True
)

TERRAIN_SHORE = TerrainType(
    name="shore",
    height_min=0.2,
    height_max=0.3,
    color="#d4ac6e",
    movement_cost=1.2,
    food_growth_rate=0.8,
    passable=True
)

TERRAIN_PLAINS = TerrainType(
    name="plains",
    height_min=0.3,
    height_max=0.55,
    color="#27ae60",
    movement_cost=1.0,
    food_growth_rate=1.5,
    passable=True
)

TERRAIN_FOREST = TerrainType(
    name="forest",
    height_min=0.55,
    height_max=0.7,
    color="#1e8449",
    movement_cost=1.3,
    food_growth_rate=1.2,
    passable=True,
    predator_avoid=0.2
)

TERRAIN_HILLS = TerrainType(
    name="hills",
    height_min=0.7,
    height_max=0.82,
    color="#a04000",
    movement_cost=1.8,
    food_growth_rate=0.5,
    passable=True
)

TERRAIN_CAVE = TerrainType(
    name="cave",
    height_min=0.82,
    height_max=0.90,
    color="#4a3728",
    movement_cost=1.5,
    food_growth_rate=0.2,
    passable=True,
    predator_avoid=0.7,
    shelter_bonus=0.3
)

TERRAIN_MOUNTAIN = TerrainType(
    name="mountain",
    height_min=0.90,
    height_max=1.0,
    color="#707070",
    movement_cost=5.0,
    food_growth_rate=0.05,
    passable=False
)

ALL_TERRAIN_TYPES = [
    TERRAIN_WATER, TERRAIN_SHORE, TERRAIN_PLAINS, 
    TERRAIN_FOREST, TERRAIN_HILLS, TERRAIN_CAVE, TERRAIN_MOUNTAIN
]


def get_terrain_type(height: float) -> TerrainType:
    """Get terrain type for a given height value."""
    for terrain in ALL_TERRAIN_TYPES:
        if terrain.height_min <= height < terrain.height_max:
            return terrain
    return TERRAIN_PLAINS


# =============================================================================
# TERRAIN CLASS
# =============================================================================

class Terrain:
    """
    2D terrain with heightmap, affecting movement and food growth.
    
    Includes scarification layer - accumulated deformation from creature activity.
    Scars affect movement cost, food growth, and create historical record.
    """
    
    def __init__(self, world_size: float, resolution: int = 100, seed: int = None):
        """
        Initialize terrain.
        
        Args:
            world_size: Size of world in units
            resolution: Grid resolution (NxN cells)
            seed: Random seed for reproducible terrain
        """
        self.world_size = world_size
        self.resolution = resolution
        self.cell_size = world_size / resolution
        
        # Generate heightmap
        self.heightmap = self._generate_heightmap(seed)
        
        # === SCARIFICATION LAYER ===
        # Accumulated terrain deformation from creature activity
        # Positive = worn down (paths), Negative = built up (mounds)
        self.scars = np.zeros((resolution, resolution))
        self.scar_decay_rate = 0.0001  # Very slow natural healing
        self.max_scar_depth = 0.15  # Maximum deformation
        
        # Cache terrain types for each cell
        self.terrain_grid = np.empty((resolution, resolution), dtype=object)
        for i in range(resolution):
            for j in range(resolution):
                self.terrain_grid[i, j] = get_terrain_type(self.heightmap[i, j])
        
        # Pre-compute color map for visualization
        self.color_map = self._generate_color_map()
        
        print(f"[Terrain] Generated {resolution}x{resolution} terrain")
        self._print_terrain_stats()
    
    def _generate_heightmap(self, seed: int = None) -> np.ndarray:
        """Generate terrain using multi-octave Perlin noise."""
        perlin = PerlinNoise(seed)
        heightmap = np.zeros((self.resolution, self.resolution))
        
        scale = 0.05
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                x = (i / self.resolution) * self.world_size * scale
                y = (j / self.resolution) * self.world_size * scale
                height = perlin.octave_noise(x, y, octaves=4, persistence=0.5)
                heightmap[i, j] = (height + 1) / 2
        
        self._add_water_features(heightmap)
        return heightmap
    
    def _add_water_features(self, heightmap: np.ndarray):
        """Add lakes/ponds - guaranteed water zones for aquatic life."""
        # Create at least one large lake
        n_lakes = np.random.randint(3, 6)
        
        for i_lake in range(n_lakes):
            cx = np.random.randint(15, self.resolution - 15)
            cy = np.random.randint(15, self.resolution - 15)
            
            # First lake is larger and deeper
            if i_lake == 0:
                radius = np.random.randint(12, 20)
                depth = 0.35  # Deep enough to guarantee water
            else:
                radius = np.random.randint(6, 12)
                depth = np.random.uniform(0.2, 0.35)
            
            for i in range(max(0, cx - radius - 2), min(self.resolution, cx + radius + 2)):
                for j in range(max(0, cy - radius - 2), min(self.resolution, cy + radius + 2)):
                    dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                    if dist < radius:
                        # Smooth falloff from center
                        factor = 1 - (dist / radius)**2
                        heightmap[i, j] -= depth * factor
                    elif dist < radius + 2:
                        # Shore gradient
                        shore_factor = 1 - (dist - radius) / 2
                        heightmap[i, j] -= depth * 0.3 * shore_factor
        
        # Clamp but allow negative (deep water)
        heightmap[:] = np.clip(heightmap, -0.1, 1)
    
    def _generate_color_map(self) -> np.ndarray:
        """Generate RGB color array for visualization."""
        colors = np.zeros((self.resolution, self.resolution, 3))
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                terrain = self.terrain_grid[i, j]
                hex_color = terrain.color.lstrip('#')
                r = int(hex_color[0:2], 16) / 255
                g = int(hex_color[2:4], 16) / 255
                b = int(hex_color[4:6], 16) / 255
                colors[i, j] = [r, g, b]
        
        return np.transpose(colors, (1, 0, 2))
    
    def _print_terrain_stats(self):
        """Print terrain composition."""
        counts = {t.name: 0 for t in ALL_TERRAIN_TYPES}
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                counts[self.terrain_grid[i, j].name] += 1
        
        total = self.resolution ** 2
        print("[Terrain] Composition:")
        for name, count in counts.items():
            pct = 100 * count / total
            if pct > 0.5:
                print(f"  {name}: {pct:.1f}%")
    
    def world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        x = int((pos[0] + self.world_size/2) / self.world_size * self.resolution)
        y = int((pos[1] + self.world_size/2) / self.world_size * self.resolution)
        x = np.clip(x, 0, self.resolution - 1)
        y = np.clip(y, 0, self.resolution - 1)
        return x, y
    
    def get_height(self, pos: np.ndarray) -> float:
        """Get terrain height at world position."""
        i, j = self.world_to_grid(pos)
        return self.heightmap[i, j]
    
    def get_terrain_at(self, pos: np.ndarray) -> TerrainType:
        """Get terrain type at world position."""
        i, j = self.world_to_grid(pos)
        return self.terrain_grid[i, j]
    
    def get_movement_cost(self, pos: np.ndarray) -> float:
        """Get movement cost multiplier at position, modified by scars."""
        base_cost = self.get_terrain_at(pos).movement_cost
        i, j = self.world_to_grid(pos)
        scar = self.scars[i, j]
        
        # Worn paths (positive scars) reduce movement cost
        # Built-up areas (negative scars) increase it slightly
        if scar > 0:
            # Well-worn path - easier to traverse
            return base_cost * (1.0 - scar * 2.0)  # Up to 30% easier
        else:
            # Built up area - slightly harder
            return base_cost * (1.0 - scar * 0.5)  # Up to 7% harder
    
    def get_food_growth_rate(self, pos: np.ndarray) -> float:
        """Get food growth rate multiplier at position, modified by scars."""
        base_rate = self.get_terrain_at(pos).food_growth_rate
        i, j = self.world_to_grid(pos)
        scar = self.scars[i, j]
        
        # Worn paths have less vegetation
        # Built-up areas (from death/activity) can be more fertile
        if scar > 0.03:
            return base_rate * max(0.3, 1.0 - scar * 3.0)  # Trampled = less growth
        elif scar < -0.01:
            return base_rate * (1.0 - scar * 3.0)  # Enriched = more growth (scar is negative)
        return base_rate
    
    def add_scar(self, pos: np.ndarray, amount: float, radius: float = 1.0):
        """
        Add scarification at position.
        
        Args:
            pos: World position
            amount: Scar amount (positive = wear down, negative = build up)
            radius: Effect radius in grid cells
        """
        ci, cj = self.world_to_grid(pos)
        r_cells = max(1, int(radius / self.cell_size))
        
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                i, j = ci + di, cj + dj
                if 0 <= i < self.resolution and 0 <= j < self.resolution:
                    dist = np.sqrt(di**2 + dj**2)
                    if dist <= r_cells:
                        falloff = 1.0 - (dist / (r_cells + 1))
                        self.scars[i, j] += amount * falloff
                        self.scars[i, j] = np.clip(
                            self.scars[i, j], 
                            -self.max_scar_depth, 
                            self.max_scar_depth
                        )
    
    def add_footprint(self, pos: np.ndarray, weight: float = 1.0):
        """Add a small wear scar from creature walking."""
        self.add_scar(pos, 0.0001 * weight, radius=0.5)
    
    def add_death_site(self, pos: np.ndarray, energy: float):
        """
        Mark a death site - nutrients enrich the soil.
        Creates negative scar (built-up/enriched).
        """
        enrichment = -0.02 * (energy / 50.0)  # Scale with creature energy
        self.add_scar(pos, enrichment, radius=2.0)
    
    def update_scars(self):
        """Slowly heal/decay scars over time."""
        # Very slow exponential decay toward zero
        self.scars *= (1.0 - self.scar_decay_rate)
    
    def get_scar_at(self, pos: np.ndarray) -> float:
        """Get scar depth at position."""
        i, j = self.world_to_grid(pos)
        return self.scars[i, j]
    
    def is_passable(self, pos: np.ndarray) -> bool:
        """Check if position is passable."""
        return self.get_terrain_at(pos).passable
    
    def get_gradient(self, pos: np.ndarray) -> np.ndarray:
        """Get terrain gradient (slope direction) at position."""
        i, j = self.world_to_grid(pos)
        
        if i > 0 and i < self.resolution - 1:
            dx = (self.heightmap[i+1, j] - self.heightmap[i-1, j]) / 2
        else:
            dx = 0
        
        if j > 0 and j < self.resolution - 1:
            dy = (self.heightmap[i, j+1] - self.heightmap[i, j-1]) / 2
        else:
            dy = 0
        
        return np.array([dx, dy])
    
    def find_nearby_passable(self, pos: np.ndarray, search_radius: float = 5.0) -> np.ndarray:
        """Find nearest passable position if current is impassable."""
        if self.is_passable(pos):
            return pos
        
        for r in np.linspace(1, search_radius, 10):
            for angle in np.linspace(0, 2*np.pi, 16):
                test_pos = pos + r * np.array([np.cos(angle), np.sin(angle)])
                if self.is_passable(test_pos):
                    return test_pos
        
        return pos
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'heightmap': self.heightmap.tolist(),
            'world_size': self.world_size,
            'resolution': self.resolution,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Terrain':
        """Deserialize from persistence."""
        terrain = cls.__new__(cls)
        terrain.heightmap = np.array(data['heightmap'])
        terrain.world_size = data['world_size']
        terrain.resolution = data['resolution']
        terrain.cell_size = terrain.world_size / terrain.resolution
        
        terrain.terrain_grid = np.empty((terrain.resolution, terrain.resolution), dtype=object)
        for i in range(terrain.resolution):
            for j in range(terrain.resolution):
                terrain.terrain_grid[i, j] = get_terrain_type(terrain.heightmap[i, j])
        
        terrain.color_map = terrain._generate_color_map()
        return terrain


def apply_terrain_movement(creature, terrain: Terrain, force: np.ndarray, dt: float):
    """
    Apply movement with terrain effects.
    
    Args:
        creature: Creature object with pos, vel attributes
        terrain: Terrain object
        force: Applied force vector
        dt: Timestep
        
    Returns:
        (new_pos, new_vel) tuple
    """
    current_terrain = terrain.get_terrain_at(creature.pos)
    movement_cost = current_terrain.movement_cost
    
    effective_force = force / movement_cost
    
    new_vel = creature.vel + effective_force * dt * 20
    new_vel *= 0.96
    new_pos = creature.pos + new_vel * dt
    
    if terrain.is_passable(new_pos):
        return new_pos, new_vel
    else:
        # Try x-only movement
        test_pos = np.array([new_pos[0], creature.pos[1]])
        if terrain.is_passable(test_pos):
            return test_pos, np.array([new_vel[0], 0])
        
        # Try y-only movement
        test_pos = np.array([creature.pos[0], new_pos[1]])
        if terrain.is_passable(test_pos):
            return test_pos, np.array([0, new_vel[1]])
        
        return creature.pos, np.zeros(2)
