"""
Ant Colony System with Reaction-Diffusion Pheromones.

A biologically-inspired ant colony simulation using Gray-Scott
reaction-diffusion equations for emergent pheromone trail patterns.

The ants don't follow hard-coded paths - they follow chemical
gradients that emerge from the RD dynamics, creating natural
foraging patterns.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    pass


# Try scipy for convolution, fallback to pure numpy
try:
    from scipy.signal import convolve2d
except ImportError:
    def convolve2d(field, kernel, mode='same', boundary='wrap'):
        """Simple convolution fallback."""
        h, w = field.shape
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        padded = np.pad(field, ((pad_h, pad_h), (pad_w, pad_w)), mode='wrap')
        
        result = np.zeros_like(field)
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        
        return result


# =============================================================================
# REACTION-DIFFUSION FIELD
# =============================================================================

class ReactionDiffusionField:
    """
    Gray-Scott reaction-diffusion system for pheromone trails.
    
    Equations:
        ∂u/∂t = Dᵤ∇²u - uv² + F(1-u)
        ∂v/∂t = Dᵥ∇²v + uv² - (F+k)v
    
    u = "food" chemical (substrate)
    v = "ant pheromone" chemical (catalyst)
    
    F = feed rate (how fast u is replenished)
    k = kill rate (how fast v decays)
    
    Different F,k values create different patterns:
    - Spots, stripes, waves, chaos
    """
    
    # Pattern presets (F, k)
    PATTERNS = {
        'spots': (0.035, 0.065),
        'stripes': (0.04, 0.06),
        'waves': (0.025, 0.05),
        'maze': (0.029, 0.057),
        'chaos': (0.026, 0.051),
        'trails': (0.037, 0.06),  # Best for ant trails
    }
    
    def __init__(self, width: int = 100, height: int = 100, 
                 pattern: str = 'trails', scale: float = 1.0):
        self.width = width
        self.height = height
        self.scale = scale
        
        # Diffusion rates
        self.Du = 0.16  # u diffuses faster
        self.Dv = 0.08  # v diffuses slower (creates patterns)
        
        # Reaction rates
        self.F, self.k = self.PATTERNS.get(pattern, self.PATTERNS['trails'])
        
        # Fields
        self.u = np.ones((height, width))
        self.v = np.zeros((height, width))
        
        # Seed initial perturbations
        self._seed_initial()
        
        # Laplacian kernel
        self.laplacian = np.array([
            [0.05, 0.2, 0.05],
            [0.2, -1.0, 0.2],
            [0.05, 0.2, 0.05]
        ])
        
        print(f"[RD] Initialized {width}x{height} field, pattern={pattern}")
    
    def _seed_initial(self, n_seeds: int = 5):
        """Add initial pheromone seeds."""
        for _ in range(n_seeds):
            cx = np.random.randint(20, self.width - 20)
            cy = np.random.randint(20, self.height - 20)
            radius = np.random.randint(3, 8)
            
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if i*i + j*j < radius*radius:
                        x = (cx + i) % self.width
                        y = (cy + j) % self.height
                        self.v[y, x] = 1.0
                        self.u[y, x] = 0.5
    
    def step(self, dt: float = 1.0):
        """Evolve one time step."""
        # Compute Laplacians
        Lu = convolve2d(self.u, self.laplacian, mode='same', boundary='wrap')
        Lv = convolve2d(self.v, self.laplacian, mode='same', boundary='wrap')
        
        # Reaction term
        uvv = self.u * self.v * self.v
        
        # Gray-Scott equations
        du = self.Du * Lu - uvv + self.F * (1 - self.u)
        dv = self.Dv * Lv + uvv - (self.F + self.k) * self.v
        
        # Update
        self.u += du * dt
        self.v += dv * dt
        
        # Clamp
        self.u = np.clip(self.u, 0, 1)
        self.v = np.clip(self.v, 0, 1)
    
    def add_pheromone(self, world_pos: np.ndarray, amount: float = 0.5, radius: float = 2.0):
        """Add pheromone at world position."""
        gx = int((world_pos[0] / self.scale + 0.5) * self.width)
        gy = int((world_pos[1] / self.scale + 0.5) * self.height)
        
        gx = gx % self.width
        gy = gy % self.height
        
        r_grid = int(radius * self.width / self.scale)
        for i in range(-r_grid, r_grid + 1):
            for j in range(-r_grid, r_grid + 1):
                if i*i + j*j < r_grid*r_grid:
                    x = (gx + i) % self.width
                    y = (gy + j) % self.height
                    self.v[y, x] = min(1.0, self.v[y, x] + amount)
                    self.u[y, x] = max(0.0, self.u[y, x] - amount * 0.3)
    
    def get_pheromone_at(self, world_pos: np.ndarray) -> float:
        """Get pheromone concentration at world position."""
        gx = int((world_pos[0] / self.scale + 0.5) * self.width)
        gy = int((world_pos[1] / self.scale + 0.5) * self.height)
        
        gx = np.clip(gx, 0, self.width - 1)
        gy = np.clip(gy, 0, self.height - 1)
        
        return self.v[gy, gx]
    
    def get_gradient_at(self, world_pos: np.ndarray) -> np.ndarray:
        """Get pheromone gradient (direction of increasing concentration)."""
        gx = int((world_pos[0] / self.scale + 0.5) * self.width)
        gy = int((world_pos[1] / self.scale + 0.5) * self.height)
        
        gx = np.clip(gx, 1, self.width - 2)
        gy = np.clip(gy, 1, self.height - 2)
        
        dx = (self.v[gy, gx+1] - self.v[gy, gx-1]) / 2
        dy = (self.v[gy+1, gx] - self.v[gy-1, gx]) / 2
        
        return np.array([dx, dy]) * self.scale
    
    def get_render_image(self) -> np.ndarray:
        """Get RGB image for visualization."""
        v_boosted = np.clip(self.v * 3.0, 0, 1)
        
        rgb = np.zeros((self.height, self.width, 3))
        rgb[:, :, 0] = v_boosted * 0.9
        rgb[:, :, 1] = v_boosted * 1.0
        rgb[:, :, 2] = self.u * 0.2
        
        return np.clip(rgb, 0, 1)
    
    def to_dict(self) -> dict:
        """Serialize."""
        return {
            'u': self.u.tolist(),
            'v': self.v.tolist(),
            'width': self.width,
            'height': self.height,
            'scale': self.scale,
            'F': self.F,
            'k': self.k
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ReactionDiffusionField':
        """Deserialize."""
        field = cls.__new__(cls)
        field.width = data['width']
        field.height = data['height']
        field.scale = data['scale']
        field.F = data['F']
        field.k = data['k']
        field.Du = 0.16
        field.Dv = 0.08
        field.u = np.array(data['u'])
        field.v = np.array(data['v'])
        field.laplacian = np.array([
            [0.05, 0.2, 0.05],
            [0.2, -1.0, 0.2],
            [0.05, 0.2, 0.05]
        ])
        return field


# =============================================================================
# ANT
# =============================================================================

@dataclass
class Ant:
    """Single ant agent."""
    pos: np.ndarray
    vel: np.ndarray
    carrying_food: bool = False
    age: int = 0
    
    def __post_init__(self):
        self.pos = np.array(self.pos, dtype=float)
        self.vel = np.array(self.vel, dtype=float)
        self._id = id(self)
    
    def __eq__(self, other):
        if not isinstance(other, Ant):
            return False
        return self._id == other._id
    
    def __hash__(self):
        return self._id


# =============================================================================
# ANT COLONY
# =============================================================================

class AntColony:
    """
    Ant colony using reaction-diffusion for pheromone trails.
    
    Ants:
    - Follow pheromone gradients
    - Deposit pheromone when moving
    - Seek food, return to nest
    """
    
    def __init__(self, nest_pos: np.ndarray, world_size: float,
                 n_ants: int = 30, rd_field: ReactionDiffusionField = None):
        self.nest_pos = np.array(nest_pos, dtype=float)
        self.world_size = world_size
        
        # Reaction-diffusion field
        if rd_field is None:
            self.rd_field = ReactionDiffusionField(
                width=100, height=100, 
                pattern='trails',
                scale=world_size
            )
        else:
            self.rd_field = rd_field
        
        # Spawn ants
        self.ants: List[Ant] = []
        for _ in range(n_ants):
            offset = np.random.randn(2) * 8
            pos = self.nest_pos + offset
            angle = np.random.uniform(0, 2*np.pi)
            vel = np.array([np.cos(angle), np.sin(angle)]) * 0.8
            self.ants.append(Ant(pos, vel))
        
        # Stats
        self.food_collected = 0
        self.total_ants_ever = n_ants
        self.nest_radius = 5.0
        self.last_swarm = False
        
        print(f"[Colony] Spawned {n_ants} ants at ({nest_pos[0]:.1f}, {nest_pos[1]:.1f})")
    
    def update(self, food_positions: List[np.ndarray], dt: float = 1.0):
        """Update colony for one step."""
        self.last_swarm = False
        
        # Evolve reaction-diffusion
        for _ in range(3):
            self.rd_field.step(dt * 0.3)
        
        # Maintain nest pheromone
        self.rd_field.add_pheromone(self.nest_pos, amount=0.1, radius=self.nest_radius)
        
        # Update each ant
        for ant in self.ants:
            self._update_ant(ant, food_positions, dt)
        
        # Spawn new ant occasionally
        if len(self.ants) < 50 and np.random.random() < 0.01:
            offset = np.random.randn(2) * 2
            self.ants.append(Ant(self.nest_pos + offset, np.random.randn(2) * 0.3))
            self.total_ants_ever += 1
    
    def _update_ant(self, ant: Ant, food_positions: List[np.ndarray], dt: float):
        """Update single ant."""
        ant.age += 1
        
        gradient = self.rd_field.get_gradient_at(ant.pos)
        
        if ant.carrying_food:
            # Return to nest
            to_nest = self.nest_pos - ant.pos
            dist_to_nest = np.linalg.norm(to_nest)
            
            if dist_to_nest < self.nest_radius:
                ant.carrying_food = False
                self.food_collected += 1
                self.rd_field.add_pheromone(ant.pos, amount=0.3, radius=1.5)
            else:
                direction = to_nest / (dist_to_nest + 0.1)
                ant.vel = 0.8 * ant.vel + 0.4 * direction + 0.1 * np.random.randn(2)
        else:
            # Search for food
            if np.linalg.norm(gradient) > 0.01:
                ant.vel = 0.7 * ant.vel + 0.3 * gradient / (np.linalg.norm(gradient) + 0.1)
            
            ant.vel += 0.15 * np.random.randn(2)
            
            for food_pos in food_positions:
                if np.linalg.norm(ant.pos - food_pos) < 2.0:
                    ant.carrying_food = True
                    break
        
        # Clamp velocity
        speed = np.linalg.norm(ant.vel)
        max_speed = 1.5
        if speed > max_speed:
            ant.vel = ant.vel / speed * max_speed
        
        # Move
        ant.pos += ant.vel * dt
        
        # Boundary wrapping
        half = self.world_size / 2
        ant.pos[0] = ((ant.pos[0] + half) % self.world_size) - half
        ant.pos[1] = ((ant.pos[1] + half) % self.world_size) - half
        
        # Deposit pheromone
        deposit = 0.05 if ant.carrying_food else 0.02
        self.rd_field.add_pheromone(ant.pos, amount=deposit, radius=0.5)
    
    def get_ants_near(self, pos: np.ndarray, radius: float) -> List[Ant]:
        """Get ants within radius of position."""
        return [a for a in self.ants if np.linalg.norm(a.pos - pos) < radius]
    
    def disturb(self, pos: np.ndarray, radius: float = 5.0):
        """Disturb ants near position (scatter them)."""
        for ant in self.get_ants_near(pos, radius):
            away = ant.pos - pos
            dist = np.linalg.norm(away)
            if dist > 0.1:
                ant.vel += (away / dist) * 2.0
        
        if len(self.get_ants_near(pos, radius)) > 5:
            self.last_swarm = True
    
    def get_swarm_positions(self) -> List[np.ndarray]:
        """Get positions for swarm visualization."""
        return [ant.pos for ant in self.ants]
    
    def to_dict(self) -> dict:
        """Serialize."""
        return {
            'nest_pos': self.nest_pos.tolist(),
            'world_size': self.world_size,
            'ants': [{'pos': a.pos.tolist(), 'vel': a.vel.tolist(), 
                     'carrying': a.carrying_food, 'age': a.age} for a in self.ants],
            'food_collected': self.food_collected,
            'rd_field': self.rd_field.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AntColony':
        """Deserialize."""
        rd_field = ReactionDiffusionField.from_dict(data['rd_field'])
        colony = cls(
            np.array(data['nest_pos']),
            data['world_size'],
            n_ants=0,
            rd_field=rd_field
        )
        colony.ants = [
            Ant(np.array(a['pos']), np.array(a['vel']), a['carrying'], a['age'])
            for a in data['ants']
        ]
        colony.food_collected = data['food_collected']
        return colony


# =============================================================================
# HERBIE-ANT SPECIAL INTERACTIONS
# =============================================================================

def herbie_sense_ants(herbie, colony: AntColony) -> dict:
    """
    Give Herbie awareness of ant pheromones and colony.
    Returns sensing data that can affect behavior.
    """
    pheromone = colony.rd_field.get_pheromone_at(herbie.pos)
    gradient = colony.rd_field.get_gradient_at(herbie.pos)
    
    dist_to_nest = np.linalg.norm(herbie.pos - colony.nest_pos)
    near_nest = dist_to_nest < 10.0
    
    nearby_ants = len(colony.get_ants_near(herbie.pos, 5.0))
    
    return {
        'pheromone_level': pheromone,
        'pheromone_gradient': gradient,
        'dist_to_nest': dist_to_nest,
        'near_nest': near_nest,
        'nearby_ants': nearby_ants,
        'danger_level': nearby_ants * (1.0 if near_nest else 0.3)
    }


def herbie_can_raid_colony(herbie, colony: AntColony) -> bool:
    """Check if Herbie can raid the ant colony for food."""
    if not hasattr(herbie, 'hands') or herbie.hands is None:
        return False
    
    dist = np.linalg.norm(herbie.pos - colony.nest_pos)
    if dist > colony.nest_radius + 2:
        return False
    
    nearby = len(colony.get_ants_near(herbie.pos, 5.0))
    return nearby < 10


def herbie_raid_colony(herbie, colony: AntColony) -> float:
    """Herbie raids colony for food."""
    if not herbie_can_raid_colony(herbie, colony):
        return 0.0
    
    grabbed = min(5, len(colony.ants))
    
    for _ in range(grabbed):
        if colony.ants:
            colony.ants.pop()
    
    food_gained = grabbed * 2.0  # ANT_FOOD_VALUE
    
    colony.disturb(herbie.pos, 8.0)
    
    if grabbed > 0:
        print(f"[Herbie] Raided colony! Got {grabbed} ants ({food_gained:.1f} food)")
    
    return food_gained
