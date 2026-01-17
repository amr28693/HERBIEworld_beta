"""
World Objects - Items that creatures can interact with.

Includes food, barriers, nutrients, meat chunks, and corpses.
"""

from typing import List, Tuple, Optional
import numpy as np

from ..core.constants import (
    WORLD_L, BODY_L, Nx, Ny, X, Y, dx, dt
)


class WorldObject:
    """
    An object in the world that creatures can interact with.
    
    Objects have:
    - Position and velocity (can be pushed)
    - Size and compliance (soft vs hard)
    - Energy (for food objects)
    - Contact detection with creature body fields
    """
    
    def __init__(self, pos, size=1.0, compliance=0.5, mass=1.0, color='green', energy=None):
        """
        Initialize a world object.
        
        Args:
            pos: (x, y) position
            size: Object radius
            compliance: 0=hard barrier, 1=soft/edible
            mass: Affects how easily it's pushed
            color: Display color
            energy: Energy content (None = auto from compliance)
        """
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2)
        self.size = size
        self.initial_size = size
        self.compliance = compliance
        self.mass = mass
        self.color = color
        self.contact = 0.0
        self.reward = 0.0
        
        if energy is None:
            self.energy = 40.0 * compliance if compliance > 0.5 else 0.0
        else:
            self.energy = energy
        self.max_energy = self.energy
        self.alive = True
        
    def compute_contact(self, creature_pos: np.ndarray, body_I: np.ndarray) -> float:
        """
        Compute contact strength with creature body field.
        
        Args:
            creature_pos: Creature's world position
            body_I: Creature's body intensity field
            
        Returns:
            Contact strength (0-1)
        """
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
        """
        Compute reward and energy extraction from contact.
        
        Returns:
            (reward, extracted_mass) tuple
        """
        if not self.alive or self.contact < 0.04:
            self.reward = 0.0
            return 0.0, 0.0
            
        if self.compliance > 0.5 and self.energy > 0:
            # Soft/edible object - extract energy
            extraction = min(self.contact * 1.5, self.energy)
            self.energy -= extraction
            self.reward = extraction * (self.compliance - 0.3)
            self.size = 0.4 + 0.6 * self.initial_size * (self.energy / (self.max_energy + 1e-6))
            if self.energy < 0.5:
                self.alive = False
            return self.reward, extraction
        else:
            # Hard object - just contact reward (negative for barriers)
            self.reward = self.contact * (self.compliance - 0.35) * 2.0
            return self.reward, 0.0
        
    def apply_push(self, creature_pos: np.ndarray, creature_vel: np.ndarray, 
                   body_momentum: np.ndarray, contact_strength: float):
        """
        Apply push force from creature contact.
        
        Args:
            creature_pos: Creature position
            creature_vel: Creature velocity
            body_momentum: Body field momentum
            contact_strength: Contact strength
        """
        if not self.alive or contact_strength < 0.08:
            return
            
        direction = self.pos - creature_pos
        dist = np.linalg.norm(direction)
        if dist < 0.1:
            return
            
        direction = direction / dist
        push_vec = creature_vel * 0.4 + body_momentum * 1.5
        push_along = np.dot(push_vec, direction)
        
        if push_along > 0:
            force = direction * push_along * contact_strength / self.mass
            force *= self.compliance * 0.5 + 0.2
            self.vel += force * dt * 15
            
    def update(self):
        """Update object physics (velocity decay, boundary check)."""
        if not self.alive:
            return
            
        self.vel *= 0.94
        self.pos += self.vel * dt
        
        # Boundary containment
        margin = self.size + 1
        self.pos = np.clip(self.pos, -WORLD_L/2 + margin, WORLD_L/2 - margin)
        
        # Bounce off boundaries
        if abs(self.pos[0]) > WORLD_L/2 - margin - 0.1:
            self.vel[0] *= -0.5
        if abs(self.pos[1]) > WORLD_L/2 - margin - 0.1:
            self.vel[1] *= -0.5
            
    def get_potential(self, creature_pos: np.ndarray) -> np.ndarray:
        """
        Get potential field for this object (attraction/repulsion).
        
        Args:
            creature_pos: Creature position
            
        Returns:
            2D potential field
        """
        if not self.alive:
            return np.zeros((Ny, Nx))
            
        rel = self.pos - creature_pos
        dist = np.sqrt((X - rel[0])**2 + (Y - rel[1])**2)
        
        if self.compliance > 0.5:
            # Attractive (food)
            strength = self.compliance * 0.5 * (1 + self.energy / (self.max_energy + 1e-6))
            return -strength * 2.5 * np.exp(-dist**2 / self.size**2)
        else:
            # Repulsive (barrier)
            barrier_dist = dist - self.size
            return (1 - self.compliance) * 5.0 * np.exp(-np.maximum(0, barrier_dist)**2 / 0.6)


class NutrientPatch:
    """
    Dropped nutrients that can sprout into new food.
    
    When creatures defecate, nutrients are deposited that can
    eventually grow into new food plants.
    """
    
    def __init__(self, pos, nutrients: float):
        """
        Initialize nutrient patch.
        
        Args:
            pos: Position
            nutrients: Nutrient amount (affects resulting plant size)
        """
        self.pos = np.array(pos, dtype=float)
        self.nutrients = nutrients
        self.age = 0
        self.sprouted = False
        self.sprout_time = int(np.random.uniform(300, 600))
        
    def update(self) -> bool:
        """
        Update nutrient patch.
        
        Returns:
            True if ready to sprout
        """
        self.age += 1
        if self.age > self.sprout_time and not self.sprouted:
            self.sprouted = True
            return True
        return False


class MeatChunk(WorldObject):
    """
    A chunk of meat from a killed creature.
    
    Can be picked up, carried, and eaten by Herbies.
    Decays over time if not consumed.
    """
    
    QUADRANT_NAMES = ['head', 'torso', 'left_limbs', 'right_limbs']
    
    def __init__(self, pos: np.ndarray, source_species: str, quadrant: str, energy: float, 
                 source_name: str = None):
        """
        Initialize meat chunk.
        
        Args:
            pos: Position
            source_species: Species the meat came from
            quadrant: Body part ('head', 'torso', 'left_limbs', 'right_limbs')
            energy: Energy content
            source_name: Name of source creature (for Herbies)
        """
        super().__init__(
            pos=pos,
            size=0.8,
            compliance=0.9,  # Very soft/edible
            mass=1.5,
            color='#8B0000',  # Dark red
            energy=energy
        )
        self.source_species = source_species
        self.source_name = source_name
        self.quadrant = quadrant
        self.decay_timer = 2000  # Decays after ~2000 steps if not eaten
        self.is_meat = True
        
    def step(self):
        """
        Meat decays over time.
        
        Returns:
            True if still alive, False if decayed
        """
        self.decay_timer -= 1
        if self.decay_timer <= 0:
            self.alive = False
            return False
        # Slowly lose energy as it rots
        if self.decay_timer < 500:
            self.energy *= 0.998
        return True
    
    def __repr__(self):
        if self.source_name:
            return f"MeatChunk({self.source_name}'s {self.quadrant}, E={self.energy:.1f})"
        return f"MeatChunk({self.source_species} {self.quadrant}, E={self.energy:.1f})"


class HerbieCorpse(WorldObject):
    """
    A dead Herbie's body.
    
    Can be:
    1. Gripped and moved (for burial, or to keep away from predators)
    2. Dismembered into meat chunks (by sustained interaction)
    
    This enables emergent behaviors:
    - Carrying fallen friends to graves
    - Cannibalism in desperate times
    - Keeping bodies away from scavengers
    """
    
    def __init__(self, pos: np.ndarray, herbie_name: str, body_energy: float, 
                 cause_of_death: str = "unknown"):
        """
        Initialize Herbie corpse.
        
        Args:
            pos: Position
            herbie_name: Name of the deceased Herbie
            body_energy: Energy remaining in body
            cause_of_death: How they died
        """
        super().__init__(
            pos=pos,
            size=1.2,
            compliance=0.4,  # Not directly edible - must be dismembered
            mass=4.0,  # Heavy - hard to move
            color='#4a4a4a',  # Gray/pale
            energy=body_energy
        )
        self.herbie_name = herbie_name
        self.cause_of_death = cause_of_death
        self.decay_timer = 3000  # Longer decay than meat
        self.is_corpse = True
        self.dismember_progress = 0.0
        self.dismember_threshold = 50.0
        self.gripped_by = None
        
    def step(self):
        """
        Corpse decays over time, eventually becomes bones.
        
        Returns:
            True if still present, False if decayed completely
        """
        self.decay_timer -= 1
        if self.decay_timer <= 0:
            self.alive = False
            print(f"[CORPSE] 💀 '{self.herbie_name}'s remains have decayed to bones")
            return False
        return True
    
    def add_dismember_progress(self, amount: float) -> bool:
        """
        Add progress toward dismemberment.
        
        Args:
            amount: Progress amount
            
        Returns:
            True if corpse should now become meat chunks
        """
        self.dismember_progress += amount
        if self.dismember_progress >= self.dismember_threshold:
            return True
        return False
    
    def to_meat_chunks(self) -> List['MeatChunk']:
        """
        Convert corpse to 4 meat chunks.
        
        Returns:
            List of MeatChunk objects
        """
        chunks = []
        energy_per_chunk = self.energy / 4
        
        offsets = [
            np.array([0.0, 1.0]),   # head
            np.array([0.0, -1.0]),  # torso  
            np.array([-1.0, 0.0]),  # left limbs
            np.array([1.0, 0.0]),   # right limbs
        ]
        
        for quadrant, offset in zip(MeatChunk.QUADRANT_NAMES, offsets):
            chunk_pos = self.pos + offset * np.random.uniform(0.3, 0.8)
            chunk = MeatChunk(
                chunk_pos, 
                "Herbie", 
                quadrant, 
                energy_per_chunk,
                source_name=self.herbie_name
            )
            chunks.append(chunk)
        
        self.alive = False
        print(f"[CORPSE] 🔪 '{self.herbie_name}'s body was dismembered into meat")
        return chunks
    
    def __repr__(self):
        return f"HerbieCorpse('{self.herbie_name}', E={self.energy:.1f}, decay={self.decay_timer})"


def spawn_meat_chunks(pos: np.ndarray, source_species: str, total_energy: float,
                      source_name: str = None) -> List[MeatChunk]:
    """
    Spawn 4 meat chunks (quadrants) from a killed creature.
    
    Used when Herbies kill an Apex with LITE_ORE.
    
    Args:
        pos: Death position
        source_species: Species that was killed
        total_energy: Total energy to distribute
        source_name: Name of killed creature (optional)
        
    Returns:
        List of 4 MeatChunk objects
    """
    chunks = []
    energy_per_chunk = total_energy / 4
    
    offsets = [
        np.array([0.0, 1.0]),   # head - north
        np.array([0.0, -1.0]),  # torso - south  
        np.array([-1.0, 0.0]),  # left limbs - west
        np.array([1.0, 0.0]),   # right limbs - east
    ]
    
    for quadrant, offset in zip(MeatChunk.QUADRANT_NAMES, offsets):
        chunk_pos = pos + offset * np.random.uniform(0.5, 1.5)
        chunk = MeatChunk(chunk_pos, source_species, quadrant, energy_per_chunk, source_name)
        chunks.append(chunk)
    
    name_str = f"'{source_name}'" if source_name else source_species
    print(f"[MEAT] 🥩 {name_str} dismembered into 4 chunks ({energy_per_chunk:.1f} energy each)")
    return chunks
