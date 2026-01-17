"""
Aquatic Life - Plants and creatures that live in water.

Adds ecological value to water zones which were previously dead space.
Includes aquatic plants (harvestable) and optional aquatic creatures.
"""

from typing import List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

from .objects import WorldObject
from ..core.constants import WORLD_L

if TYPE_CHECKING:
    from .terrain import Terrain


class AquaticPlant(WorldObject):
    """
    An aquatic plant that grows in water/shore terrain.
    
    Characteristics:
    - Grows IN water (inverted from land plants)
    - Slower to harvest (creatures move slow in water)
    - Higher energy payoff (worth the trip)
    - Can spread via water currents
    """
    
    PLANT_TYPES = {
        'kelp': {
            'color': '#2E8B57',  # Sea green
            'size': 1.2,
            'energy': 60.0,
            'growth_rate': 0.02,
            'preferred_terrain': ['water'],
        },
        'lily': {
            'color': '#98FB98',  # Pale green
            'size': 0.8,
            'energy': 35.0,
            'growth_rate': 0.03,
            'preferred_terrain': ['water', 'shore'],
        },
        'algae': {
            'color': '#9ACD32',  # Yellow-green
            'size': 0.5,
            'energy': 20.0,
            'growth_rate': 0.05,
            'preferred_terrain': ['water', 'shore'],
        },
        'reed': {
            'color': '#6B8E23',  # Olive drab
            'size': 1.0,
            'energy': 40.0,
            'growth_rate': 0.025,
            'preferred_terrain': ['shore'],
        },
    }
    
    def __init__(self, pos: np.ndarray, plant_type: str = 'kelp'):
        """
        Initialize aquatic plant.
        
        Args:
            pos: World position
            plant_type: One of 'kelp', 'lily', 'algae', 'reed'
        """
        props = self.PLANT_TYPES.get(plant_type, self.PLANT_TYPES['algae'])
        
        super().__init__(
            pos=pos,
            size=props['size'],
            compliance=0.85,  # Soft, edible
            mass=0.4,  # Light, sways in current
            color=props['color'],
            energy=props['energy']
        )
        
        self.plant_type = plant_type
        self.growth_rate = props['growth_rate']
        self.preferred_terrain = props['preferred_terrain']
        self.is_aquatic = True
        self.sway_phase = np.random.uniform(0, 2*np.pi)
        self.age = 0
        self.mature_age = 100  # Steps until full size
        
    def update_growth(self, terrain_name: str):
        """
        Update plant growth based on terrain.
        
        Args:
            terrain_name: Current terrain type
        """
        self.age += 1
        
        # Check if terrain is suitable
        if terrain_name in self.preferred_terrain:
            growth_mult = 1.5
        elif terrain_name in ['water', 'shore']:
            growth_mult = 0.8
        else:
            # Wrong terrain - decay
            self.energy -= 0.5
            if self.energy <= 0:
                self.alive = False
            return
        
        # Grow if not at max
        if self.energy < self.max_energy:
            self.energy += self.growth_rate * growth_mult
            self.energy = min(self.energy, self.max_energy * 1.2)
        
        # Update size based on energy
        self.size = self.initial_size * (0.5 + 0.5 * self.energy / self.max_energy)
        
        # Sway animation
        self.sway_phase += 0.1
    
    def get_sway_offset(self) -> np.ndarray:
        """Get visual sway offset for rendering."""
        sway_x = 0.15 * np.sin(self.sway_phase)
        sway_y = 0.1 * np.sin(self.sway_phase * 0.7 + 1.0)
        return np.array([sway_x, sway_y])
    
    def can_spread(self) -> bool:
        """Check if plant can produce offspring."""
        return self.energy > self.max_energy * 0.8 and self.age > self.mature_age


# =============================================================================
# PDE-BASED AQUATIC CREATURES
# =============================================================================
# Real emergent aquatic life with:
# - NLSE body field (water-adapted parameters)
# - Piezoelectric skeleton (single pole or dual C-start)
# - KdV tail/fin dynamics for steering
# - Optional torus brain (advanced species)
# - Schooling emerges from wave interactions

@dataclass
class PiezoSkeleton:
    """
    Piezoelectric skeleton for aquatic creatures.
    
    Single pole: Simple fish, slow but efficient
    Dual pole (C-start): Fast escape response, connected at middle
    
    Generates thrust from body field oscillations.
    """
    n_poles: int = 1  # 1 = simple, 2 = C-start capable
    pole_length: float = 1.0
    
    # Pole states (charge accumulation)
    charge: np.ndarray = None
    flex: np.ndarray = None  # Bend angle of each pole segment
    
    # C-start state
    c_start_primed: bool = False
    c_start_cooldown: int = 0
    
    def __post_init__(self):
        segments = 8 if self.n_poles == 1 else 16
        self.charge = np.zeros(segments)
        self.flex = np.zeros(segments)
        self.segments = segments
    
    def update(self, body_field_gradient: float, stress: float = 0.0) -> Tuple[float, float]:
        """
        Update skeleton from body field.
        
        Returns: (thrust, turn_rate)
        """
        # Piezoelectric response - field gradient creates charge
        new_charge = body_field_gradient * 0.3
        self.charge = 0.9 * self.charge
        self.charge[:-1] += new_charge * np.abs(np.sin(np.linspace(0, np.pi, self.segments-1)))
        
        # Charge creates flex (mechanical response)
        self.flex = 0.85 * self.flex + 0.15 * self.charge * 0.5
        
        # Thrust from tail oscillation
        tail_flex = np.sum(self.flex[-3:])
        thrust = np.abs(tail_flex) * 0.4
        
        # Turn from asymmetric flex
        if self.n_poles == 2:
            # Dual pole - left/right asymmetry
            left_flex = np.sum(self.flex[:self.segments//2])
            right_flex = np.sum(self.flex[self.segments//2:])
            turn_rate = (right_flex - left_flex) * 0.3
        else:
            # Single pole - flex direction
            turn_rate = np.sum(self.flex * np.linspace(-1, 1, self.segments)) * 0.2
        
        # C-start escape response
        if self.n_poles == 2 and stress > 0.5 and self.c_start_cooldown == 0:
            self.c_start_primed = True
        
        if self.c_start_primed:
            # Explosive flex then release
            self.flex[:self.segments//2] = 0.8
            self.flex[self.segments//2:] = -0.8
            thrust *= 4.0  # Burst speed
            self.c_start_primed = False
            self.c_start_cooldown = 50
        
        if self.c_start_cooldown > 0:
            self.c_start_cooldown -= 1
        
        return thrust, turn_rate


@dataclass
class FishTail:
    """
    KdV-based tail dynamics for steering.
    
    Soliton pulses travel down the tail, creating thrust.
    Phase relationships between pulses determine turn direction.
    """
    N: int = 24  # Tail segments
    
    # KdV field
    u: np.ndarray = None
    u_prev: np.ndarray = None
    
    # Parameters
    c: float = 0.8   # Wave speed (slower in water)
    g: float = -0.3  # Nonlinearity
    
    def __post_init__(self):
        self.u = np.zeros(self.N)
        self.u_prev = np.zeros(self.N)
        self.x = np.linspace(0, 1, self.N)
        self.dx = self.x[1] - self.x[0]
    
    def inject_pulse(self, amplitude: float, phase: float = 0.0):
        """Inject steering pulse at tail base."""
        pulse = amplitude * np.exp(-((self.x - 0.1)**2) / 0.01)
        self.u[:len(pulse)] += pulse * np.cos(phase)
    
    def evolve(self) -> Tuple[float, float]:
        """
        Evolve tail dynamics.
        
        Returns: (thrust, lateral_force)
        """
        # Simplified KdV step
        u_new = np.zeros(self.N)
        
        for i in range(1, self.N - 1):
            # Advection + nonlinear steepening
            dudx = (self.u[i+1] - self.u[i-1]) / (2 * self.dx)
            u_new[i] = self.u[i] - self.c * dudx * 0.1 + self.g * self.u[i] * dudx * 0.1
        
        # Damping at tip
        u_new *= 0.98
        u_new[-3:] *= 0.9
        
        self.u_prev = self.u.copy()
        self.u = u_new
        
        # Thrust from tip motion
        tip_vel = self.u[-1] - self.u_prev[-1]
        thrust = np.abs(tip_vel) * 2.0
        
        # Lateral force from wave asymmetry
        lateral = np.sum(self.u * np.sin(np.linspace(0, np.pi, self.N))) * 0.5
        
        return thrust, lateral


@dataclass
class AquaticCreature:
    """
    PDE-based aquatic creature with emergent behavior.
    
    Species types:
    - 'minnow': Simple single-pole, no torus, schooling instinct
    - 'fish': Dual-pole C-start, small torus, territorial
    - 'eel': Long single-pole, medium torus, solitary hunter
    
    All have NLSE body field and KdV tail dynamics.
    Couples to water NLSE field and collective school torus.
    Digestion -> poop -> fertilizer -> plant growth cycle.
    """
    pos: np.ndarray
    vel: np.ndarray = None
    species: str = 'minnow'
    
    # Core state
    alive: bool = True
    age: int = 0
    energy: float = 30.0
    size: float = 0.6
    heading: float = 0.0  # Radians
    
    # PDE components (initialized in __post_init__)
    body_psi: np.ndarray = None  # NLSE body field
    skeleton: PiezoSkeleton = None
    tail: FishTail = None
    torus_psi: np.ndarray = None  # Brain (optional)
    
    # Species-specific parameters
    max_age: int = 2000
    has_torus: bool = False
    schooling_strength: float = 0.0
    
    # Behavior state
    stress: float = 0.0
    last_seen_predator: float = 0.0
    
    # Gut/digestion system
    gut_contents: float = 0.0
    gut_capacity: float = 15.0
    poop_pending: bool = False
    last_poop_amount: float = 0.0
    
    # Reproduction
    ready_to_spawn: bool = False
    spawn_cooldown: int = 0
    maturity_age: int = 300
    
    def __post_init__(self):
        if self.vel is None:
            self.vel = np.random.randn(2) * 0.3 + np.array([0.1, 0.1])  # Start with some motion
        
        self.heading = np.arctan2(self.vel[1], self.vel[0])
        
        # Species configuration
        if self.species == 'minnow':
            self.size = 0.4
            self.energy = 20.0
            self.max_age = 1500
            self.skeleton = PiezoSkeleton(n_poles=1, pole_length=0.5)
            self.has_torus = False
            self.schooling_strength = 0.8  # Strong schooling
            self.gut_capacity = 10.0
        elif self.species == 'fish':
            self.size = 0.7
            self.energy = 35.0
            self.max_age = 2500
            self.skeleton = PiezoSkeleton(n_poles=2, pole_length=0.8)  # C-start capable
            self.has_torus = True
            self.schooling_strength = 0.3
            self.gut_capacity = 20.0
        elif self.species == 'eel':
            self.size = 0.5
            self.energy = 45.0
            self.max_age = 4000
            self.skeleton = PiezoSkeleton(n_poles=1, pole_length=1.5)  # Long body
            self.has_torus = True
            self.schooling_strength = 0.0  # Solitary
            self.gut_capacity = 25.0
        else:
            # Default - ensure skeleton exists
            self.skeleton = PiezoSkeleton(n_poles=1, pole_length=0.5)
        
        # Initialize NLSE body field (1D, water-adapted)
        N_body = 32
        self.body_psi = np.zeros(N_body, dtype=complex)
        # Start with stronger random excitation to get movement going
        self.body_psi += 0.3 * (np.random.randn(N_body) + 1j * np.random.randn(N_body))
        self.body_g = -0.5  # Focusing nonlinearity
        
        # Initialize tail
        self.tail = FishTail()
        
        # Inject initial tail pulse to get swimming started
        self.tail.inject_pulse(0.5, self.heading)
        
        # Initialize torus brain if species has one
        if self.has_torus:
            N_torus = 24  # Smaller than land creatures
            self.torus_psi = np.zeros(N_torus, dtype=complex)
            self.torus_psi += 0.1 * (np.random.randn(N_torus) + 1j * np.random.randn(N_torus))
            self.torus_g = -0.3
    
    def update(self, terrain: 'Terrain', food_plants: List[AquaticPlant],
               all_aquatic: List['AquaticCreature'] = None,
               aquatic_system: 'AquaticSystem' = None) -> bool:
        """
        Full PDE-based update step with field coupling.
        
        Couples to:
        - Water body NLSE (currents)
        - School torus (collective behavior)
        """
        if not self.alive:
            return False
        
        self.age += 1
        
        # === REPRODUCTION STATE ===
        if self.spawn_cooldown > 0:
            self.spawn_cooldown -= 1
        
        # Check if ready to reproduce (lower threshold for aquatic life)
        if (self.age > self.maturity_age and 
            self.energy > 20 and  # Lower threshold
            self.spawn_cooldown <= 0):
            self.ready_to_spawn = True
        else:
            self.ready_to_spawn = False
        
        # === NaN PROTECTION ===
        if np.any(np.isnan(self.pos)) or np.any(np.isnan(self.vel)):
            # Reset to safe state
            self.pos = np.array([0.0, 0.0])
            self.vel = np.array([0.0, 0.0])
            self.body_psi = np.zeros_like(self.body_psi)
            self.body_psi += 0.1 * (np.random.randn(len(self.body_psi)) + 1j * np.random.randn(len(self.body_psi)))
        
        # Age death
        if self.age > self.max_age or self.energy <= 0:
            self.alive = False
            return False
        
        # === SENSE ENVIRONMENT ===
        # Clamp position before terrain lookup
        half_world = WORLD_L / 2 - 1
        self.pos = np.clip(self.pos, -half_world, half_world)
        
        current_terrain = terrain.get_terrain_at(self.pos)
        in_water = current_terrain.name in ['water', 'shore']
        
        # Check for predators (land creatures near water)
        self.stress *= 0.95  # Decay
        
        # === EVOLVE BODY FIELD (NLSE) ===
        # Water resistance creates different dynamics
        N = len(self.body_psi)
        dx = 1.0 / N
        
        # Laplacian
        psi_xx = np.zeros(N, dtype=complex)
        psi_xx[1:-1] = (self.body_psi[2:] - 2*self.body_psi[1:-1] + self.body_psi[:-2]) / dx**2
        
        # NLSE evolution (water-damped) - with stability limit
        dpsi = 1j * (0.5 * psi_xx + self.body_g * np.abs(self.body_psi)**2 * self.body_psi)
        dpsi = np.clip(dpsi.real, -1, 1) + 1j * np.clip(dpsi.imag, -1, 1)
        self.body_psi += dpsi * 0.02  # Smaller timestep for stability
        
        # Water damping
        self.body_psi *= 0.99
        
        # Amplitude limit to prevent blowup
        amp = np.abs(self.body_psi)
        if np.max(amp) > 2.0:
            self.body_psi = self.body_psi / np.max(amp) * 2.0
        
        # Swimming motion injects energy
        swim_energy = min(np.linalg.norm(self.vel), 1.0) * 0.05
        self.body_psi[N//2] += swim_energy * np.exp(1j * self.heading)
        
        # === SKELETON RESPONSE ===
        # Body field gradient drives piezo skeleton
        body_gradient = np.abs(self.body_psi[-1]) - np.abs(self.body_psi[0])
        skel_thrust, skel_turn = self.skeleton.update(body_gradient, self.stress)
        
        # === TAIL DYNAMICS (KdV) ===
        # Body oscillations inject tail pulses
        body_osc = np.sum(np.abs(self.body_psi) * np.sin(np.linspace(0, 2*np.pi, N)))
        if body_osc > 0.1:
            self.tail.inject_pulse(body_osc * 0.3, self.heading)
        
        tail_thrust, tail_lateral = self.tail.evolve()
        
        # === TORUS BRAIN (if present) ===
        torus_bias = np.array([0.0, 0.0])
        if self.has_torus and self.torus_psi is not None:
            N_t = len(self.torus_psi)
            dx_t = 2 * np.pi / N_t
            
            # Torus NLSE with stability
            psi_t_xx = np.zeros(N_t, dtype=complex)
            psi_t_xx[1:-1] = (self.torus_psi[2:] - 2*self.torus_psi[1:-1] + self.torus_psi[:-2]) / dx_t**2
            # Periodic boundaries
            psi_t_xx[0] = (self.torus_psi[1] - 2*self.torus_psi[0] + self.torus_psi[-1]) / dx_t**2
            psi_t_xx[-1] = (self.torus_psi[0] - 2*self.torus_psi[-1] + self.torus_psi[-2]) / dx_t**2
            
            dpsi_t = 1j * (0.5 * psi_t_xx + self.torus_g * np.abs(self.torus_psi)**2 * self.torus_psi)
            dpsi_t = np.clip(dpsi_t.real, -0.5, 0.5) + 1j * np.clip(dpsi_t.imag, -0.5, 0.5)
            self.torus_psi += dpsi_t * 0.02
            self.torus_psi *= 0.98
            
            # Amplitude limit
            t_amp = np.abs(self.torus_psi)
            if np.max(t_amp) > 1.5:
                self.torus_psi = self.torus_psi / np.max(t_amp) * 1.5
            
            # Hunger injects arousal
            hunger = 1.0 - self.energy / 50.0
            if hunger > 0.3:
                self.torus_psi[0] += hunger * 0.03
            
            # Extract directional bias from torus
            theta = np.linspace(0, 2*np.pi, N_t, endpoint=False)
            power = np.abs(self.torus_psi)**2
            torus_bias[0] = np.sum(power * np.cos(theta))
            torus_bias[1] = np.sum(power * np.sin(theta))
            bias_norm = np.linalg.norm(torus_bias)
            if bias_norm > 0.01:
                torus_bias = torus_bias / bias_norm * min(0.3, bias_norm)
        
        # === SCHOOLING (emergent from sensing neighbors) ===
        school_force = np.array([0.0, 0.0])
        if self.schooling_strength > 0 and all_aquatic:
            neighbors = [a for a in all_aquatic 
                        if a is not self and a.alive and a.species == self.species]
            
            for neighbor in neighbors[:5]:  # Limit computation
                diff = neighbor.pos - self.pos
                dist = np.linalg.norm(diff)
                
                if dist < 1.5:
                    # Too close - repel
                    school_force -= diff / (dist + 0.1) * 0.3
                elif dist < 8.0:
                    # Cohesion - attract
                    school_force += diff / (dist + 0.1) * 0.1
                    # Alignment - match heading
                    neighbor_dir = np.array([np.cos(neighbor.heading), np.sin(neighbor.heading)])
                    school_force += neighbor_dir * 0.1
            
            school_force *= self.schooling_strength
        
        # === FOOD SEEKING AND EATING ===
        food_force = np.array([0.0, 0.0])
        if self.energy < 25 and food_plants:
            alive_plants = [p for p in food_plants if p.alive]
            if alive_plants:
                nearest = min(alive_plants, key=lambda p: np.linalg.norm(p.pos - self.pos))
                to_food = nearest.pos - self.pos
                dist = np.linalg.norm(to_food)
                if dist < 1.5:
                    # Eat - food goes to gut, not directly to energy
                    eat_amt = min(8.0, nearest.energy, self.gut_capacity - self.gut_contents)
                    nearest.energy -= eat_amt
                    self.gut_contents += eat_amt
                    if nearest.energy <= 0:
                        nearest.alive = False
                elif dist < 12.0:
                    food_force = to_food / dist * 0.5
        
        # === DIGESTION ===
        # Gut contents slowly convert to energy
        if self.gut_contents > 0:
            digest_rate = 0.02  # Slow aquatic digestion
            digested = min(digest_rate, self.gut_contents)
            self.gut_contents -= digested
            self.energy += digested * 0.8  # 80% efficiency
        
        # === POOP ===
        # When gut is full enough, poop
        self.poop_pending = False
        if self.gut_contents > self.gut_capacity * 0.7:
            self.poop_pending = True
            self.last_poop_amount = self.gut_contents * 0.5
            self.gut_contents *= 0.5
        
        # === WATER CURRENT COUPLING ===
        water_current = np.array([0.0, 0.0])
        school_angular_bias = 0.0
        school_direction = self.heading
        
        if aquatic_system is not None:
            # Get water current at position
            water_current = aquatic_system.get_water_current_at(self.pos)
            
            # Get school torus coupling (especially for minnows)
            school_angular_bias, school_direction = aquatic_system.get_school_bias(self.heading)
        
        # === COMBINE FORCES ===
        total_thrust = skel_thrust + tail_thrust
        
        # Minimum thrust to prevent complete stagnation - MORE AGGRESSIVE
        if total_thrust < 0.1:
            total_thrust = 0.1 + np.random.random() * 0.2
        
        # Update heading with multiple influences
        self.heading += skel_turn + tail_lateral * 0.1
        
        # Brain bias
        if np.linalg.norm(torus_bias) > 0.01:
            self.heading += np.arctan2(torus_bias[1], torus_bias[0]) * 0.05
        
        # Local neighbor schooling
        if np.linalg.norm(school_force) > 0.01:
            self.heading += np.arctan2(school_force[1], school_force[0]) * 0.03
        
        # SCHOOL TORUS COUPLING - key for collective behavior
        self.heading += school_angular_bias * self.schooling_strength
        
        # Tendency to align with overall school direction (for schooling species)
        if self.schooling_strength > 0.3:
            heading_diff = school_direction - self.heading
            # Wrap to [-π, π] using modulo (fast, handles any value)
            heading_diff = ((heading_diff + np.pi) % (2 * np.pi)) - np.pi
            self.heading += heading_diff * 0.02 * self.schooling_strength
        
        # Keep heading bounded to prevent drift
        self.heading = ((self.heading + np.pi) % (2 * np.pi)) - np.pi
        
        # Random heading drift for exploration
        self.heading += np.random.randn() * 0.03
        
        # Keep heading bounded and valid
        if np.isnan(self.heading) or np.isinf(self.heading):
            self.heading = np.random.uniform(0, 2 * np.pi)
        self.heading = ((self.heading + np.pi) % (2 * np.pi)) - np.pi
        
        # Thrust in heading direction
        thrust_vec = np.array([np.cos(self.heading), np.sin(self.heading)]) * total_thrust
        
        # Add all forces
        self.vel += thrust_vec * 0.5  # BOOSTED thrust
        self.vel += torus_bias * 0.2  # Brain
        self.vel += school_force * 0.25  # Local neighbors
        self.vel += food_force * 0.3  # Food seeking
        self.vel += water_current * 0.4  # Water current - STRONGER flow influence
        
        # Water resistance (reduced for more visible movement)
        self.vel *= 0.95
        
        # Speed limit and NaN protection
        if np.any(np.isnan(self.vel)) or np.any(np.isinf(self.vel)):
            self.vel = np.random.randn(2) * 0.3  # Reset with more motion
        
        speed = np.linalg.norm(self.vel)
        max_speed = 2.0 if self.species != 'eel' else 2.5  # Faster fish
        if speed > max_speed:
            self.vel = self.vel / speed * max_speed
        
        # Minimum speed - fish should always be swimming!
        if speed < 0.1:
            self.vel += np.array([np.cos(self.heading), np.sin(self.heading)]) * 0.2
        
        # === STAY IN WATER ===
        if not in_water:
            self.stress = 1.0  # Panic!
            
            # Get water center from system and head toward it
            if aquatic_system is not None:
                water_center = aquatic_system.get_water_center()
                to_water = water_center - self.pos
                dist = np.linalg.norm(to_water)
                if dist > 0.1:
                    # Strong pull toward water
                    self.vel = to_water / dist * 1.0
                    self.heading = np.arctan2(to_water[1], to_water[0])
            else:
                # Fallback - reverse direction
                self.vel *= -0.5
                self.heading += np.pi
        
        # === MOVE ===
        self.pos += self.vel * 0.5  # BOOSTED movement scale
        
        # Boundary - strict clamping
        margin = 3.0
        self.pos = np.clip(self.pos, -WORLD_L/2 + margin, WORLD_L/2 - margin)
        
        # Final NaN check
        if np.any(np.isnan(self.pos)):
            # Reset to water center if we have it
            if aquatic_system is not None:
                self.pos = aquatic_system.get_water_center().copy()
            else:
                self.pos = np.array([0.0, 0.0])
        
        # === ENERGY ===
        # Swimming costs energy proportional to thrust
        self.energy -= 0.005 + total_thrust * 0.01
        
        return self.alive


class AquaticSystem:
    """
    Manages all aquatic life in the simulation.
    
    Handles spawning, updating, and interaction between
    aquatic plants and creatures.
    """
    
    def __init__(self, world_size: float = 100.0):
        """Initialize aquatic system with PDE water body and collective school field."""
        self.world_size = world_size
        self.plants: List[AquaticPlant] = []
        self.creatures: List[AquaticCreature] = []
        self.poops: List[dict] = []  # Poop/fertilizer deposits
        self.step_count = 0
        
        # Spawning parameters - no hard caps, just soft limits based on resources
        self.max_plants = 100  # Soft limit - can exceed with good conditions
        self.max_creatures = 100  # Soft limit - resource-limited, not hard-capped
        self.plant_spawn_rate = 0.005
        self.creature_spawn_rate = 0.002
        self.poop_decay_time = 200
        
        # === WATER BODY NLSE FIELD ===
        # 2D field representing water dynamics (currents, temperature gradients, etc.)
        self.water_resolution = 32
        self.water_psi = np.zeros((self.water_resolution, self.water_resolution), dtype=complex)
        # Initialize with gentle currents
        x = np.linspace(-np.pi, np.pi, self.water_resolution)
        y = np.linspace(-np.pi, np.pi, self.water_resolution)
        X, Y = np.meshgrid(x, y)
        self.water_psi = 0.3 * np.exp(1j * (0.5 * X + 0.3 * Y))  # Initial flow pattern
        self.water_g = -0.2  # Weak focusing - creates current eddies
        self.water_dt = 0.05
        
        # === SCHOOL TORUS FIELD ===
        # Collective field that fish couple to - emergent schooling through shared dynamics
        self.school_N = 48  # Torus resolution
        self.school_psi = np.zeros(self.school_N, dtype=complex)
        # Initialize with small excitation
        self.school_psi += 0.1 * (np.random.randn(self.school_N) + 1j * np.random.randn(self.school_N))
        self.school_g = -0.4  # Focusing - creates coherent school modes
        self.school_dt = 0.03
        
    def _update_water_field(self):
        """Evolve water body NLSE - creates currents and eddies."""
        psi = self.water_psi
        N = self.water_resolution
        
        # Laplacian via finite differences (periodic boundary)
        laplacian = (
            np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) +
            np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi
        )
        
        # NLSE: i∂ψ/∂t = -∇²ψ + g|ψ|²ψ
        dpsi = 1j * (laplacian + self.water_g * np.abs(psi)**2 * psi)
        
        # Stability
        dpsi = np.clip(dpsi.real, -0.5, 0.5) + 1j * np.clip(dpsi.imag, -0.5, 0.5)
        
        self.water_psi += self.water_dt * dpsi
        self.water_psi *= 0.998  # Gentle damping
        
        # Amplitude limit
        amp = np.abs(self.water_psi)
        mask = amp > 1.5
        if np.any(mask):
            self.water_psi[mask] *= 1.5 / amp[mask]
        
        # Inject energy at boundaries (incoming currents)
        if self.step_count % 20 == 0:
            edge_excite = 0.05 * np.exp(1j * self.step_count * 0.1)
            self.water_psi[0, :] += edge_excite
            self.water_psi[-1, :] += edge_excite * np.exp(1j * np.pi)
    
    def _update_school_torus(self):
        """Evolve collective school field - fish couple to this."""
        psi = self.school_psi
        N = self.school_N
        
        # Laplacian on torus (periodic)
        laplacian = np.roll(psi, 1) + np.roll(psi, -1) - 2 * psi
        
        # NLSE evolution
        dpsi = 1j * (laplacian + self.school_g * np.abs(psi)**2 * psi)
        dpsi = np.clip(dpsi.real, -0.3, 0.3) + 1j * np.clip(dpsi.imag, -0.3, 0.3)
        
        self.school_psi += self.school_dt * dpsi
        self.school_psi *= 0.995  # Damping
        
        # Amplitude limit
        amp = np.abs(self.school_psi)
        if np.max(amp) > 2.0:
            self.school_psi *= 2.0 / np.max(amp)
        
        # Fish inject energy into the school field based on their activity
        for creature in self.creatures:
            if creature.alive:
                # Map creature heading to torus position (with NaN protection)
                if np.isnan(creature.heading) or np.isinf(creature.heading):
                    creature.heading = np.random.uniform(0, 2 * np.pi)
                theta = creature.heading % (2 * np.pi)
                torus_idx = int((theta / (2 * np.pi)) * N) % N
                
                # Activity level based on speed
                vel_mag = np.linalg.norm(creature.vel)
                if not np.isnan(vel_mag):
                    activity = vel_mag * 0.1
                    self.school_psi[torus_idx] += activity * np.exp(1j * theta)
    
    def get_water_current_at(self, pos: np.ndarray) -> np.ndarray:
        """Get water current vector at world position from NLSE field."""
        # Map world position to water grid
        half = self.world_size / 2
        nx = int((pos[0] + half) / self.world_size * self.water_resolution) % self.water_resolution
        ny = int((pos[1] + half) / self.world_size * self.water_resolution) % self.water_resolution
        
        # Current is gradient of phase
        psi = self.water_psi
        # Phase gradient via finite difference
        dx = np.angle(psi[(nx+1) % self.water_resolution, ny]) - np.angle(psi[(nx-1) % self.water_resolution, ny])
        dy = np.angle(psi[nx, (ny+1) % self.water_resolution]) - np.angle(psi[nx, (ny-1) % self.water_resolution])
        
        # Amplitude modulates current strength
        amp = np.abs(psi[nx, ny])
        
        return np.array([dx, dy]) * amp * 0.5
    
    def get_school_bias(self, heading: float) -> Tuple[float, float]:
        """Get directional bias from school torus for a fish at given heading."""
        # Map heading to torus
        theta = heading % (2 * np.pi)
        N = self.school_N
        idx = int((theta / (2 * np.pi)) * N) % N
        
        # Sample neighborhood on torus
        neighborhood = 5
        power = np.zeros(neighborhood * 2 + 1)
        for i, offset in enumerate(range(-neighborhood, neighborhood + 1)):
            power[i] = np.abs(self.school_psi[(idx + offset) % N])**2
        
        # Bias towards higher power regions (where other fish are)
        center_of_mass = np.sum(power * np.arange(-neighborhood, neighborhood + 1)) / (np.sum(power) + 1e-6)
        
        # Convert to angular bias
        angular_bias = center_of_mass * 0.1
        
        # Also get overall school direction from torus
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        weighted_angle = np.sum(np.abs(self.school_psi)**2 * np.exp(1j * angles))
        school_direction = np.angle(weighted_angle)
        
        return angular_bias, school_direction
    
    def get_water_center(self) -> np.ndarray:
        """Get center of water body for creatures to return to."""
        if hasattr(self, 'water_center'):
            return self.water_center
        return np.array([0.0, 0.0])
        
    def initialize(self, terrain: 'Terrain', n_plants: int = 15, n_creatures: int = 12):
        """
        Initialize with starting population.
        
        Args:
            terrain: Terrain for finding water
            n_plants: Initial plant count
            n_creatures: Initial creature count
        """
        # Find water/shore cells
        water_positions = []
        for i in range(terrain.resolution):
            for j in range(terrain.resolution):
                t = terrain.terrain_grid[i, j]
                if t.name in ['water', 'shore']:
                    world_pos = np.array([
                        (i / terrain.resolution) * self.world_size - self.world_size/2,
                        (j / terrain.resolution) * self.world_size - self.world_size/2
                    ])
                    water_positions.append((world_pos, t.name))
        
        if not water_positions:
            print("[Aquatic] No water found in terrain!")
            return
        
        # Spawn plants
        for _ in range(min(n_plants, len(water_positions))):
            pos, terrain_name = water_positions[np.random.randint(len(water_positions))]
            pos = pos + np.random.randn(2) * 2  # Jitter
            
            # Choose plant type based on terrain
            if terrain_name == 'water':
                plant_type = np.random.choice(['kelp', 'algae', 'lily'], p=[0.5, 0.3, 0.2])
            else:  # shore
                plant_type = np.random.choice(['reed', 'lily', 'algae'], p=[0.4, 0.4, 0.2])
            
            self.plants.append(AquaticPlant(pos, plant_type))
        
        # Spawn creatures with species variety - use water OR shore
        aquatic_cells = [p for p, t in water_positions]  # Both water and shore work
        if aquatic_cells:
            # Species distribution: mostly minnows (schooling), some fish, few eels
            species_dist = ['minnow'] * 6 + ['fish'] * 3 + ['eel'] * 1
            for _ in range(n_creatures):
                # Pick a water/shore cell and stay close to it
                base_pos = aquatic_cells[np.random.randint(len(aquatic_cells))]
                pos = base_pos + np.random.randn(2) * 1.0  # Small jitter to stay in water
                species = np.random.choice(species_dist)
                self.creatures.append(AquaticCreature(pos=pos.copy(), species=species))
        else:
            print("[Aquatic] No aquatic terrain found for creatures!")
        
        # Store water positions for boundary enforcement
        self.water_positions = aquatic_cells
        
        # Compute water center for creatures to return to if lost
        if aquatic_cells:
            self.water_center = np.mean(aquatic_cells, axis=0)
        else:
            self.water_center = np.array([0.0, 0.0])
        
        # Count species
        species_counts = {}
        for c in self.creatures:
            species_counts[c.species] = species_counts.get(c.species, 0) + 1
        species_str = ', '.join(f"{k}:{v}" for k, v in species_counts.items())
        
        print(f"[Aquatic] Initialized with {len(self.plants)} plants, {len(self.creatures)} creatures ({species_str})")
    
    def update(self, terrain: 'Terrain') -> List[str]:
        """
        Update all aquatic life.
        
        Args:
            terrain: Terrain for position checks
            
        Returns:
            List of event messages
        """
        messages = []
        self.step_count += 1
        
        # === EVOLVE PDE FIELDS ===
        self._update_water_field()
        self._update_school_torus()
        
        # Update plants
        for plant in self.plants:
            if plant.alive:
                terrain_type = terrain.get_terrain_at(plant.pos)
                plant.update_growth(terrain_type.name)
        
        # Remove dead plants
        self.plants = [p for p in self.plants if p.alive]
        
        # Plant spreading
        if self.step_count % 50 == 0:
            new_plants = []
            for plant in self.plants:
                if plant.can_spread() and len(self.plants) + len(new_plants) < self.max_plants:
                    if np.random.random() < 0.1:
                        # Spread nearby
                        offset = np.random.randn(2) * 3
                        new_pos = plant.pos + offset
                        new_pos = np.clip(new_pos, -WORLD_L/2 + 3, WORLD_L/2 - 3)
                        
                        # Check terrain
                        t = terrain.get_terrain_at(new_pos)
                        if t.name in plant.preferred_terrain:
                            new_plants.append(AquaticPlant(new_pos, plant.plant_type))
            self.plants.extend(new_plants)
        
        # Update creatures - pass system for field coupling
        new_offspring = []
        for creature in self.creatures:
            creature.update(terrain, self.plants, self.creatures, self)
            
            # Handle poop
            if creature.poop_pending and creature.last_poop_amount > 0:
                self.poops.append({
                    'pos': creature.pos.copy(),
                    'amount': creature.last_poop_amount,
                    'birth_step': self.step_count,
                    'species': creature.species
                })
                creature.poop_pending = False
            
            # Handle reproduction
            if creature.ready_to_spawn and len(self.creatures) + len(new_offspring) < self.max_creatures:
                # Find nearby mate of same species
                for other in self.creatures:
                    if (other is not creature and other.alive and 
                        other.species == creature.species and
                        other.energy > 15):
                        dist = np.linalg.norm(other.pos - creature.pos)
                        if dist < 3.0:
                            # Spawn offspring!
                            offspring_pos = (creature.pos + other.pos) / 2 + np.random.randn(2) * 0.5
                            offspring = AquaticCreature(pos=offspring_pos, species=creature.species)
                            new_offspring.append(offspring)
                            
                            # Energy cost
                            creature.energy -= 10
                            other.energy -= 10
                            creature.ready_to_spawn = False
                            creature.spawn_cooldown = 500
                            
                            messages.append(f"[Aquatic] 🐟 New {creature.species} born!")
                            break
        
        self.creatures.extend(new_offspring)
        
        # Process poop decomposition -> fertilizer -> plant growth
        new_poops = []
        for poop in self.poops:
            age = self.step_count - poop['birth_step']
            
            if age < self.poop_decay_time:
                new_poops.append(poop)
                
                # Poop fertilizes nearby plants
                if age > 20:  # After initial decomposition
                    for plant in self.plants:
                        dist = np.linalg.norm(plant.pos - poop['pos'])
                        if dist < 3.0:
                            # Boost plant growth
                            fertilizer_boost = poop['amount'] * 0.001 * (1 - age / self.poop_decay_time)
                            plant.energy += fertilizer_boost
                
                # Poop can spawn new algae (nitrogen cycle!)
                if age > 50 and age % 30 == 0 and np.random.random() < 0.1:
                    if len(self.plants) < self.max_plants:
                        offset = np.random.randn(2) * 1.5
                        new_pos = poop['pos'] + offset
                        t = terrain.get_terrain_at(np.clip(new_pos, -WORLD_L/2 + 3, WORLD_L/2 - 3))
                        if t.name in ['water', 'shore']:
                            self.plants.append(AquaticPlant(new_pos, 'algae'))
        
        self.poops = new_poops
        
        # Remove dead creatures
        self.creatures = [c for c in self.creatures if c.alive]
        
        # Spontaneous spawning in water
        if (self.step_count % 100 == 0 and 
            len(self.plants) < self.max_plants * 0.5 and
            np.random.random() < self.plant_spawn_rate * 10):
            # Find random water position
            for _ in range(10):
                pos = np.random.uniform(-WORLD_L/2 + 5, WORLD_L/2 - 5, 2)
                t = terrain.get_terrain_at(pos)
                if t.name in ['water', 'shore']:
                    plant_type = np.random.choice(['algae', 'lily', 'kelp'])
                    self.plants.append(AquaticPlant(pos, plant_type))
                    break
        
        return messages
    
    def get_harvestable_objects(self) -> List[WorldObject]:
        """Get list of plants as WorldObjects for creature interaction."""
        return [p for p in self.plants if p.alive]
    
    def get_stats(self) -> dict:
        """Get aquatic system statistics."""
        plant_counts = {}
        for p in self.plants:
            plant_counts[p.plant_type] = plant_counts.get(p.plant_type, 0) + 1
            
        return {
            'total_plants': len(self.plants),
            'plant_types': plant_counts,
            'total_creatures': len(self.creatures),
            'total_energy': sum(p.energy for p in self.plants),
        }
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'plants': [
                {
                    'pos': p.pos.tolist(),
                    'type': p.plant_type,
                    'energy': p.energy,
                    'age': p.age,
                }
                for p in self.plants
            ],
            'creatures': [
                {
                    'pos': c.pos.tolist(),
                    'vel': c.vel.tolist(),
                    'energy': c.energy,
                    'age': c.age,
                }
                for c in self.creatures
            ],
            'step_count': self.step_count,
        }
    
    @classmethod
    def from_dict(cls, d: dict, world_size: float) -> 'AquaticSystem':
        """Deserialize from persistence."""
        system = cls(world_size)
        
        for p_data in d.get('plants', []):
            plant = AquaticPlant(np.array(p_data['pos']), p_data['type'])
            plant.energy = p_data['energy']
            plant.age = p_data['age']
            system.plants.append(plant)
        
        for c_data in d.get('creatures', []):
            creature = AquaticCreature(
                pos=np.array(c_data['pos']),
                vel=np.array(c_data['vel']),
                energy=c_data['energy'],
                age=c_data['age']
            )
            system.creatures.append(creature)
        
        system.step_count = d.get('step_count', 0)
        return system
