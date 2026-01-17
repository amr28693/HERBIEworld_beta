"""
Herbie Hands - KdV-driven gripper appendages for manipulation.

Herbies have two gripper limbs that can:
- Pick up and hold objects
- Use objects as tools
- Coordinate for building/combat
- Learn grip timing through experience

The grip reflex emerges from neural dynamics - KdV soliton pulses
travel down the gripper, triggering grip when they reach the tip.
"""

from typing import List, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..world.objects import WorldObject


class GrippableProperties:
    """Properties of an object that can be gripped."""
    def __init__(
        self,
        grip_difficulty: float = 0.5,
        weight: float = 1.0,
        tool_damage: float = 0.0,
        tool_type: str = "none",
        stackable: bool = False
    ):
        self.grip_difficulty = grip_difficulty  # 0=easy, 1=hard
        self.weight = weight  # Affects carry speed
        self.tool_damage = tool_damage  # Damage when used as weapon
        self.tool_type = tool_type  # "blunt", "sharp", "material"
        self.stackable = stackable  # Can be stacked/built with


class GripperLimb:
    """
    A KdV-driven manipulator appendage for Herbie.
    
    The grip timing emerges from neural dynamics:
    - KdV soliton pulse travels down the limb
    - When pulse reaches tip with sufficient amplitude → grip reflex triggers
    - Success depends on timing, object proximity, approach angle
    
    This is learnable: well-timed pulses = successful grips.
    """
    
    def __init__(self, name: str, side: str = 'left'):
        self.name = name
        self.side = side
        
        # KdV field (shorter than locomotion limbs)
        self.N_grip = 32
        self.x_grip = np.linspace(0, 1, self.N_grip)
        self.dx = self.x_grip[1] - self.x_grip[0]
        
        # KdV state
        self.u = np.zeros(self.N_grip)
        self.u_prev = np.zeros(self.N_grip)
        
        # Parameters
        self.g = 0.3
        self.dispersion = 0.02
        self.damping = 0.95
        
        # Grip state
        self.grip_state = 0.0  # 0=open, 1=closed
        self.grip_target = 0.0
        self.grip_speed = 0.1
        self.grip_strength = 0.5  # Learnable
        
        # Held object
        self.held_object: Optional['WorldObject'] = None
        self.held_object_id: Optional[int] = None
        self.hold_duration = 0
        
        # Position
        self.base_angle = np.pi/3 if side == 'left' else -np.pi/3
        self.angle = self.base_angle
        self.length = 3.5
        self.angular_velocity = 0.0
        
        # Learning
        self.grip_successes = 0
        self.grip_attempts = 0
        self.learned_timing_offset = 0.0
        
        # Pulse tracking
        self.pulse_position = 0.0
        self.pulse_amplitude = 0.0
        self.tip_activation = 0.0
    
    def inject_efferent(self, amplitude: float, phase: float = 0.0):
        """Inject motor signal from torus brain."""
        inject_width = 3
        for i in range(inject_width):
            self.u[i] += amplitude * np.exp(-i/2) * np.cos(phase)
    
    def evolve(self, dt: float = 0.015):
        """Evolve KdV dynamics in the gripper."""
        u = self.u.copy()
        
        # KdV: u_t + u*u_x + dispersion*u_xxx = 0
        u_x = np.zeros_like(u)
        u_xxx = np.zeros_like(u)
        
        u_x[1:-1] = (u[1:-1] - u[:-2]) / self.dx
        u_xxx[2:-2] = (u[4:] - 2*u[3:-1] + 2*u[1:-3] - u[:-4]) / (2 * self.dx**3)
        
        du = -self.g * u * u_x - self.dispersion * u_xxx
        self.u = u + du * dt * 10
        self.u = np.clip(self.u, -10.0, 10.0)
        self.u *= self.damping
        
        # Boundary conditions
        self.u[0] *= 0.9
        self.u[-1] *= 0.8
        
        # Track pulse
        if np.max(np.abs(self.u)) > 0.01:
            weighted_pos = np.sum(self.x_grip * np.abs(self.u)) / (np.sum(np.abs(self.u)) + 1e-6)
            self.pulse_position = weighted_pos
            self.pulse_amplitude = np.max(np.abs(self.u))
        else:
            self.pulse_amplitude *= 0.9
        
        self.tip_activation = min(5.0, np.mean(np.abs(self.u[-5:])))
        
        # Update grip state
        if self.grip_state < self.grip_target:
            self.grip_state = min(self.grip_target, self.grip_state + self.grip_speed)
        elif self.grip_state > self.grip_target:
            self.grip_state = max(self.grip_target, self.grip_state - self.grip_speed)
    
    def get_tip_position(self, body_pos: np.ndarray) -> np.ndarray:
        """Get world position of gripper tip."""
        return body_pos + self.length * np.array([np.cos(self.angle), np.sin(self.angle)])
    
    def attempt_grip(self, body_pos: np.ndarray, objects: List,
                     world_object_map: dict = None) -> bool:
        """
        Attempt to grip a nearby object.
        
        Success depends on:
        - Tip activation (KdV pulse at tip)
        - Object proximity
        - Grip timing (learned)
        """
        if self.held_object is not None:
            return False
        
        self.grip_attempts += 1
        tip_pos = self.get_tip_position(body_pos)
        
        # Check activation threshold
        activation_threshold = 0.08 - self.learned_timing_offset * 0.03
        if self.tip_activation < activation_threshold:
            return False
        
        # Find grippable objects
        grip_radius = 2.5
        best_obj = None
        best_dist = grip_radius
        best_idx = None
        
        for idx, obj in enumerate(objects):
            if not obj.alive or obj.compliance < 0.25:  # Elements have 0.3 compliance
                continue
            
            dist = np.linalg.norm(obj.pos - tip_pos)
            if dist < best_dist:
                grip_props = getattr(obj, 'grip_props', GrippableProperties())
                success_prob = (1 - grip_props.grip_difficulty) * self.grip_strength
                success_prob *= self.tip_activation * 2
                
                if np.random.random() < success_prob:
                    best_obj = obj
                    best_dist = dist
                    best_idx = idx
        
        if best_obj is not None:
            self.held_object = best_obj
            self.held_object_id = best_idx
            self.grip_target = 1.0
            self.hold_duration = 0
            self.grip_successes += 1
            
            # Learning
            self.learned_timing_offset += 0.01
            self.learned_timing_offset = np.clip(self.learned_timing_offset, -0.5, 0.5)
            self.grip_strength = min(1.0, self.grip_strength + 0.005)
            
            return True
        
        self.learned_timing_offset -= 0.002
        return False
    
    def release(self, body_vel: np.ndarray = None) -> Tuple[Optional['WorldObject'], np.ndarray]:
        """Release held object, returning (object, release_velocity)."""
        if self.held_object is None:
            return None, np.zeros(2)
        
        obj = self.held_object
        release_vel = np.zeros(2)
        
        if body_vel is not None:
            release_vel += body_vel * 0.5
        
        # Add angular momentum
        tip_vel = self.angular_velocity * self.length
        tangent = np.array([-np.sin(self.angle), np.cos(self.angle)])
        release_vel += tangent * tip_vel * 0.3
        
        self.held_object = None
        self.held_object_id = None
        self.grip_target = 0.0
        self.hold_duration = 0
        
        return obj, release_vel
    
    def update_held_object_position(self, body_pos: np.ndarray):
        """Keep held object at gripper tip."""
        if self.held_object is not None:
            self.held_object.pos = self.get_tip_position(body_pos)
            self.hold_duration += 1
    
    def use_as_tool(self, body_pos: np.ndarray, target_pos: np.ndarray,
                    target_creature=None) -> float:
        """Use held object as a tool. Returns damage/effect strength."""
        if self.held_object is None:
            return 0.0
        
        grip_props = getattr(self.held_object, 'grip_props', GrippableProperties())
        
        tip_pos = self.get_tip_position(body_pos)
        dist = np.linalg.norm(target_pos - tip_pos)
        
        if dist > 2.5:
            return 0.0
        
        base_damage = grip_props.tool_damage
        swing_factor = 1.0 + abs(self.angular_velocity) * 0.5
        timing_factor = 1.0 + self.tip_activation * 0.5
        
        damage = base_damage * swing_factor * timing_factor
        
        if target_creature is not None and damage > 0:
            target_creature.metabolism.hunger = min(1.0,
                target_creature.metabolism.hunger + damage * 0.1)
            if hasattr(target_creature, 'afferent') and 'env_pain' in target_creature.afferent:
                target_creature.afferent['env_pain'].nucleate(damage, 0.0)
        
        return damage
    
    def get_state(self) -> dict:
        """Get gripper state for persistence/visualization."""
        return {
            'grip_state': self.grip_state,
            'angle': self.angle,
            'tip_activation': self.tip_activation,
            'holding': self.held_object is not None,
            'grip_strength': self.grip_strength,
            'successes': self.grip_successes,
            'attempts': self.grip_attempts,
        }


class HerbieHands:
    """
    Manages Herbie's two gripper appendages.
    
    Left and right hands can work independently or together.
    Coordinated actions (two-handed grip, building) emerge from
    synchronized KdV pulses from the torus brain.
    """
    
    def __init__(self):
        self.left = GripperLimb("left_hand", "left")
        self.right = GripperLimb("right_hand", "right")
        
        # Coordination
        self.coordinated_mode = False
        self.coordination_phase = 0.0
        
        # Building/combat modes
        self.build_mode = False
        self.build_target_pos: Optional[np.ndarray] = None
        self.combat_mode = False
        
        # Learning
        self.tool_use_successes = 0
        self.build_successes = 0
    
    def inject_efferent_bilateral(self, amplitude: float, phase: float,
                                   left_bias: float = 0.0):
        """
        Inject motor signals to both hands.
        left_bias: -1 = left only, 0 = both equal, 1 = right only
        """
        left_amp = amplitude * (1 - max(0, left_bias))
        right_amp = amplitude * (1 + min(0, left_bias))
        
        self.left.inject_efferent(left_amp, phase)
        self.right.inject_efferent(right_amp, phase + self.coordination_phase)
    
    def evolve(self, body_pos: np.ndarray, dt: float = 0.015):
        """Update both grippers."""
        self.left.evolve(dt)
        self.right.evolve(dt)
        
        self.left.update_held_object_position(body_pos)
        self.right.update_held_object_position(body_pos)
    
    def attempt_grip_nearest(self, body_pos: np.ndarray, objects: List,
                             prefer_hand: str = None, herbie=None) -> Tuple[bool, str]:
        """
        Try to grip with the best available hand.
        
        If herbie is provided, uses wavefunction resonance to weight selection.
        Returns (success, which_hand).
        """
        left_tip = self.left.get_tip_position(body_pos)
        right_tip = self.right.get_tip_position(body_pos)
        
        # Find nearest object (with resonance weighting if available)
        best_obj = None
        best_score = -999.0
        best_pos = None
        
        for obj in objects:
            if not obj.alive or obj.compliance < 0.25:  # Elements have 0.3 compliance
                continue
            
            dist_left = np.linalg.norm(obj.pos - left_tip)
            dist_right = np.linalg.norm(obj.pos - right_tip)
            min_dist = min(dist_left, dist_right)
            
            if min_dist > 3.0:
                continue
            
            base_score = 3.0 - min_dist
            
            if herbie is not None:
                try:
                    from ..chemistry.elements import compute_object_resonance
                    resonance = compute_object_resonance(herbie, obj)
                    
                    if resonance < -0.5 and np.random.random() < abs(resonance):
                        continue
                    
                    score = base_score + resonance * 2.0
                except ImportError:
                    score = base_score
            else:
                score = base_score
            
            if score > best_score:
                best_obj = obj
                best_score = score
                best_pos = obj.pos
        
        if best_obj is None:
            return False, ""
        
        # Determine which hand
        if prefer_hand == 'left' and self.left.held_object is None:
            use_hand = self.left
            hand_name = 'left'
        elif prefer_hand == 'right' and self.right.held_object is None:
            use_hand = self.right
            hand_name = 'right'
        else:
            dist_left = np.linalg.norm(best_pos - left_tip)
            dist_right = np.linalg.norm(best_pos - right_tip)
            
            if dist_left < dist_right and self.left.held_object is None:
                use_hand = self.left
                hand_name = 'left'
            elif self.right.held_object is None:
                use_hand = self.right
                hand_name = 'right'
            elif self.left.held_object is None:
                use_hand = self.left
                hand_name = 'left'
            else:
                return False, ""
        
        success = use_hand.attempt_grip(body_pos, objects)
        return success, hand_name if success else ""
    
    def release_all(self, body_vel: np.ndarray = None) -> List[Tuple['WorldObject', np.ndarray]]:
        """Release both hands, return list of (object, velocity)."""
        released = []
        
        obj_l, vel_l = self.left.release(body_vel)
        if obj_l is not None:
            released.append((obj_l, vel_l))
        
        obj_r, vel_r = self.right.release(body_vel)
        if obj_r is not None:
            released.append((obj_r, vel_r))
        
        return released
    
    def is_holding_anything(self) -> bool:
        return self.left.held_object is not None or self.right.held_object is not None
    
    def get_held_objects(self) -> List['WorldObject']:
        """Return list of currently held objects."""
        held = []
        if self.left.held_object is not None:
            held.append(self.left.held_object)
        if self.right.held_object is not None:
            held.append(self.right.held_object)
        return held
    
    def get_total_weight(self) -> float:
        """Get total weight of held objects."""
        weight = 0.0
        for obj in self.get_held_objects():
            grip_props = getattr(obj, 'grip_props', GrippableProperties())
            weight += grip_props.weight
        return weight
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'left_grip_strength': self.left.grip_strength,
            'left_timing_offset': self.left.learned_timing_offset,
            'left_successes': self.left.grip_successes,
            'right_grip_strength': self.right.grip_strength,
            'right_timing_offset': self.right.learned_timing_offset,
            'right_successes': self.right.grip_successes,
            'tool_use_successes': self.tool_use_successes,
            'build_successes': self.build_successes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'HerbieHands':
        """Deserialize from persistence."""
        hands = cls()
        hands.left.grip_strength = data.get('left_grip_strength', 0.5)
        hands.left.learned_timing_offset = data.get('left_timing_offset', 0.0)
        hands.left.grip_successes = data.get('left_successes', 0)
        hands.right.grip_strength = data.get('right_grip_strength', 0.5)
        hands.right.learned_timing_offset = data.get('right_timing_offset', 0.0)
        hands.right.grip_successes = data.get('right_successes', 0)
        hands.tool_use_successes = data.get('tool_use_successes', 0)
        hands.build_successes = data.get('build_successes', 0)
        return hands


def add_grip_properties_to_objects(objects: List['WorldObject']):
    """Add grippable properties to world objects based on their type."""
    for obj in objects:
        if obj.compliance > 0.7:
            obj.grip_props = GrippableProperties(
                grip_difficulty=0.2,
                weight=0.5 + obj.size * 0.3,
                tool_damage=0.05,
                tool_type="soft",
                stackable=False
            )
        elif obj.compliance > 0.4:
            obj.grip_props = GrippableProperties(
                grip_difficulty=0.4,
                weight=1.0 + obj.size * 0.5,
                tool_damage=0.15,
                tool_type="blunt",
                stackable=True
            )
        else:
            obj.grip_props = GrippableProperties(
                grip_difficulty=0.7,
                weight=2.0 + obj.size,
                tool_damage=0.3,
                tool_type="blunt",
                stackable=True
            )
