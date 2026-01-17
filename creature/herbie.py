"""
Herbie - The protagonist creature with hands and sexual reproduction.

Herbies are the most cognitively complex species with:
- Gripper hands for manipulation
- Sexual reproduction with resonance-based pair bonding
- Names and moods
- Family cohesion and parental care
- Placement memory for caching behavior
"""

from typing import List, Optional, TYPE_CHECKING
import numpy as np

from ..core.constants import WORLD_L, G_SMOOTH, dt
from ..events.logger import event_log
from ..brain.placement_memory import PlacementMemoryField, PlacementMemoryParams
from .creature import Creature
from .species import SPECIES_HERBIE
from .traits import MutatedTraits
from .herbie_hands import HerbieHands
from .herbie_social import (
    HerbieSex, HERBIE_SEX_COLORS,
    HerbieGenome, HerbieMatingState, HerbieNameGenerator,
    HERBIE_PREGNANCY_SPEED_PENALTY, HERBIE_PREGNANCY_HUNGER_MULT,
    HERBIE_JUVENILE_SCALE, HERBIE_JUVENILE_DURATION,
    HERBIE_DEFENSE_BONUS_FAMILY, HERBIE_FAMILY_COHESION_RANGE,
    HERBIE_JUVENILE_FEED_RANGE, HERBIE_PARENTAL_FEED_RATE,
    HERBIE_GESTATION_DURATION, HERBIE_COURTSHIP_DISTANCE,
    check_resonance_and_bond
)

if TYPE_CHECKING:
    from ..world.seasons import Season


# Audio sensitivity multiplier for Herbies
HERBIE_AUDIO_MULTIPLIER = 2.5


class HerbieMood:
    """Calculate and display Herbie's emotional state."""
    
    MOODS = {
        'starving': {'icon': '😰', 'color': 'red', 'desc': 'desperately hungry'},
        'hungry': {'icon': '😟', 'color': 'orange', 'desc': 'quite hungry'},
        'stressed': {'icon': '😓', 'color': 'yellow', 'desc': 'under pressure'},
        'content': {'icon': '😊', 'color': 'lightgreen', 'desc': 'doing well'},
        'happy': {'icon': '😄', 'color': 'cyan', 'desc': 'very happy'},
        'blissful': {'icon': '🥰', 'color': 'pink', 'desc': 'in heaven'},
        'sleepy': {'icon': '😴', 'color': 'lavender', 'desc': 'drowsy'},
        'curious': {'icon': '🤔', 'color': 'lightblue', 'desc': 'exploring'},
        'neutral': {'icon': '😐', 'color': 'white', 'desc': 'neutral'},
    }
    
    @classmethod
    def calculate_mood(cls, herbie) -> str:
        """Determine mood from internal state."""
        hunger = herbie.metabolism.hunger
        arousal = herbie.torus.get_arousal() if hasattr(herbie.torus, 'get_arousal') else 0.5
        dream = herbie.dream_depth
        
        if dream > 0.3:
            return 'sleepy'
        if hunger > 0.8:
            return 'starving'
        if hunger > 0.6:
            return 'hungry'
        if hunger > 0.4:
            return 'stressed'
        if arousal > 0.7 and hunger < 0.3:
            return 'blissful'
        if arousal > 0.5 and hunger < 0.4:
            return 'happy'
        if arousal > 0.3:
            return 'curious'
        if hunger < 0.2:
            return 'content'
        return 'neutral'
    
    @classmethod
    def get_mood_info(cls, mood: str) -> dict:
        return cls.MOODS.get(mood, cls.MOODS['neutral'])


def get_herbie_display_color(herbie) -> str:
    """Get display color based on sex and state."""
    base_color = HERBIE_SEX_COLORS.get(herbie.mating_state.sex, '#FFFFFF')
    
    if herbie.mating_state.is_pregnant:
        return '#FFB6C1'  # Light pink when pregnant
    if herbie.mating_state.is_juvenile:
        return '#ADD8E6'  # Light blue for juveniles
    
    return base_color


# =============================================================================
# MATING HELPER FUNCTIONS
# =============================================================================

def find_potential_mate(herbie, all_herbies: List) -> Optional['HerbieWithHands']:
    """Find a compatible mate based on sex and bonding status."""
    ms = herbie.mating_state
    
    if ms.sex == HerbieSex.CARRIER:
        if ms.is_bonded and ms.mate_id:
            for other in all_herbies:
                if other.creature_id == ms.mate_id and other.alive:
                    return other
            ms.is_bonded = False
            ms.mate_id = None
            return None
        target_sex = HerbieSex.PROVIDER
    else:
        if ms.mate_ids:
            for mate_id in ms.mate_ids:
                for other in all_herbies:
                    if other.creature_id == mate_id and other.alive:
                        if not other.mating_state.is_pregnant:
                            return other
        target_sex = HerbieSex.CARRIER
    
    # Find unbonded potential mate
    best_candidate = None
    best_dist = float('inf')
    
    for other in all_herbies:
        if other.creature_id == herbie.creature_id:
            continue
        if other.mating_state.sex != target_sex:
            continue
        if ms.sex == HerbieSex.CARRIER and other.mating_state.sex == HerbieSex.PROVIDER:
            pass  # Carriers can approach any provider
        elif ms.sex == HerbieSex.PROVIDER:
            if other.mating_state.is_bonded and other.mating_state.mate_id != herbie.creature_id:
                continue  # Can't court already-bonded carriers
        
        dist = np.linalg.norm(other.pos - herbie.pos)
        if dist < best_dist:
            best_dist = dist
            best_candidate = other
    
    return best_candidate


def process_courtship(herbie, target) -> bool:
    """Process courtship, return True if mating should occur."""
    dist = np.linalg.norm(target.pos - herbie.pos)
    
    if dist > HERBIE_COURTSHIP_DISTANCE:
        herbie.mating_state.courtship_progress = max(0, herbie.mating_state.courtship_progress - 1)
        return False
    
    # Check resonance if not already bonded
    if not herbie.mating_state.is_bonded:
        if herbie.mating_state.can_check_resonance(target.creature_id, herbie.step_count):
            should_bond, resonance = check_resonance_and_bond(herbie, target, herbie.step_count)
            if should_bond:
                herbie.mating_state.add_mate(target.creature_id, resonance)
                target.mating_state.add_mate(herbie.creature_id, resonance)
                
                name1 = herbie.mating_state.name or herbie.creature_id[:8]
                name2 = target.mating_state.name or target.creature_id[:8]
                print(f"[Herbie] ❤️ {name1} and {name2} bonded! (resonance={resonance:.2f})")
                
                # Record in world history
                from ..events.world_history import world_history
                world_history().record_bond(herbie.step_count, name1, name2, resonance)
    
    herbie.mating_state.courtship_progress += 1
    
    # Check if courtship complete
    from .herbie_social import HERBIE_COURTSHIP_DURATION
    return herbie.mating_state.courtship_progress >= HERBIE_COURTSHIP_DURATION


def execute_mating(provider, carrier, step_count: int):
    """Execute mating between provider and carrier."""
    if carrier.mating_state.is_pregnant:
        return
    
    # Create offspring genome
    offspring_genome = HerbieGenome.mix_and_mutate(
        provider.mating_state.genome,
        carrier.mating_state.genome
    )
    
    # Create embryo only if embryo mode is 'herbie' or 'all'
    import os
    embryo_mode = os.environ.get('HERBIE_EMBRYO_MODE', 'herbie')
    embryo = None
    if embryo_mode in ('herbie', 'all'):
        from .embryo import create_embryo_from_parents
        embryo = create_embryo_from_parents(provider, carrier)
    
    carrier.mating_state.is_pregnant = True
    carrier.mating_state.gestation_progress = 0
    carrier.mating_state.offspring_genome = offspring_genome
    carrier.mating_state.embryo = embryo  # Store embryo for development (may be None)
    carrier.mating_state.provider_id = provider.creature_id
    carrier.mating_state.last_mating_step = step_count
    provider.mating_state.last_mating_step = step_count
    
    print(f"[Herbie] 💕 {provider.mating_state.name or provider.creature_id[:8]} mated with "
          f"{carrier.mating_state.name or carrier.creature_id[:8]}!")
    
    event_log().log_birth(
        step=step_count,
        name="(conception)",
        sex="unknown",
        generation=max(provider.generation, carrier.generation) + 1,
        parents=(carrier.mating_state.name, provider.mating_state.name),
        pos=tuple(carrier.pos)
    )


def process_pregnancy(herbie) -> Optional[dict]:
    """Process pregnancy with embryo development, return birth data if ready."""
    if not herbie.mating_state.is_pregnant:
        return None
    
    herbie.mating_state.gestation_progress += 1
    
    # Develop embryo if present
    embryo = getattr(herbie.mating_state, 'embryo', None)
    if embryo:
        from .embryo import develop_embryo_during_gestation
        
        # Get maternal state
        mother_hunger = herbie.metabolism.hunger
        mother_stress = 0.0
        
        # Check if mother is being threatened
        if hasattr(herbie, 'afferent') and 'threat' in herbie.afferent:
            threat_channel = herbie.afferent['threat']
            if hasattr(threat_channel, 'u'):
                mother_stress = min(1.0, np.sum(np.abs(threat_channel.u)**2))
        
        # Develop embryo (5 dev steps per gestation step)
        events = develop_embryo_during_gestation(
            embryo, mother_hunger, mother_stress, steps=5
        )
        
        # Log significant developmental events
        for event in events:
            if event.event_type == 'stage_transition':
                print(f"[Embryo] 🧬 {herbie.mating_state.name}'s embryo: {event.details.get('to', '?')}")
            elif event.event_type == 'stress_perturbation':
                print(f"[Embryo] ⚠️ Maternal stress affecting {herbie.mating_state.name}'s embryo")
    
    if herbie.mating_state.gestation_progress >= HERBIE_GESTATION_DURATION:
        # Give birth!
        herbie.mating_state.is_pregnant = False
        
        # Determine baby sex
        baby_sex = HerbieSex.PROVIDER if np.random.random() < 0.5 else HerbieSex.CARRIER
        baby_name = HerbieNameGenerator.generate(baby_sex)
        
        # Get developmental trait modifiers
        dev_traits = {}
        dev_summary = None
        if embryo:
            dev_traits = embryo.get_final_traits()
            dev_summary = embryo.get_development_summary()
            print(f"[Birth] 🍼 {baby_name} born! Basins:{dev_summary['n_basins']}, "
                  f"Symmetry:{dev_summary['bilateral_symmetry']:.2f}")
        
        birth_data = {
            'pos': herbie.pos + np.random.randn(2) * 2,
            'sex': baby_sex,
            'genome': herbie.mating_state.offspring_genome,
            'mother_id': herbie.creature_id,
            'father_id': herbie.mating_state.provider_id,
            'baby_name': baby_name,
            'generation': herbie.generation + 1,
            'developmental_traits': dev_traits,
            'dev_summary': dev_summary,
        }
        
        herbie.mating_state.offspring_genome = None
        herbie.mating_state.provider_id = None
        herbie.mating_state.embryo = None
        
        return birth_data
    
    return None


def process_juvenile_maturation(herbie):
    """Process juvenile growth toward adulthood."""
    if not herbie.mating_state.is_juvenile:
        return
    
    herbie.mating_state.maturity += 1.0 / HERBIE_JUVENILE_DURATION
    
    if herbie.mating_state.maturity >= 1.0:
        herbie.mating_state.is_juvenile = False
        herbie.mating_state.maturity = 1.0
        print(f"[Herbie] 🎉 {herbie.mating_state.name or herbie.creature_id[:8]} reached adulthood!")


def process_heat_behavior(herbie, all_herbies, season) -> np.ndarray:
    """Process heat/seeking behavior, return direction vector."""
    direction = np.zeros(2)
    ms = herbie.mating_state
    
    if ms.is_juvenile or ms.is_pregnant:
        return direction
    
    # Check if should enter heat
    if not ms.in_heat:
        if herbie.step_count - ms.last_heat_step > ms.heat_cooldown:
            if herbie.metabolism.hunger < 0.4:
                ms.in_heat = True
                ms.heat_start_step = herbie.step_count
    
    # Process heat
    if ms.in_heat:
        if herbie.step_count - ms.heat_start_step > ms.heat_duration:
            ms.in_heat = False
            ms.last_heat_step = herbie.step_count
        else:
            # Seek opposite sex
            target_sex = HerbieSex.CARRIER if ms.sex == HerbieSex.PROVIDER else HerbieSex.PROVIDER
            for other in all_herbies:
                if other.mating_state.sex == target_sex and other.alive:
                    to_other = other.pos - herbie.pos
                    dist = np.linalg.norm(to_other)
                    if 0.1 < dist < 30:
                        direction += (to_other / dist) * (0.5 / (dist + 1))
    
    return direction


def provider_seek_family(herbie, all_herbies) -> np.ndarray:
    """Provider seeks pregnant mates and offspring."""
    direction = np.zeros(2)
    
    if herbie.mating_state.sex != HerbieSex.PROVIDER:
        return direction
    
    # Seek pregnant mates
    for mate_id in herbie.mating_state.mate_ids:
        for other in all_herbies:
            if other.creature_id == mate_id and other.alive:
                if other.mating_state.is_pregnant:
                    to_mate = other.pos - herbie.pos
                    dist = np.linalg.norm(to_mate)
                    if dist > HERBIE_FAMILY_COHESION_RANGE * 0.5:
                        direction += (to_mate / (dist + 0.1)) * 0.5
    
    # Seek offspring
    for offspring_id in herbie.mating_state.offspring_ids:
        for other in all_herbies:
            if other.creature_id == offspring_id and other.alive:
                if other.mating_state.is_juvenile:
                    to_child = other.pos - herbie.pos
                    dist = np.linalg.norm(to_child)
                    if dist > HERBIE_FAMILY_COHESION_RANGE * 0.3:
                        direction += (to_child / (dist + 0.1)) * 0.3
    
    return direction


def provider_feed_family(herbie, all_herbies):
    """Provider shares food with family."""
    if herbie.mating_state.sex != HerbieSex.PROVIDER:
        return
    if herbie.metabolism.hunger > 0.3:
        return  # Too hungry to share
    
    # Find hungry family members
    for mate_id in herbie.mating_state.mate_ids:
        for other in all_herbies:
            if other.creature_id == mate_id and other.alive:
                dist = np.linalg.norm(other.pos - herbie.pos)
                if dist < HERBIE_JUVENILE_FEED_RANGE:
                    if other.metabolism.hunger > 0.4:
                        transfer = HERBIE_PARENTAL_FEED_RATE * 0.3
                        other.metabolism.hunger = max(0, other.metabolism.hunger - transfer)
                        herbie.metabolism.hunger += transfer * 0.5


def get_family_defense_bonus(herbie, all_herbies) -> float:
    """Get defense bonus from nearby family members."""
    bonus = 0.0
    
    for other in all_herbies:
        if other.creature_id == herbie.creature_id:
            continue
        
        dist = np.linalg.norm(other.pos - herbie.pos)
        if dist > HERBIE_FAMILY_COHESION_RANGE:
            continue
        
        # Check if family
        is_family = False
        if herbie.mating_state.mate_id == other.creature_id:
            is_family = True
        if other.creature_id in herbie.mating_state.mate_ids:
            is_family = True
        if other.creature_id in herbie.mating_state.offspring_ids:
            is_family = True
        if herbie.creature_id in other.mating_state.offspring_ids:
            is_family = True
        
        if is_family:
            proximity_factor = 1.0 - (dist / HERBIE_FAMILY_COHESION_RANGE)
            bonus += HERBIE_DEFENSE_BONUS_FAMILY * proximity_factor
    
    return min(bonus, HERBIE_DEFENSE_BONUS_FAMILY * 2)


# =============================================================================
# MAIN HERBIE CLASS
# =============================================================================

class HerbieWithHands(Creature):
    """
    Full Herbie with hands and sexual reproduction.
    
    This is the creature class used at runtime for all Herbies.
    Combines:
    - Base creature NLSE dynamics
    - Gripper hands for manipulation
    - Sexual reproduction with mating states
    - Placement memory for caching behavior
    - Enhanced audio sensitivity
    """
    
    def __init__(self, *args, sex: HerbieSex = None, genome: HerbieGenome = None,
                 mating_state: HerbieMatingState = None,
                 inherited_hands: dict = None,
                 placement_memory_params: PlacementMemoryParams = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Hands
        if inherited_hands:
            self.hands = HerbieHands.from_dict(inherited_hands)
        else:
            self.hands = HerbieHands()
        
        # Mating state
        if mating_state:
            self.mating_state = mating_state
        else:
            self.mating_state = HerbieMatingState()
            if sex:
                self.mating_state.sex = sex
            if genome:
                self.mating_state.genome = genome
            else:
                self.mating_state.genome = HerbieGenome.random()
        
        # Placement memory
        if placement_memory_params:
            self.placement_memory = PlacementMemoryField(placement_memory_params)
        else:
            self.placement_memory = PlacementMemoryField(PlacementMemoryParams.random())
        
        # Apply genome
        self._apply_genome()
        
        # Audio sensitivity
        self.audio_sensitivity = self.mating_state.genome.audio_sensitivity * HERBIE_AUDIO_MULTIPLIER
        
        # Mood tracking
        self.current_mood = 'neutral'
        self.mood_history = []
        
        # Terrain reference
        self.terrain = None
        
        # Pending birth (set by process_pregnancy)
        self._pending_birth = None
    
    def _apply_genome(self):
        """Apply genetic traits to creature properties."""
        g = self.mating_state.genome
        self._genome_speed_mult = g.speed_modifier
        self._genome_efficiency_mult = g.energy_efficiency
        self._genome_scale = g.body_scale
        self._aggression = g.aggression
        self._curiosity = g.curiosity
        self._sociability = g.sociability
        
        if self.hands:
            self.hands.left.grip_strength = g.grip_strength
            self.hands.right.grip_strength = g.grip_strength
    
    def get_effective_speed(self, season=None) -> float:
        """Get speed with genome and pregnancy modifiers."""
        base = getattr(self, '_genome_speed_mult', 1.0)
        if self.mating_state.is_pregnant:
            base *= HERBIE_PREGNANCY_SPEED_PENALTY
        if self.mating_state.is_juvenile:
            base *= 0.8 + 0.2 * self.mating_state.maturity
        return base
    
    def get_body_scale(self) -> float:
        """Get body scale with genome and juvenile modifier."""
        base = getattr(self, '_genome_scale', 1.0)
        if self.mating_state.is_juvenile:
            base *= HERBIE_JUVENILE_SCALE + (1 - HERBIE_JUVENILE_SCALE) * self.mating_state.maturity
        return base
    
    def get_display_color(self) -> str:
        """Get display color based on sex and state."""
        return get_herbie_display_color(self)
    
    def step(self, all_creatures: List, audio_amp: float = 0.0,
             silence_frames: int = 100, audio_system=None, season: 'Season' = None):
        """Full Herbie step with hands and sexual reproduction."""
        
        if not self.alive:
            return 'dead'
        
        # Get all Herbies for social behaviors
        all_herbies = [c for c in all_creatures 
                      if c.species.name == "Herbie" and c.alive and hasattr(c, 'mating_state')]
        
        # Boost audio for Herbie
        boosted_audio = audio_amp * self.audio_sensitivity
        
        # Juvenile maturation
        if self.mating_state.is_juvenile:
            process_juvenile_maturation(self)
        
        # Heat behavior
        heat_direction = process_heat_behavior(self, all_herbies, season)
        if np.linalg.norm(heat_direction) > 0.01:
            self.vel += heat_direction * 0.08
        
        # Mating behavior
        if not self.mating_state.is_juvenile and self.mating_state.can_mate(
                self.step_count, self.metabolism.hunger, self.step_count):
            
            if self.mating_state.sex == HerbieSex.PROVIDER:
                potential_mate = find_potential_mate(self, all_herbies)
                
                if potential_mate:
                    self.mating_state.courtship_target = potential_mate.creature_id
                    
                    to_mate = potential_mate.pos - self.pos
                    dist = np.linalg.norm(to_mate)
                    if dist > 1:
                        seek_strength = 0.3 * getattr(self, '_sociability', 0.5)
                        self.vel += (to_mate / dist) * seek_strength * 0.1
                    
                    if process_courtship(self, potential_mate):
                        execute_mating(self, potential_mate, self.step_count)
        
        # Pregnancy
        if self.mating_state.sex == HerbieSex.CARRIER:
            birth_data = process_pregnancy(self)
            if birth_data:
                self._pending_birth = birth_data
        
        # Provider family behaviors
        if self.mating_state.sex == HerbieSex.PROVIDER:
            family_direction = provider_seek_family(self, all_herbies)
            if np.linalg.norm(family_direction) > 0.01:
                self.vel += family_direction * 0.05
            provider_feed_family(self, all_herbies)
        
        # Family defense bonus
        family_defense = get_family_defense_bonus(self, all_herbies)
        self.defense_bonus = getattr(self, 'learned_defense_bonus', 0) + family_defense
        
        # Placement memory
        if hasattr(self, 'placement_memory'):
            self.placement_memory.evolve()
            self.placement_memory.update_reference(self.pos)
            
            direction, strength = self.placement_memory.get_direction_bias(self.metabolism.hunger)
            if strength > 0.05:
                self.torus.inject_reward(direction, strength * 0.5)
        
        # Block asexual reproduction - Herbies reproduce sexually only
        self.metabolism.ready_to_reproduce = False
        
        # === MAIN CREATURE STEP (adapted from parent) ===
        self.step_count += 1
        self.hunt_cooldown = max(0, self.hunt_cooldown - 1)
        self.damage_taken_this_step = 0.0
        self.lifetime_tracker.update(self, self.step_count)
        
        # Dream state
        if silence_frames < 80:
            self.dream_depth = max(0.0, self.dream_depth - 0.018)
        else:
            self.dream_depth = 0.94 * self.dream_depth + 0.06 * np.clip((silence_frames - 80) / 180, 0, 1)
        self.is_dreaming = self.dream_depth > 0.15
        
        # Sense creatures
        self.sense_nearby_creatures(all_creatures)
        
        # Afferent channels
        for ch in self.afferent.values():
            ch.evolve()
            for a in ch.get_arrivals():
                self.torus.receive_afferent(ch.name, [a])
        
        # Torus
        g_target = self.species.torus_g_base + 0.6 * boosted_audio - 0.5 * self.metabolism.hunger - 1.2 * self.dream_depth
        self.torus.g = G_SMOOTH * self.torus.g + (1 - G_SMOOTH) * g_target
        self.torus.evolve(boosted_audio, self.metabolism.hunger, self.dream_depth)
        
        torus_arousal = self.torus.get_arousal()
        torus_phase = float(np.angle(self.torus.psi[np.argmax(np.abs(self.torus.psi)**2)]))
        torus_bias = self.torus.get_directional_bias()
        
        # Efferent (including hands)
        if self.torus.should_fire_efferent():
            patterns = self.torus.get_efferent_pattern()
            
            for i, (_, amp) in enumerate(patterns[:len(self.efferent)]):
                amp_scaled = amp * getattr(self, 'effective_efferent', 1.0)
                eff_keys = list(self.efferent.keys())
                if i < len(eff_keys):
                    self.efferent[eff_keys[i]].nucleate(amp_scaled, np.random.random())
                if i < len(self.limb_defs):
                    limb_name = list(self.limb_defs.keys())[i]
                    self.limbs[limb_name].inject_efferent(amp_scaled * 0.7)
                    self.morph.apply_efferent_torque(limb_name, amp_scaled * 0.6)
            
            # Hand efferents
            hand_amp = torus_arousal * 0.5
            left_bias = torus_bias[0]
            self.hands.inject_efferent_bilateral(hand_amp, torus_phase, left_bias)
        
        for ch in self.efferent.values():
            ch.evolve()
        
        # Hands evolution
        self.hands.evolve(self.pos)
        
        # Grip attempts
        if self.hands.left.tip_activation > 0.1 or self.hands.right.tip_activation > 0.1:
            success, which = self.hands.attempt_grip_nearest(self.pos, self.world.objects, herbie=self)
            if success:
                print(f"[Herbie] {self.mating_state.name or self.creature_id[:8]} gripped with {which} hand!")
        
        # Body field
        body_I = self.get_body_I()
        self.skeleton.step(body_I)
        V_total = self._compute_potential()
        
        g_base = self.species.body_g_base
        g_target = g_base + 0.5 * torus_arousal + self.metabolism.get_g_modifier() + 0.4 * boosted_audio - 0.4 * self.dream_depth
        self.body.g = G_SMOOTH * self.body.g + (1 - G_SMOOTH) * g_target
        self.body.evolve(V_total, torus_phase, torus_arousal, self.metabolism.get_g_modifier(),
                         self.dream_depth, boosted_audio)
        
        # Limbs
        limb_thrust = np.zeros(2)
        for limb_name, limb in self.limbs.items():
            origin_name = self.limb_defs[limb_name][0]
            phase, amp = self.body.get_region_state(origin_name)
            limb.inject_from_body(amp, phase + 0.15 * torus_phase, torus_arousal)
            limb.g = 0.9 * limb.g + 0.1 * (self.species.limb_g_base + 0.4 * torus_arousal)
            limb.evolve(torus_arousal, self.metabolism.hunger, self.dream_depth)
            
            if limb.pulse_position > 0.5 and limb.pulse_amplitude > 0.04:
                tip_phase, tip_amp = limb.get_tip_state()
                self.body.inject_at_region(origin_name, tip_amp * 0.15, tip_phase)
                angle = self.morph.limb_angles[limb_name]
                limb_thrust += np.array([-np.cos(angle), -np.sin(angle)]) * limb.pulse_amplitude * abs(self.morph.limb_velocities.get(limb_name, 0)) * 0.4
        
        if self.limb_defs:
            self.morph.update_limb_angles(self.limbs, torus_bias, self.skeleton.rms,
                                          self.metabolism.hunger, boosted_audio)
        
        # Movement
        body_momentum = self.body.get_momentum()
        if np.linalg.norm(torus_bias) > 0.05:
            self.body.inject_momentum(torus_bias, torus_arousal * 0.5)
        if np.linalg.norm(limb_thrust) > 0.02:
            self.body.inject_momentum(limb_thrust, 0.3)
        
        weight = self.hands.get_total_weight()
        weight_factor = 1.0 / (1.0 + weight * 0.2)
        eff_speed = self.get_effective_speed(season) * weight_factor
        hunger_boost = 1.0 + self.metabolism.hunger * 0.5
        
        force = body_momentum * 5.0 * hunger_boost * eff_speed
        force += torus_bias * 1.5 * hunger_boost * eff_speed
        force += limb_thrust * 2.5 * eff_speed
        
        speed = np.linalg.norm(self.vel)
        if speed < 0.5 or self.metabolism.hunger > 0.5:
            force += np.random.randn(2) * (0.4 + 0.3 * self.metabolism.hunger)
        
        boundary_margin = WORLD_L/2 - 5
        for i in range(2):
            if self.pos[i] < -boundary_margin:
                force[i] += 2.0 * (-boundary_margin - self.pos[i])
            elif self.pos[i] > boundary_margin:
                force[i] -= 2.0 * (self.pos[i] - boundary_margin)
        
        self.vel += force * dt * 20
        self.vel *= 0.96
        max_speed = (2.5 + self.metabolism.hunger * 0.5) * eff_speed
        if speed > max_speed:
            self.vel = self.vel / speed * max_speed
        self.pos += self.vel * dt
        self.pos = np.clip(self.pos, -WORLD_L/2 + 2, WORLD_L/2 - 2)
        
        # World interaction
        body_I = self.get_body_I()
        self.total_reward, self.mass_extracted = self.world.process_creature_interactions(
            self, body_I, body_momentum)
        
        # Consume held food
        for held_obj in self.hands.get_held_objects():
            if held_obj.compliance > 0.5 and held_obj.energy > 0:
                consumed = min(held_obj.energy, 0.5)
                held_obj.energy -= consumed
                self.total_reward += consumed
                self.mass_extracted += consumed
                
                if held_obj.energy <= 0:
                    held_obj.alive = False
                    if self.hands.left.held_object == held_obj:
                        self.hands.left.release()
                    if self.hands.right.held_object == held_obj:
                        self.hands.right.release()
        
        # Metabolism
        reward_source = self.world.get_reward_source_for_creature(self)
        vel_mag = np.linalg.norm(self.vel)
        old_hunger = self.metabolism.hunger
        self.metabolism.update(self.total_reward, self.mass_extracted, vel_mag, reward_source)
        
        # Pregnancy hunger modifier
        if self.mating_state.is_pregnant:
            hunger_increase = self.metabolism.hunger - old_hunger
            if hunger_increase > 0:
                self.metabolism.hunger += hunger_increase * (HERBIE_PREGNANCY_HUNGER_MULT - 1)
        
        if self.metabolism.defecation_pending:
            self.world.drop_nutrient(self.pos.copy(), self.metabolism.last_defecation_amount)
            self.lifetime_tracker.log_defecation(self.metabolism.last_defecation_amount, self.step_count)
        
        if self.metabolism.hunger > 0.2 and self.total_reward < 0.01:
            self._hunger_seeking()
        
        self.display_vel = 0.8 * self.display_vel + 0.2 * self.vel
        
        # Update mood
        self.current_mood = HerbieMood.calculate_mood(self)
        if self.step_count % 50 == 0:
            self.mood_history.append(self.current_mood)
            if len(self.mood_history) > 100:
                self.mood_history.pop(0)
        
        # Block asexual reproduction again
        self.metabolism.ready_to_reproduce = False
        
        # Death check
        if self.metabolism.is_dead:
            self.alive = False
            name = self.mating_state.name or self.creature_id[:8]
            print(f"[Herbie] {name} Gen{self.generation} died: {self.metabolism.cause_of_death}")
            
            event_log().log_death(
                step=self.step_count,
                name=name,
                cause=self.metabolism.cause_of_death,
                generation=self.generation,
                age=self.step_count,
                pos=(self.pos[0], self.pos[1])
            )
            
            HerbieNameGenerator.release(name)
            return 'dead'
        
        return 'alive'
    
    def get_mood_display(self):
        """Return (icon, color, description) for current mood."""
        info = HerbieMood.get_mood_info(self.current_mood)
        return info['icon'], info['color'], info['desc']
    
    def to_dict(self) -> dict:
        """Serialize Herbie for persistence."""
        base = super().to_dict()
        base.update({
            'sex': self.mating_state.sex.value,
            'mating_state': self.mating_state.to_dict(),
            'hands': self.hands.to_dict(),
            'placement_memory': self.placement_memory.to_dict() if hasattr(self, 'placement_memory') else None,
        })
        return base
