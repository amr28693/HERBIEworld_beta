"""
Metabolism - Full metabolic system with hunger, satiation, gut, and reproduction.

Tracks energy intake, digestion, hunger/satiation states, aging, and
reproduction readiness. Species-specific parameters control metabolic rates.
"""

import time
from typing import Optional
import numpy as np


class Metabolism:
    """
    Full metabolic system with hunger, satiation, gut, and reproduction.
    
    Manages:
    - Hunger and satiation levels
    - Gut contents and digestion
    - Defecation (nutrient cycling)
    - Age and lifespan
    - Reproduction readiness
    - Starvation tracking
    
    Species-specific parameters:
    - energy_efficiency: How well food converts to energy (0.5-1.5)
    - metabolism_rate: Hunger increase multiplier (affects basal + movement cost)
    """
    
    def __init__(self, inherited_vigor: float = 1.0, 
                 energy_efficiency: float = 1.0,
                 metabolism_rate: float = 1.0,
                 reproduction_refractory: int = 1000):
        """
        Initialize metabolism.
        
        Args:
            inherited_vigor: Inherited metabolic efficiency (affects lifespan, digestion)
            energy_efficiency: Species energy conversion efficiency
            metabolism_rate: Species hunger rate multiplier
            reproduction_refractory: Minimum steps between reproduction (biological recovery)
        """
        self.hunger = 0.35
        self.satiation = 0.0
        self.boredom = 0.0
        
        self.last_meal_time = time.time()
        self.time_at_current_source = 0.0
        self.last_reward_source = None
        
        # Gut system
        self.gut_contents = 0.0
        self.gut_capacity = 30.0
        self.defecation_pending = False
        self.last_defecation_amount = 0.0
        
        # Age and lifespan
        self.age = 0
        self.max_age = int(np.random.uniform(8000, 12000) * inherited_vigor)
        
        # Consumption tracking
        self.total_consumed = 0.0
        
        # Reproduction - emergent, only refractory is fixed (biological recovery time)
        self.reproduction_refractory = 0
        self.reproduction_refractory_period = reproduction_refractory
        self.ready_to_reproduce = False
        
        # Death state
        self.is_dead = False
        self.cause_of_death = None
        
        # Vigor affects various rates
        self.vigor = inherited_vigor
        
        # Species-specific metabolic parameters
        self.energy_efficiency = energy_efficiency  # Higher = better food conversion
        self.metabolism_rate = metabolism_rate      # Higher = faster hunger increase
        
        # Starvation tracking
        self.starvation_timer = 0
        
    def update(self, reward: float, mass_extracted: float, creature_vel_mag: float, 
               current_source_id: Optional[int] = None):
        """
        Update metabolism for one timestep.
        
        Args:
            reward: Reward signal from food consumption
            mass_extracted: Mass extracted from food
            creature_vel_mag: Creature's velocity magnitude
            current_source_id: ID of current food source (for habituation)
        """
        self.age += 1
        
        # Hunger increases with age (basal metabolic rate)
        # Species metabolism_rate scales the whole curve
        age_factor = 1.0 + 0.5 * (self.age / self.max_age)
        base_hunger_rate = 0.0004 * age_factor * self.metabolism_rate
        
        # Movement cost: kinetic energy dissipates as metabolic heat
        # Quadratic in velocity (KE ~ v²), creates exploration/conservation trade-off
        # Species metabolism_rate affects this too - high metabolism = expensive movement
        movement_cost = 0.00015 * creature_vel_mag ** 2 * self.metabolism_rate
        
        self.hunger = min(1.0, self.hunger + base_hunger_rate + movement_cost)
        
        # Starvation check
        if self.hunger > 0.95 and self.satiation < 0.1:
            self.starvation_timer += 1
            if self.starvation_timer > 200:
                self.is_dead = True
                self.cause_of_death = 'starvation'
        else:
            self.starvation_timer = 0
            
        # Age death
        if self.age > self.max_age:
            self.is_dead = True
            self.cause_of_death = 'old_age'
        
        # Boredom from inactivity
        if creature_vel_mag < 0.4:
            self.boredom = min(1.0, self.boredom + 0.002)
        else:
            self.boredom = max(0.0, self.boredom - 0.001)
        
        # Gut processing
        self.gut_contents += mass_extracted
        self.total_consumed += mass_extracted
        
        # Reproduction - EMERGENT from energetic state
        # No arbitrary threshold - readiness emerges from:
        # 1. Low hunger (well-fed)
        # 2. Sufficient body energy (stored reserves)
        # 3. Not recently reproduced (refractory period for biological recovery)
        # 4. Age-related fertility (mature but not too old)
        
        self.reproduction_refractory = max(0, self.reproduction_refractory - 1)
        
        if not self.ready_to_reproduce and self.reproduction_refractory == 0:
            # Calculate reproduction probability based on state
            # Well-fed and energetic = higher chance
            hunger_factor = max(0, 1.0 - self.hunger * 1.5)  # 0 at hunger=0.67+, 1 at hunger=0
            satiation_factor = 0.5 + 0.5 * min(1.0, self.satiation * 2)  # Base 0.5, up to 1.0
            
            # Age factor: need some maturity, peak in middle age
            age_ratio = self.age / max(1, self.max_age)
            if age_ratio < 0.05:
                age_factor = 0.3 + age_ratio * 14  # Start at 0.3, ramp to 1.0 by age 5%
            elif age_ratio < 0.7:
                age_factor = 1.0  # Prime: full fertility
            else:
                age_factor = max(0.1, 1.0 - (age_ratio - 0.7) * 3)  # Old: declining but not zero
            
            # Combined reproduction probability per step
            # Base rate higher for faster reproduction
            repro_probability = 0.002 * hunger_factor * satiation_factor * age_factor * self.vigor
            
            if np.random.random() < repro_probability:
                self.ready_to_reproduce = True
                self.reproduction_refractory = self.reproduction_refractory_period
        
        # Digestion - energy_efficiency affects how much hunger is reduced per unit digested
        if self.gut_contents > 0:
            digestion = min(0.002 * self.vigor, self.gut_contents)
            self.gut_contents -= digestion
            # High energy_efficiency = more hunger reduction per unit food
            self.hunger = max(0.0, self.hunger - digestion * 0.01 * self.vigor * self.energy_efficiency)
        
        # Defecation
        self.defecation_pending = False
        if self.gut_contents > self.gut_capacity * 0.8:
            self.defecation_pending = True
            self.last_defecation_amount = self.gut_contents * 0.6
            self.gut_contents *= 0.4
        
        # Reward processing with habituation
        if reward > 0:
            if current_source_id == self.last_reward_source:
                self.time_at_current_source += 1
                habituation = np.exp(-self.time_at_current_source * 0.005)
                effective_reward = reward * habituation
            else:
                self.time_at_current_source = 0
                self.last_reward_source = current_source_id
                effective_reward = reward
            self.hunger = max(0.0, self.hunger - effective_reward * 0.15 * self.vigor)
            self.satiation = min(1.0, self.satiation + effective_reward * 0.1)
            self.last_meal_time = time.time()
        else:
            self.satiation = max(0.0, self.satiation - 0.001)
            self.time_at_current_source = 0.0
            
    def get_g_modifier(self) -> float:
        """Get nonlinearity modifier for body field based on metabolic state."""
        return -0.4 * self.hunger + 0.2 * self.satiation + 0.3 * self.boredom
    
    def get_restlessness(self) -> float:
        """Get restlessness level (drives exploration behavior)."""
        return max(self.boredom * 0.8, (self.hunger - 0.6) * 0.5 if self.hunger > 0.6 else 0.0)
    
    def get_age_fraction(self) -> float:
        """Get fraction of lifespan elapsed."""
        return self.age / self.max_age
    
    def consume_for_reproduction(self):
        """Consume energy for reproduction (called when reproducing)."""
        self.ready_to_reproduce = False
        self.total_consumed *= 0.5  # Reset consumption tracking
        self.hunger = min(1.0, self.hunger + 0.3)
    
    def to_dict(self) -> dict:
        """Serialize metabolism state."""
        return {
            'hunger': self.hunger,
            'satiation': self.satiation,
            'boredom': self.boredom,
            'gut_contents': self.gut_contents,
            'age': self.age,
            'max_age': self.max_age,
            'total_consumed': self.total_consumed,
            'reproduction_refractory': self.reproduction_refractory,
            'ready_to_reproduce': self.ready_to_reproduce,
            'vigor': self.vigor,
            'starvation_timer': self.starvation_timer,
            'is_dead': self.is_dead,
            'cause_of_death': self.cause_of_death,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Metabolism':
        """Deserialize metabolism state."""
        metab = cls(d.get('vigor', 1.0))
        metab.hunger = d.get('hunger', 0.35)
        metab.satiation = d.get('satiation', 0.0)
        metab.boredom = d.get('boredom', 0.0)
        metab.gut_contents = d.get('gut_contents', 0.0)
        metab.age = d.get('age', 0)
        metab.max_age = d.get('max_age', metab.max_age)
        metab.total_consumed = d.get('total_consumed', 0.0)
        metab.reproduction_refractory = d.get('reproduction_refractory', 0)
        metab.ready_to_reproduce = d.get('ready_to_reproduce', False)
        metab.starvation_timer = d.get('starvation_timer', 0)
        metab.is_dead = d.get('is_dead', False)
        metab.cause_of_death = d.get('cause_of_death', None)
        return metab
