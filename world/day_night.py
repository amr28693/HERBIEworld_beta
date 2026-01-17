"""
Day/Night Cycle - Circadian rhythm affecting NLSE dynamics.

Day = focusing (organized behavior)
Night = reduced focusing (dreams, predator activity)
"""

from typing import Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class DayNightState:
    """Current state of the day/night cycle."""
    time_of_day: float      # 0.0 = midnight, 0.5 = noon, 1.0 = midnight
    light_level: float      # 0.0 = dark, 1.0 = bright
    is_day: bool
    phase_name: str         # 'night', 'dawn', 'day', 'dusk'
    
    # NLSE modifiers
    g_modifier: float       # Multiplier for nonlinearity
    predator_activity: float  # Multiplier for predator behavior
    food_growth: float      # Multiplier for food spawning
    dream_boost: float      # Boost to dream depth at night


class DayNightCycle:
    """
    Manages circadian rhythm for the simulation.
    
    Day/Night affects:
    - NLSE nonlinearity sign (focusing ↔ defocusing)
    - Predator activity (more active at night)
    - Food growth (only during day)
    - Dream depth (enhanced at night)
    - Visual lighting
    """
    
    # Default cycle parameters
    DEFAULT_DAY_LENGTH = 500  # Steps for full day/night cycle
    
    # Phase boundaries (fraction of day)
    DAWN_START = 0.2
    DAY_START = 0.3
    DUSK_START = 0.7
    NIGHT_START = 0.8
    
    def __init__(self, start_time: float = 0.25, day_length: int = None):
        """
        Initialize cycle.
        
        Args:
            start_time: 0.0 = midnight, 0.25 = dawn, 0.5 = noon, 0.75 = dusk
            day_length: Steps for full day/night cycle (default 500)
        """
        self.day_length = day_length or self.DEFAULT_DAY_LENGTH
        self.cycle_position = start_time
        self.total_days = 0
        self.step_count = 0
    
    def update(self) -> DayNightState:
        """Advance cycle by one step and return current state."""
        self.step_count += 1
        
        self.cycle_position += 1.0 / self.day_length
        if self.cycle_position >= 1.0:
            self.cycle_position -= 1.0
            self.total_days += 1
        
        return self.get_state()
    
    def get_state(self) -> DayNightState:
        """Get current day/night state."""
        pos = self.cycle_position
        
        # Light level: sinusoidal, peaks at noon
        light_level = 0.5 + 0.5 * np.cos(2 * np.pi * (pos - 0.5))
        light_level = np.clip(light_level, 0.05, 1.0)
        
        # Determine phase
        if pos < self.DAWN_START or pos >= self.NIGHT_START:
            phase_name = 'night'
            is_day = False
        elif pos < self.DAY_START:
            phase_name = 'dawn'
            is_day = True
        elif pos < self.DUSK_START:
            phase_name = 'day'
            is_day = True
        else:
            phase_name = 'dusk'
            is_day = True
        
        # g_modifier: 1.0 at noon, 0.3 at midnight (always positive)
        day_factor = np.cos(2 * np.pi * (pos - 0.5))
        g_modifier = 0.65 + 0.35 * day_factor
        
        # Predator activity: 1.5x at night
        predator_activity = 1.0 + 0.5 * (1 - light_level)
        
        # Food growth: only during day
        food_growth = light_level if is_day else 0.1
        
        # Dream boost at night
        dream_boost = 1.0 + 1.5 * (1 - light_level)
        
        return DayNightState(
            time_of_day=pos,
            light_level=light_level,
            is_day=is_day,
            phase_name=phase_name,
            g_modifier=g_modifier,
            predator_activity=predator_activity,
            food_growth=food_growth,
            dream_boost=dream_boost
        )
    
    def get_time_string(self) -> str:
        """Get human-readable time string."""
        pos = self.cycle_position
        hour = int(pos * 24)
        minute = int((pos * 24 - hour) * 60)
        return f"{hour:02d}:{minute:02d}"
    
    def get_sky_color(self) -> Tuple[float, float, float]:
        """Get RGB sky color for visualization."""
        state = self.get_state()
        
        if state.phase_name == 'night':
            return (0.05, 0.05, 0.15)
        elif state.phase_name == 'dawn':
            t = (self.cycle_position - self.DAWN_START) / (self.DAY_START - self.DAWN_START)
            return (0.2 + 0.3*t, 0.1 + 0.2*t, 0.15 + 0.1*t)
        elif state.phase_name == 'day':
            return (0.4, 0.6, 0.9)
        else:  # dusk
            t = (self.cycle_position - self.DUSK_START) / (self.NIGHT_START - self.DUSK_START)
            return (0.4 - 0.2*t, 0.2 - 0.1*t, 0.2)
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'cycle_position': self.cycle_position,
            'total_days': self.total_days,
            'step_count': self.step_count
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DayNightCycle':
        """Deserialize from persistence."""
        cycle = cls(data.get('cycle_position', 0.25))
        cycle.total_days = data.get('total_days', 0)
        cycle.step_count = data.get('step_count', 0)
        return cycle


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def apply_daynight_to_nlse(creature, daynight_state: DayNightState):
    """
    Apply day/night modifier to creature's NLSE parameters.
    
    NOTE: Does not modify g directly as creatures have adaptive control.
    Effects come through dream_boost instead.
    """
    if hasattr(creature, 'dream_depth'):
        creature.dream_depth = min(1.0, creature.dream_depth + 0.01 * (daynight_state.dream_boost - 1.0))


def apply_daynight_to_predator(creature, daynight_state: DayNightState):
    """Make predators more active at night."""
    if creature.species.diet in ['carnivore', 'omnivore']:
        if hasattr(creature, 'hunt_range'):
            creature.hunt_range = creature.species.hunt_range * daynight_state.predator_activity
        
        if hasattr(creature, 'hunt_cooldown') and creature.hunt_cooldown > 0:
            reduction = int(daynight_state.predator_activity - 1.0)
            creature.hunt_cooldown = max(0, creature.hunt_cooldown - reduction)


def get_daynight_overlay_alpha(daynight_state: DayNightState) -> float:
    """Get alpha for night overlay (darker at night)."""
    return 0.4 * (1 - daynight_state.light_level)


def get_daynight_status_string(cycle: DayNightCycle) -> str:
    """Get status string with emoji for display."""
    state = cycle.get_state()
    icon = {
        'night': '🌙',
        'dawn': '🌅', 
        'day': '☀️',
        'dusk': '🌆'
    }.get(state.phase_name, '?')
    
    return f"{icon} {cycle.get_time_string()} Day{cycle.total_days}"


def get_daynight_text_status(cycle: DayNightCycle) -> str:
    """Get text-only status (no emoji) for terminal."""
    state = cycle.get_state()
    phase_abbrev = {
        'day': 'DAY', 'night': 'NITE', 'dawn': 'DAWN', 'dusk': 'DUSK'
    }
    phase = phase_abbrev.get(state.phase_name, state.phase_name.upper()[:4])
    return f"{phase} {cycle.get_time_string()} D{cycle.total_days}"
