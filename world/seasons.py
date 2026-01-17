"""
Season System - Yearly cycles affecting food, metabolism, and predators.

Seasons affect spawn rates, hunger, movement costs, and enable
behaviors like hibernation.
"""

from typing import Optional
from dataclasses import dataclass

from ..core.constants import SEASON_LENGTH


@dataclass
class Season:
    """Season parameters."""
    name: str
    food_spawn_rate: float    # Multiplier on food spawning
    food_decay_rate: float    # Multiplier on food energy loss  
    metabolism_cost: float    # Multiplier on hunger increase
    movement_cost: float      # Multiplier on movement energy
    predator_activity: float  # Multiplier on predator aggression
    allows_hibernation: bool


# Season definitions
SEASONS = {
    'spring': Season('Spring', 2.0, 0.5, 0.9, 0.9, 0.8, False),
    'summer': Season('Summer', 1.0, 1.0, 1.0, 1.0, 1.0, False),
    'autumn': Season('Autumn', 0.3, 1.5, 1.1, 1.0, 1.2, False),
    'winter': Season('Winter', 0.1, 2.0, 1.4, 1.3, 0.6, True),
}
SEASON_ORDER = ['spring', 'summer', 'autumn', 'winter']


class SeasonSystem:
    """
    Manages seasonal cycles.
    
    Full year = 4 * SEASON_LENGTH steps
    """
    
    def __init__(self, food_multiplier: float = 1.0, harshness_multiplier: float = 1.0):
        """Initialize season system at start of spring."""
        self.step_count = 0
        self.current_season_idx = 0
        self.season_step = 0
        self.year = 0
        
        # Global modifiers from launcher
        self.food_multiplier = food_multiplier
        self.harshness_multiplier = harshness_multiplier
        
    @property
    def current_season(self) -> Season:
        """Get current Season object."""
        return SEASONS[SEASON_ORDER[self.current_season_idx]]
    
    @property
    def effective_season(self) -> Season:
        """Get season with global multipliers applied."""
        base = self.current_season
        
        # Apply harshness - makes winter harsher, summer more extreme
        if base.name == 'Winter':
            harsh_factor = 1.0 + (self.harshness_multiplier - 1.0) * 0.5
        elif base.name == 'Summer':
            harsh_factor = 1.0 + (self.harshness_multiplier - 1.0) * 0.3
        else:
            harsh_factor = 1.0
        
        return Season(
            name=base.name,
            food_spawn_rate=base.food_spawn_rate * self.food_multiplier,
            food_decay_rate=base.food_decay_rate * harsh_factor,
            metabolism_cost=base.metabolism_cost * harsh_factor,
            movement_cost=base.movement_cost * harsh_factor,
            predator_activity=base.predator_activity,
            allows_hibernation=base.allows_hibernation
        )
    
    @property
    def season_name(self) -> str:
        """Get current season name."""
        return self.current_season.name
    
    @property
    def season_progress(self) -> float:
        """Get progress through current season (0-1)."""
        return self.season_step / SEASON_LENGTH
    
    @property
    def year_progress(self) -> float:
        """Get progress through current year (0-1)."""
        total_step_in_year = self.current_season_idx * SEASON_LENGTH + self.season_step
        return total_step_in_year / (4 * SEASON_LENGTH)
    
    def update(self) -> Optional[str]:
        """
        Advance season by one step.
        
        Returns:
            Message string if season changed, None otherwise
        """
        self.step_count += 1
        self.season_step += 1
        
        if self.season_step >= SEASON_LENGTH:
            self.season_step = 0
            old_season = self.season_name
            self.current_season_idx = (self.current_season_idx + 1) % 4
            if self.current_season_idx == 0:
                self.year += 1
            return f"[SEASON] {old_season} → {self.season_name} (Year {self.year})"
        return None
    
    def get_food_spawn_rate(self) -> float:
        """Get food spawn rate multiplier for current season."""
        return self.current_season.food_spawn_rate
    
    def get_food_decay_rate(self) -> float:
        """Get food decay rate multiplier for current season."""
        return self.current_season.food_decay_rate
    
    def get_metabolism_cost(self) -> float:
        """Get metabolism cost multiplier for current season."""
        return self.current_season.metabolism_cost
    
    def get_movement_cost(self) -> float:
        """Get movement cost multiplier for current season."""
        return self.current_season.movement_cost
    
    def get_predator_activity(self) -> float:
        """Get predator activity multiplier for current season."""
        return self.current_season.predator_activity
    
    def allows_hibernation(self) -> bool:
        """Check if current season allows hibernation."""
        return self.current_season.allows_hibernation
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'step_count': self.step_count,
            'current_season_idx': self.current_season_idx,
            'season_step': self.season_step,
            'year': self.year,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SeasonSystem':
        """Deserialize from persistence."""
        seasons = cls()
        seasons.step_count = d.get('step_count', 0)
        seasons.current_season_idx = d.get('current_season_idx', 0)
        seasons.season_step = d.get('season_step', 0)
        seasons.year = d.get('year', 0)
        return seasons


# =============================================================================
# AGE VULNERABILITY HELPERS
# =============================================================================

def get_age_vulnerability(age: int, max_age: int) -> float:
    """
    Young (<10%) and old (>70%) are more vulnerable.
    
    Args:
        age: Current age in steps
        max_age: Maximum lifespan
        
    Returns:
        Vulnerability multiplier (1.0 = normal, >1 = more vulnerable)
    """
    if max_age <= 0:
        return 1.0
    age_ratio = age / max_age
    if age_ratio < 0.1:
        return 1.5 - 0.5 * (age_ratio / 0.1)  # 1.5x → 1.0x
    elif age_ratio > 0.7:
        return 1.0 + 0.5 * ((age_ratio - 0.7) / 0.3)  # 1.0x → 1.5x
    return 1.0


def get_age_speed_modifier(age: int, max_age: int) -> float:
    """
    Elderly creatures slow down.
    
    Args:
        age: Current age in steps
        max_age: Maximum lifespan
        
    Returns:
        Speed multiplier (1.0 = normal, <1 = slower)
    """
    if max_age <= 0:
        return 1.0
    age_ratio = age / max_age
    if age_ratio > 0.8:
        return 1.0 - 0.3 * ((age_ratio - 0.8) / 0.2)
    return 1.0
