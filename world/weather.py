"""
Weather System - Environmental events affecting food and creatures.

Manages rain, drought, storms, and bloom events that affect
food growth, energy, and survival.
"""

from typing import List, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

from ..core.constants import WORLD_L

if TYPE_CHECKING:
    from .objects import NutrientPatch


@dataclass
class WeatherEvent:
    """A weather event affecting a region."""
    event_type: str      # 'rain', 'drought', 'storm', 'bloom'
    center: np.ndarray   # World position
    radius: float        # Affected radius
    intensity: float     # 0-1
    duration: int        # Steps remaining
    start_step: int      # When event started


class WeatherSystem:
    """
    Manages weather events that affect crops/food.
    
    Event types:
    - rain: Increases food energy and size
    - drought: Decreases food energy, can kill plants
    - storm: Can destroy food, pushes objects
    - bloom: Spawns new nutrients, increases food energy
    """
    
    def __init__(self):
        """Initialize weather system."""
        self.active_events: List[WeatherEvent] = []
        self.event_history: List[dict] = []
        self.step_count = 0
        
        # Weather activity varies cyclically
        self.activity_phase = np.random.uniform(0, 2*np.pi)
        self.activity_period = np.random.uniform(3000, 8000)
        
    def get_event_probability(self) -> float:
        """Get probability of spawning a new weather event."""
        base = 0.002
        amplitude = 0.006
        phase = self.step_count / self.activity_period * 2 * np.pi + self.activity_phase
        return base + amplitude * (0.5 + 0.5 * np.sin(phase))
    
    def update(self, world) -> List[str]:
        """
        Update weather system.
        
        Args:
            world: World object with objects and nutrients lists
            
        Returns:
            List of message strings for display
        """
        self.step_count += 1
        messages = []
        
        # Spawn new event?
        if np.random.random() < self.get_event_probability():
            event = self._spawn_event()
            self.active_events.append(event)
            messages.append(f"[WEATHER] {event.event_type.upper()} at ({event.center[0]:.0f}, {event.center[1]:.0f})!")
        
        # Update active events
        for event in self.active_events:
            self._apply_event(event, world)
            event.duration -= 1
        
        # Remove expired events by rebuilding list (avoids numpy array comparison issues)
        expired_indices = [i for i, e in enumerate(self.active_events) if e.duration <= 0]
        for idx in reversed(expired_indices):
            e = self.active_events[idx]
            self.event_history.append({
                'type': e.event_type,
                'center': e.center.tolist(),
                'intensity': e.intensity,
                'start': e.start_step,
                'end': self.step_count
            })
            messages.append(f"[WEATHER] {e.event_type} ended")
            del self.active_events[idx]
        
        return messages
    
    def _spawn_event(self) -> WeatherEvent:
        """Spawn a random weather event."""
        event_types = ['rain', 'drought', 'storm', 'bloom']
        weights = [0.35, 0.25, 0.15, 0.25]
        event_type = np.random.choice(event_types, p=weights)
        
        center = np.array([
            np.random.uniform(-WORLD_L/2 + 5, WORLD_L/2 - 5),
            np.random.uniform(-WORLD_L/2 + 5, WORLD_L/2 - 5)
        ])
        
        if event_type == 'rain':
            radius = np.random.uniform(8, 15)
            intensity = np.random.uniform(0.4, 0.8)
            duration = int(np.random.uniform(200, 500))
        elif event_type == 'drought':
            radius = np.random.uniform(10, 18)
            intensity = np.random.uniform(0.5, 0.9)
            duration = int(np.random.uniform(400, 800))
        elif event_type == 'storm':
            radius = np.random.uniform(6, 12)
            intensity = np.random.uniform(0.7, 1.0)
            duration = int(np.random.uniform(50, 150))
        else:  # bloom
            radius = np.random.uniform(8, 14)
            intensity = np.random.uniform(0.5, 0.9)
            duration = int(np.random.uniform(300, 600))
        
        return WeatherEvent(event_type, center, radius, intensity, duration, self.step_count)
    
    def _apply_event(self, event: WeatherEvent, world):
        """Apply weather event effects to world."""
        # Import here to avoid circular dependency
        from .objects import NutrientPatch
        
        for obj in world.objects:
            if not obj.alive or obj.compliance < 0.5:
                continue
            
            dist = np.linalg.norm(obj.pos - event.center)
            if dist > event.radius:
                continue
            
            effect_strength = event.intensity * (1 - dist / event.radius)
            
            if event.event_type == 'rain':
                obj.energy = min(obj.max_energy * 1.5, obj.energy + effect_strength * 0.3)
                obj.size = min(obj.initial_size * 1.3, obj.size + effect_strength * 0.005)
            elif event.event_type == 'drought':
                obj.energy = max(0, obj.energy - effect_strength * 0.2)
                if obj.energy < 1:
                    obj.alive = False
            elif event.event_type == 'storm':
                if np.random.random() < effect_strength * 0.02:
                    obj.alive = False
                push_dir = obj.pos - event.center
                push_dir = push_dir / (np.linalg.norm(push_dir) + 0.1)
                obj.vel += push_dir * effect_strength * 0.5
            elif event.event_type == 'bloom':
                obj.energy = min(obj.max_energy * 2, obj.energy + effect_strength * 0.5)
        
        # Bloom can spawn new nutrients
        if event.event_type == 'bloom' and np.random.random() < event.intensity * 0.01:
            offset = np.random.randn(2) * event.radius * 0.5
            new_pos = np.clip(event.center + offset, -WORLD_L/2 + 3, WORLD_L/2 - 3)
            world.nutrients.append(NutrientPatch(new_pos, np.random.uniform(20, 40)))
        
        # Rain can spawn nutrients too
        if event.event_type == 'rain' and np.random.random() < event.intensity * 0.005:
            offset = np.random.randn(2) * event.radius * 0.3
            new_pos = np.clip(event.center + offset, -WORLD_L/2 + 3, WORLD_L/2 - 3)
            world.nutrients.append(NutrientPatch(new_pos, np.random.uniform(10, 25)))
    
    def get_visual_data(self) -> List[dict]:
        """Get data for visualization."""
        return [{
            'type': e.event_type,
            'center': e.center.copy(),
            'radius': e.radius,
            'intensity': e.intensity,
            'duration': e.duration
        } for e in self.active_events]
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'step_count': self.step_count,
            'activity_phase': self.activity_phase,
            'activity_period': self.activity_period,
            'event_history': self.event_history[-100:],  # Keep last 100
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'WeatherSystem':
        """Deserialize from persistence."""
        weather = cls()
        weather.step_count = d.get('step_count', 0)
        weather.activity_phase = d.get('activity_phase', weather.activity_phase)
        weather.activity_period = d.get('activity_period', weather.activity_period)
        weather.event_history = d.get('event_history', [])
        return weather
