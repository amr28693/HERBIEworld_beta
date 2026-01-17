"""
Event Logger for HERBIE World simulation.

Logs key simulation events to a JSONL file for easy parsing.
Each line is a self-contained JSON object.
"""

import json
import time
from typing import Optional

from ..core.constants import EVENT_LOG_FILE


class EventLogger:
    """
    Logs key simulation events to a JSONL file for easy parsing.
    Each line is a self-contained JSON object.
    
    Event types:
    - birth: Herbie born
    - death: Herbie died  
    - bond: Two Herbies bonded
    - predation: Creature killed by predator
    - element_pickup: Herbie picked up element
    - element_drop: Herbie dropped element
    - construction: Element structure formed
    - apex_bane: Herbie used LITE_ORE against Apex
    - dismember: Corpse was dismembered
    - season: Season changed
    - achievement: Achievement unlocked
    - extinction: All Herbies died
    - resurrection: Herbie restored from persistence
    - sound: Sound event (strike, drop, voice, resonator)
    - population: Periodic population snapshot
    """
    
    _instance: Optional['EventLogger'] = None
    
    def __init__(self, filepath: str = None):
        """
        Initialize event logger.
        
        Args:
            filepath: Path to JSONL log file (default: EVENT_LOG_FILE from constants)
        """
        self.filepath = filepath or EVENT_LOG_FILE
        self.enabled = True
        self.buffer = []
        self.buffer_size = 10  # Flush every N events
        
    @classmethod
    def get(cls) -> 'EventLogger':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = EventLogger()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset singleton (useful for testing)."""
        cls._instance = None
    
    def log(self, event_type: str, step: int = 0, **data):
        """
        Log an event.
        
        Args:
            event_type: Type of event (e.g., 'birth', 'death', 'bond')
            step: Simulation step when event occurred
            **data: Additional event data
        """
        if not self.enabled:
            return
            
        event = {
            'type': event_type,
            'step': step,
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            **data
        }
        
        self.buffer.append(event)
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Write buffered events to file."""
        if not self.buffer:
            return
            
        try:
            with open(self.filepath, 'a') as f:
                for event in self.buffer:
                    f.write(json.dumps(event) + '\n')
            self.buffer.clear()
        except Exception as e:
            print(f"[EventLog] Write failed: {e}")
    
    # === Convenience methods for specific event types ===
    
    def log_birth(self, step: int, name: str, sex: str, generation: int, 
                  parents: tuple = None, pos: tuple = None, 
                  traits: dict = None, creature_id: str = None,
                  species: str = "Herbie", parent_id: str = None):
        """Log a birth event with genetic/trait information."""
        self.log('birth', step, 
                 name=name, 
                 sex=sex, 
                 generation=generation,
                 species=species,
                 creature_id=creature_id,
                 parent_id=parent_id,
                 parents=list(parents) if parents else None,
                 pos=list(pos) if pos is not None else None,
                 traits=traits)
    
    def log_death(self, step: int, name: str, cause: str, generation: int,
                  age: int = 0, pos: tuple = None):
        """Log a death event."""
        self.log('death', step, name=name, cause=cause, generation=generation,
                 age=age, pos=list(pos) if pos is not None else None)
    
    def log_bond(self, step: int, name1: str, name2: str, resonance: float = 0):
        """Log a bonding event between two creatures."""
        self.log('bond', step, partners=[name1, name2], resonance=resonance)
    
    def log_predation(self, step: int, predator: str, prey: str, prey_name: str = None):
        """Log a predation event."""
        self.log('predation', step, predator=predator, prey=prey, prey_name=prey_name)
    
    def log_element(self, step: int, action: str, name: str, element: str, pos: tuple = None):
        """Log an element interaction (pickup/drop)."""
        self.log(f'element_{action}', step, name=name, element=element,
                 pos=list(pos) if pos is not None else None)
    
    def log_apex_bane(self, step: int, herbie_name: str, apex_killed: bool, damage: float):
        """Log a Herbie using LITE_ORE against Apex."""
        self.log('apex_bane', step, herbie=herbie_name, apex_killed=apex_killed, damage=damage)
    
    def log_dismember(self, step: int, actor: str, corpse: str):
        """Log a dismemberment event."""
        self.log('dismember', step, actor=actor, corpse=corpse)
    
    def log_season(self, step: int, season: str, year: int):
        """Log a season change."""
        self.log('season', step, season=season, year=year)
    
    def log_achievement(self, step: int, name: str, description: str = ""):
        """Log an achievement unlocked."""
        self.log('achievement', step, name=name, description=description)
    
    def log_extinction(self, step: int):
        """Log extinction of all Herbies."""
        self.log('extinction', step)
    
    def log_sound(self, step: int, sound_type: str, creator_name: str, 
                  frequency: float, amplitude: float, pos: tuple = None,
                  material1: str = None, material2: str = None):
        """Log a sound event (strike, drop, voice, resonator)."""
        self.log('sound', step, 
                 sound_type=sound_type,
                 creator=creator_name,
                 frequency=round(frequency, 1),
                 amplitude=round(amplitude, 3),
                 pos=list(pos) if pos is not None else None,
                 material1=material1,
                 material2=material2)
    
    def log_resurrection(self, step: int, generation: int):
        """Log a Herbie resurrection from persistence."""
        self.log('resurrection', step, generation=generation)
    
    def log_population(self, step: int, counts: dict, elements: int = 0):
        """Log periodic population snapshot."""
        self.log('population', step, counts=counts, elements=elements)


# Global accessor function
def event_log() -> EventLogger:
    """Get the singleton EventLogger instance."""
    return EventLogger.get()
