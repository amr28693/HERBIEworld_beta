"""
Console Logger - Configurable verbosity for terminal output.

Verbosity levels:
- MINIMAL: Only Herbie births/deaths, seasons, major events
- HERBIE: Herbie-centric (bonds, nests, grips) + minimal  
- ECO: Ecosystem overview (all species, but summarized)
- FULL: Everything (all creatures, ants, chemistry, etc.)

Toggle with 'V' key during simulation.
"""

from enum import IntEnum
from typing import Optional
from collections import defaultdict
import time


class Verbosity(IntEnum):
    MINIMAL = 0   # Just the essentials
    HERBIE = 1    # Herbie-focused
    ECO = 2       # Ecosystem overview
    FULL = 3      # Everything


class ConsoleLogger:
    """
    Manages console output verbosity.
    
    Filters print statements based on current verbosity level.
    Provides summaries for suppressed events.
    """
    
    _instance: Optional['ConsoleLogger'] = None
    
    def __init__(self):
        self.verbosity = Verbosity.ECO  # Default: ecosystem overview
        self.enabled = True
        
        # Event counting for summaries
        self.event_counts = defaultdict(int)
        self.last_summary_step = 0
        self.summary_interval = 200  # Steps between summaries
        
        # Grip throttling
        self.grip_count = 0
        self.last_grip_report_step = 0
        
        # Category definitions
        self.categories = {
            # MINIMAL - always show
            'essential': [
                '[GENESIS]', '[LEVIATHAN]', '[SEASON]', '[METEOR]',
                '[NEW BABY]', '[Herbie] 💀', '[EXTINCTION]', '[Manager]',
                '[Persistence]', '[Shutdown]', '[History]', '[ECOSYSTEM]',
                '[ACHIEVEMENT]', '[Params]', '[Init]', '[Birth]', '[Embryo]',
                '🥚',  # All species births
            ],
            
            # HERBIE level - bonding, mating, nesting (not grips)
            'herbie': [
                '[Herbie] ❤️', '[Herbie] 💕', '[Herbie] 🎒',
                '[NEST]', '[CULTURE', '[BRAIN',
            ],
            
            # ECO level - other species events
            'eco': [
                '[Blob]', '[Apex]', '[Biped]', '[Mono]', '[Scavenger]',
                '[Aquatic]', '[Colony]', '[MIGRATION]', '[CHEMISTRY]',
                '[Herbie]',  # General Herbie messages
            ],
            
            # FULL level - everything including spam
            'full': [
                '[Ants]', '[RD]', '[Day', '[Viz]', '[View]',
                'gripped',  # Grip messages are spammy
            ],
        }
        
        # Verbosity names for display
        self.verbosity_names = {
            Verbosity.MINIMAL: "MINIMAL (essentials only)",
            Verbosity.HERBIE: "HERBIE-CENTRIC",
            Verbosity.ECO: "ECOSYSTEM OVERVIEW", 
            Verbosity.FULL: "FULL (everything)",
        }
    
    @classmethod
    def get(cls) -> 'ConsoleLogger':
        if cls._instance is None:
            cls._instance = ConsoleLogger()
        return cls._instance
    
    def cycle_verbosity(self):
        """Cycle to next verbosity level."""
        self.verbosity = Verbosity((self.verbosity + 1) % 4)
        return self.verbosity_names[self.verbosity]
    
    def set_verbosity(self, level: Verbosity):
        """Set verbosity level directly."""
        self.verbosity = level
    
    def should_print(self, message: str) -> bool:
        """Determine if message should be printed at current verbosity."""
        if not self.enabled:
            return False
        
        # Grip messages are super spammy - only show at FULL and throttled
        if 'gripped' in message:
            if self.verbosity < Verbosity.FULL:
                return False
            # Even at FULL, throttle grips
            self.grip_count += 1
            if self.grip_count % 10 != 1:  # Only show every 10th grip
                return False
        
        # Essential messages always show
        for prefix in self.categories['essential']:
            if prefix in message:
                return True
        
        # MINIMAL only shows essentials
        if self.verbosity == Verbosity.MINIMAL:
            return False
        
        # HERBIE level - check for herbie-specific interesting events
        for prefix in self.categories['herbie']:
            if prefix in message:
                return self.verbosity >= Verbosity.HERBIE
        
        # ECO level
        for prefix in self.categories['eco']:
            if prefix in message:
                return self.verbosity >= Verbosity.ECO
        
        # FULL level - everything else with brackets
        if message.startswith('['):
            return self.verbosity >= Verbosity.FULL
        
        # Non-bracketed messages (status bars, etc.) always show
        return True
    
    def count_event(self, message: str):
        """Count suppressed event for later summary."""
        # Extract category
        if message.startswith('['):
            end = message.find(']')
            if end > 0:
                category = message[1:end]
                self.event_counts[category] += 1
    
    def get_summary(self, step: int) -> Optional[str]:
        """Get summary of suppressed events if interval passed."""
        if step - self.last_summary_step < self.summary_interval:
            return None
        
        if not self.event_counts:
            return None
        
        self.last_summary_step = step
        
        # Build summary
        parts = []
        for category, count in sorted(self.event_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                parts.append(f"{category}:{count}")
        
        self.event_counts.clear()
        
        if parts:
            return f"[Summary] {', '.join(parts[:8])}"
        return None
    
    def log(self, message: str, step: int = 0, force: bool = False) -> bool:
        """
        Log a message respecting verbosity.
        
        Args:
            message: The message to log
            step: Current simulation step
            force: If True, always print regardless of verbosity
        
        Returns True if message was printed.
        """
        if force or self.should_print(message):
            print(message)
            return True
        else:
            self.count_event(message)
            return False


# Global instance
def console_log() -> ConsoleLogger:
    """Get singleton ConsoleLogger."""
    return ConsoleLogger.get()


# Wrapper for print that respects verbosity
_original_print = print

def verbose_print(*args, **kwargs):
    """Print wrapper that respects verbosity settings."""
    message = ' '.join(str(a) for a in args)
    
    # Check if this is a loggable message (starts with bracket)
    if message.startswith('['):
        logger = console_log()
        if logger.should_print(message):
            _original_print(*args, **kwargs)
        else:
            logger.count_event(message)
    else:
        # Non-categorized messages always print
        _original_print(*args, **kwargs)
