"""
Narrative Logger - Captures interesting simulation events in readable form.

This is a simpler log that captures the "story" of what's happening,
complementing the structured JSONL for data analysis.
"""

import os
import time
from typing import Optional
from collections import deque


# Data directory
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')


class NarrativeLog:
    """
    Captures simulation narrative in human-readable format.
    
    Writes to narrative_log.txt with timestamped events.
    Filters out spam (ants, repetitive grips) to keep it interesting.
    """
    
    _instance: Optional['NarrativeLog'] = None
    
    def __init__(self, filepath: str = None):
        self.filepath = filepath or os.path.join(DATA_DIR, 'narrative_log.txt')
        self.enabled = True
        self.buffer = []
        self.buffer_size = 20
        
        # Anti-spam tracking
        self.recent_events = deque(maxlen=50)
        self.event_counts = {}
        self.last_flush_step = 0
        
        # Event categories to always log
        self.important_prefixes = [
            '[GENESIS]', '[LEVIATHAN]', '[SEASON]', '[NEST]',
            '[CULTURE', '[NEW BABY]', '[Herbie] ❤️', '[Herbie] 💕',
            '[Herbie] 🎉', '[Herbie] 💀', '[METEOR]', '[MIGRATION]',
            '[Blob]', '[Apex]', '[Biped]', '[Mono]', '[Scavenger]'
        ]
        
        # Events to filter/reduce
        self.spam_prefixes = [
            '[Ants]', '[Herbie] gripped', '[RD]'
        ]
        
        # Initialize file
        self._init_file()
    
    @classmethod
    def get(cls) -> 'NarrativeLog':
        if cls._instance is None:
            cls._instance = NarrativeLog()
        return cls._instance
    
    def _init_file(self):
        """Initialize or append to log file."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        try:
            with open(self.filepath, 'a') as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"  SESSION STARTED: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*70}\n\n")
        except Exception as e:
            print(f"[NarrativeLog] Init failed: {e}")
    
    def log(self, message: str, step: int = 0, force: bool = False):
        """
        Log a narrative event.
        
        Args:
            message: The event message (usually from print)
            step: Current simulation step
            force: If True, always log regardless of spam filter
        """
        if not self.enabled:
            return
        
        # Check if important
        is_important = force or any(message.startswith(p) for p in self.important_prefixes)
        is_spam = any(message.startswith(p) for p in self.spam_prefixes)
        
        if is_spam and not is_important:
            # Count spam events, log summary periodically
            key = message.split()[0] if message else 'unknown'
            self.event_counts[key] = self.event_counts.get(key, 0) + 1
            return
        
        # Dedupe recent identical events
        if message in self.recent_events and not force:
            return
        
        self.recent_events.append(message)
        
        # Format with timestamp
        timestamp = time.strftime('%H:%M:%S')
        entry = f"[{timestamp}] Step {step:>6}: {message}"
        self.buffer.append(entry)
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def log_summary(self, step: int):
        """Log a summary of filtered events."""
        if not self.event_counts:
            return
        
        summary_parts = []
        for key, count in sorted(self.event_counts.items(), key=lambda x: -x[1]):
            if count > 5:
                summary_parts.append(f"{key}: {count}")
        
        if summary_parts:
            timestamp = time.strftime('%H:%M:%S')
            summary = f"[{timestamp}] Step {step:>6}: [Summary] " + ", ".join(summary_parts[:5])
            self.buffer.append(summary)
        
        self.event_counts.clear()
    
    def log_ecosystem_status(self, step: int, creatures: list, world):
        """Log periodic ecosystem status."""
        # Count by species
        species_counts = {}
        for c in creatures:
            if c.alive:
                species_counts[c.species.name] = species_counts.get(c.species.name, 0) + 1
        
        total = sum(species_counts.values())
        herbies = species_counts.get('Herbie', 0)
        predators = species_counts.get('Apex', 0) + species_counts.get('Scavenger', 0)
        
        # Food count
        food_count = sum(1 for o in world.objects if o.alive and getattr(o, 'compliance', 0) > 0.5)
        
        timestamp = time.strftime('%H:%M:%S')
        status = (f"[{timestamp}] Step {step:>6}: [ECOSYSTEM] "
                  f"Pop: {total} (Herbies: {herbies}, Predators: {predators}) | "
                  f"Food: {food_count}")
        self.buffer.append(status)
        self.buffer.append("")  # Blank line for readability
    
    def flush(self):
        """Write buffer to file."""
        if not self.buffer:
            return
        
        try:
            with open(self.filepath, 'a') as f:
                for entry in self.buffer:
                    f.write(entry + '\n')
            self.buffer.clear()
        except Exception as e:
            print(f"[NarrativeLog] Flush failed: {e}")
    
    def log_section(self, title: str):
        """Log a section header."""
        self.buffer.append("")
        self.buffer.append(f"--- {title} ---")
        self.buffer.append("")


def narrative_log() -> NarrativeLog:
    """Get singleton NarrativeLog instance."""
    return NarrativeLog.get()


# Hook to capture print statements
_original_print = print

def narrative_print(*args, **kwargs):
    """Wrapper that also logs to narrative log."""
    message = ' '.join(str(a) for a in args)
    
    # Only capture bracketed events
    if message.startswith('['):
        narrative_log().log(message, step=0)
    
    _original_print(*args, **kwargs)
