"""
Audio System - Sound input/output and world soundscape.

Provides:
- AudioSystem: Real-time audio I/O with sonification
- WorldSoundscape: Physical world sounds from object manipulation
- SoundEvent: Individual sound events with decay
"""

from .audio_system import AudioSystem, DummyAudio, HAS_AUDIO
from .soundscape import (
    SoundEvent, WorldSoundscape, MATERIAL_SOUNDS, DEFAULT_SOUND,
    get_world_soundscape, init_world_soundscape
)
