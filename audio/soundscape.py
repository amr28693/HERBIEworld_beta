"""
WorldSoundscape - Physical sounds in the world.

Herbies can create external sounds by physically manipulating objects.
This creates a "musical instrument" layer on top of internal sonification.

Features:
- Strike sounds (hitting objects together)
- Drop sounds (dropping objects)
- Scrape sounds (dragging objects)
- Resonators (arranged objects that ring)
- Voice emission (creature vocalizations)
- Pattern detection (is it musical?)
"""

import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..chemistry.element_objects import ElementObject

# Import ElementType for material sounds
try:
    from ..chemistry.elements import ElementType
    _HAS_ELEMENTS = True
except ImportError:
    _HAS_ELEMENTS = False
    ElementType = None


@dataclass
class SoundEvent:
    """A physical sound event in the world."""
    pos: np.ndarray           # Where the sound originated
    frequency: float          # Base frequency in Hz
    amplitude: float          # Loudness (0-1)
    decay_rate: float         # How fast it fades
    timbre: str               # 'percussive', 'tonal', 'scrape', 'ring'
    source_type: str          # 'strike', 'drop', 'scrape', 'wind', 'voice'
    created_at: int           # Step when created
    creator_id: str = ""      # Who made the sound
    
    def __post_init__(self):
        self.age = 0
        self.current_amplitude = self.amplitude
    
    def update(self) -> bool:
        """Update sound, return False when sound has died."""
        self.age += 1
        self.current_amplitude *= (1.0 - self.decay_rate)
        return self.current_amplitude > 0.001


# Sound properties for different materials
# These map ElementType -> sound properties
MATERIAL_SOUNDS = {}
if _HAS_ELEMENTS and ElementType is not None:
    MATERIAL_SOUNDS = {
        ElementType.ITE: {
            'base_freq': 180,
            'freq_range': 60,
            'decay': 0.15,
            'timbre': 'percussive',
            'resonance': 0.3,
        },
        ElementType.ITE_LITE: {
            'base_freq': 320,
            'freq_range': 150,
            'decay': 0.08,
            'timbre': 'tonal',
            'resonance': 0.6,
        },
        ElementType.ORE: {
            'base_freq': 440,
            'freq_range': 200,
            'decay': 0.03,
            'timbre': 'ring',
            'resonance': 0.9,
        },
        ElementType.VAPOR: {
            'base_freq': 800,
            'freq_range': 400,
            'decay': 0.25,
            'timbre': 'scrape',
            'resonance': 0.1,
        },
        ElementType.MULCHITE: {
            'base_freq': 220,
            'freq_range': 80,
            'decay': 0.20,
            'timbre': 'percussive',
            'resonance': 0.2,
        },
        ElementType.LITE_ORE: {
            'base_freq': 660,
            'freq_range': 300,
            'decay': 0.02,
            'timbre': 'ring',
            'resonance': 0.95,
        },
    }

DEFAULT_SOUND = {
    'base_freq': 250,
    'freq_range': 100,
    'decay': 0.12,
    'timbre': 'percussive',
    'resonance': 0.4,
}


class WorldSoundscape:
    """
    Manages physical sounds in the world that Herbies create.
    These are external sounds from object manipulation.
    """
    
    def __init__(self, audio_system=None):
        self.audio_system = audio_system
        self.active_sounds: List[SoundEvent] = []
        self.sound_history: deque = deque(maxlen=1000)
        self.step_count = 0
        
        # Toggle - press 'M' to mute/unmute
        self.enabled = True
        
        # Pattern detection
        self.recent_frequencies: deque = deque(maxlen=50)
        self.pattern_score = 0.0
        
        # Resonant structures
        self.resonators: List[dict] = []
    
    def get_material_sound(self, obj) -> dict:
        """Get sound properties for an object based on its material."""
        if _HAS_ELEMENTS and hasattr(obj, 'element_type'):
            return MATERIAL_SOUNDS.get(obj.element_type, DEFAULT_SOUND)
        return DEFAULT_SOUND
    
    def create_strike_sound(self, pos: np.ndarray, obj1, obj2, 
                           velocity: float, creator_id: str = "") -> SoundEvent:
        """
        Create sound from striking two objects together.
        The resulting sound combines properties of both materials.
        """
        sound1 = self.get_material_sound(obj1)
        sound2 = self.get_material_sound(obj2)
        
        # Combined frequency - geometric mean biased toward harder material
        harder = sound1 if sound1['resonance'] > sound2['resonance'] else sound2
        softer = sound2 if sound1['resonance'] > sound2['resonance'] else sound1
        
        base_freq = np.sqrt(harder['base_freq'] * softer['base_freq'])
        freq_variation = harder['freq_range'] * velocity * np.random.uniform(0.8, 1.2)
        frequency = base_freq + freq_variation
        
        # Amplitude from velocity and resonance
        amplitude = np.clip(velocity * 0.5 * harder['resonance'], 0.1, 1.0)
        
        # Decay - harder materials ring longer
        decay = harder['decay'] * (1 - 0.3 * velocity)
        
        sound = SoundEvent(
            pos=pos.copy(),
            frequency=float(frequency),
            amplitude=float(amplitude),
            decay_rate=float(np.clip(decay, 0.01, 0.3)),
            timbre=harder['timbre'],
            source_type='strike',
            created_at=self.step_count,
            creator_id=creator_id,
        )
        
        self.active_sounds.append(sound)
        self.sound_history.append(sound)
        self.recent_frequencies.append(frequency)
        
        print(f"[SOUND] 🎵 Strike! {frequency:.0f}Hz ({harder['timbre']}) at ({pos[0]:+.1f}, {pos[1]:+.1f})")
        
        return sound
    
    def create_drop_sound(self, pos: np.ndarray, obj, drop_height: float,
                         creator_id: str = "") -> SoundEvent:
        """Create sound from dropping an object."""
        sound_props = self.get_material_sound(obj)
        
        frequency = sound_props['base_freq'] + sound_props['freq_range'] * np.sqrt(drop_height) * 0.5
        amplitude = np.clip(np.sqrt(drop_height) * 0.4, 0.1, 0.8)
        
        sound = SoundEvent(
            pos=pos.copy(),
            frequency=float(frequency),
            amplitude=float(amplitude),
            decay_rate=sound_props['decay'],
            timbre=sound_props['timbre'],
            source_type='drop',
            created_at=self.step_count,
            creator_id=creator_id,
        )
        
        self.active_sounds.append(sound)
        self.sound_history.append(sound)
        self.recent_frequencies.append(frequency)
        
        return sound
    
    def create_scrape_sound(self, pos: np.ndarray, obj, velocity: float,
                           creator_id: str = "") -> Optional[SoundEvent]:
        """Create continuous scraping sound (called each frame during scrape)."""
        if np.random.random() > velocity * 0.3:
            return None
        
        sound_props = self.get_material_sound(obj)
        
        frequency = sound_props['base_freq'] * np.random.uniform(0.7, 1.3)
        amplitude = velocity * 0.2 * sound_props['resonance']
        
        sound = SoundEvent(
            pos=pos.copy(),
            frequency=float(frequency),
            amplitude=float(amplitude),
            decay_rate=0.3,
            timbre='scrape',
            source_type='scrape',
            created_at=self.step_count,
            creator_id=creator_id,
        )
        
        self.active_sounds.append(sound)
        return sound
    
    def create_resonator(self, pos: np.ndarray, elements: List) -> Optional[dict]:
        """
        Create a resonant structure from arranged elements.
        Can produce ongoing sounds when disturbed.
        """
        if len(elements) < 2:
            return None
        
        freqs = []
        resonance_sum = 0.0
        for elem in elements:
            props = self.get_material_sound(elem)
            freqs.append(props['base_freq'])
            resonance_sum += props['resonance']
        
        spread = np.std([np.linalg.norm(e.pos - pos) for e in elements]) if len(elements) > 1 else 1.0
        
        resonator = {
            'pos': pos.copy(),
            'base_freq': np.mean(freqs),
            'harmonics': [f / freqs[0] for f in freqs] if freqs[0] > 0 else [1.0],
            'resonance': resonance_sum / len(elements),
            'spread': spread,
            'elements': [id(e) for e in elements],
            'created_at': self.step_count,
        }
        
        self.resonators.append(resonator)
        print(f"[SOUND] 🎼 Resonator created at ({pos[0]:+.1f}, {pos[1]:+.1f}) - {len(elements)} elements")
        
        return resonator
    
    def trigger_resonator(self, resonator: dict, intensity: float, 
                         creator_id: str = "") -> List[SoundEvent]:
        """Trigger a resonator to produce sound."""
        sounds = []
        
        base_freq = resonator['base_freq']
        
        for i, harmonic_ratio in enumerate(resonator['harmonics'][:4]):
            freq = base_freq * harmonic_ratio * (1 + resonator['spread'] * 0.1)
            amp = intensity * resonator['resonance'] * (0.5 ** i)
            
            if amp > 0.02:
                sound = SoundEvent(
                    pos=resonator['pos'].copy(),
                    frequency=float(freq),
                    amplitude=float(amp),
                    decay_rate=0.02 * (1 + i * 0.3),
                    timbre='ring',
                    source_type='resonator',
                    created_at=self.step_count,
                    creator_id=creator_id,
                )
                self.active_sounds.append(sound)
                sounds.append(sound)
                self.recent_frequencies.append(freq)
        
        if sounds:
            print(f"[SOUND] 🎶 Resonator triggered! {len(sounds)} harmonics")
        
        return sounds
    
    def step(self, wind_strength: float = 0.0):
        """Update all sounds and check for wind-triggered resonators."""
        self.step_count += 1
        
        # Update active sounds
        self.active_sounds = [s for s in self.active_sounds if s.update()]
        
        # Wind can trigger resonators
        if wind_strength > 0.1:
            for resonator in self.resonators:
                if np.random.random() < wind_strength * resonator['resonance'] * 0.05:
                    self.trigger_resonator(resonator, wind_strength * 0.5)
        
        # Update pattern score
        self._analyze_patterns()
        
        # Update audio output
        self._update_audio_output()
    
    def _analyze_patterns(self):
        """Analyze recent sounds for musical patterns."""
        if len(self.recent_frequencies) < 5:
            self.pattern_score = 0.0
            return
        
        freqs = list(self.recent_frequencies)
        
        # Check for harmonic relationships
        harmonic_score = 0.0
        for i in range(len(freqs) - 1):
            ratio = freqs[i+1] / freqs[i] if freqs[i] > 0 else 1
            for simple_ratio in [2.0, 1.5, 1.33, 1.25, 1.0, 0.5, 0.67, 0.75, 0.8]:
                if abs(ratio - simple_ratio) < 0.1:
                    harmonic_score += 0.2
                    break
        
        # Check for repetition (rhythm)
        rhythm_score = 0.0
        if len(freqs) >= 8:
            for pattern_len in [2, 3, 4]:
                pattern = freqs[-pattern_len:]
                matches = 0
                for i in range(len(freqs) - pattern_len * 2):
                    if all(abs(freqs[i+j] - pattern[j]) < 50 for j in range(pattern_len)):
                        matches += 1
                rhythm_score += matches * 0.1
        
        self.pattern_score = np.clip(harmonic_score + rhythm_score, 0.0, 1.0)
        
        if self.pattern_score > 0.5 and self.step_count % 100 == 0:
            print(f"[MUSIC] 🎹 Musical pattern detected! Score: {self.pattern_score:.2f}")
    
    def _update_audio_output(self):
        """Mix world sounds into the audio output."""
        if self.audio_system is None:
            return
        
        if not self.enabled:
            self.audio_system._world_freqs = []
            self.audio_system._world_amps = []
            return
        
        world_freqs = []
        world_amps = []
        
        for sound in self.active_sounds:
            if sound.current_amplitude > 0.01:
                freq_exists = any(abs(f - sound.frequency) < 30 for f in world_freqs)
                if not freq_exists and len(world_freqs) < 6:
                    world_freqs.append(sound.frequency)
                    world_amps.append(sound.current_amplitude * 0.4)
        
        self.audio_system._world_freqs = world_freqs
        self.audio_system._world_amps = world_amps
    
    def get_nearby_sounds(self, pos: np.ndarray, radius: float = 20.0) -> List[SoundEvent]:
        """Get sounds near a position (for creature hearing)."""
        return [s for s in self.active_sounds 
                if np.linalg.norm(s.pos - pos) < radius]
    
    def compute_sound_field_at(self, pos: np.ndarray) -> dict:
        """
        Compute the aggregate sound field at a position.
        Returns information a creature can use to perceive/respond to sounds.
        """
        nearby = self.get_nearby_sounds(pos, radius=30.0)
        
        if not nearby:
            return {
                'total_amplitude': 0.0,
                'dominant_freq': 0.0,
                'direction': np.array([0.0, 0.0]),
                'n_sources': 0,
                'is_musical': False,
                'loudest_source': None,
            }
        
        total_amp = 0.0
        weighted_freq = 0.0
        direction = np.array([0.0, 0.0])
        loudest = None
        loudest_amp = 0.0
        
        for sound in nearby:
            dist = np.linalg.norm(sound.pos - pos)
            falloff = 1.0 / (1.0 + dist * 0.1) ** 2
            local_amp = sound.current_amplitude * falloff
            
            total_amp += local_amp
            weighted_freq += sound.frequency * local_amp
            
            if dist > 0.1:
                dir_to_sound = (sound.pos - pos) / dist
                direction += dir_to_sound * local_amp
            
            if local_amp > loudest_amp:
                loudest_amp = local_amp
                loudest = sound
        
        if total_amp > 0:
            weighted_freq /= total_amp
            dir_mag = np.linalg.norm(direction)
            if dir_mag > 0.01:
                direction = direction / dir_mag
        
        return {
            'total_amplitude': float(total_amp),
            'dominant_freq': float(weighted_freq),
            'direction': direction,
            'n_sources': len(nearby),
            'is_musical': self.pattern_score > 0.4,
            'loudest_source': loudest,
        }
    
    def emit_creature_voice(self, pos: np.ndarray, creature_id: str, 
                           frequencies: List[float], amplitudes: List[float]) -> List[SoundEvent]:
        """
        Emit a creature's internal sonification as world sounds.
        This is the creature "singing" or "vocalizing" their internal state.
        """
        sounds = []
        
        total_amp = sum(amplitudes) if amplitudes else 0
        if total_amp < 0.05:
            return sounds
        
        recent_vocalizations = len([s for s in self.active_sounds 
                                   if s.creator_id == creature_id and s.source_type == 'voice'])
        if recent_vocalizations > 2:
            return sounds
        
        if frequencies and amplitudes:
            max_idx = np.argmax(amplitudes)
            freq = frequencies[max_idx]
            amp = amplitudes[max_idx]
            
            if amp > 0.03:
                sound = SoundEvent(
                    pos=pos.copy(),
                    frequency=float(freq),
                    amplitude=float(amp * 0.5),
                    decay_rate=0.08,
                    timbre='tonal',
                    source_type='voice',
                    created_at=self.step_count,
                    creator_id=creature_id,
                )
                self.active_sounds.append(sound)
                self.recent_frequencies.append(freq)
                sounds.append(sound)
        
        return sounds
    
    def get_status(self) -> str:
        """Get status string for display."""
        n_sounds = len(self.active_sounds)
        n_resonators = len(self.resonators)
        pattern = f" 🎹{self.pattern_score:.1f}" if self.pattern_score > 0.3 else ""
        return f"♪{n_sounds} ⚗{n_resonators}{pattern}"


# Global soundscape instance
_world_soundscape: Optional[WorldSoundscape] = None


def get_world_soundscape() -> Optional[WorldSoundscape]:
    """Get the global world soundscape instance."""
    return _world_soundscape


def init_world_soundscape(audio_system=None) -> WorldSoundscape:
    """Initialize the global world soundscape."""
    global _world_soundscape
    _world_soundscape = WorldSoundscape(audio_system)
    return _world_soundscape
