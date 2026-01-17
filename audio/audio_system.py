"""
AudioSystem - Combined audio input (sensing) and output (sonification).

Handles:
- Microphone input for audio-reactive behavior
- Spectral analysis (bass/mid/treble, centroid, flux)
- Output sonification based on creature state
- World sound mixing
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional

# Audio constants
SAMPLE_RATE = 44100
BLOCK_IN = 2048
BLOCK_OUT = 1024

# Check for sounddevice
try:
    import sounddevice as sd
    HAS_AUDIO = True
except (ImportError, OSError):
    HAS_AUDIO = False
    sd = None


class AudioSystem:
    """Combined audio input (sensing) and output (sonification)."""
    
    def __init__(self):
        # Input state
        self.amplitude = 0.0
        self.bass = 0.0
        self.mid = 0.0
        self.treble = 0.0
        self.amp_history = deque(maxlen=600)
        self.silence_frames = 0
        self.in_buffer = np.zeros(4096)
        
        # Spectral analysis
        self.spectral_centroid = 0.0
        self.spectral_flux = 0.0
        self.prev_spectrum = None
        
        # Output state
        self.out_freqs = [220.0]
        self.out_amps = [0.0]
        self.out_phases = np.zeros(24)
        self.master_amp = 0.0
        self.step_count = 0
        
        # World sounds (set by WorldSoundscape)
        self._world_freqs: List[float] = []
        self._world_amps: List[float] = []
        self._world_phases = np.zeros(10)
        
        self.in_stream = None
        self.out_stream = None
    
    def _in_callback(self, indata, frames, time_info, status):
        """Input stream callback - analyze incoming audio."""
        audio = indata[:, 0] if indata.ndim > 1 else indata
        self.amplitude = float(np.sqrt(np.mean(audio**2) + 1e-18))
        
        n = min(audio.size, self.in_buffer.size)
        self.in_buffer = np.roll(self.in_buffer, -n)
        self.in_buffer[-n:] = audio[:n]
        
        spec = np.abs(np.fft.rfft(self.in_buffer * np.hanning(self.in_buffer.size)))
        freqs = np.fft.rfftfreq(self.in_buffer.size, 1/SAMPLE_RATE)
        
        def band(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            return float(np.sqrt(np.mean(spec[m]**2) + 1e-18)) if np.any(m) else 0.0
        
        self.bass = band(20, 300)
        self.mid = band(300, 2000)
        self.treble = band(2000, 12000)
        
        # Spectral centroid (brightness)
        spec_sum = np.sum(spec) + 1e-12
        self.spectral_centroid = float(np.sum(freqs * spec) / spec_sum)
        
        # Spectral flux (change rate)
        if self.prev_spectrum is not None and len(self.prev_spectrum) == len(spec):
            self.spectral_flux = float(np.sqrt(np.mean((spec - self.prev_spectrum)**2)))
        self.prev_spectrum = spec.copy()
        
        self.amp_history.append(self.amplitude)
        self.silence_frames = self.silence_frames + 1 if self.amplitude < 0.008 else 0
    
    def _out_callback(self, outdata, frames, time_info, status):
        """Output stream callback - generate audio."""
        t = np.arange(frames) / SAMPLE_RATE
        out = np.zeros(frames)
        
        while len(self.out_phases) < len(self.out_freqs):
            self.out_phases = np.append(self.out_phases, np.random.uniform(0, 2*np.pi))
        
        # Creature internal sonification
        for i, (f, a) in enumerate(zip(self.out_freqs, self.out_amps)):
            if i < len(self.out_phases):
                out += np.sin(2*np.pi*f*t + self.out_phases[i]) * a * self.master_amp
                self.out_phases[i] = (self.out_phases[i] + 2*np.pi*f*frames/SAMPLE_RATE) % (2*np.pi)
        
        # World sounds (strikes, drops, resonators, voices)
        while len(self._world_phases) < len(self._world_freqs):
            self._world_phases = np.append(self._world_phases, np.random.uniform(0, 2*np.pi))
        
        for i, (f, a) in enumerate(zip(self._world_freqs, self._world_amps)):
            if i < len(self._world_phases) and a > 0.005:
                # World sounds have harmonics for richer timbre
                out += np.sin(2*np.pi*f*t + self._world_phases[i]) * a * 0.6
                out += np.sin(2*np.pi*f*2*t + self._world_phases[i]) * a * 0.2
                out += np.sin(2*np.pi*f*3*t + self._world_phases[i]) * a * 0.1
                self._world_phases[i] = (self._world_phases[i] + 2*np.pi*f*frames/SAMPLE_RATE) % (2*np.pi)
        
        out = np.tanh(out * 2.5) * 0.3
        outdata[:] = np.column_stack([out, out])
    
    def start(self):
        """Start audio streams."""
        if not HAS_AUDIO:
            return
        
        try:
            self.in_stream = sd.InputStream(
                channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_IN,
                callback=self._in_callback
            )
            self.in_stream.start()
            print("[Audio] Input started")
        except Exception as e:
            print(f"[Audio] Input failed: {e}")
        
        try:
            self.out_stream = sd.OutputStream(
                channels=2, samplerate=SAMPLE_RATE, blocksize=BLOCK_OUT,
                callback=self._out_callback
            )
            self.out_stream.start()
            print("[Audio] Output started")
        except Exception as e:
            print(f"[Audio] Output failed: {e}")
    
    def stop(self):
        """Stop audio streams."""
        for s in [self.in_stream, self.out_stream]:
            if s:
                try:
                    s.stop()
                    s.close()
                except:
                    pass
    
    def update_sonification(self, body_psi, torus_psi, limbs, dream_depth: float = 0.0):
        """
        Generate frequencies from field state.
        
        Args:
            body_psi: Body field wavefunction
            torus_psi: Torus brain wavefunction
            limbs: Dict of limb objects
            dream_depth: Dream state depth (0-1)
        """
        self.step_count += 1
        self.out_freqs = []
        self.out_amps = []
        
        # Grid dimensions from body field
        Ny, Nx = body_psi.shape
        
        # Torus contribution
        torus_I = np.abs(torus_psi)**2
        phase_var = np.mean(np.abs(np.diff(np.angle(torus_psi))))
        base_freq = 140 + 80 * phase_var
        self.out_freqs.append(float(base_freq * (1 - 0.2*dream_depth)))
        self.out_amps.append(float(np.sqrt(np.sum(torus_I)) * 0.06))
        
        # Body hotspots
        I = (np.abs(body_psi)**2).copy()
        thresh = np.percentile(I, 97) * 0.6
        for _ in range(3):
            idx = np.unravel_index(np.argmax(I), I.shape)
            if I[idx] < thresh:
                break
            freq = 100 + (idx[0] / Ny) * 350
            self.out_freqs.append(float(freq * (1 - 0.15*dream_depth)))
            self.out_amps.append(0.04)
            I[max(0,idx[0]-6):min(Ny,idx[0]+7), max(0,idx[1]-6):min(Nx,idx[1]+7)] = 0
        
        # Limb contributions
        for limb in limbs.values():
            if hasattr(limb, 'pulse_amplitude') and limb.pulse_amplitude > 0.06:
                freq = 180 + 280 * limb.pulse_position
                self.out_freqs.append(float(freq))
                self.out_amps.append(float(0.035 * limb.pulse_amplitude))
        
        # Master volume
        total_E = np.sum(np.abs(body_psi)**2) + np.sum(torus_I)
        ramp = np.clip(self.step_count / 150.0, 0, 1)
        self.master_amp = np.clip(total_E / 1200.0, 0.02, 0.22) * (1 - 0.4*dream_depth) * ramp
    
    def get_relative_amplitude(self) -> float:
        """Get amplitude relative to recent history."""
        if len(self.amp_history) < 50:
            return 1.0
        return float(np.clip(self.amplitude / (np.mean(self.amp_history) + 1e-12), 0, 3))
    
    def get_audio_features(self) -> dict:
        """Return dict of all audio features for analysis."""
        return {
            'amplitude': self.amplitude,
            'bass': self.bass,
            'mid': self.mid,
            'treble': self.treble,
            'spectral_centroid': self.spectral_centroid,
            'spectral_flux': self.spectral_flux,
            'relative_amplitude': self.get_relative_amplitude(),
            'silence_frames': self.silence_frames,
        }


class DummyAudio:
    """Stub audio system when sounddevice unavailable."""
    
    def __init__(self):
        self.amplitude = 0.0
        self.bass = 0.0
        self.mid = 0.0
        self.treble = 0.0
        self.spectral_centroid = 0.0
        self.spectral_flux = 0.0
        self.silence_frames = 0
        self._world_freqs = []
        self._world_amps = []
    
    def start(self):
        print("[Audio] Disabled (sounddevice not available)")
    
    def stop(self):
        pass
    
    def update_sonification(self, *args, **kwargs):
        pass
    
    def get_relative_amplitude(self) -> float:
        return 1.0
    
    def get_audio_features(self) -> dict:
        return {
            'amplitude': 0.0,
            'bass': 0.0,
            'mid': 0.0,
            'treble': 0.0,
            'spectral_centroid': 0.0,
            'spectral_flux': 0.0,
            'relative_amplitude': 1.0,
            'silence_frames': 0,
        }
