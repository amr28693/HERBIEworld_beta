"""
Torus Brain - Ring NLSE central pattern generator.

The torus brain is the core neural dynamics of HERBIE creatures.
It processes afferent (sensory) signals and generates efferent (motor) patterns.
"""

from typing import List, Tuple
import numpy as np

from ..core.constants import (
    N_torus, theta_torus, L_op_torus, 
    G_SMOOTH, dt
)


class TorusBrain:
    """
    Ring NLSE as central pattern generator (CPG).
    
    The torus brain processes afferent signals and generates efferent patterns.
    Key metrics:
    - Circulation: net phase winding (direction preference)
    - Coherence: concentration of |ψ|² (focus/attention)
    - Arousal: combined activity level
    """
    
    def __init__(self):
        """Initialize torus brain with random state."""
        # Random initial state
        random_phase = np.random.uniform(0, 2*np.pi, N_torus)
        random_amp = 0.3 + 0.2 * np.random.rand(N_torus)
        self.psi = random_amp * np.exp(1j * random_phase)
        
        # Add random localized bump
        bump_loc = np.random.uniform(0, 2*np.pi)
        self.psi += 0.15 * np.exp(-((theta_torus - bump_loc)**2) / 0.6)
        
        self.g = 1.0           # Nonlinearity strength
        self.circulation = 0.0  # Net phase winding
        self.coherence = 0.0    # Focus/attention metric
        self.energy = 0.0       # Total field energy
        
        # Afferent input accumulator
        self.afferent_input = np.zeros(N_torus)
        
        # Statistics
        self.efferent_fire_count = 0
        
    def receive_afferent(self, channel_name: str, arrivals: List[float]):
        """
        Process incoming sensory signals.
        Different channels map to different torus locations.
        
        Args:
            channel_name: Name of the afferent channel (e.g., 'touch_0', 'env_reward')
            arrivals: List of signal amplitudes that arrived
        """
        for amp in arrivals:
            # Map channel to torus location
            if 'touch' in channel_name:
                # Touch channels map to specific sectors
                touch_idx = int(channel_name.split('_')[1]) if '_' in channel_name else 0
                idx = (touch_idx * N_torus // 3) % N_torus
            elif 'env_reward' in channel_name:
                # Reward maps to "front" of torus
                idx = 0
            elif 'env_pain' in channel_name:
                # Pain maps to opposite side
                idx = N_torus // 2
            elif 'proprioception' in channel_name:
                # Proprioception distributed
                idx = np.random.randint(N_torus)
            else:
                idx = hash(channel_name) % N_torus
                
            # Spread input across neighboring nodes
            for di in range(-3, 4):
                self.afferent_input[(idx + di) % N_torus] += amp * 0.15 * np.exp(-di**2/2)
                
    def compute_metrics(self):
        """Compute circulation, coherence, energy."""
        phase = np.angle(self.psi)
        dph = np.diff(phase, append=phase[0])
        dph = np.where(dph > np.pi, dph - 2*np.pi,
                      np.where(dph < -np.pi, dph + 2*np.pi, dph))
        raw_circ = float(np.sum(dph) / (2*np.pi))
        
        # Clamp circulation to reasonable range
        clamped_circ = np.clip(raw_circ, -1.5, 1.5)
        
        # Smooth AND decay toward zero to prevent persistent directional bias
        self.circulation = 0.85 * self.circulation + 0.1 * clamped_circ
        # Decay toward zero
        self.circulation *= 0.98
        
        I = np.abs(self.psi)**2
        I_norm = I / (np.sum(I) + 1e-12)
        self.coherence = float(np.sum(I_norm**2) * N_torus)
        self.energy = float(np.sum(I))
        
    def get_arousal(self) -> float:
        """
        Combined arousal metric (0-1).
        
        Higher arousal = more active, alert state.
        """
        return float(np.clip(
            (self.coherence / 8.0) * (0.5 + 0.5*np.tanh(abs(self.circulation))),
            0, 1
        ))
        
    def get_directional_bias(self) -> np.ndarray:
        """
        Get unit vector from density concentration (circular mean).
        
        Returns:
            np.ndarray: 2D direction vector with magnitude representing bias strength
        """
        I = np.abs(self.psi)**2
        I_norm = I / (np.sum(I) + 1e-12)
        
        mean_x = float(np.sum(I_norm * np.cos(theta_torus)))
        mean_y = float(np.sum(I_norm * np.sin(theta_torus)))
        
        concentration = np.sqrt(mean_x**2 + mean_y**2)
        
        if concentration < 0.03:  # Higher threshold - need stronger signal
            return np.zeros(2)
            
        avg_angle = np.arctan2(mean_y, mean_x)
        # Much weaker bias - let food seeking dominate
        bias_strength = np.clip(abs(self.circulation) * 0.1 + concentration * 0.25, 0, 0.35)
        
        # Add random walk to prevent persistent drift
        avg_angle += np.random.uniform(-0.4, 0.4)
        
        return np.array([np.cos(avg_angle), np.sin(avg_angle)]) * bias_strength
    
    def inject_directional_bias(self, direction: np.ndarray, strength: float = 0.3):
        """
        Inject external directional bias (e.g., from player input or placement memory).
        
        Args:
            direction: 2D direction vector
            strength: Injection strength
        """
        if np.linalg.norm(direction) < 0.01:
            return
        
        # Convert direction to angle on torus
        target_angle = np.arctan2(direction[1], direction[0])
        
        # Inject energy at that angle
        for i in range(N_torus):
            # Gaussian around target angle
            angle_diff = theta_torus[i] - target_angle
            # Wrap to [-pi, pi]
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            
            injection = strength * np.exp(-angle_diff**2 / 0.5)
            self.psi[i] += injection * np.exp(1j * theta_torus[i])
        
    def should_fire_efferent(self) -> bool:
        """Determine if efferent signals should be sent."""
        return self.get_arousal() > 0.15 or np.random.random() < 0.08
        
    def get_efferent_pattern(self) -> List[Tuple[int, float]]:
        """
        Get pattern for efferent firing based on torus state.
        
        Returns:
            List of (index, amplitude) tuples for efferent channels
        """
        I = np.abs(self.psi)**2
        patterns = []
        
        # Find local maxima
        for i in range(N_torus):
            if I[i] > I[(i-1) % N_torus] and I[i] > I[(i+1) % N_torus]:
                patterns.append((i, I[i]))
                
        # Fallback: use highest values
        if len(patterns) < 2:
            sorted_idx = np.argsort(I)[::-1]
            for idx in sorted_idx[:3]:
                patterns.append((idx, I[idx]))
                
        return patterns[:3]
        
    def inject_reward(self, world_angle: float, strength: float):
        """
        Bias torus toward rewarding direction.
        
        Args:
            world_angle: Direction of reward in world space (radians)
            strength: Injection strength
        """
        torus_idx = int((world_angle + np.pi) / (2*np.pi) * N_torus) % N_torus
        for di in range(-4, 5):
            idx = (torus_idx + di) % N_torus
            self.psi[idx] += strength * 0.12 * np.exp(1j * world_angle) * np.exp(-di**2/3)
            
    def inject_aversion(self, world_angle: float, strength: float):
        """
        Bias torus away from aversive direction.
        
        Args:
            world_angle: Direction of threat in world space (radians)
            strength: Aversion strength
        """
        # Inject in opposite direction - STRONGER to actually flee
        opposite_angle = world_angle + np.pi
        torus_idx = int((opposite_angle + np.pi) / (2*np.pi) * N_torus) % N_torus
        for di in range(-5, 6):
            idx = (torus_idx + di) % N_torus
            self.psi[idx] += strength * 0.25 * np.exp(1j * opposite_angle) * np.exp(-di**2/4)
            
    def evolve(self, audio_amp: float, hunger: float, dream_depth: float):
        """
        Evolve torus NLSE dynamics.
        
        Args:
            audio_amp: Current audio amplitude (increases activity)
            hunger: Hunger level (decreases activity)
            dream_depth: Sleep/dream depth (decreases activity)
        """
        # Target g (nonlinearity)
        g_target = 1.2
        g_target += 0.6 * audio_amp
        g_target -= 0.5 * hunger
        g_target -= 1.2 * dream_depth
        self.g = G_SMOOTH * self.g + (1 - G_SMOOTH) * g_target
        
        # Process afferent input
        self.psi += self.afferent_input * np.exp(1j * np.angle(self.psi))
        self.afferent_input *= 0.3  # Decay
        
        # Spontaneous activity
        if np.random.random() < 0.15:
            idx = np.random.randint(N_torus)
            self.psi[idx] += 0.2 * np.exp(1j * np.random.uniform(0, 2*np.pi))
        
        # Split-step NLSE
        psi_k = np.fft.fft(self.psi) * L_op_torus
        self.psi = np.fft.ifft(psi_k)
        self.psi *= np.exp(1j * self.g * dt * np.abs(self.psi)**2)
        
        # Dissipation
        self.psi *= 0.996
        
        # Energy regulation
        E = np.sum(np.abs(self.psi)**2)
        target_E = 5.0 * (1 + 0.4 * self.get_arousal())
        if E > 1e-9:
            self.psi *= np.clip((target_E / E) ** 0.06, 0.94, 1.06)
            
        self.compute_metrics()
    
    def get_stats(self) -> dict:
        """Return brain statistics dictionary."""
        return {
            'energy': self.energy,
            'coherence': self.coherence,
            'circulation': self.circulation,
            'arousal': self.get_arousal(),
            'g': self.g,
            'efferent_fires': self.efferent_fire_count,
        }
    
    def to_dict(self) -> dict:
        """Serialize brain state for persistence."""
        return {
            'psi_real': self.psi.real.tolist(),
            'psi_imag': self.psi.imag.tolist(),
            'g': self.g,
            'circulation': self.circulation,
            'coherence': self.coherence,
            'energy': self.energy,
            'efferent_fire_count': self.efferent_fire_count,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TorusBrain':
        """Deserialize brain state from persistence."""
        brain = cls.__new__(cls)
        brain.psi = np.array(d['psi_real']) + 1j * np.array(d['psi_imag'])
        brain.g = d.get('g', 1.0)
        brain.circulation = d.get('circulation', 0.0)
        brain.coherence = d.get('coherence', 0.0)
        brain.energy = d.get('energy', 0.0)
        brain.afferent_input = np.zeros(N_torus)
        brain.efferent_fire_count = d.get('efferent_fire_count', 0)
        return brain


class GhostTorus:
    """
    A torus wavefunction cut free from its body.
    
    Pure NLSE via split-step FFT. No artificial limits.
    It does whatever it wants.
    """
    
    def __init__(self, psi: np.ndarray, pos: np.ndarray, name: str = None,
                 circulation: float = 0.0, energy: float = 0.0, g: float = -0.5):
        """
        Cut the torus free.
        """
        self.psi = psi.copy()
        self.pos = pos.copy()
        self.vel = np.zeros(2)
        
        self.name = name
        self.initial_circulation = circulation
        self.initial_energy = energy
        self.g = g
        
        self.age = 0
        self.alive = True
    
    def update(self, terrain: 'Terrain', other_ghosts: list = None) -> bool:
        """
        Pure NLSE via split-step Fourier method.
        No artificial death. It lives until it doesn't.
        """
        if not self.alive:
            return False
        
        self.age += 1
        
        # === SPLIT-STEP FOURIER METHOD ===
        N = len(self.psi)
        dt = 0.05
        
        # Wavenumbers
        k = np.fft.fftfreq(N, d=1.0/N) * 2 * np.pi
        
        # Half-step nonlinear
        nonlinear_phase = np.exp(-1j * self.g * np.abs(self.psi)**2 * dt / 2)
        self.psi = self.psi * nonlinear_phase
        
        # Full-step linear (Fourier space)
        psi_k = np.fft.fft(self.psi)
        linear_phase = np.exp(-1j * 0.5 * k**2 * dt)
        psi_k = psi_k * linear_phase
        self.psi = np.fft.ifft(psi_k)
        
        # Half-step nonlinear
        nonlinear_phase = np.exp(-1j * self.g * np.abs(self.psi)**2 * dt / 2)
        self.psi = self.psi * nonlinear_phase
        
        # === MOVEMENT FROM CIRCULATION ===
        phase = np.angle(self.psi)
        dphase = np.diff(np.unwrap(phase))
        circulation = np.mean(dphase)
        
        speed = min(0.15, np.abs(circulation) * 0.03)
        angle = self.age * circulation * 0.1
        
        self.vel = self.vel * 0.98 + np.array([np.cos(angle), np.sin(angle)]) * speed
        self.pos += self.vel
        
        # Periodic boundaries
        half = terrain.world_size / 2
        self.pos = ((self.pos + half) % terrain.world_size) - half
        
        # === SINGULARITY DETECTION ===
        # NaN or Inf means the wavefunction collapsed to a singularity
        # This is physically meaningful - not an error
        if np.any(np.isnan(self.psi)) or np.any(np.isinf(self.psi)):
            self.alive = False
            self.cause_of_death = "singularity"
            return False
        
        return True
    
    def _compute_coherence(self) -> float:
        """Coherence for visualization only."""
        power = np.abs(self.psi)**2
        total = np.sum(power)
        if total < 1e-10:
            return 0.0
        power_norm = power / total
        entropy = -np.sum(power_norm * np.log(power_norm + 1e-10)) / np.log(len(self.psi))
        return 1.0 - entropy
    
    def get_circulation(self) -> float:
        """Current circulation."""
        phase = np.angle(self.psi)
        return np.sum(np.diff(np.unwrap(phase))) / (2 * np.pi)
    
    def get_energy(self) -> float:
        """Total field energy."""
        return np.sum(np.abs(self.psi)**2)
    
    def get_influence_at(self, pos: np.ndarray) -> float:
        """Ghost's influence at a position (for creature sensing)."""
        dist = np.linalg.norm(pos - self.pos)
        if dist > 10.0:
            return 0.0
        coherence = self._compute_coherence()
        return coherence * np.exp(-dist**2 / 20.0)
    
    def get_energy_for_terrain(self) -> float:
        """Energy to return to terrain on dispersal."""
        return np.sum(np.abs(self.psi)**2) * self.initial_energy * 0.1


class GhostField:
    """
    Manages all ghost tori in the world.
    
    Tracks persisting wavefunctions, updates their dynamics,
    and handles their eventual dispersal back into the world.
    """
    
    def __init__(self):
        self.ghosts: list[GhostTorus] = []
        self.max_ghosts = 50  # Limit for performance
        self.total_dispersed = 0
    
    def spawn_ghost(self, brain: 'TorusBrain', pos: np.ndarray, 
                    name: str = None, energy: float = 0.0):
        """
        Cut a torus free from its dying body.
        """
        if len(self.ghosts) >= self.max_ghosts:
            # Remove oldest ghost
            if self.ghosts:
                oldest = min(self.ghosts, key=lambda g: -g.age)
                self.ghosts.remove(oldest)
        
        ghost = GhostTorus(
            psi=brain.psi,  # The actual wavefunction
            pos=pos,
            name=name,
            circulation=brain.circulation,
            energy=energy,
            g=brain.g  # Same nonlinearity
        )
        self.ghosts.append(ghost)
        
        if name:
            print(f"[Ghost] 👻 {name}'s torus cut free...")
    
    def update(self, terrain: 'Terrain'):
        """Update all ghosts - pure NLSE, no interaction."""
        dispersing = []
        
        for ghost in self.ghosts:
            try:
                still_exists = ghost.update(terrain, None)
                if not still_exists:
                    dispersing.append(ghost)
            except (FloatingPointError, ValueError, OverflowError):
                # Singularity - wavefunction collapsed
                ghost.alive = False
                ghost.cause_of_death = "singularity"
                dispersing.append(ghost)
        
        # Handle dispersal
        for ghost in dispersing:
            cause = getattr(ghost, 'cause_of_death', 'dispersal')
            
            if cause == "singularity":
                if ghost.name:
                    print(f"[Ghost] ⚫ {ghost.name} collapsed to singularity")
                # Singularity still deposits energy where it was last valid
                if not np.any(np.isnan(ghost.pos)):
                    terrain.add_death_site(ghost.pos, ghost.initial_energy * 0.5)
            else:
                if not np.any(np.isnan(ghost.pos)):
                    energy = ghost.get_energy_for_terrain()
                    terrain.add_death_site(ghost.pos, energy * 10)
                    if ghost.name:
                        print(f"[Ghost] 🌫️ {ghost.name} dispersed into the earth")
            
            self.total_dispersed += 1
        
        self.ghosts = [g for g in self.ghosts if g.alive]
    
    def get_congregation_centers(self, min_ghosts: int = 2) -> list:
        """Find centers where ghosts have congregated."""
        if len(self.ghosts) < min_ghosts:
            return []
        
        # Simple clustering - find groups within radius
        centers = []
        visited = set()
        
        for i, ghost in enumerate(self.ghosts):
            if i in visited:
                continue
            
            # Find nearby ghosts
            cluster = [ghost]
            cluster_indices = {i}
            
            for j, other in enumerate(self.ghosts):
                if j != i and j not in visited:
                    if np.linalg.norm(other.pos - ghost.pos) < 10.0:
                        cluster.append(other)
                        cluster_indices.add(j)
            
            if len(cluster) >= min_ghosts:
                # Compute center of mass weighted by coherence
                total_weight = sum(g._compute_coherence() for g in cluster)
                if total_weight > 0:
                    center = sum(g.pos * g._compute_coherence() for g in cluster) / total_weight
                    centers.append({
                        'pos': center,
                        'count': len(cluster),
                        'total_coherence': total_weight,
                        'names': [g.name for g in cluster if g.name]
                    })
                visited.update(cluster_indices)
        
        return centers
