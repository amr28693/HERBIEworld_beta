"""
Primordial Elements - Types, properties, and wavefunction resonance.

Each element has:
- Physical properties (density, reactivity, etc.)
- Terrain affinities (where it spawns)
- Wavefunction signatures (resonance patterns with creature body fields)

The resonance system creates individual affinities: some Herbies feel
compelled to pick up certain elements while others are repelled.
"""

from enum import Enum, auto
from typing import Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    pass  # For creature type hints


class ElementType(Enum):
    """Primordial elements found in the world."""
    ITE = auto()        # Denseite from mountains/hills - stackable, sparks
    ITE_LITE = auto()   # Fibrous lite from plains/forest - flexible, burns  
    ORE = auto()        # Metallic ore from shores/caves - conducts, malleable
    VAPOR = auto()      # Gaseous catalyst from water/rain - reactive, dissipates
    MULCHITE = auto()   # Organic mulch from forest floor - ferments, nutrient-rich
    LITE_ORE = auto()   # Volatile lite-ore from volcanic - energetic, dangerous


# Terrain affinities - which terrains spawn which elements
ELEMENT_TERRAIN_SOURCES: Dict[ElementType, set] = {
    ElementType.ITE: {'mountain', 'hill', 'cave', 'hills'},
    ElementType.ITE_LITE: {'plains', 'forest', 'shore'},
    ElementType.ORE: {'shore', 'water', 'cave', 'hills'},
    ElementType.VAPOR: {'water', 'swamp', 'shore'},
    ElementType.MULCHITE: {'forest', 'swamp', 'plains'},
    ElementType.LITE_ORE: {'cave', 'volcano', 'mountain', 'hills'},
}


@dataclass
class ElementProperties:
    """Physical properties of an element type."""
    density: float = 1.0
    reactivity: float = 0.5
    diffusion_rate: float = 0.1
    decay_rate: float = 0.001
    flammability: float = 0.0
    conductivity: float = 0.0
    volatility: float = 0.0
    stackable: bool = False
    color: str = 'gray'
    symbol: str = '?'


ELEMENT_PROPS: Dict[ElementType, ElementProperties] = {
    ElementType.ITE: ElementProperties(
        density=3.0, reactivity=0.2, diffusion_rate=0.02, decay_rate=0.0,
        flammability=0.0, conductivity=0.1, volatility=0.0,
        stackable=True, color='#7a7a7a', symbol='●'
    ),
    ElementType.ITE_LITE: ElementProperties(
        density=0.4, reactivity=0.6, diffusion_rate=0.15, decay_rate=0.000005,
        flammability=0.8, conductivity=0.0, volatility=0.1,
        stackable=False, color='#c4a574', symbol='≋'
    ),
    ElementType.ORE: ElementProperties(
        density=5.0, reactivity=0.3, diffusion_rate=0.01, decay_rate=0.0,
        flammability=0.0, conductivity=0.9, volatility=0.0,
        stackable=True, color='#b87333', symbol='⬢'
    ),
    ElementType.VAPOR: ElementProperties(
        density=0.05, reactivity=0.9, diffusion_rate=0.5, decay_rate=0.0008,
        flammability=0.0, conductivity=0.0, volatility=0.2,
        stackable=False, color='#aaddff', symbol='~'
    ),
    ElementType.MULCHITE: ElementProperties(
        density=0.8, reactivity=0.5, diffusion_rate=0.08, decay_rate=0.00001,
        flammability=0.4, conductivity=0.0, volatility=0.0,
        stackable=False, color='#4a3728', symbol='≈'
    ),
    ElementType.LITE_ORE: ElementProperties(
        density=2.5, reactivity=0.95, diffusion_rate=0.05, decay_rate=0.0,
        flammability=0.3, conductivity=0.7, volatility=0.9,
        stackable=False, color='#ff6b35', symbol='⚡'
    ),
}

# LITE_ORE is toxic to Apex predators
APEX_BANE_ELEMENT = ElementType.LITE_ORE
APEX_BANE_DAMAGE_MULTIPLIER = 5.0
APEX_BANE_FEAR_RADIUS = 8.0


@dataclass
class ElementSignature:
    """
    Wavefunction signature for an element type.
    These determine resonance patterns with creature body fields.
    """
    k_center: float = 1.0       # Preferred spatial frequency
    k_width: float = 0.5        # Bandwidth of resonance
    phase_mode: int = 0         # 0=radial, 1=spiral+, 2=spiral-, 3=linear
    phase_strength: float = 0.5
    amp_preference: str = 'core'  # 'core', 'limbs', 'diffuse', 'edge'
    freq_center: float = 0.0    # Preferred oscillation rate
    freq_sensitivity: float = 0.2


ELEMENT_SIGNATURES: Dict[ElementType, ElementSignature] = {
    # ITE (Stone) - likes stable, low-frequency, core-centered fields
    ElementType.ITE: ElementSignature(
        k_center=0.5, k_width=0.3,
        phase_mode=0, phase_strength=0.2,
        amp_preference='core',
        freq_center=0.0, freq_sensitivity=0.1
    ),
    
    # ITE_LITE (Fiber/Wood) - likes moderate activity, distributed fields
    ElementType.ITE_LITE: ElementSignature(
        k_center=1.5, k_width=0.8,
        phase_mode=1, phase_strength=0.4,
        amp_preference='limbs',
        freq_center=0.3, freq_sensitivity=0.3
    ),
    
    # ORE (Metal) - likes high-frequency, sharp, edge-concentrated fields
    ElementType.ORE: ElementSignature(
        k_center=2.5, k_width=0.4,
        phase_mode=3, phase_strength=0.6,
        amp_preference='edge',
        freq_center=0.5, freq_sensitivity=0.2
    ),
    
    # VAPOR (Mist) - likes diffuse, low-amplitude, high-frequency fields
    ElementType.VAPOR: ElementSignature(
        k_center=3.0, k_width=1.5,
        phase_mode=2, phase_strength=0.3,
        amp_preference='diffuse',
        freq_center=0.8, freq_sensitivity=0.5
    ),
    
    # MULCHITE (Organic matter) - likes warm, medium, growing patterns
    ElementType.MULCHITE: ElementSignature(
        k_center=1.0, k_width=1.0,
        phase_mode=1, phase_strength=0.5,
        amp_preference='core',
        freq_center=0.2, freq_sensitivity=0.4
    ),
    
    # LITE_ORE (Volatile Crystal) - likes chaotic, high-energy fields
    ElementType.LITE_ORE: ElementSignature(
        k_center=2.0, k_width=1.2,
        phase_mode=2, phase_strength=0.8,
        amp_preference='edge',
        freq_center=1.0, freq_sensitivity=0.6
    ),
}


def compute_herbie_spectral_signature(herbie) -> Optional[dict]:
    """
    Compute a Herbie's current wavefunction signature.
    
    Returns dict with:
    - k_spectrum: Power at different spatial frequencies
    - phase_pattern: Dominant phase structure
    - amp_distribution: Where amplitude is concentrated
    - oscillation_rate: How fast the field is changing
    """
    if not hasattr(herbie, 'body') or herbie.body is None:
        return None
    
    psi = herbie.body.psi
    I = np.abs(psi)**2
    phase = np.angle(psi)
    
    # Spatial frequency spectrum
    psi_k = np.fft.fft2(psi)
    power_spectrum = np.abs(psi_k)**2
    
    # Radial averaging
    kx = np.fft.fftfreq(psi.shape[1])
    ky = np.fft.fftfreq(psi.shape[0])
    KX, KY = np.meshgrid(kx, ky)
    K_mag = np.sqrt(KX**2 + KY**2)
    
    k_bins = np.linspace(0, 0.5, 10)
    k_spectrum = np.zeros(len(k_bins) - 1)
    for i in range(len(k_bins) - 1):
        mask = (K_mag >= k_bins[i]) & (K_mag < k_bins[i+1])
        if np.sum(mask) > 0:
            k_spectrum[i] = np.mean(power_spectrum[mask])
    
    k_spectrum = k_spectrum / (np.sum(k_spectrum) + 1e-10)
    
    # Phase pattern analysis
    dphi_dx = np.gradient(phase, axis=1)
    dphi_dy = np.gradient(phase, axis=0)
    I_norm = I / (np.sum(I) + 1e-10)
    
    curl = dphi_dx - dphi_dy
    spiral_score = float(np.sum(I_norm * curl))
    
    div = np.gradient(dphi_dx, axis=1) + np.gradient(dphi_dy, axis=0)
    radial_score = float(np.abs(np.sum(I_norm * div)))
    
    grad_consistency = np.std(dphi_dx) + np.std(dphi_dy)
    linear_score = 1.0 / (1.0 + grad_consistency)
    
    # Amplitude distribution
    total_I = np.sum(I) + 1e-10
    center = np.array(psi.shape) / 2
    Y_idx, X_idx = np.ogrid[:psi.shape[0], :psi.shape[1]]
    r_from_center = np.sqrt((X_idx - center[1])**2 + (Y_idx - center[0])**2)
    r_max = min(center)
    
    core_mask = r_from_center < r_max * 0.4
    edge_mask = r_from_center > r_max * 0.7
    
    core_frac = np.sum(I[core_mask]) / total_I
    edge_frac = np.sum(I[edge_mask]) / total_I
    diffuse_frac = 1.0 - core_frac - edge_frac
    
    # Oscillation rate from torus arousal
    osc_rate = 0.0
    if hasattr(herbie, 'torus') and herbie.torus is not None:
        osc_rate = herbie.torus.get_arousal() * 0.5
    
    return {
        'k_spectrum': k_spectrum,
        'k_center': float(np.sum(k_spectrum * (k_bins[:-1] + k_bins[1:]) / 2)),
        'spiral_score': spiral_score,
        'radial_score': radial_score,
        'linear_score': linear_score,
        'core_frac': core_frac,
        'edge_frac': edge_frac,
        'diffuse_frac': diffuse_frac,
        'osc_rate': osc_rate,
    }


def compute_element_resonance(herbie, element_type: ElementType) -> float:
    """
    Compute resonance between a Herbie's wavefunction and an element's signature.
    
    Returns value from -1 (repulsion) to +1 (strong attraction).
    """
    sig = ELEMENT_SIGNATURES.get(element_type)
    if sig is None:
        return 0.0
    
    herbie_sig = compute_herbie_spectral_signature(herbie)
    if herbie_sig is None:
        return 0.0
    
    resonance = 0.0
    
    # 1. Spatial frequency match
    k_match = np.exp(-((herbie_sig['k_center'] - sig.k_center)**2) / (2 * sig.k_width**2))
    resonance += 0.3 * k_match
    
    # 2. Phase pattern match
    if sig.phase_mode == 0:  # Radial
        phase_match = herbie_sig['radial_score']
    elif sig.phase_mode == 1:  # Spiral+
        phase_match = max(0, herbie_sig['spiral_score'])
    elif sig.phase_mode == 2:  # Spiral-
        phase_match = max(0, -herbie_sig['spiral_score'])
    else:  # Linear
        phase_match = herbie_sig['linear_score']
    
    resonance += 0.25 * sig.phase_strength * np.tanh(phase_match * 3)
    
    # 3. Amplitude distribution match
    if sig.amp_preference == 'core':
        amp_match = herbie_sig['core_frac']
    elif sig.amp_preference == 'edge':
        amp_match = herbie_sig['edge_frac']
    elif sig.amp_preference == 'diffuse':
        amp_match = herbie_sig['diffuse_frac']
    else:  # limbs
        amp_match = herbie_sig['edge_frac'] * 0.7 + herbie_sig['diffuse_frac'] * 0.3
    
    resonance += 0.25 * amp_match
    
    # 4. Oscillation frequency match
    osc_match = np.exp(-((herbie_sig['osc_rate'] - sig.freq_center)**2) / (2 * sig.freq_sensitivity**2))
    resonance += 0.2 * osc_match
    
    # Scale to [-1, 1]
    resonance = (resonance - 0.4) * 2.5
    
    return float(np.clip(resonance, -1.0, 1.0))


def compute_object_resonance(herbie, obj) -> float:
    """
    Compute resonance with any world object.
    For ElementObjects uses full resonance; others use simplified version.
    """
    # Check if it's an ElementObject (avoid circular import)
    if hasattr(obj, 'element_type'):
        return compute_element_resonance(herbie, obj.element_type)
    
    herbie_sig = compute_herbie_spectral_signature(herbie)
    if herbie_sig is None:
        return 0.0
    
    resonance = 0.0
    
    if hasattr(obj, 'compliance'):
        if obj.compliance > 0.6:
            resonance += 0.3 * herbie_sig['diffuse_frac'] - 0.1 * herbie_sig['edge_frac']
        else:
            resonance += 0.3 * herbie_sig['edge_frac'] - 0.1 * herbie_sig['diffuse_frac']
    
    if hasattr(obj, 'size'):
        if obj.size > 1.5:
            resonance += 0.2 * (1.0 - herbie_sig['k_center'] * 2)
        elif obj.size < 0.8:
            resonance += 0.2 * (herbie_sig['k_center'] * 2 - 0.5)
    
    if hasattr(obj, 'mass') and obj.mass > 3.0:
        resonance += 0.2 * herbie_sig['core_frac']
    
    return float(np.clip(resonance, -1.0, 1.0))


def get_herbie_element_affinities(herbie) -> Dict[ElementType, float]:
    """Get a Herbie's current affinities for all element types."""
    return {
        element_type: compute_element_resonance(herbie, element_type)
        for element_type in ElementType
    }
