"""
Color utilities for visualization.

Provides HSV-based complex field coloring, terrain colors,
species colors, and day/night color adjustments.
"""

import numpy as np
import matplotlib.colors as mcolors
from typing import Tuple, Optional


def complex_to_rgb(psi: np.ndarray, amp_gamma: float = 0.8) -> np.ndarray:
    """
    Convert complex field to RGB using HSV color mapping.
    
    Phase → Hue (cyclic color wheel)
    Amplitude → Value (brightness)
    
    Args:
        psi: Complex wavefunction array
        amp_gamma: Gamma correction for amplitude
        
    Returns:
        RGB array with shape (*psi.shape, 3)
    """
    amp = np.abs(psi)
    amp_n = np.clip(amp / (np.percentile(amp, 99) + 1e-12), 0, 1) ** amp_gamma
    hue = (np.angle(psi) + np.pi) / (2 * np.pi)
    
    hsv = np.zeros((*psi.shape, 3))
    hsv[..., 0] = hue
    hsv[..., 1] = 0.85
    hsv[..., 2] = amp_n
    
    return mcolors.hsv_to_rgb(hsv)


def get_species_color(species_name: str) -> str:
    """Get display color for a species."""
    colors = {
        'Herbie': '#00CED1',      # Dark cyan
        'Blob': '#98FB98',        # Pale green
        'Apex': '#DC143C',        # Crimson
        'Grazer': '#90EE90',      # Light green
        'Scavenger': '#DAA520',   # Goldenrod
        'Drifter': '#ADD8E6',     # Light blue
        'Burrower': '#DEB887',    # Burlywood
        'Floater': '#E6E6FA',     # Lavender
    }
    return colors.get(species_name, '#FFFFFF')


def get_terrain_color(terrain_name: str) -> Tuple[float, float, float]:
    """Get RGB color for a terrain type."""
    colors = {
        'water': (0.1, 0.3, 0.6),
        'shore': (0.8, 0.7, 0.5),
        'plains': (0.5, 0.7, 0.3),
        'forest': (0.2, 0.5, 0.2),
        'hills': (0.6, 0.5, 0.3),
        'mountain': (0.5, 0.5, 0.5),
        'cave': (0.2, 0.2, 0.3),
    }
    return colors.get(terrain_name.lower(), (0.4, 0.4, 0.4))


def get_terrain_color_rgba(terrain_name: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    """Get RGBA color for terrain with alpha."""
    r, g, b = get_terrain_color(terrain_name)
    return (r, g, b, alpha)


def get_sky_color_for_time(time_of_day: float, is_day: bool) -> str:
    """
    Get sky background color based on time of day.
    
    Args:
        time_of_day: 0.0-1.0, where 0.5 is noon
        is_day: Whether it's currently day
        
    Returns:
        Hex color string
    """
    if is_day:
        # Day: light blue fading toward dawn/dusk
        if time_of_day < 0.3:  # Dawn
            return '#2a3f5f'
        elif time_of_day > 0.7:  # Dusk
            return '#4a2f3f'
        else:  # Midday
            return '#1a2f4f'
    else:
        # Night: deep blue/black
        return '#0a0a15'


def apply_day_night_tint(base_color: Tuple[float, float, float], 
                         light_level: float) -> Tuple[float, float, float]:
    """
    Apply day/night lighting to a base color.
    
    Args:
        base_color: RGB tuple (0-1 each)
        light_level: 0.0 (night) to 1.0 (day)
        
    Returns:
        Tinted RGB tuple
    """
    # At night, shift toward blue and reduce brightness
    night_tint = (0.1, 0.1, 0.3)
    
    r = base_color[0] * light_level + night_tint[0] * (1 - light_level)
    g = base_color[1] * light_level + night_tint[1] * (1 - light_level)
    b = base_color[2] * light_level + night_tint[2] * (1 - light_level)
    
    # Reduce overall brightness at night
    brightness = 0.3 + 0.7 * light_level
    
    return (r * brightness, g * brightness, b * brightness)


def get_daynight_overlay_alpha(light_level: float) -> float:
    """Get alpha for night overlay based on light level."""
    return max(0.0, (0.5 - light_level) * 0.6)


def creature_state_color(is_hibernating: bool = False, 
                         is_digesting: bool = False,
                         is_defending: bool = False,
                         is_selected: bool = False) -> Tuple[str, float, float]:
    """
    Get edge color, alpha, and line width for creature state.
    
    Returns:
        (edge_color, alpha, line_width)
    """
    if is_hibernating:
        return ('lightblue', 0.3, 1.0)
    elif is_digesting:
        return ('orange', 0.5, 1.0)
    elif is_defending:
        return ('red', 0.8, 2.0)
    elif is_selected:
        return ('yellow', 0.7, 2.0)
    else:
        return ('white', 0.4, 0.5)


def interpolate_color(color1: Tuple[float, float, float], 
                      color2: Tuple[float, float, float],
                      t: float) -> Tuple[float, float, float]:
    """Linear interpolation between two RGB colors."""
    t = np.clip(t, 0.0, 1.0)
    return (
        color1[0] * (1 - t) + color2[0] * t,
        color1[1] * (1 - t) + color2[1] * t,
        color1[2] * (1 - t) + color2[2] * t,
    )


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert hex color string to RGB tuple (0-1 each)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    """Convert RGB tuple (0-1 each) to hex color string."""
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )


# Leviathan colors
LEVIATHAN_COLOR_BASE = '#8B0000'  # Dark red
LEVIATHAN_COLOR_GLOW = '#FF4500'  # Orange-red
LEVIATHAN_COLOR_GENESIS = '#FF2200'  # Bright fire


# Weather event colors
WEATHER_COLORS = {
    'rain': '#4169E1',      # Royal blue
    'drought': '#8B4513',   # Saddle brown
    'storm': '#800080',     # Purple
    'bloom': '#FFD700',     # Gold
}


def get_weather_color(event_type: str) -> str:
    """Get color for weather event type."""
    return WEATHER_COLORS.get(event_type, '#FFFFFF')
