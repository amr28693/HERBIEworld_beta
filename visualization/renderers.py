"""
Renderers - Functions for drawing individual visual elements.

Each renderer takes an axes object and draws specific elements.
These are composable building blocks for the full visualization.
"""

import numpy as np
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
from typing import List, Optional, Tuple, TYPE_CHECKING

from .colors import (
    get_terrain_color, creature_state_color, get_weather_color,
    LEVIATHAN_COLOR_BASE, LEVIATHAN_COLOR_GLOW, LEVIATHAN_COLOR_GENESIS,
    apply_day_night_tint
)

if TYPE_CHECKING:
    from ..creature.creature import Creature
    from ..world.terrain import Terrain
    from ..ecology.ants import AntColony
    from ..ecology.leviathan import LeviathanSystem


# =============================================================================
# TERRAIN RENDERING
# =============================================================================

def render_terrain_background(ax, terrain: 'Terrain', world_size: float,
                               light_level: float = 1.0) -> np.ndarray:
    """
    Render terrain as background image.
    
    Args:
        ax: Matplotlib axes
        terrain: Terrain object
        world_size: Size of world in units
        light_level: Day/night light level (0-1)
        
    Returns:
        RGB image array
    """
    resolution = terrain.resolution
    img = np.zeros((resolution, resolution, 3))
    
    for i in range(resolution):
        for j in range(resolution):
            t = terrain.terrain_grid[i, j]
            base_color = get_terrain_color(t.name)
            
            # Apply day/night tinting
            tinted = apply_day_night_tint(base_color, light_level)
            img[j, i] = tinted
    
    return img


# =============================================================================
# CREATURE RENDERING
# =============================================================================

def render_creature(ax, creature: 'Creature', selected: bool = False,
                    light_level: float = 1.0, show_label: bool = True):
    """
    Render a single creature with limbs and state indicators.
    
    Args:
        ax: Matplotlib axes
        creature: Creature to render
        selected: Whether this creature is selected
        light_level: Day/night light level
        show_label: Whether to show generation label
    """
    if not creature.alive:
        return
    
    # Get state-dependent styling
    is_hibernating = getattr(creature, 'is_hibernating', False)
    is_digesting = getattr(creature, 'is_digesting', False)
    is_defending = getattr(creature, 'is_defending', False)
    
    ec, alpha, lw = creature_state_color(
        is_hibernating, is_digesting, is_defending, selected
    )
    
    # Body color
    base_color = creature.species.color_base
    size = creature.species.body_scale * 2.0
    
    # Main body circle
    ax.add_patch(Circle(
        creature.pos, size, fc=base_color, ec=ec, 
        alpha=alpha * light_level, lw=lw
    ))
    
    # Limbs
    for limb_name in getattr(creature, 'limb_defs', {}):
        origin_name = creature.limb_defs[limb_name][0]
        if origin_name in creature.basins:
            origin = creature.basins[origin_name]['pos']
            angle = creature.morph.limb_angles.get(limb_name, 0)
            limb_len = creature.species.limb_length
            
            tip_x = creature.pos[0] + origin[0] + limb_len * np.cos(angle)
            tip_y = creature.pos[1] + origin[1] + limb_len * np.sin(angle)
            
            ax.plot(
                [creature.pos[0] + origin[0], tip_x],
                [creature.pos[1] + origin[1], tip_y],
                color=base_color, lw=2, alpha=alpha * 0.8
            )
    
    # Velocity arrow
    if np.linalg.norm(creature.display_vel) > 0.1 and not is_hibernating:
        ax.arrow(
            creature.pos[0], creature.pos[1],
            creature.display_vel[0] * 2, creature.display_vel[1] * 2,
            head_width=0.5, fc='yellow', ec='yellow', alpha=0.5
        )
    
    # Label
    if show_label:
        label = f'G{creature.generation}'
        if is_hibernating:
            label += 'z'
        if is_digesting:
            label += '*'
        
        ax.text(
            creature.pos[0], creature.pos[1] + size + 0.5,
            label, color=base_color, fontsize=6, ha='center'
        )


def render_herbie_special(ax, herbie, light_level: float = 1.0):
    """
    Render Herbie with additional features (hands, mating state, mood).
    
    Args:
        ax: Matplotlib axes
        herbie: HerbieWithHands creature
        light_level: Day/night light level
    """
    if not herbie.alive:
        return
    
    pos = herbie.pos
    size = herbie.species.body_scale * 2.0
    
    # Mating state indicator
    if hasattr(herbie, 'mating_state'):
        ms = herbie.mating_state
        
        # Name above head
        if ms.name:
            ax.text(
                pos[0], pos[1] + size + 1.5,
                ms.name, color='white', fontsize=7, ha='center',
                fontweight='bold'
            )
        
        # Bond indicator (heart) - use text instead of invalid marker
        if ms.mate_id or ms.mate_ids:
            ax.text(pos[0] + size + 0.5, pos[1] + size, '♥', 
                   color='red', fontsize=10, ha='center', va='center')
        
        # Pregnant indicator
        if ms.is_pregnant:
            ax.add_patch(Circle(
                pos, size * 1.2, fc='none', ec='pink', 
                lw=2, linestyle='--', alpha=0.7
            ))
    
    # Hands - show held objects
    if hasattr(herbie, 'hands') and herbie.hands:
        hands = herbie.hands
        
        # Left hand
        if hands.left.held_object:
            obj = hands.left.held_object
            hand_pos = pos + np.array([-size * 0.8, 0])
            ax.add_patch(Circle(
                hand_pos, 0.4, fc=getattr(obj, 'color', '#888888'),
                ec='white', alpha=0.8
            ))
        
        # Right hand
        if hands.right.held_object:
            obj = hands.right.held_object
            hand_pos = pos + np.array([size * 0.8, 0])
            ax.add_patch(Circle(
                hand_pos, 0.4, fc=getattr(obj, 'color', '#888888'),
                ec='white', alpha=0.8
            ))


# =============================================================================
# FOOD AND OBJECT RENDERING
# =============================================================================

def render_food(ax, objects: List, light_level: float = 1.0):
    """
    Render food objects.
    
    Args:
        ax: Matplotlib axes
        objects: List of WorldObjects
        light_level: Day/night light level
    """
    food_alpha = 0.5 + 0.4 * light_level
    
    for obj in objects:
        if not obj.alive:
            continue
        
        alpha = food_alpha if obj.compliance > 0.5 else 0.4
        ax.add_patch(Circle(
            obj.pos, obj.size, fc=obj.color, ec='white',
            alpha=alpha, lw=0.3
        ))


def render_nutrients(ax, nutrients: List, light_level: float = 1.0):
    """Render nutrient patches (poop/decomposing matter)."""
    for nut in nutrients:
        ax.plot(
            nut.pos[0], nut.pos[1], 'o',
            color='goldenrod', ms=3, alpha=0.5 * light_level
        )


def render_corpses(ax, creatures: List):
    """Render dead creature corpses."""
    for creature in creatures:
        if not creature.alive:
            ax.add_patch(Circle(
                creature.pos, creature.species.body_scale,
                fc='gray', ec='darkgray', alpha=0.3
            ))
            ax.plot(creature.pos[0], creature.pos[1], 'x', 
                   color='red', ms=5, alpha=0.5)


# =============================================================================
# WEATHER RENDERING
# =============================================================================

def render_weather(ax, weather_events: List):
    """
    Render weather event zones.
    
    Args:
        ax: Matplotlib axes
        weather_events: List of active weather events
    """
    for event in weather_events:
        color = get_weather_color(event.event_type)
        
        circle = Circle(
            event.center, event.radius,
            fc=color, ec=color, alpha=0.2, lw=1, linestyle='--'
        )
        ax.add_patch(circle)
        
        ax.text(
            event.center[0], event.center[1] + event.radius + 1,
            event.event_type.upper(), color=color, fontsize=7,
            ha='center', fontweight='bold', alpha=0.8
        )


# =============================================================================
# LEVIATHAN RENDERING
# =============================================================================

def render_leviathan(ax, leviathan_mgr: 'LeviathanSystem', step: int):
    """
    Render Leviathan and its effects.
    
    Args:
        ax: Matplotlib axes
        leviathan_mgr: Leviathan management system
        step: Current step for animation
    """
    vis_data = leviathan_mgr.get_visual_data()
    
    # Fertile patches
    for pos, radius, intensity in vis_data.get('patches', []):
        ax.add_patch(Circle(
            pos, radius,
            fc='orange', ec='darkorange', alpha=0.15 * intensity,
            lw=1, linestyle='--'
        ))
    
    # Active Leviathan
    if not vis_data.get('active'):
        return
    
    lev_pos = vis_data['pos']
    lev_size = vis_data['size']
    trail = vis_data.get('trail', [])
    is_hunting = vis_data.get('is_hunting', False)
    is_genesis = vis_data.get('is_genesis', False)
    
    # Color scheme
    if is_genesis:
        trail_color = '#FF4500'
        glow_color = '#FF6600'
        body_color = '#FF2200'
        eye_color = '#FFFF00'
    else:
        trail_color = LEVIATHAN_COLOR_GLOW
        glow_color = LEVIATHAN_COLOR_GLOW
        body_color = '#FF0000' if is_hunting else LEVIATHAN_COLOR_BASE
        eye_color = 'yellow'
    
    # Trail
    if len(trail) > 1:
        for i in range(len(trail) - 1):
            alpha = (i / len(trail)) * (0.7 if is_genesis else 0.5)
            lw = 6 if is_genesis else 4
            ax.plot(
                [trail[i][0], trail[i+1][0]],
                [trail[i][1], trail[i+1][1]],
                color=trail_color, alpha=alpha, lw=lw
            )
    
    # Pulsing glow
    pulse_speed = 0.2 if is_genesis else 0.1
    pulse = 0.3 + 0.2 * np.sin(step * pulse_speed)
    glow_size = lev_size * (1.3 + pulse * 0.3)
    
    glow_alpha = 0.4 if is_genesis else 0.25
    ax.add_patch(Circle(
        lev_pos, glow_size * (1.2 if is_genesis else 1.0),
        fc=glow_color, ec='none', alpha=glow_alpha
    ))
    
    # Main body
    ax.add_patch(Circle(
        lev_pos, lev_size,
        fc=body_color, ec='white', alpha=0.85, lw=3
    ))
    
    # Eyes
    eye_offset = lev_size * 0.4
    eye_size = lev_size * 0.12
    ax.add_patch(Circle(
        lev_pos + np.array([-eye_offset, eye_offset*0.5]),
        eye_size, fc=eye_color, ec='none', alpha=0.9
    ))
    ax.add_patch(Circle(
        lev_pos + np.array([eye_offset, eye_offset*0.5]),
        eye_size, fc=eye_color, ec='none', alpha=0.9
    ))
    
    # Label
    if is_genesis:
        status = "!! GENESIS !!"
        label_color = '#FF4500'
    else:
        status = "HUNTING" if is_hunting else "LEVIATHAN"
        label_color = 'darkred'
    
    ax.text(
        lev_pos[0], lev_pos[1] + lev_size + 3, status,
        color='white', fontsize=9, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=label_color, alpha=0.8)
    )


# =============================================================================
# ANT COLONY RENDERING
# =============================================================================

def render_ant_colony(ax, colony: 'AntColony', world_size: float,
                      show_pheromones: bool = True, show_ants: bool = True):
    """
    Render ant colony with pheromone trails.
    
    Args:
        ax: Matplotlib axes
        colony: AntColony object
        world_size: World size for extent
        show_pheromones: Whether to show pheromone field
        show_ants: Whether to show individual ants
    """
    half = world_size / 2
    
    # Pheromone field
    if show_pheromones and hasattr(colony, 'rd_field'):
        ph_img = colony.rd_field.get_render_image()
        alpha_mask = np.clip(colony.rd_field.v * 2.0, 0, 0.8)
        
        rgba = np.zeros((ph_img.shape[0], ph_img.shape[1], 4))
        rgba[:, :, :3] = ph_img
        rgba[:, :, 3] = alpha_mask
        
        ax.imshow(
            rgba, origin='lower',
            extent=[-half, half, -half, half],
            interpolation='bilinear', zorder=1
        )
    
    # Nest
    nest_glow = Circle(
        colony.nest_pos, colony.nest_radius * 1.5,
        fc='brown', ec='none', alpha=0.2, zorder=2
    )
    ax.add_patch(nest_glow)
    
    nest_circle = Circle(
        colony.nest_pos, colony.nest_radius,
        fc='saddlebrown', ec='brown', alpha=0.8, lw=2, zorder=3
    )
    ax.add_patch(nest_circle)
    
    ax.text(
        colony.nest_pos[0], colony.nest_pos[1],
        f'[A] {len(colony.ants)}', color='white', fontsize=8,
        ha='center', va='center', zorder=4
    )
    
    # Individual ants
    if show_ants:
        for ant in colony.ants:
            color = 'yellow' if ant.carrying_food else 'brown'
            size = 3 if ant.carrying_food else 2
            ax.plot(
                ant.pos[0], ant.pos[1], 'o',
                color=color, ms=size, alpha=0.7, zorder=3
            )


# =============================================================================
# EMERGENT SYSTEM RENDERING
# =============================================================================

def render_nests(ax, nests: List[Tuple], light_level: float = 1.0):
    """
    Render emergent Herbie nests/homesteads.
    
    Args:
        ax: Matplotlib axes
        nests: List of (center, owner_name, nutrient_count, is_active, decay) tuples
        light_level: Day/night light level
    """
    for center, owner_name, nutrient_count, is_active, decay in nests:
        alpha = 0.3 * light_level * (1 - decay * 0.5)
        color = '#8B4513' if is_active else '#696969'
        
        ax.add_patch(Circle(
            center, 8.0,
            fc=color, ec='none', alpha=alpha
        ))
        
        if is_active:
            ax.text(
                center[0], center[1], f"[H]{owner_name[:6]}",
                color='white', fontsize=6, ha='center', alpha=0.8
            )


def render_smears(ax, marks: List[Tuple], light_level: float = 1.0):
    """
    Render pigment smear marks (emergent art).
    
    Args:
        ax: Matplotlib axes
        marks: List of (pos, color, intensity) tuples
        light_level: Day/night light level
    """
    for pos, color, intensity in marks:
        alpha = intensity * 0.7 * light_level
        size = 2 + intensity * 3
        ax.scatter(
            pos[0], pos[1],
            c=[color], s=size, alpha=alpha, edgecolors='none'
        )


def render_holes(ax, holes: List[Tuple], light_level: float = 1.0):
    """
    Render dug holes.
    
    Args:
        ax: Matplotlib axes
        holes: List of (pos, depth, is_covered, contents_count, digger_name) tuples
        light_level: Day/night light level
    """
    for pos, depth, is_covered, contents, digger_name in holes:
        if is_covered:
            # Covered hole - mound
            ax.add_patch(Circle(
                pos, 2.0,
                fc='#8B7355', ec='#5C4033', alpha=0.5 * light_level
            ))
        else:
            # Open hole - dark pit
            alpha = 0.3 + 0.4 * depth
            ax.add_patch(Circle(
                pos, 2.0 * (0.5 + 0.5 * depth),
                fc='#1a1a1a', ec='#3a3a3a', alpha=alpha * light_level
            ))
            
            if contents > 0:
                ax.text(
                    pos[0], pos[1], f'{contents}',
                    color='gray', fontsize=5, ha='center'
                )


def render_mushrooms(ax, fruiting_bodies: List[Tuple], light_level: float = 1.0):
    """
    Render mycorrhizal fruiting bodies (mushrooms).
    
    Args:
        ax: Matplotlib axes
        fruiting_bodies: List of (pos, size, age_fraction) tuples
        light_level: Day/night light level
    """
    for pos, size, age_frac in fruiting_bodies:
        # Young mushrooms are lighter, old ones darker
        r = 0.9 - 0.3 * age_frac
        g = 0.8 - 0.4 * age_frac
        b = 0.7 - 0.3 * age_frac
        
        ax.add_patch(Circle(
            pos, size * 1.5,
            fc=(r, g, b), ec='#5a4a3a', alpha=0.7 * light_level, lw=0.5
        ))


def render_aquatic(ax, aquatic_system, light_level: float = 1.0):
    """
    Render aquatic plants and creatures.
    
    Args:
        ax: Matplotlib axes
        aquatic_system: AquaticSystem object
        light_level: Day/night light level
    """
    # Plants
    for plant in aquatic_system.plants:
        if not plant.alive:
            continue
        
        # Apply sway offset
        offset = plant.get_sway_offset()
        display_pos = plant.pos + offset
        
        ax.add_patch(Circle(
            display_pos, plant.size,
            fc=plant.color, ec='#2E8B57', alpha=0.6 * light_level, lw=0.5
        ))
    
    # Creatures (fish)
    for creature in aquatic_system.creatures:
        if not creature.alive:
            continue
        
        ax.plot(
            creature.pos[0], creature.pos[1], 'o',
            color='silver', ms=4, alpha=0.6 * light_level
        )


def render_mycelia_overlay(ax, mycelia_network, world_size: float,
                           light_level: float = 1.0, alpha: float = 0.3):
    """
    Render mycorrhizal network as semi-transparent overlay.
    
    Args:
        ax: Matplotlib axes
        mycelia_network: MyceliumNetwork object
        world_size: World size for extent
        light_level: Day/night light level
        alpha: Base transparency
    """
    half = world_size / 2
    
    # Density field as purple overlay
    density = mycelia_network.density
    
    # Create RGBA image
    rgba = np.zeros((density.shape[0], density.shape[1], 4))
    rgba[:, :, 0] = 0.5  # R
    rgba[:, :, 1] = 0.2  # G
    rgba[:, :, 2] = 0.6  # B
    rgba[:, :, 3] = density * alpha * light_level
    
    ax.imshow(
        rgba, origin='lower',
        extent=[-half, half, -half, half],
        interpolation='bilinear', zorder=0.5
    )
