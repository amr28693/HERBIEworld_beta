"""
Chemistry Rendering - Visualization for elements, constructions, and reactions.

Provides rendering functions for:
- ElementObject display with temperature glow
- Construction stacks with type indicators
- Reaction sites (fire, explosion, compost, tempering)
- Element field overlays (temperature, concentrations)
- Herbie held element display
"""

import numpy as np
from matplotlib.patches import Circle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..chemistry.spawner import ElementSpawner
    from ..chemistry.elements import ElementType


def render_elements(ax, element_spawner: 'ElementSpawner', view_mode: str = 'world'):
    """
    Render all elements in the world.
    
    Args:
        ax: Matplotlib axes
        element_spawner: The ElementSpawner instance
        view_mode: 'world' for full view, 'local' for Herbie-centered
    """
    for elem in element_spawner.element_objects:
        if not elem.alive:
            continue
        
        props = elem.props
        
        # Base circle for element
        circle = Circle(
            elem.pos, 
            elem.size,
            facecolor=props.color,
            edgecolor='white' if elem.temperature > 0.5 else '#333333',
            linewidth=1 + elem.temperature,  # Glowing edge when hot
            alpha=0.8,
            zorder=15
        )
        ax.add_patch(circle)
        
        # Symbol on top
        ax.text(
            elem.pos[0], elem.pos[1],
            props.symbol,
            ha='center', va='center',
            fontsize=6 + elem.amount * 2,
            color='white' if props.density > 2 else 'black',
            fontweight='bold',
            zorder=16
        )
        
        # Temperature glow
        if elem.temperature > 0.3:
            glow = Circle(
                elem.pos,
                elem.size * (1.5 + elem.temperature),
                facecolor='none',
                edgecolor='#ff6600',
                linewidth=elem.temperature * 2,
                alpha=elem.temperature * 0.5,
                zorder=14
            )
            ax.add_patch(glow)


def render_constructions(ax, element_spawner: 'ElementSpawner'):
    """Render constructions (stacked elements)."""
    for cid, construction in element_spawner.constructions.items():
        pos = construction.pos
        
        # Base size depends on height
        size = 1.5 + construction.height * 0.5
        
        # Color depends on type
        colors = {
            'pile': '#8b7355',
            'mound': '#6b5344',
            'wall': '#5a4a3a',
            'shelter': '#4a3a2a',
        }
        color = colors.get(construction.construction_type, '#666666')
        
        # Stability affects alpha
        alpha = 0.4 + construction.stability * 0.4
        
        # Draw stacked appearance
        for i in range(min(construction.height, 5)):
            offset = i * 0.3
            layer = Circle(
                (pos[0], pos[1] - offset * 0.5),
                size - i * 0.2,
                facecolor=color,
                edgecolor='#333333',
                linewidth=0.5,
                alpha=alpha * (1 - i * 0.1),
                zorder=12 + i
            )
            ax.add_patch(layer)
        
        # Label
        label = f"{construction.construction_type[:3].upper()}"
        if construction.height > 3:
            label += f" h{construction.height}"
        
        ax.text(
            pos[0], pos[1] + size + 0.5,
            label,
            ha='center', va='bottom',
            fontsize=5,
            color='#cccccc',
            zorder=20
        )
        
        # Show ownership/contributors count
        if len(construction.contributors) > 1:
            ax.text(
                pos[0], pos[1] - size - 0.3,
                f"👥{len(construction.contributors)}",
                ha='center', va='top',
                fontsize=5,
                zorder=20
            )


def render_reaction_sites(ax, element_spawner: 'ElementSpawner'):
    """Render active reaction sites with appropriate effects."""
    field = element_spawner.element_field
    
    for i, j, reaction_type in field.reaction_sites:
        pos = field.grid_to_world(i, j)
        
        if reaction_type == 'fire':
            # Fire effect - flickering orange/red
            for _ in range(3):
                offset = np.random.randn(2) * 0.5
                fire = Circle(
                    pos + offset,
                    0.3 + np.random.random() * 0.3,
                    facecolor='#ff4400',
                    edgecolor='#ffaa00',
                    linewidth=1,
                    alpha=0.6 + np.random.random() * 0.3,
                    zorder=25
                )
                ax.add_patch(fire)
            ax.text(pos[0], pos[1], '*', ha='center', va='center', 
                   fontsize=8, zorder=26)
        
        elif reaction_type == 'explosion':
            # Explosion effect - expanding ring
            for r in [1, 2, 3]:
                ring = Circle(
                    pos, r,
                    facecolor='none',
                    edgecolor='#ffff00',
                    linewidth=3 - r * 0.5,
                    alpha=0.8 - r * 0.2,
                    zorder=30
                )
                ax.add_patch(ring)
            ax.text(pos[0], pos[1], '💥', ha='center', va='center',
                   fontsize=12, zorder=31)
        
        elif reaction_type == 'compost':
            # Composting - green glow
            glow = Circle(
                pos, 1.0,
                facecolor='#44aa44',
                edgecolor='#228822',
                linewidth=1,
                alpha=0.3,
                zorder=11
            )
            ax.add_patch(glow)
        
        elif reaction_type == 'hot_ore':
            # Hot ore - orange glow
            glow = Circle(
                pos, 0.8,
                facecolor='#ff6600',
                alpha=0.4,
                zorder=11
            )
            ax.add_patch(glow)
        
        elif reaction_type == 'temper':
            # Tempering - sparkle effect
            ax.scatter(
                [pos[0] + np.random.randn() * 0.5 for _ in range(5)],
                [pos[1] + np.random.randn() * 0.5 for _ in range(5)],
                c='white', s=3, marker='*', alpha=0.8, zorder=27
            )


def render_element_field(ax, element_spawner: 'ElementSpawner', 
                         show_temperature: bool = True,
                         show_concentrations: bool = False):
    """
    Render the reaction-diffusion field as a background overlay.
    
    Args:
        ax: Matplotlib axes
        element_spawner: ElementSpawner instance
        show_temperature: Show temperature field
        show_concentrations: Show element concentration fields
    """
    field = element_spawner.element_field
    extent = [-field.world_size/2, field.world_size/2,
              -field.world_size/2, field.world_size/2]
    
    if show_temperature:
        # Temperature field - red/orange overlay
        temp = field.temperature
        if temp.max() > 0.1:
            ax.imshow(
                temp,
                extent=extent,
                origin='lower',
                cmap='hot',
                alpha=np.clip(temp.max() * 0.3, 0, 0.4),
                vmin=0, vmax=2,
                zorder=1
            )
    
    if show_concentrations:
        # Combined concentration field - subtle overlay
        combined = np.zeros((field.resolution, field.resolution, 3))
        
        # Element colors for RGB contribution
        from ..chemistry.elements import ElementType, ELEMENT_PROPS
        
        elem_colors = {
            ElementType.ITE: [0.5, 0.5, 0.5],
            ElementType.ITE_LITE: [0.7, 0.6, 0.4],
            ElementType.ORE: [0.7, 0.45, 0.2],
            ElementType.VAPOR: [0.7, 0.9, 1.0],
            ElementType.MULCHITE: [0.3, 0.2, 0.15],
            ElementType.LITE_ORE: [1.0, 0.4, 0.2],
        }
        
        for elem_type, color in elem_colors.items():
            if elem_type in field.fields:
                conc = field.fields[elem_type]
                for c in range(3):
                    combined[:, :, c] += conc * color[c] * 0.5
        
        combined = np.clip(combined, 0, 1)
        
        ax.imshow(
            combined,
            extent=extent,
            origin='lower',
            alpha=0.2,
            zorder=0
        )


def render_herbie_held_elements(ax, herbie, offset_y: float = 0):
    """Render what Herbie is holding with element-specific styling."""
    if not hasattr(herbie, 'hands'):
        return
    
    held = herbie.hands.get_held_objects()
    
    for i, obj in enumerate(held):
        # Check if it's an element
        if hasattr(obj, 'element_type') and hasattr(obj, 'props'):
            props = obj.props
            
            # Position near Herbie
            hand_offset = np.array([-1.5 if i == 0 else 1.5, -2.0 + offset_y])
            pos = herbie.pos + hand_offset
            
            # Draw held element
            circle = Circle(
                pos, obj.size * 0.8,
                facecolor=props.color,
                edgecolor='#ffff00',  # Yellow edge = held
                linewidth=2,
                alpha=0.9,
                zorder=50
            )
            ax.add_patch(circle)
            
            ax.text(
                pos[0], pos[1],
                props.symbol,
                ha='center', va='center',
                fontsize=6,
                color='white',
                fontweight='bold',
                zorder=51
            )
            
            # Temperature indicator
            if hasattr(obj, 'temperature') and obj.temperature > 0.3:
                ax.text(
                    pos[0], pos[1] + obj.size + 0.3,
                    f"🌡️{obj.temperature:.1f}",
                    ha='center', va='bottom',
                    fontsize=4,
                    zorder=51
                )


def get_chemistry_status_string(element_spawner: 'ElementSpawner') -> str:
    """Get a status string for chemistry system."""
    n_elements = len([e for e in element_spawner.element_objects if e.alive])
    n_constructions = len(element_spawner.constructions)
    n_reactions = len(element_spawner.element_field.reaction_sites)
    
    # Count by type
    type_counts = {}
    for elem in element_spawner.element_objects:
        if elem.alive:
            name = elem.element_type.name
            type_counts[name] = type_counts.get(name, 0) + 1
    
    type_str = " ".join(f"{k[:3]}:{v}" for k, v in sorted(type_counts.items()))
    
    return f"Chem: {n_elements}elem ({type_str}) | {n_constructions}build | {n_reactions}react"


def render_chemistry_legend(ax, x: float, y: float):
    """Render a small legend for element symbols."""
    from ..chemistry.elements import ElementType, ELEMENT_PROPS
    
    elements = [
        (ElementType.ITE, "ITE-Dense"),
        (ElementType.ITE_LITE, "ITE_LITE-Fiber"),
        (ElementType.ORE, "ORE-Metal"),
        (ElementType.VAPOR, "VAP-Gas"),
    ]
    
    for i, (elem_type, name) in enumerate(elements):
        props = ELEMENT_PROPS[elem_type]
        ax.text(
            x, y - i * 0.08,
            f"{props.symbol} {name}",
            ha='left', va='top',
            fontsize=5,
            color=props.color,
            transform=ax.transAxes
        )
