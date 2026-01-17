"""
Visualization System - Real-time matplotlib rendering.

Modular visualization supporting:
- World view with terrain, creatures, objects
- Day/night lighting
- Selected creature detail (body field, brain torus)
- Population charts
- Status displays
- Specialized overlays (ants, chemistry, leviathan)
- Popup views (evolution tree, art gallery, isometric)
"""

from .colors import complex_to_rgb, get_species_color, get_terrain_color
from .renderers import (
    render_creature, render_food, render_terrain_background,
    render_leviathan, render_ant_colony, render_weather,
    render_nests, render_smears, render_holes
)
from .panels import (
    render_status_panel, render_population_chart,
    render_creature_detail, render_torus_panel
)
from .chemistry_render import (
    render_elements, render_constructions, render_reaction_sites,
    render_element_field, render_herbie_held_elements,
    get_chemistry_status_string, render_chemistry_legend
)
from .specialized_views import (
    EvolutionTreeRenderer, EvolutionTreeView, render_mini_tree,
    ArtGalleryView, IsometricCityView
)
from .main_vis import HerbieVisualization
