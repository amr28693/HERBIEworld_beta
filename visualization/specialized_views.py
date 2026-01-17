"""
Specialized Visualization Views - Popup and alternate view modes.

Contains:
- EvolutionTreeRenderer: Renders evolution tree diagram
- EvolutionTreeView: Popup window for tree
- ArtGalleryView: Gallery of emergent Herbie art
- IsometricCityView: Pseudo-3D isometric view
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import TYPE_CHECKING, List, Dict, Optional

if TYPE_CHECKING:
    from ..evolution.tree import EvolutionTree, EvolutionNode
    from ..ecology.emergent import SmearSystem
    from ..manager.creature_manager import CreatureManager


# =============================================================================
# EVOLUTION TREE RENDERER
# =============================================================================

class EvolutionTreeRenderer:
    """
    Renders evolution tree as a visual diagram.
    Shows lineage, traits, and statistics.
    """
    
    def __init__(self, tree: 'EvolutionTree'):
        self.tree = tree
        self.selected_species = "Herbie"
        self.show_all_species = False
        
        # Visual settings
        self.node_radius = 0.4
        self.generation_spacing = 2.5
        self.sibling_spacing = 3.0
        
        # Colors by species
        self.species_colors = {
            'Herbie': '#00FFFF',
            'Blob': '#90EE90',
            'Biped': '#FFD700',
            'Mono': '#DDA0DD',
            'Scavenger': '#FFA500',
            'Apex': '#FF4444'
        }
    
    def render(self, ax, current_step: int):
        """Render tree on given axes."""
        ax.clear()
        ax.set_facecolor('#0a0a1a')
        
        if self.show_all_species:
            species_list = list(self.species_colors.keys())
        else:
            species_list = [self.selected_species]
        
        # Compute layout
        for species in species_list:
            self.tree.compute_tree_layout(species)
        
        # Draw edges first (behind nodes)
        for parent_id, children_ids in self.tree.children.items():
            if parent_id not in self.tree.nodes:
                continue
            parent = self.tree.nodes[parent_id]
            
            if not self.show_all_species and parent.species != self.selected_species:
                continue
            
            for child_id in children_ids:
                if child_id not in self.tree.nodes:
                    continue
                child = self.tree.nodes[child_id]
                
                color = self.species_colors.get(parent.species, 'white')
                alpha = 0.6 if child.is_alive() else 0.3
                
                ax.plot([parent.tree_x, child.tree_x], 
                       [parent.tree_y, child.tree_y],
                       color=color, alpha=alpha, lw=1.5, zorder=1)
        
        # Draw nodes
        for node in self.tree.nodes.values():
            if not self.show_all_species and node.species != self.selected_species:
                continue
            
            self._draw_node(ax, node, current_step)
        
        # Title and stats
        stats = self.tree.get_generation_stats(
            self.selected_species if not self.show_all_species else None
        )
        
        title = f"Evolution Tree: {self.selected_species}" if not self.show_all_species else "Evolution Tree: All Species"
        ax.set_title(title, color='white', fontsize=12)
        
        # Set axis limits with protection
        all_nodes = [n for n in self.tree.nodes.values() 
                    if self.show_all_species or n.species == self.selected_species]
        if all_nodes:
            xs = [n.tree_x for n in all_nodes if np.isfinite(n.tree_x)]
            ys = [n.tree_y for n in all_nodes if np.isfinite(n.tree_y)]
            
            if xs and ys:
                margin = 3
                ax.set_xlim(min(xs) - margin, max(xs) + margin)
                ax.set_ylim(min(ys) - margin, max(ys) + margin)
        
        try:
            ax.set_xscale('linear')
            ax.set_yscale('linear')
        except:
            pass
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        self._draw_legend(ax, stats)
    
    def _draw_node(self, ax, node: 'EvolutionNode', current_step: int):
        """Draw a single node."""
        x, y = node.tree_x, node.tree_y
        
        color = self.species_colors.get(node.species, 'white')
        
        if node.is_alive():
            circle = Circle((x, y), self.node_radius, 
                           fc=color, ec='white', alpha=0.9, lw=2, zorder=2)
            
            # Pulse effect for living creatures
            pulse = 0.1 * np.sin(current_step * 0.1)
            glow = Circle((x, y), self.node_radius * (1.3 + pulse),
                         fc=color, ec='none', alpha=0.3, zorder=1)
            ax.add_patch(glow)
        else:
            circle = Circle((x, y), self.node_radius,
                           fc='none', ec=color, alpha=0.5, lw=1.5, zorder=2)
        
        ax.add_patch(circle)
        
        # Generation label
        ax.text(x, y, f'G{node.generation}', 
               color='white' if node.is_alive() else 'gray',
               fontsize=7, ha='center', va='center', zorder=3)
        
        # Offspring indicator
        if node.offspring_count > 0:
            for i in range(min(node.offspring_count, 5)):
                angle = np.pi/2 + i * np.pi/6
                dot_x = x + (self.node_radius + 0.2) * np.cos(angle)
                dot_y = y + (self.node_radius + 0.2) * np.sin(angle)
                ax.plot(dot_x, dot_y, 'o', color=color, ms=3, alpha=0.7)
        
        # Herbie-specific indicators
        if node.species == "Herbie":
            if node.knowledge_count > 0:
                ax.plot(x + self.node_radius + 0.3, y + self.node_radius + 0.3,
                       '*', color='yellow', ms=8, alpha=0.8)
            
            if node.grip_successes > 0:
                ax.text(x, y - self.node_radius - 0.4,
                       f'✋{node.grip_successes}', color='cyan',
                       fontsize=5, ha='center', alpha=0.7)
    
    def _draw_legend(self, ax, stats: dict):
        """Draw legend with statistics."""
        trans = ax.transAxes
        
        y = 0.98
        ax.text(0.02, y, f"Generations: {len(stats)}", 
               transform=trans, color='white', fontsize=8)
        y -= 0.04
        ax.text(0.02, y, f"Total: {self.tree.total_births} born, {self.tree.total_deaths} died",
               transform=trans, color='gray', fontsize=7)
        
        y -= 0.06
        for gen in sorted(stats.keys())[-3:]:
            data = stats[gen]
            ax.text(0.02, y, f"G{gen}: {data['alive']}↑ {data['dead']}↓",
                   transform=trans, color='white', fontsize=6)
            y -= 0.03
        
        ax.text(0.98, 0.02, "T: toggle tree | S: switch species",
               transform=trans, color='gray', fontsize=6, ha='right')


class EvolutionTreeView:
    """Standalone popup window for evolution tree."""
    
    def __init__(self, tree: 'EvolutionTree'):
        self.tree = tree
        self.renderer = EvolutionTreeRenderer(tree)
        self.visible = False
        self.fig = None
        self.ax = None
    
    def toggle(self):
        """Toggle visibility."""
        self.visible = not self.visible
        
        if self.visible:
            self._create_window()
        else:
            self._close_window()
    
    def _create_window(self):
        """Create popup window."""
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 8), facecolor='black')
            self.fig.canvas.manager.set_window_title('Evolution Tree')
            self.ax = self.fig.add_subplot(111)
            
            self.fig.canvas.mpl_connect('key_press_event', self._on_key)
            
            plt.ion()
            self.fig.show()
    
    def _close_window(self):
        """Close popup window."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def _on_key(self, event):
        """Handle key presses."""
        if event.key == 't':
            self.toggle()
        elif event.key == ']':
            # Cycle species (was 's' but conflicts with save)
            species_list = list(self.renderer.species_colors.keys())
            try:
                current_idx = species_list.index(self.renderer.selected_species)
            except ValueError:
                current_idx = 0
            self.renderer.selected_species = species_list[(current_idx + 1) % len(species_list)]
            print(f"[Tree] Showing: {self.renderer.selected_species}")
        elif event.key == '[':
            # Cycle species backward
            species_list = list(self.renderer.species_colors.keys())
            try:
                current_idx = species_list.index(self.renderer.selected_species)
            except ValueError:
                current_idx = 0
            self.renderer.selected_species = species_list[(current_idx - 1) % len(species_list)]
            print(f"[Tree] Showing: {self.renderer.selected_species}")
        elif event.key == 'a':
            self.renderer.show_all_species = not self.renderer.show_all_species
            print(f"[Tree] All species: {self.renderer.show_all_species}")
    
    def update(self, current_step: int):
        """Update tree view if visible."""
        if not self.visible or self.ax is None:
            return
        
        try:
            self.renderer.render(self.ax, current_step)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except ValueError as e:
            if "log-scaled" in str(e) or "positive values" in str(e):
                pass
            else:
                raise
        except Exception:
            pass


def render_mini_tree(ax, tree: 'EvolutionTree', species: str = "Herbie", 
                     current_step: int = 0, max_nodes: int = 20):
    """Render a compact tree view in an existing axes."""
    ax.clear()
    ax.set_facecolor('#0a0a1a')
    
    nodes = tree.get_species_tree(species)
    if not nodes:
        ax.text(0.5, 0.5, f"No {species} yet", color='gray',
               ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        return
    
    nodes = sorted(nodes, key=lambda n: n.generation, reverse=True)[:max_nodes]
    tree.compute_tree_layout(species)
    
    color = {
        'Herbie': '#00FFFF', 'Blob': '#90EE90', 'Biped': '#FFD700',
        'Mono': '#DDA0DD', 'Scavenger': '#FFA500', 'Apex': '#FF4444'
    }.get(species, 'white')
    
    # Draw edges
    for parent_id, children_ids in tree.children.items():
        if parent_id not in tree.nodes:
            continue
        parent = tree.nodes[parent_id]
        if parent.species != species:
            continue
        
        for child_id in children_ids:
            if child_id not in tree.nodes:
                continue
            child = tree.nodes[child_id]
            
            alpha = 0.5 if child.is_alive() else 0.2
            ax.plot([parent.tree_x, child.tree_x],
                   [parent.tree_y, child.tree_y],
                   color=color, alpha=alpha, lw=1)
    
    # Draw nodes
    for node in nodes:
        x, y = node.tree_x, node.tree_y
        
        if node.is_alive():
            ax.plot(x, y, 'o', color=color, ms=8, alpha=0.9)
        else:
            ax.plot(x, y, 'o', mfc='none', mec=color, ms=6, alpha=0.4)
        
        ax.text(x, y, str(node.generation), color='white',
               fontsize=5, ha='center', va='center')
    
    if nodes:
        xs = [n.tree_x for n in nodes if np.isfinite(n.tree_x)]
        ys = [n.tree_y for n in nodes if np.isfinite(n.tree_y)]
        if xs and ys:
            margin = 2
            ax.set_xlim(min(xs) - margin, max(xs) + margin)
            ax.set_ylim(min(ys) - margin, max(ys) + margin)
    
    ax.axis('off')
    ax.set_title(f'{species} Tree', color=color, fontsize=8)


# =============================================================================
# ART GALLERY VIEW
# =============================================================================

class ArtGalleryView:
    """
    Popup window showing emergent Herbie art.
    Finds clusters of smear marks and displays them as 'artworks'.
    """
    
    def __init__(self, smear_system: 'SmearSystem'):
        self.smear_system = smear_system
        self.fig = None
        self.visible = False
        self.current_cluster_idx = 0
        self.clusters = []
    
    def toggle(self):
        """Toggle gallery visibility."""
        if self.visible:
            self.close()
        else:
            self.open()
    
    def open(self):
        """Open the gallery window."""
        self.visible = True
        self.clusters = self.smear_system.find_art_clusters(min_marks=5, cluster_radius=10.0)
        
        if not self.clusters:
            print("[ART GALLERY] No art found yet. Herbies need to smear more!")
            self.visible = False
            return
        
        print(f"[ART GALLERY] Found {len(self.clusters)} artwork(s)!")
        
        self.fig = plt.figure(figsize=(10, 8), facecolor='#1a1a1a')
        self.fig.canvas.manager.set_window_title('HERBIE Art Gallery')
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        self._render_current()
        plt.show(block=False)
    
    def close(self):
        """Close the gallery."""
        if self.fig:
            plt.close(self.fig)
        self.fig = None
        self.visible = False
    
    def _on_key(self, event):
        """Handle gallery navigation."""
        if event.key == 'escape' or event.key == 'g':
            self.close()
        elif event.key == 'right' or event.key == 'n':
            self.current_cluster_idx = (self.current_cluster_idx + 1) % len(self.clusters)
            self._render_current()
        elif event.key == 'left' or event.key == 'p':
            self.current_cluster_idx = (self.current_cluster_idx - 1) % len(self.clusters)
            self._render_current()
        elif event.key == 's':
            self._save_current()
    
    def _render_current(self):
        """Render the current artwork."""
        if not self.clusters or not self.fig:
            return
        
        self.fig.clear()
        
        cluster = self.clusters[self.current_cluster_idx]
        marks = cluster['marks']
        min_pos, max_pos = cluster['bounds']
        
        # Main art display
        ax_art = self.fig.add_axes([0.1, 0.2, 0.8, 0.7])
        ax_art.set_facecolor('#2a2a2a')
        
        padding = 3.0
        ax_art.set_xlim(min_pos[0] - padding, max_pos[0] + padding)
        ax_art.set_ylim(min_pos[1] - padding, max_pos[1] + padding)
        ax_art.set_aspect('equal')
        
        # Draw marks
        for mark in marks:
            size = 50 + 100 * mark.intensity
            ax_art.scatter(mark.pos[0], mark.pos[1], 
                          c=[mark.color], s=size, alpha=mark.intensity * 0.85,
                          edgecolors='none')
            
            if mark.intensity > 0.7:
                ax_art.scatter(mark.pos[0], mark.pos[1], 
                              c=[mark.color], s=size * 2, alpha=0.2,
                              edgecolors='none')
        
        ax_art.axis('off')
        
        # Info panel
        ax_info = self.fig.add_axes([0.1, 0.02, 0.8, 0.15])
        ax_info.set_facecolor('#1a1a1a')
        ax_info.axis('off')
        
        n_creators = len(cluster['creator_ids'])
        n_marks = len(marks)
        
        creator_text = f"{n_creators} artist{'s' if n_creators > 1 else ''}"
        
        # Analyze style
        entropies = [m.body_entropy for m in marks]
        phases = [m.torus_phase for m in marks]
        avg_entropy = np.mean(entropies)
        phase_variance = np.var(phases)
        
        if avg_entropy > 4:
            style = "Chaotic"
        elif avg_entropy > 3:
            style = "Dynamic"
        else:
            style = "Calm"
        
        if phase_variance > 2:
            style += ", Varied"
        else:
            style += ", Focused"
        
        title = f"Artwork {self.current_cluster_idx + 1} of {len(self.clusters)}"
        subtitle = f"{n_marks} marks by {creator_text} | Style: {style}"
        location = f"Location: ({cluster['center'][0]:.0f}, {cluster['center'][1]:.0f})"
        
        ax_info.text(0.5, 0.8, title, color='white', fontsize=14, 
                    ha='center', fontweight='bold', transform=ax_info.transAxes)
        ax_info.text(0.5, 0.5, subtitle, color='gray', fontsize=10,
                    ha='center', transform=ax_info.transAxes)
        ax_info.text(0.5, 0.2, location, color='#555555', fontsize=8,
                    ha='center', transform=ax_info.transAxes)
        ax_info.text(0.5, -0.2, "← → to navigate | S to save | G or ESC to close",
                    color='#333333', fontsize=8, ha='center', transform=ax_info.transAxes)
        
        self.fig.canvas.draw()
    
    def _save_current(self):
        """Save current artwork to file."""
        if not self.clusters:
            return
        
        filename = f"herbie_art_{self.current_cluster_idx + 1}.png"
        self.fig.savefig(filename, facecolor='#1a1a1a', dpi=150)
        print(f"[ART GALLERY] Saved to {filename}")


# =============================================================================
# ISOMETRIC CITY VIEW
# =============================================================================

class IsometricCityView:
    """
    Pseudo-3D isometric view of the world.
    Shows creatures as volumes, terrain has depth.
    """
    
    def __init__(self, manager: 'CreatureManager'):
        self.manager = manager
        self.visible = False
        self.fig = None
        self.ax = None
        
        # View parameters
        self.center = np.array([15.0, 10.0])
        self.view_radius = 20.0
        self.iso_angle = np.pi / 6  # 30 degrees
        self.pan_speed = 2.0
    
    def toggle(self):
        """Toggle isometric view on/off."""
        self.visible = not self.visible
        if self.visible:
            self._create_window()
            print("[View] ISOMETRIC CITY VIEW: ON")
            print("       Arrow keys to pan, +/- to zoom, I to close")
        else:
            self._close_window()
            print("[View] ISOMETRIC CITY VIEW: OFF")
    
    def _create_window(self):
        """Create the isometric view window."""
        if self.fig is not None:
            return
        
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.patch.set_facecolor('#1a1a2e')
        self.ax.set_facecolor('#1a1a2e')
        
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._center_on_population()
        
        self.fig.canvas.manager.set_window_title('Herbie City - Isometric View')
        plt.ion()
        self.fig.show()
    
    def _close_window(self):
        """Close the isometric view window."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def _center_on_population(self):
        """Center view on largest Herbie cluster."""
        herbies = [c for c in self.manager.creatures 
                   if c.species.name == "Herbie" and c.alive]
        if herbies:
            positions = np.array([h.pos for h in herbies])
            self.center = np.mean(positions, axis=0)
    
    def _on_key(self, event):
        """Handle key presses for navigation."""
        if event.key == 'i' or event.key == 'escape':
            self.toggle()
        elif event.key == 'up':
            self.center[1] += self.pan_speed
        elif event.key == 'down':
            self.center[1] -= self.pan_speed
        elif event.key == 'left':
            self.center[0] -= self.pan_speed
        elif event.key == 'right':
            self.center[0] += self.pan_speed
        elif event.key == '+' or event.key == '=':
            self.view_radius = max(5.0, self.view_radius - 2.0)
        elif event.key == '-':
            self.view_radius = min(50.0, self.view_radius + 2.0)
        elif event.key == 'c':
            self._center_on_population()
        
        if self.visible:
            self.update()
    
    def _world_to_iso(self, x, y, z=0):
        """Convert world coordinates to isometric screen coordinates."""
        iso_x = (x - y) * np.cos(self.iso_angle)
        iso_y = (x + y) * np.sin(self.iso_angle) + z * 0.5
        return iso_x, iso_y
    
    def _get_terrain_height(self, x, y):
        """Get terrain height at position."""
        if hasattr(self.manager, 'terrain') and self.manager.terrain:
            pos = np.array([x, y])
            return self.manager.terrain.get_height(pos) * 3.0
        return 0.0
    
    def _get_terrain_color(self, x, y):
        """Get terrain color at position."""
        if hasattr(self.manager, 'terrain') and self.manager.terrain:
            pos = np.array([x, y])
            terrain = self.manager.terrain.get_terrain_at(pos)
            return getattr(terrain, 'color', '#27ae60')
        return '#27ae60'
    
    def update(self):
        """Render the isometric view."""
        if not self.visible or self.fig is None:
            return
        
        self.ax.clear()
        self.ax.set_facecolor('#1a1a2e')
        
        creatures = self.manager.creatures
        world = self.manager.world
        
        # Terrain grid
        grid_step = 2.0
        x_range = np.arange(self.center[0] - self.view_radius, 
                          self.center[0] + self.view_radius, grid_step)
        y_range = np.arange(self.center[1] - self.view_radius,
                          self.center[1] + self.view_radius, grid_step)
        
        # Draw terrain tiles
        for y in reversed(y_range):
            for x in x_range:
                height = self._get_terrain_height(x, y)
                color = self._get_terrain_color(x, y)
                
                corners = [
                    self._world_to_iso(x, y, height),
                    self._world_to_iso(x + grid_step, y, height),
                    self._world_to_iso(x + grid_step, y + grid_step, height),
                    self._world_to_iso(x, y + grid_step, height)
                ]
                
                xs = [c[0] for c in corners] + [corners[0][0]]
                ys = [c[1] for c in corners] + [corners[0][1]]
                self.ax.fill(xs, ys, color=color, alpha=0.6, ec='#333333', lw=0.3)
        
        # Smear marks
        if hasattr(self.manager, 'smears'):
            for pos, color, intensity in self.manager.smears.get_marks_for_display():
                if (abs(pos[0] - self.center[0]) < self.view_radius and
                    abs(pos[1] - self.center[1]) < self.view_radius):
                    height = self._get_terrain_height(pos[0], pos[1])
                    iso_x, iso_y = self._world_to_iso(pos[0], pos[1], height + 0.1)
                    self.ax.scatter(iso_x, iso_y, c=[color], s=30 * intensity,
                                   alpha=intensity * 0.9, marker='o')
        
        # Food objects
        for obj in world.objects:
            if not obj.alive:
                continue
            if (abs(obj.pos[0] - self.center[0]) < self.view_radius and
                abs(obj.pos[1] - self.center[1]) < self.view_radius):
                height = self._get_terrain_height(obj.pos[0], obj.pos[1])
                iso_x, iso_y = self._world_to_iso(obj.pos[0], obj.pos[1], height + obj.size * 0.5)
                
                size = 40 * obj.size
                color = obj.color if hasattr(obj, 'color') else 'green'
                self.ax.scatter(iso_x, iso_y, c=[color], s=size, alpha=0.7, 
                               ec='white', lw=0.5)
        
        # Creatures (sorted back to front)
        visible_creatures = [(c, c.pos[1]) for c in creatures if c.alive and
                            abs(c.pos[0] - self.center[0]) < self.view_radius and
                            abs(c.pos[1] - self.center[1]) < self.view_radius]
        visible_creatures.sort(key=lambda x: -x[1])
        
        for creature, _ in visible_creatures:
            pos = creature.pos
            height = self._get_terrain_height(pos[0], pos[1])
            creature_height = creature.species.body_scale * 2
            
            iso_x, iso_y = self._world_to_iso(pos[0], pos[1], height)
            color = creature.species.color_base
            base_size = creature.species.body_scale * 150
            
            # Draw as stacked layers
            for layer in range(4):
                layer_height = height + creature_height * (layer / 4)
                layer_iso_x, layer_iso_y = self._world_to_iso(pos[0], pos[1], layer_height)
                layer_size = base_size * (1.0 - layer * 0.15)
                layer_alpha = 0.3 + layer * 0.15
                
                self.ax.scatter(layer_iso_x, layer_iso_y, c=[color], s=layer_size,
                               alpha=layer_alpha, ec='white' if layer == 3 else 'none',
                               lw=1 if layer == 3 else 0)
            
            # Name label for Herbies
            if creature.species.name == "Herbie" and hasattr(creature, 'mating_state'):
                name = getattr(creature.mating_state, 'name', None)
                if name:
                    label_iso_x, label_iso_y = self._world_to_iso(
                        pos[0], pos[1], height + creature_height + 1
                    )
                    self.ax.text(label_iso_x, label_iso_y, name,
                               color='white', fontsize=8, ha='center', va='bottom',
                               fontweight='bold', alpha=0.9)
        
        # Title
        herbie_count = sum(1 for c in creatures if c.species.name == "Herbie" and c.alive)
        
        self.ax.set_title(
            f'HERBIE CITY | Center: ({self.center[0]:.0f}, {self.center[1]:.0f}) | '
            f'Herbies: {herbie_count} | Zoom: {self.view_radius:.0f}',
            color='white', fontsize=12, fontweight='bold'
        )
        
        self.ax.text(0.02, 0.02, 
                    'Arrows=Pan | +/-=Zoom | C=Center on Herbies | I=Close',
                    color='gray', fontsize=8, transform=self.ax.transAxes)
        
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
