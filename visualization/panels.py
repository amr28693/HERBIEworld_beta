"""
Panel renderers - Status displays, population charts, creature details.

These render into specific axes for information display panels.
"""

import numpy as np
from collections import deque
from typing import List, Dict, Optional, TYPE_CHECKING

from .colors import complex_to_rgb, get_species_color

if TYPE_CHECKING:
    from ..creature.creature import Creature
    from ..manager.creature_manager import CreatureManager


# =============================================================================
# STATUS PANEL
# =============================================================================

def render_status_panel(ax, manager: 'CreatureManager', step: int):
    """
    Render main status information panel.
    
    Args:
        ax: Matplotlib axes (text-only)
        manager: CreatureManager for stats
        step: Current simulation step
    """
    ax.clear()
    ax.set_facecolor('black')
    ax.axis('off')
    
    stats = manager.get_statistics()
    
    # Season and time
    season = stats['season']
    year = stats['year']
    day_phase = stats['day_phase']
    
    season_emoji = {'Spring': '[Spr]', 'Summer': '[Sum]', 'Autumn': '[Aut]', 'Winter': '[Win]'}
    time_emoji = {'day': '*', 'dawn': '>', 'dusk': '<', 'night': 'o'}
    
    line1 = f"{season_emoji.get(season, '')} {season} Year {year} | {time_emoji.get(day_phase, '')} {day_phase}"
    
    # Population
    total = stats['total_creatures']
    herbie_count = stats['species_counts'].get('Herbie', 0)
    
    line2 = f"Population: {total} | Herbies: {herbie_count} | Food: {stats['food_count']}"
    
    # Special events
    events = []
    if stats.get('leviathan_active'):
        events.append("!! LEVIATHAN")
    if stats.get('meteor_active'):
        events.append("☄️ METEORS")
    if stats.get('migration_active'):
        events.append("~~ MIGRATION")
    
    line3 = " | ".join(events) if events else ""
    
    # Achievements
    unlocked, total_ach = stats['achievements']
    line4 = f"Achievements: {unlocked}/{total_ach}"
    
    # Render
    ax.text(0.5, 0.8, line1, ha='center', va='center', color='white', 
            fontsize=10, transform=ax.transAxes)
    ax.text(0.5, 0.55, line2, ha='center', va='center', color='cyan',
            fontsize=9, transform=ax.transAxes)
    if line3:
        ax.text(0.5, 0.35, line3, ha='center', va='center', color='orange',
                fontsize=9, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.15, line4, ha='center', va='center', color='gold',
            fontsize=8, transform=ax.transAxes)


def render_extended_status(ax, manager: 'CreatureManager'):
    """
    Render extended status with ecosystem details.
    
    Args:
        ax: Matplotlib axes
        manager: CreatureManager for stats
    """
    ax.clear()
    ax.set_facecolor('#0a0a15')
    ax.axis('off')
    
    stats = manager.get_statistics()
    
    y = 0.95
    dy = 0.12
    
    # Aquatic stats
    aquatic = stats.get('aquatic', {})
    if aquatic:
        ax.text(0.05, y, f"~~ Aquatic: {aquatic.get('total_plants', 0)} plants, "
                f"{aquatic.get('total_creatures', 0)} fish",
                color='#4169E1', fontsize=8, transform=ax.transAxes)
        y -= dy
    
    # Mycelia stats
    mycelia_biomass = stats.get('mycelia_biomass', 0)
    mycelia_coverage = stats.get('mycelia_coverage', 0)
    mushrooms = stats.get('mushroom_count', 0)
    ax.text(0.05, y, f"^^ Mycelia: {mycelia_biomass:.0f} biomass, "
            f"{mycelia_coverage*100:.0f}% coverage, {mushrooms} mushrooms",
            color='#9370DB', fontsize=8, transform=ax.transAxes)
    y -= dy
    
    # Species breakdown
    for species, count in stats['species_counts'].items():
        color = get_species_color(species)
        ax.text(0.05, y, f"  {species}: {count}",
                color=color, fontsize=7, transform=ax.transAxes)
        y -= dy * 0.8


# =============================================================================
# POPULATION CHART
# =============================================================================

class PopulationTracker:
    """Track population history for charting."""
    
    def __init__(self, species_names: List[str], max_history: int = 500):
        self.species_names = species_names
        self.history = {name: deque(maxlen=max_history) for name in species_names}
        self.time_points = deque(maxlen=max_history)
        self.step_counter = 0
    
    def update(self, creatures: List['Creature']):
        """Record current population counts."""
        self.step_counter += 1
        self.time_points.append(self.step_counter)
        
        for name in self.species_names:
            count = sum(1 for c in creatures if c.species.name == name and c.alive)
            self.history[name].append(count)


def render_population_chart(ax, tracker: PopulationTracker):
    """
    Render population over time chart.
    
    Args:
        ax: Matplotlib axes
        tracker: PopulationTracker with history
    """
    ax.clear()
    ax.set_facecolor('black')
    ax.set_title('Population', color='white', fontsize=9)
    
    if len(tracker.time_points) < 2:
        return
    
    times = list(tracker.time_points)
    
    for name in tracker.species_names:
        counts = list(tracker.history[name])
        if any(counts):
            color = get_species_color(name)
            ax.plot(times, counts, color=color, lw=1.5, label=name)
    
    ax.tick_params(colors='gray', labelsize=6)
    ax.set_xlim(max(0, tracker.step_counter - 500), tracker.step_counter + 10)
    
    # Y axis
    all_counts = []
    for name in tracker.species_names:
        all_counts.extend(tracker.history[name])
    if all_counts:
        ax.set_ylim(0, max(all_counts) * 1.1 + 1)


def render_species_legend(ax, species_counts: Dict[str, int]):
    """
    Render species legend with counts.
    
    Args:
        ax: Matplotlib axes
        species_counts: Dict of species name -> count
    """
    ax.clear()
    ax.set_facecolor('black')
    ax.axis('off')
    
    y = 0.95
    for name, count in species_counts.items():
        color = get_species_color(name)
        ax.text(0.1, y, f'{name}: {count}',
                color=color, fontsize=8, transform=ax.transAxes)
        y -= 0.12


# =============================================================================
# CREATURE DETAIL PANELS
# =============================================================================

def render_creature_detail(ax, creature: 'Creature'):
    """
    Render selected creature's body field.
    
    Args:
        ax: Matplotlib axes
        creature: Creature to display
    """
    ax.clear()
    ax.set_facecolor('black')
    
    if not creature.alive:
        ax.set_title('(dead)', color='gray', fontsize=9)
        return
    
    # State indicator
    state = ""
    if getattr(creature, 'is_hibernating', False):
        state = " [zzz]"
    elif getattr(creature, 'is_digesting', False):
        state = " [dig]"
    
    # Name for Herbies
    name = ""
    if hasattr(creature, 'mating_state') and creature.mating_state.name:
        name = f" '{creature.mating_state.name}'"
    
    ax.set_title(
        f'{creature.species.name}{name} G{creature.generation}{state}',
        color=creature.species.color_base, fontsize=9
    )
    
    # Body field intensity
    I = np.abs(creature.body.psi)**2
    from ..core.constants import BODY_L
    
    ax.imshow(
        I, origin='lower', cmap='inferno',
        extent=[-BODY_L/2, BODY_L/2, -BODY_L/2, BODY_L/2]
    )
    ax.set_xticks([])
    ax.set_yticks([])


def render_creature_phase(ax, creature: 'Creature'):
    """
    Render selected creature's body phase field (HSV colored).
    
    Args:
        ax: Matplotlib axes
        creature: Creature to display
    """
    ax.clear()
    ax.set_facecolor('black')
    
    if not creature.alive:
        return
    
    ax.set_title('Phase', color='white', fontsize=8)
    
    from ..core.constants import BODY_L
    phase_rgb = complex_to_rgb(creature.body.psi)
    
    ax.imshow(
        phase_rgb, origin='lower',
        extent=[-BODY_L/2, BODY_L/2, -BODY_L/2, BODY_L/2]
    )
    ax.set_xticks([])
    ax.set_yticks([])


def render_torus_panel(ax, creature: 'Creature'):
    """
    Render selected creature's torus brain (polar plot).
    
    Args:
        ax: Matplotlib axes (must have projection='polar')
        creature: Creature to display
    """
    ax.clear()
    ax.set_facecolor('black')
    
    if not creature.alive:
        return
    
    ax.set_title('Brain', color='white', fontsize=8)
    
    from ..core.constants import N_torus
    theta = np.linspace(0, 2*np.pi, N_torus, endpoint=False)
    
    ax.plot(
        theta, np.abs(creature.torus.psi),
        color=creature.species.color_base, lw=2
    )
    ax.set_ylim(0, 2)


def render_herbie_family_panel(ax, herbie):
    """
    Render Herbie family information.
    
    Args:
        ax: Matplotlib axes
        herbie: HerbieWithHands creature
    """
    ax.clear()
    ax.set_facecolor('#0a0a0a')
    ax.axis('off')
    
    if not hasattr(herbie, 'mating_state'):
        return
    
    ms = herbie.mating_state
    
    y = 0.9
    dy = 0.12
    
    # Name and sex
    sex_emoji = '♀️' if ms.sex.name == 'CARRIER' else '♂️'
    ax.text(0.1, y, f"{sex_emoji} {ms.name or 'Unnamed'}",
            color='white', fontsize=10, fontweight='bold',
            transform=ax.transAxes)
    y -= dy * 1.5
    
    # Generation
    ax.text(0.1, y, f"Generation: {herbie.generation}",
            color='cyan', fontsize=8, transform=ax.transAxes)
    y -= dy
    
    # Mate(s)
    if ms.sex.name == 'CARRIER' and ms.mate_id:
        ax.text(0.1, y, f"Bonded to: {ms.mate_name or 'partner'}",
                color='pink', fontsize=8, transform=ax.transAxes)
        y -= dy
    elif ms.sex.name == 'PROVIDER' and ms.mate_ids:
        ax.text(0.1, y, f"Mates: {len(ms.mate_ids)}",
                color='pink', fontsize=8, transform=ax.transAxes)
        y -= dy
    
    # Offspring
    if ms.offspring_ids:
        ax.text(0.1, y, f"Offspring: {len(ms.offspring_ids)}",
                color='lightgreen', fontsize=8, transform=ax.transAxes)
        y -= dy
    
    # Pregnancy
    if ms.is_pregnant:
        progress = ms.pregnancy_progress * 100
        ax.text(0.1, y, f"🤰 Pregnant: {progress:.0f}%",
                color='pink', fontsize=8, fontweight='bold',
                transform=ax.transAxes)


# =============================================================================
# HUNGER AND ENERGY BARS
# =============================================================================

def render_hunger_bars(ax, creatures: List['Creature']):
    """
    Render hunger level bars for all creatures.
    
    Args:
        ax: Matplotlib axes
        creatures: List of creatures
    """
    ax.clear()
    ax.set_facecolor('black')
    ax.set_title('Hunger Levels', color='white', fontsize=9)
    
    alive = [c for c in creatures if c.alive]
    if not alive:
        return
    
    hungers = [c.metabolism.hunger for c in alive]
    colors = [c.species.color_base for c in alive]
    
    ax.bar(range(len(hungers)), hungers, color=colors, alpha=0.7)
    ax.axhline(0.5, color='yellow', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 1)
    ax.tick_params(colors='gray', labelsize=6)


def render_energy_chart(ax, creature: 'Creature', history_len: int = 200):
    """
    Render creature energy history.
    
    Args:
        ax: Matplotlib axes
        creature: Creature with energy history
        history_len: Number of points to show
    """
    ax.clear()
    ax.set_facecolor('black')
    ax.set_title('Energy', color='orange', fontsize=8)
    
    if hasattr(creature, 'energy_history'):
        history = list(creature.energy_history)[-history_len:]
        ax.plot(history, color='orange', lw=1)
        ax.set_ylim(0, max(history) * 1.2 + 10 if history else 100)
    
    ax.tick_params(colors='gray', labelsize=6)


# =============================================================================
# ACHIEVEMENT DISPLAY
# =============================================================================

def render_achievement_popup(ax, achievement_name: str, description: str):
    """
    Render achievement unlock popup.
    
    Args:
        ax: Matplotlib axes
        achievement_name: Name of achievement
        description: Achievement description
    """
    ax.clear()
    ax.set_facecolor('#1a1a00')
    ax.axis('off')
    
    ax.text(0.5, 0.7, '🏆 ACHIEVEMENT UNLOCKED 🏆',
            ha='center', va='center', color='gold',
            fontsize=10, fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.4, achievement_name,
            ha='center', va='center', color='yellow',
            fontsize=12, fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.2, description,
            ha='center', va='center', color='white',
            fontsize=8, transform=ax.transAxes)
