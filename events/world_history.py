"""
World History - Human-readable summary of simulation events.

Generates a narrative log that tells the story of what happened
in the simulation, readable by anyone without technical knowledge.
"""

import os
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# Data directory - same as main.py uses
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')


@dataclass
class WorldHistory:
    """
    Tracks and summarizes the history of a simulation run.
    
    Generates a readable .txt file with:
    - Timeline of major events
    - Population statistics
    - Notable individuals
    - Dynasties and lineages
    - Causes of death
    - Seasonal patterns
    """
    
    # Tracking data
    births: List[dict] = field(default_factory=list)
    deaths: List[dict] = field(default_factory=list)
    bonds: List[dict] = field(default_factory=list)
    predations: List[dict] = field(default_factory=list)
    seasons: List[dict] = field(default_factory=list)
    population_snapshots: List[dict] = field(default_factory=list)
    
    # Derived stats
    prolific_parents: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    death_causes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    oldest_lived: Optional[dict] = None
    first_death: Optional[dict] = None
    last_birth: Optional[dict] = None
    max_population: int = 0
    max_generation: int = 0
    
    # Meta
    start_time: str = field(default_factory=lambda: time.strftime('%Y-%m-%d %H:%M:%S'))
    start_step: int = 0
    
    _instance: Optional['WorldHistory'] = None
    
    @classmethod
    def get(cls) -> 'WorldHistory':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = WorldHistory()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset for new simulation."""
        cls._instance = None
    
    # === EVENT RECORDING ===
    
    def record_birth(self, step: int, name: str, sex: str, generation: int,
                     parents: tuple = None, species: str = "Herbie"):
        """Record a birth."""
        event = {
            'step': step,
            'name': name,
            'sex': sex,
            'generation': generation,
            'parents': parents,
            'species': species,
            'time': time.strftime('%H:%M:%S')
        }
        self.births.append(event)
        self.last_birth = event
        self.max_generation = max(self.max_generation, generation)
        
        if parents:
            for p in parents:
                self.prolific_parents[p] += 1
    
    def record_death(self, step: int, name: str, cause: str, age: int,
                     generation: int = 0, species: str = "Herbie"):
        """Record a death."""
        event = {
            'step': step,
            'name': name,
            'cause': cause,
            'age': age,
            'generation': generation,
            'species': species,
            'time': time.strftime('%H:%M:%S')
        }
        self.deaths.append(event)
        self.death_causes[cause] += 1
        
        if self.first_death is None:
            self.first_death = event
        
        if self.oldest_lived is None or age > self.oldest_lived.get('age', 0):
            self.oldest_lived = event
    
    def record_bond(self, step: int, name1: str, name2: str, resonance: float = 0):
        """Record a bonding."""
        self.bonds.append({
            'step': step,
            'partners': (name1, name2),
            'resonance': resonance,
            'time': time.strftime('%H:%M:%S')
        })
    
    def record_predation(self, step: int, predator: str, prey_name: str, prey_species: str):
        """Record a predation event."""
        self.predations.append({
            'step': step,
            'predator': predator,
            'victim': prey_name,
            'species': prey_species,
            'time': time.strftime('%H:%M:%S')
        })
    
    def record_season(self, step: int, season: str, year: int):
        """Record season change."""
        self.seasons.append({
            'step': step,
            'season': season,
            'year': year,
            'time': time.strftime('%H:%M:%S')
        })
    
    def record_population(self, step: int, counts: dict):
        """Record population snapshot."""
        total = sum(counts.values())
        self.max_population = max(self.max_population, total)
        self.population_snapshots.append({
            'step': step,
            'counts': counts.copy(),
            'total': total
        })
    
    # === REPORT GENERATION ===
    
    def generate_report(self, final_step: int, final_counts: dict) -> str:
        """Generate the full human-readable report."""
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append("              HERBIE WORLD - SIMULATION HISTORY")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  Started: {self.start_time}")
        lines.append(f"  Ended:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  Duration: {final_step:,} steps")
        lines.append("")
        
        # Era summary
        if self.seasons:
            years = max(s['year'] for s in self.seasons)
            lines.append(f"  Simulation covered {years} year(s) and {len(self.seasons)} seasons")
        lines.append("")
        
        # Population summary
        lines.append("-" * 70)
        lines.append("  POPULATION SUMMARY")
        lines.append("-" * 70)
        lines.append("")
        lines.append(f"  Peak population: {self.max_population}")
        lines.append(f"  Final population: {sum(final_counts.values())}")
        lines.append("")
        lines.append("  Final species counts:")
        for species, count in sorted(final_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {species}: {count}")
        lines.append("")
        
        # Herbie statistics
        herbie_births = [b for b in self.births if b.get('species') == 'Herbie']
        herbie_deaths = [d for d in self.deaths if d.get('species') == 'Herbie']
        
        lines.append("-" * 70)
        lines.append("  THE HERBIE CHRONICLE")
        lines.append("-" * 70)
        lines.append("")
        lines.append(f"  Total births: {len(herbie_births)}")
        lines.append(f"  Total deaths: {len(herbie_deaths)}")
        lines.append(f"  Bonds formed: {len(self.bonds)}")
        lines.append(f"  Highest generation reached: Gen {self.max_generation}")
        lines.append("")
        
        # Notable individuals
        if self.oldest_lived:
            lines.append(f"  Longest-lived: {self.oldest_lived['name']} " +
                        f"(lived {self.oldest_lived['age']:,} steps)")
        
        # Most prolific parents
        if self.prolific_parents:
            top_parents = sorted(self.prolific_parents.items(), key=lambda x: -x[1])[:5]
            lines.append("")
            lines.append("  Most prolific parents:")
            for name, count in top_parents:
                lines.append(f"    {name}: {count} children")
        
        lines.append("")
        
        # Causes of death
        if self.death_causes:
            lines.append("-" * 70)
            lines.append("  CAUSES OF DEATH")
            lines.append("-" * 70)
            lines.append("")
            total_deaths = sum(self.death_causes.values())
            for cause, count in sorted(self.death_causes.items(), key=lambda x: -x[1]):
                pct = (count / total_deaths) * 100 if total_deaths > 0 else 0
                # Make cause readable
                readable_cause = self._readable_cause(cause)
                lines.append(f"    {readable_cause}: {count} ({pct:.1f}%)")
            lines.append("")
        
        # Predation events
        if self.predations:
            lines.append("-" * 70)
            lines.append("  PREDATION EVENTS")
            lines.append("-" * 70)
            lines.append("")
            lines.append(f"  Total kills by predators: {len(self.predations)}")
            
            # Group by predator
            by_predator = defaultdict(list)
            for p in self.predations:
                by_predator[p['predator']].append(p['victim'])
            
            for pred, victims in by_predator.items():
                lines.append(f"    {pred}: {len(victims)} kills")
                if len(victims) <= 5:
                    lines.append(f"      Victims: {', '.join(str(v) for v in victims)}")
                else:
                    lines.append(f"      Victims include: {', '.join(str(v) for v in victims[:5])}...")
            lines.append("")
        
        # Bonded pairs
        if self.bonds:
            lines.append("-" * 70)
            lines.append("  LOVE STORIES")
            lines.append("-" * 70)
            lines.append("")
            lines.append(f"  {len(self.bonds)} pairs bonded during this simulation:")
            lines.append("")
            for bond in self.bonds[:20]:  # First 20
                p1, p2 = bond['partners']
                res = bond.get('resonance', 0)
                if res > 0.95:
                    desc = "soulmates"
                elif res > 0.8:
                    desc = "deeply bonded"
                else:
                    desc = "partners"
                lines.append(f"    {p1} & {p2} - {desc}")
            if len(self.bonds) > 20:
                lines.append(f"    ... and {len(self.bonds) - 20} more pairs")
            lines.append("")
        
        # Timeline of notable events
        lines.append("-" * 70)
        lines.append("  TIMELINE OF NOTABLE EVENTS")
        lines.append("-" * 70)
        lines.append("")
        
        timeline = self._build_timeline()
        for event in timeline[:30]:  # First 30 events
            lines.append(f"    Step {event['step']:>6}: {event['description']}")
        if len(timeline) > 30:
            lines.append(f"    ... and {len(timeline) - 30} more events")
        lines.append("")
        
        # Footer
        lines.append("=" * 70)
        lines.append("  END OF HISTORY")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _readable_cause(self, cause: str) -> str:
        """Convert cause code to readable string."""
        if cause.startswith('predation:'):
            pred = cause.split(':')[1]
            return f"Killed by {pred}"
        elif cause == 'starvation':
            return "Starvation"
        elif cause == 'old_age':
            return "Old age"
        elif cause == 'genesis_cleansing':
            return "Genesis Leviathan"
        elif cause == 'disease':
            return "Disease"
        elif cause == 'unknown':
            return "Unknown causes"
        else:
            return cause.replace('_', ' ').title()
    
    def _build_timeline(self) -> List[dict]:
        """Build a timeline of notable events."""
        events = []
        
        # First events
        if self.births:
            first_birth = self.births[0]
            events.append({
                'step': first_birth['step'],
                'description': f"First birth: {first_birth['name']} ({first_birth['sex']})"
            })
        
        if self.first_death:
            events.append({
                'step': self.first_death['step'],
                'description': f"First death: {self.first_death['name']} ({self._readable_cause(self.first_death['cause'])})"
            })
        
        # First bond
        if self.bonds:
            first_bond = self.bonds[0]
            p1, p2 = first_bond['partners']
            events.append({
                'step': first_bond['step'],
                'description': f"First bond: {p1} & {p2}"
            })
        
        # Season changes
        for s in self.seasons:
            events.append({
                'step': s['step'],
                'description': f"Season changed to {s['season']} (Year {s['year']})"
            })
        
        # Predations
        for p in self.predations[:10]:  # First 10 predations
            events.append({
                'step': p['step'],
                'description': f"{p['predator']} killed {p['victim']}"
            })
        
        # Generation milestones
        gen_firsts = {}
        for b in self.births:
            gen = b.get('generation', 0)
            if gen > 1 and gen not in gen_firsts:
                gen_firsts[gen] = b
                events.append({
                    'step': b['step'],
                    'description': f"First Gen{gen} born: {b['name']}"
                })
        
        # Sort by step
        events.sort(key=lambda x: x['step'])
        return events
    
    def save_report(self, final_step: int, final_counts: dict, filepath: str = None):
        """Save the report to a file."""
        if filepath is None:
            filepath = os.path.join(DATA_DIR, 'world_history.txt')
        
        report = self.generate_report(final_step, final_counts)
        
        try:
            with open(filepath, 'w') as f:
                f.write(report)
            print(f"[History] Saved world history to {filepath}")
        except Exception as e:
            print(f"[History] Failed to save: {e}")


# Global accessor
def world_history() -> WorldHistory:
    """Get the singleton WorldHistory instance."""
    return WorldHistory.get()
