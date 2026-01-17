"""
Evolution Tree - Tracks complete evolutionary history of all creatures.

Records births, deaths, traits, lineage, and lifetime statistics.
Enables visualization of family trees and generational analysis.
"""

from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

if TYPE_CHECKING:
    pass


@dataclass
class EvolutionNode:
    """Record of a single creature in the evolution tree."""
    creature_id: str
    species: str
    generation: int
    parent_id: Optional[str]
    birth_step: int
    death_step: Optional[int] = None
    
    # Traits at birth
    traits: dict = field(default_factory=dict)
    
    # Lifetime stats
    offspring_count: int = 0
    food_consumed: float = 0.0
    distance_traveled: float = 0.0
    
    # Herbie-specific
    grip_successes: int = 0
    knowledge_count: int = 0
    
    # Cause of death
    cause_of_death: str = ""
    
    # Position in tree visualization
    tree_x: float = 0.0
    tree_y: float = 0.0
    
    def is_alive(self) -> bool:
        return self.death_step is None
    
    def lifespan(self) -> Optional[int]:
        if self.death_step is None:
            return None
        return self.death_step - self.birth_step
    
    def to_dict(self) -> dict:
        return {
            'creature_id': self.creature_id,
            'species': self.species,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'birth_step': self.birth_step,
            'death_step': self.death_step,
            'traits': self.traits,
            'offspring_count': self.offspring_count,
            'food_consumed': self.food_consumed,
            'distance_traveled': self.distance_traveled,
            'grip_successes': self.grip_successes,
            'knowledge_count': self.knowledge_count,
            'cause_of_death': self.cause_of_death
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EvolutionNode':
        return cls(
            creature_id=data['creature_id'],
            species=data['species'],
            generation=data['generation'],
            parent_id=data.get('parent_id'),
            birth_step=data['birth_step'],
            death_step=data.get('death_step'),
            traits=data.get('traits', {}),
            offspring_count=data.get('offspring_count', 0),
            food_consumed=data.get('food_consumed', 0.0),
            distance_traveled=data.get('distance_traveled', 0.0),
            grip_successes=data.get('grip_successes', 0),
            knowledge_count=data.get('knowledge_count', 0),
            cause_of_death=data.get('cause_of_death', '')
        )


class EvolutionTree:
    """
    Tracks complete evolutionary history of all creatures.
    
    Structure:
    - nodes: Dict[creature_id, EvolutionNode]
    - children: Dict[creature_id, List[creature_id]]
    - roots: List[creature_id] (creatures with no parent)
    """
    
    def __init__(self):
        """Initialize empty evolution tree."""
        self.nodes: Dict[str, EvolutionNode] = {}
        self.children: Dict[str, List[str]] = {}
        self.roots: List[str] = []
        
        # Statistics
        self.total_births = 0
        self.total_deaths = 0
        self.max_generation = 0
        self.longest_lineage = 0
    
    def register_birth(self, creature, step: int) -> EvolutionNode:
        """Register a new creature birth."""
        node = EvolutionNode(
            creature_id=creature.creature_id,
            species=creature.species.name,
            generation=creature.generation,
            parent_id=creature.parent_id if hasattr(creature, 'parent_id') and creature.parent_id else None,
            birth_step=step,
            traits=creature.traits.to_dict() if hasattr(creature, 'traits') and hasattr(creature.traits, 'to_dict') else {}
        )
        
        # Add knowledge count for Herbie
        if creature.species.name == "Herbie":
            node.knowledge_count = len(getattr(creature, 'learned_potential', {}))
            if hasattr(creature, 'hands') and creature.hands:
                node.grip_successes = creature.hands.left.grip_successes + creature.hands.right.grip_successes
        
        self.nodes[creature.creature_id] = node
        
        # Update parent's children list
        if node.parent_id and node.parent_id in self.nodes:
            if node.parent_id not in self.children:
                self.children[node.parent_id] = []
            self.children[node.parent_id].append(creature.creature_id)
            self.nodes[node.parent_id].offspring_count += 1
        else:
            self.roots.append(creature.creature_id)
        
        self.total_births += 1
        self.max_generation = max(self.max_generation, creature.generation)
        
        return node
    
    def register_death(self, creature, step: int, cause: str = ""):
        """Register a creature death."""
        if creature.creature_id in self.nodes:
            node = self.nodes[creature.creature_id]
            node.death_step = step
            node.cause_of_death = cause
            
            # Update final stats
            if hasattr(creature, 'metabolism'):
                node.food_consumed = creature.metabolism.total_consumed
            if hasattr(creature, 'total_distance'):
                node.distance_traveled = creature.total_distance
            
            if creature.species.name == "Herbie":
                node.knowledge_count = len(getattr(creature, 'learned_potential', {}))
                if hasattr(creature, 'hands') and creature.hands:
                    node.grip_successes = creature.hands.left.grip_successes + creature.hands.right.grip_successes
            
            self.total_deaths += 1
    
    def update_living(self, creature):
        """Update stats for living creature."""
        if creature.creature_id in self.nodes:
            node = self.nodes[creature.creature_id]
            if hasattr(creature, 'metabolism'):
                node.food_consumed = creature.metabolism.total_consumed
            if hasattr(creature, 'offspring_count'):
                node.offspring_count = creature.offspring_count
            
            if creature.species.name == "Herbie":
                node.knowledge_count = len(getattr(creature, 'learned_potential', {}))
                if hasattr(creature, 'hands') and creature.hands:
                    node.grip_successes = creature.hands.left.grip_successes + creature.hands.right.grip_successes
    
    def get_lineage(self, creature_id: str) -> List[str]:
        """Get ancestors from creature back to root."""
        lineage = [creature_id]
        current = creature_id
        
        while current in self.nodes:
            parent = self.nodes[current].parent_id
            if parent and parent in self.nodes:
                lineage.append(parent)
                current = parent
            else:
                break
        
        return list(reversed(lineage))
    
    def get_descendants(self, creature_id: str) -> List[str]:
        """Get all descendants of a creature."""
        descendants = []
        queue = [creature_id]
        
        while queue:
            current = queue.pop(0)
            children = self.children.get(current, [])
            descendants.extend(children)
            queue.extend(children)
        
        return descendants
    
    def get_species_tree(self, species: str) -> List[EvolutionNode]:
        """Get all nodes for a specific species."""
        return [node for node in self.nodes.values() if node.species == species]
    
    def get_generation_stats(self, species: str = None) -> Dict[int, dict]:
        """Get statistics per generation."""
        stats = {}
        
        for node in self.nodes.values():
            if species and node.species != species:
                continue
            
            gen = node.generation
            if gen not in stats:
                stats[gen] = {
                    'count': 0,
                    'alive': 0,
                    'dead': 0,
                    'avg_lifespan': 0,
                    'total_offspring': 0,
                    'total_food': 0,
                    'lifespans': []
                }
            
            stats[gen]['count'] += 1
            if node.is_alive():
                stats[gen]['alive'] += 1
            else:
                stats[gen]['dead'] += 1
                if node.lifespan():
                    stats[gen]['lifespans'].append(node.lifespan())
            
            stats[gen]['total_offspring'] += node.offspring_count
            stats[gen]['total_food'] += node.food_consumed
        
        # Calculate averages
        for gen, data in stats.items():
            if data['lifespans']:
                data['avg_lifespan'] = np.mean(data['lifespans'])
            data['avg_offspring'] = data['total_offspring'] / max(data['count'], 1)
            data['avg_food'] = data['total_food'] / max(data['count'], 1)
        
        return stats
    
    def compute_tree_layout(self, species: str = "Herbie"):
        """Compute x,y positions for tree visualization."""
        species_nodes = self.get_species_tree(species)
        if not species_nodes:
            return
        
        by_generation = {}
        for node in species_nodes:
            gen = node.generation
            if gen not in by_generation:
                by_generation[gen] = []
            by_generation[gen].append(node)
        
        for gen, nodes in by_generation.items():
            y = -gen * 2.0
            n = len(nodes)
            for i, node in enumerate(nodes):
                if n == 1:
                    x = 0
                else:
                    x = (i - (n-1)/2) * 3.0
                node.tree_x = x
                node.tree_y = y
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'nodes': {k: v.to_dict() for k, v in self.nodes.items()},
            'children': self.children,
            'roots': self.roots,
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'max_generation': self.max_generation
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EvolutionTree':
        """Deserialize from persistence."""
        tree = cls()
        tree.nodes = {k: EvolutionNode.from_dict(v) for k, v in data.get('nodes', {}).items()}
        tree.children = data.get('children', {})
        tree.roots = data.get('roots', [])
        tree.total_births = data.get('total_births', 0)
        tree.total_deaths = data.get('total_deaths', 0)
        tree.max_generation = data.get('max_generation', 0)
        return tree
    
    def get_summary_string(self, species: str = None) -> str:
        """Get summary string for display."""
        if species:
            nodes = self.get_species_tree(species)
            alive = sum(1 for n in nodes if n.is_alive())
            dead = sum(1 for n in nodes if not n.is_alive())
            max_gen = max((n.generation for n in nodes), default=0)
            return f"{species}: {alive} alive, {dead} dead, max G{max_gen}"
        else:
            return f"Total: {self.total_births} births, {self.total_deaths} deaths, max G{self.max_generation}"


def sync_evolution_tree(tree: EvolutionTree, creatures: List, step: int):
    """Sync evolution tree with current creature list."""
    for creature in creatures:
        if creature.creature_id not in tree.nodes:
            tree.register_birth(creature, step)
        else:
            tree.update_living(creature)
