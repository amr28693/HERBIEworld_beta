"""
ElementSpawner - Manages terrain-based element spawning.

Elements appear naturally in their source terrains:
- ITE (stone) from mountains/caves
- ITE_LITE (fiber/wood) from forests
- ORE (metal) from caves/hills
- VAPOR (mist) from water/shore (boosted during rain)
- MULCHITE (organic) from forest/plains
- LITE_ORE (crystal) from caves - rare but powerful
"""

import numpy as np
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from .elements import (
    ElementType, ElementProperties, ELEMENT_PROPS,
    ELEMENT_TERRAIN_SOURCES
)
from .element_objects import ElementObject, Construction, ElementField
from ..core.constants import WORLD_L, dt

if TYPE_CHECKING:
    from ..world.multi_world import MultiWorld
    from ..world.terrain import Terrain
    from ..world.objects import WorldObject


class ElementSpawner:
    """
    Manages spawning of elements from terrain.
    Elements appear naturally in their source terrains.
    """
    
    def __init__(self, world: 'MultiWorld', terrain_system: 'Terrain'):
        self.world = world
        self.terrain = terrain_system
        
        # Spawn tracking
        self.element_objects: List[ElementObject] = []
        self.constructions: Dict[str, Construction] = {}
        self.next_construction_id = 0
        
        # Element field for reaction-diffusion
        self.element_field = ElementField(
            world_size=getattr(world, 'world_size', WORLD_L),
            resolution=48
        )
        
        # Spawn rates per terrain type (abundant world)
        self.spawn_rates = {
            ElementType.ITE: 0.008,        # Abundant stone
            ElementType.ITE_LITE: 0.012,   # Very abundant fiber/wood
            ElementType.ORE: 0.004,        # Common metal
            ElementType.VAPOR: 0.003,      # Mist - boosted more during rain
            ElementType.MULCHITE: 0.010,   # Abundant organic matter
            ElementType.LITE_ORE: 0.002,   # Uncommon but findable - THE APEX BANE
        }
        
        # Max elements in world (much higher)
        self.max_elements = 1500  # Allow a rich chemical world
    
    def step(self, step_count: int, is_raining: bool = False):
        """Update element system each step."""
        
        # 1. Maybe spawn new elements from terrain (every 10 steps)
        if len(self.element_objects) < self.max_elements and step_count % 10 == 0:
            self._try_spawn_elements(is_raining)
            self._try_spawn_elements(is_raining)  # Always try twice
            # Extra spawn when population is low
            if len(self.element_objects) < self.max_elements * 0.5:
                self._try_spawn_elements(is_raining)
                self._try_spawn_elements(is_raining)
                self._try_spawn_elements(is_raining)
        
        # 2. Update element field (reaction-diffusion)
        self.element_field.step(self.element_objects)
        
        # 3. Update individual elements
        alive_elements = []
        for elem in self.element_objects:
            if elem.alive:
                # Get nearby elements for interaction
                nearby = [e for e in self.element_objects 
                         if e != elem and e.alive and 
                         np.linalg.norm(e.pos - elem.pos) < 5.0]
                
                # Get ambient temperature from field
                grid_pos = self.element_field.world_to_grid(elem.pos)
                ambient_temp = self.element_field.temperature[grid_pos[1], grid_pos[0]]
                
                elem.update_element(dt, nearby, ambient_temp=ambient_temp)
                alive_elements.append(elem)
        self.element_objects = alive_elements
        
        # 4. Update constructions
        self._update_constructions(step_count)
        
        # 5. Check for reaction products to spawn
        self._process_reaction_products(step_count)
    
    def _try_spawn_elements(self, is_raining: bool):
        """Try to spawn elements based on terrain."""
        world_size = getattr(self.world, 'world_size', WORLD_L)
        half = world_size / 2
        
        # Pick random position
        x = np.random.uniform(-half + 5, half - 5)
        y = np.random.uniform(-half + 5, half - 5)
        pos = np.array([x, y])
        
        # Get terrain name at position
        terrain_name = 'plains'
        if self.terrain:
            terrain_obj = self.terrain.get_terrain_at(pos)
            terrain_name = terrain_obj.name if hasattr(terrain_obj, 'name') else str(terrain_obj)
        
        # Check each element type
        for elem_type, sources in ELEMENT_TERRAIN_SOURCES.items():
            if terrain_name in sources:
                rate = self.spawn_rates.get(elem_type, 0.005)
                
                # Boost vapor during rain
                if elem_type == ElementType.VAPOR and is_raining:
                    rate *= 5.0
                
                if np.random.random() < rate:
                    amount = np.random.uniform(0.3, 1.0)
                    elem = ElementObject(pos, elem_type, amount)
                    self.element_objects.append(elem)
                    return  # Only spawn one per check
    
    def _update_constructions(self, step_count: int):
        """Update construction stability and remove collapsed ones."""
        to_remove = []
        for cid, construction in self.constructions.items():
            steps_since = step_count - construction.last_modified
            construction.decay(steps_since)
            if construction.is_collapsed():
                to_remove.append(cid)
                # Scatter elements when collapsed
                for elem_id in construction.elements:
                    # Find element and scatter it
                    for elem in self.element_objects:
                        if '_' in elem_id:
                            try:
                                target_id = int(elem_id.split('_')[-1])
                                if id(elem) == target_id:
                                    scatter = np.random.randn(2) * 2
                                    elem.pos = elem.pos + scatter
                            except ValueError:
                                pass
        
        for cid in to_remove:
            del self.constructions[cid]
    
    def _process_reaction_products(self, step_count: int):
        """Spawn new elements/objects from reactions."""
        for elem in self.element_objects:
            for product_type, amount in elem.pending_products:
                if len(self.element_objects) < self.max_elements:
                    offset = np.random.randn(2) * 0.5
                    new_elem = ElementObject(
                        elem.pos + offset, product_type, amount,
                        placed_by=elem.placed_by
                    )
                    self.element_objects.append(new_elem)
            elem.pending_products.clear()
    
    def place_element(self, element: ElementObject, creature_id: str, step_count: int):
        """
        Handle a creature placing an element.
        May create or add to a construction.
        """
        if not element.props.stackable:
            return  # Non-stackable elements just get dropped
        
        element.placed_by = creature_id
        element.placed_at = step_count
        
        # Check for nearby constructions
        for cid, construction in self.constructions.items():
            dist = np.linalg.norm(element.pos - construction.pos)
            if dist < 3.0:  # Close enough to add to existing
                old_type = construction.construction_type
                construction.add_element(str(id(element)), creature_id, step_count)
                # Average position toward center
                construction.pos = (construction.pos + element.pos) / 2
                # Log the expansion
                if construction.construction_type != old_type:
                    print(f"[CONSTRUCTION] {cid} upgraded: {old_type} → {construction.construction_type} (h={construction.height})")
                else:
                    print(f"[CONSTRUCTION] {cid} expanded: {construction.construction_type} h={construction.height} (+1 element)")
                return
        
        # Check for nearby stackable elements (create new construction)
        nearby_stackable = [e for e in self.element_objects 
                          if e != element and e.alive and e.props.stackable
                          and np.linalg.norm(e.pos - element.pos) < 2.0]
        
        if len(nearby_stackable) >= 1:  # At least 2 stackable elements together
            # Create new construction
            cid = f"C{self.next_construction_id:05d}"
            self.next_construction_id += 1
            
            center = element.pos.copy()
            elem_ids = [str(id(element))]
            for e in nearby_stackable:
                center = center + e.pos
                elem_ids.append(str(id(e)))
            center = center / (len(nearby_stackable) + 1)
            
            construction = Construction(
                construction_id=cid,
                pos=center,
                elements=elem_ids,
                builder_id=creature_id,
                contributors={creature_id},
                created_at=step_count,
                last_modified=step_count,
            )
            self.constructions[cid] = construction
            print(f"[CONSTRUCTION] ★ New {cid} built at ({center[0]:+.1f}, {center[1]:+.1f}) by {creature_id[:12]}!")
    
    def get_elements_near(self, pos: np.ndarray, radius: float = 5.0) -> List[ElementObject]:
        """Get all elements within radius of position."""
        return [e for e in self.element_objects 
                if e.alive and np.linalg.norm(e.pos - pos) < radius]
    
    def get_construction_at(self, pos: np.ndarray, radius: float = 3.0) -> Optional[Construction]:
        """Get construction near position."""
        for construction in self.constructions.values():
            if np.linalg.norm(construction.pos - pos) < radius:
                return construction
        return None
    
    def get_shelter_at(self, pos: np.ndarray) -> float:
        """Get total shelter value at position."""
        shelter = 0.0
        for construction in self.constructions.values():
            dist = np.linalg.norm(construction.pos - pos)
            if dist < 5.0:
                shelter += construction.get_shelter_value() * (1 - dist/5.0)
        return min(1.0, shelter)
    
    def get_grippable_elements(self) -> List:
        """Get elements as world objects for grip system."""
        return [e for e in self.element_objects if e.alive]
    
    def get_status(self) -> str:
        """Get status string for display."""
        n_elem = len(self.element_objects)
        n_const = len(self.constructions)
        
        # Count by type
        type_counts = {}
        for elem in self.element_objects:
            name = elem.element_type.name[:3]
            type_counts[name] = type_counts.get(name, 0) + 1
        
        types_str = " ".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
        
        return f"Elem:{n_elem} Const:{n_const} | {types_str}"
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'elements': [e.to_dict() for e in self.element_objects if e.alive],
            'constructions': {cid: c.to_dict() 
                             for cid, c in self.constructions.items()},
            'next_construction_id': self.next_construction_id,
            'element_field': self.element_field.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, d: dict, world, terrain) -> 'ElementSpawner':
        """Deserialize from persistence."""
        spawner = cls(world, terrain)
        
        # Load elements
        for elem_data in d.get('elements', []):
            try:
                elem = ElementObject.from_dict(elem_data)
                spawner.element_objects.append(elem)
            except Exception as e:
                print(f"[Chemistry] Failed to load element: {e}")
        
        # Load constructions
        for cid, const_data in d.get('constructions', {}).items():
            try:
                construction = Construction.from_dict(const_data)
                spawner.constructions[cid] = construction
            except Exception as e:
                print(f"[Chemistry] Failed to load construction: {e}")
        
        spawner.next_construction_id = d.get('next_construction_id', 0)
        
        # Load element field
        if 'element_field' in d:
            spawner.element_field = ElementField.from_dict(
                d['element_field'], 
                getattr(world, 'world_size', WORLD_L)
            )
        
        return spawner


def integrate_chemistry_to_world_objects(world_objects: List, 
                                         element_spawner: ElementSpawner) -> List:
    """
    Integrate element objects into the world objects list for grip detection.
    Call this before grip checks.
    """
    grippable_elements = element_spawner.get_grippable_elements()
    return world_objects + grippable_elements


def check_element_interaction(creature, element_spawner: ElementSpawner):
    """
    Check if creature is interacting with elements in meaningful ways.
    Called during creature step. Includes sound generation!
    """
    if not hasattr(creature, 'hands'):
        return
    
    # Import here to avoid circular dependency
    from ..audio.soundscape import get_world_soundscape
    from ..events.logger import event_log
    
    # Check what creature is holding
    held = creature.hands.get_held_objects()
    for obj in held:
        if isinstance(obj, ElementObject):
            # Element is being carried - deposit to field at creature position
            element_spawner.element_field.deposit(
                creature.pos, obj.element_type, obj.amount * 0.02
            )
            
            # Check if creature is near a reaction site
            reaction = element_spawner.element_field.get_reaction_at(creature.pos)
            if reaction == 'fire' and obj.props.flammability > 0.3:
                obj.temperature += 0.1
    
    # Strike detection - Both hands have objects
    left_obj = creature.hands.left.held_object
    right_obj = creature.hands.right.held_object
    
    if left_obj is not None and right_obj is not None:
        left_tip = creature.hands.left.get_tip_position(creature.pos)
        right_tip = creature.hands.right.get_tip_position(creature.pos)
        
        hand_dist = np.linalg.norm(left_tip - right_tip)
        
        if hand_dist < 2.0:
            left_active = creature.hands.left.tip_activation
            right_active = creature.hands.right.tip_activation
            
            combined_activation = left_active + right_active
            
            if combined_activation > 4.0:
                last_strike = getattr(creature, '_last_strike_step', -100)
                if creature.step_count - last_strike > 30:
                    soundscape = get_world_soundscape()
                    if soundscape is not None:
                        strike_pos = (left_tip + right_tip) / 2
                        velocity = combined_activation * 0.3
                        
                        creature_id = creature.creature_id if hasattr(creature, 'creature_id') else ""
                        soundscape.create_strike_sound(
                            strike_pos, left_obj, right_obj, velocity, creature_id
                        )
                        creature._last_strike_step = creature.step_count


def on_element_dropped(element: ElementObject, creature_id: str, 
                       step_count: int, element_spawner: ElementSpawner):
    """
    Called when a creature drops an element.
    Handles construction creation/addition and creates drop sound.
    """
    from ..audio.soundscape import get_world_soundscape
    
    elem_name = element.element_type.name
    print(f"[CHEMISTRY] Creature dropped {elem_name} at ({element.pos[0]:+.1f}, {element.pos[1]:+.1f})")
    element_spawner.place_element(element, creature_id, step_count)
    
    soundscape = get_world_soundscape()
    if soundscape is not None:
        drop_height = 1.0 + np.random.uniform(0, 0.5)
        soundscape.create_drop_sound(element.pos, element, drop_height, creature_id)


def save_chemistry_state(element_spawner: ElementSpawner, filepath: str = None):
    """Save chemistry/construction state to disk."""
    import json
    import os
    
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'herbie_chemistry.json')
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(element_spawner.to_dict(), f, indent=2)
        print(f"[Chemistry] Saved {len(element_spawner.element_objects)} elements, "
              f"{len(element_spawner.constructions)} constructions")
    except Exception as e:
        print(f"[Chemistry] Save failed: {e}")


def load_chemistry_state(world, terrain, filepath: str = None) -> Optional[ElementSpawner]:
    """Load chemistry/construction state from disk."""
    import json
    import os
    
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'herbie_chemistry.json')
    
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        spawner = ElementSpawner.from_dict(data, world, terrain)
        print(f"[Chemistry] Loaded {len(spawner.element_objects)} elements, "
              f"{len(spawner.constructions)} constructions")
        return spawner
    except Exception as e:
        print(f"[Chemistry] Load failed: {e}")
        return None
