"""
Element Objects and Fields - Physical elements and reaction-diffusion dynamics.

ElementObject: A physical element that can be picked up, carried, and reacts
ElementField: Global reaction-diffusion field for element chemistry
"""

from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

from .elements import ElementType, ELEMENT_PROPS, ElementProperties
from ..world.objects import WorldObject
from ..core.constants import WORLD_L

if TYPE_CHECKING:
    pass


@dataclass
class GrippableProperties:
    """Properties of an object that can be gripped."""
    grip_difficulty: float = 0.5  # 0=easy, 1=hard
    weight: float = 1.0           # Affects carry speed
    tool_damage: float = 0.0      # Damage when used as weapon
    tool_type: str = "none"       # "blunt", "sharp", "material"
    stackable: bool = False       # Can be stacked/built with


class ElementObject(WorldObject):
    """
    A physical element that can be picked up, carried, placed, and reacts
    with other elements via reaction-diffusion field dynamics.
    """
    
    def __init__(self, pos, element_type: ElementType, amount: float = 1.0,
                 placed_by: str = None):
        props = ELEMENT_PROPS[element_type]
        
        mass = props.density * amount
        size = 0.3 + 0.4 * (amount ** 0.33)
        compliance = 0.3 if props.stackable else 0.6
        
        super().__init__(
            pos=pos, size=size, compliance=compliance,
            mass=mass, color=props.color, energy=0.0
        )
        
        self.element_type = element_type
        self.amount = amount
        self.props = props
        self.placed_by = placed_by
        self.placed_at = 0
        self.age = 0
        
        self.field_radius = 2.0 + amount * 0.5
        self.field_strength = amount * props.reactivity
        self.temperature = 0.0
        self.pending_products: List[Tuple[ElementType, float]] = []
        
        self.grip_props = GrippableProperties(
            grip_difficulty=0.3 + props.density * 0.15,
            weight=mass,
            tool_damage=props.density * 0.5 if props.stackable else 0.1,
            tool_type='blunt' if props.stackable else 'material',
            stackable=props.stackable
        )
    
    def get_field_contribution(self, query_pos: np.ndarray) -> float:
        """Get this element's field strength at a query position."""
        if not self.alive:
            return 0.0
        dist = np.linalg.norm(query_pos - self.pos)
        if dist > self.field_radius * 2:
            return 0.0
        return self.field_strength * np.exp(-dist**2 / (self.field_radius**2))
    
    def update_element(self, dt_step: float, nearby_elements: List['ElementObject'],
                       ambient_temp: float = 0.0):
        """Update element state."""
        self.age += 1
        self.temperature = self.temperature * 0.98 + ambient_temp * 0.02
        
        if self.props.decay_rate > 0:
            decay = self.props.decay_rate * (1 + self.temperature * 0.018)
            self.amount = max(0.01, self.amount - decay)
        
        self.size = 0.3 + 0.4 * (self.amount ** 0.33)
        self.field_strength = self.amount * self.props.reactivity
        
        if self.props.decay_rate > 0 and self.amount < 0.02:
            self.alive = False
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'type': 'element',
            'element_type': self.element_type.name,
            'pos': self.pos.tolist(),
            'amount': self.amount,
            'age': self.age,
            'temperature': self.temperature,
            'placed_by': self.placed_by,
            'placed_at': self.placed_at,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ElementObject':
        """Deserialize from persistence."""
        elem = cls(
            pos=np.array(d['pos']),
            element_type=ElementType[d['element_type']],
            amount=d.get('amount', 1.0),
            placed_by=d.get('placed_by'),
        )
        elem.age = d.get('age', 0)
        elem.temperature = d.get('temperature', 0.0)
        elem.placed_at = d.get('placed_at', 0)
        return elem


class ElementField:
    """
    Manages the global reaction-diffusion field for element chemistry.
    Each element type has its own concentration field that diffuses and reacts.
    """
    
    def __init__(self, world_size: float = 100.0, resolution: int = 64):
        """Initialize element field."""
        self.world_size = world_size
        self.resolution = resolution
        self.cell_size = world_size / resolution
        
        self.fields: Dict[ElementType, np.ndarray] = {
            elem: np.zeros((resolution, resolution))
            for elem in ElementType
        }
        self.temperature = np.zeros((resolution, resolution))
        self.reaction_sites: List[Tuple[int, int, str]] = []
        self.total_reactions = 0
        
        # Diffusion kernel
        self.diff_kernel = np.array([
            [0.05, 0.2, 0.05],
            [0.2, 0.0, 0.2],
            [0.05, 0.2, 0.05]
        ])
    
    def world_to_grid(self, pos: np.ndarray) -> Tuple[int, int]:
        """Convert world position to grid indices."""
        i = int((pos[0] + self.world_size/2) / self.cell_size)
        j = int((pos[1] + self.world_size/2) / self.cell_size)
        return (
            np.clip(i, 0, self.resolution - 1),
            np.clip(j, 0, self.resolution - 1)
        )
    
    def grid_to_world(self, i: int, j: int) -> np.ndarray:
        """Convert grid indices to world position."""
        x = (i + 0.5) * self.cell_size - self.world_size/2
        y = (j + 0.5) * self.cell_size - self.world_size/2
        return np.array([x, y])
    
    def deposit(self, pos: np.ndarray, element_type: ElementType, amount: float):
        """Deposit element concentration at a position."""
        i, j = self.world_to_grid(pos)
        self.fields[element_type][j, i] += amount
    
    def sample(self, pos: np.ndarray, element_type: ElementType) -> float:
        """Sample element concentration at a position."""
        i, j = self.world_to_grid(pos)
        return self.fields[element_type][j, i]
    
    def sample_all(self, pos: np.ndarray) -> Dict[ElementType, float]:
        """Sample all element concentrations at a position."""
        i, j = self.world_to_grid(pos)
        return {elem: self.fields[elem][j, i] for elem in ElementType}
    
    def add_heat(self, pos: np.ndarray, amount: float, radius: float = 2.0):
        """Add heat at a position."""
        i, j = self.world_to_grid(pos)
        for di in range(-3, 4):
            for dj in range(-3, 4):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.resolution and 0 <= nj < self.resolution:
                    dist = np.sqrt(di**2 + dj**2) * self.cell_size
                    if dist < radius:
                        self.temperature[nj, ni] += amount * (1 - dist/radius)
    
    def step(self, elements: List[ElementObject]):
        """
        Step the reaction-diffusion system.
        """
        self.reaction_sites.clear()
        
        # Deposit from element objects
        for elem_obj in elements:
            if elem_obj.alive:
                self.deposit(elem_obj.pos, elem_obj.element_type, 
                           elem_obj.amount * 0.05)
                if elem_obj.temperature > 0.5:
                    self.add_heat(elem_obj.pos, elem_obj.temperature * 0.1)
        
        # Diffuse each field
        for elem_type in ElementType:
            props = ELEMENT_PROPS[elem_type]
            field = self.fields[elem_type]
            
            diffused = np.zeros_like(field)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    diffused += np.roll(np.roll(field, di, axis=1), dj, axis=0) * \
                               self.diff_kernel[dj+1, di+1]
            
            self.fields[elem_type] = field * (1 - props.diffusion_rate) + \
                                     diffused * props.diffusion_rate
            
            if props.decay_rate > 0:
                self.fields[elem_type] *= (1 - props.decay_rate)
        
        # Diffuse temperature
        temp_diffused = np.zeros_like(self.temperature)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                temp_diffused += np.roll(np.roll(self.temperature, di, axis=1), dj, axis=0) * \
                                self.diff_kernel[dj+1, di+1]
        self.temperature = self.temperature * 0.85 + temp_diffused * 0.15
        self.temperature *= 0.98
        
        # Check for reactions
        self._check_reactions(elements)
    
    def _check_reactions(self, elements: List[ElementObject]):
        """Check for chemical reactions based on field concentrations."""
        REACT_THRESHOLD = 0.6
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                ite = self.fields[ElementType.ITE][j, i]
                fiber = self.fields[ElementType.ITE_LITE][j, i]
                ore = self.fields[ElementType.ORE][j, i]
                vapor = self.fields[ElementType.VAPOR][j, i]
                mulch = self.fields[ElementType.MULCHITE][j, i]
                volatile = self.fields[ElementType.LITE_ORE][j, i]
                temp = self.temperature[j, i]
                
                # FIRE: ITE_LITE + heat
                if fiber > REACT_THRESHOLD and temp > 1.2:
                    burn_amount = min(fiber, 0.02) * temp * 0.5
                    self.fields[ElementType.ITE_LITE][j, i] -= burn_amount
                    self.temperature[j, i] += burn_amount * 1.5
                    self.reaction_sites.append((i, j, 'fire'))
                    self.total_reactions += 1
                    self._log_reaction('fire', i, j, burn_amount)
                
                # COMPOSTING: ITE + ITE_LITE + VAPOR → MULCHITE
                if ite > 0.3 and fiber > 0.3 and vapor > 0.4:
                    react_amount = min(ite, fiber, vapor * 0.5) * 0.02
                    self.fields[ElementType.ITE][j, i] -= react_amount * 0.1
                    self.fields[ElementType.ITE_LITE][j, i] -= react_amount * 0.5
                    self.fields[ElementType.VAPOR][j, i] -= react_amount * 0.3
                    self.fields[ElementType.MULCHITE][j, i] += react_amount * 1.5
                    self.reaction_sites.append((i, j, 'compost'))
                    self.total_reactions += 1
                    self._log_reaction('compost', i, j, react_amount)
                
                # HOT ORE
                if ore > REACT_THRESHOLD and temp > 1.5:
                    self.reaction_sites.append((i, j, 'hot_ore'))
                
                # TEMPERING: ITE + ORE + high heat
                if ite > 0.4 and ore > 0.4 and temp > 2.0:
                    react_amount = min(ite, ore) * 0.01
                    self.fields[ElementType.ITE][j, i] -= react_amount
                    self.fields[ElementType.ORE][j, i] -= react_amount
                    self.temperature[j, i] -= 0.2
                    self.reaction_sites.append((i, j, 'temper'))
                    self.total_reactions += 1
                    self._log_reaction('temper', i, j, react_amount)
                
                # EXPLOSION: LITE_ORE + heat
                if volatile > 0.5 and temp > 1.5:
                    explode_amount = volatile * temp * 0.1
                    self.fields[ElementType.LITE_ORE][j, i] *= 0.7
                    self.temperature[j, i] += explode_amount * 2
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.resolution and 0 <= nj < self.resolution:
                                self.temperature[nj, ni] += explode_amount * 0.3
                    self.reaction_sites.append((i, j, 'explosion'))
                    self.total_reactions += 1
                    self._log_reaction('explosion', i, j, explode_amount)
                
                # CONDENSATION: VAPOR + cold
                if vapor > 0.7 and temp < 0.1:
                    self.fields[ElementType.VAPOR][j, i] *= 0.95
    
    def _log_reaction(self, reaction_type: str, grid_i: int, grid_j: int, amount: float):
        """Log significant reactions to event system."""
        # Only log significant reactions (not every tiny tick)
        if amount < 0.01:
            return
        
        # Convert grid to world coords
        world_x = (grid_i / self.resolution - 0.5) * self.world_size
        world_y = (grid_j / self.resolution - 0.5) * self.world_size
        
        try:
            from ..events.logger import event_log
            from ..events.narrative_log import narrative_log
            
            reaction_names = {
                'fire': '🔥 FIRE',
                'compost': '🌱 COMPOSTING', 
                'temper': '⚒️ TEMPERING',
                'explosion': '💥 EXPLOSION'
            }
            
            readable = reaction_names.get(reaction_type, reaction_type.upper())
            
            event_log().log('reaction', 0,
                reaction_type=reaction_type,
                pos=(world_x, world_y),
                amount=amount,
                total_reactions=self.total_reactions
            )
            
            # Only narrative log explosions and significant events
            if reaction_type == 'explosion' or amount > 0.05:
                narrative_log().log(
                    f"[CHEMISTRY] {readable} at ({world_x:.0f}, {world_y:.0f})!",
                    step=0, force=True
                )
        except ImportError:
            pass
    
    def get_reaction_at(self, pos: np.ndarray) -> Optional[str]:
        """Check if there's an active reaction at a position."""
        i, j = self.world_to_grid(pos)
        for ri, rj, rtype in self.reaction_sites:
            if ri == i and rj == j:
                return rtype
        return None
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'fields': {elem.name: field.tolist() 
                      for elem, field in self.fields.items()},
            'temperature': self.temperature.tolist(),
            'total_reactions': self.total_reactions,
        }
    
    @classmethod
    def from_dict(cls, d: dict, world_size: float = 100.0) -> 'ElementField':
        """Deserialize from persistence."""
        resolution = len(d['temperature'])
        field = cls(world_size=world_size, resolution=resolution)
        for elem_name, data in d.get('fields', {}).items():
            try:
                elem_type = ElementType[elem_name]
                field.fields[elem_type] = np.array(data)
            except KeyError:
                pass
        field.temperature = np.array(d['temperature'])
        field.total_reactions = d.get('total_reactions', 0)
        return field


# =============================================================================
# CONSTRUCTION - Built structures from placed elements
# =============================================================================

@dataclass
class Construction:
    """
    A persistent construction made from placed elements.
    Emerges when stackable elements are placed near each other.
    """
    construction_id: str
    pos: np.ndarray
    elements: List[str]  # element object IDs that form this
    builder_id: str  # First creature to place an element here
    contributors: set  # All creatures who added to it
    created_at: int  # Step when first element placed
    last_modified: int  # Step when last element added
    height: int = 1  # Stack height
    stability: float = 1.0  # 0-1, decreases over time without maintenance
    construction_type: str = "pile"  # "pile", "wall", "shelter", "unknown"
    
    def __post_init__(self):
        if isinstance(self.pos, list):
            self.pos = np.array(self.pos)
        if isinstance(self.contributors, list):
            self.contributors = set(self.contributors)
    
    def add_element(self, element_id: str, contributor_id: str, step: int):
        """Add an element to this construction."""
        self.elements.append(element_id)
        self.contributors.add(contributor_id)
        self.last_modified = step
        self.height = min(10, len(self.elements))
        self.stability = min(1.0, self.stability + 0.1)
        self._update_type()
    
    def _update_type(self):
        """Determine construction type based on shape."""
        n = len(self.elements)
        if n < 3:
            self.construction_type = "pile"
        elif n < 6:
            self.construction_type = "mound"
        elif self.height >= 4:
            self.construction_type = "wall"
        else:
            self.construction_type = "shelter"
    
    def decay(self, steps_since_modify: int):
        """Constructions decay without maintenance."""
        if steps_since_modify > 2000:
            self.stability -= 0.001
        if steps_since_modify > 5000:
            self.stability -= 0.002
        self.stability = max(0, self.stability)
    
    def is_collapsed(self) -> bool:
        """Check if construction has collapsed."""
        return self.stability < 0.1
    
    def get_shelter_value(self) -> float:
        """How much shelter does this provide?"""
        if self.construction_type == "shelter":
            return 0.3 * self.height * self.stability
        elif self.construction_type == "wall":
            return 0.2 * self.height * self.stability
        return 0.0
    
    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            'construction_id': self.construction_id,
            'pos': self.pos.tolist() if isinstance(self.pos, np.ndarray) else self.pos,
            'elements': self.elements,
            'builder_id': self.builder_id,
            'contributors': list(self.contributors),
            'created_at': self.created_at,
            'last_modified': self.last_modified,
            'height': self.height,
            'stability': self.stability,
            'construction_type': self.construction_type,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Construction':
        """Deserialize from persistence."""
        return cls(
            construction_id=d['construction_id'],
            pos=np.array(d['pos']),
            elements=d['elements'],
            builder_id=d['builder_id'],
            contributors=set(d.get('contributors', [])),
            created_at=d['created_at'],
            last_modified=d['last_modified'],
            height=d.get('height', 1),
            stability=d.get('stability', 1.0),
            construction_type=d.get('construction_type', 'pile'),
        )
