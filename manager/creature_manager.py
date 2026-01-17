"""
CreatureManager - Main simulation orchestration.

This is a faithful extraction of the full CreatureManagerX chain:
- Base creature lifecycle (spawning, stepping, death/reproduction)
- Weather and seasons  
- Disease outbreaks
- Day/night cycle
- Ant colony
- Leviathan (including Genesis event at step 50)
- Meteor showers
- Blob migrations
- Achievement system
- Culture/nest/brain tracking (observational)
- Digging and smear systems
- Herbie sexual reproduction with full mating

The step_all method maintains the same layered behavior accumulation
as the original D17X -> D17O -> Sexual -> N -> K -> V2 -> Base chain.
"""

from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
import numpy as np

from ..core.constants import WORLD_L
from ..creature.species import (
    SpeciesParams, SPECIES_HERBIE, SPECIES_BLOB, ALL_SPECIES, SPECIES_BY_NAME,
    SPECIES_SPAWN_WEIGHTS
)
from ..creature.creature import Creature
from ..creature.traits import MutatedTraits
from ..creature.herbie import HerbieWithHands
from ..creature.herbie_social import (
    HerbieSex, HerbieGenome, HerbieMatingState, HerbieNameGenerator
)
from ..creature.herbie_hands import add_grip_properties_to_objects
from ..world.multi_world import MultiWorld
from ..world.objects import WorldObject
from ..world.terrain import Terrain
from ..world.weather import WeatherSystem
from ..world.seasons import SeasonSystem, SEASONS
from ..world.day_night import DayNightCycle
from ..world.aquatic import AquaticSystem
from ..world.mycelia import MyceliumNetwork
from ..ecology.disease import DiseaseSystem
from ..ecology.leviathan import Leviathan, LeviathanSystem
from ..ecology.ants import AntColony, herbie_sense_ants
from ..ecology.emergent import (
    CultureTracker, NestTracker, BrainStateTracker,
    DiggingSystem, SmearSystem, FavoriteHerbieTracker, AntCreatureInteraction,
    spawn_pigment_object
)
from ..brain.placement_memory import PlacementMemoryParams
from ..events.logger import event_log
from ..events.world_history import world_history
from ..events.narrative_log import narrative_log
from ..events.console_log import console_log
from ..evolution.tree import EvolutionTree, sync_evolution_tree

if TYPE_CHECKING:
    pass


# =============================================================================
# ACHIEVEMENT SYSTEM
# =============================================================================

class Achievement:
    """A single achievement."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.achieved = False
        self.achieved_step = 0


class AchievementSystem:
    """Tracks and displays achievements."""
    
    ACHIEVEMENTS = {
        'first_bond': ('First Love', 'Two Herbies bonded for the first time'),
        'first_birth': ('New Life', 'A baby Herbie was born'),
        'family_of_five': ('Growing Family', 'A Herbie family reached 5 members'),
        'three_generations': ('Legacy', 'Three generations of Herbies alive'),
        'survived_winter': ('Winter Warriors', 'Herbies survived their first winter'),
        'centenarian': ('Old Soul', 'A Herbie lived 5000 steps'),
        'leviathan_appears': ('Ancient One', 'The Leviathan appeared'),
        'predator_purge': ('Balance Restored', 'Leviathan killed 3+ predators'),
        'ant_swarm': ('Tiny Terrors', 'Ants swarmed a creature'),
        'meteor_shower': ('Starfall', 'Witnessed a meteor shower'),
        'blob_migration': ('The Wandering', 'A blob migration passed through'),
        'first_bloom': ('Spring Awakening', 'First bloom event'),
        'full_year': ('Seasons Turn', 'Completed a full year cycle'),
    }
    
    def __init__(self):
        self.achievements = {
            key: Achievement(name, desc) 
            for key, (name, desc) in self.ACHIEVEMENTS.items()
        }
        self.pending_display = []
        self.display_timer = 0
        self.current_display = None
    
    def unlock(self, key: str, step_count: int) -> bool:
        """Unlock an achievement. Returns True if newly unlocked."""
        if key not in self.achievements:
            return False
        
        ach = self.achievements[key]
        if not ach.achieved:
            ach.achieved = True
            ach.achieved_step = step_count
            self.pending_display.append(ach)
            print(f"[ACHIEVEMENT] {ach.name}: {ach.description}")
            return True
        return False
    
    def update(self):
        """Update display timer."""
        if self.display_timer > 0:
            self.display_timer -= 1
            if self.display_timer <= 0:
                self.current_display = None
        
        if self.current_display is None and self.pending_display:
            self.current_display = self.pending_display.pop(0)
            self.display_timer = 150
        
        return self.current_display
    
    def get_unlocked_count(self) -> Tuple[int, int]:
        """Return (unlocked, total)."""
        unlocked = sum(1 for a in self.achievements.values() if a.achieved)
        return (unlocked, len(self.achievements))
    
    def get_current_display(self):
        """Get the current achievement being displayed (if any)."""
        return self.current_display
        unlocked = sum(1 for a in self.achievements.values() if a.achieved)
        return unlocked, len(self.achievements)


# =============================================================================
# METEOR SHOWER MANAGER
# =============================================================================

class MeteorShowerManager:
    """Manages meteor shower events."""
    
    def __init__(self, world_size: float):
        self.world_size = world_size
        self.active = False
        self.shower_step = 0
        self.shower_duration = 200
        self.impacts = []
        self.last_shower = -10000
        self.shower_interval = 3000  # Steps between showers
    
    def update(self, world) -> dict:
        """Update meteor shower state."""
        events = {'started': False, 'impacts': [], 'crater_food': []}
        
        if self.active:
            self.shower_step += 1
            
            # Random impacts during shower
            if np.random.random() < 0.1:
                pos = np.array([
                    np.random.uniform(-self.world_size/2 + 10, self.world_size/2 - 10),
                    np.random.uniform(-self.world_size/2 + 10, self.world_size/2 - 10)
                ])
                self.impacts.append(pos)
                events['impacts'].append(pos)
                
                # Chance to spawn special food
                if np.random.random() < 0.3:
                    events['crater_food'].append(pos + np.random.randn(2) * 2)
            
            if self.shower_step >= self.shower_duration:
                self.active = False
                print(f"[METEOR] Shower ended. {len(self.impacts)} impacts.")
                self.impacts = []
        else:
            # Check for new shower
            if np.random.random() < 0.0003:
                self.active = True
                self.shower_step = 0
                self.impacts = []
                events['started'] = True
                print("[METEOR] ☄️ Meteor shower beginning!")
        
        return events


# =============================================================================
# BLOB MIGRATION MANAGER
# =============================================================================

class BlobMigrationManager:
    """Manages blob migration events."""
    
    def __init__(self, world_size: float):
        self.world_size = world_size
        self.active = False
        self.migration_blobs = []
        self.direction = np.array([1.0, 0.0])
        self.last_migration = -10000
    
    def update(self, creatures: List, world) -> dict:
        """Update migration state."""
        events = {'started': False, 'left': []}
        
        if self.active:
            # Move migration blobs
            for blob in self.migration_blobs[:]:
                blob['pos'] += self.direction * 0.8
                
                # Check if left world
                if abs(blob['pos'][0]) > self.world_size/2 or abs(blob['pos'][1]) > self.world_size/2:
                    # Chance to leave a blob behind
                    if np.random.random() < 0.2:
                        leave_pos = blob['pos'] - self.direction * 5
                        leave_pos = np.clip(leave_pos, -self.world_size/2 + 5, self.world_size/2 - 5)
                        events['left'].append(leave_pos)
                    self.migration_blobs.remove(blob)
            
            if not self.migration_blobs:
                self.active = False
                print("[MIGRATION] Blob migration passed through.")
        else:
            # Check for new migration
            if np.random.random() < 0.0001:
                self.active = True
                self.direction = np.array([np.random.choice([-1, 1]), 0])
                start_x = -self.direction[0] * (self.world_size/2 - 5)
                
                n_blobs = np.random.randint(5, 15)
                self.migration_blobs = []
                for _ in range(n_blobs):
                    pos = np.array([start_x, np.random.uniform(-self.world_size/3, self.world_size/3)])
                    self.migration_blobs.append({'pos': pos})
                
                events['started'] = True
                print(f"[MIGRATION] 🫧 Blob migration approaching! ({n_blobs} blobs)")
        
        return events


# =============================================================================
# CREATURE MANAGER
# =============================================================================

class CreatureManager:
    """
    Full simulation manager with all D17X features.
    
    Handles:
    - Creature lifecycle (spawning, stepping, death/reproduction)
    - Weather and seasons
    - Day/night cycle
    - Disease outbreaks
    - Ant colony
    - Leviathan (including Genesis event at step 50)
    - Meteor showers
    - Blob migrations
    - Achievement system
    - Herbie sexual reproduction
    - Evolution tree tracking
    """
    
    # Genesis Leviathan spawns at step 50
    GENESIS_STEP = 50
    
    def __init__(self, world: MultiWorld, terrain: Terrain = None):
        """
        Initialize creature manager.
        
        Args:
            world: The world containing objects and terrain
            terrain: Terrain reference (uses world.terrain if None)
        """
        self.world = world
        self.terrain = terrain or world.terrain
        
        # Creatures
        self.creatures: List[Creature] = []
        self.pending_births: List[Tuple] = []
        
        # Launch parameters (set by launcher or defaults)
        self.launch_params = {}
        
        # Core systems - will be re-initialized with params in spawn_initial_population
        self.weather = WeatherSystem()
        self.seasons = SeasonSystem()  # Will be replaced if launch_params set
        self.disease = DiseaseSystem(world_size=world.world_size)
        self.daynight = DayNightCycle(start_time=0.3)  # Start at dawn
        
        # Ghost field - persisting torus wavefunctions after death
        from ..brain.torus import GhostField
        self.ghost_field = GhostField()
        
        # Ant colony (spawn in forest)
        colony_pos = self._find_good_spawn_position('forest')
        self.ant_colony = AntColony(colony_pos, world.world_size, n_ants=25)
        
        # Leviathan system
        self.leviathan_mgr = LeviathanSystem(world.world_size)
        self.genesis_triggered = False
        
        # Event systems
        self.meteor_mgr = MeteorShowerManager(world.world_size)
        self.migration_mgr = BlobMigrationManager(world.world_size)
        self.achievements = AchievementSystem()
        
        # Evolution tracking
        self.evo_tree = EvolutionTree()
        self._evo_tree_ready = False
        
        # Emergent behavior systems (PURELY OBSERVATIONAL)
        self.culture_tracker = CultureTracker()
        self.nest_tracker = NestTracker()
        self.brain_tracker = BrainStateTracker()
        self.digging = DiggingSystem()
        self.smears = SmearSystem()
        self.favorites = FavoriteHerbieTracker()
        
        # Aquatic life system (will be initialized with params in spawn_initial_population)
        self.aquatic = AquaticSystem(world.world_size)
        
        # Mycorrhizal network (underground fungal highways)
        self.mycelia = MyceliumNetwork(world.world_size, resolution=50)
        
        # Herbie lineage tracking
        self.herbie_knowledge = {}
        self.herbie_generation = 0
        self.herbie_lineage_depth = 0
        
        # Protagonist tracking
        self.protagonist_herbie_id = None
        
        # Reset name generator
        HerbieNameGenerator.reset()
        
        # Chemistry system (6 primordial elements)
        self.element_spawner = None
        self.init_chemistry()
        
        self._evo_tree_ready = True
        
        print(f"[Manager] Initialized with {world.world_size}x{world.world_size} world")
        print(f"[Manager] Day/Night cycle: {self.daynight.day_length} steps/day")
        print(f"[Manager] Ant colony at ({colony_pos[0]:.1f}, {colony_pos[1]:.1f})")
        print(f"[Manager] Genesis Leviathan at step {self.GENESIS_STEP}")
        print(f"[Manager] Aquatic system: resource-limited (no hard caps)")
        print(f"[Manager] Mycorrhizal network: {self.mycelia.resolution}x{self.mycelia.resolution} grid")
    
    # =========================================================================
    # SPAWNING
    # =========================================================================
    
    def spawn_initial_population(self):
        """Spawn initial creatures including Herbie tribe."""
        # Get launch parameters
        params = getattr(self, 'launch_params', {})
        
        # Re-initialize season system with launch params
        food_mult = params.get('food_mult', 1.0)
        season_harsh = params.get('season_harsh', 1.0)
        self.seasons = SeasonSystem(
            food_multiplier=food_mult,
            harshness_multiplier=season_harsh
        )
        
        # Re-initialize day/night with custom day length
        day_length = params.get('day_length', 500)
        self.daynight = DayNightCycle(start_time=0.3, day_length=day_length)
        
        # Store mutation rate for creature spawning
        self.mutation_rate = params.get('mutation_rate', 1.0)
        
        # Store embryo development mode: 'herbie', 'all', 'none'
        self.embryo_mode = params.get('embryo_mode', 'herbie')
        
        # Initialize aquatic system with params
        n_fish = params.get('start_fish', 5)
        if self.terrain:
            n_plants = max(10, n_fish * 3)  # More fish = more plants needed
            self.aquatic.initialize(self.terrain, n_plants=n_plants, n_creatures=n_fish)
        
        print(f"[Params] Food abundance: {food_mult:.1f}x")
        print(f"[Params] Seasonal harshness: {season_harsh:.1f}x")
        print(f"[Params] Day length: {day_length} steps")
        print(f"[Params] Mutation rate: {self.mutation_rate:.1f}x")
        print(f"[Params] Aquatic life: {n_fish} fish")
        print(f"[Params] Embryo development: {self.embryo_mode}")
        
        # Spawn Herbies
        spawn_radius = self.world.world_size * 0.20
        n_herbies = params.get('start_pop', 8)
        
        spawn_positions = []
        for i in range(max(n_herbies, 8)):  # At least 8 positions
            angle = (i / max(n_herbies, 8)) * 2 * np.pi + np.random.uniform(-0.2, 0.2)
            pos = np.array([
                spawn_radius * np.cos(angle),
                spawn_radius * np.sin(angle)
            ]) + np.random.randn(2) * 3
            pos = np.clip(pos, -self.world.world_size/2 + 5, self.world.world_size/2 - 5)
            spawn_positions.append(pos)
        
        np.random.shuffle(spawn_positions)
        n_herbies = params.get('start_pop', 8)
        predators_enabled = params.get('predators', True)
        
        # Split into providers and carriers
        n_providers = n_herbies // 2
        n_carriers = n_herbies - n_providers
        
        # Spawn Providers
        providers = []
        for i in range(n_providers):
            provider = self.spawn_herbie(pos=spawn_positions[i % len(spawn_positions)], sex=HerbieSex.PROVIDER)
            providers.append(provider)
            if i == 0:
                self.protagonist_herbie_id = provider.creature_id
        
        # Spawn Carriers
        carriers = []
        for i in range(n_carriers):
            carrier = self.spawn_herbie(pos=spawn_positions[(i + n_providers) % len(spawn_positions)], sex=HerbieSex.CARRIER)
            carriers.append(carrier)
        
        print(f"\n{'='*60}")
        print(f"[Herbie] Tribal Spawn - {n_herbies} founders!")
        print(f"{'='*60}")
        print(f"  PROVIDERS (♂) - {n_providers} males:")
        for i, p in enumerate(providers):
            star = " ★ PROTAGONIST" if i == 0 else ""
            print(f"    '{p.mating_state.name}' at ({p.pos[0]:+6.1f}, {p.pos[1]:+6.1f}){star}")
        print(f"  CARRIERS (♀) - {n_carriers} females:")
        for c in carriers:
            print(f"    '{c.mating_state.name}' at ({c.pos[0]:+6.1f}, {c.pos[1]:+6.1f})")
        print(f"{'='*60}\n")
        
        # Get apex count from params
        apex_count = params.get('start_apex', 1)
        
        # Spawn other species
        for species in ALL_SPECIES:
            if species.name == "Herbie":
                continue
            if species.name == "Apex":
                if not predators_enabled or apex_count == 0:
                    continue
                n_spawn = apex_count
            elif species.name == "Scavenger":
                n_spawn = 1 if predators_enabled else 0
            else:
                n_spawn = 3
            
            for _ in range(n_spawn):
                pos = self._find_good_spawn_position('plains')
                self.spawn_creature(species, pos)
        
        # Spawn pigment objects for art/smearing
        self._spawn_pigment_objects()
        
        # Register all in evolution tree
        for creature in self.creatures:
            if creature.creature_id not in self.evo_tree.nodes:
                self.evo_tree.register_birth(creature, 0)
        
        print(f"[Manager] Evolution tree: {len(self.evo_tree.nodes)} creatures registered")
    
    def spawn_creature(self, species: SpeciesParams, pos: np.ndarray = None,
                       generation: int = 0, parent_id: str = "",
                       lineage_depth: int = 0, inherited_knowledge: dict = None,
                       inherited_traits: MutatedTraits = None) -> Creature:
        """Spawn a creature of any species."""
        if pos is None:
            pos = self._find_good_spawn_position('plains')
        
        if species.name == "Herbie":
            return self.spawn_herbie(pos=pos, generation=generation)
        
        creature = Creature(species, self.world, pos, generation, parent_id,
                           lineage_depth, inherited_knowledge, inherited_traits)
        creature.terrain = self.terrain
        
        self.creatures.append(creature)
        
        # Register with evolution tree
        if self._evo_tree_ready:
            step = self.seasons.step_count if self.seasons else 0
            self.evo_tree.register_birth(creature, step)
        
        return creature
    
    def spawn_herbie(self, pos: np.ndarray = None, sex: HerbieSex = None,
                     genome: HerbieGenome = None, generation: int = 0,
                     parent_ids: Tuple[str, str] = None,
                     is_juvenile: bool = False,
                     parent_placement_params: PlacementMemoryParams = None) -> HerbieWithHands:
        """Spawn a Herbie with full mating system."""
        if pos is None:
            pos = self._find_good_spawn_position('plains')
        
        if sex is None:
            sex = HerbieSex.PROVIDER if np.random.random() < 0.5 else HerbieSex.CARRIER
        
        if genome is None:
            genome = HerbieGenome.random()
        
        # Create mating state
        mating_state = HerbieMatingState()
        mating_state.sex = sex
        mating_state.genome = genome
        mating_state.name = HerbieNameGenerator.generate(sex)
        
        if parent_ids:
            mating_state.parent_ids = parent_ids
        
        if is_juvenile:
            mating_state.is_juvenile = True
            mating_state.maturity = 0.0
        
        # Determine generation from parents
        if parent_ids:
            for c in self.creatures:
                if c.creature_id in parent_ids:
                    generation = max(generation, c.generation + 1)
        
        # Placement memory params
        if parent_placement_params:
            child_pmem_params = PlacementMemoryParams.inherit(parent_placement_params)
        else:
            child_pmem_params = PlacementMemoryParams.random()
        
        # Create Herbie
        herbie = HerbieWithHands(
            SPECIES_HERBIE, self.world, pos,
            generation=generation,
            parent_id=parent_ids[0] if parent_ids else "",
            sex=sex,
            genome=genome,
            mating_state=mating_state,
            placement_memory_params=child_pmem_params
        )
        herbie.terrain = self.terrain
        
        self.creatures.append(herbie)
        
        # Register with evolution tree
        if self._evo_tree_ready:
            step = self.seasons.step_count if self.seasons else 0
            self.evo_tree.register_birth(herbie, step)
        
        sex_str = "Provider ♂" if sex == HerbieSex.PROVIDER else "Carrier ♀"
        juvenile_str = " (baby)" if is_juvenile else ""
        print(f"[Herbie] Spawned Gen{generation} {sex_str} '{mating_state.name}'{juvenile_str}")
        
        # Log birth for genetics analysis (founders too!)
        trait_data = herbie.traits.to_dict() if hasattr(herbie, 'traits') and herbie.traits else None
        event_log().log_birth(
            step=self.seasons.step_count if self.seasons else 0,
            name=mating_state.name,
            sex='male' if sex == HerbieSex.PROVIDER else 'female',
            generation=generation,
            parents=parent_ids,
            pos=(pos[0], pos[1]),
            traits=trait_data,
            creature_id=herbie.creature_id,
            species="Herbie",
            parent_id=parent_ids[0] if parent_ids else None
        )
        
        return herbie
    
    def _find_good_spawn_position(self, preferred_terrain: str = 'plains') -> np.ndarray:
        """Find spawn position on preferred terrain type."""
        for _ in range(50):
            x = np.random.uniform(-self.world.world_size/3, self.world.world_size/3)
            y = np.random.uniform(-self.world.world_size/3, self.world.world_size/3)
            pos = np.array([x, y])
            
            terrain_type = self.terrain.get_terrain_at(pos)
            if terrain_type.name == preferred_terrain:
                return pos
        
        return np.array([0.0, 0.0])
    
    # =========================================================================
    # MAIN STEP
    # =========================================================================
    
    def step_all(self, audio_amp: float = 0.0, silence_frames: int = 100, 
                 audio_system=None):
        """
        Advance simulation by one step.
        
        This is the full step_all from the D17X chain, including:
        - Day/night cycle
        - Genesis Leviathan at step 50
        - Meteor showers
        - Blob migrations
        - Ant colony
        - Achievement tracking
        """
        step = self.seasons.step_count
        
        # === UPDATE DAY/NIGHT ===
        dn_state = self.daynight.update()
        
        # === UPDATE SEASONS ===
        season_msg = self.seasons.update()
        if season_msg:
            print(season_msg)
            event_log().log_season(
                step=step,
                season=self.seasons.season_name,
                year=self.seasons.year
            )
            # Record in world history
            world_history().record_season(
                step=step,
                season=self.seasons.season_name,
                year=self.seasons.year
            )
            # Narrative log
            narrative_log().log_section(f"YEAR {self.seasons.year} - {self.seasons.season_name.upper()}")
            narrative_log().log(season_msg, step, force=True)
        season = self.seasons.effective_season  # Uses multipliers from launcher
        
        # === GENESIS LEVIATHAN (step 50) ===
        if step == self.GENESIS_STEP and not self.genesis_triggered:
            self._trigger_genesis_leviathan()
            self.genesis_triggered = True
        
        # === LEVIATHAN UPDATE ===
        lev_events = self.leviathan_mgr.update(self.creatures, audio_system, world=self.world)
        
        for kill_id in lev_events.get('kills', []):
            print(f"[LEVIATHAN] * Killed predator {kill_id}!")
        
        if lev_events.get('spawned'):
            self.achievements.unlock('leviathan_appears', step)
        
        if len(lev_events.get('kills', [])) >= 3:
            self.achievements.unlock('predator_purge', step)
        
        # === METEOR SHOWER UPDATE ===
        meteor_events = self.meteor_mgr.update(self.world)
        
        if meteor_events.get('started'):
            self.achievements.unlock('meteor_shower', step)
        
        # Spawn star food from craters
        for food_pos in meteor_events.get('crater_food', []):
            star_food = WorldObject(
                pos=food_pos, size=1.5, compliance=0.9,
                mass=0.4, color='gold', energy=60.0
            )
            self.world.objects.append(star_food)
        
        # === BLOB MIGRATION UPDATE ===
        migration_events = self.migration_mgr.update(self.creatures, self.world)
        
        if migration_events.get('started'):
            self.achievements.unlock('blob_migration', step)
        
        for blob_pos in migration_events.get('left', []):
            self.spawn_creature(SPECIES_BLOB, pos=blob_pos)
        
        # === FERTILE PATCH BONUS ===
        self.world._fertile_bonus_fn = lambda pos: self.leviathan_mgr.get_food_growth_bonus(pos)
        
        # === ANT COLONY UPDATE ===
        food_positions = [obj.pos for obj in self.world.objects 
                         if obj.alive and obj.compliance > 0.5]
        self.ant_colony.update(food_positions)
        
        # Ant-creature interactions
        AntCreatureInteraction.process_interactions(
            self.ant_colony, self.creatures, self.world, step)
        
        if hasattr(self.ant_colony, 'last_swarm') and self.ant_colony.last_swarm:
            self.achievements.unlock('ant_swarm', step)
        
        # === UPDATE WEATHER ===
        weather_messages = self.weather.update(self.world)
        for msg in weather_messages:
            print(msg)
        
        # === UPDATE CHEMISTRY ===
        is_raining = any(e.event_type == 'rain' for e in self.weather.active_events)
        wind_strength = 0.5 if any(e.event_type == 'storm' for e in self.weather.active_events) else 0.1
        self.step_chemistry(is_raining, wind_strength)
        
        # === HERBIE BIRTHS ===
        self._process_herbie_births()
        
        # === HERBIE EXTINCTION CHECK ===
        self._check_herbie_extinction()
        
        # === STEP CREATURES ===
        dead_before = set(c.creature_id for c in self.creatures if not c.alive)
        dead_creatures = []
        reproducing_creatures = []
        
        for creature in self.creatures[:]:
            if not creature.alive:
                continue
            
            result = creature.step(self.creatures, audio_amp, silence_frames, 
                                   audio_system)
            
            if result == 'dead':
                dead_creatures.append(creature)
            elif result == 'reproduce':
                reproducing_creatures.append(creature)
        
        # === HANDLE DEATHS ===
        for creature in dead_creatures:
            self._handle_death(creature)
        
        # === CHECK NEW DEATHS FOR EVOLUTION TREE ===
        for creature in self.creatures:
            if not creature.alive and creature.creature_id not in dead_before:
                cause = creature.metabolism.cause_of_death if hasattr(creature, 'metabolism') else 'unknown'
                self.evo_tree.register_death(creature, step, cause)
                
                # === LOG DEATH EVENT ===
                name = None
                if creature.species.name == "Herbie" and hasattr(creature, 'mating_state'):
                    name = creature.mating_state.name
                else:
                    name = f"{creature.species.name}_{creature.creature_id[-3:]}"
                
                event_log().log_death(
                    step=step,
                    name=name,
                    cause=cause,
                    generation=creature.generation,
                    age=creature.step_count,
                    pos=tuple(creature.pos)
                )
                
                # Record in human-readable history
                world_history().record_death(
                    step=step,
                    name=name,
                    cause=cause,
                    age=creature.step_count,
                    generation=creature.generation,
                    species=creature.species.name
                )
                
                # Inject stress signal into mycorrhizal network
                # Death sends a chemical warning through connected plants
                self.mycelia.inject_signal(creature.pos, strength=0.5)
                
                # === GHOST SPAWNING ===
                # Torus brain persists as ghost after death
                if hasattr(creature, 'brain') and creature.brain is not None:
                    ghost_name = None
                    if creature.species.name == "Herbie" and hasattr(creature, 'mating_state'):
                        ghost_name = creature.mating_state.name
                    energy = creature.body.energy if hasattr(creature, 'body') else 0
                    self.ghost_field.spawn_ghost(creature.brain, creature.pos, ghost_name, energy)
                
                # === TERRAIN SCARIFICATION ===
                # Death enriches the soil at that location
                if self.terrain:
                    energy = creature.body.energy if hasattr(creature, 'body') else 10
                    self.terrain.add_death_site(creature.pos, energy)
                
                if creature.species.name == "Herbie" and hasattr(creature, 'mating_state'):
                    HerbieNameGenerator.release(creature.mating_state.name or "")
        
        # === UPDATE LIVING IN TREE ===
        for creature in self.creatures:
            if creature.alive:
                self.evo_tree.update_living(creature)
        
        # === HANDLE REPRODUCTION (non-Herbie) ===
        for creature in reproducing_creatures:
            if creature.species.name != "Herbie":
                self._handle_reproduction(creature)
        
        # === DISEASE ===
        if step % 10 == 0:
            self.disease.update(self.creatures, step)
        
        # === ECOSYSTEM ===
        self.world.update_ecosystem(season, self.world._fertile_bonus_fn)
        
        # === AQUATIC LIFE ===
        self.aquatic.update(self.terrain)
        
        # === GHOST FIELD ===
        # Update persisting torus wavefunctions
        self.ghost_field.update(self.terrain)
        
        # === TERRAIN SCARIFICATION ===
        # Slow healing of terrain scars + footprint accumulation
        if step % 10 == 0 and self.terrain:
            self.terrain.update_scars()
            # Add footprints from moving creatures
            for creature in self.creatures:
                if creature.alive:
                    speed = np.linalg.norm(creature.vel) if hasattr(creature, 'vel') else 0
                    if speed > 0.1:
                        weight = creature.mass if hasattr(creature, 'mass') else 1.0
                        self.terrain.add_footprint(creature.pos, weight * speed * 0.1)
        
        # === MYCORRHIZAL NETWORK ===
        # Update underground fungal network
        self.mycelia.update(self.terrain, self.world.objects)
        # Redistribute nutrients between connected food sources
        if step % 20 == 0:
            self.mycelia.redistribute_nutrients(self.world.objects)
        
        # === CLEAN DEAD ===
        self.creatures = [c for c in self.creatures if c.alive]
        
        # === EMERGENT BEHAVIOR SYSTEMS (PURELY OBSERVATIONAL) ===
        
        # Culture observation - tracks lineage divergence and territorial patterns
        self.culture_tracker.observe(self.creatures, step, self.leviathan_mgr)
        
        # Nest tracking - recognizes emergent homesteading
        self.nest_tracker.update(self.creatures, self.world.nutrients, step)
        
        # Brain state tracking - entropy/KL for proto-emotional states
        brain_snapshots = self.brain_tracker.observe(self.creatures, step)
        if step % 500 == 0 and brain_snapshots:
            self.brain_tracker.log_current_states(brain_snapshots, self.creatures)
        
        # Digging system - mechanical hole physics (emergent burial/caching)
        dig_events = self.digging.update(self.creatures, self.world.objects, step)
        for event in dig_events:
            print(event)
        
        # Smear system - pigment marks encoding brain state (emergent art)
        smear_events = self.smears.update(self.creatures, self.terrain, step)
        for event in smear_events:
            print(event)
        
        # === BEHAVIORAL SNAPSHOT (every 100 steps) ===
        if step % 100 == 0:
            self._log_behavioral_snapshot(step)
        
        # === POPULATION REBALANCING ===
        if step % 100 == 0:
            self._rebalance_population()
        
        # === PERIODIC POPULATION SNAPSHOT ===
        if step % 500 == 0:
            species_counts = {}
            for c in self.creatures:
                if c.alive:
                    species_counts[c.species.name] = species_counts.get(c.species.name, 0) + 1
            event_log().log_population(
                step=step,
                counts=species_counts,
                elements=len([o for o in self.world.objects if hasattr(o, 'element_type')])
            )
            # Record in world history
            world_history().record_population(step=step, counts=species_counts)
            
            # Log ecosystem status to narrative log
            narrative_log().log_ecosystem_status(step, self.creatures, self.world)
            narrative_log().log_summary(step)
            narrative_log().flush()
        
        # === ACHIEVEMENT CHECKS ===
        self._check_herbie_achievements()
        self.achievements.update()
    
    def _trigger_genesis_leviathan(self):
        """Trigger the Genesis Leviathan at step 50.
        
        The Genesis Leviathan spawns and begins its journey across the world.
        It does NOT instantly kill all predators - it hunts them as it moves,
        so survival depends on predator positions and luck.
        """
        step = self.seasons.step_count
        
        narrative_log().log_section("THE GENESIS EVENT")
        narrative_log().log("[GENESIS] 🔥 In the beginning, the First Leviathan crossed the void...", step, force=True)
        
        print("\n" + "="*70)
        print("[GENESIS] 🔥 In the beginning, the First Leviathan crossed the void...")
        print("="*70)
        
        # Count predators that COULD be hunted (for narrative)
        predators_present = sum(1 for c in self.creatures if c.alive and c.species.diet == 'carnivore')
        if predators_present > 0:
            print(f"[GENESIS] {predators_present} predator(s) sense the coming fire...")
            narrative_log().log(f"[GENESIS] {predators_present} predator(s) sense the coming fire...", step, force=True)
        else:
            print("[GENESIS] The land is already free of predators...")
            narrative_log().log("[GENESIS] The land is already free of predators...", step, force=True)
        
        # Spawn the Genesis Leviathan - it will hunt predators as it moves
        # The leviathan manager handles the actual hunting probabilistically
        self.leviathan_mgr.spawn_genesis_leviathan(self.world.world_size)
        
        # Create sacred ground (fertile patches)
        n_patches = np.random.randint(3, 6)
        for _ in range(n_patches):
            patch_pos = np.array([
                np.random.uniform(-self.world.world_size/3, self.world.world_size/3),
                np.random.uniform(-self.world.world_size/3, self.world.world_size/3)
            ])
            self.leviathan_mgr.add_sacred_ground(patch_pos, radius=15.0)
        
        msg = f"[GENESIS] The First Leviathan emerges from the edge... {n_patches} sacred groves bloom."
        print(f"[GENESIS] The First Leviathan emerges from the edge of the world...")
        print("="*70 + "\n")
        narrative_log().log(msg, step, force=True)
        
        self.achievements.unlock('leviathan_appears', self.seasons.step_count)
    
    def _process_herbie_births(self):
        """Process pending Herbie births from pregnancy."""
        for creature in self.creatures[:]:
            if creature.species.name != "Herbie" or not creature.alive:
                continue
            
            if hasattr(creature, '_pending_birth') and creature._pending_birth:
                birth_data = creature._pending_birth
                creature._pending_birth = None
                
                parent_pmem = None
                if hasattr(creature, 'placement_memory'):
                    parent_pmem = creature.placement_memory.params
                
                baby_sex = birth_data.get('sex', 
                    HerbieSex.PROVIDER if np.random.random() < 0.5 else HerbieSex.CARRIER)
                
                baby = self.spawn_herbie(
                    pos=birth_data['pos'],
                    sex=baby_sex,
                    genome=birth_data.get('genome'),
                    parent_ids=(birth_data.get('mother_id', ''), birth_data.get('father_id', '')),
                    is_juvenile=True,
                    parent_placement_params=parent_pmem
                )
                
                # Apply developmental trait modifiers from embryogenesis
                dev_traits = birth_data.get('developmental_traits', {})
                if dev_traits and hasattr(baby, 'traits') and baby.traits:
                    for trait_name, modifier in dev_traits.items():
                        current = getattr(baby.traits, trait_name, None)
                        if current is not None:
                            # Apply modifier
                            setattr(baby.traits, trait_name, current * modifier)
                
                # Log developmental summary
                dev_summary = birth_data.get('dev_summary')
                if dev_summary:
                    event_log().log('development', self.seasons.step_count,
                        creature_id=baby.creature_id,
                        n_basins=dev_summary.get('n_basins', 0),
                        bilateral_symmetry=dev_summary.get('bilateral_symmetry', 1.0),
                        n_perturbations=dev_summary.get('n_perturbations', 0),
                        trait_modifiers=dev_summary.get('trait_modifiers', {})
                    )
                
                total_herbies = sum(1 for c in self.creatures if c.species.name == "Herbie" and c.alive)
                sex_icon = '♂' if baby_sex == HerbieSex.PROVIDER else '♀'
                birth_msg = f"[NEW BABY] {sex_icon} '{baby.mating_state.name}' born | TOTAL HERBIES: {total_herbies}"
                print(birth_msg)
                narrative_log().log(birth_msg, self.seasons.step_count, force=True)
                
                baby_sex_str = 'male' if baby_sex == HerbieSex.PROVIDER else 'female'
                mother_name = birth_data.get('mother_name', birth_data.get('mother_id', ''))
                father_name = birth_data.get('father_name', birth_data.get('father_id', ''))
                
                # Collect trait data for genetics analysis
                trait_data = None
                if hasattr(baby, 'traits') and baby.traits:
                    trait_data = baby.traits.to_dict()
                
                event_log().log_birth(
                    step=self.seasons.step_count,
                    name=baby.mating_state.name,
                    sex=baby_sex_str,
                    generation=baby.generation,
                    parents=(mother_name, father_name),
                    pos=(baby.pos[0], baby.pos[1]),
                    traits=trait_data,
                    creature_id=baby.creature_id,
                    species="Herbie",
                    parent_id=birth_data.get('mother_id')
                )
                
                # Record in human-readable history
                world_history().record_birth(
                    step=self.seasons.step_count,
                    name=baby.mating_state.name,
                    sex=baby_sex_str,
                    generation=baby.generation,
                    parents=(mother_name, father_name),
                    species="Herbie"
                )
                
                for c in self.creatures:
                    if c.creature_id in [birth_data.get('mother_id'), birth_data.get('father_id')]:
                        if hasattr(c, 'mating_state'):
                            c.mating_state.offspring_ids.append(baby.creature_id)
    
    def _check_herbie_extinction(self):
        """Check and handle Herbie extinction."""
        herbies = [c for c in self.creatures if c.species.name == "Herbie" and c.alive]
        
        if len(herbies) == 0:
            print("[Herbie] EXTINCTION! Respawning mated pair...")
            provider = self.spawn_herbie(sex=HerbieSex.PROVIDER)
            self.protagonist_herbie_id = provider.creature_id
            carrier_pos = provider.pos + np.array([8.0, 0.0])
            carrier = self.spawn_herbie(pos=carrier_pos, sex=HerbieSex.CARRIER)
            provider.mating_state.mate_ids.append(carrier.creature_id)
            carrier.mating_state.mate_id = provider.creature_id
            provider.mating_state.is_bonded = True
            carrier.mating_state.is_bonded = True
        
        elif len(herbies) == 1:
            existing = herbies[0]
            if hasattr(existing, 'mating_state') and not existing.mating_state.is_bonded:
                new_sex = HerbieSex.CARRIER if existing.mating_state.sex == HerbieSex.PROVIDER else HerbieSex.PROVIDER
                new_pos = existing.pos + np.random.randn(2) * 15
                new_pos = np.clip(new_pos, -self.world.world_size/2 + 5, self.world.world_size/2 - 5)
                self.spawn_herbie(pos=new_pos, sex=new_sex)
                print(f"[Herbie] Spawned new mate for lonely Herbie")
    
    def _handle_death(self, creature: Creature):
        """Handle creature death."""
        pass  # Death handling is done in the main step_all loop
    
    def _handle_reproduction(self, creature: Creature):
        """Handle non-Herbie creature reproduction with optional embryonic development."""
        offset = np.random.randn(2) * 5
        child_pos = creature.pos + offset
        child_pos = np.clip(child_pos, -self.world.world_size/2 + 5, self.world.world_size/2 - 5)
        
        # Apply mutation rate from launcher params
        mut_rate = getattr(self, 'mutation_rate', 1.0)
        child_traits = MutatedTraits.mutate_from_parent(creature.traits, creature.species, mut_rate)
        
        # Check embryo mode setting
        embryo_mode = getattr(self, 'embryo_mode', 'herbie')
        dev_traits = {}
        dev_summary = None
        
        if embryo_mode == 'all':
            # Full embryonic development for all creatures
            from ..creature.embryo import EmbryoField
            
            embryo = EmbryoField(total_steps=50)  # Faster development for asexual
            
            # Set parental genetics (asexual = same parent twice effectively)
            g_parent = creature.body.g if hasattr(creature, 'body') else 0.5
            parent_traits_dict = child_traits.to_dict() if child_traits else {}
            embryo.set_parental_genetics(g_parent, g_parent, parent_traits_dict)
            
            # Run rapid development (asexual = faster, less complex)
            mother_hunger = creature.metabolism.hunger
            mother_stress = 0.0
            if hasattr(creature, 'afferent') and 'threat' in creature.afferent:
                threat_channel = creature.afferent['threat']
                if hasattr(threat_channel, 'u'):
                    mother_stress = min(1.0, np.sum(np.abs(threat_channel.u)**2))
            
            # Develop entire embryo now (instant for asexual species)
            while not embryo.is_ready():
                embryo.set_maternal_environment(mother_hunger, mother_stress)
                embryo.step()
            
            # Get developmental modifiers
            dev_traits = embryo.get_final_traits()
            dev_summary = embryo.get_development_summary()
            
            # Apply developmental modifiers to child traits
            if dev_traits and child_traits:
                for trait_name, modifier in dev_traits.items():
                    current = getattr(child_traits, trait_name, None)
                    if current is not None:
                        setattr(child_traits, trait_name, current * modifier)
        
        child = self.spawn_creature(
            creature.species,
            pos=child_pos,
            generation=creature.generation + 1,
            parent_id=creature.creature_id,
            lineage_depth=creature.lineage_depth + 1,
            inherited_knowledge=creature.learned_potential,
            inherited_traits=child_traits
        )
        
        # Log birth with traits for genetics analysis
        species = creature.species.name
        gen = creature.generation + 1
        
        trait_data = child_traits.to_dict() if child_traits else None
        event_log().log_birth(
            step=self.seasons.step_count,
            name=child.creature_id[:8],
            sex='unknown',
            generation=gen,
            parents=(creature.creature_id[:8], None),
            pos=(child_pos[0], child_pos[1]),
            traits=trait_data,
            creature_id=child.creature_id,
            species=species,
            parent_id=creature.creature_id
        )
        
        # Log developmental data
        if dev_summary:
            event_log().log('development', self.seasons.step_count,
                creature_id=child.creature_id,
                species=species,
                n_basins=dev_summary.get('n_basins', 0),
                bilateral_symmetry=dev_summary.get('bilateral_symmetry', 1.0),
                n_perturbations=dev_summary.get('n_perturbations', 0),
                trait_modifiers=dev_summary.get('trait_modifiers', {})
            )
        
        # Include dev info in log message
        sym_str = f" sym:{dev_summary['bilateral_symmetry']:.2f}" if dev_summary else ""
        console_log().log(f"[{species}] 🥚 New Gen{gen} {species} born at ({child_pos[0]:.0f}, {child_pos[1]:.0f}){sym_str}")
    
    def _rebalance_population(self):
        """Rebalance species populations."""
        species_counts = {}
        for c in self.creatures:
            if c.alive:
                species_counts[c.species.name] = species_counts.get(c.species.name, 0) + 1
        
        for species in ALL_SPECIES:
            if species.name == "Herbie":
                continue
            count = species_counts.get(species.name, 0)
            
            if count == 0:
                # Extinct - respawn with boosted chance
                weight = SPECIES_SPAWN_WEIGHTS.get(species.name, 1.0)
                # Apex gets guaranteed respawn when extinct
                if species.name == "Apex":
                    respawn_chance = 0.3  # 30% per check when extinct
                else:
                    respawn_chance = weight * 0.1
                    
                if np.random.random() < respawn_chance:
                    pos = self._find_good_spawn_position('plains')
                    self.spawn_creature(species, pos)
                    print(f"[{species.name}] Respawned after extinction")
    
    def _log_behavioral_snapshot(self, step: int):
        """Log interesting behavioral events happening in the world."""
        herbies = [c for c in self.creatures if c.species.name == "Herbie" and c.alive]
        predators = [c for c in self.creatures if c.species.diet != 'herbivore' and c.alive]
        
        # Find interesting behaviors
        events = []
        
        # Check for fleeing herbies
        for h in herbies:
            if hasattr(h, 'mating_state') and h.mating_state:
                name = h.mating_state.name
                
                # Check if being hunted (KdVChannel uses .u not .psi)
                if hasattr(h, 'afferent') and 'threat' in h.afferent:
                    threat_channel = h.afferent['threat']
                    threat_level = np.sum(np.abs(threat_channel.u)**2) if hasattr(threat_channel, 'u') else 0
                    if threat_level > 0.5:
                        events.append(f"[Herbie] 🏃 {name} is fleeing from danger!")
                
                # Check if very hungry
                if h.metabolism.hunger > 0.8:
                    events.append(f"[Herbie] 😰 {name} is starving (hunger={h.metabolism.hunger:.0%})")
                
                # Check if dreaming
                if getattr(h, 'is_dreaming', False) and getattr(h, 'dream_depth', 0) > 0.5:
                    events.append(f"[Herbie] 💤 {name} is deep in dreams...")
        
        # Check predator hunting status
        for p in predators:
            if hasattr(p, 'current_prey_target') and p.current_prey_target:
                # Find prey name
                prey = next((c for c in self.creatures if c.creature_id == p.current_prey_target), None)
                if prey and prey.alive:
                    prey_name = prey.mating_state.name if hasattr(prey, 'mating_state') and prey.mating_state else prey.species.name
                    events.append(f"[{p.species.name}] 🎯 Stalking {prey_name}...")
        
        # Check for herbie social gatherings
        for h in herbies:
            if hasattr(h, 'mating_state') and h.mating_state:
                nearby_herbies = sum(1 for other in herbies 
                                    if other != h and np.linalg.norm(other.pos - h.pos) < 8)
                if nearby_herbies >= 3:
                    events.append(f"[Herbie] 👥 {h.mating_state.name} is in a gathering of {nearby_herbies+1} Herbies")
                    break  # Only report one gathering per snapshot
        
        # Print up to 3 interesting events per snapshot
        for event in events[:3]:
            console_log().log(event, step)
    
    def _check_herbie_achievements(self):
        """Check for Herbie-specific achievements."""
        step = self.seasons.step_count
        herbies = [c for c in self.creatures if c.species.name == "Herbie" and c.alive and hasattr(c, 'mating_state')]
        
        for h in herbies:
            if h.mating_state.is_bonded:
                self.achievements.unlock('first_bond', step)
                break
        
        for h in herbies:
            if h.mating_state.is_juvenile:
                self.achievements.unlock('first_birth', step)
                break
        
        for h in herbies:
            family_size = 1 + len(h.mating_state.offspring_ids)
            if h.mating_state.mate_id:
                family_size += 1
            if family_size >= 5:
                self.achievements.unlock('family_of_five', step)
                break
        
        generations = set(h.generation for h in herbies)
        if len(generations) >= 3:
            self.achievements.unlock('three_generations', step)
        
        for h in herbies:
            if h.step_count >= 5000:
                self.achievements.unlock('centenarian', step)
                break
        
        if self.seasons.current_season.name == 'Winter' and len(herbies) > 0:
            self.achievements.unlock('survived_winter', step)
        
        if self.seasons.year >= 1:
            self.achievements.unlock('full_year', step)
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_protagonist_herbie(self) -> Optional[HerbieWithHands]:
        """Get the protagonist Herbie."""
        for c in self.creatures:
            if c.creature_id == self.protagonist_herbie_id and c.alive:
                return c
        
        for c in self.creatures:
            if c.species.name == "Herbie" and c.alive and hasattr(c, 'mating_state'):
                if c.mating_state.sex == HerbieSex.PROVIDER:
                    self.protagonist_herbie_id = c.creature_id
                    return c
        
        return None
    
    def get_statistics(self) -> dict:
        """Get current simulation statistics."""
        species_counts = {}
        for c in self.creatures:
            if c.alive:
                species_counts[c.species.name] = species_counts.get(c.species.name, 0) + 1
        
        return {
            'step': self.seasons.step_count,
            'season': self.seasons.current_season.name,
            'year': self.seasons.year,
            'day_phase': self.daynight.get_state().time_of_day,
            'species_counts': species_counts,
            'total_creatures': sum(species_counts.values()),
            'food_count': self.world.count_food(),
            'leviathan_active': self.leviathan_mgr.is_active(),
            'meteor_active': self.meteor_mgr.active,
            'migration_active': self.migration_mgr.active,
            'achievements': self.achievements.get_unlocked_count(),
            'aquatic': self.aquatic.get_stats(),
            'mycelia_biomass': self.mycelia.get_total_biomass(),
            'mycelia_coverage': self.mycelia.get_coverage(),
            'mushroom_count': len(self.mycelia.fruiting_bodies),
        }
    
    # =========================================================================
    # CHEMISTRY INTEGRATION
    # =========================================================================
    
    def init_chemistry(self):
        """Initialize chemistry system if not already present."""
        if hasattr(self, 'element_spawner') and self.element_spawner is not None:
            return
        
        from ..chemistry import (
            ElementSpawner, load_chemistry_state,
            ElementType, ElementObject, ELEMENT_TERRAIN_SOURCES
        )
        
        # Try to load existing chemistry state
        self.element_spawner = load_chemistry_state(self.world, self.terrain)
        
        if self.element_spawner is None:
            self.element_spawner = ElementSpawner(self.world, self.terrain)
            print("[Chemistry] Initialized new element system")
            self._seed_initial_elements()
        
        print(f"[Chemistry] {len(self.element_spawner.element_objects)} elements, "
              f"{len(self.element_spawner.constructions)} constructions")
    
    def _seed_initial_elements(self):
        """Seed world with initial elements based on terrain."""
        from ..chemistry import ElementType, ElementObject, ELEMENT_TERRAIN_SOURCES
        from ..core.constants import WORLD_L
        
        world_size = getattr(self.world, 'world_size', WORLD_L)
        half = world_size / 2
        
        for _ in range(200):
            x = np.random.uniform(-half + 5, half - 5)
            y = np.random.uniform(-half + 5, half - 5)
            pos = np.array([x, y])
            
            terrain_name = 'plains'
            if self.terrain:
                terrain_obj = self.terrain.get_terrain_at(pos)
                terrain_name = terrain_obj.name if hasattr(terrain_obj, 'name') else str(terrain_obj)
            
            for elem_type, sources in ELEMENT_TERRAIN_SOURCES.items():
                if terrain_name in sources:
                    if np.random.random() < 0.7:
                        amount = np.random.uniform(0.8, 2.0)
                        elem = ElementObject(pos, elem_type, amount)
                        self.element_spawner.element_objects.append(elem)
                        break
        
        # Guaranteed LITE_ORE spawns for Herbie defense
        for _ in range(15):
            x = np.random.uniform(-half + 10, half - 10)
            y = np.random.uniform(-half + 10, half - 10)
            pos = np.array([x, y])
            elem = ElementObject(pos, ElementType.LITE_ORE, np.random.uniform(1.0, 2.0))
            self.element_spawner.element_objects.append(elem)
        
        print(f"[Chemistry] Seeded {len(self.element_spawner.element_objects)} elements")
    
    def _spawn_pigment_objects(self):
        """Spawn smearable pigment objects around the world for art creation."""
        world_size = getattr(self.world, 'world_size', 100)
        half = world_size / 2
        
        # Pigment types and their terrain associations
        pigment_terrains = {
            'charcoal': ['forest'],      # From burnt wood
            'berry': ['forest', 'plains'],  # Berry bushes
            'mud': ['shore', 'plains'],    # Near water
            'ochre': ['hills', 'cave'],    # Mineral deposits
            'mineral_blue': ['cave'],      # Rare cave pigment
            'mineral_green': ['hills'],    # Copper deposits
        }
        
        n_pigments = 0
        for pigment_type, terrains in pigment_terrains.items():
            # Spawn 5-10 of each type
            n_spawn = np.random.randint(5, 11)
            for _ in range(n_spawn):
                # Try to find appropriate terrain
                for attempt in range(20):
                    x = np.random.uniform(-half + 5, half - 5)
                    y = np.random.uniform(-half + 5, half - 5)
                    pos = np.array([x, y])
                    
                    if self.terrain:
                        terrain_obj = self.terrain.get_terrain_at(pos)
                        terrain_name = terrain_obj.name if hasattr(terrain_obj, 'name') else 'plains'
                        if terrain_name in terrains:
                            pigment = spawn_pigment_object(pos, pigment_type)
                            self.world.objects.append(pigment)
                            n_pigments += 1
                            break
                    else:
                        # No terrain, spawn anywhere
                        pigment = spawn_pigment_object(pos, pigment_type)
                        self.world.objects.append(pigment)
                        n_pigments += 1
                        break
        
        print(f"[Art] Spawned {n_pigments} pigment objects for smearing")
    
    def step_chemistry(self, is_raining: bool = False, wind_strength: float = 0.0):
        """Step the chemistry system."""
        if not hasattr(self, 'element_spawner') or self.element_spawner is None:
            return
        
        from ..chemistry import (
            integrate_chemistry_to_world_objects,
            check_element_interaction, on_element_dropped
        )
        from ..audio.soundscape import get_world_soundscape
        
        # Step element spawner
        self.element_spawner.step(self.seasons.step_count, is_raining)
        
        # Make elements grippable
        original_objects = self.world.objects.copy()
        self.world.objects = integrate_chemistry_to_world_objects(
            original_objects, self.element_spawner
        )
        
        # Process creature-element interactions
        for creature in self.creatures:
            if not creature.alive:
                continue
            
            check_element_interaction(creature, self.element_spawner)
            
            # Handle element drops
            if hasattr(creature, 'hands'):
                for hand in [creature.hands.left, creature.hands.right]:
                    dropped = getattr(hand, '_just_dropped', None)
                    if dropped and hasattr(dropped, 'element_type'):
                        name = getattr(creature, 'mating_state', None)
                        creature_name = name.name if name else creature.creature_id[:8]
                        on_element_dropped(
                            dropped, creature_name,
                            self.seasons.step_count, self.element_spawner
                        )
                        hand._just_dropped = None
        
        # Apply shelter effects
        for creature in self.creatures:
            if not creature.alive:
                continue
            
            shelter = self.element_spawner.get_shelter_at(creature.pos)
            if shelter > 0.1:
                if hasattr(creature, 'metabolism'):
                    creature.metabolism.hunger_rate_modifier = 1.0 - shelter * 0.15
                creature._shelter_bonus = shelter
        
        # Restore original objects
        self.world.objects = original_objects
        
        # Step world soundscape
        soundscape = get_world_soundscape()
        if soundscape is not None:
            soundscape.step(wind_strength)
    
    def get_chemistry_status(self) -> str:
        """Get chemistry status string for display."""
        if not hasattr(self, 'element_spawner') or self.element_spawner is None:
            return ""
        return self.element_spawner.get_status()
    
    def save_full_state(self):
        """Save complete world state for persistence."""
        if hasattr(self, 'persistence') and self.persistence is not None:
            try:
                self.persistence.save(
                    creatures=self.creatures,
                    world=self.world,
                    terrain=self.terrain,
                    step=self.step_count,
                    evolution_tree=self.evolution_tree,
                    daynight=self.daynight,
                    season_cycle=self.season_cycle
                )
                print(f"[Persistence] Saved state at step {self.step_count}")
            except Exception as e:
                print(f"[Persistence] Save failed: {e}")
