"""World systems - terrain, objects, weather, seasons, day/night, ecology."""

from .objects import WorldObject, NutrientPatch, MeatChunk, HerbieCorpse, spawn_meat_chunks
from .terrain import Terrain, TerrainType
from .weather import WeatherSystem
from .seasons import SeasonSystem, Season, SEASONS, get_age_speed_modifier
from .day_night import DayNightCycle
from .mycelia import MyceliumNetwork, FruitingBody
from .aquatic import AquaticPlant, AquaticCreature, AquaticSystem
from .multi_world import MultiWorld
