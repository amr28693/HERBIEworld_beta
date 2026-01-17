"""
World Persistence - Unified save/load for complete world state.

This module handles saving and loading:
- Terrain
- All creatures (all species, generations, knowledge)
- World objects (food, barriers, elements)
- Nutrients
- Season/year/day-night
- Ecosystem statistics
- Herbie lineage and state
- Evolution tree
- Mycelia network
- Aquatic system

CRITICAL: This unified persistence layer addresses the bugs caused by
having separate HERBIE_MULTI_STATE and WORLD_STATE files that could
get out of sync.
"""

import os
import time
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np

from ..core.constants import WORLD_L, SCRIPT_DIR

if TYPE_CHECKING:
    from ..world.terrain import Terrain
    from ..world.objects import WorldObject, NutrientPatch
    from ..world.seasons import SeasonSystem
    from ..world.day_night import DayNightCycle
    from ..world.mycelia import MyceliumNetwork
    from ..world.aquatic import AquaticSystem
    from ..ecology.disease import DiseaseSystem
    from ..ecology.leviathan import LeviathanSystem
    from ..evolution.tree import EvolutionTree
    from ..chemistry.element_objects import ElementField

# Default save path
WORLD_STATE_FILE = os.path.join(SCRIPT_DIR, "herbie_world_state.npz")


class WorldPersistence:
    """
    Unified persistence for complete world state.
    
    Key improvement over original: ALL state goes into a single file,
    eliminating the desync bugs from separate Herbie/World saves.
    """
    
    @staticmethod
    def save_world(manager, filepath: str = None) -> bool:
        """
        Save complete world state to single file.
        
        Args:
            manager: CreatureManager with all world state
            filepath: Save path (defaults to WORLD_STATE_FILE)
            
        Returns:
            True if successful
        """
        if filepath is None:
            filepath = WORLD_STATE_FILE
        
        print(f"[Persistence] Saving unified world state...")
        
        try:
            # === TERRAIN ===
            terrain_data = manager.terrain.to_dict() if hasattr(manager, 'terrain') else {}
            
            # === SEASON ===
            season_data = manager.seasons.to_dict() if hasattr(manager, 'seasons') else {}
            
            # === DAY/NIGHT ===
            daynight_data = manager.day_night.to_dict() if hasattr(manager, 'day_night') else {}
            
            # === WORLD OBJECTS ===
            objects_data = []
            if hasattr(manager, 'world') and hasattr(manager.world, 'objects'):
                for obj in manager.world.objects:
                    if hasattr(obj, 'to_dict'):
                        objects_data.append(obj.to_dict())
                    else:
                        objects_data.append({
                            'pos': obj.pos.tolist() if hasattr(obj.pos, 'tolist') else list(obj.pos),
                            'size': obj.size,
                            'compliance': obj.compliance,
                            'energy': obj.energy,
                            'alive': obj.alive,
                        })
            
            # === NUTRIENTS ===
            nutrients_data = []
            if hasattr(manager, 'world') and hasattr(manager.world, 'nutrients'):
                for nut in manager.world.nutrients:
                    nutrients_data.append({
                        'pos': nut.pos.tolist() if hasattr(nut.pos, 'tolist') else list(nut.pos),
                        'nutrients': nut.nutrients,
                        'age': nut.age,
                    })
            
            # === CREATURES ===
            creatures_data = []
            if hasattr(manager, 'creatures'):
                for creature in manager.creatures:
                    if not creature.alive:
                        continue
                    if hasattr(creature, 'to_dict'):
                        creatures_data.append(creature.to_dict())
                    else:
                        # Fallback for minimal creature data
                        creatures_data.append({
                            'species_name': creature.species.name,
                            'pos': creature.pos.tolist() if hasattr(creature.pos, 'tolist') else list(creature.pos),
                            'creature_id': creature.creature_id,
                            'generation': creature.generation,
                        })
            
            # === MYCELIA ===
            mycelia_data = {}
            if hasattr(manager, 'mycelia') and manager.mycelia:
                mycelia_data = manager.mycelia.to_dict()
            
            # === AQUATIC ===
            aquatic_data = {}
            if hasattr(manager, 'aquatic') and manager.aquatic:
                aquatic_data = manager.aquatic.to_dict()
            
            # === DISEASE ===
            disease_data = {}
            if hasattr(manager, 'disease') and manager.disease:
                disease_data = manager.disease.to_dict()
            
            # === LEVIATHAN ===
            leviathan_data = {}
            if hasattr(manager, 'leviathan') and manager.leviathan:
                leviathan_data = manager.leviathan.to_dict()
            
            # === EVOLUTION TREE ===
            evolution_data = {}
            if hasattr(manager, 'evolution_tree') and manager.evolution_tree:
                evolution_data = manager.evolution_tree.to_dict()
            
            # === ELEMENT FIELD ===
            element_field_data = {}
            if hasattr(manager, 'element_field') and manager.element_field:
                element_field_data = manager.element_field.to_dict()
            
            # === STEP COUNT ===
            step_count = getattr(manager, 'step_count', 0)
            
            # === SAVE ===
            np.savez_compressed(
                filepath,
                # Core
                terrain=terrain_data,
                season=season_data,
                daynight=daynight_data,
                step_count=step_count,
                
                # World contents
                objects=objects_data,
                nutrients=nutrients_data,
                creatures=creatures_data,
                
                # Subsystems
                mycelia=mycelia_data,
                aquatic=aquatic_data,
                disease=disease_data,
                leviathan=leviathan_data,
                evolution=evolution_data,
                element_field=element_field_data,
                
                # Metadata
                save_version="unified_v1",
                save_time=time.time(),
            )
            
            file_size = os.path.getsize(filepath)
            print(f"[Persistence] Saved to {filepath}")
            print(f"[Persistence]   - {len(creatures_data)} creatures")
            print(f"[Persistence]   - {len(objects_data)} objects")
            print(f"[Persistence]   - File size: {file_size:,} bytes")
            return True
            
        except Exception as e:
            print(f"[Persistence] Save failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def load_world(filepath: str = None) -> Optional[Dict[str, Any]]:
        """
        Load world state from file.
        
        Args:
            filepath: Load path (defaults to WORLD_STATE_FILE)
            
        Returns:
            Dict with all world data, or None if load fails
        """
        if filepath is None:
            filepath = WORLD_STATE_FILE
        
        if not os.path.exists(filepath):
            print(f"[Persistence] No save file found at {filepath}")
            return None
        
        print(f"[Persistence] Loading unified world state...")
        
        try:
            d = np.load(filepath, allow_pickle=True)
            
            version = str(d.get('save_version', 'unknown'))
            print(f"[Persistence] Save version: {version}")
            
            result = {
                'terrain': d['terrain'].item() if 'terrain' in d else {},
                'season': d['season'].item() if 'season' in d else {},
                'daynight': d['daynight'].item() if 'daynight' in d else {},
                'step_count': int(d['step_count']) if 'step_count' in d else 0,
                'objects': list(d['objects']) if 'objects' in d else [],
                'nutrients': list(d['nutrients']) if 'nutrients' in d else [],
                'creatures': list(d['creatures']) if 'creatures' in d else [],
                'mycelia': d['mycelia'].item() if 'mycelia' in d else {},
                'aquatic': d['aquatic'].item() if 'aquatic' in d else {},
                'disease': d['disease'].item() if 'disease' in d else {},
                'leviathan': d['leviathan'].item() if 'leviathan' in d else {},
                'evolution': d['evolution'].item() if 'evolution' in d else {},
                'element_field': d['element_field'].item() if 'element_field' in d else {},
            }
            
            d.close()
            
            print(f"[Persistence] Loaded {len(result['creatures'])} creatures, "
                  f"{len(result['objects'])} objects")
            
            return result
            
        except Exception as e:
            print(f"[Persistence] Load failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def exists(filepath: str = None) -> bool:
        """Check if save file exists."""
        if filepath is None:
            filepath = WORLD_STATE_FILE
        return os.path.exists(filepath)
