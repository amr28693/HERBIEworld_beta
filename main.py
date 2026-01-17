#!/usr/bin/env python3
"""
HERBIE World - Artificial Life Simulation
=========================================

An emergent artificial life simulation featuring:
- NLSE-based wave equation consciousness
- Torus brain architecture
- Sexual reproduction with genetics
- Emergent behaviors (nesting, art, burial)
- Chemistry system with primordial elements
- Multi-species ecosystem
- Day/night cycle, seasons, weather
- Disease outbreaks
- Leviathan apex predator

Usage:
    python -m herbie_world                 # Normal mode with visualization
    python -m herbie_world --overnight     # Headless mode for long runs
    python -m herbie_world --help          # Show help

Controls (in visualization):
    Arrow keys: Select creature
    Space: Pause/resume
    L: Toggle local view (follow selected creature)
    H: Cycle through Herbies only
    A: Toggle ants display
    P: Toggle pheromone display
    M: Toggle mycelia display
    N: Toggle nest display
    S: Toggle smear display
    C: Toggle chemistry temperature field
    X: Toggle chemistry concentration field
    Z: Toggle chemistry display
"""

import sys
import time
import os
import numpy as np

# === READ LAUNCHER PARAMETERS ===
LAUNCH_PARAMS = {
    'food_mult': float(os.environ.get('HERBIE_FOOD_MULT', '1.0')),
    'season_harsh': float(os.environ.get('HERBIE_SEASON_HARSH', '1.0')),
    'start_pop': int(os.environ.get('HERBIE_START_POP', '8')),
    'start_apex': int(os.environ.get('HERBIE_START_APEX', '1')),
    'start_fish': int(os.environ.get('HERBIE_START_FISH', '5')),
    'world_size': int(os.environ.get('HERBIE_WORLD_SIZE', '100')),
    'day_length': int(os.environ.get('HERBIE_DAY_LENGTH', '500')),
    'mutation_rate': float(os.environ.get('HERBIE_MUTATION_RATE', '1.0')),
    'predators': os.environ.get('HERBIE_PREDATORS', '1') == '1',
    'embryo_mode': os.environ.get('HERBIE_EMBRYO_MODE', 'herbie'),  # 'herbie', 'all', 'none'
}

# Package imports
from .core.constants import WORLD_L, TERRAIN_RESOLUTION
from .world.terrain import Terrain
from .world.multi_world import MultiWorld
from .manager.creature_manager import CreatureManager
from .audio.audio_system import AudioSystem, DummyAudio, HAS_AUDIO
from .audio.soundscape import init_world_soundscape
from .visualization.main_vis import HerbieVisualization
from .persistence.world_state import WorldPersistence
from .events.logger import event_log

# Matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


# Persistence paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
HERBIE_STATE_FILE = os.path.join(DATA_DIR, 'herbie_state.npz')
WORLD_STATE_FILE = os.path.join(DATA_DIR, 'world_state.npz')


def ensure_dirs():
    """Ensure data directories exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def print_banner(mode: str = 'normal'):
    """Print startup banner."""
    print("=" * 72)
    print("HERBIE WORLD - Emergent Artificial Life Simulation")
    print("=" * 72)
    
    if mode == 'overnight':
        print("OVERNIGHT MODE (Headless)")
        print("Running without visualization for maximum speed.")
        print("Press Ctrl+C to stop gracefully.")
    else:
        print("Features:")
        print("  🧠 NLSE wave equation consciousness (torus brain)")
        print("  🧬 Sexual reproduction with Mendelian genetics")
        print("  🏠 Emergent behaviors (nesting, art, burial)")
        print("  🧪 Chemistry system with 6 primordial elements")
        print("  🌍 Multi-species ecosystem with predator-prey dynamics")
        print("  🌙 Day/night cycle, seasons, weather events")
        print("  🦠 Disease outbreaks with transmission")
        print("  🐉 Leviathan apex predator (Genesis event at step 50)")
    
    print("=" * 72)


def main_visual():
    """Main loop with visualization."""
    if not HAS_MATPLOTLIB:
        print("[Error] Matplotlib not available. Use --overnight mode.")
        return
    
    print_banner('visual')
    
    print(f"[Persistence] Data directory: {DATA_DIR}")
    print(f"[Persistence] World state file: {WORLD_STATE_FILE}")
    
    ensure_dirs()
    
    manager = None
    audio = AudioSystem() if HAS_AUDIO else DummyAudio()
    soundscape = init_world_soundscape(audio)
    
    print("[Init] World soundscape initialized")
    
    try:
        audio.start()
        
        # Try to load existing world
        saved_data = WorldPersistence.load_world(WORLD_STATE_FILE)
        
        if saved_data is not None:
            print("[Init] Restoring saved world...")
            terrain = WorldPersistence.restore_terrain(saved_data.get('terrain'))
            world = MultiWorld(terrain)
            manager = CreatureManager(world, terrain=terrain)
            
            if manager.load_full_state(saved_data):
                print("[Init] World restored successfully!")
            else:
                print("[Init] Restore failed, starting fresh")
                manager.spawn_initial_population()
        else:
            print("[Init] Starting new world...")
            # Use launcher parameters
            world_size = LAUNCH_PARAMS['world_size']
            terrain = Terrain(world_size, TERRAIN_RESOLUTION)
            world = MultiWorld(terrain)
            manager = CreatureManager(world, terrain=terrain)
            
            # Apply launch parameters
            manager.launch_params = LAUNCH_PARAMS
            manager.spawn_initial_population()
        
        # Create visualization
        viz = HerbieVisualization(manager, terrain=terrain)
        
        print(f"\n[Init] Starting simulation...")
        print(f"[Init] Creatures: {len([c for c in manager.creatures if c.alive])}")
        print(f"[Init] Food: {manager.world.count_food()}")
        print()
        
        # Main loop
        while True:
            audio_amp = audio.amplitude
            silence = audio.silence_frames
            manager.step_all(audio_amp, silence, audio)
            
            # Render every 3 steps
            if manager.seasons.step_count % 3 == 0:
                viz.update()
                plt.pause(0.02)
            
            # Status every 100 steps
            if manager.seasons.step_count % 100 == 0:
                step = manager.seasons.step_count
                season = manager.seasons.current_season
                dn_state = manager.daynight.get_state()
                day = step // 500 + 1  # Day number (500 steps per day)
                
                counts = {}
                for c in manager.creatures:
                    if c.alive:
                        name = c.species.name[:3]
                        counts[name] = counts.get(name, 0) + 1
                
                count_str = " | ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
                phase = dn_state.phase_name.upper()[:3] if dn_state else "???"
                
                print(f"[Day {day:3d} Step {step:5d}] {phase} {season.name[:3]} | {count_str}")
            
            # Auto-save every 500 steps
            if manager.seasons.step_count % 500 == 0:
                herbies = [c for c in manager.creatures 
                          if c.species.name == "Herbie" and c.alive]
                if herbies:
                    best = max(herbies, key=lambda h: h.generation)
                    try:
                        np.savez(HERBIE_STATE_FILE, 
                                generation=best.generation,
                                pos=best.pos,
                                step=manager.seasons.step_count)
                    except Exception as e:
                        print(f"[Persistence] Save failed: {e}")
    
    except KeyboardInterrupt:
        print("\n[Shutdown] Saving state...")
    finally:
        if manager is not None:
            manager.save_full_state()
            # Save human-readable history
            from .events.world_history import world_history
            from .events.narrative_log import narrative_log
            final_counts = {}
            for c in manager.creatures:
                if c.alive:
                    final_counts[c.species.name] = final_counts.get(c.species.name, 0) + 1
            world_history().save_report(manager.seasons.step_count, final_counts)
            narrative_log().flush()
        event_log().flush()
        audio.stop()
        if plt:
            plt.close('all')


def main_overnight():
    """Headless overnight mode - no visualization."""
    print_banner('overnight')
    
    print(f"[Persistence] Data directory: {DATA_DIR}")
    print(f"[Persistence] World state file: {WORLD_STATE_FILE}")
    
    ensure_dirs()
    
    manager = None
    audio = AudioSystem() if HAS_AUDIO else None
    soundscape = init_world_soundscape(audio)
    
    try:
        if audio:
            audio.start()
            print("[Overnight] Audio system started")
        
        # Try to load existing world
        saved_data = WorldPersistence.load_world(WORLD_STATE_FILE)
        
        if saved_data is not None:
            print("[Overnight] Restoring saved world...")
            terrain = WorldPersistence.restore_terrain(saved_data.get('terrain'))
            world = MultiWorld(terrain)
            manager = CreatureManager(world, terrain=terrain)
            
            if manager.load_full_state(saved_data):
                print("[Overnight] World restored successfully!")
            else:
                print("[Overnight] Restore failed, starting fresh")
                saved_data = None
        
        if saved_data is None:
            print("[Overnight] Creating new world...")
            terrain = Terrain(WORLD_L, TERRAIN_RESOLUTION)
            world = MultiWorld(terrain)
            manager = CreatureManager(world, terrain=terrain)
            manager.spawn_initial_population()
        
        print(f"\n[Overnight] Starting simulation...")
        print(f"[Overnight] Creatures: {len([c for c in manager.creatures if c.alive])}")
        print()
        
        event_log().log('session_start', 0, mode='overnight')
        last_save_time = time.time()
        
        # Main loop - no visualization
        while True:
            audio_amp = audio.amplitude if audio else 0.0
            silence = audio.silence_frames if audio else 0
            manager.step_all(audio_amp, silence, audio)
            
            step = manager.seasons.step_count
            
            # Status every 500 steps
            if step % 500 == 0:
                season = manager.seasons.current_season
                
                counts = {}
                for c in manager.creatures:
                    if c.alive:
                        name = c.species.name[:3]
                        counts[name] = counts.get(name, 0) + 1
                
                count_str = " | ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
                food_count = manager.world.count_food()
                elapsed = time.time() - last_save_time
                
                print(f"[{step:6d}] {season.name[:3]} | {count_str} | Food:{food_count} | {elapsed:.1f}s")
            
            # Population snapshot every 1000 steps
            if step % 1000 == 0:
                full_counts = {}
                for c in manager.creatures:
                    if c.alive:
                        full_counts[c.species.name] = full_counts.get(c.species.name, 0) + 1
                event_log().log_population(step, full_counts)
                event_log().flush()
            
            # Save state every 2000 steps
            if step % 2000 == 0:
                herbies = [c for c in manager.creatures 
                          if c.species.name == "Herbie" and c.alive]
                if herbies:
                    best = max(herbies, key=lambda h: h.generation)
                    try:
                        np.savez(HERBIE_STATE_FILE,
                                generation=best.generation,
                                pos=best.pos,
                                step=step)
                        print(f"[Persistence] Saved Gen{best.generation} @ step {step}")
                    except Exception as e:
                        print(f"[Persistence] Save failed: {e}")
                
                last_save_time = time.time()
            
            # Full world save every 10000 steps
            if step % 10000 == 0:
                try:
                    manager.save_full_state()
                    print(f"[Persistence] Full world state saved @ step {step}")
                except Exception as e:
                    print(f"[Persistence] World save failed: {e}")
    
    except KeyboardInterrupt:
        print("\n[Overnight] Shutting down gracefully...")
    finally:
        if manager is not None:
            print("[Overnight] Saving final state...")
            manager.save_full_state()
            
            step = manager.seasons.step_count if manager.seasons else 0
            event_log().log('session_end', step, mode='overnight')
            
            # Save human-readable history
            from .events.world_history import world_history
            final_counts = {}
            for c in manager.creatures:
                if c.alive:
                    final_counts[c.species.name] = final_counts.get(c.species.name, 0) + 1
            world_history().save_report(step, final_counts)
        
        if audio:
            audio.stop()
        
        event_log().flush()
        print("[Overnight] Goodbye!")


def main():
    """Entry point - choose mode based on arguments."""
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        return
    
    if '--overnight' in sys.argv:
        main_overnight()
    else:
        main_visual()


if __name__ == "__main__":
    main()
