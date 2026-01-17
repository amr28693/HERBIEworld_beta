#!/usr/bin/env python3
"""
HERBIE World Launcher - Tron-style GUI for setting simulation parameters.

Simple tkinter interface - no extra dependencies needed.
"""

import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os


class TronStyle:
    """Tron-inspired color scheme."""
    BG_DARK = "#0a0a0f"
    BG_PANEL = "#12121a"
    CYAN = "#00d4ff"
    CYAN_DIM = "#006680"
    MAGENTA = "#ff00ff"
    MAGENTA_DIM = "#660066"
    WHITE = "#e0e0e0"
    GRID = "#1a1a2e"
    

class HerbieLauncher:
    """Main launcher window."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HERBIE WORLD")
        self.root.geometry("620x1000")  # Slightly taller for new option
        self.root.configure(bg=TronStyle.BG_DARK)
        self.root.resizable(False, False)
        
        # Parameters with defaults
        self.params = {
            'food_abundance': tk.DoubleVar(value=1.0),
            'seasonal_harshness': tk.DoubleVar(value=1.0),
            'starting_herbies': tk.IntVar(value=8),
            'starting_apex': tk.IntVar(value=1),
            'starting_fish': tk.IntVar(value=5),
            'world_size': tk.IntVar(value=100),
            'day_length': tk.IntVar(value=500),
            'mutation_rate': tk.DoubleVar(value=1.0),
            'predators_enabled': tk.BooleanVar(value=True),
            'embryo_mode': tk.StringVar(value='herbie'),  # 'herbie', 'all', 'none'
        }
        
        self._build_ui()
        
    def _build_ui(self):
        """Build the launcher interface."""
        # Title
        title_frame = tk.Frame(self.root, bg=TronStyle.BG_DARK)
        title_frame.pack(pady=20)
        
        title = tk.Label(
            title_frame,
            text="HERBIE WORLD",
            font=("Courier", 36, "bold"),
            fg=TronStyle.CYAN,
            bg=TronStyle.BG_DARK
        )
        title.pack()
        
        subtitle = tk.Label(
            title_frame,
            text="Artificial Life Simulation",
            font=("Courier", 12),
            fg=TronStyle.CYAN_DIM,
            bg=TronStyle.BG_DARK
        )
        subtitle.pack()
        
        # Divider line
        self._add_divider()
        
        # Parameters panel
        params_frame = tk.Frame(self.root, bg=TronStyle.BG_PANEL, padx=30, pady=20)
        params_frame.pack(fill='x', padx=30, pady=10)
        
        # Food Abundance
        self._add_slider(
            params_frame,
            "FOOD ABUNDANCE",
            self.params['food_abundance'],
            0.2, 3.0,
            "Scarce ←→ Bountiful",
            row=0
        )
        
        # Seasonal Harshness
        self._add_slider(
            params_frame,
            "SEASONAL HARSHNESS",
            self.params['seasonal_harshness'],
            0.0, 2.0,
            "Mild ←→ Brutal",
            row=1
        )
        
        # Starting Herbies
        self._add_slider(
            params_frame,
            "STARTING HERBIES",
            self.params['starting_herbies'],
            2, 20,
            "Few ←→ Many",
            row=2,
            is_int=True
        )
        
        # Starting Apex Predators
        self._add_slider(
            params_frame,
            "APEX PREDATORS",
            self.params['starting_apex'],
            0, 5,
            "None ←→ Pack",
            row=3,
            is_int=True
        )
        
        # Starting Fish
        self._add_slider(
            params_frame,
            "AQUATIC LIFE",
            self.params['starting_fish'],
            0, 15,
            "None ←→ Teeming",
            row=4,
            is_int=True
        )
        
        # World Size
        self._add_slider(
            params_frame,
            "WORLD SIZE",
            self.params['world_size'],
            60, 150,
            "Small ←→ Large",
            row=5,
            is_int=True
        )
        
        # Day Length
        self._add_slider(
            params_frame,
            "DAY LENGTH",
            self.params['day_length'],
            200, 1000,
            "Fast days ←→ Slow days",
            row=6,
            is_int=True
        )
        
        # Mutation Rate
        self._add_slider(
            params_frame,
            "MUTATION RATE",
            self.params['mutation_rate'],
            0.0, 3.0,
            "Stable ←→ Chaotic",
            row=7
        )
        
        # Other predators toggle
        pred_frame = tk.Frame(params_frame, bg=TronStyle.BG_PANEL)
        pred_frame.grid(row=16, column=0, columnspan=2, sticky='w', pady=10)
        
        pred_label = tk.Label(
            pred_frame,
            text="OTHER PREDATORS",
            font=("Courier", 11, "bold"),
            fg=TronStyle.CYAN,
            bg=TronStyle.BG_PANEL
        )
        pred_label.pack(side='left')
        
        pred_toggle = tk.Checkbutton(
            pred_frame,
            variable=self.params['predators_enabled'],
            bg=TronStyle.BG_PANEL,
            fg=TronStyle.CYAN,
            selectcolor=TronStyle.BG_DARK,
            activebackground=TronStyle.BG_PANEL,
            activeforeground=TronStyle.MAGENTA
        )
        pred_toggle.pack(side='left', padx=10)
        
        pred_status = tk.Label(
            pred_frame,
            text="(adds danger)",
            font=("Courier", 9),
            fg=TronStyle.CYAN_DIM,
            bg=TronStyle.BG_PANEL
        )
        pred_status.pack(side='left')
        
        # Embryo Development Mode
        embryo_frame = tk.Frame(params_frame, bg=TronStyle.BG_PANEL)
        embryo_frame.grid(row=17, column=0, columnspan=2, sticky='w', pady=10)
        
        embryo_label = tk.Label(
            embryo_frame,
            text="EMBRYO DEVELOPMENT",
            font=("Courier", 11, "bold"),
            fg=TronStyle.MAGENTA,
            bg=TronStyle.BG_PANEL
        )
        embryo_label.pack(side='left')
        
        # Radio buttons for embryo mode
        modes = [
            ('herbie', 'Herbies Only', '(recommended)'),
            ('all', 'All Creatures', '(slower, more data)'),
            ('none', 'Disabled', '(fastest)'),
        ]
        
        for mode_val, mode_text, mode_hint in modes:
            rb_frame = tk.Frame(embryo_frame, bg=TronStyle.BG_PANEL)
            rb_frame.pack(side='left', padx=5)
            
            rb = tk.Radiobutton(
                rb_frame,
                text=mode_text,
                variable=self.params['embryo_mode'],
                value=mode_val,
                bg=TronStyle.BG_PANEL,
                fg=TronStyle.CYAN,
                selectcolor=TronStyle.BG_DARK,
                activebackground=TronStyle.BG_PANEL,
                activeforeground=TronStyle.MAGENTA,
                font=("Courier", 9)
            )
            rb.pack(side='left')
        
        # Divider
        self._add_divider()
        
        # Preview panel
        preview_frame = tk.Frame(self.root, bg=TronStyle.BG_PANEL, padx=20, pady=15)
        preview_frame.pack(fill='x', padx=30, pady=10)
        
        preview_title = tk.Label(
            preview_frame,
            text="[ SIMULATION PREVIEW ]",
            font=("Courier", 10),
            fg=TronStyle.MAGENTA,
            bg=TronStyle.BG_PANEL
        )
        preview_title.pack()
        
        self.preview_text = tk.Label(
            preview_frame,
            text="",
            font=("Courier", 9),
            fg=TronStyle.WHITE,
            bg=TronStyle.BG_PANEL,
            justify='left'
        )
        self.preview_text.pack(pady=10)
        self._update_preview()
        
        # Bind updates for all variable types
        for var in self.params.values():
            if isinstance(var, (tk.DoubleVar, tk.IntVar, tk.BooleanVar)):
                var.trace_add('write', lambda *args: self._update_preview())
        
        # Divider
        self._add_divider()
        
        # Launch button
        launch_btn = tk.Button(
            self.root,
            text="▶  START SIMULATION  ▶",
            font=("Courier", 18, "bold"),
            fg=TronStyle.BG_DARK,
            bg=TronStyle.CYAN,
            activeforeground=TronStyle.BG_DARK,
            activebackground=TronStyle.MAGENTA,
            relief='flat',
            padx=30,
            pady=15,
            cursor='hand2',
            command=self._launch
        )
        launch_btn.pack(pady=20)
        
        # Footer
        footer = tk.Label(
            self.root,
            text="Controls: Arrow keys to select • Space to pause • H to cycle Herbies",
            font=("Courier", 8),
            fg=TronStyle.CYAN_DIM,
            bg=TronStyle.BG_DARK
        )
        footer.pack(side='bottom', pady=10)
        
    def _add_divider(self):
        """Add a glowing divider line."""
        divider = tk.Frame(self.root, height=2, bg=TronStyle.CYAN_DIM)
        divider.pack(fill='x', padx=50, pady=5)
        
    def _add_slider(self, parent, label, var, min_val, max_val, hint, row, is_int=False):
        """Add a styled slider with label."""
        # Label
        lbl = tk.Label(
            parent,
            text=label,
            font=("Courier", 11, "bold"),
            fg=TronStyle.CYAN,
            bg=TronStyle.BG_PANEL,
            anchor='w'
        )
        lbl.grid(row=row*2, column=0, sticky='w', pady=(10, 0))
        
        # Hint
        hint_lbl = tk.Label(
            parent,
            text=hint,
            font=("Courier", 8),
            fg=TronStyle.CYAN_DIM,
            bg=TronStyle.BG_PANEL,
            anchor='w'
        )
        hint_lbl.grid(row=row*2, column=1, sticky='e', pady=(10, 0))
        
        # Slider frame
        slider_frame = tk.Frame(parent, bg=TronStyle.BG_PANEL)
        slider_frame.grid(row=row*2+1, column=0, columnspan=2, sticky='ew', pady=(0, 5))
        
        # Value display
        if is_int:
            val_text = f"{int(var.get())}"
        else:
            val_text = f"{var.get():.1f}"
        val_label = tk.Label(
            slider_frame,
            text=val_text,
            font=("Courier", 10, "bold"),
            fg=TronStyle.MAGENTA,
            bg=TronStyle.BG_PANEL,
            width=5
        )
        val_label.pack(side='right', padx=5)
        
        # Slider
        resolution = 1 if is_int else 0.1
        slider = tk.Scale(
            slider_frame,
            variable=var,
            from_=min_val,
            to=max_val,
            orient='horizontal',
            length=350,
            resolution=resolution,
            showvalue=False,
            bg=TronStyle.BG_PANEL,
            fg=TronStyle.CYAN,
            troughcolor=TronStyle.GRID,
            activebackground=TronStyle.MAGENTA,
            highlightthickness=0,
            command=lambda v, vl=val_label, ii=is_int: self._update_val_label(v, vl, ii)
        )
        slider.pack(side='left', fill='x', expand=True)
        
    def _update_val_label(self, val, label, is_int):
        """Update value label when slider moves."""
        if is_int:
            label.config(text=f"{int(float(val))}")
        else:
            label.config(text=f"{float(val):.1f}")
        self._update_preview()
        
    def _update_preview(self):
        """Update the preview text."""
        food = self.params['food_abundance'].get()
        harsh = self.params['seasonal_harshness'].get()
        herbies = self.params['starting_herbies'].get()
        apex = self.params['starting_apex'].get()
        fish = self.params['starting_fish'].get()
        size = self.params['world_size'].get()
        day_len = self.params['day_length'].get()
        mutation = self.params['mutation_rate'].get()
        preds = self.params['predators_enabled'].get()
        embryo_mode = self.params['embryo_mode'].get()
        
        # Describe the world
        if food < 0.5:
            food_desc = "Barren wasteland"
        elif food < 1.0:
            food_desc = "Sparse resources"
        elif food < 1.5:
            food_desc = "Balanced ecosystem"
        elif food < 2.5:
            food_desc = "Lush environment"
        else:
            food_desc = "Garden of plenty"
            
        if harsh < 0.3:
            season_desc = "eternal spring"
        elif harsh < 0.8:
            season_desc = "mild seasons"
        elif harsh < 1.3:
            season_desc = "natural cycles"
        elif harsh < 1.8:
            season_desc = "harsh winters"
        else:
            season_desc = "brutal extremes"
        
        # Apex description
        if apex == 0:
            apex_desc = "No apex predators"
        elif apex == 1:
            apex_desc = "1 apex predator lurks"
        else:
            apex_desc = f"{apex} apex predators roam"
        
        # Fish description
        if fish == 0:
            fish_desc = ""
        elif fish <= 3:
            fish_desc = f"🐟 {fish} fish"
        else:
            fish_desc = f"🐟 {fish} fish (teeming)"
            
        pred_desc = "+ other species" if preds else "(solo ecosystem)"
        
        # Day length description
        if day_len < 300:
            time_desc = "⚡ Fast time"
        elif day_len > 700:
            time_desc = "🐢 Slow time"
        else:
            time_desc = "Normal time"
        
        # Mutation description
        if mutation < 0.3:
            mut_desc = "🧬 Stable genetics"
        elif mutation > 2.0:
            mut_desc = "🎲 High mutation"
        else:
            mut_desc = ""
        
        # Embryo mode description
        if embryo_mode == 'all':
            embryo_desc = "🧫 Full evo-devo (all species)"
        elif embryo_mode == 'herbie':
            embryo_desc = "🧬 Herbie evo-devo"
        else:
            embryo_desc = ""
        
        preview = f"""
  World: {size}x{size} | {time_desc}
  
  🌿 {herbies} Herbies
  🔴 {apex_desc}
     {pred_desc}
  {fish_desc}
  
  {food_desc} with {season_desc}
  {mut_desc}
  {embryo_desc}
  
  {'⚠️ SURVIVAL MODE' if (food < 0.7 and harsh > 1.3) else ''}
  {'🌸 PARADISE MODE' if (food > 2.0 and harsh < 0.5 and apex == 0) else ''}
  {'💀 NIGHTMARE MODE' if (apex >= 3 and food < 1.0) else ''}
  {'🧪 EVOLUTION LAB' if mutation > 2.0 else ''}
"""
        self.preview_text.config(text=preview)
        
    def _launch(self):
        """Launch the simulation with selected parameters."""
        # Build environment variables for parameters
        env = os.environ.copy()
        env['HERBIE_FOOD_MULT'] = str(self.params['food_abundance'].get())
        env['HERBIE_SEASON_HARSH'] = str(self.params['seasonal_harshness'].get())
        env['HERBIE_START_POP'] = str(self.params['starting_herbies'].get())
        env['HERBIE_START_APEX'] = str(self.params['starting_apex'].get())
        env['HERBIE_START_FISH'] = str(self.params['starting_fish'].get())
        env['HERBIE_WORLD_SIZE'] = str(self.params['world_size'].get())
        env['HERBIE_DAY_LENGTH'] = str(self.params['day_length'].get())
        env['HERBIE_MUTATION_RATE'] = str(self.params['mutation_rate'].get())
        env['HERBIE_PREDATORS'] = '1' if self.params['predators_enabled'].get() else '0'
        env['HERBIE_EMBRYO_MODE'] = self.params['embryo_mode'].get()  # 'herbie', 'all', 'none'
        
        # Close launcher
        self.root.destroy()
        
        # Launch simulation
        # Get the directory where this launcher lives
        launcher_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Run the main module
        subprocess.run(
            [sys.executable, '-m', 'herbie_world'],
            env=env,
            cwd=os.path.dirname(launcher_dir)
        )
        
    def run(self):
        """Start the launcher."""
        self.root.mainloop()


def main():
    """Entry point."""
    launcher = HerbieLauncher()
    launcher.run()


if __name__ == '__main__':
    main()
