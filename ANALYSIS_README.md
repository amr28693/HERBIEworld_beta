# HERBIE World Analysis Scripts - User Guide

## Overview

HERBIE World includes two powerful analysis scripts that help you understand 
your simulation data:

1. **analyze_simulation.py** - General simulation analysis (populations, 
   information theory, dynamics)
2. **analyze_genetics.py** - Mendelian genetics analysis (allele frequencies, 
   Hardy-Weinberg, selection)

---

## Prerequisites

Before running the scripts, make sure you have:

1. **Python 3.8+** installed
2. **Required packages**: numpy, matplotlib (optional for plots)

To check if you have these:
```bash
python --version          # Should show 3.8 or higher
python -c "import numpy"  # Should show no error
python -c "import matplotlib"  # Optional, for plots
```

If you need to install packages:
```bash
pip install numpy matplotlib
```

---

## Running the Simulation First

The analysis scripts need data from a simulation run. Here's how:

### Step 1: Run the Simulation

```bash
# Navigate to the herbie_world folder
cd /path/to/herbie_world

# Run the simulation
python -m herbie_world
```

Or use the launcher:
```bash
python launcher.py
```

### Step 2: Let it Run

- Let the simulation run for at least **1000-5000 steps** for meaningful data
- Watch the terminal for events being logged
- Press **Space** to pause if needed
- Close the window when done (data auto-saves)

### Step 3: Find Your Data

After running, your data is saved in:
```
herbie_world/
└── data/
    ├── event_log.jsonl      ← Main data file (required)
    ├── world_history.txt    ← Human-readable summary
    └── narrative_log.txt    ← Story of events
```

---

## Running the Simulation Analyzer

This script analyzes population dynamics, information theory metrics, and more.

### Basic Usage

```bash
# Navigate to herbie_world folder
cd /path/to/herbie_world

# Run with default data location
python analyze_simulation.py

# Or specify a specific log file
python analyze_simulation.py data/event_log.jsonl

# Or analyze a log from another location
python analyze_simulation.py /path/to/other/event_log.jsonl
```

### What It Produces

After running, you'll find these new files in the same folder as your log:

| File | Description |
|------|-------------|
| `analysis_report.txt` | Comprehensive text report with all metrics |
| `analysis_plots.png` | 6-panel visualization of dynamics |

### What's Analyzed

1. **Summary Statistics** - Total births, deaths, predations
2. **Population Dynamics** - Growth rates, stability, carrying capacity
3. **Derivative Analysis** - First and second derivatives (acceleration)
4. **Information Geometry** - Fisher information, KL divergence
5. **Information Dynamics** - Transfer entropy, mutual information
6. **Correlation Analysis** - Species interactions
7. **Phase Space** - Equilibrium detection

---

## Running the Genetics Analyzer

This script performs classical population genetics analysis.

### Basic Usage

```bash
# Navigate to herbie_world folder
cd /path/to/herbie_world

# Run with default data location
python analyze_genetics.py

# Or specify a specific log file
python analyze_genetics.py data/event_log.jsonl
```

### What It Produces

| File | Description |
|------|-------------|
| `genetics_report.txt` | Full genetics analysis report |
| `genetics_plots.png` | 6-panel genetic visualization |
| `allele_frequencies.csv` | Time series data (spreadsheet-compatible) |
| `pedigree.json` | Family tree structure (for visualization tools) |

### What's Analyzed

1. **Genetic Diversity** - Alleles per gene, Shannon diversity
2. **Inbreeding Coefficient (F)** - Population structure
3. **Hardy-Weinberg Tests** - Chi-squared equilibrium tests
4. **Selection Coefficients** - Which traits are being selected
5. **Linkage Disequilibrium** - Gene associations
6. **Drift vs Selection** - Detecting evolutionary forces
7. **Pedigree Analysis** - Lineage tracking

---

## Example Workflow

Here's a complete example session:

```bash
# 1. Navigate to herbie_world
cd ~/Desktop/herbie_world

# 2. Run simulation with launcher
python launcher.py

# 3. (Simulation runs... let it go for 5000+ steps, then close)

# 4. Run simulation analysis
python analyze_simulation.py
# Output: analysis_report.txt, analysis_plots.png

# 5. Run genetics analysis
python analyze_genetics.py
# Output: genetics_report.txt, genetics_plots.png, 
#         allele_frequencies.csv, pedigree.json

# 6. View results
cat data/analysis_report.txt
cat data/genetics_report.txt
open data/analysis_plots.png    # Mac
# or: xdg-open data/analysis_plots.png  # Linux
# or: start data/analysis_plots.png     # Windows
```

---

## Troubleshooting

### "No module named 'numpy'"
```bash
pip install numpy
```

### "No module named 'matplotlib'"
```bash
pip install matplotlib
```
(Plots will be skipped if matplotlib is missing, but reports still work)

### "File not found: event_log.jsonl"
You need to run a simulation first! The log file is created when creatures 
are born, die, bond, etc. Run the simulation for at least a few minutes.

### "No genetic data found"
The genetics analyzer needs birth events with genetic info. Make sure your 
simulation ran long enough for reproduction to occur (usually 500+ steps).

### Empty or sparse results
Let the simulation run longer! Good analysis needs:
- **Minimum**: 1,000 steps
- **Recommended**: 5,000+ steps
- **Best**: 10,000+ steps (overnight run)

---

## Understanding the Output

### Analysis Report Key Metrics

| Metric | What It Means |
|--------|---------------|
| **Fisher Information** | High = population is informative; Low = uncertain |
| **KL Divergence** | How much the population distribution changed |
| **Transfer Entropy** | Information flow between species |
| **Carrying Capacity** | Estimated stable population size |
| **Selection Coefficient (s)** | Positive = favored; Negative = disfavored |
| **Inbreeding (F)** | 0 = random mating; >0.15 = significant inbreeding |
| **Hardy-Weinberg χ²** | <3.84 = equilibrium; >3.84 = selection/drift |

---

## Tips for Better Analysis

1. **Longer runs = better data** - Overnight runs give excellent results
2. **Higher mutation rate** - Use launcher slider for more genetic variation
3. **Larger populations** - More Herbies = better statistics
4. **Multiple runs** - Compare different parameter settings

---

## Contact & Issues

These scripts are part of HERBIE World, a PDE-based artificial life simulation.
For issues or questions, check the simulation logs for error messages.
