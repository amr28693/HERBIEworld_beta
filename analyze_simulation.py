#!/usr/bin/env python3
"""
HERBIE World Analysis Suite
===========================

Comprehensive post-simulation analysis with:
- Population dynamics and derivatives
- Information geometry (Fisher information, KL divergence)
- Information dynamics (transfer entropy, mutual information)
- Phase space analysis
- Correlation analysis across all variables

Run from the herbie_world folder:
    python analyze_simulation.py

Or with specific log file:
    python analyze_simulation.py path/to/event_log.jsonl

Outputs analysis_report.txt and analysis_plots.png
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# =============================================================================
# DATA LOADING
# =============================================================================

def load_events(filepath: str) -> List[dict]:
    """Load events from JSONL file."""
    events = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except FileNotFoundError:
        print(f"[Error] File not found: {filepath}")
        sys.exit(1)
    return events


def extract_time_series(events: List[dict]) -> dict:
    """Extract time series from events."""
    data = {
        'steps': [],
        'population': defaultdict(list),
        'births': [],
        'deaths': [],
        'predations': [],
        'seasons': [],
        'reactions': [],
        'nests': [],
    }
    
    birth_counts = defaultdict(int)
    death_counts = defaultdict(int)
    
    for e in events:
        etype = e.get('type')
        step = e.get('step', 0)
        
        if etype == 'population':
            data['steps'].append(step)
            counts = e.get('counts', {})
            for species, count in counts.items():
                data['population'][species].append((step, count))
        
        elif etype == 'birth':
            data['births'].append({
                'step': step,
                'name': e.get('name'),
                'generation': e.get('generation', 0),
                'parents': e.get('parents', [])
            })
            birth_counts[step // 500] += 1
        
        elif etype == 'death':
            data['deaths'].append({
                'step': step,
                'name': e.get('name'),
                'cause': e.get('cause', 'unknown'),
                'age': e.get('age', 0)
            })
            death_counts[step // 500] += 1
        
        elif etype == 'predation':
            data['predations'].append({
                'step': step,
                'predator': e.get('predator'),
                'prey': e.get('prey_name')
            })
        
        elif etype == 'season':
            data['seasons'].append({
                'step': step,
                'season': e.get('season'),
                'year': e.get('year', 0)
            })
        
        elif etype == 'reaction':
            data['reactions'].append({
                'step': step,
                'type': e.get('reaction_type'),
                'amount': e.get('amount', 0)
            })
        
        elif etype == 'nest_established':
            data['nests'].append({
                'step': step,
                'owner': e.get('owner')
            })
    
    data['birth_rate'] = birth_counts
    data['death_rate'] = death_counts
    
    return data


# =============================================================================
# DERIVATIVE ANALYSIS
# =============================================================================

def compute_derivatives(x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute first and second derivatives of a time series.
    
    Returns: (dx/dt, d²x/dt², smoothed_x)
    """
    if len(x) < 3:
        return np.array([]), np.array([]), x
    
    # Smooth with Gaussian kernel
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(x.astype(float), sigma=2)
    
    # First derivative (central difference)
    dt = np.diff(t)
    dt = np.where(dt == 0, 1, dt)  # Avoid division by zero
    dx = np.diff(smoothed)
    dxdt = dx / dt
    
    # Second derivative
    if len(dxdt) > 1:
        dt2 = dt[:-1]
        dt2 = np.where(dt2 == 0, 1, dt2)
        d2xdt2 = np.diff(dxdt) / dt2
    else:
        d2xdt2 = np.array([])
    
    return dxdt, d2xdt2, smoothed


def analyze_population_dynamics(data: dict) -> dict:
    """
    Comprehensive population dynamics analysis.
    
    Computes:
    - Growth rates (dx/dt)
    - Acceleration (d²x/dt²)
    - Carrying capacity estimates
    - Stability metrics
    """
    results = {}
    
    for species, pop_data in data['population'].items():
        if len(pop_data) < 3:
            continue
        
        steps = np.array([p[0] for p in pop_data])
        counts = np.array([p[1] for p in pop_data])
        
        dxdt, d2xdt2, smoothed = compute_derivatives(counts, steps)
        
        results[species] = {
            'steps': steps,
            'counts': counts,
            'smoothed': smoothed,
            'growth_rate': dxdt,
            'acceleration': d2xdt2,
            'mean_population': np.mean(counts),
            'std_population': np.std(counts),
            'max_population': np.max(counts),
            'min_population': np.min(counts),
            'final_population': counts[-1] if len(counts) > 0 else 0,
            # Stability: low variance in growth rate = stable
            'stability': 1.0 / (1.0 + np.std(dxdt)) if len(dxdt) > 0 else 0,
            # Growth tendency
            'net_growth': np.sum(dxdt) if len(dxdt) > 0 else 0,
        }
        
        # Estimate carrying capacity using logistic fit
        if np.max(counts) > np.min(counts) + 1:
            # Simple estimate: population where growth rate crosses zero from above
            if len(dxdt) > 2:
                zero_crossings = np.where(np.diff(np.sign(dxdt)) < 0)[0]
                if len(zero_crossings) > 0:
                    results[species]['carrying_capacity_est'] = smoothed[zero_crossings[-1]]
    
    return results


# =============================================================================
# INFORMATION GEOMETRY
# =============================================================================

def compute_fisher_information(distribution: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Fisher information of a probability distribution.
    
    Fisher information measures the "curvature" of the likelihood function,
    indicating how much information the distribution carries about its parameter.
    
    I(θ) = E[(d/dθ log p(x|θ))²]
    
    For a discrete distribution, we use the score function approximation.
    """
    p = distribution / (np.sum(distribution) + epsilon)
    p = np.clip(p, epsilon, 1.0)
    
    # Score function: d/dp log(p) = 1/p
    # Fisher info ≈ sum of 1/p weighted by p = sum of 1 = n (trivial)
    # Better: use gradient of log-likelihood
    log_p = np.log(p)
    
    # Approximate derivative using finite differences
    grad_log_p = np.gradient(log_p)
    
    # Fisher information
    fisher = np.sum(p * grad_log_p**2)
    
    return fisher


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Kullback-Leibler divergence D_KL(P || Q).
    
    Measures how much P diverges from Q.
    D_KL(P || Q) = Σ P(x) log(P(x) / Q(x))
    
    This is the "information gained" when moving from prior Q to posterior P.
    """
    p = p / (np.sum(p) + epsilon)
    q = q / (np.sum(q) + epsilon)
    
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    
    return np.sum(p * np.log(p / q))


def compute_jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric KL).
    
    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    
    Bounded between 0 and ln(2).
    """
    m = 0.5 * (p + q)
    return 0.5 * compute_kl_divergence(p, m) + 0.5 * compute_kl_divergence(q, m)


def analyze_information_geometry(data: dict) -> dict:
    """
    Information-geometric analysis of population distributions.
    
    Computes:
    - Fisher information over time
    - KL divergence between consecutive snapshots
    - Jensen-Shannon divergence
    - Information flow rates
    """
    results = {
        'fisher_info': {},
        'kl_divergence': {},
        'js_divergence': {},
        'info_flow_rate': {}
    }
    
    species_list = list(data['population'].keys())
    if not species_list:
        return results
    
    # Get population vectors at each step
    steps = sorted(set(s for sp in data['population'].values() for s, _ in sp))
    
    if len(steps) < 2:
        return results
    
    # Build population matrix (steps x species)
    pop_matrix = []
    for step in steps:
        row = []
        for sp in species_list:
            sp_data = dict(data['population'][sp])
            row.append(sp_data.get(step, 0))
        pop_matrix.append(row)
    
    pop_matrix = np.array(pop_matrix, dtype=float)
    
    # Fisher information at each timestep
    fisher_vals = []
    for i, row in enumerate(pop_matrix):
        if np.sum(row) > 0:
            fisher_vals.append(compute_fisher_information(row))
        else:
            fisher_vals.append(0)
    
    results['fisher_info'] = {
        'values': fisher_vals,
        'steps': steps,
        'mean': np.mean(fisher_vals),
        'trend': np.polyfit(range(len(fisher_vals)), fisher_vals, 1)[0] if len(fisher_vals) > 1 else 0
    }
    
    # KL divergence between consecutive timesteps
    kl_vals = []
    for i in range(1, len(pop_matrix)):
        if np.sum(pop_matrix[i-1]) > 0 and np.sum(pop_matrix[i]) > 0:
            kl = compute_kl_divergence(pop_matrix[i], pop_matrix[i-1])
            kl_vals.append(kl)
    
    results['kl_divergence'] = {
        'values': kl_vals,
        'mean': np.mean(kl_vals) if kl_vals else 0,
        'max': np.max(kl_vals) if kl_vals else 0,
        'total_info_change': np.sum(kl_vals) if kl_vals else 0
    }
    
    # JS divergence (symmetric)
    js_vals = []
    for i in range(1, len(pop_matrix)):
        if np.sum(pop_matrix[i-1]) > 0 and np.sum(pop_matrix[i]) > 0:
            js = compute_jensen_shannon(pop_matrix[i], pop_matrix[i-1])
            js_vals.append(js)
    
    results['js_divergence'] = {
        'values': js_vals,
        'mean': np.mean(js_vals) if js_vals else 0
    }
    
    return results


# =============================================================================
# INFORMATION DYNAMICS
# =============================================================================

def compute_entropy(x: np.ndarray, bins: int = 20) -> float:
    """Compute Shannon entropy of a distribution."""
    if len(x) == 0:
        return 0.0
    
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zeros
    if len(hist) == 0:
        return 0.0
    
    # Normalize
    hist = hist / np.sum(hist)
    return -np.sum(hist * np.log(hist + 1e-10))


def compute_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """
    Compute mutual information I(X; Y).
    
    I(X; Y) = H(X) + H(Y) - H(X, Y)
    
    Measures how much knowing X reduces uncertainty about Y.
    """
    if len(x) < 10 or len(y) < 10:
        return 0.0
    
    # Make same length
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    
    # Individual entropies
    h_x = compute_entropy(x, bins)
    h_y = compute_entropy(y, bins)
    
    # Joint entropy
    joint, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    joint = joint[joint > 0]
    if len(joint) == 0:
        return 0.0
    joint = joint / np.sum(joint)
    h_xy = -np.sum(joint * np.log(joint + 1e-10))
    
    return max(0, h_x + h_y - h_xy)


def compute_transfer_entropy(source: np.ndarray, target: np.ndarray, 
                              lag: int = 1, bins: int = 10) -> float:
    """
    Compute transfer entropy from source to target.
    
    TE(X → Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
    
    Measures directional information flow: how much does knowing X's past
    help predict Y's future beyond knowing Y's own past?
    """
    if len(source) < lag + 10 or len(target) < lag + 10:
        return 0.0
    
    n = min(len(source), len(target)) - lag
    
    # Create lagged versions
    y_t = target[lag:lag+n]
    y_past = target[:n]
    x_past = source[:n]
    
    # H(Y_t | Y_past) ≈ H(Y_t, Y_past) - H(Y_past)
    h_y_past = compute_entropy(y_past, bins)
    
    joint_y, _, _ = np.histogram2d(y_t, y_past, bins=bins)
    joint_y = joint_y / (np.sum(joint_y) + 1e-10)
    joint_y = joint_y[joint_y > 0]
    h_y_joint = -np.sum(joint_y * np.log(joint_y + 1e-10)) if len(joint_y) > 0 else 0
    
    h_y_given_past = h_y_joint - h_y_past
    
    # H(Y_t | Y_past, X_past) - need 3D histogram approximation
    # Simplify: use conditional mutual information approximation
    mi_y_x_given_past = compute_mutual_information(y_t - y_past, x_past, bins)
    
    # TE ≈ MI(Y_t; X_past | Y_past)
    te = max(0, mi_y_x_given_past)
    
    return te


def analyze_information_dynamics(data: dict) -> dict:
    """
    Information dynamics analysis.
    
    Computes:
    - Transfer entropy between species
    - Mutual information between birth/death rates
    - Entropy trends over time
    - Information flow network
    """
    results = {
        'transfer_entropy': {},
        'mutual_information': {},
        'entropy_trends': {},
        'info_flow_network': []
    }
    
    # Get population time series for each species
    species_series = {}
    for species, pop_data in data['population'].items():
        if len(pop_data) >= 10:
            species_series[species] = np.array([p[1] for p in pop_data], dtype=float)
    
    if len(species_series) < 2:
        return results
    
    # Transfer entropy between all pairs
    species_names = list(species_series.keys())
    for i, sp1 in enumerate(species_names):
        for j, sp2 in enumerate(species_names):
            if i != j:
                te = compute_transfer_entropy(species_series[sp1], species_series[sp2])
                key = f"{sp1} → {sp2}"
                results['transfer_entropy'][key] = te
                
                if te > 0.01:
                    results['info_flow_network'].append({
                        'source': sp1,
                        'target': sp2,
                        'strength': te
                    })
    
    # Mutual information between species
    for i, sp1 in enumerate(species_names):
        for j, sp2 in enumerate(species_names):
            if i < j:
                mi = compute_mutual_information(species_series[sp1], species_series[sp2])
                key = f"{sp1} ↔ {sp2}"
                results['mutual_information'][key] = mi
    
    # Entropy trends
    for species, series in species_series.items():
        # Rolling entropy
        window = min(10, len(series) // 3)
        if window < 3:
            continue
        
        entropies = []
        for i in range(len(series) - window):
            h = compute_entropy(series[i:i+window])
            entropies.append(h)
        
        if entropies:
            results['entropy_trends'][species] = {
                'values': entropies,
                'mean': np.mean(entropies),
                'trend': np.polyfit(range(len(entropies)), entropies, 1)[0] if len(entropies) > 1 else 0
            }
    
    return results


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def analyze_correlations(data: dict) -> dict:
    """
    Cross-correlation analysis between all measurable quantities.
    """
    results = {
        'population_correlations': {},
        'birth_death_correlation': 0,
        'predation_population_correlation': {},
        'seasonal_effects': {}
    }
    
    # Population correlations
    species_series = {}
    for species, pop_data in data['population'].items():
        if len(pop_data) >= 5:
            species_series[species] = np.array([p[1] for p in pop_data], dtype=float)
    
    species_names = list(species_series.keys())
    for i, sp1 in enumerate(species_names):
        for j, sp2 in enumerate(species_names):
            if i < j:
                n = min(len(species_series[sp1]), len(species_series[sp2]))
                if n >= 5:
                    corr = np.corrcoef(species_series[sp1][:n], species_series[sp2][:n])[0, 1]
                    results['population_correlations'][f"{sp1}-{sp2}"] = corr
    
    # Birth-death correlation
    birth_steps = [b['step'] for b in data['births']]
    death_steps = [d['step'] for d in data['deaths']]
    
    if birth_steps and death_steps:
        # Bin into time windows
        max_step = max(max(birth_steps) if birth_steps else 0, 
                      max(death_steps) if death_steps else 0)
        bins = np.arange(0, max_step + 500, 500)
        
        birth_hist, _ = np.histogram(birth_steps, bins=bins)
        death_hist, _ = np.histogram(death_steps, bins=bins)
        
        if len(birth_hist) >= 3:
            results['birth_death_correlation'] = np.corrcoef(birth_hist, death_hist)[0, 1]
    
    return results


# =============================================================================
# PHASE SPACE ANALYSIS
# =============================================================================

def analyze_phase_space(pop_dynamics: dict) -> dict:
    """
    Phase space analysis: population vs growth rate trajectories.
    
    Identifies:
    - Fixed points (equilibria)
    - Limit cycles
    - Attractors/repellers
    """
    results = {}
    
    for species, dynamics in pop_dynamics.items():
        if 'growth_rate' not in dynamics or len(dynamics['growth_rate']) < 5:
            continue
        
        pop = dynamics['smoothed'][:-1] if len(dynamics['smoothed']) > len(dynamics['growth_rate']) else dynamics['smoothed']
        growth = dynamics['growth_rate']
        
        n = min(len(pop), len(growth))
        pop = pop[:n]
        growth = growth[:n]
        
        # Find zero crossings of growth rate (potential equilibria)
        zero_crossings = np.where(np.diff(np.sign(growth)))[0]
        equilibria = pop[zero_crossings] if len(zero_crossings) > 0 else []
        
        # Classify equilibria as stable (growth decreasing) or unstable
        stable_eq = []
        unstable_eq = []
        for idx in zero_crossings:
            if idx > 0 and idx < len(growth) - 1:
                if growth[idx-1] > 0 and growth[idx+1] < 0:
                    stable_eq.append(pop[idx])
                else:
                    unstable_eq.append(pop[idx])
        
        results[species] = {
            'equilibria': list(equilibria),
            'stable_equilibria': stable_eq,
            'unstable_equilibria': unstable_eq,
            'phase_portrait': list(zip(pop.tolist(), growth.tolist()))
        }
    
    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(data: dict, pop_dynamics: dict, info_geom: dict, 
                   info_dyn: dict, correlations: dict, phase: dict) -> str:
    """Generate comprehensive analysis report."""
    lines = []
    
    # Header
    lines.append("=" * 80)
    lines.append("         HERBIE WORLD - COMPREHENSIVE ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary statistics
    lines.append("-" * 80)
    lines.append("  1. SUMMARY STATISTICS")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"  Total events analyzed: {sum(len(v) if isinstance(v, list) else 0 for v in data.values())}")
    lines.append(f"  Total births: {len(data['births'])}")
    lines.append(f"  Total deaths: {len(data['deaths'])}")
    lines.append(f"  Predation events: {len(data['predations'])}")
    lines.append(f"  Season changes: {len(data['seasons'])}")
    lines.append(f"  Chemical reactions: {len(data['reactions'])}")
    lines.append(f"  Nests established: {len(data['nests'])}")
    lines.append("")
    
    # Population dynamics
    lines.append("-" * 80)
    lines.append("  2. POPULATION DYNAMICS")
    lines.append("-" * 80)
    lines.append("")
    
    for species, dynamics in pop_dynamics.items():
        lines.append(f"  {species}:")
        lines.append(f"    Mean population: {dynamics['mean_population']:.1f} ± {dynamics['std_population']:.1f}")
        lines.append(f"    Range: {dynamics['min_population']} - {dynamics['max_population']}")
        lines.append(f"    Stability index: {dynamics['stability']:.3f} (1.0 = perfectly stable)")
        lines.append(f"    Net growth tendency: {dynamics['net_growth']:+.2f}")
        if 'carrying_capacity_est' in dynamics:
            lines.append(f"    Estimated carrying capacity: {dynamics['carrying_capacity_est']:.1f}")
        lines.append("")
    
    # Derivative analysis
    lines.append("-" * 80)
    lines.append("  3. DERIVATIVE ANALYSIS (Rate of Change)")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  First derivative (dx/dt) = instantaneous growth rate")
    lines.append("  Second derivative (d²x/dt²) = acceleration of growth")
    lines.append("")
    
    for species, dynamics in pop_dynamics.items():
        if len(dynamics.get('growth_rate', [])) > 0:
            gr = dynamics['growth_rate']
            lines.append(f"  {species}:")
            lines.append(f"    Growth rate: mean={np.mean(gr):.3f}, std={np.std(gr):.3f}")
            lines.append(f"    Max growth: {np.max(gr):.3f}, Max decline: {np.min(gr):.3f}")
            if len(dynamics.get('acceleration', [])) > 0:
                acc = dynamics['acceleration']
                lines.append(f"    Acceleration: mean={np.mean(acc):.4f}")
            lines.append("")
    
    # Information geometry
    lines.append("-" * 80)
    lines.append("  4. INFORMATION GEOMETRY")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Fisher Information: measures parameter sensitivity")
    lines.append("    High Fisher info = distribution is 'sharp', informative")
    lines.append("    Low Fisher info = distribution is 'flat', uncertain")
    lines.append("")
    
    if 'fisher_info' in info_geom and info_geom['fisher_info']:
        fi = info_geom['fisher_info']
        lines.append(f"  Mean Fisher information: {fi['mean']:.4f}")
        lines.append(f"  Fisher info trend: {fi['trend']:+.6f} per step")
        trend_desc = "increasing (system becoming more structured)" if fi['trend'] > 0 else "decreasing (system becoming more uncertain)"
        lines.append(f"    Interpretation: {trend_desc}")
        lines.append("")
    
    lines.append("  KL Divergence: measures distribution change over time")
    lines.append("    D_KL(P||Q) = information gained from Q to P")
    lines.append("")
    
    if 'kl_divergence' in info_geom and info_geom['kl_divergence']:
        kl = info_geom['kl_divergence']
        lines.append(f"  Mean KL divergence: {kl['mean']:.4f} nats/step")
        lines.append(f"  Max KL divergence: {kl['max']:.4f} nats (biggest distribution shift)")
        lines.append(f"  Total information change: {kl['total_info_change']:.4f} nats")
        lines.append("")
    
    # Information dynamics
    lines.append("-" * 80)
    lines.append("  5. INFORMATION DYNAMICS")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Transfer Entropy: measures directed information flow")
    lines.append("    TE(X→Y) = how much X's past predicts Y's future")
    lines.append("")
    
    if info_dyn.get('transfer_entropy'):
        lines.append("  Information flow between species:")
        for pair, te in sorted(info_dyn['transfer_entropy'].items(), key=lambda x: -x[1])[:10]:
            if te > 0.001:
                lines.append(f"    {pair}: {te:.4f} nats")
        lines.append("")
    
    if info_dyn.get('mutual_information'):
        lines.append("  Mutual Information (symmetric coupling):")
        for pair, mi in sorted(info_dyn['mutual_information'].items(), key=lambda x: -x[1])[:10]:
            if mi > 0.001:
                lines.append(f"    {pair}: {mi:.4f} nats")
        lines.append("")
    
    # Correlations
    lines.append("-" * 80)
    lines.append("  6. CORRELATION ANALYSIS")
    lines.append("-" * 80)
    lines.append("")
    
    if correlations.get('population_correlations'):
        lines.append("  Population correlations (Pearson r):")
        for pair, corr in sorted(correlations['population_correlations'].items(), 
                                 key=lambda x: -abs(x[1])):
            interp = "strongly positive" if corr > 0.7 else \
                     "positive" if corr > 0.3 else \
                     "weakly positive" if corr > 0 else \
                     "weakly negative" if corr > -0.3 else \
                     "negative" if corr > -0.7 else "strongly negative"
            lines.append(f"    {pair}: r = {corr:+.3f} ({interp})")
        lines.append("")
    
    if correlations.get('birth_death_correlation'):
        corr = correlations['birth_death_correlation']
        lines.append(f"  Birth-death rate correlation: r = {corr:+.3f}")
        if corr > 0.5:
            lines.append("    → High births correlate with high deaths (dynamic equilibrium)")
        elif corr < -0.5:
            lines.append("    → High births correlate with low deaths (population expansion)")
        lines.append("")
    
    # Phase space
    lines.append("-" * 80)
    lines.append("  7. PHASE SPACE ANALYSIS")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Equilibria: where growth rate = 0")
    lines.append("    Stable: population returns after perturbation")
    lines.append("    Unstable: population diverges after perturbation")
    lines.append("")
    
    for species, ph in phase.items():
        if ph.get('stable_equilibria') or ph.get('unstable_equilibria'):
            lines.append(f"  {species}:")
            if ph['stable_equilibria']:
                lines.append(f"    Stable equilibria at: {[f'{e:.1f}' for e in ph['stable_equilibria']]}")
            if ph['unstable_equilibria']:
                lines.append(f"    Unstable equilibria at: {[f'{e:.1f}' for e in ph['unstable_equilibria']]}")
            lines.append("")
    
    # Death cause analysis
    lines.append("-" * 80)
    lines.append("  8. MORTALITY ANALYSIS")
    lines.append("-" * 80)
    lines.append("")
    
    cause_counts = defaultdict(int)
    ages_by_cause = defaultdict(list)
    for d in data['deaths']:
        cause = d.get('cause', 'unknown')
        cause_counts[cause] += 1
        ages_by_cause[cause].append(d.get('age', 0))
    
    total_deaths = sum(cause_counts.values())
    for cause, count in sorted(cause_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_deaths if total_deaths > 0 else 0
        avg_age = np.mean(ages_by_cause[cause]) if ages_by_cause[cause] else 0
        lines.append(f"  {cause}: {count} ({pct:.1f}%), avg age at death: {avg_age:.0f} steps")
    lines.append("")
    
    # Footer
    lines.append("=" * 80)
    lines.append("  END OF ANALYSIS")
    lines.append("=" * 80)
    
    return "\n".join(lines)


# =============================================================================
# PLOTTING
# =============================================================================

def generate_plots(data: dict, pop_dynamics: dict, info_geom: dict,
                  output_path: str):
    """Generate analysis plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("[Warning] matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('HERBIE World Simulation Analysis', fontsize=14, fontweight='bold')
    
    # 1. Population time series
    ax = axes[0, 0]
    for species, dynamics in pop_dynamics.items():
        steps = dynamics['steps']
        counts = dynamics['counts']
        ax.plot(steps, counts, label=species, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Population')
    ax.set_title('Population Dynamics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Growth rates
    ax = axes[0, 1]
    for species, dynamics in pop_dynamics.items():
        if 'growth_rate' in dynamics and len(dynamics['growth_rate']) > 0:
            gr = dynamics['growth_rate']
            steps = dynamics['steps'][1:len(gr)+1]
            ax.plot(steps, gr, label=species, linewidth=1.5, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('dx/dt')
    ax.set_title('Growth Rates (First Derivative)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Phase portrait (Herbie if available)
    ax = axes[0, 2]
    if 'Herbie' in pop_dynamics:
        dynamics = pop_dynamics['Herbie']
        if 'growth_rate' in dynamics and len(dynamics['growth_rate']) > 0:
            pop = dynamics['smoothed'][:-1][:len(dynamics['growth_rate'])]
            growth = dynamics['growth_rate'][:len(pop)]
            ax.scatter(pop, growth, c=range(len(pop)), cmap='viridis', s=20, alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Population')
            ax.set_ylabel('Growth Rate')
            ax.set_title('Phase Portrait (Herbie)')
    ax.grid(True, alpha=0.3)
    
    # 4. Fisher information
    ax = axes[1, 0]
    if info_geom.get('fisher_info') and info_geom['fisher_info'].get('values'):
        fi = info_geom['fisher_info']
        ax.plot(fi['steps'][:len(fi['values'])], fi['values'], 'b-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Fisher Information')
        ax.set_title('Fisher Information Over Time')
    ax.grid(True, alpha=0.3)
    
    # 5. KL divergence
    ax = axes[1, 1]
    if info_geom.get('kl_divergence') and info_geom['kl_divergence'].get('values'):
        kl_vals = info_geom['kl_divergence']['values']
        ax.plot(range(len(kl_vals)), kl_vals, 'r-', linewidth=2)
        ax.set_xlabel('Transition')
        ax.set_ylabel('KL Divergence (nats)')
        ax.set_title('Distribution Change Rate')
    ax.grid(True, alpha=0.3)
    
    # 6. Death causes pie chart
    ax = axes[1, 2]
    cause_counts = defaultdict(int)
    for d in data['deaths']:
        cause = d.get('cause', 'unknown')
        # Simplify cause names
        if cause.startswith('predation:'):
            cause = 'Predation'
        cause_counts[cause] += 1
    
    if cause_counts:
        labels = list(cause_counts.keys())
        sizes = list(cause_counts.values())
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Causes of Death')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Analysis] Saved plots to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main analysis entry point."""
    # Find event log
    if len(sys.argv) > 1:
        event_file = sys.argv[1]
    else:
        # Look in common locations
        candidates = [
            'data/event_log.jsonl',
            '../data/event_log.jsonl',
            'event_log.jsonl',
        ]
        event_file = None
        for c in candidates:
            if os.path.exists(c):
                event_file = c
                break
        
        if event_file is None:
            print("Usage: python analyze_simulation.py [path/to/event_log.jsonl]")
            print("\nNo event_log.jsonl found in common locations.")
            sys.exit(1)
    
    print(f"[Analysis] Loading events from {event_file}...")
    events = load_events(event_file)
    print(f"[Analysis] Loaded {len(events)} events")
    
    if len(events) == 0:
        print("[Error] No events found in log file")
        sys.exit(1)
    
    print("[Analysis] Extracting time series...")
    data = extract_time_series(events)
    
    print("[Analysis] Computing population dynamics...")
    pop_dynamics = analyze_population_dynamics(data)
    
    print("[Analysis] Computing information geometry...")
    info_geom = analyze_information_geometry(data)
    
    print("[Analysis] Computing information dynamics...")
    info_dyn = analyze_information_dynamics(data)
    
    print("[Analysis] Computing correlations...")
    correlations = analyze_correlations(data)
    
    print("[Analysis] Analyzing phase space...")
    phase = analyze_phase_space(pop_dynamics)
    
    print("[Analysis] Generating report...")
    report = generate_report(data, pop_dynamics, info_geom, info_dyn, correlations, phase)
    
    # Save report
    report_path = os.path.join(os.path.dirname(event_file), 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"[Analysis] Saved report to {report_path}")
    
    # Print to console too
    print("\n" + report)
    
    # Generate plots
    plot_path = os.path.join(os.path.dirname(event_file), 'analysis_plots.png')
    generate_plots(data, pop_dynamics, info_geom, plot_path)
    
    print("\n[Analysis] Complete!")


if __name__ == '__main__':
    main()
