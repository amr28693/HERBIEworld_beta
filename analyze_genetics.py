#!/usr/bin/env python3
"""
Genetics Analysis Script - Classical and Population Genetics Analysis

Analyzes genetic data from HERBIE World simulations:
- Allele frequency tracking over generations
- Hardy-Weinberg equilibrium tests
- Linkage disequilibrium analysis
- Heterozygosity and inbreeding coefficients
- Selection coefficient estimation
- Punnett square predictions
- Phylogenetic reconstruction
- Drift vs selection detection

Usage:
    python analyze_genetics.py [path/to/event_log.jsonl]
    
Outputs:
    genetics_report.txt - Comprehensive text report
    genetics_plots.png - Multi-panel visualization
    allele_frequencies.csv - Time series data
    pedigree.json - Family tree structure
"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plots will be skipped.")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Individual:
    """An individual creature with genetic data."""
    creature_id: str
    species: str
    generation: int
    birth_step: int
    death_step: Optional[int]
    parent_id: Optional[str]
    mate_id: Optional[str]
    offspring_ids: List[str]
    
    # Genetic data
    genome_id: str
    maternal_genome_id: Optional[str]
    paternal_genome_id: Optional[str]
    genotypes: Dict[str, str]  # gene_name -> genotype string (e.g., "Aa")
    phenotypes: Dict[str, float]  # gene_name -> expressed value
    
    # Traits (old system, for compatibility)
    traits: Dict[str, float]


@dataclass 
class PopulationSnapshot:
    """Population state at a given time."""
    step: int
    n_individuals: int
    allele_frequencies: Dict[str, Dict[str, float]]
    genotype_frequencies: Dict[str, Dict[str, float]]
    heterozygosity: Dict[str, float]
    mean_phenotypes: Dict[str, float]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_genetic_data(filepath: str) -> Tuple[List[Individual], List[PopulationSnapshot]]:
    """
    Load genetic data from event log.
    
    Extracts:
    - Birth events with genetic info
    - Death events
    - Mating events
    - Population snapshots
    """
    individuals: Dict[str, Individual] = {}
    snapshots: List[PopulationSnapshot] = []
    
    snapshot_interval = 500  # Create snapshots every N steps
    last_snapshot_step = 0
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        return [], []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            event_type = event.get('type') or event.get('event')  # Support both formats
            step = event.get('step', 0)
            
            # Birth events
            if event_type == 'birth':
                creature_id = event.get('creature_id', str(len(individuals)))
                
                # Extract genetic data
                genome_data = event.get('genome', {})
                traits_data = event.get('traits', {})
                
                ind = Individual(
                    creature_id=creature_id,
                    species=event.get('species', 'Unknown'),
                    generation=event.get('generation', 0),
                    birth_step=step,
                    death_step=None,
                    parent_id=event.get('parent_id'),
                    mate_id=None,
                    offspring_ids=[],
                    genome_id=genome_data.get('genome_id', creature_id),
                    maternal_genome_id=genome_data.get('maternal_genome_id'),
                    paternal_genome_id=genome_data.get('paternal_genome_id'),
                    genotypes=genome_data.get('genotypes', {}),
                    phenotypes=genome_data.get('phenotypes', {}),
                    traits=traits_data
                )
                
                individuals[creature_id] = ind
                
                # Link to parent
                if ind.parent_id and ind.parent_id in individuals:
                    individuals[ind.parent_id].offspring_ids.append(creature_id)
            
            # Death events
            elif event_type in ['death', 'predation']:
                creature_id = event.get('creature_id')
                if creature_id in individuals:
                    individuals[creature_id].death_step = step
            
            # Development events - merge trait modifiers into individuals
            elif event_type == 'development':
                creature_id = event.get('creature_id')
                if creature_id in individuals:
                    trait_mods = event.get('trait_modifiers', {})
                    # Add developmental data to traits
                    for trait_name, modifier in trait_mods.items():
                        individuals[creature_id].traits[trait_name] = modifier
                    # Also store developmental metrics
                    individuals[creature_id].traits['_n_basins'] = event.get('n_basins', 0)
                    individuals[creature_id].traits['_bilateral_symmetry'] = event.get('bilateral_symmetry', 1.0)
            
            # Mating/bonding events
            elif event_type == 'bond':
                id1 = event.get('creature_1_id')
                id2 = event.get('creature_2_id')
                if id1 in individuals:
                    individuals[id1].mate_id = id2
                if id2 in individuals:
                    individuals[id2].mate_id = id1
            
            # Create population snapshots
            if step - last_snapshot_step >= snapshot_interval:
                # Get living individuals at this step
                living = [ind for ind in individuals.values() 
                         if ind.birth_step <= step and 
                         (ind.death_step is None or ind.death_step > step)]
                
                if living:
                    snapshot = create_population_snapshot(living, step)
                    snapshots.append(snapshot)
                
                last_snapshot_step = step
    
    # Final snapshot
    all_individuals = list(individuals.values())
    if all_individuals:
        final_step = max(ind.death_step or ind.birth_step for ind in all_individuals)
        living = [ind for ind in all_individuals 
                 if ind.death_step is None or ind.death_step >= final_step - 100]
        if living:
            snapshots.append(create_population_snapshot(living, final_step))
    
    return list(individuals.values()), snapshots


def create_population_snapshot(individuals: List[Individual], step: int) -> PopulationSnapshot:
    """Create a population snapshot from living individuals."""
    
    # Collect all genotypes
    all_genotypes: Dict[str, List[str]] = defaultdict(list)
    all_phenotypes: Dict[str, List[float]] = defaultdict(list)
    
    for ind in individuals:
        for gene, geno in ind.genotypes.items():
            all_genotypes[gene].append(geno)
        for gene, pheno in ind.phenotypes.items():
            all_phenotypes[gene].append(pheno)
        
        # Convert continuous traits to pseudo-genotypes for population genetics
        for trait, value in ind.traits.items():
            all_phenotypes[f"trait_{trait}"].append(value)
            
            # Bin continuous traits into pseudo-alleles (H=high, M=medium, L=low)
            if value > 1.1:
                pseudo_geno = "HH"  # High homozygote
            elif value > 1.0:
                pseudo_geno = "HM"  # High/medium heterozygote
            elif value > 0.9:
                pseudo_geno = "MM"  # Medium homozygote
            elif value > 0.8:
                pseudo_geno = "ML"  # Medium/low heterozygote
            else:
                pseudo_geno = "LL"  # Low homozygote
            
            all_genotypes[f"trait_{trait}"].append(pseudo_geno)
    
    # Calculate allele frequencies
    allele_freqs: Dict[str, Dict[str, float]] = {}
    for gene, genotypes in all_genotypes.items():
        allele_counts: Dict[str, int] = defaultdict(int)
        total_alleles = 0
        
        for geno in genotypes:
            # Parse genotype string (e.g., "Aa" -> ['A', 'a'])
            if len(geno) >= 2:
                allele_counts[geno[0]] += 1
                allele_counts[geno[1]] += 1
                total_alleles += 2
        
        if total_alleles > 0:
            allele_freqs[gene] = {a: c/total_alleles for a, c in allele_counts.items()}
    
    # Calculate genotype frequencies
    genotype_freqs: Dict[str, Dict[str, float]] = {}
    for gene, genotypes in all_genotypes.items():
        geno_counts: Dict[str, int] = defaultdict(int)
        for geno in genotypes:
            # Normalize genotype (sort alleles)
            if len(geno) >= 2:
                normalized = ''.join(sorted(geno[:2]))
                geno_counts[normalized] += 1
        
        total = len(genotypes)
        if total > 0:
            genotype_freqs[gene] = {g: c/total for g, c in geno_counts.items()}
    
    # Calculate heterozygosity
    heterozygosity: Dict[str, float] = {}
    for gene, genotypes in all_genotypes.items():
        n_het = sum(1 for g in genotypes if len(g) >= 2 and g[0] != g[1])
        heterozygosity[gene] = n_het / len(genotypes) if genotypes else 0.0
    
    # Mean phenotypes
    mean_phenotypes = {gene: np.mean(values) for gene, values in all_phenotypes.items() if values}
    
    return PopulationSnapshot(
        step=step,
        n_individuals=len(individuals),
        allele_frequencies=allele_freqs,
        genotype_frequencies=genotype_freqs,
        heterozygosity=heterozygosity,
        mean_phenotypes=mean_phenotypes
    )


# ============================================================================
# GENETIC ANALYSES
# ============================================================================

def hardy_weinberg_test(genotype_freqs: Dict[str, float], 
                        allele_freqs: Dict[str, float]) -> Dict[str, any]:
    """
    Test for Hardy-Weinberg equilibrium.
    
    Returns chi-squared statistic, p-value interpretation, and expected frequencies.
    """
    if len(allele_freqs) != 2:
        return {'error': 'HW test requires exactly 2 alleles'}
    
    alleles = list(allele_freqs.keys())
    p = allele_freqs[alleles[0]]
    q = allele_freqs[alleles[1]]
    
    # Expected frequencies under HW
    expected = {
        alleles[0] + alleles[0]: p * p,
        alleles[0] + alleles[1]: 2 * p * q,
        alleles[1] + alleles[1]: q * q,
    }
    
    # Chi-squared test
    chi_sq = 0
    for geno, exp_freq in expected.items():
        # Try both orderings
        obs_freq = genotype_freqs.get(geno, 0)
        if obs_freq == 0:
            obs_freq = genotype_freqs.get(geno[::-1], 0)
        
        if exp_freq > 0:
            chi_sq += (obs_freq - exp_freq) ** 2 / exp_freq
    
    # df = 1 for HW test
    # Critical values: 3.84 (p=0.05), 6.63 (p=0.01)
    if chi_sq < 3.84:
        interpretation = "IN EQUILIBRIUM (p > 0.05)"
    elif chi_sq < 6.63:
        interpretation = "DEVIATION (p < 0.05)"
    else:
        interpretation = "STRONG DEVIATION (p < 0.01)"
    
    return {
        'chi_squared': chi_sq,
        'interpretation': interpretation,
        'expected_frequencies': expected,
        'observed_frequencies': dict(genotype_freqs),
        'allele_frequencies': {'p': p, 'q': q},
    }


def calculate_inbreeding_coefficient(individuals: List[Individual]) -> float:
    """
    Calculate population inbreeding coefficient (F).
    
    F = 1 - (Ho / He)
    where Ho = observed heterozygosity, He = expected heterozygosity
    """
    if not individuals:
        return 0.0
    
    # Calculate across all genes
    Ho_values = []
    He_values = []
    
    for gene in set(g for ind in individuals for g in ind.genotypes.keys()):
        genotypes = [ind.genotypes.get(gene, '') for ind in individuals if gene in ind.genotypes]
        
        if not genotypes:
            continue
        
        # Observed heterozygosity
        n_het = sum(1 for g in genotypes if len(g) >= 2 and g[0] != g[1])
        Ho = n_het / len(genotypes)
        Ho_values.append(Ho)
        
        # Expected heterozygosity (from allele frequencies)
        allele_counts: Dict[str, int] = defaultdict(int)
        total = 0
        for g in genotypes:
            if len(g) >= 2:
                allele_counts[g[0]] += 1
                allele_counts[g[1]] += 1
                total += 2
        
        if total > 0:
            freqs = [c/total for c in allele_counts.values()]
            He = 1 - sum(f**2 for f in freqs)
            He_values.append(He)
    
    if not Ho_values or not He_values:
        return 0.0
    
    mean_Ho = np.mean(Ho_values)
    mean_He = np.mean(He_values)
    
    if mean_He == 0:
        return 0.0
    
    return 1 - (mean_Ho / mean_He)


def estimate_selection_coefficients(snapshots: List[PopulationSnapshot]) -> Dict[str, Dict[str, float]]:
    """
    Estimate selection coefficients from allele frequency changes.
    
    Uses: Δp = s * p * q * (p*h + q*(1-h)) / mean_fitness
    Simplified to: s ≈ Δp / (p * q) for additive selection
    """
    if len(snapshots) < 2:
        return {}
    
    selection = {}
    
    # Get genes from first snapshot
    genes = list(snapshots[0].allele_frequencies.keys())
    
    for gene in genes:
        selection[gene] = {}
        
        # Track allele frequency changes
        for allele in snapshots[0].allele_frequencies.get(gene, {}).keys():
            freq_changes = []
            
            for i in range(1, len(snapshots)):
                prev_freq = snapshots[i-1].allele_frequencies.get(gene, {}).get(allele, 0.5)
                curr_freq = snapshots[i].allele_frequencies.get(gene, {}).get(allele, 0.5)
                
                delta_p = curr_freq - prev_freq
                
                # Avoid division by zero
                pq = prev_freq * (1 - prev_freq)
                if pq > 0.01:  # Only if not near fixation
                    s_estimate = delta_p / pq
                    freq_changes.append(s_estimate)
            
            if freq_changes:
                selection[gene][allele] = {
                    'mean_s': np.mean(freq_changes),
                    'std_s': np.std(freq_changes),
                    'direction': 'positive' if np.mean(freq_changes) > 0 else 'negative',
                }
    
    return selection


def calculate_linkage_disequilibrium(individuals: List[Individual], 
                                      gene1: str, gene2: str) -> float:
    """
    Calculate linkage disequilibrium (D) between two genes.
    
    D = freq(AB) - freq(A)*freq(B)
    
    Standardized D' = D / D_max
    """
    # Get haplotype frequencies (assuming we can reconstruct from diploids)
    haplotypes: Dict[str, int] = defaultdict(int)
    total = 0
    
    for ind in individuals:
        g1 = ind.genotypes.get(gene1, '')
        g2 = ind.genotypes.get(gene2, '')
        
        if len(g1) >= 2 and len(g2) >= 2:
            # Create haplotypes (simplified - assumes linkage phase)
            haplotypes[g1[0] + g2[0]] += 1
            haplotypes[g1[1] + g2[1]] += 1
            total += 2
    
    if total == 0:
        return 0.0
    
    # Allele frequencies
    freq_A = sum(c for h, c in haplotypes.items() if h[0].isupper()) / total
    freq_B = sum(c for h, c in haplotypes.items() if h[1].isupper()) / total
    
    # Haplotype frequency for AB
    freq_AB = sum(c for h, c in haplotypes.items() 
                  if h[0].isupper() and h[1].isupper()) / total
    
    # D = freq(AB) - freq(A)*freq(B)
    D = freq_AB - freq_A * freq_B
    
    # Standardize
    if D > 0:
        D_max = min(freq_A * (1-freq_B), (1-freq_A) * freq_B)
    else:
        D_max = min(freq_A * freq_B, (1-freq_A) * (1-freq_B))
    
    if D_max > 0:
        return D / D_max
    return 0.0


def build_pedigree(individuals: List[Individual]) -> Dict:
    """
    Build pedigree structure for phylogenetic analysis.
    """
    pedigree = {
        'individuals': {},
        'generations': defaultdict(list),
        'lineages': defaultdict(list),
    }
    
    for ind in individuals:
        pedigree['individuals'][ind.creature_id] = {
            'id': ind.creature_id,
            'species': ind.species,
            'generation': ind.generation,
            'parent_id': ind.parent_id,
            'mate_id': ind.mate_id,
            'offspring': ind.offspring_ids,
            'genome_id': ind.genome_id,
            'birth_step': ind.birth_step,
            'death_step': ind.death_step,
        }
        
        pedigree['generations'][ind.generation].append(ind.creature_id)
        
        # Track lineages
        lineage_id = ind.genome_id[:4] if ind.genome_id else 'unknown'
        pedigree['lineages'][lineage_id].append(ind.creature_id)
    
    return pedigree


def calculate_genetic_diversity(individuals: List[Individual]) -> Dict[str, float]:
    """
    Calculate various genetic diversity metrics.
    """
    if not individuals:
        return {}
    
    # Number of alleles per gene
    alleles_per_gene: Dict[str, Set[str]] = defaultdict(set)
    for ind in individuals:
        for gene, geno in ind.genotypes.items():
            if len(geno) >= 2:
                alleles_per_gene[gene].add(geno[0])
                alleles_per_gene[gene].add(geno[1])
    
    # Shannon diversity index per gene
    shannon_per_gene = {}
    for gene, alleles in alleles_per_gene.items():
        # Count alleles
        counts = defaultdict(int)
        total = 0
        for ind in individuals:
            geno = ind.genotypes.get(gene, '')
            if len(geno) >= 2:
                counts[geno[0]] += 1
                counts[geno[1]] += 1
                total += 2
        
        if total > 0:
            freqs = [c/total for c in counts.values()]
            H = -sum(f * np.log(f) for f in freqs if f > 0)
            shannon_per_gene[gene] = H
    
    return {
        'n_genes_tracked': len(alleles_per_gene),
        'mean_alleles_per_gene': np.mean([len(a) for a in alleles_per_gene.values()]) if alleles_per_gene else 0,
        'max_alleles_per_gene': max(len(a) for a in alleles_per_gene.values()) if alleles_per_gene else 0,
        'mean_shannon_diversity': np.mean(list(shannon_per_gene.values())) if shannon_per_gene else 0,
        'alleles_per_gene': {g: len(a) for g, a in alleles_per_gene.items()},
        'shannon_per_gene': shannon_per_gene,
    }


def detect_selection_vs_drift(snapshots: List[PopulationSnapshot], 
                               threshold: float = 0.1) -> Dict[str, str]:
    """
    Detect whether allele frequency changes are due to selection or drift.
    
    Uses variance in allele frequency change - drift should be random,
    selection should show consistent direction.
    """
    if len(snapshots) < 3:
        return {}
    
    results = {}
    genes = list(snapshots[0].allele_frequencies.keys())
    
    for gene in genes:
        alleles = list(snapshots[0].allele_frequencies.get(gene, {}).keys())
        if not alleles:
            continue
        
        allele = alleles[0]  # Track first allele
        
        # Get frequency trajectory
        freqs = [s.allele_frequencies.get(gene, {}).get(allele, 0.5) 
                 for s in snapshots]
        
        if len(freqs) < 3:
            continue
        
        # Calculate changes
        changes = np.diff(freqs)
        
        if len(changes) == 0:
            continue
        
        # Metrics
        mean_change = np.mean(changes)
        std_change = np.std(changes)
        
        # Consistent direction suggests selection
        same_sign = np.sum(np.sign(changes) == np.sign(mean_change)) / len(changes)
        
        # Net change
        net_change = freqs[-1] - freqs[0]
        
        if abs(net_change) < 0.05:
            results[gene] = "NEUTRAL (stable)"
        elif same_sign > 0.7 and abs(mean_change) > threshold / 10:
            direction = "positive" if mean_change > 0 else "negative"
            results[gene] = f"SELECTION ({direction})"
        elif std_change > abs(mean_change) * 2:
            results[gene] = "DRIFT (high variance)"
        else:
            results[gene] = "MIXED (drift + weak selection)"
    
    return results


# ============================================================================
# REPORTING
# ============================================================================

def generate_report(individuals: List[Individual], 
                    snapshots: List[PopulationSnapshot],
                    output_path: str):
    """Generate comprehensive genetics report."""
    
    lines = []
    lines.append("=" * 70)
    lines.append("HERBIE WORLD - GENETICS ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Summary
    lines.append("POPULATION SUMMARY")
    lines.append("-" * 40)
    lines.append(f"Total individuals tracked: {len(individuals)}")
    
    species_counts = defaultdict(int)
    for ind in individuals:
        species_counts[ind.species] += 1
    for species, count in sorted(species_counts.items()):
        lines.append(f"  {species}: {count}")
    
    max_gen = max(ind.generation for ind in individuals) if individuals else 0
    lines.append(f"Maximum generation: {max_gen}")
    lines.append(f"Population snapshots: {len(snapshots)}")
    lines.append("")
    
    # Genetic diversity
    lines.append("GENETIC DIVERSITY")
    lines.append("-" * 40)
    diversity = calculate_genetic_diversity(individuals)
    lines.append(f"Genes tracked: {diversity.get('n_genes_tracked', 0)}")
    lines.append(f"Mean alleles per gene: {diversity.get('mean_alleles_per_gene', 0):.2f}")
    lines.append(f"Mean Shannon diversity: {diversity.get('mean_shannon_diversity', 0):.3f}")
    lines.append("")
    
    # Inbreeding
    lines.append("INBREEDING ANALYSIS")
    lines.append("-" * 40)
    F = calculate_inbreeding_coefficient(individuals)
    lines.append(f"Population inbreeding coefficient (F): {F:.4f}")
    if F < 0.05:
        lines.append("  Interpretation: Low inbreeding (random mating)")
    elif F < 0.15:
        lines.append("  Interpretation: Moderate inbreeding")
    else:
        lines.append("  Interpretation: High inbreeding (population structure)")
    lines.append("")
    
    # Hardy-Weinberg tests
    if snapshots:
        lines.append("HARDY-WEINBERG EQUILIBRIUM TESTS")
        lines.append("-" * 40)
        latest = snapshots[-1]
        
        for gene in list(latest.genotype_frequencies.keys())[:5]:  # Top 5 genes
            geno_freq = latest.genotype_frequencies.get(gene, {})
            allele_freq = latest.allele_frequencies.get(gene, {})
            
            if len(allele_freq) == 2:
                hw = hardy_weinberg_test(geno_freq, allele_freq)
                lines.append(f"\n  {gene}:")
                lines.append(f"    Chi-squared: {hw.get('chi_squared', 0):.3f}")
                lines.append(f"    Status: {hw.get('interpretation', 'N/A')}")
                
                if 'allele_frequencies' in hw:
                    af = hw['allele_frequencies']
                    lines.append(f"    Allele frequencies: p={af['p']:.3f}, q={af['q']:.3f}")
        lines.append("")
    
    # Selection analysis
    if len(snapshots) >= 2:
        lines.append("SELECTION ANALYSIS")
        lines.append("-" * 40)
        
        selection = estimate_selection_coefficients(snapshots)
        for gene, allele_data in list(selection.items())[:5]:
            lines.append(f"\n  {gene}:")
            for allele, data in allele_data.items():
                s = data.get('mean_s', 0)
                direction = data.get('direction', 'unknown')
                lines.append(f"    {allele}: s = {s:+.4f} ({direction})")
        
        lines.append("\n  Selection vs Drift Detection:")
        sel_drift = detect_selection_vs_drift(snapshots)
        for gene, status in list(sel_drift.items())[:10]:
            lines.append(f"    {gene}: {status}")
        lines.append("")
    
    # Linkage analysis
    lines.append("LINKAGE DISEQUILIBRIUM")
    lines.append("-" * 40)
    genes = list(set(g for ind in individuals for g in ind.genotypes.keys()))
    if len(genes) >= 2:
        # Test a few gene pairs
        for i in range(min(3, len(genes)-1)):
            for j in range(i+1, min(i+3, len(genes))):
                D_prime = calculate_linkage_disequilibrium(individuals, genes[i], genes[j])
                lines.append(f"  {genes[i]} - {genes[j]}: D' = {D_prime:.3f}")
    else:
        lines.append("  Insufficient genes for LD analysis")
    lines.append("")
    
    # Pedigree summary
    lines.append("PEDIGREE ANALYSIS")
    lines.append("-" * 40)
    pedigree = build_pedigree(individuals)
    n_generations = len(pedigree['generations'])
    n_lineages = len(pedigree['lineages'])
    lines.append(f"Generations: {n_generations}")
    lines.append(f"Distinct lineages: {n_lineages}")
    
    # Largest lineages
    lineage_sizes = [(lin, len(members)) for lin, members in pedigree['lineages'].items()]
    lineage_sizes.sort(key=lambda x: -x[1])
    lines.append("\nLargest lineages:")
    for lin, size in lineage_sizes[:5]:
        lines.append(f"  Lineage {lin}: {size} individuals")
    lines.append("")
    
    # Allele frequency trajectories
    if snapshots:
        lines.append("ALLELE FREQUENCY TRAJECTORIES")
        lines.append("-" * 40)
        
        # Track major changes
        if len(snapshots) >= 2:
            first = snapshots[0]
            last = snapshots[-1]
            
            changes = []
            for gene in first.allele_frequencies:
                for allele in first.allele_frequencies[gene]:
                    freq_start = first.allele_frequencies[gene].get(allele, 0)
                    freq_end = last.allele_frequencies.get(gene, {}).get(allele, 0)
                    delta = freq_end - freq_start
                    changes.append((gene, allele, freq_start, freq_end, delta))
            
            # Sort by absolute change
            changes.sort(key=lambda x: -abs(x[4]))
            
            lines.append("Largest allele frequency changes:")
            for gene, allele, start, end, delta in changes[:10]:
                lines.append(f"  {gene} [{allele}]: {start:.3f} → {end:.3f} (Δ = {delta:+.3f})")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Report saved to: {output_path}")
    return '\n'.join(lines)


def generate_plots(snapshots: List[PopulationSnapshot], 
                   individuals: List[Individual],
                   output_path: str):
    """Generate visualization plots."""
    
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    if not snapshots:
        print("No data for plots")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Genetics Analysis', fontsize=14, fontweight='bold')
    
    steps = [s.step for s in snapshots]
    
    # 1. Population size over time
    ax = axes[0, 0]
    pop_sizes = [s.n_individuals for s in snapshots]
    ax.plot(steps, pop_sizes, 'b-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Population Size')
    ax.set_title('Population Over Time')
    ax.grid(True, alpha=0.3)
    
    # 2. Allele frequencies over time
    ax = axes[0, 1]
    genes = list(snapshots[0].allele_frequencies.keys())[:3]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    color_idx = 0
    for gene in genes:
        alleles = list(snapshots[0].allele_frequencies.get(gene, {}).keys())
        for allele in alleles[:2]:
            freqs = [s.allele_frequencies.get(gene, {}).get(allele, 0) for s in snapshots]
            ax.plot(steps, freqs, color=colors[color_idx], label=f"{gene}:{allele}")
            color_idx += 1
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Allele Frequency')
    ax.set_title('Allele Frequency Dynamics')
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 3. Heterozygosity over time
    ax = axes[0, 2]
    for gene in genes[:3]:
        het = [s.heterozygosity.get(gene, 0) for s in snapshots]
        ax.plot(steps, het, label=gene)
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Heterozygosity')
    ax.set_title('Observed Heterozygosity')
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # 4. Generation distribution
    ax = axes[1, 0]
    generations = [ind.generation for ind in individuals]
    if generations:
        ax.hist(generations, bins=max(1, max(generations)), color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Count')
    ax.set_title('Generation Distribution')
    ax.grid(True, alpha=0.3)
    
    # 5. Hardy-Weinberg comparison (latest snapshot)
    ax = axes[1, 1]
    if snapshots:
        latest = snapshots[-1]
        genes_for_hw = [g for g in latest.genotype_frequencies 
                        if len(latest.allele_frequencies.get(g, {})) == 2][:3]
        
        x_pos = np.arange(len(genes_for_hw))
        width = 0.35
        
        observed_het = []
        expected_het = []
        
        for gene in genes_for_hw:
            af = latest.allele_frequencies.get(gene, {})
            if len(af) == 2:
                p = list(af.values())[0]
                exp_het = 2 * p * (1-p)
                obs_het = latest.heterozygosity.get(gene, 0)
                expected_het.append(exp_het)
                observed_het.append(obs_het)
        
        if observed_het:
            ax.bar(x_pos - width/2, observed_het, width, label='Observed', color='blue', alpha=0.7)
            ax.bar(x_pos + width/2, expected_het, width, label='Expected (HW)', color='orange', alpha=0.7)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(genes_for_hw, rotation=45, ha='right')
            ax.legend()
    
    ax.set_ylabel('Heterozygosity')
    ax.set_title('Hardy-Weinberg: Observed vs Expected')
    ax.grid(True, alpha=0.3)
    
    # 6. Mean phenotypes over time
    ax = axes[1, 2]
    phenotype_genes = list(snapshots[0].mean_phenotypes.keys())[:4]
    
    for gene in phenotype_genes:
        values = [s.mean_phenotypes.get(gene, 1.0) for s in snapshots]
        ax.plot(steps, values, label=gene[:15])
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Phenotype')
    ax.set_title('Phenotype Evolution')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {output_path}")


def export_allele_frequencies(snapshots: List[PopulationSnapshot], output_path: str):
    """Export allele frequency time series to CSV."""
    
    if not snapshots:
        return
    
    # Collect all genes and alleles
    all_genes = set()
    all_alleles: Dict[str, Set[str]] = defaultdict(set)
    
    for snap in snapshots:
        for gene, alleles in snap.allele_frequencies.items():
            all_genes.add(gene)
            all_alleles[gene].update(alleles.keys())
    
    # Build header
    header = ['step', 'n_individuals']
    for gene in sorted(all_genes):
        for allele in sorted(all_alleles[gene]):
            header.append(f"{gene}_{allele}")
    
    # Build rows
    rows = []
    for snap in snapshots:
        row = [snap.step, snap.n_individuals]
        for gene in sorted(all_genes):
            for allele in sorted(all_alleles[gene]):
                freq = snap.allele_frequencies.get(gene, {}).get(allele, 0)
                row.append(f"{freq:.4f}")
        rows.append(','.join(str(x) for x in row))
    
    with open(output_path, 'w') as f:
        f.write(','.join(header) + '\n')
        f.write('\n'.join(rows))
    
    print(f"Allele frequencies saved to: {output_path}")


def export_pedigree(individuals: List[Individual], output_path: str):
    """Export pedigree to JSON for visualization."""
    
    pedigree = build_pedigree(individuals)
    
    with open(output_path, 'w') as f:
        json.dump(pedigree, f, indent=2, default=str)
    
    print(f"Pedigree saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Find event log
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        # Default path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(script_dir, 'data', 'event_log.jsonl')
    
    print("=" * 60)
    print("HERBIE WORLD - GENETICS ANALYZER")
    print("=" * 60)
    print(f"\nLoading data from: {log_path}")
    
    # Load data
    individuals, snapshots = load_genetic_data(log_path)
    
    if not individuals:
        print("\nNo genetic data found in log.")
        print("Run a simulation first, or check that genetic logging is enabled.")
        
        # Create sample analysis with synthetic data
        print("\nGenerating sample analysis with synthetic population...")
        individuals, snapshots = generate_synthetic_data()
    
    print(f"\nLoaded {len(individuals)} individuals")
    print(f"Created {len(snapshots)} population snapshots")
    
    # Output directory
    output_dir = os.path.dirname(log_path) if log_path else '.'
    
    # Generate outputs
    print("\nGenerating analysis...")
    
    report_path = os.path.join(output_dir, 'genetics_report.txt')
    generate_report(individuals, snapshots, report_path)
    
    plots_path = os.path.join(output_dir, 'genetics_plots.png')
    generate_plots(snapshots, individuals, plots_path)
    
    csv_path = os.path.join(output_dir, 'allele_frequencies.csv')
    export_allele_frequencies(snapshots, csv_path)
    
    pedigree_path = os.path.join(output_dir, 'pedigree.json')
    export_pedigree(individuals, pedigree_path)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - {report_path}")
    print(f"  - {plots_path}")
    print(f"  - {csv_path}")
    print(f"  - {pedigree_path}")


def generate_synthetic_data() -> Tuple[List[Individual], List[PopulationSnapshot]]:
    """Generate synthetic population for demo."""
    
    individuals = []
    
    # Create founding population
    for i in range(20):
        ind = Individual(
            creature_id=f"founder_{i}",
            species="Herbie",
            generation=0,
            birth_step=0,
            death_step=None if i < 10 else 1000,
            parent_id=None,
            mate_id=None,
            offspring_ids=[],
            genome_id=f"G{i:04d}",
            maternal_genome_id=None,
            paternal_genome_id=None,
            genotypes={
                'speed_base': np.random.choice(['SS', 'Ss', 'ss'], p=[0.25, 0.5, 0.25]),
                'metabolism_base': np.random.choice(['MM', 'Mm', 'mm'], p=[0.25, 0.5, 0.25]),
                'sociability': np.random.choice(['CC', 'Cc', 'cc'], p=[0.25, 0.5, 0.25]),
            },
            phenotypes={
                'speed_base': np.random.uniform(0.8, 1.2),
                'metabolism_base': np.random.uniform(0.8, 1.2),
            },
            traits={}
        )
        individuals.append(ind)
    
    # Create offspring over generations
    for gen in range(1, 5):
        for i in range(15):
            parent = np.random.choice([ind for ind in individuals if ind.generation == gen-1])
            
            ind = Individual(
                creature_id=f"gen{gen}_{i}",
                species="Herbie",
                generation=gen,
                birth_step=gen * 500,
                death_step=None if np.random.random() > 0.3 else (gen+1) * 500,
                parent_id=parent.creature_id,
                mate_id=None,
                offspring_ids=[],
                genome_id=f"G{gen}{i:04d}",
                maternal_genome_id=parent.genome_id,
                paternal_genome_id=None,
                genotypes={
                    'speed_base': np.random.choice(['SS', 'Ss', 'ss'], p=[0.3, 0.5, 0.2]),  # Selection for speed
                    'metabolism_base': np.random.choice(['MM', 'Mm', 'mm'], p=[0.25, 0.5, 0.25]),
                    'sociability': np.random.choice(['CC', 'Cc', 'cc'], p=[0.25, 0.5, 0.25]),
                },
                phenotypes={
                    'speed_base': np.random.uniform(0.9, 1.3),  # Trending faster
                    'metabolism_base': np.random.uniform(0.8, 1.2),
                },
                traits={}
            )
            individuals.append(ind)
            parent.offspring_ids.append(ind.creature_id)
    
    # Create snapshots
    snapshots = []
    for step in range(0, 2500, 500):
        living = [ind for ind in individuals 
                 if ind.birth_step <= step and 
                 (ind.death_step is None or ind.death_step > step)]
        if living:
            snapshots.append(create_population_snapshot(living, step))
    
    return individuals, snapshots


if __name__ == '__main__':
    main()
