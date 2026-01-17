"""
Mendelian Genetics System - Discrete alleles with dominance relationships.

This implements classical genetics with:
- Discrete alleles (not just continuous drift)
- Dominant/recessive relationships
- Codominance and incomplete dominance
- Linked genes (chromosomes)
- Crossing over during meiosis
- Hardy-Weinberg tracking
- Punnett square predictions

Each creature has a diploid genome (two copies of each gene).
Gametes are haploid (one copy) produced via meiosis with recombination.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import numpy as np
import uuid


class DominanceType(Enum):
    """How alleles interact in heterozygotes."""
    COMPLETE = 'complete'       # Dominant masks recessive
    INCOMPLETE = 'incomplete'   # Blend (intermediate phenotype)
    CODOMINANT = 'codominant'   # Both expressed


@dataclass
class Allele:
    """A single allele variant."""
    id: str                     # Unique identifier (e.g., 'A', 'a', 'A1')
    value: float                # Phenotypic effect (0.0 - 1.0 scale)
    dominance_weight: float     # Higher = more dominant (0.0 - 1.0)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Allele):
            return self.id == other.id
        return False


@dataclass 
class Gene:
    """A gene with possible alleles and dominance type."""
    name: str
    chromosome: int             # Which linkage group (0-3)
    position: float             # Position on chromosome (0.0 - 1.0) for crossing over
    dominance_type: DominanceType
    alleles: Dict[str, Allele]  # id -> Allele
    
    def get_phenotype(self, allele1: Allele, allele2: Allele) -> float:
        """Calculate phenotype from diploid genotype."""
        if self.dominance_type == DominanceType.COMPLETE:
            # Higher dominance weight wins
            if allele1.dominance_weight > allele2.dominance_weight:
                return allele1.value
            elif allele2.dominance_weight > allele1.dominance_weight:
                return allele2.value
            else:
                return (allele1.value + allele2.value) / 2
        
        elif self.dominance_type == DominanceType.INCOMPLETE:
            # Weighted average by dominance
            total_weight = allele1.dominance_weight + allele2.dominance_weight
            if total_weight == 0:
                return (allele1.value + allele2.value) / 2
            w1 = allele1.dominance_weight / total_weight
            w2 = allele2.dominance_weight / total_weight
            return allele1.value * w1 + allele2.value * w2
        
        else:  # CODOMINANT
            # Both fully expressed - return average for continuous traits
            return (allele1.value + allele2.value) / 2


# ============================================================================
# GENE DEFINITIONS
# ============================================================================

# Define the genes that control creature traits
GENE_DEFINITIONS: Dict[str, Gene] = {}

def _init_genes():
    """Initialize all gene definitions."""
    global GENE_DEFINITIONS
    
    # CHROMOSOME 0: Metabolism genes
    GENE_DEFINITIONS['metabolism_base'] = Gene(
        name='metabolism_base',
        chromosome=0,
        position=0.1,
        dominance_type=DominanceType.INCOMPLETE,
        alleles={
            'M': Allele('M', 1.2, 0.7),    # Fast metabolism (slightly dominant)
            'm': Allele('m', 0.8, 0.3),    # Slow metabolism
            'M+': Allele('M+', 1.4, 0.5),  # Ultra-fast (rare)
        }
    )
    
    GENE_DEFINITIONS['energy_storage'] = Gene(
        name='energy_storage',
        chromosome=0,
        position=0.4,
        dominance_type=DominanceType.COMPLETE,
        alleles={
            'E': Allele('E', 1.3, 0.8),    # Efficient storage (dominant)
            'e': Allele('e', 0.9, 0.2),    # Poor storage (recessive)
        }
    )
    
    GENE_DEFINITIONS['hunger_sensitivity'] = Gene(
        name='hunger_sensitivity',
        chromosome=0,
        position=0.7,
        dominance_type=DominanceType.INCOMPLETE,
        alleles={
            'H': Allele('H', 1.2, 0.5),    # High sensitivity
            'h': Allele('h', 0.8, 0.5),    # Low sensitivity
        }
    )
    
    # CHROMOSOME 1: Locomotion genes
    GENE_DEFINITIONS['speed_base'] = Gene(
        name='speed_base',
        chromosome=1,
        position=0.2,
        dominance_type=DominanceType.INCOMPLETE,
        alleles={
            'S': Allele('S', 1.3, 0.6),    # Fast
            's': Allele('s', 0.7, 0.4),    # Slow
            'S+': Allele('S+', 1.5, 0.5),  # Sprint gene (rare)
        }
    )
    
    GENE_DEFINITIONS['endurance'] = Gene(
        name='endurance',
        chromosome=1,
        position=0.5,
        dominance_type=DominanceType.COMPLETE,
        alleles={
            'N': Allele('N', 1.2, 0.7),    # High endurance (dominant)
            'n': Allele('n', 0.85, 0.3),   # Low endurance
        }
    )
    
    GENE_DEFINITIONS['agility'] = Gene(
        name='agility',
        chromosome=1,
        position=0.8,
        dominance_type=DominanceType.CODOMINANT,
        alleles={
            'A': Allele('A', 1.25, 0.5),   # Agile
            'a': Allele('a', 0.8, 0.5),    # Clumsy
        }
    )
    
    # CHROMOSOME 2: Sensory genes
    GENE_DEFINITIONS['vision_range'] = Gene(
        name='vision_range',
        chromosome=2,
        position=0.15,
        dominance_type=DominanceType.INCOMPLETE,
        alleles={
            'V': Allele('V', 1.3, 0.6),    # Far sight
            'v': Allele('v', 0.8, 0.4),    # Near sight
        }
    )
    
    GENE_DEFINITIONS['smell_sensitivity'] = Gene(
        name='smell_sensitivity',
        chromosome=2,
        position=0.45,
        dominance_type=DominanceType.COMPLETE,
        alleles={
            'O': Allele('O', 1.4, 0.8),    # Keen smell (dominant)
            'o': Allele('o', 0.7, 0.2),    # Poor smell
        }
    )
    
    GENE_DEFINITIONS['hearing_range'] = Gene(
        name='hearing_range',
        chromosome=2,
        position=0.75,
        dominance_type=DominanceType.INCOMPLETE,
        alleles={
            'R': Allele('R', 1.2, 0.5),    # Good hearing
            'r': Allele('r', 0.85, 0.5),   # Poor hearing
        }
    )
    
    # CHROMOSOME 3: Social/behavioral genes
    GENE_DEFINITIONS['sociability'] = Gene(
        name='sociability',
        chromosome=3,
        position=0.2,
        dominance_type=DominanceType.INCOMPLETE,
        alleles={
            'C': Allele('C', 1.3, 0.5),    # Social
            'c': Allele('c', 0.7, 0.5),    # Solitary
        }
    )
    
    GENE_DEFINITIONS['aggression'] = Gene(
        name='aggression',
        chromosome=3,
        position=0.5,
        dominance_type=DominanceType.INCOMPLETE,
        alleles={
            'G': Allele('G', 1.4, 0.55),   # Aggressive (slightly dominant)
            'g': Allele('g', 0.6, 0.45),   # Passive
        }
    )
    
    GENE_DEFINITIONS['parental_care'] = Gene(
        name='parental_care',
        chromosome=3,
        position=0.8,
        dominance_type=DominanceType.COMPLETE,
        alleles={
            'P': Allele('P', 1.3, 0.75),   # Nurturing (dominant)
            'p': Allele('p', 0.8, 0.25),   # Neglectful
        }
    )

_init_genes()


@dataclass
class Chromosome:
    """One copy of a chromosome with alleles at each locus."""
    genes: Dict[str, Allele] = field(default_factory=dict)  # gene_name -> allele
    
    def copy(self) -> 'Chromosome':
        return Chromosome(genes=self.genes.copy())


@dataclass
class Genome:
    """
    Diploid genome with maternal and paternal chromosome sets.
    
    Each creature has two copies of each chromosome (one from each parent).
    """
    maternal: Dict[int, Chromosome] = field(default_factory=dict)  # chrom_num -> Chromosome
    paternal: Dict[int, Chromosome] = field(default_factory=dict)
    
    # Unique genome ID for tracking lineage
    genome_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Parent genome IDs
    maternal_genome_id: Optional[str] = None
    paternal_genome_id: Optional[str] = None
    
    def get_genotype(self, gene_name: str) -> Tuple[Allele, Allele]:
        """Get both alleles for a gene."""
        gene = GENE_DEFINITIONS.get(gene_name)
        if not gene:
            raise ValueError(f"Unknown gene: {gene_name}")
        
        chrom = gene.chromosome
        mat_allele = self.maternal.get(chrom, Chromosome()).genes.get(gene_name)
        pat_allele = self.paternal.get(chrom, Chromosome()).genes.get(gene_name)
        
        # Default to first allele if not set
        if mat_allele is None:
            mat_allele = list(gene.alleles.values())[0]
        if pat_allele is None:
            pat_allele = list(gene.alleles.values())[0]
            
        return (mat_allele, pat_allele)
    
    def get_phenotype(self, gene_name: str) -> float:
        """Get expressed phenotype for a gene."""
        gene = GENE_DEFINITIONS.get(gene_name)
        if not gene:
            return 1.0
        
        allele1, allele2 = self.get_genotype(gene_name)
        return gene.get_phenotype(allele1, allele2)
    
    def get_genotype_string(self, gene_name: str) -> str:
        """Get genotype as string (e.g., 'Aa', 'MM', 'ss')."""
        allele1, allele2 = self.get_genotype(gene_name)
        # Sort by dominance (dominant first)
        if allele1.dominance_weight >= allele2.dominance_weight:
            return f"{allele1.id}{allele2.id}"
        return f"{allele2.id}{allele1.id}"
    
    def is_homozygous(self, gene_name: str) -> bool:
        """Check if homozygous for a gene."""
        a1, a2 = self.get_genotype(gene_name)
        return a1.id == a2.id
    
    def is_heterozygous(self, gene_name: str) -> bool:
        """Check if heterozygous for a gene."""
        return not self.is_homozygous(gene_name)
    
    def get_all_phenotypes(self) -> Dict[str, float]:
        """Get all phenotypes."""
        return {name: self.get_phenotype(name) for name in GENE_DEFINITIONS}
    
    def get_full_genotype(self) -> Dict[str, str]:
        """Get all genotypes as strings."""
        return {name: self.get_genotype_string(name) for name in GENE_DEFINITIONS}
    
    def to_dict(self) -> dict:
        """Serialize genome."""
        return {
            'genome_id': self.genome_id,
            'maternal_genome_id': self.maternal_genome_id,
            'paternal_genome_id': self.paternal_genome_id,
            'maternal': {
                chrom: {g: a.id for g, a in c.genes.items()}
                for chrom, c in self.maternal.items()
            },
            'paternal': {
                chrom: {g: a.id for g, a in c.genes.items()}
                for chrom, c in self.paternal.items()
            }
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Genome':
        """Deserialize genome."""
        genome = cls()
        genome.genome_id = d.get('genome_id', str(uuid.uuid4())[:8])
        genome.maternal_genome_id = d.get('maternal_genome_id')
        genome.paternal_genome_id = d.get('paternal_genome_id')
        
        for chrom_str, genes in d.get('maternal', {}).items():
            chrom = int(chrom_str)
            genome.maternal[chrom] = Chromosome()
            for gene_name, allele_id in genes.items():
                gene = GENE_DEFINITIONS.get(gene_name)
                if gene and allele_id in gene.alleles:
                    genome.maternal[chrom].genes[gene_name] = gene.alleles[allele_id]
        
        for chrom_str, genes in d.get('paternal', {}).items():
            chrom = int(chrom_str)
            genome.paternal[chrom] = Chromosome()
            for gene_name, allele_id in genes.items():
                gene = GENE_DEFINITIONS.get(gene_name)
                if gene and allele_id in gene.alleles:
                    genome.paternal[chrom].genes[gene_name] = gene.alleles[allele_id]
        
        return genome


# ============================================================================
# MEIOSIS AND REPRODUCTION
# ============================================================================

CROSSOVER_RATE = 0.3  # Probability of crossover per chromosome
MUTATION_RATE_PER_GENE = 0.02  # Probability of new mutation per gene


def create_gamete(genome: Genome, mutation_rate_mult: float = 1.0) -> Dict[int, Chromosome]:
    """
    Produce a haploid gamete via meiosis.
    
    Includes:
    - Random segregation of homologs
    - Crossing over between homologs
    - Rare new mutations
    """
    gamete: Dict[int, Chromosome] = {}
    
    for chrom_num in range(4):  # 4 chromosomes
        mat = genome.maternal.get(chrom_num, Chromosome())
        pat = genome.paternal.get(chrom_num, Chromosome())
        
        # Decide which parent's chromosome to start with
        use_maternal = np.random.random() < 0.5
        
        # Maybe do crossing over
        if np.random.random() < CROSSOVER_RATE:
            # Crossover point
            crossover_point = np.random.random()
            
            new_chrom = Chromosome()
            for gene_name, gene in GENE_DEFINITIONS.items():
                if gene.chromosome != chrom_num:
                    continue
                
                # Before crossover point: use initial parent
                # After crossover point: use other parent
                if gene.position < crossover_point:
                    source = mat if use_maternal else pat
                else:
                    source = pat if use_maternal else mat
                
                if gene_name in source.genes:
                    new_chrom.genes[gene_name] = source.genes[gene_name]
            
            gamete[chrom_num] = new_chrom
        else:
            # No crossover - use one parent's chromosome
            source = mat if use_maternal else pat
            gamete[chrom_num] = source.copy()
        
        # Apply rare mutations
        effective_mut_rate = MUTATION_RATE_PER_GENE * mutation_rate_mult
        for gene_name, gene in GENE_DEFINITIONS.items():
            if gene.chromosome != chrom_num:
                continue
            
            if np.random.random() < effective_mut_rate:
                # Pick a random allele (could be same or different)
                new_allele = np.random.choice(list(gene.alleles.values()))
                gamete[chrom_num].genes[gene_name] = new_allele
    
    return gamete


def reproduce(parent1: Genome, parent2: Genome, 
              mutation_rate_mult: float = 1.0) -> Genome:
    """
    Sexual reproduction between two genomes.
    
    Each parent contributes a haploid gamete (via meiosis).
    Results in a new diploid genome.
    """
    gamete1 = create_gamete(parent1, mutation_rate_mult)
    gamete2 = create_gamete(parent2, mutation_rate_mult)
    
    child = Genome()
    child.maternal = gamete1
    child.paternal = gamete2
    child.maternal_genome_id = parent1.genome_id
    child.paternal_genome_id = parent2.genome_id
    
    return child


def create_random_genome(allele_frequencies: Dict[str, Dict[str, float]] = None) -> Genome:
    """
    Create a random genome, optionally with specified allele frequencies.
    
    allele_frequencies: {gene_name: {allele_id: frequency}}
    If not specified, uses uniform frequencies.
    """
    genome = Genome()
    
    for chrom_num in range(4):
        genome.maternal[chrom_num] = Chromosome()
        genome.paternal[chrom_num] = Chromosome()
    
    for gene_name, gene in GENE_DEFINITIONS.items():
        chrom = gene.chromosome
        
        # Get frequencies
        if allele_frequencies and gene_name in allele_frequencies:
            freqs = allele_frequencies[gene_name]
            allele_ids = list(freqs.keys())
            probs = [freqs[a] for a in allele_ids]
            # Normalize
            total = sum(probs)
            probs = [p/total for p in probs]
        else:
            # Uniform frequencies
            allele_ids = list(gene.alleles.keys())
            probs = [1.0/len(allele_ids)] * len(allele_ids)
        
        # Sample alleles
        mat_id = np.random.choice(allele_ids, p=probs)
        pat_id = np.random.choice(allele_ids, p=probs)
        
        genome.maternal[chrom].genes[gene_name] = gene.alleles[mat_id]
        genome.paternal[chrom].genes[gene_name] = gene.alleles[pat_id]
    
    return genome


# ============================================================================
# POPULATION GENETICS
# ============================================================================

def calculate_allele_frequencies(genomes: List[Genome]) -> Dict[str, Dict[str, float]]:
    """
    Calculate allele frequencies across a population.
    
    Returns: {gene_name: {allele_id: frequency}}
    """
    if not genomes:
        return {}
    
    frequencies: Dict[str, Dict[str, int]] = {}
    
    for gene_name in GENE_DEFINITIONS:
        frequencies[gene_name] = {}
    
    # Count alleles
    for genome in genomes:
        for gene_name in GENE_DEFINITIONS:
            a1, a2 = genome.get_genotype(gene_name)
            frequencies[gene_name][a1.id] = frequencies[gene_name].get(a1.id, 0) + 1
            frequencies[gene_name][a2.id] = frequencies[gene_name].get(a2.id, 0) + 1
    
    # Convert to frequencies
    n_alleles = len(genomes) * 2  # Diploid
    result = {}
    for gene_name, counts in frequencies.items():
        result[gene_name] = {a: c / n_alleles for a, c in counts.items()}
    
    return result


def calculate_genotype_frequencies(genomes: List[Genome]) -> Dict[str, Dict[str, float]]:
    """
    Calculate genotype frequencies.
    
    Returns: {gene_name: {genotype_string: frequency}}
    """
    if not genomes:
        return {}
    
    frequencies: Dict[str, Dict[str, int]] = {}
    
    for gene_name in GENE_DEFINITIONS:
        frequencies[gene_name] = {}
    
    for genome in genomes:
        for gene_name in GENE_DEFINITIONS:
            geno = genome.get_genotype_string(gene_name)
            frequencies[gene_name][geno] = frequencies[gene_name].get(geno, 0) + 1
    
    n = len(genomes)
    return {
        gene: {g: c/n for g, c in genos.items()}
        for gene, genos in frequencies.items()
    }


def hardy_weinberg_expected(p: float) -> Tuple[float, float, float]:
    """
    Calculate expected Hardy-Weinberg genotype frequencies.
    
    For allele frequency p (dominant) and q = 1-p (recessive):
    Returns (p², 2pq, q²) = (AA, Aa, aa)
    """
    q = 1 - p
    return (p*p, 2*p*q, q*q)


def hardy_weinberg_chi_squared(observed: Dict[str, float], 
                                allele_freq: float) -> Tuple[float, bool]:
    """
    Chi-squared test for Hardy-Weinberg equilibrium.
    
    Returns: (chi_squared_statistic, is_in_equilibrium)
    """
    expected = hardy_weinberg_expected(allele_freq)
    
    # Map genotypes to expected frequencies
    # Assumes 2-allele system for simplicity
    obs_values = list(observed.values())
    if len(obs_values) != 3:
        return (0.0, True)  # Can't test with != 3 genotypes
    
    chi_sq = sum((o - e)**2 / e if e > 0 else 0 
                 for o, e in zip(obs_values, expected))
    
    # df = 1 for HW test, critical value at p=0.05 is 3.84
    return (chi_sq, chi_sq < 3.84)


def calculate_heterozygosity(genomes: List[Genome]) -> Dict[str, float]:
    """
    Calculate observed heterozygosity for each gene.
    
    Returns: {gene_name: proportion_heterozygous}
    """
    if not genomes:
        return {}
    
    result = {}
    for gene_name in GENE_DEFINITIONS:
        n_het = sum(1 for g in genomes if g.is_heterozygous(gene_name))
        result[gene_name] = n_het / len(genomes)
    
    return result


def calculate_fst(subpopulations: List[List[Genome]]) -> Dict[str, float]:
    """
    Calculate Fst (fixation index) for population structure.
    
    Fst = (Ht - Hs) / Ht
    where Ht = total heterozygosity, Hs = mean subpop heterozygosity
    
    Returns: {gene_name: Fst}
    """
    if len(subpopulations) < 2:
        return {}
    
    # Combine all genomes for total frequencies
    all_genomes = [g for subpop in subpopulations for g in subpop]
    if not all_genomes:
        return {}
    
    total_freqs = calculate_allele_frequencies(all_genomes)
    
    result = {}
    for gene_name in GENE_DEFINITIONS:
        # Expected heterozygosity in total population
        freqs = total_freqs.get(gene_name, {})
        Ht = 1 - sum(f**2 for f in freqs.values())
        
        # Mean expected heterozygosity in subpopulations
        Hs_values = []
        for subpop in subpopulations:
            if subpop:
                sub_freqs = calculate_allele_frequencies(subpop).get(gene_name, {})
                Hs = 1 - sum(f**2 for f in sub_freqs.values())
                Hs_values.append(Hs)
        
        if Hs_values and Ht > 0:
            Hs = np.mean(Hs_values)
            result[gene_name] = (Ht - Hs) / Ht
        else:
            result[gene_name] = 0.0
    
    return result


def predict_offspring_ratios(parent1: Genome, parent2: Genome, 
                             gene_name: str) -> Dict[str, float]:
    """
    Predict offspring genotype ratios (Punnett square).
    
    Returns: {genotype_string: expected_frequency}
    """
    a1, a2 = parent1.get_genotype(gene_name)
    b1, b2 = parent2.get_genotype(gene_name)
    
    # All possible offspring genotypes
    offspring = [
        tuple(sorted([a1.id, b1.id])),
        tuple(sorted([a1.id, b2.id])),
        tuple(sorted([a2.id, b1.id])),
        tuple(sorted([a2.id, b2.id])),
    ]
    
    # Count frequencies
    counts: Dict[str, int] = {}
    for o in offspring:
        key = ''.join(o)
        counts[key] = counts.get(key, 0) + 1
    
    return {k: v/4 for k, v in counts.items()}
