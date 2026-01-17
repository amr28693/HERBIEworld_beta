"""
Herbie Social Systems - Mating, genetics, names, and family behavior.

Herbies have:
- Sexual reproduction with genome mixing
- Resonance-based pair bonding (NLSE field correlation)  
- Polygyny (providers can have multiple carriers)
- Named individuals with gendered name pools
- Family cohesion and parental care
"""

from enum import Enum
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

if TYPE_CHECKING:
    pass


# =============================================================================
# SEX AND VISUAL DISTINCTION
# =============================================================================

class HerbieSex(Enum):
    PROVIDER = "provider"   # Cyan, brings food, defends
    CARRIER = "carrier"     # Magenta, gestates, nurtures


HERBIE_SEX_COLORS = {
    HerbieSex.PROVIDER: '#00FFFF',  # Cyan
    HerbieSex.CARRIER: '#FF69B4',   # Hot pink
}


# =============================================================================
# MATING PARAMETERS
# =============================================================================

HERBIE_COURTSHIP_DISTANCE = 8.0
HERBIE_COURTSHIP_DURATION = 150
HERBIE_MATING_HUNGER_MAX = 0.4
HERBIE_MATING_MIN_AGE = 250

# Pregnancy
HERBIE_GESTATION_DURATION = 400
HERBIE_PREGNANCY_SPEED_PENALTY = 0.6
HERBIE_PREGNANCY_HUNGER_MULT = 1.5

# Juvenile
HERBIE_JUVENILE_DURATION = 600
HERBIE_JUVENILE_SCALE = 0.6
HERBIE_JUVENILE_FEED_RANGE = 10.0
HERBIE_PARENTAL_FEED_RATE = 0.3

# Family
HERBIE_FAMILY_COHESION_RANGE = 20.0
HERBIE_DEFENSE_BONUS_FAMILY = 0.3

# Resonance
HERBIE_RESONANCE_THRESHOLD = 0.5
HERBIE_RESONANCE_BOND_THRESHOLD = 0.7
HERBIE_RESONANCE_CHECK_DISTANCE = 15.0
HERBIE_RESONANCE_MEMORY = 5


# =============================================================================
# NAME SYSTEM
# =============================================================================

HERBIE_NAMES_MALE = [
    # English/Nature
    "Atlas", "Birch", "Cosmo", "Dash", "Echo", "Finn", "Grove", "Harbor",
    "Moss", "North", "Oak", "Pike", "Ridge", "Stone", "Thorn", "Vale",
    "Alder", "Brook", "Clay", "Dune", "Ember", "Flint", "Glen", "Heath",
    "Knox", "Leo", "Mars", "Onyx", "Sage", "Tide", "Wren", "Yew",
    
    # Chinese/Sino
    "Chen", "Wei", "Jun", "Feng", "Long", "Ming", "Tao", "Xian",
    "Bao", "Hai", "Jin", "Lei", "Ping", "Shan", "Yang", "Zhi",
    
    # Japanese
    "Akira", "Hiro", "Kai", "Kenji", "Ren", "Ryu", "Sora", "Yuki",
    "Haru", "Kenta", "Nobu", "Shin", "Taka", "Yori", "Zen", "Dai",
    
    # Slavic
    "Alexei", "Boris", "Dmitri", "Ivan", "Lev", "Misha", "Nikolai", "Pavel",
    "Sasha", "Viktor", "Yuri", "Zoran", "Danko", "Goran", "Mirko", "Vlad",
    
    # African
    "Amadi", "Bakari", "Chidi", "Dakarai", "Emeka", "Farai", "Jabari", "Kofi",
    "Kwame", "Mandla", "Oba", "Sefu", "Tau", "Uzoma", "Zuberi", "Akin",
    
    # Latin American/Spanish
    "Alejandro", "Braulio", "Carlos", "Diego", "Esteban", "Felipe", "Gilberto", "Hugo",
    "Ignacio", "Javier", "Lorenzo", "Marco", "Nico", "Pablo", "Rafael", "Santos",
    
    # Indian
    "Arjun", "Dev", "Kiran", "Nikhil", "Raj", "Sanjay", "Ved", "Vikram",
    "Anil", "Deepak", "Hari", "Mohan", "Prem", "Ravi", "Sunil", "Vijay",
    
    # Arabic/Middle Eastern
    "Amir", "Farid", "Hassan", "Jamal", "Khalil", "Nabil", "Omar", "Rashid",
    "Samir", "Tariq", "Yusuf", "Zahir", "Idris", "Malik", "Rami", "Walid",
    
    # Celtic/Gaelic
    "Aidan", "Brennan", "Cian", "Declan", "Eamon", "Fionn", "Killian", "Liam",
    "Niall", "Oran", "Ronan", "Seamus", "Tadhg", "Cormac", "Donal", "Padraig"
]

HERBIE_NAMES_FEMALE = [
    # English/Nature
    "Aurora", "Blossom", "Coral", "Dawn", "Fern", "Gale", "Haven", "Iris",
    "Juniper", "Luna", "Maple", "Nova", "Olive", "Pearl", "River", "Sierra",
    "Terra", "Violet", "Willow", "Aria", "Briar", "Cedar", "Dove", "Eden",
    "Flora", "Grace", "Honey", "Isla", "Joy", "Lily", "Meadow", "Opal",
    
    # Chinese/Sino
    "Mei", "Lan", "Xiu", "Ying", "Hua", "Lin", "Qing", "Shu",
    "Ai", "Fang", "Hong", "Jing", "Li", "Ming", "Ning", "Wen",
    
    # Japanese
    "Akiko", "Hana", "Kiko", "Mai", "Nori", "Rei", "Saki", "Yumi",
    "Aiko", "Emi", "Haruka", "Keiko", "Mika", "Natsuki", "Sakura", "Yuki",
    
    # Slavic
    "Anya", "Daria", "Irina", "Katya", "Lena", "Mila", "Nadia", "Olga",
    "Sasha", "Tanya", "Vera", "Yelena", "Zoya", "Kira", "Lara", "Nina",
    
    # African
    "Abena", "Amara", "Esi", "Imani", "Kezia", "Nia", "Sanaa", "Zuri",
    "Adaeze", "Chioma", "Ebele", "Folami", "Makena", "Nyala", "Sade", "Yaa",
    
    # Latin American/Spanish
    "Alma", "Camila", "Dulce", "Elena", "Fernanda", "Graciela", "Isabella", "Jimena",
    "Lucia", "Marisol", "Natalia", "Paloma", "Rosa", "Sofia", "Valentina", "Ximena",
    
    # Indian
    "Ananya", "Devi", "Indira", "Kavya", "Lakshmi", "Maya", "Nila", "Priya",
    "Rani", "Sita", "Uma", "Veda", "Anita", "Deepa", "Gita", "Leela",
    
    # Arabic/Middle Eastern
    "Amina", "Fatima", "Layla", "Nadia", "Samira", "Yasmin", "Zahra", "Leila",
    "Aisha", "Dalia", "Hana", "Jamila", "Karima", "Maryam", "Noor", "Rania",
    
    # Celtic/Gaelic
    "Aisling", "Brigid", "Ciara", "Deirdre", "Eileen", "Fiona", "Grainne", "Maeve",
    "Niamh", "Orla", "Roisin", "Siobhan", "Sinead", "Aoife", "Caoimhe", "Saoirse"
]


class HerbieNameGenerator:
    """Generates unique names for Herbies."""
    _used_names = set()
    _name_counter = {}
    
    @classmethod
    def generate(cls, sex: HerbieSex) -> str:
        """Generate a unique name based on sex."""
        from random import choice
        
        pool = HERBIE_NAMES_MALE if sex == HerbieSex.PROVIDER else HERBIE_NAMES_FEMALE
        available = [n for n in pool if n not in cls._used_names]
        
        if available:
            name = choice(available)
            cls._used_names.add(name)
            return name
        else:
            base_name = choice(pool)
            cls._name_counter[base_name] = cls._name_counter.get(base_name, 1) + 1
            return f"{base_name}{cls._name_counter[base_name]}"
    
    @classmethod
    def release(cls, name: str):
        """Release a name when Herbie dies."""
        base_name = ''.join(c for c in name if not c.isdigit())
        cls._used_names.discard(base_name)
    
    @classmethod
    def reset(cls):
        """Reset for new world."""
        cls._used_names.clear()
        cls._name_counter.clear()


# =============================================================================
# GENOME
# =============================================================================

@dataclass
class HerbieGenome:
    """Genetic traits that get mixed during sexual reproduction."""
    # Physical
    body_scale: float = 1.0
    speed_modifier: float = 1.0
    energy_efficiency: float = 1.0
    
    # Behavioral
    aggression: float = 0.5
    curiosity: float = 0.5
    sociability: float = 0.5
    
    # Neural
    torus_g_modifier: float = 0.0
    audio_sensitivity: float = 1.5
    
    # Hand dexterity
    grip_strength: float = 0.5
    reach_modifier: float = 1.0
    
    @classmethod
    def random(cls) -> 'HerbieGenome':
        """Create random genome for founder."""
        return cls(
            body_scale=np.random.uniform(0.9, 1.1),
            speed_modifier=np.random.uniform(0.9, 1.1),
            energy_efficiency=np.random.uniform(0.9, 1.1),
            aggression=np.random.uniform(0.3, 0.7),
            curiosity=np.random.uniform(0.3, 0.7),
            sociability=np.random.uniform(0.3, 0.7),
            torus_g_modifier=np.random.uniform(-0.3, 0.3),
            audio_sensitivity=np.random.uniform(1.2, 1.8),
            grip_strength=np.random.uniform(0.4, 0.6),
            reach_modifier=np.random.uniform(0.9, 1.1),
        )
    
    @classmethod
    def mix_and_mutate(cls, genome_a: 'HerbieGenome', genome_b: 'HerbieGenome',
                       mutation_rate: float = 0.1) -> 'HerbieGenome':
        """Sexual reproduction: mix two genomes with mutation."""
        child = cls()
        
        for field_name in ['body_scale', 'speed_modifier', 'energy_efficiency',
                          'aggression', 'curiosity', 'sociability',
                          'torus_g_modifier', 'audio_sensitivity',
                          'grip_strength', 'reach_modifier']:
            val_a = getattr(genome_a, field_name)
            val_b = getattr(genome_b, field_name)
            
            base_val = val_a if np.random.random() < 0.5 else val_b
            
            if np.random.random() < mutation_rate:
                base_val += np.random.normal(0, 0.1)
            
            setattr(child, field_name, base_val)
        
        # Clamp values
        child.body_scale = np.clip(child.body_scale, 0.7, 1.4)
        child.speed_modifier = np.clip(child.speed_modifier, 0.6, 1.5)
        child.energy_efficiency = np.clip(child.energy_efficiency, 0.7, 1.4)
        child.aggression = np.clip(child.aggression, 0.1, 0.9)
        child.curiosity = np.clip(child.curiosity, 0.1, 0.9)
        child.sociability = np.clip(child.sociability, 0.1, 0.9)
        child.audio_sensitivity = np.clip(child.audio_sensitivity, 0.8, 2.5)
        child.grip_strength = np.clip(child.grip_strength, 0.2, 0.8)
        child.reach_modifier = np.clip(child.reach_modifier, 0.7, 1.3)
        
        return child
    
    def to_dict(self) -> dict:
        return {
            'body_scale': self.body_scale,
            'speed_modifier': self.speed_modifier,
            'energy_efficiency': self.energy_efficiency,
            'aggression': self.aggression,
            'curiosity': self.curiosity,
            'sociability': self.sociability,
            'torus_g_modifier': self.torus_g_modifier,
            'audio_sensitivity': self.audio_sensitivity,
            'grip_strength': self.grip_strength,
            'reach_modifier': self.reach_modifier,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'HerbieGenome':
        return cls(**d)


# =============================================================================
# MATING STATE
# =============================================================================

@dataclass
class HerbieMatingState:
    """Tracks mating, pregnancy, and parenting state."""
    sex: HerbieSex = HerbieSex.PROVIDER
    genome: HerbieGenome = field(default_factory=HerbieGenome.random)
    name: str = ""
    
    # Mating (polygyny: Providers can have multiple mates)
    mate_id: Optional[str] = None       # For Carriers: their one Provider
    mate_ids: List[str] = field(default_factory=list)  # For Providers: their Carriers
    courtship_target: Optional[str] = None
    courtship_progress: int = 0
    last_mating_step: int = 0
    mating_cooldown: int = 500
    
    # Resonance bonding
    is_bonded: bool = False
    resonance_score: float = 0.0
    resonance_scores: dict = field(default_factory=dict)
    failed_resonances: List[str] = field(default_factory=list)
    last_resonance_check: int = 0
    
    # Heat/seeking
    in_heat: bool = False
    heat_start_step: int = 0
    heat_duration: int = 300
    last_heat_step: int = 0
    heat_cooldown: int = 800
    
    # Pregnancy (Carriers only)
    is_pregnant: bool = False
    gestation_progress: int = 0
    offspring_genome: Optional[HerbieGenome] = None
    provider_id: Optional[str] = None
    
    # Parenting
    offspring_ids: List[str] = field(default_factory=list)
    parent_ids: Tuple[str, str] = ("", "")
    
    # Juvenile state
    is_juvenile: bool = False
    maturity: float = 0.0
    
    def can_mate(self, step_count: int, hunger: float, age: int) -> bool:
        """Check if ready to mate."""
        if self.is_pregnant or self.is_juvenile:
            return False
        if hunger > HERBIE_MATING_HUNGER_MAX:
            return False
        if age < HERBIE_MATING_MIN_AGE:
            return False
        if step_count - self.last_mating_step < self.mating_cooldown:
            return False
        return True
    
    def can_check_resonance(self, other_id: str, step_count: int) -> bool:
        """Check if we should test resonance with another Herbie."""
        if self.sex == HerbieSex.CARRIER and self.is_bonded:
            return False
        if self.sex == HerbieSex.PROVIDER and other_id in self.mate_ids:
            return False
        if step_count - self.last_resonance_check < 50:
            return False
        if other_id in self.failed_resonances[-HERBIE_RESONANCE_MEMORY:]:
            return False
        return True
    
    def add_mate(self, mate_id: str, resonance: float):
        """Add a mate (Providers can have multiple)."""
        if self.sex == HerbieSex.PROVIDER:
            if mate_id not in self.mate_ids:
                self.mate_ids.append(mate_id)
            self.resonance_scores[mate_id] = resonance
            self.is_bonded = True
            self.resonance_score = max(self.resonance_scores.values()) if self.resonance_scores else resonance
        else:
            self.mate_id = mate_id
            self.resonance_score = resonance
            self.is_bonded = True
    
    def get_mate_count(self) -> int:
        """Get number of mates."""
        if self.sex == HerbieSex.PROVIDER:
            return len(self.mate_ids)
        else:
            return 1 if self.mate_id else 0
    
    def record_failed_resonance(self, other_id: str):
        """Record a failed resonance attempt."""
        self.failed_resonances.append(other_id)
        if len(self.failed_resonances) > HERBIE_RESONANCE_MEMORY * 2:
            self.failed_resonances = self.failed_resonances[-HERBIE_RESONANCE_MEMORY:]
    
    def to_dict(self) -> dict:
        return {
            'sex': self.sex.value,
            'genome': self.genome.to_dict(),
            'name': self.name,
            'mate_id': self.mate_id,
            'mate_ids': self.mate_ids.copy(),
            'courtship_target': self.courtship_target,
            'courtship_progress': self.courtship_progress,
            'last_mating_step': self.last_mating_step,
            'is_bonded': self.is_bonded,
            'resonance_score': self.resonance_score,
            'resonance_scores': self.resonance_scores.copy(),
            'failed_resonances': self.failed_resonances.copy(),
            'is_pregnant': self.is_pregnant,
            'gestation_progress': self.gestation_progress,
            'offspring_genome': self.offspring_genome.to_dict() if self.offspring_genome else None,
            'provider_id': self.provider_id,
            'offspring_ids': self.offspring_ids.copy(),
            'parent_ids': self.parent_ids,
            'is_juvenile': self.is_juvenile,
            'maturity': self.maturity,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'HerbieMatingState':
        state = cls()
        state.sex = HerbieSex(d.get('sex', 'provider'))
        state.genome = HerbieGenome.from_dict(d['genome']) if d.get('genome') else HerbieGenome.random()
        state.name = d.get('name', '')
        state.mate_id = d.get('mate_id')
        state.mate_ids = d.get('mate_ids', [])
        state.courtship_target = d.get('courtship_target')
        state.courtship_progress = d.get('courtship_progress', 0)
        state.last_mating_step = d.get('last_mating_step', 0)
        state.is_pregnant = d.get('is_pregnant', False)
        state.gestation_progress = d.get('gestation_progress', 0)
        state.offspring_genome = HerbieGenome.from_dict(d['offspring_genome']) if d.get('offspring_genome') else None
        state.provider_id = d.get('provider_id')
        state.offspring_ids = d.get('offspring_ids', [])
        state.parent_ids = tuple(d.get('parent_ids', ("", "")))
        state.is_juvenile = d.get('is_juvenile', False)
        state.maturity = d.get('maturity', 0.0)
        state.is_bonded = d.get('is_bonded', False)
        state.resonance_score = d.get('resonance_score', 0.0)
        state.resonance_scores = d.get('resonance_scores', {})
        state.failed_resonances = d.get('failed_resonances', [])
        return state


# =============================================================================
# RESONANCE COMPUTATION
# =============================================================================

def compute_nlse_resonance(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """
    Compute resonance between two NLSE body fields.
    Returns 0-1 score where 1 = perfect resonance.
    """
    I_a = np.abs(psi_a)**2
    I_b = np.abs(psi_b)**2
    
    I_a_norm = I_a / (np.sum(I_a) + 1e-10)
    I_b_norm = I_b / (np.sum(I_b) + 1e-10)
    
    # Spatial correlation
    corr = np.corrcoef(I_a_norm.flatten(), I_b_norm.flatten())[0, 1]
    if np.isnan(corr):
        corr = 0.0
    spatial_score = (corr + 1) / 2
    
    # Energy similarity
    E_a, E_b = np.sum(I_a), np.sum(I_b)
    energy_score = min(E_a, E_b) / (max(E_a, E_b) + 1e-10)
    
    # Phase coherence
    fft_a = np.abs(np.fft.fft2(psi_a))
    fft_b = np.abs(np.fft.fft2(psi_b))
    fft_a_norm = fft_a / (np.sum(fft_a) + 1e-10)
    fft_b_norm = fft_b / (np.sum(fft_b) + 1e-10)
    freq_overlap = np.sum(np.minimum(fft_a_norm, fft_b_norm))
    phase_score = min(1.0, freq_overlap * 2)
    
    # Entropy similarity
    def field_entropy(I):
        I_flat = I.flatten()
        I_flat = I_flat / (np.sum(I_flat) + 1e-10)
        I_flat = I_flat[I_flat > 1e-10]
        return -np.sum(I_flat * np.log(I_flat))
    
    H_a, H_b = field_entropy(I_a), field_entropy(I_b)
    entropy_score = min(H_a, H_b) / (max(H_a, H_b) + 1e-10)
    
    resonance = (
        0.35 * spatial_score +
        0.25 * energy_score +
        0.25 * phase_score +
        0.15 * entropy_score
    )
    resonance += np.random.uniform(-0.1, 0.1)
    
    return float(np.clip(resonance, 0.0, 1.0))


def check_resonance_and_bond(herbie_a, herbie_b, step_count: int) -> Tuple[bool, float]:
    """
    Check NLSE resonance between two Herbies.
    Returns (should_bond, resonance_score).
    """
    if not hasattr(herbie_a, 'body') or not hasattr(herbie_b, 'body'):
        return False, 0.0
    
    psi_a = herbie_a.body.psi
    psi_b = herbie_b.body.psi
    
    resonance = compute_nlse_resonance(psi_a, psi_b)
    
    herbie_a.mating_state.last_resonance_check = step_count
    herbie_b.mating_state.last_resonance_check = step_count
    
    if resonance >= HERBIE_RESONANCE_BOND_THRESHOLD:
        return True, resonance
    elif resonance < HERBIE_RESONANCE_THRESHOLD:
        herbie_a.mating_state.record_failed_resonance(herbie_b.creature_id)
        herbie_b.mating_state.record_failed_resonance(herbie_a.creature_id)
        return False, resonance
    else:
        return False, resonance
