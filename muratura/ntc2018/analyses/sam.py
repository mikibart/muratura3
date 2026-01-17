"""
Modulo per Simplified Analysis of Masonry (SAM)
Analisi semplificata di strutture in muratura secondo approccio per componenti
Versione 8.2 - Production Ready Definitiva (Corrected)

CHANGELOG v8.2:
- FIX: Formula As fasce corretta (area di sezione, non dipende da lunghezza)
- FIX: BUCKLING effettivamente assegnato (non sovrascritto da INVALID)
- FIX: Sintassi f-string nel logging finale corretta
- FIX: Documentazione rigidezza fasce allineata (I/L, A/L)
- FIX: Gestione coerente strict validation per SpandrelType
- FIX: Rimossi blocchi duplicati nel loop analisi fasce
- FIX: Chiarezza su global_DCR vs max_DCR_overall
- NEW: max_DCR_overall nel summary
- NEW: Contatori globali (tau_max attivo, saved_by_interaction, maschi in trazione)
- NEW: Assunzioni dettagliate per tipo componente
- IMPROVED: Gerarchia modi di rottura rispettata completamente
- IMPROVED: Trasparenza totale su cap e riduzioni applicate

NOTA VERIFICA GLOBALE:
- global_DCR: massimo DCR assoluto tra tutti i componenti
- global_interaction: valore di interazione del componente critico (con interazione attiva)
- La verifica globale dipende dal peggior valore di interazione quando attiva
- Gerarchia modi rottura: BUCKLING/CRUSHING > INVALID > COMBINED > (FLEXURE|SHEAR) > SAFE
"""

import logging
import math
from typing import Dict, List, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum

# Import centralizzato da enums.py
from ..enums import LoadDistribution, LoadDistributionMethod

# Configurazione logger con NullHandler per uso come libreria
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ===============================
# COSTANTI
# ===============================

# Soglie numeriche per robustezza
MIN_DIMENSION = 0.05  # Dimensione minima elementi [m]
MIN_AREA = 0.0025  # Area minima [m²]
EPSILON = 1e-10  # Tolleranza numerica
ZERO_TOLERANCE = 0.001  # Tolleranza per considerare N≈0 [kN]
DEMAND_TOLERANCE = 0.001  # Tolleranza per considerare domanda≈0 [kN o kNm]

# Limiti fisici
MAX_FRICTION_TO_SHEAR_RATIO = 5.0  # Limite rapporto attrito/taglio
TAU_MAX_DEFAULT = 1.5  # Limite assoluto tensione taglio [MPa]

# Limiti snellezza
SLENDERNESS_LIMITS = {
    'OUT_OF_PLANE': {
        'warning': 15.0,  # λ = H/t
        'critical': 20.0,
        'max': 25.0
    },
    'IN_PLANE': {
        'warning': 2.0,  # λ = H/L
        'critical': 3.0,
        'max': 4.0
    }
}

# ===============================
# ENUMERAZIONI
# ===============================

class FailureMode(Enum):
    """Modi di rottura per componenti in muratura (con priorità)"""
    SAFE = "safe"  # Priorità 0 (solo se verificato)
    FLEXURE = "flexure"  # Priorità 1
    DIAGONAL_SHEAR = "diagonal_shear"  # Priorità 1
    SLIDING_SHEAR = "sliding_shear"  # Priorità 1
    ARCH_SHEAR = "arch_shear"  # Priorità 1
    DIRECT_SHEAR = "direct_shear"  # Priorità 1
    COMBINED = "combined"  # Priorità 2
    BUCKLING = "buckling"  # Priorità 3 (critico)
    CRUSHING = "crushing"  # Priorità 3 (critico)
    INVALID = "invalid"  # Priorità 4 (massima)

# Definizione priorità modi di rottura
# Gerarchia: BUCKLING/CRUSHING > INVALID > COMBINED > (FLEXURE|SHEAR) > SAFE
FAILURE_MODE_PRIORITY = {
    FailureMode.INVALID: 4,
    FailureMode.CRUSHING: 3,
    FailureMode.BUCKLING: 3,
    FailureMode.COMBINED: 2,
    FailureMode.FLEXURE: 1,
    FailureMode.DIAGONAL_SHEAR: 1,
    FailureMode.SLIDING_SHEAR: 1,
    FailureMode.ARCH_SHEAR: 1,
    FailureMode.DIRECT_SHEAR: 1,
    FailureMode.SAFE: 0
}

class ComponentType(Enum):
    """Tipo di componente strutturale"""
    PIER = "pier"
    SPANDREL = "spandrel"

class SlendernessType(Enum):
    """Tipo di snellezza"""
    OUT_OF_PLANE = "out_of_plane"  # H/t
    IN_PLANE = "in_plane"  # H/L

class SpandrelType(Enum):
    """Tipo di fascia"""
    UNREINFORCED = "unreinforced"  # Non armata
    REINFORCED = "reinforced"  # Armata
    ARCHED = "arched"  # Ad arco

# LoadDistributionMethod importato da ..enums (alias di LoadDistribution)
# Valori: UNIFORM, STIFFNESS, AREA, EQUAL, LENGTH, COUPLED

# ===============================
# CONFIGURAZIONE ANALISI
# ===============================

@dataclass
class AnalysisConfig:
    """Configurazione parametri di analisi estesa"""
    # Coefficienti di sicurezza
    gamma_m: float = 2.0
    FC: float = 1.0
    
    # Ripartizione carichi orizzontali
    pier_load_share: float = 0.7
    spandrel_load_share: float = 0.3
    load_distribution_method: LoadDistributionMethod = LoadDistributionMethod.UNIFORM
    
    # Ripartizione carichi verticali
    vertical_load_to_piers_only: bool = True
    
    # Parametri geometrici
    pier_spacing: float = 0.5  # [m]
    slenderness_type: SlendernessType = SlendernessType.OUT_OF_PLANE
    
    # Fattori di riduzione
    tension_reduction_sliding: float = 0.5
    tension_reduction_diagonal: float = 0.7
    arch_shear_reduction: float = 0.5
    arch_without_tie_reduction: float = 0.5
    
    # Coefficienti attrito e limiti
    mu_friction: float = 0.4
    max_friction_absolute: float = 0.5  # [MPa]
    tau_max: float = TAU_MAX_DEFAULT  # [MPa]
    
    # Parametri interazione σn-taglio
    diagonal_shear_n_factor: float = 0.3
    
    # Soglie
    crushing_limit: float = 0.95
    crushing_warning: float = 0.85
    crushing_tolerance: float = 0.02
    safety_threshold: float = 0.8
    
    # Parametri snellezza
    enable_slenderness_effects: bool = True
    slenderness_reduction_method: str = "linear"  # "linear" o "parabolic"
    
    # Parametri interazione M-V
    enable_mv_interaction: bool = True
    mv_interaction_alpha: float = 2.0
    mv_interaction_beta: float = 2.0
    
    # Opzioni modello
    create_default_pier: bool = False
    consider_spandrel_axial: bool = False
    check_pier_overlap: bool = True
    strict_input_validation: bool = True  # ValueError invece di fallback
    
    def __post_init__(self):
        """Validazione completa post-inizializzazione"""
        self.validate_all()
    
    def validate_all(self):
        """Validazione completa di tutti i parametri"""
        # Coefficienti > 0
        if self.gamma_m <= 0:
            raise ValueError(f"gamma_m deve essere > 0, ricevuto {self.gamma_m}")
        if self.FC <= 0:
            raise ValueError(f"FC deve essere > 0, ricevuto {self.FC}")
        
        # Quote in [0,1]
        if not (0 <= self.pier_load_share <= 1):
            raise ValueError(f"pier_load_share deve essere in [0,1], ricevuto {self.pier_load_share}")
        if not (0 <= self.spandrel_load_share <= 1):
            raise ValueError(f"spandrel_load_share deve essere in [0,1], ricevuto {self.spandrel_load_share}")
        
        # Parametri geometrici
        if self.pier_spacing < 0:
            logger.warning(f"pier_spacing negativo ({self.pier_spacing}), impostato a 0")
            self.pier_spacing = 0.0
        
        # Coefficienti attrito
        if self.mu_friction < 0:
            raise ValueError(f"mu_friction deve essere >= 0, ricevuto {self.mu_friction}")
        if self.max_friction_absolute < 0:
            raise ValueError(f"max_friction_absolute deve essere >= 0, ricevuto {self.max_friction_absolute}")
        
        # Validazione tau_max
        if self.tau_max <= 0:
            raise ValueError(f"tau_max deve essere > 0, ricevuto {self.tau_max}")
        
        # Soglie crushing
        if not (0 < self.crushing_warning < self.crushing_limit <= 1):
            raise ValueError("Richiesto: 0 < crushing_warning < crushing_limit <= 1")
        
        # Validazione parametri interazione
        if not (-1 <= self.diagonal_shear_n_factor <= 1):
            raise ValueError(f"diagonal_shear_n_factor deve essere in [-1,1], ricevuto {self.diagonal_shear_n_factor}")
        
        # Validazione esponenti interazione M-V
        if self.enable_mv_interaction:
            if self.mv_interaction_alpha <= 0 or self.mv_interaction_beta <= 0:
                raise ValueError("Esponenti interazione M-V devono essere > 0")
    
    def validate_and_normalize(self) -> Tuple[float, float]:
        """Valida e normalizza le quote di ripartizione orizzontale"""
        total = self.pier_load_share + self.spandrel_load_share
        
        if abs(total) < EPSILON:
            logger.warning("Quote di carico entrambe zero, uso default 70/30")
            return 0.7, 0.3
        
        if abs(total - 1.0) > 0.001:
            logger.warning(f"Ripartizione carichi non unitaria: {total:.3f}, normalizzata a 1.0")
            pier_norm = self.pier_load_share / total
            spandrel_norm = self.spandrel_load_share / total
            return pier_norm, spandrel_norm
        
        return self.pier_load_share, self.spandrel_load_share

# ===============================
# FUNZIONI UTILITÀ
# ===============================

def parse_slenderness_type(value: Any, strict: bool = False) -> SlendernessType:
    """
    Converte un valore in SlendernessType
    
    Args:
        value: Valore da convertire
        strict: Se True, lancia ValueError invece di default
    """
    if isinstance(value, SlendernessType):
        return value
    
    if isinstance(value, str):
        value_upper = value.upper()
        for st in SlendernessType:
            if st.value.upper() == value_upper:
                return st
        
        # Alias
        if value_upper in ['OUT', 'OUT_PLANE', 'OUTOFPLANE', 'OOP']:
            return SlendernessType.OUT_OF_PLANE
        elif value_upper in ['IN', 'IN_PLANE', 'INPLANE', 'IP']:
            return SlendernessType.IN_PLANE
    
    if strict:
        raise ValueError(f"Tipo snellezza non valido: {value}")
    else:
        logger.warning(f"Tipo snellezza non valido: {value}, uso default OUT_OF_PLANE")
        return SlendernessType.OUT_OF_PLANE

def parse_load_distribution_method(value: Any, strict: bool = False) -> LoadDistributionMethod:
    """
    Converte un valore in LoadDistributionMethod
    
    Args:
        value: Valore da convertire
        strict: Se True, lancia ValueError invece di default
    """
    if isinstance(value, LoadDistributionMethod):
        return value
    
    if isinstance(value, str):
        value_upper = value.upper()
        for method in LoadDistributionMethod:
            if method.value.upper() == value_upper:
                return method
    
    if strict:
        raise ValueError(f"Metodo distribuzione non valido: {value}")
    else:
        logger.warning(f"Metodo distribuzione non valido: {value}, uso default UNIFORM")
        return LoadDistributionMethod.UNIFORM

def format_dcr(dcr: float) -> str:
    """Formatta un valore DCR per output leggibile"""
    if math.isnan(dcr):
        return "N/D"
    elif dcr == float('inf'):
        return "∞"
    elif dcr > 999:
        return ">999"
    else:
        return f"{dcr:.3f}"

def calculate_dcr(demand: float, capacity: float) -> float:
    """
    Calcola DCR gestendo correttamente il caso domanda=0
    
    Args:
        demand: Domanda (valore assoluto)
        capacity: Capacità
        
    Returns:
        DCR (Demand/Capacity Ratio)
    """
    abs_demand = abs(demand)
    
    # Se domanda≈0, DCR=0 anche se capacità nulla
    if abs_demand <= DEMAND_TOLERANCE:
        return 0.0
    
    # Se capacità≈0 ma domanda>0, DCR=∞
    if capacity <= EPSILON:
        return float('inf')
    
    return abs_demand / capacity

def describe_axial_state(N: float, is_compression: bool, is_tension: bool) -> str:
    """Descrive lo stato di sforzo assiale"""
    if abs(N) <= ZERO_TOLERANCE:
        return "N≈0"
    elif is_compression:
        return "compressione"
    elif is_tension:
        return "trazione"
    else:
        return "neutro"

def calculate_slenderness_knockdown(slenderness: float, slenderness_type: SlendernessType, 
                                   method: str = "linear") -> Tuple[float, str, bool]:
    """
    Calcola il fattore di riduzione per snellezza
    
    Args:
        slenderness: Valore di snellezza
        slenderness_type: Tipo (IP/OOP)
        method: Metodo di riduzione ("linear" o "parabolic")
        
    Returns:
        Tuple (fattore_riduzione, warning_message, is_critical)
    """
    limits = SLENDERNESS_LIMITS[slenderness_type.value.upper()]
    warning_msg = None
    is_critical = False
    
    if slenderness <= limits['warning']:
        return 1.0, None, False
    elif slenderness <= limits['critical']:
        # Zona warning - riduzione leggera
        if method == "linear":
            factor = 1.0 - 0.2 * (slenderness - limits['warning']) / (limits['critical'] - limits['warning'])
        else:  # parabolic
            ratio = (slenderness - limits['warning']) / (limits['critical'] - limits['warning'])
            factor = 1.0 - 0.2 * ratio**2
        warning_msg = f"Snellezza elevata ({slenderness:.1f}), riduzione capacità {(1-factor)*100:.0f}%"
    elif slenderness <= limits['max']:
        # Zona critica - riduzione severa
        if method == "linear":
            factor = 0.8 - 0.6 * (slenderness - limits['critical']) / (limits['max'] - limits['critical'])
        else:  # parabolic
            ratio = (slenderness - limits['critical']) / (limits['max'] - limits['critical'])
            factor = 0.8 - 0.6 * ratio**2
        warning_msg = f"Snellezza critica ({slenderness:.1f}), forte riduzione capacità {(1-factor)*100:.0f}%"
    else:
        # Oltre il limite massimo - instabilità
        factor = 0.2  # Capacità residua minima
        warning_msg = f"Snellezza eccessiva ({slenderness:.1f} > {limits['max']}), instabilità"
        is_critical = True
    
    return factor, warning_msg, is_critical

def check_mv_interaction(dcr_m: float, dcr_v: float, alpha: float = 2.0, 
                        beta: float = 2.0) -> Tuple[float, bool, bool]:
    """
    Verifica interazione M-V con superficie bilineare/ellittica
    
    Args:
        dcr_m: DCR per momento
        dcr_v: DCR per taglio
        alpha: Esponente per M
        beta: Esponente per V
        
    Returns:
        Tuple (valore_interazione, verificato, saved_by_interaction)
    """
    if dcr_m == float('inf') or dcr_v == float('inf'):
        return float('inf'), False, False
    
    # Superficie di interazione: (DCR_M)^α + (DCR_V)^β ≤ 1
    interaction = (dcr_m ** alpha) + (dcr_v ** beta)
    verified = interaction <= 1.0
    
    # Flag se l'interazione "salva" un elemento con DCR singolo >1
    saved_by_interaction = verified and (dcr_m > 1.0 or dcr_v > 1.0)
    
    return interaction, verified, saved_by_interaction

# ===============================
# CLASSI GEOMETRICHE
# ===============================

@dataclass
class GeometryPier:
    """Geometria di un maschio murario"""
    length: float  # [m]
    height: float  # [m]
    thickness: float  # [m]
    position_x: float = 0.0  # [m]
    
    def __post_init__(self):
        """Validazione parametri"""
        if self.thickness < MIN_DIMENSION:
            raise ValueError(f"Spessore deve essere >= {MIN_DIMENSION}m, ricevuto {self.thickness}")
        if self.length < MIN_DIMENSION:
            raise ValueError(f"Lunghezza deve essere >= {MIN_DIMENSION}m, ricevuto {self.length}")
        if self.height < MIN_DIMENSION:
            raise ValueError(f"Altezza deve essere >= {MIN_DIMENSION}m, ricevuto {self.height}")
    
    @property
    def area(self) -> float:
        """Area sezione trasversale [m²]"""
        return self.length * self.thickness
    
    @property
    def section_modulus(self) -> float:
        """Modulo di resistenza [m³]"""
        return self.thickness * self.length**2 / 6
    
    @property
    def moment_of_inertia(self) -> float:
        """Momento d'inerzia [m⁴]"""
        return self.thickness * self.length**3 / 12
    
    @property
    def flexural_stiffness(self) -> float:
        """Rigidezza flessionale relativa [m³]"""
        return self.moment_of_inertia / self.height
    
    @property
    def shear_stiffness(self) -> float:
        """Rigidezza a taglio relativa [m²]"""
        return self.area / self.height
    
    def get_slenderness(self, slenderness_type: SlendernessType) -> float:
        """Calcola la snellezza"""
        if slenderness_type == SlendernessType.OUT_OF_PLANE:
            return self.height / self.thickness
        else:  # IN_PLANE
            return self.height / self.length
    
    def get_boundaries(self) -> Tuple[float, float]:
        """
        Restituisce i limiti geometrici del maschio
        
        Returns:
            Tuple (x_min, x_max)
        """
        x_min = self.position_x - self.length / 2
        x_max = self.position_x + self.length / 2
        return x_min, x_max

@dataclass
class GeometrySpandrel:
    """Geometria di una fascia di piano"""
    length: float  # [m]
    height: float  # [m]
    thickness: float  # [m]
    arch_rise: float = 0.0  # [m]
    has_tie_beam: bool = True
    spandrel_type: SpandrelType = SpandrelType.UNREINFORCED
    
    def __post_init__(self):
        """Validazione parametri"""
        if self.thickness < MIN_DIMENSION:
            raise ValueError(f"Spessore deve essere >= {MIN_DIMENSION}m, ricevuto {self.thickness}")
        if self.length < MIN_DIMENSION:
            raise ValueError(f"Lunghezza deve essere >= {MIN_DIMENSION}m, ricevuto {self.length}")
        if self.height < MIN_DIMENSION:
            raise ValueError(f"Altezza deve essere >= {MIN_DIMENSION}m, ricevuto {self.height}")
        if self.arch_rise < 0:
            raise ValueError(f"Freccia arco deve essere >= 0, ricevuto {self.arch_rise}")
        
        # Validazioni geometriche fasce ad arco
        if self.arch_rise > 0:
            if self.arch_rise > self.length / 2:
                logger.warning(f"Freccia arco ({self.arch_rise:.2f}m) > L/2, limitata a L/2")
                self.arch_rise = self.length / 2
            
            if self.arch_rise > self.height:
                logger.warning(f"Freccia arco ({self.arch_rise:.2f}m) > altezza, limitata all'altezza")
                self.arch_rise = self.height
            
            # Imposta tipo automaticamente se ha arco
            if self.spandrel_type == SpandrelType.UNREINFORCED:
                self.spandrel_type = SpandrelType.ARCHED
    
    @property
    def area(self) -> float:
        """Area sezione trasversale [m²]"""
        return self.height * self.thickness
    
    @property
    def shear_area(self) -> float:
        """Area resistente a taglio [m²]"""
        return self.length * self.thickness
    
    @property
    def is_arched(self) -> bool:
        """Verifica se è una fascia ad arco"""
        return self.arch_rise > 0 or self.spandrel_type == SpandrelType.ARCHED
    
    @property
    def is_reinforced(self) -> bool:
        """Verifica se è armata"""
        return self.spandrel_type == SpandrelType.REINFORCED
    
    @property
    def flexural_stiffness(self) -> float:
        """Rigidezza flessionale relativa [m³]"""
        I = self.thickness * self.height**3 / 12
        return I / self.length
    
    @property
    def shear_stiffness(self) -> float:
        """Rigidezza a taglio relativa [m²]"""
        return self.shear_area / self.length

# ===============================
# PROPRIETÀ MATERIALI
# ===============================

@dataclass
class MaterialProperties:
    """Proprietà meccaniche della muratura"""
    # Resistenze caratteristiche
    fk: float = 2.4  # [MPa]
    fvk0: float = 0.1  # [MPa]
    fvk: float = 0.15  # [MPa]
    
    # Moduli elastici
    E: float = 1000.0  # [MPa]
    G: float = 400.0  # [MPa]
    
    # Scelta resistenza taglio
    use_fvd0_for_piers: bool = True
    use_fvd0_for_spandrels: bool = False
    
    # Parametri per elementi armati
    fyk_reinforcement: float = 450.0  # Tensione snervamento armatura [MPa]
    reinforcement_ratio: float = 0.002  # Rapporto armatura [-]
    
    def __post_init__(self):
        """Validazione proprietà materiali"""
        # Controllo resistenze non negative
        if self.fk < 0:
            raise ValueError(f"fk deve essere >= 0, ricevuto {self.fk}")
        if self.fvk0 < 0:
            raise ValueError(f"fvk0 deve essere >= 0, ricevuto {self.fvk0}")
        if self.fvk < 0:
            raise ValueError(f"fvk deve essere >= 0, ricevuto {self.fvk}")
        
        # Errore per moduli elastici ≤0
        if self.E <= 0:
            raise ValueError(f"Modulo elastico E deve essere > 0, ricevuto {self.E}")
        if self.G <= 0:
            raise ValueError(f"Modulo di taglio G deve essere > 0, ricevuto {self.G}")
        
        # Warning per valori sospetti
        if self.fk == 0 and self.fvk0 == 0 and self.fvk == 0:
            logger.warning("Tutte le resistenze del materiale sono nulle")
        
        if self.fvk0 > self.fvk:
            logger.warning(f"fvk0 ({self.fvk0} MPa) > fvk ({self.fvk} MPa)")
    
    def get_design_values(self, config: AnalysisConfig) -> Dict[str, float]:
        """Calcola i valori di progetto delle resistenze"""
        gamma_m = config.gamma_m
        FC = config.FC
        
        # Resistenza armatura
        fyd_reinforcement = self.fyk_reinforcement / 1.15  # Coefficiente sicurezza acciaio
        
        return {
            'fcd': self.fk / (gamma_m * FC),
            'fvd0': self.fvk0 / (gamma_m * FC),
            'fvd': self.fvk / (gamma_m * FC),
            'fyd_s': fyd_reinforcement,
            'reinforcement_ratio': self.reinforcement_ratio  # Propagato per uso corretto
        }

# ===============================
# COMPONENTE SAM
# ===============================

@dataclass
class SAMComponent:
    """Componente strutturale per analisi SAM"""
    geometry: Union[GeometryPier, GeometrySpandrel]
    component_type: ComponentType
    axial_load: float = 0.0  # [kN]
    
    # Flag validità strutturale (oltre a quella tensionale)
    is_structurally_valid: bool = True
    structural_warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validazione post-inizializzazione"""
        if isinstance(self.geometry, GeometryPier) and self.component_type != ComponentType.PIER:
            raise ValueError("Geometria pier deve essere associata a ComponentType.PIER")
        if isinstance(self.geometry, GeometrySpandrel) and self.component_type != ComponentType.SPANDREL:
            raise ValueError("Geometria spandrel deve essere associata a ComponentType.SPANDREL")
        
        # Check validità strutturale base
        if self.geometry.area < MIN_AREA:
            self.is_structurally_valid = False
            self.structural_warnings.append(f"Area insufficiente ({self.geometry.area:.4f} m²)")
    
    def get_stress_state(self, mat_values: Dict[str, float]) -> Dict[str, Any]:
        """Calcola lo stato tensionale del componente"""
        A = self.geometry.area
        
        if A < MIN_AREA:
            logger.warning(f"Area sotto soglia minima: {A:.4f} m² < {MIN_AREA} m²")
            return {
                'sigma_0_MPa': 0.0,
                'fcd_MPa': mat_values['fcd'],
                'stress_ratio': 0.0,
                'is_compression': False,
                'is_tension': False,
                'is_valid': False
            }
        
        sigma_0_kN_m2 = self.axial_load / A
        sigma_0_MPa = sigma_0_kN_m2 / 1000
        fcd_MPa = mat_values['fcd']
        
        is_compression = self.axial_load > ZERO_TOLERANCE
        is_tension = self.axial_load < -ZERO_TOLERANCE
        
        if is_compression:
            stress_ratio = sigma_0_MPa / fcd_MPa if fcd_MPa > 0 else 0
        else:
            stress_ratio = 0.0
        
        return {
            'sigma_0_MPa': sigma_0_MPa,
            'fcd_MPa': fcd_MPa,
            'stress_ratio': stress_ratio,
            'is_compression': is_compression,
            'is_tension': is_tension,
            'is_valid': True
        }
    
    def flexure_capacity(self, material_values: Dict[str, float], 
                        material_props: MaterialProperties,
                        config: AnalysisConfig) -> Tuple[float, float, List[str]]:
        """
        Calcola la capacità flessionale con effetti snellezza
        
        Returns:
            Tuple (Momento resistente [kNm], fattore_riduzione_snellezza, warnings)
        """
        if self.component_type == ComponentType.PIER:
            return self._pier_flexure_capacity(material_values, config)
        else:
            return self._spandrel_flexure_capacity(material_values, material_props, config)
    
    def _pier_flexure_capacity(self, mat: Dict[str, float], 
                              config: AnalysisConfig) -> Tuple[float, float, List[str]]:
        """Capacità flessionale maschio con effetti snellezza"""
        warnings = []
        geom = self.geometry
        W = geom.section_modulus
        
        stress = self.get_stress_state(mat)
        
        if not stress['is_valid']:
            warnings.append("Capacità flessionale nulla (area invalida)")
            return 0.0, 1.0, warnings
        
        sigma_0_MPa = stress['sigma_0_MPa']
        fcd_MPa = stress['fcd_MPa']
        
        # Capacità base
        if stress['is_compression']:
            stress_ratio = stress['stress_ratio']
            
            if stress_ratio > config.crushing_limit:
                Mu_base = 0.0
                warnings.append("Schiacciamento completo")
            elif stress_ratio > config.crushing_warning:
                reduction = (config.crushing_limit - stress_ratio) / \
                           (config.crushing_limit - config.crushing_warning)
                Mu_base = W * (fcd_MPa - sigma_0_MPa) * reduction * 1000
                warnings.append(f"Vicino a schiacciamento (σ/fcd={stress_ratio:.1%})")
            else:
                Mu_base = W * (fcd_MPa - sigma_0_MPa) * 1000
        else:
            Mu_base = 0.0
            if stress['is_tension']:
                warnings.append("Capacità flessionale nulla (trazione)")
        
        # Effetti snellezza
        slenderness_factor = 1.0
        if config.enable_slenderness_effects and Mu_base > 0:
            slenderness = geom.get_slenderness(config.slenderness_type)
            slenderness_factor, warning, is_critical = calculate_slenderness_knockdown(
                slenderness, config.slenderness_type, config.slenderness_reduction_method
            )
            if warning:
                warnings.append(warning)
                if is_critical:
                    self.structural_warnings.append(f"Instabilità per snellezza eccessiva (λ={slenderness:.1f})")
        
        Mu = Mu_base * slenderness_factor
        
        return max(Mu, 0.0), slenderness_factor, warnings
    
    def _spandrel_flexure_capacity(self, mat: Dict[str, float],
                                  material_props: MaterialProperties,
                                  config: AnalysisConfig) -> Tuple[float, float, List[str]]:
        """Capacità flessionale fascia con distinzione tipologie"""
        warnings = []
        geom = self.geometry
        
        if geom.area < MIN_AREA:
            warnings.append("Capacità flessionale nulla (area invalida)")
            return 0.0, 1.0, warnings
        
        # Stato tensionale
        if config.consider_spandrel_axial:
            stress = self.get_stress_state(mat)
            if not stress['is_valid']:
                warnings.append("Stato tensionale invalido")
                return 0.0, 1.0, warnings
            
            if stress['is_compression'] and stress['stress_ratio'] > config.crushing_warning:
                reduction = max(0, 1 - stress['stress_ratio'])
                warnings.append(f"Riduzione per compressione elevata ({stress['stress_ratio']:.1%})")
            else:
                reduction = 1.0
        else:
            reduction = 1.0
        
        fcd_MPa = mat['fcd']
        
        # Contributo armatura con formula corretta
        reinforcement_contribution = 0.0
        if geom.is_reinforced:
            # Uso reinforcement_ratio da istanza materiale
            rho = mat.get('reinforcement_ratio', material_props.reinforcement_ratio)
            # As basata su SEZIONE (spessore × altezza), NON lunghezza
            As = geom.thickness * geom.height * rho  # [m²] - area armatura di sezione
            fyd = mat.get('fyd_s', 391.3)  # [MPa]
            z = 0.9 * geom.height  # Braccio di leva [m]
            # Conversione unità corretta MPa → kN
            reinforcement_contribution = As * fyd * z * 1000  # [m²] * [MPa] * [m] * 1000 = [kNm]
            warnings.append(f"Contributo armatura: {reinforcement_contribution:.1f} kNm (As={As*1e4:.2f} cm²)")
        
        if geom.is_arched:
            # Fascia ad arco
            t = geom.thickness
            f = geom.arch_rise
            L = geom.length
            
            # Capacità base arco
            Mu_base = fcd_MPa * t * f * L * 1000 / 8
            
            # Riduzione per assenza tiranti
            if not geom.has_tie_beam:
                Mu_base *= config.arch_without_tie_reduction
                warnings.append(f"Arco senza tirante (riduzione {config.arch_without_tie_reduction})")
        else:
            # Fascia rettilinea
            h = geom.height
            W = geom.thickness * h**2 / 6
            Mu_base = W * fcd_MPa * 1000
        
        Mu = (Mu_base + reinforcement_contribution) * reduction
        
        return max(Mu, 0.0), 1.0, warnings
    
    def shear_capacity(self, material_values: Dict[str, float],
                      material_props: MaterialProperties,
                      config: AnalysisConfig) -> Tuple[float, str, str, float, List[str]]:
        """
        Calcola la capacità a taglio con dipendenza da σn e cap assoluto
        
        Returns:
            Tuple (Taglio resistente [kN], Meccanismo, Resistenza usata, tau_actual [MPa], warnings)
        """
        if self.component_type == ComponentType.PIER:
            return self._pier_shear_capacity(material_values, material_props, config)
        else:
            return self._spandrel_shear_capacity(material_values, material_props, config)
    
    def _pier_shear_capacity(self, mat: Dict[str, float], 
                           mat_props: MaterialProperties,
                           config: AnalysisConfig) -> Tuple[float, str, str, float, List[str]]:
        """Capacità a taglio maschio con effetti σn e tau_max"""
        warnings = []
        geom = self.geometry
        A = geom.area
        
        stress = self.get_stress_state(mat)
        
        if not stress['is_valid']:
            self.is_structurally_valid = False
            warnings.append("Capacità taglio nulla (area invalida)")
            return 0.0, "invalid", "none", 0.0, warnings
        
        sigma_0_MPa = stress['sigma_0_MPa']
        
        # Scelta resistenza base
        if mat_props.use_fvd0_for_piers:
            fv_base = mat['fvd0']
            resistance_type = "fvd0"
        else:
            fv_base = mat['fvd']
            resistance_type = "fvd"
        
        # 1. Taglio per scorrimento
        if stress['is_compression']:
            friction_term = config.mu_friction * sigma_0_MPa
            
            if fv_base <= 0:
                friction_term = min(friction_term, config.max_friction_absolute)
                fv_sliding = friction_term
                warnings.append(f"Resistenza base nulla, solo attrito (limitato a {config.max_friction_absolute} MPa)")
            else:
                friction_ratio = friction_term / fv_base
                if friction_ratio > MAX_FRICTION_TO_SHEAR_RATIO:
                    friction_term = fv_base * MAX_FRICTION_TO_SHEAR_RATIO
                    warnings.append(f"Attrito limitato (ratio {friction_ratio:.1f}→{MAX_FRICTION_TO_SHEAR_RATIO})")
                fv_sliding = fv_base + friction_term
        else:
            if fv_base <= 0:
                fv_sliding = 0.0
            else:
                fv_sliding = fv_base * config.tension_reduction_sliding
                if stress['is_tension']:
                    warnings.append(f"Riduzione per trazione ({config.tension_reduction_sliding})")
        
        # Applicazione tau_max
        if fv_sliding > config.tau_max:
            fv_sliding = config.tau_max
            warnings.append(f"Cap τ_max applicato ({config.tau_max} MPa)")
        
        Vt = A * fv_sliding * 1000
        
        # 2. Taglio per fessurazione diagonale con effetto σn
        if fv_base <= 0:
            tau_diagonal = 0.0
        else:
            # Formula base con dipendenza da σn
            if stress['is_compression']:
                n_factor = 1.0 + config.diagonal_shear_n_factor * stress['stress_ratio']
            else:
                n_factor = config.tension_reduction_diagonal
            
            tau_diagonal = 1.5 * fv_base * n_factor
            
            # Cap con tau_max
            if tau_diagonal > config.tau_max:
                tau_diagonal = config.tau_max
                warnings.append(f"Cap τ_max su diagonale ({config.tau_max} MPa)")
        
        b = geom.length
        Vd = b * geom.thickness * tau_diagonal * 1000
        
        # Selezione meccanismo critico
        if Vt <= Vd:
            return max(Vt, 0.0), "sliding", resistance_type, fv_sliding, warnings
        else:
            return max(Vd, 0.0), "diagonal", resistance_type, tau_diagonal, warnings
    
    def _spandrel_shear_capacity(self, mat: Dict[str, float],
                                mat_props: MaterialProperties,
                                config: AnalysisConfig) -> Tuple[float, str, str, float, List[str]]:
        """Capacità a taglio fascia con distinzione tipologie"""
        warnings = []
        geom = self.geometry
        
        # Meccanismo "invalid" per aree insufficienti
        if geom.area < MIN_AREA or geom.shear_area < MIN_AREA:
            self.is_structurally_valid = False
            warnings.append("Capacità taglio nulla (area invalida)")
            return 0.0, "invalid", "none", 0.0, warnings
        
        A_shear = geom.shear_area
        
        # Scelta resistenza base
        if mat_props.use_fvd0_for_spandrels:
            fv_base = mat['fvd0']
            resistance_type = "fvd0"
        else:
            fv_base = mat['fvd']
            resistance_type = "fvd"
        
        # Meccanismo "invalid" per fv_base≤0
        if fv_base <= 0:
            self.is_structurally_valid = False
            warnings.append(f"Resistenza base taglio ≤0 ({fv_base:.3f} MPa)")
            return 0.0, "invalid", resistance_type, 0.0, warnings
        
        # Effetto sforzo normale
        if config.consider_spandrel_axial:
            stress = self.get_stress_state(mat)
            if stress['is_compression']:
                enhancement = 1.0 + 0.2 * min(stress['stress_ratio'], 0.5)
            elif stress['is_tension']:
                enhancement = config.tension_reduction_sliding
            else:
                enhancement = 1.0
        else:
            enhancement = 1.0
        
        # Contributo armatura
        reinforcement_factor = 1.0
        if geom.is_reinforced:
            reinforcement_factor = 1.3  # Fattore empirico
            warnings.append("Incremento per armatura (×1.3)")
        
        # Calcolo capacità base
        fv_effective = fv_base * enhancement * reinforcement_factor
        
        # Applicazione tau_max
        if fv_effective > config.tau_max:
            fv_effective = config.tau_max
            warnings.append(f"Cap τ_max applicato ({config.tau_max} MPa)")
        
        if geom.is_arched:
            # Riduzione aggiuntiva per archi senza tiranti
            arch_factor = config.arch_shear_reduction
            if not geom.has_tie_beam:
                arch_factor *= config.arch_without_tie_reduction
                warnings.append(f"Arco senza tirante (riduzione totale {arch_factor})")
            
            Vu = arch_factor * A_shear * fv_effective * 1000
            mechanism = "arch_shear"
        else:
            Vu = A_shear * fv_effective * 1000
            mechanism = "direct_shear"
        
        return max(Vu, 0.0), mechanism, resistance_type, fv_effective, warnings
    
    def determine_failure_mode(self, Mu: float, Vu: float, 
                             demand_M: float, demand_V: float,
                             shear_mechanism: str,
                             stress_state: Dict[str, Any],
                             config: AnalysisConfig,
                             has_buckling: bool = False) -> Tuple[FailureMode, float, Dict[str, Any]]:
        """
        Determina il modo di rottura con priorità e interazione M-V
        Gerarchia: BUCKLING/CRUSHING > INVALID > COMBINED > (FLEXURE|SHEAR) > SAFE
        
        Returns:
            Tuple (FailureMode, interaction_value, extra_info)
        """
        extra_info = {}
        
        # Controllo instabilità BUCKLING (priorità massima)
        if has_buckling:
            return FailureMode.BUCKLING, float('inf'), {'reason': 'slenderness_exceeded'}
        
        # Controllo schiacciamento CRUSHING
        if (self.component_type == ComponentType.PIER and 
            stress_state.get('is_compression', False)):
            
            stress_ratio = stress_state.get('stress_ratio', 0)
            if stress_ratio > config.crushing_limit * (1 - config.crushing_tolerance):
                if Mu < EPSILON or stress_ratio > config.crushing_limit:
                    return FailureMode.CRUSHING, float('inf'), {'stress_ratio': stress_ratio}
        
        # Controllo validità strutturale INVALID
        if not self.is_structurally_valid or not stress_state.get('is_valid', True):
            return FailureMode.INVALID, float('inf'), {'reason': 'structural_invalid'}
        
        # Calcolo DCR con gestione domanda=0
        DCR_M = calculate_dcr(demand_M, Mu)
        DCR_V = calculate_dcr(demand_V, Vu)
        
        # Interazione M-V
        if config.enable_mv_interaction:
            interaction_value, verified, saved_by_interaction = check_mv_interaction(
                DCR_M, DCR_V, 
                config.mv_interaction_alpha, 
                config.mv_interaction_beta
            )
            
            extra_info['saved_by_interaction'] = saved_by_interaction
            
            if verified:
                if max(DCR_M, DCR_V) <= config.safety_threshold:
                    return FailureMode.SAFE, interaction_value, extra_info
                else:
                    # Near limit ma verificato con interazione
                    if DCR_M > DCR_V:
                        return FailureMode.FLEXURE, interaction_value, extra_info
                    else:
                        return self._map_shear_mechanism(shear_mechanism), interaction_value, extra_info
            else:
                # Combined se entrambi falliscono
                if DCR_M > 1.0 and DCR_V > 1.0:
                    return FailureMode.COMBINED, interaction_value, extra_info
                # Modo singolo
                elif DCR_M > 1.0:
                    return FailureMode.FLEXURE, interaction_value, extra_info
                else:
                    return self._map_shear_mechanism(shear_mechanism), interaction_value, extra_info
        else:
            # Modalità classica senza interazione
            max_DCR = max(DCR_M, DCR_V)
            
            if max_DCR <= config.safety_threshold:
                return FailureMode.SAFE, max_DCR, extra_info
            elif DCR_M > 1.0 and DCR_V > 1.0:
                return FailureMode.COMBINED, max_DCR, extra_info
            elif DCR_M > 1.0:
                return FailureMode.FLEXURE, max_DCR, extra_info
            elif DCR_V > 1.0:
                return self._map_shear_mechanism(shear_mechanism), max_DCR, extra_info
            else:
                if DCR_M > DCR_V:
                    return FailureMode.FLEXURE, max_DCR, extra_info
                else:
                    return self._map_shear_mechanism(shear_mechanism), max_DCR, extra_info
    
    def _map_shear_mechanism(self, mechanism: str) -> FailureMode:
        """Mappa il meccanismo di taglio al modo di rottura"""
        mapping = {
            "sliding": FailureMode.SLIDING_SHEAR,
            "diagonal": FailureMode.DIAGONAL_SHEAR,
            "arch_shear": FailureMode.ARCH_SHEAR,
            "direct_shear": FailureMode.DIRECT_SHEAR,
            "invalid": FailureMode.INVALID
        }
        return mapping.get(mechanism, FailureMode.DIAGONAL_SHEAR)

# ===============================
# FUNZIONI ANALISI SAM (continua...)
# ===============================

def validate_and_sort_piers(piers: List[GeometryPier], 
                           check_overlap: bool = True) -> List[GeometryPier]:
    """
    Valida e ordina i maschi per position_x
    
    Args:
        piers: Lista di maschi
        check_overlap: Se True, controlla sovrapposizioni
        
    Returns:
        Lista ordinata di maschi
        
    Raises:
        ValueError: Se ci sono sovrapposizioni
    """
    if not piers:
        return piers
    
    # Ordina per position_x
    sorted_piers = sorted(piers, key=lambda p: p.position_x)
    
    # Controllo sovrapposizioni
    if check_overlap:
        for i in range(len(sorted_piers) - 1):
            pier1 = sorted_piers[i]
            pier2 = sorted_piers[i + 1]
            
            x1_min, x1_max = pier1.get_boundaries()
            x2_min, x2_max = pier2.get_boundaries()
            
            if x1_max > x2_min:
                overlap = x1_max - x2_min
                raise ValueError(f"Sovrapposizione maschi {i+1} e {i+2}: {overlap:.3f}m "
                               f"(x1_max={x1_max:.3f}, x2_min={x2_min:.3f})")
    
    return sorted_piers

def identify_components(wall_data: Dict, config: AnalysisConfig) -> Tuple[List[GeometryPier], List[GeometrySpandrel], float]:
    """
    Identifica e crea le geometrie con validazioni estese
    
    Returns:
        Tuple (piers, spandrels, pier_spacing_effective)
    """
    piers = []
    spandrels = []
    
    # Estrazione maschi
    piers_data = wall_data.get('piers', [])
    
    if piers_data is not None and not isinstance(piers_data, list):
        raise TypeError(f"'piers' deve essere una lista, ricevuto {type(piers_data)}")
    
    if not piers_data:
        if config.create_default_pier:
            logger.warning("Nessun maschio specificato, creazione maschio di default")
            piers_data = [{'length': 1.0, 'height': 3.0, 'thickness': 0.3}]
        else:
            logger.info("Nessun maschio specificato nel modello")
    
    # Validazione pier_spacing da wall_data
    pier_spacing_effective = wall_data.get('pier_spacing', config.pier_spacing)
    if pier_spacing_effective < 0:
        logger.warning(f"pier_spacing negativo da wall_data ({pier_spacing_effective}), uso 0")
        pier_spacing_effective = 0.0
    
    # Check policy position_x
    if piers_data:
        has_explicit = [('position_x' in p) for p in piers_data]
        all_explicit = all(has_explicit)
        none_explicit = not any(has_explicit)
        
        if not all_explicit and not none_explicit:
            raise ValueError("Position_x mista non ammessa: tutti espliciti o tutti impliciti")
        
        # Calcolo posizioni
        if none_explicit:
            # Calcolo automatico
            x_current = 0.0
            for i, pier_data in enumerate(piers_data):
                pier_data['position_x'] = x_current + pier_data['length']/2
                x_current += pier_data['length'] + pier_spacing_effective
        
        # Creazione maschi
        for i, pier_data in enumerate(piers_data):
            try:
                pier = GeometryPier(
                    length=pier_data['length'],
                    height=pier_data['height'],
                    thickness=pier_data['thickness'],
                    position_x=pier_data['position_x']
                )
                piers.append(pier)
            except (KeyError, ValueError) as e:
                logger.error(f"Errore nel maschio {i+1}: {e}")
                raise
    
    # Validazione e ordinamento maschi
    if config.check_pier_overlap:
        piers = validate_and_sort_piers(piers, check_overlap=True)
    
    # Estrazione fasce
    spandrels_data = wall_data.get('spandrels', [])
    
    if spandrels_data is not None and not isinstance(spandrels_data, list):
        raise TypeError(f"'spandrels' deve essere una lista")
    
    if spandrels_data:
        for i, spandrel_data in enumerate(spandrels_data):
            try:
                # Gestione coerente SpandrelType con strict validation
                spandrel_type_str = spandrel_data.get('type', 'UNREINFORCED').upper()
                try:
                    spandrel_type = SpandrelType[spandrel_type_str]
                except KeyError:
                    if config.strict_input_validation:
                        raise ValueError(f"Tipo fascia non valido: {spandrel_type_str}")
                    else:
                        logger.warning(f"Tipo fascia non valido: {spandrel_type_str}, uso UNREINFORCED")
                        spandrel_type = SpandrelType.UNREINFORCED
                
                # Parametri estesi
                spandrel = GeometrySpandrel(
                    length=spandrel_data['length'],
                    height=spandrel_data['height'],
                    thickness=spandrel_data['thickness'],
                    arch_rise=spandrel_data.get('arch_rise', 0.0),
                    has_tie_beam=spandrel_data.get('has_tie_beam', True),
                    spandrel_type=spandrel_type
                )
                spandrels.append(spandrel)
            except (KeyError, ValueError) as e:
                logger.error(f"Errore nella fascia {i+1}: {e}")
                raise
    
    return piers, spandrels, pier_spacing_effective

def distribute_loads(loads: Dict, piers: List[GeometryPier], 
                    spandrels: List[GeometrySpandrel],
                    pier_share: float, spandrel_share: float,
                    method: LoadDistributionMethod) -> Dict[str, Any]:
    """
    Distribuisce le sollecitazioni con metodi avanzati
    
    Args:
        loads: Carichi applicati
        piers: Lista geometrie maschi
        spandrels: Lista geometrie fasce
        pier_share: Quota maschi
        spandrel_share: Quota fasce
        method: Metodo di distribuzione
        
    Returns:
        Dizionario con sollecitazioni per componente e assunzioni
    """
    total_M = loads.get('moment', 0.0)
    total_V = loads.get('shear', 0.0)
    
    n_piers = len(piers)
    n_spandrels = len(spandrels)
    
    if n_piers == 0 and n_spandrels == 0:
        raise ValueError("Nessun componente strutturale presente")
    
    # Quote effettive
    if n_spandrels == 0:
        pier_share_actual = 1.0
        spandrel_share_actual = 0.0
    elif n_piers == 0:
        pier_share_actual = 0.0
        spandrel_share_actual = 1.0
    else:
        pier_share_actual = pier_share
        spandrel_share_actual = spandrel_share
    
    # Assunzioni dettagliate per tipo componente
    stiffness_assumptions = None
    
    # Distribuzione pesata
    if method == LoadDistributionMethod.STIFFNESS:
        # Distribuzione per rigidezza
        pier_distributions = _distribute_by_stiffness(
            piers, total_M * pier_share_actual, total_V * pier_share_actual
        )
        spandrel_distributions = _distribute_by_stiffness(
            spandrels, total_M * spandrel_share_actual, total_V * spandrel_share_actual
        )
        # Documentazione allineata alle formule reali
        stiffness_assumptions = {
            'method': 'STIFFNESS',
            'piers': 'Flessionale: I/H, Taglio: A/H',
            'spandrels': 'Flessionale: I/L, Taglio: A/L'
        }
    elif method == LoadDistributionMethod.AREA:
        # Distribuzione per area
        pier_distributions = _distribute_by_area(
            piers, total_M * pier_share_actual, total_V * pier_share_actual
        )
        spandrel_distributions = _distribute_by_area(
            spandrels, total_M * spandrel_share_actual, total_V * spandrel_share_actual
        )
        stiffness_assumptions = {
            'method': 'AREA',
            'description': 'Distribuzione proporzionale all\'area trasversale'
        }
    else:
        # Distribuzione uniforme (default)
        if n_piers > 0:
            M_pier = (total_M * pier_share_actual) / n_piers
            V_pier = (total_V * pier_share_actual) / n_piers
            pier_distributions = [(M_pier, V_pier)] * n_piers
        else:
            pier_distributions = []
        
        if n_spandrels > 0:
            M_spandrel = (total_M * spandrel_share_actual) / n_spandrels
            V_spandrel = (total_V * spandrel_share_actual) / n_spandrels
            spandrel_distributions = [(M_spandrel, V_spandrel)] * n_spandrels
        else:
            spandrel_distributions = []
        stiffness_assumptions = {
            'method': 'UNIFORM',
            'description': 'Distribuzione uniforme tra componenti'
        }
    
    logger.info(f"Metodo distribuzione: {method.value}")
    logger.info(f"Riparto effettivo - Maschi: {pier_share_actual:.1%}, "
                f"Fasce: {spandrel_share_actual:.1%}")
    
    return {
        'pier_distributions': pier_distributions,
        'spandrel_distributions': spandrel_distributions,
        'pier_share': pier_share_actual,
        'spandrel_share': spandrel_share_actual,
        'moment_pier_total': total_M * pier_share_actual,
        'distribution_method': method.value,
        'stiffness_assumptions': stiffness_assumptions
    }

def _distribute_by_stiffness(components: List, total_M: float, 
                            total_V: float) -> List[Tuple[float, float]]:
    """Distribuisce per rigidezza"""
    if not components:
        return []
    
    # Calcolo rigidezze
    flex_stiffnesses = [c.flexural_stiffness for c in components]
    shear_stiffnesses = [c.shear_stiffness for c in components]
    
    total_flex_stiff = sum(flex_stiffnesses)
    total_shear_stiff = sum(shear_stiffnesses)
    
    distributions = []
    for flex_k, shear_k in zip(flex_stiffnesses, shear_stiffnesses):
        if total_flex_stiff > 0:
            M_i = total_M * (flex_k / total_flex_stiff)
        else:
            M_i = total_M / len(components)
        
        if total_shear_stiff > 0:
            V_i = total_V * (shear_k / total_shear_stiff)
        else:
            V_i = total_V / len(components)
        
        distributions.append((M_i, V_i))
    
    return distributions

def _distribute_by_area(components: List, total_M: float, 
                       total_V: float) -> List[Tuple[float, float]]:
    """Distribuisce per area"""
    if not components:
        return []
    
    areas = [c.area for c in components]
    total_area = sum(areas)
    
    if total_area <= MIN_AREA:
        # Fallback a uniforme
        n = len(components)
        return [(total_M/n, total_V/n)] * n
    
    distributions = []
    for area in areas:
        factor = area / total_area
        distributions.append((total_M * factor, total_V * factor))
    
    return distributions

def calculate_axial_loads(loads: Dict, piers: List[GeometryPier], 
                         spandrels: List[GeometrySpandrel],
                         moment_pier_system: float,
                         config: AnalysisConfig) -> Tuple[List[float], List[float], bool, bool, List[str]]:
    """
    Calcola i carichi assiali con warning per degrado
    
    Returns:
        Tuple (pier_axials, spandrel_axials, to_piers_only_effective, axial_effect_active, warnings)
    """
    total_vertical = loads.get('vertical', 0.0)
    warnings = []
    
    pier_axials = []
    spandrel_axials = []
    
    # Controllo coerenza configurazione
    if len(piers) == 0 and config.vertical_load_to_piers_only:
        warning = "vertical_load_to_piers_only ignorato (nessun maschio)"
        logger.warning(warning)
        warnings.append(warning)
        distribute_to_piers_only = False
    else:
        distribute_to_piers_only = config.vertical_load_to_piers_only
    
    # Distribuzione
    if distribute_to_piers_only:
        vertical_to_piers = total_vertical
        vertical_to_spandrels = 0.0
    else:
        total_pier_area = sum(p.area for p in piers) if piers else 0
        total_spandrel_area = sum(s.area for s in spandrels) if spandrels else 0
        total_area = total_pier_area + total_spandrel_area
        
        if total_area > MIN_AREA:
            vertical_to_piers = total_vertical * (total_pier_area / total_area)
            vertical_to_spandrels = total_vertical * (total_spandrel_area / total_area)
            
            # Warning degrado
            if total_spandrel_area < 0.1 * total_area and vertical_to_spandrels > ZERO_TOLERANCE:
                warning = f"Area fasce <10% totale: distribuzione degenera (fasce: {vertical_to_spandrels:.1f} kN)"
                logger.warning(warning)
                warnings.append(warning)
        else:
            vertical_to_piers = total_vertical
            vertical_to_spandrels = 0.0
            if not distribute_to_piers_only:
                warning = "Area totale insufficiente: tutto il carico ai maschi"
                logger.warning(warning)
                warnings.append(warning)
    
    # Effetto N fasce
    axial_effect_active = config.consider_spandrel_axial and abs(vertical_to_spandrels) > ZERO_TOLERANCE
    
    # Carichi maschi con presso-flessione
    if piers:
        total_pier_area = sum(p.area for p in piers)
        
        if total_pier_area > MIN_AREA:
            x_bar = sum(p.position_x * p.area for p in piers) / total_pier_area
            I_tot = sum(p.area * (p.position_x - x_bar)**2 for p in piers)
            
            for pier in piers:
                N_base = vertical_to_piers * (pier.area / total_pier_area)
                
                if abs(moment_pier_system) > EPSILON and I_tot > EPSILON:
                    N_moment = moment_pier_system * (pier.position_x - x_bar) * pier.area / I_tot
                    N_total = N_base + N_moment
                else:
                    N_total = N_base
                
                pier_axials.append(N_total)
        else:
            pier_axials = [0.0] * len(piers)
    
    # Carichi fasce
    if spandrels:
        if abs(vertical_to_spandrels) > ZERO_TOLERANCE:
            total_spandrel_area = sum(s.area for s in spandrels)
            if total_spandrel_area > MIN_AREA:
                for spandrel in spandrels:
                    N_spandrel = vertical_to_spandrels * (spandrel.area / total_spandrel_area)
                    spandrel_axials.append(N_spandrel)
            else:
                spandrel_axials = [0.0] * len(spandrels)
        else:
            spandrel_axials = [0.0] * len(spandrels)
    
    return pier_axials, spandrel_axials, distribute_to_piers_only, axial_effect_active, warnings

def analyze_sam(wall_data: Dict, material: MaterialProperties,
                loads: Dict, options: Dict = None) -> Dict:
    """
    Esegue l'analisi SAM v8.2 completa
    
    Args:
        wall_data: Dati geometrici della parete
        material: Proprietà del materiale
        loads: Carichi applicati
        options: Opzioni di analisi
        
    Returns:
        Dizionario con risultati dell'analisi
    """
    # Configurazione analisi
    config = AnalysisConfig()
    if options:
        for key, value in options.items():
            if key == 'slenderness_type':
                setattr(config, key, parse_slenderness_type(value, config.strict_input_validation))
            elif key == 'load_distribution_method':
                setattr(config, key, parse_load_distribution_method(value, config.strict_input_validation))
            elif hasattr(config, key):
                setattr(config, key, value)
        
        # Validazione completa dopo override
        config.validate_all()
    
    # Validazione configurazione
    try:
        pier_share_norm, spandrel_share_norm = config.validate_and_normalize()
    except ValueError as e:
        logger.error(f"Errore configurazione: {e}")
        raise
    
    logger.info("=== INIZIO ANALISI SAM v8.2 ===")
    logger.info("Production Ready Definitiva - Correzioni complete e robustezza migliorata")
    
    # Valori di progetto del materiale
    mat_values = material.get_design_values(config.gamma_m, config.FC)
    logger.info(f"Valori materiale: fcd={mat_values['fcd']:.2f} MPa, "
                f"fvd0={mat_values['fvd0']:.3f} MPa, fvd={mat_values['fvd']:.3f} MPa")
    
    # Identificazione componenti
    try:
        piers, spandrels, pier_spacing_effective = identify_components(wall_data, config)
    except Exception as e:
        logger.error(f"Errore nell'identificazione componenti: {e}")
        raise
    
    logger.info(f"Identificati {len(piers)} maschi e {len(spandrels)} fasce")
    
    # Controllo componenti vuoti
    if not piers and not spandrels:
        logger.error("Nessun componente strutturale definito")
        raise ValueError("Almeno un maschio o una fascia deve essere definito")
    
    # Distribuzione carichi orizzontali
    load_distribution = distribute_loads(
        loads, piers, spandrels, 
        pier_share_norm, spandrel_share_norm,
        config.load_distribution_method
    )
    
    # Carichi assiali
    moment_pier_system = load_distribution['moment_pier_total']
    pier_axials, spandrel_axials, vertical_to_piers_effective, axial_effect_active, axial_warnings = calculate_axial_loads(
        loads, piers, spandrels, moment_pier_system, config
    )
    
    # Inizializzazione risultati
    results = {
        'method': 'SAM',
        'version': '8.2',
        'analysis_type': 'Simplified Analysis of Masonry - v8.2 Production Ready',
        'units': {
            'forces': 'kN',
            'moments': 'kNm',
            'stresses': 'MPa',
            'stress_ratio': '-',
            'lengths': 'm',
            'areas': 'm²',
            'section_modulus': 'm³',
            'slenderness': '-'
        },
        'configuration': {
            'gamma_m': config.gamma_m,
            'FC': config.FC,
            'horizontal_load_sharing': {
                'pier_share': load_distribution['pier_share'],
                'spandrel_share': load_distribution['spandrel_share'],
                'distribution_method': load_distribution['distribution_method']
            },
            'vertical_load_distribution': {
                'to_piers_only_input': config.vertical_load_to_piers_only,
                'to_piers_only_effective': vertical_to_piers_effective,
                'consider_spandrel_axial': config.consider_spandrel_axial
            },
            'pier_spacing_input': config.pier_spacing,
            'pier_spacing_effective': pier_spacing_effective,
            'slenderness_type': config.slenderness_type.value,
            'slenderness_effects': config.enable_slenderness_effects,
            'mv_interaction': config.enable_mv_interaction,
            'tension_reductions': {
                'sliding': config.tension_reduction_sliding,
                'diagonal': config.tension_reduction_diagonal
            },
            'friction_parameters': {
                'mu': config.mu_friction,
                'max_friction_ratio': MAX_FRICTION_TO_SHEAR_RATIO,
                'max_friction_absolute': config.max_friction_absolute,
                'tau_max': config.tau_max
            },
            'thresholds': {
                'crushing_limit': config.crushing_limit,
                'crushing_warning': config.crushing_warning,
                'safety': config.safety_threshold
            }
        },
        'material_values': mat_values,
        'n_piers': len(piers),
        'n_spandrels': len(spandrels),
        'pier_results': [],
        'spandrel_results': [],
        'summary': {},
        'warnings': {
            'global': axial_warnings,
            'components': []
        }
    }
    
    # ANALISI MASCHI
    logger.info("--- Analisi Maschi ---")
    all_component_results = []  # Per top-3
    pier_distributions = load_distribution['pier_distributions']
    
    # Contatori globali
    count_tau_max_applied = 0
    count_saved_by_interaction = 0
    count_piers_in_tension = 0
    
    for i, (pier, axial_load, (M_d, V_d)) in enumerate(zip(piers, pier_axials, pier_distributions)):
        component = SAMComponent(pier, ComponentType.PIER, axial_load)
        
        # Stato tensionale
        stress_state = component.get_stress_state(mat_values)
        
        # Contatore trazione
        if stress_state.get('is_tension', False):
            count_piers_in_tension += 1
        
        # Capacità con effetti snellezza
        Mu, slenderness_factor, flex_warnings = component.flexure_capacity(mat_values, material, config)
        Vu, shear_mechanism, resistance_used, tau_actual, shear_warnings = component.shear_capacity(
            mat_values, material, config)
        
        # Contatore tau_max
        if any('Cap τ_max' in w for w in shear_warnings):
            count_tau_max_applied += 1
        
        # Controllo instabilità
        slenderness = pier.get_slenderness(config.slenderness_type)
        has_buckling = False
        if config.enable_slenderness_effects:
            limits = SLENDERNESS_LIMITS[config.slenderness_type.value.upper()]
            if slenderness > limits['max']:
                has_buckling = True
        
        # DCR con gestione domanda=0
        DCR_flex = calculate_dcr(M_d, Mu)
        DCR_shear = calculate_dcr(V_d, Vu)
        
        # Log per DCR infiniti
        if math.isinf(DCR_flex):
            logger.debug(f"Maschio {i+1}: DCR_flex=∞ (Mu={Mu:.3f})")
        if math.isinf(DCR_shear):
            logger.debug(f"Maschio {i+1}: DCR_shear=∞ (Vu={Vu:.3f})")
        
        # Modo di rottura e interazione
        failure_mode, interaction_value, extra_info = component.determine_failure_mode(
            Mu, Vu, M_d, V_d, shear_mechanism, stress_state, config, has_buckling
        )
        
        # Contatore saved_by_interaction
        if extra_info.get('saved_by_interaction', False):
            count_saved_by_interaction += 1
        
        # Descrizione stato
        axial_state = describe_axial_state(
            axial_load,
            stress_state['is_compression'],
            stress_state['is_tension']
        )
        
        # Verifica
        if config.enable_mv_interaction:
            is_verified = (interaction_value <= 1.0 and 
                          FAILURE_MODE_PRIORITY[failure_mode] < 3)
        else:
            max_DCR = max(DCR_flex, DCR_shear)
            is_verified = (max_DCR <= 1.0 and 
                          FAILURE_MODE_PRIORITY[failure_mode] < 3)
        
        max_DCR = max(DCR_flex, DCR_shear)
        safety_state = "safe" if max_DCR <= config.safety_threshold else (
                      "near_limit" if is_verified else "failed")
        
        # Raccolta warnings componente
        component_warnings = flex_warnings + shear_warnings + component.structural_warnings
        
        pier_result = {
            'id': i + 1,
            'component_type': 'pier',
            'geometry': {
                'length': pier.length,
                'height': pier.height,
                'thickness': pier.thickness,
                'position_x': pier.position_x,
                'area': pier.area,
                'section_modulus': pier.section_modulus,
                'slenderness': slenderness,
                'slenderness_type': config.slenderness_type.value,
                'slenderness_factor': slenderness_factor
            },
            'loads': {
                'axial': axial_load,
                'axial_state': axial_state,
                'moment_demand': M_d,
                'shear_demand': V_d,
                'stress_ratio': stress_state['stress_ratio'] if stress_state['is_compression'] else None
            },
            'capacity': {
                'moment': Mu,
                'shear': Vu,
                'shear_mechanism': shear_mechanism,
                'shear_resistance': resistance_used,
                'tau_actual': tau_actual
            },
            'DCR': {
                'flexure': DCR_flex,
                'shear': DCR_shear,
                'max': max_DCR,
                'interaction': interaction_value if config.enable_mv_interaction else None
            },
            'failure_mode': failure_mode.value,
            'safety_state': safety_state,
            'verified': is_verified,
            'warnings': component_warnings,
            'extra_info': extra_info
        }
        
        results['pier_results'].append(pier_result)
        all_component_results.append(pier_result)
        
        if component_warnings:
            results['warnings']['components'].append({
                'component': f"Maschio {i+1}",
                'warnings': component_warnings
            })
        
        logger.info(f"Maschio {i+1}: N={axial_load:.1f}kN ({axial_state}), "
                   f"DCR_max={format_dcr(max_DCR)}, Modo={failure_mode.value}, "
                   f"λ={slenderness:.1f} (factor={slenderness_factor:.2f})")
    
    # ANALISI FASCE (CORRETTA - senza duplicazioni)
    logger.info("--- Analisi Fasce ---")
    spandrel_distributions = load_distribution['spandrel_distributions']
    
    for i, (spandrel, axial_load, (M_d, V_d)) in enumerate(zip(spandrels, spandrel_axials, spandrel_distributions)):
        component = SAMComponent(spandrel, ComponentType.SPANDREL, axial_load)
        
        # Stato tensionale
        stress_state = component.get_stress_state(mat_values)
        
        # Capacità
        Mu, _, flex_warnings = component.flexure_capacity(mat_values, material, config)
        Vu, shear_mechanism, resistance_used, tau_actual, shear_warnings = component.shear_capacity(
            mat_values, material, config)
        
        # Contatore tau_max
        if any('Cap τ_max' in w for w in shear_warnings):
            count_tau_max_applied += 1
        
        # DCR con gestione domanda=0
        DCR_flex = calculate_dcr(M_d, Mu)
        DCR_shear = calculate_dcr(V_d, Vu)
        
        # Log per DCR infiniti
        if math.isinf(DCR_flex):
            logger.debug(f"Fascia {i+1}: DCR_flex=∞ (Mu={Mu:.3f})")
        if math.isinf(DCR_shear):
            logger.debug(f"Fascia {i+1}: DCR_shear=∞ (Vu={Vu:.3f})")
        
        # Modo di rottura e interazione
        failure_mode, interaction_value, extra_info = component.determine_failure_mode(
            Mu, Vu, M_d, V_d, shear_mechanism, stress_state, config
        )
        
        # Contatore saved_by_interaction
        if extra_info.get('saved_by_interaction', False):
            count_saved_by_interaction += 1
        
        # Descrizione stato
        axial_state = describe_axial_state(
            axial_load,
            stress_state['is_compression'],
            stress_state['is_tension']
        )
        
        # Verifica
        if config.enable_mv_interaction:
            is_verified = (interaction_value <= 1.0 and 
                          FAILURE_MODE_PRIORITY[failure_mode] < 3)
        else:
            max_DCR = max(DCR_flex, DCR_shear)
            is_verified = (max_DCR <= 1.0 and 
                          FAILURE_MODE_PRIORITY[failure_mode] < 3)
        
        max_DCR = max(DCR_flex, DCR_shear)
        safety_state = "safe" if max_DCR <= config.safety_threshold else (
                      "near_limit" if is_verified else "failed")
        
        # Raccolta warnings componente
        component_warnings = flex_warnings + shear_warnings + component.structural_warnings
        
        spandrel_result = {
            'id': i + 1,
            'component_type': 'spandrel',
            'geometry': {
                'length': spandrel.length,
                'height': spandrel.height,
                'thickness': spandrel.thickness,
                'area': spandrel.area,
                'shear_area': spandrel.shear_area,
                'is_arched': spandrel.is_arched,
                'arch_rise': spandrel.arch_rise,
                'has_tie_beam': spandrel.has_tie_beam,
                'type': spandrel.spandrel_type.value
            },
            'loads': {
                'axial': axial_load,
                'axial_state': axial_state,
                'moment_demand': M_d,
                'shear_demand': V_d,
                'stress_ratio': stress_state['stress_ratio'] if stress_state['is_compression'] else None
            },
            'capacity': {
                'moment': Mu,
                'shear': Vu,
                'shear_mechanism': shear_mechanism,
                'shear_resistance': resistance_used,
                'tau_actual': tau_actual
            },
            'DCR': {
                'flexure': DCR_flex,
                'shear': DCR_shear,
                'max': max_DCR,
                'interaction': interaction_value if config.enable_mv_interaction else None
            },
            'failure_mode': failure_mode.value,
            'safety_state': safety_state,
            'verified': is_verified,
            'warnings': component_warnings,
            'extra_info': extra_info
        }
        
        results['spandrel_results'].append(spandrel_result)
        all_component_results.append(spandrel_result)
        
        if component_warnings:
            results['warnings']['components'].append({
                'component': f"Fascia {i+1}",
                'warnings': component_warnings
            })
        
        logger.info(f"Fascia {i+1}: DCR_max={format_dcr(max_DCR)}, "
                   f"Modo={failure_mode.value}, Tipo={spandrel.spandrel_type.value}")
    
    # TOP-3 COMPONENTI CRITICI e CALCOLO DCR GLOBALI
    # Calcolo max_DCR_overall (massimo DCR assoluto)
    all_dcr_values = [comp['DCR']['max'] for comp in all_component_results 
                      if comp['DCR']['max'] != float('inf')]
    max_DCR_overall = max(all_dcr_values) if all_dcr_values else 0.0
    
    if config.enable_mv_interaction:
        sorted_components = sorted(all_component_results, 
                                 key=lambda x: x['DCR']['interaction'] if x['DCR']['interaction'] != float('inf') else 1e10,
                                 reverse=True)
    else:
        sorted_components = sorted(all_component_results, 
                                 key=lambda x: x['DCR']['max'] if x['DCR']['max'] != float('inf') else 1e10,
                                 reverse=True)
    
    top_3_critical = []
    for comp in sorted_components[:3]:
        comp_type = comp['component_type'].capitalize()
        comp_id = comp['id']
        top_3_critical.append({
            'component': f"{comp_type} {comp_id}",
            'DCR': comp['DCR']['max'],  # Sempre DCR max
            'interaction': comp['DCR']['interaction'] if config.enable_mv_interaction else None,
            'failure_mode': comp['failure_mode'],
            'verified': comp['verified']
        })
    
    # RIEPILOGO GLOBALE
    if sorted_components:
        global_critical = sorted_components[0]
        # global_DCR ora è il max_DCR_overall per coerenza semantica
        global_DCR = max_DCR_overall
        global_interaction = global_critical['DCR']['interaction'] if config.enable_mv_interaction else None
        critical_component = f"{global_critical['component_type']}_{global_critical['id']}"
        dcr_of_critical = global_critical['DCR']['max']  # DCR del componente critico
        
        # Mapping corretto per value invece di indice enum
        failure_mode_by_value = {mode.value: mode for mode in FailureMode}
        
        # Controllo modi critici
        has_critical_failures = any(
            FAILURE_MODE_PRIORITY[failure_mode_by_value.get(comp['failure_mode'], FailureMode.SAFE)] >= 3 
            for comp in all_component_results
        )
        
        critical_components_list = [
            f"{comp['component_type'].capitalize()} {comp['id']} ({comp['failure_mode']})"
            for comp in all_component_results
            if FAILURE_MODE_PRIORITY[failure_mode_by_value.get(comp['failure_mode'], FailureMode.SAFE)] >= 3
        ]
    else:
        global_DCR = 0.0
        global_interaction = None
        critical_component = "none"
        dcr_of_critical = 0.0
        has_critical_failures = False
        critical_components_list = []
    
    # Verifica globale con check esplicito su None
    if config.enable_mv_interaction:
        global_verified = (global_interaction is None or global_interaction <= 1.0) and not has_critical_failures
    else:
        global_verified = (global_DCR <= 1.0) and not has_critical_failures
    
    results['global_DCR'] = global_DCR
    results['global_interaction'] = global_interaction
    results['verified'] = global_verified
    
    # Summary esteso v8.2
    results['summary'] = {
        'global_DCR': global_DCR,  # Ora è il max assoluto
        'max_DCR_overall': max_DCR_overall,  # Mantenuto per compatibilità
        'dcr_of_critical_component': dcr_of_critical,  # DCR del componente critico
        'global_interaction': global_interaction,
        'verification_passed': global_verified,
        'critical_component': critical_component,
        'top_3_critical': top_3_critical,
        'has_critical_failures': has_critical_failures,
        'critical_components_list': critical_components_list,
        'counters': {
            'tau_max_applied': count_tau_max_applied,
            'saved_by_interaction': count_saved_by_interaction,
            'piers_in_tension': count_piers_in_tension,
            'total_components': len(all_component_results)
        },
        'effective_parameters': {
            'pier_spacing': pier_spacing_effective,
            'vertical_to_piers_only': vertical_to_piers_effective,
            'axial_effect_active': axial_effect_active
        },
        'assumptions': {
            'stiffness_distribution': load_distribution.get('stiffness_assumptions', 'N/A'),  # Dict dettagliato
            'stiffness_summary': _get_stiffness_summary(load_distribution.get('stiffness_assumptions')),  # Stringa per compatibilità
            'reinforcement_ratio_used': mat_values.get('reinforcement_ratio', material.reinforcement_ratio),
            'reinforcement_shear_factor': 1.3
        },
        'applied_reductions': {
            'arch_without_tie': config.arch_without_tie_reduction if any(
                s.is_arched and not s.has_tie_beam for s in spandrels
            ) else None,
            'arch_shear': config.arch_shear_reduction if any(s.is_arched for s in spandrels) else None
        },
        'safety_notes': {
            'SAFE_threshold': config.safety_threshold,
            'VERIFIED_threshold': 1.0,
            'description': f"SAFE: DCR≤{config.safety_threshold}, VERIFIED: DCR/Interaction≤1.0",
            'mv_interaction_active': config.enable_mv_interaction,
            'mv_exponents': f"α={config.mv_interaction_alpha}, β={config.mv_interaction_beta}" if config.enable_mv_interaction else None
        }
    }
    
    logger.info("=== RISULTATI FINALI ===")
    logger.info(f"DCR Globale (max assoluto): {format_dcr(global_DCR)}")
    
    if config.enable_mv_interaction:
        if global_interaction is not None:
            logger.info(f"Interazione M-V Globale: {global_interaction:.3f}")
        else:
            logger.info("Interazione M-V Globale: N/A")
    
    if has_critical_failures:
        logger.warning("ATTENZIONE: Modi di rottura critici presenti")
        logger.info("Verifica: NON SUPERATA (modi critici)")
    else:
        logger.info(f"Verifica: {'SUPERATA' if global_verified else 'NON SUPERATA'}")
    
    logger.info(f"Componente critico: {critical_component}")
    
    # Log contatori
    counters = results['summary']['counters']
    logger.info(f"Contatori: τ_max applicato {counters['tau_max_applied']} volte, "
                f"{counters['saved_by_interaction']} salvati da interazione, "
                f"{counters['piers_in_tension']} maschi in trazione")
    
    return results

def _get_stiffness_summary(stiffness_assumptions):
    """Helper per compatibilità stiffness_assumptions (stringa riassuntiva per retrocompatibilità)"""
    if stiffness_assumptions is None or stiffness_assumptions == 'N/A':
        return 'N/A'
    if isinstance(stiffness_assumptions, dict):
        method = stiffness_assumptions.get('method', 'UNKNOWN')
        if method == 'STIFFNESS':
            return 'Rigidezza: Maschi I/H-A/H, Fasce I/L-A/L'
        elif method == 'AREA':
            return 'Distribuzione proporzionale all\'area'
        elif method == 'UNIFORM':
            return 'Distribuzione uniforme'
        else:
            return stiffness_assumptions.get('description', 'N/A')
    return str(stiffness_assumptions)