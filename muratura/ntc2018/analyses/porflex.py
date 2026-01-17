"""
Modulo per l'analisi PORFLEX (Pier Only Resistance with Flexible Spandrels) secondo NTC 2018
Architetto Michelangelo Bartolotta - Via Domenico Scinà n. 28, 90139 Palermo

Versione 2.1.3 - Gennaio 2025

CORREZIONI APPLICATE v2.1.3:
- ID 1-based per TUTTE le fasce nei log (coerenza completa)
- Regolarizzazione robusta anche con diagonale media nulla
- Guard su distribuzione carichi con lista maschi vuota
- Test suite aggiunta per validazione edge cases

CORREZIONI APPLICATE v2.1.2:
- Ramo esplicito per PINNED_PINNED in GeometryPier per maggiore chiarezza
- ID 1-based nei log per coerenza con output tabellari
- Metadata diagnostics con modalità critiche per debug avanzato

CORREZIONI APPLICATE v2.1.1:
- Corretta indicizzazione duplicata in build_compatibility_system
- Migliorata gestione eccezioni con messaggi più specifici
- Aggiunta validazione input per evitare divisioni per zero
- Corretta gestione del caso PINNED_PINNED nei maschi
- Sistemato calcolo critical_aspect in verifications
- Migliorata coerenza unità di misura nei calcoli

CORREZIONI PRINCIPALI v2.1:
- Sistema nodale 1-DOF: solo spostamenti traslazionali u (no rotazioni)
- Indicizzazione DOF: corretta per sistema 1-DOF (indice i, non 2*i)
- Domanda fasce: calcolata da forze elastiche nel sistema accoppiato
- Capacità maschi: integrazione con modulo POR o fallback documentato
- CouplingModel.RIGID: penalty configurabile con controllo condizionamento
- Identificazione fasce: usa coordinate reali maschi per lunghezza
- Metadata: popolamento warnings e info numeriche complete
- Tracciamento modo critico: flexure vs shear per ogni elemento

Analisi sismica di pareti murarie con metodo PORFLEX conforme a:
- NTC 2018 §7.8.2.2 (Modellazione e analisi)
- Circolare 617/2019 §C8.7.1 (Criteri di applicazione)
- Tab. 7.8.II NTC 2018 (Fattore di forma b)
- Tab. C8.5.I Circolare (Parametri meccanici)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

# Import centralizzato da enums.py
from ..enums import LoadDistribution

logger = logging.getLogger(__name__)

# Tolleranza numerica per confronti con zero
EPS = 1e-9

# Fattori di conversione unità (tutto in SI base: m, kN, MPa)
UNIT_CONVERSIONS = {
    'mm_to_m': 1e-3,
    'mm2_to_m2': 1e-6,
    'mm3_to_m3': 1e-9,
    'mm4_to_m4': 1e-12,
    'MPa_to_kN_m2': 1e3,
    'N_to_kN': 1e-3,
    'Nmm_to_kNm': 1e-6
}

# ========================= ENUMERAZIONI =========================

class FailureMode(Enum):
    """Modi di rottura secondo NTC 2018 §7.8.2.2"""
    FLEXURE = "Rottura per pressoflessione/roccheggio"
    DIAGONAL_SHEAR = "Rottura per taglio diagonale"
    SLIDING_SHEAR = "Rottura per scorrimento"
    CRUSHING = "Schiacciamento"
    NO_CAPACITY = "Assenza di capacità (trazione)"
    SPANDREL_FLEXURE = "Rottura flessionale fascia"
    SPANDREL_SHEAR = "Rottura a taglio fascia"

class MaterialType(Enum):
    """Tipologie murarie secondo Tab. C8.5.I"""
    PIETRAME_DISORDINATA = "Muratura in pietrame disordinata"
    PIETRAME_SBOZZATA = "Muratura a conci sbozzati"
    PIETRAME_BUONA = "Muratura in pietre a spacco con buona tessitura"
    BLOCCHI_LATERIZIO = "Muratura in blocchi laterizi semipieni"
    BLOCCHI_CALCESTRUZZO = "Muratura in blocchi di calcestruzzo"
    MATTONI_PIENI = "Muratura in mattoni pieni e malta di calce"

class BoundaryConditions(Enum):
    """Condizioni di vincolo"""
    CANTILEVER = "cantilever"
    FIXED_FIXED = "fixed-fixed"
    FIXED_PINNED = "fixed-pinned"
    PINNED_PINNED = "pinned-pinned"

# LoadDistribution importato da ..enums (vedi import sopra)
# Valori: AREA, EQUAL, UNIFORM, LENGTH, STIFFNESS, COUPLED

class CouplingModel(Enum):
    """Modelli di accoppiamento maschi-fasce"""
    ELASTIC = "elastic"  # Accoppiamento elastico completo
    RIGID = "rigid"      # Ipotesi di solaio rigido
    NONE = "none"        # Nessun accoppiamento (POR standard)

class SpandrelConstraint(Enum):
    """Condizioni di vincolo delle fasce"""
    FIXED_FIXED = "fixed-fixed"      # Doppio incastro
    FIXED_PINNED = "fixed-pinned"    # Incastro-cerniera
    PINNED_PINNED = "pinned-pinned"  # Doppia cerniera

# ========================= IMPORT SIMULATO DAL MODULO POR =========================

@dataclass
class MaterialProperties:
    """Proprietà meccaniche della muratura secondo Tab. C8.5.I"""
    fm: float  # Resistenza media a compressione [MPa]
    tau0: float  # Resistenza media a taglio [MPa]
    E: float  # Modulo elastico medio [MPa]
    G: float  # Modulo di taglio medio [MPa]
    w: float  # Peso specifico [kN/m³]
    mu: float = 0.4  # Coefficiente di attrito
    material_type: Optional[MaterialType] = None
    
    def get_design_values(self, gamma_m: float = 2.0, FC: float = 1.35) -> Dict:
        """Calcola valori di progetto secondo NTC 2018"""
        return {
            'fmd': self.fm / gamma_m,
            'fvd0': self.tau0 / gamma_m,
            'fcd': self.fm / (gamma_m * FC),
            'Ed': self.E / gamma_m,
            'Gd': self.G / gamma_m,
            'mu': self.mu
        }

@dataclass
class GeometryPier:
    """Geometria del maschio murario"""
    length: float  # Lunghezza [m]
    height: float  # Altezza [m]
    thickness: float  # Spessore [m]
    h0: float = field(init=False)  # Altezza di taglio [m]
    area: float = field(init=False)  # Area trasversale [m²]
    shape_factor: float = field(init=False)  # Fattore di forma b
    boundary_conditions: BoundaryConditions = BoundaryConditions.CANTILEVER
    x_position: float = 0.0  # Posizione x del centro maschio [m]
    
    def __post_init__(self):
        """Calcola proprietà derivate"""
        # Validazione dimensioni
        if self.length <= 0 or self.height <= 0 or self.thickness <= 0:
            raise ValueError(f"Dimensioni maschio devono essere positive: L={self.length}, H={self.height}, t={self.thickness}")
        
        self.area = self.length * self.thickness
        
        # Altezza di taglio secondo vincoli
        if self.boundary_conditions == BoundaryConditions.CANTILEVER:
            self.h0 = self.height
        elif self.boundary_conditions == BoundaryConditions.FIXED_FIXED:
            self.h0 = self.height / 2
        elif self.boundary_conditions == BoundaryConditions.FIXED_PINNED:
            self.h0 = 2 * self.height / 3
        else:  # PINNED_PINNED
            self.h0 = 2 * self.height / 3  # esplicito come nel caso FIXED_PINNED
        
        # Fattore di forma
        self.shape_factor = self._calculate_shape_factor()
    
    def _calculate_shape_factor(self) -> float:
        """Calcola fattore di forma secondo Tab. 7.8.II NTC 2018"""
        h_l_ratio = self.h0 / self.length
        if h_l_ratio <= 1.0:
            return 1.0
        elif h_l_ratio <= 1.5:
            return 0.85
        elif h_l_ratio <= 2.0:
            return 0.70
        else:
            return 0.70  # Limite conservativo

@dataclass
class PORPier:
    """Maschio murario per analisi POR (classe base semplificata)"""
    geometry: GeometryPier
    axial_load: float = 0.0  # Carico assiale N [kN]
    material: Optional[MaterialProperties] = None
    
    def flexure_capacity_ntc(self, mat: Dict) -> Tuple[float, FailureMode, Dict]:
        """
        Capacità flessionale secondo NTC 2018 - FALLBACK DOCUMENTATO
        
        Returns: (Mu [kNm], mode, details)
        """
        # Geometria
        A = self.geometry.area  # m²
        l = self.geometry.length  # m
        t = self.geometry.thickness  # m
        
        # Verifica area valida
        if A <= 0:
            return 0.0, FailureMode.NO_CAPACITY, {'reason': 'Area nulla'}
        
        # Tensioni in MPa
        sigma_n = self.axial_load / A / UNIT_CONVERSIONS['MPa_to_kN_m2']  # MPa
        fcd = mat['fcd']  # MPa
        
        # Verifica compressione
        if sigma_n <= 0:
            return 0.0, FailureMode.NO_CAPACITY, {'reason': 'Trazione'}
        
        if sigma_n >= 0.85 * fcd:
            return 0.0, FailureMode.CRUSHING, {'reason': 'Schiacciamento'}
        
        # Lunghezza compressa per stress-block rettangolare
        lc = min(l, self.axial_load / (0.85 * fcd * t * UNIT_CONVERSIONS['MPa_to_kN_m2']))
        
        # Braccio di leva
        lever_arm = (l - lc) / 2
        
        # Momento ultimo
        Mu = self.axial_load * lever_arm  # kNm
        
        details = {
            'method': 'NTC stress-block',
            'N': self.axial_load,
            'sigma_n': sigma_n,
            'fcd': fcd,
            'lc': lc,
            'lever_arm': lever_arm
        }
        
        return Mu, FailureMode.FLEXURE, details
    
    def shear_capacity_ntc(self, mat: Dict) -> Tuple[float, FailureMode, Dict]:
        """
        Capacità a taglio secondo NTC 2018 - FALLBACK DOCUMENTATO
        
        Returns: (Vu [kN], mode, details)
        """
        # Geometria
        A = self.geometry.area  # m²
        b = self.geometry.shape_factor
        
        # Verifica area valida
        if A <= 0:
            return 0.0, FailureMode.NO_CAPACITY, {'reason': 'Area nulla'}
        
        # Tensioni in MPa
        sigma_n = max(0, self.axial_load / A / UNIT_CONVERSIONS['MPa_to_kN_m2'])  # MPa
        
        # Parametri resistenza
        fvd0 = mat['fvd0']  # MPa
        fmd = mat.get('fmd', mat['fcd'] * 2)  # MPa
        mu = mat.get('mu', 0.4)
        
        # Taglio diagonale - Formula NTC
        alpha = 0.4
        fvd_diagonal = fvd0 + alpha * sigma_n
        fvd_max = 0.065 * fmd
        fvd_diagonal = min(fvd_diagonal, fvd_max)
        
        Vt = A * fvd_diagonal * b * UNIT_CONVERSIONS['MPa_to_kN_m2']  # kN
        
        # Taglio scorrimento
        fvd_sliding = fvd0 + mu * sigma_n
        Vs = A * fvd_sliding * UNIT_CONVERSIONS['MPa_to_kN_m2']  # kN
        
        # Minimo tra i meccanismi
        if Vt <= Vs:
            return Vt, FailureMode.DIAGONAL_SHEAR, {
                'mechanism': 'diagonal',
                'fvd': fvd_diagonal,
                'sigma_n': sigma_n,
                'b': b
            }
        else:
            return Vs, FailureMode.SLIDING_SHEAR, {
                'mechanism': 'sliding',
                'fvd': fvd_sliding,
                'sigma_n': sigma_n,
                'mu': mu
            }

# ========================= CLASSI SPECIFICHE PORFLEX =========================

@dataclass
class SpandrelGeometry:
    """
    Geometria della fascia di piano (spandrel)
    
    Fascia come trave orizzontale tra maschi:
    - Sezione resistente: b × h (thickness × height)
    - Luce libera: L (length = distanza tra assi maschi)
    - Inerzia: I = b·h³/12
    """
    length: float  # L - luce libera [m]
    height: float  # h - altezza sezione [m]
    thickness: float  # b - spessore sezione [m]
    level: int = 0
    left_pier_id: int = -1
    right_pier_id: int = -1
    constraint: SpandrelConstraint = SpandrelConstraint.FIXED_FIXED
    
    # Proprietà calcolate
    area: float = field(init=False)  # Area sezione resistente [m²]
    inertia: float = field(init=False)  # Momento d'inerzia [m⁴]
    
    def __post_init__(self):
        """Calcola proprietà derivate e valida geometria"""
        # Validazione dimensioni
        if self.length <= 0:
            raise ValueError(f"Luce fascia deve essere > 0, ricevuto {self.length}")
        if self.height <= 0:
            raise ValueError(f"Altezza fascia deve essere > 0, ricevuto {self.height}")
        if self.thickness <= 0:
            raise ValueError(f"Spessore fascia deve essere > 0, ricevuto {self.thickness}")
        
        # Validazione dimensioni minime per fasce strutturali
        if self.height < 0.20:
            logger.warning(f"Altezza fascia {self.height}m < 0.20m minimo strutturale")
        if self.length < 0.60:
            logger.warning(f"Luce fascia {self.length}m < 0.60m minimo per accoppiamento")
        
        # Area sezione resistente
        self.area = self.thickness * self.height  # b × h [m²]
        
        # Momento d'inerzia per flessione nel piano
        self.inertia = self.thickness * self.height**3 / 12  # b·h³/12 [m⁴]
        
        # Validazione connessioni
        if self.left_pier_id == -1 and self.right_pier_id == -1:
            logger.warning(f"Fascia al livello {self.level} senza maschi collegati")

@dataclass
class PORFLEXSpandrel:
    """Fascia di piano per analisi PORFLEX"""
    geometry: SpandrelGeometry
    axial_load: float = 0.0  # Carico assiale N [kN]
    material: Optional[MaterialProperties] = None
    
    def calculate_tributary_load(self, wall_data: Dict, loads: Dict) -> float:
        """Calcola carico assiale da area tributaria sopra la fascia"""
        if not self.material:
            return 0.0
        
        # Area tributaria sopra la fascia
        height_above = wall_data.get('height', 3.0) - self.geometry.level * wall_data.get('floor_height', 3.0)
        height_above = max(0, height_above - self.geometry.height)
        
        # Volume muratura sopra
        volume_above = self.geometry.length * self.geometry.thickness * height_above  # m³
        
        # Peso proprio muratura
        self_weight = volume_above * self.material.w  # kN
        
        # Carichi permanenti e variabili (stima da solaio)
        floor_loads = loads.get('floor_loads', 5.0)  # kN/m²
        tributary_area = self.geometry.length * wall_data.get('floor_span', 3.0) / 2  # m²
        applied_loads = floor_loads * tributary_area  # kN
        
        return self_weight + applied_loads
    
    def flexure_capacity(self, mat: Dict) -> Tuple[float, Dict]:
        """Capacità flessionale della fascia"""
        # Geometria in unità SI
        b = self.geometry.thickness  # m
        h = self.geometry.height  # m
        L = self.geometry.length  # m
        
        # Tensione normale da carico assiale
        A = self.geometry.area  # m²
        if A <= 0:
            return 0.0, {'error': 'Area nulla'}
        
        sigma_n = self.axial_load / A  # kN/m²
        
        # Resistenza a compressione di progetto
        fcd = mat['fcd'] * UNIT_CONVERSIONS['MPa_to_kN_m2']  # kN/m²
        
        # Modulo di resistenza elastico
        W = b * h**2 / 6  # m³
        
        # Momento resistente per pressoflessione
        if sigma_n > 0:  # Compressione
            nu = sigma_n / fcd if fcd > 0 else 0  # Sforzo normale adimensionalizzato
            factor = min(1.5, 1 + 0.5 * nu)  # Incremento limitato
        else:  # Trazione
            ft = 0.1 * fcd  # Resistenza a trazione fittizia
            factor = max(0.1, 1 + sigma_n / ft) if ft > 0 else 0.1
        
        # Momento ultimo con coefficiente per muratura
        coeff = 0.12  # Coefficiente empirico per fasce in muratura
        Mu = coeff * fcd * W * factor  # kNm
        
        details = {
            'N': self.axial_load,
            'sigma_n': sigma_n / UNIT_CONVERSIONS['MPa_to_kN_m2'],  # MPa
            'fcd': fcd / UNIT_CONVERSIONS['MPa_to_kN_m2'],  # MPa
            'W': W,
            'factor': factor,
            'coeff': coeff
        }
        
        return Mu, details
    
    def shear_capacity(self, mat: Dict) -> Tuple[float, Dict]:
        """Capacità a taglio della fascia"""
        # Geometria e tensioni
        A = self.geometry.area  # m²
        if A <= 0:
            return 0.0, {'error': 'Area nulla'}
        
        # Conversione corretta σn da kN/m² a MPa
        sigma_n_kN_m2 = self.axial_load / A  # kN/m²
        sigma_n = sigma_n_kN_m2 / UNIT_CONVERSIONS['MPa_to_kN_m2']  # MPa
        sigma_n = max(0, sigma_n)  # Solo compressione
        
        # Parametri di resistenza NTC
        fvd0 = mat['fvd0']  # MPa
        fmd = mat.get('fmd', mat['fcd'] * 2)  # MPa
        mu = mat.get('mu', 0.4)
        
        # Taglio per fessurazione diagonale
        alpha = 0.4
        fvd_diagonal = fvd0 + alpha * sigma_n  # MPa
        
        # Limite superiore secondo NTC
        fvd_max = 0.065 * fmd  # MPa
        fvd_diagonal = min(fvd_diagonal, fvd_max)
        
        # Fattore di forma per fasce
        b_spandrel = 0.8
        
        # Capacità taglio diagonale
        Vt = A * fvd_diagonal * b_spandrel * UNIT_CONVERSIONS['MPa_to_kN_m2']  # kN
        
        # Taglio per scorrimento
        fvd_sliding = fvd0 + mu * sigma_n  # MPa
        Vs = A * fvd_sliding * UNIT_CONVERSIONS['MPa_to_kN_m2']  # kN
        
        # Minimo tra i due meccanismi
        Vu = min(Vt, Vs)
        mechanism = "diagonal" if Vt <= Vs else "sliding"
        
        details = {
            'mechanism': mechanism,
            'Vt_diagonal': Vt,
            'Vs_sliding': Vs,
            'fvd_diagonal': fvd_diagonal,
            'fvd_sliding': fvd_sliding,
            'sigma_n': sigma_n,
            'b_spandrel': b_spandrel,
            'alpha': alpha,
            'fvd_max': fvd_max,
            'capped': fvd_diagonal == fvd_max
        }
        
        return Vu, details
    
    def stiffness(self, mat: Dict) -> Dict:
        """Calcola rigidezze della fascia"""
        # Moduli elastici
        E = mat.get('Ed', mat.get('E', 1000))  # MPa
        G = mat.get('Gd', mat.get('G', 400))  # MPa
        
        # Conversione in kN/m²
        E_kN = E * UNIT_CONVERSIONS['MPa_to_kN_m2']  # kN/m²
        G_kN = G * UNIT_CONVERSIONS['MPa_to_kN_m2']  # kN/m²
        
        # Geometria
        I = self.geometry.inertia  # m⁴
        L = self.geometry.length  # m
        
        # Area a taglio con fattore di forma
        kappa = 5/6  # Fattore di taglio per sezione rettangolare
        As = kappa * self.geometry.area  # m²
        
        # Rigidezza flessionale secondo vincoli
        if self.geometry.constraint == SpandrelConstraint.FIXED_FIXED:
            k_flex = 12 * E_kN * I / L**3  # kN/m
        elif self.geometry.constraint == SpandrelConstraint.FIXED_PINNED:
            k_flex = 3 * E_kN * I / L**3  # kN/m
        else:  # PINNED_PINNED
            k_flex = 0.01 * E_kN * I / L**3  # kN/m - valore nominale
        
        # Rigidezza tagliante
        k_shear = G_kN * As / L if L > 0 else 0  # kN/m
        
        # Rigidezza totale (in serie)
        if k_flex > 0 and k_shear > 0:
            k_total = 1 / (1/k_flex + 1/k_shear)
        else:
            k_total = 0
        
        return {
            'k_flex': k_flex,
            'k_shear': k_shear,
            'k_total': k_total,
            'flexibility_ratio': k_flex / k_shear if k_shear > 0 else np.inf,
            'I': I,
            'As': As,
            'kappa': kappa
        }

@dataclass
class NodeDisplacement:
    """Spostamenti nodali per sistema accoppiato - Sistema 1-DOF"""
    node_id: int
    level: int
    u_horizontal: float = 0.0  # Spostamento orizzontale [m]
    connected_piers: List[int] = field(default_factory=list)
    connected_spandrels: List[int] = field(default_factory=list)

@dataclass
class CouplingEffect:
    """Effetto di accoppiamento tra maschio e fasce"""
    pier_id: int
    V_uncoupled: float  # Taglio senza accoppiamento [kN]
    V_coupled: float  # Taglio con accoppiamento [kN]
    M_uncoupled: float  # Momento senza accoppiamento [kNm]
    M_coupled: float  # Momento con accoppiamento [kNm]
    coupling_ratio: float  # V_coupled/V_uncoupled
    top_spandrel: Optional[int] = None
    bottom_spandrel: Optional[int] = None

@dataclass
class PORFLEXOptions:
    """Opzioni per analisi PORFLEX"""
    # Coefficienti di sicurezza
    gamma_m: float = 2.0
    FC: float = 1.35
    
    # Distribuzione carichi
    load_distribution: LoadDistribution = LoadDistribution.COUPLED
    
    # Modello di accoppiamento
    coupling_model: CouplingModel = CouplingModel.ELASTIC
    
    # Parametri fasce
    consider_spandrels: bool = True
    spandrel_stiffness_factor: float = 0.5  # Riduzione per incertezze
    spandrel_strength_factor: float = 0.7  # Riduzione resistenza
    spandrel_constraint: SpandrelConstraint = SpandrelConstraint.FIXED_FIXED
    
    # Parametri numerici
    regularization_factor: float = 1e-6  # Per matrice K
    rigid_penalty_factor: float = 1000  # Fattore penalty per RIGID
    max_condition_number: float = 1e10  # Soglia warning condizionamento
    max_iterations: int = 100
    tolerance: float = 1e-4
    
    # Output
    verify_spandrels: bool = True
    detailed_output: bool = True
    include_deformation: bool = True
    
    def __post_init__(self):
        """Validazione opzioni"""
        if self.gamma_m <= 0:
            raise ValueError(f"gamma_m deve essere > 0, ricevuto {self.gamma_m}")
        if self.FC < 1:
            raise ValueError(f"FC deve essere >= 1, ricevuto {self.FC}")
        if not 0 < self.spandrel_stiffness_factor <= 1:
            raise ValueError(f"spandrel_stiffness_factor deve essere in (0,1]")
        if not 0 < self.spandrel_strength_factor <= 1:
            raise ValueError(f"spandrel_strength_factor deve essere in (0,1]")
        if self.rigid_penalty_factor < 100:
            raise ValueError(f"rigid_penalty_factor deve essere >= 100")
        if self.max_condition_number < 1e6:
            raise ValueError(f"max_condition_number deve essere >= 1e6")

# ========================= SISTEMA DI COMPATIBILITÀ NODALE =========================

def build_compatibility_system(piers: List[GeometryPier], 
                              spandrels: List[PORFLEXSpandrel],
                              material: MaterialProperties,
                              options: PORFLEXOptions) -> Tuple[np.ndarray, List[NodeDisplacement], List[str]]:
    """
    Costruisce sistema di compatibilità elastica
    
    MODELLO 1-DOF: Solo spostamenti orizzontali u
    - Nodi con solo DOF traslazionale
    - Maschi = molle verticali k = 3EI/h³
    - Fasce = molle orizzontali tra nodi
    
    Returns:
        K_global: Matrice di rigidezza globale [n × n] per n nodi
        nodes: Lista dei nodi con connettività
        warnings: Lista di warning generati
    """
    n_piers = len(piers)
    warnings = []
    
    # Sistema 1-DOF: un nodo per maschio con solo u
    nodes = []
    for i, pier in enumerate(piers):
        node = NodeDisplacement(
            node_id=i,
            level=0,  # Singolo livello per ora
            connected_piers=[i]
        )
        nodes.append(node)
    
    # Matrice rigidezza globale (1 DOF per nodo)
    n_dof = n_piers
    K = np.zeros((n_dof, n_dof))
    
    # Valori di progetto
    mat = material.get_design_values(options.gamma_m, options.FC)
    E = mat.get('Ed', material.E) * UNIT_CONVERSIONS['MPa_to_kN_m2']  # kN/m²
    
    # Assembla rigidezze maschi (diagonale principale)
    for i, pier in enumerate(piers):
        # Rigidezza laterale maschio
        I_pier = pier.thickness * pier.length**3 / 12  # m⁴
        h_eff = pier.height  # Altezza effettiva per rigidezza
        
        # Coefficiente secondo vincoli - CORRETTO: gestione unificata
        if pier.boundary_conditions == BoundaryConditions.CANTILEVER:
            k_coeff = 3
        elif pier.boundary_conditions == BoundaryConditions.FIXED_FIXED:
            k_coeff = 12
        elif pier.boundary_conditions == BoundaryConditions.FIXED_PINNED:
            k_coeff = 3  # Default conservativo per 1-DOF
        else:  # PINNED_PINNED
            k_coeff = 0.01  # Valore molto piccolo ma non nullo
            logger.warning(f"Maschio {i+1} con vincolo PINNED_PINNED: rigidezza quasi nulla")
            warnings.append(f"Maschio {i+1} PINNED_PINNED: possibile instabilità numerica")
        
        if h_eff > 0:
            k_pier = k_coeff * E * I_pier / h_eff**3  # kN/m
        else:
            k_pier = 1e-6  # Valore minimo per evitare singolarità
            warnings.append(f"Maschio {i+1} con altezza nulla")
        
        # Inserisci nella matrice globale
        K[i, i] += k_pier
    
    # Gestione modello di accoppiamento
    if options.coupling_model == CouplingModel.NONE:
        logger.info("CouplingModel.NONE: nessun accoppiamento tra maschi")
        return K, nodes, warnings
    
    # Assembla accoppiamento attraverso fasce
    n_valid_spandrels = 0
    for j, spandrel in enumerate(spandrels):
        geom = spandrel.geometry
        
        # Valida connessioni
        if geom.left_pier_id < 0 or geom.right_pier_id < 0:
            continue
        if geom.left_pier_id >= n_piers or geom.right_pier_id >= n_piers:
            msg = f"Fascia {j+1} con ID maschi non validi: {geom.left_pier_id}, {geom.right_pier_id}"
            logger.warning(msg)
            warnings.append(msg)
            continue
        
        # Escludi fasce con stesso nodo
        if geom.left_pier_id == geom.right_pier_id:
            msg = f"Fascia {j+1} con left=right={geom.left_pier_id}, esclusa"
            logger.warning(msg)
            warnings.append(msg)
            continue
        
        # Calcola rigidezza fascia
        stiffness = spandrel.stiffness(mat)
        k_span = stiffness['k_total'] * options.spandrel_stiffness_factor  # kN/m
        
        # Applica modello di accoppiamento
        if options.coupling_model == CouplingModel.RIGID:
            # RIGID: penalty method con fattore configurabile
            k_span = k_span * options.rigid_penalty_factor
            logger.debug(f"Fascia {j+1}: modalità RIGID, k_span amplificato a {k_span:.1e} kN/m")
            if n_valid_spandrels == 0:
                warnings.append(f"Modalità RIGID attiva con penalty factor={options.rigid_penalty_factor}")
        
        n_valid_spandrels += 1
        
        # Indici dei nodi collegati
        i_left = geom.left_pier_id
        i_right = geom.right_pier_id
        
        # Matrice locale fascia (2×2) nel sistema globale
        K[i_left, i_left] += k_span
        K[i_left, i_right] -= k_span
        K[i_right, i_left] -= k_span
        K[i_right, i_right] += k_span
        
        # Aggiorna connettività nodi
        nodes[i_left].connected_spandrels.append(j)
        nodes[i_right].connected_spandrels.append(j)
    
    # Condizionamento prima della regolarizzazione
    try:
        cond_before = np.linalg.cond(K)
    except:
        cond_before = np.inf
    
    # Regolarizzazione per stabilità numerica
    diag_mean = np.mean(np.diag(K))
    if options.regularization_factor > 0:
        if diag_mean > 0:
            epsilon = options.regularization_factor * diag_mean
        else:
            # Caso estremo: diagonale media nulla
            max_diag = float(np.max(np.diag(K)))
            base = max(max_diag, 1.0)
            epsilon = options.regularization_factor * base
            warnings.append("Regolarizzazione attiva con base su max_diag (diag_mean=0)")
        K += epsilon * np.eye(n_dof)
        logger.debug(f"Matrice K regolarizzata con ε = {epsilon:.2e} kN/m")
    
    # Verifica condizionamento dopo regolarizzazione
    try:
        cond_after = np.linalg.cond(K)
        logger.info(f"Condizionamento K: {cond_before:.2e} → {cond_after:.2e}")
        
        if cond_after > options.max_condition_number:
            msg = f"Matrice K mal condizionata: cond(K) = {cond_after:.2e}"
            logger.warning(msg)
            warnings.append(msg)
    except:
        logger.warning("Impossibile calcolare condizionamento matrice K")
    
    # Verifica rapporto diagonali per stabilità numerica
    min_diag = np.min(np.diag(K))
    max_diag = np.max(np.diag(K))
    if min_diag > 0:
        diag_ratio = max_diag / min_diag
        if diag_ratio > 1e8:
            msg = f"Rapporto diagonali K elevato: {diag_ratio:.2e}"
            logger.warning(msg)
            warnings.append(msg)
    
    logger.info(f"Sistema assemblato: {n_piers} nodi, {n_valid_spandrels} fasce attive")
    
    return K, nodes, warnings

def solve_coupled_system(K: np.ndarray, F: np.ndarray, 
                        nodes: List[NodeDisplacement],
                        piers: List[GeometryPier],
                        spandrels: List[PORFLEXSpandrel],
                        material: MaterialProperties,
                        options: PORFLEXOptions) -> Tuple[List[CouplingEffect], Dict[int, float], np.ndarray]:
    """
    Risolve sistema accoppiato
    
    Sistema 1-DOF: K·u = F → u → forze interne → momenti domanda
    
    Returns:
        coupling_effects: Lista di CouplingEffect con domande accoppiate
        spandrel_forces: Dict con forze nelle fasce {spandrel_id: F_span}
        u: Vettore spostamenti nodali [m]
    """
    n_dof = len(F)
    
    # Valori di progetto coerenti
    mat = material.get_design_values(options.gamma_m, options.FC)
    E = mat.get('Ed', material.E) * UNIT_CONVERSIONS['MPa_to_kN_m2']  # kN/m²
    
    try:
        # Risolvi sistema lineare
        u = np.linalg.solve(K, F)
        logger.debug(f"Sistema risolto: max |u| = {np.max(np.abs(u)):.3e} m")
    except np.linalg.LinAlgError as e:
        logger.error(f"Errore soluzione sistema: {e}")
        # Fallback a distribuzione uniforme
        max_diag = np.max(np.diag(K))
        u = F / max_diag if max_diag > 0 else F
    
    # Calcola forze nelle fasce dal sistema accoppiato
    spandrel_forces = {}
    for j, spandrel in enumerate(spandrels):
        geom = spandrel.geometry
        if geom.left_pier_id >= 0 and geom.right_pier_id >= 0:
            # Spostamenti dei nodi collegati
            u_left = u[geom.left_pier_id]
            u_right = u[geom.right_pier_id]
            
            # Forza nella fascia = k_span * (u_right - u_left)
            stiffness = spandrel.stiffness(mat)
            k_span = stiffness['k_total'] * options.spandrel_stiffness_factor
            if options.coupling_model == CouplingModel.RIGID:
                k_span *= options.rigid_penalty_factor
            
            F_span = k_span * (u_right - u_left)  # kN
            spandrel_forces[j] = F_span
    
    # Estrai forze e momenti per ogni maschio
    coupling_effects = []
    
    for i, pier in enumerate(piers):
        # Spostamento nodale
        u_node = u[i]
        
        # Rigidezza maschio coerente con build_compatibility_system
        I_pier = pier.thickness * pier.length**3 / 12  # m⁴
        h_eff = pier.height
        
        # Coefficiente secondo vincoli
        if pier.boundary_conditions == BoundaryConditions.CANTILEVER:
            k_coeff = 3
        elif pier.boundary_conditions == BoundaryConditions.FIXED_FIXED:
            k_coeff = 12
        elif pier.boundary_conditions == BoundaryConditions.FIXED_PINNED:
            k_coeff = 3
        else:  # PINNED_PINNED
            k_coeff = 0.01
        
        if h_eff > 0:
            k_pier = k_coeff * E * I_pier / h_eff**3  # kN/m
        else:
            k_pier = 1e-6
        
        # Forze con accoppiamento
        V_coupled = k_pier * u_node  # kN
        M_coupled = V_coupled * pier.h0  # kNm - usa h0 per il momento
        
        # Confronto con caso non accoppiato
        V_uncoupled = F[i]  # Forza diretta applicata
        M_uncoupled = V_uncoupled * pier.h0
        
        # Rapporto di accoppiamento
        coupling_ratio = V_coupled / V_uncoupled if V_uncoupled > EPS else 1.0
        
        effect = CouplingEffect(
            pier_id=i,
            V_uncoupled=V_uncoupled,
            V_coupled=V_coupled,
            M_uncoupled=M_uncoupled,
            M_coupled=M_coupled,
            coupling_ratio=coupling_ratio
        )
        
        coupling_effects.append(effect)
    
    return coupling_effects, spandrel_forces, u

# ========================= IDENTIFICAZIONE ELEMENTI =========================

def identify_spandrels_from_wall(wall_data: Dict, piers: List[GeometryPier], 
                                 default_constraint: SpandrelConstraint = SpandrelConstraint.FIXED_FIXED) -> Tuple[List[SpandrelGeometry], List[str]]:
    """
    Identifica fasce orizzontali tra maschi
    Usa coordinate reali dei maschi per calcolare lunghezze
    
    Returns:
        spandrels: Lista di SpandrelGeometry validate
        warnings: Lista di warning generati
    """
    spandrels = []
    warnings = []
    
    # Se le fasce sono definite esplicitamente
    if 'spandrels' in wall_data:
        for i, spandrel_data in enumerate(wall_data['spandrels']):
            thickness = spandrel_data.get('thickness', wall_data.get('thickness', 0.3))
            
            # Validazione ID maschi
            left_id = spandrel_data.get('left_pier', -1)
            right_id = spandrel_data.get('right_pier', -1)
            
            if left_id >= len(piers) or right_id >= len(piers):
                logger.warning(f"Fascia {i+1} con ID maschi non validi, ignorata")
                continue
            
            # Calcola luce effettiva da posizioni maschi
            if left_id >= 0 and right_id >= 0:
                # Usa x_position se disponibile
                if hasattr(piers[left_id], 'x_position') and hasattr(piers[right_id], 'x_position'):
                    x_left = piers[left_id].x_position
                    x_right = piers[right_id].x_position
                else:
                    # Stima da sequenza maschi
                    x_left = sum(p.length for p in piers[:left_id]) + piers[left_id].length/2
                    x_right = sum(p.length for p in piers[:right_id]) + piers[right_id].length/2
                
                length = abs(x_right - x_left)
            else:
                length = spandrel_data.get('length', 2.0)
            
            # Altezza fascia da dati o da geometria aperture
            height = spandrel_data.get('height', 0.8)
            if 'openings' in wall_data and height <= 0:
                # Stima da architrave apertura e intradosso solaio
                for opening in wall_data.get('openings', []):
                    opening_top = opening.get('y', 0) + opening.get('height', 2.0)
                    floor_height = wall_data.get('floor_height', 3.0)
                    height = max(0.2, floor_height - opening_top)
                    break
            
            # Vincoli fascia
            constraint_str = spandrel_data.get('constraint', None)
            if constraint_str:
                try:
                    constraint = SpandrelConstraint(constraint_str)
                except ValueError:
                    constraint = default_constraint
                    logger.warning(f"Vincolo fascia '{constraint_str}' non valido, uso {default_constraint.value}")
            else:
                constraint = default_constraint
            
            # Validazione minimi strutturali
            if height < 0.20:
                logger.info(f"Fascia {i+1}: h={height}m < 0.20m minimo, esclusa")
                warnings.append(f"Fascia {i+1} sotto altezza minima")
                continue
            if length < 0.60:
                logger.info(f"Fascia {i+1}: L={length}m < 0.60m minimo, esclusa")
                warnings.append(f"Fascia {i+1} sotto luce minima")
                continue
            
            try:
                spandrel = SpandrelGeometry(
                    length=length,
                    height=height,
                    thickness=thickness,
                    level=spandrel_data.get('level', 0),
                    left_pier_id=left_id,
                    right_pier_id=right_id,
                    constraint=constraint
                )
                spandrels.append(spandrel)
            except ValueError as e:
                logger.warning(f"Errore creazione fascia {i+1}: {e}")
                warnings.append(f"Fascia {i+1} non valida: {str(e)}")
    
    # Identificazione automatica migliorata
    elif 'openings' in wall_data and len(piers) > 1:
        logger.info("Identificazione automatica fasce da aperture")
        
        n_levels = wall_data.get('n_levels', 1)
        floor_height = wall_data.get('floor_height', 3.0)
        wall_thickness = wall_data.get('thickness', 0.3)
        
        for level in range(n_levels):
            h_level = level * floor_height
            
            # Ordina maschi per posizione x
            level_piers = [(i, p) for i, p in enumerate(piers)]
            level_piers.sort(key=lambda x: x[1].x_position if hasattr(x[1], 'x_position') else x[0])
            
            # Crea fasce tra maschi adiacenti
            for j in range(len(level_piers) - 1):
                left_idx, left_pier = level_piers[j]
                right_idx, right_pier = level_piers[j + 1]
                
                # Calcola luce reale
                if hasattr(left_pier, 'x_position') and hasattr(right_pier, 'x_position'):
                    span = abs(right_pier.x_position - left_pier.x_position)
                else:
                    span = 2.0  # Default se non disponibili coordinate
                
                # Stima altezza fascia
                height = 0.8  # Default
                for opening in wall_data.get('openings', []):
                    if opening.get('level', 0) == level:
                        opening_top = opening.get('y', 0) + opening.get('height', 2.0)
                        height = min(height, floor_height - opening_top)
                
                # Validazione e creazione
                if height >= 0.20 and span >= 0.60:
                    try:
                        spandrel = SpandrelGeometry(
                            length=span,
                            height=height,
                            thickness=wall_thickness,
                            level=level,
                            left_pier_id=left_idx,
                            right_pier_id=right_idx,
                            constraint=SpandrelConstraint.FIXED_FIXED
                        )
                        spandrels.append(spandrel)
                    except ValueError as e:
                        logger.warning(f"Errore creazione fascia automatica: {e}")
    
    logger.info(f"Identificate {len(spandrels)} fasce strutturali")
    return spandrels, warnings

# ========================= FUNZIONE PRINCIPALE PORFLEX =========================

def analyze_porflex(wall_data: Dict, material: MaterialProperties,
                    loads: Dict, options: PORFLEXOptions = None) -> Dict:
    """
    Analisi PORFLEX completa
    
    Sistema nodale 1-DOF con compatibilità elastica:
    - Nodi ai top dei maschi (solo u)
    - Maschi come molle verticali
    - Fasce come molle orizzontali
    - Soluzione per spostamenti → forze → verifiche
    """
    # Log header
    logger.info("=" * 70)
    logger.info("ANALISI PORFLEX v2.1.3 - POR with Flexible Spandrels")
    logger.info("=" * 70)
    logger.info(f"Parete: L={wall_data.get('length','N/A')}m, H={wall_data.get('height','N/A')}m")
    logger.info(f"Materiale: {material.material_type.value if material.material_type else 'Custom'}")
    
    # Default options
    if options is None:
        options = PORFLEXOptions()
    
    logger.info(f"Opzioni: coupling={options.coupling_model.value}, "
               f"spandrels={'SI' if options.consider_spandrels else 'NO'}")
    
    # Raccolta warnings per metadata
    warnings_list = []
    
    # Import funzioni POR necessarie (simulato)
    def identify_piers_from_wall(wall_data):
        """Funzione simulata - in produzione da por.py"""
        piers = []
        if 'piers' in wall_data:
            x_current = 0
            for pier_data in wall_data['piers']:
                try:
                    pier = GeometryPier(
                        length=pier_data.get('length', 1.0),
                        height=pier_data.get('height', 3.0),
                        thickness=pier_data.get('thickness', wall_data.get('thickness', 0.3)),
                        x_position=x_current + pier_data.get('length', 1.0) / 2
                    )
                    piers.append(pier)
                    x_current += pier_data.get('length', 1.0) + pier_data.get('spacing', 0.5)
                except ValueError as e:
                    logger.warning(f"Errore creazione maschio: {e}")
        else:
            # Default: 2 maschi
            piers = [
                GeometryPier(2.0, 3.0, 0.3, x_position=1.0),
                GeometryPier(2.0, 3.0, 0.3, x_position=4.0)
            ]
        return piers
    
    def distribute_loads(N_total, piers, method, material):
        """Distribuzione carichi verticali - simulata"""
        if not piers:  # Guard su lista vuota
            return []
        if method == 'area':
            total_area = sum(p.area for p in piers)
            if total_area > 0:
                return [N_total * p.area / total_area for p in piers]
            # fallback uniforme se area totale = 0
        return [N_total / len(piers) for _ in piers]
    
    # Identifica elementi strutturali
    try:
        piers = identify_piers_from_wall(wall_data)
        spandrels_geom, identify_warnings = identify_spandrels_from_wall(
            wall_data, piers, options.spandrel_constraint
        )
        warnings_list.extend(identify_warnings)
    except Exception as e:
        logger.error(f"Errore identificazione elementi: {e}")
        raise ValueError(f"Impossibile identificare elementi strutturali: {str(e)}")
    
    logger.info(f"Identificati: {len(piers)} maschi, {len(spandrels_geom)} fasce")
    
    # Crea oggetti fascia con carichi
    spandrels = []
    for spandrel_geom in spandrels_geom:
        spandrel = PORFLEXSpandrel(spandrel_geom, 0.0, material)
        spandrel.axial_load = spandrel.calculate_tributary_load(wall_data, loads)
        spandrels.append(spandrel)
        logger.debug(f"Fascia {len(spandrels)}: N={spandrel.axial_load:.1f} kN da area tributaria")
    
    # Valori di progetto materiale
    mat = material.get_design_values(options.gamma_m, options.FC)
    
    # Carichi totali
    V_total = abs(loads.get('horizontal', 0.0))  # kN
    N_total = loads.get('vertical', 0.0)  # kN
    
    logger.info(f"Carichi: V={V_total:.1f} kN, N={N_total:.1f} kN")
    
    # Sistema accoppiato o standard
    spandrel_forces = {}
    u_solution = None
    K_global = None  # Inizializza per evitare errori
    
    if options.load_distribution == LoadDistribution.COUPLED and options.consider_spandrels and spandrels:
        # Costruisci sistema accoppiato
        logger.info("ACCOPPIAMENTO ATTIVO - Costruzione sistema nodale 1-DOF")
        K_global, nodes, build_warnings = build_compatibility_system(piers, spandrels, material, options)
        warnings_list.extend(build_warnings)
        
        # Vettore forze (distribuzione iniziale uniforme)
        n_piers = len(piers)
        F = np.zeros(n_piers)
        for i in range(n_piers):
            F[i] = V_total / n_piers if n_piers > 0 else 0
        
        # Risolvi sistema accoppiato
        coupling_effects, spandrel_forces, u_solution = solve_coupled_system(
            K_global, F, nodes, piers, spandrels, material, options
        )
        
        # Log rigidezze
        logger.info("Riepilogo rigidezze fasce:")
        for i, spandrel in enumerate(spandrels[:3]):
            stiff = spandrel.stiffness(mat)
            logger.info(f"  Fascia {i+1}: k_flex={stiff['k_flex']:.1f}, "
                       f"k_shear={stiff['k_shear']:.1f}, ratio={stiff['flexibility_ratio']:.2f}")
    else:
        # Distribuzione standard senza accoppiamento
        logger.info("Distribuzione standard (no accoppiamento)")
        coupling_effects = []
        for i in range(len(piers)):
            V_demand = V_total/len(piers) if len(piers) > 0 else 0
            effect = CouplingEffect(
                pier_id=i,
                V_uncoupled=V_demand,
                V_coupled=V_demand,
                M_uncoupled=V_demand * piers[i].h0,
                M_coupled=V_demand * piers[i].h0,
                coupling_ratio=1.0
            )
            coupling_effects.append(effect)
    
    # Distribuisci carichi verticali
    pier_loads = distribute_loads(N_total, piers, 'area', material)
    
    # Inizializza risultati
    results = {
        'method': 'PORFLEX v2.1.3 - NTC 2018',
        'wall_geometry': wall_data,
        'material': {
            'type': material.material_type.value if material.material_type else 'Custom',
            'gamma_m': options.gamma_m,
            'FC': options.FC,
            'fm': material.fm,
            'tau0': material.tau0,
            'E': material.E,
            'mu': mat.get('mu', 0.4)
        },
        'loads': {
            'vertical': N_total,
            'horizontal': V_total
        },
        'options': {
            'coupling_model': options.coupling_model.value,
            'consider_spandrels': options.consider_spandrels,
            'spandrel_constraint': options.spandrel_constraint.value if options.consider_spandrels else 'N/A',
            'regularization_factor': options.regularization_factor,
            'rigid_penalty_factor': options.rigid_penalty_factor if options.coupling_model == CouplingModel.RIGID else 'N/A'
        },
        'n_piers': len(piers),
        'n_spandrels': len(spandrels),
        'piers_analysis': [],
        'spandrels_analysis': [],
        'coupling_effects': [],
        'global_capacity': {},
        'verifications': {},
        'metadata': {
            'version': '2.1.3',
            'warnings': [],
            'units_scale': {
                'geometry': 'm',
                'forces': 'kN',
                'stresses': 'MPa',
                'stiffness': 'kN/m'
            },
            'numerical_info': {}
        }
    }
    
    # Analizza maschi con capacità dal POR
    total_Mu = 0.0
    total_Vu = 0.0
    min_safety_pier = np.inf
    min_fs_flex_pier = np.inf
    min_fs_shear_pier = np.inf
    critical_pier = -1
    critical_pier_mode = None
    
    logger.info("\nAnalisi maschi:")
    logger.info("ID | L[m] | N[kN] | V_demand | M_demand | Mu | Vu | FS")
    logger.info("-" * 60)
    
    for i, (pier, N, effect) in enumerate(zip(piers, pier_loads, coupling_effects)):
        # Crea oggetto maschio
        por_pier = PORPier(pier, N, material)
        
        # Capacità con metodi POR o fallback documentato
        try:
            # Tentativo metodi reali se disponibili
            if hasattr(por_pier, 'flexure_capacity_rocking'):
                Mu, flexure_mode, flex_details = por_pier.flexure_capacity_rocking(mat)
            else:
                # Fallback NTC documentato
                Mu, flexure_mode, flex_details = por_pier.flexure_capacity_ntc(mat)
                if 'method' not in flex_details:
                    flex_details['method'] = 'fallback_ntc'
        except Exception as e:
            msg = f"Errore calcolo Mu maschio {i+1}: {e}"
            logger.error(msg)
            warnings_list.append(msg)
            raise ValueError(f"Impossibile calcolare capacità flessionale maschio {i+1}")
        
        # Capacità a taglio
        try:
            if hasattr(por_pier, 'shear_capacity'):
                Vu, shear_mode, shear_details = por_pier.shear_capacity(mat, {'shear_cap_factor': 0.065})
            else:
                # Fallback NTC documentato
                Vu, shear_mode, shear_details = por_pier.shear_capacity_ntc(mat)
                if 'method' not in shear_details:
                    shear_details['method'] = 'fallback_ntc'
        except Exception as e:
            msg = f"Errore calcolo Vu maschio {i+1}: {e}"
            logger.error(msg)
            warnings_list.append(msg)
            raise ValueError(f"Impossibile calcolare capacità taglio maschio {i+1}")
        
        # Domanda DA ACCOPPIAMENTO
        V_demand = effect.V_coupled
        M_demand = effect.M_coupled
        
        # Fattori di sicurezza
        fs_flex = Mu / M_demand if M_demand > EPS else 999
        fs_shear = Vu / V_demand if V_demand > EPS else 999
        fs_local = min(fs_flex, fs_shear)
        
        # Traccia modo critico
        local_critical_mode = 'flexure' if fs_flex <= fs_shear else 'shear'
        
        # Aggiorna minimi globali
        if fs_flex < min_fs_flex_pier:
            min_fs_flex_pier = fs_flex
        if fs_shear < min_fs_shear_pier:
            min_fs_shear_pier = fs_shear
        
        if fs_local < min_safety_pier:
            min_safety_pier = fs_local
            critical_pier = i
            critical_pier_mode = local_critical_mode
        
        # Accumula capacità
        total_Mu += Mu
        total_Vu += Vu
        
        logger.info(f"{i+1:2} | {pier.length:4.2f} | {N:6.1f} | "
                   f"{V_demand:6.1f} | {M_demand:7.1f} | "
                   f"{Mu:6.1f} | {Vu:6.1f} | {fs_local:4.2f}")
        
        # Memorizza risultati
        pier_result = {
            'id': i + 1,
            'geometry': {
                'length': pier.length,
                'height': pier.height,
                'thickness': pier.thickness
            },
            'loads': {
                'N': N,
                'V_demand': V_demand,
                'M_demand': M_demand,
                'coupling_ratio': effect.coupling_ratio
            },
            'capacity': {
                'Mu': Mu,
                'Vu': Vu,
                'flexure_mode': flexure_mode.value,
                'shear_mode': shear_mode.value
            },
            'safety_factors': {
                'flexure': fs_flex,
                'shear': fs_shear,
                'global': fs_local
            },
            'critical_mode': local_critical_mode,
            'verification': 'VERIFICATO' if fs_local >= 1.0 else 'NON VERIFICATO'
        }
        results['piers_analysis'].append(pier_result)
    
    logger.info("-" * 60)
    
    # Analizza fasce se richiesto
    min_safety_spandrel = np.inf
    min_fs_flex_spandrel = np.inf
    min_fs_shear_spandrel = np.inf
    critical_spandrel = -1
    critical_spandrel_mode = None
    
    if options.verify_spandrels and spandrels:
        logger.info("\nAnalisi fasce:")
        logger.info("ID | L[m] | h[m] | N[kN] | V_demand | M_demand | Mu | Vu | FS")
        logger.info("-" * 60)
        
        for i, spandrel in enumerate(spandrels):
            # Capacità fascia
            Mu_span, flex_details = spandrel.flexure_capacity(mat)
            Vu_span, shear_details = spandrel.shear_capacity(mat)
            
            # Applica fattori di riduzione
            Mu_span *= options.spandrel_strength_factor
            Vu_span *= options.spandrel_strength_factor
            
            # Domanda fascia DA SISTEMA ACCOPPIATO
            if i in spandrel_forces:
                # Forza elastica dalla soluzione
                F_span = abs(spandrel_forces[i])
                V_demand_span = F_span
                
                # Momento secondo schema di vincolo
                L = spandrel.geometry.length
                if spandrel.geometry.constraint == SpandrelConstraint.FIXED_FIXED:
                    M_demand_span = F_span * L / 8
                elif spandrel.geometry.constraint == SpandrelConstraint.PINNED_PINNED:
                    M_demand_span = F_span * L / 4
                else:  # FIXED_PINNED
                    M_demand_span = F_span * L / 6
            else:
                # Stima se non nel sistema
                V_demand_span = V_total * 0.1 / max(1, len(spandrels))
                M_demand_span = V_demand_span * spandrel.geometry.length / 8
            
            # Fattori di sicurezza
            fs_flex_span = Mu_span / M_demand_span if M_demand_span > EPS else 999
            fs_shear_span = Vu_span / V_demand_span if V_demand_span > EPS else 999
            fs_span = min(fs_flex_span, fs_shear_span)
            
            # Traccia modo critico
            local_critical_mode = 'flexure' if fs_flex_span <= fs_shear_span else 'shear'
            
            # Aggiorna minimi globali
            if fs_flex_span < min_fs_flex_spandrel:
                min_fs_flex_spandrel = fs_flex_span
            if fs_shear_span < min_fs_shear_spandrel:
                min_fs_shear_spandrel = fs_shear_span
            
            if fs_span < min_safety_spandrel:
                min_safety_spandrel = fs_span
                critical_spandrel = i
                critical_spandrel_mode = local_critical_mode
            
            logger.info(f"{i+1:2} | {spandrel.geometry.length:4.2f} | "
                       f"{spandrel.geometry.height:4.2f} | {spandrel.axial_load:6.1f} | "
                       f"{V_demand_span:6.1f} | {M_demand_span:7.1f} | "
                       f"{Mu_span:6.1f} | {Vu_span:6.1f} | {fs_span:4.2f}")
            
            # Memorizza risultati
            spandrel_result = {
                'id': i + 1,
                'geometry': {
                    'length': spandrel.geometry.length,
                    'height': spandrel.geometry.height,
                    'thickness': spandrel.geometry.thickness,
                    'level': spandrel.geometry.level
                },
                'capacity': {
                    'Mu': Mu_span,
                    'Vu': Vu_span,
                    'mechanism': shear_details['mechanism']
                },
                'demand': {
                    'V': V_demand_span,
                    'M': M_demand_span,
                    'from_system': i in spandrel_forces
                },
                'safety_factors': {
                    'flexure': fs_flex_span,
                    'shear': fs_shear_span,
                    'global': fs_span
                },
                'critical_mode': local_critical_mode,
                'verification': 'VERIFICATO' if fs_span >= 1.0 else 'NON VERIFICATO'
            }
            results['spandrels_analysis'].append(spandrel_result)
        
        logger.info("-" * 60)
    
    # Effetti di accoppiamento per output dettagliato
    if options.detailed_output:
        for effect in coupling_effects:
            results['coupling_effects'].append({
                'pier_id': effect.pier_id + 1,
                'V_uncoupled': effect.V_uncoupled,
                'V_coupled': effect.V_coupled,
                'M_uncoupled': effect.M_uncoupled,
                'M_coupled': effect.M_coupled,
                'coupling_ratio': effect.coupling_ratio
            })
    
    # Capacità e verifiche globali
    fs_global = min(min_safety_pier, min_safety_spandrel) if spandrels else min_safety_pier
    
    # Aggiungi info numeriche al metadata
    if options.coupling_model != CouplingModel.NONE and options.consider_spandrels and K_global is not None:
        try:
            cond_K_final = np.linalg.cond(K_global)
            diag_mean = np.mean(np.diag(K_global))
            epsilon_real = options.regularization_factor * diag_mean if diag_mean > 0 else 0
            
            numerical_info = {
                'matrix_condition': cond_K_final,
                'regularization_epsilon': epsilon_real,
                'matrix_size': K_global.shape[0],
                'spandrels_in_system': len(spandrel_forces)
            }
            
            # Aggiungi deformazioni se richiesto
            if options.include_deformation and u_solution is not None:
                numerical_info['max_displacement'] = float(np.max(np.abs(u_solution)))
                numerical_info['mean_displacement'] = float(np.mean(np.abs(u_solution)))
                # Spostamenti per nodo (primi 5 per brevità)
                if len(u_solution) <= 5:
                    numerical_info['displacements'] = u_solution.tolist()
                else:
                    numerical_info['displacements_sample'] = u_solution[:5].tolist()
            
            results['metadata']['numerical_info'] = numerical_info
        except Exception as e:
            logger.debug(f"Impossibile calcolare info numeriche: {e}")
    
    # Popola warnings nel metadata
    if len(warnings_list) > 0:
        results['metadata']['warnings'] = list(set(warnings_list))
    
    # Capacità globale con minimi per modo
    results['global_capacity'] = {
        'total_Mu': total_Mu,
        'total_Vu': total_Vu,
        'min_safety_pier': min_safety_pier,
        'min_safety_spandrel': min_safety_spandrel if spandrels else 999,
        'critical_pier': critical_pier + 1 if critical_pier >= 0 else 0,
        'critical_spandrel': critical_spandrel + 1 if critical_spandrel >= 0 else 0,
        'min_fs_pier': {
            'flexure': min_fs_flex_pier,
            'shear': min_fs_shear_pier
        }
    }
    
    # Aggiungi minimi per fasce se verificate
    if options.verify_spandrels and spandrels:
        results['global_capacity']['min_fs_spandrel'] = {
            'flexure': min_fs_flex_spandrel,
            'shear': min_fs_shear_spandrel
        }
    
    # Determina elemento critico e modo - CORRETTO
    if spandrels and min_safety_spandrel < min_safety_pier:
        critical_element = 'SPANDREL'
        critical_fs = min_safety_spandrel
        critical_mode = critical_spandrel_mode
    else:
        critical_element = 'PIER'
        critical_fs = min_safety_pier
        critical_mode = critical_pier_mode
    
    # Mappa il modo critico per l'output - CORRETTO
    if critical_mode == 'flexure':
        critical_aspect_output = 'Flexure'
    elif critical_mode == 'shear':
        critical_aspect_output = 'Shear'
    else:
        critical_aspect_output = 'N/A'
    
    # Verifications allineate al POR per compatibilità
    results['verifications'] = {
        'safety_factor_global': fs_global,
        'verification': 'VERIFICATO' if fs_global >= 1.0 else 'NON VERIFICATO',
        'critical_element': critical_element,
        'critical_aspect': critical_aspect_output,  # CORRETTO: usa il valore mappato
        'coupling_active': options.consider_spandrels and len(spandrels) > 0,
        'demand': {
            'V_total': V_total,
            'M_total': V_total * np.mean([p.height for p in piers]) if piers else 0
        },
        'safety_factors': {
            'flexure': min_fs_flex_pier,
            'shear': min_fs_shear_pier,
            'global': fs_global
        }
    }
    
    # Log riassunto finale
    logger.info("\n" + "=" * 70)
    logger.info("RIASSUNTO ANALISI PORFLEX v2.1.3")
    logger.info("=" * 70)
    logger.info(f"Capacità globale: Mu={total_Mu:.1f} kNm, Vu={total_Vu:.1f} kN")
    logger.info(f"Fattore sicurezza maschi: {min_safety_pier:.2f}")
    if options.verify_spandrels and spandrels:
        logger.info(f"Fattore sicurezza fasce: {min_safety_spandrel:.2f}")
    logger.info(f"Fattore sicurezza globale: {fs_global:.2f}")
    logger.info(f"Elemento critico: {critical_element}")
    if critical_element == 'PIER':
        logger.info(f"  Maschio critico: #{critical_pier + 1} (modo: {critical_aspect_output})")
    else:
        logger.info(f"  Fascia critica: #{critical_spandrel + 1} (modo: {critical_aspect_output})")
    logger.info(f"Verifica: {results['verifications']['verification']}")
    
    # Info numeriche se disponibili
    if 'numerical_info' in results['metadata'] and results['metadata']['numerical_info']:
        info = results['metadata']['numerical_info']
        logger.info(f"Condizionamento finale K: {info.get('matrix_condition', 0):.2e}")
        logger.info(f"Regolarizzazione ε: {info.get('regularization_epsilon', 0):.2e} kN/m")
        if 'max_displacement' in info:
            logger.info(f"Spostamento max: {info['max_displacement']:.3e} m")
            logger.info(f"Spostamento medio: {info['mean_displacement']:.3e} m")
    
    # Warnings se presenti
    if results['metadata']['warnings']:
        logger.info(f"Avvertimenti: {len(results['metadata']['warnings'])}")
        for w in results['metadata']['warnings'][:3]:
            logger.info(f"  - {w}")
    
    # Modalità critiche nei metadata (aiuta il debug/report)
    results['metadata'].setdefault('diagnostics', {})
    results['metadata']['diagnostics'].update({
        'critical_modes': {
            'pier': critical_pier_mode,
            'spandrel': critical_spandrel_mode if spandrels else None,
            'global': critical_mode
        }
    })
    
    logger.info("=" * 70)
    
    return results