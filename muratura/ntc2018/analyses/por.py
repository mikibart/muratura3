"""
Modulo per l'analisi POR (Pier Only Resistance) secondo NTC 2018
Architetto Michelangelo Bartolotta - Via Domenico Scinà n. 28, 90139 Palermo

Versione 1.0.2 - Gennaio 2025

Analisi sismica di pareti murarie con metodo POR conforme a:
- NTC 2018 §7.8.2.2 (Modellazione e analisi)
- Circolare 617/2019 §C8.7.1 (Criteri di applicazione)
- Tab. 7.8.II NTC 2018 (Fattore di forma b)
- Tab. C8.5.I Circolare (Parametri meccanici)

Caratteristiche principali:
- Modello di roccheggio con stress block (σ_max = 0.85·fcd)
- Taglio per fessurazione diagonale e scorrimento
- Gestione completa modi di rottura (FLEXURE, SHEAR, CRUSHING, NO_CAPACITY)
- Modelli di momento globale configurabili (uniform/triangular/at_top)
- Distribuzione carichi avanzata (area/length/stiffness)
- Altezza efficace con pesatura per domande reali
- Validazione robusta e tracciabilità completa

Note operative:
- Carichi in input assumono peso proprio già incluso
- Resistenza laterale annullata per N ≤ 0 (trazione)
- Fattore b limitato a 0.70 per snellezze h0/l > 2.0
- Configurare logging per dettagli analisi: logging.getLogger(__name__).setLevel(logging.INFO)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import centralizzato da enums.py
from ..enums import LoadDistribution

logger = logging.getLogger(__name__)

# Tolleranza numerica per confronti con zero
EPS = 1e-9

# ========================= ENUMERAZIONI =========================

class FailureMode(Enum):
    """Modi di rottura secondo NTC 2018 §7.8.2.2"""
    FLEXURE = "Rottura per pressoflessione/roccheggio"
    DIAGONAL_SHEAR = "Rottura per taglio diagonale"
    SLIDING_SHEAR = "Rottura per scorrimento"
    CRUSHING = "Schiacciamento"
    NO_CAPACITY = "Assenza di capacità (trazione)"

class MomentModel(Enum):
    """Modelli per il calcolo del momento ribaltante globale"""
    UNIFORM = "uniform"      # Distribuzione uniforme: M = 0.5 * V * h
    TRIANGULAR = "triangular"  # Distribuzione triangolare: M = (2/3) * V * h
    AT_TOP = "at_top"       # Forza in sommità: M = V * h
    # Alias per retrocompatibilità
    TWO_THIRDS = "triangular"

class HeightAggregation(Enum):
    """Metodi di aggregazione altezza per momento globale"""
    MEAN = "mean"  # Media semplice
    WEIGHTED_BY_V = "weighted_by_V"  # Pesata per taglio reale
    WEIGHTED_BY_AREA = "weighted_by_area"  # Pesata per area

class MaterialType(Enum):
    """Tipologie murarie secondo Tab. C8.5.I"""
    PIETRAME_DISORDINATA = "Muratura in pietrame disordinata"
    PIETRAME_SBOZZATA = "Muratura a conci sbozzati"
    PIETRAME_BUONA = "Muratura in pietre a spacco con buona tessitura"
    BLOCCHI_LATERIZIO = "Muratura in blocchi laterizi semipieni"
    BLOCCHI_CALCESTRUZZO = "Muratura in blocchi di calcestruzzo"
    MATTONI_PIENI = "Muratura in mattoni pieni e malta di calce"

class BoundaryConditions(Enum):
    """Condizioni di vincolo del maschio"""
    CANTILEVER = "cantilever"  # Mensola
    FIXED_FIXED = "fixed-fixed"  # Doppio incastro
    FIXED_PINNED = "fixed-pinned"  # Incastro-cerniera

# LoadDistribution importato da ..enums (vedi import sopra)
# Valori: AREA, EQUAL, UNIFORM, LENGTH, STIFFNESS, COUPLED

# ========================= CLASSI OPZIONI =========================

@dataclass
class AnalysisOptions:
    """
    Opzioni per l'analisi POR con validazione e valori di default
    
    Parametri di sicurezza:
        gamma_m: Coefficiente parziale materiale (default 2.0)
        FC: Fattore di confidenza (default 1.35)
    
    Distribuzione carichi:
        load_distribution: Metodo distribuzione carichi verticali
        demand_distribution: Metodo distribuzione domande taglio (default 'area')
        
    Modello globale:
        moment_model: Modello momento ribaltante
        height_aggregation: Metodo aggregazione altezza efficace
        
    Parametri materiale:
        shear_cap_factor: Coefficiente limite superiore taglio (default 0.065)
        mu: Override coefficiente attrito (optional)
        stiffness_uses_design_modulus: Usa Ed invece di E per rigidezza (default False)
        
    Output:
        include_self_weight: Considera peso proprio in aggiunta ai carichi (default False)
        detailed_warnings: Include warning estesi nell'output (default True)
    """
    # Coefficienti di sicurezza
    gamma_m: float = 2.0
    FC: float = 1.35
    
    # Distribuzione carichi e domande
    load_distribution: str = 'area'
    demand_distribution: str = 'area'  # Per domande taglio locali
    
    # Modello globale
    moment_model: str = 'uniform'
    height_aggregation: str = 'mean'
    
    # Parametri materiale
    shear_cap_factor: float = 0.065
    mu: Optional[float] = None
    stiffness_uses_design_modulus: bool = False
    
    # Opzioni output
    include_self_weight: bool = False
    detailed_warnings: bool = True
    
    def __post_init__(self):
        """Validazione opzioni"""
        if self.gamma_m <= 0:
            raise ValueError(f"gamma_m deve essere > 0, ricevuto {self.gamma_m}")
        if self.FC < 1:
            raise ValueError(f"FC deve essere >= 1, ricevuto {self.FC}")
        if self.shear_cap_factor <= 0:
            raise ValueError(f"shear_cap_factor deve essere > 0")
        if self.mu is not None and (self.mu < 0 or self.mu > 1):
            raise ValueError(f"mu deve essere in [0,1], ricevuto {self.mu}")
        
        # Validazione metodi
        valid_load_methods = [m.value for m in LoadDistribution]
        if self.load_distribution not in valid_load_methods:
            raise ValueError(f"load_distribution deve essere in {valid_load_methods}")
        if self.demand_distribution not in valid_load_methods:
            raise ValueError(f"demand_distribution deve essere in {valid_load_methods}")
        
        valid_moment_models = [m.value for m in MomentModel if m != MomentModel.TWO_THIRDS]
        if self.moment_model not in valid_moment_models:
            raise ValueError(f"moment_model deve essere in {valid_moment_models}")
        
        valid_height_methods = [m.value for m in HeightAggregation]
        if self.height_aggregation not in valid_height_methods:
            raise ValueError(f"height_aggregation deve essere in {valid_height_methods}")

# ========================= CLASSI GEOMETRIA =========================

@dataclass
class GeometryPier:
    """Geometria del maschio murario con validazione"""
    length: float  # Lunghezza [m]
    height: float  # Altezza [m]
    thickness: float  # Spessore [m]
    h0: float = field(init=False)  # Altezza di taglio [m]
    area: float = field(init=False)  # Area trasversale [m²]
    shape_factor: float = field(init=False)  # Fattore di forma b
    boundary_conditions: str = "cantilever"  # Condizioni di vincolo
    
    def __post_init__(self):
        """Calcola proprietà derivate con validazione"""
        # Validazione geometria
        if self.length <= 0:
            raise ValueError(f"Lunghezza maschio deve essere > 0, ricevuto {self.length}")
        if self.height <= 0:
            raise ValueError(f"Altezza maschio deve essere > 0, ricevuto {self.height}")
        if self.thickness <= 0:
            raise ValueError(f"Spessore maschio deve essere > 0, ricevuto {self.thickness}")
        
        # Validazione vincoli
        valid_conditions = [bc.value for bc in BoundaryConditions]
        if self.boundary_conditions not in valid_conditions:
            logger.warning(f"Condizione di vincolo '{self.boundary_conditions}' non riconosciuta, uso 'cantilever'")
            self.boundary_conditions = BoundaryConditions.CANTILEVER.value
        
        # Area trasversale
        self.area = self.length * self.thickness
        
        # Altezza di taglio secondo NTC 2018 §7.8.2.2.1
        if self.boundary_conditions == BoundaryConditions.CANTILEVER.value:
            self.h0 = self.height
        elif self.boundary_conditions == BoundaryConditions.FIXED_FIXED.value:
            self.h0 = self.height / 2
        else:  # FIXED_PINNED
            self.h0 = 2 * self.height / 3
        
        # Fattore di forma b secondo Tab. 7.8.II NTC 2018
        self.shape_factor = self._calculate_shape_factor()
    
    def _calculate_shape_factor(self) -> float:
        """
        Calcola fattore di forma secondo Tab. 7.8.II NTC 2018
        Valori tabellari discreti per intervalli di snellezza
        """
        h_l_ratio = self.h0 / self.length
        
        # Tabella 7.8.II - Valori discreti conformi NTC
        if self.boundary_conditions == BoundaryConditions.FIXED_FIXED.value:
            # Doppio incastro
            if h_l_ratio <= 1.0:
                b = 1.0
            elif h_l_ratio <= 1.5:
                b = 0.85
            elif h_l_ratio <= 2.0:
                b = 0.70
            else:
                b = 0.70  # Valore limite
                logger.warning(f"Snellezza h0/l = {h_l_ratio:.2f} fuori range Tab. 7.8.II, uso b={b}")
        elif self.boundary_conditions == BoundaryConditions.CANTILEVER.value:
            # Mensola
            if h_l_ratio <= 1.0:
                b = 1.0
            elif h_l_ratio <= 1.5:
                b = 0.80
            elif h_l_ratio <= 2.0:
                b = 0.70
            else:
                b = 0.70  # Valore limite
                logger.warning(f"Snellezza h0/l = {h_l_ratio:.2f} fuori range Tab. 7.8.II, uso b={b}")
        else:
            # Fixed-pinned: uso valori intermedi tra fixed-fixed e cantilever
            if h_l_ratio <= 1.0:
                b = 1.0
            elif h_l_ratio <= 1.5:
                b = 0.82  # Media tra 0.85 e 0.80
            elif h_l_ratio <= 2.0:
                b = 0.70
            else:
                b = 0.70
                logger.warning(f"Snellezza h0/l = {h_l_ratio:.2f} fuori range, uso b={b}")
        
        return b

# ========================= CLASSI MATERIALI =========================

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
    
    def __post_init__(self):
        """Validazione proprietà materiale"""
        if self.fm <= 0:
            raise ValueError(f"Resistenza a compressione fm deve essere > 0")
        if self.tau0 <= 0:
            raise ValueError(f"Resistenza a taglio tau0 deve essere > 0")
        if self.E <= 0:
            raise ValueError(f"Modulo elastico E deve essere > 0")
        if self.G <= 0:
            raise ValueError(f"Modulo di taglio G deve essere > 0")
        if self.mu < 0 or self.mu > 1:
            raise ValueError(f"Coefficiente di attrito mu deve essere in [0,1]")
    
    @classmethod
    def from_type(cls, material_type: MaterialType, LC: int = 1) -> 'MaterialProperties':
        """Crea proprietà da tipologia muraria e livello di conoscenza"""
        # Validazione LC
        if LC not in [1, 2, 3]:
            raise ValueError(f"Livello di conoscenza LC deve essere 1, 2 o 3, ricevuto {LC}")
        
        # Valori medi da Tab. C8.5.I - Completo con BLOCCHI_CALCESTRUZZO
        properties = {
            MaterialType.PIETRAME_DISORDINATA: {
                'fm': [1.0, 1.8], 'tau0': [0.020, 0.032], 
                'E': [690, 1050], 'G': [230, 350], 'w': 19
            },
            MaterialType.PIETRAME_SBOZZATA: {
                'fm': [2.0, 3.0], 'tau0': [0.035, 0.051],
                'E': [1020, 1440], 'G': [340, 480], 'w': 20
            },
            MaterialType.PIETRAME_BUONA: {
                'fm': [2.6, 3.8], 'tau0': [0.056, 0.074],
                'E': [1500, 1980], 'G': [500, 660], 'w': 21
            },
            MaterialType.BLOCCHI_LATERIZIO: {
                'fm': [5.0, 8.0], 'tau0': [0.080, 0.170],
                'E': [3500, 5600], 'G': [875, 1400], 'w': 12
            },
            MaterialType.BLOCCHI_CALCESTRUZZO: {  # Aggiunto
                'fm': [3.0, 5.0], 'tau0': [0.060, 0.090],
                'E': [2700, 3600], 'G': [1080, 1440], 'w': 14
            },
            MaterialType.MATTONI_PIENI: {
                'fm': [2.4, 4.0], 'tau0': [0.060, 0.092],
                'E': [1200, 1800], 'G': [400, 600], 'w': 18
            }
        }
        
        if material_type not in properties:
            raise ValueError(f"Tipologia muraria {material_type.value} non supportata")
        
        props = properties[material_type]
        # Interpola in base al livello di conoscenza
        factor = (LC - 1) / 2  # LC1=0, LC2=0.5, LC3=1
        
        return cls(
            fm=props['fm'][0] + factor * (props['fm'][1] - props['fm'][0]),
            tau0=props['tau0'][0] + factor * (props['tau0'][1] - props['tau0'][0]),
            E=props['E'][0] + factor * (props['E'][1] - props['E'][0]),
            G=props['G'][0] + factor * (props['G'][1] - props['G'][0]),
            w=props['w'],
            material_type=material_type
        )
    
    def get_design_values(self, gamma_m: float = 2.0, FC: float = 1.35) -> Dict:
        """Calcola valori di progetto secondo NTC 2018 §4.5.6.1"""
        # Validazione coefficienti di sicurezza
        if gamma_m <= 0:
            raise ValueError(f"Coefficiente gamma_m deve essere > 0")
        if FC < 1:
            raise ValueError(f"Fattore di confidenza FC deve essere >= 1")
        
        return {
            'fmd': self.fm / gamma_m,  # Resistenza di progetto a compressione
            'fvd0': self.tau0 / gamma_m,  # Resistenza base a taglio (senza compressione)
            'fcd': self.fm / (gamma_m * FC),  # Per verifica schiacciamento
            'Ed': self.E / gamma_m,
            'Gd': self.G / gamma_m,
            'mu': self.mu
        }

# ========================= CLASSE ANALISI POR =========================

@dataclass
class PORPier:
    """Maschio murario per analisi POR con modelli conformi NTC/EC6"""
    geometry: GeometryPier
    axial_load: float = 0.0  # Carico assiale N [kN]
    material: Optional[MaterialProperties] = None
    
    def flexure_capacity_rocking(self, mat: Dict) -> Tuple[float, FailureMode, Dict]:
        """
        Capacità flessionale per roccheggio secondo NTC 2018 §7.8.2.2.1
        Modello con zona compressa al piede e apertura fessura in sommità
        Returns: (Mu [kNm], modo_flessione, dettagli calcolo)
        """
        A = self.geometry.area * 1e6  # mm²
        l = self.geometry.length * 1e3  # mm
        t = self.geometry.thickness * 1e3  # mm
        N = self.axial_load * 1e3  # N
        
        # Gestione trazione: nessuna capacità a roccheggio
        if N <= 0:
            logger.warning("Carico assiale di trazione o nullo: nessuna capacità a roccheggio")
            return 0.0, FailureMode.NO_CAPACITY, {
                'failure': 'NO_CAPACITY',
                'reason': 'Assenza di compressione',
                'N': self.axial_load
            }
        
        sigma0 = N / A  # MPa - tensione media
        fcd = mat['fcd']  # MPa - resistenza di progetto
        
        # Verifica schiacciamento
        if sigma0 >= 0.85 * fcd:
            return 0.0, FailureMode.CRUSHING, {
                'failure': 'CRUSHING',
                'sigma0': sigma0,
                'sigma_lim': 0.85 * fcd
            }
        
        # Modello di roccheggio: equilibrio con distribuzione triangolare/rettangolare
        # Lunghezza zona compressa per distribuzione rettangolare equivalente
        # σ_max = 0.85 * fcd (stress block)
        # N = σ_max * lc * t → lc = N / (σ_max * t)
        
        sigma_max = 0.85 * fcd
        lc = N / (sigma_max * t)  # mm - lunghezza zona compressa
        
        # Limite fisico: lc non può superare la lunghezza del maschio
        if lc > l:
            # Distribuzione uniforme su tutta la sezione
            lc = l
            sigma_max = N / (l * t)
        
        # Braccio della risultante di compressione rispetto al centro
        # Per distribuzione rettangolare: braccio = (l/2 - lc/2)
        lever_arm = (l - lc) / 2  # mm
        
        # Momento ultimo
        Mu = N * lever_arm  # Nmm
        
        details = {
            'N': self.axial_load,  # kN
            'sigma0': sigma0,  # MPa
            'sigma_max': sigma_max,  # MPa
            'lc_m': lc / 1e3,  # m - lunghezza zona compressa
            'lever_arm_m': lever_arm / 1e3,  # m - braccio
            'l_m': l / 1e3,  # m - lunghezza maschio
            't_m': t / 1e3,  # m - spessore maschio
            'fcd': fcd,  # MPa
            'units': {
                'N': 'kN',
                'sigma': 'MPa',
                'lengths': 'm'
            }
        }
        
        return Mu / 1e6, FailureMode.FLEXURE, details  # kNm
    
    def shear_capacity_diagonal(self, mat: Dict, shear_cap_factor: float = 0.065) -> Tuple[float, Dict]:
        """
        Taglio per fessurazione diagonale §7.8.2.2.2
        
        Args:
            mat: Dizionario proprietà materiale di progetto
            shear_cap_factor: Coefficiente per limite superiore (default 0.065)
        
        Returns: (Vt [kN], dettagli)
        """
        A = self.geometry.area * 1e6  # mm²
        l = self.geometry.length * 1e3  # mm
        t = self.geometry.thickness * 1e3  # mm
        b = self.geometry.shape_factor
        N = self.axial_load * 1e3  # N
        
        # Clamp tensione normale a valori non negativi
        sigma_n = max(0, N / A) if A > 0 else 0  # MPa
        
        # Parametri di resistenza
        fvd0 = mat['fvd0']  # Resistenza base a taglio
        
        # Formula NTC - CORREZIONE DEL BUG NELLA FORMULA
        # La formula corretta è: ft = fvd0 * sqrt(1 + σn/(1.5 * fvd0))
        
        if fvd0 > EPS:
            term = 1 + sigma_n / (1.5 * fvd0)
            term = max(EPS, term)  # Clamp per sicurezza numerica con tolleranza
            ft = fvd0 * np.sqrt(term)
        else:
            ft = 0
        
        # Limite superiore secondo normativa
        ft_max = shear_cap_factor * mat.get('fmd', mat['fcd'] * 2)
        if ft > ft_max:
            logger.info(f"Taglio diagonale limitato: ft={ft:.3f} > ft_max={ft_max:.3f} MPa")
            ft = ft_max
        
        Vt = l * t * ft * b / 1e3  # kN
        
        details = {
            'mechanism': 'Taglio diagonale',
            'ft': ft,  # MPa
            'sigma_n': sigma_n,  # MPa
            'fvd0': fvd0,  # MPa
            'b': b,  # adimensionale
            'ft_max': ft_max,  # MPa
            'capped': ft == ft_max,
            'units': {
                'stresses': 'MPa',
                'b': 'adimensionale'
            }
        }
        
        return Vt, details
    
    def shear_capacity_sliding(self, mat: Dict) -> Tuple[float, Dict]:
        """
        Taglio per scorrimento §7.8.2.2.3
        Returns: (Vs [kN], dettagli)
        """
        A = self.geometry.area * 1e6  # mm²
        N = self.axial_load * 1e3  # N
        
        # Clamp tensione normale a valori non negativi
        sigma_n = max(0, N / A) if A > 0 else 0  # MPa
        
        # Formula: Vs = A * (fvd0 + μ * σn)
        fvd0 = mat['fvd0']  # Resistenza base
        mu = mat['mu']  # Coefficiente di attrito
        
        fv = fvd0 + mu * sigma_n
        Vs = A * fv / 1e3  # kN
        
        details = {
            'mechanism': 'Scorrimento',
            'fv': fv,  # MPa
            'sigma_n': sigma_n,  # MPa
            'fvd0': fvd0,  # MPa
            'mu': mu,  # adimensionale
            'units': {
                'stresses': 'MPa',
                'mu': 'adimensionale'
            }
        }
        
        return Vs, details
    
    def shear_capacity(self, mat: Dict, options: Dict = None) -> Tuple[float, FailureMode, Dict]:
        """
        Capacità a taglio minima tra i meccanismi
        
        Args:
            mat: Proprietà materiale di progetto
            options: Opzioni aggiuntive (shear_cap_factor, mu override)
        
        Returns: (Vu [kN], modo rottura, dettagli)
        """
        if options is None:
            options = {}
        
        # Se in trazione, nessuna capacità
        if self.axial_load <= 0:
            return 0.0, FailureMode.NO_CAPACITY, {
                'reason': 'Assenza di compressione',
                'N': self.axial_load
            }
        
        # Override coefficiente di attrito se specificato
        if 'mu' in options:
            mat_copy = mat.copy()
            mat_copy['mu'] = options['mu']
            mat = mat_copy
        
        # Capacità taglio diagonale con cap factor personalizzabile
        shear_cap_factor = options.get('shear_cap_factor', 0.065)
        Vt, details_t = self.shear_capacity_diagonal(mat, shear_cap_factor)
        
        # Capacità taglio per scorrimento
        Vs, details_s = self.shear_capacity_sliding(mat)
        
        if Vt <= Vs:
            return Vt, FailureMode.DIAGONAL_SHEAR, details_t
        else:
            return Vs, FailureMode.SLIDING_SHEAR, details_s
    
    def failure_mode_analysis(self, Mu: float, Vu: float, 
                            flexure_mode: FailureMode,
                            shear_mode: FailureMode) -> Tuple[FailureMode, float]:
        """
        Determina modo di rottura governante e taglio associato alla flessione
        Gestisce correttamente CRUSHING e NO_CAPACITY
        
        Returns: (modo rottura governante, V_associato)
        """
        h0 = self.geometry.h0
        
        # Casi speciali con priorità massima
        if flexure_mode == FailureMode.CRUSHING:
            return FailureMode.CRUSHING, 0
        
        if flexure_mode == FailureMode.NO_CAPACITY or shear_mode == FailureMode.NO_CAPACITY:
            return FailureMode.NO_CAPACITY, 0
        
        # Calcolo V_flex
        V_flex = Mu / h0 if h0 > 0 and Mu > 0 else 0
        
        # Confronto capacità standard
        if Vu < V_flex:
            # Rottura a taglio governa - usa il modo specifico
            return shear_mode, Vu
        else:
            # Rottura a flessione governa
            return FailureMode.FLEXURE, V_flex

# ========================= FUNZIONI DI SUPPORTO =========================

def identify_piers_from_wall(wall_data: Dict) -> List[GeometryPier]:
    """
    Identifica maschi murari dalla geometria della parete con validazione robusta
    """
    piers = []
    
    # Validazione wall_data
    if 'piers' in wall_data:
        # Maschi definiti esplicitamente
        for i, pier_data in enumerate(wall_data['piers']):
            # Priorità thickness: pier → wall → errore
            if 'thickness' in pier_data:
                thickness = pier_data['thickness']
            elif 'thickness' in wall_data:
                thickness = wall_data['thickness']
            else:
                raise ValueError(f"Spessore non definito per maschio {i+1}")
            
            try:
                pier = GeometryPier(
                    length=pier_data['length'],
                    height=pier_data['height'],
                    thickness=thickness,
                    boundary_conditions=pier_data.get('boundary_conditions', 'cantilever')
                )
                piers.append(pier)
            except ValueError as e:
                logger.error(f"Errore creazione maschio {i+1}: {e}")
                raise
    else:
        # Parete piena senza aperture
        required = ['length', 'height', 'thickness']
        for field in required:
            if field not in wall_data:
                raise ValueError(f"Campo richiesto '{field}' mancante in wall_data")
        
        pier = GeometryPier(
            length=wall_data['length'],
            height=wall_data['height'],
            thickness=wall_data['thickness'],
            boundary_conditions=wall_data.get('boundary_conditions', 'cantilever')
        )
        piers.append(pier)
    
    return piers

def distribute_loads(total_load: float, piers: List[GeometryPier], 
                    method: str = 'area', material: MaterialProperties = None,
                    options: Dict = None) -> List[float]:
    """
    Distribuisce carichi verticali sui maschi con gestione robusta
    
    Args:
        total_load: Carico totale da distribuire [kN]
        piers: Lista dei maschi
        method: Metodo di distribuzione
        material: Proprietà materiale (necessario per 'stiffness')
        options: Opzioni aggiuntive per calcolo rigidezza
    """
    if not piers:
        logger.error("Nessun maschio definito per distribuzione carichi")
        return []
    
    # Validazione metodo
    valid_methods = [m.value for m in LoadDistribution]
    if method not in valid_methods:
        logger.warning(f"Metodo '{method}' non valido, uso 'area'")
        method = LoadDistribution.AREA.value
    
    if method == LoadDistribution.AREA.value:
        total_area = sum(p.area for p in piers)
        if total_area <= 0:
            logger.error(f"Area totale maschi = {total_area} <= 0")
            logger.warning("Fallback a distribuzione uniforme")
            method = LoadDistribution.EQUAL.value
        else:
            return [total_load * p.area / total_area for p in piers]
    
    if method == LoadDistribution.LENGTH.value:
        total_length = sum(p.length for p in piers)
        if total_length <= 0:
            logger.error(f"Lunghezza totale maschi = {total_length} <= 0")
            method = LoadDistribution.EQUAL.value
        else:
            return [total_load * p.length / total_length for p in piers]
    
    if method == LoadDistribution.STIFFNESS.value:
        # Stima rigidezza flessionale k ∝ E·I/h0³
        # Usa Ed (valore di progetto) se richiesto, altrimenti E (valore medio)
        if material:
            if hasattr(material, 'get_design_values') and options and options.get('stiffness_uses_design_modulus', False):
                try:
                    mat_design = material.get_design_values(options.get('gamma_m', 2.0), options.get('FC', 1.35))
                    E = mat_design['Ed']
                    logger.debug("Distribuzione STIFFNESS: usando Ed (modulo di progetto)")
                except:
                    E = material.E
                    logger.warning("Errore calcolo Ed, uso E (modulo medio)")
            else:
                E = material.E
                logger.debug("Distribuzione STIFFNESS: usando E (modulo medio)")
        else:
            E = 1.0  # Distribuzione relativa
            logger.debug("Distribuzione STIFFNESS: rapporti relativi (E=1)")
        
        stiffnesses = []
        for p in piers:
            I = p.thickness * p.length**3 / 12  # Momento d'inerzia
            k = E * I / (p.h0**3) if p.h0 > 0 else 0
            stiffnesses.append(k)
        
        total_stiff = sum(stiffnesses)
        if total_stiff <= 0:
            logger.warning("Rigidezza totale nulla, fallback a distribuzione uniforme")
            method = LoadDistribution.EQUAL.value
        else:
            return [total_load * s / total_stiff for s in stiffnesses]
    
    # Equal distribution (default fallback)
    n_piers = len(piers)
    return [total_load / n_piers for _ in piers]

def calculate_global_moment(V_total: float, piers: List[GeometryPier],
                           pier_v_demands: List[float] = None,
                           model: str = 'uniform',
                           height_aggregation: str = 'mean') -> Tuple[float, float]:
    """
    Calcola momento ribaltante globale con diversi modelli
    
    Args:
        V_total: Taglio totale [kN]
        piers: Lista dei maschi
        pier_v_demands: Domande di taglio reali per maschio [kN] (per weighted_by_V)
        model: Modello di distribuzione ('uniform', 'triangular', 'at_top')
        height_aggregation: Metodo aggregazione altezza
    
    Returns: (M_total [kNm], h_effective [m])
    """
    if not piers:
        return 0.0, 0.0
    
    # Calcolo altezza efficace
    if height_aggregation == HeightAggregation.WEIGHTED_BY_V.value and pier_v_demands:
        # Pesata per taglio reale
        total_v = sum(pier_v_demands)
        if total_v > 0:
            weights = [v / total_v for v in pier_v_demands]
            h_eff = sum(p.height * w for p, w in zip(piers, weights))
        else:
            h_eff = sum(p.height for p in piers) / len(piers)
    elif height_aggregation == HeightAggregation.WEIGHTED_BY_AREA.value:
        # Pesata per area
        areas = [p.area for p in piers]
        total_area = sum(areas)
        if total_area > 0:
            h_eff = sum(p.height * p.area for p in piers) / total_area
        else:
            h_eff = sum(p.height for p in piers) / len(piers)
    else:
        # Media semplice (default)
        h_eff = sum(p.height for p in piers) / len(piers)
    
    # Gestione alias e retrocompatibilità
    if model == 'two_thirds':
        model = MomentModel.TRIANGULAR.value
        logger.debug("Alias 'two_thirds' mappato a 'triangular'")
    
    # Calcolo momento secondo modello
    if model == MomentModel.AT_TOP.value or model == 'at_top':
        M_total = V_total * h_eff
        factor = 1.0
    elif model == MomentModel.TRIANGULAR.value or model == 'triangular':
        M_total = V_total * h_eff * 2/3
        factor = 2/3
    elif model == MomentModel.UNIFORM.value or model == 'uniform':
        M_total = V_total * h_eff * 0.5
        factor = 0.5
    else:
        logger.warning(f"Modello momento '{model}' non riconosciuto, uso 'uniform'")
        M_total = V_total * h_eff * 0.5
        factor = 0.5
    
    logger.info(f"Momento con modello {model}: M = {factor:.2f} * V * h = {factor:.2f} * {V_total:.1f} * {h_eff:.2f}")
    
    return M_total, h_eff

# ========================= FUNZIONE PRINCIPALE ANALISI =========================

def analyze_por(wall_data: Dict, material: MaterialProperties,
                loads: Dict, options: AnalysisOptions = None) -> Dict:
    """
    Analisi POR completa secondo NTC 2018 con gestione robusta
    
    Args:
        wall_data: Geometria parete {
            'length': float [m],
            'height': float [m], 
            'thickness': float [m],
            'boundary_conditions': str (optional),
            'piers': list of dicts (optional)
        }
        material: Proprietà materiale (MaterialProperties object)
        loads: {
            'vertical': float [kN] (peso proprio incluso di default),
            'horizontal': float [kN]
        }
        options: AnalysisOptions object con parametri di calcolo
    
    Returns:
        Dict con risultati analisi, capacità e verifiche.
        Include campo 'metadata' con flag di condizionamento numerico.
    
    Note:
        - I carichi verticali includono il peso proprio di default
        - Usare include_self_weight=True solo per aggiungere automaticamente il peso proprio geometrico
        - Per N ≤ 0 la resistenza laterale viene annullata
        - Snellezze h0/l > 2.0 usano b = 0.70 (valore limite conservativo)
        - h_eff è l'altezza fisica media dei maschi (NON h0), pesabile con area o taglio
        - FS=999 è un valore sentinel per "divisione evitata/non rilevante"
        - Tolleranza numerica EPS=1e-9 per confronti con zero
    """
    logger.info("=== ANALISI POR - Pier Only Resistance ===")
    logger.info(f"Parete: L={wall_data.get('length','N/A')}m, H={wall_data.get('height','N/A')}m")
    
    # Default options con validazione
    if options is None:
        options = AnalysisOptions()
    
    # Estrai parametri da AnalysisOptions
    gamma_m = options.gamma_m
    FC = options.FC
    load_dist = options.load_distribution
    demand_dist = options.demand_distribution
    moment_model = options.moment_model
    height_agg = options.height_aggregation
    shear_cap_factor = options.shear_cap_factor
    
    # Validazione carichi
    if not isinstance(loads, dict):
        raise TypeError("loads deve essere un dizionario")
    
    V_total = loads.get('horizontal', 0.0)
    N_total = loads.get('vertical', 0.0)
    
    if V_total < 0:
        logger.warning(f"Carico orizzontale negativo V={V_total}, uso valore assoluto")
        V_total = abs(V_total)
    
    # Identifica maschi con gestione errori
    try:
        piers = identify_piers_from_wall(wall_data)
    except Exception as e:
        logger.error(f"Errore identificazione maschi: {e}")
        raise
    
    logger.info(f"Identificati {len(piers)} maschi murari")
    
    # Aggiunta peso proprio se richiesto (dopo aver calcolato piers una sola volta)
    self_weight_added = 0.0
    if options.include_self_weight and material:
        total_volume = sum(p.area * p.height for p in piers)
        self_weight_added = total_volume * getattr(material, 'w', material.weight)  # kN
        N_total += self_weight_added
        logger.info(f"Peso proprio aggiunto: {self_weight_added:.1f} kN (totale N={N_total:.1f} kN)")
    
    # Valori di progetto materiale
    try:
        mat = material.get_design_values(gamma_m, FC)
        # Aggiungi mu se non presente (MaterialProperties da materials.py non lo include)
        if 'mu' not in mat:
            mat['mu'] = getattr(material, 'mu', 0.4)  # Default 0.4 per muratura
    except Exception as e:
        logger.error(f"Errore calcolo valori di progetto: {e}")
        raise

    # Override mu se specificato
    if options.mu is not None:
        mat['mu'] = options.mu
        logger.info(f"Override coefficiente attrito: mu={options.mu}")
    
    # Distribuisci carichi verticali
    options_dict = {
        'gamma_m': gamma_m, 
        'FC': FC, 
        'stiffness_uses_design_modulus': options.stiffness_uses_design_modulus
    }
    
    pier_loads = distribute_loads(N_total, piers, load_dist, material, options_dict)
    
    # Gestione caso area totale = 0
    total_area = sum(p.area for p in piers)
    if total_area <= 0:
        logger.error(f"Area totale maschi = {total_area} <= 0")
        return {
            'error': 'Area totale maschi nulla o negativa',
            'method': 'POR - NTC 2018',
            'verification': 'ERRORE GEOMETRIA'
        }
    
    # Inizializza risultati
    results = {
        'method': 'POR - NTC 2018',
        'wall_geometry': wall_data,
        'material': {
            'type': getattr(material.material_type, 'value', material.material_type) if material.material_type else 'Custom',
            'fm': material.fm if hasattr(material, 'fm') else material.fcm,
            'tau0': material.tau0,
            'E': material.E,
            'G': material.G,
            'w': material.w if hasattr(material, 'w') else material.weight,
            'gamma_m': gamma_m,
            'FC': FC,
            'mu': mat.get('mu', getattr(material, 'mu', 0.4))
        },
        'loads': {
            'vertical': N_total,
            'horizontal': V_total,
            'self_weight_included': options.include_self_weight,
            'self_weight_added_kN': self_weight_added
        },
        'options': {
            'load_distribution': load_dist,
            'demand_distribution': demand_dist,
            'moment_model': moment_model,
            'height_aggregation': height_agg,
            'shear_cap_factor': shear_cap_factor,
            'stiffness_uses_design_modulus': options.stiffness_uses_design_modulus
        },
        'n_piers': len(piers),
        'piers_analysis': [],
        'global_capacity': {},
        'verifications': {},
        'metadata': {
            'version': '1.0.2',
            'analysis_method': 'POR - NTC 2018',
            'ill_conditioned_cases': [],
            'numerical_warnings': [],
            'notes': []  # Per tracciabilità peso proprio e altre note
        }
    }
    
    # Verifica conservazione della somma (DOPO inizializzazione results)
    sum_loads = sum(pier_loads)
    if abs(sum_loads - N_total) > EPS * abs(N_total) if N_total != 0 else EPS:
        delta = abs(sum_loads - N_total)
        results['metadata']['numerical_warnings'].append(
            f"Distribuzione carichi verticali: somma={sum_loads:.6f} vs totale={N_total:.6f} (delta={delta:.2e})"
        )
    
    # Opzioni per capacità taglio
    shear_options = {
        'shear_cap_factor': shear_cap_factor
    }
    if options.mu is not None:
        shear_options['mu'] = options.mu
    
    # Analizza ogni maschio
    total_Mu = 0.0
    total_Vu = 0.0
    min_safety = np.inf
    governing_pier = 0
    critical_mode = None
    
    # Calcola domande di taglio per maschi con metodo specificato (riusa options_dict)
    pier_v_demands = distribute_loads(V_total, piers, demand_dist, material, options_dict)
    
    # Verifica conservazione della somma per domande di taglio
    sum_v_demands = sum(pier_v_demands)
    if abs(sum_v_demands - V_total) > EPS * abs(V_total) if V_total != 0 else EPS:
        delta = abs(sum_v_demands - V_total)
        results['metadata']['numerical_warnings'].append(
            f"Distribuzione domande taglio: somma={sum_v_demands:.6f} vs totale={V_total:.6f} (delta={delta:.2e})"
        )
    
    # Log compatto header
    logger.info("")  # Linea vuota per separazione
    logger.info("Riepilogo maschi:")
    logger.info("ID | L[m] | H[m] | N[kN] | Mu[kNm] | Vu[kN] | Modo | FS | DCR")
    logger.info("-" * 70)
    
    for i, (pier, N) in enumerate(zip(piers, pier_loads)):
        # Crea oggetto maschio
        por_pier = PORPier(pier, N, material)
        
        # Capacità flessionale (roccheggio) con modo
        Mu, flexure_mode, flex_details = por_pier.flexure_capacity_rocking(mat)
        
        # Capacità a taglio con opzioni
        Vu, shear_mode, shear_details = por_pier.shear_capacity(mat, shear_options)
        
        # Modo di rottura governante
        failure_mode, V_associated = por_pier.failure_mode_analysis(
            Mu, Vu, flexure_mode, shear_mode)
        
        # Domanda locale secondo metodo specificato (già calcolata)
        V_demand = pier_v_demands[i]
        M_demand = V_demand * pier.h0
        
        # Fattori di sicurezza locali con gestione divisione per zero
        if M_demand > EPS:
            fs_flex = Mu / M_demand
        else:
            fs_flex = 999.0  # Sentinel per divisione evitata
            results['metadata']['ill_conditioned_cases'].append(f"Maschio {i+1}: M_demand≈0")
        
        if V_demand > EPS:
            fs_shear = Vu / V_demand
        else:
            fs_shear = 999.0  # Sentinel per divisione evitata
            results['metadata']['ill_conditioned_cases'].append(f"Maschio {i+1}: V_demand≈0")
            
        fs_local = min(fs_flex, fs_shear)
        
        # DCR (Demand/Capacity Ratio)
        if Mu > EPS:
            dcr_flex = M_demand / Mu
        else:
            dcr_flex = 999.0  # Sentinel per divisione evitata
            results['metadata']['ill_conditioned_cases'].append(f"Maschio {i+1}: Mu≈0")
            
        if Vu > EPS:
            dcr_shear = V_demand / Vu
        else:
            dcr_shear = 999.0  # Sentinel per divisione evitata
            results['metadata']['ill_conditioned_cases'].append(f"Maschio {i+1}: Vu≈0")
            
        dcr_max = max(dcr_flex, dcr_shear)
        
        # Log compatto per maschio
        logger.info(f"{i+1:2} | {pier.length:4.2f} | {pier.height:4.2f} | "
                   f"{N:6.1f} | {Mu:7.2f} | {Vu:6.1f} | "
                   f"{failure_mode.name:12} | {fs_local:4.2f} | {dcr_max:4.2f}")
        
        if fs_local < min_safety:
            min_safety = fs_local
            governing_pier = i
            critical_mode = failure_mode
        
        # Memorizza risultati maschio
        pier_result = {
            'id': i + 1,
            'geometry': {
                'length': pier.length,
                'height': pier.height,
                'thickness': pier.thickness,
                'area': pier.area,
                'h0': pier.h0,
                'shape_factor': pier.shape_factor,
                'boundary_conditions': pier.boundary_conditions
            },
            'loads': {
                'N': N,
                'V_demand': V_demand,
                'M_demand': M_demand
            },
            'capacity': {
                'Mu': Mu,
                'Vu': Vu,
                'V_associated': V_associated,
                'failure_mode': failure_mode.value,
                'flexure_mode': flexure_mode.value,
                'shear_mode': shear_mode.value
            },
            'details': {
                'flexure': flex_details,
                'shear': shear_details
            },
            'safety_factors': {
                'flexure': fs_flex,
                'shear': fs_shear,
                'global': fs_local
            },
            'dcr': {
                'flexure': dcr_flex,
                'shear': dcr_shear,
                'max': dcr_max
            },
            'verification': 'VERIFICATO' if fs_local >= 1.0 else 'NON VERIFICATO',
            'warnings': []
        }
        
        # Aggiungi warning se necessario
        pier_warnings = []
        if pier.h0 / pier.length > 2.0:
            pier_warnings.append(f"Snellezza elevata h0/l={pier.h0/pier.length:.2f} (b limitato a 0.70)")
        if N <= 0:
            pier_warnings.append("Maschio in trazione - resistenza laterale annullata")
        if flexure_mode == FailureMode.CRUSHING:
            pier_warnings.append("Schiacciamento rilevato")
        if 'capped' in shear_details and shear_details['capped']:
            pier_warnings.append(f"Taglio limitato a ft_max={shear_details['ft_max']:.3f} MPa")
        
        # Warning estesi se richiesti
        if options.detailed_warnings:
            if pier.thickness / pier.length < 0.1:
                pier_warnings.append(f"Spessore molto piccolo t/l={pier.thickness/pier.length:.3f}")
            if pier.height / pier.thickness > 15:
                pier_warnings.append(f"Snellezza fuori piano h/t={pier.height/pier.thickness:.1f}")
        
        pier_result['warnings'] = pier_warnings
        
        results['piers_analysis'].append(pier_result)
        
        # Accumula capacità globali
        total_Mu += Mu
        total_Vu += Vu
    
    logger.info("-" * 70)
    
    # Verifiche globali con altezza efficace e domande reali
    M_total, h_eff = calculate_global_moment(
        V_total, piers, pier_v_demands, moment_model, height_agg)
    
    # Metadati globali per casi ill-conditioned (DOPO calcolo M_total)
    if abs(V_total) < EPS:
        results['metadata']['ill_conditioned_cases'].append("Taglio globale V_total≈0")
    if abs(M_total) < EPS:
        results['metadata']['ill_conditioned_cases'].append("Momento globale M_total≈0")
    
    # Gestione caso non applicabile (domanda nulla)
    is_not_applicable = (abs(V_total) < EPS and abs(M_total) < EPS)
    if is_not_applicable:
        results['metadata']['ill_conditioned_cases'].append("Domanda globale nulla")
    
    # Capacità globale parete (DOPO definizione is_not_applicable)
    if is_not_applicable:
        # Caso non applicabile: nessun modo critico significativo
        critical_mode_value = "N/A"
        governing_pier_id = 0
    else:
        critical_mode_value = critical_mode.value if critical_mode else 'N/A'
        if critical_mode == FailureMode.CRUSHING:
            critical_mode_value = "Schiacciamento"
        governing_pier_id = governing_pier + 1
    
    results['global_capacity'] = {
        'total_Mu': total_Mu,
        'total_Vu': total_Vu,
        'governing_pier': governing_pier_id,
        'critical_mode': critical_mode_value
    }
    
    # Aggiungi note per peso proprio
    if options.include_self_weight and self_weight_added > 0:
        results['metadata']['notes'].append(f"Self weight added: {self_weight_added:.1f} kN")
    
    fs_global_flex = total_Mu / M_total if M_total > EPS else 999.0
    fs_global_shear = total_Vu / V_total if V_total > EPS else 999.0
    fs_global = min(fs_global_flex, fs_global_shear)
    
    # DCR globali
    dcr_global_flex = M_total / total_Mu if total_Mu > EPS else 999.0
    dcr_global_shear = V_total / total_Vu if total_Vu > EPS else 999.0
    
    # Calcola fattore del modello di momento per tracciabilità (con guard EPS uniforme)
    factor_actual = M_total / (V_total * h_eff) if abs(V_total) > EPS and h_eff > EPS else 0.0
    
    # Deduplica metadati ill-conditioned preservando l'ordine
    results['metadata']['ill_conditioned_cases'] = list(dict.fromkeys(results['metadata']['ill_conditioned_cases']))
    
    # Gestione verifiche per caso non applicabile
    if is_not_applicable:
        results['verifications'] = {
            'demand': {
                'V_total': V_total,
                'M_total': M_total,
                'h_effective': h_eff,
                'moment_model': moment_model,
                'height_aggregation': height_agg,
                'moment_factor': factor_actual
            },
            'safety_factors': {
                'flexure': 999.0,  # Valore convenzionale per compatibilità
                'shear': 999.0,
                'global': 999.0
            },
            'dcr': {
                'flexure': 0.0,
                'shear': 0.0,
                'max': 0.0
            },
            'verification': 'NON APPLICABILE',
            'critical_aspect': 'N/A'
        }
    else:
        results['verifications'] = {
            'demand': {
                'V_total': V_total,
                'M_total': M_total,
                'h_effective': h_eff,
                'moment_model': moment_model,
                'height_aggregation': height_agg,
                'moment_factor': factor_actual
            },
            'safety_factors': {
                'flexure': fs_global_flex,
                'shear': fs_global_shear,
                'global': fs_global
            },
            'dcr': {
                'flexure': dcr_global_flex,
                'shear': dcr_global_shear,
                'max': max(dcr_global_flex, dcr_global_shear)
            },
            'verification': 'VERIFICATO' if fs_global >= 1.0 else 'NON VERIFICATO',
            'critical_aspect': 'Flessione' if fs_global_flex < fs_global_shear else 'Taglio'
        }
    
    # Log riassunto finale con fattore M/(V*h_eff)
    logger.info("=== RIASSUNTO ANALISI POR ===")
    if is_not_applicable:
        logger.warning("Domanda globale nulla: analisi non applicabile")
        logger.info(f"Domanda: V={V_total:.1f} kN, M={M_total:.1f} kNm")
        logger.info(f"Verifica: NON APPLICABILE")
    else:
        logger.info(f"Capacità globale: Vu={total_Vu:.1f} kN, Mu={total_Mu:.1f} kNm")
        logger.info(f"Domanda: V={V_total:.1f} kN, M={M_total:.1f} kNm (h_eff={h_eff:.2f}m)")
        logger.info(f"Modello momento: {moment_model} - Fattore M/(V*h) = {factor_actual:.2f}")
        logger.info(f"Fattore sicurezza globale: {fs_global:.2f}")
        logger.info(f"DCR massimo: {results['verifications']['dcr']['max']:.2f}")
        logger.info(f"Modo critico: {results['global_capacity']['critical_mode']}")
        logger.info(f"Verifica: {results['verifications']['verification']}")
    
    return results

# ========================= FUNZIONI DI UTILITÀ E TEST =========================

def format_results(results: Dict, decimal_places: int = 2) -> Dict:
    """
    Formatta i risultati numerici per output leggibile
    """
    import copy
    formatted = copy.deepcopy(results)
    
    # Funzione ricorsiva per formattare
    def format_value(v, dp=decimal_places):
        if isinstance(v, float):
            return round(v, dp)
        elif isinstance(v, dict):
            return {k: format_value(val, dp) for k, val in v.items()}
        elif isinstance(v, list):
            return [format_value(item, dp) for item in v]
        return v
    
    # Applica formattazione
    if 'piers_analysis' in formatted:
        for pier in formatted['piers_analysis']:
            pier['capacity'] = format_value(pier['capacity'])
            pier['loads'] = format_value(pier['loads'])
            pier['safety_factors'] = format_value(pier['safety_factors'])
            pier['dcr'] = format_value(pier['dcr'])
    
    if 'global_capacity' in formatted:
        formatted['global_capacity']['total_Mu'] = round(
            formatted['global_capacity']['total_Mu'], decimal_places)
        formatted['global_capacity']['total_Vu'] = round(
            formatted['global_capacity']['total_Vu'], decimal_places)
    
    if 'verifications' in formatted:
        formatted['verifications'] = format_value(formatted['verifications'])
    
    return formatted

def create_user_report(results: Dict, include_details: bool = False) -> Dict:
    """
    Crea report user-friendly per interfacce/export
    
    Args:
        results: Output di analyze_por
        include_details: Include dettagli tecnici (default False)
    
    Returns:
        Dict con report strutturato per UI/JSON export
    """
    if 'error' in results:
        return {
            'status': 'ERROR',
            'message': results['error'],
            'method': results.get('method', 'POR')
        }
    
    # Estrai dati principali
    verification = results['verifications']['verification']
    fs_global = results['verifications']['safety_factors']['global']
    critical_mode = results['global_capacity']['critical_mode']
    governing_pier = results['global_capacity']['governing_pier']
    
    # Conta warning e rileva schiacciamento (robusta ma snella)
    total_warnings = sum(len(pier['warnings']) for pier in results['piers_analysis'])
    has_crushing = any(
        ('Schiacciamento' in pier['capacity']['failure_mode']) or
        ('Schiacciamento' in pier['capacity']['flexure_mode'])
        for pier in results['piers_analysis']
    )
    
    report = {
        'summary': {
            'verification': verification,
            'safety_factor': round(fs_global, 2),
            'critical_mode': critical_mode,
            'governing_pier': governing_pier,
            'total_warnings': total_warnings,
            'has_crushing': has_crushing
        },
        'global_capacity': {
            'moment_ultimate': round(results['global_capacity']['total_Mu'], 1),
            'shear_ultimate': round(results['global_capacity']['total_Vu'], 1),
            'moment_demand': round(results['verifications']['demand']['M_total'], 1),
            'shear_demand': round(results['verifications']['demand']['V_total'], 1)
        },
        'piers_summary': []
    }
    
    # Riassunto per maschi
    for pier in results['piers_analysis']:
        pier_summary = {
            'id': pier['id'],
            'geometry': f"{pier['geometry']['length']:.1f}×{pier['geometry']['height']:.1f}×{pier['geometry']['thickness']:.1f}m",
            'verification': pier['verification'],
            'safety_factor': round(pier['safety_factors']['global'], 2),
            'failure_mode': pier['capacity']['failure_mode'],
            'warning_count': len(pier['warnings'])
        }
        report['piers_summary'].append(pier_summary)
    
    # Dettagli tecnici se richiesti
    if include_details:
        report['technical_details'] = {
            'material': results['material'],
            'options': results['options'],
            'metadata': results['metadata'],
            'pier_details': results['piers_analysis']
        }
    
    return report

# Export per import
__all__ = [
    # Funzione principale
    'analyze_por',
    # Classi principali
    'PORPier', 
    'GeometryPier',
    'MaterialProperties',
    'AnalysisOptions',
    # Enumerazioni
    'MaterialType',
    'FailureMode',
    'BoundaryConditions',
    'LoadDistribution',
    'MomentModel',
    'HeightAggregation',
    # Funzioni di supporto
    'identify_piers_from_wall',
    'distribute_loads',
    'calculate_global_moment',
    'format_results',
    'create_user_report'
]

# Versioning
__version__ = "1.0.2"
__author__ = "Arch. Michelangelo Bartolotta"
__email__ = "michelangelo.bartolotta@ordine-architetti-ag.it"
__status__ = "Production"

# Changelog
__changelog__ = """
v1.0.2 (Gennaio 2025):
- FIX: Corretta formula taglio diagonale secondo NTC 2018
- NEW: Gestione caso "NON APPLICABILE" per domanda globale nulla
- NEW: critical_mode="N/A" quando non applicabile
- FIX: Deduplica automatica metadati ill-conditioned
- IMPROVE: Chiarezza unità di misura nei dettagli (suffissi _m, _MPa, sezione units)
- IMPROVE: Unità esplicite anche nei dettagli di taglio
- IMPROVE: Tolleranza numerica EPS=1e-9 per confronti con zero
- IMPROVE: Verifica conservazione somme con warning se delta > EPS
- IMPROVE: Logging più consistente (warning per ill-posed, info per cap)
- IMPROVE: Note in metadata per tracciabilità peso proprio
- DOC: Chiarito che h_eff usa altezza fisica (non h0)
- DOC: Specificato che FS=999 è un sentinel per divisione evitata

v1.0.1 (Gennaio 2025):
- Prima versione di produzione
"""