# enums.py - VERSIONE COMPLETA v2.0
"""
Enumerazioni complete per il sistema FEM Muratura secondo NTC 2018.

Questo modulo contiene tutte le enumerazioni utilizzate nel sistema,
organizzate per categoria e con documentazione dettagliata.

Categorie:
- Metodi di analisi
- Tipologie di analisi
- Modi di rottura
- Legami costitutivi
- Cinematismi EC8
- Carichi e combinazioni
- Stati limite
- Livelli di conoscenza
- Verifiche normative
"""

from enum import Enum

# ============================================================================
# DISTRIBUZIONE CARICHI (UNIFICATA per tutti i moduli di analisi)
# ============================================================================

class LoadDistribution(Enum):
    """
    Metodi di distribuzione dei carichi - ENUM UNIFICATA

    Utilizzata da: POR, SAM, PORFLEX, FRAME

    Note:
        - UNIFORM e EQUAL sono equivalenti (distribuzione uniforme)
        - COUPLED è specifico per PORFLEX (accoppiamento maschi-fasce)
    """
    # Metodi base
    AREA = "area"           # Proporzionale all'area tributaria
    EQUAL = "equal"         # Distribuzione uniforme (alias: UNIFORM)
    UNIFORM = "uniform"     # Alias per EQUAL (retrocompatibilità SAM)
    LENGTH = "length"       # Proporzionale alla lunghezza
    STIFFNESS = "stiffness" # Proporzionale alla rigidezza

    # Metodi avanzati (PORFLEX)
    COUPLED = "coupled"     # Accoppiamento elastico maschi-fasce

# Alias per retrocompatibilità con vecchi moduli
LoadDistributionMethod = LoadDistribution  # Alias per SAM

# ============================================================================
# METODI E TIPOLOGIE DI ANALISI
# ============================================================================

class AnalysisMethod(Enum):
    """Metodi di analisi disponibili secondo NTC 2018"""
    FEM = "FEM"                        # Finite Element Method
    POR = "POR"                        # Pushover su modello continuo
    SAM = "SAM"                        # Simplified Analysis Method
    FRAME = "TELAIO_EQUIVALENTE"      # Telaio equivalente
    LIMIT = "ANALISI_LIMITE"          # Analisi limite cinematica
    FIBER = "MODELLO_FIBRE"           # Modello a fibre
    MICRO = "MICRO_MODELLO"           # Micro-modellazione

class AnalysisType(Enum):
    """Tipo di analisi strutturale da eseguire"""
    STATIC = "STATICA"                # Analisi statica lineare
    MODAL = "MODALE"                  # Analisi modale
    PUSHOVER = "PUSHOVER"             # Analisi pushover
    TIME_HISTORY = "TIME_HISTORY"     # Analisi dinamica al passo
    CYCLIC = "CICLICA"                # Analisi ciclica
    DYNAMIC = "DINAMICA"              # Analisi dinamica generica
    NONLINEAR_STATIC = "STATICA_NON_LINEARE"
    NONLINEAR_DYNAMIC = "DINAMICA_NON_LINEARE"

# ============================================================================
# MODI DI ROTTURA E DANNEGGIAMENTO
# ============================================================================

class FailureMode(Enum):
    """Modi di rottura per elementi murari"""
    FLEXURE = "PRESSOFLESSIONE"              # Rottura per pressoflessione
    DIAGONAL_SHEAR = "TAGLIO_DIAGONALE"      # Taglio con fessurazione diagonale
    SLIDING_SHEAR = "TAGLIO_SCORRIMENTO"     # Taglio per scorrimento
    CRUSHING = "SCHIACCIAMENTO"              # Schiacciamento locale
    ROCKING = "RIBALTAMENTO"                 # Ribaltamento rigido
    ARCH = "MECCANISMO_ARCO"                 # Formazione di arco
    COMBINED = "COMBINATO"                   # Meccanismo combinato
    BUCKLING = "INSTABILITA"                 # Instabilità fuori piano
    SPLITTING = "SPLITTING"                  # Splitting verticale

class DamageLevel(Enum):
    """Livelli di danno secondo EMS-98"""
    D0 = "NESSUN_DANNO"              # No damage
    D1 = "DANNO_LIEVE"               # Slight damage
    D2 = "DANNO_MODERATO"            # Moderate damage
    D3 = "DANNO_SEVERO"              # Severe damage
    D4 = "DANNO_MOLTO_SEVERO"       # Very severe damage
    D5 = "COLLASSO"                  # Collapse

# ============================================================================
# LEGAMI COSTITUTIVI
# ============================================================================

class ConstitutiveLaw(Enum):
    """Legami costitutivi per analisi non lineari"""
    LINEAR = "LINEARE"                # Elastico lineare
    BILINEAR = "BILINEARE"           # Bilineare con softening
    PARABOLIC = "PARABOLICO"         # Parabola di Hognestad
    MANDER = "MANDER"                # Modello di Mander (confinamento)
    KENT_PARK = "KENT_PARK"          # Modello Kent-Park
    POPOVICS = "POPOVICS"            # Modello di Popovics
    THORENFELDT = "THORENFELDT"      # Modello di Thorenfeldt
    
    # Modelli aggiuntivi per muratura
    TURNSEK_CACOVIC = "TURNSEK_CACOVIC"     # Modello per taglio
    MOHR_COULOMB = "MOHR_COULOMB"           # Criterio Mohr-Coulomb
    DRUCKER_PRAGER = "DRUCKER_PRAGER"       # Criterio Drucker-Prager

# ============================================================================
# CINEMATISMI SECONDO EC8
# ============================================================================

class KinematicMechanism(Enum):
    """24 Cinematismi di collasso secondo EC8/NTC2018"""
    
    # === Meccanismi fuori piano (più critici) ===
    OVERTURNING_SIMPLE = "RIBALTAMENTO_SEMPLICE"           # Ribaltamento semplice parete
    OVERTURNING_COMPOUND = "RIBALTAMENTO_COMPOSTO"         # Ribaltamento a blocchi
    VERTICAL_FLEXURE = "FLESSIONE_VERTICALE"               # Flessione su asse verticale
    HORIZONTAL_FLEXURE = "FLESSIONE_ORIZZONTALE"           # Flessione su asse orizzontale
    CORNER_OVERTURNING = "RIBALTAMENTO_CANTONALE"          # Ribaltamento del cantonale
    
    # === Meccanismi nel piano ===
    ROCKING_PIER = "ROCKING_MASCHIO"                       # Rocking del maschio murario
    SLIDING_PIER = "SCORRIMENTO_MASCHIO"                   # Scorrimento alla base
    DIAGONAL_CRACKING = "FESSURAZIONE_DIAGONALE"           # Fessurazione diagonale
    FLEXURAL_PIER = "FLESSIONE_MASCHIO"                    # Flessione nel piano
    
    # === Meccanismi di piano ===
    SOFT_STORY = "PIANO_SOFFICE"                           # Meccanismo di piano soffice
    FLOOR_SLIDING = "SCORRIMENTO_PIANO"                    # Scorrimento dell'impalcato
    IN_PLANE_FLOOR = "DEFORMAZIONE_PIANO"                  # Deformazione nel piano
    
    # === Meccanismi locali ===
    ARCH_THRUST = "SPINTA_ARCHI"                           # Spinta non contrastata di archi
    VAULT_MECHANISM = "MECCANISMO_VOLTE"                   # Meccanismi delle volte
    GABLE_OVERTURNING = "RIBALTAMENTO_TIMPANO"             # Ribaltamento del timpano
    CHIMNEY_OVERTURNING = "RIBALTAMENTO_COMIGNOLO"         # Ribaltamento del comignolo
    PARAPET_OVERTURNING = "RIBALTAMENTO_PARAPETTO"         # Ribaltamento del parapetto
    INFILL_EXPULSION = "ESPULSIONE_TAMPONAMENTO"           # Espulsione tamponamenti
    
    # === Meccanismi combinati ===
    WEDGE_SLIDING = "SCORRIMENTO_CUNEO"                    # Scorrimento di cuneo
    LEAF_SEPARATION = "SEPARAZIONE_PARAMENTI"              # Separazione dei paramenti
    HAMMERING = "MARTELLAMENTO"                            # Martellamento tra edifici
    
    # === Meccanismi specifici edifici storici ===
    COLONNADE_ROCKING = "RIBALTAMENTO_COLONNATO"           # Oscillazione colonnato
    DOME_CRACKING = "FESSURAZIONE_CUPOLA"                  # Fessurazione meridiana cupola
    BELL_TOWER_ROCKING = "OSCILLAZIONE_CAMPANILE"          # Oscillazione campanile
    FACADE_DETACHMENT = "DISTACCO_FACCIATA"                # Distacco della facciata
    TRIUMPHAL_ARCH = "ARCO_TRIONFALE"                      # Meccanismo arco trionfale

# ============================================================================
# CARICHI E COMBINAZIONI
# ============================================================================

class LoadType(Enum):
    """Tipologie di carico secondo NTC 2018"""
    # Carichi permanenti
    G1 = "PERMANENTE_STRUTTURALE"    # Peso proprio struttura
    G2 = "PERMANENTE_NON_STRUTTURALE" # Permanenti non strutturali
    
    # Carichi variabili
    Q = "VARIABILE"                   # Carico variabile generico
    QK_CAT_A = "ABITAZIONI"          # Cat. A - Abitazioni
    QK_CAT_B = "UFFICI"              # Cat. B - Uffici
    QK_CAT_C = "AFFOLLAMENTO"        # Cat. C - Luoghi affollati
    QK_CAT_D = "COMMERCIALE"         # Cat. D - Attività commerciali
    QK_CAT_E = "MAGAZZINI"           # Cat. E - Magazzini e archivi
    QK_CAT_F = "RIMESSE"             # Cat. F - Rimesse e parcheggi
    QK_CAT_G = "COPERTURE"           # Cat. G - Coperture
    QK_CAT_H = "COPERTURE_SPECIALI"  # Cat. H - Coperture speciali
    
    # Azioni ambientali
    W = "VENTO"                      # Azione del vento
    S = "NEVE"                       # Carico neve
    T = "TEMPERATURA"                # Variazioni termiche
    
    # Azioni eccezionali
    E = "SISMA"                      # Azione sismica
    F = "INCENDIO"                   # Azione incendio
    A = "ESPLOSIONE"                 # Azione esplosioni
    U = "URTO"                       # Azioni da urto

class LoadCombination(Enum):
    """Combinazioni di carico secondo NTC 2018"""
    SLU_FONDAMENTALE = "SLU_FONDAMENTALE"     # γG1·G1 + γG2·G2 + γQ·Qk
    SLU_SISMICA = "SLU_SISMICA"               # G1 + G2 + E + Σψ2j·Qkj
    SLE_RARA = "SLE_RARA"                     # G1 + G2 + Qk1 + Σψ0j·Qkj
    SLE_FREQUENTE = "SLE_FREQUENTE"           # G1 + G2 + ψ11·Qk1 + Σψ2j·Qkj
    SLE_QUASI_PERMANENTE = "SLE_QUASI_PERMANENTE" # G1 + G2 + Σψ2j·Qkj
    SLU_ECCEZIONALE = "SLU_ECCEZIONALE"       # G1 + G2 + Ad + ψ21·Qk1

# ============================================================================
# STATI LIMITE E LIVELLI PRESTAZIONALI
# ============================================================================

class LimitState(Enum):
    """Stati limite secondo NTC 2018"""
    # Stati limite ultimi
    SLU = "STATO_LIMITE_ULTIMO"              # Generico SLU
    SLC = "STATO_LIMITE_COLLASSO"            # Collasso (TR=975 anni)
    SLV = "STATO_LIMITE_SALVAGUARDIA_VITA"   # Salvaguardia vita (TR=475 anni)
    
    # Stati limite di esercizio
    SLE = "STATO_LIMITE_ESERCIZIO"           # Generico SLE
    SLD = "STATO_LIMITE_DANNO"               # Danno (TR=63 anni)
    SLO = "STATO_LIMITE_OPERATIVITA"         # Operatività (TR=30 anni)

class PerformanceLevel(Enum):
    """Livelli prestazionali secondo FEMA"""
    IO = "IMMEDIATE_OCCUPANCY"       # Immediata occupazione
    DC = "DAMAGE_CONTROL"           # Controllo del danno
    LS = "LIFE_SAFETY"              # Salvaguardia della vita
    CP = "COLLAPSE_PREVENTION"      # Prevenzione del collasso
    C = "COLLAPSE"                  # Collasso

# ============================================================================
# LIVELLI DI CONOSCENZA E FATTORI DI CONFIDENZA
# ============================================================================

class KnowledgeLevel(Enum):
    """Livelli di conoscenza secondo NTC 2018 - C8.5.4"""
    LC1 = "CONOSCENZA_LIMITATA"     # FC = 1.35
    LC2 = "CONOSCENZA_ADEGUATA"     # FC = 1.20
    LC3 = "CONOSCENZA_ACCURATA"     # FC = 1.00

class SurveyType(Enum):
    """Tipologie di indagine per raggiungere LC"""
    VISUAL = "RILIEVO_VISIVO"              # Ispezione visiva
    GEOMETRIC = "RILIEVO_GEOMETRICO"       # Rilievo geometrico completo
    STRUCTURAL = "RILIEVO_STRUTTURALE"     # Rilievo dettagli strutturali
    MATERIAL_LIMITED = "INDAGINI_LIMITATE" # Indagini limitate sui materiali
    MATERIAL_EXTENDED = "INDAGINI_ESTESE"  # Indagini estese sui materiali
    MATERIAL_EXHAUSTIVE = "INDAGINI_ESAUSTIVE" # Indagini esaustive

# ============================================================================
# TIPOLOGIE STRUTTURALI E VERIFICHE
# ============================================================================

class StructuralType(Enum):
    """Tipologia strutturale edificio"""
    ISOLATED_WALL = "PARETE_ISOLATA"         # Parete singola
    CELLULAR = "CELLULARE"                   # Struttura cellulare
    MIXED = "MISTO"                          # Sistema misto
    IRREGULAR = "IRREGOLARE"                 # Struttura irregolare
    HISTORIC = "STORICO"                     # Edificio storico
    MONUMENTAL = "MONUMENTALE"               # Edificio monumentale

class VerificationType(Enum):
    """Tipologie di verifica secondo NTC 2018"""
    # Verifiche statiche
    COMPRESSION = "COMPRESSIONE"             # Verifica a compressione
    FLEXURE = "PRESSOFLESSIONE"             # Verifica a pressoflessione
    SHEAR = "TAGLIO"                        # Verifica a taglio
    BUCKLING = "INSTABILITA"                # Verifica di instabilità
    
    # Verifiche sismiche
    GLOBAL_SEISMIC = "SISMICA_GLOBALE"      # Verifica sismica globale
    LOCAL_MECHANISM = "MECCANISMI_LOCALI"   # Verifica meccanismi locali
    OUT_OF_PLANE = "FUORI_PIANO"            # Verifica fuori piano
    IN_PLANE = "NEL_PIANO"                  # Verifica nel piano
    
    # Verifiche di deformabilità
    DRIFT = "DRIFT_INTERPIANO"              # Verifica drift di interpiano
    DISPLACEMENT = "SPOSTAMENTO"            # Verifica spostamenti
    
    # Verifiche geotecniche
    FOUNDATION = "FONDAZIONI"               # Verifica fondazioni
    SLIDING = "SCORRIMENTO"                 # Verifica scorrimento
    OVERTURNING = "RIBALTAMENTO"            # Verifica ribaltamento globale

# ============================================================================
# CLASSI DI DUTTILITÀ E FATTORI DI COMPORTAMENTO
# ============================================================================

class DuctilityClass(Enum):
    """Classe di duttilità secondo EC8"""
    DCL = "BASSA_DUTTILITA"         # Low ductility
    DCM = "MEDIA_DUTTILITA"         # Medium ductility
    DCH = "ALTA_DUTTILITA"          # High ductility

class BehaviorFactor(Enum):
    """Fattori di comportamento q per muratura"""
    UNREINFORCED_REGULAR = 1.5      # Muratura non armata regolare
    UNREINFORCED_IRREGULAR = 1.0    # Muratura non armata irregolare
    CONFINED_REGULAR = 2.0          # Muratura confinata regolare
    CONFINED_IRREGULAR = 1.5        # Muratura confinata irregolare
    REINFORCED_REGULAR = 2.5        # Muratura armata regolare
    REINFORCED_IRREGULAR = 2.0      # Muratura armata irregolare

# ============================================================================
# ZONE SISMICHE E CATEGORIE DI SUOLO
# ============================================================================

class SeismicZone(Enum):
    """Zone sismiche Italia"""
    ZONE_1 = 1  # ag > 0.25g - Alta sismicità
    ZONE_2 = 2  # 0.15g < ag ≤ 0.25g - Media sismicità
    ZONE_3 = 3  # 0.05g < ag ≤ 0.15g - Bassa sismicità
    ZONE_4 = 4  # ag ≤ 0.05g - Molto bassa sismicità

class SoilCategory(Enum):
    """Categorie di sottosuolo secondo NTC 2018"""
    A = "ROCCIA"                    # Ammassi rocciosi, Vs30 > 800 m/s
    B = "DEPOSITI_DENSI"           # Depositi molto densi, 360 < Vs30 ≤ 800 m/s
    C = "DEPOSITI_MEDIO_DENSI"     # Depositi mediamente densi, 180 < Vs30 ≤ 360 m/s
    D = "DEPOSITI_SCIOLTI"         # Depositi sciolti, Vs30 ≤ 180 m/s
    E = "TERRENI_TIPO_C_D"        # Terreni C o D con spessore < 20m

class TopographicCategory(Enum):
    """Categorie topografiche secondo NTC 2018"""
    T1 = "PIANEGGIANTE"            # Superficie pianeggiante
    T2 = "PENDIO"                  # Pendii con inclinazione > 15°
    T3 = "CRESTA_RILIEVO"         # Creste di rilievi
    T4 = "CRESTA_APPUNTITA"       # Creste molto appuntite

# ============================================================================
# STATI DI PROGETTO E FASI COSTRUTTIVE
# ============================================================================

class DesignSituation(Enum):
    """Situazioni di progetto secondo NTC 2018"""
    PERSISTENT = "PERSISTENTE"      # Situazione persistente
    TRANSIENT = "TRANSITORIA"       # Situazione transitoria
    ACCIDENTAL = "ECCEZIONALE"      # Situazione eccezionale
    SEISMIC = "SISMICA"            # Situazione sismica

class ConstructionPhase(Enum):
    """Fasi costruttive"""
    EXISTING = "ESISTENTE"          # Stato esistente
    DEMOLITION = "DEMOLIZIONE"      # Fase di demolizione
    STRENGTHENING = "RINFORZO"      # Fase di rinforzo
    CONSTRUCTION = "COSTRUZIONE"    # Fase di costruzione
    COMPLETED = "COMPLETATO"        # Stato finale

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_fc_from_knowledge_level(level: KnowledgeLevel) -> float:
    """
    Restituisce il fattore di confidenza per il livello di conoscenza.
    
    Args:
        level: Livello di conoscenza
        
    Returns:
        Fattore di confidenza FC
    """
    fc_values = {
        KnowledgeLevel.LC1: 1.35,
        KnowledgeLevel.LC2: 1.20,
        KnowledgeLevel.LC3: 1.00
    }
    return fc_values[level]

def get_behavior_factor(structural_type: str, regularity: bool = True) -> float:
    """
    Restituisce il fattore di comportamento q.
    
    Args:
        structural_type: Tipo di struttura ('unreinforced', 'confined', 'reinforced')
        regularity: True se struttura regolare
        
    Returns:
        Fattore di comportamento q
    """
    q_values = {
        ('unreinforced', True): 1.5,
        ('unreinforced', False): 1.0,
        ('confined', True): 2.0,
        ('confined', False): 1.5,
        ('reinforced', True): 2.5,
        ('reinforced', False): 2.0,
    }
    return q_values.get((structural_type, regularity), 1.0)

def get_load_combination_factors(combination: LoadCombination) -> dict:
    """
    Restituisce i coefficienti per la combinazione di carico.
    
    Args:
        combination: Tipo di combinazione
        
    Returns:
        Dizionario con i coefficienti γ e ψ
    """
    if combination == LoadCombination.SLU_FONDAMENTALE:
        return {
            'gamma_G1': 1.3,    # Sfavorevole
            'gamma_G2': 1.5,    # Sfavorevole
            'gamma_Q': 1.5,     # Sfavorevole
            'psi_0': 0.7,       # Combinazione
            'psi_1': 0.5,       # Frequente
            'psi_2': 0.3        # Quasi permanente
        }
    elif combination == LoadCombination.SLU_SISMICA:
        return {
            'gamma_G1': 1.0,
            'gamma_G2': 1.0,
            'gamma_E': 1.0,
            'psi_2': 0.3        # Quasi permanente per variabili
        }
    elif combination == LoadCombination.SLE_RARA:
        return {
            'gamma_G1': 1.0,
            'gamma_G2': 1.0,
            'gamma_Q': 1.0,
            'psi_0': 0.7
        }
    else:
        return {}

# ============================================================================
# MAPPING E CONVERSIONI
# ============================================================================

# Mapping italiano-inglese per compatibilità
ITALIAN_TO_ENGLISH = {
    'RIBALTAMENTO_SEMPLICE': 'SIMPLE_OVERTURNING',
    'PRESSOFLESSIONE': 'FLEXURE',
    'TAGLIO_DIAGONALE': 'DIAGONAL_SHEAR',
    'TAGLIO_SCORRIMENTO': 'SLIDING_SHEAR',
    'SCHIACCIAMENTO': 'CRUSHING',
    'MECCANISMO_ARCO': 'ARCH_MECHANISM'
}

# Mapping per categorie di importanza edifici
IMPORTANCE_CLASS = {
    'I': 0.7,    # Costruzioni con presenza occasionale di persone
    'II': 1.0,   # Costruzioni normali
    'III': 1.5,  # Costruzioni importanti
    'IV': 2.0    # Costruzioni strategiche
}

# Return periods for limit states (years)
RETURN_PERIODS = {
    LimitState.SLO: 30,
    LimitState.SLD: 50,
    LimitState.SLV: 475,
    LimitState.SLC: 975
}