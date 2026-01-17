# floors.py - Modulo Solai e Coperture
"""
Definizione e calcolo di solai e coperture secondo NTC 2018.

Tipologie supportate:
- Solai in laterocemento (travetti + pignatte)
- Solai in legno (travi + tavolato)
- Solai in acciaio (putrelle + tavelloni)
- Solai in c.a. pieno
- Volte (a botte, a crociera, a padiglione)
- Coperture piane e a falde

Calcoli:
- Peso proprio automatico da tipologia e stratigrafia
- Rigidezza nel piano (rigido/flessibile/semi-rigido)
- Distribuzione carichi alle pareti
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math


# ============================================================================
# ENUMERAZIONI
# ============================================================================

class FloorType(Enum):
    """Tipologie di solaio"""
    LATEROCEMENTO = "laterocemento"      # Travetti prefabbricati + pignatte
    LEGNO_SEMPLICE = "legno_semplice"    # Travi in legno + tavolato
    LEGNO_CONNESSO = "legno_connesso"    # Legno con soletta collaborante
    ACCIAIO = "acciaio"                   # Putrelle in acciaio + tavelloni
    CA_PIENO = "ca_pieno"                 # Calcestruzzo armato pieno
    PREDALLES = "predalles"               # Lastre predalles
    VOLTA_BOTTE = "volta_botte"           # Volta a botte
    VOLTA_CROCIERA = "volta_crociera"     # Volta a crociera
    VOLTA_PADIGLIONE = "volta_padiglione" # Volta a padiglione


class FloorStiffness(Enum):
    """Rigidezza del solaio nel piano orizzontale"""
    RIGID = "rigido"           # Solaio infinitamente rigido (soletta >= 5cm)
    SEMI_RIGID = "semi_rigido" # Solaio con rigidezza finita
    FLEXIBLE = "flessibile"    # Solaio deformabile (legno tradizionale)


class RoofType(Enum):
    """Tipologie di copertura"""
    FLAT = "piana"              # Terrazzo/lastrico solare
    SINGLE_PITCH = "una_falda"  # Una falda
    DOUBLE_PITCH = "due_falde"  # Due falde (a capanna)
    FOUR_PITCH = "quattro_falde" # Quattro falde (a padiglione)
    HIP = "a_padiglione"        # Copertura a padiglione


class RoofStructure(Enum):
    """Struttura portante della copertura"""
    TIMBER_TRUSS_SIMPLE = "capriata_semplice"
    TIMBER_TRUSS_PALLADIANA = "capriata_palladiana"
    TIMBER_RAFTERS = "travi_inclinate"
    STEEL_TRUSS = "capriata_acciaio"
    RC_SLAB = "soletta_ca"


# ============================================================================
# PESI MATERIALI (kN/m3)
# ============================================================================

MATERIAL_WEIGHTS = {
    # Calcestruzzi
    'cls_normale': 25.0,        # Calcestruzzo normale
    'cls_leggero': 18.0,        # Calcestruzzo alleggerito
    'cls_armato': 25.0,         # Calcestruzzo armato

    # Laterizi
    'pignatte': 8.0,            # Blocchi in laterizio forato
    'tavelle': 12.0,            # Tavelle in laterizio
    'tavelloni': 10.0,          # Tavelloni

    # Legno
    'legno_abete': 4.5,         # Abete
    'legno_larice': 5.5,        # Larice
    'legno_castagno': 5.8,      # Castagno
    'legno_quercia': 7.0,       # Quercia
    'legno_lamellare': 4.5,     # Legno lamellare

    # Acciaio
    'acciaio': 78.5,            # Acciaio strutturale

    # Finiture
    'massetto': 22.0,           # Massetto cementizio
    'pavimento': 22.0,          # Pavimento generico
    'intonaco': 20.0,           # Intonaco
    'controsoffitto': 0.3,      # kN/m2 (carico superficiale)

    # Impermeabilizzazioni
    'guaina': 0.05,             # kN/m2
    'coibente': 0.02,           # kN/m2 per cm spessore

    # Manti copertura
    'coppi': 0.60,              # kN/m2
    'tegole': 0.50,             # kN/m2
    'lamiera': 0.15,            # kN/m2
}


# ============================================================================
# DATABASE SOLAI TIPICI
# ============================================================================

FLOOR_DATABASE = {
    # Solai in laterocemento
    'LAT_16+4': {
        'type': FloorType.LATEROCEMENTO,
        'description': 'Laterocemento 16+4',
        'total_height': 0.20,       # m
        'joist_height': 0.16,       # m
        'slab_thickness': 0.04,     # m
        'joist_spacing': 0.50,      # m
        'self_weight': 2.80,        # kN/m2
        'stiffness': FloorStiffness.RIGID
    },
    'LAT_20+4': {
        'type': FloorType.LATEROCEMENTO,
        'description': 'Laterocemento 20+4',
        'total_height': 0.24,
        'joist_height': 0.20,
        'slab_thickness': 0.04,
        'joist_spacing': 0.50,
        'self_weight': 3.20,
        'stiffness': FloorStiffness.RIGID
    },
    'LAT_24+4': {
        'type': FloorType.LATEROCEMENTO,
        'description': 'Laterocemento 24+4',
        'total_height': 0.28,
        'joist_height': 0.24,
        'slab_thickness': 0.04,
        'joist_spacing': 0.50,
        'self_weight': 3.50,
        'stiffness': FloorStiffness.RIGID
    },

    # Solai in legno
    'LEGNO_14x20': {
        'type': FloorType.LEGNO_SEMPLICE,
        'description': 'Travi legno 14x20 + tavolato',
        'beam_width': 0.14,
        'beam_height': 0.20,
        'beam_spacing': 0.60,
        'planking_thickness': 0.03,
        'self_weight': 0.70,        # kN/m2
        'stiffness': FloorStiffness.FLEXIBLE
    },
    'LEGNO_16x24': {
        'type': FloorType.LEGNO_SEMPLICE,
        'description': 'Travi legno 16x24 + tavolato',
        'beam_width': 0.16,
        'beam_height': 0.24,
        'beam_spacing': 0.80,
        'planking_thickness': 0.03,
        'self_weight': 0.80,
        'stiffness': FloorStiffness.FLEXIBLE
    },
    'LEGNO_CONN_16x24': {
        'type': FloorType.LEGNO_CONNESSO,
        'description': 'Travi legno 16x24 + soletta connessa 5cm',
        'beam_width': 0.16,
        'beam_height': 0.24,
        'beam_spacing': 0.80,
        'slab_thickness': 0.05,
        'self_weight': 2.00,
        'stiffness': FloorStiffness.SEMI_RIGID
    },

    # Solai in acciaio
    'ACCIAIO_IPE200': {
        'type': FloorType.ACCIAIO,
        'description': 'IPE 200 + tavelloni',
        'profile': 'IPE200',
        'beam_spacing': 1.00,
        'self_weight': 1.80,
        'stiffness': FloorStiffness.FLEXIBLE
    },
    'ACCIAIO_IPE240': {
        'type': FloorType.ACCIAIO,
        'description': 'IPE 240 + tavelloni',
        'profile': 'IPE240',
        'beam_spacing': 1.20,
        'self_weight': 2.00,
        'stiffness': FloorStiffness.FLEXIBLE
    },

    # Solai in c.a. pieno
    'CA_15': {
        'type': FloorType.CA_PIENO,
        'description': 'Soletta piena 15cm',
        'thickness': 0.15,
        'self_weight': 3.75,        # 0.15 * 25
        'stiffness': FloorStiffness.RIGID
    },
    'CA_20': {
        'type': FloorType.CA_PIENO,
        'description': 'Soletta piena 20cm',
        'thickness': 0.20,
        'self_weight': 5.00,
        'stiffness': FloorStiffness.RIGID
    },

    # Volte
    'VOLTA_BOTTE_30': {
        'type': FloorType.VOLTA_BOTTE,
        'description': 'Volta a botte mattoni 30cm freccia',
        'thickness': 0.12,
        'rise': 0.30,               # freccia
        'fill_weight': 1.50,        # riempimento medio
        'self_weight': 4.00,
        'stiffness': FloorStiffness.SEMI_RIGID
    },
    'VOLTA_CROCIERA': {
        'type': FloorType.VOLTA_CROCIERA,
        'description': 'Volta a crociera mattoni',
        'thickness': 0.10,
        'rise': 0.40,
        'fill_weight': 1.20,
        'self_weight': 3.50,
        'stiffness': FloorStiffness.SEMI_RIGID
    },
}


# ============================================================================
# STRATIGRAFIE TIPICHE
# ============================================================================

@dataclass
class Layer:
    """Singolo strato di una stratigrafia"""
    name: str
    thickness: float  # m (0 per carichi superficiali)
    weight: float     # kN/m3 (o kN/m2 se thickness=0)

    @property
    def load(self) -> float:
        """Carico per unita' di superficie [kN/m2]"""
        if self.thickness == 0:
            return self.weight  # Gia' in kN/m2
        return self.thickness * self.weight


TYPICAL_STRATIFICATIONS = {
    'civile_standard': [
        Layer('Massetto alleggerito', 0.05, 14.0),
        Layer('Massetto impianti', 0.05, 22.0),
        Layer('Pavimento', 0.02, 22.0),
        Layer('Intonaco', 0.015, 20.0),
    ],
    'civile_riscaldamento': [
        Layer('Massetto alleggerito', 0.03, 14.0),
        Layer('Pannello radiante', 0.05, 12.0),
        Layer('Massetto', 0.05, 22.0),
        Layer('Pavimento', 0.02, 22.0),
        Layer('Intonaco', 0.015, 20.0),
    ],
    'terrazzo': [
        Layer('Massetto pendenze', 0.08, 20.0),
        Layer('Guaina', 0, 0.05),
        Layer('Protezione', 0.05, 22.0),
    ],
    'sottotetto_non_praticabile': [
        Layer('Intonaco', 0.015, 20.0),
    ],
}


# ============================================================================
# DATACLASS PRINCIPALI
# ============================================================================

@dataclass
class FloorStratigraphy:
    """Stratigrafia di un solaio"""
    layers: List[Layer] = field(default_factory=list)

    @property
    def total_weight(self) -> float:
        """Peso totale della stratigrafia [kN/m2]"""
        return sum(layer.load for layer in self.layers)

    @property
    def total_thickness(self) -> float:
        """Spessore totale [m]"""
        return sum(layer.thickness for layer in self.layers)

    @classmethod
    def from_preset(cls, preset_name: str) -> 'FloorStratigraphy':
        """Crea stratigrafia da preset"""
        if preset_name not in TYPICAL_STRATIFICATIONS:
            raise ValueError(f"Preset '{preset_name}' non trovato")
        return cls(layers=list(TYPICAL_STRATIFICATIONS[preset_name]))


@dataclass
class Floor:
    """Definizione di un solaio"""
    name: str
    floor_level: int                    # Piano (0 = terra)
    floor_type: FloorType               # Tipo strutturale
    preset: str = ""                    # Nome preset da database

    # Geometria
    span_direction: float = 0.0         # Angolo orditura [gradi, 0=X]
    span_length: float = 5.0            # Luce [m]
    width: float = 5.0                  # Larghezza [m]
    area: float = 25.0                  # Area [m2]

    # Carichi
    self_weight: float = 0.0            # Peso proprio struttura [kN/m2]
    stratigraphy: FloorStratigraphy = field(default_factory=FloorStratigraphy)
    live_load: float = 2.0              # Carico variabile [kN/m2]
    live_load_category: str = "A"       # Categoria d'uso NTC

    # Rigidezza
    stiffness: FloorStiffness = FloorStiffness.RIGID
    connection_type: str = "appoggio"   # appoggio, incastro

    # Coordinate del contorno (per disegno)
    vertices: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        """Inizializza da preset se specificato"""
        if self.preset and self.preset in FLOOR_DATABASE:
            data = FLOOR_DATABASE[self.preset]
            self.floor_type = data['type']
            self.self_weight = data['self_weight']
            self.stiffness = data['stiffness']

    @property
    def G1(self) -> float:
        """Carico permanente strutturale [kN/m2]"""
        return self.self_weight

    @property
    def G2(self) -> float:
        """Carico permanente non strutturale [kN/m2]"""
        return self.stratigraphy.total_weight

    @property
    def Gk(self) -> float:
        """Carico permanente totale [kN/m2]"""
        return self.G1 + self.G2

    @property
    def Qk(self) -> float:
        """Carico variabile caratteristico [kN/m2]"""
        return self.live_load

    @property
    def total_load(self) -> float:
        """Carico totale caratteristico [kN/m2]"""
        return self.Gk + self.Qk

    def get_design_load(self, combination: str = 'SLU') -> float:
        """
        Carico di progetto per combinazione.

        Args:
            combination: 'SLU' o 'SLE' o 'SISMA'

        Returns:
            Carico di progetto [kN/m2]
        """
        if combination == 'SLU':
            # 1.3*G1 + 1.5*G2 + 1.5*Qk
            return 1.3 * self.G1 + 1.5 * self.G2 + 1.5 * self.Qk
        elif combination == 'SLE':
            return self.Gk + self.Qk
        elif combination == 'SISMA':
            # G1 + G2 + 0.3*Qk (psi2)
            return self.G1 + self.G2 + 0.3 * self.Qk
        else:
            return self.total_load

    def summary(self) -> str:
        """Riepilogo solaio"""
        lines = [
            f"Solaio: {self.name}",
            f"  Tipo: {self.floor_type.value}",
            f"  Piano: {self.floor_level}",
            f"  Area: {self.area:.1f} m2",
            f"  Orditura: {self.span_direction:.0f} gradi",
            f"",
            f"  Carichi:",
            f"    G1 (struttura): {self.G1:.2f} kN/m2",
            f"    G2 (finiture):  {self.G2:.2f} kN/m2",
            f"    Qk (variabile): {self.Qk:.2f} kN/m2",
            f"    Totale:         {self.total_load:.2f} kN/m2",
            f"",
            f"  Rigidezza: {self.stiffness.value}",
        ]
        return "\n".join(lines)


@dataclass
class Roof:
    """Definizione di una copertura"""
    name: str
    roof_type: RoofType = RoofType.DOUBLE_PITCH
    structure_type: RoofStructure = RoofStructure.TIMBER_RAFTERS

    # Geometria
    area: float = 50.0              # Area in pianta [m2]
    pitch: float = 30.0             # Pendenza [gradi]
    ridge_height: float = 2.0       # Altezza colmo [m]
    eave_height: float = 0.0        # Altezza gronda [m]

    # Carichi
    structure_weight: float = 0.50  # Peso struttura [kN/m2]
    covering_weight: float = 0.60   # Peso manto [kN/m2]
    insulation_weight: float = 0.10 # Peso coibente [kN/m2]

    # Sottotetto
    attic_accessible: bool = False   # Praticabile
    attic_live_load: float = 0.5     # Carico variabile sottotetto

    @property
    def slope_area(self) -> float:
        """Area inclinata [m2]"""
        return self.area / math.cos(math.radians(self.pitch))

    @property
    def self_weight(self) -> float:
        """Peso proprio totale [kN/m2 in pianta]"""
        # Peso sulla falda inclinata riportato in pianta
        inclined = self.structure_weight + self.covering_weight + self.insulation_weight
        return inclined / math.cos(math.radians(self.pitch))

    @property
    def Gk(self) -> float:
        """Carico permanente totale [kN/m2]"""
        return self.self_weight

    @property
    def Qk(self) -> float:
        """Carico variabile [kN/m2]"""
        if self.attic_accessible:
            return self.attic_live_load
        return 0.5  # Copertura non accessibile

    def get_snow_load(self, qsk: float, Ce: float = 1.0, Ct: float = 1.0) -> float:
        """
        Calcola il carico neve.

        Args:
            qsk: Carico neve al suolo [kN/m2]
            Ce: Coefficiente di esposizione
            Ct: Coefficiente termico

        Returns:
            Carico neve su copertura [kN/m2]
        """
        # Coefficiente di forma
        alpha = self.pitch
        if alpha <= 30:
            mu = 0.8
        elif alpha <= 60:
            mu = 0.8 * (60 - alpha) / 30
        else:
            mu = 0.0

        return mu * qsk * Ce * Ct

    def summary(self) -> str:
        """Riepilogo copertura"""
        lines = [
            f"Copertura: {self.name}",
            f"  Tipo: {self.roof_type.value}",
            f"  Struttura: {self.structure_type.value}",
            f"  Pendenza: {self.pitch:.0f} gradi",
            f"  Area pianta: {self.area:.1f} m2",
            f"  Area inclinata: {self.slope_area:.1f} m2",
            f"",
            f"  Carichi:",
            f"    Peso proprio: {self.self_weight:.2f} kN/m2",
            f"    Variabile:    {self.Qk:.2f} kN/m2",
            f"  Sottotetto praticabile: {'Si' if self.attic_accessible else 'No'}",
        ]
        return "\n".join(lines)


# ============================================================================
# FUNZIONI DI UTILITA'
# ============================================================================

def get_floor_presets() -> List[str]:
    """Restituisce la lista dei preset disponibili"""
    return list(FLOOR_DATABASE.keys())


def get_floor_preset_info(preset: str) -> Dict:
    """Restituisce le info di un preset"""
    return FLOOR_DATABASE.get(preset, {})


def calculate_floor_reaction(floor: Floor, wall_length: float,
                             wall_position: str = 'support') -> float:
    """
    Calcola la reazione del solaio su una parete.

    Args:
        floor: Oggetto Floor
        wall_length: Lunghezza della parete [m]
        wall_position: 'support' = parete di appoggio, 'parallel' = parallela

    Returns:
        Carico lineare sulla parete [kN/m]
    """
    if wall_position == 'parallel':
        # Parete parallela all'orditura: nessun carico
        return 0.0

    # Parete di appoggio: carico = q * luce / 2
    q = floor.total_load  # kN/m2
    span = floor.span_length  # m

    # Carico lineare (per metro di parete)
    return q * span / 2


def calculate_seismic_mass(floor: Floor) -> float:
    """
    Calcola la massa sismica del solaio [tonnellate].

    Formula NTC: G1 + G2 + psi2 * Qk
    """
    psi2 = 0.3  # Abitazioni
    W = (floor.G1 + floor.G2 + psi2 * floor.Qk) * floor.area  # kN
    return W / 10  # tonnellate (1 kN = 0.1 t)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Test modulo floors.py")
    print("=" * 60)

    # Test solaio da preset
    floor = Floor(
        name="S1",
        floor_level=1,
        floor_type=FloorType.LATEROCEMENTO,
        preset='LAT_20+4',
        area=30.0,
        span_length=5.0
    )

    # Aggiungi stratigrafia
    floor.stratigraphy = FloorStratigraphy.from_preset('civile_standard')

    print(floor.summary())
    print()

    print(f"Carico SLU: {floor.get_design_load('SLU'):.2f} kN/m2")
    print(f"Carico SISMA: {floor.get_design_load('SISMA'):.2f} kN/m2")
    print(f"Massa sismica: {calculate_seismic_mass(floor):.2f} t")
    print()

    # Test copertura
    roof = Roof(
        name="Copertura",
        roof_type=RoofType.DOUBLE_PITCH,
        area=60.0,
        pitch=30.0
    )

    print(roof.summary())
    print()

    # Carico neve esempio (Palermo, zona II, 100m s.l.m.)
    qsk = 0.60  # kN/m2
    snow = roof.get_snow_load(qsk)
    print(f"Carico neve (qsk={qsk}): {snow:.2f} kN/m2")
