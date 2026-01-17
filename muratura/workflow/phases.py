# -*- coding: utf-8 -*-
"""
Definizione delle 12 fasi del workflow Muratura.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable


class PhaseStatus(Enum):
    """Stato di una fase del workflow."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    ERROR = auto()
    SKIPPED = auto()


@dataclass
class Phase:
    """Definizione di una fase del workflow."""
    id: int
    name: str
    description: str
    icon: str
    required: bool = True
    depends_on: List[int] = field(default_factory=list)
    validators: List[Callable] = field(default_factory=list)

    def __post_init__(self):
        self.status = PhaseStatus.NOT_STARTED
        self.completion_percent = 0.0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.data: Dict = {}


# Definizione delle 12 fasi del workflow
WORKFLOW_PHASES = [
    Phase(
        id=1,
        name="Progetto",
        description="Dati generali, vita nominale, classe d'uso, livello conoscenza",
        icon="document-new",
        required=True,
        depends_on=[]
    ),
    Phase(
        id=2,
        name="Geometria",
        description="Import DXF/IFC, disegno muri e aperture con Arch",
        icon="draw-rectangle",
        required=True,
        depends_on=[1]
    ),
    Phase(
        id=3,
        name="Materiali",
        description="Assegnazione materiali da database NTC 2018 Tab. C8.5.I",
        icon="color-fill",
        required=True,
        depends_on=[2]
    ),
    Phase(
        id=4,
        name="Struttura",
        description="Cordoli, travi, pilastri, catene, piattabande, fondazioni",
        icon="draw-cuboid",
        required=True,
        depends_on=[2]
    ),
    Phase(
        id=5,
        name="Solai",
        description="Tipologia, rigidezza, orditura, coperture",
        icon="view-grid",
        required=True,
        depends_on=[2]
    ),
    Phase(
        id=6,
        name="Carichi",
        description="G1, G2, Q, neve, vento, combinazioni",
        icon="go-down",
        required=True,
        depends_on=[5]
    ),
    Phase(
        id=7,
        name="Sismica",
        description="Località, sottosuolo, spettro, fattore q",
        icon="earthquake",
        required=True,
        depends_on=[1]
    ),
    Phase(
        id=8,
        name="Modello",
        description="Generazione automatica telaio equivalente",
        icon="network-wired",
        required=True,
        depends_on=[2, 3, 4, 5]
    ),
    Phase(
        id=9,
        name="Analisi",
        description="POR, SAM, FRAME, FEM, LIMIT, FIBER, MICRO",
        icon="applications-science",
        required=True,
        depends_on=[6, 7, 8]
    ),
    Phase(
        id=10,
        name="Verifiche",
        description="DCR, indice rischio, classe sismica",
        icon="dialog-ok-apply",
        required=True,
        depends_on=[9]
    ),
    Phase(
        id=11,
        name="Rinforzi",
        description="Catene, FRP, intonaco armato, iniezioni",
        icon="tools",
        required=False,
        depends_on=[10]
    ),
    Phase(
        id=12,
        name="Relazione",
        description="Report PDF, DOCX, export IFC/DXF",
        icon="document-export",
        required=True,
        depends_on=[10]
    ),
]


def get_phase_by_id(phase_id: int) -> Optional[Phase]:
    """Ottiene una fase per ID."""
    for phase in WORKFLOW_PHASES:
        if phase.id == phase_id:
            return phase
    return None


def get_phase_by_name(name: str) -> Optional[Phase]:
    """Ottiene una fase per nome."""
    for phase in WORKFLOW_PHASES:
        if phase.name.lower() == name.lower():
            return phase
    return None


def can_start_phase(phase_id: int, completed_phases: List[int]) -> bool:
    """Verifica se una fase può essere avviata."""
    phase = get_phase_by_id(phase_id)
    if not phase:
        return False

    # Verifica dipendenze
    for dep_id in phase.depends_on:
        if dep_id not in completed_phases:
            return False

    return True


def get_workflow_progress(phases_status: Dict[int, PhaseStatus]) -> float:
    """Calcola la percentuale di completamento del workflow."""
    completed = sum(1 for s in phases_status.values() if s == PhaseStatus.COMPLETED)
    total = len(WORKFLOW_PHASES)
    return (completed / total) * 100 if total > 0 else 0
