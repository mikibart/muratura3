# -*- coding: utf-8 -*-
"""
Muratura NTC 2018 - Modulo calcoli strutturali

Contiene:
- materials: Proprietà muratura da tabella C8.5.I
- seismic: Database parametri sismici comuni italiani
- loads: Carichi neve, vento, permanenti
- constitutive: Leggi costitutive materiali
- analyses: Motori di analisi (POR, SAM, FEM, ecc.)
- ingv_database: Database parametri sismici INGV
- equivalent_frame: Generazione telaio equivalente
- analysis_runner: Interfaccia unificata analisi
"""

from .materials import *
from .seismic import *
from .loads import *
from .enums import *
from .ingv_database import (
    get_seismic_params,
    calculate_response_spectrum,
    COMUNI_DATABASE,
    STATI_LIMITE,
)
from .equivalent_frame import (
    Pier,
    Spandrel,
    RigidNode,
    Opening,
    EquivalentFrameGenerator,
    generate_from_freecad,
)
from .analysis_runner import (
    AnalysisMethod,
    AnalysisInput,
    AnalysisResult,
    PierResult,
    AnalysisRunner,
    run_analysis,
    run_complete_analysis,
)
