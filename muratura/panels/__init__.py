# -*- coding: utf-8 -*-
"""
Muratura Panels Module

Pannelli Qt per le 12 fasi del workflow NTC 2018.
"""

from .base_panel import BasePhasePanel
from .project_panel import ProjectPanel
from .geometry_panel import GeometryPanel
from .materials_panel import MaterialsPanel
from .structural_panel import StructuralPanel
from .floors_panel import FloorsPanel
from .loads_panel import LoadsPanel
from .seismic_panel import SeismicPanel
from .model_panel import ModelPanel
from .analysis_panel import AnalysisPanel
from .results_panel import ResultsPanel
from .reinforcement_panel import ReinforcementPanel
from .report_panel import ReportPanel

__all__ = [
    'BasePhasePanel',
    'ProjectPanel',
    'GeometryPanel',
    'MaterialsPanel',
    'StructuralPanel',
    'FloorsPanel',
    'LoadsPanel',
    'SeismicPanel',
    'ModelPanel',
    'AnalysisPanel',
    'ResultsPanel',
    'ReinforcementPanel',
    'ReportPanel',
]

# Mapping fase -> pannello
PHASE_PANELS = {
    1: ProjectPanel,
    2: GeometryPanel,
    3: MaterialsPanel,
    4: StructuralPanel,
    5: FloorsPanel,
    6: LoadsPanel,
    7: SeismicPanel,
    8: ModelPanel,
    9: AnalysisPanel,
    10: ResultsPanel,
    11: ReinforcementPanel,
    12: ReportPanel,
}


def get_panel_for_phase(phase_id: int):
    """Restituisce la classe pannello per una fase."""
    return PHASE_PANELS.get(phase_id)
