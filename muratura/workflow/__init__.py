# -*- coding: utf-8 -*-
"""
Muratura Workflow Module

Gestione del workflow a 12 fasi per analisi strutturale edifici in muratura.
"""

from .manager import WorkflowManager
from .phases import Phase, PhaseStatus, WORKFLOW_PHASES
from .validators import (
    PhaseValidator,
    ValidationResult,
    get_validator,
    validate_phase,
    validate_all_phases,
)

# PanelController richiede PySide2 (disponibile solo in FreeCAD)
try:
    from .panel_controller import PanelController
    _HAS_PANEL_CONTROLLER = True
except ImportError:
    PanelController = None
    _HAS_PANEL_CONTROLLER = False

__all__ = [
    'WorkflowManager',
    'Phase',
    'PhaseStatus',
    'WORKFLOW_PHASES',
    'PhaseValidator',
    'ValidationResult',
    'get_validator',
    'validate_phase',
    'validate_all_phases',
    'PanelController',
]
