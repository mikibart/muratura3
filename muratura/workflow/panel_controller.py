# -*- coding: utf-8 -*-
"""
Panel Controller - Integrazione pannelli con WorkflowManager.

Gestisce la sincronizzazione tra pannelli UI e stato workflow.
"""

from typing import Dict, Any, Optional, Callable

try:
    from PySide2 import QtWidgets, QtCore
except ImportError:
    from PySide import QtWidgets, QtCore

from .manager import WorkflowManager
from .phases import PhaseStatus, WORKFLOW_PHASES


class PanelController(QtCore.QObject):
    """
    Controller che integra i pannelli fase con il WorkflowManager.
    Gestisce navigazione, validazione e persistenza dati.
    """

    # Segnali
    phase_changed = QtCore.Signal(int)  # Emesso quando si cambia fase
    data_saved = QtCore.Signal()  # Emesso quando i dati vengono salvati
    validation_failed = QtCore.Signal(int, str)  # (fase, messaggio errore)

    def __init__(self, workflow_manager: WorkflowManager = None, parent=None):
        super().__init__(parent)
        self.workflow = workflow_manager or WorkflowManager()
        self.panels: Dict[int, Any] = {}  # fase_id -> panel instance
        self.current_panel: Optional[Any] = None
        self.panel_container: Optional[QtWidgets.QStackedWidget] = None

        # Connetti callback workflow
        self.workflow.add_callback('phase_changed', self._on_workflow_phase_changed)
        self.workflow.add_callback('phase_completed', self._on_workflow_phase_completed)

    def set_panel_container(self, container: QtWidgets.QStackedWidget):
        """Imposta il container per i pannelli."""
        self.panel_container = container

    def register_panel(self, phase_id: int, panel):
        """Registra un pannello per una fase."""
        self.panels[phase_id] = panel

        # Connetti segnali pannello
        if hasattr(panel, 'data_changed'):
            panel.data_changed.connect(lambda d: self._on_panel_data_changed(phase_id, d))
        if hasattr(panel, 'phase_completed'):
            panel.phase_completed.connect(lambda: self._complete_phase(phase_id))
        if hasattr(panel, 'validation_error'):
            panel.validation_error.connect(lambda m: self.validation_failed.emit(phase_id, m))

        # Aggiungi al container se presente
        if self.panel_container:
            self.panel_container.addWidget(panel)

    def show_phase(self, phase_id: int) -> bool:
        """Mostra il pannello per una fase specifica."""
        if phase_id not in self.panels:
            return False

        # Salva dati fase corrente
        if self.current_panel and hasattr(self.current_panel, 'get_data'):
            current_phase = self._get_phase_for_panel(self.current_panel)
            if current_phase:
                self.workflow.set_phase_data(current_phase, self.current_panel.get_data())

        # Cambia pannello
        panel = self.panels[phase_id]
        self.current_panel = panel

        if self.panel_container:
            self.panel_container.setCurrentWidget(panel)

        # Carica dati esistenti
        existing_data = self.workflow.get_phase_data(phase_id)
        if existing_data and hasattr(panel, 'set_data'):
            panel.set_data(existing_data)

        # Aggiorna workflow
        self.workflow.go_to_phase(phase_id)
        self.phase_changed.emit(phase_id)

        return True

    def _get_phase_for_panel(self, panel) -> Optional[int]:
        """Trova l'ID fase per un pannello."""
        for phase_id, p in self.panels.items():
            if p is panel:
                return phase_id
        return None

    def _on_panel_data_changed(self, phase_id: int, data: Dict):
        """Gestisce cambio dati in un pannello."""
        self.workflow.set_phase_data(phase_id, data)

    def _complete_phase(self, phase_id: int):
        """Completa una fase."""
        panel = self.panels.get(phase_id)
        if panel and hasattr(panel, 'validate'):
            if panel.validate():
                self.workflow.complete_phase(phase_id)
                # Auto-avanza alla fase successiva
                if phase_id < 12:
                    self.show_phase(phase_id + 1)

    def _on_workflow_phase_changed(self, phase_id: int):
        """Callback per cambio fase nel workflow."""
        pass  # Già gestito da show_phase

    def _on_workflow_phase_completed(self, phase_id: int):
        """Callback per completamento fase."""
        pass  # Gestito internamente

    def save_all_data(self):
        """Salva tutti i dati dei pannelli."""
        for phase_id, panel in self.panels.items():
            if hasattr(panel, 'get_data'):
                self.workflow.set_phase_data(phase_id, panel.get_data())
        self.data_saved.emit()

    def load_project(self, filepath: str) -> bool:
        """Carica un progetto."""
        if self.workflow.load_project(filepath):
            # Aggiorna tutti i pannelli con i dati caricati
            for phase_id, panel in self.panels.items():
                data = self.workflow.get_phase_data(phase_id)
                if data and hasattr(panel, 'set_data'):
                    panel.set_data(data)
            return True
        return False

    def new_project(self, name: str, **kwargs) -> bool:
        """Crea un nuovo progetto."""
        if self.workflow.new_project(name, **kwargs):
            # Reset tutti i pannelli
            for panel in self.panels.values():
                if hasattr(panel, 'reset_data'):
                    panel.reset_data()
            self.show_phase(1)
            return True
        return False

    def get_progress(self) -> float:
        """Restituisce percentuale completamento."""
        return self.workflow.get_progress()

    def get_phase_status(self, phase_id: int) -> str:
        """Restituisce stato di una fase."""
        status = self.workflow.state.phases_status.get(phase_id, PhaseStatus.NOT_STARTED)
        return status.name.lower()

    def create_all_panels(self, parent=None) -> Dict[int, Any]:
        """Crea tutti i pannelli per le 12 fasi."""
        from muratura.panels import PHASE_PANELS

        for phase_id, panel_class in PHASE_PANELS.items():
            try:
                panel = panel_class(parent)
                self.register_panel(phase_id, panel)
            except Exception as e:
                print(f"Errore creazione pannello fase {phase_id}: {e}")

        return self.panels
