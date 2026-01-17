# -*- coding: utf-8 -*-
"""
WorkflowManager - Gestione centrale del workflow Muratura.

Coordina tutte le 12 fasi del workflow, gestisce lo stato del progetto,
e fornisce API per l'interfaccia utente e MCP.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .phases import (
    Phase, PhaseStatus, WORKFLOW_PHASES,
    get_phase_by_id, can_start_phase, get_workflow_progress
)


@dataclass
class ProjectData:
    """Dati del progetto."""
    # Fase 1: Progetto
    name: str = ""
    code: str = ""
    client: str = ""
    designer: str = ""
    date: str = ""
    building_type: str = "Residenziale"  # Residenziale, Produttivo, Agricolo, Storico, Strategico
    construction_type: str = "Esistente"  # Nuova, Esistente
    construction_year: int = 0
    intervention_type: str = "Miglioramento"  # Nuovo, Miglioramento, Adeguamento, Locale

    # Parametri normativi
    VN: int = 50  # Vita nominale (anni)
    CU: str = "II"  # Classe d'uso
    CU_coeff: float = 1.0
    VR: float = 50.0  # Periodo riferimento

    # Livello conoscenza (solo esistenti)
    LC: str = "LC2"  # LC1, LC2, LC3
    FC: float = 1.20  # Fattore confidenza

    # Fase 7: Sismica
    municipality: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    soil_category: str = "B"  # A, B, C, D, E
    topographic_category: str = "T1"  # T1, T2, T3, T4
    ag_SLV: float = 0.0
    F0_SLV: float = 0.0
    Tc_star_SLV: float = 0.0
    q_factor: float = 2.0

    # Metadati
    created: str = ""
    modified: str = ""
    software_version: str = "3.2.0"


@dataclass
class WorkflowState:
    """Stato corrente del workflow."""
    current_phase: int = 1
    phases_status: Dict[int, PhaseStatus] = field(default_factory=dict)
    phases_data: Dict[int, Dict] = field(default_factory=dict)

    def __post_init__(self):
        # Inizializza status per tutte le fasi
        if not self.phases_status:
            self.phases_status = {p.id: PhaseStatus.NOT_STARTED for p in WORKFLOW_PHASES}


class WorkflowManager:
    """
    Gestore centrale del workflow Muratura.

    Coordina:
    - Stato del progetto
    - Navigazione tra fasi
    - Validazione dati
    - Persistenza
    - Eventi e callback
    """

    def __init__(self):
        self.project = ProjectData()
        self.state = WorkflowState()
        self.document = None  # FreeCAD document
        self.project_path: Optional[Path] = None

        # Callback per eventi
        self._callbacks: Dict[str, List[Callable]] = {
            'phase_changed': [],
            'phase_completed': [],
            'project_saved': [],
            'project_loaded': [],
            'analysis_completed': [],
            'error': [],
        }

        # Cache risultati analisi
        self.analysis_results: Dict[str, Any] = {}

    # === GESTIONE PROGETTO ===

    def new_project(self, name: str, **kwargs) -> bool:
        """
        Crea un nuovo progetto.

        Args:
            name: Nome del progetto
            **kwargs: Altri parametri progetto (client, designer, building_type, etc.)

        Returns:
            True se creato con successo
        """
        # Reset stato
        self.project = ProjectData()
        self.state = WorkflowState()
        self.analysis_results = {}

        # Imposta dati base
        self.project.name = name
        self.project.code = self._generate_project_code()
        self.project.date = datetime.now().strftime("%Y-%m-%d")
        self.project.created = datetime.now().isoformat()
        self.project.modified = self.project.created

        # Imposta parametri opzionali
        for key, value in kwargs.items():
            if hasattr(self.project, key):
                setattr(self.project, key, value)

        # Calcola VR
        self._update_VR()

        # Calcola FC
        self._update_FC()

        # Crea documento FreeCAD
        try:
            import FreeCAD
            doc_name = name.replace(" ", "_")
            self.document = FreeCAD.newDocument(doc_name)
            self._add_project_properties()
        except ImportError:
            # FreeCAD non disponibile (test mode)
            self.document = None

        # Segna fase 1 come in corso
        self.state.phases_status[1] = PhaseStatus.IN_PROGRESS
        self.state.current_phase = 1

        return True

    def save_project(self, path: Optional[str] = None) -> bool:
        """Salva il progetto."""
        if path:
            self.project_path = Path(path)

        if not self.project_path:
            return False

        self.project.modified = datetime.now().isoformat()

        # Salva documento FreeCAD
        if self.document:
            fcstd_path = self.project_path.with_suffix('.FCStd')
            self.document.saveAs(str(fcstd_path))

        # Salva stato workflow in JSON separato
        state_path = self.project_path.with_suffix('.muratura.json')
        state_data = {
            'project': asdict(self.project),
            'workflow': {
                'current_phase': self.state.current_phase,
                'phases_status': {k: v.name for k, v in self.state.phases_status.items()},
                'phases_data': self.state.phases_data,
            },
            'analysis_results': self._serialize_results(self.analysis_results),
        }

        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)

        self._trigger_event('project_saved', self.project_path)
        return True

    def load_project(self, path: str) -> bool:
        """Carica un progetto esistente."""
        self.project_path = Path(path)

        # Carica stato workflow
        state_path = self.project_path.with_suffix('.muratura.json')
        if state_path.exists():
            with open(state_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)

            # Ripristina project data
            for key, value in state_data.get('project', {}).items():
                if hasattr(self.project, key):
                    setattr(self.project, key, value)

            # Ripristina workflow state
            wf_data = state_data.get('workflow', {})
            self.state.current_phase = wf_data.get('current_phase', 1)
            self.state.phases_status = {
                int(k): PhaseStatus[v]
                for k, v in wf_data.get('phases_status', {}).items()
            }
            self.state.phases_data = wf_data.get('phases_data', {})

            # Ripristina risultati
            self.analysis_results = state_data.get('analysis_results', {})

        # Carica documento FreeCAD
        fcstd_path = self.project_path.with_suffix('.FCStd')
        if fcstd_path.exists():
            try:
                import FreeCAD
                self.document = FreeCAD.openDocument(str(fcstd_path))
            except ImportError:
                self.document = None

        self._trigger_event('project_loaded', self.project_path)
        return True

    # === NAVIGAZIONE FASI ===

    def go_to_phase(self, phase_id: int) -> bool:
        """
        Naviga a una fase specifica.

        Args:
            phase_id: ID della fase (1-12)

        Returns:
            True se la navigazione è permessa
        """
        if phase_id < 1 or phase_id > 12:
            return False

        # Verifica dipendenze
        completed = [
            pid for pid, status in self.state.phases_status.items()
            if status == PhaseStatus.COMPLETED
        ]

        if not can_start_phase(phase_id, completed):
            # Permetti comunque di tornare a fasi precedenti
            if phase_id > self.state.current_phase:
                self._trigger_event('error', f"Completa prima le fasi prerequisite")
                return False

        old_phase = self.state.current_phase
        self.state.current_phase = phase_id

        # Aggiorna status se non ancora iniziata
        if self.state.phases_status[phase_id] == PhaseStatus.NOT_STARTED:
            self.state.phases_status[phase_id] = PhaseStatus.IN_PROGRESS

        self._trigger_event('phase_changed', {'from': old_phase, 'to': phase_id})
        return True

    def complete_phase(self, phase_id: int) -> bool:
        """Marca una fase come completata."""
        phase = get_phase_by_id(phase_id)
        if not phase:
            return False

        # Valida fase
        errors = self._validate_phase(phase_id)
        if errors:
            self._trigger_event('error', f"Fase {phase_id} non valida: {errors}")
            return False

        self.state.phases_status[phase_id] = PhaseStatus.COMPLETED
        self._trigger_event('phase_completed', phase_id)

        # Auto-avanza alla prossima fase
        if phase_id < 12:
            next_phase = phase_id + 1
            if can_start_phase(next_phase, self._get_completed_phases()):
                self.go_to_phase(next_phase)

        return True

    def get_current_phase(self) -> Phase:
        """Ottiene la fase corrente."""
        return get_phase_by_id(self.state.current_phase)

    def get_progress(self) -> float:
        """Ottiene la percentuale di completamento."""
        return get_workflow_progress(self.state.phases_status)

    def get_phases_summary(self) -> List[Dict]:
        """Ottiene riepilogo di tutte le fasi."""
        summary = []
        completed = self._get_completed_phases()

        for phase in WORKFLOW_PHASES:
            status = self.state.phases_status.get(phase.id, PhaseStatus.NOT_STARTED)
            can_start = can_start_phase(phase.id, completed)

            summary.append({
                'id': phase.id,
                'name': phase.name,
                'description': phase.description,
                'status': status.name,
                'can_start': can_start,
                'is_current': phase.id == self.state.current_phase,
                'required': phase.required,
            })

        return summary

    # === GESTIONE DATI FASI ===

    def set_phase_data(self, phase_id: int, key: str, value: Any):
        """Imposta un dato per una fase."""
        if phase_id not in self.state.phases_data:
            self.state.phases_data[phase_id] = {}
        self.state.phases_data[phase_id][key] = value

    def get_phase_data(self, phase_id: int, key: str = None) -> Any:
        """Ottiene dati di una fase."""
        data = self.state.phases_data.get(phase_id, {})
        if key:
            return data.get(key)
        return data

    # === API PER OGNI FASE ===

    # Fase 1: Progetto
    def set_project_info(self, **kwargs):
        """Imposta informazioni progetto."""
        for key, value in kwargs.items():
            if hasattr(self.project, key):
                setattr(self.project, key, value)

        self._update_VR()
        self._update_FC()
        self._sync_to_document()

    # Fase 7: Sismica
    def set_seismic_params(self, municipality: str = None, **kwargs):
        """Imposta parametri sismici."""
        if municipality:
            self.project.municipality = municipality
            # Cerca nel database
            params = self._lookup_seismic_params(municipality)
            if params:
                self.project.ag_SLV = params.get('ag', 0)
                self.project.F0_SLV = params.get('F0', 0)
                self.project.Tc_star_SLV = params.get('Tc_star', 0)

        for key, value in kwargs.items():
            if hasattr(self.project, key):
                setattr(self.project, key, value)

        self._sync_to_document()

    # Fase 9: Analisi
    def run_analysis(self, methods: List[str]) -> Dict[str, Any]:
        """
        Esegue le analisi selezionate.

        Args:
            methods: Lista metodi ['POR', 'SAM', 'FRAME', 'FEM', 'LIMIT', 'FIBER', 'MICRO']

        Returns:
            Risultati per ogni metodo
        """
        results = {}

        for method in methods:
            method = method.upper()
            try:
                if method == 'POR':
                    from ..ntc2018.analyses.por import analyze_por
                    results['POR'] = self._run_por_analysis()
                elif method == 'PORFLEX':
                    from ..ntc2018.analyses.porflex import analyze_porflex
                    results['PORFLEX'] = self._run_porflex_analysis()
                elif method == 'SAM':
                    from ..ntc2018.analyses.sam import analyze_sam
                    results['SAM'] = self._run_sam_analysis()
                elif method == 'FRAME':
                    from ..ntc2018.analyses.frame import _analyze_frame
                    results['FRAME'] = self._run_frame_analysis()
                elif method == 'FEM':
                    from ..ntc2018.analyses.fem import analyze_fem
                    results['FEM'] = self._run_fem_analysis()
                elif method == 'LIMIT':
                    from ..ntc2018.analyses.limit import analyze_limit
                    results['LIMIT'] = self._run_limit_analysis()
                elif method == 'FIBER':
                    from ..ntc2018.analyses.fiber import analyze_fiber
                    results['FIBER'] = self._run_fiber_analysis()
                elif method == 'MICRO':
                    from ..ntc2018.analyses.micro import analyze_micro
                    results['MICRO'] = self._run_micro_analysis()
                else:
                    results[method] = {'error': f'Metodo {method} non riconosciuto'}
            except Exception as e:
                results[method] = {'error': str(e)}

        self.analysis_results.update(results)
        self._trigger_event('analysis_completed', results)
        return results

    # === EVENTI ===

    def on(self, event: str, callback: Callable):
        """Registra callback per evento."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable):
        """Rimuove callback per evento."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    # === METODI PRIVATI ===

    def _generate_project_code(self) -> str:
        """Genera codice progetto univoco."""
        now = datetime.now()
        return f"MUR-{now.year}{now.month:02d}{now.day:02d}-{now.hour:02d}{now.minute:02d}"

    def _update_VR(self):
        """Aggiorna periodo di riferimento."""
        cu_coeffs = {'I': 0.7, 'II': 1.0, 'III': 1.5, 'IV': 2.0}
        self.project.CU_coeff = cu_coeffs.get(self.project.CU, 1.0)
        self.project.VR = self.project.VN * self.project.CU_coeff

    def _update_FC(self):
        """Aggiorna fattore di confidenza."""
        fc_values = {'LC1': 1.35, 'LC2': 1.20, 'LC3': 1.00}
        self.project.FC = fc_values.get(self.project.LC, 1.20)

    def _add_project_properties(self):
        """Aggiunge proprietà progetto al documento FreeCAD."""
        if not self.document:
            return

        props = [
            ("App::PropertyString", "ProjectName", "Muratura"),
            ("App::PropertyString", "ProjectCode", "Muratura"),
            ("App::PropertyString", "Client", "Muratura"),
            ("App::PropertyString", "Designer", "Muratura"),
            ("App::PropertyInteger", "VN", "Muratura"),
            ("App::PropertyString", "CU", "Muratura"),
            ("App::PropertyFloat", "VR", "Muratura"),
            ("App::PropertyString", "LC", "Muratura"),
            ("App::PropertyFloat", "FC", "Muratura"),
            ("App::PropertyString", "Municipality", "Muratura"),
            ("App::PropertyFloat", "ag_SLV", "Muratura"),
            ("App::PropertyFloat", "q_factor", "Muratura"),
        ]

        for prop_type, prop_name, prop_group in props:
            if not hasattr(self.document, prop_name):
                self.document.addProperty(prop_type, prop_name, prop_group)

        self._sync_to_document()

    def _sync_to_document(self):
        """Sincronizza dati progetto con documento FreeCAD."""
        if not self.document:
            return

        mapping = {
            'ProjectName': 'name',
            'ProjectCode': 'code',
            'Client': 'client',
            'Designer': 'designer',
            'VN': 'VN',
            'CU': 'CU',
            'VR': 'VR',
            'LC': 'LC',
            'FC': 'FC',
            'Municipality': 'municipality',
            'ag_SLV': 'ag_SLV',
            'q_factor': 'q_factor',
        }

        for doc_prop, project_attr in mapping.items():
            if hasattr(self.document, doc_prop):
                setattr(self.document, doc_prop, getattr(self.project, project_attr))

    def _lookup_seismic_params(self, municipality: str) -> Optional[Dict]:
        """Cerca parametri sismici per comune."""
        try:
            from ..ntc2018.seismic import get_seismic_params
            return get_seismic_params(municipality)
        except ImportError:
            return None

    def _get_completed_phases(self) -> List[int]:
        """Ottiene lista fasi completate."""
        return [
            pid for pid, status in self.state.phases_status.items()
            if status == PhaseStatus.COMPLETED
        ]

    def _validate_phase(self, phase_id: int) -> List[str]:
        """Valida una fase."""
        errors = []

        if phase_id == 1:
            if not self.project.name:
                errors.append("Nome progetto obbligatorio")
        elif phase_id == 2:
            if self.document and len(self.document.Objects) == 0:
                errors.append("Nessun elemento geometrico")
        # ... altre validazioni

        return errors

    def _trigger_event(self, event: str, data: Any = None):
        """Attiva callback per evento."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                print(f"Errore callback {event}: {e}")

    def _serialize_results(self, results: Dict) -> Dict:
        """Serializza risultati per JSON."""
        # Converti numpy arrays e altri oggetti non serializzabili
        import json

        def convert(obj):
            try:
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.generic):
                    return obj.item()
            except ImportError:
                pass

            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]

            return obj

        return convert(results)

    # === PLACEHOLDER METODI ANALISI ===
    # (da implementare con integrazione effettiva)

    def _run_por_analysis(self) -> Dict:
        """Esegue analisi POR."""
        # TODO: Implementare estrazione dati da modello e chiamata a analyze_por
        return {'method': 'POR', 'status': 'not_implemented'}

    def _run_porflex_analysis(self) -> Dict:
        return {'method': 'PORFLEX', 'status': 'not_implemented'}

    def _run_sam_analysis(self) -> Dict:
        return {'method': 'SAM', 'status': 'not_implemented'}

    def _run_frame_analysis(self) -> Dict:
        return {'method': 'FRAME', 'status': 'not_implemented'}

    def _run_fem_analysis(self) -> Dict:
        return {'method': 'FEM', 'status': 'not_implemented'}

    def _run_limit_analysis(self) -> Dict:
        return {'method': 'LIMIT', 'status': 'not_implemented'}

    def _run_fiber_analysis(self) -> Dict:
        return {'method': 'FIBER', 'status': 'not_implemented'}

    def _run_micro_analysis(self) -> Dict:
        return {'method': 'MICRO', 'status': 'not_implemented'}
