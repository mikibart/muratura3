# -*- coding: utf-8 -*-
"""
Muratura Workbench - Inizializzazione GUI

Registra il workbench Muratura in FreeCAD.
Pannello laterale workflow con 12 fasi NTC 2018.
"""

import FreeCAD
import FreeCADGui

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide import QtWidgets, QtCore, QtGui


class WorkflowPanel(QtWidgets.QDockWidget):
    """
    Pannello laterale con le 12 fasi del workflow.
    Stile simile a 3Muri/PCM con fasi sempre visibili.
    """

    # Definizione delle 12 fasi
    PHASES = [
        (1, "Progetto", "Dati generali, VN, CU, LC"),
        (2, "Geometria", "Import DXF/IFC, muri, aperture"),
        (3, "Materiali", "Database NTC Tab. C8.5.I"),
        (4, "Struttura", "Cordoli, travi, pilastri, catene"),
        (5, "Solai", "Tipologia, rigidezza, orditura"),
        (6, "Carichi", "G1, G2, Q, neve, vento"),
        (7, "Sismica", "Località, sottosuolo, spettro"),
        (8, "Modello", "Telaio equivalente automatico"),
        (9, "Analisi", "POR, SAM, FRAME, FEM, LIMIT..."),
        (10, "Verifiche", "DCR, indice rischio, classe"),
        (11, "Rinforzi", "Catene, FRP, iniezioni"),
        (12, "Relazione", "Report PDF, export IFC/DXF"),
    ]

    # Icone per stato (Unicode)
    STATUS_ICONS = {
        'not_started': '○',
        'in_progress': '◐',
        'completed': '●',
        'error': '✗',
        'skipped': '◌',
    }

    phase_clicked = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__("Workflow Muratura", parent)
        self.setObjectName("MuraturaWorkflowPanel")
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFloatable
        )

        # Stato fasi
        self.phase_status = {i: 'not_started' for i in range(1, 13)}
        self.current_phase = 1
        self.phase_buttons = {}

        self._setup_ui()

    def _setup_ui(self):
        """Costruisce l'interfaccia del pannello."""
        # Widget principale
        main_widget = QtWidgets.QWidget()
        self.setWidget(main_widget)

        layout = QtWidgets.QVBoxLayout(main_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        # Header con progresso
        header = QtWidgets.QFrame()
        header.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        header_layout = QtWidgets.QVBoxLayout(header)

        title = QtWidgets.QLabel("MURATURA 3.0")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #8B4513;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        header_layout.addWidget(title)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 12)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Completamento: %v/12")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        header_layout.addWidget(self.progress_bar)

        self.status_label = QtWidgets.QLabel("Pronto")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 11px;")
        header_layout.addWidget(self.status_label)

        layout.addWidget(header)

        # Separatore
        layout.addSpacing(5)

        # Lista fasi
        phases_frame = QtWidgets.QFrame()
        phases_frame.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        phases_layout = QtWidgets.QVBoxLayout(phases_frame)
        phases_layout.setContentsMargins(3, 3, 3, 3)
        phases_layout.setSpacing(1)

        for phase_id, name, desc in self.PHASES:
            btn = self._create_phase_button(phase_id, name, desc)
            self.phase_buttons[phase_id] = btn
            phases_layout.addWidget(btn)

        layout.addWidget(phases_frame)

        # Pulsanti azione
        layout.addSpacing(5)

        action_frame = QtWidgets.QFrame()
        action_layout = QtWidgets.QHBoxLayout(action_frame)
        action_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_prev = QtWidgets.QPushButton("◀ Precedente")
        self.btn_prev.clicked.connect(self._on_prev)
        self.btn_prev.setEnabled(False)
        action_layout.addWidget(self.btn_prev)

        self.btn_next = QtWidgets.QPushButton("Successiva ▶")
        self.btn_next.clicked.connect(self._on_next)
        action_layout.addWidget(self.btn_next)

        layout.addWidget(action_frame)

        # Spacer finale
        layout.addStretch()

        # Info corrente
        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("""
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 3px;
            padding: 5px;
            font-size: 11px;
        """)
        layout.addWidget(self.info_label)

        # Aggiorna stato iniziale
        self._update_display()

    def _create_phase_button(self, phase_id, name, desc):
        """Crea un pulsante per una fase."""
        btn = QtWidgets.QPushButton()
        btn.setCheckable(True)
        btn.setProperty("phase_id", phase_id)
        btn.clicked.connect(lambda: self._on_phase_clicked(phase_id))

        # Stile
        btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 5px 8px;
                border: 1px solid #ddd;
                border-radius: 3px;
                background-color: #fff;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #e8f4fc;
                border-color: #4a90d9;
            }
            QPushButton:checked {
                background-color: #d4edda;
                border-color: #28a745;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #999;
            }
        """)

        self._update_button_text(btn, phase_id, name)
        btn.setToolTip(f"Fase {phase_id}: {name}\n{desc}")

        return btn

    def _update_button_text(self, btn, phase_id, name):
        """Aggiorna il testo del pulsante con stato."""
        status = self.phase_status.get(phase_id, 'not_started')
        icon = self.STATUS_ICONS.get(status, '○')
        btn.setText(f"{icon}  {phase_id}. {name}")

    def _on_phase_clicked(self, phase_id):
        """Gestisce click su una fase."""
        # Deseleziona altri
        for pid, btn in self.phase_buttons.items():
            btn.setChecked(pid == phase_id)

        self.current_phase = phase_id
        self._update_display()
        self.phase_clicked.emit(phase_id)

    def _on_prev(self):
        """Vai alla fase precedente."""
        if self.current_phase > 1:
            self._on_phase_clicked(self.current_phase - 1)

    def _on_next(self):
        """Vai alla fase successiva."""
        if self.current_phase < 12:
            self._on_phase_clicked(self.current_phase + 1)

    def _update_display(self):
        """Aggiorna visualizzazione."""
        # Aggiorna pulsanti navigazione
        self.btn_prev.setEnabled(self.current_phase > 1)
        self.btn_next.setEnabled(self.current_phase < 12)

        # Aggiorna testi pulsanti
        for phase_id, name, desc in self.PHASES:
            btn = self.phase_buttons.get(phase_id)
            if btn:
                self._update_button_text(btn, phase_id, name)
                btn.setChecked(phase_id == self.current_phase)

        # Aggiorna info
        for pid, name, desc in self.PHASES:
            if pid == self.current_phase:
                status = self.phase_status.get(pid, 'not_started')
                status_text = {
                    'not_started': 'Non iniziata',
                    'in_progress': 'In corso',
                    'completed': 'Completata',
                    'error': 'Errore',
                    'skipped': 'Saltata',
                }.get(status, status)
                self.info_label.setText(
                    f"<b>Fase {pid}: {name}</b><br>"
                    f"{desc}<br>"
                    f"<i>Stato: {status_text}</i>"
                )
                break

        # Aggiorna progresso
        completed = sum(1 for s in self.phase_status.values() if s == 'completed')
        self.progress_bar.setValue(completed)

        # Aggiorna status label
        if completed == 0:
            self.status_label.setText("Inizia con Fase 1: Progetto")
        elif completed == 12:
            self.status_label.setText("✓ Workflow completato!")
        else:
            self.status_label.setText(f"Fase corrente: {self.current_phase}")

    def set_phase_status(self, phase_id, status):
        """Imposta lo stato di una fase."""
        if 1 <= phase_id <= 12:
            self.phase_status[phase_id] = status
            self._update_display()

    def go_to_phase(self, phase_id):
        """Naviga a una fase specifica."""
        if 1 <= phase_id <= 12:
            self._on_phase_clicked(phase_id)

    def get_current_phase(self):
        """Restituisce la fase corrente."""
        return self.current_phase


class MuraturaWorkbench(FreeCADGui.Workbench):
    """
    Workbench per analisi strutturale edifici in muratura.
    Conforme a NTC 2018.
    """

    MenuText = "Muratura"
    ToolTip = "Analisi strutturale edifici in muratura - NTC 2018"

    # Icona workbench (path relativo a resources/)
    Icon = """
        /* XPM */
        static char * muratura_xpm[] = {
        "16 16 4 1",
        " 	c None",
        ".	c #8B4513",
        "+	c #D2691E",
        "@	c #F4A460",
        "                ",
        " .............. ",
        " .@@.++.@@.++.@ ",
        " .............. ",
        " .++.@@.++.@@.+ ",
        " .............. ",
        " .@@.++.@@.++.@ ",
        " .............. ",
        " .++.@@.++.@@.+ ",
        " .............. ",
        " .@@.++.@@.++.@ ",
        " .............. ",
        " .++.@@.++.@@.+ ",
        " .............. ",
        " .@@.++.@@.++.@ ",
        "                "};
        """

    def Initialize(self):
        """Inizializza il workbench - carica comandi e toolbar."""

        # Import comandi
        from muratura.workbench.commands import (
            project, geometry, structural, analysis, export
        )

        # Lista comandi per categoria
        self.project_commands = [
            "Sketcher_NewSketch",  # Riusa Sketcher di FreeCAD
        ]

        self.geometry_commands = [
            "Muratura_NewWall",
            "Muratura_NewOpening",
            "Muratura_NewColumn",
            "Muratura_NewBeam",
            "Muratura_NewSlab",
            "Muratura_NewStair",
            "Muratura_NewRoof",
        ]

        self.structural_commands = [
            "Muratura_GenFoundations",
            "Muratura_GenRingBeams",
            "Muratura_SetMaterial",
            "Muratura_SetLoads",
        ]

        self.analysis_commands = [
            "Muratura_SetSeismic",
            "Muratura_RunPOR",
            "Muratura_RunSAM",
            "Muratura_ShowDCR",
        ]

        self.export_commands = [
            "Muratura_ExportReport",
            "Muratura_ExportIFC",
            "Muratura_ExportDXF",
        ]

        # Tutti i comandi
        all_commands = (
            self.geometry_commands +
            self.structural_commands +
            self.analysis_commands +
            self.export_commands
        )

        # Crea toolbar
        self.appendToolbar("Geometria", self.geometry_commands)
        self.appendToolbar("Struttura", self.structural_commands)
        self.appendToolbar("Analisi", self.analysis_commands)
        self.appendToolbar("Export", self.export_commands)

        # Crea menu
        self.appendMenu("Muratura", all_commands)
        self.appendMenu(["Muratura", "Geometria"], self.geometry_commands)
        self.appendMenu(["Muratura", "Struttura"], self.structural_commands)
        self.appendMenu(["Muratura", "Analisi"], self.analysis_commands)
        self.appendMenu(["Muratura", "Export"], self.export_commands)

    def Activated(self):
        """Chiamato quando il workbench viene attivato."""
        FreeCAD.Console.PrintMessage("Muratura Workbench attivato\n")

        # Imposta vista di default
        if FreeCAD.ActiveDocument is None:
            FreeCAD.newDocument("Muratura")

        # Crea e mostra il pannello workflow
        self._show_workflow_panel()

    def _show_workflow_panel(self):
        """Mostra il pannello workflow laterale."""
        main_window = FreeCADGui.getMainWindow()

        # Controlla se esiste già
        existing = main_window.findChild(QtWidgets.QDockWidget, "MuraturaWorkflowPanel")
        if existing:
            existing.show()
            return

        # Crea nuovo pannello
        self.workflow_panel = WorkflowPanel(main_window)
        main_window.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.workflow_panel)

        # Connetti segnale cambio fase
        self.workflow_panel.phase_clicked.connect(self._on_phase_changed)

        FreeCAD.Console.PrintMessage("Pannello Workflow Muratura attivato\n")

    def _on_phase_changed(self, phase_id):
        """Gestisce cambio fase nel workflow."""
        FreeCAD.Console.PrintMessage(f"Fase selezionata: {phase_id}\n")
        # TODO: Cambiare toolbar/pannello in base alla fase

    def _hide_workflow_panel(self):
        """Nasconde il pannello workflow."""
        main_window = FreeCADGui.getMainWindow()
        panel = main_window.findChild(QtWidgets.QDockWidget, "MuraturaWorkflowPanel")
        if panel:
            panel.hide()

    def Deactivated(self):
        """Chiamato quando si esce dal workbench."""
        # Nascondi pannello quando si cambia workbench
        self._hide_workflow_panel()

    def ContextMenu(self, recipient):
        """Menu contestuale."""
        self.appendContextMenu("Muratura", self.geometry_commands[:3])

    def GetClassName(self):
        return "Gui::PythonWorkbench"


# Registra il workbench
FreeCADGui.addWorkbench(MuraturaWorkbench())
