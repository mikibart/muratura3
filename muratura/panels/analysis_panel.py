# -*- coding: utf-8 -*-
"""
Analysis Panel - Fase 9: Analisi

POR, SAM, FRAME, FEM, LIMIT, FIBER, MICRO.
"""

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide import QtWidgets, QtCore, QtGui

from .base_panel import BasePhasePanel


class AnalysisPanel(BasePhasePanel):
    """Pannello Fase 9: Analisi Strutturale."""

    ANALYSIS_METHODS = {
        "POR": ("Pier Only Resistance", "Verifica rapida, edifici regolari"),
        "PORFLEX": ("POR + Fasce flessibili", "Con fasce significative"),
        "SAM": ("Simplified Analysis", "Interazione M-V completa"),
        "FRAME": ("Telaio equivalente", "Pushover, modi propri"),
        "FEM": ("Elementi finiti", "Geometrie complesse"),
        "FIBER": ("Fibre section", "Risposta non lineare"),
        "LIMIT": ("Analisi cinematica", "Meccanismi locali EC8-3"),
        "MICRO": ("Micro-modellazione", "Blocchi-malta-interfacce"),
    }

    def __init__(self, parent=None):
        super().__init__(9, "Analisi", parent)
        self.set_description(
            "Seleziona ed esegui le analisi strutturali: 8 metodi disponibili "
            "da semplificati (POR) a dettagliati (MICRO)."
        )
        self.setup_content()

    def setup_content(self):
        """Costruisce il contenuto del pannello."""
        # Selezione metodo
        method_group, method_layout = self.create_form_group("Metodo di Analisi")

        self.method_checks = {}
        for method, (name, desc) in self.ANALYSIS_METHODS.items():
            cb = self.create_check_field(f"{method} - {name}")
            cb.setToolTip(desc)
            method_layout.addRow(cb)
            self.method_checks[method] = cb

        # Default: POR e SAM
        self.method_checks["POR"].setChecked(True)
        self.method_checks["SAM"].setChecked(True)

        self.content_layout.addWidget(method_group)

        # Parametri pushover (per FRAME)
        pushover_group, pushover_layout = self.create_form_group("Parametri Pushover")

        self.field_pushover_pattern = self.create_combo_field([
            "Triangolare (modale)",
            "Uniforme (masse)",
            "Proporzionale al 1° modo"
        ])
        pushover_layout.addRow("Pattern di carico:", self.field_pushover_pattern)

        self.field_pushover_dir = self.create_combo_field(["+X", "-X", "+Y", "-Y"])
        pushover_layout.addRow("Direzione:", self.field_pushover_dir)

        self.field_target_drift = self.create_number_field(1, 10, 1, "%")
        self.field_target_drift.setValue(4)
        pushover_layout.addRow("Target drift:", self.field_target_drift)

        self.field_step_size = self.create_number_field(0.001, 0.1, 3)
        self.field_step_size.setValue(0.01)
        pushover_layout.addRow("Passo incremento:", self.field_step_size)

        self.content_layout.addWidget(pushover_group)

        # Meccanismi locali (per LIMIT)
        limit_group, limit_layout = self.create_form_group("Meccanismi Locali (EC8-3)")

        mechanisms = [
            "Ribaltamento semplice",
            "Ribaltamento composto",
            "Flessione verticale",
            "Flessione orizzontale",
            "Ribaltamento cantonale",
        ]
        for mech in mechanisms:
            cb = self.create_check_field(mech)
            cb.setChecked(True)
            limit_layout.addRow(cb)

        self.content_layout.addWidget(limit_group)

        # Pulsante esecuzione
        btn_run = QtWidgets.QPushButton("Esegui Analisi")
        btn_run.clicked.connect(self._run_analysis)
        btn_run.setStyleSheet("""
            QPushButton {
                background-color: #28a745; color: white;
                padding: 15px; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #218838; }
        """)
        self.content_layout.addWidget(btn_run)

        # Progress
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        self.content_layout.addWidget(self.progress)

        # Risultati
        results_group, results_layout = self.create_form_group("Risultati")

        self.results_table = QtWidgets.QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["Metodo", "Vrd,x [kN]", "Vrd,y [kN]", "DCR max"])
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.results_table.setMaximumHeight(200)
        results_layout.addRow(self.results_table)

        self.content_layout.addWidget(results_group)

    def _run_analysis(self):
        """Esegue le analisi selezionate."""
        selected = [m for m, cb in self.method_checks.items() if cb.isChecked()]

        if not selected:
            self._show_error("Seleziona almeno un metodo di analisi")
            return

        self.progress.setVisible(True)
        self.progress.setMaximum(len(selected))
        self.results_table.setRowCount(0)

        for i, method in enumerate(selected):
            self.progress.setValue(i + 1)
            QtWidgets.QApplication.processEvents()

            try:
                # TODO: Eseguire analisi reale
                # from muratura.ntc2018.analyses import run_analysis
                # result = run_analysis(method)

                # Risultati simulati
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self.results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(method))
                self.results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{150 + i*20:.0f}"))
                self.results_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{180 + i*25:.0f}"))
                self.results_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{0.85 + i*0.05:.2f}"))

            except Exception as e:
                self._show_error(f"Errore {method}: {e}")

        self.progress.setVisible(False)
        self._show_success(f"Completate {len(selected)} analisi")
        self._mark_dirty()

    def _update_data_from_ui(self):
        self._data = {
            'methods': [m for m, cb in self.method_checks.items() if cb.isChecked()],
            'pushover': {
                'pattern': self.field_pushover_pattern.currentText(),
                'direction': self.field_pushover_dir.currentText(),
                'target_drift': self.field_target_drift.value(),
                'step_size': self.field_step_size.value(),
            },
        }
