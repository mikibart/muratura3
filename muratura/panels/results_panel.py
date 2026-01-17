# -*- coding: utf-8 -*-
"""
Results Panel - Fase 10: Verifiche e Risultati

DCR, indice rischio, classe sismica.
"""

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide import QtWidgets, QtCore, QtGui

from .base_panel import BasePhasePanel


class ResultsPanel(BasePhasePanel):
    """Pannello Fase 10: Verifiche e Risultati."""

    # Classi rischio sismico DM 58/2017
    RISK_CLASSES = {
        "A+": (0, 0.50, 100),
        "A": (0.50, 1.0, 80),
        "B": (1.0, 1.5, 60),
        "C": (1.5, 2.5, 45),
        "D": (2.5, 3.5, 30),
        "E": (3.5, 4.5, 15),
        "F": (4.5, 7.5, 10),
        "G": (7.5, 100, 0),
    }

    def __init__(self, parent=None):
        super().__init__(10, "Verifiche", parent)
        self.set_description(
            "Visualizza i risultati delle verifiche: DCR, indice di rischio sismico "
            "e classificazione sismica dell'edificio."
        )
        self.setup_content()

    def setup_content(self):
        """Costruisce il contenuto del pannello."""
        # Riepilogo verifiche
        summary_group, summary_layout = self.create_form_group("Riepilogo Verifiche SLU")

        self.label_verified = QtWidgets.QLabel()
        self.label_verified.setStyleSheet("""
            font-size: 18px; font-weight: bold; padding: 15px;
            background: #d4edda; color: #155724; border-radius: 5px;
        """)
        self.label_verified.setAlignment(QtCore.Qt.AlignCenter)
        summary_layout.addRow(self.label_verified)

        stats_widget = QtWidgets.QWidget()
        stats_layout = QtWidgets.QHBoxLayout(stats_widget)

        self.label_piers_ok = QtWidgets.QLabel("Maschi OK: -/-")
        stats_layout.addWidget(self.label_piers_ok)

        self.label_spandrels_ok = QtWidgets.QLabel("Fasce OK: -/-")
        stats_layout.addWidget(self.label_spandrels_ok)

        summary_layout.addRow(stats_widget)

        self.content_layout.addWidget(summary_group)

        # Indice di Rischio
        ir_group, ir_layout = self.create_form_group("Indice di Rischio Sismico")

        self.label_ir = QtWidgets.QLabel()
        self.label_ir.setStyleSheet("""
            font-size: 24px; font-weight: bold; padding: 20px;
            background: #fff3cd; border-radius: 5px;
        """)
        self.label_ir.setAlignment(QtCore.Qt.AlignCenter)
        ir_layout.addRow(self.label_ir)

        ir_info = QtWidgets.QLabel(
            "IR = PGA_Capacità / PGA_Domanda\n"
            "IR ≥ 1.0 → Edificio verificato a SLV\n"
            "IR < 1.0 → Edificio NON verificato"
        )
        ir_info.setStyleSheet("color: #666; font-size: 11px;")
        ir_layout.addRow(ir_info)

        self.content_layout.addWidget(ir_group)

        # Classe Rischio
        class_group, class_layout = self.create_form_group("Classe di Rischio Sismico (DM 58/2017)")

        self.label_class = QtWidgets.QLabel()
        self.label_class.setStyleSheet("""
            font-size: 36px; font-weight: bold; padding: 20px;
            background: #17a2b8; color: white; border-radius: 10px;
        """)
        self.label_class.setAlignment(QtCore.Qt.AlignCenter)
        class_layout.addRow(self.label_class)

        self.content_layout.addWidget(class_group)

        # Elementi critici
        critical_group, critical_layout = self.create_form_group("Elementi Critici (Top 10)")

        self.critical_table = QtWidgets.QTableWidget(0, 5)
        self.critical_table.setHorizontalHeaderLabels([
            "Elemento", "Piano", "DCR", "Modo rottura", "Azione"
        ])
        self.critical_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.critical_table.setMaximumHeight(200)
        critical_layout.addRow(self.critical_table)

        self.content_layout.addWidget(critical_group)

        # Mappa DCR
        dcr_group, dcr_layout = self.create_form_group("Mappa DCR")

        legend = QtWidgets.QLabel(
            "🟢 DCR ≤ 0.5: Molto verificato | "
            "🟡 DCR ≤ 0.8: Verificato | "
            "🟠 DCR ≤ 1.0: Al limite | "
            "🔴 DCR > 1.0: NON verificato"
        )
        legend.setStyleSheet("font-size: 11px;")
        dcr_layout.addRow(legend)

        btn_show_dcr = QtWidgets.QPushButton("Mostra Mappa DCR 3D")
        btn_show_dcr.clicked.connect(self._show_dcr_map)
        dcr_layout.addRow(btn_show_dcr)

        self.content_layout.addWidget(dcr_group)

        # Carica risultati demo
        self._load_demo_results()

    def _load_demo_results(self):
        """Carica risultati demo."""
        # Verifiche
        self.label_verified.setText("✓ EDIFICIO VERIFICATO (con riserva)")
        self.label_piers_ok.setText("Maschi OK: 22/24")
        self.label_spandrels_ok.setText("Fasce OK: 16/18")

        # Indice rischio
        ir = 0.78
        self.label_ir.setText(f"IR = {ir:.2f}")
        if ir < 1.0:
            self.label_ir.setStyleSheet("""
                font-size: 24px; font-weight: bold; padding: 20px;
                background: #f8d7da; color: #721c24; border-radius: 5px;
            """)

        # Classe rischio
        self.label_class.setText("C")

        # Elementi critici
        critical = [
            ("Maschio M12", "1", "1.15", "Taglio diagonale", "Rinforzo"),
            ("Maschio M08", "0", "1.08", "Presso-flessione", "Rinforzo"),
            ("Fascia F05", "1", "0.98", "Flessione", "Monitoraggio"),
            ("Maschio M15", "1", "0.95", "Scorrimento", "Monitoraggio"),
            ("Maschio M03", "0", "0.92", "Taglio diagonale", "OK"),
        ]

        self.critical_table.setRowCount(len(critical))
        for i, (elem, floor, dcr, mode, action) in enumerate(critical):
            self.critical_table.setItem(i, 0, QtWidgets.QTableWidgetItem(elem))
            self.critical_table.setItem(i, 1, QtWidgets.QTableWidgetItem(floor))
            self.critical_table.setItem(i, 2, QtWidgets.QTableWidgetItem(dcr))
            self.critical_table.setItem(i, 3, QtWidgets.QTableWidgetItem(mode))
            self.critical_table.setItem(i, 4, QtWidgets.QTableWidgetItem(action))

    def _show_dcr_map(self):
        """Mostra mappa DCR 3D."""
        self._show_info("Visualizzazione mappa DCR in sviluppo")

    def _update_data_from_ui(self):
        self._data = {
            'ir': 0.78,
            'risk_class': 'C',
            'verified': True,
        }
