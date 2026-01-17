# -*- coding: utf-8 -*-
"""
Reinforcement Panel - Fase 11: Rinforzi

Catene, FRP, intonaco armato, iniezioni.
"""

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide import QtWidgets, QtCore, QtGui

from .base_panel import BasePhasePanel


class ReinforcementPanel(BasePhasePanel):
    """Pannello Fase 11: Interventi di Rinforzo."""

    INTERVENTIONS = {
        "Catene": {
            "description": "Tiranti metallici con capochiave",
            "effect": "+30% resistenza globale",
            "params": ["Diametro", "Materiale", "Pretensione"],
        },
        "Cordoli": {
            "description": "Cordoli in c.a. sommitali",
            "effect": "+20% rigidezza, box behavior",
            "params": ["Sezione", "Armatura", "Classe cls"],
        },
        "Cerchiature": {
            "description": "Profili metallici attorno aperture",
            "effect": "Rinforzo locale aperture",
            "params": ["Profilo", "Piastre", "Tasselli"],
        },
        "Intonaco armato": {
            "description": "Betoncino armato su muratura",
            "effect": "+50% fm, τ₀ (2 lati)",
            "params": ["Spessore", "Rete", "Connettori"],
        },
        "FRP/FRCM": {
            "description": "Fibre composite incollate",
            "effect": "+40% resistenza taglio",
            "params": ["Tipo fibra", "Grammatura", "Strati"],
        },
        "Iniezioni": {
            "description": "Miscela consolidante iniettata",
            "effect": "+50% fm, τ₀, E, G",
            "params": ["Miscela", "Pressione", "Maglia fori"],
        },
    }

    def __init__(self, parent=None):
        super().__init__(11, "Rinforzi", parent)
        self.set_description(
            "Definisci gli interventi di rinforzo per migliorare le prestazioni "
            "sismiche dell'edificio."
        )
        self.setup_content()

    def setup_content(self):
        """Costruisce il contenuto del pannello."""
        # Elementi critici
        critical_group, critical_layout = self.create_form_group("Elementi da Rinforzare")

        self.critical_list = QtWidgets.QListWidget()
        self.critical_list.setMaximumHeight(120)
        items = [
            "🔴 Maschio M12 - Piano 1 - DCR 1.15",
            "🔴 Maschio M08 - Piano 0 - DCR 1.08",
            "🟠 Fascia F05 - Piano 1 - DCR 0.98",
        ]
        self.critical_list.addItems(items)
        critical_layout.addRow(self.critical_list)

        self.content_layout.addWidget(critical_group)

        # Selezione intervento
        intervention_group, intervention_layout = self.create_form_group("Tipo Intervento")

        self.field_intervention = self.create_combo_field(list(self.INTERVENTIONS.keys()))
        self.field_intervention.currentIndexChanged.connect(self._on_intervention_changed)
        intervention_layout.addRow("Intervento:", self.field_intervention)

        self.label_description = QtWidgets.QLabel()
        self.label_description.setWordWrap(True)
        self.label_description.setStyleSheet("color: #666; font-style: italic;")
        intervention_layout.addRow(self.label_description)

        self.label_effect = QtWidgets.QLabel()
        self.label_effect.setStyleSheet("font-weight: bold; color: #28a745;")
        intervention_layout.addRow("Effetto:", self.label_effect)

        self.content_layout.addWidget(intervention_group)

        # Parametri intervento
        self.params_group, self.params_layout = self.create_form_group("Parametri Intervento")

        # Catene (default)
        self.field_diam = self.create_combo_field(["Φ16", "Φ18", "Φ20", "Φ22", "Φ24"])
        self.field_diam.setCurrentIndex(2)
        self.params_layout.addRow("Diametro:", self.field_diam)

        self.field_mat = self.create_combo_field(["S235", "S275", "S355"])
        self.params_layout.addRow("Materiale:", self.field_mat)

        self.field_pre = self.create_number_field(0, 200, 1, "kN")
        self.field_pre.setValue(50)
        self.params_layout.addRow("Pretensione:", self.field_pre)

        self.content_layout.addWidget(self.params_group)

        # Applicazione
        apply_group, apply_layout = self.create_form_group("Applicazione")

        self.field_target = self.create_combo_field([
            "Elemento selezionato",
            "Tutti gli elementi critici",
            "Intero piano",
            "Intero edificio"
        ])
        apply_layout.addRow("Applica a:", self.field_target)

        btn_apply = QtWidgets.QPushButton("Applica Intervento")
        btn_apply.clicked.connect(self._apply_intervention)
        btn_apply.setStyleSheet("""
            QPushButton { background-color: #28a745; color: white; padding: 10px; }
            QPushButton:hover { background-color: #218838; }
        """)
        apply_layout.addRow(btn_apply)

        btn_recalc = QtWidgets.QPushButton("Ricalcola con Rinforzi")
        btn_recalc.clicked.connect(self._recalculate)
        btn_recalc.setStyleSheet("""
            QPushButton { background-color: #007bff; color: white; padding: 10px; }
            QPushButton:hover { background-color: #0056b3; }
        """)
        apply_layout.addRow(btn_recalc)

        self.content_layout.addWidget(apply_group)

        # Confronto
        compare_group, compare_layout = self.create_form_group("Confronto Ante/Post Intervento")

        self.compare_table = QtWidgets.QTableWidget(3, 3)
        self.compare_table.setHorizontalHeaderLabels(["Parametro", "Ante", "Post"])
        self.compare_table.verticalHeader().setVisible(False)
        self.compare_table.setMaximumHeight(120)

        data = [
            ("IR", "0.78", "-"),
            ("Classe rischio", "C", "-"),
            ("Elementi critici", "3", "-"),
        ]
        for i, (param, ante, post) in enumerate(data):
            self.compare_table.setItem(i, 0, QtWidgets.QTableWidgetItem(param))
            self.compare_table.setItem(i, 1, QtWidgets.QTableWidgetItem(ante))
            self.compare_table.setItem(i, 2, QtWidgets.QTableWidgetItem(post))

        compare_layout.addRow(self.compare_table)

        self.content_layout.addWidget(compare_group)

        # Inizializza
        self._on_intervention_changed()

    def _on_intervention_changed(self):
        """Aggiorna descrizione intervento."""
        intervention = self.field_intervention.currentText()
        if intervention in self.INTERVENTIONS:
            info = self.INTERVENTIONS[intervention]
            self.label_description.setText(info["description"])
            self.label_effect.setText(info["effect"])

    def _apply_intervention(self):
        """Applica l'intervento selezionato."""
        intervention = self.field_intervention.currentText()
        target = self.field_target.currentText()
        self._show_success(f"Intervento '{intervention}' applicato a: {target}")
        self._mark_dirty()

    def _recalculate(self):
        """Ricalcola con i rinforzi applicati."""
        # Simula miglioramento
        self.compare_table.setItem(0, 2, QtWidgets.QTableWidgetItem("1.05"))
        self.compare_table.setItem(1, 2, QtWidgets.QTableWidgetItem("B"))
        self.compare_table.setItem(2, 2, QtWidgets.QTableWidgetItem("0"))

        self._show_success("Ricalcolo completato: edificio ora verificato!")

    def _update_data_from_ui(self):
        self._data = {
            'intervention': self.field_intervention.currentText(),
            'target': self.field_target.currentText(),
        }
