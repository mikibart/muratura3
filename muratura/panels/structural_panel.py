# -*- coding: utf-8 -*-
"""
Structural Panel - Fase 4: Struttura

Cordoli, travi, pilastri, catene, piattabande, fondazioni.
"""

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide import QtWidgets, QtCore, QtGui

from .base_panel import BasePhasePanel


class StructuralPanel(BasePhasePanel):
    """Pannello Fase 4: Elementi strutturali."""

    def __init__(self, parent=None):
        super().__init__(4, "Struttura", parent)
        self.set_description(
            "Aggiungi elementi strutturali: cordoli, travi, pilastri, catene e fondazioni."
        )
        self.setup_content()

    def setup_content(self):
        """Costruisce il contenuto del pannello."""
        tabs = QtWidgets.QTabWidget()

        # Tab Cordoli
        cordoli_tab = QtWidgets.QWidget()
        cordoli_layout = QtWidgets.QVBoxLayout(cordoli_tab)

        group1, layout1 = self.create_form_group("Cordoli di Piano")
        self.field_cordolo_b = self.create_number_field(0.20, 1.0, 2, "m")
        self.field_cordolo_b.setValue(0.30)
        layout1.addRow("Base:", self.field_cordolo_b)

        self.field_cordolo_h = self.create_number_field(0.20, 0.50, 2, "m")
        self.field_cordolo_h.setValue(0.25)
        layout1.addRow("Altezza:", self.field_cordolo_h)

        self.field_cordolo_cls = self.create_combo_field(["C20/25", "C25/30", "C28/35", "C32/40"])
        self.field_cordolo_cls.setCurrentIndex(1)
        layout1.addRow("Classe cls:", self.field_cordolo_cls)

        self.field_cordolo_arm = self.create_combo_field(["4Φ12", "4Φ14", "4Φ16", "6Φ16"])
        self.field_cordolo_arm.setCurrentIndex(2)
        layout1.addRow("Armatura:", self.field_cordolo_arm)

        btn_gen_cordoli = QtWidgets.QPushButton("Genera Cordoli Automatici")
        btn_gen_cordoli.clicked.connect(self._on_generate_cordoli)
        layout1.addRow("", btn_gen_cordoli)

        cordoli_layout.addWidget(group1)
        cordoli_layout.addStretch()
        tabs.addTab(cordoli_tab, "Cordoli")

        # Tab Catene
        catene_tab = QtWidgets.QWidget()
        catene_layout = QtWidgets.QVBoxLayout(catene_tab)

        group2, layout2 = self.create_form_group("Tiranti e Catene")
        self.field_tirante_tipo = self.create_combo_field(["Barra piena", "Piattina", "Cavo"])
        layout2.addRow("Tipo:", self.field_tirante_tipo)

        self.field_tirante_mat = self.create_combo_field(["S235", "S275", "S355"])
        layout2.addRow("Materiale:", self.field_tirante_mat)

        self.field_tirante_diam = self.create_combo_field(["Φ16", "Φ18", "Φ20", "Φ22", "Φ24", "Φ26"])
        self.field_tirante_diam.setCurrentIndex(2)
        layout2.addRow("Diametro:", self.field_tirante_diam)

        self.field_tirante_pre = self.create_number_field(0, 200, 1, "kN")
        layout2.addRow("Pretensione:", self.field_tirante_pre)

        self.field_capochiave = self.create_combo_field(["Piastra 30×30", "Paletto", "Ancora a coda di rondine"])
        layout2.addRow("Capochiave:", self.field_capochiave)

        catene_layout.addWidget(group2)
        catene_layout.addStretch()
        tabs.addTab(catene_tab, "Catene")

        # Tab Fondazioni
        fond_tab = QtWidgets.QWidget()
        fond_layout = QtWidgets.QVBoxLayout(fond_tab)

        group3, layout3 = self.create_form_group("Fondazioni Continue")
        self.field_fond_tipo = self.create_combo_field(["Continua", "A platea", "A travi rovesce"])
        layout3.addRow("Tipologia:", self.field_fond_tipo)

        self.field_fond_b = self.create_number_field(0.40, 2.0, 2, "m")
        self.field_fond_b.setValue(0.80)
        layout3.addRow("Larghezza:", self.field_fond_b)

        self.field_fond_h = self.create_number_field(0.30, 1.0, 2, "m")
        self.field_fond_h.setValue(0.50)
        layout3.addRow("Altezza:", self.field_fond_h)

        btn_gen_fond = QtWidgets.QPushButton("Genera Fondazioni")
        btn_gen_fond.clicked.connect(self._on_generate_fondazioni)
        layout3.addRow("", btn_gen_fond)

        fond_layout.addWidget(group3)
        fond_layout.addStretch()
        tabs.addTab(fond_tab, "Fondazioni")

        self.content_layout.addWidget(tabs)

    def _on_generate_cordoli(self):
        """Genera cordoli automaticamente."""
        self._show_info("Generazione cordoli in sviluppo")

    def _on_generate_fondazioni(self):
        """Genera fondazioni automaticamente."""
        self._show_info("Generazione fondazioni in sviluppo")

    def _update_data_from_ui(self):
        self._data = {
            'cordolo': {
                'base': self.field_cordolo_b.value(),
                'height': self.field_cordolo_h.value(),
                'concrete': self.field_cordolo_cls.currentText(),
                'reinforcement': self.field_cordolo_arm.currentText(),
            },
            'tirante': {
                'type': self.field_tirante_tipo.currentText(),
                'material': self.field_tirante_mat.currentText(),
                'diameter': self.field_tirante_diam.currentText(),
                'pretension': self.field_tirante_pre.value(),
            },
            'foundation': {
                'type': self.field_fond_tipo.currentText(),
                'width': self.field_fond_b.value(),
                'height': self.field_fond_h.value(),
            },
        }
