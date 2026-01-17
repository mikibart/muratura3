# -*- coding: utf-8 -*-
"""
Floors Panel - Fase 5: Solai e Coperture

Tipologia, rigidezza, orditura, coperture.
"""

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide import QtWidgets, QtCore, QtGui

from .base_panel import BasePhasePanel


class FloorsPanel(BasePhasePanel):
    """Pannello Fase 5: Solai e Coperture."""

    FLOOR_TYPES = {
        "Latero-cemento": [
            ("16+4 travetti", 0.20, 2.8),
            ("20+4 travetti", 0.24, 3.2),
            ("20+5 predalles", 0.25, 3.5),
            ("25+5 predalles", 0.30, 4.0),
        ],
        "Legno": [
            ("Semplice orditura", 0.20, 1.0),
            ("Doppia orditura", 0.25, 1.3),
            ("Con tavolato", 0.22, 1.5),
            ("Con soletta collaborante", 0.30, 2.2),
        ],
        "Acciaio": [
            ("Putrelle + voltine", 0.25, 3.0),
            ("Putrelle + tavelloni", 0.22, 2.5),
            ("Lamiera grecata", 0.15, 2.8),
        ],
        "Volte": [
            ("A botte", 0.20, 4.0),
            ("A crociera", 0.18, 3.5),
            ("A padiglione", 0.18, 3.5),
        ],
    }

    STIFFNESS_CLASSES = ["Rigido", "Semirigido", "Flessibile"]

    def __init__(self, parent=None):
        super().__init__(5, "Solai", parent)
        self.set_description(
            "Definisci tipologia solai, rigidezza di piano e orditura. "
            "Configura anche le coperture."
        )
        self.setup_content()

    def setup_content(self):
        """Costruisce il contenuto del pannello."""
        tabs = QtWidgets.QTabWidget()

        # Tab Solai
        floor_tab = QtWidgets.QWidget()
        floor_layout = QtWidgets.QVBoxLayout(floor_tab)

        group1, layout1 = self.create_form_group("Tipologia Solaio")

        self.field_floor_category = self.create_combo_field(list(self.FLOOR_TYPES.keys()))
        self.field_floor_category.currentIndexChanged.connect(self._on_floor_category_changed)
        layout1.addRow("Categoria:", self.field_floor_category)

        self.field_floor_type = QtWidgets.QComboBox()
        layout1.addRow("Tipologia:", self.field_floor_type)

        self.field_floor_thickness = self.create_number_field(0.10, 0.60, 2, "m")
        self.field_floor_thickness.setValue(0.24)
        layout1.addRow("Spessore:", self.field_floor_thickness)

        self.field_floor_weight = self.create_number_field(1.0, 8.0, 2, "kN/m²")
        self.field_floor_weight.setValue(3.2)
        layout1.addRow("Peso proprio:", self.field_floor_weight)

        self.field_floor_direction = self.create_combo_field(["Direzione X", "Direzione Y", "Bidirezionale"])
        layout1.addRow("Orditura:", self.field_floor_direction)

        self.field_joist_spacing = self.create_number_field(0.30, 1.20, 2, "m")
        self.field_joist_spacing.setValue(0.50)
        layout1.addRow("Interasse travetti:", self.field_joist_spacing)

        floor_layout.addWidget(group1)

        # Rigidezza
        group2, layout2 = self.create_form_group("Rigidezza di Piano")

        self.field_stiffness = self.create_combo_field(self.STIFFNESS_CLASSES)
        layout2.addRow("Classificazione:", self.field_stiffness)

        info_label = QtWidgets.QLabel(
            "• Rigido: Soletta c.a. ≥4cm, trasmissione completa forze\n"
            "• Semirigido: Tavolato incrociato, rigidezza finita\n"
            "• Flessibile: Travetti senza collegamento, volte"
        )
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        layout2.addRow(info_label)

        floor_layout.addWidget(group2)
        floor_layout.addStretch()
        tabs.addTab(floor_tab, "Solai")

        # Tab Coperture
        roof_tab = QtWidgets.QWidget()
        roof_layout = QtWidgets.QVBoxLayout(roof_tab)

        group3, layout3 = self.create_form_group("Copertura")

        self.field_roof_type = self.create_combo_field([
            "Piana",
            "A una falda",
            "A due falde",
            "A padiglione",
            "Con capriate"
        ])
        layout3.addRow("Tipologia:", self.field_roof_type)

        self.field_roof_slope = self.create_number_field(0, 45, 1, "°")
        self.field_roof_slope.setValue(25)
        layout3.addRow("Inclinazione:", self.field_roof_slope)

        self.field_roof_weight = self.create_number_field(0.5, 5.0, 2, "kN/m²")
        self.field_roof_weight.setValue(1.5)
        layout3.addRow("Peso copertura:", self.field_roof_weight)

        self.field_roof_overhang = self.create_number_field(0, 1.5, 2, "m")
        self.field_roof_overhang.setValue(0.50)
        layout3.addRow("Sporto gronda:", self.field_roof_overhang)

        roof_layout.addWidget(group3)
        roof_layout.addStretch()
        tabs.addTab(roof_tab, "Coperture")

        self.content_layout.addWidget(tabs)
        self._on_floor_category_changed()

    def _on_floor_category_changed(self):
        """Aggiorna lista tipologie solaio."""
        category = self.field_floor_category.currentText()
        types = self.FLOOR_TYPES.get(category, [])

        self.field_floor_type.clear()
        for name, thickness, weight in types:
            self.field_floor_type.addItem(name)

        if types:
            self.field_floor_thickness.setValue(types[0][1])
            self.field_floor_weight.setValue(types[0][2])

    def _update_data_from_ui(self):
        self._data = {
            'floor': {
                'category': self.field_floor_category.currentText(),
                'type': self.field_floor_type.currentText(),
                'thickness': self.field_floor_thickness.value(),
                'weight': self.field_floor_weight.value(),
                'direction': self.field_floor_direction.currentText(),
                'joist_spacing': self.field_joist_spacing.value(),
                'stiffness': self.field_stiffness.currentText(),
            },
            'roof': {
                'type': self.field_roof_type.currentText(),
                'slope': self.field_roof_slope.value(),
                'weight': self.field_roof_weight.value(),
                'overhang': self.field_roof_overhang.value(),
            },
        }
