# -*- coding: utf-8 -*-
"""
Loads Panel - Fase 6: Carichi

G1, G2, Q, neve, vento, combinazioni NTC 2018.
"""

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide import QtWidgets, QtCore, QtGui

from .base_panel import BasePhasePanel


class LoadsPanel(BasePhasePanel):
    """Pannello Fase 6: Carichi."""

    # Carichi variabili NTC 2018 Tab. 3.1.II
    VARIABLE_LOADS = {
        "A - Residenziale": (2.00, 2.00, 0.7, 0.5, 0.3),
        "B - Uffici": (3.00, 2.00, 0.7, 0.5, 0.3),
        "C1 - Sale riunioni": (3.00, 4.00, 0.7, 0.7, 0.6),
        "C2 - Sale spettacolo": (4.00, 4.00, 0.7, 0.7, 0.6),
        "C3 - Musei": (5.00, 5.00, 0.7, 0.7, 0.6),
        "D1 - Negozi": (4.00, 4.00, 0.7, 0.7, 0.6),
        "D2 - Centri commerciali": (5.00, 5.00, 0.7, 0.7, 0.6),
        "E - Magazzini": (6.00, 7.00, 1.0, 0.9, 0.8),
        "F - Autorimesse ≤30kN": (2.50, 20.00, 0.7, 0.7, 0.6),
        "H - Coperture": (0.50, 1.20, 0.0, 0.0, 0.0),
    }

    # Zone neve NTC 2018
    SNOW_ZONES = {
        "I-Alpina": 1.50,
        "I-Mediterranea": 1.50,
        "II": 1.00,
        "III": 0.60,
    }

    def __init__(self, parent=None):
        super().__init__(6, "Carichi", parent)
        self.set_description(
            "Definisci i carichi secondo NTC 2018: permanenti (G1, G2), "
            "variabili (Q), neve e vento."
        )
        self.setup_content()

    def setup_content(self):
        """Costruisce il contenuto del pannello."""
        tabs = QtWidgets.QTabWidget()

        # Tab Permanenti
        perm_tab = QtWidgets.QWidget()
        perm_layout = QtWidgets.QVBoxLayout(perm_tab)

        group1, layout1 = self.create_form_group("Permanenti Non Strutturali (G2)")

        self.field_g2_massetto = self.create_number_field(0, 3.0, 2, "kN/m²")
        self.field_g2_massetto.setValue(1.00)
        layout1.addRow("Massetto:", self.field_g2_massetto)

        self.field_g2_pavimento = self.create_number_field(0, 2.0, 2, "kN/m²")
        self.field_g2_pavimento.setValue(0.50)
        layout1.addRow("Pavimento:", self.field_g2_pavimento)

        self.field_g2_intonaco = self.create_number_field(0, 1.0, 2, "kN/m²")
        self.field_g2_intonaco.setValue(0.40)
        layout1.addRow("Intonaco:", self.field_g2_intonaco)

        self.field_g2_tramezze = self.create_number_field(0, 2.0, 2, "kN/m²")
        self.field_g2_tramezze.setValue(1.20)
        layout1.addRow("Tramezze:", self.field_g2_tramezze)

        self.label_g2_total = QtWidgets.QLabel("Totale G2: 3.10 kN/m²")
        self.label_g2_total.setStyleSheet("font-weight: bold;")
        layout1.addRow(self.label_g2_total)

        perm_layout.addWidget(group1)
        perm_layout.addStretch()
        tabs.addTab(perm_tab, "Permanenti")

        # Tab Variabili
        var_tab = QtWidgets.QWidget()
        var_layout = QtWidgets.QVBoxLayout(var_tab)

        group2, layout2 = self.create_form_group("Carichi Variabili (Q)")

        self.field_use_category = self.create_combo_field(list(self.VARIABLE_LOADS.keys()))
        self.field_use_category.currentIndexChanged.connect(self._on_category_changed)
        layout2.addRow("Categoria d'uso:", self.field_use_category)

        self.field_qk = self.create_number_field(0, 20, 2, "kN/m²")
        self.field_qk.setValue(2.00)
        layout2.addRow("qk distribuito:", self.field_qk)

        self.field_Qk = self.create_number_field(0, 50, 2, "kN")
        self.field_Qk.setValue(2.00)
        layout2.addRow("Qk concentrato:", self.field_Qk)

        coeff_label = QtWidgets.QLabel("Coefficienti ψ:")
        coeff_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout2.addRow(coeff_label)

        self.field_psi0 = self.create_number_field(0, 1, 2)
        self.field_psi0.setValue(0.7)
        layout2.addRow("ψ₀:", self.field_psi0)

        self.field_psi1 = self.create_number_field(0, 1, 2)
        self.field_psi1.setValue(0.5)
        layout2.addRow("ψ₁:", self.field_psi1)

        self.field_psi2 = self.create_number_field(0, 1, 2)
        self.field_psi2.setValue(0.3)
        layout2.addRow("ψ₂:", self.field_psi2)

        var_layout.addWidget(group2)
        var_layout.addStretch()
        tabs.addTab(var_tab, "Variabili")

        # Tab Neve/Vento
        env_tab = QtWidgets.QWidget()
        env_layout = QtWidgets.QVBoxLayout(env_tab)

        group3, layout3 = self.create_form_group("Carico Neve (NTC §3.4)")

        self.field_snow_zone = self.create_combo_field(list(self.SNOW_ZONES.keys()))
        self.field_snow_zone.currentIndexChanged.connect(self._update_snow)
        layout3.addRow("Zona:", self.field_snow_zone)

        self.field_altitude = self.create_number_field(0, 3000, 0, "m s.l.m.")
        self.field_altitude.setValue(200)
        self.field_altitude.valueChanged.connect(self._update_snow)
        layout3.addRow("Quota:", self.field_altitude)

        self.field_snow_mu = self.create_number_field(0, 1.5, 2)
        self.field_snow_mu.setValue(0.8)
        layout3.addRow("μi (forma):", self.field_snow_mu)

        self.field_snow_ce = self.create_combo_field(["0.9 (ventosa)", "1.0 (normale)", "1.1 (riparata)"])
        self.field_snow_ce.setCurrentIndex(1)
        layout3.addRow("CE (esposizione):", self.field_snow_ce)

        self.label_qs = QtWidgets.QLabel("qs = 0.80 kN/m²")
        self.label_qs.setStyleSheet("font-weight: bold; background: #e3f2fd; padding: 5px;")
        layout3.addRow("Carico neve:", self.label_qs)

        env_layout.addWidget(group3)

        group4, layout4 = self.create_form_group("Carico Vento (NTC §3.3)")

        self.field_wind_zone = self.create_combo_field(["1", "2", "3", "4", "5", "6", "7", "8", "9"])
        layout4.addRow("Zona:", self.field_wind_zone)

        self.field_terrain = self.create_combo_field(["A", "B", "C", "D"])
        self.field_terrain.setCurrentIndex(1)
        layout4.addRow("Categoria terreno:", self.field_terrain)

        self.field_building_height = self.create_number_field(3, 50, 1, "m")
        self.field_building_height.setValue(9)
        layout4.addRow("Altezza edificio:", self.field_building_height)

        self.label_qb = QtWidgets.QLabel("qb = 0.39 kN/m²")
        self.label_qb.setStyleSheet("font-weight: bold; background: #e3f2fd; padding: 5px;")
        layout4.addRow("Pressione cinetica:", self.label_qb)

        env_layout.addWidget(group4)
        env_layout.addStretch()
        tabs.addTab(env_tab, "Neve/Vento")

        # Tab Combinazioni
        comb_tab = QtWidgets.QWidget()
        comb_layout = QtWidgets.QVBoxLayout(comb_tab)

        group5, layout5 = self.create_form_group("Combinazioni di Carico")

        btn_generate = QtWidgets.QPushButton("Genera Combinazioni NTC 2018")
        btn_generate.clicked.connect(self._generate_combinations)
        layout5.addRow(btn_generate)

        self.comb_text = QtWidgets.QTextEdit()
        self.comb_text.setReadOnly(True)
        self.comb_text.setMaximumHeight(200)
        layout5.addRow(self.comb_text)

        comb_layout.addWidget(group5)
        comb_layout.addStretch()
        tabs.addTab(comb_tab, "Combinazioni")

        self.content_layout.addWidget(tabs)

    def _on_category_changed(self):
        """Aggiorna carichi da categoria."""
        cat = self.field_use_category.currentText()
        if cat in self.VARIABLE_LOADS:
            qk, Qk, psi0, psi1, psi2 = self.VARIABLE_LOADS[cat]
            self.field_qk.setValue(qk)
            self.field_Qk.setValue(Qk)
            self.field_psi0.setValue(psi0)
            self.field_psi1.setValue(psi1)
            self.field_psi2.setValue(psi2)

    def _update_snow(self):
        """Calcola carico neve."""
        zone = self.field_snow_zone.currentText()
        qsk = self.SNOW_ZONES.get(zone, 1.0)
        mu = self.field_snow_mu.value()
        ce_text = self.field_snow_ce.currentText()
        ce = float(ce_text.split()[0])

        qs = mu * ce * qsk
        self.label_qs.setText(f"qs = {qs:.2f} kN/m²")

    def _generate_combinations(self):
        """Genera combinazioni di carico."""
        text = "COMBINAZIONI SLU (NTC §2.5.3):\n"
        text += "─" * 40 + "\n"
        text += "1. 1.3×G1 + 1.5×G2 + 1.5×Qk\n"
        text += "2. 1.0×G1 + 0×G2 + 0×Qk\n\n"
        text += "COMBINAZIONI SLE:\n"
        text += "─" * 40 + "\n"
        text += "Rara: G1 + G2 + Qk\n"
        text += "Frequente: G1 + G2 + ψ1×Qk\n"
        text += "Q.permanente: G1 + G2 + ψ2×Qk\n\n"
        text += "COMBINAZIONE SISMICA:\n"
        text += "─" * 40 + "\n"
        text += "G1 + G2 + E + ψ2×Qk\n"

        self.comb_text.setText(text)

    def _update_data_from_ui(self):
        g2_total = (
            self.field_g2_massetto.value() +
            self.field_g2_pavimento.value() +
            self.field_g2_intonaco.value() +
            self.field_g2_tramezze.value()
        )
        self.label_g2_total.setText(f"Totale G2: {g2_total:.2f} kN/m²")

        self._data = {
            'G2': {
                'massetto': self.field_g2_massetto.value(),
                'pavimento': self.field_g2_pavimento.value(),
                'intonaco': self.field_g2_intonaco.value(),
                'tramezze': self.field_g2_tramezze.value(),
                'total': g2_total,
            },
            'Q': {
                'category': self.field_use_category.currentText(),
                'qk': self.field_qk.value(),
                'Qk': self.field_Qk.value(),
                'psi0': self.field_psi0.value(),
                'psi1': self.field_psi1.value(),
                'psi2': self.field_psi2.value(),
            },
            'snow': {
                'zone': self.field_snow_zone.currentText(),
                'altitude': self.field_altitude.value(),
            },
            'wind': {
                'zone': self.field_wind_zone.currentText(),
                'terrain': self.field_terrain.currentText(),
            },
        }
