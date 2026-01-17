# -*- coding: utf-8 -*-
"""
Seismic Panel - Fase 7: Parametri Sismici

Località, sottosuolo, spettro di risposta, fattore q.
"""

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide import QtWidgets, QtCore, QtGui

from .base_panel import BasePhasePanel


class SeismicPanel(BasePhasePanel):
    """Pannello Fase 7: Parametri Sismici."""

    # Categorie sottosuolo NTC 2018 Tab. 3.2.II
    SOIL_CATEGORIES = {
        "A": ("Roccia affiorante", ">800 m/s", 1.00, 1.00),
        "B": ("Depositi sabbie/ghiaie", "360-800 m/s", 1.20, 1.10),
        "C": ("Depositi argille/sabbie", "180-360 m/s", 1.50, 1.05),
        "D": ("Depositi scadenti", "<180 m/s", 1.80, 1.25),
        "E": ("Strato su roccia", "-", 1.60, 1.15),
    }

    # Categorie topografiche NTC 2018 Tab. 3.2.IV
    TOPO_CATEGORIES = {
        "T1": ("Pianura", 1.0),
        "T2": ("Pendii <15°", 1.2),
        "T3": ("Rilievi H/L<0.2", 1.2),
        "T4": ("Rilievi H/L≥0.5", 1.4),
    }

    def __init__(self, parent=None):
        super().__init__(7, "Sismica", parent)
        self.set_description(
            "Definisci i parametri sismici: località, categoria sottosuolo, "
            "topografia e calcola lo spettro di risposta NTC 2018."
        )
        self.setup_content()

    def setup_content(self):
        """Costruisce il contenuto del pannello."""
        # Località
        loc_group, loc_layout = self.create_form_group("Localizzazione")

        self.field_comune = self.create_input_field("Nome comune")
        loc_layout.addRow("Comune:", self.field_comune)

        coord_widget = QtWidgets.QWidget()
        coord_layout = QtWidgets.QHBoxLayout(coord_widget)
        coord_layout.setContentsMargins(0, 0, 0, 0)

        self.field_lat = self.create_number_field(35, 48, 4, "°")
        self.field_lat.setValue(41.9028)
        coord_layout.addWidget(QtWidgets.QLabel("Lat:"))
        coord_layout.addWidget(self.field_lat)

        self.field_lon = self.create_number_field(6, 19, 4, "°")
        self.field_lon.setValue(12.4964)
        coord_layout.addWidget(QtWidgets.QLabel("Lon:"))
        coord_layout.addWidget(self.field_lon)

        loc_layout.addRow("Coordinate:", coord_widget)

        btn_search = QtWidgets.QPushButton("Cerca Parametri INGV")
        btn_search.clicked.connect(self._search_ingv)
        loc_layout.addRow("", btn_search)

        self.content_layout.addWidget(loc_group)

        # Parametri sismici
        params_group, params_layout = self.create_form_group("Parametri di Pericolosità")

        self.params_table = QtWidgets.QTableWidget(4, 4)
        self.params_table.setHorizontalHeaderLabels(["SL", "TR [anni]", "ag [g]", "F0"])
        self.params_table.verticalHeader().setVisible(False)
        self.params_table.setMaximumHeight(150)

        sl_names = ["SLO", "SLD", "SLV", "SLC"]
        tr_values = [30, 50, 475, 975]
        for i, (sl, tr) in enumerate(zip(sl_names, tr_values)):
            self.params_table.setItem(i, 0, QtWidgets.QTableWidgetItem(sl))
            self.params_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(tr)))
            self.params_table.setItem(i, 2, QtWidgets.QTableWidgetItem("-"))
            self.params_table.setItem(i, 3, QtWidgets.QTableWidgetItem("-"))

        params_layout.addRow(self.params_table)
        self.content_layout.addWidget(params_group)

        # Sottosuolo
        soil_group, soil_layout = self.create_form_group("Categoria Sottosuolo (Tab. 3.2.II)")

        self.field_soil = self.create_combo_field(
            [f"{k} - {v[0]} (Vs30 {v[1]})" for k, v in self.SOIL_CATEGORIES.items()]
        )
        self.field_soil.setCurrentIndex(2)  # C default
        self.field_soil.currentIndexChanged.connect(self._update_amplification)
        soil_layout.addRow("Categoria:", self.field_soil)

        self.label_ss = QtWidgets.QLabel("SS = 1.50")
        self.label_ss.setStyleSheet("font-weight: bold;")
        soil_layout.addRow("Coeff. stratigrafico:", self.label_ss)

        self.content_layout.addWidget(soil_group)

        # Topografia
        topo_group, topo_layout = self.create_form_group("Categoria Topografica (Tab. 3.2.IV)")

        self.field_topo = self.create_combo_field(
            [f"{k} - {v[0]}" for k, v in self.TOPO_CATEGORIES.items()]
        )
        self.field_topo.currentIndexChanged.connect(self._update_amplification)
        topo_layout.addRow("Categoria:", self.field_topo)

        self.label_st = QtWidgets.QLabel("ST = 1.0")
        self.label_st.setStyleSheet("font-weight: bold;")
        topo_layout.addRow("Coeff. topografico:", self.label_st)

        self.content_layout.addWidget(topo_group)

        # Fattore q
        q_group, q_layout = self.create_form_group("Fattore di Struttura q")

        self.field_regularity_plan = self.create_check_field("Regolare in pianta")
        self.field_regularity_plan.setChecked(True)
        self.field_regularity_plan.stateChanged.connect(self._update_q)
        q_layout.addRow(self.field_regularity_plan)

        self.field_regularity_elev = self.create_check_field("Regolare in elevazione")
        self.field_regularity_elev.setChecked(True)
        self.field_regularity_elev.stateChanged.connect(self._update_q)
        q_layout.addRow(self.field_regularity_elev)

        self.field_alpha_ratio = self.create_number_field(1.0, 1.8, 2)
        self.field_alpha_ratio.setValue(1.3)
        self.field_alpha_ratio.valueChanged.connect(self._update_q)
        q_layout.addRow("αu/α1:", self.field_alpha_ratio)

        self.label_q = QtWidgets.QLabel()
        self.label_q.setStyleSheet("""
            font-weight: bold; font-size: 14px;
            padding: 10px; background: #e3f2fd; border-radius: 5px;
        """)
        q_layout.addRow("Fattore q:", self.label_q)

        self.content_layout.addWidget(q_group)

        # Inizializza
        self._update_amplification()
        self._update_q()

    def _search_ingv(self):
        """Cerca parametri sismici da database INGV."""
        # TODO: Implementare ricerca reale
        comune = self.field_comune.text()
        if not comune:
            self._show_error("Inserisci il nome del comune")
            return

        # Valori esempio per Roma
        ag_values = [0.055, 0.069, 0.141, 0.177]
        f0_values = [2.45, 2.44, 2.44, 2.42]

        for i, (ag, f0) in enumerate(zip(ag_values, f0_values)):
            self.params_table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{ag:.3f}"))
            self.params_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{f0:.2f}"))

        self._show_info(f"Parametri caricati per: {comune}")

    def _update_amplification(self):
        """Aggiorna coefficienti di amplificazione."""
        # Sottosuolo
        soil_idx = self.field_soil.currentIndex()
        soil_keys = list(self.SOIL_CATEGORIES.keys())
        if 0 <= soil_idx < len(soil_keys):
            _, _, ss, cc = self.SOIL_CATEGORIES[soil_keys[soil_idx]]
            self.label_ss.setText(f"SS = {ss:.2f}, CC = {cc:.2f}")

        # Topografia
        topo_idx = self.field_topo.currentIndex()
        topo_keys = list(self.TOPO_CATEGORIES.keys())
        if 0 <= topo_idx < len(topo_keys):
            _, st = self.TOPO_CATEGORIES[topo_keys[topo_idx]]
            self.label_st.setText(f"ST = {st:.1f}")

        self._mark_dirty()

    def _update_q(self):
        """Calcola fattore di struttura q."""
        reg_plan = self.field_regularity_plan.isChecked()
        reg_elev = self.field_regularity_elev.isChecked()
        alpha = self.field_alpha_ratio.value()

        # q0 base per muratura ordinaria
        if reg_plan and reg_elev:
            q0 = 2.0 * alpha
            reg_text = "regolare"
        else:
            q0 = 1.5 * alpha
            reg_text = "non regolare"

        # KR per regolarità in elevazione
        kr = 1.0 if reg_elev else 0.8
        q = q0 * kr

        self.label_q.setText(f"q = {q:.2f} (edificio {reg_text})")
        self._mark_dirty()

    def _update_data_from_ui(self):
        soil_idx = self.field_soil.currentIndex()
        soil_keys = list(self.SOIL_CATEGORIES.keys())
        soil_cat = soil_keys[soil_idx] if 0 <= soil_idx < len(soil_keys) else "C"

        topo_idx = self.field_topo.currentIndex()
        topo_keys = list(self.TOPO_CATEGORIES.keys())
        topo_cat = topo_keys[topo_idx] if 0 <= topo_idx < len(topo_keys) else "T1"

        self._data = {
            'location': {
                'comune': self.field_comune.text(),
                'lat': self.field_lat.value(),
                'lon': self.field_lon.value(),
            },
            'soil': soil_cat,
            'topography': topo_cat,
            'regularity': {
                'plan': self.field_regularity_plan.isChecked(),
                'elevation': self.field_regularity_elev.isChecked(),
            },
            'alpha_ratio': self.field_alpha_ratio.value(),
        }
