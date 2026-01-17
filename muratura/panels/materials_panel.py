# -*- coding: utf-8 -*-
"""
Materials Panel - Fase 3: Materiali

Assegnazione materiali muratura da database NTC 2018 Tab. C8.5.I.
"""

from typing import Dict, Any, List

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide import QtWidgets, QtCore, QtGui

from .base_panel import BasePhasePanel


class MaterialsPanel(BasePhasePanel):
    """
    Pannello Fase 3: Materiali.
    Database NTC 2018 Tab. C8.5.I con coefficienti correttivi.
    """

    # Database materiali NTC 2018 Tab. C8.5.I
    # (nome, fm_min, fm_max, tau0_min, tau0_max, E_min, E_max, G_min, G_max, w)
    MATERIALS_DATABASE = {
        "Murature irregolari": [
            ("Pietrame disordinato", 1.0, 1.8, 0.020, 0.032, 690, 1050, 230, 350, 19),
            ("Pietrame a spacco", 2.0, 3.0, 0.035, 0.051, 1020, 1440, 340, 480, 20),
            ("Pietrame buona tessitura", 2.6, 3.8, 0.056, 0.074, 1500, 1980, 500, 660, 21),
            ("Pietrame listata", 2.0, 3.0, 0.040, 0.058, 1200, 1620, 400, 540, 20),
        ],
        "Murature regolari": [
            ("Mattoni pieni, malta buona", 2.4, 4.0, 0.060, 0.092, 1200, 1800, 400, 600, 18),
            ("Mattoni pieni, malta scarsa", 1.4, 2.4, 0.040, 0.060, 900, 1260, 300, 420, 17),
            ("Mattoni semipieni", 3.0, 5.0, 0.076, 0.106, 2400, 3520, 800, 1170, 15),
            ("Blocchi laterizio", 3.0, 4.4, 0.080, 0.120, 2400, 3200, 800, 1070, 12),
            ("Blocchi cls", 1.5, 3.0, 0.095, 0.142, 1200, 2400, 400, 800, 14),
        ],
        "Murature in tufo": [
            ("Tufo irregolare", 1.4, 2.4, 0.028, 0.042, 900, 1260, 300, 420, 16),
            ("Tufo a filari", 2.0, 3.2, 0.042, 0.062, 1200, 1620, 400, 540, 16),
        ],
    }

    # Coefficienti correttivi Tab. C8.5.II
    CORRECTION_FACTORS = {
        "malta_buona": ("Malta buona", 1.3, "fm, τ₀"),
        "ricorsi": ("Ricorsi/Listature", 1.2, "fm, τ₀"),
        "diatoni": ("Diatoni (elementi passanti)", 1.2, "fm, τ₀"),
        "iniezioni": ("Iniezioni di miscela", 1.5, "fm, τ₀, E, G"),
        "intonaco_1": ("Intonaco armato 1 lato", 1.3, "fm, τ₀, E"),
        "intonaco_2": ("Intonaco armato 2 lati", 1.5, "fm, τ₀, E"),
        "nucleo_scadente": ("Nucleo interno scadente", 0.8, "fm, τ₀"),
        "doppio_paramento": ("Doppio paramento scollegato", 0.8, "fm, τ₀"),
    }

    def __init__(self, parent=None):
        super().__init__(3, "Materiali", parent)
        self.set_description(
            "Assegna i materiali muratura secondo NTC 2018 Tab. C8.5.I. "
            "Seleziona i muri e applica il materiale con i coefficienti correttivi appropriati."
        )
        self.setup_content()
        self._setup_validators()

    def setup_content(self):
        """Costruisce il contenuto del pannello."""
        # Selezione materiale
        material_group, material_layout = self.create_form_group("Database Materiali NTC 2018")

        # Categoria
        self.field_category = self.create_combo_field(list(self.MATERIALS_DATABASE.keys()))
        self.field_category.currentIndexChanged.connect(self._on_category_changed)
        material_layout.addRow("Categoria:", self.field_category)

        # Tipologia
        self.field_material = QtWidgets.QComboBox()
        self.field_material.currentIndexChanged.connect(self._on_material_changed)
        material_layout.addRow("Tipologia:", self.field_material)

        # Tabella proprietà
        props_label = QtWidgets.QLabel("Proprietà meccaniche:")
        props_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        material_layout.addRow(props_label)

        self.props_table = QtWidgets.QTableWidget(5, 3)
        self.props_table.setHorizontalHeaderLabels(["Proprietà", "Min", "Max"])
        self.props_table.verticalHeader().setVisible(False)
        self.props_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.props_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.props_table.setMaximumHeight(180)

        props = ["fm [MPa]", "τ₀ [MPa]", "E [MPa]", "G [MPa]", "w [kN/m³]"]
        for i, prop in enumerate(props):
            self.props_table.setItem(i, 0, QtWidgets.QTableWidgetItem(prop))

        material_layout.addRow(self.props_table)

        self.content_layout.addWidget(material_group)

        # Coefficienti correttivi
        corr_group, corr_layout = self.create_form_group("Coefficienti Correttivi (Tab. C8.5.II)")

        self.correction_checks = {}
        for key, (label, factor, applies_to) in self.CORRECTION_FACTORS.items():
            cb = self.create_check_field(f"{label} (×{factor})")
            cb.setToolTip(f"Si applica a: {applies_to}")
            cb.stateChanged.connect(self._on_correction_changed)
            corr_layout.addRow(cb)
            self.correction_checks[key] = cb

        # Fattore totale calcolato
        self.label_total_factor = QtWidgets.QLabel("Fattore totale: 1.00")
        self.label_total_factor.setStyleSheet("""
            font-weight: bold;
            padding: 8px;
            background-color: #e8f5e9;
            border-radius: 4px;
            margin-top: 5px;
        """)
        corr_layout.addRow(self.label_total_factor)

        self.label_limit_warning = QtWidgets.QLabel()
        self.label_limit_warning.setStyleSheet("color: #ff9800; font-size: 11px;")
        self.label_limit_warning.setWordWrap(True)
        corr_layout.addRow(self.label_limit_warning)

        self.content_layout.addWidget(corr_group)

        # Proprietà finali
        final_group, final_layout = self.create_form_group("Proprietà Finali")

        self.final_table = QtWidgets.QTableWidget(5, 2)
        self.final_table.setHorizontalHeaderLabels(["Proprietà", "Valore"])
        self.final_table.verticalHeader().setVisible(False)
        self.final_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.final_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.final_table.setMaximumHeight(180)

        props_final = ["fm [MPa]", "τ₀ [MPa]", "E [MPa]", "G [MPa]", "w [kN/m³]"]
        for i, prop in enumerate(props_final):
            self.final_table.setItem(i, 0, QtWidgets.QTableWidgetItem(prop))

        final_layout.addRow(self.final_table)

        self.content_layout.addWidget(final_group)

        # Assegnazione
        assign_group, assign_layout = self.create_form_group("Assegnazione")

        self.field_wall_selection = self.create_combo_field(["Tutti i muri", "Selezione attuale", "Per piano..."])
        assign_layout.addRow("Applica a:", self.field_wall_selection)

        btn_assign = QtWidgets.QPushButton("Assegna Materiale")
        btn_assign.clicked.connect(self._on_assign_material)
        btn_assign.setStyleSheet("""
            QPushButton { background-color: #28a745; color: white; padding: 10px; font-weight: bold; }
            QPushButton:hover { background-color: #218838; }
        """)
        assign_layout.addRow("", btn_assign)

        self.content_layout.addWidget(assign_group)

        # Inizializza
        self._on_category_changed()

    def _on_category_changed(self):
        """Gestisce cambio categoria materiale."""
        category = self.field_category.currentText()
        materials = self.MATERIALS_DATABASE.get(category, [])

        self.field_material.clear()
        for mat in materials:
            self.field_material.addItem(mat[0])

        if materials:
            self._on_material_changed()

    def _on_material_changed(self):
        """Gestisce cambio materiale."""
        category = self.field_category.currentText()
        material_name = self.field_material.currentText()

        materials = self.MATERIALS_DATABASE.get(category, [])
        for mat in materials:
            if mat[0] == material_name:
                # Aggiorna tabella proprietà
                # (nome, fm_min, fm_max, tau0_min, tau0_max, E_min, E_max, G_min, G_max, w)
                self.props_table.setItem(0, 1, QtWidgets.QTableWidgetItem(f"{mat[1]:.2f}"))
                self.props_table.setItem(0, 2, QtWidgets.QTableWidgetItem(f"{mat[2]:.2f}"))
                self.props_table.setItem(1, 1, QtWidgets.QTableWidgetItem(f"{mat[3]:.3f}"))
                self.props_table.setItem(1, 2, QtWidgets.QTableWidgetItem(f"{mat[4]:.3f}"))
                self.props_table.setItem(2, 1, QtWidgets.QTableWidgetItem(f"{mat[5]:.0f}"))
                self.props_table.setItem(2, 2, QtWidgets.QTableWidgetItem(f"{mat[6]:.0f}"))
                self.props_table.setItem(3, 1, QtWidgets.QTableWidgetItem(f"{mat[7]:.0f}"))
                self.props_table.setItem(3, 2, QtWidgets.QTableWidgetItem(f"{mat[8]:.0f}"))
                self.props_table.setItem(4, 1, QtWidgets.QTableWidgetItem(f"{mat[9]:.1f}"))
                self.props_table.setItem(4, 2, QtWidgets.QTableWidgetItem(f"{mat[9]:.1f}"))

                self._current_material = mat
                break

        self._update_final_properties()
        self._mark_dirty()

    def _on_correction_changed(self):
        """Gestisce cambio coefficienti correttivi."""
        self._update_final_properties()
        self._mark_dirty()

    def _get_total_factor(self) -> float:
        """Calcola il fattore correttivo totale."""
        total = 1.0
        for key, cb in self.correction_checks.items():
            if cb.isChecked():
                _, factor, _ = self.CORRECTION_FACTORS[key]
                total *= factor
        return total

    def _update_final_properties(self):
        """Aggiorna le proprietà finali con i coefficienti."""
        if not hasattr(self, '_current_material'):
            return

        mat = self._current_material
        factor = self._get_total_factor()

        # Limita a 1.5 come da normativa
        limited_factor = min(factor, 1.5)

        # Usa valori medi
        fm = (mat[1] + mat[2]) / 2 * limited_factor
        tau0 = (mat[3] + mat[4]) / 2 * limited_factor
        E = (mat[5] + mat[6]) / 2 * limited_factor
        G = (mat[7] + mat[8]) / 2 * limited_factor
        w = mat[9]

        self.final_table.setItem(0, 1, QtWidgets.QTableWidgetItem(f"{fm:.2f}"))
        self.final_table.setItem(1, 1, QtWidgets.QTableWidgetItem(f"{tau0:.3f}"))
        self.final_table.setItem(2, 1, QtWidgets.QTableWidgetItem(f"{E:.0f}"))
        self.final_table.setItem(3, 1, QtWidgets.QTableWidgetItem(f"{G:.0f}"))
        self.final_table.setItem(4, 1, QtWidgets.QTableWidgetItem(f"{w:.1f}"))

        # Aggiorna label fattore
        self.label_total_factor.setText(f"Fattore totale: {factor:.2f}")

        if factor > 1.5:
            self.label_total_factor.setStyleSheet("""
                font-weight: bold; padding: 8px; background-color: #fff3e0;
                border-radius: 4px; margin-top: 5px;
            """)
            self.label_limit_warning.setText(
                f"⚠ Il fattore calcolato ({factor:.2f}) supera il limite 1.5. "
                f"Viene applicato il valore massimo 1.5."
            )
        else:
            self.label_total_factor.setStyleSheet("""
                font-weight: bold; padding: 8px; background-color: #e8f5e9;
                border-radius: 4px; margin-top: 5px;
            """)
            self.label_limit_warning.setText("")

    def _on_assign_material(self):
        """Assegna il materiale ai muri selezionati."""
        if not hasattr(self, '_current_material'):
            self._show_error("Seleziona prima un materiale")
            return

        try:
            from muratura.bim import set_material_properties
            import FreeCAD

            if not FreeCAD.ActiveDocument:
                self._show_error("Nessun documento attivo")
                return

            mat = self._current_material
            factor = min(self._get_total_factor(), 1.5)

            # Raccogli correzioni attive
            corrections = {}
            for key, cb in self.correction_checks.items():
                if cb.isChecked():
                    corrections[key] = True

            # Trova muri
            selection = self.field_wall_selection.currentText()
            walls = []

            if selection == "Tutti i muri":
                for obj in FreeCAD.ActiveDocument.Objects:
                    if hasattr(obj, 'IfcType') and obj.IfcType == "Wall":
                        walls.append(obj)
                    elif obj.TypeId == "Part::FeaturePython" and "Wall" in obj.Name:
                        walls.append(obj)
            elif selection == "Selezione attuale":
                import FreeCADGui
                walls = [s.Object for s in FreeCADGui.Selection.getSelection()
                         if hasattr(s, 'Object')]

            # Applica materiale
            for wall in walls:
                set_material_properties(
                    wall,
                    mat[0],  # material_type
                    "buona",  # mortar_quality
                    corrections
                )

            self._show_success(f"Materiale assegnato a {len(walls)} muri")

        except Exception as e:
            self._show_error(f"Errore assegnazione: {e}")

    def _setup_validators(self):
        """Configura i validatori."""
        def validate_material():
            if not hasattr(self, '_current_material'):
                return False, "Seleziona un materiale"
            return True, ""

        self.add_validator(validate_material)

    def _update_data_from_ui(self):
        """Aggiorna i dati dall'interfaccia."""
        corrections = {k: cb.isChecked() for k, cb in self.correction_checks.items()}

        self._data = {
            'category': self.field_category.currentText(),
            'material': self.field_material.currentText(),
            'corrections': corrections,
            'factor': self._get_total_factor(),
        }

    def _update_ui_from_data(self):
        """Aggiorna l'interfaccia dai dati."""
        if not self._data:
            return

        # Categoria
        category = self._data.get('category', '')
        idx = self.field_category.findText(category)
        if idx >= 0:
            self.field_category.setCurrentIndex(idx)

        # Materiale
        material = self._data.get('material', '')
        idx = self.field_material.findText(material)
        if idx >= 0:
            self.field_material.setCurrentIndex(idx)

        # Correzioni
        corrections = self._data.get('corrections', {})
        for key, checked in corrections.items():
            if key in self.correction_checks:
                self.correction_checks[key].setChecked(checked)
