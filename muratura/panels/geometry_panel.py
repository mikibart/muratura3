# -*- coding: utf-8 -*-
"""
Geometry Panel - Fase 2: Geometria

Import DXF/IFC, definizione piani, disegno muri e aperture con Arch.
"""

from typing import Dict, Any, List

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide import QtWidgets, QtCore, QtGui

from .base_panel import BasePhasePanel


class GeometryPanel(BasePhasePanel):
    """
    Pannello Fase 2: Geometria.
    Import DXF/IFC, gestione piani, muri e aperture.
    """

    def __init__(self, parent=None):
        super().__init__(2, "Geometria", parent)
        self.set_description(
            "Definisci la geometria dell'edificio: importa da DXF/IFC oppure "
            "disegna muri e aperture direttamente. Gestisci i piani dell'edificio."
        )
        self.setup_content()
        self._setup_validators()

    def setup_content(self):
        """Costruisce il contenuto del pannello."""
        # Tab widget per organizzare le sottosezioni
        tabs = QtWidgets.QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #ccc; }
            QTabBar::tab { padding: 8px 15px; }
            QTabBar::tab:selected { background-color: #fff; font-weight: bold; }
        """)

        # Tab 1: Piani
        levels_tab = self._create_levels_tab()
        tabs.addTab(levels_tab, "Piani")

        # Tab 2: Import
        import_tab = self._create_import_tab()
        tabs.addTab(import_tab, "Import")

        # Tab 3: Muri
        walls_tab = self._create_walls_tab()
        tabs.addTab(walls_tab, "Muri")

        # Tab 4: Aperture
        openings_tab = self._create_openings_tab()
        tabs.addTab(openings_tab, "Aperture")

        self.content_layout.addWidget(tabs)

    def _create_levels_tab(self) -> QtWidgets.QWidget:
        """Crea tab definizione piani."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Configurazione rapida
        quick_group, quick_layout = self.create_form_group("Configurazione Rapida")

        self.field_num_floors = self.create_int_field(1, 10)
        self.field_num_floors.setValue(2)
        quick_layout.addRow("Numero piani:", self.field_num_floors)

        self.field_floor_height = self.create_number_field(2.0, 6.0, 2, "m")
        self.field_floor_height.setValue(3.00)
        quick_layout.addRow("Altezza interpiano:", self.field_floor_height)

        self.field_ground_level = self.create_number_field(-5.0, 10.0, 2, "m")
        self.field_ground_level.setValue(0.00)
        quick_layout.addRow("Quota piano terra:", self.field_ground_level)

        self.field_has_basement = self.create_check_field("Includi piano interrato")
        quick_layout.addRow("", self.field_has_basement)

        btn_generate = QtWidgets.QPushButton("Genera Piani")
        btn_generate.clicked.connect(self._on_generate_levels)
        btn_generate.setStyleSheet("""
            QPushButton { background-color: #17a2b8; color: white; padding: 8px; }
            QPushButton:hover { background-color: #138496; }
        """)
        quick_layout.addRow("", btn_generate)

        layout.addWidget(quick_group)

        # Tabella piani
        table_group, table_layout = self.create_form_group("Definizione Piani")

        self.levels_table = QtWidgets.QTableWidget(0, 4)
        self.levels_table.setHorizontalHeaderLabels(["Piano", "Nome", "Quota [m]", "Altezza [m]"])
        self.levels_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.levels_table.setMinimumHeight(200)

        table_layout.addRow(self.levels_table)

        # Pulsanti tabella
        btn_row = QtWidgets.QWidget()
        btn_layout = QtWidgets.QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)

        btn_add = QtWidgets.QPushButton("+ Aggiungi")
        btn_add.clicked.connect(self._add_level_row)
        btn_layout.addWidget(btn_add)

        btn_remove = QtWidgets.QPushButton("- Rimuovi")
        btn_remove.clicked.connect(self._remove_level_row)
        btn_layout.addWidget(btn_remove)

        btn_layout.addStretch()
        table_layout.addRow(btn_row)

        layout.addWidget(table_group)
        layout.addStretch()

        return tab

    def _create_import_tab(self) -> QtWidgets.QWidget:
        """Crea tab import DXF/IFC."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Import DXF
        dxf_group, dxf_layout = self.create_form_group("Import DXF")

        dxf_file_row = QtWidgets.QWidget()
        dxf_file_layout = QtWidgets.QHBoxLayout(dxf_file_row)
        dxf_file_layout.setContentsMargins(0, 0, 0, 0)

        self.field_dxf_file = self.create_input_field("Seleziona file DXF...")
        self.field_dxf_file.setReadOnly(True)
        dxf_file_layout.addWidget(self.field_dxf_file)

        btn_browse_dxf = QtWidgets.QPushButton("Sfoglia...")
        btn_browse_dxf.clicked.connect(lambda: self._browse_file("dxf"))
        dxf_file_layout.addWidget(btn_browse_dxf)

        dxf_layout.addRow("File DXF:", dxf_file_row)

        self.field_dxf_layer = self.create_input_field("MURI")
        dxf_layout.addRow("Layer muri:", self.field_dxf_layer)

        self.field_dxf_scale = self.create_combo_field(["1:1", "1:50", "1:100", "1:200"])
        dxf_layout.addRow("Scala:", self.field_dxf_scale)

        self.field_dxf_units = self.create_combo_field(["mm", "cm", "m"])
        self.field_dxf_units.setCurrentIndex(2)
        dxf_layout.addRow("Unità:", self.field_dxf_units)

        btn_import_dxf = QtWidgets.QPushButton("Importa DXF")
        btn_import_dxf.clicked.connect(self._on_import_dxf)
        btn_import_dxf.setStyleSheet("""
            QPushButton { background-color: #28a745; color: white; padding: 8px; }
            QPushButton:hover { background-color: #218838; }
        """)
        dxf_layout.addRow("", btn_import_dxf)

        layout.addWidget(dxf_group)

        # Import IFC
        ifc_group, ifc_layout = self.create_form_group("Import IFC (BIM)")

        ifc_file_row = QtWidgets.QWidget()
        ifc_file_layout = QtWidgets.QHBoxLayout(ifc_file_row)
        ifc_file_layout.setContentsMargins(0, 0, 0, 0)

        self.field_ifc_file = self.create_input_field("Seleziona file IFC...")
        self.field_ifc_file.setReadOnly(True)
        ifc_file_layout.addWidget(self.field_ifc_file)

        btn_browse_ifc = QtWidgets.QPushButton("Sfoglia...")
        btn_browse_ifc.clicked.connect(lambda: self._browse_file("ifc"))
        ifc_file_layout.addWidget(btn_browse_ifc)

        ifc_layout.addRow("File IFC:", ifc_file_row)

        self.field_ifc_import_walls = self.create_check_field("Importa muri (IfcWall)")
        self.field_ifc_import_walls.setChecked(True)
        ifc_layout.addRow("", self.field_ifc_import_walls)

        self.field_ifc_import_windows = self.create_check_field("Importa aperture (IfcWindow, IfcDoor)")
        self.field_ifc_import_windows.setChecked(True)
        ifc_layout.addRow("", self.field_ifc_import_windows)

        self.field_ifc_import_slabs = self.create_check_field("Importa solai (IfcSlab)")
        self.field_ifc_import_slabs.setChecked(True)
        ifc_layout.addRow("", self.field_ifc_import_slabs)

        btn_import_ifc = QtWidgets.QPushButton("Importa IFC")
        btn_import_ifc.clicked.connect(self._on_import_ifc)
        btn_import_ifc.setStyleSheet("""
            QPushButton { background-color: #28a745; color: white; padding: 8px; }
            QPushButton:hover { background-color: #218838; }
        """)
        ifc_layout.addRow("", btn_import_ifc)

        layout.addWidget(ifc_group)
        layout.addStretch()

        return tab

    def _create_walls_tab(self) -> QtWidgets.QWidget:
        """Crea tab disegno muri."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Strumento muro
        wall_group, wall_layout = self.create_form_group("Nuovo Muro")

        # Coordinate
        coord_row = QtWidgets.QWidget()
        coord_layout = QtWidgets.QGridLayout(coord_row)
        coord_layout.setContentsMargins(0, 0, 0, 0)

        coord_layout.addWidget(QtWidgets.QLabel("X1:"), 0, 0)
        self.field_wall_x1 = self.create_number_field(-100, 100, 2, "m")
        coord_layout.addWidget(self.field_wall_x1, 0, 1)

        coord_layout.addWidget(QtWidgets.QLabel("Y1:"), 0, 2)
        self.field_wall_y1 = self.create_number_field(-100, 100, 2, "m")
        coord_layout.addWidget(self.field_wall_y1, 0, 3)

        coord_layout.addWidget(QtWidgets.QLabel("X2:"), 1, 0)
        self.field_wall_x2 = self.create_number_field(-100, 100, 2, "m")
        self.field_wall_x2.setValue(5.0)
        coord_layout.addWidget(self.field_wall_x2, 1, 1)

        coord_layout.addWidget(QtWidgets.QLabel("Y2:"), 1, 2)
        self.field_wall_y2 = self.create_number_field(-100, 100, 2, "m")
        coord_layout.addWidget(self.field_wall_y2, 1, 3)

        wall_layout.addRow("Coordinate:", coord_row)

        self.field_wall_thickness = self.create_number_field(0.10, 1.50, 2, "m")
        self.field_wall_thickness.setValue(0.40)
        wall_layout.addRow("Spessore:", self.field_wall_thickness)

        self.field_wall_height = self.create_number_field(0.50, 10.0, 2, "m")
        self.field_wall_height.setValue(3.00)
        wall_layout.addRow("Altezza:", self.field_wall_height)

        self.field_wall_floor = self.create_int_field(0, 10)
        wall_layout.addRow("Piano:", self.field_wall_floor)

        self.field_wall_baseline = self.create_combo_field(["Centro", "Sinistra", "Destra"])
        wall_layout.addRow("Baseline:", self.field_wall_baseline)

        btn_create_wall = QtWidgets.QPushButton("Crea Muro")
        btn_create_wall.clicked.connect(self._on_create_wall)
        btn_create_wall.setStyleSheet("""
            QPushButton { background-color: #007bff; color: white; padding: 10px; }
            QPushButton:hover { background-color: #0056b3; }
        """)
        wall_layout.addRow("", btn_create_wall)

        layout.addWidget(wall_group)

        # Rettangolo rapido
        rect_group, rect_layout = self.create_form_group("Rettangolo Rapido")

        rect_coord = QtWidgets.QWidget()
        rect_coord_layout = QtWidgets.QGridLayout(rect_coord)
        rect_coord_layout.setContentsMargins(0, 0, 0, 0)

        rect_coord_layout.addWidget(QtWidgets.QLabel("X:"), 0, 0)
        self.field_rect_x = self.create_number_field(-100, 100, 2, "m")
        rect_coord_layout.addWidget(self.field_rect_x, 0, 1)

        rect_coord_layout.addWidget(QtWidgets.QLabel("Y:"), 0, 2)
        self.field_rect_y = self.create_number_field(-100, 100, 2, "m")
        rect_coord_layout.addWidget(self.field_rect_y, 0, 3)

        rect_coord_layout.addWidget(QtWidgets.QLabel("L:"), 1, 0)
        self.field_rect_length = self.create_number_field(1, 100, 2, "m")
        self.field_rect_length.setValue(10.0)
        rect_coord_layout.addWidget(self.field_rect_length, 1, 1)

        rect_coord_layout.addWidget(QtWidgets.QLabel("W:"), 1, 2)
        self.field_rect_width = self.create_number_field(1, 100, 2, "m")
        self.field_rect_width.setValue(8.0)
        rect_coord_layout.addWidget(self.field_rect_width, 1, 3)

        rect_layout.addRow("Posizione e dimensioni:", rect_coord)

        btn_create_rect = QtWidgets.QPushButton("Crea Rettangolo")
        btn_create_rect.clicked.connect(self._on_create_rectangle)
        btn_create_rect.setStyleSheet("""
            QPushButton { background-color: #007bff; color: white; padding: 10px; }
            QPushButton:hover { background-color: #0056b3; }
        """)
        rect_layout.addRow("", btn_create_rect)

        layout.addWidget(rect_group)
        layout.addStretch()

        return tab

    def _create_openings_tab(self) -> QtWidgets.QWidget:
        """Crea tab aperture."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # Tipologie predefinite
        preset_group, preset_layout = self.create_form_group("Aperture Predefinite")

        presets = [
            ("Finestra piccola", "0.80 × 1.20 m, davanzale 0.90 m"),
            ("Finestra media", "1.20 × 1.40 m, davanzale 0.90 m"),
            ("Finestra grande", "1.60 × 1.60 m, davanzale 0.80 m"),
            ("Portafinestra", "1.20 × 2.20 m"),
            ("Porta interna", "0.90 × 2.10 m"),
            ("Porta esterna", "1.20 × 2.20 m"),
        ]

        self.field_opening_preset = self.create_combo_field([p[0] for p in presets])
        self.field_opening_preset.setToolTip("\n".join([f"{p[0]}: {p[1]}" for p in presets]))
        preset_layout.addRow("Tipologia:", self.field_opening_preset)

        layout.addWidget(preset_group)

        # Apertura personalizzata
        custom_group, custom_layout = self.create_form_group("Apertura Personalizzata")

        self.field_opening_wall = self.create_input_field("Nome muro (es. Wall001)")
        custom_layout.addRow("Muro:", self.field_opening_wall)

        self.field_opening_x = self.create_number_field(0, 50, 2, "m")
        self.field_opening_x.setValue(2.0)
        custom_layout.addRow("Posizione X:", self.field_opening_x)

        self.field_opening_width = self.create_number_field(0.40, 5.0, 2, "m")
        self.field_opening_width.setValue(1.20)
        custom_layout.addRow("Larghezza:", self.field_opening_width)

        self.field_opening_height = self.create_number_field(0.40, 4.0, 2, "m")
        self.field_opening_height.setValue(1.40)
        custom_layout.addRow("Altezza:", self.field_opening_height)

        self.field_opening_sill = self.create_number_field(0, 2.0, 2, "m")
        self.field_opening_sill.setValue(0.90)
        custom_layout.addRow("Davanzale/Soglia:", self.field_opening_sill)

        self.field_opening_type = self.create_combo_field(["Finestra", "Porta"])
        custom_layout.addRow("Tipo:", self.field_opening_type)

        btn_create_opening = QtWidgets.QPushButton("Crea Apertura")
        btn_create_opening.clicked.connect(self._on_create_opening)
        btn_create_opening.setStyleSheet("""
            QPushButton { background-color: #17a2b8; color: white; padding: 10px; }
            QPushButton:hover { background-color: #138496; }
        """)
        custom_layout.addRow("", btn_create_opening)

        layout.addWidget(custom_group)
        layout.addStretch()

        return tab

    # Handlers

    def _on_generate_levels(self):
        """Genera piani automaticamente."""
        num_floors = self.field_num_floors.value()
        height = self.field_floor_height.value()
        ground = self.field_ground_level.value()
        has_basement = self.field_has_basement.isChecked()

        # Pulisci tabella
        self.levels_table.setRowCount(0)

        # Piano interrato
        start_floor = -1 if has_basement else 0
        current_level = ground - height if has_basement else ground

        for i in range(start_floor, num_floors):
            row = self.levels_table.rowCount()
            self.levels_table.insertRow(row)

            # Piano
            floor_item = QtWidgets.QTableWidgetItem(str(i))
            floor_item.setFlags(floor_item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.levels_table.setItem(row, 0, floor_item)

            # Nome
            if i < 0:
                name = "Interrato"
            elif i == 0:
                name = "Terra"
            else:
                names = ["Primo", "Secondo", "Terzo", "Quarto", "Quinto",
                         "Sesto", "Settimo", "Ottavo", "Nono", "Decimo"]
                name = names[i-1] if i <= 10 else f"Piano {i}"
            self.levels_table.setItem(row, 1, QtWidgets.QTableWidgetItem(name))

            # Quota
            quota = current_level if i < 0 else ground + (i * height if i > 0 else 0)
            self.levels_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{quota:.2f}"))

            # Altezza
            self.levels_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{height:.2f}"))

            current_level = quota

        self._mark_dirty()

    def _add_level_row(self):
        """Aggiunge riga alla tabella piani."""
        row = self.levels_table.rowCount()
        self.levels_table.insertRow(row)
        self._mark_dirty()

    def _remove_level_row(self):
        """Rimuove riga selezionata dalla tabella piani."""
        row = self.levels_table.currentRow()
        if row >= 0:
            self.levels_table.removeRow(row)
            self._mark_dirty()

    def _browse_file(self, file_type: str):
        """Apre dialogo selezione file."""
        if file_type == "dxf":
            filter_str = "File DXF (*.dxf)"
            field = self.field_dxf_file
        else:
            filter_str = "File IFC (*.ifc)"
            field = self.field_ifc_file

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Seleziona file {file_type.upper()}", "", filter_str
        )
        if file_path:
            field.setText(file_path)
            self._mark_dirty()

    def _on_import_dxf(self):
        """Importa file DXF."""
        file_path = self.field_dxf_file.text()
        if not file_path:
            self._show_error("Seleziona un file DXF da importare")
            return

        try:
            from muratura.bim import import_dxf
            layer = self.field_dxf_layer.text() or "MURI"
            # TODO: applicare scala e unità
            walls = import_dxf(file_path, layer)
            self._show_success(f"Importati {len(walls)} muri da DXF")
        except Exception as e:
            self._show_error(f"Errore import DXF: {e}")

    def _on_import_ifc(self):
        """Importa file IFC."""
        file_path = self.field_ifc_file.text()
        if not file_path:
            self._show_error("Seleziona un file IFC da importare")
            return

        try:
            from muratura.bim import import_ifc
            objs = import_ifc(file_path)
            self._show_success(f"Importati {len(objs)} oggetti da IFC")
        except Exception as e:
            self._show_error(f"Errore import IFC: {e}")

    def _on_create_wall(self):
        """Crea un nuovo muro."""
        try:
            from muratura.bim import create_wall

            x1 = self.field_wall_x1.value()
            y1 = self.field_wall_y1.value()
            x2 = self.field_wall_x2.value()
            y2 = self.field_wall_y2.value()
            thickness = self.field_wall_thickness.value()
            height = self.field_wall_height.value()
            floor = self.field_wall_floor.value()

            baseline_map = {"Centro": "Center", "Sinistra": "Left", "Destra": "Right"}
            baseline = baseline_map.get(self.field_wall_baseline.currentText(), "Center")

            wall = create_wall(
                start=(x1, y1),
                end=(x2, y2),
                thickness=thickness,
                height=height,
                floor=floor,
                baseline=baseline
            )

            if wall:
                self._show_success(f"Muro creato: {wall.Name}")
                self._mark_dirty()
            else:
                self._show_error("Impossibile creare il muro")
        except Exception as e:
            self._show_error(f"Errore creazione muro: {e}")

    def _on_create_rectangle(self):
        """Crea un rettangolo di muri."""
        try:
            x = self.field_rect_x.value()
            y = self.field_rect_y.value()
            length = self.field_rect_length.value()
            width = self.field_rect_width.value()
            thickness = self.field_wall_thickness.value()
            height = self.field_wall_height.value()
            floor = self.field_wall_floor.value()

            from muratura.bim import create_wall

            # Crea 4 muri
            walls = []
            points = [
                ((x, y), (x + length, y)),           # Sud
                ((x + length, y), (x + length, y + width)),  # Est
                ((x + length, y + width), (x, y + width)),   # Nord
                ((x, y + width), (x, y)),            # Ovest
            ]

            for start, end in points:
                wall = create_wall(start, end, thickness, height, floor)
                if wall:
                    walls.append(wall)

            if walls:
                self._show_success(f"Creati {len(walls)} muri")
                self._mark_dirty()
            else:
                self._show_error("Impossibile creare i muri")
        except Exception as e:
            self._show_error(f"Errore creazione rettangolo: {e}")

    def _on_create_opening(self):
        """Crea un'apertura nel muro."""
        try:
            from muratura.bim import create_window, create_door

            wall_name = self.field_opening_wall.text()
            if not wall_name:
                self._show_error("Specifica il nome del muro")
                return

            width = self.field_opening_width.value()
            height = self.field_opening_height.value()
            sill = self.field_opening_sill.value()
            opening_type = self.field_opening_type.currentText()

            # TODO: trovare il muro per nome e inserire l'apertura
            # Per ora mostriamo solo un messaggio
            self._show_info(
                f"Apertura {opening_type} {width:.2f}×{height:.2f}m\n"
                f"Davanzale: {sill:.2f}m\n"
                f"Su muro: {wall_name}"
            )
        except Exception as e:
            self._show_error(f"Errore creazione apertura: {e}")

    def _setup_validators(self):
        """Configura i validatori."""
        def validate_levels():
            if self.levels_table.rowCount() < 1:
                return False, "Definisci almeno un piano"
            return True, ""

        self.add_validator(validate_levels)

    def _update_data_from_ui(self):
        """Aggiorna i dati dall'interfaccia."""
        # Piani
        levels = []
        for row in range(self.levels_table.rowCount()):
            level = {}
            for col in range(self.levels_table.columnCount()):
                item = self.levels_table.item(row, col)
                if item:
                    headers = ["floor", "name", "level", "height"]
                    level[headers[col]] = item.text()
            levels.append(level)

        self._data = {
            'levels': levels,
            'dxf_file': self.field_dxf_file.text(),
            'ifc_file': self.field_ifc_file.text(),
        }

    def _update_ui_from_data(self):
        """Aggiorna l'interfaccia dai dati."""
        if not self._data:
            return

        # Piani
        levels = self._data.get('levels', [])
        self.levels_table.setRowCount(len(levels))
        for row, level in enumerate(levels):
            self.levels_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(level.get('floor', ''))))
            self.levels_table.setItem(row, 1, QtWidgets.QTableWidgetItem(level.get('name', '')))
            self.levels_table.setItem(row, 2, QtWidgets.QTableWidgetItem(level.get('level', '')))
            self.levels_table.setItem(row, 3, QtWidgets.QTableWidgetItem(level.get('height', '')))

        # File
        self.field_dxf_file.setText(self._data.get('dxf_file', ''))
        self.field_ifc_file.setText(self._data.get('ifc_file', ''))
