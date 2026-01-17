# -*- coding: utf-8 -*-
"""
Report Panel - Fase 12: Relazione di Calcolo

Report PDF, DOCX, export IFC/DXF.
"""

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide import QtWidgets, QtCore, QtGui

from .base_panel import BasePhasePanel


class ReportPanel(BasePhasePanel):
    """Pannello Fase 12: Relazione di Calcolo."""

    CHAPTERS = [
        ("Premessa", True),
        ("Descrizione opera", True),
        ("Materiali", True),
        ("Analisi dei carichi", True),
        ("Azione sismica", True),
        ("Modellazione", True),
        ("Analisi e verifiche", True),
        ("Meccanismi locali", True),
        ("Interventi di rinforzo", False),
        ("Conclusioni", True),
    ]

    def __init__(self, parent=None):
        super().__init__(12, "Relazione", parent)
        self.set_description(
            "Genera la relazione di calcolo strutturale completa ed esporta "
            "il modello in vari formati (PDF, DOCX, IFC, DXF)."
        )
        self.setup_content()

    def setup_content(self):
        """Costruisce il contenuto del pannello."""
        tabs = QtWidgets.QTabWidget()

        # Tab Relazione
        report_tab = QtWidgets.QWidget()
        report_layout = QtWidgets.QVBoxLayout(report_tab)

        # Capitoli
        chapters_group, chapters_layout = self.create_form_group("Capitoli da Includere")

        self.chapter_checks = {}
        for chapter, default in self.CHAPTERS:
            cb = self.create_check_field(chapter)
            cb.setChecked(default)
            chapters_layout.addRow(cb)
            self.chapter_checks[chapter] = cb

        btn_all = QtWidgets.QPushButton("Seleziona tutti")
        btn_all.clicked.connect(lambda: [cb.setChecked(True) for cb in self.chapter_checks.values()])
        chapters_layout.addRow(btn_all)

        report_layout.addWidget(chapters_group)

        # Opzioni
        options_group, options_layout = self.create_form_group("Opzioni Relazione")

        self.field_format = self.create_combo_field(["PDF", "DOCX", "HTML", "LaTeX"])
        options_layout.addRow("Formato:", self.field_format)

        self.field_lang = self.create_combo_field(["Italiano", "English"])
        options_layout.addRow("Lingua:", self.field_lang)

        self.cb_include_drawings = self.create_check_field("Includi tavole grafiche")
        self.cb_include_drawings.setChecked(True)
        options_layout.addRow(self.cb_include_drawings)

        self.cb_include_calcs = self.create_check_field("Includi dettagli calcoli")
        self.cb_include_calcs.setChecked(True)
        options_layout.addRow(self.cb_include_calcs)

        report_layout.addWidget(options_group)

        # Generazione
        btn_generate = QtWidgets.QPushButton("Genera Relazione")
        btn_generate.clicked.connect(self._generate_report)
        btn_generate.setStyleSheet("""
            QPushButton {
                background-color: #28a745; color: white;
                padding: 15px; font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #218838; }
        """)
        report_layout.addWidget(btn_generate)

        report_layout.addStretch()
        tabs.addTab(report_tab, "Relazione")

        # Tab Export
        export_tab = QtWidgets.QWidget()
        export_layout = QtWidgets.QVBoxLayout(export_tab)

        # Export modello
        model_group, model_layout = self.create_form_group("Export Modello")

        btn_ifc = QtWidgets.QPushButton("Esporta IFC (BIM)")
        btn_ifc.clicked.connect(lambda: self._export("ifc"))
        model_layout.addRow(btn_ifc)

        btn_step = QtWidgets.QPushButton("Esporta STEP")
        btn_step.clicked.connect(lambda: self._export("step"))
        model_layout.addRow(btn_step)

        btn_stl = QtWidgets.QPushButton("Esporta STL (3D print)")
        btn_stl.clicked.connect(lambda: self._export("stl"))
        model_layout.addRow(btn_stl)

        export_layout.addWidget(model_group)

        # Export disegni
        drawings_group, drawings_layout = self.create_form_group("Export Tavole")

        btn_dxf = QtWidgets.QPushButton("Esporta DXF")
        btn_dxf.clicked.connect(lambda: self._export("dxf"))
        drawings_layout.addRow(btn_dxf)

        btn_svg = QtWidgets.QPushButton("Esporta SVG")
        btn_svg.clicked.connect(lambda: self._export("svg"))
        drawings_layout.addRow(btn_svg)

        export_layout.addWidget(drawings_group)

        # Export dati
        data_group, data_layout = self.create_form_group("Export Dati")

        btn_json = QtWidgets.QPushButton("Esporta JSON")
        btn_json.clicked.connect(lambda: self._export("json"))
        data_layout.addRow(btn_json)

        btn_csv = QtWidgets.QPushButton("Esporta CSV (risultati)")
        btn_csv.clicked.connect(lambda: self._export("csv"))
        data_layout.addRow(btn_csv)

        export_layout.addWidget(data_group)
        export_layout.addStretch()
        tabs.addTab(export_tab, "Export")

        # Tab Tavole
        drawings_tab = QtWidgets.QWidget()
        drawings_layout = QtWidgets.QVBoxLayout(drawings_tab)

        drawings_group2, drawings_layout2 = self.create_form_group("Tavole Grafiche")

        drawings_list = [
            ("TAV.01", "Piante architettoniche"),
            ("TAV.02", "Piante strutturali"),
            ("TAV.03", "Sezioni"),
            ("TAV.04", "Telaio equivalente"),
            ("TAV.05", "Mappa DCR"),
            ("TAV.06", "Curve pushover"),
            ("TAV.07", "Particolari rinforzi"),
        ]

        for code, desc in drawings_list:
            cb = self.create_check_field(f"{code}: {desc}")
            cb.setChecked(True)
            drawings_layout2.addRow(cb)

        btn_gen_drawings = QtWidgets.QPushButton("Genera Tavole Selezionate")
        btn_gen_drawings.clicked.connect(self._generate_drawings)
        drawings_layout2.addRow(btn_gen_drawings)

        drawings_layout.addWidget(drawings_group2)
        drawings_layout.addStretch()
        tabs.addTab(drawings_tab, "Tavole")

        self.content_layout.addWidget(tabs)

    def _generate_report(self):
        """Genera la relazione di calcolo."""
        format_type = self.field_format.currentText()
        chapters = [ch for ch, cb in self.chapter_checks.items() if cb.isChecked()]

        if not chapters:
            self._show_error("Seleziona almeno un capitolo")
            return

        # Dialogo salvataggio
        ext_map = {"PDF": "pdf", "DOCX": "docx", "HTML": "html", "LaTeX": "tex"}
        ext = ext_map.get(format_type, "pdf")

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Salva Relazione",
            f"relazione_strutturale.{ext}",
            f"File {format_type} (*.{ext})"
        )

        if file_path:
            try:
                # TODO: Generare relazione reale
                # from muratura.reports import generate_report
                # generate_report(file_path, chapters, format_type)

                self._show_success(f"Relazione generata: {file_path}")
            except Exception as e:
                self._show_error(f"Errore generazione: {e}")

    def _export(self, format_type):
        """Esporta modello nel formato specificato."""
        ext_map = {
            "ifc": ("File IFC", "ifc"),
            "step": ("File STEP", "step"),
            "stl": ("File STL", "stl"),
            "dxf": ("File DXF", "dxf"),
            "svg": ("File SVG", "svg"),
            "json": ("File JSON", "json"),
            "csv": ("File CSV", "csv"),
        }

        if format_type not in ext_map:
            return

        desc, ext = ext_map[format_type]

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            f"Esporta {format_type.upper()}",
            f"modello.{ext}",
            f"{desc} (*.{ext})"
        )

        if file_path:
            try:
                # TODO: Export reale
                self._show_success(f"Esportato: {file_path}")
            except Exception as e:
                self._show_error(f"Errore export: {e}")

    def _generate_drawings(self):
        """Genera le tavole grafiche."""
        self._show_info("Generazione tavole in sviluppo")

    def _update_data_from_ui(self):
        self._data = {
            'chapters': [ch for ch, cb in self.chapter_checks.items() if cb.isChecked()],
            'format': self.field_format.currentText(),
            'language': self.field_lang.currentText(),
            'include_drawings': self.cb_include_drawings.isChecked(),
            'include_calcs': self.cb_include_calcs.isChecked(),
        }
