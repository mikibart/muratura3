# -*- coding: utf-8 -*-
"""
Project Panel - Fase 1: Dati Progetto

Gestisce dati generali, vita nominale, classe d'uso, livello conoscenza.
"""

from datetime import date
from typing import Dict, Any

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide import QtWidgets, QtCore, QtGui

from .base_panel import BasePhasePanel


class ProjectPanel(BasePhasePanel):
    """
    Pannello Fase 1: Progetto.
    Dati generali, VN, CU, LC secondo NTC 2018.
    """

    # Classi d'uso NTC 2018 Tab. 2.4.II
    USE_CLASSES = {
        "I": ("Scarsa presenza persone", 0.7),
        "II": ("Normali affollamenti", 1.0),
        "III": ("Affollamenti significativi", 1.5),
        "IV": ("Funzioni pubbliche strategiche", 2.0),
    }

    # Livelli di conoscenza NTC 2018 §8.5.4
    KNOWLEDGE_LEVELS = {
        "LC1": ("Limitata", 1.35),
        "LC2": ("Adeguata", 1.20),
        "LC3": ("Accurata", 1.00),
    }

    # Categorie intervento NTC 2018 §8.4
    INTERVENTION_CATEGORIES = [
        "Nuova costruzione",
        "Miglioramento sismico",
        "Adeguamento sismico",
        "Riparazione/Intervento locale",
    ]

    # Tipologie edificio
    BUILDING_TYPES = [
        "Residenziale",
        "Produttivo/Commerciale",
        "Agricolo",
        "Storico/Monumentale",
        "Strategico",
    ]

    def __init__(self, parent=None):
        super().__init__(1, "Progetto", parent)
        self.set_description(
            "Inserisci i dati generali del progetto: committente, progettista, "
            "vita nominale, classe d'uso e livello di conoscenza (per edifici esistenti)."
        )
        self.setup_content()
        self._setup_validators()

    def setup_content(self):
        """Costruisce il contenuto del pannello."""
        # Gruppo: Dati generali
        general_group, general_layout = self.create_form_group("Dati Generali")

        self.field_name = self.create_input_field("Nome progetto")
        general_layout.addRow("Nome progetto:", self.field_name)

        self.field_client = self.create_input_field("Committente")
        general_layout.addRow("Committente:", self.field_client)

        self.field_designer = self.create_input_field("Progettista strutturale")
        general_layout.addRow("Progettista:", self.field_designer)

        self.field_date = QtWidgets.QDateEdit()
        self.field_date.setDate(QtCore.QDate.currentDate())
        self.field_date.setCalendarPopup(True)
        self.field_date.dateChanged.connect(self._mark_dirty)
        general_layout.addRow("Data:", self.field_date)

        self.field_code = self.create_input_field("Generato automaticamente")
        self.field_code.setReadOnly(True)
        self.field_code.setStyleSheet("background-color: #f0f0f0;")
        general_layout.addRow("Codice:", self.field_code)

        self.content_layout.addWidget(general_group)

        # Gruppo: Tipologia edificio
        type_group, type_layout = self.create_form_group("Tipologia Edificio")

        self.field_type = self.create_combo_field(self.BUILDING_TYPES)
        type_layout.addRow("Tipologia:", self.field_type)

        self.field_construction = QtWidgets.QButtonGroup(self)
        rb_new = QtWidgets.QRadioButton("Nuova costruzione")
        rb_existing = QtWidgets.QRadioButton("Edificio esistente")
        rb_new.setChecked(True)
        self.field_construction.addButton(rb_new, 0)
        self.field_construction.addButton(rb_existing, 1)
        rb_new.toggled.connect(self._on_construction_changed)
        rb_existing.toggled.connect(self._on_construction_changed)

        construction_widget = QtWidgets.QWidget()
        construction_layout = QtWidgets.QHBoxLayout(construction_widget)
        construction_layout.setContentsMargins(0, 0, 0, 0)
        construction_layout.addWidget(rb_new)
        construction_layout.addWidget(rb_existing)
        construction_layout.addStretch()
        type_layout.addRow("Costruzione:", construction_widget)

        self.field_year = self.create_int_field(1800, 2100)
        self.field_year.setValue(1960)
        self.field_year.setEnabled(False)
        type_layout.addRow("Anno costruzione:", self.field_year)

        self.field_intervention = self.create_combo_field(self.INTERVENTION_CATEGORIES)
        type_layout.addRow("Categoria intervento:", self.field_intervention)

        self.content_layout.addWidget(type_group)

        # Gruppo: Parametri normativi NTC 2018 §2.4
        ntc_group, ntc_layout = self.create_form_group("Parametri Normativi (NTC 2018 §2.4)")

        # Vita nominale
        vn_widget = QtWidgets.QWidget()
        vn_layout = QtWidgets.QHBoxLayout(vn_widget)
        vn_layout.setContentsMargins(0, 0, 0, 0)

        self.field_vn = QtWidgets.QButtonGroup(self)
        rb_vn50 = QtWidgets.QRadioButton("50 anni (ordinario)")
        rb_vn100 = QtWidgets.QRadioButton("100 anni (strategico)")
        rb_vn50.setChecked(True)
        self.field_vn.addButton(rb_vn50, 50)
        self.field_vn.addButton(rb_vn100, 100)
        rb_vn50.toggled.connect(self._update_vr)
        rb_vn100.toggled.connect(self._update_vr)

        vn_layout.addWidget(rb_vn50)
        vn_layout.addWidget(rb_vn100)
        vn_layout.addStretch()
        ntc_layout.addRow("Vita nominale VN:", vn_widget)

        # Classe d'uso
        cu_widget = QtWidgets.QWidget()
        cu_layout = QtWidgets.QGridLayout(cu_widget)
        cu_layout.setContentsMargins(0, 0, 0, 0)

        self.field_cu = QtWidgets.QButtonGroup(self)
        for i, (class_id, (desc, coeff)) in enumerate(self.USE_CLASSES.items()):
            rb = QtWidgets.QRadioButton(f"Classe {class_id} (CU={coeff})")
            rb.setToolTip(desc)
            if class_id == "II":
                rb.setChecked(True)
            self.field_cu.addButton(rb, i)
            rb.toggled.connect(self._update_vr)
            cu_layout.addWidget(rb, i // 2, i % 2)

        ntc_layout.addRow("Classe d'uso:", cu_widget)

        # VR calcolato
        self.label_vr = QtWidgets.QLabel()
        self.label_vr.setStyleSheet("""
            font-weight: bold;
            padding: 5px;
            background-color: #e3f2fd;
            border-radius: 3px;
        """)
        ntc_layout.addRow("Periodo riferimento VR:", self.label_vr)

        self.content_layout.addWidget(ntc_group)

        # Gruppo: Livello di Conoscenza (solo edifici esistenti)
        self.lc_group, lc_layout = self.create_form_group(
            "Livello di Conoscenza (NTC 2018 §8.5.4)"
        )

        lc_widget = QtWidgets.QWidget()
        lc_vlayout = QtWidgets.QVBoxLayout(lc_widget)
        lc_vlayout.setContentsMargins(0, 0, 0, 0)

        self.field_lc = QtWidgets.QButtonGroup(self)
        lc_descriptions = {
            "LC1": "Geometria: Rilievo completo\nDettagli costruttivi: Limitate\nMateriali: Da normativa",
            "LC2": "Geometria: Rilievo completo\nDettagli costruttivi: Estese\nMateriali: Da prove limitate",
            "LC3": "Geometria: Rilievo completo\nDettagli costruttivi: Esaustive\nMateriali: Da prove estese",
        }

        for i, (lc, (desc, fc)) in enumerate(self.KNOWLEDGE_LEVELS.items()):
            rb = QtWidgets.QRadioButton(f"{lc} - {desc} (FC = {fc})")
            rb.setToolTip(lc_descriptions.get(lc, ""))
            if lc == "LC2":
                rb.setChecked(True)
            self.field_lc.addButton(rb, i)
            rb.toggled.connect(self._update_fc)
            lc_vlayout.addWidget(rb)

        lc_layout.addRow("Livello:", lc_widget)

        # FC calcolato
        self.label_fc = QtWidgets.QLabel()
        self.label_fc.setStyleSheet("""
            font-weight: bold;
            padding: 5px;
            background-color: #fff3e0;
            border-radius: 3px;
        """)
        lc_layout.addRow("Fattore confidenza FC:", self.label_fc)

        self.lc_group.setEnabled(False)  # Disabilitato per nuove costruzioni
        self.content_layout.addWidget(self.lc_group)

        # Spacer
        self.content_layout.addStretch()

        # Aggiorna valori calcolati
        self._update_vr()
        self._update_fc()
        self._generate_code()

    def _on_construction_changed(self):
        """Gestisce cambio tipo costruzione."""
        is_existing = self.field_construction.checkedId() == 1
        self.field_year.setEnabled(is_existing)
        self.lc_group.setEnabled(is_existing)
        self._mark_dirty()

    def _update_vr(self):
        """Calcola e aggiorna VR = VN × CU."""
        vn = self.field_vn.checkedId()
        if vn <= 0:
            vn = 50

        cu_id = self.field_cu.checkedId()
        cu_values = [0.7, 1.0, 1.5, 2.0]
        cu = cu_values[cu_id] if 0 <= cu_id < 4 else 1.0

        vr = vn * cu
        self.label_vr.setText(f"VR = {vn} × {cu} = {vr:.0f} anni")
        self._mark_dirty()

    def _update_fc(self):
        """Aggiorna il fattore di confidenza."""
        lc_id = self.field_lc.checkedId()
        fc_values = [1.35, 1.20, 1.00]
        fc = fc_values[lc_id] if 0 <= lc_id < 3 else 1.20
        lc_names = ["LC1", "LC2", "LC3"]
        lc_name = lc_names[lc_id] if 0 <= lc_id < 3 else "LC2"

        self.label_fc.setText(f"FC ({lc_name}) = {fc}")
        self._mark_dirty()

    def _generate_code(self):
        """Genera codice progetto automatico."""
        today = date.today()
        code = f"MUR-{today.year}{today.month:02d}{today.day:02d}-001"
        self.field_code.setText(code)

    def _setup_validators(self):
        """Configura i validatori."""
        def validate_name():
            if not self.field_name.text().strip():
                return False, "Il nome del progetto è obbligatorio"
            return True, ""

        self.add_validator(validate_name)

    def _update_data_from_ui(self):
        """Aggiorna i dati dall'interfaccia."""
        # Vita nominale
        vn = self.field_vn.checkedId()
        if vn <= 0:
            vn = 50

        # Classe d'uso
        cu_id = self.field_cu.checkedId()
        cu_classes = ["I", "II", "III", "IV"]
        cu_class = cu_classes[cu_id] if 0 <= cu_id < 4 else "II"
        cu_values = [0.7, 1.0, 1.5, 2.0]
        cu_value = cu_values[cu_id] if 0 <= cu_id < 4 else 1.0

        # Livello conoscenza
        lc_id = self.field_lc.checkedId()
        lc_names = ["LC1", "LC2", "LC3"]
        lc_name = lc_names[lc_id] if 0 <= lc_id < 3 else "LC2"
        fc_values = [1.35, 1.20, 1.00]
        fc = fc_values[lc_id] if 0 <= lc_id < 3 else 1.20

        is_existing = self.field_construction.checkedId() == 1

        self._data = {
            # Dati generali
            'name': self.field_name.text().strip(),
            'client': self.field_client.text().strip(),
            'designer': self.field_designer.text().strip(),
            'date': self.field_date.date().toString("yyyy-MM-dd"),
            'code': self.field_code.text(),

            # Tipologia
            'building_type': self.field_type.currentText(),
            'is_existing': is_existing,
            'construction_year': self.field_year.value() if is_existing else None,
            'intervention_category': self.field_intervention.currentText(),

            # NTC 2018
            'VN': vn,
            'CU_class': cu_class,
            'CU': cu_value,
            'VR': vn * cu_value,

            # Livello conoscenza (solo esistenti)
            'LC': lc_name if is_existing else None,
            'FC': fc if is_existing else 1.0,
        }

    def _update_ui_from_data(self):
        """Aggiorna l'interfaccia dai dati."""
        if not self._data:
            return

        # Dati generali
        self.field_name.setText(self._data.get('name', ''))
        self.field_client.setText(self._data.get('client', ''))
        self.field_designer.setText(self._data.get('designer', ''))

        date_str = self._data.get('date', '')
        if date_str:
            self.field_date.setDate(QtCore.QDate.fromString(date_str, "yyyy-MM-dd"))

        code = self._data.get('code', '')
        if code:
            self.field_code.setText(code)

        # Tipologia
        btype = self._data.get('building_type', '')
        idx = self.field_type.findText(btype)
        if idx >= 0:
            self.field_type.setCurrentIndex(idx)

        is_existing = self._data.get('is_existing', False)
        for btn in self.field_construction.buttons():
            if self.field_construction.id(btn) == (1 if is_existing else 0):
                btn.setChecked(True)

        if is_existing and self._data.get('construction_year'):
            self.field_year.setValue(self._data['construction_year'])

        intervention = self._data.get('intervention_category', '')
        idx = self.field_intervention.findText(intervention)
        if idx >= 0:
            self.field_intervention.setCurrentIndex(idx)

        # VN
        vn = self._data.get('VN', 50)
        for btn in self.field_vn.buttons():
            if self.field_vn.id(btn) == vn:
                btn.setChecked(True)

        # CU
        cu_class = self._data.get('CU_class', 'II')
        cu_map = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
        cu_id = cu_map.get(cu_class, 1)
        for btn in self.field_cu.buttons():
            if self.field_cu.id(btn) == cu_id:
                btn.setChecked(True)

        # LC
        if is_existing:
            lc = self._data.get('LC', 'LC2')
            lc_map = {'LC1': 0, 'LC2': 1, 'LC3': 2}
            lc_id = lc_map.get(lc, 1)
            for btn in self.field_lc.buttons():
                if self.field_lc.id(btn) == lc_id:
                    btn.setChecked(True)
