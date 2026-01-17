# -*- coding: utf-8 -*-
"""
Base Panel - Classe base per tutti i pannelli fase.
"""

from typing import Dict, Any, Optional, Callable

try:
    from PySide2 import QtWidgets, QtCore, QtGui
except ImportError:
    from PySide import QtWidgets, QtCore, QtGui


class BasePhasePanel(QtWidgets.QWidget):
    """
    Classe base per i pannelli delle fasi del workflow.
    Fornisce struttura comune e metodi di utilità.
    """

    # Segnali
    data_changed = QtCore.Signal(dict)  # Emesso quando i dati cambiano
    phase_completed = QtCore.Signal(int)  # Emesso quando la fase è completata
    validation_error = QtCore.Signal(str)  # Emesso su errore validazione

    def __init__(self, phase_id: int, phase_name: str, parent=None):
        super().__init__(parent)
        self.phase_id = phase_id
        self.phase_name = phase_name
        self._data = {}
        self._validators = []
        self._is_dirty = False

        self._setup_base_ui()

    def _setup_base_ui(self):
        """Costruisce la struttura base del pannello."""
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # Header
        self.header = QtWidgets.QFrame()
        self.header.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.header.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        header_layout = QtWidgets.QVBoxLayout(self.header)

        self.title_label = QtWidgets.QLabel(f"Fase {self.phase_id}: {self.phase_name}")
        self.title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #495057;
        """)
        header_layout.addWidget(self.title_label)

        self.description_label = QtWidgets.QLabel("")
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color: #6c757d; font-size: 12px;")
        header_layout.addWidget(self.description_label)

        self.main_layout.addWidget(self.header)

        # Area contenuto (da implementare nelle sottoclassi)
        self.content_area = QtWidgets.QScrollArea()
        self.content_area.setWidgetResizable(True)
        self.content_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.content_widget = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)

        self.content_area.setWidget(self.content_widget)
        self.main_layout.addWidget(self.content_area, 1)

        # Footer con pulsanti
        self.footer = QtWidgets.QFrame()
        footer_layout = QtWidgets.QHBoxLayout(self.footer)
        footer_layout.setContentsMargins(0, 5, 0, 0)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_data)
        footer_layout.addWidget(self.btn_reset)

        footer_layout.addStretch()

        self.btn_validate = QtWidgets.QPushButton("Valida")
        self.btn_validate.clicked.connect(self.validate)
        footer_layout.addWidget(self.btn_validate)

        self.btn_complete = QtWidgets.QPushButton("Completa Fase")
        self.btn_complete.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.btn_complete.clicked.connect(self._on_complete)
        footer_layout.addWidget(self.btn_complete)

        self.main_layout.addWidget(self.footer)

    def set_description(self, text: str):
        """Imposta la descrizione della fase."""
        self.description_label.setText(text)

    def setup_content(self):
        """
        Metodo da sovrascrivere per costruire il contenuto specifico.
        Chiamato automaticamente dopo __init__.
        """
        pass

    def get_data(self) -> Dict[str, Any]:
        """Restituisce i dati del pannello."""
        return self._data.copy()

    def set_data(self, data: Dict[str, Any]):
        """Imposta i dati del pannello."""
        self._data = data.copy()
        self._update_ui_from_data()
        self._is_dirty = False

    def _update_ui_from_data(self):
        """Aggiorna l'interfaccia dai dati. Da sovrascrivere."""
        pass

    def _update_data_from_ui(self):
        """Aggiorna i dati dall'interfaccia. Da sovrascrivere."""
        pass

    def _mark_dirty(self):
        """Marca il pannello come modificato."""
        self._is_dirty = True
        self._update_data_from_ui()
        self.data_changed.emit(self._data)

    def is_dirty(self) -> bool:
        """Verifica se ci sono modifiche non salvate."""
        return self._is_dirty

    def add_validator(self, validator: Callable[[], tuple]):
        """
        Aggiunge un validatore.
        Il validatore deve restituire (bool, str) dove bool indica successo
        e str è il messaggio di errore.
        """
        self._validators.append(validator)

    def validate(self) -> bool:
        """Esegue la validazione. Restituisce True se valido."""
        self._update_data_from_ui()

        for validator in self._validators:
            is_valid, message = validator()
            if not is_valid:
                self.validation_error.emit(message)
                self._show_error(message)
                return False

        self._show_success("Validazione completata con successo!")
        return True

    def reset_data(self):
        """Resetta i dati del pannello."""
        self._data = {}
        self._update_ui_from_data()
        self._is_dirty = False

    def _on_complete(self):
        """Gestisce il click su Completa Fase."""
        if self.validate():
            self.phase_completed.emit(self.phase_id)

    def _show_error(self, message: str):
        """Mostra un messaggio di errore."""
        QtWidgets.QMessageBox.warning(
            self,
            f"Errore - Fase {self.phase_id}",
            message
        )

    def _show_success(self, message: str):
        """Mostra un messaggio di successo."""
        QtWidgets.QMessageBox.information(
            self,
            f"Successo - Fase {self.phase_id}",
            message
        )

    def _show_info(self, message: str):
        """Mostra un messaggio informativo."""
        QtWidgets.QMessageBox.information(
            self,
            f"Info - Fase {self.phase_id}",
            message
        )

    # Metodi helper per creare widget comuni

    def create_form_group(self, title: str) -> tuple:
        """
        Crea un gruppo form con titolo.
        Restituisce (group_box, layout).
        """
        group = QtWidgets.QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        layout = QtWidgets.QFormLayout(group)
        layout.setLabelAlignment(QtCore.Qt.AlignRight)
        layout.setSpacing(8)
        return group, layout

    def create_input_field(self, placeholder: str = "", width: int = 200) -> QtWidgets.QLineEdit:
        """Crea un campo di input standard."""
        field = QtWidgets.QLineEdit()
        field.setPlaceholderText(placeholder)
        field.setMinimumWidth(width)
        field.textChanged.connect(self._mark_dirty)
        return field

    def create_number_field(
        self,
        min_val: float = 0,
        max_val: float = 9999,
        decimals: int = 2,
        suffix: str = ""
    ) -> QtWidgets.QDoubleSpinBox:
        """Crea un campo numerico."""
        field = QtWidgets.QDoubleSpinBox()
        field.setRange(min_val, max_val)
        field.setDecimals(decimals)
        if suffix:
            field.setSuffix(f" {suffix}")
        field.setMinimumWidth(120)
        field.valueChanged.connect(self._mark_dirty)
        return field

    def create_int_field(
        self,
        min_val: int = 0,
        max_val: int = 9999,
        suffix: str = ""
    ) -> QtWidgets.QSpinBox:
        """Crea un campo intero."""
        field = QtWidgets.QSpinBox()
        field.setRange(min_val, max_val)
        if suffix:
            field.setSuffix(f" {suffix}")
        field.setMinimumWidth(100)
        field.valueChanged.connect(self._mark_dirty)
        return field

    def create_combo_field(self, items: list) -> QtWidgets.QComboBox:
        """Crea un campo combo box."""
        field = QtWidgets.QComboBox()
        field.addItems(items)
        field.setMinimumWidth(150)
        field.currentIndexChanged.connect(self._mark_dirty)
        return field

    def create_check_field(self, text: str = "") -> QtWidgets.QCheckBox:
        """Crea un campo checkbox."""
        field = QtWidgets.QCheckBox(text)
        field.stateChanged.connect(self._mark_dirty)
        return field

    def create_table(self, headers: list, rows: int = 5) -> QtWidgets.QTableWidget:
        """Crea una tabella."""
        table = QtWidgets.QTableWidget(rows, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        table.itemChanged.connect(self._mark_dirty)
        return table
