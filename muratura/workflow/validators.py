# -*- coding: utf-8 -*-
"""
Validators - Validazione completezza fasi workflow.

Ogni fase ha validatori specifici che verificano la completezza dei dati.
"""

from typing import Dict, Any, List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Risultato di una validazione."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class PhaseValidator:
    """Validatore base per una fase."""

    def __init__(self, phase_id: int, phase_name: str):
        self.phase_id = phase_id
        self.phase_name = phase_name
        self._rules: List[Tuple[str, Callable[[Dict], Tuple[bool, str]]]] = []

    def add_rule(self, name: str, check_func: Callable[[Dict], Tuple[bool, str]]):
        """
        Aggiunge una regola di validazione.

        Args:
            name: Nome della regola
            check_func: Funzione che prende i dati e restituisce (is_valid, message)
        """
        self._rules.append((name, check_func))

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Esegue tutte le validazioni."""
        errors = []
        warnings = []

        for rule_name, check_func in self._rules:
            try:
                is_valid, message = check_func(data)
                if not is_valid:
                    if message.startswith("WARNING:"):
                        warnings.append(message.replace("WARNING:", "").strip())
                    else:
                        errors.append(message)
            except Exception as e:
                errors.append(f"Errore validazione '{rule_name}': {e}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


# =============================================================================
# VALIDATORI PER OGNI FASE
# =============================================================================

def create_phase1_validator() -> PhaseValidator:
    """Validatore Fase 1: Progetto."""
    v = PhaseValidator(1, "Progetto")

    def check_name(data):
        name = data.get('name', '').strip()
        if not name:
            return False, "Il nome del progetto è obbligatorio"
        if len(name) < 3:
            return False, "Il nome deve avere almeno 3 caratteri"
        return True, ""

    def check_vn(data):
        vn = data.get('VN', 0)
        if vn not in [50, 100]:
            return False, "Vita nominale deve essere 50 o 100 anni"
        return True, ""

    def check_cu(data):
        cu = data.get('CU', 0)
        if cu not in [0.7, 1.0, 1.5, 2.0]:
            return False, "Classe d'uso non valida"
        return True, ""

    def check_lc(data):
        if data.get('is_existing', False):
            lc = data.get('LC', '')
            if lc not in ['LC1', 'LC2', 'LC3']:
                return False, "Livello di conoscenza obbligatorio per edifici esistenti"
        return True, ""

    v.add_rule("nome_progetto", check_name)
    v.add_rule("vita_nominale", check_vn)
    v.add_rule("classe_uso", check_cu)
    v.add_rule("livello_conoscenza", check_lc)

    return v


def create_phase2_validator() -> PhaseValidator:
    """Validatore Fase 2: Geometria."""
    v = PhaseValidator(2, "Geometria")

    def check_levels(data):
        levels = data.get('levels', [])
        if not levels:
            return False, "Definire almeno un piano"
        return True, ""

    def check_walls(data):
        # Verifica che ci siano muri nel documento FreeCAD
        # Questo richiede accesso al documento
        return True, ""  # TODO: implementare con FreeCAD

    v.add_rule("piani", check_levels)
    v.add_rule("muri", check_walls)

    return v


def create_phase3_validator() -> PhaseValidator:
    """Validatore Fase 3: Materiali."""
    v = PhaseValidator(3, "Materiali")

    def check_material(data):
        material = data.get('material', '')
        if not material:
            return False, "Selezionare un tipo di muratura"
        return True, ""

    def check_corrections(data):
        corrections = data.get('corrections', {})
        # Verifica incompatibilità
        if corrections.get('nucleo_scadente') and corrections.get('iniezioni'):
            return False, "WARNING: Iniezioni incompatibili con nucleo scadente non consolidato"
        return True, ""

    v.add_rule("materiale", check_material)
    v.add_rule("correzioni", check_corrections)

    return v


def create_phase4_validator() -> PhaseValidator:
    """Validatore Fase 4: Struttura."""
    v = PhaseValidator(4, "Struttura")

    def check_cordoli(data):
        cordolo = data.get('cordolo', {})
        if cordolo:
            h = cordolo.get('height', 0)
            if h < 0.20:
                return False, "WARNING: Altezza cordolo inferiore a 20cm"
        return True, ""

    v.add_rule("cordoli", check_cordoli)

    return v


def create_phase5_validator() -> PhaseValidator:
    """Validatore Fase 5: Solai."""
    v = PhaseValidator(5, "Solai")

    def check_floor_type(data):
        floor = data.get('floor', {})
        if not floor.get('type'):
            return False, "Specificare tipologia solaio"
        return True, ""

    def check_stiffness(data):
        floor = data.get('floor', {})
        stiffness = floor.get('stiffness', '')
        if not stiffness:
            return False, "Classificare rigidezza di piano"
        return True, ""

    v.add_rule("tipologia", check_floor_type)
    v.add_rule("rigidezza", check_stiffness)

    return v


def create_phase6_validator() -> PhaseValidator:
    """Validatore Fase 6: Carichi."""
    v = PhaseValidator(6, "Carichi")

    def check_g2(data):
        g2 = data.get('G2', {})
        total = g2.get('total', 0)
        if total <= 0:
            return False, "Definire carichi permanenti non strutturali G2"
        return True, ""

    def check_q(data):
        q = data.get('Q', {})
        qk = q.get('qk', 0)
        if qk <= 0:
            return False, "Definire carichi variabili Q"
        return True, ""

    v.add_rule("permanenti", check_g2)
    v.add_rule("variabili", check_q)

    return v


def create_phase7_validator() -> PhaseValidator:
    """Validatore Fase 7: Sismica."""
    v = PhaseValidator(7, "Sismica")

    def check_location(data):
        loc = data.get('location', {})
        if not loc.get('comune') and not (loc.get('lat') and loc.get('lon')):
            return False, "Specificare località o coordinate"
        return True, ""

    def check_soil(data):
        soil = data.get('soil', '')
        if soil not in ['A', 'B', 'C', 'D', 'E']:
            return False, "Selezionare categoria sottosuolo"
        return True, ""

    v.add_rule("localita", check_location)
    v.add_rule("sottosuolo", check_soil)

    return v


def create_phase8_validator() -> PhaseValidator:
    """Validatore Fase 8: Modello."""
    v = PhaseValidator(8, "Modello")

    def check_model_generated(data):
        # TODO: verificare che il modello telaio equivalente sia generato
        return True, ""

    v.add_rule("modello", check_model_generated)

    return v


def create_phase9_validator() -> PhaseValidator:
    """Validatore Fase 9: Analisi."""
    v = PhaseValidator(9, "Analisi")

    def check_methods(data):
        methods = data.get('methods', [])
        if not methods:
            return False, "Selezionare almeno un metodo di analisi"
        return True, ""

    v.add_rule("metodi", check_methods)

    return v


def create_phase10_validator() -> PhaseValidator:
    """Validatore Fase 10: Verifiche."""
    v = PhaseValidator(10, "Verifiche")

    # Le verifiche sono output, non richiedono input specifici
    return v


def create_phase11_validator() -> PhaseValidator:
    """Validatore Fase 11: Rinforzi."""
    v = PhaseValidator(11, "Rinforzi")

    # Fase opzionale
    return v


def create_phase12_validator() -> PhaseValidator:
    """Validatore Fase 12: Relazione."""
    v = PhaseValidator(12, "Relazione")

    def check_chapters(data):
        chapters = data.get('chapters', [])
        if not chapters:
            return False, "Selezionare almeno un capitolo"
        return True, ""

    v.add_rule("capitoli", check_chapters)

    return v


# =============================================================================
# FACTORY
# =============================================================================

VALIDATORS = {
    1: create_phase1_validator,
    2: create_phase2_validator,
    3: create_phase3_validator,
    4: create_phase4_validator,
    5: create_phase5_validator,
    6: create_phase6_validator,
    7: create_phase7_validator,
    8: create_phase8_validator,
    9: create_phase9_validator,
    10: create_phase10_validator,
    11: create_phase11_validator,
    12: create_phase12_validator,
}


def get_validator(phase_id: int) -> PhaseValidator:
    """Ottiene il validatore per una fase."""
    factory = VALIDATORS.get(phase_id)
    if factory:
        return factory()
    return PhaseValidator(phase_id, f"Fase {phase_id}")


def validate_phase(phase_id: int, data: Dict[str, Any]) -> ValidationResult:
    """Valida i dati di una fase."""
    validator = get_validator(phase_id)
    return validator.validate(data)


def validate_all_phases(phases_data: Dict[int, Dict]) -> Dict[int, ValidationResult]:
    """Valida tutte le fasi."""
    results = {}
    for phase_id in range(1, 13):
        data = phases_data.get(phase_id, {})
        results[phase_id] = validate_phase(phase_id, data)
    return results
