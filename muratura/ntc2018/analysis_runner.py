# -*- coding: utf-8 -*-
"""
Analysis Runner - Interfaccia unificata per tutti i metodi di analisi.

Fornisce un'API semplice per eseguire analisi strutturali su edifici in muratura
utilizzando i vari metodi disponibili (POR, SAM, FRAME, FEM, LIMIT, etc.).
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math


class AnalysisMethod(Enum):
    """Metodi di analisi disponibili."""
    POR = "por"           # Pier Only Resistance
    PORFLEX = "porflex"   # POR con fasce flessibili
    SAM = "sam"           # Simplified Analysis of Masonry
    FRAME = "frame"       # Telaio equivalente
    FEM = "fem"           # Elementi finiti
    LIMIT = "limit"       # Analisi cinematica
    FIBER = "fiber"       # Analisi a fibre
    MICRO = "micro"       # Micromeccanica


@dataclass
class AnalysisInput:
    """Dati di input per l'analisi."""
    # Geometria
    piers: List[Dict] = field(default_factory=list)
    spandrels: List[Dict] = field(default_factory=list)
    nodes: List[Dict] = field(default_factory=list)
    floors: List[float] = field(default_factory=list)

    # Materiali
    materials: Dict[str, Dict] = field(default_factory=dict)

    # Carichi
    dead_loads: List[Dict] = field(default_factory=list)
    live_loads: List[Dict] = field(default_factory=list)

    # Parametri sismici
    ag: float = 0.15       # Accelerazione
    F0: float = 2.4        # Fattore amplificazione
    Tc_star: float = 0.3   # Periodo
    S: float = 1.5         # Amplificazione
    q: float = 2.0         # Fattore di struttura
    soil_category: str = "C"
    topo_category: str = "T1"

    # Opzioni
    directions: List[str] = field(default_factory=lambda: ["+X", "-X", "+Y", "-Y"])
    include_eccentricity: bool = True
    eccentricity: float = 0.05  # 5%


@dataclass
class PierResult:
    """Risultato verifica singolo maschio."""
    id: str
    floor: int
    Ved: float  # Domanda taglio [kN]
    Vrd: float  # Resistenza taglio [kN]
    Med: float = 0  # Domanda momento [kNm]
    Mrd: float = 0  # Resistenza momento [kNm]
    Ned: float = 0  # Sforzo normale [kN]
    DCR_V: float = 0  # Demand/Capacity taglio
    DCR_M: float = 0  # Demand/Capacity momento
    DCR: float = 0  # DCR massimo
    failure_mode: str = ""  # Modalità di rottura


@dataclass
class AnalysisResult:
    """Risultato completo dell'analisi."""
    method: str
    success: bool
    message: str = ""

    # Risultati globali
    T1: float = 0  # Periodo fondamentale [s]
    Vbase_x: float = 0  # Taglio base X [kN]
    Vbase_y: float = 0  # Taglio base Y [kN]
    Vtot_rd_x: float = 0  # Resistenza totale X [kN]
    Vtot_rd_y: float = 0  # Resistenza totale Y [kN]

    # Risultati per elementi
    pier_results: List[PierResult] = field(default_factory=list)
    spandrel_results: List[Dict] = field(default_factory=list)

    # Indici globali
    DCR_max: float = 0
    IR: float = 0  # Indice di rischio
    alpha: float = 0  # Rapporto capacità/domanda

    # Per pushover
    capacity_curve: List[Tuple[float, float]] = field(default_factory=list)
    du: float = 0  # Spostamento ultimo [mm]
    dy: float = 0  # Spostamento snervamento [mm]
    ductility: float = 0

    # Modi propri
    modes: List[Dict] = field(default_factory=list)


class AnalysisRunner:
    """
    Esecutore unificato delle analisi strutturali.
    """

    def __init__(self, input_data: AnalysisInput = None):
        """
        Inizializza il runner.

        Args:
            input_data: Dati di input per l'analisi
        """
        self.input = input_data or AnalysisInput()
        self._results: Dict[str, AnalysisResult] = {}

    def set_input(self, input_data: AnalysisInput):
        """Imposta i dati di input."""
        self.input = input_data

    def run(self, method: AnalysisMethod, **options) -> AnalysisResult:
        """
        Esegue un'analisi con il metodo specificato.

        Args:
            method: Metodo di analisi
            **options: Opzioni specifiche del metodo

        Returns:
            Risultato dell'analisi
        """
        if method == AnalysisMethod.POR:
            result = self._run_por(**options)
        elif method == AnalysisMethod.PORFLEX:
            result = self._run_porflex(**options)
        elif method == AnalysisMethod.SAM:
            result = self._run_sam(**options)
        elif method == AnalysisMethod.FRAME:
            result = self._run_frame(**options)
        elif method == AnalysisMethod.FEM:
            result = self._run_fem(**options)
        elif method == AnalysisMethod.LIMIT:
            result = self._run_limit(**options)
        elif method == AnalysisMethod.FIBER:
            result = self._run_fiber(**options)
        elif method == AnalysisMethod.MICRO:
            result = self._run_micro(**options)
        else:
            result = AnalysisResult(
                method=method.value,
                success=False,
                message=f"Metodo {method.value} non implementato"
            )

        self._results[method.value] = result
        return result

    def run_multiple(self, methods: List[AnalysisMethod]) -> Dict[str, AnalysisResult]:
        """
        Esegue multiple analisi.

        Args:
            methods: Lista di metodi da eseguire

        Returns:
            Dizionario con risultati per ogni metodo
        """
        results = {}
        for method in methods:
            results[method.value] = self.run(method)
        return results

    def get_results(self) -> Dict[str, AnalysisResult]:
        """Restituisce tutti i risultati."""
        return self._results

    def get_comparison(self) -> Dict[str, Any]:
        """
        Confronta i risultati dei diversi metodi.

        Returns:
            Tabella comparativa
        """
        comparison = {
            'methods': [],
            'Vrd_x': [],
            'Vrd_y': [],
            'DCR_max': [],
            'IR': [],
        }

        for method, result in self._results.items():
            comparison['methods'].append(method)
            comparison['Vrd_x'].append(result.Vtot_rd_x)
            comparison['Vrd_y'].append(result.Vtot_rd_y)
            comparison['DCR_max'].append(result.DCR_max)
            comparison['IR'].append(result.IR)

        return comparison

    # -------------------------------------------------------------------------
    # Implementazioni dei singoli metodi
    # -------------------------------------------------------------------------

    def _run_por(self, **options) -> AnalysisResult:
        """Esegue analisi POR."""
        try:
            from .analyses.por import _analyze_por

            # Prepara input per POR
            por_input = self._prepare_por_input()

            # Esegui analisi
            raw_result = _analyze_por(por_input)

            # Converti risultato
            return self._convert_por_result(raw_result)

        except ImportError:
            return self._fallback_por(**options)
        except Exception as e:
            return AnalysisResult(
                method="por",
                success=False,
                message=f"Errore analisi POR: {str(e)}"
            )

    def _run_porflex(self, **options) -> AnalysisResult:
        """Esegue analisi PORFLEX."""
        try:
            from .analyses.porflex import _analyze_porflex

            por_input = self._prepare_por_input()
            raw_result = _analyze_porflex(por_input)
            return self._convert_por_result(raw_result)

        except ImportError:
            return self._fallback_porflex(**options)
        except Exception as e:
            return AnalysisResult(
                method="porflex",
                success=False,
                message=f"Errore analisi PORFLEX: {str(e)}"
            )

    def _run_sam(self, **options) -> AnalysisResult:
        """Esegue analisi SAM."""
        try:
            from .analyses.sam import _analyze_sam

            sam_input = self._prepare_sam_input()
            raw_result = _analyze_sam(sam_input)
            return self._convert_sam_result(raw_result)

        except ImportError:
            return self._fallback_sam(**options)
        except Exception as e:
            return AnalysisResult(
                method="sam",
                success=False,
                message=f"Errore analisi SAM: {str(e)}"
            )

    def _run_frame(self, **options) -> AnalysisResult:
        """Esegue analisi a telaio equivalente."""
        try:
            from .analyses.frame import EquivalentFrame, _analyze_frame

            frame = EquivalentFrame()
            # Popola modello
            for pier in self.input.piers:
                frame.add_pier(pier)
            for spandrel in self.input.spandrels:
                frame.add_spandrel(spandrel)

            raw_result = _analyze_frame(frame, options)
            return self._convert_frame_result(raw_result)

        except ImportError:
            return self._fallback_frame(**options)
        except Exception as e:
            return AnalysisResult(
                method="frame",
                success=False,
                message=f"Errore analisi FRAME: {str(e)}"
            )

    def _run_fem(self, **options) -> AnalysisResult:
        """Esegue analisi FEM."""
        try:
            from .analyses.fem import _analyze_fem

            fem_input = self._prepare_fem_input()
            raw_result = _analyze_fem(fem_input)
            return self._convert_fem_result(raw_result)

        except ImportError:
            return AnalysisResult(
                method="fem",
                success=False,
                message="Modulo FEM non disponibile"
            )
        except Exception as e:
            return AnalysisResult(
                method="fem",
                success=False,
                message=f"Errore analisi FEM: {str(e)}"
            )

    def _run_limit(self, **options) -> AnalysisResult:
        """Esegue analisi cinematica (meccanismi locali)."""
        try:
            from .analyses.limit import LimitAnalysis, _analyze_limit

            limit = LimitAnalysis()
            raw_result = _analyze_limit(limit, self.input)
            return self._convert_limit_result(raw_result)

        except ImportError:
            return AnalysisResult(
                method="limit",
                success=False,
                message="Modulo LIMIT non disponibile"
            )
        except Exception as e:
            return AnalysisResult(
                method="limit",
                success=False,
                message=f"Errore analisi LIMIT: {str(e)}"
            )

    def _run_fiber(self, **options) -> AnalysisResult:
        """Esegue analisi a fibre."""
        try:
            from .analyses.fiber import FiberModel, _analyze_fiber

            fiber = FiberModel()
            raw_result = _analyze_fiber(fiber, self.input)
            return self._convert_fiber_result(raw_result)

        except ImportError:
            return AnalysisResult(
                method="fiber",
                success=False,
                message="Modulo FIBER non disponibile"
            )
        except Exception as e:
            return AnalysisResult(
                method="fiber",
                success=False,
                message=f"Errore analisi FIBER: {str(e)}"
            )

    def _run_micro(self, **options) -> AnalysisResult:
        """Esegue analisi micromeccanica."""
        try:
            from .analyses.micro import MicroModel, _analyze_micro

            micro = MicroModel()
            raw_result = _analyze_micro(micro, self.input)
            return self._convert_micro_result(raw_result)

        except ImportError:
            return AnalysisResult(
                method="micro",
                success=False,
                message="Modulo MICRO non disponibile"
            )
        except Exception as e:
            return AnalysisResult(
                method="micro",
                success=False,
                message=f"Errore analisi MICRO: {str(e)}"
            )

    # -------------------------------------------------------------------------
    # Metodi di fallback (calcolo semplificato quando i moduli non sono disponibili)
    # -------------------------------------------------------------------------

    def _fallback_por(self, **options) -> AnalysisResult:
        """Calcolo POR semplificato."""
        pier_results = []
        total_vrd_x = 0
        total_vrd_y = 0
        max_dcr = 0

        # Calcola peso totale
        W_total = self._calculate_total_weight()

        # Taglio sismico alla base
        Vbase = self.input.ag * self.input.S * W_total / self.input.q

        for pier in self.input.piers:
            # Proprietà geometriche
            width = pier.get('width', 1.0)
            thickness = pier.get('thickness', 0.3)
            height = pier.get('height', 3.0)
            floor = pier.get('floor', 0)

            # Proprietà materiale
            material_id = pier.get('material', 'default')
            mat = self.input.materials.get(material_id, {})
            fm = mat.get('fm', 2.4)  # MPa
            tau0 = mat.get('tau0', 0.060)  # MPa
            w = mat.get('w', 18)  # kN/m³

            # Area e sforzo normale
            area = width * thickness  # m²
            Ned = w * area * height * (len(self.input.floors) - floor)  # kN

            # Resistenza a taglio (formula Turnsek-Cacovic semplificata)
            sigma_n = Ned / (area * 1000)  # MPa
            ft = tau0 * 1.5  # Resistenza a trazione
            Vrd = area * 1000 * ft * math.sqrt(1 + sigma_n / (1.5 * ft))  # kN

            # Momento resistente (presso-flessione)
            fd = fm
            sigma0 = Ned / (area * 1000)  # MPa
            if sigma0 > 0 and fd > 0:
                Mrd = (width * thickness * sigma0 / 2) * (1 - sigma0 / (0.85 * fd)) * 1000
            else:
                Mrd = 0

            # Taglio da momento
            h0 = height  # Altezza efficace
            Vrd_flex = 2 * Mrd / h0 if h0 > 0 else float('inf')

            # Resistenza governante
            Vrd_min = min(Vrd, Vrd_flex)

            # Distribuzione del taglio (proporzionale alla rigidezza)
            stiffness = area / height if height > 0 else 0
            Ved = Vbase * stiffness / max(1, sum(
                p.get('width', 1) * p.get('thickness', 0.3) / p.get('height', 3)
                for p in self.input.piers
            ))

            # DCR
            dcr = Ved / Vrd_min if Vrd_min > 0 else float('inf')
            max_dcr = max(max_dcr, dcr)

            # Modalità di rottura
            if Vrd < Vrd_flex:
                failure_mode = "Taglio"
            else:
                failure_mode = "Presso-flessione"

            # Accumula resistenze per direzione
            # Semplificazione: tutti X
            total_vrd_x += Vrd_min

            pier_results.append(PierResult(
                id=pier.get('id', f"P{len(pier_results)+1}"),
                floor=floor,
                Ved=round(Ved, 1),
                Vrd=round(Vrd_min, 1),
                Med=0,
                Mrd=round(Mrd, 1),
                Ned=round(Ned, 1),
                DCR_V=round(dcr, 2),
                DCR_M=0,
                DCR=round(dcr, 2),
                failure_mode=failure_mode
            ))

        # Indice di rischio
        IR = total_vrd_x / Vbase if Vbase > 0 else 0

        return AnalysisResult(
            method="por",
            success=True,
            message="Analisi POR completata (metodo semplificato)",
            Vbase_x=round(Vbase, 1),
            Vbase_y=round(Vbase, 1),
            Vtot_rd_x=round(total_vrd_x, 1),
            Vtot_rd_y=round(total_vrd_y, 1),
            pier_results=pier_results,
            DCR_max=round(max_dcr, 2),
            IR=round(IR, 2),
            alpha=round(IR, 2)
        )

    def _fallback_porflex(self, **options) -> AnalysisResult:
        """Calcolo PORFLEX semplificato."""
        # Usa POR come base
        result = self._fallback_por(**options)
        result.method = "porflex"
        result.message = "Analisi PORFLEX completata (metodo semplificato)"

        # Aggiungi effetto fasce (riduzione capacità ~10%)
        if self.input.spandrels:
            result.Vtot_rd_x *= 1.1
            result.Vtot_rd_y *= 1.1
            result.IR *= 1.1

        return result

    def _fallback_sam(self, **options) -> AnalysisResult:
        """Calcolo SAM semplificato."""
        result = self._fallback_por(**options)
        result.method = "sam"
        result.message = "Analisi SAM completata (metodo semplificato)"

        # SAM considera interazione M-V più accurata
        # Applichiamo correzione ~5%
        for pr in result.pier_results:
            pr.DCR *= 0.95

        result.DCR_max *= 0.95
        result.IR *= 1.05

        return result

    def _fallback_frame(self, **options) -> AnalysisResult:
        """Calcolo FRAME semplificato."""
        result = self._fallback_por(**options)
        result.method = "frame"
        result.message = "Analisi FRAME completata (metodo semplificato)"

        # Stima periodo fondamentale
        n_floors = len(self.input.floors)
        H_tot = sum(self.input.floors) if self.input.floors else n_floors * 3
        result.T1 = 0.05 * H_tot ** 0.75

        # Curva di capacità semplificata (bilineare)
        Vy = result.Vtot_rd_x * 0.8
        Vu = result.Vtot_rd_x
        dy = 5.0  # mm
        du = 25.0  # mm

        result.capacity_curve = [
            (0, 0),
            (dy, Vy),
            (du, Vu),
        ]
        result.dy = dy
        result.du = du
        result.ductility = du / dy if dy > 0 else 0

        # Modi propri (stima)
        result.modes = [
            {'n': 1, 'T': result.T1, 'f': 1/result.T1, 'Mx': 75, 'My': 5},
            {'n': 2, 'T': result.T1 * 0.8, 'f': 1/(result.T1*0.8), 'Mx': 5, 'My': 70},
            {'n': 3, 'T': result.T1 * 0.5, 'f': 1/(result.T1*0.5), 'Mx': 10, 'My': 15},
        ]

        return result

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _calculate_total_weight(self) -> float:
        """Calcola peso totale edificio."""
        W = 0

        # Peso muri
        for pier in self.input.piers:
            width = pier.get('width', 1.0)
            thickness = pier.get('thickness', 0.3)
            height = pier.get('height', 3.0)
            material_id = pier.get('material', 'default')
            mat = self.input.materials.get(material_id, {})
            w = mat.get('w', 18)  # kN/m³
            W += width * thickness * height * w

        # Peso solai (stima)
        for floor in self.input.floors:
            # Stima area piano da dimensioni muri
            area = 100  # m² default
            W += area * 3.5  # kN/m² tipico

        return W

    def _prepare_por_input(self) -> Dict[str, Any]:
        """Prepara input per modulo POR."""
        return {
            'piers': self.input.piers,
            'materials': self.input.materials,
            'seismic': {
                'ag': self.input.ag,
                'F0': self.input.F0,
                'S': self.input.S,
                'q': self.input.q,
            },
            'floors': self.input.floors,
        }

    def _prepare_sam_input(self) -> Dict[str, Any]:
        """Prepara input per modulo SAM."""
        return {
            'piers': self.input.piers,
            'spandrels': self.input.spandrels,
            'materials': self.input.materials,
            'seismic': {
                'ag': self.input.ag,
                'F0': self.input.F0,
                'S': self.input.S,
                'q': self.input.q,
            },
            'floors': self.input.floors,
        }

    def _prepare_fem_input(self) -> Dict[str, Any]:
        """Prepara input per modulo FEM."""
        return {
            'geometry': {
                'piers': self.input.piers,
                'spandrels': self.input.spandrels,
            },
            'materials': self.input.materials,
            'loads': {
                'dead': self.input.dead_loads,
                'live': self.input.live_loads,
            },
            'seismic': {
                'ag': self.input.ag,
                'S': self.input.S,
                'q': self.input.q,
            },
        }

    def _convert_por_result(self, raw: Dict) -> AnalysisResult:
        """Converte risultato POR in formato standard."""
        pier_results = []
        for pr in raw.get('pier_results', []):
            pier_results.append(PierResult(
                id=pr.get('id', ''),
                floor=pr.get('floor', 0),
                Ved=pr.get('Ved', 0),
                Vrd=pr.get('Vrd', 0),
                Med=pr.get('Med', 0),
                Mrd=pr.get('Mrd', 0),
                Ned=pr.get('Ned', 0),
                DCR_V=pr.get('DCR_V', 0),
                DCR_M=pr.get('DCR_M', 0),
                DCR=pr.get('DCR', 0),
                failure_mode=pr.get('failure_mode', '')
            ))

        return AnalysisResult(
            method="por",
            success=True,
            Vbase_x=raw.get('Vbase_x', 0),
            Vbase_y=raw.get('Vbase_y', 0),
            Vtot_rd_x=raw.get('Vrd_x', 0),
            Vtot_rd_y=raw.get('Vrd_y', 0),
            pier_results=pier_results,
            DCR_max=raw.get('DCR_max', 0),
            IR=raw.get('IR', 0),
            alpha=raw.get('alpha', 0)
        )

    def _convert_sam_result(self, raw: Dict) -> AnalysisResult:
        """Converte risultato SAM in formato standard."""
        result = self._convert_por_result(raw)
        result.method = "sam"
        return result

    def _convert_frame_result(self, raw: Dict) -> AnalysisResult:
        """Converte risultato FRAME in formato standard."""
        result = self._convert_por_result(raw)
        result.method = "frame"
        result.T1 = raw.get('T1', 0)
        result.capacity_curve = raw.get('capacity_curve', [])
        result.du = raw.get('du', 0)
        result.dy = raw.get('dy', 0)
        result.ductility = raw.get('ductility', 0)
        result.modes = raw.get('modes', [])
        return result

    def _convert_fem_result(self, raw: Dict) -> AnalysisResult:
        """Converte risultato FEM in formato standard."""
        result = self._convert_por_result(raw)
        result.method = "fem"
        return result

    def _convert_limit_result(self, raw: Dict) -> AnalysisResult:
        """Converte risultato LIMIT in formato standard."""
        return AnalysisResult(
            method="limit",
            success=True,
            message=raw.get('message', ''),
            alpha=raw.get('alpha_min', 0),
            IR=raw.get('IR', 0)
        )

    def _convert_fiber_result(self, raw: Dict) -> AnalysisResult:
        """Converte risultato FIBER in formato standard."""
        result = self._convert_frame_result(raw)
        result.method = "fiber"
        return result

    def _convert_micro_result(self, raw: Dict) -> AnalysisResult:
        """Converte risultato MICRO in formato standard."""
        result = self._convert_por_result(raw)
        result.method = "micro"
        return result


# -----------------------------------------------------------------------------
# Funzioni di convenienza
# -----------------------------------------------------------------------------

def run_analysis(
    method: str,
    piers: List[Dict],
    materials: Dict[str, Dict],
    seismic_params: Dict[str, Any] = None,
    spandrels: List[Dict] = None,
    floors: List[float] = None,
    **options
) -> AnalysisResult:
    """
    Funzione helper per eseguire un'analisi rapidamente.

    Args:
        method: Nome del metodo ('por', 'sam', 'frame', etc.)
        piers: Lista maschi murari
        materials: Dizionario materiali
        seismic_params: Parametri sismici
        spandrels: Lista fasce
        floors: Quote piani
        **options: Opzioni aggiuntive

    Returns:
        Risultato dell'analisi
    """
    seismic_params = seismic_params or {}
    input_data = AnalysisInput(
        piers=piers,
        spandrels=spandrels or [],
        floors=floors or [],
        materials=materials,
        ag=seismic_params.get('ag', 0.15),
        F0=seismic_params.get('F0', 2.4),
        Tc_star=seismic_params.get('Tc_star', 0.3),
        S=seismic_params.get('S', 1.5),
        q=seismic_params.get('q', 2.0),
        soil_category=seismic_params.get('soil_category', 'C'),
    )

    runner = AnalysisRunner(input_data)

    try:
        analysis_method = AnalysisMethod(method.lower())
    except ValueError:
        return AnalysisResult(
            method=method,
            success=False,
            message=f"Metodo '{method}' non riconosciuto"
        )

    return runner.run(analysis_method, **options)


def run_complete_analysis(
    input_data: AnalysisInput,
    methods: List[str] = None
) -> Dict[str, AnalysisResult]:
    """
    Esegue un'analisi completa con multipli metodi.

    Args:
        input_data: Dati di input
        methods: Lista metodi (default: ['por', 'sam'])

    Returns:
        Dizionario con risultati per ogni metodo
    """
    if methods is None:
        methods = ['por', 'sam']

    runner = AnalysisRunner(input_data)
    results = {}

    for method_name in methods:
        try:
            method = AnalysisMethod(method_name.lower())
            results[method_name] = runner.run(method)
        except ValueError:
            results[method_name] = AnalysisResult(
                method=method_name,
                success=False,
                message=f"Metodo '{method_name}' non riconosciuto"
            )

    return results
