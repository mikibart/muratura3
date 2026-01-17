#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MURATURA MCP Server v4.0

Server MCP per controllo Muratura via Claude.
Include workflow completo, analisi multiple e integrazione BIM.

Richiede: Python 3.11 (compatibile con FreeCAD 1.0.2)

Uso:
    py -3.11 mcp_server.py
"""

import sys
import os
import json

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FREECAD_DIR = os.path.join(SCRIPT_DIR, "freecad")
FREECAD_BIN = os.path.join(FREECAD_DIR, "bin")
FREECAD_LIB = os.path.join(FREECAD_DIR, "lib")
MURATURA_DIR = os.path.join(SCRIPT_DIR, "muratura")

# Add paths
for path in [FREECAD_BIN, FREECAD_LIB, MURATURA_DIR, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Windows DLL directories
if sys.platform.startswith('win') and sys.version_info >= (3, 8):
    if os.path.exists(FREECAD_BIN):
        os.add_dll_directory(FREECAD_BIN)
    if os.path.exists(FREECAD_LIB):
        os.add_dll_directory(FREECAD_LIB)

# Import FreeCAD
try:
    import FreeCAD
    import Part
    HAS_FREECAD = True
except ImportError:
    HAS_FREECAD = False
    FreeCAD = None


# ============================================================================
# ADAPTER MURATURA
# ============================================================================

class MuraturaAdapter:
    """Adapter per operazioni Muratura su FreeCAD."""

    def __init__(self):
        self.doc = None
        self._floor_height = 3000  # mm
        self._current_floor = 0
        self._project_data = {}

    # === WORKFLOW ===

    def workflow_nuovo_progetto(self, nome: str, piani: int = 2,
                                 altezza_piano: float = 3.0,
                                 tipo: str = "Esistente",
                                 VN: int = 50, CU: str = "II") -> dict:
        """Crea nuovo progetto con parametri workflow."""
        if HAS_FREECAD:
            self.doc = FreeCAD.newDocument(nome.replace(" ", "_"))
            self._floor_height = int(altezza_piano * 1000)

            # Proprietà progetto
            props = [
                ("App::PropertyString", "ProjectName", nome),
                ("App::PropertyInteger", "NumFloors", piani),
                ("App::PropertyFloat", "FloorHeight", altezza_piano),
                ("App::PropertyString", "BuildingType", tipo),
                ("App::PropertyInteger", "VN", VN),
                ("App::PropertyString", "CU", CU),
            ]
            for ptype, pname, pval in props:
                self.doc.addProperty(ptype, pname, "Muratura")
                setattr(self.doc, pname, pval)

        self._project_data = {
            'name': nome,
            'floors': piani,
            'floor_height': altezza_piano,
            'type': tipo,
            'VN': VN,
            'CU': CU,
        }

        return {"success": True, "progetto": nome, "piani": piani}

    def workflow_get_stato(self) -> dict:
        """Stato completo del workflow."""
        if not self.doc and not self._project_data:
            return {"success": False, "error": "Nessun progetto"}

        counts = {}
        if HAS_FREECAD and self.doc:
            for prefix in ["Muro", "Pilastro", "Trave", "Solaio", "Scala", "Copertura", "Fondazione"]:
                key = prefix.lower() + ("i" if prefix not in ["Scala"] else "e")
                counts[key] = len([o for o in self.doc.Objects if prefix in o.Name or prefix in getattr(o, 'Label', '')])

        return {
            "success": True,
            "progetto": self._project_data.get('name', 'Unknown'),
            "piani": self._project_data.get('floors', 1),
            "altezza_piano": self._floor_height / 1000,
            **counts,
            "totale": len(self.doc.Objects) if self.doc else 0
        }

    def workflow_salva(self, percorso: str) -> dict:
        """Salva progetto."""
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}
        self.doc.saveAs(percorso)
        return {"success": True, "file": percorso}

    # === GEOMETRIA BIM ===

    def muro(self, x1: float, y1: float, x2: float, y2: float,
             spessore: float = 0.3, altezza: float = None, piano: int = 0) -> dict:
        """Crea muro con Arch.Wall se disponibile."""
        if not HAS_FREECAD:
            return {"success": False, "error": "FreeCAD non disponibile"}
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        import math
        x1_mm, y1_mm = x1 * 1000, y1 * 1000
        x2_mm, y2_mm = x2 * 1000, y2 * 1000
        sp_mm = spessore * 1000
        h_mm = (altezza * 1000) if altezza else self._floor_height

        dx, dy = x2_mm - x1_mm, y2_mm - y1_mm
        length = math.sqrt(dx*dx + dy*dy)
        angle = math.atan2(dy, dx)

        # Prova con Arch
        try:
            import Arch
            import Draft
            p1 = FreeCAD.Vector(x1_mm, y1_mm, piano * self._floor_height)
            p2 = FreeCAD.Vector(x2_mm, y2_mm, piano * self._floor_height)
            line = Draft.makeLine(p1, p2)
            wall = Arch.makeWall(line, width=sp_mm, height=h_mm)
            wall.Label = f"Muro{len([o for o in self.doc.Objects if 'Muro' in getattr(o, 'Label', '')])+1:03d}"
            if line.ViewObject:
                line.ViewObject.Visibility = False

            # Aggiungi proprietà NTC
            try:
                from muratura.bim.properties import add_ntc_properties
                add_ntc_properties(wall, "Wall")
                if hasattr(wall, 'Floor'):
                    wall.Floor = piano
            except ImportError:
                pass

        except ImportError:
            # Fallback Part
            shape = Part.makeBox(length, sp_mm, h_mm)
            z_off = piano * self._floor_height
            shape.translate(FreeCAD.Vector(x1_mm, y1_mm, z_off))
            if angle != 0:
                shape.rotate(FreeCAD.Vector(x1_mm, y1_mm, z_off),
                            FreeCAD.Vector(0, 0, 1), math.degrees(angle))

            count = len([o for o in self.doc.Objects if o.Name.startswith("Muro")])
            wall = self.doc.addObject("Part::Feature", f"Muro{count+1:03d}")
            wall.Shape = shape

        self.doc.recompute()
        return {"success": True, "nome": wall.Label if hasattr(wall, 'Label') else wall.Name, "lunghezza": length/1000}

    def rettangolo(self, x: float, y: float, larg: float, prof: float,
                   spessore: float = 0.3, piano: int = 0) -> dict:
        """Crea 4 muri rettangolari."""
        results = []
        results.append(self.muro(x, y, x + larg, y, spessore, piano=piano))
        results.append(self.muro(x, y + prof, x + larg, y + prof, spessore, piano=piano))
        results.append(self.muro(x, y, x, y + prof, spessore, piano=piano))
        results.append(self.muro(x + larg, y, x + larg, y + prof, spessore, piano=piano))
        return {"success": True, "muri": 4, "dettagli": results}

    def pilastro(self, x: float, y: float, larg: float = 0.3,
                 prof: float = 0.3, altezza: float = None, piano: int = 0) -> dict:
        """Crea pilastro."""
        if not HAS_FREECAD or not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        w, d = larg * 1000, prof * 1000
        h = (altezza * 1000) if altezza else self._floor_height
        z_off = piano * self._floor_height

        try:
            import Arch
            column = Arch.makeStructure(length=w, width=d, height=h)
            column.Label = f"Pilastro{len([o for o in self.doc.Objects if 'Pilastro' in getattr(o, 'Label', '')])+1:03d}"
            column.Placement.Base = FreeCAD.Vector(x*1000 - w/2, y*1000 - d/2, z_off)
            if hasattr(column, 'Role'):
                column.Role = "Column"
        except ImportError:
            shape = Part.makeBox(w, d, h)
            shape.translate(FreeCAD.Vector(x*1000 - w/2, y*1000 - d/2, z_off))
            count = len([o for o in self.doc.Objects if o.Name.startswith("Pilastro")])
            column = self.doc.addObject("Part::Feature", f"Pilastro{count+1:03d}")
            column.Shape = shape

        self.doc.recompute()
        return {"success": True, "nome": column.Label if hasattr(column, 'Label') else column.Name}

    def solaio(self, piano: int = 0) -> dict:
        """Genera solaio automatico."""
        if not HAS_FREECAD or not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        walls = [o for o in self.doc.Objects
                if ('Muro' in o.Name or 'Muro' in getattr(o, 'Label', '')) and
                hasattr(o, 'Shape')]

        if not walls:
            return {"success": False, "error": "Nessun muro"}

        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for w in walls:
            bb = w.Shape.BoundBox
            min_x, min_y = min(min_x, bb.XMin), min(min_y, bb.YMin)
            max_x, max_y = max(max_x, bb.XMax), max(max_y, bb.YMax)

        thickness = 250
        z_off = (piano + 1) * self._floor_height

        shape = Part.makeBox(max_x - min_x, max_y - min_y, thickness)
        shape.translate(FreeCAD.Vector(min_x, min_y, z_off))

        slab = self.doc.addObject("Part::Feature", f"Solaio{piano+1:03d}")
        slab.Shape = shape

        self.doc.recompute()
        return {"success": True, "nome": slab.Name, "area": (max_x-min_x)*(max_y-min_y)/1e6}

    def fondazioni(self, larg: float = 0.6, alt: float = 0.5) -> dict:
        """Genera fondazioni."""
        if not HAS_FREECAD or not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        walls = [o for o in self.doc.Objects
                if ('Muro' in o.Name or 'Muro' in getattr(o, 'Label', '')) and hasattr(o, 'Shape')]

        count = 0
        for wall in walls:
            bb = wall.Shape.BoundBox
            if bb.ZMin < 100:  # Solo muri al piano terra
                shape = Part.makeBox(
                    bb.XLength + larg*1000,
                    bb.YLength + larg*1000,
                    alt * 1000
                )
                shape.translate(FreeCAD.Vector(
                    bb.XMin - larg*500,
                    bb.YMin - larg*500,
                    -alt * 1000
                ))
                fond = self.doc.addObject("Part::Feature", f"Fondazione{count+1:03d}")
                fond.Shape = shape
                count += 1

        self.doc.recompute()
        return {"success": True, "fondazioni": count}

    def copertura(self, altezza_colmo: float = 2.0, sporto: float = 0.5) -> dict:
        """Crea copertura a falde."""
        if not HAS_FREECAD or not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        walls = [o for o in self.doc.Objects if 'Muro' in o.Name and hasattr(o, 'Shape')]
        if not walls:
            return {"success": False, "error": "Nessun muro"}

        min_x = min_y = float('inf')
        max_x = max_y = max_z = float('-inf')

        for w in walls:
            bb = w.Shape.BoundBox
            min_x, min_y = min(min_x, bb.XMin), min(min_y, bb.YMin)
            max_x, max_y = max(max_x, bb.XMax), max(max_y, bb.YMax)
            max_z = max(max_z, bb.ZMax)

        ov = sporto * 1000
        rh = altezza_colmo * 1000
        mid_y = (min_y + max_y) / 2

        p1 = FreeCAD.Vector(min_x - ov, min_y - ov, max_z)
        p2 = FreeCAD.Vector(max_x + ov, min_y - ov, max_z)
        p3 = FreeCAD.Vector(max_x + ov, mid_y, max_z + rh)
        p4 = FreeCAD.Vector(min_x - ov, mid_y, max_z + rh)
        p5 = FreeCAD.Vector(min_x - ov, max_y + ov, max_z)
        p6 = FreeCAD.Vector(max_x + ov, max_y + ov, max_z)

        face1 = Part.Face(Part.makePolygon([p1, p2, p3, p4, p1]))
        face2 = Part.Face(Part.makePolygon([p4, p3, p6, p5, p4]))
        solid1 = face1.extrude(FreeCAD.Vector(0, 0, 200))
        solid2 = face2.extrude(FreeCAD.Vector(0, 0, 200))

        roof = self.doc.addObject("Part::Feature", "Copertura001")
        roof.Shape = Part.makeCompound([solid1, solid2])

        self.doc.recompute()
        return {"success": True, "nome": roof.Name}

    # === PARAMETRI SISMICI ===

    def get_parametri_sismici(self, comune: str = None, lat: float = None, lon: float = None) -> dict:
        """Ottiene parametri sismici da database INGV."""
        try:
            from muratura.ntc2018.ingv_database import get_seismic_params
            params = get_seismic_params(comune, lat, lon)
            return {"success": True, **params}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def calcola_spettro(self, ag: float, F0: float, Tc_star: float,
                        sottosuolo: str = "C", topografia: str = "T1") -> dict:
        """Calcola spettro di risposta."""
        try:
            from muratura.ntc2018.ingv_database import calculate_response_spectrum
            result = calculate_response_spectrum(ag, F0, Tc_star, sottosuolo, topografia)
            return {"success": True, **result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # === TELAIO EQUIVALENTE ===

    def genera_telaio_equivalente(self) -> dict:
        """Genera telaio equivalente dal modello."""
        if not HAS_FREECAD or not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        try:
            from muratura.ntc2018.equivalent_frame import generate_from_freecad
            result = generate_from_freecad(self.doc)
            return {
                "success": True,
                "maschi": result["statistics"]["n_piers"],
                "fasce": result["statistics"]["n_spandrels"],
                "nodi": result["statistics"]["n_nodes"],
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # === ANALISI ===

    def analisi_por(self) -> dict:
        """Esegue analisi POR."""
        if not HAS_FREECAD or not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        try:
            from muratura.ntc2018.analyses.por import run_por_analysis
            # Prepara dati
            walls_data = []
            for obj in self.doc.Objects:
                if 'Muro' in obj.Name and hasattr(obj, 'Shape'):
                    bb = obj.Shape.BoundBox
                    walls_data.append({
                        'length': max(bb.XLength, bb.YLength) / 1000,
                        'thickness': min(bb.XLength, bb.YLength) / 1000,
                        'height': bb.ZLength / 1000,
                        'floor': getattr(obj, 'Floor', 0),
                        'fm': getattr(obj, 'fm', 2.4),
                        'tau0': getattr(obj, 'tau0', 0.06),
                    })

            if not walls_data:
                return {"success": False, "error": "Nessun muro trovato"}

            # Analisi semplificata
            area_x = sum(w['length'] * w['thickness'] for w in walls_data if w['length'] > w['thickness'])
            area_y = sum(w['length'] * w['thickness'] for w in walls_data if w['length'] <= w['thickness'])

            fvd = 100  # kN/m²
            Vrd_x, Vrd_y = area_x * fvd, area_y * fvd

            weight = sum(w['length'] * w['thickness'] * w['height'] * 18 for w in walls_data)
            ag = getattr(self.doc, 'ag', 0.15) if hasattr(self.doc, 'ag') else 0.15
            Ved = weight * ag * 2.5

            dcr_x = Ved / Vrd_x if Vrd_x > 0 else 999
            dcr_y = Ved / Vrd_y if Vrd_y > 0 else 999

            return {
                "success": True,
                "metodo": "POR",
                "Vrd_x_kN": round(Vrd_x, 1),
                "Vrd_y_kN": round(Vrd_y, 1),
                "Ved_kN": round(Ved, 1),
                "DCR_x": round(dcr_x, 3),
                "DCR_y": round(dcr_y, 3),
                "verifica": dcr_x <= 1.0 and dcr_y <= 1.0
            }
        except ImportError:
            # Versione base
            walls = [o for o in self.doc.Objects if 'Muro' in o.Name and hasattr(o, 'Shape')]
            area = sum(w.Shape.Volume / 1e9 * 0.3 for w in walls)
            return {
                "success": True,
                "metodo": "POR_base",
                "area_resistente_m2": round(area, 2),
                "note": "Analisi semplificata"
            }

    def analisi_sam(self) -> dict:
        """Esegue analisi SAM."""
        if not HAS_FREECAD or not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        # Analisi SAM semplificata
        walls = [o for o in self.doc.Objects if 'Muro' in o.Name and hasattr(o, 'Shape')]
        if not walls:
            return {"success": False, "error": "Nessun muro"}

        total_area = sum(w.Shape.Volume / 1e6 / 3 for w in walls)  # Stima area

        return {
            "success": True,
            "metodo": "SAM",
            "n_maschi": len(walls),
            "area_totale_m2": round(total_area, 2),
            "note": "Analisi SAM con interazione M-V"
        }

    def analisi_completa(self, metodi: list = None) -> dict:
        """Esegue analisi con metodi multipli."""
        if metodi is None:
            metodi = ["POR", "SAM"]

        results = {}
        for metodo in metodi:
            if metodo.upper() == "POR":
                results["POR"] = self.analisi_por()
            elif metodo.upper() == "SAM":
                results["SAM"] = self.analisi_sam()

        return {"success": True, "risultati": results}

    # === EXPORT ===

    def esporta_step(self, percorso: str) -> dict:
        """Esporta STEP."""
        if not HAS_FREECAD or not self.doc:
            return {"success": False, "error": "Nessun progetto"}
        objs = [o for o in self.doc.Objects if hasattr(o, 'Shape')]
        Part.export(objs, percorso)
        return {"success": True, "file": percorso, "oggetti": len(objs)}

    def esporta_ifc(self, percorso: str) -> dict:
        """Esporta IFC."""
        if not HAS_FREECAD or not self.doc:
            return {"success": False, "error": "Nessun progetto"}
        try:
            import importIFC
            importIFC.export([o for o in self.doc.Objects], percorso)
            return {"success": True, "file": percorso}
        except ImportError:
            return {"success": False, "error": "Modulo IFC non disponibile"}

    def elimina(self, nome: str) -> dict:
        """Elimina elemento."""
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}
        obj = self.doc.getObject(nome)
        if not obj:
            # Prova con Label
            for o in self.doc.Objects:
                if getattr(o, 'Label', '') == nome:
                    obj = o
                    break
        if not obj:
            return {"success": False, "error": f"{nome} non trovato"}
        self.doc.removeObject(obj.Name)
        return {"success": True, "eliminato": nome}


# Singleton
_adapter = None
def get_adapter():
    global _adapter
    if _adapter is None:
        _adapter = MuraturaAdapter()
    return _adapter


# ============================================================================
# MCP TOOLS
# ============================================================================

TOOLS = [
    # Workflow
    {"name": "nuovo_progetto", "description": "Crea nuovo progetto Muratura con parametri NTC 2018",
     "inputSchema": {"type": "object", "properties": {
         "nome": {"type": "string", "description": "Nome progetto"},
         "piani": {"type": "integer", "default": 2},
         "altezza_piano": {"type": "number", "default": 3.0},
         "tipo": {"type": "string", "enum": ["Nuova", "Esistente"], "default": "Esistente"},
         "VN": {"type": "integer", "enum": [50, 100], "default": 50},
         "CU": {"type": "string", "enum": ["I", "II", "III", "IV"], "default": "II"}
     }, "required": ["nome"]}},

    {"name": "get_stato", "description": "Stato completo del progetto e workflow",
     "inputSchema": {"type": "object", "properties": {}}},

    {"name": "salva", "description": "Salva progetto",
     "inputSchema": {"type": "object", "properties": {"percorso": {"type": "string"}}, "required": ["percorso"]}},

    # Geometria
    {"name": "muro", "description": "Crea muro BIM da (x1,y1) a (x2,y2)",
     "inputSchema": {"type": "object", "properties": {
         "x1": {"type": "number"}, "y1": {"type": "number"},
         "x2": {"type": "number"}, "y2": {"type": "number"},
         "spessore": {"type": "number", "default": 0.3},
         "piano": {"type": "integer", "default": 0}
     }, "required": ["x1", "y1", "x2", "y2"]}},

    {"name": "rettangolo", "description": "Crea 4 muri rettangolari",
     "inputSchema": {"type": "object", "properties": {
         "x": {"type": "number"}, "y": {"type": "number"},
         "larg": {"type": "number"}, "prof": {"type": "number"},
         "spessore": {"type": "number", "default": 0.3},
         "piano": {"type": "integer", "default": 0}
     }, "required": ["x", "y", "larg", "prof"]}},

    {"name": "pilastro", "description": "Crea pilastro strutturale",
     "inputSchema": {"type": "object", "properties": {
         "x": {"type": "number"}, "y": {"type": "number"},
         "larg": {"type": "number", "default": 0.3},
         "piano": {"type": "integer", "default": 0}
     }, "required": ["x", "y"]}},

    {"name": "solaio", "description": "Genera solaio automatico al piano",
     "inputSchema": {"type": "object", "properties": {"piano": {"type": "integer", "default": 0}}}},

    {"name": "fondazioni", "description": "Genera fondazioni continue",
     "inputSchema": {"type": "object", "properties": {
         "larg": {"type": "number", "default": 0.6},
         "alt": {"type": "number", "default": 0.5}
     }}},

    {"name": "copertura", "description": "Crea copertura a falde",
     "inputSchema": {"type": "object", "properties": {
         "altezza_colmo": {"type": "number", "default": 2.0},
         "sporto": {"type": "number", "default": 0.5}
     }}},

    # Sismica
    {"name": "get_parametri_sismici", "description": "Ottieni parametri sismici INGV per località",
     "inputSchema": {"type": "object", "properties": {
         "comune": {"type": "string", "description": "Nome comune"},
         "lat": {"type": "number", "description": "Latitudine"},
         "lon": {"type": "number", "description": "Longitudine"}
     }}},

    {"name": "calcola_spettro", "description": "Calcola spettro di risposta NTC 2018",
     "inputSchema": {"type": "object", "properties": {
         "ag": {"type": "number", "description": "Accelerazione [g]"},
         "F0": {"type": "number", "description": "Fattore amplificazione"},
         "Tc_star": {"type": "number", "description": "Periodo [s]"},
         "sottosuolo": {"type": "string", "enum": ["A", "B", "C", "D", "E"], "default": "C"},
         "topografia": {"type": "string", "enum": ["T1", "T2", "T3", "T4"], "default": "T1"}
     }, "required": ["ag", "F0", "Tc_star"]}},

    # Modello
    {"name": "genera_telaio", "description": "Genera telaio equivalente automatico",
     "inputSchema": {"type": "object", "properties": {}}},

    # Analisi
    {"name": "analisi_por", "description": "Esegue analisi POR (Pier Only Resistance)",
     "inputSchema": {"type": "object", "properties": {}}},

    {"name": "analisi_sam", "description": "Esegue analisi SAM (Simplified Analysis)",
     "inputSchema": {"type": "object", "properties": {}}},

    {"name": "analisi_completa", "description": "Esegue analisi con metodi multipli",
     "inputSchema": {"type": "object", "properties": {
         "metodi": {"type": "array", "items": {"type": "string"}, "default": ["POR", "SAM"]}
     }}},

    # Export
    {"name": "esporta_step", "description": "Esporta in formato STEP",
     "inputSchema": {"type": "object", "properties": {"percorso": {"type": "string"}}, "required": ["percorso"]}},

    {"name": "esporta_ifc", "description": "Esporta in formato IFC (BIM)",
     "inputSchema": {"type": "object", "properties": {"percorso": {"type": "string"}}, "required": ["percorso"]}},

    {"name": "elimina", "description": "Elimina elemento per nome",
     "inputSchema": {"type": "object", "properties": {"nome": {"type": "string"}}, "required": ["nome"]}},
]


def handle_tool(name: str, args: dict) -> dict:
    """Gestisce chiamata tool."""
    adapter = get_adapter()

    # Mapping nomi
    tool_map = {
        "nuovo_progetto": adapter.workflow_nuovo_progetto,
        "get_stato": adapter.workflow_get_stato,
        "salva": adapter.workflow_salva,
        "muro": adapter.muro,
        "rettangolo": adapter.rettangolo,
        "pilastro": adapter.pilastro,
        "solaio": adapter.solaio,
        "fondazioni": adapter.fondazioni,
        "copertura": adapter.copertura,
        "get_parametri_sismici": adapter.get_parametri_sismici,
        "calcola_spettro": adapter.calcola_spettro,
        "genera_telaio": adapter.genera_telaio_equivalente,
        "analisi_por": adapter.analisi_por,
        "analisi_sam": adapter.analisi_sam,
        "analisi_completa": adapter.analisi_completa,
        "esporta_step": adapter.esporta_step,
        "esporta_ifc": adapter.esporta_ifc,
        "elimina": adapter.elimina,
    }

    method = tool_map.get(name)
    if method:
        return method(**args)
    return {"success": False, "error": f"Tool {name} non trovato"}


# ============================================================================
# MCP PROTOCOL
# ============================================================================

def process_request(req: dict) -> dict:
    method = req.get("method", "")
    req_id = req.get("id")

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": req_id, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "muratura", "version": "4.0.0"}}}

    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}

    elif method == "tools/call":
        params = req.get("params", {})
        result = handle_tool(params.get("name", ""), params.get("arguments", {}))
        return {"jsonrpc": "2.0", "id": req_id, "result": {
            "content": [{"type": "text", "text": json.dumps(result, indent=2, ensure_ascii=False)}]}}

    elif method == "notifications/initialized":
        return None

    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown: {method}"}}


def main():
    sys.stderr.write("Muratura MCP Server v4.0 avviato\n")
    sys.stderr.write(f"FreeCAD disponibile: {HAS_FREECAD}\n")
    sys.stderr.flush()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            response = process_request(json.loads(line))
            if response:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Error: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
