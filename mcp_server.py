#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MURATURA MCP Server v3.0

Server MCP per controllo Muratura via Claude.
Usa FreeCAD direttamente come backend geometrico.

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
for path in [FREECAD_BIN, FREECAD_LIB, MURATURA_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Windows DLL directories
if sys.platform.startswith('win') and sys.version_info >= (3, 8):
    os.add_dll_directory(FREECAD_BIN)
    os.add_dll_directory(FREECAD_LIB)

# Import FreeCAD
import FreeCAD
import Part

# ============================================================================
# ADAPTER FREECAD
# ============================================================================

class MuraturaAdapter:
    """Adapter per operazioni Muratura su FreeCAD."""

    def __init__(self):
        self.doc = None
        self._floor_height = 3000  # mm
        self._current_floor = 0

    # === PROGETTO ===

    def nuovo_progetto(self, nome: str, piani: int = 2, altezza_piano: float = 3.0) -> dict:
        """Crea nuovo progetto."""
        self.doc = FreeCAD.newDocument(nome.replace(" ", "_"))
        self._floor_height = int(altezza_piano * 1000)

        # Proprietà progetto
        self.doc.addProperty("App::PropertyString", "ProjectName", "Muratura")
        self.doc.addProperty("App::PropertyInteger", "NumFloors", "Muratura")
        self.doc.addProperty("App::PropertyFloat", "FloorHeight", "Muratura")

        self.doc.ProjectName = nome
        self.doc.NumFloors = piani
        self.doc.FloorHeight = altezza_piano

        return {"success": True, "documento": self.doc.Name, "piani": piani}

    def get_stato(self) -> dict:
        """Stato progetto."""
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        counts = {}
        for prefix in ["Muro", "Pilastro", "Trave", "Solaio", "Scala", "Copertura", "Fondazione"]:
            counts[prefix.lower() + "i" if prefix != "Scala" else "scale"] = len(
                [o for o in self.doc.Objects if o.Name.startswith(prefix)]
            )

        return {
            "success": True,
            "progetto": getattr(self.doc, 'ProjectName', self.doc.Name),
            "piani": getattr(self.doc, 'NumFloors', 1),
            "altezza_piano": self._floor_height / 1000,
            **counts,
            "totale": len(self.doc.Objects)
        }

    def salva(self, percorso: str) -> dict:
        """Salva progetto."""
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}
        self.doc.saveAs(percorso)
        return {"success": True, "file": percorso}

    # === GEOMETRIA ===

    def muro(self, x1: float, y1: float, x2: float, y2: float,
             spessore: float = 0.3, altezza: float = None, piano: int = 0) -> dict:
        """Crea muro."""
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

        shape = Part.makeBox(length, sp_mm, h_mm)
        z_off = piano * self._floor_height
        shape.translate(FreeCAD.Vector(x1_mm, y1_mm, z_off))

        if angle != 0:
            shape.rotate(FreeCAD.Vector(x1_mm, y1_mm, z_off),
                        FreeCAD.Vector(0, 0, 1), math.degrees(angle))

        count = len([o for o in self.doc.Objects if o.Name.startswith("Muro")])
        wall = self.doc.addObject("Part::Feature", f"Muro{count+1:03d}")
        wall.Shape = shape

        for prop, val in [("Length", length/1000), ("Thickness", spessore),
                          ("Height", h_mm/1000), ("Floor", piano)]:
            ptype = "App::PropertyFloat" if isinstance(val, float) else "App::PropertyInteger"
            wall.addProperty(ptype, prop, "Geometry")
            setattr(wall, prop, val)

        self.doc.recompute()
        return {"success": True, "nome": wall.Name, "lunghezza": length/1000}

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
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        w, d = larg * 1000, prof * 1000
        h = (altezza * 1000) if altezza else self._floor_height
        z_off = piano * self._floor_height

        shape = Part.makeBox(w, d, h)
        shape.translate(FreeCAD.Vector(x*1000 - w/2, y*1000 - d/2, z_off))

        count = len([o for o in self.doc.Objects if o.Name.startswith("Pilastro")])
        col = self.doc.addObject("Part::Feature", f"Pilastro{count+1:03d}")
        col.Shape = shape

        self.doc.recompute()
        return {"success": True, "nome": col.Name}

    def trave(self, x1: float, y1: float, x2: float, y2: float,
              larg: float = 0.3, alt: float = 0.5, piano: int = 0) -> dict:
        """Crea trave."""
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        import math
        x1_mm, y1_mm = x1 * 1000, y1 * 1000
        x2_mm, y2_mm = x2 * 1000, y2 * 1000
        w_mm, h_mm = larg * 1000, alt * 1000

        dx, dy = x2_mm - x1_mm, y2_mm - y1_mm
        length = math.sqrt(dx*dx + dy*dy)
        angle = math.atan2(dy, dx)
        z_off = (piano + 1) * self._floor_height - h_mm

        shape = Part.makeBox(length, w_mm, h_mm)
        shape.translate(FreeCAD.Vector(x1_mm, y1_mm - w_mm/2, z_off))

        if angle != 0:
            shape.rotate(FreeCAD.Vector(x1_mm, y1_mm, z_off),
                        FreeCAD.Vector(0, 0, 1), math.degrees(angle))

        count = len([o for o in self.doc.Objects if o.Name.startswith("Trave")])
        beam = self.doc.addObject("Part::Feature", f"Trave{count+1:03d}")
        beam.Shape = shape

        self.doc.recompute()
        return {"success": True, "nome": beam.Name, "lunghezza": length/1000}

    def solaio(self, piano: int = 0) -> dict:
        """Genera solaio automatico."""
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        walls = [o for o in self.doc.Objects
                if o.Name.startswith("Muro") and
                hasattr(o, 'Floor') and o.Floor == piano]

        if not walls:
            return {"success": False, "error": "Nessun muro al piano"}

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
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        walls = [o for o in self.doc.Objects
                if o.Name.startswith("Muro") and
                hasattr(o, 'Floor') and o.Floor == 0]

        count = 0
        for wall in walls:
            bb = wall.Shape.BoundBox
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

    def scala(self, x: float, y: float, larg: float = 1.2, piano: int = 0) -> dict:
        """Crea scala."""
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        w_mm = larg * 1000
        h_total = self._floor_height
        n_steps = int(h_total / 170)
        step_h = h_total / n_steps
        step_d = 280

        shapes = []
        for i in range(n_steps):
            step = Part.makeBox(w_mm, step_d, step_h * (i + 1))
            step.translate(FreeCAD.Vector(x*1000, y*1000 + i*step_d, piano * self._floor_height))
            shapes.append(step)

        compound = Part.makeCompound(shapes)
        count = len([o for o in self.doc.Objects if o.Name.startswith("Scala")])
        stair = self.doc.addObject("Part::Feature", f"Scala{count+1:03d}")
        stair.Shape = compound

        self.doc.recompute()
        return {"success": True, "nome": stair.Name, "gradini": n_steps}

    def copertura(self, altezza_colmo: float = 2.0, sporto: float = 0.5) -> dict:
        """Crea copertura a falde."""
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        walls = [o for o in self.doc.Objects if o.Name.startswith("Muro")]
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

        # Due falde
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

    # === ANALISI ===

    def analisi_por(self) -> dict:
        """Esegue analisi POR."""
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}

        walls = [o for o in self.doc.Objects if o.Name.startswith("Muro")]
        if not walls:
            return {"success": False, "error": "Nessun muro"}

        area_x = area_y = 0
        for w in walls:
            bb = w.Shape.BoundBox
            t = getattr(w, 'Thickness', 0.3)
            if bb.XLength > bb.YLength:
                area_x += (bb.XLength / 1000) * t
            else:
                area_y += (bb.YLength / 1000) * t

        fvd = 100  # kN/m²
        Vrd_x, Vrd_y = area_x * fvd, area_y * fvd

        weight = sum(w.Shape.Volume / 1e9 * 18 for w in walls)
        ag = getattr(self.doc, 'ag', 0.15) if hasattr(self.doc, 'ag') else 0.15
        Ved = weight * ag * 2.5

        dcr_x = Ved / Vrd_x if Vrd_x > 0 else 999
        dcr_y = Ved / Vrd_y if Vrd_y > 0 else 999

        results = {
            "metodo": "POR",
            "area_x": round(area_x, 2),
            "area_y": round(area_y, 2),
            "Vrd_x_kN": round(Vrd_x, 1),
            "Vrd_y_kN": round(Vrd_y, 1),
            "Ved_kN": round(Ved, 1),
            "DCR_x": round(dcr_x, 3),
            "DCR_y": round(dcr_y, 3),
            "verifica_x": dcr_x <= 1.0,
            "verifica_y": dcr_y <= 1.0
        }

        if not hasattr(self.doc, "AnalysisResults"):
            self.doc.addProperty("App::PropertyString", "AnalysisResults", "Analisi")
        self.doc.AnalysisResults = json.dumps(results)

        return {"success": True, **results}

    # === EXPORT ===

    def esporta_step(self, percorso: str) -> dict:
        """Esporta STEP."""
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}
        objs = [o for o in self.doc.Objects if hasattr(o, 'Shape')]
        Part.export(objs, percorso)
        return {"success": True, "file": percorso, "oggetti": len(objs)}

    def elimina(self, nome: str) -> dict:
        """Elimina elemento."""
        if not self.doc:
            return {"success": False, "error": "Nessun progetto"}
        obj = self.doc.getObject(nome)
        if not obj:
            return {"success": False, "error": f"{nome} non trovato"}
        self.doc.removeObject(nome)
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
    {"name": "nuovo_progetto", "description": "Crea nuovo progetto Muratura",
     "inputSchema": {"type": "object", "properties": {
         "nome": {"type": "string"}, "piani": {"type": "integer", "default": 2},
         "altezza_piano": {"type": "number", "default": 3.0}}, "required": ["nome"]}},

    {"name": "get_stato", "description": "Stato del progetto corrente",
     "inputSchema": {"type": "object", "properties": {}}},

    {"name": "salva", "description": "Salva progetto",
     "inputSchema": {"type": "object", "properties": {"percorso": {"type": "string"}}, "required": ["percorso"]}},

    {"name": "muro", "description": "Crea muro da (x1,y1) a (x2,y2)",
     "inputSchema": {"type": "object", "properties": {
         "x1": {"type": "number"}, "y1": {"type": "number"},
         "x2": {"type": "number"}, "y2": {"type": "number"},
         "spessore": {"type": "number", "default": 0.3},
         "piano": {"type": "integer", "default": 0}}, "required": ["x1", "y1", "x2", "y2"]}},

    {"name": "rettangolo", "description": "Crea 4 muri rettangolari",
     "inputSchema": {"type": "object", "properties": {
         "x": {"type": "number"}, "y": {"type": "number"},
         "larg": {"type": "number"}, "prof": {"type": "number"},
         "spessore": {"type": "number", "default": 0.3},
         "piano": {"type": "integer", "default": 0}}, "required": ["x", "y", "larg", "prof"]}},

    {"name": "pilastro", "description": "Crea pilastro",
     "inputSchema": {"type": "object", "properties": {
         "x": {"type": "number"}, "y": {"type": "number"},
         "larg": {"type": "number", "default": 0.3},
         "prof": {"type": "number", "default": 0.3},
         "piano": {"type": "integer", "default": 0}}, "required": ["x", "y"]}},

    {"name": "trave", "description": "Crea trave",
     "inputSchema": {"type": "object", "properties": {
         "x1": {"type": "number"}, "y1": {"type": "number"},
         "x2": {"type": "number"}, "y2": {"type": "number"},
         "larg": {"type": "number", "default": 0.3},
         "alt": {"type": "number", "default": 0.5},
         "piano": {"type": "integer", "default": 0}}, "required": ["x1", "y1", "x2", "y2"]}},

    {"name": "solaio", "description": "Genera solaio automatico",
     "inputSchema": {"type": "object", "properties": {"piano": {"type": "integer", "default": 0}}}},

    {"name": "fondazioni", "description": "Genera fondazioni",
     "inputSchema": {"type": "object", "properties": {
         "larg": {"type": "number", "default": 0.6},
         "alt": {"type": "number", "default": 0.5}}}},

    {"name": "scala", "description": "Crea scala",
     "inputSchema": {"type": "object", "properties": {
         "x": {"type": "number"}, "y": {"type": "number"},
         "larg": {"type": "number", "default": 1.2},
         "piano": {"type": "integer", "default": 0}}, "required": ["x", "y"]}},

    {"name": "copertura", "description": "Crea copertura a falde",
     "inputSchema": {"type": "object", "properties": {
         "altezza_colmo": {"type": "number", "default": 2.0},
         "sporto": {"type": "number", "default": 0.5}}}},

    {"name": "analisi_por", "description": "Esegue analisi POR",
     "inputSchema": {"type": "object", "properties": {}}},

    {"name": "esporta_step", "description": "Esporta in STEP",
     "inputSchema": {"type": "object", "properties": {"percorso": {"type": "string"}}, "required": ["percorso"]}},

    {"name": "elimina", "description": "Elimina elemento",
     "inputSchema": {"type": "object", "properties": {"nome": {"type": "string"}}, "required": ["nome"]}},
]


def handle_tool(name: str, args: dict) -> dict:
    """Gestisce chiamata tool."""
    adapter = get_adapter()
    method = getattr(adapter, name, None)
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
            "serverInfo": {"name": "muratura", "version": "3.0.0"}}}

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
    sys.stderr.write("Muratura MCP Server v3.0 avviato\n")
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
