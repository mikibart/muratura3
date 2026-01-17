# -*- coding: utf-8 -*-
"""Comandi geometria Muratura - Elementi BIM"""

import FreeCAD
import FreeCADGui
import Part
import math


def get_floor_height():
    """Ottiene altezza piano dal documento."""
    doc = FreeCAD.ActiveDocument
    if doc and hasattr(doc, "FloorHeight"):
        return doc.FloorHeight * 1000  # m → mm
    return 3000  # default 3m


class CmdNewWall:
    """Crea un nuovo muro."""

    def GetResources(self):
        return {
            "MenuText": "Nuovo Muro",
            "ToolTip": "Disegna un muro in muratura",
            "Pixmap": "Arch_Wall"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            doc = FreeCAD.newDocument("Muratura")

        # Parametri default
        length = 5000  # mm
        thickness = 300  # mm
        height = get_floor_height()

        # Crea shape
        shape = Part.makeBox(length, thickness, height)

        # Conta muri esistenti
        count = len([o for o in doc.Objects if o.Name.startswith("Muro")])

        # Crea oggetto
        wall = doc.addObject("Part::Feature", f"Muro{count+1:03d}")
        wall.Shape = shape

        # Proprietà
        wall.addProperty("App::PropertyFloat", "Length", "Geometry", "Lunghezza (m)")
        wall.addProperty("App::PropertyFloat", "Thickness", "Geometry", "Spessore (m)")
        wall.addProperty("App::PropertyFloat", "Height", "Geometry", "Altezza (m)")
        wall.addProperty("App::PropertyInteger", "Floor", "Geometry", "Piano")
        wall.addProperty("App::PropertyString", "Material", "NTC2018", "Tipo muratura")

        wall.Length = length / 1000
        wall.Thickness = thickness / 1000
        wall.Height = height / 1000
        wall.Floor = 0
        wall.Material = "Muratura in mattoni pieni"

        # Colore mattone
        if hasattr(wall, "ViewObject"):
            wall.ViewObject.ShapeColor = (0.8, 0.4, 0.2)

        doc.recompute()
        FreeCADGui.Selection.clearSelection()
        FreeCADGui.Selection.addSelection(wall)

    def IsActive(self):
        return True


class CmdNewOpening:
    """Crea un'apertura (finestra/porta) in un muro."""

    def GetResources(self):
        return {
            "MenuText": "Nuova Apertura",
            "ToolTip": "Aggiungi finestra o porta a un muro",
            "Pixmap": "Arch_Window"
        }

    def Activated(self):
        sel = FreeCADGui.Selection.getSelection()
        if not sel or not sel[0].Name.startswith("Muro"):
            FreeCAD.Console.PrintWarning("Seleziona un muro prima\n")
            return

        wall = sel[0]
        doc = FreeCAD.ActiveDocument

        # Parametri apertura (finestra default)
        width = 1200  # mm
        height = 1400  # mm
        sill = 900  # mm da terra

        # Crea apertura come box
        opening_shape = Part.makeBox(width, wall.Thickness * 1000 + 100, height)

        # Posiziona al centro del muro
        wall_length = wall.Length * 1000
        x_offset = (wall_length - width) / 2
        opening_shape.translate(FreeCAD.Vector(x_offset, -50, sill))

        # Boolean cut
        new_shape = wall.Shape.cut(opening_shape)
        wall.Shape = new_shape

        doc.recompute()
        FreeCAD.Console.PrintMessage(f"Apertura aggiunta a {wall.Name}\n")

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class CmdNewColumn:
    """Crea un pilastro."""

    def GetResources(self):
        return {
            "MenuText": "Nuovo Pilastro",
            "ToolTip": "Inserisci un pilastro",
            "Pixmap": "Arch_Structure"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            doc = FreeCAD.newDocument("Muratura")

        # Parametri default
        width = 300  # mm
        depth = 300  # mm
        height = get_floor_height()

        shape = Part.makeBox(width, depth, height)

        count = len([o for o in doc.Objects if o.Name.startswith("Pilastro")])
        column = doc.addObject("Part::Feature", f"Pilastro{count+1:03d}")
        column.Shape = shape

        column.addProperty("App::PropertyFloat", "Width", "Geometry", "Larghezza (m)")
        column.addProperty("App::PropertyFloat", "Depth", "Geometry", "Profondità (m)")
        column.addProperty("App::PropertyFloat", "Height", "Geometry", "Altezza (m)")
        column.addProperty("App::PropertyInteger", "Floor", "Geometry", "Piano")

        column.Width = width / 1000
        column.Depth = depth / 1000
        column.Height = height / 1000
        column.Floor = 0

        if hasattr(column, "ViewObject"):
            column.ViewObject.ShapeColor = (0.6, 0.6, 0.6)

        doc.recompute()

    def IsActive(self):
        return True


class CmdNewBeam:
    """Crea una trave."""

    def GetResources(self):
        return {
            "MenuText": "Nuova Trave",
            "ToolTip": "Inserisci una trave",
            "Pixmap": "Arch_Structure"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            doc = FreeCAD.newDocument("Muratura")

        length = 5000  # mm
        width = 300  # mm
        height = 500  # mm
        z_offset = get_floor_height() - height

        shape = Part.makeBox(length, width, height)
        shape.translate(FreeCAD.Vector(0, 0, z_offset))

        count = len([o for o in doc.Objects if o.Name.startswith("Trave")])
        beam = doc.addObject("Part::Feature", f"Trave{count+1:03d}")
        beam.Shape = shape

        beam.addProperty("App::PropertyFloat", "Length", "Geometry", "Lunghezza (m)")
        beam.addProperty("App::PropertyFloat", "Width", "Geometry", "Larghezza (m)")
        beam.addProperty("App::PropertyFloat", "Height", "Geometry", "Altezza (m)")
        beam.addProperty("App::PropertyInteger", "Floor", "Geometry", "Piano")

        beam.Length = length / 1000
        beam.Width = width / 1000
        beam.Height = height / 1000
        beam.Floor = 0

        if hasattr(beam, "ViewObject"):
            beam.ViewObject.ShapeColor = (0.5, 0.5, 0.5)

        doc.recompute()

    def IsActive(self):
        return True


class CmdNewSlab:
    """Crea un solaio."""

    def GetResources(self):
        return {
            "MenuText": "Nuovo Solaio",
            "ToolTip": "Inserisci un solaio",
            "Pixmap": "Arch_Floor"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            doc = FreeCAD.newDocument("Muratura")

        # Trova bounding box muri
        walls = [o for o in doc.Objects if o.Name.startswith("Muro")]

        if walls:
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            floor = 0

            for wall in walls:
                bbox = wall.Shape.BoundBox
                min_x = min(min_x, bbox.XMin)
                min_y = min(min_y, bbox.YMin)
                max_x = max(max_x, bbox.XMax)
                max_y = max(max_y, bbox.YMax)
                if hasattr(wall, 'Floor'):
                    floor = max(floor, wall.Floor)

            width = max_x - min_x
            depth = max_y - min_y
            z_offset = get_floor_height()
        else:
            min_x = min_y = 0
            width = 10000
            depth = 8000
            floor = 0
            z_offset = get_floor_height()

        thickness = 250  # mm

        shape = Part.makeBox(width, depth, thickness)
        shape.translate(FreeCAD.Vector(min_x, min_y, z_offset))

        count = len([o for o in doc.Objects if o.Name.startswith("Solaio")])
        slab = doc.addObject("Part::Feature", f"Solaio{count+1:03d}")
        slab.Shape = shape

        slab.addProperty("App::PropertyFloat", "Area", "Geometry", "Area (m²)")
        slab.addProperty("App::PropertyFloat", "Thickness", "Geometry", "Spessore (m)")
        slab.addProperty("App::PropertyInteger", "Floor", "Geometry", "Piano")

        slab.Area = (width * depth) / 1e6
        slab.Thickness = thickness / 1000
        slab.Floor = floor

        if hasattr(slab, "ViewObject"):
            slab.ViewObject.ShapeColor = (0.9, 0.9, 0.8)
            slab.ViewObject.Transparency = 50

        doc.recompute()

    def IsActive(self):
        return True


class CmdNewStair:
    """Crea una scala."""

    def GetResources(self):
        return {
            "MenuText": "Nuova Scala",
            "ToolTip": "Inserisci una scala",
            "Pixmap": "Arch_Stairs"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            doc = FreeCAD.newDocument("Muratura")

        # Scala a rampa singola
        width = 1200  # mm
        height = get_floor_height()
        num_steps = int(height / 170)  # alzata ~17cm
        step_height = height / num_steps
        step_depth = 280  # pedata

        shapes = []
        for i in range(num_steps):
            step = Part.makeBox(width, step_depth, step_height * (i + 1))
            step.translate(FreeCAD.Vector(0, i * step_depth, 0))
            shapes.append(step)

        compound = Part.makeCompound(shapes)

        count = len([o for o in doc.Objects if o.Name.startswith("Scala")])
        stair = doc.addObject("Part::Feature", f"Scala{count+1:03d}")
        stair.Shape = compound

        stair.addProperty("App::PropertyInteger", "NumSteps", "Geometry", "Numero gradini")
        stair.addProperty("App::PropertyFloat", "Width", "Geometry", "Larghezza (m)")
        stair.addProperty("App::PropertyInteger", "Floor", "Geometry", "Piano")

        stair.NumSteps = num_steps
        stair.Width = width / 1000
        stair.Floor = 0

        if hasattr(stair, "ViewObject"):
            stair.ViewObject.ShapeColor = (0.7, 0.7, 0.7)

        doc.recompute()

    def IsActive(self):
        return True


class CmdNewRoof:
    """Crea una copertura."""

    def GetResources(self):
        return {
            "MenuText": "Nuova Copertura",
            "ToolTip": "Inserisci una copertura a falde",
            "Pixmap": "Arch_Roof"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            doc = FreeCAD.newDocument("Muratura")

        # Trova bbox edificio
        walls = [o for o in doc.Objects if o.Name.startswith("Muro")]

        if walls:
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            max_z = 0

            for wall in walls:
                bbox = wall.Shape.BoundBox
                min_x = min(min_x, bbox.XMin)
                min_y = min(min_y, bbox.YMin)
                max_x = max(max_x, bbox.XMax)
                max_y = max(max_y, bbox.YMax)
                max_z = max(max_z, bbox.ZMax)
        else:
            min_x = min_y = 0
            max_x = 10000
            max_y = 8000
            max_z = 3000

        width = max_x - min_x
        depth = max_y - min_y
        overhang = 500  # sporto
        ridge_height = 2000  # altezza colmo

        # Copertura a due falde
        p1 = FreeCAD.Vector(min_x - overhang, min_y - overhang, max_z)
        p2 = FreeCAD.Vector(max_x + overhang, min_y - overhang, max_z)
        p3 = FreeCAD.Vector(max_x + overhang, (min_y + max_y) / 2, max_z + ridge_height)
        p4 = FreeCAD.Vector(min_x - overhang, (min_y + max_y) / 2, max_z + ridge_height)
        p5 = FreeCAD.Vector(min_x - overhang, max_y + overhang, max_z)
        p6 = FreeCAD.Vector(max_x + overhang, max_y + overhang, max_z)

        # Crea due falde
        face1 = Part.Face(Part.makePolygon([p1, p2, p3, p4, p1]))
        face2 = Part.Face(Part.makePolygon([p4, p3, p6, p5, p4]))

        # Estrudi per spessore
        thickness = 200
        solid1 = face1.extrude(FreeCAD.Vector(0, 0, thickness))
        solid2 = face2.extrude(FreeCAD.Vector(0, 0, thickness))

        compound = Part.makeCompound([solid1, solid2])

        count = len([o for o in doc.Objects if o.Name.startswith("Copertura")])
        roof = doc.addObject("Part::Feature", f"Copertura{count+1:03d}")
        roof.Shape = compound

        roof.addProperty("App::PropertyFloat", "RidgeHeight", "Geometry", "Altezza colmo (m)")
        roof.addProperty("App::PropertyFloat", "Overhang", "Geometry", "Sporto (m)")

        roof.RidgeHeight = ridge_height / 1000
        roof.Overhang = overhang / 1000

        if hasattr(roof, "ViewObject"):
            roof.ViewObject.ShapeColor = (0.6, 0.3, 0.2)

        doc.recompute()

    def IsActive(self):
        return True


# Registra comandi
FreeCADGui.addCommand("Muratura_NewWall", CmdNewWall())
FreeCADGui.addCommand("Muratura_NewOpening", CmdNewOpening())
FreeCADGui.addCommand("Muratura_NewColumn", CmdNewColumn())
FreeCADGui.addCommand("Muratura_NewBeam", CmdNewBeam())
FreeCADGui.addCommand("Muratura_NewSlab", CmdNewSlab())
FreeCADGui.addCommand("Muratura_NewStair", CmdNewStair())
FreeCADGui.addCommand("Muratura_NewRoof", CmdNewRoof())
