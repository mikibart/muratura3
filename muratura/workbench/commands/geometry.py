# -*- coding: utf-8 -*-
"""
Comandi geometria Muratura - Elementi BIM con Arch

Usa esclusivamente oggetti Arch per compatibilità IFC.
"""

import FreeCAD
import FreeCADGui
import math

# Import condizionale Arch
try:
    import Arch
    import Draft
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    import Part


def get_floor_height():
    """Ottiene altezza piano dal documento."""
    doc = FreeCAD.ActiveDocument
    if doc and hasattr(doc, "FloorHeight"):
        return doc.FloorHeight * 1000  # m → mm
    return 3000  # default 3m


def add_ntc_properties(obj, element_type="Wall"):
    """Aggiunge proprietà NTC 2018 a un oggetto."""
    try:
        from muratura.bim.properties import add_ntc_properties as add_props
        add_props(obj, element_type)
    except ImportError:
        # Fallback: aggiungi proprietà base
        props = [
            ("App::PropertyInteger", "Floor", "Muratura", "Piano"),
            ("App::PropertyString", "ElementType", "Muratura", "Tipo elemento"),
        ]
        for ptype, name, group, desc in props:
            if not hasattr(obj, name):
                obj.addProperty(ptype, name, group, desc)


class CmdNewWall:
    """Crea un nuovo muro usando Arch.makeWall."""

    def GetResources(self):
        return {
            "MenuText": "Nuovo Muro",
            "ToolTip": "Disegna un muro in muratura (Arch.Wall)",
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

        if HAS_ARCH:
            # Crea linea base
            p1 = FreeCAD.Vector(0, 0, 0)
            p2 = FreeCAD.Vector(length, 0, 0)
            baseline = Draft.makeLine(p1, p2)

            # Crea muro Arch
            wall = Arch.makeWall(baseline, width=thickness, height=height)
            wall.Label = f"Muro{len([o for o in doc.Objects if 'Muro' in o.Label])+1:03d}"

            # Nascondi baseline
            if baseline.ViewObject:
                baseline.ViewObject.Visibility = False

            # Proprietà NTC
            add_ntc_properties(wall, "Wall")
            if hasattr(wall, 'Floor'):
                wall.Floor = 0

            # Colore mattone
            if wall.ViewObject:
                wall.ViewObject.ShapeColor = (0.8, 0.4, 0.2)

        else:
            # Fallback Part
            shape = Part.makeBox(length, thickness, height)
            count = len([o for o in doc.Objects if o.Name.startswith("Muro")])
            wall = doc.addObject("Part::Feature", f"Muro{count+1:03d}")
            wall.Shape = shape

            wall.addProperty("App::PropertyFloat", "Length", "Geometry", "Lunghezza (m)")
            wall.addProperty("App::PropertyFloat", "Thickness", "Geometry", "Spessore (m)")
            wall.addProperty("App::PropertyFloat", "Height", "Geometry", "Altezza (m)")
            wall.addProperty("App::PropertyInteger", "Floor", "Geometry", "Piano")

            wall.Length = length / 1000
            wall.Thickness = thickness / 1000
            wall.Height = height / 1000
            wall.Floor = 0

            if wall.ViewObject:
                wall.ViewObject.ShapeColor = (0.8, 0.4, 0.2)

        doc.recompute()
        FreeCADGui.Selection.clearSelection()
        FreeCADGui.Selection.addSelection(wall)

    def IsActive(self):
        return True


class CmdNewOpening:
    """Crea un'apertura usando Arch.makeWindow."""

    def GetResources(self):
        return {
            "MenuText": "Nuova Apertura",
            "ToolTip": "Aggiungi finestra o porta (Arch.Window)",
            "Pixmap": "Arch_Window"
        }

    def Activated(self):
        sel = FreeCADGui.Selection.getSelection()
        if not sel:
            FreeCAD.Console.PrintWarning("Seleziona un muro prima\n")
            return

        wall = sel[0]
        doc = FreeCAD.ActiveDocument

        # Parametri finestra default
        width = 1200  # mm
        height = 1400  # mm
        sill = 900  # mm

        if HAS_ARCH:
            # Usa Arch.makeWindowPreset
            try:
                window = Arch.makeWindowPreset(
                    "Simple door",
                    width=width,
                    height=height,
                    h1=100, h2=100, h3=100,
                    w1=100, w2=100,
                    o1=0, o2=0
                )

                # Posiziona nel muro
                if hasattr(wall, 'Shape'):
                    bb = wall.Shape.BoundBox
                    x_pos = bb.XMin + (bb.XLength - width) / 2
                    y_pos = bb.YMin + bb.YLength / 2
                    z_pos = bb.ZMin + sill

                    window.Placement.Base = FreeCAD.Vector(x_pos, y_pos, z_pos)

                # Collega al muro
                window.Hosts = [wall]

                window.Label = f"Apertura{len([o for o in doc.Objects if 'Apertura' in o.Label])+1:03d}"

            except Exception as e:
                FreeCAD.Console.PrintError(f"Errore Arch.Window: {e}\n")
                # Fallback: boolean cut
                self._fallback_opening(wall, width, height, sill)
        else:
            self._fallback_opening(wall, width, height, sill)

        doc.recompute()

    def _fallback_opening(self, wall, width, height, sill):
        """Crea apertura con boolean cut."""
        doc = FreeCAD.ActiveDocument
        bb = wall.Shape.BoundBox

        opening_shape = Part.makeBox(width, bb.YLength + 100, height)
        x_offset = (bb.XLength - width) / 2
        opening_shape.translate(FreeCAD.Vector(bb.XMin + x_offset, bb.YMin - 50, bb.ZMin + sill))

        new_shape = wall.Shape.cut(opening_shape)
        wall.Shape = new_shape

        FreeCAD.Console.PrintMessage(f"Apertura aggiunta a {wall.Label}\n")

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class CmdNewColumn:
    """Crea un pilastro usando Arch.makeStructure."""

    def GetResources(self):
        return {
            "MenuText": "Nuovo Pilastro",
            "ToolTip": "Inserisci un pilastro (Arch.Structure)",
            "Pixmap": "Arch_Structure"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            doc = FreeCAD.newDocument("Muratura")

        width = 300  # mm
        depth = 300  # mm
        height = get_floor_height()

        if HAS_ARCH:
            column = Arch.makeStructure(length=width, width=depth, height=height)
            column.Label = f"Pilastro{len([o for o in doc.Objects if 'Pilastro' in o.Label])+1:03d}"

            # Imposta come pilastro
            if hasattr(column, 'Role'):
                column.Role = "Column"

            add_ntc_properties(column, "Column")

            if column.ViewObject:
                column.ViewObject.ShapeColor = (0.6, 0.6, 0.6)
        else:
            shape = Part.makeBox(width, depth, height)
            count = len([o for o in doc.Objects if o.Name.startswith("Pilastro")])
            column = doc.addObject("Part::Feature", f"Pilastro{count+1:03d}")
            column.Shape = shape

            if column.ViewObject:
                column.ViewObject.ShapeColor = (0.6, 0.6, 0.6)

        doc.recompute()

    def IsActive(self):
        return True


class CmdNewBeam:
    """Crea una trave usando Arch.makeStructure."""

    def GetResources(self):
        return {
            "MenuText": "Nuova Trave",
            "ToolTip": "Inserisci una trave (Arch.Structure)",
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

        if HAS_ARCH:
            # Crea linea base per la trave
            p1 = FreeCAD.Vector(0, 0, z_offset)
            p2 = FreeCAD.Vector(length, 0, z_offset)
            baseline = Draft.makeLine(p1, p2)

            beam = Arch.makeStructure(baseline, width=width, height=height)
            beam.Label = f"Trave{len([o for o in doc.Objects if 'Trave' in o.Label])+1:03d}"

            if hasattr(beam, 'Role'):
                beam.Role = "Beam"

            # Nascondi baseline
            if baseline.ViewObject:
                baseline.ViewObject.Visibility = False

            add_ntc_properties(beam, "Beam")

            if beam.ViewObject:
                beam.ViewObject.ShapeColor = (0.5, 0.5, 0.5)
        else:
            shape = Part.makeBox(length, width, height)
            shape.translate(FreeCAD.Vector(0, 0, z_offset))

            count = len([o for o in doc.Objects if o.Name.startswith("Trave")])
            beam = doc.addObject("Part::Feature", f"Trave{count+1:03d}")
            beam.Shape = shape

            if beam.ViewObject:
                beam.ViewObject.ShapeColor = (0.5, 0.5, 0.5)

        doc.recompute()

    def IsActive(self):
        return True


class CmdNewSlab:
    """Crea un solaio usando Arch.makeFloor/Structure."""

    def GetResources(self):
        return {
            "MenuText": "Nuovo Solaio",
            "ToolTip": "Inserisci un solaio (Arch.Structure/Floor)",
            "Pixmap": "Arch_Floor"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            doc = FreeCAD.newDocument("Muratura")

        # Trova bounding box muri
        walls = [o for o in doc.Objects if 'Muro' in o.Label or o.Name.startswith("Muro")]

        if walls:
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            floor = 0

            for wall in walls:
                if hasattr(wall, 'Shape') and wall.Shape:
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

        if HAS_ARCH:
            # Crea rettangolo base
            rect = Draft.makeRectangle(width, depth)
            rect.Placement.Base = FreeCAD.Vector(min_x, min_y, z_offset)

            slab = Arch.makeStructure(rect, height=thickness)
            slab.Label = f"Solaio{floor+1:03d}"

            if hasattr(slab, 'Role'):
                slab.Role = "Slab"

            # Nascondi rettangolo
            if rect.ViewObject:
                rect.ViewObject.Visibility = False

            add_ntc_properties(slab, "Slab")

            if slab.ViewObject:
                slab.ViewObject.ShapeColor = (0.9, 0.9, 0.8)
                slab.ViewObject.Transparency = 50
        else:
            shape = Part.makeBox(width, depth, thickness)
            shape.translate(FreeCAD.Vector(min_x, min_y, z_offset))

            slab = doc.addObject("Part::Feature", f"Solaio{floor+1:03d}")
            slab.Shape = shape

            if slab.ViewObject:
                slab.ViewObject.ShapeColor = (0.9, 0.9, 0.8)
                slab.ViewObject.Transparency = 50

        doc.recompute()

    def IsActive(self):
        return True


class CmdNewStair:
    """Crea una scala usando Arch.makeStairs."""

    def GetResources(self):
        return {
            "MenuText": "Nuova Scala",
            "ToolTip": "Inserisci una scala (Arch.Stairs)",
            "Pixmap": "Arch_Stairs"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            doc = FreeCAD.newDocument("Muratura")

        width = 1200  # mm
        height = get_floor_height()
        num_steps = int(height / 170)

        if HAS_ARCH:
            try:
                stair = Arch.makeStairs(
                    width=width,
                    height=height,
                    steps=num_steps
                )
                stair.Label = f"Scala{len([o for o in doc.Objects if 'Scala' in o.Label])+1:03d}"

                if stair.ViewObject:
                    stair.ViewObject.ShapeColor = (0.7, 0.7, 0.7)
            except Exception as e:
                FreeCAD.Console.PrintWarning(f"Arch.Stairs non disponibile: {e}\n")
                self._fallback_stair(doc, width, height, num_steps)
        else:
            self._fallback_stair(doc, width, height, num_steps)

        doc.recompute()

    def _fallback_stair(self, doc, width, height, num_steps):
        """Crea scala con Part."""
        step_height = height / num_steps
        step_depth = 280

        shapes = []
        for i in range(num_steps):
            step = Part.makeBox(width, step_depth, step_height * (i + 1))
            step.translate(FreeCAD.Vector(0, i * step_depth, 0))
            shapes.append(step)

        compound = Part.makeCompound(shapes)
        count = len([o for o in doc.Objects if o.Name.startswith("Scala")])
        stair = doc.addObject("Part::Feature", f"Scala{count+1:03d}")
        stair.Shape = compound

        if stair.ViewObject:
            stair.ViewObject.ShapeColor = (0.7, 0.7, 0.7)

    def IsActive(self):
        return True


class CmdNewRoof:
    """Crea una copertura usando Arch.makeRoof."""

    def GetResources(self):
        return {
            "MenuText": "Nuova Copertura",
            "ToolTip": "Inserisci una copertura a falde (Arch.Roof)",
            "Pixmap": "Arch_Roof"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            doc = FreeCAD.newDocument("Muratura")

        # Trova bbox edificio
        walls = [o for o in doc.Objects if 'Muro' in o.Label or o.Name.startswith("Muro")]

        if walls:
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            max_z = 0

            for wall in walls:
                if hasattr(wall, 'Shape') and wall.Shape:
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

        overhang = 500
        ridge_height = 2000

        if HAS_ARCH:
            try:
                # Crea wire perimetrale
                points = [
                    FreeCAD.Vector(min_x - overhang, min_y - overhang, max_z),
                    FreeCAD.Vector(max_x + overhang, min_y - overhang, max_z),
                    FreeCAD.Vector(max_x + overhang, max_y + overhang, max_z),
                    FreeCAD.Vector(min_x - overhang, max_y + overhang, max_z),
                    FreeCAD.Vector(min_x - overhang, min_y - overhang, max_z),
                ]
                wire = Draft.makeWire(points, closed=True)

                roof = Arch.makeRoof(wire, angles=[25, 25, 25, 25])
                roof.Label = "Copertura001"

                if wire.ViewObject:
                    wire.ViewObject.Visibility = False

                if roof.ViewObject:
                    roof.ViewObject.ShapeColor = (0.6, 0.3, 0.2)

            except Exception as e:
                FreeCAD.Console.PrintWarning(f"Arch.Roof non disponibile: {e}\n")
                self._fallback_roof(doc, min_x, min_y, max_x, max_y, max_z, overhang, ridge_height)
        else:
            self._fallback_roof(doc, min_x, min_y, max_x, max_y, max_z, overhang, ridge_height)

        doc.recompute()

    def _fallback_roof(self, doc, min_x, min_y, max_x, max_y, max_z, overhang, ridge_height):
        """Crea copertura con Part."""
        mid_y = (min_y + max_y) / 2

        p1 = FreeCAD.Vector(min_x - overhang, min_y - overhang, max_z)
        p2 = FreeCAD.Vector(max_x + overhang, min_y - overhang, max_z)
        p3 = FreeCAD.Vector(max_x + overhang, mid_y, max_z + ridge_height)
        p4 = FreeCAD.Vector(min_x - overhang, mid_y, max_z + ridge_height)
        p5 = FreeCAD.Vector(min_x - overhang, max_y + overhang, max_z)
        p6 = FreeCAD.Vector(max_x + overhang, max_y + overhang, max_z)

        face1 = Part.Face(Part.makePolygon([p1, p2, p3, p4, p1]))
        face2 = Part.Face(Part.makePolygon([p4, p3, p6, p5, p4]))

        thickness = 200
        solid1 = face1.extrude(FreeCAD.Vector(0, 0, thickness))
        solid2 = face2.extrude(FreeCAD.Vector(0, 0, thickness))

        compound = Part.makeCompound([solid1, solid2])
        roof = doc.addObject("Part::Feature", "Copertura001")
        roof.Shape = compound

        if roof.ViewObject:
            roof.ViewObject.ShapeColor = (0.6, 0.3, 0.2)

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
