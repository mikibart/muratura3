# -*- coding: utf-8 -*-
"""Comandi strutturali Muratura"""

import FreeCAD
import FreeCADGui
import Part


class CmdGenFoundations:
    """Genera fondazioni automatiche sotto i muri."""

    def GetResources(self):
        return {
            "MenuText": "Genera Fondazioni",
            "ToolTip": "Crea fondazioni continue sotto i muri del piano terra",
            "Pixmap": "Arch_Foundation"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            FreeCAD.Console.PrintWarning("Nessun documento attivo\n")
            return

        walls = [o for o in doc.Objects
                if o.Name.startswith("Muro") and
                hasattr(o, 'Floor') and o.Floor == 0]

        if not walls:
            FreeCAD.Console.PrintWarning("Nessun muro al piano terra\n")
            return

        width = 600  # mm - larghezza fondazione
        height = 500  # mm - altezza fondazione
        count = 0

        for wall in walls:
            bbox = wall.Shape.BoundBox

            # Fondazione più larga del muro
            fond_shape = Part.makeBox(
                bbox.XLength + width,
                bbox.YLength + width,
                height
            )
            fond_shape.translate(FreeCAD.Vector(
                bbox.XMin - width/2,
                bbox.YMin - width/2,
                -height
            ))

            fond = doc.addObject("Part::Feature", f"Fondazione{count+1:03d}")
            fond.Shape = fond_shape

            fond.addProperty("App::PropertyFloat", "Width", "Geometry", "Larghezza (m)")
            fond.addProperty("App::PropertyFloat", "Height", "Geometry", "Altezza (m)")
            fond.Width = width / 1000
            fond.Height = height / 1000

            if hasattr(fond, "ViewObject"):
                fond.ViewObject.ShapeColor = (0.5, 0.5, 0.5)

            count += 1

        doc.recompute()
        FreeCAD.Console.PrintMessage(f"Create {count} fondazioni\n")

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class CmdGenRingBeams:
    """Genera cordoli in sommità ai muri."""

    def GetResources(self):
        return {
            "MenuText": "Genera Cordoli",
            "ToolTip": "Crea cordoli in c.a. in sommità ai muri",
            "Pixmap": "Arch_Rebar"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            return

        walls = [o for o in doc.Objects if o.Name.startswith("Muro")]
        if not walls:
            FreeCAD.Console.PrintWarning("Nessun muro nel progetto\n")
            return

        height = 300  # mm
        count = 0

        for wall in walls:
            bbox = wall.Shape.BoundBox

            cordolo_shape = Part.makeBox(
                bbox.XLength,
                bbox.YLength,
                height
            )
            cordolo_shape.translate(FreeCAD.Vector(
                bbox.XMin,
                bbox.YMin,
                bbox.ZMax
            ))

            cordolo = doc.addObject("Part::Feature", f"Cordolo{count+1:03d}")
            cordolo.Shape = cordolo_shape

            cordolo.addProperty("App::PropertyFloat", "Height", "Geometry", "Altezza (m)")
            cordolo.addProperty("App::PropertyString", "WallRef", "Reference", "Muro di riferimento")
            cordolo.Height = height / 1000
            cordolo.WallRef = wall.Name

            if hasattr(cordolo, "ViewObject"):
                cordolo.ViewObject.ShapeColor = (0.7, 0.7, 0.7)

            count += 1

        doc.recompute()
        FreeCAD.Console.PrintMessage(f"Creati {count} cordoli\n")

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class CmdSetMaterial:
    """Imposta il materiale per gli elementi selezionati."""

    def GetResources(self):
        return {
            "MenuText": "Imposta Materiale",
            "ToolTip": "Assegna tipo muratura NTC 2018 agli elementi",
            "Pixmap": "Arch_Material"
        }

    def Activated(self):
        sel = FreeCADGui.Selection.getSelection()
        if not sel:
            FreeCAD.Console.PrintWarning("Seleziona elementi prima\n")
            return

        # Tipi muratura NTC 2018 Tabella C8.5.I
        material_types = [
            "Muratura in mattoni pieni e malta di calce",
            "Muratura in mattoni pieni e malta cementizia",
            "Muratura in mattoni semipieni",
            "Muratura in blocchi di calcestruzzo",
            "Muratura in pietra squadrata",
            "Muratura in pietra irregolare",
            "Muratura in blocchi di tufo"
        ]

        # Per ora assegna il primo tipo
        material = material_types[0]

        for obj in sel:
            if not hasattr(obj, "Material"):
                obj.addProperty("App::PropertyString", "Material", "NTC2018", "Tipo muratura")
            obj.Material = material

        FreeCAD.Console.PrintMessage(f"Materiale '{material}' assegnato a {len(sel)} elementi\n")

    def IsActive(self):
        return len(FreeCADGui.Selection.getSelection()) > 0


class CmdSetLoads:
    """Imposta i carichi sugli elementi."""

    def GetResources(self):
        return {
            "MenuText": "Imposta Carichi",
            "ToolTip": "Definisce carichi permanenti e variabili",
            "Pixmap": "Arch_AxisSystem"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            return

        # Aggiungi proprietà carichi al documento
        if not hasattr(doc, "G1k"):
            doc.addProperty("App::PropertyFloat", "G1k", "Carichi", "Peso proprio strutturale (kN/m²)")
            doc.addProperty("App::PropertyFloat", "G2k", "Carichi", "Permanenti portati (kN/m²)")
            doc.addProperty("App::PropertyFloat", "Qk", "Carichi", "Variabili (kN/m²)")
            doc.addProperty("App::PropertyFloat", "Qs", "Carichi", "Neve (kN/m²)")
            doc.addProperty("App::PropertyFloat", "Qw", "Carichi", "Vento (kN/m²)")

        # Valori default per abitazione
        doc.G1k = 3.0   # solaio latero-cemento
        doc.G2k = 2.0   # pavimento, tramezzi
        doc.Qk = 2.0    # categoria A - residenziale
        doc.Qs = 0.0    # da calcolare in base al comune
        doc.Qw = 0.0    # da calcolare

        FreeCAD.Console.PrintMessage("Carichi impostati (valori default residenziale)\n")

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


# Registra comandi
FreeCADGui.addCommand("Muratura_GenFoundations", CmdGenFoundations())
FreeCADGui.addCommand("Muratura_GenRingBeams", CmdGenRingBeams())
FreeCADGui.addCommand("Muratura_SetMaterial", CmdSetMaterial())
FreeCADGui.addCommand("Muratura_SetLoads", CmdSetLoads())
