# -*- coding: utf-8 -*-
"""Comandi analisi Muratura - NTC 2018"""

import FreeCAD
import FreeCADGui
import json


class CmdSetSeismic:
    """Imposta parametri sismici del sito."""

    def GetResources(self):
        return {
            "MenuText": "Parametri Sismici",
            "ToolTip": "Imposta comune e parametri sismici NTC 2018",
            "Pixmap": "Arch_Site"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            return

        # Aggiungi proprietà sismiche
        props = [
            ("App::PropertyString", "Comune", "Comune di riferimento"),
            ("App::PropertyString", "CategoriaSuolo", "Categoria suolo (A-E)"),
            ("App::PropertyInteger", "ClasseUso", "Classe d'uso (1-4)"),
            ("App::PropertyInteger", "VitaNominale", "Vita nominale (anni)"),
            ("App::PropertyFloat", "ag", "Accelerazione (g)"),
            ("App::PropertyFloat", "F0", "Fattore amplificazione"),
            ("App::PropertyFloat", "Tc_star", "Periodo Tc* (s)"),
        ]

        for prop_type, prop_name, prop_doc in props:
            if not hasattr(doc, prop_name):
                doc.addProperty(prop_type, prop_name, "Sismico", prop_doc)

        # Valori default (zona sismica media)
        doc.Comune = "Roma"
        doc.CategoriaSuolo = "B"
        doc.ClasseUso = 2
        doc.VitaNominale = 50
        doc.ag = 0.15
        doc.F0 = 2.4
        doc.Tc_star = 0.28

        FreeCAD.Console.PrintMessage("Parametri sismici impostati\n")

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class CmdRunPOR:
    """Esegue analisi POR."""

    def GetResources(self):
        return {
            "MenuText": "Analisi POR",
            "ToolTip": "Esegue analisi sismica con metodo POR",
            "Pixmap": "Arch_Check"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            return

        walls = [o for o in doc.Objects if o.Name.startswith("Muro")]
        if not walls:
            FreeCAD.Console.PrintWarning("Nessun muro nel progetto\n")
            return

        # Calcola area resistente per direzione
        area_x = 0.0
        area_y = 0.0

        for wall in walls:
            bbox = wall.Shape.BoundBox
            thickness = wall.Thickness if hasattr(wall, 'Thickness') else 0.3

            # Orientamento muro
            if bbox.XLength > bbox.YLength:
                # Muro in direzione X
                area_x += (bbox.XLength / 1000) * thickness
            else:
                # Muro in direzione Y
                area_y += (bbox.YLength / 1000) * thickness

        # Resistenza a taglio muratura (valore semplificato)
        # fvd = fvk / gamma_M = 0.2 / 2.0 = 0.1 N/mm² = 100 kN/m²
        fvd = 100  # kN/m²

        Vrd_x = area_x * fvd
        Vrd_y = area_y * fvd

        # Calcola taglio sismico (semplificato)
        total_weight = sum(w.Shape.Volume / 1e9 * 18 for w in walls)  # kN (18 kN/m³)
        ag = doc.ag if hasattr(doc, 'ag') else 0.15
        Ved = total_weight * ag * 2.5  # taglio alla base semplificato

        # DCR (Demand Capacity Ratio)
        dcr_x = Ved / Vrd_x if Vrd_x > 0 else float('inf')
        dcr_y = Ved / Vrd_y if Vrd_y > 0 else float('inf')

        # Salva risultati
        results = {
            "metodo": "POR",
            "area_x_m2": round(area_x, 2),
            "area_y_m2": round(area_y, 2),
            "Vrd_x_kN": round(Vrd_x, 1),
            "Vrd_y_kN": round(Vrd_y, 1),
            "Ved_kN": round(Ved, 1),
            "DCR_x": round(dcr_x, 3),
            "DCR_y": round(dcr_y, 3),
            "verificato_x": dcr_x <= 1.0,
            "verificato_y": dcr_y <= 1.0
        }

        if not hasattr(doc, "AnalysisResults"):
            doc.addProperty("App::PropertyString", "AnalysisResults", "Analisi", "Risultati JSON")
        doc.AnalysisResults = json.dumps(results, indent=2)

        # Output
        status_x = "OK" if dcr_x <= 1.0 else "NON VERIFICATO"
        status_y = "OK" if dcr_y <= 1.0 else "NON VERIFICATO"

        FreeCAD.Console.PrintMessage("\n=== RISULTATI ANALISI POR ===\n")
        FreeCAD.Console.PrintMessage(f"Area resistente X: {area_x:.2f} m²\n")
        FreeCAD.Console.PrintMessage(f"Area resistente Y: {area_y:.2f} m²\n")
        FreeCAD.Console.PrintMessage(f"Taglio resistente X: {Vrd_x:.1f} kN\n")
        FreeCAD.Console.PrintMessage(f"Taglio resistente Y: {Vrd_y:.1f} kN\n")
        FreeCAD.Console.PrintMessage(f"Taglio sollecitante: {Ved:.1f} kN\n")
        FreeCAD.Console.PrintMessage(f"DCR X: {dcr_x:.3f} - {status_x}\n")
        FreeCAD.Console.PrintMessage(f"DCR Y: {dcr_y:.3f} - {status_y}\n")
        FreeCAD.Console.PrintMessage("==============================\n")

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class CmdRunSAM:
    """Esegue analisi SAM."""

    def GetResources(self):
        return {
            "MenuText": "Analisi SAM",
            "ToolTip": "Esegue analisi con metodo SAM (Shear Analysis Method)",
            "Pixmap": "Arch_Check"
        }

    def Activated(self):
        FreeCAD.Console.PrintMessage("Analisi SAM: implementazione in corso...\n")
        # TODO: Implementare metodo SAM completo

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class CmdShowDCR:
    """Mostra mappa colori DCR."""

    def GetResources(self):
        return {
            "MenuText": "Mostra DCR",
            "ToolTip": "Colora gli elementi in base al DCR (verde=OK, rosso=critico)",
            "Pixmap": "Arch_ToggleSubs"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            return

        if not hasattr(doc, "AnalysisResults"):
            FreeCAD.Console.PrintWarning("Esegui prima un'analisi\n")
            return

        results = json.loads(doc.AnalysisResults)
        dcr_max = max(results.get("DCR_x", 1), results.get("DCR_y", 1))

        walls = [o for o in doc.Objects if o.Name.startswith("Muro")]

        for wall in walls:
            if hasattr(wall, "ViewObject"):
                # Colore in base a DCR
                if dcr_max <= 0.7:
                    color = (0.0, 0.8, 0.0)  # Verde
                elif dcr_max <= 1.0:
                    color = (1.0, 0.8, 0.0)  # Giallo
                else:
                    color = (1.0, 0.0, 0.0)  # Rosso

                wall.ViewObject.ShapeColor = color

        FreeCAD.Console.PrintMessage(f"Colorazione DCR applicata (DCR max = {dcr_max:.3f})\n")

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


# Registra comandi
FreeCADGui.addCommand("Muratura_SetSeismic", CmdSetSeismic())
FreeCADGui.addCommand("Muratura_RunPOR", CmdRunPOR())
FreeCADGui.addCommand("Muratura_RunSAM", CmdRunSAM())
FreeCADGui.addCommand("Muratura_ShowDCR", CmdShowDCR())
