# -*- coding: utf-8 -*-
"""
Muratura Workbench - Inizializzazione GUI

Registra il workbench Muratura in FreeCAD.
"""

import FreeCAD
import FreeCADGui


class MuraturaWorkbench(FreeCADGui.Workbench):
    """
    Workbench per analisi strutturale edifici in muratura.
    Conforme a NTC 2018.
    """

    MenuText = "Muratura"
    ToolTip = "Analisi strutturale edifici in muratura - NTC 2018"

    # Icona workbench (path relativo a resources/)
    Icon = """
        /* XPM */
        static char * muratura_xpm[] = {
        "16 16 4 1",
        " 	c None",
        ".	c #8B4513",
        "+	c #D2691E",
        "@	c #F4A460",
        "                ",
        " .............. ",
        " .@@.++.@@.++.@ ",
        " .............. ",
        " .++.@@.++.@@.+ ",
        " .............. ",
        " .@@.++.@@.++.@ ",
        " .............. ",
        " .++.@@.++.@@.+ ",
        " .............. ",
        " .@@.++.@@.++.@ ",
        " .............. ",
        " .++.@@.++.@@.+ ",
        " .............. ",
        " .@@.++.@@.++.@ ",
        "                "};
        """

    def Initialize(self):
        """Inizializza il workbench - carica comandi e toolbar."""

        # Import comandi
        from muratura.workbench.commands import (
            project, geometry, structural, analysis, export
        )

        # Lista comandi per categoria
        self.project_commands = [
            "Sketcher_NewSketch",  # Riusa Sketcher di FreeCAD
        ]

        self.geometry_commands = [
            "Muratura_NewWall",
            "Muratura_NewOpening",
            "Muratura_NewColumn",
            "Muratura_NewBeam",
            "Muratura_NewSlab",
            "Muratura_NewStair",
            "Muratura_NewRoof",
        ]

        self.structural_commands = [
            "Muratura_GenFoundations",
            "Muratura_GenRingBeams",
            "Muratura_SetMaterial",
            "Muratura_SetLoads",
        ]

        self.analysis_commands = [
            "Muratura_SetSeismic",
            "Muratura_RunPOR",
            "Muratura_RunSAM",
            "Muratura_ShowDCR",
        ]

        self.export_commands = [
            "Muratura_ExportReport",
            "Muratura_ExportIFC",
            "Muratura_ExportDXF",
        ]

        # Tutti i comandi
        all_commands = (
            self.geometry_commands +
            self.structural_commands +
            self.analysis_commands +
            self.export_commands
        )

        # Crea toolbar
        self.appendToolbar("Geometria", self.geometry_commands)
        self.appendToolbar("Struttura", self.structural_commands)
        self.appendToolbar("Analisi", self.analysis_commands)
        self.appendToolbar("Export", self.export_commands)

        # Crea menu
        self.appendMenu("Muratura", all_commands)
        self.appendMenu(["Muratura", "Geometria"], self.geometry_commands)
        self.appendMenu(["Muratura", "Struttura"], self.structural_commands)
        self.appendMenu(["Muratura", "Analisi"], self.analysis_commands)
        self.appendMenu(["Muratura", "Export"], self.export_commands)

    def Activated(self):
        """Chiamato quando il workbench viene attivato."""
        FreeCAD.Console.PrintMessage("Muratura Workbench attivato\n")

        # Imposta vista di default
        if FreeCAD.ActiveDocument is None:
            FreeCAD.newDocument("Muratura")

    def Deactivated(self):
        """Chiamato quando si esce dal workbench."""
        pass

    def ContextMenu(self, recipient):
        """Menu contestuale."""
        self.appendContextMenu("Muratura", self.geometry_commands[:3])

    def GetClassName(self):
        return "Gui::PythonWorkbench"


# Registra il workbench
FreeCADGui.addWorkbench(MuraturaWorkbench())
