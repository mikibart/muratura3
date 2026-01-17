# -*- coding: utf-8 -*-
"""Comandi progetto Muratura"""

import FreeCAD
import FreeCADGui


class CmdNewProject:
    """Crea un nuovo progetto Muratura."""

    def GetResources(self):
        return {
            "MenuText": "Nuovo Progetto",
            "ToolTip": "Crea un nuovo progetto Muratura",
            "Pixmap": "document-new"
        }

    def Activated(self):
        doc = FreeCAD.newDocument("Muratura")

        # Aggiungi proprietà progetto
        doc.addProperty("App::PropertyString", "ProjectName", "Muratura", "Nome progetto")
        doc.addProperty("App::PropertyInteger", "NumFloors", "Muratura", "Numero piani")
        doc.addProperty("App::PropertyFloat", "FloorHeight", "Muratura", "Altezza interpiano (m)")
        doc.addProperty("App::PropertyString", "Comune", "Sismico", "Comune di riferimento")
        doc.addProperty("App::PropertyString", "CategoriaSuolo", "Sismico", "Categoria suolo")

        doc.ProjectName = "Nuovo Edificio"
        doc.NumFloors = 2
        doc.FloorHeight = 3.0
        doc.Comune = ""
        doc.CategoriaSuolo = "B"

        FreeCAD.Console.PrintMessage("Progetto Muratura creato\n")

    def IsActive(self):
        return True


# Registra comando
FreeCADGui.addCommand("Muratura_NewProject", CmdNewProject())
