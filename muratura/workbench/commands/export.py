# -*- coding: utf-8 -*-
"""Comandi export Muratura"""

import FreeCAD
import FreeCADGui
import Part
import os
import json


class CmdExportReport:
    """Genera relazione tecnica."""

    def GetResources(self):
        return {
            "MenuText": "Genera Relazione",
            "ToolTip": "Esporta relazione tecnica in HTML/PDF",
            "Pixmap": "document-save"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            return

        # Raccogli dati
        walls = [o for o in doc.Objects if o.Name.startswith("Muro")]
        columns = [o for o in doc.Objects if o.Name.startswith("Pilastro")]
        slabs = [o for o in doc.Objects if o.Name.startswith("Solaio")]

        project_name = doc.ProjectName if hasattr(doc, 'ProjectName') else doc.Name
        comune = doc.Comune if hasattr(doc, 'Comune') else "N/D"

        # Risultati analisi
        results = {}
        if hasattr(doc, "AnalysisResults"):
            results = json.loads(doc.AnalysisResults)

        # Genera HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Relazione Tecnica - {project_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #8B4513; border-bottom: 2px solid #8B4513; }}
        h2 {{ color: #D2691E; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #8B4513; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .ok {{ color: green; font-weight: bold; }}
        .ko {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>RELAZIONE TECNICA STRUTTURALE</h1>
    <h2>Edificio in Muratura - NTC 2018</h2>

    <h2>1. DATI GENERALI</h2>
    <table>
        <tr><th>Parametro</th><th>Valore</th></tr>
        <tr><td>Nome progetto</td><td>{project_name}</td></tr>
        <tr><td>Comune</td><td>{comune}</td></tr>
        <tr><td>Numero muri</td><td>{len(walls)}</td></tr>
        <tr><td>Numero pilastri</td><td>{len(columns)}</td></tr>
        <tr><td>Numero solai</td><td>{len(slabs)}</td></tr>
    </table>

    <h2>2. PARAMETRI SISMICI</h2>
    <table>
        <tr><th>Parametro</th><th>Valore</th></tr>
        <tr><td>Categoria suolo</td><td>{getattr(doc, 'CategoriaSuolo', 'B')}</td></tr>
        <tr><td>Classe d'uso</td><td>{getattr(doc, 'ClasseUso', 2)}</td></tr>
        <tr><td>Vita nominale</td><td>{getattr(doc, 'VitaNominale', 50)} anni</td></tr>
        <tr><td>ag/g</td><td>{getattr(doc, 'ag', 0.15)}</td></tr>
    </table>

    <h2>3. RISULTATI ANALISI</h2>
"""

        if results:
            dcr_x = results.get('DCR_x', 0)
            dcr_y = results.get('DCR_y', 0)
            status_x = 'ok' if dcr_x <= 1.0 else 'ko'
            status_y = 'ok' if dcr_y <= 1.0 else 'ko'

            html += f"""
    <table>
        <tr><th>Parametro</th><th>Direzione X</th><th>Direzione Y</th></tr>
        <tr><td>Area resistente (m²)</td><td>{results.get('area_x_m2', 0)}</td><td>{results.get('area_y_m2', 0)}</td></tr>
        <tr><td>Taglio resistente (kN)</td><td>{results.get('Vrd_x_kN', 0)}</td><td>{results.get('Vrd_y_kN', 0)}</td></tr>
        <tr><td>Taglio sollecitante (kN)</td><td colspan="2">{results.get('Ved_kN', 0)}</td></tr>
        <tr><td>DCR</td><td class="{status_x}">{dcr_x:.3f}</td><td class="{status_y}">{dcr_y:.3f}</td></tr>
        <tr><td>Verifica</td><td class="{status_x}">{'VERIFICATO' if status_x == 'ok' else 'NON VERIFICATO'}</td><td class="{status_y}">{'VERIFICATO' if status_y == 'ok' else 'NON VERIFICATO'}</td></tr>
    </table>
"""
        else:
            html += "<p>Analisi non ancora eseguita.</p>"

        html += """
    <h2>4. CONCLUSIONI</h2>
    <p>La presente relazione è stata redatta in conformità alle NTC 2018 (D.M. 17/01/2018).</p>

    <hr>
    <p><em>Generato con Muratura - Software analisi strutturale</em></p>
</body>
</html>
"""

        # Salva file
        filepath = os.path.join(os.path.dirname(doc.FileName) if doc.FileName else ".", f"{project_name}_relazione.html")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        FreeCAD.Console.PrintMessage(f"Relazione salvata: {filepath}\n")

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class CmdExportIFC:
    """Esporta in formato IFC."""

    def GetResources(self):
        return {
            "MenuText": "Esporta IFC",
            "ToolTip": "Esporta modello in formato IFC (BIM)",
            "Pixmap": "IFC"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            return

        filepath = os.path.join(
            os.path.dirname(doc.FileName) if doc.FileName else ".",
            f"{doc.Name}.ifc"
        )

        try:
            import importIFC
            importIFC.export(doc.Objects, filepath)
            FreeCAD.Console.PrintMessage(f"IFC esportato: {filepath}\n")
        except ImportError:
            FreeCAD.Console.PrintWarning("Modulo IFC non disponibile\n")

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


class CmdExportDXF:
    """Esporta in formato DXF."""

    def GetResources(self):
        return {
            "MenuText": "Esporta DXF",
            "ToolTip": "Esporta piante in formato DXF (AutoCAD)",
            "Pixmap": "Draft_Draft"
        }

    def Activated(self):
        doc = FreeCAD.ActiveDocument
        if not doc:
            return

        filepath = os.path.join(
            os.path.dirname(doc.FileName) if doc.FileName else ".",
            f"{doc.Name}.dxf"
        )

        try:
            import importDXF
            importDXF.export(doc.Objects, filepath)
            FreeCAD.Console.PrintMessage(f"DXF esportato: {filepath}\n")
        except ImportError:
            # Fallback a Part export
            objects = [o for o in doc.Objects if hasattr(o, 'Shape')]
            Part.export(objects, filepath.replace('.dxf', '.step'))
            FreeCAD.Console.PrintMessage(f"STEP esportato (DXF non disponibile): {filepath.replace('.dxf', '.step')}\n")

    def IsActive(self):
        return FreeCAD.ActiveDocument is not None


# Registra comandi
FreeCADGui.addCommand("Muratura_ExportReport", CmdExportReport())
FreeCADGui.addCommand("Muratura_ExportIFC", CmdExportIFC())
FreeCADGui.addCommand("Muratura_ExportDXF", CmdExportDXF())
