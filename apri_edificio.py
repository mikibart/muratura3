# Script per aprire edificio in FreeCAD con vista corretta
import FreeCAD
import FreeCADGui

# Apri documento
doc = FreeCAD.open("D:/muratura3/edificio_mcp.FCStd")

# Rendi visibili tutti gli oggetti
for obj in doc.Objects:
    if hasattr(obj, "ViewObject") and obj.ViewObject:
        obj.ViewObject.Visibility = True

# Recompute
doc.recompute()

# Fit all
FreeCADGui.ActiveDocument.ActiveView.fitAll()

# Imposta vista isometrica
FreeCADGui.ActiveDocument.ActiveView.viewIsometric()

print(f"Aperto: {doc.Name} con {len(doc.Objects)} oggetti")
