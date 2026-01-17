# Test semplice - crea un cubo e lo mostra
import FreeCAD
import FreeCADGui
import Part

# Crea documento
doc = FreeCAD.newDocument("Test")

# Metodo 1: Part.show() - dovrebbe funzionare sempre
box = Part.makeBox(5000, 5000, 3000)
Part.show(box, "Cubo_Test")

# Recompute
doc.recompute()

# Fit all
try:
    FreeCADGui.activeDocument().activeView().viewIsometric()
    FreeCADGui.activeDocument().activeView().fitAll()
    FreeCADGui.updateGui()
except:
    pass

print(f"Creato documento con {len(doc.Objects)} oggetti")
print("Se non vedi il cubo, c'è un problema con OpenGL/driver grafici")
