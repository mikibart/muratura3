# -*- coding: utf-8 -*-
"""
Script per FreeCAD GUI - Crea edificio e visualizza
Eseguire con: FreeCAD.exe gui_build.py
"""

import FreeCAD
import FreeCADGui
import Part
import time

print("=" * 50)
print("MURATURA 3.0 - Creazione Edificio in GUI")
print("=" * 50)

# Crea documento
doc = FreeCAD.newDocument("Edificio_Muratura")
FreeCADGui.ActiveDocument = FreeCADGui.getDocument(doc.Name)

def crea_muro(x1, y1, x2, y2, spessore=300, altezza=3000, piano=0, colore=(0.8, 0.4, 0.2)):
    """Crea un muro."""
    import math
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    angle = math.atan2(dy, dx)

    shape = Part.makeBox(length, spessore, altezza)
    shape.translate(FreeCAD.Vector(x1, y1, piano * 3000))
    if angle != 0:
        shape.rotate(FreeCAD.Vector(x1, y1, piano * 3000), FreeCAD.Vector(0,0,1), math.degrees(angle))

    count = len([o for o in doc.Objects if o.Name.startswith("Muro")])
    wall = doc.addObject("Part::Feature", f"Muro{count+1:03d}")
    wall.Shape = shape
    wall.ViewObject.ShapeColor = colore
    return wall

def crea_pilastro(x, y, larg=300, prof=300, altezza=3000, piano=0):
    """Crea pilastro."""
    shape = Part.makeBox(larg, prof, altezza)
    shape.translate(FreeCAD.Vector(x - larg/2, y - prof/2, piano * 3000))

    count = len([o for o in doc.Objects if o.Name.startswith("Pilastro")])
    col = doc.addObject("Part::Feature", f"Pilastro{count+1:03d}")
    col.Shape = shape
    col.ViewObject.ShapeColor = (0.6, 0.6, 0.6)
    return col

def crea_solaio(x_min, y_min, x_max, y_max, z, spessore=250):
    """Crea solaio."""
    shape = Part.makeBox(x_max - x_min, y_max - y_min, spessore)
    shape.translate(FreeCAD.Vector(x_min, y_min, z))

    count = len([o for o in doc.Objects if o.Name.startswith("Solaio")])
    slab = doc.addObject("Part::Feature", f"Solaio{count+1:03d}")
    slab.Shape = shape
    slab.ViewObject.ShapeColor = (0.9, 0.9, 0.8)
    slab.ViewObject.Transparency = 30
    return slab

def crea_fondazione(x_min, y_min, x_max, y_max, altezza=500, extra=300):
    """Crea fondazione."""
    shape = Part.makeBox(x_max - x_min + extra*2, y_max - y_min + extra*2, altezza)
    shape.translate(FreeCAD.Vector(x_min - extra, y_min - extra, -altezza))

    count = len([o for o in doc.Objects if o.Name.startswith("Fond")])
    fond = doc.addObject("Part::Feature", f"Fondazione{count+1:03d}")
    fond.Shape = shape
    fond.ViewObject.ShapeColor = (0.5, 0.5, 0.5)
    return fond

# ============ COSTRUZIONE EDIFICIO ============

print("\nCostruzione edificio 10x8m, 2 piani...")

# Piano Terra - Muri perimetrali (mm)
print("- Muri perimetrali...")
crea_muro(0, 0, 10000, 0, spessore=400)      # Sud
crea_muro(0, 8000, 10000, 8000, spessore=400) # Nord
crea_muro(0, 0, 0, 8000, spessore=400)        # Ovest
crea_muro(10000, 0, 10000, 8000, spessore=400) # Est

# Muro divisorio interno
print("- Muro interno...")
crea_muro(5000, 0, 5000, 8000, spessore=300, colore=(0.9, 0.5, 0.3))

# Pilastri angolari
print("- Pilastri...")
for x, y in [(0, 0), (10000, 0), (10000, 8000), (0, 8000)]:
    crea_pilastro(x, y, larg=400, prof=400)

# Solaio piano terra
print("- Solaio...")
crea_solaio(-200, -200, 10200, 8200, 3000)

# Fondazioni
print("- Fondazioni...")
crea_fondazione(0, 0, 10000, 400)      # Sud
crea_fondazione(0, 7600, 10000, 8000)  # Nord
crea_fondazione(0, 0, 400, 8000)       # Ovest
crea_fondazione(9600, 0, 10000, 8000)  # Est
crea_fondazione(4700, 0, 5300, 8000)   # Centrale

# Copertura semplice
print("- Copertura...")
roof_shape = Part.makeBox(11000, 9000, 200)
roof_shape.translate(FreeCAD.Vector(-500, -500, 6000))
roof = doc.addObject("Part::Feature", "Copertura001")
roof.Shape = roof_shape
roof.ViewObject.ShapeColor = (0.6, 0.3, 0.2)

# ============ FINALIZZAZIONE ============

doc.recompute()

# Imposta vista
print("\nImpostazione vista...")
FreeCADGui.activeDocument().activeView().viewIsometric()
FreeCADGui.activeDocument().activeView().fitAll()

# Conta elementi
muri = len([o for o in doc.Objects if o.Name.startswith("Muro")])
pilastri = len([o for o in doc.Objects if o.Name.startswith("Pilastro")])
solai = len([o for o in doc.Objects if o.Name.startswith("Solaio")])
fond = len([o for o in doc.Objects if o.Name.startswith("Fond")])

print(f"\n{'='*50}")
print(f"EDIFICIO COMPLETATO!")
print(f"{'='*50}")
print(f"Muri: {muri}")
print(f"Pilastri: {pilastri}")
print(f"Solai: {solai}")
print(f"Fondazioni: {fond}")
print(f"Copertura: 1")
print(f"TOTALE: {len(doc.Objects)} elementi")
print(f"{'='*50}")
