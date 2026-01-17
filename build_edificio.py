# -*- coding: utf-8 -*-
"""
MURATURA 3.0 - Costruzione edificio completo
Usa Part.show() per visualizzazione corretta
"""
import FreeCAD
import FreeCADGui
import Part
import math

print("=" * 50)
print("MURATURA 3.0 - Costruzione Edificio")
print("=" * 50)

# Crea documento
doc = FreeCAD.newDocument("Edificio_Muratura")

# Parametri
FLOOR_H = 3000  # mm
WALL_COLOR = (0.76, 0.38, 0.18)  # Mattone
PILLAR_COLOR = (0.6, 0.6, 0.6)  # Grigio
SLAB_COLOR = (0.9, 0.88, 0.75)  # Beige
FOUND_COLOR = (0.5, 0.5, 0.5)  # Grigio scuro
ROOF_COLOR = (0.55, 0.27, 0.15)  # Tegola

n_muri = 0
n_pilastri = 0

def muro(x1, y1, x2, y2, sp=300, h=FLOOR_H, z=0):
    """Crea muro e lo mostra."""
    global n_muri
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    angle = math.atan2(dy, dx)

    shape = Part.makeBox(length, sp, h)
    shape.translate(FreeCAD.Vector(x1, y1 - sp/2, z))
    if angle != 0:
        shape.rotate(FreeCAD.Vector(x1, y1, z), FreeCAD.Vector(0,0,1), math.degrees(angle))

    n_muri += 1
    obj = Part.show(shape, f"Muro{n_muri:03d}")
    obj.ViewObject.ShapeColor = WALL_COLOR
    return obj

def pilastro(x, y, w=350, d=350, h=FLOOR_H, z=0):
    """Crea pilastro e lo mostra."""
    global n_pilastri
    shape = Part.makeBox(w, d, h)
    shape.translate(FreeCAD.Vector(x - w/2, y - d/2, z))

    n_pilastri += 1
    obj = Part.show(shape, f"Pilastro{n_pilastri:03d}")
    obj.ViewObject.ShapeColor = PILLAR_COLOR
    return obj

# ============ PIANO TERRA ============
print("\n[Piano Terra]")

# Muri perimetrali (10m x 8m)
print("  Muri perimetrali...")
muro(0, 0, 10000, 0, sp=400)        # Sud
muro(0, 8000, 10000, 8000, sp=400)  # Nord
muro(0, 0, 0, 8000, sp=400)         # Ovest
muro(10000, 0, 10000, 8000, sp=400) # Est

# Muri interni
print("  Muri interni...")
muro(5000, 0, 5000, 8000, sp=250)   # Divisorio centrale
muro(0, 4000, 5000, 4000, sp=150)   # Tramezza

# Pilastri angolari
print("  Pilastri...")
for x, y in [(0, 0), (10000, 0), (10000, 8000), (0, 8000)]:
    pilastro(x, y, w=400, d=400)

# Pilastri centrali
pilastro(5000, 0, w=350, d=350)
pilastro(5000, 8000, w=350, d=350)

# ============ SOLAIO ============
print("\n[Solaio]")
slab_shape = Part.makeBox(10800, 8800, 250)
slab_shape.translate(FreeCAD.Vector(-400, -400, FLOOR_H))
slab = Part.show(slab_shape, "Solaio001")
slab.ViewObject.ShapeColor = SLAB_COLOR
slab.ViewObject.Transparency = 20

# ============ FONDAZIONI ============
print("\n[Fondazioni]")
found_shapes = []

# Fondazione perimetrale
f1 = Part.makeBox(11200, 800, 500)
f1.translate(FreeCAD.Vector(-600, -600, -500))
found_shapes.append(f1)

f2 = Part.makeBox(11200, 800, 500)
f2.translate(FreeCAD.Vector(-600, 7800, -500))
found_shapes.append(f2)

f3 = Part.makeBox(800, 9200, 500)
f3.translate(FreeCAD.Vector(-600, -600, -500))
found_shapes.append(f3)

f4 = Part.makeBox(800, 9200, 500)
f4.translate(FreeCAD.Vector(9800, -600, -500))
found_shapes.append(f4)

# Fondazione centrale
f5 = Part.makeBox(600, 9200, 500)
f5.translate(FreeCAD.Vector(4700, -600, -500))
found_shapes.append(f5)

for i, fs in enumerate(found_shapes):
    f = Part.show(fs, f"Fondazione{i+1:03d}")
    f.ViewObject.ShapeColor = FOUND_COLOR

# ============ SCALA ============
print("\n[Scala]")
steps = []
n_steps = 17
step_h = FLOOR_H / n_steps
step_d = 280

for i in range(n_steps):
    step = Part.makeBox(1200, step_d, (i+1) * step_h)
    step.translate(FreeCAD.Vector(1000, 1000 + i * step_d, 0))
    steps.append(step)

scala_compound = Part.makeCompound(steps)
scala = Part.show(scala_compound, "Scala001")
scala.ViewObject.ShapeColor = (0.7, 0.7, 0.7)

# ============ COPERTURA ============
print("\n[Copertura]")

# Copertura a due falde
ridge_h = 2000
p1 = FreeCAD.Vector(-500, -500, FLOOR_H + 250)
p2 = FreeCAD.Vector(10500, -500, FLOOR_H + 250)
p3 = FreeCAD.Vector(10500, 4000, FLOOR_H + 250 + ridge_h)
p4 = FreeCAD.Vector(-500, 4000, FLOOR_H + 250 + ridge_h)
p5 = FreeCAD.Vector(-500, 8500, FLOOR_H + 250)
p6 = FreeCAD.Vector(10500, 8500, FLOOR_H + 250)

# Falda sud
face1 = Part.Face(Part.makePolygon([p1, p2, p3, p4, p1]))
roof1 = face1.extrude(FreeCAD.Vector(0, 0, 150))

# Falda nord
face2 = Part.Face(Part.makePolygon([p4, p3, p6, p5, p4]))
roof2 = face2.extrude(FreeCAD.Vector(0, 0, 150))

roof_compound = Part.makeCompound([roof1, roof2])
roof = Part.show(roof_compound, "Copertura001")
roof.ViewObject.ShapeColor = ROOF_COLOR

# ============ FINALIZZAZIONE ============
doc.recompute()

# Vista isometrica e fit
print("\n[Vista]")
FreeCADGui.activeDocument().activeView().viewIsometric()
FreeCADGui.activeDocument().activeView().fitAll()
FreeCADGui.updateGui()

# Riepilogo
print("\n" + "=" * 50)
print("EDIFICIO COMPLETATO!")
print("=" * 50)
print(f"  Muri:       {n_muri}")
print(f"  Pilastri:   {n_pilastri}")
print(f"  Solaio:     1")
print(f"  Fondazioni: {len(found_shapes)}")
print(f"  Scala:      1 ({n_steps} gradini)")
print(f"  Copertura:  1 (2 falde)")
print(f"  TOTALE:     {len(doc.Objects)} elementi")
print("=" * 50)
