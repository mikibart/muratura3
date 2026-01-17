#!/usr/bin/env python3
"""
Crea edificio con FreeCAD headless, poi apre GUI per visualizzare.
"""
import subprocess
import sys
import os
import time

# Setup
FREECAD_DIR = "D:/muratura3/freecad"
FREECAD_EXE = f"{FREECAD_DIR}/bin/FreeCAD.exe"
FREECADCMD_EXE = f"{FREECAD_DIR}/bin/FreeCADCmd.exe"
OUTPUT_FILE = "D:/muratura3/edificio_gui.FCStd"

# Script da eseguire in FreeCAD
BUILD_SCRIPT = '''
import FreeCAD
import Part
import math

print("Creazione edificio...")

doc = FreeCAD.newDocument("Edificio_GUI")

def muro(x1, y1, x2, y2, sp=300, h=3000, piano=0, colore=(0.8, 0.4, 0.2)):
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx*dx + dy*dy)
    angle = math.atan2(dy, dx)
    shape = Part.makeBox(length, sp, h)
    shape.translate(FreeCAD.Vector(x1, y1, piano * 3000))
    if angle != 0:
        shape.rotate(FreeCAD.Vector(x1, y1, piano * 3000), FreeCAD.Vector(0,0,1), math.degrees(angle))
    count = len([o for o in doc.Objects if o.Name.startswith("Muro")])
    wall = doc.addObject("Part::Feature", f"Muro{count+1:03d}")
    wall.Shape = shape
    return wall

def pilastro(x, y, w=350, d=350, h=3000, piano=0):
    shape = Part.makeBox(w, d, h)
    shape.translate(FreeCAD.Vector(x - w/2, y - d/2, piano * 3000))
    count = len([o for o in doc.Objects if o.Name.startswith("Pilastro")])
    col = doc.addObject("Part::Feature", f"Pilastro{count+1:03d}")
    col.Shape = shape
    return col

# Costruzione
print("- Muri perimetrali")
muro(0, 0, 10000, 0, sp=400)
muro(0, 8000, 10000, 8000, sp=400)
muro(0, 0, 0, 8000, sp=400)
muro(10000, 0, 10000, 8000, sp=400)

print("- Muro interno")
muro(5000, 0, 5000, 8000, sp=300)

print("- Pilastri")
for x, y in [(0, 0), (10000, 0), (10000, 8000), (0, 8000)]:
    pilastro(x, y)

print("- Solaio")
slab = Part.makeBox(10400, 8400, 250)
slab.translate(FreeCAD.Vector(-200, -200, 3000))
s = doc.addObject("Part::Feature", "Solaio001")
s.Shape = slab

print("- Fondazioni")
for i, wall in enumerate([o for o in doc.Objects if o.Name.startswith("Muro")]):
    bb = wall.Shape.BoundBox
    fshape = Part.makeBox(bb.XLength + 600, bb.YLength + 600, 500)
    fshape.translate(FreeCAD.Vector(bb.XMin - 300, bb.YMin - 300, -500))
    f = doc.addObject("Part::Feature", f"Fondazione{i+1:03d}")
    f.Shape = fshape

print("- Copertura")
roof = Part.makeBox(11000, 9000, 200)
roof.translate(FreeCAD.Vector(-500, -500, 6000))
r = doc.addObject("Part::Feature", "Copertura001")
r.Shape = roof

doc.recompute()
doc.saveAs("''' + OUTPUT_FILE.replace("\\", "/") + '''")

print(f"Salvato: {len(doc.Objects)} elementi")
'''

print("=" * 50)
print("MURATURA 3.0 - Build & View")
print("=" * 50)

# Scrivi script temporaneo
script_file = "D:/muratura3/_temp_build.py"
with open(script_file, 'w') as f:
    f.write(BUILD_SCRIPT)

# Esegui con FreeCADCmd
print("\n1. Creazione edificio (headless)...")
result = subprocess.run(
    [FREECADCMD_EXE, script_file],
    capture_output=True,
    text=True,
    timeout=60
)
print(result.stdout)
if result.returncode != 0:
    print(f"Errore: {result.stderr}")
    sys.exit(1)

# Verifica file
if os.path.exists(OUTPUT_FILE):
    size = os.path.getsize(OUTPUT_FILE)
    print(f"\n2. File creato: {OUTPUT_FILE} ({size/1024:.1f} KB)")
else:
    print("ERRORE: File non creato!")
    sys.exit(1)

# Apri con FreeCAD GUI
print("\n3. Apertura FreeCAD GUI...")
subprocess.Popen([FREECAD_EXE, OUTPUT_FILE])

print("\nFatto! FreeCAD si sta aprendo con l'edificio.")
print("Premi V poi F per fit all, oppure usa la rotella del mouse.")
