#!/usr/bin/env python3
"""Test Muratura 3.0 - Verifica completa"""
import sys
import os

# Setup
sys.path.insert(0, r'D:\muratura3')
sys.path.insert(0, r'D:\muratura3\muratura')

FREECAD_DIR = r'D:\muratura3\freecad'
sys.path.insert(0, os.path.join(FREECAD_DIR, 'bin'))
sys.path.insert(0, os.path.join(FREECAD_DIR, 'lib'))

if sys.platform.startswith('win'):
    os.add_dll_directory(os.path.join(FREECAD_DIR, 'bin'))
    os.add_dll_directory(os.path.join(FREECAD_DIR, 'lib'))

print("=" * 60)
print("MURATURA 3.0 - TEST COMPLETO")
print("=" * 60)

# Import MCP adapter
from mcp_server import get_adapter

adapter = get_adapter()

# 1. Progetto
print("\n--- 1. PROGETTO ---")
r = adapter.nuovo_progetto("Test Muratura 3", piani=2, altezza_piano=3.2)
print(f"nuovo_progetto: {r}")

# 2. Geometria piano terra
print("\n--- 2. GEOMETRIA PIANO TERRA ---")
r = adapter.rettangolo(0, 0, 12, 10, spessore=0.4, piano=0)
print(f"rettangolo: {r['muri']} muri")

r = adapter.muro(6, 0, 6, 10, spessore=0.3, piano=0)
print(f"muro divisorio: {r}")

# 3. Pilastri
print("\n--- 3. PILASTRI ---")
for x, y in [(0, 0), (12, 0), (12, 10), (0, 10)]:
    r = adapter.pilastro(x, y, larg=0.4, prof=0.4, piano=0)
    print(f"pilastro ({x},{y}): {r['nome']}")

# 4. Travi
print("\n--- 4. TRAVI ---")
r = adapter.trave(0, 0, 12, 0, piano=0)
print(f"trave sud: {r}")
r = adapter.trave(0, 10, 12, 10, piano=0)
print(f"trave nord: {r}")

# 5. Scala
print("\n--- 5. SCALA ---")
r = adapter.scala(1, 1, larg=1.2, piano=0)
print(f"scala: {r}")

# 6. Solaio
print("\n--- 6. SOLAIO ---")
r = adapter.solaio(piano=0)
print(f"solaio: {r}")

# 7. Fondazioni
print("\n--- 7. FONDAZIONI ---")
r = adapter.fondazioni(larg=0.6, alt=0.5)
print(f"fondazioni: {r}")

# 8. Copertura
print("\n--- 8. COPERTURA ---")
r = adapter.copertura(altezza_colmo=2.5, sporto=0.6)
print(f"copertura: {r}")

# 9. Stato
print("\n--- 9. STATO ---")
r = adapter.get_stato()
for k, v in r.items():
    if k != 'success':
        print(f"  {k}: {v}")

# 10. Analisi
print("\n--- 10. ANALISI POR ---")
r = adapter.analisi_por()
print(f"DCR_x: {r['DCR_x']:.3f} - {'OK' if r['verifica_x'] else 'NO'}")
print(f"DCR_y: {r['DCR_y']:.3f} - {'OK' if r['verifica_y'] else 'NO'}")

# 11. Export
print("\n--- 11. EXPORT ---")
r = adapter.esporta_step(r"D:\muratura3\test_output.step")
print(f"STEP: {r}")

r = adapter.salva(r"D:\muratura3\test_output.FCStd")
print(f"FCStd: {r}")

print("\n" + "=" * 60)
print("TEST COMPLETATO CON SUCCESSO!")
print("=" * 60)
