#!/usr/bin/env python3
"""
MURATURA 3.0 - Test Automatico Completo
Verifica tutte le funzionalità senza GUI
"""
import sys
import os

# Setup FreeCAD
sys.path.insert(0, 'D:/muratura3')
sys.path.insert(0, 'D:/muratura3/freecad/bin')
sys.path.insert(0, 'D:/muratura3/freecad/lib')
os.add_dll_directory('D:/muratura3/freecad/bin')
os.add_dll_directory('D:/muratura3/freecad/lib')

import FreeCAD
import Part

print("=" * 70)
print("MURATURA 3.0 - TEST AUTOMATICO")
print("=" * 70)

# Import adapter
from mcp_server import MuraturaAdapter

def test_completo():
    """Esegue test completo del sistema."""

    adapter = MuraturaAdapter()
    errori = []
    successi = 0

    # ========== TEST 1: PROGETTO ==========
    print("\n[TEST 1] Creazione progetto...")
    r = adapter.nuovo_progetto("Test_Auto", piani=2, altezza_piano=3.0)
    if r["success"]:
        print(f"  ✓ Progetto creato: {r['documento']}")
        successi += 1
    else:
        errori.append(f"Progetto: {r.get('error')}")

    # ========== TEST 2: MURI ==========
    print("\n[TEST 2] Creazione muri...")
    r = adapter.rettangolo(0, 0, 10, 8, spessore=0.3, piano=0)
    if r["success"] and r["muri"] == 4:
        print(f"  ✓ Rettangolo: {r['muri']} muri creati")
        successi += 1
    else:
        errori.append("Rettangolo fallito")

    r = adapter.muro(5, 0, 5, 8, spessore=0.25, piano=0)
    if r["success"]:
        print(f"  ✓ Muro interno: {r['nome']}")
        successi += 1
    else:
        errori.append(f"Muro: {r.get('error')}")

    # ========== TEST 3: PILASTRI ==========
    print("\n[TEST 3] Creazione pilastri...")
    pilastri_ok = 0
    for x, y in [(0,0), (10,0), (10,8), (0,8)]:
        r = adapter.pilastro(x, y, larg=0.3, prof=0.3, piano=0)
        if r["success"]:
            pilastri_ok += 1
    if pilastri_ok == 4:
        print(f"  ✓ Pilastri: {pilastri_ok} creati")
        successi += 1
    else:
        errori.append(f"Pilastri: solo {pilastri_ok}/4")

    # ========== TEST 4: TRAVI ==========
    print("\n[TEST 4] Creazione travi...")
    r1 = adapter.trave(0, 0, 10, 0, piano=0)
    r2 = adapter.trave(0, 8, 10, 8, piano=0)
    if r1["success"] and r2["success"]:
        print(f"  ✓ Travi: {r1['nome']}, {r2['nome']}")
        successi += 1
    else:
        errori.append("Travi fallite")

    # ========== TEST 5: SCALA ==========
    print("\n[TEST 5] Creazione scala...")
    r = adapter.scala(1, 1, larg=1.2, piano=0)
    if r["success"]:
        print(f"  ✓ Scala: {r['nome']} ({r['gradini']} gradini)")
        successi += 1
    else:
        errori.append(f"Scala: {r.get('error')}")

    # ========== TEST 6: SOLAIO ==========
    print("\n[TEST 6] Generazione solaio...")
    r = adapter.solaio(piano=0)
    if r["success"]:
        print(f"  ✓ Solaio: {r['nome']} (area: {r['area']:.1f} m²)")
        successi += 1
    else:
        errori.append(f"Solaio: {r.get('error')}")

    # ========== TEST 7: FONDAZIONI ==========
    print("\n[TEST 7] Generazione fondazioni...")
    r = adapter.fondazioni(larg=0.5, alt=0.4)
    if r["success"]:
        print(f"  ✓ Fondazioni: {r['fondazioni']} create")
        successi += 1
    else:
        errori.append(f"Fondazioni: {r.get('error')}")

    # ========== TEST 8: COPERTURA ==========
    print("\n[TEST 8] Creazione copertura...")
    r = adapter.copertura(altezza_colmo=2.0, sporto=0.5)
    if r["success"]:
        print(f"  ✓ Copertura: {r['nome']}")
        successi += 1
    else:
        errori.append(f"Copertura: {r.get('error')}")

    # ========== TEST 9: STATO ==========
    print("\n[TEST 9] Verifica stato...")
    r = adapter.get_stato()
    if r["success"] and r["totale"] > 15:
        print(f"  ✓ Stato: {r['totale']} elementi totali")
        print(f"    - Muri: {r['muroi']}")
        print(f"    - Pilastri: {r['pilastroi']}")
        print(f"    - Travi: {r['travei']}")
        print(f"    - Solai: {r['solaioi']}")
        print(f"    - Scale: {r['scale']}")
        print(f"    - Coperture: {r['coperturai']}")
        print(f"    - Fondazioni: {r['fondazionei']}")
        successi += 1
    else:
        errori.append("Stato non corretto")

    # ========== TEST 10: ANALISI POR ==========
    print("\n[TEST 10] Analisi strutturale POR...")
    r = adapter.analisi_por()
    if r["success"]:
        print(f"  ✓ Analisi POR completata:")
        print(f"    - Area resistente X: {r['area_x']:.2f} m²")
        print(f"    - Area resistente Y: {r['area_y']:.2f} m²")
        print(f"    - Vrd X: {r['Vrd_x_kN']:.0f} kN")
        print(f"    - Vrd Y: {r['Vrd_y_kN']:.0f} kN")
        print(f"    - Ved: {r['Ved_kN']:.0f} kN")
        print(f"    - DCR X: {r['DCR_x']:.3f} {'✓' if r['verifica_x'] else '✗'}")
        print(f"    - DCR Y: {r['DCR_y']:.3f} {'✓' if r['verifica_y'] else '✗'}")
        successi += 1
    else:
        errori.append(f"Analisi: {r.get('error')}")

    # ========== TEST 11: EXPORT STEP ==========
    print("\n[TEST 11] Export STEP...")
    r = adapter.esporta_step("D:/muratura3/test_auto.step")
    if r["success"]:
        size = os.path.getsize("D:/muratura3/test_auto.step")
        print(f"  ✓ STEP: {r['file']} ({size/1024:.1f} KB, {r['oggetti']} oggetti)")
        successi += 1
    else:
        errori.append(f"Export STEP: {r.get('error')}")

    # ========== TEST 12: SALVATAGGIO ==========
    print("\n[TEST 12] Salvataggio progetto...")
    r = adapter.salva("D:/muratura3/test_auto.FCStd")
    if r["success"]:
        size = os.path.getsize("D:/muratura3/test_auto.FCStd")
        print(f"  ✓ FCStd: {r['file']} ({size/1024:.1f} KB)")
        successi += 1
    else:
        errori.append(f"Salvataggio: {r.get('error')}")

    # ========== TEST 13: ELIMINA ELEMENTO ==========
    print("\n[TEST 13] Eliminazione elemento...")
    r = adapter.elimina("Fondazione001")
    if r["success"]:
        print(f"  ✓ Eliminato: {r['eliminato']}")
        successi += 1
    else:
        errori.append(f"Elimina: {r.get('error')}")

    # ========== TEST 14: VERIFICA GEOMETRIE ==========
    print("\n[TEST 14] Verifica integrità geometrie...")
    doc = adapter.doc
    invalid = 0
    for obj in doc.Objects:
        if hasattr(obj, 'Shape'):
            if not obj.Shape.isValid():
                invalid += 1
                print(f"  ✗ {obj.Name}: geometria non valida")
    if invalid == 0:
        print(f"  ✓ Tutte le {len(doc.Objects)} geometrie sono valide")
        successi += 1
    else:
        errori.append(f"{invalid} geometrie non valide")

    # ========== TEST 15: BOUNDING BOX ==========
    print("\n[TEST 15] Verifica dimensioni edificio...")
    shapes = [obj.Shape for obj in doc.Objects if hasattr(obj, 'Shape')]
    if shapes:
        compound = Part.makeCompound(shapes)
        bb = compound.BoundBox
        print(f"  ✓ Dimensioni edificio:")
        print(f"    - Larghezza (X): {bb.XLength/1000:.2f} m")
        print(f"    - Profondità (Y): {bb.YLength/1000:.2f} m")
        print(f"    - Altezza (Z): {bb.ZLength/1000:.2f} m")
        print(f"    - Volume totale: {compound.Volume/1e9:.2f} m³")
        successi += 1
    else:
        errori.append("Nessuna geometria")

    # ========== RISULTATO FINALE ==========
    print("\n" + "=" * 70)
    print("RISULTATO TEST")
    print("=" * 70)
    print(f"Test superati: {successi}/15")
    print(f"Test falliti: {len(errori)}/15")

    if errori:
        print("\nErrori:")
        for e in errori:
            print(f"  ✗ {e}")

    if successi == 15:
        print("\n" + "★" * 35)
        print("  TUTTI I TEST SUPERATI!")
        print("  MURATURA 3.0 FUNZIONA CORRETTAMENTE")
        print("★" * 35)
        return True
    else:
        print(f"\n⚠ {len(errori)} test falliti")
        return False


if __name__ == "__main__":
    success = test_completo()
    sys.exit(0 if success else 1)
