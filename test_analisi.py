# -*- coding: utf-8 -*-
"""Test tutte le analisi MCP"""
import sys
sys.path.insert(0, 'D:/muratura3')
from mcp_gui_client import *
import json

def pretty(r):
    return json.dumps(r, indent=2, ensure_ascii=False)

print("=" * 60)
print("TEST ANALISI MURATURA 3.0")
print("=" * 60)

# Crea nuovo progetto
print("\n[1] Nuovo progetto...")
r = nuovo_progetto("Test_Analisi", piani=2, altezza_piano=3.0)
print(f"    {r}")

# Crea edificio semplice
print("\n[2] Creazione edificio...")
r = rettangolo(0, 0, 10, 8, spessore=0.4)
print(f"    Muri perimetrali: {r.get('muri', 'error')}")

r = muro(5, 0, 5, 8, spessore=0.3)
print(f"    Muro divisorio: {r.get('nome', 'error')}")

r = pilastro(0, 0, larg=0.4, prof=0.4)
print(f"    Pilastro 1: {r.get('nome', 'error')}")
r = pilastro(10, 0, larg=0.4, prof=0.4)
print(f"    Pilastro 2: {r.get('nome', 'error')}")
r = pilastro(10, 8, larg=0.4, prof=0.4)
print(f"    Pilastro 3: {r.get('nome', 'error')}")
r = pilastro(0, 8, larg=0.4, prof=0.4)
print(f"    Pilastro 4: {r.get('nome', 'error')}")

r = solaio()
print(f"    Solaio: {r.get('nome', 'error')}")

r = fit()
print("    Vista: OK")

# Stato
print("\n[3] Stato progetto...")
r = get_stato()
print(f"    Muri: {r['muri']}, Pilastri: {r['pilastri']}, Solai: {r['solai']}, Totale: {r['totale']}")

# ==================== ANALISI ====================
print("\n" + "=" * 60)
print("ANALISI STRUTTURALI")
print("=" * 60)

# 1. POR
print("\n[4] ANALISI POR...")
r = analisi_por()
if r.get('success'):
    print(f"    Area X: {r['area_resistente_x_m2']} m²")
    print(f"    Area Y: {r['area_resistente_y_m2']} m²")
    print(f"    DCR X: {r['DCR_x']} - {r['verifica_x']}")
    print(f"    DCR Y: {r['DCR_y']} - {r['verifica_y']}")
    print(f"    Colore: {r['colore']}")
else:
    print(f"    ERRORE: {r}")

# 2. SAM
print("\n[5] ANALISI SAM...")
r = analisi_sam()
if r.get('success'):
    print(f"    Muri analizzati: {r['n_muri']}")
    print(f"    K totale X: {r['K_tot_x']} kN/m")
    print(f"    K totale Y: {r['K_tot_y']} kN/m")
    print(f"    Ved: {r['Ved_kN']} kN")
    print(f"    Verificato: {r['verificato']}")
else:
    print(f"    ERRORE: {r}")

# 3. Carichi
print("\n[6] ANALISI CARICHI VERTICALI...")
r = analisi_carichi()
if r.get('success'):
    print(f"    qEd: {r['carichi']['qEd']} kN/m²")
    print(f"    Verificato: {r['verificato']}")
else:
    print(f"    ERRORE: {r}")

# 4. Pressoflessione
print("\n[7] ANALISI PRESSOFLESSIONE...")
r = analisi_pressoflessione()
if r.get('success'):
    print(f"    Verificato: {r['verificato']}")
else:
    print(f"    ERRORE: {r}")

# 5. Taglio
print("\n[8] ANALISI TAGLIO...")
r = analisi_taglio()
if r.get('success'):
    print(f"    fvd: {r['parametri']['fvd']} MPa")
    print(f"    Verificato: {r['verificato']}")
else:
    print(f"    ERRORE: {r}")

# 6. Ribaltamento
print("\n[9] ANALISI RIBALTAMENTO...")
r = analisi_ribaltamento()
if r.get('success'):
    print(f"    Verificato: {r['verificato']}")
else:
    print(f"    ERRORE: {r}")

# 7. Report completo
print("\n[10] REPORT COMPLETO...")
r = report_completo()
if r.get('success'):
    print(f"    Verifiche:")
    for k, v in r['verifiche'].items():
        status = "✓" if v else "✗"
        print(f"      {status} {k}")
    print(f"\n    TUTTO VERIFICATO: {r['tutto_verificato']}")
else:
    print(f"    ERRORE: {r}")

print("\n" + "=" * 60)
print("TEST COMPLETATO")
print("=" * 60)
