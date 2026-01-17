# -*- coding: utf-8 -*-
"""
MCP GUI Server - Esegue comandi MCP in FreeCAD GUI
Legge comandi da file e li esegue con timer FreeCAD
"""
import FreeCAD
import FreeCADGui
import Part
import math
import json
import os

# File per comunicazione
CMD_FILE = "D:/muratura3/_mcp_command.json"
RESP_FILE = "D:/muratura3/_mcp_response.json"

# Stato
FLOOR_H = 3000
counters = {"muro": 0, "pilastro": 0, "trave": 0, "solaio": 0, "scala": 0, "fond": 0}

# Colori
COLORS = {
    "muro": (0.76, 0.38, 0.18),
    "pilastro": (0.6, 0.6, 0.6),
    "trave": (0.5, 0.5, 0.55),
    "solaio": (0.9, 0.88, 0.75),
    "fondazione": (0.5, 0.5, 0.5),
    "scala": (0.7, 0.7, 0.7),
    "copertura": (0.55, 0.27, 0.15),
}

def get_doc():
    """Ottiene o crea documento."""
    if FreeCAD.ActiveDocument is None:
        FreeCAD.newDocument("Muratura")
    return FreeCAD.ActiveDocument

# ========== COMANDI ==========

def cmd_nuovo_progetto(nome="Muratura", piani=2, altezza_piano=3.0, **kw):
    global FLOOR_H, counters
    FLOOR_H = int(altezza_piano * 1000)
    counters = {"muro": 0, "pilastro": 0, "trave": 0, "solaio": 0, "scala": 0, "fond": 0}
    doc = FreeCAD.newDocument(nome.replace(" ", "_"))
    return {"documento": doc.Name}

def cmd_muro(x1, y1, x2, y2, spessore=0.3, altezza=None, piano=0, **kw):
    doc = get_doc()
    x1_mm, y1_mm = x1 * 1000, y1 * 1000
    x2_mm, y2_mm = x2 * 1000, y2 * 1000
    sp_mm = spessore * 1000
    h_mm = (altezza * 1000) if altezza else FLOOR_H

    dx, dy = x2_mm - x1_mm, y2_mm - y1_mm
    length = math.sqrt(dx*dx + dy*dy)
    angle = math.atan2(dy, dx)

    shape = Part.makeBox(length, sp_mm, h_mm)
    shape.translate(FreeCAD.Vector(x1_mm, y1_mm - sp_mm/2, piano * FLOOR_H))
    if angle != 0:
        shape.rotate(FreeCAD.Vector(x1_mm, y1_mm, piano * FLOOR_H),
                    FreeCAD.Vector(0,0,1), math.degrees(angle))

    counters["muro"] += 1
    obj = Part.show(shape, f"Muro{counters['muro']:03d}")
    obj.ViewObject.ShapeColor = COLORS["muro"]
    doc.recompute()
    return {"nome": obj.Name, "lunghezza": length/1000}

def cmd_rettangolo(x, y, larg, prof, spessore=0.3, piano=0, **kw):
    r = []
    r.append(cmd_muro(x, y, x + larg, y, spessore, piano=piano))
    r.append(cmd_muro(x, y + prof, x + larg, y + prof, spessore, piano=piano))
    r.append(cmd_muro(x, y, x, y + prof, spessore, piano=piano))
    r.append(cmd_muro(x + larg, y, x + larg, y + prof, spessore, piano=piano))
    return {"muri": 4, "dettagli": r}

def cmd_pilastro(x, y, larg=0.3, prof=0.3, altezza=None, piano=0, **kw):
    doc = get_doc()
    w, d = larg * 1000, prof * 1000
    h = (altezza * 1000) if altezza else FLOOR_H

    shape = Part.makeBox(w, d, h)
    shape.translate(FreeCAD.Vector(x*1000 - w/2, y*1000 - d/2, piano * FLOOR_H))

    counters["pilastro"] += 1
    obj = Part.show(shape, f"Pilastro{counters['pilastro']:03d}")
    obj.ViewObject.ShapeColor = COLORS["pilastro"]
    doc.recompute()
    return {"nome": obj.Name}

def cmd_trave(x1, y1, x2, y2, larg=0.3, alt=0.5, piano=0, **kw):
    doc = get_doc()
    x1_mm, y1_mm = x1 * 1000, y1 * 1000
    x2_mm, y2_mm = x2 * 1000, y2 * 1000
    w_mm, h_mm = larg * 1000, alt * 1000

    dx, dy = x2_mm - x1_mm, y2_mm - y1_mm
    length = math.sqrt(dx*dx + dy*dy)
    angle = math.atan2(dy, dx)
    z = (piano + 1) * FLOOR_H - h_mm

    shape = Part.makeBox(length, w_mm, h_mm)
    shape.translate(FreeCAD.Vector(x1_mm, y1_mm - w_mm/2, z))
    if angle != 0:
        shape.rotate(FreeCAD.Vector(x1_mm, y1_mm, z), FreeCAD.Vector(0,0,1), math.degrees(angle))

    counters["trave"] += 1
    obj = Part.show(shape, f"Trave{counters['trave']:03d}")
    obj.ViewObject.ShapeColor = COLORS["trave"]
    doc.recompute()
    return {"nome": obj.Name, "lunghezza": length/1000}

def cmd_solaio(piano=0, spessore=0.25, **kw):
    doc = get_doc()
    walls = [o for o in doc.Objects if o.Name.startswith("Muro")]
    if not walls:
        return {"error": "Nessun muro"}

    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for w in walls:
        bb = w.Shape.BoundBox
        min_x, min_y = min(min_x, bb.XMin), min(min_y, bb.YMin)
        max_x, max_y = max(max_x, bb.XMax), max(max_y, bb.YMax)

    sp_mm = spessore * 1000
    shape = Part.makeBox(max_x - min_x + 400, max_y - min_y + 400, sp_mm)
    shape.translate(FreeCAD.Vector(min_x - 200, min_y - 200, (piano + 1) * FLOOR_H))

    counters["solaio"] += 1
    obj = Part.show(shape, f"Solaio{counters['solaio']:03d}")
    obj.ViewObject.ShapeColor = COLORS["solaio"]
    obj.ViewObject.Transparency = 20
    doc.recompute()
    return {"nome": obj.Name, "area": (max_x - min_x) * (max_y - min_y) / 1e6}

def cmd_fondazioni(larg=0.6, alt=0.5, **kw):
    doc = get_doc()
    walls = [o for o in doc.Objects if o.Name.startswith("Muro")]
    count = 0
    for wall in walls:
        bb = wall.Shape.BoundBox
        shape = Part.makeBox(bb.XLength + larg*1000, bb.YLength + larg*1000, alt*1000)
        shape.translate(FreeCAD.Vector(bb.XMin - larg*500, bb.YMin - larg*500, -alt*1000))

        counters["fond"] += 1
        obj = Part.show(shape, f"Fondazione{counters['fond']:03d}")
        obj.ViewObject.ShapeColor = COLORS["fondazione"]
        count += 1
    doc.recompute()
    return {"fondazioni": count}

def cmd_scala(x, y, larg=1.2, piano=0, **kw):
    doc = get_doc()
    w_mm = larg * 1000
    n_steps = int(FLOOR_H / 175)
    step_h = FLOOR_H / n_steps
    step_d = 280

    steps = []
    for i in range(n_steps):
        step = Part.makeBox(w_mm, step_d, (i+1) * step_h)
        step.translate(FreeCAD.Vector(x*1000, y*1000 + i * step_d, piano * FLOOR_H))
        steps.append(step)

    compound = Part.makeCompound(steps)
    counters["scala"] += 1
    obj = Part.show(compound, f"Scala{counters['scala']:03d}")
    obj.ViewObject.ShapeColor = COLORS["scala"]
    doc.recompute()
    return {"nome": obj.Name, "gradini": n_steps}

def cmd_copertura(altezza_colmo=2.0, sporto=0.5, **kw):
    doc = get_doc()
    walls = [o for o in doc.Objects if o.Name.startswith("Muro")]
    if not walls:
        return {"error": "Nessun muro"}

    min_x = min_y = float('inf')
    max_x = max_y = max_z = float('-inf')
    for w in walls:
        bb = w.Shape.BoundBox
        min_x, min_y = min(min_x, bb.XMin), min(min_y, bb.YMin)
        max_x, max_y = max(max_x, bb.XMax), max(max_y, bb.YMax)
        max_z = max(max_z, bb.ZMax)

    ov = sporto * 1000
    rh = altezza_colmo * 1000
    mid_y = (min_y + max_y) / 2

    p1 = FreeCAD.Vector(min_x - ov, min_y - ov, max_z)
    p2 = FreeCAD.Vector(max_x + ov, min_y - ov, max_z)
    p3 = FreeCAD.Vector(max_x + ov, mid_y, max_z + rh)
    p4 = FreeCAD.Vector(min_x - ov, mid_y, max_z + rh)
    p5 = FreeCAD.Vector(min_x - ov, max_y + ov, max_z)
    p6 = FreeCAD.Vector(max_x + ov, max_y + ov, max_z)

    face1 = Part.Face(Part.makePolygon([p1, p2, p3, p4, p1]))
    face2 = Part.Face(Part.makePolygon([p4, p3, p6, p5, p4]))
    roof1 = face1.extrude(FreeCAD.Vector(0, 0, 150))
    roof2 = face2.extrude(FreeCAD.Vector(0, 0, 150))

    compound = Part.makeCompound([roof1, roof2])
    obj = Part.show(compound, "Copertura001")
    obj.ViewObject.ShapeColor = COLORS["copertura"]
    doc.recompute()
    return {"nome": obj.Name}

def cmd_fit(**kw):
    FreeCADGui.activeDocument().activeView().viewIsometric()
    FreeCADGui.activeDocument().activeView().fitAll()
    return {}

def cmd_get_stato(**kw):
    doc = get_doc()
    return {
        "muri": len([o for o in doc.Objects if o.Name.startswith("Muro")]),
        "pilastri": len([o for o in doc.Objects if o.Name.startswith("Pilastro")]),
        "travi": len([o for o in doc.Objects if o.Name.startswith("Trave")]),
        "solai": len([o for o in doc.Objects if o.Name.startswith("Solaio")]),
        "scale": len([o for o in doc.Objects if o.Name.startswith("Scala")]),
        "fondazioni": len([o for o in doc.Objects if o.Name.startswith("Fond")]),
        "coperture": len([o for o in doc.Objects if o.Name.startswith("Copertura")]),
        "totale": len(doc.Objects)
    }

def cmd_salva(percorso, **kw):
    doc = get_doc()
    doc.saveAs(percorso)
    return {"file": percorso}

def cmd_esporta_step(percorso, **kw):
    doc = get_doc()
    objs = [o for o in doc.Objects if hasattr(o, 'Shape')]
    Part.export(objs, percorso)
    return {"file": percorso, "oggetti": len(objs)}

def get_wall_properties(wall):
    """Estrae proprietà geometriche di un muro."""
    bb = wall.Shape.BoundBox
    if bb.XLength > bb.YLength:
        length = bb.XLength / 1000
        thickness = bb.YLength / 1000
        direction = "X"
    else:
        length = bb.YLength / 1000
        thickness = bb.XLength / 1000
        direction = "Y"
    height = bb.ZLength / 1000
    area = length * thickness
    volume = wall.Shape.Volume / 1e9
    return {
        "name": wall.Name,
        "length": length,
        "thickness": thickness,
        "height": height,
        "area": area,
        "volume": volume,
        "direction": direction,
        "centroid": (bb.Center.x/1000, bb.Center.y/1000, bb.Center.z/1000)
    }

def colora_elementi(elementi, dcr_values):
    """Colora elementi in base a DCR."""
    for elem, dcr in zip(elementi, dcr_values):
        if dcr <= 0.7:
            elem.ViewObject.ShapeColor = (0.2, 0.8, 0.2)  # Verde
        elif dcr <= 1.0:
            elem.ViewObject.ShapeColor = (1.0, 0.8, 0.0)  # Giallo
        else:
            elem.ViewObject.ShapeColor = (1.0, 0.2, 0.2)  # Rosso
    FreeCADGui.updateGui()

# ==================== ANALISI POR ====================
def cmd_analisi_por(ag=0.15, suolo="B", q=2.0, **kw):
    """Analisi POR (Per Ogni Resistente) - NTC 2018."""
    doc = get_doc()
    walls = [o for o in doc.Objects if o.Name.startswith("Muro")]
    if not walls:
        return {"error": "Nessun muro"}

    # Coefficienti suolo
    S_values = {"A": 1.0, "B": 1.2, "C": 1.25, "D": 1.35, "E": 1.4}
    S = S_values.get(suolo, 1.2)

    area_x = area_y = 0.0
    for wall in walls:
        props = get_wall_properties(wall)
        if props["direction"] == "X":
            area_x += props["area"]
        else:
            area_y += props["area"]

    fvd = 100  # kN/m² (fvk/gamma_M = 0.2/2.0)
    Vrd_x = area_x * fvd
    Vrd_y = area_y * fvd

    peso_muri = sum(w.Shape.Volume / 1e9 * 18 for w in walls)
    peso_solai = sum(s.Shape.Volume / 1e9 * 25 for s in doc.Objects if s.Name.startswith("Solaio"))
    peso_totale = peso_muri + peso_solai

    Ved = peso_totale * ag * S * (2.5 / q)

    dcr_x = Ved / Vrd_x if Vrd_x > 0 else 999
    dcr_y = Ved / Vrd_y if Vrd_y > 0 else 999
    dcr_max = max(dcr_x, dcr_y)

    colora_elementi(walls, [dcr_max] * len(walls))

    return {
        "metodo": "POR", "n_muri": len(walls),
        "parametri": {"ag": ag, "suolo": suolo, "S": S, "q": q},
        "area_resistente_x_m2": round(area_x, 2),
        "area_resistente_y_m2": round(area_y, 2),
        "Vrd_x_kN": round(Vrd_x, 1), "Vrd_y_kN": round(Vrd_y, 1),
        "peso_kN": round(peso_totale, 1), "Ved_kN": round(Ved, 1),
        "DCR_x": round(dcr_x, 3), "DCR_y": round(dcr_y, 3),
        "verifica_x": "OK" if dcr_x <= 1.0 else "NO",
        "verifica_y": "OK" if dcr_y <= 1.0 else "NO",
        "colore": "VERDE" if dcr_max <= 0.7 else ("GIALLO" if dcr_max <= 1.0 else "ROSSO")
    }

# ==================== ANALISI SAM ====================
def cmd_analisi_sam(ag=0.15, suolo="B", **kw):
    """Analisi SAM (Simplified Analysis Method) - Telaio equivalente."""
    doc = get_doc()
    walls = [o for o in doc.Objects if o.Name.startswith("Muro")]
    if not walls:
        return {"error": "Nessun muro"}

    S_values = {"A": 1.0, "B": 1.2, "C": 1.25, "D": 1.35, "E": 1.4}
    S = S_values.get(suolo, 1.2)

    # Analisi per singolo maschio murario
    results = []
    for wall in walls:
        props = get_wall_properties(wall)
        L, t, H = props["length"], props["thickness"], props["height"]

        # Rigidezza a taglio
        G = 500  # MPa (modulo taglio muratura)
        A = L * t  # m²
        Ks = G * 1000 * A / H  # kN/m

        # Rigidezza a flessione
        E = 1500  # MPa (modulo elastico)
        I = t * L**3 / 12  # m⁴
        Kf = 12 * E * 1000 * I / H**3  # kN/m

        # Rigidezza combinata (serie)
        K = 1 / (1/Ks + 1/Kf) if (Ks > 0 and Kf > 0) else 0

        # Resistenza a taglio
        fvd = 0.1  # N/mm² = 100 kN/m²
        Vt = fvd * 1000 * A  # kN

        # Resistenza a pressoflessione (semplificata)
        fd = 2.4  # N/mm² resistenza compressione
        sigma_0 = 0.3  # MPa carico verticale medio
        Vp = L * t * 1000 * (1.5 * fvd * 1000 * (1 + sigma_0 / (1.5 * fvd)))

        Vrd = min(Vt, Vp)

        results.append({
            "muro": wall.Name,
            "L": round(L, 2), "t": round(t, 2), "H": round(H, 2),
            "K_kN_m": round(K, 1),
            "Vrd_kN": round(Vrd, 1),
            "direzione": props["direction"]
        })

    # Distribuzione taglio in base a rigidezza
    K_tot_x = sum(r["K_kN_m"] for r in results if r["direzione"] == "X")
    K_tot_y = sum(r["K_kN_m"] for r in results if r["direzione"] == "Y")

    peso_totale = sum(w.Shape.Volume / 1e9 * 18 for w in walls)
    peso_totale += sum(s.Shape.Volume / 1e9 * 25 for s in doc.Objects if s.Name.startswith("Solaio"))
    Ved = peso_totale * ag * S * 1.25  # q=2

    dcr_list = []
    for r in results:
        if r["direzione"] == "X" and K_tot_x > 0:
            Vi = Ved * r["K_kN_m"] / K_tot_x
        elif r["direzione"] == "Y" and K_tot_y > 0:
            Vi = Ved * r["K_kN_m"] / K_tot_y
        else:
            Vi = 0
        dcr = Vi / r["Vrd_kN"] if r["Vrd_kN"] > 0 else 999
        r["Ved_kN"] = round(Vi, 1)
        r["DCR"] = round(dcr, 3)
        r["verifica"] = "OK" if dcr <= 1.0 else "NO"
        dcr_list.append(dcr)

    colora_elementi(walls, dcr_list)

    return {
        "metodo": "SAM",
        "n_muri": len(walls),
        "K_tot_x": round(K_tot_x, 1),
        "K_tot_y": round(K_tot_y, 1),
        "Ved_kN": round(Ved, 1),
        "muri": results,
        "verificato": all(r["verifica"] == "OK" for r in results)
    }

# ==================== ANALISI CARICHI VERTICALI ====================
def cmd_analisi_carichi(**kw):
    """Analisi carichi verticali - verifica a compressione."""
    doc = get_doc()
    walls = [o for o in doc.Objects if o.Name.startswith("Muro")]
    if not walls:
        return {"error": "Nessun muro"}

    # Carichi
    g1 = 3.0   # kN/m² peso proprio solaio
    g2 = 2.0   # kN/m² permanenti portati
    qk = 2.0   # kN/m² variabili (abitazione)
    gamma_g = 1.3
    gamma_q = 1.5

    qEd = gamma_g * (g1 + g2) + gamma_q * qk  # kN/m² carico combinato

    results = []
    for wall in walls:
        props = get_wall_properties(wall)
        L, t, H = props["length"], props["thickness"], props["height"]

        # Area di influenza (semplificata: metà campata per lato)
        area_inf = L * 2.5  # m² (ipotesi campata 5m)

        # Carico dal solaio
        N_solaio = qEd * area_inf  # kN

        # Peso proprio muro
        N_muro = props["volume"] * 18  # kN

        # Carico totale
        NEd = N_solaio + N_muro  # kN

        # Resistenza a compressione
        fd = 2.4  # N/mm² = 2400 kN/m²
        phi = 0.65  # fattore riduzione snellezza
        NRd = phi * fd * 1000 * L * t  # kN

        dcr = NEd / NRd if NRd > 0 else 999

        results.append({
            "muro": wall.Name,
            "NEd_kN": round(NEd, 1),
            "NRd_kN": round(NRd, 1),
            "sigma_kN_m2": round(NEd / (L * t), 1) if L * t > 0 else 0,
            "DCR": round(dcr, 3),
            "verifica": "OK" if dcr <= 1.0 else "NO"
        })

    dcr_list = [r["DCR"] for r in results]
    colora_elementi(walls, dcr_list)

    return {
        "metodo": "CARICHI_VERTICALI",
        "carichi": {"g1": g1, "g2": g2, "qk": qk, "qEd": round(qEd, 2)},
        "muri": results,
        "verificato": all(r["verifica"] == "OK" for r in results)
    }

# ==================== ANALISI PRESSOFLESSIONE ====================
def cmd_analisi_pressoflessione(ag=0.15, **kw):
    """Verifica a pressoflessione nel piano."""
    doc = get_doc()
    walls = [o for o in doc.Objects if o.Name.startswith("Muro")]
    if not walls:
        return {"error": "Nessun muro"}

    results = []
    for wall in walls:
        props = get_wall_properties(wall)
        L, t, H = props["length"], props["thickness"], props["height"]

        # Carico verticale (semplificato)
        sigma_0 = 0.3  # MPa = 300 kN/m²
        N = sigma_0 * 1000 * L * t  # kN

        # Resistenza a compressione
        fd = 2.4  # MPa

        # Momento resistente (formula NTC 2018 7.8.2.2.1)
        # Mu = (L² * t * sigma_0 / 2) * (1 - sigma_0 / (0.85 * fd))
        Mu = (L**2 * t * sigma_0 * 1000 / 2) * (1 - sigma_0 / (0.85 * fd))  # kNm

        # Taglio resistente associato
        Vu = 2 * Mu / H  # kN (mensola)

        # Taglio sollecitante (sismico)
        peso = props["volume"] * 18 + 5 * L  # peso proprio + solaio
        Ved = peso * ag * 1.5  # kN

        dcr = Ved / Vu if Vu > 0 else 999

        results.append({
            "muro": wall.Name,
            "N_kN": round(N, 1),
            "Mu_kNm": round(Mu, 1),
            "Vu_kN": round(Vu, 1),
            "Ved_kN": round(Ved, 1),
            "DCR": round(dcr, 3),
            "verifica": "OK" if dcr <= 1.0 else "NO"
        })

    dcr_list = [r["DCR"] for r in results]
    colora_elementi(walls, dcr_list)

    return {
        "metodo": "PRESSOFLESSIONE",
        "muri": results,
        "verificato": all(r["verifica"] == "OK" for r in results)
    }

# ==================== ANALISI TAGLIO ====================
def cmd_analisi_taglio(fvk=0.2, gamma_m=2.0, **kw):
    """Verifica a taglio per scorrimento."""
    doc = get_doc()
    walls = [o for o in doc.Objects if o.Name.startswith("Muro")]
    if not walls:
        return {"error": "Nessun muro"}

    fvd = fvk / gamma_m  # MPa

    results = []
    for wall in walls:
        props = get_wall_properties(wall)
        L, t, H = props["length"], props["thickness"], props["height"]

        # Carico verticale
        sigma_n = 0.3  # MPa (compressione media)

        # Resistenza a taglio (Turnsek-Cacovic)
        # fvd = fvk0 + 0.4 * sigma_n (con limite)
        fvd_eff = min(fvd + 0.4 * sigma_n, 0.2)  # MPa

        # Taglio resistente
        VRd = fvd_eff * 1000 * L * t  # kN

        # Taglio sollecitante
        peso = props["volume"] * 18 + 5 * L
        Ved = peso * 0.15 * 1.5  # ag=0.15, amplificato

        dcr = Ved / VRd if VRd > 0 else 999

        results.append({
            "muro": wall.Name,
            "fvd_MPa": round(fvd_eff, 3),
            "VRd_kN": round(VRd, 1),
            "Ved_kN": round(Ved, 1),
            "DCR": round(dcr, 3),
            "verifica": "OK" if dcr <= 1.0 else "NO"
        })

    dcr_list = [r["DCR"] for r in results]
    colora_elementi(walls, dcr_list)

    return {
        "metodo": "TAGLIO",
        "parametri": {"fvk": fvk, "gamma_m": gamma_m, "fvd": round(fvd, 3)},
        "muri": results,
        "verificato": all(r["verifica"] == "OK" for r in results)
    }

# ==================== ANALISI RIBALTAMENTO ====================
def cmd_analisi_ribaltamento(**kw):
    """Verifica a ribaltamento fuori piano."""
    doc = get_doc()
    walls = [o for o in doc.Objects if o.Name.startswith("Muro")]
    if not walls:
        return {"error": "Nessun muro"}

    results = []
    for wall in walls:
        props = get_wall_properties(wall)
        L, t, H = props["length"], props["thickness"], props["height"]

        # Snellezza
        lambda_val = H / t

        # Snellezza limite NTC 2018 (Tab. 7.8.II)
        lambda_lim = 12 if t >= 0.24 else 10

        # Verifica geometrica
        verifica_geo = lambda_val <= lambda_lim

        # Momento stabilizzante (peso proprio)
        W = props["volume"] * 18  # kN
        Ms = W * t / 2  # kNm

        # Momento ribaltante (sisma fuori piano)
        ag = 0.15
        Sa = ag * 2.5  # accelerazione fuori piano
        Mr = W * Sa * H / 2  # kNm

        dcr = Mr / Ms if Ms > 0 else 999

        results.append({
            "muro": wall.Name,
            "snellezza": round(lambda_val, 1),
            "snellezza_lim": lambda_lim,
            "verifica_geo": "OK" if verifica_geo else "NO",
            "Ms_kNm": round(Ms, 1),
            "Mr_kNm": round(Mr, 1),
            "DCR": round(dcr, 3),
            "verifica": "OK" if dcr <= 1.0 and verifica_geo else "NO"
        })

    dcr_list = [r["DCR"] for r in results]
    colora_elementi(walls, dcr_list)

    return {
        "metodo": "RIBALTAMENTO",
        "muri": results,
        "verificato": all(r["verifica"] == "OK" for r in results)
    }

# ==================== REPORT COMPLETO ====================
def cmd_report_completo(ag=0.15, suolo="B", **kw):
    """Genera report completo con tutte le verifiche."""
    results = {
        "POR": cmd_analisi_por(ag=ag, suolo=suolo),
        "SAM": cmd_analisi_sam(ag=ag, suolo=suolo),
        "CARICHI": cmd_analisi_carichi(),
        "PRESSOFLESSIONE": cmd_analisi_pressoflessione(ag=ag),
        "TAGLIO": cmd_analisi_taglio(),
        "RIBALTAMENTO": cmd_analisi_ribaltamento()
    }

    verifiche = {k: v.get("verificato", v.get("verifica_x") == "OK" and v.get("verifica_y") == "OK")
                 for k, v in results.items()}

    return {
        "metodo": "REPORT_COMPLETO",
        "verifiche": verifiche,
        "tutto_verificato": all(verifiche.values()),
        "dettagli": results
    }

# Mappa comandi
COMMANDS = {
    "nuovo_progetto": cmd_nuovo_progetto,
    "muro": cmd_muro,
    "rettangolo": cmd_rettangolo,
    "pilastro": cmd_pilastro,
    "trave": cmd_trave,
    "solaio": cmd_solaio,
    "fondazioni": cmd_fondazioni,
    "scala": cmd_scala,
    "copertura": cmd_copertura,
    "fit": cmd_fit,
    "get_stato": cmd_get_stato,
    "salva": cmd_salva,
    "esporta_step": cmd_esporta_step,
    # Analisi
    "analisi_por": cmd_analisi_por,
    "analisi_sam": cmd_analisi_sam,
    "analisi_carichi": cmd_analisi_carichi,
    "analisi_pressoflessione": cmd_analisi_pressoflessione,
    "analisi_taglio": cmd_analisi_taglio,
    "analisi_ribaltamento": cmd_analisi_ribaltamento,
    "report_completo": cmd_report_completo,
}

# ========== TIMER LOOP ==========

def check_commands():
    """Controlla e esegue comandi dal file."""
    try:
        if os.path.exists(CMD_FILE):
            with open(CMD_FILE, 'r') as f:
                data = json.load(f)

            # Rimuovi file comando
            os.remove(CMD_FILE)

            cmd_name = data.get("command", "")
            args = data.get("args", {})

            if cmd_name in COMMANDS:
                result = COMMANDS[cmd_name](**args)
                result["success"] = True
            else:
                result = {"success": False, "error": f"Comando sconosciuto: {cmd_name}"}

            # Scrivi risposta
            with open(RESP_FILE, 'w') as f:
                json.dump(result, f)

            # Aggiorna GUI
            FreeCADGui.updateGui()

    except Exception as e:
        with open(RESP_FILE, 'w') as f:
            json.dump({"success": False, "error": str(e)}, f)

# Avvia timer (controlla ogni 100ms)
from PySide2 import QtCore
timer = QtCore.QTimer()
timer.timeout.connect(check_commands)
timer.start(100)

print("=" * 50)
print("MURATURA MCP GUI Server Attivo")
print("=" * 50)
print(f"Comando: {CMD_FILE}")
print(f"Risposta: {RESP_FILE}")
print("Pronto per ricevere comandi MCP!")
print("=" * 50)
