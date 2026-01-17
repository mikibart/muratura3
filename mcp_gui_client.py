#!/usr/bin/env python3
"""
MCP GUI Client - Invia comandi a FreeCAD GUI
"""
import json
import os
import time

CMD_FILE = "D:/muratura3/_mcp_command.json"
RESP_FILE = "D:/muratura3/_mcp_response.json"
TIMEOUT = 5.0

def send(command, **args):
    """Invia comando e attende risposta."""
    # Rimuovi vecchia risposta
    if os.path.exists(RESP_FILE):
        os.remove(RESP_FILE)

    # Scrivi comando
    with open(CMD_FILE, 'w') as f:
        json.dump({"command": command, "args": args}, f)

    # Attendi risposta
    start = time.time()
    while time.time() - start < TIMEOUT:
        if os.path.exists(RESP_FILE):
            time.sleep(0.05)  # Aspetta scrittura completa
            with open(RESP_FILE, 'r') as f:
                result = json.load(f)
            os.remove(RESP_FILE)
            return result
        time.sleep(0.05)

    return {"success": False, "error": "Timeout - FreeCAD GUI non risponde"}

# Helper functions
def nuovo_progetto(nome="Muratura", piani=2, altezza_piano=3.0):
    return send("nuovo_progetto", nome=nome, piani=piani, altezza_piano=altezza_piano)

def muro(x1, y1, x2, y2, spessore=0.3, piano=0):
    return send("muro", x1=x1, y1=y1, x2=x2, y2=y2, spessore=spessore, piano=piano)

def rettangolo(x, y, larg, prof, spessore=0.3, piano=0):
    return send("rettangolo", x=x, y=y, larg=larg, prof=prof, spessore=spessore, piano=piano)

def pilastro(x, y, larg=0.3, prof=0.3, piano=0):
    return send("pilastro", x=x, y=y, larg=larg, prof=prof, piano=piano)

def trave(x1, y1, x2, y2, larg=0.3, alt=0.5, piano=0):
    return send("trave", x1=x1, y1=y1, x2=x2, y2=y2, larg=larg, alt=alt, piano=piano)

def solaio(piano=0):
    return send("solaio", piano=piano)

def fondazioni(larg=0.6, alt=0.5):
    return send("fondazioni", larg=larg, alt=alt)

def scala(x, y, larg=1.2, piano=0):
    return send("scala", x=x, y=y, larg=larg, piano=piano)

def copertura(altezza_colmo=2.0, sporto=0.5):
    return send("copertura", altezza_colmo=altezza_colmo, sporto=sporto)

def fit():
    return send("fit")

def get_stato():
    return send("get_stato")

def salva(percorso):
    return send("salva", percorso=percorso)

def esporta_step(percorso):
    return send("esporta_step", percorso=percorso)

def analisi_por():
    return send("analisi_por")

def analisi_sam():
    return send("analisi_sam")

def analisi_carichi():
    return send("analisi_carichi")

def analisi_pressoflessione():
    return send("analisi_pressoflessione")

def analisi_taglio():
    return send("analisi_taglio")

def analisi_ribaltamento():
    return send("analisi_ribaltamento")

def report_completo(percorso="D:/muratura3/report_ntc2018.txt"):
    return send("report_completo", percorso=percorso)


if __name__ == "__main__":
    print("Test MCP GUI Client...")
    r = get_stato()
    print(f"Stato: {r}")
