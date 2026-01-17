#!/usr/bin/env python3
"""
Client MCP per controllare FreeCAD GUI
"""
import socket
import json

MCP_HOST = '127.0.0.1'
MCP_PORT = 9999

def send_command(command, **args):
    """Invia comando al bridge MCP in FreeCAD."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((MCP_HOST, MCP_PORT))

        data = json.dumps({"command": command, "args": args})
        sock.send(data.encode('utf-8'))

        response = sock.recv(4096).decode('utf-8')
        sock.close()

        return json.loads(response)
    except ConnectionRefusedError:
        return {"success": False, "error": "FreeCAD non in ascolto. Avvia prima FreeCAD con start_gui_mcp.py"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Funzioni helper
def nuovo_progetto(nome="Muratura", piani=2, altezza_piano=3.0):
    return send_command("nuovo_progetto", nome=nome, piani=piani, altezza_piano=altezza_piano)

def muro(x1, y1, x2, y2, spessore=0.3, piano=0):
    return send_command("muro", x1=x1, y1=y1, x2=x2, y2=y2, spessore=spessore, piano=piano)

def rettangolo(x, y, larg, prof, spessore=0.3, piano=0):
    return send_command("rettangolo", x=x, y=y, larg=larg, prof=prof, spessore=spessore, piano=piano)

def pilastro(x, y, larg=0.3, prof=0.3, piano=0):
    return send_command("pilastro", x=x, y=y, larg=larg, prof=prof, piano=piano)

def solaio(piano=0):
    return send_command("solaio", piano=piano)

def fondazioni():
    return send_command("fondazioni")

def fit():
    return send_command("fit")

def get_stato():
    return send_command("get_stato")


if __name__ == "__main__":
    print("Test connessione MCP...")
    r = get_stato()
    if r.get("success", False) or "error" not in r:
        print(f"Connesso! Stato: {r}")
    else:
        print(f"Errore: {r.get('error')}")
