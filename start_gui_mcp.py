# -*- coding: utf-8 -*-
"""
Avvia FreeCAD GUI con MCP Bridge attivo
"""
import FreeCAD
import FreeCADGui

# Importa e avvia bridge
from Muratura import mcp_bridge
mcp_bridge.start_bridge()

print("=" * 50)
print("MURATURA 3.0 - GUI con MCP Bridge")
print("=" * 50)
print("MCP Bridge attivo su porta 9999")
print("Invia comandi JSON a 127.0.0.1:9999")
print("=" * 50)
