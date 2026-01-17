# -*- coding: utf-8 -*-
"""
Muratura Workbench - Inizializzazione (headless)

Questo file viene caricato sia in modalità GUI che console.
Non importare moduli GUI qui.
"""

import FreeCAD

# Aggiungi path per moduli Muratura
import os
import sys

MURATURA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MURATURA_ROOT not in sys.path:
    sys.path.insert(0, MURATURA_ROOT)

FreeCAD.Console.PrintMessage("Muratura Workbench inizializzato\n")
