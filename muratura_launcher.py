#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MURATURA - Launcher

Avvia FreeCAD configurato come Muratura Edition.
"""

import os
import sys
import subprocess

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FREECAD_DIR = os.path.join(SCRIPT_DIR, "freecad")
FREECAD_BIN = os.path.join(FREECAD_DIR, "bin", "FreeCAD.exe")
MURATURA_DIR = os.path.join(SCRIPT_DIR, "muratura")
WORKBENCH_DIR = os.path.join(MURATURA_DIR, "workbench")

def setup_user_config():
    """Crea configurazione utente per Muratura."""
    # FreeCAD user config directory
    appdata = os.environ.get("APPDATA", "")
    fc_config = os.path.join(appdata, "FreeCAD")

    if not os.path.exists(fc_config):
        os.makedirs(fc_config)

    # Crea user.cfg con workbench Muratura come default
    user_cfg = os.path.join(fc_config, "user.cfg")

    # Aggiungi path workbench Muratura ai Mod
    mod_dir = os.path.join(FREECAD_DIR, "Mod", "Muratura")
    if not os.path.exists(mod_dir):
        # Link simbolico al workbench
        try:
            os.symlink(WORKBENCH_DIR, mod_dir)
            print(f"Workbench linkato: {mod_dir}")
        except OSError:
            # Windows senza privilegi admin - copia i file
            import shutil
            if os.path.exists(mod_dir):
                shutil.rmtree(mod_dir)
            shutil.copytree(WORKBENCH_DIR, mod_dir)
            print(f"Workbench copiato: {mod_dir}")

def main():
    print("=" * 50)
    print("MURATURA - Analisi Strutturale Edifici in Muratura")
    print("Basato su FreeCAD 1.0.2")
    print("=" * 50)

    # Setup
    setup_user_config()

    # Aggiungi path muratura al PYTHONPATH
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{MURATURA_DIR};{pythonpath}"

    # Avvia FreeCAD
    print(f"\nAvvio FreeCAD: {FREECAD_BIN}")

    if os.path.exists(FREECAD_BIN):
        subprocess.run([FREECAD_BIN], env=env)
    else:
        print(f"ERRORE: FreeCAD non trovato in {FREECAD_BIN}")
        sys.exit(1)


if __name__ == "__main__":
    main()
