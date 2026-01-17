# MURATURA 3.0 - Documento di Coscienza

**Versione**: 3.1.0
**Data**: 2026-01-17
**Base**: FreeCAD 1.0.2 Portable
**Stato**: MCP GUI FUNZIONANTE

---

## FILOSOFIA

**Muratura è FreeCAD specializzato per analisi strutturale edifici in muratura.**

- FreeCAD fornisce: GUI, viewport 3D, geometria BREP, export
- Muratura aggiunge: Workbench NTC 2018, analisi POR/SAM, report
- MCP controlla FreeCAD GUI in tempo reale

---

## STRUTTURA PROGETTO

```
D:\muratura3\
├── freecad/              # FreeCAD 1.0.2 Portable
│   ├── bin/
│   ├── lib/
│   └── Mod/Muratura/     # Link al workbench
│
├── muratura/             # Codice Muratura
│   ├── __init__.py
│   ├── workbench/        # Workbench FreeCAD
│   │   ├── Init.py
│   │   ├── InitGui.py
│   │   └── commands/     # Comandi GUI
│   │       ├── geometry.py
│   │       ├── structural.py
│   │       ├── analysis.py
│   │       └── export.py
│   └── ntc2018/          # Moduli calcolo NTC
│       ├── materials.py
│       ├── seismic.py
│       ├── loads.py
│       └── analyses/
│
├── mcp_server.py         # Server MCP v3.0
├── muratura.bat          # Launcher Windows
├── muratura_launcher.py  # Launcher Python
│
└── .claude/
    └── claude.md         # Questo file
```

---

## AVVIO

### MCP GUI (RACCOMANDATO)
```batch
# 1. Avvia FreeCAD con MCP Server
start "" "D:/muratura3/freecad/bin/FreeCAD.exe" "D:/muratura3/mcp_gui_server.py"

# 2. Usa client per inviare comandi
py -3.11 -c "from mcp_gui_client import *; rettangolo(0,0,10,8); fit()"
```

### MCP Headless (senza GUI)
```batch
py -3.11 mcp_server.py
```

### Configurazione Claude Desktop
```json
{
  "mcpServers": {
    "muratura": {
      "command": "py",
      "args": ["-3.11", "D:\\muratura3\\mcp_server.py"]
    }
  }
}
```

---

## COMANDI MCP (13 tools)

| Tool | Descrizione |
|------|-------------|
| `nuovo_progetto` | Crea progetto (nome, piani, altezza) |
| `get_stato` | Stato progetto |
| `salva` | Salva .FCStd |
| `muro` | Muro da x1,y1 a x2,y2 |
| `rettangolo` | 4 muri rettangolari |
| `pilastro` | Pilastro in x,y |
| `trave` | Trave da x1,y1 a x2,y2 |
| `solaio` | Solaio automatico |
| `fondazioni` | Fondazioni automatiche |
| `scala` | Scala a rampa |
| `copertura` | Copertura a falde |
| `analisi_por` | Analisi POR |
| `esporta_step` | Export STEP |
| `elimina` | Elimina elemento |

---

## COMANDI WORKBENCH GUI

### Toolbar Geometria
- Nuovo Muro
- Nuova Apertura (finestra/porta)
- Nuovo Pilastro
- Nuova Trave
- Nuovo Solaio
- Nuova Scala
- Nuova Copertura

### Toolbar Struttura
- Genera Fondazioni
- Genera Cordoli
- Imposta Materiale
- Imposta Carichi

### Toolbar Analisi
- Parametri Sismici
- Analisi POR
- Analisi SAM
- Mostra DCR

### Toolbar Export
- Genera Relazione
- Esporta IFC
- Esporta DXF

---

## MODULI NTC 2018

Da `muratura/ntc2018/`:

| Modulo | Contenuto |
|--------|-----------|
| `materials.py` | Proprietà muratura Tabella C8.5.I |
| `seismic.py` | Database parametri sismici comuni |
| `loads.py` | Carichi neve, vento NTC 3.3-3.4 |
| `constitutive.py` | Leggi costitutive |
| `analyses/por.py` | Metodo POR |
| `analyses/sam.py` | Metodo SAM |
| `analyses/fem.py` | Analisi FEM |

---

## WORKFLOW TIPICO

1. **Geometria**: Disegna muri con `rettangolo` o `muro`
2. **Elementi**: Aggiungi `pilastro`, `trave`, `scala`
3. **Struttura**: `solaio` + `fondazioni` + `copertura`
4. **Analisi**: Imposta sismico, esegui `analisi_por`
5. **Export**: `esporta_step`, genera relazione

---

## ESEMPIO MCP

```python
# Edificio 2 piani
nuovo_progetto("Casa Esempio", piani=2, altezza_piano=3.0)

# Piano terra
rettangolo(0, 0, 12, 10, spessore=0.4, piano=0)
muro(6, 0, 6, 10, spessore=0.3, piano=0)  # divisorio

# Pilastri angolari
for x, y in [(0,0), (12,0), (12,10), (0,10)]:
    pilastro(x, y, larg=0.4, piano=0)

# Strutture
solaio(piano=0)
fondazioni()
copertura(altezza_colmo=2.5)

# Analisi
analisi_por()

# Export
esporta_step("D:/output/casa.step")
salva("D:/output/casa.FCStd")
```

---

## MCP GUI - CONTROLLO TEMPO REALE

Il sistema MCP GUI permette di controllare FreeCAD in tempo reale:

**File di comunicazione:**
- `_mcp_command.json` - Comando da eseguire
- `_mcp_response.json` - Risposta dal server

**Come funziona:**
1. `mcp_gui_server.py` gira dentro FreeCAD GUI
2. Timer Qt controlla ogni 100ms se c'è un comando
3. Esegue comando con `Part.show()` per visibilità immediata
4. Scrive risposta e aggiorna GUI

**Esempio uso:**
```python
from mcp_gui_client import *

nuovo_progetto("Casa", piani=2)
rettangolo(0, 0, 10, 8, spessore=0.4)
pilastro(0, 0)
pilastro(10, 0)
solaio()
fondazioni()
copertura()
fit()  # Centra vista
```

---

## NOTE

- Python 3.11 richiesto (compatibilità FreeCAD)
- Coordinate in metri, FreeCAD lavora in mm internamente
- Piano 0 = piano terra
- Usare `Part.show()` per oggetti visibili in GUI (non `addObject`)
