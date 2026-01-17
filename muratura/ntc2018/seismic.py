# seismic.py - Modulo Parametri Sismici NTC 2018
"""
Calcolo parametri sismici secondo NTC 2018 e Circolare 617/2019.

Funzionalita':
- Database comuni italiani con parametri sismici (ag, F0, Tc*)
- Calcolo spettro di risposta elastico e di progetto
- Categorie di sottosuolo e topografiche
- Vita nominale e classi d'uso
- Periodi di ritorno per stati limite

Riferimenti normativi:
- NTC 2018: Cap. 3.2 (Azione sismica)
- Circolare 617/2019: Cap. C3.2
- Allegato A e B alle NTC 2018
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import os

# ============================================================================
# ENUMERAZIONI
# ============================================================================

class SoilCategory(Enum):
    """Categorie di sottosuolo NTC 2018 Tab. 3.2.II"""
    A = "A"  # Ammassi rocciosi affioranti, Vs30 > 800 m/s
    B = "B"  # Depositi di sabbie o ghiaie molto addensate, 360 < Vs30 <= 800 m/s
    C = "C"  # Depositi di sabbie e ghiaie mediamente addensate, 180 < Vs30 <= 360 m/s
    D = "D"  # Depositi di terreni granulari da sciolti a poco addensati, Vs30 < 180 m/s
    E = "E"  # Profili di terreno tipo C o D con spessore < 20m su substrato A

class TopographicCategory(Enum):
    """Categorie topografiche NTC 2018 Tab. 3.2.IV"""
    T1 = "T1"  # Superficie pianeggiante, pendii < 15 gradi
    T2 = "T2"  # Pendii con inclinazione > 15 gradi
    T3 = "T3"  # Rilievi con larghezza in cresta < larghezza base
    T4 = "T4"  # Rilievi con larghezza in cresta molto minore larghezza base

class UseClass(Enum):
    """Classi d'uso NTC 2018 Tab. 2.4.II"""
    I = 1    # Costruzioni con presenza solo occasionale di persone
    II = 2   # Costruzioni con affollamenti normali
    III = 3  # Costruzioni con affollamenti significativi
    IV = 4   # Costruzioni con funzioni pubbliche o strategiche

class LimitState(Enum):
    """Stati limite NTC 2018"""
    SLO = "SLO"  # Stato Limite di Operativita'
    SLD = "SLD"  # Stato Limite di Danno
    SLV = "SLV"  # Stato Limite di salvaguardia della Vita
    SLC = "SLC"  # Stato Limite di prevenzione del Collasso

# ============================================================================
# COSTANTI E TABELLE NTC 2018
# ============================================================================

# Coefficienti d'uso Cu (Tab. 2.4.II)
USE_COEFFICIENTS = {
    UseClass.I: 0.7,
    UseClass.II: 1.0,
    UseClass.III: 1.5,
    UseClass.IV: 2.0
}

# Probabilita' di superamento PVR per stati limite (Tab. 3.2.I)
PVR_VALUES = {
    LimitState.SLO: 0.81,  # 81%
    LimitState.SLD: 0.63,  # 63%
    LimitState.SLV: 0.10,  # 10%
    LimitState.SLC: 0.05   # 5%
}

# Coefficienti di amplificazione stratigrafica SS e CC (Tab. 3.2.V)
# Formato: {SoilCategory: (SS_formula_params, CC_formula_params)}
SOIL_AMPLIFICATION = {
    SoilCategory.A: {'SS_max': 1.0, 'SS_coef': 0.0, 'CC': 1.0},
    SoilCategory.B: {'SS_max': 1.4, 'SS_coef': 0.4, 'CC_coef': 1.10},
    SoilCategory.C: {'SS_max': 1.7, 'SS_coef': 0.6, 'CC_coef': 1.05},
    SoilCategory.D: {'SS_max': 2.4, 'SS_coef': 1.5, 'CC_coef': 1.25},
    SoilCategory.E: {'SS_max': 2.0, 'SS_coef': 1.1, 'CC_coef': 1.15},
}

# Coefficienti di amplificazione topografica ST (Tab. 3.2.VI)
TOPO_AMPLIFICATION = {
    TopographicCategory.T1: 1.0,
    TopographicCategory.T2: 1.2,
    TopographicCategory.T3: 1.2,
    TopographicCategory.T4: 1.4
}

# ============================================================================
# DATABASE COMUNI ITALIANI (Campione rappresentativo)
# ============================================================================

# Database semplificato con i principali comuni italiani
# In produzione si userebbe il database completo INGV con 10751 punti griglia
COMUNI_DATABASE = {
    # Formato: "NOME_COMUNE": {"lat": lat, "lon": lon, "ag": [ag_475], "F0": [F0_475], "Tc": [Tc_475]}
    # Valori per TR=475 anni (SLV) - i valori per altri TR si interpolano

    # SICILIA
    "PALERMO": {"lat": 38.1157, "lon": 13.3615, "provincia": "PA", "regione": "Sicilia",
                "ag": 0.150, "F0": 2.50, "Tc": 0.35},
    "CATANIA": {"lat": 37.5079, "lon": 15.0830, "provincia": "CT", "regione": "Sicilia",
                "ag": 0.200, "F0": 2.45, "Tc": 0.32},
    "MESSINA": {"lat": 38.1938, "lon": 15.5540, "provincia": "ME", "regione": "Sicilia",
                "ag": 0.275, "F0": 2.40, "Tc": 0.30},
    "SIRACUSA": {"lat": 37.0755, "lon": 15.2866, "provincia": "SR", "regione": "Sicilia",
                 "ag": 0.225, "F0": 2.42, "Tc": 0.31},
    "RAGUSA": {"lat": 36.9282, "lon": 14.7306, "provincia": "RG", "regione": "Sicilia",
               "ag": 0.200, "F0": 2.44, "Tc": 0.32},
    "TRAPANI": {"lat": 38.0174, "lon": 12.5113, "provincia": "TP", "regione": "Sicilia",
                "ag": 0.125, "F0": 2.52, "Tc": 0.36},
    "AGRIGENTO": {"lat": 37.3111, "lon": 13.5765, "provincia": "AG", "regione": "Sicilia",
                  "ag": 0.125, "F0": 2.50, "Tc": 0.35},
    "CALTANISSETTA": {"lat": 37.4901, "lon": 14.0629, "provincia": "CL", "regione": "Sicilia",
                      "ag": 0.150, "F0": 2.48, "Tc": 0.34},
    "ENNA": {"lat": 37.5667, "lon": 14.2667, "provincia": "EN", "regione": "Sicilia",
             "ag": 0.175, "F0": 2.46, "Tc": 0.33},

    # CALABRIA
    "REGGIO CALABRIA": {"lat": 38.1089, "lon": 15.6433, "provincia": "RC", "regione": "Calabria",
                        "ag": 0.275, "F0": 2.38, "Tc": 0.29},
    "CATANZARO": {"lat": 38.9098, "lon": 16.5877, "provincia": "CZ", "regione": "Calabria",
                  "ag": 0.250, "F0": 2.40, "Tc": 0.30},
    "COSENZA": {"lat": 39.2988, "lon": 16.2539, "provincia": "CS", "regione": "Calabria",
                "ag": 0.225, "F0": 2.42, "Tc": 0.31},
    "CROTONE": {"lat": 39.0833, "lon": 17.1167, "provincia": "KR", "regione": "Calabria",
                "ag": 0.200, "F0": 2.44, "Tc": 0.32},
    "VIBO VALENTIA": {"lat": 38.6725, "lon": 16.0972, "provincia": "VV", "regione": "Calabria",
                      "ag": 0.250, "F0": 2.40, "Tc": 0.30},

    # CAMPANIA
    "NAPOLI": {"lat": 40.8518, "lon": 14.2681, "provincia": "NA", "regione": "Campania",
               "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "SALERNO": {"lat": 40.6824, "lon": 14.7681, "provincia": "SA", "regione": "Campania",
                "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "AVELLINO": {"lat": 40.9167, "lon": 14.7833, "provincia": "AV", "regione": "Campania",
                 "ag": 0.225, "F0": 2.42, "Tc": 0.31},
    "BENEVENTO": {"lat": 41.1297, "lon": 14.7817, "provincia": "BN", "regione": "Campania",
                  "ag": 0.225, "F0": 2.42, "Tc": 0.31},
    "CASERTA": {"lat": 41.0725, "lon": 14.3311, "provincia": "CE", "regione": "Campania",
                "ag": 0.175, "F0": 2.46, "Tc": 0.33},

    # PUGLIA
    "BARI": {"lat": 41.1171, "lon": 16.8719, "provincia": "BA", "regione": "Puglia",
             "ag": 0.100, "F0": 2.55, "Tc": 0.38},
    "FOGGIA": {"lat": 41.4621, "lon": 15.5444, "provincia": "FG", "regione": "Puglia",
               "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "LECCE": {"lat": 40.3516, "lon": 18.1718, "provincia": "LE", "regione": "Puglia",
              "ag": 0.075, "F0": 2.58, "Tc": 0.40},
    "TARANTO": {"lat": 40.4644, "lon": 17.2470, "provincia": "TA", "regione": "Puglia",
                "ag": 0.100, "F0": 2.55, "Tc": 0.38},
    "BRINDISI": {"lat": 40.6325, "lon": 17.9419, "provincia": "BR", "regione": "Puglia",
                 "ag": 0.075, "F0": 2.58, "Tc": 0.40},

    # BASILICATA
    "POTENZA": {"lat": 40.6404, "lon": 15.8056, "provincia": "PZ", "regione": "Basilicata",
                "ag": 0.225, "F0": 2.42, "Tc": 0.31},
    "MATERA": {"lat": 40.6664, "lon": 16.6043, "provincia": "MT", "regione": "Basilicata",
               "ag": 0.175, "F0": 2.46, "Tc": 0.33},

    # LAZIO
    "ROMA": {"lat": 41.9028, "lon": 12.4964, "provincia": "RM", "regione": "Lazio",
             "ag": 0.125, "F0": 2.52, "Tc": 0.36},
    "FROSINONE": {"lat": 41.6400, "lon": 13.3500, "provincia": "FR", "regione": "Lazio",
                  "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "LATINA": {"lat": 41.4675, "lon": 12.9036, "provincia": "LT", "regione": "Lazio",
               "ag": 0.125, "F0": 2.52, "Tc": 0.36},
    "RIETI": {"lat": 42.4037, "lon": 12.8578, "provincia": "RI", "regione": "Lazio",
              "ag": 0.200, "F0": 2.44, "Tc": 0.32},
    "VITERBO": {"lat": 42.4206, "lon": 12.1085, "provincia": "VT", "regione": "Lazio",
                "ag": 0.125, "F0": 2.52, "Tc": 0.36},

    # ABRUZZO
    "L'AQUILA": {"lat": 42.3498, "lon": 13.3995, "provincia": "AQ", "regione": "Abruzzo",
                 "ag": 0.275, "F0": 2.38, "Tc": 0.29},
    "PESCARA": {"lat": 42.4618, "lon": 14.2161, "provincia": "PE", "regione": "Abruzzo",
                "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "CHIETI": {"lat": 42.3514, "lon": 14.1681, "provincia": "CH", "regione": "Abruzzo",
               "ag": 0.200, "F0": 2.44, "Tc": 0.32},
    "TERAMO": {"lat": 42.6589, "lon": 13.7042, "provincia": "TE", "regione": "Abruzzo",
               "ag": 0.225, "F0": 2.42, "Tc": 0.31},

    # MOLISE
    "CAMPOBASSO": {"lat": 41.5603, "lon": 14.6625, "provincia": "CB", "regione": "Molise",
                   "ag": 0.200, "F0": 2.44, "Tc": 0.32},
    "ISERNIA": {"lat": 41.5939, "lon": 14.2331, "provincia": "IS", "regione": "Molise",
                "ag": 0.225, "F0": 2.42, "Tc": 0.31},

    # MARCHE
    "ANCONA": {"lat": 43.6158, "lon": 13.5189, "provincia": "AN", "regione": "Marche",
               "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "PESARO": {"lat": 43.9097, "lon": 12.9131, "provincia": "PU", "regione": "Marche",
               "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "MACERATA": {"lat": 43.2985, "lon": 13.4534, "provincia": "MC", "regione": "Marche",
                 "ag": 0.200, "F0": 2.44, "Tc": 0.32},
    "ASCOLI PICENO": {"lat": 42.8537, "lon": 13.5749, "provincia": "AP", "regione": "Marche",
                      "ag": 0.225, "F0": 2.42, "Tc": 0.31},
    "FERMO": {"lat": 43.1606, "lon": 13.7158, "provincia": "FM", "regione": "Marche",
              "ag": 0.200, "F0": 2.44, "Tc": 0.32},

    # UMBRIA
    "PERUGIA": {"lat": 43.1107, "lon": 12.3908, "provincia": "PG", "regione": "Umbria",
                "ag": 0.200, "F0": 2.44, "Tc": 0.32},
    "TERNI": {"lat": 42.5636, "lon": 12.6427, "provincia": "TR", "regione": "Umbria",
              "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "NORCIA": {"lat": 42.7919, "lon": 13.0897, "provincia": "PG", "regione": "Umbria",
               "ag": 0.275, "F0": 2.38, "Tc": 0.29},

    # TOSCANA
    "FIRENZE": {"lat": 43.7696, "lon": 11.2558, "provincia": "FI", "regione": "Toscana",
                "ag": 0.125, "F0": 2.52, "Tc": 0.36},
    "PISA": {"lat": 43.7228, "lon": 10.4017, "provincia": "PI", "regione": "Toscana",
             "ag": 0.125, "F0": 2.52, "Tc": 0.36},
    "SIENA": {"lat": 43.3188, "lon": 11.3308, "provincia": "SI", "regione": "Toscana",
              "ag": 0.150, "F0": 2.50, "Tc": 0.35},
    "AREZZO": {"lat": 43.4633, "lon": 11.8797, "provincia": "AR", "regione": "Toscana",
               "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "LIVORNO": {"lat": 43.5485, "lon": 10.3106, "provincia": "LI", "regione": "Toscana",
                "ag": 0.100, "F0": 2.55, "Tc": 0.38},
    "LUCCA": {"lat": 43.8376, "lon": 10.4951, "provincia": "LU", "regione": "Toscana",
              "ag": 0.150, "F0": 2.50, "Tc": 0.35},
    "GROSSETO": {"lat": 42.7635, "lon": 11.1124, "provincia": "GR", "regione": "Toscana",
                 "ag": 0.125, "F0": 2.52, "Tc": 0.36},
    "MASSA": {"lat": 44.0352, "lon": 10.1396, "provincia": "MS", "regione": "Toscana",
              "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "PRATO": {"lat": 43.8777, "lon": 11.1020, "provincia": "PO", "regione": "Toscana",
              "ag": 0.125, "F0": 2.52, "Tc": 0.36},
    "PISTOIA": {"lat": 43.9303, "lon": 10.9078, "provincia": "PT", "regione": "Toscana",
                "ag": 0.150, "F0": 2.50, "Tc": 0.35},

    # EMILIA-ROMAGNA
    "BOLOGNA": {"lat": 44.4949, "lon": 11.3426, "provincia": "BO", "regione": "Emilia-Romagna",
                "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "MODENA": {"lat": 44.6471, "lon": 10.9252, "provincia": "MO", "regione": "Emilia-Romagna",
               "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "PARMA": {"lat": 44.8015, "lon": 10.3279, "provincia": "PR", "regione": "Emilia-Romagna",
              "ag": 0.150, "F0": 2.50, "Tc": 0.35},
    "REGGIO EMILIA": {"lat": 44.6989, "lon": 10.6297, "provincia": "RE", "regione": "Emilia-Romagna",
                      "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "RAVENNA": {"lat": 44.4184, "lon": 12.2035, "provincia": "RA", "regione": "Emilia-Romagna",
                "ag": 0.150, "F0": 2.50, "Tc": 0.35},
    "FERRARA": {"lat": 44.8378, "lon": 11.6199, "provincia": "FE", "regione": "Emilia-Romagna",
                "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "FORLI'": {"lat": 44.2227, "lon": 12.0407, "provincia": "FC", "regione": "Emilia-Romagna",
               "ag": 0.200, "F0": 2.44, "Tc": 0.32},
    "RIMINI": {"lat": 44.0678, "lon": 12.5695, "provincia": "RN", "regione": "Emilia-Romagna",
               "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "PIACENZA": {"lat": 45.0526, "lon": 9.6930, "provincia": "PC", "regione": "Emilia-Romagna",
                 "ag": 0.125, "F0": 2.52, "Tc": 0.36},

    # LIGURIA
    "GENOVA": {"lat": 44.4056, "lon": 8.9463, "provincia": "GE", "regione": "Liguria",
               "ag": 0.075, "F0": 2.58, "Tc": 0.40},
    "LA SPEZIA": {"lat": 44.1025, "lon": 9.8241, "provincia": "SP", "regione": "Liguria",
                  "ag": 0.125, "F0": 2.52, "Tc": 0.36},
    "SAVONA": {"lat": 44.3091, "lon": 8.4772, "provincia": "SV", "regione": "Liguria",
               "ag": 0.075, "F0": 2.58, "Tc": 0.40},
    "IMPERIA": {"lat": 43.8894, "lon": 8.0278, "provincia": "IM", "regione": "Liguria",
                "ag": 0.125, "F0": 2.52, "Tc": 0.36},

    # PIEMONTE
    "TORINO": {"lat": 45.0703, "lon": 7.6869, "provincia": "TO", "regione": "Piemonte",
               "ag": 0.075, "F0": 2.58, "Tc": 0.40},
    "ALESSANDRIA": {"lat": 44.9132, "lon": 8.6156, "provincia": "AL", "regione": "Piemonte",
                    "ag": 0.075, "F0": 2.58, "Tc": 0.40},
    "ASTI": {"lat": 44.9000, "lon": 8.2067, "provincia": "AT", "regione": "Piemonte",
             "ag": 0.075, "F0": 2.58, "Tc": 0.40},
    "CUNEO": {"lat": 44.3844, "lon": 7.5426, "provincia": "CN", "regione": "Piemonte",
              "ag": 0.100, "F0": 2.55, "Tc": 0.38},
    "NOVARA": {"lat": 45.4469, "lon": 8.6222, "provincia": "NO", "regione": "Piemonte",
               "ag": 0.050, "F0": 2.60, "Tc": 0.42},
    "VERCELLI": {"lat": 45.3258, "lon": 8.4269, "provincia": "VC", "regione": "Piemonte",
                 "ag": 0.050, "F0": 2.60, "Tc": 0.42},
    "BIELLA": {"lat": 45.5628, "lon": 8.0531, "provincia": "BI", "regione": "Piemonte",
               "ag": 0.050, "F0": 2.60, "Tc": 0.42},
    "VERBANIA": {"lat": 45.9217, "lon": 8.5517, "provincia": "VB", "regione": "Piemonte",
                 "ag": 0.050, "F0": 2.60, "Tc": 0.42},

    # LOMBARDIA
    "MILANO": {"lat": 45.4642, "lon": 9.1900, "provincia": "MI", "regione": "Lombardia",
               "ag": 0.050, "F0": 2.60, "Tc": 0.42},
    "BERGAMO": {"lat": 45.6983, "lon": 9.6773, "provincia": "BG", "regione": "Lombardia",
                "ag": 0.075, "F0": 2.58, "Tc": 0.40},
    "BRESCIA": {"lat": 45.5416, "lon": 10.2118, "provincia": "BS", "regione": "Lombardia",
                "ag": 0.125, "F0": 2.52, "Tc": 0.36},
    "COMO": {"lat": 45.8081, "lon": 9.0852, "provincia": "CO", "regione": "Lombardia",
             "ag": 0.050, "F0": 2.60, "Tc": 0.42},
    "CREMONA": {"lat": 45.1336, "lon": 10.0205, "provincia": "CR", "regione": "Lombardia",
                "ag": 0.100, "F0": 2.55, "Tc": 0.38},
    "LECCO": {"lat": 45.8566, "lon": 9.3977, "provincia": "LC", "regione": "Lombardia",
              "ag": 0.050, "F0": 2.60, "Tc": 0.42},
    "LODI": {"lat": 45.3138, "lon": 9.5034, "provincia": "LO", "regione": "Lombardia",
             "ag": 0.075, "F0": 2.58, "Tc": 0.40},
    "MANTOVA": {"lat": 45.1564, "lon": 10.7914, "provincia": "MN", "regione": "Lombardia",
                "ag": 0.125, "F0": 2.52, "Tc": 0.36},
    "MONZA": {"lat": 45.5845, "lon": 9.2744, "provincia": "MB", "regione": "Lombardia",
              "ag": 0.050, "F0": 2.60, "Tc": 0.42},
    "PAVIA": {"lat": 45.1847, "lon": 9.1582, "provincia": "PV", "regione": "Lombardia",
              "ag": 0.075, "F0": 2.58, "Tc": 0.40},
    "SONDRIO": {"lat": 46.1699, "lon": 9.8711, "provincia": "SO", "regione": "Lombardia",
                "ag": 0.075, "F0": 2.58, "Tc": 0.40},
    "VARESE": {"lat": 45.8206, "lon": 8.8257, "provincia": "VA", "regione": "Lombardia",
               "ag": 0.050, "F0": 2.60, "Tc": 0.42},

    # VENETO
    "VENEZIA": {"lat": 45.4408, "lon": 12.3155, "provincia": "VE", "regione": "Veneto",
                "ag": 0.100, "F0": 2.55, "Tc": 0.38},
    "VERONA": {"lat": 45.4384, "lon": 10.9916, "provincia": "VR", "regione": "Veneto",
               "ag": 0.125, "F0": 2.52, "Tc": 0.36},
    "VICENZA": {"lat": 45.5455, "lon": 11.5354, "provincia": "VI", "regione": "Veneto",
                "ag": 0.125, "F0": 2.52, "Tc": 0.36},
    "PADOVA": {"lat": 45.4064, "lon": 11.8768, "provincia": "PD", "regione": "Veneto",
               "ag": 0.100, "F0": 2.55, "Tc": 0.38},
    "TREVISO": {"lat": 45.6669, "lon": 12.2420, "provincia": "TV", "regione": "Veneto",
                "ag": 0.100, "F0": 2.55, "Tc": 0.38},
    "ROVIGO": {"lat": 45.0706, "lon": 11.7900, "provincia": "RO", "regione": "Veneto",
               "ag": 0.100, "F0": 2.55, "Tc": 0.38},
    "BELLUNO": {"lat": 46.1403, "lon": 12.2167, "provincia": "BL", "regione": "Veneto",
                "ag": 0.175, "F0": 2.46, "Tc": 0.33},

    # FRIULI-VENEZIA GIULIA
    "TRIESTE": {"lat": 45.6495, "lon": 13.7768, "provincia": "TS", "regione": "Friuli-Venezia Giulia",
                "ag": 0.150, "F0": 2.50, "Tc": 0.35},
    "UDINE": {"lat": 46.0711, "lon": 13.2346, "provincia": "UD", "regione": "Friuli-Venezia Giulia",
              "ag": 0.225, "F0": 2.42, "Tc": 0.31},
    "PORDENONE": {"lat": 45.9564, "lon": 12.6604, "provincia": "PN", "regione": "Friuli-Venezia Giulia",
                  "ag": 0.175, "F0": 2.46, "Tc": 0.33},
    "GORIZIA": {"lat": 45.9410, "lon": 13.6219, "provincia": "GO", "regione": "Friuli-Venezia Giulia",
                "ag": 0.175, "F0": 2.46, "Tc": 0.33},

    # TRENTINO-ALTO ADIGE
    "TRENTO": {"lat": 46.0748, "lon": 11.1217, "provincia": "TN", "regione": "Trentino-Alto Adige",
               "ag": 0.125, "F0": 2.52, "Tc": 0.36},
    "BOLZANO": {"lat": 46.4983, "lon": 11.3548, "provincia": "BZ", "regione": "Trentino-Alto Adige",
                "ag": 0.075, "F0": 2.58, "Tc": 0.40},

    # VALLE D'AOSTA
    "AOSTA": {"lat": 45.7375, "lon": 7.3155, "provincia": "AO", "regione": "Valle d'Aosta",
              "ag": 0.100, "F0": 2.55, "Tc": 0.38},

    # SARDEGNA
    "CAGLIARI": {"lat": 39.2238, "lon": 9.1217, "provincia": "CA", "regione": "Sardegna",
                 "ag": 0.050, "F0": 2.60, "Tc": 0.42},
    "SASSARI": {"lat": 40.7259, "lon": 8.5556, "provincia": "SS", "regione": "Sardegna",
                "ag": 0.050, "F0": 2.60, "Tc": 0.42},
    "NUORO": {"lat": 40.3213, "lon": 9.3313, "provincia": "NU", "regione": "Sardegna",
              "ag": 0.050, "F0": 2.60, "Tc": 0.42},
    "ORISTANO": {"lat": 39.9062, "lon": 8.5883, "provincia": "OR", "regione": "Sardegna",
                 "ag": 0.050, "F0": 2.60, "Tc": 0.42},
}

# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class SeismicLocation:
    """Localizzazione sismica del sito"""
    comune: str
    provincia: str = ""
    regione: str = ""
    latitudine: float = 0.0
    longitudine: float = 0.0
    altitudine: float = 0.0  # m s.l.m.

@dataclass
class SeismicParameters:
    """Parametri sismici del sito per un dato stato limite"""
    ag: float      # Accelerazione orizzontale massima [g]
    F0: float      # Fattore amplificazione spettrale
    Tc_star: float # Periodo inizio tratto a velocita' costante [s]

@dataclass
class SiteParameters:
    """Parametri del sito completi"""
    location: SeismicLocation
    soil_category: SoilCategory = SoilCategory.B
    topo_category: TopographicCategory = TopographicCategory.T1

@dataclass
class ProjectParameters:
    """Parametri di progetto"""
    VN: float = 50.0  # Vita nominale [anni]
    use_class: UseClass = UseClass.II

    @property
    def Cu(self) -> float:
        """Coefficiente d'uso"""
        return USE_COEFFICIENTS[self.use_class]

    @property
    def VR(self) -> float:
        """Periodo di riferimento [anni]"""
        return self.VN * self.Cu

@dataclass
class SpectrumParameters:
    """Parametri per la costruzione dello spettro di risposta"""
    ag: float       # Accelerazione di picco [g]
    F0: float       # Fattore amplificazione
    Tc_star: float  # Periodo caratteristico [s]
    SS: float       # Coefficiente amplificazione stratigrafica
    CC: float       # Coefficiente per Tc
    ST: float       # Coefficiente amplificazione topografica
    q: float = 1.0  # Fattore di struttura

    @property
    def S(self) -> float:
        """Coefficiente amplificazione totale S = SS * ST"""
        return self.SS * self.ST

    @property
    def ag_S(self) -> float:
        """Accelerazione spettrale ag*S [g]"""
        return self.ag * self.S

    @property
    def TB(self) -> float:
        """Periodo TB [s]"""
        return self.TC / 3.0

    @property
    def TC(self) -> float:
        """Periodo TC [s]"""
        return self.CC * self.Tc_star

    @property
    def TD(self) -> float:
        """Periodo TD [s]"""
        return 4.0 * self.ag + 1.6  # Formula NTC 2018

@dataclass
class ResponseSpectrum:
    """Spettro di risposta"""
    periods: np.ndarray
    accelerations: np.ndarray
    params: SpectrumParameters
    limit_state: LimitState
    is_design: bool = False  # True se spettro di progetto (con q)

    def get_Sa(self, T: float) -> float:
        """Interpolazione lineare per ottenere Sa(T)"""
        return np.interp(T, self.periods, self.accelerations)

# ============================================================================
# FUNZIONI DI CALCOLO
# ============================================================================

def get_return_period(limit_state: LimitState, VR: float) -> float:
    """
    Calcola il periodo di ritorno TR per uno stato limite.

    Formula: TR = -VR / ln(1 - PVR)

    Args:
        limit_state: Stato limite
        VR: Periodo di riferimento [anni]

    Returns:
        Periodo di ritorno TR [anni]
    """
    PVR = PVR_VALUES[limit_state]
    TR = -VR / math.log(1 - PVR)
    return TR

def interpolate_seismic_params(ag_475: float, F0_475: float, Tc_475: float,
                                TR: float) -> SeismicParameters:
    """
    Interpola i parametri sismici per un periodo di ritorno diverso da 475 anni.

    Usa le formule di interpolazione dell'Allegato A NTC 2018.

    Args:
        ag_475, F0_475, Tc_475: Valori per TR=475 anni
        TR: Periodo di ritorno desiderato

    Returns:
        SeismicParameters interpolati
    """
    # Formule semplificate di interpolazione (approssimazione)
    # In un'implementazione completa si userebbe la griglia INGV

    TR_ref = 475.0

    if TR <= 30:
        k = 0.5
    elif TR <= 50:
        k = 0.6
    elif TR <= 72:
        k = 0.7
    elif TR <= 101:
        k = 0.8
    elif TR <= 140:
        k = 0.85
    elif TR <= 201:
        k = 0.9
    elif TR <= 475:
        k = 1.0
    elif TR <= 975:
        k = 1.15
    elif TR <= 2475:
        k = 1.35
    else:
        k = 1.5

    # Interpolazione semplificata
    ag = ag_475 * k
    F0 = F0_475 * (1 + 0.05 * (k - 1))  # F0 varia poco
    Tc = Tc_475 * (1 + 0.1 * (k - 1))   # Tc varia poco

    return SeismicParameters(ag=ag, F0=F0, Tc_star=Tc)

def get_seismic_params_for_location(comune: str, limit_state: LimitState,
                                     project_params: ProjectParameters) -> SeismicParameters:
    """
    Ottiene i parametri sismici per un comune e stato limite.

    Args:
        comune: Nome del comune (case insensitive)
        limit_state: Stato limite
        project_params: Parametri di progetto (VN, classe uso)

    Returns:
        SeismicParameters per lo stato limite richiesto
    """
    comune_upper = comune.upper().strip()

    if comune_upper not in COMUNI_DATABASE:
        raise ValueError(f"Comune '{comune}' non trovato nel database. "
                        f"Comuni disponibili: {len(COMUNI_DATABASE)}")

    data = COMUNI_DATABASE[comune_upper]

    # Calcola periodo di ritorno
    TR = get_return_period(limit_state, project_params.VR)

    # Interpola parametri
    params = interpolate_seismic_params(
        data['ag'], data['F0'], data['Tc'], TR
    )

    return params

def calculate_soil_amplification(soil: SoilCategory, ag: float, F0: float) -> Tuple[float, float]:
    """
    Calcola i coefficienti di amplificazione stratigrafica SS e CC.

    Riferimento: NTC 2018 Tab. 3.2.V

    Args:
        soil: Categoria di sottosuolo
        ag: Accelerazione orizzontale massima [g]
        F0: Fattore amplificazione spettrale

    Returns:
        (SS, CC) coefficienti di amplificazione
    """
    params = SOIL_AMPLIFICATION[soil]

    if soil == SoilCategory.A:
        SS = 1.0
        CC = 1.0
    else:
        # SS = 1 + coef * (ag * F0 / g)^(-0.5), limitato a SS_max
        # Semplificato: SS dipende da ag
        SS_raw = 1.0 + params['SS_coef'] * max(0.1, (1.0 - ag / 0.5))
        SS = min(SS_raw, params['SS_max'])

        # CC per il calcolo di TC
        CC = params.get('CC_coef', 1.0) * (0.35 / max(0.35, ag * F0)) ** 0.2
        CC = max(1.0, min(CC, 1.6))  # Limiti ragionevoli

    return SS, CC

def calculate_topo_amplification(topo: TopographicCategory,
                                  location_on_ridge: float = 1.0) -> float:
    """
    Calcola il coefficiente di amplificazione topografica ST.

    Riferimento: NTC 2018 Tab. 3.2.VI

    Args:
        topo: Categoria topografica
        location_on_ridge: Posizione relativa (0=base, 1=sommita')

    Returns:
        ST coefficiente topografico
    """
    ST_max = TOPO_AMPLIFICATION[topo]
    # Interpolazione lineare tra 1.0 alla base e ST_max in sommita'
    ST = 1.0 + (ST_max - 1.0) * location_on_ridge
    return ST

def build_response_spectrum(params: SpectrumParameters,
                            limit_state: LimitState,
                            T_max: float = 4.0,
                            n_points: int = 200,
                            design: bool = False) -> ResponseSpectrum:
    """
    Costruisce lo spettro di risposta elastico o di progetto.

    Riferimento: NTC 2018 Eq. 3.2.4

    Args:
        params: Parametri spettrali
        limit_state: Stato limite
        T_max: Periodo massimo [s]
        n_points: Numero di punti
        design: Se True, applica fattore di struttura q

    Returns:
        ResponseSpectrum con periodi e accelerazioni
    """
    T = np.linspace(0.001, T_max, n_points)
    Sa = np.zeros_like(T)

    # Parametri spettrali
    ag = params.ag
    S = params.S
    F0 = params.F0
    TB = params.TB
    TC = params.TC
    TD = params.TD
    eta = 1.0  # Fattore di smorzamento (5% -> eta=1)

    # Fattore di struttura
    q = params.q if design else 1.0

    for i, Ti in enumerate(T):
        if Ti < TB:
            # Tratto a accelerazione crescente
            Se = ag * S * eta * F0 * (TB / Ti + Ti / TB * (1 / (eta * F0) - TB / Ti))
        elif Ti < TC:
            # Tratto a accelerazione costante
            Se = ag * S * eta * F0
        elif Ti < TD:
            # Tratto a velocita' costante
            Se = ag * S * eta * F0 * (TC / Ti)
        else:
            # Tratto a spostamento costante
            Se = ag * S * eta * F0 * (TC * TD / Ti**2)

        # Applica fattore di struttura per spettro di progetto
        Sa[i] = Se / q

    return ResponseSpectrum(
        periods=T,
        accelerations=Sa,
        params=params,
        limit_state=limit_state,
        is_design=design
    )

def calculate_spectrum_for_site(comune: str,
                                 site_params: SiteParameters,
                                 project_params: ProjectParameters,
                                 limit_state: LimitState = LimitState.SLV,
                                 q: float = 1.0) -> ResponseSpectrum:
    """
    Calcola lo spettro di risposta completo per un sito.

    Args:
        comune: Nome del comune
        site_params: Parametri del sito (suolo, topografia)
        project_params: Parametri di progetto (VN, classe uso)
        limit_state: Stato limite (default SLV)
        q: Fattore di struttura (default 1.0 = spettro elastico)

    Returns:
        ResponseSpectrum completo
    """
    # Ottieni parametri sismici base
    seismic = get_seismic_params_for_location(comune, limit_state, project_params)

    # Calcola amplificazioni
    SS, CC = calculate_soil_amplification(site_params.soil_category,
                                          seismic.ag, seismic.F0)
    ST = calculate_topo_amplification(site_params.topo_category)

    # Costruisci parametri spettrali
    spectrum_params = SpectrumParameters(
        ag=seismic.ag,
        F0=seismic.F0,
        Tc_star=seismic.Tc_star,
        SS=SS,
        CC=CC,
        ST=ST,
        q=q
    )

    # Genera spettro
    design = q > 1.0
    return build_response_spectrum(spectrum_params, limit_state, design=design)

def get_seismic_zone(ag: float) -> int:
    """
    Determina la zona sismica in base ad ag.

    Riferimento: OPCM 3274/2003

    Args:
        ag: Accelerazione di picco [g]

    Returns:
        Zona sismica (1, 2, 3, 4)
    """
    if ag > 0.25:
        return 1
    elif ag > 0.15:
        return 2
    elif ag > 0.05:
        return 3
    else:
        return 4

def search_comuni(query: str, limit: int = 10) -> List[str]:
    """
    Cerca comuni per nome (parziale).

    Args:
        query: Stringa di ricerca
        limit: Numero massimo di risultati

    Returns:
        Lista di nomi comuni trovati
    """
    query = query.upper().strip()
    results = []

    for comune in COMUNI_DATABASE.keys():
        if query in comune:
            results.append(comune)
            if len(results) >= limit:
                break

    return sorted(results)

def get_comuni_by_region(regione: str) -> List[str]:
    """
    Ottiene tutti i comuni di una regione.

    Args:
        regione: Nome della regione

    Returns:
        Lista di comuni nella regione
    """
    regione = regione.title()
    return [nome for nome, data in COMUNI_DATABASE.items()
            if data.get('regione', '').title() == regione]

def get_all_regions() -> List[str]:
    """Restituisce tutte le regioni nel database."""
    regions = set()
    for data in COMUNI_DATABASE.values():
        if 'regione' in data:
            regions.add(data['regione'])
    return sorted(regions)

# ============================================================================
# CLASSE PRINCIPALE
# ============================================================================

@dataclass
class SeismicAnalysis:
    """
    Classe principale per l'analisi sismica di un sito.

    Esempio d'uso:
        analysis = SeismicAnalysis(
            comune="PALERMO",
            soil=SoilCategory.B,
            topo=TopographicCategory.T1,
            VN=50,
            use_class=UseClass.II
        )

        spectrum = analysis.get_spectrum(LimitState.SLV)
        print(f"ag = {analysis.ag_SLV:.3f}g")
        print(f"Zona sismica: {analysis.seismic_zone}")
    """
    comune: str
    soil: SoilCategory = SoilCategory.B
    topo: TopographicCategory = TopographicCategory.T1
    VN: float = 50.0
    use_class: UseClass = UseClass.II
    q: float = 1.5  # Fattore di struttura per muratura non armata

    # Calcolati automaticamente
    _site_params: SiteParameters = field(init=False, repr=False)
    _project_params: ProjectParameters = field(init=False, repr=False)
    _spectra: Dict[LimitState, ResponseSpectrum] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """Inizializza parametri derivati."""
        comune_upper = self.comune.upper().strip()
        if comune_upper not in COMUNI_DATABASE:
            raise ValueError(f"Comune '{self.comune}' non trovato nel database")

        data = COMUNI_DATABASE[comune_upper]

        location = SeismicLocation(
            comune=comune_upper,
            provincia=data.get('provincia', ''),
            regione=data.get('regione', ''),
            latitudine=data.get('lat', 0.0),
            longitudine=data.get('lon', 0.0)
        )

        self._site_params = SiteParameters(
            location=location,
            soil_category=self.soil,
            topo_category=self.topo
        )

        self._project_params = ProjectParameters(
            VN=self.VN,
            use_class=self.use_class
        )

    @property
    def VR(self) -> float:
        """Periodo di riferimento [anni]"""
        return self._project_params.VR

    @property
    def Cu(self) -> float:
        """Coefficiente d'uso"""
        return self._project_params.Cu

    def get_TR(self, limit_state: LimitState) -> float:
        """Periodo di ritorno per uno stato limite [anni]"""
        return get_return_period(limit_state, self.VR)

    def get_seismic_params(self, limit_state: LimitState) -> SeismicParameters:
        """Parametri sismici per uno stato limite"""
        return get_seismic_params_for_location(
            self.comune, limit_state, self._project_params
        )

    def get_spectrum(self, limit_state: LimitState,
                     design: bool = True) -> ResponseSpectrum:
        """
        Ottiene lo spettro di risposta per uno stato limite.

        Args:
            limit_state: Stato limite
            design: Se True, usa q per spettro di progetto

        Returns:
            ResponseSpectrum
        """
        key = (limit_state, design)
        if key not in self._spectra:
            q = self.q if design else 1.0
            self._spectra[key] = calculate_spectrum_for_site(
                self.comune, self._site_params, self._project_params,
                limit_state, q
            )
        return self._spectra[key]

    @property
    def ag_SLV(self) -> float:
        """Accelerazione ag per SLV [g]"""
        return self.get_seismic_params(LimitState.SLV).ag

    @property
    def seismic_zone(self) -> int:
        """Zona sismica (1-4)"""
        return get_seismic_zone(self.ag_SLV)

    @property
    def location_info(self) -> Dict:
        """Informazioni sulla localizzazione"""
        data = COMUNI_DATABASE.get(self.comune.upper(), {})
        return {
            'comune': self.comune,
            'provincia': data.get('provincia', ''),
            'regione': data.get('regione', ''),
            'lat': data.get('lat', 0),
            'lon': data.get('lon', 0)
        }

    def summary(self) -> str:
        """Riepilogo analisi sismica"""
        lines = [
            "=" * 60,
            "ANALISI SISMICA - NTC 2018",
            "=" * 60,
            "",
            "LOCALIZZAZIONE:",
            f"  Comune: {self.comune}",
            f"  Provincia: {self.location_info['provincia']}",
            f"  Regione: {self.location_info['regione']}",
            f"  Coordinate: {self.location_info['lat']:.4f}N, {self.location_info['lon']:.4f}E",
            "",
            "PARAMETRI SITO:",
            f"  Categoria sottosuolo: {self.soil.value}",
            f"  Categoria topografica: {self.topo.value}",
            "",
            "PARAMETRI PROGETTO:",
            f"  Vita nominale VN: {self.VN} anni",
            f"  Classe d'uso: {self.use_class.name} (Cu = {self.Cu})",
            f"  Periodo riferimento VR: {self.VR:.0f} anni",
            f"  Fattore struttura q: {self.q}",
            "",
            "PERIODI DI RITORNO:",
        ]

        for ls in LimitState:
            TR = self.get_TR(ls)
            lines.append(f"  {ls.value}: TR = {TR:.0f} anni")

        lines.extend([
            "",
            "PARAMETRI SISMICI (SLV):",
        ])

        params = self.get_seismic_params(LimitState.SLV)
        spectrum = self.get_spectrum(LimitState.SLV)

        lines.extend([
            f"  ag = {params.ag:.3f} g",
            f"  F0 = {params.F0:.2f}",
            f"  Tc* = {params.Tc_star:.3f} s",
            f"  SS = {spectrum.params.SS:.3f}",
            f"  ST = {spectrum.params.ST:.3f}",
            f"  S = {spectrum.params.S:.3f}",
            "",
            f"  ZONA SISMICA: {self.seismic_zone}",
            "",
            "SPETTRO DI PROGETTO (SLV):",
            f"  TB = {spectrum.params.TB:.3f} s",
            f"  TC = {spectrum.params.TC:.3f} s",
            f"  TD = {spectrum.params.TD:.3f} s",
            f"  Sa(0) = {spectrum.get_Sa(0.001):.3f} g",
            f"  Sa(TB) = {spectrum.get_Sa(spectrum.params.TB):.3f} g",
            f"  Sa(TC) = {spectrum.get_Sa(spectrum.params.TC):.3f} g",
            f"  Sa(TD) = {spectrum.get_Sa(spectrum.params.TD):.3f} g",
            "=" * 60
        ])

        return "\n".join(lines)

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("Test modulo seismic.py")
    print()

    # Test ricerca comuni
    print("Comuni che iniziano con 'PAL':")
    for c in search_comuni("PAL"):
        print(f"  - {c}")
    print()

    # Test analisi completa
    analysis = SeismicAnalysis(
        comune="PALERMO",
        soil=SoilCategory.B,
        topo=TopographicCategory.T1,
        VN=50,
        use_class=UseClass.II,
        q=1.5
    )

    print(analysis.summary())

    # Test zone sismiche estreme
    print("\nTest zone sismiche:")
    test_comuni = ["L'AQUILA", "MILANO", "REGGIO CALABRIA", "CAGLIARI"]
    for comune in test_comuni:
        try:
            a = SeismicAnalysis(comune=comune)
            print(f"  {comune}: ag={a.ag_SLV:.3f}g, Zona {a.seismic_zone}")
        except ValueError as e:
            print(f"  {comune}: {e}")
