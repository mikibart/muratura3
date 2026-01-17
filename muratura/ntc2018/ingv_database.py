# -*- coding: utf-8 -*-
"""
Database parametri sismici INGV.

Contiene parametri di pericolosità per i principali comuni italiani.
Interpolazione da reticolo INGV per coordinate personalizzate.
"""

import math
from typing import Dict, Tuple, Optional

# Parametri sismici per comuni principali
# (ag_SLO, F0_SLO, ag_SLD, F0_SLD, ag_SLV, F0_SLV, ag_SLC, F0_SLC)
# ag in g, F0 adimensionale
COMUNI_DATABASE = {
    "roma": {
        "lat": 41.9028, "lon": 12.4964,
        "ag": [0.055, 0.069, 0.141, 0.177],
        "F0": [2.45, 2.44, 2.44, 2.42],
        "Tc_star": [0.28, 0.29, 0.32, 0.33],
    },
    "milano": {
        "lat": 45.4642, "lon": 9.1900,
        "ag": [0.024, 0.030, 0.059, 0.073],
        "F0": [2.50, 2.49, 2.47, 2.46],
        "Tc_star": [0.24, 0.25, 0.27, 0.28],
    },
    "napoli": {
        "lat": 40.8518, "lon": 14.2681,
        "ag": [0.071, 0.089, 0.168, 0.207],
        "F0": [2.40, 2.40, 2.42, 2.41],
        "Tc_star": [0.30, 0.31, 0.34, 0.35],
    },
    "l'aquila": {
        "lat": 42.3498, "lon": 13.3995,
        "ag": [0.104, 0.131, 0.261, 0.321],
        "F0": [2.35, 2.36, 2.39, 2.38],
        "Tc_star": [0.33, 0.34, 0.37, 0.38],
    },
    "firenze": {
        "lat": 43.7696, "lon": 11.2558,
        "ag": [0.049, 0.062, 0.126, 0.156],
        "F0": [2.45, 2.44, 2.43, 2.42],
        "Tc_star": [0.27, 0.28, 0.31, 0.32],
    },
    "bologna": {
        "lat": 44.4949, "lon": 11.3426,
        "ag": [0.060, 0.076, 0.150, 0.185],
        "F0": [2.43, 2.43, 2.42, 2.41],
        "Tc_star": [0.28, 0.29, 0.32, 0.33],
    },
    "palermo": {
        "lat": 38.1157, "lon": 13.3615,
        "ag": [0.062, 0.078, 0.153, 0.189],
        "F0": [2.41, 2.41, 2.41, 2.40],
        "Tc_star": [0.29, 0.30, 0.33, 0.34],
    },
    "catania": {
        "lat": 37.5079, "lon": 15.0830,
        "ag": [0.102, 0.129, 0.256, 0.315],
        "F0": [2.37, 2.38, 2.40, 2.39],
        "Tc_star": [0.32, 0.33, 0.36, 0.37],
    },
    "torino": {
        "lat": 45.0703, "lon": 7.6869,
        "ag": [0.035, 0.044, 0.087, 0.107],
        "F0": [2.48, 2.47, 2.45, 2.44],
        "Tc_star": [0.26, 0.27, 0.29, 0.30],
    },
    "genova": {
        "lat": 44.4056, "lon": 8.9463,
        "ag": [0.032, 0.040, 0.079, 0.097],
        "F0": [2.48, 2.47, 2.46, 2.45],
        "Tc_star": [0.25, 0.26, 0.28, 0.29],
    },
    "venezia": {
        "lat": 45.4408, "lon": 12.3155,
        "ag": [0.045, 0.057, 0.113, 0.140],
        "F0": [2.46, 2.45, 2.44, 2.43],
        "Tc_star": [0.27, 0.28, 0.30, 0.31],
    },
    "bari": {
        "lat": 41.1171, "lon": 16.8719,
        "ag": [0.043, 0.054, 0.108, 0.133],
        "F0": [2.46, 2.46, 2.44, 2.43],
        "Tc_star": [0.26, 0.27, 0.30, 0.31],
    },
    "perugia": {
        "lat": 43.1107, "lon": 12.3908,
        "ag": [0.082, 0.103, 0.203, 0.250],
        "F0": [2.39, 2.40, 2.41, 2.40],
        "Tc_star": [0.31, 0.32, 0.35, 0.36],
    },
    "ancona": {
        "lat": 43.6158, "lon": 13.5189,
        "ag": [0.075, 0.095, 0.186, 0.229],
        "F0": [2.40, 2.41, 2.42, 2.41],
        "Tc_star": [0.30, 0.31, 0.34, 0.35],
    },
    "reggio calabria": {
        "lat": 38.1089, "lon": 15.6436,
        "ag": [0.135, 0.170, 0.336, 0.414],
        "F0": [2.32, 2.34, 2.37, 2.36],
        "Tc_star": [0.35, 0.36, 0.39, 0.40],
    },
    "messina": {
        "lat": 38.1937, "lon": 15.5542,
        "ag": [0.130, 0.164, 0.324, 0.399],
        "F0": [2.33, 2.35, 2.38, 2.37],
        "Tc_star": [0.34, 0.35, 0.38, 0.39],
    },
    "potenza": {
        "lat": 40.6404, "lon": 15.8054,
        "ag": [0.095, 0.120, 0.237, 0.292],
        "F0": [2.38, 2.39, 2.40, 2.39],
        "Tc_star": [0.32, 0.33, 0.36, 0.37],
    },
    "campobasso": {
        "lat": 41.5603, "lon": 14.6621,
        "ag": [0.088, 0.111, 0.219, 0.270],
        "F0": [2.39, 2.40, 2.41, 2.40],
        "Tc_star": [0.31, 0.32, 0.35, 0.36],
    },
    "trento": {
        "lat": 46.0748, "lon": 11.1217,
        "ag": [0.051, 0.064, 0.127, 0.157],
        "F0": [2.45, 2.44, 2.43, 2.42],
        "Tc_star": [0.27, 0.28, 0.31, 0.32],
    },
    "trieste": {
        "lat": 45.6495, "lon": 13.7768,
        "ag": [0.070, 0.088, 0.174, 0.214],
        "F0": [2.41, 2.42, 2.42, 2.41],
        "Tc_star": [0.29, 0.30, 0.33, 0.34],
    },
    "aosta": {
        "lat": 45.7375, "lon": 7.3150,
        "ag": [0.042, 0.053, 0.105, 0.130],
        "F0": [2.47, 2.46, 2.45, 2.44],
        "Tc_star": [0.26, 0.27, 0.29, 0.30],
    },
}

# Stati limite con relativi TR
STATI_LIMITE = {
    "SLO": {"TR": 30, "index": 0},
    "SLD": {"TR": 50, "index": 1},
    "SLV": {"TR": 475, "index": 2},
    "SLC": {"TR": 975, "index": 3},
}


def get_seismic_params(comune: str = None, lat: float = None, lon: float = None) -> Dict:
    """
    Ottiene i parametri sismici per un comune o coordinate.

    Args:
        comune: Nome del comune (case insensitive)
        lat: Latitudine (se comune non specificato)
        lon: Longitudine (se comune non specificato)

    Returns:
        Dizionario con parametri per ogni stato limite
    """
    if comune:
        comune_lower = comune.lower().strip()
        if comune_lower in COMUNI_DATABASE:
            data = COMUNI_DATABASE[comune_lower]
            return _format_params(data, comune.title())

        # Cerca corrispondenza parziale
        for key, data in COMUNI_DATABASE.items():
            if comune_lower in key or key in comune_lower:
                return _format_params(data, key.title())

    if lat is not None and lon is not None:
        # Interpola dai comuni più vicini
        return _interpolate_params(lat, lon)

    # Default: valori medi Italia centrale
    return _format_params({
        "lat": 42.0, "lon": 12.5,
        "ag": [0.070, 0.088, 0.175, 0.215],
        "F0": [2.42, 2.42, 2.42, 2.41],
        "Tc_star": [0.29, 0.30, 0.33, 0.34],
    }, "Default")


def _format_params(data: Dict, nome: str) -> Dict:
    """Formatta i parametri nel formato standard."""
    result = {
        "comune": nome,
        "lat": data["lat"],
        "lon": data["lon"],
        "stati_limite": {}
    }

    for sl, info in STATI_LIMITE.items():
        idx = info["index"]
        result["stati_limite"][sl] = {
            "TR": info["TR"],
            "ag": round(data["ag"][idx], 3),
            "F0": round(data["F0"][idx], 2),
            "Tc_star": round(data["Tc_star"][idx], 2),
        }

    return result


def _interpolate_params(lat: float, lon: float) -> Dict:
    """Interpola parametri da comuni vicini."""
    # Trova i 3 comuni più vicini
    distances = []
    for nome, data in COMUNI_DATABASE.items():
        d = _haversine(lat, lon, data["lat"], data["lon"])
        distances.append((d, nome, data))

    distances.sort(key=lambda x: x[0])
    nearest = distances[:3]

    # Interpolazione pesata inversa alla distanza
    total_weight = sum(1/d[0] for d in nearest if d[0] > 0)

    ag = [0, 0, 0, 0]
    F0 = [0, 0, 0, 0]
    Tc_star = [0, 0, 0, 0]

    for d, nome, data in nearest:
        if d > 0:
            weight = (1/d) / total_weight
            for i in range(4):
                ag[i] += data["ag"][i] * weight
                F0[i] += data["F0"][i] * weight
                Tc_star[i] += data["Tc_star"][i] * weight

    return _format_params({
        "lat": lat,
        "lon": lon,
        "ag": ag,
        "F0": F0,
        "Tc_star": Tc_star,
    }, f"Coord ({lat:.4f}, {lon:.4f})")


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcola distanza tra due punti (km)."""
    R = 6371  # Raggio Terra in km

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


def calculate_response_spectrum(
    ag: float,
    F0: float,
    Tc_star: float,
    soil_category: str = "C",
    topo_category: str = "T1",
    damping: float = 5.0,
    T_range: Tuple[float, float] = (0, 4.0),
    n_points: int = 100
) -> Dict:
    """
    Calcola lo spettro di risposta NTC 2018.

    Args:
        ag: Accelerazione di ancoraggio [g]
        F0: Fattore di amplificazione
        Tc_star: Periodo inizio tratto a velocità costante [s]
        soil_category: Categoria sottosuolo (A-E)
        topo_category: Categoria topografica (T1-T4)
        damping: Smorzamento [%]
        T_range: Range periodi [s]
        n_points: Numero punti spettro

    Returns:
        Dizionario con periodi T e accelerazioni Se
    """
    # Coefficienti sottosuolo NTC Tab. 3.2.II
    SOIL_COEFFS = {
        "A": (1.00, 1.00),
        "B": (1.20, 1.10),
        "C": (1.50, 1.05),
        "D": (1.80, 1.25),
        "E": (1.60, 1.15),
    }

    # Coefficienti topografici NTC Tab. 3.2.IV
    TOPO_COEFFS = {
        "T1": 1.0,
        "T2": 1.2,
        "T3": 1.2,
        "T4": 1.4,
    }

    Ss, Cc = SOIL_COEFFS.get(soil_category, (1.5, 1.05))
    St = TOPO_COEFFS.get(topo_category, 1.0)

    S = Ss * St

    # Periodi caratteristici
    Tb = Tc_star * Cc / 3
    Tc = Tc_star * Cc
    Td = 4.0 * ag + 1.6

    # Fattore smorzamento
    eta = max(0.55, math.sqrt(10 / (5 + damping)))

    # Genera spettro
    T_values = []
    Se_values = []

    for i in range(n_points):
        T = T_range[0] + (T_range[1] - T_range[0]) * i / (n_points - 1)
        T_values.append(T)

        if T < Tb:
            Se = ag * S * eta * F0 * (T/Tb + 1/(eta*F0) * (1 - T/Tb))
        elif T < Tc:
            Se = ag * S * eta * F0
        elif T < Td:
            Se = ag * S * eta * F0 * (Tc/T)
        else:
            Se = ag * S * eta * F0 * (Tc*Td/(T**2))

        Se_values.append(round(Se, 4))

    return {
        "T": T_values,
        "Se": Se_values,
        "params": {
            "ag": ag,
            "F0": F0,
            "S": S,
            "Ss": Ss,
            "St": St,
            "Tb": round(Tb, 3),
            "Tc": round(Tc, 3),
            "Td": round(Td, 3),
            "eta": round(eta, 3),
        }
    }
