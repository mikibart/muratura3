# loads.py - Modulo Carichi Neve e Vento
"""
Calcolo carichi neve e vento secondo NTC 2018.

NTC 2018 Cap. 3.3 - Azione della neve
NTC 2018 Cap. 3.4 - Azione del vento

"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum
import math


# ============================================================================
# CARICO NEVE (NTC 2018 - 3.4)
# ============================================================================

class SnowZone(Enum):
    """Zone neve secondo NTC 2018"""
    I_ALPINA = "I-Alpina"       # Aosta, Belluno, Bergamo, Biella, Bolzano, Brescia, Como,
                                # Cuneo, Lecco, Pordenone, Sondrio, Torino, Trento,
                                # Udine, Verbania, Vercelli, Vicenza
    I_MEDITERRANEA = "I-Med"    # Alessandria, Ancona, Asti, Bologna, Ferrara, Forli,
                                # Gorizia, Modena, Novara, Padova, Parma, Pavia, Pesaro,
                                # Piacenza, Ravenna, Reggio Emilia, Rimini, Rovigo, Treviso,
                                # Trieste, Venezia, Verona
    II = "II"                   # Arezzo, Ascoli, Avellino, Bari, Campobasso, Chieti, Firenze,
                                # Foggia, Frosinone, Isernia, L'Aquila, Macerata, Matera,
                                # Perugia, Pescara, Potenza, Rieti, Siena, Taranto, Teramo,
                                # Terni, Viterbo
    III = "III"                 # Agrigento, Barletta, Benevento, Brindisi, Cagliari, Caltanissetta,
                                # Caserta, Catania, Catanzaro, Cosenza, Crotone, Enna, Genova,
                                # Grosseto, Imperia, La Spezia, Latina, Lecce, Livorno, Lucca,
                                # Massa, Messina, Napoli, Nuoro, Oristano, Palermo, Pisa, Pistoia,
                                # Prato, Ragusa, Reggio Calabria, Roma, Salerno, Sassari, Savona,
                                # Siracusa, Sud Sardegna, Trapani, Vibo Valentia


# Formule carico neve al suolo qsk [kN/m2] in funzione di as (altitudine)
# NTC 2018 Tabella 3.4.I
def calc_qsk(zone: SnowZone, altitude: float) -> float:
    """
    Calcola il carico neve al suolo qsk [kN/m2]

    Args:
        zone: Zona neve
        altitude: Altitudine sul livello del mare [m]

    Returns:
        qsk in kN/m2
    """
    as_m = altitude  # altitudine in metri

    if zone == SnowZone.I_ALPINA:
        # qsk = 1.50 kN/m2  as <= 200m
        # qsk = 1.39 [1 + (as/728)^2]  as > 200m
        if as_m <= 200:
            return 1.50
        else:
            return 1.39 * (1 + (as_m / 728) ** 2)

    elif zone == SnowZone.I_MEDITERRANEA:
        # qsk = 1.50 kN/m2  as <= 200m
        # qsk = 1.35 [1 + (as/602)^2]  as > 200m
        if as_m <= 200:
            return 1.50
        else:
            return 1.35 * (1 + (as_m / 602) ** 2)

    elif zone == SnowZone.II:
        # qsk = 1.00 kN/m2  as <= 200m
        # qsk = 0.85 [1 + (as/481)^2]  as > 200m
        if as_m <= 200:
            return 1.00
        else:
            return 0.85 * (1 + (as_m / 481) ** 2)

    elif zone == SnowZone.III:
        # qsk = 0.60 kN/m2  as <= 200m
        # qsk = 0.51 [1 + (as/481)^2]  as > 200m
        if as_m <= 200:
            return 0.60
        else:
            return 0.51 * (1 + (as_m / 481) ** 2)

    return 1.50  # Default sicuro


class SnowExposure(Enum):
    """Coefficiente di esposizione CE (NTC 2018 Tab. 3.4.II)"""
    BATTUTA_VENTI = ("Battuta dai venti", 0.9)      # Zona aperta, spazzata dal vento
    NORMALE = ("Normale", 1.0)                       # Condizioni ordinarie
    RIPARATA = ("Riparata", 1.1)                    # Zona riparata, accumulo neve

    def __init__(self, description: str, value: float):
        self.description = description
        self._value = value

    @property
    def CE(self) -> float:
        return self._value


class SnowThermal(Enum):
    """Coefficiente termico Ct (NTC 2018 par. 3.4.4)"""
    STANDARD = ("Standard (non riscaldato)", 1.0)
    RISCALDATO = ("Copertura riscaldata", 0.8)

    def __init__(self, description: str, value: float):
        self.description = description
        self._value = value

    @property
    def Ct(self) -> float:
        return self._value


@dataclass
class SnowLoad:
    """Calcolo carico neve secondo NTC 2018"""
    zone: SnowZone
    altitude: float                          # Altitudine [m]
    exposure: SnowExposure = SnowExposure.NORMALE
    thermal: SnowThermal = SnowThermal.STANDARD
    roof_slope: float = 0.0                  # Pendenza falda [gradi]

    # Coefficienti forma (semplificati per coperture semplici)
    mu1: float = 0.8                         # Copertura piana

    @property
    def qsk(self) -> float:
        """Carico neve al suolo [kN/m2]"""
        return calc_qsk(self.zone, self.altitude)

    @property
    def CE(self) -> float:
        """Coefficiente esposizione"""
        return self.exposure.CE

    @property
    def Ct(self) -> float:
        """Coefficiente termico"""
        return self.thermal.Ct

    @property
    def mu(self) -> float:
        """Coefficiente di forma (semplificato)"""
        # Per coperture a una o due falde
        alpha = self.roof_slope

        if alpha <= 30:
            return 0.8
        elif alpha < 60:
            return 0.8 * (60 - alpha) / 30
        else:
            return 0.0

    @property
    def qs(self) -> float:
        """Carico neve sulla copertura [kN/m2]"""
        return self.mu * self.qsk * self.CE * self.Ct

    def summary(self) -> str:
        """Riepilogo calcolo neve"""
        lines = [
            "=== CARICO NEVE (NTC 2018) ===",
            f"Zona: {self.zone.value}",
            f"Altitudine: {self.altitude:.0f} m s.l.m.",
            f"",
            f"Parametri:",
            f"  qsk = {self.qsk:.2f} kN/m2 (carico al suolo)",
            f"  CE  = {self.CE:.1f} ({self.exposure.description})",
            f"  Ct  = {self.Ct:.1f} ({self.thermal.description})",
            f"  mu  = {self.mu:.2f} (pendenza {self.roof_slope:.0f})",
            f"",
            f"CARICO NEVE: qs = {self.qs:.2f} kN/m2",
        ]
        return "\n".join(lines)


# ============================================================================
# CARICO VENTO (NTC 2018 - 3.3)
# ============================================================================

class WindZone(Enum):
    """Zone vento secondo NTC 2018 Tab. 3.3.I"""
    # Zona, vb0 [m/s], a0 [m], ka [1/s]
    ZONE_1 = (1, 25, 1000, 0.010)  # Valle d'Aosta, Piemonte, Lombardia, Trentino-Alto Adige,
                                   # Veneto, Friuli-Venezia Giulia (tranne provincia Trieste)
    ZONE_2 = (2, 25, 1000, 0.015)  # Emilia-Romagna
    ZONE_3 = (3, 27, 1500, 0.010)  # Toscana, Marche, Umbria, Lazio, Abruzzo, Molise,
                                   # Puglia, Campania, Basilicata, Calabria (tranne Reggio Calabria)
    ZONE_4 = (4, 28, 1500, 0.010)  # Sicilia e provincia Reggio Calabria
    ZONE_5 = (5, 28, 1500, 0.015)  # Sardegna (tranne zona 6), provincia Trieste
    ZONE_6 = (6, 28, 1500, 0.020)  # Cagliari e costa orientale Sardegna (Ogliastra, Olbia-Tempio)
    ZONE_7 = (7, 29, 1500, 0.015)  # Liguria
    ZONE_8 = (8, 31, 1500, 0.015)  # Isole: Lampedusa, Linosa, Pantelleria
    ZONE_9 = (9, 31, 1500, 0.020)  # Isole: Ponza, Palmarola

    def __init__(self, zone_num: int, vb0: float, a0: float, ka: float):
        self.zone_num = zone_num
        self.vb0 = vb0   # velocita' base riferimento [m/s]
        self.a0 = a0     # altitudine riferimento [m]
        self.ka = ka     # coefficiente [1/s]


class ExposureCategory(Enum):
    """Categorie di esposizione (NTC 2018 Tab. 3.3.II)"""
    # Categoria, kr, z0 [m], zmin [m]
    I = (1, 0.17, 0.01, 2)     # Mare aperto, laghi con >= 5km fetch, costa piatta
    II = (2, 0.19, 0.05, 4)    # Aree agricole con vegetazione bassa, ostacoli isolati
    III = (3, 0.20, 0.10, 5)   # Aree suburbane, industriali, boschive
    IV = (4, 0.22, 0.30, 8)    # Aree urbane in cui >= 15% superficie coperta edifici > 15m
    V = (5, 0.23, 0.70, 12)    # Centri di citta' con >= 15% superficie coperta edifici > 15m

    def __init__(self, cat: int, kr: float, z0: float, zmin: float):
        self.cat = cat
        self.kr = kr
        self.z0 = z0
        self.zmin = zmin


class TopographicClass(Enum):
    """Classi topografiche (NTC 2018 par. 3.3.7)"""
    PIANEGGIANTE = ("Pianeggiante o leggermente ondulato", 1.0)
    PENDII_DOLCI = ("Pendii dolci", 1.0)
    COLLINE = ("Rilievi collinari", 1.1)
    VALLI_STRETTE = ("Valli strette", 1.2)

    def __init__(self, description: str, ct: float):
        self.description = description
        self._ct = ct

    @property
    def ct(self) -> float:
        return self._ct


@dataclass
class WindLoad:
    """Calcolo carico vento secondo NTC 2018"""
    zone: WindZone
    altitude: float                          # Altitudine sito [m]
    building_height: float                   # Altezza edificio [m]
    exposure: ExposureCategory = ExposureCategory.III
    topography: TopographicClass = TopographicClass.PIANEGGIANTE

    # Coefficienti pressione (semplificati per edifici rettangolari)
    cpe_sopravento: float = 0.8              # Parete sopravento
    cpe_sottovento: float = -0.4             # Parete sottovento
    cpi: float = 0.2                         # Pressione interna (+-0.2)

    @property
    def vb(self) -> float:
        """Velocita' di riferimento vb [m/s]"""
        vb0 = self.zone.vb0
        a0 = self.zone.a0
        ka = self.zone.ka

        if self.altitude <= a0:
            return vb0
        else:
            return vb0 + ka * (self.altitude - a0)

    @property
    def qb(self) -> float:
        """Pressione cinetica di riferimento qb [kN/m2]"""
        # qb = 0.5 * rho * vb^2
        # rho = 1.25 kg/m3 (aria)
        rho = 1.25
        return 0.5 * rho * self.vb ** 2 / 1000  # kN/m2

    @property
    def ce(self) -> float:
        """Coefficiente di esposizione ce(z)"""
        z = max(self.building_height, self.exposure.zmin)
        kr = self.exposure.kr
        z0 = self.exposure.z0
        ct = self.topography.ct

        # ce(z) = kr^2 * ct * ln(z/z0) * [7 + ct * ln(z/z0)]
        ln_ratio = math.log(z / z0)
        return kr ** 2 * ct * ln_ratio * (7 + ct * ln_ratio)

    @property
    def cd(self) -> float:
        """Coefficiente dinamico cd (semplificato)"""
        # Per edifici normali cd = 1
        # Per edifici alti e snelli serve analisi specifica
        if self.building_height <= 50:
            return 1.0
        else:
            return 1.0  # Semplificazione, richiede analisi

    @property
    def p_sopravento(self) -> float:
        """Pressione parete sopravento [kN/m2]"""
        cp = self.cpe_sopravento - self.cpi
        return self.qb * self.ce * cp * self.cd

    @property
    def p_sottovento(self) -> float:
        """Pressione parete sottovento [kN/m2] (suzione)"""
        cp = self.cpe_sottovento + self.cpi
        return self.qb * self.ce * cp * self.cd

    @property
    def p_totale(self) -> float:
        """Pressione totale netta [kN/m2]"""
        # Per edificio chiuso: somma sopravento + |sottovento|
        return self.p_sopravento + abs(self.p_sottovento)

    @property
    def p(self) -> float:
        """Pressione vento di progetto [kN/m2]"""
        return self.qb * self.ce * self.cd

    def summary(self) -> str:
        """Riepilogo calcolo vento"""
        lines = [
            "=== CARICO VENTO (NTC 2018) ===",
            f"Zona: {self.zone.zone_num}",
            f"Altitudine sito: {self.altitude:.0f} m s.l.m.",
            f"Altezza edificio: {self.building_height:.1f} m",
            f"",
            f"Parametri:",
            f"  vb  = {self.vb:.1f} m/s (velocita' riferimento)",
            f"  qb  = {self.qb:.3f} kN/m2 (pressione cinetica)",
            f"  ce  = {self.ce:.2f} (esposizione cat. {self.exposure.cat})",
            f"  cd  = {self.cd:.2f} (dinamico)",
            f"",
            f"Pressioni:",
            f"  Sopravento: p = {self.p_sopravento:.3f} kN/m2",
            f"  Sottovento: p = {self.p_sottovento:.3f} kN/m2",
            f"",
            f"PRESSIONE VENTO: p = {self.p:.3f} kN/m2",
        ]
        return "\n".join(lines)


# ============================================================================
# DATABASE ZONE PER PROVINCIA
# ============================================================================

# Mapping provincia -> (zona neve, zona vento)
PROVINCE_ZONES = {
    # Valle d'Aosta
    'Aosta': (SnowZone.I_ALPINA, WindZone.ZONE_1),

    # Piemonte
    'Torino': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Cuneo': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Alessandria': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Asti': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Biella': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Novara': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Verbano-Cusio-Ossola': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Vercelli': (SnowZone.I_ALPINA, WindZone.ZONE_1),

    # Lombardia
    'Milano': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Bergamo': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Brescia': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Como': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Cremona': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Lecco': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Lodi': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Mantova': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Monza e Brianza': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Pavia': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Sondrio': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Varese': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),

    # Trentino-Alto Adige
    'Trento': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Bolzano': (SnowZone.I_ALPINA, WindZone.ZONE_1),

    # Veneto
    'Venezia': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Verona': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Vicenza': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Belluno': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Padova': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Rovigo': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Treviso': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),

    # Friuli-Venezia Giulia
    'Udine': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Pordenone': (SnowZone.I_ALPINA, WindZone.ZONE_1),
    'Gorizia': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_1),
    'Trieste': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_5),

    # Liguria
    'Genova': (SnowZone.III, WindZone.ZONE_7),
    'Imperia': (SnowZone.III, WindZone.ZONE_7),
    'La Spezia': (SnowZone.III, WindZone.ZONE_7),
    'Savona': (SnowZone.III, WindZone.ZONE_7),

    # Emilia-Romagna
    'Bologna': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_2),
    'Ferrara': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_2),
    "Forli'-Cesena": (SnowZone.I_MEDITERRANEA, WindZone.ZONE_2),
    'Modena': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_2),
    'Parma': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_2),
    'Piacenza': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_2),
    'Ravenna': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_2),
    'Reggio Emilia': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_2),
    'Rimini': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_2),

    # Toscana
    'Firenze': (SnowZone.II, WindZone.ZONE_3),
    'Arezzo': (SnowZone.II, WindZone.ZONE_3),
    'Grosseto': (SnowZone.III, WindZone.ZONE_3),
    'Livorno': (SnowZone.III, WindZone.ZONE_3),
    'Lucca': (SnowZone.III, WindZone.ZONE_3),
    'Massa-Carrara': (SnowZone.III, WindZone.ZONE_3),
    'Pisa': (SnowZone.III, WindZone.ZONE_3),
    'Pistoia': (SnowZone.III, WindZone.ZONE_3),
    'Prato': (SnowZone.III, WindZone.ZONE_3),
    'Siena': (SnowZone.II, WindZone.ZONE_3),

    # Marche
    'Ancona': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_3),
    'Ascoli Piceno': (SnowZone.II, WindZone.ZONE_3),
    'Fermo': (SnowZone.II, WindZone.ZONE_3),
    'Macerata': (SnowZone.II, WindZone.ZONE_3),
    'Pesaro e Urbino': (SnowZone.I_MEDITERRANEA, WindZone.ZONE_3),

    # Umbria
    'Perugia': (SnowZone.II, WindZone.ZONE_3),
    'Terni': (SnowZone.II, WindZone.ZONE_3),

    # Lazio
    'Roma': (SnowZone.III, WindZone.ZONE_3),
    'Frosinone': (SnowZone.II, WindZone.ZONE_3),
    'Latina': (SnowZone.III, WindZone.ZONE_3),
    'Rieti': (SnowZone.II, WindZone.ZONE_3),
    'Viterbo': (SnowZone.II, WindZone.ZONE_3),

    # Abruzzo
    "L'Aquila": (SnowZone.II, WindZone.ZONE_3),
    'Chieti': (SnowZone.II, WindZone.ZONE_3),
    'Pescara': (SnowZone.II, WindZone.ZONE_3),
    'Teramo': (SnowZone.II, WindZone.ZONE_3),

    # Molise
    'Campobasso': (SnowZone.II, WindZone.ZONE_3),
    'Isernia': (SnowZone.II, WindZone.ZONE_3),

    # Campania
    'Napoli': (SnowZone.III, WindZone.ZONE_3),
    'Avellino': (SnowZone.II, WindZone.ZONE_3),
    'Benevento': (SnowZone.III, WindZone.ZONE_3),
    'Caserta': (SnowZone.III, WindZone.ZONE_3),
    'Salerno': (SnowZone.III, WindZone.ZONE_3),

    # Puglia
    'Bari': (SnowZone.II, WindZone.ZONE_3),
    'Barletta-Andria-Trani': (SnowZone.III, WindZone.ZONE_3),
    'Brindisi': (SnowZone.III, WindZone.ZONE_3),
    'Foggia': (SnowZone.II, WindZone.ZONE_3),
    'Lecce': (SnowZone.III, WindZone.ZONE_3),
    'Taranto': (SnowZone.II, WindZone.ZONE_3),

    # Basilicata
    'Potenza': (SnowZone.II, WindZone.ZONE_3),
    'Matera': (SnowZone.II, WindZone.ZONE_3),

    # Calabria
    'Catanzaro': (SnowZone.III, WindZone.ZONE_3),
    'Cosenza': (SnowZone.III, WindZone.ZONE_3),
    'Crotone': (SnowZone.III, WindZone.ZONE_3),
    'Reggio Calabria': (SnowZone.III, WindZone.ZONE_4),
    'Vibo Valentia': (SnowZone.III, WindZone.ZONE_3),

    # Sicilia
    'Palermo': (SnowZone.III, WindZone.ZONE_4),
    'Agrigento': (SnowZone.III, WindZone.ZONE_4),
    'Caltanissetta': (SnowZone.III, WindZone.ZONE_4),
    'Catania': (SnowZone.III, WindZone.ZONE_4),
    'Enna': (SnowZone.III, WindZone.ZONE_4),
    'Messina': (SnowZone.III, WindZone.ZONE_4),
    'Ragusa': (SnowZone.III, WindZone.ZONE_4),
    'Siracusa': (SnowZone.III, WindZone.ZONE_4),
    'Trapani': (SnowZone.III, WindZone.ZONE_4),

    # Sardegna
    'Cagliari': (SnowZone.III, WindZone.ZONE_6),
    'Nuoro': (SnowZone.III, WindZone.ZONE_5),
    'Oristano': (SnowZone.III, WindZone.ZONE_5),
    'Sassari': (SnowZone.III, WindZone.ZONE_5),
    'Sud Sardegna': (SnowZone.III, WindZone.ZONE_6),
}


def get_zones_by_province(provincia: str) -> Optional[Tuple[SnowZone, WindZone]]:
    """Restituisce le zone neve e vento per una provincia"""
    # Normalizza nome provincia
    provincia = provincia.strip()

    # Cerca corrispondenza esatta
    if provincia in PROVINCE_ZONES:
        return PROVINCE_ZONES[provincia]

    # Cerca corrispondenza parziale
    provincia_lower = provincia.lower()
    for prov, zones in PROVINCE_ZONES.items():
        if provincia_lower in prov.lower() or prov.lower() in provincia_lower:
            return zones

    return None


# ============================================================================
# FUNZIONI DI UTILITA'
# ============================================================================

def calcola_carichi_climatici(
    provincia: str,
    altitudine: float,
    altezza_edificio: float,
    pendenza_copertura: float = 0.0,
    exposure_snow: SnowExposure = SnowExposure.NORMALE,
    exposure_wind: ExposureCategory = ExposureCategory.III,
) -> Tuple[SnowLoad, WindLoad]:
    """
    Calcola carichi neve e vento per una localita'.

    Args:
        provincia: Nome provincia
        altitudine: Altitudine [m s.l.m.]
        altezza_edificio: Altezza edificio [m]
        pendenza_copertura: Pendenza falda [gradi]
        exposure_snow: Esposizione neve
        exposure_wind: Categoria esposizione vento

    Returns:
        Tupla (SnowLoad, WindLoad)
    """
    zones = get_zones_by_province(provincia)
    if zones is None:
        # Default: zona II neve, zona 3 vento
        zones = (SnowZone.II, WindZone.ZONE_3)

    snow_zone, wind_zone = zones

    snow = SnowLoad(
        zone=snow_zone,
        altitude=altitudine,
        exposure=exposure_snow,
        roof_slope=pendenza_copertura
    )

    wind = WindLoad(
        zone=wind_zone,
        altitude=altitudine,
        building_height=altezza_edificio,
        exposure=exposure_wind
    )

    return snow, wind


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test carico neve
    print("=== TEST CARICO NEVE ===")
    snow = SnowLoad(
        zone=SnowZone.I_MEDITERRANEA,
        altitude=300,
        exposure=SnowExposure.NORMALE,
        roof_slope=20
    )
    print(snow.summary())
    print()

    # Test carico vento
    print("=== TEST CARICO VENTO ===")
    wind = WindLoad(
        zone=WindZone.ZONE_3,
        altitude=200,
        building_height=12,
        exposure=ExposureCategory.III
    )
    print(wind.summary())
    print()

    # Test funzione utilita'
    print("=== TEST CARICHI PER LOCALITA' ===")
    snow, wind = calcola_carichi_climatici(
        provincia="Roma",
        altitudine=50,
        altezza_edificio=9,
        pendenza_copertura=15
    )
    print(f"Roma, 50m s.l.m., edificio 9m:")
    print(f"  Neve: qs = {snow.qs:.2f} kN/m2")
    print(f"  Vento: p = {wind.p:.3f} kN/m2")
