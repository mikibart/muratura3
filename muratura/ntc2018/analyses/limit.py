# analyses/limit.py
import numpy as np
import copy
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize, minimize_scalar
from ..enums import KinematicMechanism
from ..materials import MaterialProperties
from ..utils import probabilistic_limit_analysis, sensitivity_analysis_limit
import logging

logger = logging.getLogger(__name__)

@dataclass
class LimitAnalysis:
    """Modello per analisi limite con meccanismi cinematici EC8-3"""
    geometry: Dict
    material: MaterialProperties
    
    def analyze_all_mechanisms(self, loads: Dict) -> Dict:
        """Analizza tutti i 24 meccanismi e trova il minimo alpha"""
        results = {}
        mechanism_results = {}
        min_alpha = float('inf')
        governing_mechanism = None
        
        # Analizza ogni meccanismo
        for mech in KinematicMechanism:
            try:
                alpha = self._analyze_mechanism(mech, loads)
                mechanism_results[mech.value] = {
                    'alpha': alpha,
                    'description': self._get_mechanism_description(mech),
                    'critical': False
                }
                
                if alpha < min_alpha and alpha > 0:
                    min_alpha = alpha
                    governing_mechanism = mech.value
                    
            except Exception as e:
                logger.error(f"Errore analisi meccanismo {mech.value}: {str(e)}")
                mechanism_results[mech.value] = {
                    'alpha': float('inf'),
                    'description': self._get_mechanism_description(mech),
                    'error': str(e)
                }
        
        # Marca il meccanismo critico
        if governing_mechanism:
            mechanism_results[governing_mechanism]['critical'] = True
        
        # Calcola fattore di sicurezza
        ag = loads.get('seismic_acceleration', 0.25)
        safety_factor = min_alpha / ag if ag > 0 else float('inf')
        
        results = {
            'mechanisms': mechanism_results,
            'min_alpha': min_alpha,
            'governing_mechanism': governing_mechanism,
            'safety_factor': safety_factor,
            'verification': safety_factor >= 1.0,
            'seismic_acceleration': ag
        }
        
        # Aggiungi analisi per tipologia
        results['by_category'] = self._categorize_mechanisms(mechanism_results)
        
        return results
    
    def _analyze_mechanism(self, mechanism: KinematicMechanism, loads: Dict) -> float:
        """Calcola fattore alpha per meccanismo specifico secondo EC8-3"""
        
        # Estrai parametri geometrici
        h = self.geometry.get('height', 3.0)
        t = self.geometry.get('thickness', 0.3)
        L = self.geometry.get('length', 5.0)
        h_piano = self.geometry.get('floor_height', 3.0)
        n_piani = self.geometry.get('n_floors', 1)
        
        # Parametri materiale
        gamma = self.material.weight  # kN/m³
        fcm = self.material.fcm  # MPa
        tau0 = self.material.tau0  # MPa
        mu = self.material.mu  # coefficiente attrito
        E = self.material.E  # MPa
        
        # Carichi
        P = loads.get('vertical', 0)  # kN - carico verticale totale
        q = loads.get('floor_load', 0)  # kN/m² - carico solaio
        wind = loads.get('wind_pressure', 0)  # kN/m² - pressione vento
        
        # Peso proprio del muro
        W = gamma * t * L * h  # kN
        
        # Massa partecipante
        M = self._calculate_participating_mass(mechanism, loads)
        
        # Calcolo alpha per ogni meccanismo
        if mechanism == KinematicMechanism.OVERTURNING_SIMPLE:
            # Ribaltamento semplice fuori piano
            # EC8-3 C8.7.1.1
            b = t  # braccio stabilizzante
            h_eff = h  # altezza efficace
            N = P + W  # forza normale totale
            
            # Momento stabilizzante / Momento ribaltante
            alpha = (N * b/2) / (M * 9.81 * h_eff * 2/3)
            
        elif mechanism == KinematicMechanism.OVERTURNING_COMPOUND:
            # Ribaltamento composto (cunei multipli)
            # EC8-3 C8.7.1.2
            n_cunei = min(3, int(h / h_piano))
            alpha_cunei = []
            
            for i in range(n_cunei):
                h_i = h - i * h_piano
                W_i = gamma * t * L * h_i
                N_i = P * (n_piani - i) / n_piani + W_i
                M_i = (N_i + q * L * (n_piani - i)) / 9.81
                
                alpha_i = (N_i * t/2) / (M_i * 9.81 * h_i * 2/3)
                alpha_cunei.append(alpha_i)
            
            alpha = min(alpha_cunei) if alpha_cunei else 0.1
            
        elif mechanism == KinematicMechanism.VERTICAL_FLEXURE:
            # Flessione verticale (presso-flessione fuori piano)
            # EC8-3 C8.7.1.3
            sigma0 = P / (L * t) if L * t > 0 else 0  # MPa
            psi = sigma0 / fcm  # livello di compressione
            
            # Momento resistente ridotto per compressione
            M_rd = fcm * L * t**2 / 6 * (1 - psi**2)
            
            # Momento sollecitante per accelerazione orizzontale
            M_sd = M * 9.81 * h**2 / 8
            
            alpha = M_rd / M_sd if M_sd > 0 else 1.0
            
        elif mechanism == KinematicMechanism.HORIZONTAL_FLEXURE:
            # Flessione orizzontale (nel piano)
            # EC8-3 C8.7.1.4
            sigma0 = P / (L * t) if L * t > 0 else 0
            psi = sigma0 / fcm
            
            # Lunghezza efficace fascia orizzontale
            L_eff = min(L, 1.5 * h)
            
            M_rd = fcm * t * L_eff**2 / 6 * (1 - psi**2)
            M_sd = M * 9.81 * L_eff**2 / 8
            
            alpha = M_rd / M_sd if M_sd > 0 else 1.0
            
        elif mechanism == KinematicMechanism.CORNER_OVERTURNING:
            # Ribaltamento del cantonale
            # EC8-3 C8.7.1.5
            L_cant = min(L/3, h)  # lunghezza cantonale
            t_eff = t * 1.5  # spessore efficace angolo
            
            W_cant = gamma * t_eff * L_cant * h
            M_cant = W_cant / 9.81
            
            # Diagonale stabilizzante
            d = np.sqrt(L_cant**2 + t_eff**2)
            
            alpha = (W_cant * d/2) / (M_cant * 9.81 * h * 2/3)
            
        elif mechanism == KinematicMechanism.ROCKING_PIER:
            # Rocking del maschio murario
            # EC8-3 C8.7.2.1
            h_pier = self.geometry.get('pier_height', h/2)
            L_pier = self.geometry.get('pier_length', L/3)
            
            N_pier = P * L_pier / L + gamma * t * L_pier * h_pier
            M_pier = N_pier / 9.81
            
            # Verifica pressoflessione
            sigma0 = N_pier / (L_pier * t) if L_pier * t > 0 else 0
            psi = sigma0 / fcm
            
            # Moltiplicatore per rocking
            alpha = (L_pier / h_pier) * (1 - psi) * t / L_pier
            
        elif mechanism == KinematicMechanism.SLIDING_PIER:
            # Scorrimento del maschio
            # EC8-3 C8.7.2.2
            N_pier = P / 3 + gamma * t * L/3 * h/2  # Carico su singolo maschio
            
            # Resistenza a taglio
            V_rd = (tau0 + mu * N_pier / (L/3 * t)) * L/3 * t
            
            # Taglio sollecitante
            V_sd = M * 9.81 / 3  # distribuito su 3 maschi
            
            alpha = V_rd / V_sd if V_sd > 0 else 1.0
            
        elif mechanism == KinematicMechanism.DIAGONAL_CRACKING:
            # Fessurazione diagonale (taglio)
            # EC8-3 C8.7.2.3
            sigma0 = P / (L * t) if L * t > 0 else 0
            
            # Resistenza a taglio con Mohr-Coulomb
            tau_max = tau0 + mu * sigma0
            
            # Area resistente diagonale
            A_diag = t * L * np.sqrt(2)
            
            V_rd = tau_max * A_diag
            V_sd = M * 9.81
            
            alpha = V_rd / V_sd if V_sd > 0 else 1.0
            
        elif mechanism == KinematicMechanism.SOFT_STORY:
            # Meccanismo di piano soffice
            # EC8-3 C8.7.3.1
            # Minimo tra rocking e taglio del piano
            alpha_rock = self._analyze_mechanism(KinematicMechanism.ROCKING_PIER, loads)
            alpha_diag = self._analyze_mechanism(KinematicMechanism.DIAGONAL_CRACKING, loads)
            
            alpha = min(alpha_rock, alpha_diag) * 0.8  # fattore riduttivo
            
        elif mechanism == KinematicMechanism.FLOOR_SLIDING:
            # Scorrimento all'interfaccia solaio-muro
            # EC8-3 C8.7.3.2
            N_floor = q * L + P/n_piani if n_piani > 0 else P
            
            # Resistenza per attrito
            F_rd = mu * N_floor
            
            # Forza inerziale del solaio
            F_sd = (N_floor / 9.81) * 9.81  # ag = 1.0
            
            alpha = F_rd / F_sd if F_sd > 0 else 1.0
            
        elif mechanism == KinematicMechanism.ARCH_THRUST:
            # Spinta di archi e volte
            # EC8-3 C8.7.4.1
            f = self.geometry.get('arch_rise', L/10)  # freccia arco
            s = self.geometry.get('arch_thickness', 0.3)  # spessore arco
            
            # Spinta statica arco
            H_stat = (gamma * s * L**2) / (8 * f)
            
            # Incremento sismico spinta
            lambda_thrust = 1.5  # fattore amplificazione
            
            # Resistenza del piedritto
            N_pier = P + W/2
            H_res = mu * N_pier
            
            alpha = H_res / (lambda_thrust * H_stat) if H_stat > 0 else 1.0
            
        elif mechanism == KinematicMechanism.VAULT_MECHANISM:
            # Meccanismo delle volte
            # EC8-3 C8.7.4.2
            tipo_volta = self.geometry.get('vault_type', 'barrel')
            
            if tipo_volta == 'barrel':
                # Volta a botte
                R = L/2  # raggio
                s = self.geometry.get('vault_thickness', 0.3)
                
                # Moltiplicatore per formazione cerniere
                alpha = (s / R) * np.sqrt(fcm / gamma)
                
            elif tipo_volta == 'cross':
                # Volta a crociera
                alpha = 1.2 * (t / L) * np.sqrt(fcm / gamma)
            else:
                alpha = 0.8 * (t / L)
                
        elif mechanism == KinematicMechanism.GABLE_OVERTURNING:
            # Ribaltamento del timpano
            # EC8-3 C8.7.5.1
            h_gable = self.geometry.get('gable_height', h/3)
            t_gable = self.geometry.get('gable_thickness', t)
            
            W_gable = gamma * t_gable * L * h_gable / 2  # triangolare
            M_gable = W_gable / 9.81
            
            # Centro di massa timpano triangolare a h/3 dalla base
            alpha = (W_gable * t_gable/2) / (M_gable * 9.81 * h_gable/3)
            
        elif mechanism == KinematicMechanism.CHIMNEY_OVERTURNING:
            # Ribaltamento di comignoli/pinnacoli
            # EC8-3 C8.7.5.2
            h_chim = self.geometry.get('chimney_height', 1.5)
            b_chim = self.geometry.get('chimney_base', 0.5)
            
            W_chim = gamma * b_chim**2 * h_chim
            M_chim = W_chim / 9.81
            
            alpha = (W_chim * b_chim/2) / (M_chim * 9.81 * h_chim/2)
            
        elif mechanism == KinematicMechanism.PARAPET_OVERTURNING:
            # Ribaltamento del parapetto
            # EC8-3 C8.7.5.3
            h_par = self.geometry.get('parapet_height', 1.0)
            t_par = self.geometry.get('parapet_thickness', t*0.7)
            
            W_par = gamma * t_par * L * h_par
            M_par = W_par / 9.81
            
            # Connessione alla base
            conn_factor = 0.5 if self.geometry.get('parapet_connected', False) else 1.0
            
            alpha = conn_factor * (W_par * t_par/2) / (M_par * 9.81 * h_par/2)
            
        elif mechanism == KinematicMechanism.INFILL_EXPULSION:
            # Espulsione tamponature
            # EC8-3 C8.7.5.4
            t_infill = self.geometry.get('infill_thickness', 0.12)
            h_infill = self.geometry.get('infill_height', h)
            
            # Verifica fuori piano tamponatura
            slenderness = h_infill / t_infill
            
            if slenderness > 15:
                alpha = 0.1  # molto vulnerabile
            else:
                alpha = 0.4 * (15 / slenderness)
                
        elif mechanism == KinematicMechanism.WEDGE_SLIDING:
            # Scorrimento del cuneo
            # EC8-3 C8.7.6.1
            angle = self.geometry.get('wedge_angle', 30) * np.pi / 180
            
            # Resistenza lungo il piano inclinato
            N_wedge = P * np.cos(angle)
            T_resist = mu * N_wedge + tau0 * L * t
            T_sliding = M * 9.81 * np.sin(angle)
            
            alpha = T_resist / T_sliding if T_sliding > 0 else 1.0
            
        elif mechanism == KinematicMechanism.LEAF_SEPARATION:
            # Disgregazione della tessitura (muratura a sacco)
            # EC8-3 C8.7.6.2
            quality = self.geometry.get('masonry_quality', 'poor')
            
            if quality == 'good':
                alpha = 0.8
            elif quality == 'fair':
                alpha = 0.4
            else:  # poor
                alpha = 0.2
                
        elif mechanism == KinematicMechanism.HAMMERING:
            # Martellamento tra edifici adiacenti
            # EC8-3 C8.7.7.1
            gap = self.geometry.get('building_gap', 0.05)  # m
            h_impact = self.geometry.get('impact_height', h*0.7)
            
            # Spostamento atteso
            d_max = alpha * 9.81 * (h_impact / (2*np.pi))**2
            
            if gap > 0:
                alpha = gap / (0.01 * h_impact)  # 1% drift
            else:
                alpha = 0.1
                
        elif mechanism == KinematicMechanism.COLONNADE_ROCKING:
            # Rocking di colonnati
            # EC8-3 C8.7.8.1
            D = self.geometry.get('column_diameter', 0.6)
            h_col = self.geometry.get('column_height', 4.0)
            
            # Snellezza colonna
            lambda_col = h_col / D
            
            # Moltiplicatore per colonna circolare
            alpha = (D / h_col) * np.sqrt(1 - (P / (fcm * np.pi * D**2 / 4)))
            
        elif mechanism == KinematicMechanism.DOME_CRACKING:
            # Fessurazione cupole
            # EC8-3 C8.7.8.2
            R_dome = self.geometry.get('dome_radius', 5.0)
            t_dome = self.geometry.get('dome_thickness', 0.5)
            
            # Rapporto spessore/raggio
            t_R = t_dome / R_dome
            
            # Moltiplicatore per cupola
            alpha = 2 * t_R * np.sqrt(fcm / gamma)
            
        elif mechanism == KinematicMechanism.BELL_TOWER_ROCKING:
            # Rocking campanili
            # EC8-3 C8.7.8.3
            h_tower = self.geometry.get('tower_height', 20.0)
            b_tower = self.geometry.get('tower_base', 5.0)
            
            # Massa campanile
            M_tower = gamma * b_tower**2 * h_tower / 9.81
            
            # Moltiplicatore considerando rastremazione
            taper = self.geometry.get('tower_taper', 0.8)
            
            alpha = (b_tower / h_tower) * (1 + taper) / 2
            
        elif mechanism == KinematicMechanism.FACADE_DETACHMENT:
            # Distacco della facciata
            # EC8-3 C8.7.9.1
            conn_quality = self.geometry.get('facade_connection', 'poor')
            
            if conn_quality == 'good':
                # Buon ammorsamento
                alpha = 0.6
            elif conn_quality == 'fair':
                alpha = 0.3
            else:  # poor
                alpha = 0.15
                
        elif mechanism == KinematicMechanism.TRIUMPHAL_ARCH:
            # Archi trionfali/monumentali
            # EC8-3 C8.7.9.2
            h_arch = self.geometry.get('arch_height', 10.0)
            w_arch = self.geometry.get('arch_width', 8.0)
            t_pier = self.geometry.get('arch_pier_thickness', 2.0)
            
            # Vulnerabilità proporzionale a snellezza
            slenderness_arch = h_arch / t_pier
            
            alpha = 2.0 / slenderness_arch
            
        else:
            logger.warning(f"Meccanismo {mechanism} non implementato")
            alpha = 0.5  # valore di default conservativo
        
        # Applica fattori correttivi
        alpha = self._apply_correction_factors(alpha, mechanism, loads)
        
        return max(alpha, 0.001)  # Evita valori negativi o nulli
    
    def _calculate_participating_mass(self, mechanism: KinematicMechanism, 
                                     loads: Dict) -> float:
        """Calcola massa partecipante per il meccanismo"""
        
        # Estrai parametri
        h = self.geometry.get('height', 3.0)
        t = self.geometry.get('thickness', 0.3)
        L = self.geometry.get('length', 5.0)
        gamma = self.material.weight
        
        # Massa muro
        M_wall = gamma * t * L * h / 9.81
        
        # Massa solai
        q = loads.get('floor_load', 0)
        n_floors = self.geometry.get('n_floors', 1)
        M_floors = q * L * n_floors / 9.81
        
        # Frazione di massa partecipante per meccanismo
        participation_factors = {
            KinematicMechanism.OVERTURNING_SIMPLE: 1.0,
            KinematicMechanism.OVERTURNING_COMPOUND: 0.8,
            KinematicMechanism.VERTICAL_FLEXURE: 0.7,
            KinematicMechanism.HORIZONTAL_FLEXURE: 0.6,
            KinematicMechanism.CORNER_OVERTURNING: 0.3,
            KinematicMechanism.ROCKING_PIER: 0.4,
            KinematicMechanism.SLIDING_PIER: 0.5,
            KinematicMechanism.DIAGONAL_CRACKING: 0.6,
            KinematicMechanism.SOFT_STORY: 0.9,
            KinematicMechanism.FLOOR_SLIDING: 0.2,
            KinematicMechanism.ARCH_THRUST: 0.4,
            KinematicMechanism.VAULT_MECHANISM: 0.5,
            KinematicMechanism.GABLE_OVERTURNING: 0.2,
            KinematicMechanism.CHIMNEY_OVERTURNING: 0.1,
            KinematicMechanism.PARAPET_OVERTURNING: 0.1,
            KinematicMechanism.INFILL_EXPULSION: 0.3,
            KinematicMechanism.WEDGE_SLIDING: 0.4,
            KinematicMechanism.LEAF_SEPARATION: 0.7,
            KinematicMechanism.HAMMERING: 0.8,
            KinematicMechanism.COLONNADE_ROCKING: 0.5,
            KinematicMechanism.DOME_CRACKING: 0.6,
            KinematicMechanism.BELL_TOWER_ROCKING: 0.9,
            KinematicMechanism.FACADE_DETACHMENT: 0.7,
            KinematicMechanism.TRIUMPHAL_ARCH: 0.8,
        }
        
        psi = participation_factors.get(mechanism, 0.5)
        
        return psi * (M_wall + M_floors)
    
    def _apply_correction_factors(self, alpha: float, 
                                  mechanism: KinematicMechanism,
                                  loads: Dict) -> float:
        """Applica fattori correttivi secondo EC8-3"""
        
        # Fattore di confidenza (LC)
        LC = loads.get('confidence_factor', 1.35)
        
        # Fattore di comportamento
        q = self._get_behavior_factor(mechanism)
        
        # Correzione per effetti locali
        local_factor = 1.0
        if mechanism in [KinematicMechanism.GABLE_OVERTURNING,
                        KinematicMechanism.CHIMNEY_OVERTURNING,
                        KinematicMechanism.PARAPET_OVERTURNING]:
            local_factor = 1.5  # amplificazione locale
        
        # Applica correzioni
        alpha_corr = alpha * q / (LC * local_factor)
        
        return alpha_corr
    
    def _get_behavior_factor(self, mechanism: KinematicMechanism) -> float:
        """Fattore di comportamento q per meccanismo"""
        
        # Valori secondo EC8-3 Tabella C8.2
        q_factors = {
            KinematicMechanism.OVERTURNING_SIMPLE: 2.0,
            KinematicMechanism.OVERTURNING_COMPOUND: 2.0,
            KinematicMechanism.VERTICAL_FLEXURE: 1.5,
            KinematicMechanism.HORIZONTAL_FLEXURE: 1.5,
            KinematicMechanism.CORNER_OVERTURNING: 2.0,
            KinematicMechanism.ROCKING_PIER: 2.0,
            KinematicMechanism.SLIDING_PIER: 1.0,
            KinematicMechanism.DIAGONAL_CRACKING: 1.5,
            KinematicMechanism.SOFT_STORY: 1.0,
            KinematicMechanism.FLOOR_SLIDING: 1.0,
            KinematicMechanism.ARCH_THRUST: 1.5,
            KinematicMechanism.VAULT_MECHANISM: 1.5,
            KinematicMechanism.GABLE_OVERTURNING: 2.0,
            KinematicMechanism.CHIMNEY_OVERTURNING: 2.0,
            KinematicMechanism.PARAPET_OVERTURNING: 2.0,
            KinematicMechanism.INFILL_EXPULSION: 1.0,
            KinematicMechanism.WEDGE_SLIDING: 1.0,
            KinematicMechanism.LEAF_SEPARATION: 1.0,
            KinematicMechanism.HAMMERING: 1.0,
            KinematicMechanism.COLONNADE_ROCKING: 2.0,
            KinematicMechanism.DOME_CRACKING: 1.5,
            KinematicMechanism.BELL_TOWER_ROCKING: 2.0,
            KinematicMechanism.FACADE_DETACHMENT: 1.0,
            KinematicMechanism.TRIUMPHAL_ARCH: 2.0,
        }
        
        return q_factors.get(mechanism, 1.5)
    
    def _get_mechanism_description(self, mechanism: KinematicMechanism) -> str:
        """Descrizione del meccanismo"""
        
        descriptions = {
            KinematicMechanism.OVERTURNING_SIMPLE: "Ribaltamento semplice fuori piano",
            KinematicMechanism.OVERTURNING_COMPOUND: "Ribaltamento composto",
            KinematicMechanism.VERTICAL_FLEXURE: "Flessione verticale",
            KinematicMechanism.HORIZONTAL_FLEXURE: "Flessione orizzontale",
            KinematicMechanism.CORNER_OVERTURNING: "Ribaltamento del cantonale",
            KinematicMechanism.ROCKING_PIER: "Rocking del maschio murario",
            KinematicMechanism.SLIDING_PIER: "Scorrimento del maschio",
            KinematicMechanism.DIAGONAL_CRACKING: "Fessurazione diagonale",
            KinematicMechanism.SOFT_STORY: "Piano soffice",
            KinematicMechanism.FLOOR_SLIDING: "Scorrimento piano",
            KinematicMechanism.ARCH_THRUST: "Spinta di archi",
            KinematicMechanism.VAULT_MECHANISM: "Meccanismo delle volte",
            KinematicMechanism.GABLE_OVERTURNING: "Ribaltamento timpano",
            KinematicMechanism.CHIMNEY_OVERTURNING: "Ribaltamento comignolo",
            KinematicMechanism.PARAPET_OVERTURNING: "Ribaltamento parapetto",
            KinematicMechanism.INFILL_EXPULSION: "Espulsione tamponature",
            KinematicMechanism.WEDGE_SLIDING: "Scorrimento del cuneo",
            KinematicMechanism.LEAF_SEPARATION: "Disgregazione muratura",
            KinematicMechanism.HAMMERING: "Martellamento",
            KinematicMechanism.COLONNADE_ROCKING: "Rocking colonnato",
            KinematicMechanism.DOME_CRACKING: "Fessurazione cupola",
            KinematicMechanism.BELL_TOWER_ROCKING: "Rocking campanile",
            KinematicMechanism.FACADE_DETACHMENT: "Distacco facciata",
            KinematicMechanism.TRIUMPHAL_ARCH: "Arco trionfale",
        }
        
        return descriptions.get(mechanism, "Meccanismo non definito")
    
    def _categorize_mechanisms(self, mechanism_results: Dict) -> Dict:
        """Categorizza meccanismi per tipologia"""
        
        categories = {
            'fuori_piano': [
                'OVERTURNING_SIMPLE', 'OVERTURNING_COMPOUND',
                'VERTICAL_FLEXURE', 'CORNER_OVERTURNING',
                'GABLE_OVERTURNING', 'CHIMNEY_OVERTURNING',
                'PARAPET_OVERTURNING', 'FACADE_DETACHMENT'
            ],
            'nel_piano': [
                'HORIZONTAL_FLEXURE', 'ROCKING_PIER',
                'SLIDING_PIER', 'DIAGONAL_CRACKING',
                'SOFT_STORY', 'FLOOR_SLIDING'
            ],
            'volte_archi': [
                'ARCH_THRUST', 'VAULT_MECHANISM',
                'DOME_CRACKING', 'TRIUMPHAL_ARCH'
            ],
            'locali': [
                'INFILL_EXPULSION', 'WEDGE_SLIDING',
                'LEAF_SEPARATION', 'HAMMERING'
            ],
            'monumentali': [
                'COLONNADE_ROCKING', 'BELL_TOWER_ROCKING'
            ]
        }
        
        categorized = {}
        for cat, mechs in categories.items():
            categorized[cat] = {
                'mechanisms': [],
                'min_alpha': float('inf'),
                'critical': None
            }
            
            for mech in mechs:
                if mech in mechanism_results:
                    result = mechanism_results[mech]
                    categorized[cat]['mechanisms'].append({
                        'name': mech,
                        'alpha': result['alpha'],
                        'critical': result.get('critical', False)
                    })
                    
                    if result['alpha'] < categorized[cat]['min_alpha']:
                        categorized[cat]['min_alpha'] = result['alpha']
                        categorized[cat]['critical'] = mech
        
        return categorized
    
    def optimize_strengthening(self, target_alpha: float, 
                             mechanism: Optional[KinematicMechanism] = None) -> Dict:
        """Ottimizza interventi di rinforzo per raggiungere alpha target"""
        
        current_results = self.analyze_all_mechanisms({})
        
        if mechanism is None:
            # Usa meccanismo critico
            mechanism = KinematicMechanism[current_results['governing_mechanism']]
        
        current_alpha = current_results['mechanisms'][mechanism.value]['alpha']
        
        # Strategie di rinforzo per meccanismo
        strengthening_strategies = self._get_strengthening_strategies(mechanism)
        
        optimization_results = {
            'current_alpha': current_alpha,
            'target_alpha': target_alpha,
            'mechanism': mechanism.value,
            'strategies': []
        }
        
        for strategy in strengthening_strategies:
            result = self._optimize_strategy(strategy, mechanism, target_alpha)
            optimization_results['strategies'].append(result)
        
        # Trova strategia ottimale
        optimal = min(optimization_results['strategies'], 
                     key=lambda x: x.get('cost_index', float('inf')))
        optimization_results['optimal_strategy'] = optimal
        
        return optimization_results
    
    def _get_strengthening_strategies(self, 
                                     mechanism: KinematicMechanism) -> List[Dict]:
        """Definisce strategie di rinforzo per meccanismo"""
        
        strategies = []
        
        # Strategie comuni
        strategies.append({
            'name': 'tie_rods',
            'description': 'Inserimento tiranti',
            'parameters': ['tie_force', 'tie_spacing'],
            'applicable': True
        })
        
        strategies.append({
            'name': 'injections',
            'description': 'Iniezioni di malta',
            'parameters': ['injection_depth', 'grout_strength'],
            'applicable': mechanism not in [
                KinematicMechanism.FLOOR_SLIDING,
                KinematicMechanism.HAMMERING
            ]
        })
        
        # Strategie specifiche per meccanismo
        if mechanism in [KinematicMechanism.OVERTURNING_SIMPLE,
                        KinematicMechanism.OVERTURNING_COMPOUND,
                        KinematicMechanism.CORNER_OVERTURNING]:
            strategies.append({
                'name': 'buttresses',
                'description': 'Contrafforti',
                'parameters': ['buttress_width', 'buttress_spacing'],
                'applicable': True
            })
            
        if mechanism in [KinematicMechanism.VERTICAL_FLEXURE,
                        KinematicMechanism.HORIZONTAL_FLEXURE]:
            strategies.append({
                'name': 'frp_wrapping',
                'description': 'Fasciature FRP',
                'parameters': ['frp_layers', 'frp_type'],
                'applicable': True
            })
            
        if mechanism in [KinematicMechanism.SLIDING_PIER,
                        KinematicMechanism.DIAGONAL_CRACKING]:
            strategies.append({
                'name': 'jacketing',
                'description': 'Camicia armata',
                'parameters': ['jacket_thickness', 'reinforcement_ratio'],
                'applicable': True
            })
            
        return strategies
    
    def _optimize_strategy(self, strategy: Dict, 
                          mechanism: KinematicMechanism,
                          target_alpha: float) -> Dict:
        """Ottimizza parametri di una strategia di rinforzo"""
        
        def objective(params):
            # Crea modello rinforzato
            reinforced_model = self._create_reinforced_model(strategy, params)
            
            # Analizza meccanismo
            loads = {'seismic_acceleration': 0.25}  # default
            alpha = reinforced_model._analyze_mechanism(mechanism, loads)
            
            # Penalizza se non raggiunge target
            if alpha < target_alpha:
                penalty = 100 * (target_alpha - alpha)**2
            else:
                penalty = 0
            
            # Costo indicizzato
            cost = self._calculate_cost_index(strategy, params)
            
            return cost + penalty
        
        # Ottimizza con vincoli
        if strategy['name'] == 'tie_rods':
            bounds = [(50, 500), (1.0, 5.0)]  # forza (kN), spacing (m)
            x0 = [100, 2.0]
        elif strategy['name'] == 'injections':
            bounds = [(0.1, 0.5), (5, 30)]  # depth (m), strength (MPa)
            x0 = [0.3, 15]
        elif strategy['name'] == 'frp_wrapping':
            bounds = [(1, 5), (1, 3)]  # layers, type (1=CFRP, 2=GFRP, 3=AFRP)
            x0 = [2, 1]
        else:
            bounds = [(0.1, 1.0), (0.1, 1.0)]
            x0 = [0.5, 0.5]
        
        from scipy.optimize import minimize
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        # Verifica finale
        final_model = self._create_reinforced_model(strategy, result.x)
        final_alpha = final_model._analyze_mechanism(mechanism, 
                                                    {'seismic_acceleration': 0.25})
        
        return {
            'strategy': strategy['name'],
            'description': strategy['description'],
            'optimal_params': dict(zip(strategy['parameters'], result.x)),
            'achieved_alpha': final_alpha,
            'target_reached': final_alpha >= target_alpha,
            'cost_index': self._calculate_cost_index(strategy, result.x),
            'success': result.success
        }
    
    def _create_reinforced_model(self, strategy: Dict, params: List) -> 'LimitAnalysis':
        """Crea modello con rinforzi applicati"""
        
        # Copia geometria e materiale
        reinforced_geom = copy.deepcopy(self.geometry)
        reinforced_mat = copy.deepcopy(self.material)
        
        # Applica modifiche secondo strategia
        if strategy['name'] == 'tie_rods':
            # Tiranti aumentano capacità contro ribaltamento
            tie_force, tie_spacing = params
            reinforced_geom['tie_force'] = tie_force
            reinforced_geom['tie_spacing'] = tie_spacing
            
        elif strategy['name'] == 'injections':
            # Iniezioni migliorano proprietà meccaniche
            depth_ratio, grout_strength = params
            reinforced_mat.fcm *= (1 + 0.5 * depth_ratio)
            reinforced_mat.tau0 *= (1 + 0.3 * depth_ratio)
            reinforced_mat.E *= (1 + 0.2 * depth_ratio)
            
        elif strategy['name'] == 'frp_wrapping':
            # FRP aumenta resistenza a flessione e taglio
            layers, frp_type = params
            frp_factors = {1: 1.5, 2: 1.3, 3: 1.4}  # CFRP, GFRP, AFRP
            factor = frp_factors.get(int(frp_type), 1.3)
            reinforced_mat.fcm *= (1 + 0.2 * layers * factor)
            reinforced_mat.fctm *= (1 + 0.3 * layers * factor)
            
        elif strategy['name'] == 'buttresses':
            # Contrafforti aumentano spessore efficace
            width, spacing = params
            eff_thickness_increase = width / spacing
            reinforced_geom['thickness'] *= (1 + eff_thickness_increase)
            
        elif strategy['name'] == 'jacketing':
            # Camicia armata
            thickness, reinf_ratio = params
            reinforced_geom['thickness'] += thickness
            reinforced_mat.fcm *= (1 + reinf_ratio)
            reinforced_mat.tau0 *= (1 + 1.5 * reinf_ratio)
        
        return LimitAnalysis(reinforced_geom, reinforced_mat)
    
    def _calculate_cost_index(self, strategy: Dict, params: List) -> float:
        """Calcola indice di costo normalizzato per strategia"""
        
        # Costi unitari indicativi (€/unità)
        unit_costs = {
            'tie_rods': 150,  # €/m
            'injections': 200,  # €/m²
            'frp_wrapping': 350,  # €/m²
            'buttresses': 500,  # €/m³
            'jacketing': 400   # €/m²
        }
        
        base_cost = unit_costs.get(strategy['name'], 300)
        
        # Calcola quantità
        if strategy['name'] == 'tie_rods':
            force, spacing = params
            quantity = self.geometry['length'] / spacing
            cost = base_cost * quantity * (1 + force/200)
            
        elif strategy['name'] == 'injections':
            depth, strength = params
            area = self.geometry['height'] * self.geometry['length']
            cost = base_cost * area * depth * (strength/15)
            
        elif strategy['name'] == 'frp_wrapping':
            layers, frp_type = params
            area = 2 * self.geometry['height'] * self.geometry['length']
            cost = base_cost * area * layers * frp_type
            
        else:
            # Stima semplificata
            cost = base_cost * np.prod(params) * self.geometry['length']
        
        # Normalizza rispetto a costo di riferimento
        ref_cost = 10000  # €
        return cost / ref_cost
    
    def perform_sensitivity_analysis(self, base_loads: Dict, 
                                   parameters: List[str] = None) -> Dict:
        """Analisi di sensitività sui parametri"""
        
        if parameters is None:
            parameters = ['thickness', 'height', 'fcm', 'mu', 'seismic_acceleration']
        
        sensitivity_results = {}
        base_results = self.analyze_all_mechanisms(base_loads)
        base_alpha = base_results['min_alpha']
        
        for param in parameters:
            variations = np.linspace(0.5, 1.5, 11)  # ±50%
            alphas = []
            
            for var in variations:
                # Crea modello variato
                varied_model = self._create_varied_model(param, var)
                varied_loads = copy.deepcopy(base_loads)
                
                if param == 'seismic_acceleration':
                    varied_loads['seismic_acceleration'] = base_loads.get(
                        'seismic_acceleration', 0.25) * var
                    varied_results = self.analyze_all_mechanisms(varied_loads)
                else:
                    varied_results = varied_model.analyze_all_mechanisms(base_loads)
                
                alphas.append(varied_results['min_alpha'])
            
            # Calcola sensitività
            sensitivity = np.gradient(alphas, variations)
            
            sensitivity_results[param] = {
                'variations': variations.tolist(),
                'alphas': alphas,
                'sensitivity': sensitivity.tolist(),
                'mean_sensitivity': np.mean(np.abs(sensitivity)),
                'max_variation': (max(alphas) - min(alphas)) / base_alpha
            }
        
        return sensitivity_results
    
    def _create_varied_model(self, parameter: str, factor: float) -> 'LimitAnalysis':
        """Crea modello con parametro variato"""
        
        varied_geom = copy.deepcopy(self.geometry)
        varied_mat = copy.deepcopy(self.material)
        
        if parameter in ['thickness', 'height', 'length']:
            varied_geom[parameter] *= factor
        elif parameter in ['fcm', 'fctm', 'tau0', 'mu', 'E']:
            setattr(varied_mat, parameter, getattr(varied_mat, parameter) * factor)
        
        return LimitAnalysis(varied_geom, varied_mat)


def perform_limit_analysis(wall_data: Dict, material: MaterialProperties,
                          loads: Dict, options: Dict) -> Dict:
    """Funzione principale per analisi limite completa"""
    
    logger.info("Esecuzione analisi limite completa EC8-3")
    
    # Prepara geometria
    geometry = {
        'height': wall_data.get('height', 3.0),
        'thickness': wall_data.get('thickness', 0.3),
        'length': wall_data.get('length', 5.0),
        'n_floors': wall_data.get('n_floors', 1),
        'floor_height': wall_data.get('floor_height', 3.0),
        **wall_data  # Include tutti i parametri aggiuntivi
    }
    
    # Crea modello
    model = LimitAnalysis(geometry, material)
    
    # Analisi base
    results = model.analyze_all_mechanisms(loads)
    
    # Analisi aggiuntive opzionali
    if options.get('probabilistic', False):
        results['probabilistic'] = probabilistic_limit_analysis(model, loads, options)
    
    if options.get('optimize_strengthening', False):
        target_alpha = options.get('target_alpha', 0.3)
        results['strengthening'] = model.optimize_strengthening(target_alpha)
    
    if options.get('sensitivity', False):
        results['sensitivity'] = model.perform_sensitivity_analysis(loads)
    
    # Report dettagliato
    results['report'] = _generate_limit_report(results, geometry, loads)
    
    return results


def _generate_limit_report(results: Dict, geometry: Dict, loads: Dict) -> Dict:
    """Genera report dettagliato analisi limite"""
    
    report = {
        'summary': {
            'governing_mechanism': results['governing_mechanism'],
            'min_alpha': results['min_alpha'],
            'safety_factor': results['safety_factor'],
            'verification': results['verification'],
            'seismic_demand': loads.get('seismic_acceleration', 0.25)
        },
        'critical_mechanisms': [],
        'recommendations': []
    }
    
    # Identifica meccanismi critici (alpha < 0.5)
    for mech, data in results['mechanisms'].items():
        if data['alpha'] < 0.5:
            report['critical_mechanisms'].append({
                'mechanism': mech,
                'alpha': data['alpha'],
                'description': data['description'],
                'severity': 'high' if data['alpha'] < 0.2 else 'medium'
            })
    
    # Raccomandazioni
    if results['safety_factor'] < 1.0:
        report['recommendations'].append(
            "Struttura non verifica per azioni sismiche. Interventi di rinforzo necessari."
        )
        
        if results.get('strengthening'):
            optimal = results['strengthening'].get('optimal_strategy', {})
            report['recommendations'].append(
                f"Strategia ottimale: {optimal.get('description', 'N/A')}"
            )
    
    return report