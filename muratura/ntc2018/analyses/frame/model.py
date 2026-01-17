# analyses/frame/model.py
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy.sparse import lil_matrix, csr_matrix, eye
from scipy.sparse.linalg import spsolve, eigsh
from .element import FrameElement
from ...materials import MaterialProperties
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AnalysisOptions:
    """Opzioni per l'analisi del telaio"""
    analysis_type: str = "pushover"  # "static", "modal", "pushover"
    n_modes: int = 6
    lateral_pattern: str = "triangular"  # "triangular", "uniform", "modal"
    target_drift: float = 0.04
    max_iterations: int = 100
    convergence_tolerance: float = 1e-6
    include_pdelta: bool = True
    include_material_nonlinearity: bool = True
    
@dataclass
class LoadCase:
    """Definizione di un caso di carico"""
    name: str
    forces: Dict[int, np.ndarray]  # node_id -> [Fx, Fy, Mz]
    load_type: str = "static"  # "static", "seismic", "wind"
    
@dataclass
class PerformanceLevel:
    """Livelli di prestazione secondo EC8/NTC"""
    name: str
    drift_limit: float
    damage_description: str
    
    @classmethod
    def get_default_levels(cls) -> List['PerformanceLevel']:
        return [
            cls("DL", 0.005, "Damage Limitation - No structural damage"),
            cls("SD", 0.015, "Significant Damage - Repairable damage"),
            cls("NC", 0.030, "Near Collapse - Heavy damage, possible collapse")
        ]

class EquivalentFrame:
    """Modello completo di telaio equivalente per murature"""
    
    def __init__(self):
        self.nodes = {}  # id -> coordinate
        self.elements = []  # Lista FrameElement
        self.constraints = []  # Nodi vincolati
        self.K_global = None
        self.M_global = None
        self.node_dofs = {}  # Mappa nodo -> DOF globali
        self.performance_levels = PerformanceLevel.get_default_levels()
        self.analysis_history = []
        
    def add_node(self, node_id: int, x: float, y: float):
        """Aggiunge nodo al telaio"""
        self.nodes[node_id] = np.array([x, y])
        logger.debug(f"Aggiunto nodo {node_id} in posizione ({x}, {y})")
        
    def add_element(self, element: FrameElement):
        """Aggiunge elemento al telaio"""
        # Verifica che i nodi esistano
        if element.i_node not in self.nodes or element.j_node not in self.nodes:
            raise ValueError(f"Nodi {element.i_node} o {element.j_node} non esistono")
            
        # Imposta trasformazione
        element.set_transformation_matrix(self.nodes)
        self.elements.append(element)
        logger.debug(f"Aggiunto elemento tra nodi {element.i_node} e {element.j_node}")
        
    def add_constraint(self, node_id: int, dof_type: str = "fixed"):
        """Aggiunge vincolo a nodo"""
        if node_id not in self.nodes:
            raise ValueError(f"Nodo {node_id} non esiste")
            
        self.constraints.append({
            'node': node_id,
            'type': dof_type,
            'dofs': self._get_constrained_dofs(dof_type)
        })
        logger.debug(f"Aggiunto vincolo {dof_type} al nodo {node_id}")
        
    def _get_constrained_dofs(self, dof_type: str) -> List[int]:
        """Restituisce DOF vincolati per tipo"""
        constraint_types = {
            "fixed": [0, 1, 2],      # u, v, theta
            "pinned": [0, 1],        # u, v
            "roller_x": [1],         # v
            "roller_y": [0],         # u
            "rotation": [2]          # theta
        }
        return constraint_types.get(dof_type, [])
            
    def assemble_stiffness_matrix(self):
        """Assembla matrice di rigidezza globale"""
        n_nodes = len(self.nodes)
        n_dof = 3 * n_nodes
        
        # Crea mapping nodo -> DOF
        for i, node_id in enumerate(sorted(self.nodes.keys())):
            self.node_dofs[node_id] = [3*i, 3*i+1, 3*i+2]
            
        # Inizializza matrice sparsa
        self.K_global = lil_matrix((n_dof, n_dof))
        
        # Assembla contributi elementi
        for elem in self.elements:
            dof_i = self.node_dofs[elem.i_node]
            dof_j = self.node_dofs[elem.j_node]
            elem_dofs = dof_i + dof_j
            
            K_elem = elem.get_global_stiffness()
            
            # Aggiungi alla matrice globale
            for i in range(6):
                for j in range(6):
                    self.K_global[elem_dofs[i], elem_dofs[j]] += K_elem[i, j]
                    
        # Converti a formato CSR per efficienza
        self.K_global = self.K_global.tocsr()
        logger.info(f"Matrice di rigidezza assemblata: {n_dof}x{n_dof}")
        
    def assemble_mass_matrix(self, floor_masses: Dict[int, float]):
        """Assembla matrice delle masse"""
        n_dof = self.K_global.shape[0]
        self.M_global = lil_matrix((n_dof, n_dof))
        
        # Masse concentrate ai nodi
        for node_id, mass in floor_masses.items():
            if node_id in self.node_dofs:
                dofs = self.node_dofs[node_id]
                # Masse traslazionali
                self.M_global[dofs[0], dofs[0]] = mass
                self.M_global[dofs[1], dofs[1]] = mass
                # Massa rotazionale (momento d'inerzia polare)
                I_polar = mass * 1.0  # Raggio giratorio stimato
                self.M_global[dofs[2], dofs[2]] = I_polar
                
        # Aggiungi masse distribuite degli elementi
        for elem in self.elements:
            elem_mass = elem.get_mass_matrix()
            dof_i = self.node_dofs[elem.i_node]
            dof_j = self.node_dofs[elem.j_node]
            elem_dofs = dof_i + dof_j
            
            for i in range(6):
                for j in range(6):
                    self.M_global[elem_dofs[i], elem_dofs[j]] += elem_mass[i, j]
                
        self.M_global = self.M_global.tocsr()
        logger.info("Matrice delle masse assemblata")
        
    def apply_constraints(self, K: csr_matrix, F: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """Applica vincoli con metodo penalità"""
        K_mod = K.tolil()
        F_mod = F.copy()
        
        penalty = np.max(K.diagonal()) * 1e12
        
        for constraint in self.constraints:
            node_id = constraint['node']
            if node_id in self.node_dofs:
                global_dofs = self.node_dofs[node_id]
                for local_dof in constraint['dofs']:
                    dof = global_dofs[local_dof]
                    # Metodo penalità
                    K_mod[dof, dof] += penalty
                    F_mod[dof] = 0
                    
        return K_mod.tocsr(), F_mod
    
    def solve_static(self, forces: Dict[int, np.ndarray]) -> Dict:
        """Risolve analisi statica lineare"""
        if self.K_global is None:
            self.assemble_stiffness_matrix()
        
        n_dof = self.K_global.shape[0]
        F = np.zeros(n_dof)
        
        # Assembla vettore forze
        for node_id, force in forces.items():
            if node_id in self.node_dofs:
                dofs = self.node_dofs[node_id]
                F[dofs[0]] = force[0] if len(force) >= 1 else 0
                F[dofs[1]] = force[1] if len(force) >= 2 else 0
                F[dofs[2]] = force[2] if len(force) >= 3 else 0
        
        # Applica vincoli
        K_mod, F_mod = self.apply_constraints(self.K_global, F)
        
        # Risolvi sistema
        try:
            # Aggiungi piccola perturbazione per stabilità numerica
            K_mod = K_mod + eye(n_dof) * 1e-10
            u = spsolve(K_mod, F_mod)
            
            # Verifica soluzione
            if np.any(np.isnan(u)) or np.any(np.isinf(u)):
                logger.warning("Soluzione contiene NaN o Inf, reset a zero")
                u = np.zeros(n_dof)
            
            # Limita spostamenti massimi per stabilità
            max_disp = 1.0  # metro
            if np.max(np.abs(u)) > max_disp:
                scale = max_disp / np.max(np.abs(u))
                u = u * scale
                logger.warning(f"Spostamenti scalati di {scale:.3f}")
                
        except Exception as e:
            logger.error(f"Errore nella soluzione: {e}")
            u = np.zeros(n_dof)
        
        # Calcola reazioni
        R = self.K_global @ u - F
        
        # Calcola forze interne elementi
        element_forces = []
        for elem in self.elements:
            try:
                dof_i = self.node_dofs[elem.i_node]
                dof_j = self.node_dofs[elem.j_node]
                u_elem = np.concatenate([u[dof_i], u[dof_j]])
                forces = elem.compute_internal_forces(u_elem)
                element_forces.append(forces)
            except Exception as e:
                logger.error(f"Errore calcolo forze elemento: {e}")
                element_forces.append({'N': 0, 'V': 0, 'M_i': 0, 'M_j': 0})
        
        return {
            'displacements': u,
            'reactions': R,
            'element_forces': element_forces,
            'max_displacement': np.max(np.abs(u))
        }
        
    def solve_modal(self, n_modes: int = 6) -> Dict:
        """Analisi modale per frequenze e modi di vibrare"""
        if self.K_global is None:
            self.assemble_stiffness_matrix()
            
        if self.M_global is None:
            raise ValueError("Matrice delle masse non definita")
            
        # Applica vincoli
        K_mod, _ = self.apply_constraints(self.K_global, np.zeros(self.K_global.shape[0]))
        
        # Risolvi problema agli autovalori generalizzato
        try:
            eigenvalues, eigenvectors = eigsh(K_mod, k=min(n_modes, K_mod.shape[0]-2), 
                                            M=self.M_global, sigma=0, which='LM')
            
            # Ordina per frequenza crescente
            idx = eigenvalues.argsort()
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
        except Exception as e:
            logger.error(f"Errore analisi modale: {e}")
            return {
                'frequencies': np.zeros(n_modes),
                'periods': np.zeros(n_modes),
                'mode_shapes': np.zeros((self.K_global.shape[0], n_modes)),
                'modal_masses': np.zeros(n_modes)
            }
        
        # Calcola frequenze e periodi
        omega = np.sqrt(np.abs(eigenvalues))
        frequencies = omega / (2 * np.pi)
        periods = 1 / frequencies
        
        # Calcola masse modali effettive
        modal_masses = []
        participation_factors = []
        
        for i in range(len(eigenvalues)):
            phi = eigenvectors[:, i]
            # Normalizza rispetto alla massa
            phi = phi / np.sqrt(phi.T @ self.M_global @ phi)
            eigenvectors[:, i] = phi
            
            # Massa modale effettiva (direzione X)
            r_x = np.zeros(len(phi))
            r_x[::3] = 1  # DOF in X
            L_x = phi.T @ self.M_global @ r_x
            M_eff_x = L_x**2
            modal_masses.append(M_eff_x)
            participation_factors.append(L_x)
            
        return {
            'frequencies': frequencies,
            'periods': periods,
            'mode_shapes': eigenvectors,
            'modal_masses': modal_masses,
            'participation_factors': participation_factors
        }
        
    def pushover_analysis(self, lateral_pattern: str = "triangular", 
                         target_drift: float = 0.04,
                         options: Optional[AnalysisOptions] = None) -> Dict:
        """Analisi pushover completa con controllo di spostamento"""
        
        if options is None:
            options = AnalysisOptions(lateral_pattern=lateral_pattern, 
                                    target_drift=target_drift)
        
        results = {
            'steps': [],
            'curve': {
                'V_base': [], 
                'delta_top': [],
                'drift': []
            },
            'performance_point': None,
            'hinges': [],
            'performance_levels': {},
            'bilinear': None
        }
        
        # Pattern di carico laterale
        forces = self._get_lateral_pattern(options.lateral_pattern)
        
        # Altezza edificio
        H_tot = self._get_building_height()
        
        # Incremento di carico
        lambda_increment = 0.01
        lambda_current = 0.0
        
        # Storia delle cerniere
        hinge_history = []
        
        for step in range(options.max_iterations):
            lambda_current += lambda_increment
            
            # Applica carico corrente
            F_current = {k: lambda_current * v for k, v in forces.items()}
            
            try:
                # Risolvi per configurazione corrente
                solution = self.solve_static(F_current)
                
                # Verifica formazione cerniere
                hinges_formed = self._check_hinges(solution)
                
                if hinges_formed:
                    # Aggiorna rigidezza con cerniere
                    self._update_model_with_hinges(hinges_formed)
                    hinge_history.extend(hinges_formed)
                    
                    # Riduci incremento dopo formazione cerniere
                    lambda_increment *= 0.5
                    
                # Calcola quantità di risposta
                V_base = self._calculate_base_shear(solution['reactions'])
                delta_top = self._calculate_top_displacement(solution['displacements'])
                drift = delta_top / H_tot if H_tot > 0 else 0
                
                # Salva risultati step
                results['curve']['V_base'].append(V_base)
                results['curve']['delta_top'].append(delta_top)
                results['curve']['drift'].append(drift)
                
                results['steps'].append({
                    'step': step,
                    'lambda': lambda_current,
                    'solution': solution,
                    'hinges': hinges_formed,
                    'V_base': V_base,
                    'delta_top': delta_top,
                    'drift': drift
                })
                
                # Verifica livelli di prestazione
                for level in self.performance_levels:
                    if drift >= level.drift_limit and level.name not in results['performance_levels']:
                        results['performance_levels'][level.name] = {
                            'V_base': V_base,
                            'delta_top': delta_top,
                            'drift': drift,
                            'step': step
                        }
                
                # Controllo convergenza
                if drift > options.target_drift:
                    logger.info(f"Raggiunto drift target: {drift:.3f}")
                    break
                    
                # Controllo instabilità
                if len(results['curve']['V_base']) > 10:
                    if V_base < 0.8 * max(results['curve']['V_base'][-10:]):
                        logger.warning("Degrado significativo della capacità")
                        break
                        
            except Exception as e:
                logger.error(f"Errore al passo {step}: {e}")
                break
        
        # Post-processing risultati
        if len(results['curve']['V_base']) > 2:
            # Bilinearizzazione
            results['bilinear'] = self._bilinearize_pushover(results['curve'])
            
            # Punto di performance
            results['performance_point'] = self._find_performance_point_N2(
                results['curve'], results['bilinear']
            )
            
            # Fattore di duttilità
            if results['bilinear']:
                delta_y = results['bilinear']['delta_y']
                delta_u = results['curve']['delta_top'][-1]
                results['ductility'] = delta_u / delta_y if delta_y > 0 else 1.0
        
        results['hinges'] = hinge_history
        
        return results
        
    def _get_lateral_pattern(self, pattern_type: str) -> Dict[int, np.ndarray]:
        """Genera pattern di carico laterale"""
        if pattern_type == "triangular":
            return self._triangular_load_pattern()
        elif pattern_type == "uniform":
            return self._uniform_load_pattern()
        elif pattern_type == "modal":
            return self._modal_load_pattern()
        else:
            raise ValueError(f"Pattern non riconosciuto: {pattern_type}")
            
    def _triangular_load_pattern(self) -> Dict[int, np.ndarray]:
        """Pattern triangolare (lineare con altezza)"""
        forces = {}
        
        # Trova livelli
        levels = {}
        for node_id, coord in self.nodes.items():
            level = round(coord[1], 2)
            if level not in levels:
                levels[level] = []
            levels[level].append(node_id)
        
        # Ordina livelli
        sorted_levels = sorted(levels.keys())
        if len(sorted_levels) < 2:
            return forces
            
        y_min = sorted_levels[0]
        y_max = sorted_levels[-1]
        H_tot = y_max - y_min
        
        # Forza totale normalizzata
        F_tot = 100.0  # kN
        
        # Distribuisci forze
        sum_wi_hi = 0
        for level in sorted_levels[1:]:  # Escludi base
            wi = len(levels[level])  # Peso proporzionale ai nodi
            hi = level - y_min
            sum_wi_hi += wi * hi
            
        if sum_wi_hi > 0:
            for level in sorted_levels[1:]:
                hi = level - y_min
                Fi = F_tot * len(levels[level]) * hi / sum_wi_hi
                force_per_node = Fi / len(levels[level])
                
                for node_id in levels[level]:
                    forces[node_id] = np.array([force_per_node, 0, 0])
                    
        return forces
        
    def _uniform_load_pattern(self) -> Dict[int, np.ndarray]:
        """Pattern uniforme"""
        forces = {}
        
        # Trova nodi non alla base
        y_min = min(coord[1] for coord in self.nodes.values())
        active_nodes = [nid for nid, coord in self.nodes.items() if coord[1] > y_min]
        
        if active_nodes:
            F_tot = 100.0  # kN
            force_per_node = F_tot / len(active_nodes)
            
            for node_id in active_nodes:
                forces[node_id] = np.array([force_per_node, 0, 0])
                
        return forces
        
    def _modal_load_pattern(self) -> Dict[int, np.ndarray]:
        """Pattern proporzionale al primo modo"""
        try:
            # Analisi modale per primo modo
            modal = self.solve_modal(n_modes=1)
            phi = modal['mode_shapes'][:, 0]
            
            forces = {}
            F_tot = 100.0  # kN
            
            # Estrai componenti X del modo
            phi_x = phi[::3]
            sum_phi = np.sum(np.abs(phi_x))
            
            if sum_phi > 0:
                for i, node_id in enumerate(sorted(self.nodes.keys())):
                    if self.nodes[node_id][1] > min(coord[1] for coord in self.nodes.values()):
                        Fi = F_tot * np.abs(phi_x[i]) / sum_phi
                        forces[node_id] = np.array([Fi, 0, 0])
                        
        except Exception as e:
            logger.warning(f"Errore in modal pattern, uso triangolare: {e}")
            return self._triangular_load_pattern()
            
        return forces
        
    def _check_hinges(self, solution: Dict) -> List[Dict]:
        """Verifica formazione cerniere plastiche"""
        hinges = []
        
        for i, elem in enumerate(self.elements):
            if i < len(solution['element_forces']):
                forces = solution['element_forces'][i]
                
                # Calcola capacità elemento
                if elem.type == "pier":
                    capacity = self._pier_capacity(elem, forces['N'])
                else:  # spandrel
                    capacity = self._spandrel_capacity(elem)
                    
                # Verifica domanda/capacità
                if capacity['M_max'] > 0:
                    DCR_M_i = abs(forces['M_i']) / capacity['M_max']
                    DCR_M_j = abs(forces['M_j']) / capacity['M_max']
                else:
                    DCR_M_i = DCR_M_j = 999
                    
                if capacity['V_max'] > 0:
                    DCR_V = abs(forces['V']) / capacity['V_max']
                else:
                    DCR_V = 999
                
                # Formazione cerniera flessionale
                if DCR_M_i > 1.0:
                    hinges.append({
                        'element': i,
                        'location': 'i',
                        'type': 'flexure',
                        'DCR': DCR_M_i
                    })
                    
                if DCR_M_j > 1.0:
                    hinges.append({
                        'element': i,
                        'location': 'j',
                        'type': 'flexure',
                        'DCR': DCR_M_j
                    })
                    
                # Formazione cerniera a taglio
                if DCR_V > 1.0:
                    hinges.append({
                        'element': i,
                        'location': 'center',
                        'type': 'shear',
                        'DCR': DCR_V
                    })
                    
        return hinges
        
    def _pier_capacity(self, elem: FrameElement, N: float) -> Dict:
        """Calcola capacità maschio murario secondo NTC2018"""
        geom = elem.geometry
        mat = elem.material.get_design_values()
        
        # Tensione normale media
        sigma0 = abs(N) / geom.area if geom.area > 0 else 0
        
        # Momento ultimo (presso-flessione)
        if sigma0 < 0.85 * mat['fcd'] * 1000:  # kPa
            l = geom.length
            t = geom.thickness
            fcd = mat['fcd'] * 1000  # kPa
            
            # Formula NTC per muratura non armata
            Mu = (l * t * sigma0 / 2) * (1 - sigma0 / (0.85 * fcd))
        else:
            Mu = 0
            
        # Taglio ultimo - minimo tra tre meccanismi
        fvd0 = mat['fvd0'] * 1000  # kPa
        
        # 1. Taglio-scorrimento
        Vt1 = geom.length * geom.thickness * fvd0 * np.sqrt(1 + sigma0 / fvd0)
        
        # 2. Taglio-trazione diagonale
        b = geom.shape_factor  # fattore di forma
        Vt2 = geom.length * geom.thickness * b * (fvd0 + 0.4 * sigma0)
        
        # 3. Taglio-presso flessione
        h0 = geom.h0 if hasattr(geom, 'h0') else geom.height * 0.5
        Vt3 = Mu / h0 if h0 > 0 else 0
        
        V_max = min(Vt1, Vt2, Vt3)
        
        return {
            'M_max': max(Mu, 0),
            'V_max': max(V_max, 0),
            'mechanism': 'shear' if V_max < Vt3 else 'flexure'
        }
        
    def _spandrel_capacity(self, elem: FrameElement) -> Dict:
        """Calcola capacità fascia di piano"""
        geom = elem.geometry
        mat = elem.material.get_design_values()
        
        # Verifica presenza tirante o arco
        if hasattr(geom, 'tie_rod') and geom.tie_rod:
            # Meccanismo con tirante
            T = geom.tie_rod.capacity  # kN
            z = 0.9 * geom.height  # braccio
            Mr = T * z
            Vr = 2 * Mr / geom.length if geom.length > 0 else 0
            
        elif hasattr(geom, 'arch_rise') and geom.arch_rise > 0:
            # Meccanismo ad arco
            f = geom.arch_rise  # monta dell'arco
            t = geom.thickness
            fcd = mat['fcd'] * 1000  # kPa
            
            # Spinta orizzontale
            H = 0.5 * fcd * t * f
            
            # Momento resistente
            Mr = H * f * 8 / 3
            Vr = 2 * H * f / geom.length if geom.length > 0 else 0
            
        else:
            # Nessun meccanismo resistente affidabile
            Mr = 0
            Vr = geom.length * geom.thickness * mat['fvd0'] * 1000 * 0.1  # Minimo
            
        return {
            'M_max': max(Mr, 0),
            'V_max': max(Vr, 0),
            'mechanism': 'tie_rod' if hasattr(geom, 'tie_rod') else 'arch'
        }
        
    def _update_model_with_hinges(self, hinges: List[Dict]):
        """Aggiorna rigidezza elementi con cerniere plastiche"""
        for hinge in hinges:
            elem = self.elements[hinge['element']]
            
            # Fattore di riduzione rigidezza
            reduction = 0.001
            
            if hinge['type'] == 'flexure':
                # Riduce rigidezza flessionale
                if hinge['location'] == 'i':
                    elem.hinge_i = True
                    elem.k_local[2, 2] *= reduction  # Rotazione nodo i
                    elem.k_local[2, 5] *= reduction
                    elem.k_local[5, 2] *= reduction
                elif hinge['location'] == 'j':
                    elem.hinge_j = True
                    elem.k_local[5, 5] *= reduction  # Rotazione nodo j
                    elem.k_local[5, 2] *= reduction
                    elem.k_local[2, 5] *= reduction
                    
            elif hinge['type'] == 'shear':
                # Riduce rigidezza a taglio
                elem.shear_failure = True
                elem.k_local[1, 1] *= reduction
                elem.k_local[4, 4] *= reduction
                elem.k_local[1, 4] *= reduction
                elem.k_local[4, 1] *= reduction
                
        # Riassembla matrice di rigidezza globale
        self.assemble_stiffness_matrix()
        
    def _calculate_base_shear(self, reactions: np.ndarray) -> float:
        """Calcola taglio totale alla base"""
        V_base = 0
        
        # Somma reazioni orizzontali ai vincoli
        for constraint in self.constraints:
            if constraint['type'] in ['fixed', 'pinned']:
                node_id = constraint['node']
                if node_id in self.node_dofs:
                    dof_x = self.node_dofs[node_id][0]
                    V_base += abs(reactions[dof_x])
                    
        return V_base
        
    def _calculate_top_displacement(self, displacements: np.ndarray) -> float:
        """Calcola spostamento in sommità"""
        # Trova nodo più alto
        if not self.nodes:
            return 0
            
        top_node = max(self.nodes.items(), key=lambda x: x[1][1])[0]
        
        if top_node in self.node_dofs:
            dof_x = self.node_dofs[top_node][0]
            return abs(displacements[dof_x])
            
        return 0
        
    def _get_building_height(self) -> float:
        """Altezza totale edificio"""
        if not self.nodes:
            return 0
            
        y_coords = [coord[1] for coord in self.nodes.values()]
        return max(y_coords) - min(y_coords)
        
    def _bilinearize_pushover(self, curve: Dict) -> Dict:
        """Bilinearizzazione curva pushover con equal energy"""
        V = np.array(curve['V_base'])
        delta = np.array(curve['delta_top'])
        
        if len(V) < 3:
            return None
            
        # Trova punto di snervamento con criterio energetico
        V_max = np.max(V)
        
        # Stima iniziale: 70% del massimo
        V_y_target = 0.7 * V_max
        
        # Trova indice corrispondente
        idx_y = np.argmin(np.abs(V - V_y_target))
        
        # Energia sotto la curva
        # Compatibilità numpy (trapz deprecato in numpy 2.0)
        try:
            E_total = np.trapezoid(V[:idx_y+1], delta[:idx_y+1])
        except AttributeError:
            E_total = np.trapz(V[:idx_y+1], delta[:idx_y+1])
        
        # Ottimizza V_y per equal energy
        best_error = float('inf')
        best_params = None
        
        for i in range(max(1, idx_y-5), min(len(V)-1, idx_y+5)):
            V_y = V[i]
            delta_y = delta[i]
            
            # Pendenza elastica
            K_el = V_y / delta_y if delta_y > 0 else 0
            
            # Pendenza plastica (assumendo che arrivi a V_max)
            idx_max = np.argmax(V)
            if idx_max > i and delta[idx_max] > delta_y:
                K_pl = (V[idx_max] - V_y) / (delta[idx_max] - delta_y)
            else:
                K_pl = 0
                
            # Energia bilineare fino a delta_y
            E_bilinear = 0.5 * V_y * delta_y
            
            # Errore
            error = abs(E_bilinear - E_total) / E_total if E_total > 0 else float('inf')
            
            if error < best_error:
                best_error = error
                best_params = {
                    'V_y': V_y,
                    'delta_y': delta_y,
                    'K_el': K_el,
                    'K_pl': K_pl,
                    'V_max': V_max,
                    'delta_u': delta[-1]
                }
                
        return best_params
        
    def _find_performance_point_N2(self, curve: Dict, bilinear: Dict) -> Dict:
        """Trova punto di performance con metodo N2 (Fajfar)"""
        if not bilinear:
            return None
            
        # Parametri sistema SDOF equivalente
        V_y = bilinear['V_y']
        delta_y = bilinear['delta_y']
        K_star = V_y / delta_y if delta_y > 0 else 1e6
        
        # Massa totale partecipante (stima)
        if self.M_global is not None:
            M_tot = np.sum(self.M_global.diagonal()[:len(self.nodes)])
        else:
            M_tot = 100.0  # Stima default
            
        # Fattore di partecipazione (primo modo)
        Gamma = 1.3  # Valore tipico per edifici regolari
        
        # Sistema SDOF equivalente
        M_star = M_tot / Gamma
        T_star = 2 * np.pi * np.sqrt(M_star / K_star)
        
        # Spettro di risposta di progetto
        Sa_d = self._design_spectrum(T_star)
        
        # Domanda in termini di spostamento
        if T_star < 0.5:  # Periodo corto
            Sd_demand = Sa_d * T_star**2 / (4 * np.pi**2)
        else:
            # Riduzione per duttilità
            mu = 3.0  # Duttilità target
            R_mu = (mu - 1) * T_star / 0.5 + 1
            Sd_demand = Sa_d * T_star**2 / (4 * np.pi**2 * R_mu)
            
        # Spostamento target struttura MDOF
        delta_target = Gamma * Sd_demand
        
        # Trova punto sulla curva pushover
        V = np.array(curve['V_base'])
        delta = np.array(curve['delta_top'])
        
        idx = np.argmin(np.abs(delta - delta_target))
        
        return {
            'V_base': V[idx],
            'delta_top': delta[idx],
            'delta_target': delta_target,
            'T_star': T_star,
            'Sa_demand': Sa_d,
            'Sd_demand': Sd_demand,
            'ductility_demand': delta[idx] / delta_y if delta_y > 0 else 1.0
        }
        
    def _design_spectrum(self, T: float, ag: float = 0.25, soil: str = "B") -> float:
        """Spettro di progetto elastico secondo NTC2018/EC8"""
        # Parametri spettrali per suolo B
        S = 1.2
        TB = 0.15
        TC = 0.5
        TD = 2.0
        
        # Fattore di struttura base
        q = 2.0
        
        # Spettro elastico
        if T <= TB:
            Se = ag * S * (1 + T/TB * 2.5)
        elif T <= TC:
            Se = ag * S * 2.5
        elif T <= TD:
            Se = ag * S * 2.5 * (TC/T)
        else:
            Se = ag * S * 2.5 * (TC*TD/T**2)
            
        # Spettro di progetto
        Sd = Se / q
        
        return Sd
        
    def plot_pushover_curve(self, results: Dict, filename: str = None):
        """Visualizza curva pushover con punti notevoli"""
        plt.figure(figsize=(10, 6))
        
        # Curva pushover
        V = results['curve']['V_base']
        delta = results['curve']['delta_top']
        plt.plot(delta, V, 'b-', linewidth=2, label='Curva Pushover')
        
        # Bilineare
        if results['bilinear']:
            bil = results['bilinear']
            delta_y = bil['delta_y']
            V_y = bil['V_y']
            delta_u = bil['delta_u']
            
            # Tratto elastico
            delta_el = np.linspace(0, delta_y, 50)
            V_el = bil['K_el'] * delta_el
            plt.plot(delta_el, V_el, 'r--', linewidth=2, label='Bilineare')
            
            # Tratto plastico
            if bil['K_pl'] >= 0:
                delta_pl = np.linspace(delta_y, delta_u, 50)
                V_pl = V_y + bil['K_pl'] * (delta_pl - delta_y)
                plt.plot(delta_pl, V_pl, 'r--', linewidth=2)
            
            plt.plot(delta_y, V_y, 'ro', markersize=8, label=f'Snervamento')
            
        # Punto di performance
        if results['performance_point']:
            pp = results['performance_point']
            plt.plot(pp['delta_top'], pp['V_base'], 'go', markersize=10, 
                    label=f'Punto Performance (μ={pp["ductility_demand"]:.2f})')
            
        # Livelli prestazionali
        for level_name, level_data in results['performance_levels'].items():
            plt.axvline(level_data['delta_top'], color='gray', linestyle=':', 
                       alpha=0.5, label=f'{level_name}')
            
        plt.xlabel('Spostamento in sommità [m]')
        plt.ylabel('Taglio alla base [kN]')
        plt.title('Curva di Capacità - Analisi Pushover')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
    def export_results(self, results: Dict, filename: str):
        """Esporta risultati in formato JSON"""
        import json
        
        # Converti array numpy in liste
        def convert_arrays(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            return obj
            
        results_serializable = convert_arrays(results)
        
        with open(filename, 'w') as f:
            json.dump(results_serializable, f, indent=2)
            
        logger.info(f"Risultati esportati in {filename}")


def create_frame_from_wall_data(wall_data: Dict, material: MaterialProperties) -> EquivalentFrame:
    """Crea modello di telaio equivalente da dati parete"""
    frame = EquivalentFrame()

    # Estrai geometria - supporta sia formato complesso che semplice
    walls = wall_data.get('walls', [])
    openings = wall_data.get('openings', [])
    floors_data = wall_data.get('floors', [])

    # Se non ci sono walls ma c'è length/height, crea parete singola
    if not walls and 'length' in wall_data:
        L = wall_data['length']
        H = wall_data['height']
        t = wall_data['thickness']
        n_floors = wall_data.get('floors', 1) if isinstance(wall_data.get('floors'), int) else 1

        # Crea modello semplificato per singola parete
        node_id = 0

        # Nodi: base sinistra, base destra, e per ogni piano
        h_floor = H / n_floors if n_floors > 0 else H

        # Nodi alla base
        frame.add_node(0, 0, 0)          # Base sinistra
        frame.add_node(1, L, 0)          # Base destra
        frame.add_constraint(0, "fixed")
        frame.add_constraint(1, "fixed")

        node_id = 2
        prev_left = 0
        prev_right = 1

        # Nodi per ogni piano
        for i in range(1, n_floors + 1):
            y = i * h_floor
            left_node = node_id
            right_node = node_id + 1
            frame.add_node(left_node, 0, y)
            frame.add_node(right_node, L, y)

            # Crea elementi pier verticali (maschi)
            from ...geometry import GeometryPier

            # Maschio sinistro (length=larghezza muro in pianta, height=altezza piano)
            geom_left = GeometryPier(length=t, height=h_floor, thickness=t)
            elem_left = FrameElement(
                element_id=len(frame.elements),
                i_node=prev_left,
                j_node=left_node,
                geometry=geom_left,
                material=material,
                element_type='pier'
            )
            frame.add_element(elem_left)

            # Maschio destro
            geom_right = GeometryPier(length=t, height=h_floor, thickness=t)
            elem_right = FrameElement(
                element_id=len(frame.elements),
                i_node=prev_right,
                j_node=right_node,
                geometry=geom_right,
                material=material,
                element_type='pier'
            )
            frame.add_element(elem_right)

            # Fascia orizzontale (spandrel) a ogni piano
            from ...geometry import GeometrySpandrel
            geom_span = GeometrySpandrel(length=L, height=h_floor*0.3, thickness=t)
            elem_span = FrameElement(
                element_id=len(frame.elements),
                i_node=left_node,
                j_node=right_node,
                geometry=geom_span,
                material=material,
                element_type='spandrel'
            )
            frame.add_element(elem_span)

            prev_left = left_node
            prev_right = right_node
            node_id += 2

        return frame

    # Formato complesso con coordinate walls
    node_id = 0
    node_map = {}  # (x,y) -> node_id

    # Nodi da pareti
    for wall in walls:
        for point in [wall['start'], wall['end']]:
            key = (round(point[0], 3), round(point[1], 3))
            if key not in node_map:
                frame.add_node(node_id, point[0], point[1])
                node_map[key] = node_id
                node_id += 1

    # Nodi da solai
    if isinstance(floors_data, list):
        for floor in floors_data:
            y = floor['level']
            for wall in walls:
                if wall['start'][1] <= y <= wall['end'][1] or wall['end'][1] <= y <= wall['start'][1]:
                    x = wall['start'][0]
                    key = (round(x, 3), round(y, 3))
                    if key not in node_map:
                        frame.add_node(node_id, x, y)
                        node_map[key] = node_id
                        node_id += 1

    # Aggiungi vincoli alla base
    if node_map:
        y_min = min(coord[1] for coord in node_map.keys())
        for (x, y), nid in node_map.items():
            if abs(y - y_min) < 0.01:
                frame.add_constraint(nid, "fixed")

    return frame


def _analyze_frame(wall_data: Dict, material: MaterialProperties,
                   loads: Dict, options: Dict) -> Dict:
    """Funzione principale per analisi con telaio equivalente"""
    logger.info("Avvio analisi con metodo del telaio equivalente")
    
    # Crea modello
    frame = create_frame_from_wall_data(wall_data, material)
    
    # Opzioni analisi
    analysis_options = AnalysisOptions(
        analysis_type=options.get('analysis_type', 'pushover'),
        lateral_pattern=options.get('lateral_pattern', 'triangular'),
        target_drift=options.get('target_drift', 0.04),
        n_modes=options.get('n_modes', 6)
    )
    
    # Risultati
    results = {
        'method': 'FRAME',
        'model': frame,
        'analyses': {}
    }
    
    # Analisi statica (carichi verticali)
    if 'vertical' in loads and loads['vertical']:
        logger.info("Esecuzione analisi statica per carichi verticali")
        # Converti carico scalare in dict di forze nodali
        V_load = loads['vertical']
        if isinstance(V_load, (int, float)):
            # Distribuisci il carico ai nodi superiori
            if frame.nodes:
                y_max = max(coord[1] for coord in frame.nodes.values())
                top_nodes = [nid for nid, coord in frame.nodes.items()
                            if abs(coord[1] - y_max) < 0.01]
                if top_nodes:
                    load_per_node = V_load / len(top_nodes)
                    V_load = {nid: np.array([0, -load_per_node, 0]) for nid in top_nodes}
                else:
                    V_load = {}
            else:
                V_load = {}
        if V_load:
            static_results = frame.solve_static(V_load)
            results['analyses']['static'] = static_results
        
    # Assembla matrice di rigidezza
    frame.assemble_stiffness_matrix()

    # Analisi modale
    if analysis_options.analysis_type in ['modal', 'pushover']:
        logger.info(f"Esecuzione analisi modale ({analysis_options.n_modes} modi)")

        # Masse di piano
        floor_masses = {}
        floors_data = wall_data.get('floors', [])

        # Se floors è un intero (numero piani) invece di lista
        if isinstance(floors_data, int):
            n_floors = floors_data
            H = wall_data.get('height', 3.0)
            h_floor = H / n_floors if n_floors > 0 else H
            default_mass = 10000  # kg per piano (stima)

            for nid, coord in frame.nodes.items():
                # Assegna massa ai nodi non alla base
                if coord[1] > 0.01:
                    floor_masses[nid] = default_mass / 2  # Divide per nodi sx e dx

        elif isinstance(floors_data, list):
            for floor in floors_data:
                # Gestisce sia lista di float (quote) che lista di dict
                if isinstance(floor, (int, float)):
                    level = float(floor)
                    mass = 1000  # massa default per piano
                else:
                    level = floor.get('level', 0)
                    mass = floor.get('mass', 1000)

                level_nodes = [nid for nid, coord in frame.nodes.items()
                              if abs(coord[1] - level) < 0.01]

                if level_nodes:
                    mass_per_node = mass / len(level_nodes)
                    for nid in level_nodes:
                        floor_masses[nid] = mass_per_node

        # Se non ci sono masse, usa default
        if not floor_masses:
            for nid, coord in frame.nodes.items():
                if coord[1] > 0.01:
                    floor_masses[nid] = 5000  # kg default

        frame.assemble_mass_matrix(floor_masses)
        modal_results = frame.solve_modal(analysis_options.n_modes)
        results['analyses']['modal'] = modal_results
        
        # Report modi
        logger.info("Periodi propri:")
        for i, T in enumerate(modal_results['periods'][:3]):
            logger.info(f"  Modo {i+1}: T = {T:.3f} s, f = {1/T:.2f} Hz")
            
    # Analisi pushover
    if analysis_options.analysis_type == 'pushover':
        logger.info(f"Esecuzione analisi pushover (pattern: {analysis_options.lateral_pattern})")
        
        pushover_results = frame.pushover_analysis(
            lateral_pattern=analysis_options.lateral_pattern,
            target_drift=analysis_options.target_drift,
            options=analysis_options
        )
        results['analyses']['pushover'] = pushover_results
        
        # Report risultati chiave
        if pushover_results['performance_point']:
            pp = pushover_results['performance_point']
            logger.info(f"Punto di performance:")
            logger.info(f"  V_base = {pp['V_base']:.1f} kN")
            logger.info(f"  Delta = {pp['delta_top']:.3f} m")
            logger.info(f"  Duttilità = {pp['ductility_demand']:.2f}")
            
        # Genera grafici se richiesto
        if options.get('plot_results', False):
            frame.plot_pushover_curve(pushover_results, 
                                    filename=options.get('plot_filename'))
            
    # Esporta risultati se richiesto
    if 'export_filename' in options:
        frame.export_results(results, options['export_filename'])
        
    # Sommario finale
    results['summary'] = {
        'n_nodes': len(frame.nodes),
        'n_elements': len(frame.elements),
        'n_constraints': len(frame.constraints),
        'analyses_performed': list(results['analyses'].keys())
    }
    
    if 'pushover' in results['analyses']:
        pushover = results['analyses']['pushover']
        results['summary']['pushover'] = {
            'max_base_shear': max(pushover['curve']['V_base']) if pushover['curve']['V_base'] else 0,
            'max_drift': max(pushover['curve']['drift']) if pushover['curve']['drift'] else 0,
            'n_hinges': len(pushover['hinges']),
            'performance_levels': list(pushover['performance_levels'].keys())
        }
        
    logger.info("Analisi con telaio equivalente completata")
    
    return results