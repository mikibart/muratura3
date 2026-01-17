# analyses/frame/element.py
"""
Modulo per la definizione degli elementi del telaio equivalente per analisi di strutture in muratura.
Implementa elementi maschio (pier) e fascia (spandrel) con deformabilità a taglio.
"""

import numpy as np
from typing import Dict, Union, Tuple, Optional, List
from scipy.sparse import csr_matrix
from dataclasses import dataclass
import warnings

# Import dalla struttura del progetto
from ...geometry import GeometryPier, GeometrySpandrel
from ...materials import MaterialProperties


@dataclass
class ElementState:
    """Stato corrente dell'elemento"""
    is_elastic: bool = True
    is_cracked: bool = False
    is_failed: bool = False
    damage_level: float = 0.0
    plastic_hinge_i: bool = False
    plastic_hinge_j: bool = False
    failure_mode: Optional[str] = None  # 'flexure', 'shear', 'compression'


class FrameElement:
    """
    Elemento per telaio equivalente con comportamento non lineare.
    
    Attributes:
        i_node: Nodo iniziale
        j_node: Nodo finale
        geometry: Geometria dell'elemento (pier o spandrel)
        material: Proprietà del materiale
        element_type: Tipo di elemento ("pier" o "spandrel")
    """
    
    def __init__(self, 
                 element_id: int,
                 i_node: int, 
                 j_node: int, 
                 geometry: Union[GeometryPier, GeometrySpandrel],
                 material: MaterialProperties, 
                 element_type: str):
        """
        Inizializza l'elemento frame.
        
        Args:
            element_id: ID univoco dell'elemento
            i_node: Nodo iniziale
            j_node: Nodo finale
            geometry: Geometria dell'elemento
            material: Proprietà del materiale
            element_type: "pier" o "spandrel"
        """
        self.id = element_id
        self.i_node = i_node
        self.j_node = j_node
        self.geometry = geometry
        self.material = material
        self.type = element_type.lower()
        
        if self.type not in ["pier", "spandrel"]:
            raise ValueError(f"Tipo elemento non valido: {element_type}")
        
        # Matrice di rigidezza locale iniziale (elastica)
        self.k_local = self._compute_local_stiffness()
        self.k_local_elastic = self.k_local.copy()  # Backup rigidezza elastica
        
        # Matrice di trasformazione
        self.T = None
        
        # Forze interne
        self.forces = {'N': 0.0, 'V': 0.0, 'M_i': 0.0, 'M_j': 0.0}
        
        # Spostamenti nodali
        self.displacements = np.zeros(6)
        
        # Stato dell'elemento
        self.state = ElementState()
        
        # Storia delle sollecitazioni per analisi non lineare
        self.force_history = []
        self.displacement_history = []
        
        # Capacità dell'elemento
        self.capacities = self._compute_capacities()
        
    def _compute_local_stiffness(self) -> np.ndarray:
        """
        Calcola matrice di rigidezza locale 6x6.
        
        Returns:
            Matrice di rigidezza locale [kN, m]
        """
        if self.type == "pier":
            return self._pier_stiffness_matrix()
        else:
            return self._spandrel_stiffness_matrix()
            
    def _pier_stiffness_matrix(self) -> np.ndarray:
        """
        Matrice di rigidezza maschio murario con deformabilità a taglio.
        Basata sul modello di Timoshenko per elementi tozzi.
        
        Returns:
            Matrice di rigidezza 6x6 [kN, m]
        """
        # Proprietà materiale in unità SI
        E = self.material.E * 1e6  # MPa -> Pa
        G = self.material.G * 1e6  # MPa -> Pa
        
        # Proprietà geometriche
        A = self.geometry.area  # m²
        I = self.geometry.inertia  # m⁴
        L = self.geometry.height  # m
        
        # Fattore di forma per taglio (rettangolare)
        k_s = 5/6  # Fattore di correzione taglio per sezione rettangolare
        
        # Area efficace a taglio
        A_v = k_s * A
        
        # Parametro di deformabilità a taglio (φ)
        phi = 12 * E * I / (G * A_v * L**2)
        
        # Matrice 6x6 con effetti del taglio
        k = np.zeros((6, 6))
        
        # Termini assiali (u1, u4)
        EA_L = E * A / L
        k[0, 0] = EA_L
        k[0, 3] = -EA_L
        k[3, 0] = -EA_L
        k[3, 3] = EA_L
        
        # Termini flessionali con deformabilità a taglio
        EI_L3 = E * I / L**3
        EI_L2 = E * I / L**2
        EI_L = E * I / L
        
        # Coefficienti modificati per taglio
        k11 = 12 * EI_L3 / (1 + phi)
        k12 = 6 * EI_L2 / (1 + phi)
        k22 = (4 + phi) * EI_L / (1 + phi)
        k23 = (2 - phi) * EI_L / (1 + phi)
        
        # Assemblaggio termini flessionali
        # Riga/colonna 1 (v1)
        k[1, 1] = k11
        k[1, 2] = k12
        k[1, 4] = -k11
        k[1, 5] = k12
        
        # Riga/colonna 2 (θ1)
        k[2, 1] = k12
        k[2, 2] = k22
        k[2, 4] = -k12
        k[2, 5] = k23
        
        # Riga/colonna 4 (v2)
        k[4, 1] = -k11
        k[4, 2] = -k12
        k[4, 4] = k11
        k[4, 5] = -k12
        
        # Riga/colonna 5 (θ2)
        k[5, 1] = k12
        k[5, 2] = k23
        k[5, 4] = -k12
        k[5, 5] = k22
        
        return k / 1000  # Pa -> kN/m
        
    def _spandrel_stiffness_matrix(self) -> np.ndarray:
        """
        Matrice di rigidezza fascia di piano (trave tozza).
        Considera l'alta deformabilità a taglio tipica delle fasce.
        
        Returns:
            Matrice di rigidezza 6x6 [kN, m]
        """
        # Proprietà materiale
        E = self.material.E * 1e6  # MPa -> Pa
        G = self.material.G * 1e6  # MPa -> Pa
        
        # Proprietà geometriche
        t = self.geometry.thickness  # m
        h = self.geometry.height  # m
        L = self.geometry.length  # m
        A = t * h  # m²
        
        # Momento d'inerzia per sezione rettangolare
        I = t * h**3 / 12  # m⁴
        
        # Area efficace a taglio (rettangolare)
        A_v = 5/6 * A
        
        # Parametro di deformabilità a taglio aumentato per trave tozza
        # Le fasce hanno rapporto L/h basso, quindi alta deformabilità a taglio
        factor = max(3.0, 10.0 * h / L)  # Fattore amplificativo per trave tozza
        phi = factor * 12 * E * I / (G * A_v * L**2)
        
        k = np.zeros((6, 6))
        
        # Rigidezza assiale molto alta (fascia sempre compressa)
        # Amplificata per simulare l'effetto puntone
        EA_L = 10 * E * A / L
        k[0, 0] = EA_L
        k[0, 3] = -EA_L
        k[3, 0] = -EA_L
        k[3, 3] = EA_L
        
        # Rigidezza flessionale ridotta per alta deformabilità a taglio
        EI_L3 = E * I / L**3
        EI_L2 = E * I / L**2
        EI_L = E * I / L
        
        k11 = 12 * EI_L3 / (1 + phi)
        k12 = 6 * EI_L2 / (1 + phi)
        k22 = (4 + phi) * EI_L / (1 + phi)
        k23 = (2 - phi) * EI_L / (1 + phi)
        
        # Assemblaggio con segni corretti per convenzione
        k[1, 1] = k11
        k[1, 2] = -k12  # Segno negativo per convenzione fascia
        k[1, 4] = -k11
        k[1, 5] = -k12
        
        k[2, 1] = -k12
        k[2, 2] = k22
        k[2, 4] = k12
        k[2, 5] = k23
        
        k[4, 1] = -k11
        k[4, 2] = k12
        k[4, 4] = k11
        k[4, 5] = k12
        
        k[5, 1] = -k12
        k[5, 2] = k23
        k[5, 4] = k12
        k[5, 5] = k22
        
        return k / 1000  # Pa -> kN/m
        
    def set_transformation_matrix(self, node_coords: Dict[int, Tuple[float, float]]):
        """
        Imposta matrice di trasformazione da coordinate locali a globali.
        
        Args:
            node_coords: Dizionario {node_id: (x, y)} delle coordinate nodali
        """
        xi, yi = node_coords[self.i_node]
        xj, yj = node_coords[self.j_node]
        
        # Lunghezza elemento
        dx = xj - xi
        dy = yj - yi
        L = np.sqrt(dx**2 + dy**2)
        
        if L < 1e-6:
            raise ValueError(f"Elemento {self.id} ha lunghezza zero")
        
        # Coseni direttori
        c = dx / L  # cos(θ)
        s = dy / L  # sin(θ)
        
        # Matrice di rotazione 3x3
        R = np.array([
            [c,  s, 0],
            [-s, c, 0],
            [0,  0, 1]
        ])
        
        # Matrice di trasformazione 6x6
        self.T = np.zeros((6, 6))
        self.T[0:3, 0:3] = R
        self.T[3:6, 3:6] = R
        
    def get_global_stiffness(self) -> np.ndarray:
        """
        Restituisce matrice di rigidezza in coordinate globali.
        
        Returns:
            Matrice di rigidezza globale 6x6
        """
        if self.T is None:
            raise ValueError("Matrice di trasformazione non impostata")
        return self.T.T @ self.k_local @ self.T

    def get_mass_matrix(self) -> np.ndarray:
        """
        Calcola matrice di massa consistente 6x6.

        Returns:
            Matrice di massa locale [kg]
        """
        # Proprietà
        if self.type == "pier":
            L = self.geometry.height
            A = self.geometry.area
        else:
            L = self.geometry.length
            A = self.geometry.area

        # Densità muratura tipica
        rho = 1800  # kg/m³

        # Massa totale elemento
        m_tot = rho * A * L

        # Matrice di massa consistente (simplified lumped)
        m = np.zeros((6, 6))

        # Masse traslazionali (metà per nodo)
        m[0, 0] = m_tot / 2  # u1
        m[1, 1] = m_tot / 2  # v1
        m[3, 3] = m_tot / 2  # u2
        m[4, 4] = m_tot / 2  # v2

        # Inerzia rotazionale (piccola)
        I_rot = m_tot * L**2 / 12
        m[2, 2] = I_rot / 2  # θ1
        m[5, 5] = I_rot / 2  # θ2

        return m

    def compute_internal_forces(self, u_global: np.ndarray) -> Dict[str, float]:
        """
        Calcola forze interne dall'spostamento nodale globale.
        
        Args:
            u_global: Vettore spostamenti globali [u1, v1, θ1, u2, v2, θ2]
            
        Returns:
            Dizionario con forze interne {N, V, M_i, M_j}
        """
        if self.T is None:
            raise ValueError("Matrice di trasformazione non impostata")
            
        # Spostamenti locali
        u_local = self.T @ u_global
        self.displacements = u_local.copy()
        
        # Forze locali
        f_local = self.k_local @ u_local
        
        # Estrai componenti significative
        self.forces['N'] = f_local[0]    # Forza assiale
        self.forces['V'] = f_local[1]    # Taglio
        self.forces['M_i'] = f_local[2]  # Momento nodo i
        self.forces['M_j'] = f_local[5]  # Momento nodo j
        
        # Aggiorna storia
        self.force_history.append(self.forces.copy())
        self.displacement_history.append(u_local.copy())
        
        return self.forces
        
    def _compute_capacities(self) -> Dict[str, float]:
        """
        Calcola le capacità resistenti dell'elemento.
        
        Returns:
            Dizionario con capacità {N_max, V_max, M_max}
        """
        capacities = {}
        
        if self.type == "pier":
            # Capacità maschio murario
            t = self.geometry.thickness
            b = self.geometry.width
            h = self.geometry.height
            
            # Resistenza a compressione
            f_m = self.material.fcm * 1000  # MPa -> kPa
            capacities['N_max'] = f_m * t * b
            
            # Resistenza a taglio (criterio di Mohr-Coulomb)
            f_v0 = self.material.tau0 * 1000  # MPa -> kPa
            mu = 0.4  # Coefficiente d'attrito
            sigma_n = 0.5 * f_m  # Tensione normale media stimata
            capacities['V_max'] = (f_v0 + mu * sigma_n) * t * b
            
            # Momento ultimo (presso-flessione)
            capacities['M_max'] = capacities['N_max'] * b / 6
            
        else:
            # Capacità fascia di piano
            t = self.geometry.thickness
            h = self.geometry.height
            L = self.geometry.length
            
            # Alta capacità assiale (sempre compressa)
            f_m = self.material.fcm * 1000
            capacities['N_max'] = 2 * f_m * t * h
            
            # Bassa capacità a taglio
            f_v0 = self.material.tau0 * 1000
            capacities['V_max'] = f_v0 * t * h * 0.5  # Ridotta
            
            # Bassa capacità flessionale
            capacities['M_max'] = capacities['V_max'] * L / 4
            
        return capacities
        
    def check_failure(self, safety_factor: float = 1.0) -> bool:
        """
        Verifica lo stato di rottura dell'elemento.
        
        Args:
            safety_factor: Fattore di sicurezza
            
        Returns:
            True se l'elemento è in rottura
        """
        # Verifica compressione
        if abs(self.forces['N']) > self.capacities['N_max'] / safety_factor:
            self.state.is_failed = True
            self.state.failure_mode = 'compression'
            return True
            
        # Verifica taglio
        if abs(self.forces['V']) > self.capacities['V_max'] / safety_factor:
            self.state.is_failed = True
            self.state.failure_mode = 'shear'
            return True
            
        # Verifica flessione
        M_max = max(abs(self.forces['M_i']), abs(self.forces['M_j']))
        if M_max > self.capacities['M_max'] / safety_factor:
            self.state.is_failed = True
            self.state.failure_mode = 'flexure'
            
            # Identifica cerniere plastiche
            if abs(self.forces['M_i']) > self.capacities['M_max'] / safety_factor:
                self.state.plastic_hinge_i = True
            if abs(self.forces['M_j']) > self.capacities['M_max'] / safety_factor:
                self.state.plastic_hinge_j = True
                
            return True
            
        return False
        
    def update_stiffness(self, reduction_factor: float = None):
        """
        Aggiorna la rigidezza per comportamento non lineare.
        
        Args:
            reduction_factor: Fattore di riduzione rigidezza (0-1)
        """
        if reduction_factor is not None:
            self.k_local = self.k_local_elastic * reduction_factor
            self.state.is_cracked = True
            self.state.damage_level = 1 - reduction_factor
        else:
            # Riduzione automatica basata sul danneggiamento
            if self.state.is_failed:
                self.k_local = self.k_local_elastic * 0.01  # Rigidezza residua
            elif self.state.plastic_hinge_i or self.state.plastic_hinge_j:
                self.k_local = self.k_local_elastic * 0.1
            elif self.state.is_cracked:
                self.k_local = self.k_local_elastic * 0.5
                
    def get_element_forces_vector(self) -> np.ndarray:
        """
        Restituisce vettore forze interne in coordinate locali.
        
        Returns:
            Vettore forze [Ni, Vi, Mi, Nj, Vj, Mj]
        """
        return np.array([
            self.forces['N'],
            self.forces['V'],
            self.forces['M_i'],
            -self.forces['N'],  # Equilibrio
            -self.forces['V'],  # Equilibrio
            self.forces['M_j']
        ])
        
    def get_stress_state(self) -> Dict[str, float]:
        """
        Calcola lo stato tensionale nell'elemento.
        
        Returns:
            Dizionario con tensioni {sigma_max, tau_max, sigma_vm}
        """
        A = self.geometry.area
        
        if self.type == "pier":
            W = self.geometry.thickness * self.geometry.width**2 / 6
        else:
            W = self.geometry.thickness**2 * self.geometry.height / 6
            
        # Tensione normale massima
        sigma_N = self.forces['N'] / A
        sigma_M = max(abs(self.forces['M_i']), abs(self.forces['M_j'])) / W
        sigma_max = abs(sigma_N) + sigma_M
        
        # Tensione tangenziale
        tau_max = 1.5 * abs(self.forces['V']) / A
        
        # Tensione di Von Mises
        sigma_vm = np.sqrt(sigma_max**2 + 3 * tau_max**2)
        
        return {
            'sigma_max': sigma_max,
            'tau_max': tau_max,
            'sigma_vm': sigma_vm,
            'sigma_N': sigma_N,
            'sigma_M': sigma_M
        }
        
    def get_drift(self) -> float:
        """
        Calcola il drift (spostamento relativo) dell'elemento.
        
        Returns:
            Drift normalizzato rispetto all'altezza
        """
        if self.type == "pier":
            # Drift laterale per maschi
            delta_v = self.displacements[4] - self.displacements[1]
            return abs(delta_v) / self.geometry.height
        else:
            # Drift verticale per fasce
            delta_u = self.displacements[3] - self.displacements[0]
            return abs(delta_u) / self.geometry.length
            
    def get_ductility_demand(self) -> float:
        """
        Calcola la domanda di duttilità.
        
        Returns:
            Rapporto spostamento/spostamento al limite elastico
        """
        # Spostamento al limite elastico stimato
        if self.type == "pier":
            # Rotazione al limite elastico
            theta_y = self.capacities['V_max'] * self.geometry.height / (3 * self.k_local_elastic[1, 1])
            theta_current = abs(self.displacements[2] + self.displacements[5]) / 2
            return theta_current / theta_y if theta_y > 0 else 0
        else:
            # Deformazione assiale per fasce
            eps_y = self.capacities['N_max'] / (self.material.E * 1000 * self.geometry.area)
            eps_current = abs(self.displacements[3] - self.displacements[0]) / self.geometry.length
            return eps_current / eps_y if eps_y > 0 else 0
            
    def __repr__(self) -> str:
        """Rappresentazione stringa dell'elemento"""
        return (f"FrameElement(id={self.id}, type={self.type}, "
                f"nodes=[{self.i_node}, {self.j_node}], "
                f"state={'Failed' if self.state.is_failed else 'Active'})")
                
    def to_dict(self) -> Dict:
        """
        Esporta elemento come dizionario per serializzazione.
        
        Returns:
            Dizionario con tutti i dati dell'elemento
        """
        return {
            'id': self.id,
            'type': self.type,
            'i_node': self.i_node,
            'j_node': self.j_node,
            'geometry': {
                'thickness': self.geometry.thickness,
                'width': getattr(self.geometry, 'width', None),
                'height': self.geometry.height,
                'length': getattr(self.geometry, 'length', None)
            },
            'material': {
                'E': self.material.E,
                'G': self.material.G,
                'f_m': self.material.fcm,
                'f_v0': self.material.tau0
            },
            'forces': self.forces,
            'capacities': self.capacities,
            'state': {
                'is_failed': self.state.is_failed,
                'failure_mode': self.state.failure_mode,
                'damage_level': self.state.damage_level
            },
            'drift': self.get_drift(),
            'ductility': self.get_ductility_demand()
        }