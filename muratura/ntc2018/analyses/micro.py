# analyses/micro.py
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import LineCollection

logger = logging.getLogger(__name__)

@dataclass
class MicroElement:
    """Elemento micro (blocco o malta)"""
    id: int
    type: str  # 'block' or 'mortar'
    nodes: List[int]
    coords: np.ndarray  # [4, 2] for Q4
    material: Dict
    thickness: float = 1.0
    
    def get_centroid(self) -> np.ndarray:
        """Calcola il centroide dell'elemento"""
        return np.mean(self.coords, axis=0)
    
    def get_area(self) -> float:
        """Calcola l'area dell'elemento (2D)"""
        # Shoelace formula per quadrilatero
        x = self.coords[:, 0]
        y = self.coords[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

@dataclass
class Interface:
    """Interfaccia tra elementi"""
    id: int
    nodes: List[int]  # Nodi dell'interfaccia
    elem1_id: int  # Elemento 1
    elem2_id: int  # Elemento 2
    normal: np.ndarray  # Versore normale
    tangent: np.ndarray  # Versore tangente
    length: float  # Lunghezza interfaccia
    props: Dict  # Proprietà interfaccia
    
    def get_midpoint(self, nodes_dict: Dict) -> np.ndarray:
        """Calcola punto medio dell'interfaccia"""
        coords = [nodes_dict[n] for n in self.nodes]
        return np.mean(coords, axis=0)

@dataclass
class MicroModel:
    """Micro-modello dettagliato della muratura"""
    block_props: Dict
    mortar_props: Dict
    interface_props: Dict
    elements: List[MicroElement] = field(default_factory=list)
    interfaces: List[Interface] = field(default_factory=list)
    nodes: Dict[int, np.ndarray] = field(default_factory=dict)
    dof_map: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    K_global: Optional[csr_matrix] = None
    
    def generate_micro_mesh(self, wall_data: Dict, block_size: Dict):
        """Genera mesh micro dettagliata con blocchi, malta e interfacce"""
        length = wall_data.get('length', 5.0)
        height = wall_data.get('height', 3.0)
        thickness = wall_data.get('thickness', 0.3)
        
        b_length = block_size.get('length', 0.25)
        b_height = block_size.get('height', 0.12)
        m_horiz = block_size.get('mortar_horizontal', 0.01)
        m_vert = block_size.get('mortar_vertical', 0.01)
        
        # Calcola numero di blocchi
        n_blocks_x = int(length / (b_length + m_vert))
        n_blocks_y = int(height / (b_height + m_horiz))
        
        # Genera griglia di nodi
        node_id = 0
        x_coords = []
        y_coords = []
        
        # Coordinate x (alternando blocchi e malta verticale)
        x = 0.0
        for j in range(n_blocks_x):
            x_coords.append(x)  # Inizio blocco
            x += b_length
            x_coords.append(x)  # Fine blocco/inizio malta
            if j < n_blocks_x - 1:
                x += m_vert
        
        # Coordinate y (alternando blocchi e malta orizzontale)
        y = 0.0
        for i in range(n_blocks_y):
            y_coords.append(y)  # Inizio blocco
            y += b_height
            y_coords.append(y)  # Fine blocco/inizio malta
            if i < n_blocks_y - 1:
                y += m_horiz
        
        # Crea nodi
        for y in y_coords:
            for x in x_coords:
                self.nodes[node_id] = np.array([x, y])
                node_id += 1
        
        nx = len(x_coords)
        elem_id = 0
        
        # Crea elementi blocco
        for i in range(0, len(y_coords)-1, 2):  # Solo righe dei blocchi
            for j in range(0, len(x_coords)-1, 2):  # Solo colonne dei blocchi
                n1 = i * nx + j
                n2 = n1 + 2  # Salta colonna malta
                n3 = (i + 2) * nx + j + 2  # Salta riga malta
                n4 = (i + 2) * nx + j
                
                if n3 < len(self.nodes):  # Verifica validità nodi
                    coords = np.array([self.nodes[n1], self.nodes[n2], 
                                     self.nodes[n3], self.nodes[n4]])
                    elem = MicroElement(elem_id, 'block', [n1, n2, n3, n4], 
                                      coords, self.block_props, thickness)
                    self.elements.append(elem)
                    elem_id += 1
        
        # Crea elementi malta orizzontale
        for i in range(1, len(y_coords)-1, 2):  # Righe malta
            for j in range(0, len(x_coords)-1, 2):  # Colonne blocchi
                n1 = i * nx + j
                n2 = n1 + 2
                n3 = (i + 1) * nx + j + 2
                n4 = (i + 1) * nx + j
                
                if n3 < len(self.nodes):
                    coords = np.array([self.nodes[n1], self.nodes[n2], 
                                     self.nodes[n3], self.nodes[n4]])
                    elem = MicroElement(elem_id, 'mortar', [n1, n2, n3, n4], 
                                      coords, self.mortar_props, thickness)
                    self.elements.append(elem)
                    elem_id += 1
        
        # Crea elementi malta verticale
        for i in range(0, len(y_coords)-1, 2):  # Righe blocchi
            for j in range(1, len(x_coords)-1, 2):  # Colonne malta
                n1 = i * nx + j
                n2 = n1 + 1
                n3 = (i + 2) * nx + j + 1
                n4 = (i + 2) * nx + j
                
                if n3 < len(self.nodes) and j+1 < nx:
                    coords = np.array([self.nodes[n1], self.nodes[n2], 
                                     self.nodes[n3], self.nodes[n4]])
                    elem = MicroElement(elem_id, 'mortar', [n1, n2, n3, n4], 
                                      coords, self.mortar_props, thickness)
                    self.elements.append(elem)
                    elem_id += 1
        
        # Genera interfacce tra elementi adiacenti
        self._generate_interfaces()
        
        logger.info(f"Mesh micro generata: {len(self.elements)} elementi, "
                   f"{len(self.interfaces)} interfacce, {len(self.nodes)} nodi")
    
    def _generate_interfaces(self):
        """Genera interfacce tra elementi adiacenti"""
        interface_id = 0
        
        # Per ogni elemento, trova elementi adiacenti
        for i, elem1 in enumerate(self.elements):
            for j, elem2 in enumerate(self.elements):
                if i >= j:  # Evita duplicati
                    continue
                
                # Trova nodi condivisi
                shared_nodes = list(set(elem1.nodes) & set(elem2.nodes))
                
                if len(shared_nodes) == 2:  # Interfaccia = 2 nodi condivisi
                    # Calcola normale e tangente
                    p1 = self.nodes[shared_nodes[0]]
                    p2 = self.nodes[shared_nodes[1]]
                    
                    # Vettore dell'interfaccia
                    v = p2 - p1
                    length = np.linalg.norm(v)
                    if length > 1e-6:
                        tangent = v / length
                        normal = np.array([-tangent[1], tangent[0]])
                        
                        # Orienta normale da elem1 a elem2
                        c1 = elem1.get_centroid()
                        c2 = elem2.get_centroid()
                        if np.dot(c2 - c1, normal) < 0:
                            normal = -normal
                        
                        interface = Interface(
                            interface_id, shared_nodes, elem1.id, elem2.id,
                            normal, tangent, length, self.interface_props
                        )
                        self.interfaces.append(interface)
                        interface_id += 1
    
    def _compute_element_stiffness(self, elem: MicroElement) -> np.ndarray:
        """Calcola matrice di rigidezza elemento (plane stress)"""
        E = elem.material.get('E', 3000)
        nu = elem.material.get('nu', 0.2)
        t = elem.thickness
        
        # Matrice costitutiva plane stress
        D = E / (1 - nu**2) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ])
        
        # Punti di Gauss per integrazione
        gauss_points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / np.sqrt(3)
        weights = [1, 1, 1, 1]
        
        Ke = np.zeros((8, 8))
        
        for gp, w in zip(gauss_points, weights):
            xi, eta = gp
            
            # Funzioni di forma Q4
            N = 0.25 * np.array([
                (1-xi)*(1-eta),
                (1+xi)*(1-eta),
                (1+xi)*(1+eta),
                (1-xi)*(1+eta)
            ])
            
            # Derivate funzioni di forma
            dN_dxi = 0.25 * np.array([
                [-(1-eta), (1-eta), (1+eta), -(1+eta)],
                [-(1-xi), -(1+xi), (1+xi), (1-xi)]
            ])
            
            # Jacobiano
            J = dN_dxi @ elem.coords
            det_J = np.linalg.det(J)
            
            if det_J > 1e-6:
                J_inv = np.linalg.inv(J)
                dN_dx = J_inv @ dN_dxi
                
                # Matrice B
                B = np.zeros((3, 8))
                for i in range(4):
                    B[0, 2*i] = dN_dx[0, i]
                    B[1, 2*i+1] = dN_dx[1, i]
                    B[2, 2*i] = dN_dx[1, i]
                    B[2, 2*i+1] = dN_dx[0, i]
                
                # Contributo al Ke
                Ke += B.T @ D @ B * det_J * t * w
        
        return Ke
    
    def _compute_interface_stiffness(self, interface: Interface) -> np.ndarray:
        """Calcola matrice di rigidezza interfaccia"""
        kn = interface.props.get('k_normal', 1e6)
        kt = interface.props.get('k_tangent', 1e5)
        L = interface.length
        
        # Matrice in coordinate locali
        K_local = L * np.array([
            [kn, 0, -kn, 0],
            [0, kt, 0, -kt],
            [-kn, 0, kn, 0],
            [0, -kt, 0, kt]
        ])
        
        # Matrice di rotazione
        c = interface.normal[0]
        s = interface.normal[1]
        T = np.array([
            [c, s, 0, 0],
            [-s, c, 0, 0],
            [0, 0, c, s],
            [0, 0, -s, c]
        ])
        
        # Trasforma in coordinate globali
        K_global = T.T @ K_local @ T
        
        return K_global
    
    def assemble_stiffness(self):
        """Assembla matrice di rigidezza globale"""
        n_nodes = len(self.nodes)
        n_dof = 2 * n_nodes
        
        # Mappa DOF
        self.dof_map = {nid: (2*nid, 2*nid+1) for nid in self.nodes.keys()}
        
        # Inizializza matrice sparsa
        self.K_global = lil_matrix((n_dof, n_dof))
        
        # Assembla elementi
        for elem in self.elements:
            Ke = self._compute_element_stiffness(elem)
            dofs = []
            for node in elem.nodes:
                dofs.extend(self.dof_map[node])
            
            for i in range(8):
                for j in range(8):
                    self.K_global[dofs[i], dofs[j]] += Ke[i, j]
        
        # Assembla interfacce
        for interface in self.interfaces:
            Ki = self._compute_interface_stiffness(interface)
            dofs = []
            for node in interface.nodes:
                dofs.extend(self.dof_map[node])
            
            for i in range(4):
                for j in range(4):
                    self.K_global[dofs[i], dofs[j]] += Ki[i, j]
        
        self.K_global = self.K_global.tocsr()
    
    def apply_boundary_conditions(self, boundary: Dict, F: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """Applica condizioni al contorno"""
        penalty = 1e15
        K_mod = self.K_global.copy().tolil()
        
        # Base fissa
        if boundary.get('bottom_fixed', True):
            tol = 1e-6
            bottom_nodes = [nid for nid, coord in self.nodes.items() 
                          if coord[1] < tol]
            
            for nid in bottom_nodes:
                dof_x, dof_y = self.dof_map[nid]
                K_mod[dof_x, dof_x] += penalty
                K_mod[dof_y, dof_y] += penalty
                F[dof_x] = 0
                F[dof_y] = 0
        
        # Spostamenti prescritti
        for bc in boundary.get('prescribed_displacements', []):
            node_id = bc['node']
            if 'ux' in bc:
                dof = self.dof_map[node_id][0]
                K_mod[dof, dof] += penalty
                F[dof] = bc['ux'] * penalty
            if 'uy' in bc:
                dof = self.dof_map[node_id][1]
                K_mod[dof, dof] += penalty
                F[dof] = bc['uy'] * penalty
        
        return K_mod.tocsr(), F
    
    def analyze_micro(self, loads: Dict, boundary: Dict) -> Dict:
        """Esegue analisi micro-strutturale"""
        logger.info("Inizio analisi micro-strutturale")
        
        # Assembla sistema
        self.assemble_stiffness()
        n_dof = self.K_global.shape[0]
        F = np.zeros(n_dof)
        
        # Applica carichi
        self._apply_loads(loads, F)
        
        # Applica condizioni al contorno
        K_mod, F_mod = self.apply_boundary_conditions(boundary, F)
        
        # Risolvi sistema
        logger.info("Risoluzione sistema lineare...")
        u = spsolve(K_mod, F_mod)
        
        # Calcola spostamenti e deformazioni
        displacements = self._extract_displacements(u)
        strains = self._compute_strains(u)
        stresses = self._compute_stresses(strains)
        
        # Analisi del danno
        damage = self._compute_damage_pattern(stresses)
        cracks = self._identify_cracks(damage, stresses)
        
        # Calcola reazioni vincolari
        reactions = self._compute_reactions(u, F)
        
        results = {
            'displacements': displacements,
            'strains': strains,
            'stresses': stresses,
            'damage': damage,
            'crack_pattern': cracks,
            'reactions': reactions,
            'max_displacement': np.max(np.abs(u)),
            'n_damaged_elements': len(damage['crushing']) + len(damage['cracking']),
            'n_failed_interfaces': len(damage['sliding'])
        }
        
        logger.info(f"Analisi completata: max spostamento = {results['max_displacement']:.3e} m")
        
        return results
    
    def _apply_loads(self, loads: Dict, F: np.ndarray):
        """Applica i carichi al vettore delle forze"""
        # Carico distribuito orizzontale in sommità
        if 'horizontal_top' in loads:
            H = loads['horizontal_top']
            y_max = max(coord[1] for coord in self.nodes.values())
            top_nodes = [nid for nid, coord in self.nodes.items() 
                        if abs(coord[1] - y_max) < 1e-6]
            
            if top_nodes:
                H_node = H / len(top_nodes)
                for nid in top_nodes:
                    F[self.dof_map[nid][0]] += H_node
        
        # Carico verticale distribuito
        if 'vertical_top' in loads:
            V = loads['vertical_top']
            y_max = max(coord[1] for coord in self.nodes.values())
            top_nodes = [nid for nid, coord in self.nodes.items() 
                        if abs(coord[1] - y_max) < 1e-6]
            
            if top_nodes:
                V_node = V / len(top_nodes)
                for nid in top_nodes:
                    F[self.dof_map[nid][1]] -= V_node  # Negativo = verso il basso
        
        # Peso proprio
        if loads.get('self_weight', True):
            for elem in self.elements:
                gamma = elem.material.get('weight', 20.0)  # kN/m³
                area = elem.get_area()
                weight = gamma * area * elem.thickness
                
                # Distribuisci su 4 nodi
                for node in elem.nodes:
                    F[self.dof_map[node][1]] -= weight / 4
        
        # Carichi nodali puntuali
        for load in loads.get('point_loads', []):
            node_id = load['node']
            if 'Fx' in load:
                F[self.dof_map[node_id][0]] += load['Fx']
            if 'Fy' in load:
                F[self.dof_map[node_id][1]] += load['Fy']
    
    def _extract_displacements(self, u: np.ndarray) -> Dict:
        """Estrae spostamenti nodali"""
        displacements = {
            'nodes': {},
            'max_ux': 0.0,
            'max_uy': 0.0,
            'max_total': 0.0
        }
        
        for nid in self.nodes:
            dof_x, dof_y = self.dof_map[nid]
            ux = u[dof_x]
            uy = u[dof_y]
            utot = np.sqrt(ux**2 + uy**2)
            
            displacements['nodes'][nid] = {
                'ux': ux,
                'uy': uy,
                'utotal': utot
            }
            
            displacements['max_ux'] = max(displacements['max_ux'], abs(ux))
            displacements['max_uy'] = max(displacements['max_uy'], abs(uy))
            displacements['max_total'] = max(displacements['max_total'], utot)
        
        return displacements
    
    def _compute_strains(self, u: np.ndarray) -> Dict:
        """Calcola deformazioni negli elementi"""
        strains = {'elements': {}}
        
        for elem in self.elements:
            # Estrai spostamenti elemento
            u_elem = []
            for node in elem.nodes:
                dof_x, dof_y = self.dof_map[node]
                u_elem.extend([u[dof_x], u[dof_y]])
            u_elem = np.array(u_elem)
            
            # Calcola deformazioni al centroide
            xi = eta = 0.0  # Centroide
            
            # Derivate funzioni di forma
            dN_dxi = 0.25 * np.array([
                [-(1-eta), (1-eta), (1+eta), -(1+eta)],
                [-(1-xi), -(1+xi), (1+xi), (1-xi)]
            ])
            
            J = dN_dxi @ elem.coords
            J_inv = np.linalg.inv(J)
            dN_dx = J_inv @ dN_dxi
            
            # Matrice B
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2*i] = dN_dx[0, i]
                B[1, 2*i+1] = dN_dx[1, i]
                B[2, 2*i] = dN_dx[1, i]
                B[2, 2*i+1] = dN_dx[0, i]
            
            # Deformazioni
            epsilon = B @ u_elem
            
            strains['elements'][elem.id] = {
                'exx': epsilon[0],
                'eyy': epsilon[1],
                'exy': epsilon[2],
                'principal': self._compute_principal_strains(epsilon)
            }
        
        return strains
    
    def _compute_principal_strains(self, epsilon: np.ndarray) -> Dict:
        """Calcola deformazioni principali"""
        exx, eyy, exy = epsilon
        
        # Deformazioni principali
        e_avg = (exx + eyy) / 2
        e_diff = (exx - eyy) / 2
        e_max = e_avg + np.sqrt(e_diff**2 + exy**2)
        e_min = e_avg - np.sqrt(e_diff**2 + exy**2)
        
        # Direzione principale
        if abs(exy) > 1e-10:
            theta = 0.5 * np.arctan2(2*exy, exx - eyy)
        else:
            theta = 0.0
        
        return {
            'e1': e_max,
            'e2': e_min,
            'theta': np.degrees(theta)
        }
    
    def _compute_stresses(self, strains: Dict) -> Dict:
        """Calcola tensioni negli elementi e interfacce"""
        stresses = {'elements': {}, 'interfaces': {}}
        
        # Tensioni negli elementi
        for elem_id, strain in strains['elements'].items():
            elem = self.elements[elem_id]
            E = elem.material.get('E', 3000)
            nu = elem.material.get('nu', 0.2)
            
            # Matrice costitutiva
            D = E / (1 - nu**2) * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1-nu)/2]
            ])
            
            epsilon = np.array([strain['exx'], strain['eyy'], strain['exy']])
            sigma = D @ epsilon
            
            stresses['elements'][elem_id] = {
                'sxx': sigma[0],
                'syy': sigma[1],
                'sxy': sigma[2],
                'principal': self._compute_principal_stresses(sigma),
                'von_mises': self._compute_von_mises(sigma)
            }
        
        # Tensioni nelle interfacce
        for interface in self.interfaces:
            # Calcola spostamenti relativi
            u_rel = self._compute_interface_displacement(interface, strains)
            
            # Tensioni interfaccia
            kn = interface.props.get('k_normal', 1e6)
            kt = interface.props.get('k_tangent', 1e5)
            
            sigma_n = kn * u_rel['normal']
            tau = kt * u_rel['tangential']
            
            stresses['interfaces'][interface.id] = {
                'sigma_n': sigma_n,
                'tau': tau,
                'u_normal': u_rel['normal'],
                'u_tangential': u_rel['tangential']
            }
        
        return stresses
    
    def _compute_principal_stresses(self, sigma: np.ndarray) -> Dict:
        """Calcola tensioni principali"""
        sxx, syy, sxy = sigma
        
        s_avg = (sxx + syy) / 2
        s_diff = (sxx - syy) / 2
        s1 = s_avg + np.sqrt(s_diff**2 + sxy**2)
        s2 = s_avg - np.sqrt(s_diff**2 + sxy**2)
        
        if abs(sxy) > 1e-10:
            theta = 0.5 * np.arctan2(2*sxy, sxx - syy)
        else:
            theta = 0.0
        
        return {
            'sigma1': s1,
            'sigma2': s2,
            'theta': np.degrees(theta)
        }
    
    def _compute_von_mises(self, sigma: np.ndarray) -> float:
        """Calcola tensione di von Mises"""
        sxx, syy, sxy = sigma
        return np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)
    
    def _compute_interface_displacement(self, interface: Interface, strains: Dict) -> Dict:
        """Calcola spostamenti relativi interfaccia"""
        # Placeholder - implementazione semplificata
        return {
            'normal': 0.0,
            'tangential': 0.0
        }
    
    def _compute_damage_pattern(self, stresses: Dict) -> Dict:
        """Identifica pattern di danno"""
        damage = {
            'crushing': [],
            'cracking': [],
            'sliding': [],
            'mortar_failure': []
        }
        
        # Verifica elementi
        for elem_id, stress in stresses['elements'].items():
            elem = self.elements[elem_id]
            
            # Compressione
            fc = elem.material.get('fc', 5.0)
            if stress['principal']['sigma2'] < -fc:
                damage['crushing'].append({
                    'element': elem_id,
                    'stress_ratio': abs(stress['principal']['sigma2']) / fc,
                    'location': elem.get_centroid().tolist()
                })
            
            # Trazione
            ft = elem.material.get('ft', 0.1)
            if stress['principal']['sigma1'] > ft:
                damage['cracking'].append({
                    'element': elem_id,
                    'stress_ratio': stress['principal']['sigma1'] / ft,
                    'angle': stress['principal']['theta'],
                    'location': elem.get_centroid().tolist()
                })
            
            # Failure malta
            if elem.type == 'mortar' and stress['von_mises'] > elem.material.get('fc', 2.0):
                damage['mortar_failure'].append({
                    'element': elem_id,
                    'stress': stress['von_mises']
                })
        
        # Verifica interfacce
        for intf_id, stress in stresses['interfaces'].items():
            interface = next(i for i in self.interfaces if i.id == intf_id)
            
            # Criterio di Mohr-Coulomb
            cohesion = interface.props.get('cohesion', 0.1)
            friction = interface.props.get('friction', 0.6)
            
            tau_max = cohesion + friction * max(0, -stress['sigma_n'])
            
            if abs(stress['tau']) > tau_max:
                damage['sliding'].append({
                    'interface': intf_id,
                    'tau': stress['tau'],
                    'tau_max': tau_max,
                    'location': interface.get_midpoint(self.nodes).tolist()
                })
        
        return damage
    
    def _identify_cracks(self, damage: Dict, stresses: Dict) -> List[Dict]:
        """Identifica e classifica fessure"""
        cracks = []
        
        # Fessure da trazione
        for crack_info in damage['cracking']:
            elem = self.elements[crack_info['element']]
            crack = {
                'type': 'tensile',
                'element': crack_info['element'],
                'location': crack_info['location'],
                'orientation': crack_info['angle'],
                'severity': 'high' if crack_info['stress_ratio'] > 2.0 else 'medium',
                'width_estimate': self._estimate_crack_width(crack_info['stress_ratio']),
                'propagation_risk': crack_info['stress_ratio'] > 1.5
            }
            cracks.append(crack)
        
        # Fessure da scorrimento
        for slide_info in damage['sliding']:
            interface = next(i for i in self.interfaces if i.id == slide_info['interface'])
            crack = {
                'type': 'sliding',
                'interface': slide_info['interface'],
                'location': slide_info['location'],
                'orientation': np.degrees(np.arctan2(interface.tangent[1], interface.tangent[0])),
                'severity': 'high' if abs(slide_info['tau']) > 2 * slide_info['tau_max'] else 'medium',
                'displacement': abs(slide_info['tau']) / interface.props.get('k_tangent', 1e5)
            }
            cracks.append(crack)
        
        # Fessure da schiacciamento
        for crush_info in damage['crushing']:
            elem = self.elements[crush_info['element']]
            crack = {
                'type': 'crushing',
                'element': crush_info['element'],
                'location': crush_info['location'],
                'severity': 'critical' if crush_info['stress_ratio'] > 1.5 else 'high',
                'extent': 'localized' if crush_info['stress_ratio'] < 1.2 else 'extensive'
            }
            cracks.append(crack)
        
        # Analisi pattern globale
        if len(cracks) > 3:
            pattern = self._analyze_crack_pattern(cracks)
            for crack in cracks:
                crack['pattern'] = pattern
        
        return cracks
    
    def _estimate_crack_width(self, stress_ratio: float) -> float:
        """Stima larghezza fessura in mm"""
        if stress_ratio < 1.0:
            return 0.0
        elif stress_ratio < 1.5:
            return 0.1 * (stress_ratio - 1.0)
        elif stress_ratio < 2.0:
            return 0.05 + 0.3 * (stress_ratio - 1.5)
        else:
            return 0.2 + 0.5 * (stress_ratio - 2.0)
    
    def _analyze_crack_pattern(self, cracks: List[Dict]) -> str:
        """Analizza pattern globale delle fessure"""
        # Conta tipi di fessure
        n_tensile = sum(1 for c in cracks if c['type'] == 'tensile')
        n_sliding = sum(1 for c in cracks if c['type'] == 'sliding')
        n_crushing = sum(1 for c in cracks if c['type'] == 'crushing')
        
        # Analizza distribuzione spaziale
        locations = [c['location'] for c in cracks]
        y_coords = [loc[1] for loc in locations]
        y_mean = np.mean(y_coords)
        y_std = np.std(y_coords)
        
        # Determina pattern
        if n_crushing > n_tensile + n_sliding:
            return 'compression_failure'
        elif n_sliding > n_tensile:
            return 'shear_failure'
        elif n_tensile > 0 and y_std < 0.1:
            return 'horizontal_bending'
        elif n_tensile > 0:
            return 'diagonal_cracking'
        else:
            return 'mixed_mode'
    
    def _compute_reactions(self, u: np.ndarray, F_applied: np.ndarray) -> Dict:
        """Calcola reazioni vincolari"""
        F_total = self.K_global @ u
        R = F_total - F_applied
        
        reactions = {
            'nodes': {},
            'total_Rx': 0.0,
            'total_Ry': 0.0,
            'total_moment': 0.0
        }
        
        # Estrai reazioni per nodi vincolati
        tol = 1e-6
        bottom_nodes = [nid for nid, coord in self.nodes.items() if coord[1] < tol]
        
        for nid in bottom_nodes:
            dof_x, dof_y = self.dof_map[nid]
            Rx = R[dof_x]
            Ry = R[dof_y]
            
            if abs(Rx) > 1e-3 or abs(Ry) > 1e-3:
                reactions['nodes'][nid] = {
                    'Rx': Rx,
                    'Ry': Ry,
                    'R_total': np.sqrt(Rx**2 + Ry**2)
                }
                
                reactions['total_Rx'] += Rx
                reactions['total_Ry'] += Ry
                
                # Momento rispetto all'origine
                x = self.nodes[nid][0]
                reactions['total_moment'] += x * Ry
        
        return reactions
    
    def homogenization(self) -> Dict:
        """Omogeneizzazione per ottenere proprietà equivalenti macro"""
        logger.info("Esecuzione omogeneizzazione micro->macro")
        
        # Volume RVE (Representative Volume Element)
        x_coords = [n[0] for n in self.nodes.values()]
        y_coords = [n[1] for n in self.nodes.values()]
        L_rve = max(x_coords) - min(x_coords)
        H_rve = max(y_coords) - min(y_coords)
        V_rve = L_rve * H_rve * self.elements[0].thickness
        
        # Volumi componenti
        V_block = sum(elem.get_area() * elem.thickness 
                     for elem in self.elements if elem.type == 'block')
        V_mortar = sum(elem.get_area() * elem.thickness 
                      for elem in self.elements if elem.type == 'mortar')
        
        # Frazioni volumetriche
        f_block = V_block / V_rve
        f_mortar = V_mortar / V_rve
        
        # Test di carico per omogeneizzazione
        test_cases = {
            'uniaxial_x': {'exx': 1e-3, 'eyy': 0, 'exy': 0},
            'uniaxial_y': {'exx': 0, 'eyy': 1e-3, 'exy': 0},
            'shear': {'exx': 0, 'eyy': 0, 'exy': 1e-3}
        }
        
        C_hom = np.zeros((3, 3))  # Matrice costitutiva omogeneizzata
        
        for i, (case, strain) in enumerate(test_cases.items()):
            # Applica deformazione media al RVE
            stress_avg = self._apply_homogenization_bc(strain)
            
            # Estrai colonna della matrice C
            C_hom[:, i] = stress_avg
        
        # Calcola proprietà ingegneristiche
        E_x = C_hom[0, 0]
        E_y = C_hom[1, 1]
        G_xy = C_hom[2, 2]
        nu_xy = -C_hom[0, 1] / C_hom[1, 1]
        nu_yx = -C_hom[1, 0] / C_hom[0, 0]
        
        # Resistenze omogeneizzate (approccio semplificato)
        fc_block = self.block_props.get('fc', 5.0)
        fc_mortar = self.mortar_props.get('fc', 2.0)
        fc_hom = f_block * fc_block + f_mortar * fc_mortar
        
        ft_block = self.block_props.get('ft', 0.1)
        ft_mortar = self.mortar_props.get('ft', 0.05)
        ft_hom = min(ft_block, ft_mortar)  # Governato dal componente più debole
        
        # Resistenza a taglio (Mohr-Coulomb medio)
        cohesion = self.interface_props.get('cohesion', 0.1)
        friction = self.interface_props.get('friction', 0.6)
        tau0_hom = cohesion
        
        results = {
            'elastic_moduli': {
                'E_x': E_x,
                'E_y': E_y,
                'G_xy': G_xy,
                'nu_xy': nu_xy,
                'nu_yx': nu_yx
            },
            'strengths': {
                'fc': fc_hom,
                'ft': ft_hom,
                'tau0': tau0_hom,
                'mu': friction
            },
            'volume_fractions': {
                'blocks': f_block,
                'mortar': f_mortar
            },
            'anisotropy': {
                'E_ratio': E_x / E_y,
                'strength_ratio': fc_hom / ft_hom
            }
        }
        
        return results
    
    def _apply_homogenization_bc(self, strain_avg: Dict) -> np.ndarray:
        """Applica BC per omogeneizzazione e calcola stress medio"""
        # Implementazione semplificata
        # In pratica servirebbe risolvere problema BC periodiche
        exx = strain_avg['exx']
        eyy = strain_avg['eyy']
        exy = strain_avg['exy']
        
        # Stima stress medio con rule of mixtures
        E_block = self.block_props.get('E', 3000)
        E_mortar = self.mortar_props.get('E', 1000)
        nu_block = self.block_props.get('nu', 0.2)
        nu_mortar = self.mortar_props.get('nu', 0.2)
        
        # Calcola frazioni
        V_total = sum(elem.get_area() * elem.thickness for elem in self.elements)
        f_block = sum(elem.get_area() * elem.thickness 
                     for elem in self.elements if elem.type == 'block') / V_total
        f_mortar = 1 - f_block
        
        # Matrici costitutive
        D_block = E_block / (1 - nu_block**2) * np.array([
            [1, nu_block, 0],
            [nu_block, 1, 0],
            [0, 0, (1-nu_block)/2]
        ])
        
        D_mortar = E_mortar / (1 - nu_mortar**2) * np.array([
            [1, nu_mortar, 0],
            [nu_mortar, 1, 0],
            [0, 0, (1-nu_mortar)/2]
        ])
        
        # Media pesata
        D_avg = f_block * D_block + f_mortar * D_mortar
        
        # Stress medio
        epsilon = np.array([exx, eyy, exy])
        sigma_avg = D_avg @ epsilon
        
        return sigma_avg
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Visualizza risultati analisi micro"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Mesh deformata
        ax = axes[0, 0]
        scale = 100  # Fattore di scala deformazioni
        
        for elem in self.elements:
            # Coordinate originali
            x_orig = [self.nodes[n][0] for n in elem.nodes]
            y_orig = [self.nodes[n][1] for n in elem.nodes]
            x_orig.append(x_orig[0])
            y_orig.append(y_orig[0])
            
            # Coordinate deformate
            x_def = []
            y_def = []
            for n in elem.nodes:
                disp = results['displacements']['nodes'][n]
                x_def.append(self.nodes[n][0] + scale * disp['ux'])
                y_def.append(self.nodes[n][1] + scale * disp['uy'])
            x_def.append(x_def[0])
            y_def.append(y_def[0])
            
            # Plot
            ax.plot(x_orig, y_orig, 'b-', alpha=0.3, linewidth=0.5)
            ax.plot(x_def, y_def, 'r-', linewidth=1)
            
            # Colora elementi per tipo
            if elem.type == 'block':
                poly = Polygon(list(zip(x_def[:-1], y_def[:-1])), 
                             facecolor='lightcoral', alpha=0.5)
            else:
                poly = Polygon(list(zip(x_def[:-1], y_def[:-1])), 
                             facecolor='lightgray', alpha=0.5)
            ax.add_patch(poly)
        
        ax.set_aspect('equal')
        ax.set_title(f'Configurazione Deformata (scala {scale}x)')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        
        # 2. Mappa delle tensioni principali
        ax = axes[0, 1]
        
        for elem_id, stress in results['stresses']['elements'].items():
            elem = self.elements[elem_id]
            centroid = elem.get_centroid()
            s1 = stress['principal']['sigma1']
            s2 = stress['principal']['sigma2']
            theta = np.radians(stress['principal']['theta'])
            
            # Direzioni principali
            v1 = np.array([np.cos(theta), np.sin(theta)])
            v2 = np.array([-np.sin(theta), np.cos(theta)])
            
            # Scala vettori
            scale_s = 0.01
            
            # Plot tensioni principali
            if s1 > 0:  # Trazione
                ax.arrow(centroid[0], centroid[1], 
                        scale_s * s1 * v1[0], scale_s * s1 * v1[1],
                        head_width=0.02, head_length=0.01, fc='red', ec='red')
            if s2 < 0:  # Compressione
                ax.arrow(centroid[0], centroid[1], 
                        -scale_s * abs(s2) * v2[0], -scale_s * abs(s2) * v2[1],
                        head_width=0.02, head_length=0.01, fc='blue', ec='blue')
        
        # Contorno elementi
        for elem in self.elements:
            x = [self.nodes[n][0] for n in elem.nodes]
            y = [self.nodes[n][1] for n in elem.nodes]
            x.append(x[0])
            y.append(y[0])
            ax.plot(x, y, 'k-', linewidth=0.5)
        
        ax.set_aspect('equal')
        ax.set_title('Tensioni Principali')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.legend(['Trazione', 'Compressione'])
        
        # 3. Pattern di danno
        ax = axes[1, 0]
        
        # Plot elementi
        for elem in self.elements:
            x = [self.nodes[n][0] for n in elem.nodes]
            y = [self.nodes[n][1] for n in elem.nodes]
            
            # Colora per danno
            elem_damaged = False
            color = 'lightgreen'
            
            for crack in results['damage']['cracking']:
                if crack['element'] == elem.id:
                    color = 'red'
                    elem_damaged = True
                    break
                    
            for crush in results['damage']['crushing']:
                if crush['element'] == elem.id:
                    color = 'darkred'
                    elem_damaged = True
                    break
            
            if elem.type == 'mortar' and not elem_damaged:
                color = 'lightgray'
            
            poly = Polygon(list(zip(x, y)), facecolor=color, 
                         edgecolor='black', linewidth=0.5)
            ax.add_patch(poly)
        
        # Plot interfacce danneggiate
        for slide in results['damage']['sliding']:
            intf_id = slide['interface']
            interface = next(i for i in self.interfaces if i.id == intf_id)
            
            p1 = self.nodes[interface.nodes[0]]
            p2 = self.nodes[interface.nodes[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'orange', linewidth=3)
        
        ax.set_aspect('equal')
        ax.set_title('Pattern di Danno')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        
        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', label='Integro'),
            Patch(facecolor='red', label='Fessurato'),
            Patch(facecolor='darkred', label='Schiacciato'),
            Patch(facecolor='orange', label='Scorrimento')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 4. Evoluzione fessure
        ax = axes[1, 1]
        
        # Mappa fessure
        for crack in results['crack_pattern']:
            if crack['type'] == 'tensile':
                location = crack['location']
                angle = np.radians(crack['orientation'])
                length = 0.05 * crack['width_estimate']
                
                dx = length * np.cos(angle + np.pi/2)
                dy = length * np.sin(angle + np.pi/2)
                
                ax.plot([location[0] - dx, location[0] + dx],
                       [location[1] - dy, location[1] + dy],
                       'r-', linewidth=2 * crack['width_estimate'])
            
            elif crack['type'] == 'sliding':
                location = crack['location']
                ax.plot(location[0], location[1], 'o', 
                       color='orange', markersize=8)
        
        # Contorno
        for elem in self.elements:
            x = [self.nodes[n][0] for n in elem.nodes]
            y = [self.nodes[n][1] for n in elem.nodes]
            x.append(x[0])
            y.append(y[0])
            ax.plot(x, y, 'k-', linewidth=0.5, alpha=0.3)
        
        ax.set_aspect('equal')
        ax.set_title('Pattern Fessurativo')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Risultati salvati in {save_path}")
        
        plt.show()


def analyze_micro(wall_data: Dict, material: Dict, loads: Dict, 
                 options: Dict = None) -> Dict:
    """Funzione principale per analisi micro-strutturale"""
    if options is None:
        options = {}
    
    logger.info("=" * 50)
    logger.info("ANALISI MICRO-STRUTTURALE MURATURA")
    logger.info("=" * 50)
    
    # Proprietà di default
    block_props = options.get('block_properties', {
        'E': material.get('E', 3000) * 1.5,  # MPa
        'nu': 0.2,
        'fc': material.get('fcm', 5.0) * 2.0,  # MPa
        'ft': material.get('ftm', 0.1) * 1.5,  # MPa
        'weight': 20.0  # kN/m³
    })
    
    mortar_props = options.get('mortar_properties', {
        'E': material.get('E', 3000) * 0.3,  # MPa
        'nu': 0.2,
        'fc': material.get('fcm', 5.0) * 0.3,  # MPa
        'ft': material.get('ftm', 0.1) * 0.5,  # MPa
        'weight': 18.0  # kN/m³
    })
    
    interface_props = options.get('interface_properties', {
        'k_normal': 1e6,  # MPa/m
        'k_tangent': 1e5,  # MPa/m
        'cohesion': material.get('tau0', 0.1),  # MPa
        'friction': material.get('mu', 0.6),
        'tensile_strength': material.get('ftm', 0.1) * 0.1,  # MPa
    })
    
    # Dimensioni blocco
    block_size = options.get('block_size', {
        'length': 0.25,  # m
        'height': 0.12,  # m
        'mortar_horizontal': 0.01,  # m
        'mortar_vertical': 0.01  # m
    })
    
    # Crea modello
    model = MicroModel(block_props, mortar_props, interface_props)
    
    # Genera mesh
    model.generate_micro_mesh(wall_data, block_size)
    
    # Condizioni al contorno
    boundary = options.get('boundary_conditions', {
        'bottom_fixed': True,
        'prescribed_displacements': []
    })
    
    # Tipo di analisi
    analysis_type = options.get('analysis_type', 'static')
    
    results = {
        'method': 'MICRO_MODEL',
        'n_elements': len(model.elements),
        'n_blocks': len([e for e in model.elements if e.type == 'block']),
        'n_mortar': len([e for e in model.elements if e.type == 'mortar']),
        'n_interfaces': len(model.interfaces),
        'n_nodes': len(model.nodes),
        'analysis_type': analysis_type
    }
    
    if analysis_type == 'static':
        # Analisi statica non lineare
        micro_results = model.analyze_micro(loads, boundary)
        results.update(micro_results)
        
        # Sommario risultati
        results['summary'] = {
            'max_displacement': micro_results['max_displacement'],
            'n_cracks': len(micro_results['crack_pattern']),
            'damage_level': 'severe' if micro_results['n_damaged_elements'] > 10 else 'moderate',
            'failure_mode': micro_results['crack_pattern'][0]['pattern'] if micro_results['crack_pattern'] else 'none'
        }
        
    elif analysis_type == 'homogenization':
        # Omogeneizzazione
        hom_results = model.homogenization()
        results['homogenized'] = hom_results
        
        # Confronto con proprietà macro
        results['comparison'] = {
            'E_ratio': hom_results['elastic_moduli']['E_x'] / material.get('E', 3000),
            'fc_ratio': hom_results['strengths']['fc'] / material.get('fcm', 5.0),
            'anisotropy': hom_results['anisotropy']
        }
    
    # Visualizzazione
    if options.get('plot_results', True) and analysis_type == 'static':
        save_path = options.get('save_path', 'micro_analysis_results.png')
        model.plot_results(results, save_path)
    
    logger.info("Analisi micro completata con successo")
    
    return results