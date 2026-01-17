# analyses/fem.py
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from ..materials import MaterialProperties
from ..constitutive import ConstitutiveModel
from ..enums import ConstitutiveLaw
from ..geometry import GeometryPier  # Usato per esempio, ma adattabile
from ..utils import logger  # Assumi logger da utils

logger = logging.getLogger(__name__)

@dataclass
class FEMElement:
    """Elemento FEM quadrilatero Q4 con supporto non lineare"""
    nodes: List[int]  # ID nodi (4 per Q4)
    material: MaterialProperties
    constitutive_law: ConstitutiveLaw = ConstitutiveLaw.LINEAR
    thickness: float = 1.0  # Spessore per piano stress/strain
    _model: Optional[ConstitutiveModel] = None
    
    def __post_init__(self):
        self._model = self.material.get_constitutive_law(self.constitutive_law)
    
    def shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """Funzioni di forma per Q4"""
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta)
        ])
        return N
    
    def dN_dxi(self, xi: float, eta: float) -> np.ndarray:
        """Derivate parziali rispetto a xi, eta"""
        dN_dxi = 0.25 * np.array([
            -(1 - eta), (1 - eta), (1 + eta), -(1 + eta)
        ])
        dN_deta = 0.25 * np.array([
            -(1 - xi), -(1 + xi), (1 + xi), (1 - xi)
        ])
        return dN_dxi, dN_deta
    
    def jacobian(self, coords: np.ndarray, xi: float, eta: float) -> Tuple[np.ndarray, float]:
        """Matrice Jacobiana e determinante"""
        dN_dxi, dN_deta = self.dN_dxi(xi, eta)
        J = np.zeros((2, 2))
        J[0, 0] = np.dot(dN_dxi, coords[:, 0])  # dx/dxi
        J[0, 1] = np.dot(dN_dxi, coords[:, 1])  # dy/dxi
        J[1, 0] = np.dot(dN_deta, coords[:, 0])  # dx/deta
        J[1, 1] = np.dot(dN_deta, coords[:, 1])  # dy/deta
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError(f"Jacobian determinant non-positive: {detJ}")
        return J, detJ
    
    def B_matrix(self, coords: np.ndarray, xi: float, eta: float) -> np.ndarray:
        """Matrice B per deformazioni"""
        J, detJ = self.jacobian(coords, xi, eta)
        invJ = np.linalg.inv(J)
        dN_dxi, dN_deta = self.dN_dxi(xi, eta)
        
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        
        B = np.zeros((3, 8))  # 3 strain components, 8 DOFs (4 nodes * 2)
        for i in range(4):
            B[0, 2*i] = dN_dx[i]    # epsilon_x = du/dx
            B[1, 2*i + 1] = dN_dy[i]  # epsilon_y = dv/dy
            B[2, 2*i] = dN_dy[i]      # gamma_xy = du/dy + dv/dx
            B[2, 2*i + 1] = dN_dx[i]
        
        return B
    
    def D_matrix(self, strain: Optional[np.ndarray] = None) -> np.ndarray:
        """Matrice costitutiva D, tangente per non lineare"""
        if self.constitutive_law == ConstitutiveLaw.LINEAR:
            E = self.material.E
            nu = self.material.nu
            # Piano stress
            D = (E / (1 - nu**2)) * np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu)/2]
            ])
            return D
        else:
            # Per non lineare, usa modulo tangente
            if strain is None:
                strain = np.zeros(3)
            
            # Calcola strain equivalente per modulo tangente
            strain_eq = np.sqrt(strain[0]**2 + strain[1]**2 + 2 * strain[2]**2)
            Et = self._model.tangent_modulus(strain_eq)
            
            # Assicura che Et sia positivo per stabilità
            Et = max(Et, 0.01 * self.material.E)
            
            nu_eff = min(self.material.nu, 0.45)  # Limita per stabilità
            
            D = (Et / (1 - nu_eff**2)) * np.array([
                [1, nu_eff, 0],
                [nu_eff, 1, 0],
                [0, 0, (1 - nu_eff)/2]
            ])
            return D
    
    def element_stiffness(self, coords: np.ndarray, u_elem: Optional[np.ndarray] = None) -> np.ndarray:
        """Matrice di rigidezza elemento K_e, tangente per non lineare"""
        K_e = np.zeros((8, 8))
        # Punti di Gauss 2x2
        gauss_points = np.array([
            [-1/np.sqrt(3), -1/np.sqrt(3)],
            [1/np.sqrt(3), -1/np.sqrt(3)],
            [1/np.sqrt(3), 1/np.sqrt(3)],
            [-1/np.sqrt(3), 1/np.sqrt(3)]
        ])
        weights = np.ones(4)
        
        for i in range(4):
            xi, eta = gauss_points[i]
            w = weights[i]
            
            try:
                B = self.B_matrix(coords, xi, eta)
                _, detJ = self.jacobian(coords, xi, eta)
                
                # Per non lineare, calcola strain al punto di Gauss
                strain = np.zeros(3)
                if u_elem is not None:
                    strain = B @ u_elem
                
                D = self.D_matrix(strain)
                
                K_e += B.T @ D @ B * detJ * w * self.thickness
                
            except ValueError as e:
                logger.error(f"Errore nel calcolo della rigidezza elemento: {e}")
                raise
        
        return K_e
    
    def internal_forces(self, coords: np.ndarray, u_elem: np.ndarray) -> np.ndarray:
        """Forze interne per non lineare"""
        f_int = np.zeros(8)
        gauss_points = np.array([
            [-1/np.sqrt(3), -1/np.sqrt(3)],
            [1/np.sqrt(3), -1/np.sqrt(3)],
            [1/np.sqrt(3), 1/np.sqrt(3)],
            [-1/np.sqrt(3), 1/np.sqrt(3)]
        ])
        weights = np.ones(4)
        
        for i in range(4):
            xi, eta = gauss_points[i]
            w = weights[i]
            
            try:
                B = self.B_matrix(coords, xi, eta)
                _, detJ = self.jacobian(coords, xi, eta)
                strain = B @ u_elem
                
                # Calcola stress con modello costitutivo
                if self.constitutive_law == ConstitutiveLaw.LINEAR:
                    stress = self.D_matrix() @ strain
                else:
                    # Per non lineare, usa stress dal modello costitutivo
                    strain_eq = np.sqrt(strain[0]**2 + strain[1]**2 + 2 * strain[2]**2)
                    sigma_eq = self._model.stress(strain_eq)
                    # Distribuzione proporzionale delle tensioni
                    if strain_eq > 1e-12:
                        stress = sigma_eq / strain_eq * strain
                    else:
                        stress = np.zeros(3)
                
                f_int += B.T @ stress * detJ * w * self.thickness
                
            except ValueError as e:
                logger.error(f"Errore nel calcolo forze interne: {e}")
                raise
        
        return f_int

class FEMModel:
    """Modello FEM 2D completo con supporto non lineare"""
    def __init__(self):
        self.nodes: Dict[int, np.ndarray] = {}  # id -> [x, y]
        self.elements: List[FEMElement] = []
        self.constraints: List[Dict] = []  # {'node': id, 'dofs': [0,1] for ux,uy}
        self.loads: Dict[int, np.ndarray] = {}  # node_id -> [Fx, Fy]
        self.K_global: csr_matrix = None
        self.dof_map: Dict[int, Tuple[int, int]] = {}  # node_id -> (dof_x, dof_y)
        self._setup_dof_mapping()
    
    def _setup_dof_mapping(self):
        """Inizializza la mappatura DOF - ricostruisce sempre per sincronizzazione"""
        if self.nodes:
            # Ricostruisci sempre per garantire consistenza con i nodi attuali
            self.dof_map = {}
            for i, node_id in enumerate(sorted(self.nodes.keys())):
                self.dof_map[int(node_id)] = (2*i, 2*i + 1)
    
    def add_node(self, node_id: int, x: float, y: float):
        self.nodes[int(node_id)] = np.array([x, y])
        # dof_map viene ricostruito in assemble_global_stiffness
    
    def add_element(self, elem: FEMElement):
        # Verifica che tutti i nodi esistano
        for node_id in elem.nodes:
            if int(node_id) not in self.nodes:
                raise ValueError(f"Nodo {node_id} non definito")
        self.elements.append(elem)
    
    def add_constraint(self, node_id: int, dofs: List[int]):
        if int(node_id) not in self.nodes:
            raise ValueError(f"Nodo {node_id} non definito per vincolo")
        self.constraints.append({'node': int(node_id), 'dofs': dofs})
    
    def add_load(self, node_id: int, Fx: float = 0.0, Fy: float = 0.0):
        if int(node_id) not in self.nodes:
            raise ValueError(f"Nodo {node_id} non definito per carico")
        self.loads[int(node_id)] = np.array([Fx, Fy])
    
    def generate_mesh(self, wall_data: Dict, material: MaterialProperties, 
                     n_x: int = 10, n_y: int = 5, law: ConstitutiveLaw = ConstitutiveLaw.LINEAR):
        """Genera mesh rettangolare semplice per parete, con handling aperture"""
        length = wall_data.get('length', 5.0)
        height = wall_data.get('height', 3.0)
        thickness = wall_data.get('thickness', 0.3)
        openings = wall_data.get('openings', [])  # Lista di {'x_start':, 'x_end':, 'y_start':, 'y_end':}
        
        dx = length / n_x
        dy = height / n_y
        
        # Crea griglia di nodi
        node_id = 0
        node_grid = np.full((n_y + 1, n_x + 1), -1, dtype=int)
        
        for i in range(n_y + 1):
            for j in range(n_x + 1):
                x = j * dx
                y = i * dy
                self.add_node(node_id, x, y)
                node_grid[i, j] = node_id
                node_id += 1
        
        # Crea elementi evitando le aperture
        for i in range(n_y):
            for j in range(n_x):
                # Verifica sovrapposizione con aperture
                x_min = j * dx
                x_max = (j + 1) * dx
                y_min = i * dy
                y_max = (i + 1) * dy
                
                overlap = False
                for op in openings:
                    if (x_min < op['x_end'] and x_max > op['x_start'] and
                        y_min < op['y_end'] and y_max > op['y_start']):
                        overlap = True
                        break
                
                if not overlap:
                    n1 = node_grid[i, j]
                    n2 = node_grid[i, j + 1]
                    n3 = node_grid[i + 1, j + 1]
                    n4 = node_grid[i + 1, j]
                    elem = FEMElement(nodes=[n1, n2, n3, n4], material=material, 
                                    thickness=thickness, constitutive_law=law)
                    self.add_element(elem)
        
        # Vincoli alla base (fixed bottom)
        for j in range(n_x + 1):
            bottom_node = node_grid[0, j]
            self.add_constraint(bottom_node, [0, 1])  # ux=0, uy=0
    
    def assemble_global_stiffness(self, u: Optional[np.ndarray] = None):
        """Assembla matrice di rigidezza globale"""
        n_nodes = len(self.nodes)
        n_dof = 2 * n_nodes
        
        self._setup_dof_mapping()
        self.K_global = lil_matrix((n_dof, n_dof))
        
        for elem in self.elements:
            coords = np.array([self.nodes[int(n)] for n in elem.nodes])
            elem_dofs = []
            for n in elem.nodes:
                elem_dofs.extend(self.dof_map[int(n)])
            
            u_elem = None if u is None else u[elem_dofs]
            
            try:
                K_e = elem.element_stiffness(coords, u_elem)
                
                for i in range(8):
                    for j in range(8):
                        self.K_global[elem_dofs[i], elem_dofs[j]] += K_e[i, j]
                        
            except Exception as e:
                logger.error(f"Errore nell'assemblaggio elemento: {e}")
                raise
        
        self.K_global = self.K_global.tocsr()
    
    def apply_boundary_conditions(self, K: csr_matrix, F: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """Applica vincoli con metodo della penalità"""
        penalty = 1e12
        K_mod = K.copy().tolil()
        F_mod = F.copy()
        
        for const in self.constraints:
            node_dofs = self.dof_map[int(const['node'])]
            for local_dof in const['dofs']:
                global_dof = node_dofs[local_dof]
                K_mod[global_dof, global_dof] += penalty
                F_mod[global_dof] = 0
        
        return K_mod.tocsr(), F_mod
    
    def assemble_load_vector(self) -> np.ndarray:
        """Assembla vettore dei carichi"""
        n_dof = 2 * len(self.nodes)
        F = np.zeros(n_dof)
        
        for node_id, load in self.loads.items():
            dofs = self.dof_map[int(node_id)]
            F[dofs[0]] = load[0]
            F[dofs[1]] = load[1]
        
        return F
    
    def compute_internal_forces(self, u: np.ndarray) -> np.ndarray:
        """Calcola forze interne per analisi non lineare"""
        n_dof = len(u)
        f_int = np.zeros(n_dof)
        
        for elem in self.elements:
            coords = np.array([self.nodes[int(n)] for n in elem.nodes])
            elem_dofs = []
            for n in elem.nodes:
                elem_dofs.extend(self.dof_map[int(n)])
            u_elem = u[elem_dofs]
            
            try:
                f_e = elem.internal_forces(coords, u_elem)
                f_int[elem_dofs] += f_e
            except Exception as e:
                logger.error(f"Errore nel calcolo forze interne elemento: {e}")
                raise
        
        return f_int
    
    def solve_linear(self) -> np.ndarray:
        """Risolve sistema lineare"""
        logger.info("Inizio risoluzione lineare FEM")
        
        self.assemble_global_stiffness()
        F = self.assemble_load_vector()
        K_mod, F_mod = self.apply_boundary_conditions(self.K_global, F)
        
        # Verifica condizionamento
        if K_mod.nnz == 0:
            raise ValueError("Matrice di rigidezza vuota")
        
        u = spsolve(K_mod, F_mod)
        logger.info("Risoluzione lineare completata")
        return u
    
    def solve_nonlinear(self, tol: float = 1e-6, max_iter: int = 50) -> np.ndarray:
        """Newton-Raphson per analisi non lineare"""
        logger.info("Inizio risoluzione non lineare con Newton-Raphson")
        
        F_ext = self.assemble_load_vector()
        u = np.zeros(len(F_ext))
        iter_count = 0
        
        # Applica vincoli al vettore dei carichi esterni
        _, F_ext = self.apply_boundary_conditions(csr_matrix((len(F_ext), len(F_ext))), F_ext)
        
        while iter_count < max_iter:
            # Calcola forze interne e residuo
            f_int = self.compute_internal_forces(u)
            R = F_ext - f_int
            
            # Applica vincoli al residuo
            for const in self.constraints:
                node_dofs = self.dof_map[int(const['node'])]
                for local_dof in const['dofs']:
                    global_dof = node_dofs[local_dof]
                    R[global_dof] = 0
                    u[global_dof] = 0  # Mantieni spostamento nullo
            
            norm_R = np.linalg.norm(R)
            logger.debug(f"Iterazione {iter_count}: ||R|| = {norm_R}")
            
            if norm_R < tol:
                logger.info(f"Convergenza raggiunta in {iter_count} iterazioni")
                break
            
            # Assembla matrice tangente
            self.assemble_global_stiffness(u)
            K_mod, R_mod = self.apply_boundary_conditions(self.K_global, R)
            
            try:
                du = spsolve(K_mod, R_mod)
                u += du
            except Exception as e:
                logger.error(f"Errore nella risoluzione sistema lineare: {e}")
                break
            
            iter_count += 1
        
        if iter_count >= max_iter:
            logger.warning(f"Solver non lineare non convergente dopo {max_iter} iterazioni")
        
        return u
    
    def compute_stresses(self, u: np.ndarray) -> List[Dict]:
        """Calcola tensioni medie per elemento"""
        stresses = []
        
        for elem_idx, elem in enumerate(self.elements):
            coords = np.array([self.nodes[int(n)] for n in elem.nodes])
            elem_dofs = []
            for n in elem.nodes:
                elem_dofs.extend(self.dof_map[int(n)])
            u_elem = u[elem_dofs]
            
            try:
                # Punto centrale per media
                xi, eta = 0.0, 0.0
                B = elem.B_matrix(coords, xi, eta)
                strain = B @ u_elem
                
                if elem.constitutive_law == ConstitutiveLaw.LINEAR:
                    stress = elem.D_matrix() @ strain
                    sigma_x, sigma_y, tau_xy = stress[0], stress[1], stress[2]
                else:
                    # Stress dal modello costitutivo non lineare
                    strain_eq = np.sqrt(strain[0]**2 + strain[1]**2 + 2 * strain[2]**2)
                    sigma_eq = elem._model.stress(strain_eq)
                    if strain_eq > 1e-12:
                        stress = sigma_eq / strain_eq * strain
                    else:
                        stress = np.zeros(3)
                    sigma_x, sigma_y, tau_xy = stress[0], stress[1], stress[2]
                
                # Von Mises
                von_mises = np.sqrt(sigma_x**2 - sigma_x*sigma_y + sigma_y**2 + 3*tau_xy**2)
                
                stresses.append({
                    'element_id': elem_idx,
                    'sigma_x': float(sigma_x),
                    'sigma_y': float(sigma_y),
                    'tau_xy': float(tau_xy),
                    'von_mises': float(von_mises),
                    'strain_x': float(strain[0]),
                    'strain_y': float(strain[1]),
                    'gamma_xy': float(strain[2])
                })
                
            except Exception as e:
                logger.error(f"Errore nel calcolo tensioni elemento {elem_idx}: {e}")
                stresses.append({
                    'element_id': elem_idx,
                    'sigma_x': 0.0,
                    'sigma_y': 0.0,
                    'tau_xy': 0.0,
                    'von_mises': 0.0,
                    'strain_x': 0.0,
                    'strain_y': 0.0,
                    'gamma_xy': 0.0
                })
        
        return stresses

def _analyze_fem(wall_data: Dict, material: MaterialProperties,
                 loads: Dict, options: Dict) -> Dict:
    """
    Analisi FEM 2D completa con elementi Q4, supporto non lineare.
    Genera mesh con handling aperture, risolve lineare/non-lineare e calcola tensioni.
    """
    logger.info("Avvio analisi FEM 2D con elementi Q4")
    
    try:
        # Parametri di mesh e analisi
        n_x = options.get('n_x', 10)
        n_y = options.get('n_y', 5)
        law = options.get('constitutive_law', ConstitutiveLaw.LINEAR)
        nonlinear = options.get('nonlinear', law != ConstitutiveLaw.LINEAR)
        
        # Crea modello e genera mesh
        model = FEMModel()
        model.generate_mesh(wall_data, material, n_x, n_y, law)
        
        logger.info(f"Mesh generata: {len(model.nodes)} nodi, {len(model.elements)} elementi")
        
        # Applica carichi sui nodi superiori
        length = wall_data.get('length', 5.0)
        height = wall_data.get('height', 3.0)
        n_nodes_x = n_x + 1
        
        # Identifica nodi superiori correttamente
        top_node_ids = []
        for node_id, coords in model.nodes.items():
            if abs(coords[1] - height) < 1e-6:  # Nodi alla quota massima
                top_node_ids.append(node_id)
        
        if top_node_ids:
            Fx_per_node = loads.get('horizontal', 0.0) / len(top_node_ids)
            Fy_per_node = loads.get('vertical', 0.0) / len(top_node_ids)
            
            for node_id in top_node_ids:
                model.add_load(node_id, Fx=Fx_per_node, Fy=Fy_per_node)
        
        logger.info(f"Applicati carichi su {len(top_node_ids)} nodi superiori")
        
        # Risoluzione
        if nonlinear:
            logger.info("Risoluzione non lineare")
            u = model.solve_nonlinear()
        else:
            logger.info("Risoluzione lineare")
            u = model.solve_linear()
        
        # Calcola tensioni
        stresses = model.compute_stresses(u)
        
        # Verifica convergenza per non lineare
        converged = True
        if nonlinear:
            f_int = model.compute_internal_forces(u)
            f_ext = model.assemble_load_vector()
            residual_norm = np.linalg.norm(f_ext - f_int)
            converged = residual_norm < 1e-4
            logger.info(f"Residuo finale: {residual_norm}, Convergenza: {converged}")
        
        # Statistiche finali
        max_displacement = np.max(np.abs(u))
        max_von_mises = max([s['von_mises'] for s in stresses]) if stresses else 0.0
        
        results = {
            'method': 'FEM',
            'analysis_type': 'nonlinear' if nonlinear else 'linear',
            'displacements': u.tolist(),
            'max_displacement': float(max_displacement),
            'stresses': stresses,
            'max_von_mises': float(max_von_mises),
            'n_nodes': len(model.nodes),
            'n_elements': len(model.elements),
            'n_constraints': len(model.constraints),
            'n_loads': len(model.loads),
            'nonlinear_converged': converged,
            'mesh_info': {
                'n_x': n_x,
                'n_y': n_y,
                'constitutive_law': law.value if hasattr(law, 'value') else str(law)
            }
        }
        
        logger.info("Analisi FEM completata con successo")
        return results
        
    except Exception as e:
        logger.error(f"Errore nell'analisi FEM: {e}")
        return {
            'method': 'FEM',
            'error': str(e),
            'displacements': [],
            'stresses': [],
            'n_nodes': 0,
            'n_elements': 0,
            'nonlinear_converged': False
        }