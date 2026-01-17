# analyses/fiber.py
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from scipy.optimize import fsolve
from scipy.linalg import solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from ..materials import MaterialProperties
from ..constitutive import ConstitutiveModel
from ..enums import ConstitutiveLaw
from ..geometry import GeometryPier, GeometrySpandrel
from ..utils import calculate_damage_indices, extract_hysteretic_params, calculate_section_ductility, compare_constitutive_laws, distribute_vertical_loads

logger = logging.getLogger(__name__)

@dataclass
class Fiber:
    """Singola fibra in sezione"""
    y: float  # Posizione verticale dalla neutra [m]
    area: float  # Area fibra [m²]
    model: ConstitutiveModel  # Modello costitutivo
    strain: float = 0.0  # Deformazione corrente
    stress: float = 0.0  # Tensione corrente
    
    def update_state(self, strain: float) -> float:
        """Aggiorna stato fibra e ritorna tensione"""
        self.strain = strain
        self.stress = self.model.stress(strain)
        return self.stress
    
    def get_stress(self, strain: float) -> float:
        """Calcola tensione senza aggiornare stato"""
        return self.model.stress(strain)
    
    def tangent_modulus(self, strain: float = None) -> float:
        """Modulo tangente (usa strain corrente se non specificato)"""
        if strain is None:
            strain = self.strain
        return self.model.tangent_modulus(strain)

@dataclass
class FiberSection:
    """Sezione a fibre"""
    fibers: List[Fiber] = field(default_factory=list)
    width: float = 1.0  # Larghezza sezione [m]
    height: float = 0.0  # Altezza totale [m]
    centroid: float = 0.0  # Posizione centroide [m]
    
    # Stato corrente sezione
    epsilon0: float = 0.0  # Deformazione assiale
    curvature: float = 0.0  # Curvatura
    
    def add_fibers(self, material: MaterialProperties, law_type: ConstitutiveLaw,
                   n_fibers: int = 20, height: float = 0.3, width: float = 1.0):
        """Aggiunge fibre uniformi"""
        self.height = height
        self.width = width
        dy = height / n_fibers
        area = width * dy
        
        model = material.get_constitutive_law(law_type)
        
        for i in range(n_fibers):
            y = -height/2 + (i + 0.5) * dy  # Dal basso all'alto
            self.fibers.append(Fiber(y=y, area=area, model=model))
    
    def add_reinforcement(self, positions: List[float], areas: List[float], 
                         steel_model: ConstitutiveModel):
        """Aggiunge barre di armatura"""
        for y, area in zip(positions, areas):
            self.fibers.append(Fiber(y=y, area=area, model=steel_model))
    
    def axial_moment(self, curvature: float, epsilon0: float, 
                    update: bool = False) -> Tuple[float, float]:
        """Calcola N e M per data curvatura e epsilon0"""
        N = 0.0
        M = 0.0
        
        for fiber in self.fibers:
            strain = epsilon0 + curvature * fiber.y
            if update:
                stress = fiber.update_state(strain)
            else:
                stress = fiber.get_stress(strain)
            N += stress * fiber.area
            M += stress * fiber.area * fiber.y
        
        if update:
            self.epsilon0 = epsilon0
            self.curvature = curvature
            
        return N, M
    
    def solve_equilibrium(self, N_target: float, M_target: float,
                         tol: float = 1e-6, max_iter: int = 50) -> Tuple[float, float]:
        """Risolve per epsilon0 e curvatura dato N e M target"""
        def residual(x):
            eps0, kappa = x
            N, M = self.axial_moment(kappa, eps0)
            return [N - N_target, M - M_target]
        
        # Stima iniziale
        x0 = [self.epsilon0, self.curvature]
        
        try:
            sol, info, ier, msg = fsolve(residual, x0, full_output=True, xtol=tol)
            if ier == 1:
                eps0, kappa = sol
                self.axial_moment(kappa, eps0, update=True)
                return eps0, kappa
            else:
                logger.warning(f"Convergenza non raggiunta: {msg}")
                return self.epsilon0, self.curvature
        except Exception as e:
            logger.error(f"Errore in solve_equilibrium: {e}")
            return self.epsilon0, self.curvature
    
    def get_moment_curvature(self, N: float = 0.0, max_curvature: float = 0.1,
                             n_points: int = 50) -> Dict:
        """Genera diagramma M-chi per dato N"""
        curvatures = []
        moments = []
        epsilon0s = []
        
        kappa = np.linspace(0, max_curvature, n_points)
        
        for k in kappa:
            # Risolvi per epsilon0 tale che N = N_target
            def residual(eps0):
                N_calc, _ = self.axial_moment(k, eps0[0])
                return N_calc - N
            
            # Stima iniziale migliorata
            if len(epsilon0s) > 0:
                eps0_guess = epsilon0s[-1]
            else:
                eps0_guess = -N / (self.width * self.height * self.fibers[0].model.E)
            
            try:
                eps0 = fsolve(residual, eps0_guess, xtol=1e-8)[0]
                _, M = self.axial_moment(k, eps0)
                
                curvatures.append(k)
                moments.append(M)
                epsilon0s.append(eps0)
            except:
                logger.warning(f"Convergenza fallita per curvatura {k}")
                if len(moments) > 0:
                    curvatures.append(k)
                    moments.append(moments[-1])  # Usa ultimo valore
                    epsilon0s.append(epsilon0s[-1])
        
        return {
            'curvature': np.array(curvatures),
            'moment': np.array(moments),
            'epsilon0': np.array(epsilon0s)
        }
    
    def tangent_stiffness(self) -> Tuple[float, float, float]:
        """Calcola EA, EI, S per matrice sezionale tangente (stato corrente)"""
        EA = 0.0
        S = 0.0
        EI = 0.0
        
        for fiber in self.fibers:
            Et = fiber.tangent_modulus()
            EA += Et * fiber.area
            S += Et * fiber.area * fiber.y
            EI += Et * fiber.area * fiber.y**2
        
        return EA, S, EI
    
    def get_section_matrix(self) -> np.ndarray:
        """Matrice costitutiva sezionale 2x2 [N,M] = D * [eps0, kappa]"""
        EA, S, EI = self.tangent_stiffness()
        return np.array([[EA, S], [S, EI]])

@dataclass
class IntegrationPoint:
    """Punto di integrazione Gauss"""
    xi: float  # Coordinata locale [-1, 1]
    weight: float  # Peso integrazione
    section: FiberSection  # Sezione associata
    
class FiberElement:
    """Elemento beam-column con sezioni a fibre"""
    def __init__(self, section_template: FiberSection, length: float, 
                 n_ip: int = 3, id: str = None):
        self.length = length
        self.n_ip = n_ip
        self.id = id or f"elem_{id(self)}"
        
        # Crea punti integrazione con copie della sezione
        self.integration_points = []
        xi_gauss, w_gauss = self._gauss_points(n_ip)
        
        for xi, w in zip(xi_gauss, w_gauss):
            section_copy = FiberSection(
                fibers=[Fiber(f.y, f.area, f.model) for f in section_template.fibers],
                width=section_template.width,
                height=section_template.height
            )
            self.integration_points.append(IntegrationPoint(xi, w, section_copy))
        
        # Stato elemento
        self.u_local = np.zeros(6)  # [u1, v1, θ1, u2, v2, θ2]
        
    def _gauss_points(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Punti e pesi di Gauss"""
        if n == 1:
            return np.array([0.0]), np.array([2.0])
        elif n == 2:
            xi = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
            w = np.array([1.0, 1.0])
        elif n == 3:
            xi = np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
            w = np.array([5/9, 8/9, 5/9])
        elif n == 4:
            xi1 = np.sqrt(3/7 - 2/7*np.sqrt(6/5))
            xi2 = np.sqrt(3/7 + 2/7*np.sqrt(6/5))
            xi = np.array([-xi2, -xi1, xi1, xi2])
            w1 = (18 + np.sqrt(30))/36
            w2 = (18 - np.sqrt(30))/36
            w = np.array([w2, w1, w1, w2])
        else:
            raise ValueError(f"Gauss points non implementati per n={n}")
        return xi, w
    
    def shape_functions(self, xi: float) -> Tuple[np.ndarray, np.ndarray]:
        """Funzioni di forma e derivate per elemento beam"""
        L = self.length
        
        # Funzioni di forma Hermitiane
        N1 = 1 - xi
        N2 = xi
        N3 = (1 - xi)**2 * (1 + 2*xi)
        N4 = (1 - xi)**2 * xi * L
        N5 = xi**2 * (3 - 2*xi)
        N6 = xi**2 * (xi - 1) * L
        
        # Matrice B (strain-displacement)
        B_axial = np.array([-1/L, 0, 0, 1/L, 0, 0])
        B_flexure = np.array([0, -6/L**2 + 12*xi/L**2, -4/L + 6*xi/L,
                             0, 6/L**2 - 12*xi/L**2, -2/L + 6*xi/L])
        
        return B_axial, B_flexure
    
    def element_strains(self, u_elem: np.ndarray, xi: float) -> Tuple[float, float]:
        """Calcola deformazioni in punto di integrazione"""
        B_axial, B_flexure = self.shape_functions(xi)
        epsilon0 = B_axial @ u_elem
        curvature = B_flexure @ u_elem
        return epsilon0, curvature
    
    def stiffness_matrix(self) -> np.ndarray:
        """Matrice rigidezza elemento tangente"""
        K = np.zeros((6, 6))
        L = self.length
        
        for ip in self.integration_points:
            # Coordinate globali
            x = L/2 * (1 + ip.xi)
            
            # Matrici B
            B_axial, B_flexure = self.shape_functions(ip.xi)
            B = np.vstack([B_axial, B_flexure])
            
            # Matrice costitutiva sezionale
            D = ip.section.get_section_matrix()
            
            # Contributo punto integrazione
            K += B.T @ D @ B * ip.weight * L/2
        
        return K
    
    def internal_forces(self, u_elem: np.ndarray) -> np.ndarray:
        """Forze interne elemento"""
        F = np.zeros(6)
        L = self.length
        
        for ip in self.integration_points:
            # Deformazioni
            epsilon0, curvature = self.element_strains(u_elem, ip.xi)
            
            # Aggiorna stato sezione e ottieni N, M
            N, M = ip.section.axial_moment(curvature, epsilon0, update=True)
            
            # Matrici B
            B_axial, B_flexure = self.shape_functions(ip.xi)
            
            # Contributo forze
            F += B_axial * N * ip.weight * L/2
            F += B_flexure * M * ip.weight * L/2
        
        self.u_local = u_elem
        return F
    
    def update_state(self, u_elem: np.ndarray):
        """Aggiorna stato elemento e sezioni"""
        for ip in self.integration_points:
            epsilon0, curvature = self.element_strains(u_elem, ip.xi)
            ip.section.axial_moment(curvature, epsilon0, update=True)
        self.u_local = u_elem

class FiberModel:
    """Modello globale a fibre per struttura"""
    def __init__(self, material: MaterialProperties, law_type: ConstitutiveLaw = ConstitutiveLaw.MANDER):
        self.material = material
        self.law_type = law_type
        self.elements: List[FiberElement] = []
        self.nodes: Dict[int, np.ndarray] = {}
        self.constraints: Dict[int, List[int]] = {}  # DOF vincolati
        self.dof_map: Dict[Tuple[int, int], int] = {}  # (node, local_dof) -> global_dof
        self.n_dof = 0
        
    def add_node(self, node_id: int, x: float, y: float) -> int:
        """Aggiunge nodo"""
        self.nodes[node_id] = np.array([x, y])
        return node_id
    
    def add_element(self, elem_id: str, node1: int, node2: int, 
                   geometry: Union[GeometryPier, GeometrySpandrel],
                   n_fibers: int = 20, n_ip: int = 3):
        """Aggiunge elemento"""
        # Crea sezione template
        section = FiberSection()
        if isinstance(geometry, GeometryPier):
            width = geometry.thickness
            height = geometry.thickness  # Assumo sezione quadrata
        else:
            width = geometry.length
            height = geometry.thickness
            
        section.add_fibers(self.material, self.law_type, n_fibers, height, width)
        
        # Calcola lunghezza elemento
        p1 = self.nodes[node1]
        p2 = self.nodes[node2]
        length = np.linalg.norm(p2 - p1)
        
        # Crea elemento
        elem = FiberElement(section, length, n_ip, elem_id)
        elem.node1 = node1
        elem.node2 = node2
        self.elements.append(elem)
        
    def add_constraint(self, node_id: int, dofs: List[int]):
        """Vincola gradi di libertà (0=u, 1=v, 2=θ)"""
        self.constraints[node_id] = dofs
        
    def _build_dof_map(self):
        """Costruisce mappatura DOF locali -> globali"""
        self.dof_map.clear()
        dof_counter = 0
        
        for node_id in sorted(self.nodes.keys()):
            for local_dof in range(3):  # u, v, θ
                if node_id not in self.constraints or local_dof not in self.constraints[node_id]:
                    self.dof_map[(node_id, local_dof)] = dof_counter
                    dof_counter += 1
        
        self.n_dof = dof_counter
        
    def _element_dof(self, elem: FiberElement) -> List[int]:
        """Ottiene DOF globali per elemento"""
        dofs = []
        for node in [elem.node1, elem.node2]:
            for local_dof in range(3):
                if (node, local_dof) in self.dof_map:
                    dofs.append(self.dof_map[(node, local_dof)])
                else:
                    dofs.append(-1)  # DOF vincolato
        return dofs
    
    def _rotation_matrix(self, elem: FiberElement) -> np.ndarray:
        """Matrice rotazione da locale a globale"""
        p1 = self.nodes[elem.node1]
        p2 = self.nodes[elem.node2]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        L = np.sqrt(dx**2 + dy**2)
        c = dx/L
        s = dy/L
        
        T = np.zeros((6, 6))
        T[0:2, 0:2] = T[3:4, 3:4] = [[c, s], [-s, c]]
        T[2, 2] = T[5, 5] = 1.0
        return T
    
    def assemble_stiffness(self) -> csr_matrix:
        """Assembla matrice rigidezza globale"""
        self._build_dof_map()
        K = np.zeros((self.n_dof, self.n_dof))
        
        for elem in self.elements:
            # Matrice locale
            K_local = elem.stiffness_matrix()
            
            # Trasforma in globale
            T = self._rotation_matrix(elem)
            K_global = T.T @ K_local @ T
            
            # Assembla
            dofs = self._element_dof(elem)
            for i, dof_i in enumerate(dofs):
                if dof_i >= 0:
                    for j, dof_j in enumerate(dofs):
                        if dof_j >= 0:
                            K[dof_i, dof_j] += K_global[i, j]
        
        return csr_matrix(K)
    
    def assemble_forces(self) -> np.ndarray:
        """Assembla vettore forze interne globali"""
        F = np.zeros(self.n_dof)
        
        for elem in self.elements:
            # Forze locali
            u_local = self._extract_element_disp(elem)
            F_local = elem.internal_forces(u_local)
            
            # Trasforma in globale
            T = self._rotation_matrix(elem)
            F_global = T.T @ F_local
            
            # Assembla
            dofs = self._element_dof(elem)
            for i, dof_i in enumerate(dofs):
                if dof_i >= 0:
                    F[dof_i] += F_global[i]
        
        return F
    
    def _extract_element_disp(self, elem: FiberElement) -> np.ndarray:
        """Estrae spostamenti elemento da soluzione globale"""
        u_elem = np.zeros(6)
        if hasattr(self, 'u_global'):
            dofs = self._element_dof(elem)
            for i, dof in enumerate(dofs):
                if dof >= 0:
                    u_elem[i] = self.u_global[dof]
        return u_elem
    
    def apply_loads(self, loads: Dict) -> np.ndarray:
        """Applica carichi esterni e ritorna vettore forze"""
        F_ext = np.zeros(self.n_dof)
        
        for node_id, load in loads.items():
            if node_id in self.nodes:
                for direction, value in load.items():
                    if direction == 'Fx':
                        local_dof = 0
                    elif direction == 'Fy':
                        local_dof = 1
                    elif direction == 'M':
                        local_dof = 2
                    else:
                        continue
                        
                    if (node_id, local_dof) in self.dof_map:
                        global_dof = self.dof_map[(node_id, local_dof)]
                        F_ext[global_dof] += value
        
        return F_ext
    
    def solve_step(self, F_ext: np.ndarray, tol: float = 1e-6, 
                   max_iter: int = 20) -> bool:
        """Risolve passo con Newton-Raphson"""
        if not hasattr(self, 'u_global'):
            self.u_global = np.zeros(self.n_dof)
            
        for iter in range(max_iter):
            # Forze interne
            F_int = self.assemble_forces()
            
            # Residuo
            R = F_ext - F_int
            norm_R = np.linalg.norm(R)
            
            if norm_R < tol:
                logger.info(f"Convergenza raggiunta in {iter+1} iterazioni")
                return True
            
            # Matrice tangente
            K = self.assemble_stiffness()
            
            # Risolvi incremento
            du = spsolve(K, R)
            
            # Aggiorna spostamenti
            self.u_global += du
            
            # Aggiorna stato elementi
            for elem in self.elements:
                u_elem = self._extract_element_disp(elem)
                elem.update_state(u_elem)
        
        logger.warning(f"Convergenza non raggiunta dopo {max_iter} iterazioni")
        return False
    
    def pushover_analysis(self, vertical_loads: Dict, lateral_pattern: str = 'triangular',
                         max_drift: float = 0.05, n_steps: int = 100) -> Dict:
        """Analisi pushover completa"""
        logger.info("Inizio analisi pushover con modello a fibre")
        
        # Reset stato
        self.u_global = np.zeros(self.n_dof)
        for elem in self.elements:
            elem.update_state(np.zeros(6))
        
        # Applica carichi verticali
        F_vert = self.apply_loads(vertical_loads)
        if not self.solve_step(F_vert):
            logger.error("Fallita applicazione carichi verticali")
            return {}
        
        # Pattern laterale
        F_lat_ref = self._lateral_pattern(lateral_pattern)
        
        # Trova DOF controllo (top displacement)
        top_node = max(self.nodes.keys(), key=lambda n: self.nodes[n][1])
        control_dof = self.dof_map.get((top_node, 0), 0)
        
        # Storia risposta
        results = {
            'curve': [],
            'convergence': []
        }
        
        # Incrementa carico laterale
        target_disp = max_drift * max(self.nodes[n][1] for n in self.nodes)
        disp_inc = target_disp / n_steps
        
        for step in range(n_steps):
            # Controllo in spostamento
            target = (step + 1) * disp_inc
            
            # Stima lambda
            if step == 0:
                lambda_inc = 0.1
            else:
                lambda_inc *= target / self.u_global[control_dof]
            
            converged = False
            for _ in range(10):  # Iterazioni per trovare lambda
                F_total = F_vert + lambda_inc * F_lat_ref
                
                # Salva stato
                u_prev = self.u_global.copy()
                elem_states = [(e.u_local.copy(), 
                              [(ip.section.epsilon0, ip.section.curvature) 
                               for ip in e.integration_points]) 
                              for e in self.elements]
                
                # Prova passo
                if self.solve_step(F_total):
                    if abs(self.u_global[control_dof] - target) < 1e-4:
                        converged = True
                        break
                    else:
                        # Aggiusta lambda
                        lambda_inc *= target / self.u_global[control_dof]
                else:
                    # Riduci incremento
                    lambda_inc *= 0.5
                    
                # Ripristina stato
                self.u_global = u_prev
                for elem, (u_elem, ip_states) in zip(self.elements, elem_states):
                    elem.u_local = u_elem
                    for ip, (eps0, kappa) in zip(elem.integration_points, ip_states):
                        ip.section.epsilon0 = eps0
                        ip.section.curvature = kappa
            
            if converged:
                # Salva risultati
                base_shear = sum(F_lat_ref[self.dof_map.get((n, 0), -1)] * lambda_inc 
                               for n in self.nodes if (n, 0) in self.dof_map)
                
                results['curve'].append({
                    'top_drift': self.u_global[control_dof] / max(self.nodes[n][1] for n in self.nodes),
                    'top_displacement': self.u_global[control_dof],
                    'base_shear': base_shear,
                    'lambda': lambda_inc
                })
                
                logger.info(f"Step {step+1}: drift={results['curve'][-1]['top_drift']:.4f}, "
                           f"V={base_shear:.1f}")
            else:
                logger.warning(f"Step {step+1} non converge, termino analisi")
                break
        
        # Post-processing
        if results['curve']:
            results['performance_levels'] = self._extract_performance_levels(results['curve'])
            results['damage_indices'] = calculate_damage_indices(results)
            
        return results
    
    def _lateral_pattern(self, pattern: str) -> np.ndarray:
        """Genera pattern di carico laterale"""
        F = np.zeros(self.n_dof)
        
        if pattern == 'uniform':
            # Forza uniforme per piano
            for node_id in self.nodes:
                if (node_id, 0) in self.dof_map:
                    F[self.dof_map[(node_id, 0)]] = 1.0
                    
        elif pattern == 'triangular':
            # Forza proporzionale all'altezza
            h_max = max(self.nodes[n][1] for n in self.nodes)
            for node_id, pos in self.nodes.items():
                if (node_id, 0) in self.dof_map:
                    F[self.dof_map[(node_id, 0)]] = pos[1] / h_max
                    
        elif pattern == 'modal':
            # Approssimazione primo modo
            h_max = max(self.nodes[n][1] for n in self.nodes)
            for node_id, pos in self.nodes.items():
                if (node_id, 0) in self.dof_map:
                    F[self.dof_map[(node_id, 0)]] = (pos[1] / h_max) ** 1.5
        
        # Normalizza
        if np.linalg.norm(F) > 0:
            F = F / np.linalg.norm(F)
            
        return F
    
    def _extract_performance_levels(self, curve: List[Dict]) -> Dict:
        """Estrae livelli prestazionali da curva pushover"""
        levels = {}
        
        # Trova snervamento con criterio bilineare equivalente
        if len(curve) > 3:
            drifts = [p['top_drift'] for p in curve]
            shears = [p['base_shear'] for p in curve]
            
            # Stima yield
            K_elastic = shears[2] / drifts[2] if drifts[2] > 0 else 0
            for i, (d, v) in enumerate(zip(drifts, shears)):
                if i > 0 and v > 0:
                    K_secant = v / d
                    if K_secant < 0.7 * K_elastic:
                        levels['yield'] = curve[i-1]
                        break
            
            # Altri livelli basati su drift
            for point in curve:
                drift = point['top_drift']
                if 'yield' in levels:
                    if drift >= 1.5 * levels['yield']['top_drift'] and 'IO' not in levels:
                        levels['IO'] = point  # Immediate Occupancy
                    if drift >= 2.5 * levels['yield']['top_drift'] and 'LS' not in levels:
                        levels['LS'] = point  # Life Safety
                    if drift >= 4.0 * levels['yield']['top_drift'] and 'CP' not in levels:
                        levels['CP'] = point  # Collapse Prevention
        
        return levels
    
    def cyclic_analysis(self, protocol: List[float], vertical_loads: Dict) -> Dict:
        """Analisi ciclica"""
        logger.info("Inizio analisi ciclica")
        
        # Reset stato
        self.u_global = np.zeros(self.n_dof)
        for elem in self.elements:
            elem.update_state(np.zeros(6))
        
        # Applica carichi verticali
        F_vert = self.apply_loads(vertical_loads)
        if not self.solve_step(F_vert):
            logger.error("Fallita applicazione carichi verticali")
            return {}
        
        # Pattern laterale di riferimento
        F_lat_ref = self._lateral_pattern('triangular')
        
        # DOF controllo
        top_node = max(self.nodes.keys(), key=lambda n: self.nodes[n][1])
        control_dof = self.dof_map.get((top_node, 0), 0)
        height = max(self.nodes[n][1] for n in self.nodes)
        
        # Risultati
        results = {
            'cycles': [],
            'backbone': {'drift': [], 'shear': []},
            'energy_dissipated': []
        }
        
        # Esegui cicli
        for cycle_num, target_drift in enumerate(protocol):
            logger.info(f"Ciclo {cycle_num+1}: drift target = ±{target_drift}")
            
            cycle_data = {
                'drift': target_drift,
                'positive': {'drift': [], 'shear': []},
                'negative': {'drift': [], 'shear': []},
                'energy': 0.0
            }
            
            # Ciclo positivo e negativo
            for direction in [1, -1]:
                target_disp = direction * target_drift * height
                current_disp = self.u_global[control_dof]
                n_substeps = max(10, int(abs(target_disp - current_disp) / (0.001 * height)))
                
                disp_path = np.linspace(current_disp, target_disp, n_substeps)
                
                for disp in disp_path:
                    # Controllo in spostamento
                    lambda_factor = self._find_load_factor(F_vert, F_lat_ref, control_dof, disp)
                    
                    if lambda_factor is not None:
                        F_total = F_vert + lambda_factor * F_lat_ref
                        
                        if self.solve_step(F_total):
                            # Calcola taglio alla base
                            base_shear = lambda_factor * np.sum(F_lat_ref)
                            current_drift = self.u_global[control_dof] / height
                            
                            if direction > 0:
                                cycle_data['positive']['drift'].append(current_drift)
                                cycle_data['positive']['shear'].append(base_shear)
                            else:
                                cycle_data['negative']['drift'].append(current_drift)
                                cycle_data['negative']['shear'].append(base_shear)
                        else:
                            logger.warning(f"Non convergenza a drift={disp/height:.4f}")
                            break
            
            # Calcola energia dissipata
            if len(cycle_data['positive']['drift']) > 1 and len(cycle_data['negative']['drift']) > 1:
                # Area del ciclo isteretico
                pos_energy = np.trapz(cycle_data['positive']['shear'], 
                                     cycle_data['positive']['drift'])
                neg_energy = np.trapz(cycle_data['negative']['shear'], 
                                     cycle_data['negative']['drift'])
                cycle_data['energy'] = abs(pos_energy) + abs(neg_energy)
            
            results['cycles'].append(cycle_data)
            results['energy_dissipated'].append(cycle_data['energy'])
            
            # Aggiorna backbone
            if cycle_data['positive']['drift']:
                max_idx = np.argmax(np.abs(cycle_data['positive']['drift']))
                results['backbone']['drift'].append(cycle_data['positive']['drift'][max_idx])
                results['backbone']['shear'].append(cycle_data['positive']['shear'][max_idx])
        
        # Parametri isteretici
        results['hysteretic_parameters'] = extract_hysteretic_params(results)
        
        return results
    
    def _find_load_factor(self, F_vert: np.ndarray, F_lat: np.ndarray, 
                         control_dof: int, target_disp: float) -> Optional[float]:
        """Trova fattore di carico per raggiungere spostamento target"""
        # Stima iniziale
        if hasattr(self, '_last_lambda'):
            lambda_est = self._last_lambda * target_disp / self.u_global[control_dof]
        else:
            lambda_est = 100.0  # Stima iniziale
        
        # Iterazioni per convergere su target
        for _ in range(10):
            # Salva stato
            u_prev = self.u_global.copy()
            elem_states = self._save_element_states()
            
            # Prova con lambda corrente
            F_total = F_vert + lambda_est * F_lat
            
            if self.solve_step(F_total):
                error = self.u_global[control_dof] - target_disp
                
                if abs(error) < 1e-4:
                    self._last_lambda = lambda_est
                    return lambda_est
                else:
                    # Correzione lambda
                    lambda_est *= target_disp / self.u_global[control_dof]
            else:
                # Riduci lambda
                lambda_est *= 0.8
            
            # Ripristina stato
            self.u_global = u_prev
            self._restore_element_states(elem_states)
        
        return None
    
    def _save_element_states(self) -> List:
        """Salva stato corrente elementi"""
        states = []
        for elem in self.elements:
            elem_state = {
                'u_local': elem.u_local.copy(),
                'ip_states': []
            }
            for ip in elem.integration_points:
                ip_state = {
                    'epsilon0': ip.section.epsilon0,
                    'curvature': ip.section.curvature,
                    'fiber_strains': [f.strain for f in ip.section.fibers],
                    'fiber_stresses': [f.stress for f in ip.section.fibers]
                }
                elem_state['ip_states'].append(ip_state)
            states.append(elem_state)
        return states
    
    def _restore_element_states(self, states: List):
        """Ripristina stato elementi"""
        for elem, state in zip(self.elements, states):
            elem.u_local = state['u_local']
            for ip, ip_state in zip(elem.integration_points, state['ip_states']):
                ip.section.epsilon0 = ip_state['epsilon0']
                ip.section.curvature = ip_state['curvature']
                for fiber, strain, stress in zip(ip.section.fibers, 
                                                ip_state['fiber_strains'],
                                                ip_state['fiber_stresses']):
                    fiber.strain = strain
                    fiber.stress = stress
    
    def moment_curvature_analysis(self, element_id: str, N: float = 0.0,
                                 max_curvature: float = 0.1) -> Dict:
        """Analisi momento-curvatura per elemento specifico"""
        elem = next((e for e in self.elements if e.id == element_id), None)
        
        if not elem:
            logger.error(f"Elemento {element_id} non trovato")
            return {}
        
        # Usa sezione a metà elemento
        mid_section = elem.integration_points[len(elem.integration_points)//2].section
        
        # Calcola diagramma M-chi
        mc_data = mid_section.get_moment_curvature(N, max_curvature)
        
        # Aggiungi parametri derivati
        mc_data['ductility'] = calculate_section_ductility(mc_data)
        
        # Trova punti caratteristici
        moments = mc_data['moment']
        curvatures = mc_data['curvature']
        
        if len(moments) > 0:
            # Punto di prima fessurazione (cambio pendenza)
            for i in range(1, len(moments)-1):
                if i > 2:
                    k1 = (moments[i] - moments[i-1]) / (curvatures[i] - curvatures[i-1])
                    k2 = (moments[i+1] - moments[i]) / (curvatures[i+1] - curvatures[i])
                    if k2 < 0.7 * k1:
                        mc_data['cracking'] = {
                            'moment': moments[i],
                            'curvature': curvatures[i]
                        }
                        break
            
            # Punto di snervamento (criterio area uguale)
            if 'cracking' in mc_data:
                idx_cr = curvatures.tolist().index(mc_data['cracking']['curvature'])
                area_el = 0
                for i in range(idx_cr, len(moments)):
                    area_tot = np.trapz(moments[:i+1], curvatures[:i+1])
                    m_el = moments[idx_cr] + (moments[i] - moments[idx_cr]) * (curvatures[i] / curvatures[i])
                    area_el_rect = m_el * curvatures[i]
                    if area_el_rect >= area_tot:
                        mc_data['yielding'] = {
                            'moment': moments[i],
                            'curvature': curvatures[i]
                        }
                        break
            
            # Punto ultimo (90% momento max)
            m_max = np.max(moments)
            idx_max = np.argmax(moments)
            for i in range(idx_max, len(moments)):
                if moments[i] < 0.9 * m_max:
                    mc_data['ultimate'] = {
                        'moment': moments[i-1],
                        'curvature': curvatures[i-1]
                    }
                    break
        
        return mc_data

def _analyze_fiber(wall_data: Dict, material: MaterialProperties,
                   loads: Dict, options: Dict) -> Dict:
    """Interfaccia per analisi FIBER da engine principale"""
    logger.info("Analisi FIBER completa")
    
    # Crea modello
    law_type = options.get('constitutive_law', ConstitutiveLaw.MANDER)
    model = FiberModel(material, law_type)
    
    # Costruisci geometria da wall_data
    _build_geometry_from_wall(model, wall_data, options)
    
    # Tipo analisi
    analysis_type = options.get('analysis_type', 'pushover')
    
    results = {
        'method': 'FIBER_MODEL',
        'constitutive_law': law_type.value,
        'elements': len(model.elements),
        'nodes': len(model.nodes),
        'dofs': model.n_dof
    }
    
    if analysis_type == 'pushover':
        vertical = distribute_vertical_loads(loads, model.elements)
        pattern = options.get('lateral_pattern', 'triangular')
        max_drift = options.get('max_drift', 0.05)
        n_steps = options.get('n_steps', 100)
        
        pushover_results = model.pushover_analysis(vertical, pattern, max_drift, n_steps)
        results.update(pushover_results)
        
    elif analysis_type == 'cyclic':
        protocol = options.get('protocol', [0.001, 0.002, 0.005, 0.01, 0.02])
        vertical = distribute_vertical_loads(loads, model.elements)
        
        cyclic_results = model.cyclic_analysis(protocol, vertical)
        results.update(cyclic_results)
        
    elif analysis_type == 'moment_curvature':
        results['moment_curvature'] = {}
        
        for elem in model.elements:
            N = loads.get(elem.id, {}).get('N', 0)
            max_curv = options.get('max_curvature', 0.1)
            
            mc_data = model.moment_curvature_analysis(elem.id, N, max_curv)
            results['moment_curvature'][elem.id] = mc_data
    
    # Confronto leggi costitutive se richiesto
    if options.get('compare_laws', False):
        results['law_comparison'] = compare_constitutive_laws(
            model.elements, material, loads
        )
    
    return results

def _build_geometry_from_wall(model: FiberModel, wall_data: Dict, options: Dict):
    """Costruisce modello geometrico da dati parete"""
    # Esempio semplificato - da adattare a formato wall_data
    nodes = wall_data.get('nodes', {})
    elements = wall_data.get('elements', [])
    
    # Aggiungi nodi
    for node_id, coords in nodes.items():
        model.add_node(node_id, coords['x'], coords['y'])
    
    # Aggiungi elementi
    for elem_data in elements:
        model.add_element(
            elem_data['id'],
            elem_data['node1'],
            elem_data['node2'],
            elem_data['geometry'],
            elem_data.get('n_fibers', 20),
            elem_data.get('n_ip', 3)
        )
    
    # Vincoli
    constraints = wall_data.get('constraints', {})
    for node_id, dofs in constraints.items():
        model.add_constraint(node_id, dofs)