# -*- coding: utf-8 -*-
"""
Equivalent Frame Generator

Generazione automatica del telaio equivalente da geometria BIM.
Identifica maschi murari (piers) e fasce di piano (spandrels).
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import math


@dataclass
class Pier:
    """Maschio murario."""
    id: str
    wall_id: str
    floor: int
    x: float  # Centro X
    y: float  # Centro Y
    z_bottom: float  # Quota inferiore
    z_top: float  # Quota superiore
    width: float  # Larghezza
    thickness: float  # Spessore
    height: float  # Altezza
    h0: float = 0  # Altezza efficace
    material: str = ""

    # Proprietà calcolate
    area: float = 0
    moment_of_inertia: float = 0
    shear_area: float = 0

    # Nodi
    node_bottom: int = 0
    node_top: int = 0

    def calculate_properties(self):
        """Calcola proprietà geometriche."""
        self.area = self.width * self.thickness
        self.moment_of_inertia = self.thickness * self.width**3 / 12
        self.shear_area = self.area * 5/6  # Fattore di taglio rettangolare
        if self.h0 == 0:
            self.h0 = self.height  # Default: altezza intera


@dataclass
class Spandrel:
    """Fascia di piano."""
    id: str
    floor: int
    x: float
    y: float
    z: float
    width: float  # Larghezza (luce apertura)
    height: float  # Altezza fascia
    thickness: float
    material: str = ""

    # Connessioni
    pier_left: str = ""
    pier_right: str = ""

    # Nodi
    node_left: int = 0
    node_right: int = 0


@dataclass
class RigidNode:
    """Nodo rigido."""
    id: int
    x: float
    y: float
    z: float
    connected_elements: List[str] = field(default_factory=list)


@dataclass
class Opening:
    """Apertura in un muro."""
    id: str
    wall_id: str
    x: float  # Posizione X nel muro (da inizio)
    width: float
    height: float
    z_bottom: float  # Quota davanzale


class EquivalentFrameGenerator:
    """
    Genera il telaio equivalente da un modello BIM.
    """

    def __init__(self):
        self.walls: List[Dict] = []
        self.openings: List[Opening] = []
        self.floors: List[float] = []  # Quote piani

        self.piers: List[Pier] = []
        self.spandrels: List[Spandrel] = []
        self.nodes: List[RigidNode] = []

        self._node_counter = 0

    def add_wall(self, wall_data: Dict):
        """Aggiunge un muro al modello."""
        self.walls.append(wall_data)

    def add_opening(self, opening: Opening):
        """Aggiunge un'apertura."""
        self.openings.append(opening)

    def set_floors(self, floor_levels: List[float]):
        """Imposta le quote dei piani."""
        self.floors = sorted(floor_levels)

    def generate(self) -> Dict[str, Any]:
        """
        Genera il telaio equivalente.

        Returns:
            Dizionario con piers, spandrels, nodes
        """
        self.piers = []
        self.spandrels = []
        self.nodes = []
        self._node_counter = 0

        # Per ogni muro, identifica maschi e fasce
        for wall in self.walls:
            wall_openings = [o for o in self.openings if o.wall_id == wall.get('id')]
            self._process_wall(wall, wall_openings)

        # Crea nodi
        self._create_nodes()

        # Calcola proprietà
        for pier in self.piers:
            pier.calculate_properties()

        return {
            "piers": self.piers,
            "spandrels": self.spandrels,
            "nodes": self.nodes,
            "statistics": {
                "n_piers": len(self.piers),
                "n_spandrels": len(self.spandrels),
                "n_nodes": len(self.nodes),
                "n_floors": len(self.floors),
            }
        }

    def _process_wall(self, wall: Dict, openings: List[Opening]):
        """Processa un singolo muro."""
        wall_id = wall.get('id', 'wall')
        x_start = wall.get('x_start', 0)
        y_start = wall.get('y_start', 0)
        x_end = wall.get('x_end', x_start + wall.get('length', 5))
        y_end = wall.get('y_end', y_start)
        thickness = wall.get('thickness', 0.3)
        z_bottom = wall.get('z_bottom', 0)
        z_top = wall.get('z_top', 3)
        floor = wall.get('floor', 0)
        material = wall.get('material', '')

        # Lunghezza muro
        wall_length = math.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
        wall_height = z_top - z_bottom

        # Direzione muro
        if wall_length > 0:
            dx = (x_end - x_start) / wall_length
            dy = (y_end - y_start) / wall_length
        else:
            dx, dy = 1, 0

        # Ordina aperture per posizione
        openings_sorted = sorted(openings, key=lambda o: o.x)

        if not openings_sorted:
            # Muro pieno → un unico maschio
            pier = Pier(
                id=f"P_{wall_id}_1",
                wall_id=wall_id,
                floor=floor,
                x=(x_start + x_end) / 2,
                y=(y_start + y_end) / 2,
                z_bottom=z_bottom,
                z_top=z_top,
                width=wall_length,
                thickness=thickness,
                height=wall_height,
                material=material
            )
            self.piers.append(pier)
        else:
            # Identifica maschi tra aperture
            prev_x = 0  # Posizione fine apertura precedente

            for i, opening in enumerate(openings_sorted):
                # Maschio prima dell'apertura
                pier_width = opening.x - prev_x
                if pier_width > 0.1:  # Minimo 10cm
                    pier_x = x_start + dx * (prev_x + pier_width/2)
                    pier_y = y_start + dy * (prev_x + pier_width/2)

                    # Altezza maschio = minore tra altezza aperture adiacenti
                    pier_height = wall_height
                    if i > 0:
                        prev_opening = openings_sorted[i-1]
                        pier_height = min(pier_height, prev_opening.height)

                    pier = Pier(
                        id=f"P_{wall_id}_{len(self.piers)+1}",
                        wall_id=wall_id,
                        floor=floor,
                        x=pier_x,
                        y=pier_y,
                        z_bottom=z_bottom,
                        z_top=z_top,
                        width=pier_width,
                        thickness=thickness,
                        height=pier_height,
                        material=material
                    )
                    self.piers.append(pier)

                # Fascia sopra apertura
                spandrel_height = z_top - (opening.z_bottom + opening.height)
                if spandrel_height > 0.1:
                    spandrel = Spandrel(
                        id=f"S_{wall_id}_{len(self.spandrels)+1}",
                        floor=floor,
                        x=x_start + dx * (opening.x + opening.width/2),
                        y=y_start + dy * (opening.x + opening.width/2),
                        z=opening.z_bottom + opening.height + spandrel_height/2,
                        width=opening.width,
                        height=spandrel_height,
                        thickness=thickness,
                        material=material
                    )
                    self.spandrels.append(spandrel)

                prev_x = opening.x + opening.width

            # Maschio dopo ultima apertura
            pier_width = wall_length - prev_x
            if pier_width > 0.1:
                pier_x = x_start + dx * (prev_x + pier_width/2)
                pier_y = y_start + dy * (prev_x + pier_width/2)

                pier = Pier(
                    id=f"P_{wall_id}_{len(self.piers)+1}",
                    wall_id=wall_id,
                    floor=floor,
                    x=pier_x,
                    y=pier_y,
                    z_bottom=z_bottom,
                    z_top=z_top,
                    width=pier_width,
                    thickness=thickness,
                    height=wall_height,
                    material=material
                )
                self.piers.append(pier)

    def _create_nodes(self):
        """Crea nodi rigidi alle estremità degli elementi."""
        node_positions = {}  # (x, y, z) -> node_id

        # Nodi per maschi
        for pier in self.piers:
            # Nodo inferiore
            pos_bottom = (round(pier.x, 2), round(pier.y, 2), round(pier.z_bottom, 2))
            if pos_bottom not in node_positions:
                node = self._create_node(pier.x, pier.y, pier.z_bottom)
                node_positions[pos_bottom] = node.id
            pier.node_bottom = node_positions[pos_bottom]

            # Nodo superiore
            pos_top = (round(pier.x, 2), round(pier.y, 2), round(pier.z_top, 2))
            if pos_top not in node_positions:
                node = self._create_node(pier.x, pier.y, pier.z_top)
                node_positions[pos_top] = node.id
            pier.node_top = node_positions[pos_top]

        # Nodi per fasce
        for spandrel in self.spandrels:
            # Nodo sinistro
            x_left = spandrel.x - spandrel.width/2
            pos_left = (round(x_left, 2), round(spandrel.y, 2), round(spandrel.z, 2))
            if pos_left not in node_positions:
                node = self._create_node(x_left, spandrel.y, spandrel.z)
                node_positions[pos_left] = node.id
            spandrel.node_left = node_positions[pos_left]

            # Nodo destro
            x_right = spandrel.x + spandrel.width/2
            pos_right = (round(x_right, 2), round(spandrel.y, 2), round(spandrel.z, 2))
            if pos_right not in node_positions:
                node = self._create_node(x_right, spandrel.y, spandrel.z)
                node_positions[pos_right] = node.id
            spandrel.node_right = node_positions[pos_right]

    def _create_node(self, x: float, y: float, z: float) -> RigidNode:
        """Crea un nuovo nodo."""
        self._node_counter += 1
        node = RigidNode(
            id=self._node_counter,
            x=x,
            y=y,
            z=z
        )
        self.nodes.append(node)
        return node


def generate_from_freecad(doc) -> Dict[str, Any]:
    """
    Genera telaio equivalente da documento FreeCAD.

    Args:
        doc: Documento FreeCAD

    Returns:
        Dizionario con telaio equivalente
    """
    generator = EquivalentFrameGenerator()

    # Estrai quote piani
    floors = set()

    # Trova muri
    for obj in doc.Objects:
        if hasattr(obj, 'IfcType') and obj.IfcType == "Wall":
            # Oggetto Arch.Wall
            bb = obj.Shape.BoundBox
            wall_data = {
                'id': obj.Name,
                'x_start': bb.XMin / 1000,
                'y_start': bb.YMin / 1000,
                'x_end': bb.XMax / 1000,
                'y_end': bb.YMax / 1000,
                'z_bottom': bb.ZMin / 1000,
                'z_top': bb.ZMax / 1000,
                'thickness': min(bb.XLength, bb.YLength) / 1000,
                'length': max(bb.XLength, bb.YLength) / 1000,
                'floor': getattr(obj, 'Floor', 0),
                'material': getattr(obj, 'MasonryType', ''),
            }
            generator.add_wall(wall_data)
            floors.add(bb.ZMin / 1000)
            floors.add(bb.ZMax / 1000)

        elif 'Muro' in obj.Name or 'Wall' in obj.Name:
            # Oggetto Part
            if hasattr(obj, 'Shape'):
                bb = obj.Shape.BoundBox
                wall_data = {
                    'id': obj.Name,
                    'x_start': bb.XMin / 1000,
                    'y_start': bb.YMin / 1000,
                    'x_end': bb.XMax / 1000,
                    'y_end': bb.YMax / 1000,
                    'z_bottom': bb.ZMin / 1000,
                    'z_top': bb.ZMax / 1000,
                    'thickness': min(bb.XLength, bb.YLength) / 1000,
                    'length': max(bb.XLength, bb.YLength) / 1000,
                    'floor': getattr(obj, 'Floor', 0),
                    'material': getattr(obj, 'Material', ''),
                }
                generator.add_wall(wall_data)
                floors.add(bb.ZMin / 1000)
                floors.add(bb.ZMax / 1000)

    # Trova aperture (Window objects)
    for obj in doc.Objects:
        if hasattr(obj, 'IfcType') and obj.IfcType in ["Window", "Door"]:
            if hasattr(obj, 'Hosts') and obj.Hosts:
                wall = obj.Hosts[0]
                bb = obj.Shape.BoundBox
                opening = Opening(
                    id=obj.Name,
                    wall_id=wall.Name,
                    x=(bb.XMin + bb.XLength/2) / 1000,
                    width=bb.XLength / 1000,
                    height=bb.ZLength / 1000,
                    z_bottom=bb.ZMin / 1000,
                )
                generator.add_opening(opening)

    generator.set_floors(sorted(list(floors)))

    return generator.generate()
