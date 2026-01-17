# -*- coding: utf-8 -*-
"""
Wrapper per FreeCAD Arch workbench.

Fornisce API semplificate per creare elementi BIM con proprietà NTC 2018.
"""

from typing import Tuple, List, Optional, Dict, Any
import math


def _get_freecad():
    """Importa FreeCAD con gestione errori."""
    try:
        import FreeCAD
        return FreeCAD
    except ImportError:
        raise ImportError("FreeCAD non disponibile. Assicurati di eseguire da FreeCAD.")


def _get_arch():
    """Importa modulo Arch con gestione errori."""
    try:
        import Arch
        return Arch
    except ImportError:
        raise ImportError("Modulo Arch non disponibile.")


def _get_draft():
    """Importa modulo Draft con gestione errori."""
    try:
        import Draft
        return Draft
    except ImportError:
        raise ImportError("Modulo Draft non disponibile.")


def create_wall(
    start: Tuple[float, float],
    end: Tuple[float, float],
    thickness: float = 0.40,
    height: float = 3.0,
    name: str = None,
    floor: int = 0,
    floor_height: float = 3.0,
    material_type: str = None,
    **kwargs
) -> Any:
    """
    Crea un muro usando Arch.makeWall.

    Args:
        start: Punto iniziale (x, y) in metri
        end: Punto finale (x, y) in metri
        thickness: Spessore muro in metri
        height: Altezza muro in metri
        name: Nome del muro (auto se None)
        floor: Piano (0 = terra)
        floor_height: Altezza interpiano per calcolo Z
        material_type: Tipo materiale NTC (opzionale)

    Returns:
        Oggetto Arch.Wall creato
    """
    FreeCAD = _get_freecad()
    Arch = _get_arch()
    Draft = _get_draft()

    doc = FreeCAD.ActiveDocument
    if not doc:
        doc = FreeCAD.newDocument("Muratura")

    # Converti metri in mm (FreeCAD usa mm internamente)
    x1, y1 = start[0] * 1000, start[1] * 1000
    x2, y2 = end[0] * 1000, end[1] * 1000
    z = floor * floor_height * 1000
    h = height * 1000
    t = thickness * 1000

    # Crea linea base per il muro
    p1 = FreeCAD.Vector(x1, y1, z)
    p2 = FreeCAD.Vector(x2, y2, z)

    baseline = Draft.makeLine(p1, p2)
    baseline.ViewObject.Visibility = False  # Nascondi la linea

    # Crea muro
    wall = Arch.makeWall(baseline, height=h, width=t, name=name)

    # Aggiungi proprietà Muratura
    _add_wall_properties(wall, floor, thickness, height, material_type)

    # Colore mattone default
    if hasattr(wall, 'ViewObject'):
        wall.ViewObject.ShapeColor = (0.8, 0.4, 0.2)

    doc.recompute()
    return wall


def create_window(
    wall: Any,
    position: float,
    width: float = 1.2,
    height: float = 1.4,
    sill_height: float = 0.9,
    window_type: str = "Finestra",
    name: str = None,
    **kwargs
) -> Any:
    """
    Crea una finestra in un muro usando Arch.makeWindow.

    Args:
        wall: Oggetto muro in cui inserire la finestra
        position: Posizione lungo il muro (da inizio) in metri
        width: Larghezza finestra in metri
        height: Altezza finestra in metri
        sill_height: Altezza davanzale in metri
        window_type: Tipo (Finestra, Portafinestra)
        name: Nome (auto se None)

    Returns:
        Oggetto Arch.Window creato
    """
    FreeCAD = _get_freecad()
    Arch = _get_arch()

    doc = FreeCAD.ActiveDocument

    # Converti in mm
    w = width * 1000
    h = height * 1000
    sill = sill_height * 1000
    pos = position * 1000

    # Crea sketch rettangolare per la finestra
    import Part

    # Ottieni direzione del muro
    if hasattr(wall, 'Base') and wall.Base:
        base = wall.Base
        if hasattr(base, 'Start') and hasattr(base, 'End'):
            wall_start = base.Start
            wall_end = base.End
        else:
            bb = wall.Shape.BoundBox
            wall_start = FreeCAD.Vector(bb.XMin, bb.YMin, bb.ZMin)
            wall_end = FreeCAD.Vector(bb.XMax, bb.YMax, bb.ZMin)
    else:
        bb = wall.Shape.BoundBox
        wall_start = FreeCAD.Vector(bb.XMin, bb.YMin, bb.ZMin)
        wall_end = FreeCAD.Vector(bb.XMax, bb.YMax, bb.ZMin)

    # Calcola posizione finestra
    dx = wall_end.x - wall_start.x
    dy = wall_end.y - wall_start.y
    length = math.sqrt(dx**2 + dy**2)

    if length > 0:
        dir_x = dx / length
        dir_y = dy / length
    else:
        dir_x, dir_y = 1, 0

    # Posizione centro finestra
    cx = wall_start.x + dir_x * pos
    cy = wall_start.y + dir_y * pos
    cz = wall_start.z + sill

    # Crea finestra semplice (rettangolo)
    # Nota: per finestre più complesse usare Window Presets
    window = Arch.makeWindowPreset(
        "Fixed",
        width=w,
        height=h,
        h1=100,  # Frame width
        h2=100,
        h3=0,
        w1=0,
        w2=0,
        o1=0,
        o2=0,
        placement=FreeCAD.Placement(
            FreeCAD.Vector(cx, cy, cz),
            FreeCAD.Rotation(FreeCAD.Vector(0, 0, 1), math.degrees(math.atan2(dir_y, dir_x)))
        )
    )

    if window:
        # Associa finestra al muro
        window.Hosts = [wall]

        # Aggiungi proprietà
        if not hasattr(window, 'WindowType'):
            window.addProperty("App::PropertyString", "WindowType", "Muratura")
        window.WindowType = window_type

        if not hasattr(window, 'SillHeight'):
            window.addProperty("App::PropertyFloat", "SillHeight", "Muratura")
        window.SillHeight = sill_height

    doc.recompute()
    return window


def create_door(
    wall: Any,
    position: float,
    width: float = 0.90,
    height: float = 2.10,
    door_type: str = "Porta interna",
    name: str = None,
    **kwargs
) -> Any:
    """
    Crea una porta in un muro.

    Args:
        wall: Oggetto muro
        position: Posizione lungo il muro in metri
        width: Larghezza porta in metri
        height: Altezza porta in metri
        door_type: Tipo (Porta interna, Porta esterna, Portone)
        name: Nome (auto se None)

    Returns:
        Oggetto Arch.Window (porta) creato
    """
    # Porta = finestra con davanzale 0
    return create_window(
        wall=wall,
        position=position,
        width=width,
        height=height,
        sill_height=0,
        window_type=door_type,
        name=name,
        **kwargs
    )


def create_structure(
    position: Tuple[float, float],
    width: float = 0.30,
    depth: float = 0.30,
    height: float = 3.0,
    structure_type: str = "Pilastro",
    floor: int = 0,
    floor_height: float = 3.0,
    material: str = "C.A.",
    name: str = None,
    **kwargs
) -> Any:
    """
    Crea elemento strutturale (pilastro, trave) usando Arch.makeStructure.

    Args:
        position: Posizione (x, y) in metri
        width: Larghezza in metri
        depth: Profondità in metri
        height: Altezza in metri
        structure_type: Tipo (Pilastro, Trave, Cordolo)
        floor: Piano
        floor_height: Altezza interpiano
        material: Materiale (C.A., Acciaio, Legno)
        name: Nome (auto se None)

    Returns:
        Oggetto Arch.Structure creato
    """
    FreeCAD = _get_freecad()
    Arch = _get_arch()

    doc = FreeCAD.ActiveDocument
    if not doc:
        doc = FreeCAD.newDocument("Muratura")

    # Converti in mm
    x, y = position[0] * 1000, position[1] * 1000
    z = floor * floor_height * 1000
    w = width * 1000
    d = depth * 1000
    h = height * 1000

    # Crea struttura
    struct = Arch.makeStructure(
        length=w,
        width=d,
        height=h,
        name=name
    )

    # Posiziona
    struct.Placement.Base = FreeCAD.Vector(x - w/2, y - d/2, z)

    # Proprietà
    if not hasattr(struct, 'StructureType'):
        struct.addProperty("App::PropertyString", "StructureType", "Muratura")
    struct.StructureType = structure_type

    if not hasattr(struct, 'StructureMaterial'):
        struct.addProperty("App::PropertyString", "StructureMaterial", "Muratura")
    struct.StructureMaterial = material

    if not hasattr(struct, 'Floor'):
        struct.addProperty("App::PropertyInteger", "Floor", "Muratura")
    struct.Floor = floor

    # Colore per tipo
    if hasattr(struct, 'ViewObject'):
        if material == "C.A.":
            struct.ViewObject.ShapeColor = (0.7, 0.7, 0.7)
        elif material == "Acciaio":
            struct.ViewObject.ShapeColor = (0.3, 0.3, 0.5)
        elif material == "Legno":
            struct.ViewObject.ShapeColor = (0.6, 0.4, 0.2)

    doc.recompute()
    return struct


def create_floor(
    boundary: List[Tuple[float, float]] = None,
    floor: int = 0,
    floor_height: float = 3.0,
    thickness: float = 0.25,
    floor_type: str = "Latero-cemento",
    name: str = None,
    **kwargs
) -> Any:
    """
    Crea un solaio usando Arch.makeFloor o Arch.makeSlab.

    Args:
        boundary: Lista punti perimetro [(x,y), ...] in metri (auto da muri se None)
        floor: Piano
        floor_height: Altezza interpiano
        thickness: Spessore solaio in metri
        floor_type: Tipologia solaio
        name: Nome (auto se None)

    Returns:
        Oggetto Arch.Floor o Arch.Structure (slab) creato
    """
    FreeCAD = _get_freecad()
    Arch = _get_arch()

    doc = FreeCAD.ActiveDocument
    if not doc:
        return None

    z = (floor + 1) * floor_height * 1000  # Sopra il piano
    t = thickness * 1000

    if boundary:
        # Crea da boundary esplicito
        import Part
        points = [FreeCAD.Vector(p[0]*1000, p[1]*1000, z) for p in boundary]
        points.append(points[0])  # Chiudi poligono
        wire = Part.makePolygon(points)
        face = Part.Face(wire)
        slab_shape = face.extrude(FreeCAD.Vector(0, 0, t))

        slab = doc.addObject("Part::Feature", name or f"Solaio_{floor}")
        slab.Shape = slab_shape
    else:
        # Auto-detect da muri esistenti
        walls = [o for o in doc.Objects if hasattr(o, 'IfcType') and o.IfcType == 'Wall']
        if not walls:
            walls = [o for o in doc.Objects if o.Name.startswith('Wall') or o.Name.startswith('Muro')]

        if walls:
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')

            for wall in walls:
                if hasattr(wall, 'Floor') and wall.Floor != floor:
                    continue
                bb = wall.Shape.BoundBox
                min_x = min(min_x, bb.XMin)
                min_y = min(min_y, bb.YMin)
                max_x = max(max_x, bb.XMax)
                max_y = max(max_y, bb.YMax)

            if min_x < float('inf'):
                import Part
                slab_shape = Part.makeBox(max_x - min_x, max_y - min_y, t)
                slab_shape.translate(FreeCAD.Vector(min_x, min_y, z))

                slab = doc.addObject("Part::Feature", name or f"Solaio_{floor}")
                slab.Shape = slab_shape
            else:
                return None
        else:
            return None

    # Proprietà
    if not hasattr(slab, 'Floor'):
        slab.addProperty("App::PropertyInteger", "Floor", "Muratura")
    slab.Floor = floor

    if not hasattr(slab, 'FloorType'):
        slab.addProperty("App::PropertyString", "FloorType", "Muratura")
    slab.FloorType = floor_type

    if not hasattr(slab, 'Thickness'):
        slab.addProperty("App::PropertyFloat", "Thickness", "Muratura")
    slab.Thickness = thickness

    # Colore
    if hasattr(slab, 'ViewObject'):
        slab.ViewObject.ShapeColor = (0.9, 0.9, 0.8)
        slab.ViewObject.Transparency = 50

    doc.recompute()
    return slab


def create_roof(
    ridge_height: float = 2.0,
    overhang: float = 0.5,
    roof_type: str = "Due falde",
    name: str = None,
    **kwargs
) -> Any:
    """
    Crea una copertura.

    Args:
        ridge_height: Altezza colmo in metri
        overhang: Sporto gronda in metri
        roof_type: Tipo (Piana, Una falda, Due falde, Padiglione)
        name: Nome (auto se None)

    Returns:
        Oggetto copertura creato
    """
    FreeCAD = _get_freecad()
    import Part

    doc = FreeCAD.ActiveDocument
    if not doc:
        return None

    # Trova bounding box edificio
    walls = [o for o in doc.Objects if o.Name.startswith('Wall') or o.Name.startswith('Muro')]
    if not walls:
        return None

    min_x = min_y = float('inf')
    max_x = max_y = max_z = float('-inf')

    for wall in walls:
        bb = wall.Shape.BoundBox
        min_x = min(min_x, bb.XMin)
        min_y = min(min_y, bb.YMin)
        max_x = max(max_x, bb.XMax)
        max_y = max(max_y, bb.YMax)
        max_z = max(max_z, bb.ZMax)

    ov = overhang * 1000
    rh = ridge_height * 1000
    mid_y = (min_y + max_y) / 2

    # Due falde
    p1 = FreeCAD.Vector(min_x - ov, min_y - ov, max_z)
    p2 = FreeCAD.Vector(max_x + ov, min_y - ov, max_z)
    p3 = FreeCAD.Vector(max_x + ov, mid_y, max_z + rh)
    p4 = FreeCAD.Vector(min_x - ov, mid_y, max_z + rh)
    p5 = FreeCAD.Vector(min_x - ov, max_y + ov, max_z)
    p6 = FreeCAD.Vector(max_x + ov, max_y + ov, max_z)

    face1 = Part.Face(Part.makePolygon([p1, p2, p3, p4, p1]))
    face2 = Part.Face(Part.makePolygon([p4, p3, p6, p5, p4]))

    thickness = 200
    solid1 = face1.extrude(FreeCAD.Vector(0, 0, thickness))
    solid2 = face2.extrude(FreeCAD.Vector(0, 0, thickness))

    roof = doc.addObject("Part::Feature", name or "Copertura")
    roof.Shape = Part.makeCompound([solid1, solid2])

    # Proprietà
    if not hasattr(roof, 'RoofType'):
        roof.addProperty("App::PropertyString", "RoofType", "Muratura")
    roof.RoofType = roof_type

    if not hasattr(roof, 'RidgeHeight'):
        roof.addProperty("App::PropertyFloat", "RidgeHeight", "Muratura")
    roof.RidgeHeight = ridge_height

    if not hasattr(roof, 'Overhang'):
        roof.addProperty("App::PropertyFloat", "Overhang", "Muratura")
    roof.Overhang = overhang

    # Colore
    if hasattr(roof, 'ViewObject'):
        roof.ViewObject.ShapeColor = (0.6, 0.3, 0.2)

    doc.recompute()
    return roof


def import_dxf(
    filepath: str,
    layer: str = None,
    scale: float = 1.0,
    offset: Tuple[float, float] = (0, 0),
    wall_height: float = 3.0,
    wall_thickness: float = 0.40,
    **kwargs
) -> List[Any]:
    """
    Importa DXF e converte polilinee in muri.

    Args:
        filepath: Percorso file DXF
        layer: Layer da importare (tutti se None)
        scale: Fattore scala (1.0 = metri, 0.001 = mm)
        offset: Offset origine (x, y) in metri
        wall_height: Altezza muri da creare
        wall_thickness: Spessore muri da creare

    Returns:
        Lista muri creati
    """
    FreeCAD = _get_freecad()

    # Import DXF
    import importDXF
    importDXF.open(filepath)

    doc = FreeCAD.ActiveDocument
    walls_created = []

    # Trova polilinee importate
    for obj in doc.Objects:
        if obj.TypeId in ['Part::Feature', 'Sketcher::SketchObject', 'Part::Part2DObjectPython']:
            if layer and hasattr(obj, 'Layer') and obj.Layer != layer:
                continue

            # Estrai segmenti
            if hasattr(obj, 'Shape') and obj.Shape:
                edges = obj.Shape.Edges
                for edge in edges:
                    if len(edge.Vertexes) >= 2:
                        p1 = edge.Vertexes[0].Point
                        p2 = edge.Vertexes[1].Point

                        # Applica scala e offset
                        start = (
                            p1.x * scale + offset[0],
                            p1.y * scale + offset[1]
                        )
                        end = (
                            p2.x * scale + offset[0],
                            p2.y * scale + offset[1]
                        )

                        wall = create_wall(
                            start=start,
                            end=end,
                            height=wall_height,
                            thickness=wall_thickness
                        )
                        walls_created.append(wall)

    return walls_created


def import_ifc(filepath: str, **kwargs) -> List[Any]:
    """
    Importa file IFC e converte elementi in oggetti Muratura.

    Args:
        filepath: Percorso file IFC

    Returns:
        Lista oggetti creati
    """
    FreeCAD = _get_freecad()

    try:
        import importIFC
        importIFC.open(filepath)
    except ImportError:
        raise ImportError("Modulo importIFC non disponibile")

    doc = FreeCAD.ActiveDocument
    objects_created = []

    # Processa oggetti importati
    for obj in doc.Objects:
        if hasattr(obj, 'IfcType'):
            if obj.IfcType == 'Wall':
                _add_wall_properties(obj)
                objects_created.append(obj)
            elif obj.IfcType == 'Window':
                objects_created.append(obj)
            elif obj.IfcType == 'Column' or obj.IfcType == 'Beam':
                objects_created.append(obj)

    return objects_created


def _add_wall_properties(
    wall: Any,
    floor: int = 0,
    thickness: float = None,
    height: float = None,
    material_type: str = None
):
    """Aggiunge proprietà NTC a un muro."""
    # Piano
    if not hasattr(wall, 'Floor'):
        wall.addProperty("App::PropertyInteger", "Floor", "Muratura", "Piano")
    wall.Floor = floor

    # Spessore (in metri)
    if not hasattr(wall, 'WallThickness'):
        wall.addProperty("App::PropertyFloat", "WallThickness", "Muratura", "Spessore [m]")
    if thickness:
        wall.WallThickness = thickness

    # Altezza (in metri)
    if not hasattr(wall, 'WallHeight'):
        wall.addProperty("App::PropertyFloat", "WallHeight", "Muratura", "Altezza [m]")
    if height:
        wall.WallHeight = height

    # Materiale
    if not hasattr(wall, 'MasonryType'):
        wall.addProperty("App::PropertyString", "MasonryType", "NTC2018", "Tipo muratura")
    if material_type:
        wall.MasonryType = material_type

    # Proprietà meccaniche
    for prop_name, prop_desc in [
        ("fm", "Resistenza compressione [MPa]"),
        ("tau0", "Resistenza taglio [MPa]"),
        ("E", "Modulo elastico [MPa]"),
        ("G", "Modulo taglio [MPa]"),
        ("w", "Peso specifico [kN/m³]"),
    ]:
        if not hasattr(wall, prop_name):
            wall.addProperty("App::PropertyFloat", prop_name, "NTC2018", prop_desc)

    # Coefficienti correttivi
    if not hasattr(wall, 'CorrectionFactors'):
        wall.addProperty("App::PropertyString", "CorrectionFactors", "NTC2018", "Fattori correttivi JSON")
    wall.CorrectionFactors = "{}"
