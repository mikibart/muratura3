# -*- coding: utf-8 -*-
"""
Gestione proprietà NTC 2018 su oggetti Arch.
"""

from typing import Dict, Any, Optional
import json


def add_ntc_properties(obj: Any, category: str = "Wall") -> None:
    """
    Aggiunge proprietà NTC 2018 a un oggetto FreeCAD.

    Args:
        obj: Oggetto FreeCAD (Wall, Structure, etc.)
        category: Categoria oggetto (Wall, Column, Beam, Slab)
    """
    # Proprietà comuni
    common_props = [
        ("App::PropertyInteger", "Floor", "Muratura", "Piano"),
        ("App::PropertyString", "ElementType", "Muratura", "Tipo elemento"),
    ]

    for prop_type, name, group, desc in common_props:
        if not hasattr(obj, name):
            obj.addProperty(prop_type, name, group, desc)

    if category == "Wall":
        _add_wall_ntc_properties(obj)
    elif category in ["Column", "Beam"]:
        _add_structure_ntc_properties(obj)
    elif category == "Slab":
        _add_slab_ntc_properties(obj)


def _add_wall_ntc_properties(obj: Any) -> None:
    """Aggiunge proprietà NTC per muri."""
    props = [
        # Geometria
        ("App::PropertyFloat", "WallThickness", "Geometry", "Spessore [m]"),
        ("App::PropertyFloat", "WallHeight", "Geometry", "Altezza [m]"),
        ("App::PropertyFloat", "WallLength", "Geometry", "Lunghezza [m]"),

        # Materiale NTC 2018
        ("App::PropertyString", "MasonryType", "NTC2018", "Tipo muratura Tab. C8.5.I"),
        ("App::PropertyString", "MortarQuality", "NTC2018", "Qualità malta"),
        ("App::PropertyFloat", "fm", "NTC2018", "Resistenza compressione [MPa]"),
        ("App::PropertyFloat", "tau0", "NTC2018", "Resistenza taglio [MPa]"),
        ("App::PropertyFloat", "fv0", "NTC2018", "Resistenza taglio muratura regolare [MPa]"),
        ("App::PropertyFloat", "E", "NTC2018", "Modulo elastico [MPa]"),
        ("App::PropertyFloat", "G", "NTC2018", "Modulo taglio [MPa]"),
        ("App::PropertyFloat", "w", "NTC2018", "Peso specifico [kN/m³]"),

        # Coefficienti correttivi
        ("App::PropertyBool", "GoodMortar", "NTC2018_Corr", "Malta buona"),
        ("App::PropertyBool", "HasCourses", "NTC2018_Corr", "Ricorsi"),
        ("App::PropertyBool", "HasThroughStones", "NTC2018_Corr", "Diatoni"),
        ("App::PropertyBool", "HasInjections", "NTC2018_Corr", "Iniezioni"),
        ("App::PropertyBool", "HasReinforcedPlaster", "NTC2018_Corr", "Intonaco armato"),
        ("App::PropertyInteger", "ReinforcedPlasterSides", "NTC2018_Corr", "Lati intonaco armato (1-2)"),

        # Risultati analisi
        ("App::PropertyFloat", "DCR", "Results", "Demand/Capacity Ratio"),
        ("App::PropertyString", "FailureMode", "Results", "Modo di rottura"),
        ("App::PropertyBool", "Verified", "Results", "Verifica soddisfatta"),
    ]

    for prop_type, name, group, desc in props:
        if not hasattr(obj, name):
            obj.addProperty(prop_type, name, group, desc)


def _add_structure_ntc_properties(obj: Any) -> None:
    """Aggiunge proprietà NTC per elementi strutturali."""
    props = [
        # Geometria
        ("App::PropertyFloat", "Width", "Geometry", "Larghezza [m]"),
        ("App::PropertyFloat", "Depth", "Geometry", "Profondità [m]"),
        ("App::PropertyFloat", "Height", "Geometry", "Altezza [m]"),
        ("App::PropertyFloat", "Length", "Geometry", "Lunghezza [m]"),

        # Materiale
        ("App::PropertyString", "StructureMaterial", "Material", "Materiale (C.A., Acciaio, Legno)"),
        ("App::PropertyString", "ConcreteClass", "Material", "Classe cls (C25/30, etc.)"),
        ("App::PropertyString", "SteelGrade", "Material", "Classe acciaio (B450C, S235, etc.)"),

        # Armatura (se C.A.)
        ("App::PropertyString", "LongReinf", "Reinforcement", "Armatura longitudinale"),
        ("App::PropertyString", "TransReinf", "Reinforcement", "Armatura trasversale"),
        ("App::PropertyFloat", "Cover", "Reinforcement", "Copriferro [cm]"),
    ]

    for prop_type, name, group, desc in props:
        if not hasattr(obj, name):
            obj.addProperty(prop_type, name, group, desc)


def _add_slab_ntc_properties(obj: Any) -> None:
    """Aggiunge proprietà NTC per solai."""
    props = [
        # Tipologia
        ("App::PropertyString", "SlabType", "Slab", "Tipologia solaio"),
        ("App::PropertyFloat", "Thickness", "Slab", "Spessore totale [m]"),
        ("App::PropertyString", "Direction", "Slab", "Direzione orditura"),
        ("App::PropertyFloat", "JoistSpacing", "Slab", "Interasse travetti [m]"),

        # Rigidezza
        ("App::PropertyString", "StiffnessClass", "Slab", "Classe rigidezza (Rigido, Semirigido, Flessibile)"),
        ("App::PropertyFloat", "ShearModulus", "Slab", "Modulo taglio equivalente [MPa]"),

        # Carichi
        ("App::PropertyFloat", "SelfWeight", "Loads", "Peso proprio [kN/m²]"),
        ("App::PropertyFloat", "G2", "Loads", "Permanenti non strutturali [kN/m²]"),
        ("App::PropertyFloat", "Q", "Loads", "Variabili [kN/m²]"),
        ("App::PropertyString", "UseCategory", "Loads", "Categoria d'uso (A-H)"),
    ]

    for prop_type, name, group, desc in props:
        if not hasattr(obj, name):
            obj.addProperty(prop_type, name, group, desc)


def set_material_properties(
    obj: Any,
    material_type: str,
    mortar_quality: str = "buona",
    corrections: Dict[str, bool] = None
) -> None:
    """
    Imposta proprietà materiale da database NTC 2018.

    Args:
        obj: Oggetto muro
        material_type: Tipo muratura (es. "MATTONI_PIENI")
        mortar_quality: Qualità malta ("scadente", "buona")
        corrections: Coefficienti correttivi
    """
    try:
        from ..ntc2018.materials import get_material_properties
        props = get_material_properties(material_type, mortar_quality)
    except ImportError:
        # Fallback valori default
        props = {
            'fm': 2.4,
            'tau0': 0.060,
            'E': 1500,
            'G': 500,
            'w': 18.0,
        }

    # Imposta proprietà
    if hasattr(obj, 'MasonryType'):
        obj.MasonryType = material_type
    if hasattr(obj, 'MortarQuality'):
        obj.MortarQuality = mortar_quality

    for prop_name in ['fm', 'tau0', 'E', 'G', 'w']:
        if hasattr(obj, prop_name):
            setattr(obj, prop_name, props.get(prop_name, 0))

    # Applica correzioni
    if corrections:
        total_factor = 1.0

        if corrections.get('good_mortar'):
            total_factor *= 1.3
            if hasattr(obj, 'GoodMortar'):
                obj.GoodMortar = True

        if corrections.get('courses'):
            total_factor *= 1.2
            if hasattr(obj, 'HasCourses'):
                obj.HasCourses = True

        if corrections.get('through_stones'):
            total_factor *= 1.2
            if hasattr(obj, 'HasThroughStones'):
                obj.HasThroughStones = True

        if corrections.get('injections'):
            total_factor *= 1.5
            if hasattr(obj, 'HasInjections'):
                obj.HasInjections = True

        if corrections.get('reinforced_plaster'):
            sides = corrections.get('plaster_sides', 1)
            factor = 1.5 if sides == 2 else 1.3
            total_factor *= factor
            if hasattr(obj, 'HasReinforcedPlaster'):
                obj.HasReinforcedPlaster = True
            if hasattr(obj, 'ReinforcedPlasterSides'):
                obj.ReinforcedPlasterSides = sides

        # Limita fattore totale a 1.5
        total_factor = min(total_factor, 1.5)

        # Applica fattore
        if hasattr(obj, 'tau0'):
            obj.tau0 *= total_factor
        if hasattr(obj, 'E'):
            obj.E *= total_factor


def get_wall_geometry(obj: Any) -> Dict[str, float]:
    """
    Estrae geometria da un oggetto muro.

    Args:
        obj: Oggetto muro FreeCAD

    Returns:
        Dizionario con length, height, thickness in metri
    """
    geometry = {
        'length': 0,
        'height': 0,
        'thickness': 0,
    }

    # Prova proprietà dirette
    if hasattr(obj, 'WallLength'):
        geometry['length'] = obj.WallLength
    if hasattr(obj, 'WallHeight'):
        geometry['height'] = obj.WallHeight
    if hasattr(obj, 'WallThickness'):
        geometry['thickness'] = obj.WallThickness

    # Fallback da Shape
    if hasattr(obj, 'Shape') and obj.Shape:
        bb = obj.Shape.BoundBox
        # In mm, converti in m
        dims = sorted([bb.XLength, bb.YLength, bb.ZLength])
        geometry['thickness'] = dims[0] / 1000
        geometry['length'] = dims[1] / 1000
        geometry['height'] = dims[2] / 1000

    # Fallback da proprietà Arch
    if hasattr(obj, 'Width'):
        geometry['thickness'] = float(obj.Width) / 1000
    if hasattr(obj, 'Height'):
        geometry['height'] = float(obj.Height) / 1000
    if hasattr(obj, 'Length'):
        geometry['length'] = float(obj.Length) / 1000

    return geometry


def get_material_properties(obj: Any) -> Dict[str, float]:
    """
    Estrae proprietà materiale da un oggetto muro.

    Args:
        obj: Oggetto muro FreeCAD

    Returns:
        Dizionario con fm, tau0, E, G, w
    """
    props = {
        'fm': 0,
        'tau0': 0,
        'fv0': 0,
        'E': 0,
        'G': 0,
        'w': 0,
    }

    for prop_name in props.keys():
        if hasattr(obj, prop_name):
            props[prop_name] = float(getattr(obj, prop_name))

    return props
