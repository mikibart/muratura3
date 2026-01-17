# -*- coding: utf-8 -*-
"""
Muratura BIM Module

Integrazione con FreeCAD Arch workbench per modellazione BIM.
Usa esclusivamente oggetti Arch (Wall, Window, Structure, Floor).
"""

from .arch_wrapper import (
    create_wall,
    create_window,
    create_door,
    create_structure,
    create_floor,
    create_roof,
    import_dxf,
    import_ifc,
)

from .properties import (
    add_ntc_properties,
    set_material_properties,
    get_wall_geometry,
)

__all__ = [
    'create_wall',
    'create_window',
    'create_door',
    'create_structure',
    'create_floor',
    'create_roof',
    'import_dxf',
    'import_ifc',
    'add_ntc_properties',
    'set_material_properties',
    'get_wall_geometry',
]
