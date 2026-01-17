# masonry_fem_engine/analyses/frame/__init__.py
"""
Modulo Frame (Telaio Equivalente)
"""

try:
    from .element import FrameElement
    from .model import EquivalentFrame, _analyze_frame as analyze_frame
except ImportError as e:
    import logging
    logging.warning(f"Frame module import error: {e}")
    FrameElement = None
    EquivalentFrame = None
    analyze_frame = None

__all__ = ['FrameElement', 'EquivalentFrame', 'analyze_frame']