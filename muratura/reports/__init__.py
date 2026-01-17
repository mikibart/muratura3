# -*- coding: utf-8 -*-
"""
Muratura Reports Module

Generazione relazioni di calcolo strutturale.
"""

from .generator import ReportGenerator
from .template_renderer import (
    TemplateRenderer,
    ReportBuilder,
    render_report,
)

__all__ = [
    'ReportGenerator',
    'TemplateRenderer',
    'ReportBuilder',
    'render_report',
]
