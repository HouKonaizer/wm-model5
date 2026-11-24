# -*- coding: utf-8 -*-
"""wm — 5D-решётчатая модель: время, энергия, вероятность, гравитация, поля.

Публичный API:
    Volume3D, GRID_SIZE, run_simulation, export_viewer (model)
    run_all (experiments), load_config (config)
"""
from .model import GRID_SIZE, N_LAYERS, LAYER_NAMES, Volume3D, run_simulation, export_viewer
from .experiments import run_all
from .config import load_config

__version__ = "0.1.0"

__all__ = [
    "GRID_SIZE",
    "N_LAYERS",
    "LAYER_NAMES",
    "Volume3D",
    "run_simulation",
    "export_viewer",
    "run_all",
    "load_config",
    "__version__",
]
