# -*- coding: utf-8 -*-
"""Совместимый shim: код перенесён в пакет wm/model.py (Option 5).

`import model` даёт прежний API (Volume3D, GRID_SIZE, run_simulation, ...) —
имена делегируются в wm.model (PEP 562), т.е. всегда актуальны.
`python model.py` — полный сценарий; snapshots.json + viewer.html в корне проекта
(здесь же шаблон viewer_template.html).
"""
import os

from wm import model as _wm_model

__all__ = [n for n in dir(_wm_model) if not n.startswith("_")]


def __getattr__(name):
    return getattr(_wm_model, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_wm_model)))


if __name__ == "__main__":
    import sys

    _wm_model.main(out_dir=os.path.dirname(os.path.abspath(__file__)),
                   animate="--animate" in sys.argv)
