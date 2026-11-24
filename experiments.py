# -*- coding: utf-8 -*-
"""Совместимый shim: код перенесён в пакет wm/experiments.py (Option 5).

`import experiments` даёт прежний API (Check, CHECKS, run_all, main, check_*) —
имена делегируются в wm.experiments (PEP 562), т.е. всегда актуальны
(CHECKS пересоздаётся run_all() — shim отражает свежий список).
`python experiments.py` — таблица 76 сверок + verification.json в корне проекта.
"""
import os

from wm import experiments as _wm_experiments

__all__ = [n for n in dir(_wm_experiments) if not n.startswith("_")]


def __getattr__(name):
    return getattr(_wm_experiments, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_wm_experiments)))


if __name__ == "__main__":
    _wm_experiments.main(out_dir=os.path.dirname(os.path.abspath(__file__)))
