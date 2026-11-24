# -*- coding: utf-8 -*-
"""Тесты wm.model: smoke, параметризация сценария, валидация."""
import numpy as np
import pytest

from wm.model import GRID_SIZE, Volume3D, run_simulation


def test_smoke_small_grid():
    v = Volume3D(size=8, seed=1)
    for _ in range(20):
        v.step()
    for arr in (v.time_rate, v.energy, v.psi_abs, v.grav, v.field):
        assert np.all(np.isfinite(arr)), "не-конечные значения в слоях"
    assert v.total_energy >= 0
    assert v.matter_count >= 0


def test_run_simulation_returns_11_frames():
    res = run_simulation(size=8, quiet=True)
    assert set(res) == {"world", "lab", "frames"}
    assert len(res["frames"]) == 11
    # Основной мир: фазы 1–8 = 20+30+40+10+70+40+100+500 = 810 (бит-идентично).
    assert res["world"].time_step == 810
    assert res["world"].size == 8
    # Лаборатория (size=8): H2 в чистом вакууме живёт все 250 шагов (фаза 11).
    assert res["lab"].time_step == 250


def test_default_size_is_grid_size():
    assert GRID_SIZE == 20
    v = Volume3D(seed=1)
    assert v.size == 20


def test_run_simulation_size_validation():
    with pytest.raises(ValueError):
        run_simulation(size=2)
