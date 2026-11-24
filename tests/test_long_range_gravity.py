# -*- coding: utf-8 -*-
"""П.0: дальняя гравитация — поле Пуассона на торе (FFT).

Флаг `long_range_gravity` (по умолчанию False: контактный 1-клеточный
шаблон, старый сценарий фаз 1–8 бит-идентичен). С включённым флагом
g = g_контактный + g_Пуассон, где ∇²g = −4πG·(ρ − ρ̄) (k=0 → 0).

П.3.1: в ИСТИННОМ слабом поле (дальнее поле, E-flat) коэффициент
κ = d(rate)/dg следует за диалом TIME_DILATION_GRAV:
- дефолт (0.3) — острая проверка против GPS/GP-B (−70% от ОТО);
- tdg=1.0 — соответствие слабому полю ОТО (κ≈1).
Дефолт модели (0.3) не меняется; tdg=1.0 включается monkeypatch'ом.
"""
import numpy as np
import pytest

import wm.model as model
from wm.model import TIME_DILATION_GRAV, Volume3D


def _run(size=40, T=120, seed=42, long_range=False):
    v = Volume3D(size=size, seed=seed)
    v.entropy_regulator_active = False
    v.long_range_gravity = long_range
    c = size // 2
    v.inject_particle((c, c, c), "n")
    for _ in range(T):
        v.step()
    return v


def _g_profile(v):
    """g(d) вдоль +Z от центра."""
    s = v.size
    c = s // 2
    return {d: float(v.grav[v._flat(c, c, (c + d) % s)]) for d in range(1, s // 2)}


def test_flag_default_off_contact():
    # дефолт: флаг выключен, дальнего поля нет (d>=4 — фоновый пол)
    v = _run(size=40, T=120, long_range=False)
    assert v.long_range_gravity is False
    gp = _g_profile(v)
    for d in (4, 6, 8):
        assert abs(gp[d]) < 5e-4, f"d={d}: g={gp[d]}"


def test_long_range_tail_exists_and_decays():
    # дальнее поле: хвост заметнее фонового пола, монотонно к 0, ~1/r
    v = _run(size=40, T=120, long_range=True)
    gp = _g_profile(v)
    g1, g3, g6, g10 = gp[1], gp[3], gp[6], gp[10]
    assert g6 < -5e-4, f"g(6)={g6}: хвост не выше фонового пола"
    assert g1 < g3 < g6 < g10, "колодец не монотонный"
    assert g10 > -1e-3, "у горизонта тора поле всё ещё глубокое"
    # 1/r: g·r ≈ const в среднем диапазоне (допуск на коррекции тора)
    ratio = (g6 * 6) / (g3 * 3)
    assert 0.4 < ratio < 2.5, f"g·r не константа: ratio={ratio}"
    # 128³-прогон (п.3.1): g·r = const в d=2..S/5 (±10%)
    ratio2 = (gp[8] * 8) / (gp[4] * 4)
    assert 0.7 < ratio2 < 1.4, f"g·r не константа (d=4..8): ratio={ratio2}"


@pytest.mark.parametrize("tdg", [TIME_DILATION_GRAV, 1.0],
                         ids=["dial-default", "tdg-1.0-oto"])
def test_weak_field_kappa_locks_prediction(monkeypatch, tdg):
    """П.3.1: κ в истинном слабом поле (дальнее поле, |g| << 1, E flat)
    следует за диалом TIME_DILATION_GRAV: структура линейна
    (rate ≈ 1 + tdg·g − 0.1E), коэффициент = диал.

    - tdg = дефолт (0.3): ОСТРАЯ проверка — отклонение от ОТО −70%;
    - tdg = 1.0: соответствие слабому полю ОТО (κ ≈ 1) — фиксируется,
      если диал поставить в физическое значение (без правки дефолта;
      бит-закреп фаз 1–8 при этом не меняется: цепочка событий
      lifetime-управляема, проверено).

    Измерение — в области, где хвост энергетического пула спал (E flat):
    иначе ΔE подмешивается в κ (эффективный потенциал модели = tdg·g − 0.1E)."""
    monkeypatch.setattr(model, "TIME_DILATION_GRAV", tdg)
    v = _run(size=40, T=200, long_range=True)
    s = v.size
    c = s // 2
    gs, rs, es = [], [], []
    for d in range(s // 4, s // 4 + 9):      # дальнее поле: d=10..18
        i = v._flat(c, c, c + d)
        gs.append(float(v.grav[i]))
        rs.append(float(v.time_rate[i]))
        es.append(float(v.energy[i]))
    # условия слабого поля: |g| << 1 и E flat (пул не вносит вклад)
    assert max(abs(g) for g in gs) < 0.05
    assert max(es) - min(es) < 5e-4, "E не flat — хвост пула в измерении"
    kappa = float(np.polyfit(np.array(gs), np.array(rs), 1)[0])
    assert abs(kappa - tdg) < 0.1, f"κ={kappa}: κ не следует за диалом {tdg}"
    if tdg < 0.5:  # дефолтный диал: острая предсказуемость против ОТО
        assert 0.25 < kappa < 0.35, f"κ={kappa}"
        assert abs(kappa - TIME_DILATION_GRAV) < 0.05
        assert kappa < 0.6, "κ близко к ОТО (1.0) — предсказание снято"
    else:          # диал в физическом значении: соответствие ОТО
        assert abs(kappa - 1.0) < 0.1, f"κ={kappa}: соответствие ОТО не выполнено"


def test_long_range_deterministic():
    a = _run(size=24, T=60, long_range=True)
    b = _run(size=24, T=60, long_range=True)
    assert np.array_equal(a.grav, b.grav)
    assert np.array_equal(a.time_rate, b.time_rate)
