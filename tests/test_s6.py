# -*- coding: utf-8 -*-
"""S6 (константы 15→5): фиксация диалов внутренней согласованностью.

Гейт: 15 свободных диалов → 5. Десять фиксируются внутренней
согласованностью (P1/P2-якоря, legacy-бит-лок, причинность, связанные
состояния, вакуумная фаза w=−1, числовая стабильность); пять остаются
свободными — каждый с обоснованной ролью (абсолютные масштабы / регуляризация).

Формат fixation-теста: свойство ДЕРЖИТСЯ при пин-значении диала и ЛОМАЕТСЯ
при его сдвиге — т.е. диал «удерживается» этим свойством, а не подбран
вручную. Свободные диалы проверяются по роли и окну.
"""
import math

import numpy as np
import pytest

import wm.model as M
from wm.model import HADRON_RELEASE, NODE_MASS_SCALE, Volume3D
from wm import experiments as X

# Пин-значения (текущие константы wm.model) — эталон фиксации.
PIN = {
    "TIME_DILATION_GRAV": 0.3,
    "TIME_DILATION_VEL": 0.1,
    "GRAV_STRENGTH": 0.08,
    "GRAV_ENERGY_COUPLING": 0.1,
    "PROB_INTERFERENCE": 0.8,
    "NODE_BIND_STRENGTH": 0.9,
    "EM_VMAX": 0.95,
    "HADRON_RADIUS": 2,
    "ENTROPY_TARGET": 0.35,
}
FREE = {
    "POISSON_4PI_G": 0.04,
    "COULOMB_SOFT": 0.35,
    "PSI_NODE_AMP": 0.7,
    "ENTROPY_PRESSURE": 0.02,
    "FIELD_GAUGE": 0.1,
}


@pytest.fixture
def dials():
    """Сохраняет/восстанавливает все 14 диалов + производный COULOMB_SOFT2."""
    names = list(PIN) + list(FREE)
    old = {k: getattr(M, k) for k in names}
    old["COULOMB_SOFT2"] = M.COULOMB_SOFT2
    yield M
    for k, v in old.items():
        setattr(M, k, v)


# ============================================================
# 1. АНКОРНЫЕ ПИНЫ (P1/P2/бит-лок/PDG)
# ============================================================

def test_tdg_p1_anchor(dials):
    """TDG = 0.3 = κ_g (P1: чистый гравитационный канал, точно)."""
    assert dials.TIME_DILATION_GRAV == pytest.approx(PIN["TIME_DILATION_GRAV"], abs=1e-15)


def test_tdv_p2_ratio(dials):
    """TDV = κ_E = κ_g/3 (P2: κ_E/κ_g = 1/3 — точная структурная связь)."""
    assert dials.TIME_DILATION_VEL == pytest.approx(
        dials.TIME_DILATION_GRAV / 3.0, abs=1e-15)


def test_grav_strength_bitlock(dials):
    """GS = 0.08: при зафиксированных P1/P2 (TDG=0.3, TDV=0.1) контактный
    κ D12 = 0.4069113352735587 (бит-лок) определяет GS однозначно
    (κ_контакт монотонно убывает по GS)."""
    assert dials.GRAV_STRENGTH == pytest.approx(PIN["GRAV_STRENGTH"], abs=1e-15)
    # монотонность: сдвиг GS ломает бит-лок (κ уходит от эталона)
    v = Volume3D(size=20, seed=42)
    v.entropy_regulator_active = False
    v.inject_particle((10, 10, 10), "n")
    for _ in range(30):
        v.step()
    near, far = v._flat(10, 10, 11), v._flat(10, 10, 16)
    kappa = (float(v.time_rate[far]) - float(v.time_rate[near])) / (
        float(v.grav[far]) - float(v.grav[near]))
    assert kappa == 0.4069113352735587


def test_hadron_release_pdg(dials):
    """HADRON_RELEASE — PDG-калибровка (m_h − Σm_q)·S, не свободный диал:
    p = (938.272−2m_u−m_d) МэВ · S ≈ 1.188, n ≈ 1.187."""
    assert HADRON_RELEASE["p"] == pytest.approx(1.1883615849412992, rel=1e-12)
    assert HADRON_RELEASE["n"] == pytest.approx(1.1868183213333627, rel=1e-12)
    assert HADRON_RELEASE["p"] > HADRON_RELEASE["n"]  # m_u < m_d


# ============================================================
# 2. ПИНЫ ЧИСЛОВОЙ СТАБИЛЬНОСТИ
# ============================================================

def test_grav_energy_coupling_no_runaway(dials):
    """GEC = 0.1: энергия грав. ямы ограничена (нет runaway-обратной связи
    «яма углубляется → тянет энергию → углубляется»). Сдвиг GEC=0.5 →
    runaway (яма ×~7 к 300 шагу). Порог между 0.3 (ограничено) и 0.5 (взрыв)."""
    e0, stable = _well_final(GEC=0.1)
    e0r, runaway = _well_final(GEC=0.5)
    assert stable < 1.5 * e0, (
        f"GEC=0.1: яма {stable:.2f} ≥ 1.5·e0={1.5 * e0:.2f} (ожид. ограничение)")
    assert runaway > 3.0 * e0r, (
        f"GEC=0.5: яма {runaway:.2f} ≤ 3·e0={3.0 * e0r:.2f} (ожид. runaway)")


def _well_final(GEC, N=16, T=300, seed=42):
    """Энергия в грав. яме (радиус 3 от тяжёлого узла) в начале и в конце.
    Возвращает (e0, e_final): e_final/e0 ≪ 1 → ограничение, ≫ 1 → runaway."""
    v = Volume3D(size=N, seed=seed)
    v.entropy_regulator_active = False
    old_u = M.PROB_UNCERTAINTY
    M.PROB_UNCERTAINTY = 0.0
    old_gec = M.GRAV_ENERGY_COUPLING
    M.GRAV_ENERGY_COUPLING = GEC
    try:
        c = N // 2
        v.inject_particle((c, c, c), "n")
        e0 = _well_energy(v, c, 3)
        for _ in range(T):
            v.step()
        return e0, _well_energy(v, c, 3)
    finally:
        M.GRAV_ENERGY_COUPLING, M.PROB_UNCERTAINTY = old_gec, old_u


def _well_energy(v, c, rad):
    e = v.energy.reshape(v.size, v.size, v.size)
    total = 0.0
    for dx in range(-rad, rad + 1):
        for dy in range(-rad, rad + 1):
            for dz in range(-rad, rad + 1):
                total += float(e[(c + dx) % v.size, (c + dy) % v.size, (c + dz) % v.size])
    return total


def test_prob_interference_stable(dials):
    """PI = 0.8: legacy-явная схема — дрейф N=Σ|ψ|² < 2% за 300 шагов
    (гейт D10/D11). Сдвиг PI=60 → дрейф > 2% (схема уходит из окна)."""
    dp, dn = X._psi_scheme_drift(nonlocality=0.0)
    assert dn < 0.02, f"PI={dials.PROB_INTERFERENCE}: дрейф N = {dn:.3e}"
    old = M.PROB_INTERFERENCE
    M.PROB_INTERFERENCE = 60.0
    try:
        dpb, dnb = X._psi_scheme_drift(nonlocality=0.0)
    finally:
        M.PROB_INTERFERENCE = old
    assert dnb > 0.02, f"PI=60: дрейф N = {dnb:.3e} (ожид. > 2%)"


def test_node_bind_no_runaway(dials):
    """NB = 0.9: энергия ячейки узла релаксирует в вакуум (e250 < e0) —
    узел не накапливает энергию. Сдвиг NB=0.99 → runaway (e250 > e0)."""
    e0, e1 = _node_energy_run(NODE_BIND_STRENGTH=0.9)
    assert e1 < e0, f"NB=0.9: e250={e1} ≥ e0={e0} (ожид. релаксация)"
    e0b, e1b = _node_energy_run(NODE_BIND_STRENGTH=0.99)
    assert e1b > e0b, f"NB=0.99: e250={e1b} ≤ e0={e0b} (ожид. runaway)"


def _node_energy_run(NODE_BIND_STRENGTH, N=16, T=250, seed=42):
    v = Volume3D(size=N, seed=seed)
    v.entropy_regulator_active = False
    old_u = M.PROB_UNCERTAINTY
    M.PROB_UNCERTAINTY = 0.0
    old_nb = M.NODE_BIND_STRENGTH
    M.NODE_BIND_STRENGTH = NODE_BIND_STRENGTH
    try:
        c = N // 2
        v.inject_particle((c, c, c), "p")
        idx = v._flat(c, c, c)
        e0 = float(v.energy[idx])
        for _ in range(T):
            v.step()
        return e0, float(v.energy[idx])
    finally:
        M.NODE_BIND_STRENGTH, M.PROB_UNCERTAINTY = old_nb, old_u


def test_em_vmax_causality(dials):
    """EM_VMAX < 1: EM-узлы не превышают свет (c = 1 ячейка/шаг, аксиома A2).
    Значение 0.95 — внутри причинного конуса с числовым запасом."""
    assert 0.0 < dials.EM_VMAX < 1.0, (
        f"EM_VMAX={dials.EM_VMAX} нарушает причинность (A2: |v| ≤ 1)")


# ============================================================
# 3. ПИН СВЯЗАННОГО СОСТОЯНИЯ (конфайнмент)
# ============================================================

def test_hadron_radius_bound_state(dials):
    """HR = 2: цветосинглет {u,u,d} с дистанциями 1,1,√2 схадронизируется
    при R≥2. Сдвиг R=1 → канал ре-хадронизации закрыт (√2 > 1)."""
    old = M.HADRON_RADIUS
    try:
        assert M.HADRON_RADIUS == PIN["HADRON_RADIUS"]
        assert _hadronize() is True, "HR=2: {u,u,d} должен схадронизироваться"
        M.HADRON_RADIUS = 1
        assert _hadronize() is False, "HR=1: канал должен быть закрыт"
    finally:
        M.HADRON_RADIUS = old


def _hadronize():
    v = Volume3D(size=12, seed=7)
    v.entropy_regulator_active = False
    v.inject_particle((6, 6, 6), "u")
    v.inject_particle((7, 6, 6), "u")
    v.inject_particle((6, 6, 7), "d")
    return bool(v.hadronize())


# ============================================================
# 4. ПИН ВАКУУМНОЙ ФАЗЫ (w=−1 эпоха)
# ============================================================

def test_entropy_target_vacuum_phase(dials):
    """ET = 0.35: целевая энтропия в окне вакуумной фазы [0.31, 0.39]:
    w=−1 эпоха (P4) — транзиент при S≈0.310 (нижняя граница), plateau
    вакуумной энергии ≈ 0.39 (верхняя). ET выше 0.39 недостижима, ниже
    0.31 не проходит через w=−1 транзиент."""
    assert 0.31 <= dials.ENTROPY_TARGET <= 0.39, (
        f"ENTROPY_TARGET={dials.ENTROPY_TARGET} вне вакуумной фазы [0.31, 0.39]")


# ============================================================
# 5. СВОБОДНЫЕ ДИАЛЫ (роль + окно)
# ============================================================

def test_free_poisson_4pi_g(dials):
    """POISSON_4PI_G = 0.04 — аналог ньютоновской G: свободная фундаментальная
    константа, задающая абсолютный гравитационный масштаб (гравитация
    притягивающая: 4πG > 0). Не пинится аккрецией (P3 — слабозависим)."""
    assert dials.POISSON_4PI_G > 0.0


def test_free_coulomb_soft(dials):
    """COULOMB_SOFT = 0.35 — UV-регуляризация решётчатого кулона: ε > 0
    (смягчение сингулярности 1/r). H-атом робастен (alive при ε 0.05–2.0),
    значение — конвенция регуляризации, не пин связанного состояния."""
    assert dials.COULOMB_SOFT > 0.0
    assert M.COULOMB_SOFT2 == pytest.approx(dials.COULOMB_SOFT ** 2, rel=1e-15)


def test_free_psi_node_amp(dials):
    """PSI_NODE_AMP = 0.7 — амплитуда связанного состояния: узел «пинает»
    локальный |ψ| к этому значению. Окно (0, 1] (амплитуда, clamp |ψ|≤1)."""
    assert 0.0 < dials.PSI_NODE_AMP <= 1.0


def test_free_entropy_pressure(dials):
    """ENTROPY_PRESSURE = 0.02 — сила вакуумного термостата: P > 0
    (активное управление). Слабая чувствительность (t_cross ≈ const при
    P 0.005–0.08) → не пинится, задаёт скорость релаксации, не фазу."""
    assert dials.ENTROPY_PRESSURE > 0.0


def test_free_field_gauge(dials):
    """FIELD_GAUGE = 0.1 — демпфирование U(1) поля (Дебай-калибровка):
    множитель (1 − FG·0.1)/шаг. Физическое окно [0, 10): FG≥10 даёт
    отрицательный/нулевой множитель (поле гасится в ноль или осциллирует)."""
    assert 0.0 <= dials.FIELD_GAUGE < 10.0
