# -*- coding: utf-8 -*-
"""S2: единое ядро (unified_action, DESIGN_S2.md).

Гейты:
1. legacy-бит-закреп: κ D12 = 0.4069113352735587 (все 76 сверок не тронулись);
2. F1: единый масштаб массы — m_dyn := M_ledger (протон = 1.2 ед.);
3. Нётер структурно: N = Σ|ψ|² и решёточный p в SSFM-ядре — машинная точность;
4. H-атом стабилен в unified-ядре (m_dyn = 1.2-масштаб, v_circ = v_leg/√1.2);
5. ledger: ΔΣE(T) = ledger(T) в unified-ядре (аналог D9).
"""
import math

import numpy as np
import pytest

import wm.model as M
from wm.model import _NODE_MEFF, _NODE_MEFF_UNI, NODE_MASS_SCALE, Volume3D


# ============================================================
# 1. LEGACY БИТ-ЗАКРЕП (новый флаг не тронул старое ядро)
# ============================================================

def test_legacy_bit_lock_kappa_d12():
    """κ D12 (GPS-коэффициент слабого поля) — бит-в-бит с до-S2 значением."""
    v = Volume3D(size=20, seed=42)
    v.entropy_regulator_active = False
    v.inject_particle((10, 10, 10), "n")
    for _ in range(30):
        v.step()
    near, far = v._flat(10, 10, 11), v._flat(10, 10, 16)
    gn, gf = float(v.grav[near]), float(v.grav[far])
    rn, rf = float(v.time_rate[near]), float(v.time_rate[far])
    kappa = (rf - rn) / (gf - gn)
    assert kappa == 0.4069113352735587


def test_unified_flag_default_off():
    """По умолчанию legacy-ядро: флаг False, масса = legacy-таблица."""
    v = Volume3D(size=8, seed=1)
    assert v.unified_action is False
    assert v._mef is _NODE_MEFF


# ============================================================
# 2. F1: ЕДИНЫЙ МАСШТАБ МАССЫ (m_dyn := M_ledger)
# ============================================================

def test_mass_unification_scale():
    """Unified-масса = ledger-масса: p = 1.2 ед., e⁻ = 6.535e-4;
    для всех частиц: m_uni = 1.2·m_legacy (оба масштаба ∝ mass_eV)."""
    assert _NODE_MEFF_UNI["p"] == pytest.approx(1.2, rel=1e-12)
    assert _NODE_MEFF_UNI["e⁻"] == pytest.approx(0.51099895e6 * NODE_MASS_SCALE, rel=1e-9)
    for sym in _NODE_MEFF:
        if _NODE_MEFF[sym] > 0.0:
            assert _NODE_MEFF_UNI[sym] == pytest.approx(1.2 * _NODE_MEFF[sym], rel=1e-12)
        else:
            assert _NODE_MEFF_UNI[sym] == 0.0


def test_unified_kernel_uses_ledger_mass():
    v = Volume3D(size=8, seed=1, unified_action=True)
    assert v.unified_action is True
    assert v._mef is _NODE_MEFF_UNI
    # v_circ H-атома в unified-ядре = v_legacy/√1.2 (m в 1.2 раза больше)
    r0 = 3.0
    v_leg = Volume3D(size=20, seed=1)
    v_uni = Volume3D(size=20, seed=1, unified_action=True)
    v_uni.em_dynamics = True
    v_leg.em_dynamics = True
    # скорость круговой орбиты: m_uni = 1.2·m_leg → v_uni = v_leg/√1.2
    v_leg_v = v_leg._orbital_velocity(r0)
    v_uni_v = v_uni._orbital_velocity(r0)
    assert v_uni_v == pytest.approx(v_leg_v / math.sqrt(1.2), rel=1e-12)


# ============================================================
# 3. НЁТЕР СТРУКТУРНО: SSFM-ЯДРО (N и p — машинная точность)
# ============================================================

def _free_psi_setup(v):
    """Свободное L2: плоская волна, V = const, без узлов/полей."""
    v.entropy_regulator_active = False
    v.energy.fill(M.ENERGY_ZERO_POINT)
    v.grav.fill(0.0)
    v.field.fill(0.0)
    v.charge.fill(0.0)
    x = np.arange(v.size, dtype=float)
    qx = 2.0 * np.pi / v.size
    v.prob_real[:] = np.broadcast_to(
        0.5 * np.cos(qx * x)[:, None, None], (v.size,) * 3).reshape(-1)
    v.prob_imag[:] = np.broadcast_to(
        0.5 * np.sin(qx * x)[:, None, None], (v.size,) * 3).reshape(-1)


def test_unified_unitarity():
    """N = Σ|ψ|² консервативно до машинной точности (300 шагов SSFM-ядра):
    Нётер U(1) — структурное свойство ядра, а не O(dt²) явной схемы."""
    v = Volume3D(size=20, seed=3, unified_action=True)
    old_u, old_nl = M.PROB_UNCERTAINTY, M.PROB_NONLOCALITY
    M.PROB_UNCERTAINTY, M.PROB_NONLOCALITY = 0.0, 0.0
    try:
        _free_psi_setup(v)
        n0 = float(np.sum(v.prob_real ** 2 + v.prob_imag ** 2))
        for _ in range(300):
            v._update_probability_uni()
        n1 = float(np.sum(v.prob_real ** 2 + v.prob_imag ** 2))
        drift = abs(n1 - n0) / n0
    finally:
        M.PROB_UNCERTAINTY, M.PROB_NONLOCALITY = old_u, old_nl
    assert drift < 1e-12, f"дрейф N = {drift:.2e}"


def test_unified_momentum_conserved():
    """Решёточный p_x = Im∫ψ*∂xψ консервативен до машинной точности
    (инвариантность относительно сдвигов пространства)."""
    v = Volume3D(size=20, seed=4, unified_action=True)
    old_u, old_nl = M.PROB_UNCERTAINTY, M.PROB_NONLOCALITY
    M.PROB_UNCERTAINTY, M.PROB_NONLOCALITY = 0.0, 0.0

    def p_x():
        r = v.prob_real.reshape(v.size, v.size, v.size)
        s = v.prob_imag.reshape(v.size, v.size, v.size)
        gr = (np.roll(r, -1, 0) - np.roll(r, 1, 0)) * 0.5
        gs = (np.roll(s, -1, 0) - np.roll(s, 1, 0)) * 0.5
        return float(np.sum(r * gs - s * gr))

    try:
        _free_psi_setup(v)
        p0 = p_x()
        assert p0 != 0.0
        for _ in range(300):
            v._update_probability_uni()
        drift = abs(p_x() - p0) / abs(p0)
    finally:
        M.PROB_UNCERTAINTY, M.PROB_NONLOCALITY = old_u, old_nl
    assert drift < 1e-12, f"дрейф p = {drift:.2e}"


# ============================================================
# 4. H-АТОМ В UNIFIED-ЯДРЕ (F1: масса 1.2-масштаба)
# ============================================================

def test_hydrogen_atom_stable_unified():
    """e⁻ на круговой орбите (m_dyn = 6.535e-4 ед.): за 250 шагов не падает
    и не улетает (те же допуски, что у legacy-теста)."""
    v = Volume3D(size=20, seed=11, unified_action=True)
    v.em_dynamics = True
    h = v.inject_hydrogen((10, 10, 10), r0=3)
    rmin, rmax = float("inf"), 0.0
    for _ in range(250):
        v.step()
        r = v.node_distance(h["p"], h["e"])
        rmin, rmax = min(rmin, r), max(rmax, r)
    assert v.find_node(h["p"]) is not None, "протон исчез"
    assert v.find_node(h["e"]) is not None, "электрон исчез"
    assert rmin >= 0.5 * h["r0"], f"электрон упал: rmin={rmin:.2f}"
    assert rmax <= 2.2 * h["r0"], f"электрон улетел: rmax={rmax:.2f}"


# ============================================================
# 5. LEDGER В UNIFIED-ЯДРЕ (аналог D9: ΔΣE = ledger)
# ============================================================

def test_energy_ledger_unified():
    """Полные шаги unified-ядра с источниками, регулятором, силой и впрыском:
    |ΔΣE(T) − ledger(T)| < 1e-9·scale."""
    v = Volume3D(size=16, seed=19, unified_action=True)
    v.inject_particle((4, 6, 4), "n")
    v.inject_particle((9, 6, 7), "n")
    v.inject_particle((6, 10, 6), "n")
    v.inject_particle((3, 3, 3), "τ⁻")
    v.inject_particle((11, 11, 11), "μ⁻")
    v.inject_particle((5, 12, 5), "e⁻")
    v.inject_quark_pair((8, 4, 8), dist=3)
    e0, l0 = float(v.energy.sum()), v.energy_ledger
    for t in range(200):
        if t == 50:
            v.apply_force((8, 8, 8), 0.3)
        if t == 100:
            v.step(external_injection={"pos": (5, 5, 5), "energy": 2.0,
                                       "charge": 0.5, "radius": 2})
            continue
        v.step()
    dE = float(v.energy.sum()) - e0
    led = v.energy_ledger - l0
    scale = max(1.0, abs(e0), abs(led))
    assert abs(dE - led) < 1e-9 * scale, f"|ΔΣE − ledger| = {abs(dE - led):.2e}"
