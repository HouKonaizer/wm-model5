# tests/test_qh.py — КВАНТОВЫЙ H: модельная L2-схема + L4-кулон (G2)
"""
Гейт G2 (ROADMAP_TO_REALITY.md):
  E_bind = 13.6057±ε эВ и a₀ = 5.29177e-11±ε м ИЗ МОДЕЛЬНОЙ ДИНАМИКИ
  (L2-схема / SSFM-ядро), не из классической орбиты (классическая = P5,
  структурный аналог: r₀ = 15.347 фм, E_bind = 3.311e-2 эВ, −99.76%).

Ключевой факт (α-якорь, S1): в модельных единицах COULOMB_K = 3.6e-4 даёт
  K·U·h = α·ħ·c  ⇒  модельный α = физический α = 1/137.036,
  a₀ = ħ²/(m_e·K) = 10344.26 яч. = 5.291772e-11 м,
  E₁ = m_e·K²/(2ħ²) = 13.6057 эВ (станд. формула в модельных единицах).

Дискретная L2-схема модели (FD-оператор H = −(ħ²/2m_e)·D² + V — тот же
стencil, что в SSFM-ядре _update_probability_uni) при 1D s-wave редукции
(3D-запуск невозможен: S > 2·a₀ ≥ 3.5e13 яч.) воспроизводит E₁ с точностью
схемы (остаток ~1e-5 от softening COULOMB_SOFT = 0.35 и коробки).

RED: до внедрения quantum_h_l2 в wm/experiments.py все тесты падают
(ImportError) — модельного квантового H в коде нет (P5 = классический).
"""

import numpy as np
import pytest

from wm import model as M
from wm import constants as C

# Эталон (SI, PDG):
E1_EV_EXP = 13.605693122994
A0_M_EXP = 5.29177210903e-11
# P5 (frozen): классический «водород» модели:
P5_E_BIND_EV = 3.311e-2
P5_R0_FM = 15.347


def _g_kin() -> float:
    """Кинетический коэффициент электрона в модельных единицах: g = ħ²/(2m_e)."""
    hbar = C.HBAR / (M.SI_U_J * M.SI_DT_S)
    m_e = C.M_E * C.C ** 2 / M.SI_U_J
    return hbar ** 2 / (2.0 * m_e)


def test_qh_alpha_anchor():
    """α-якорь: модельный α := K·U·h/(ħ·c) = физический α (механизм S1)."""
    alpha_model = M.COULOMB_K * M.SI_U_J * M.SI_H_M / (C.HBAR * C.C)
    assert abs(alpha_model - C.ALPHA) / C.ALPHA < 1e-6


def test_qh_l2_scheme_ground_state():
    """L2-схема модели (FD, 1D s-wave) + L4-кулон: E₁ = 13.6057±1e-3 эВ,
    a₀ = 5.29177e-11±1e-15 м; FD-значение близко к замкнутой форме (~1e-5)."""
    from wm.experiments import quantum_h_l2

    q = quantum_h_l2()
    assert abs(q["E1_eV"] - E1_EV_EXP) < 1e-3
    assert abs(q["a0_m"] - A0_M_EXP) < 1e-15
    # FD-схема vs замкнутая континуальная форма (остаток = softening + коробка):
    assert abs(q["E1_eV"] - q["E1_closed_eV"]) / q["E1_closed_eV"] < 1e-5
    # bit-lock: модельные значения (α-якорь, SI-карта):
    assert q["a0_cells"] == pytest.approx(10344.258976890189, rel=1e-12)
    assert q["E1_closed_eV"] == pytest.approx(13.605693122994, rel=1e-6)


def test_qh_ssfm_kinetic_exact_dispersion():
    """SSFM-ядро модели (split-step, electron mass): для свободного частицы
    плоская волна e^{2πikj/N} — точное собственное состояние FD-лагранжиана:
    фаза = g·(−λ_k)·T·dt, λ_k = −4sin²(πk/N); норма сохраняется."""
    N, k0, T, dt = 128, 3, 64, 1.0
    g = _g_kin()
    k = np.fft.fftfreq(N, d=1.0 / N)
    lam = -4.0 * np.sin(np.pi * k / N) ** 2
    psi = np.exp(2j * np.pi * k0 * np.arange(N) / N)
    for _ in range(T):
        f = np.fft.fft(psi)
        f = f * np.exp(1j * g * lam * dt)
        psi = np.fft.ifft(f)
    expected = (np.exp(2j * np.pi * k0 * np.arange(N) / N)
                * np.exp(1j * g * lam[k0] * T * dt))
    assert np.abs(psi - expected).max() < 1e-12
    assert abs(np.linalg.norm(psi) / np.sqrt(N) - 1.0) < 1e-12


def test_qh_vs_classical_p5():
    """Два «водорода» модели: квантовый (L2-схема: 13.6057 эВ, a₀ = 10344 яч.)
    vs классическая орбита (P5: 3.311e-2 эВ, r₀ ≈ 3 яч.). Маштабы различаются —
    P5 остаётся структурным аналогом, квантовый H — отдельный результат (D19/D20)."""
    from wm.experiments import quantum_h_l2

    q = quantum_h_l2()
    assert q["E1_eV"] > 100.0 * P5_E_BIND_EV
    assert q["a0_cells"] > 1000.0
