# -*- coding: utf-8 -*-
"""G3 (микросектор 2b): вывод β_g = 1/κ ∈ [2.46, 3.33] из структуры + якорей.

История TDD: RED — `beta_g_from_structure()` (β_g из структуры/якорей без
подгонки, в окне [2.46, 3.33]) — не существует (ImportError). Механизм-аудит
(`wm/experiments.bg_structure_audit`) показал, что вывода нет; KILL выполнен:
β_g = конвенция (третий «якорь» карты — по определению, не предсказание).

Эти тесты ЗАКРЕПЛЯЮТ отрицательный результат (регрессия kill):
  1. окно [2.46, 3.33] ⊂ [1/κ_eff, 1/κ_g] — таутология (независимого
     определения β_g нет);
  2. ψ-часы (V = 2·grav + E, SSFM-ядро): абсолютный сдвиг 2·Δg универсален,
     но относительный сдвиг режимозависим → универсального κ_ψ нет;
  3. карта потенциала: β_map = 4πG_model/4πG_phys = 1.57e37 — граница G1,
     вне окна (4πG_phys = 2.54e-39 модельных ед., кросс-чек: Земля);
  4. κ_g = 0.3 — свободный диал: не совпадает ни с одним безразмерным
     числом модели/якорей (α, S0, g_e, PDG-отношения, ψ-коэффициент 2, a0/h).
PREDICTIONS.md P1 не тронут: в естественном прочтении (β_g = 1) P1 остаётся
фальсифицированной (κ_eff = 0.407 vs GR 1.0; D12: −59.3%).
"""
import math

from wm.experiments import KAPPA_EFF_D12, bg_structure_audit


def test_bg_window_is_tautology():
    """Окно G3 [2.46, 3.33] — срез таутологического интервала [1/κ_eff, 1/κ_g]."""
    a = bg_structure_audit()
    w = a["window"]
    assert w["k_g"] == 0.3                      # TIME_DILATION_GRAV (live)
    assert w["k_eff"] == KAPPA_EFF_D12          # D12 бит-закреп
    assert abs(w["1/k_g"] - 1.0 / 0.3) < 1e-12
    assert abs(w["1/k_eff"] - 1.0 / 0.4069113352735587) < 1e-12
    assert abs(w["1/k_eff"] - 2.457538) < 1e-5  # 1/κ_eff = 2.45754
    assert w["tautology"]
    assert w["lo"] > w["1/k_eff"] and w["hi"] < w["1/k_g"]
    assert a["conclusion"] == "convention"


def test_bg_psi_clock_mode_dependent():
    """ψ-поле (V = 2·grav + E) — не универсальные часы: сдвиг зависит от режима."""
    a = bg_structure_audit()["psi"]
    g_a, g_b = a["g"]
    # Абсолютный сдвиг частоты = 2·Δg — универсален (структурный коэфф. 2 ядра).
    assert abs(a["abs_shift"] - 2.0 * abs(g_a - g_b)) < 1e-15
    # Относительный сдвиг (красное смещение) различается по режимам:
    r1, r2 = a["ratio"][a["k"][0]], a["ratio"][a["k"][1]]
    assert abs(r1 - r2) > 0.1
    assert a["mode_dependent"]


def test_bg_potential_map_excluded():
    """β_map = 4πG_model/4πG_phys = 1.57e37 (G1) — вне окна [2.46, 3.33]."""
    pm = bg_structure_audit()["potential_map"]
    # Физ. 4πG в модельных ед. (масса = U/c², длина = h, время = dt):
    assert 2.5e-39 < pm["4pi_G_phys_model"] < 2.6e-39
    assert 1.5e37 < pm["beta_map"] < 1.6e37
    assert not (2.46 <= pm["beta_map"] <= 3.33)
    # Кросс-чек: потенциал Земли g(R) = GM/(R·c²) = 6.96e-10 (измеренное).
    assert pm["earth_rel"] < 1e-3


def test_bg_kappa_g_is_free_dial():
    """κ_g = 0.3 не совпадает ни с одним безразмерным числом без диалов."""
    a = bg_structure_audit()
    k_g = a["window"]["k_g"]
    assert k_g == 0.3
    for name, v in a["structural"].items():
        assert abs(k_g - v) / abs(v) > 0.01, f"κ_g совпал со структурным {name} = {v}"
    assert a["kappa_differs"]
