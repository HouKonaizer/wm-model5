# -*- coding: utf-8 -*-
"""S5 (континуальный предел h→0): D15–D17 из wm/experiments.py.

Честная классификация наблюдаемых:
  - АРТЕФАКТ (схема/конвенция): κ контактного шаблона (нет h-предела),
    IC-смещение массы ямы (σ/√(2π) — клипнутый гаусс), абсолютный ΔM
    аккреции (∝ N³ — топливо = смещение начальных условий, UV-конвенция);
  - ФИЗИКА (сходится): κ слабого поля L3b → g-канал 0.3·(1−0.1·Ē),
    соотношения роста аккреции (конвергенция N=40→80 ≤ 0.5%),
    разделение g/E-каналов κ_eff (регуляризация источника, ε-насыщение).
"""
import math

from wm import experiments as X


def test_kappa_contact_no_h_limit():
    """D15a: κ контактного шаблона НЕ сходится при h→0 — артефакт 1-ячеечной
    матрицы: при N=40 ближняя точка (2 яч.) выходит за пределы шаблона."""
    kc = [X._s5_kappa_run(N) for N in (10, 20, 40)]
    e20 = abs(kc[1]["kappa"] - kc[0]["kappa"]) / abs(kc[0]["kappa"])
    e40 = abs(kc[2]["kappa"] - kc[1]["kappa"]) / abs(kc[1]["kappa"])
    assert e20 < 0.02   # грубые сетки согласованы
    assert e40 > 0.2    # тонкая расходится: шаблон — фиксированный 1-яч. объект


def test_kappa_l3b_converges_g_channel():
    """D15b: κ слабого поля (L3b) сходится к g-каналу 0.3·(1−0.1·Ē);
    остаток = E-канал в остаточном пуле ближней точки (∝ h)."""
    kl = [X._s5_kappa_run(N, lr=True) for N in (20, 40, 80)]
    ebar = 0.5 * (kl[2]["en"] + kl[2]["ef"])
    target = 0.3 * (1.0 - 0.1 * ebar)
    assert abs(kl[2]["kappa"] - target) / target < 0.05
    assert kl[2]["kappa"] < kl[0]["kappa"]   # поправка убывает при h→0


def test_accretion_ic_bias():
    """D16: смещение начальной массы ямы = σ/√(2π) (IC: max(E0+N(0,σ), E0))."""
    a = X._s5_accr_run(20)
    ref = 0.003 / math.sqrt(2.0 * math.pi)
    assert abs(a["bias"] - ref) / ref < 0.15


def test_accretion_growth_resolution():
    """D16: рост аккреции — физика (соотношения растут с разрешением:
    грубая сетка демпфирует транзиент); абсолютный ΔM — IC-конвенция."""
    a20 = X._s5_accr_run(20)
    a40 = X._s5_accr_run(40)
    r20 = a20["M"][250] / a20["M"][200]
    r40 = a40["M"][250] / a40["M"][200]
    assert 1.0 < r40 < 1.25
    assert r40 > r20


def test_source_regularization_channels():
    """D17: κ_eff сглаженного источника выше точечного (E-канал компенсирует
    g-канал); насыщение при ε ≥ 2 ячеек."""
    kn = X._s5_kappa_run(20, src="node")
    b8 = X._s5_kappa_run(20, src="blob8")
    b27 = X._s5_kappa_run(20, src="blob27")
    assert b27["kappa"] - kn["kappa"] > 0.4
    assert abs(b8["kappa"] - b27["kappa"]) < 0.05
