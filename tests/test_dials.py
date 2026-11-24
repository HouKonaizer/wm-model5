# -*- coding: utf-8 -*-
"""G4: вывод ≥1 из 5 свободных диалов из структуры решётки.

История TDD: RED — `dial_structure_audit()` в wm/experiments не существует
(ImportError). Аудит механизмов (`wm/experiments.dial_structure_audit`)
показал, что ни один из 5 диалов {POISSON_4PI_G, COULOMB_SOFT, PSI_NODE_AMP,
ENTROPY_PRESSURE, FIELD_GAUGE} не выводится из структуры + якорей:
  1. POISSON_4PI_G = 0.04: κ D12 (бит-лок 0.4069113352735587, якорь P1)
     scale-инвариантен к 4πG (идентичен при 4πG = 0.01/0.04/0.16); физический
     4πG = 2.54e-39 модельных ед. (G1) — якоря нет. Роль: абсолютный
     гравитационный масштаб (toy).
  2. COULOMB_SOFT = 0.35: E₁ (α-якорь, D19) нечувствителен к ε (spread 1e-6
     rel при ε = 0.05/0.35/2.0). Роль: UV-регуляризация кулона.
  3. PSI_NODE_AMP = 0.7: порог распада (синтез < 0.15) не выделяет значение —
     маржа ≥ 2.1× на всём окне (0, 1]; 0.7 ≈ 1/√2 (1.005%) — нумерология.
     Роль: амплитуда pin |ψ| связанного состояния.
  4. ENTROPY_PRESSURE = 0.02: плато вакуума (S ≈ 0.310, P4) не зависит от
     rates (spread 3e-4 при EP = 0.005/0.02/0.08). Роль: скорость термостата.
  5. FIELD_GAUGE = 0.1: закон Дебая 0.999^T держит hardcoded charge *= 0.999
     (dev 1e-15 при FG = 0.01/0.1/0.5). Совпадение FG = κ_E = 0.1 — с
     зафиксированным диалом (P2), без связки в коде и механизма → нумерология.
     Роль: демпфирование U(1) поля.

KILL выполнен: все 5 диалов остаются свободными с ролью (status quo S6,
EQUATIONS.md §12, §5 THEORY); счётчик свободных диалов 5 → 5.
Эти тесты закрепляют отрицательный результат (регрессия kill).
"""
import pytest

from wm import experiments as X


@pytest.fixture(scope="module")
def audit():
    return X.dial_structure_audit()


def test_dials_no_structural_match(audit):
    """Ни один диал не совпадает с чистым структурным/якорным числом (1%)."""
    for dial in ("POISSON_4PI_G", "COULOMB_SOFT", "PSI_NODE_AMP",
                 "ENTROPY_PRESSURE", "FIELD_GAUGE"):
        name, dev = audit["min_dev"][dial]
        assert dev > 0.01, f"{dial} совпал со структурным {name}: dev = {dev:.3%}"
    assert audit["no_match"]


def test_poisson_g_kappa_scale_invariant(audit):
    """κ D12 (бит-лок, якорь P1) не зависит от POISSON_4PI_G: long_range off
    по умолчанию, а κ = d(rate)/dΦ — ratio, инвариантный к масштабу G."""
    p = audit["poisson"]
    for k in p["kappas"]:
        assert k == 0.4069113352735587
    assert p["invariant"]


def test_coulomb_soft_alpha_invariant(audit):
    """E₁ (α-якорь, D19) нечувствителен к COULOMB_SOFT: вес r < ε в trial
    u = r·e^(−r/a₀) пренебрежимо мал (u(0) = 0)."""
    c = audit["coulomb"]
    e1 = c["E1_eV"]
    spread = (max(e1) - min(e1)) / (sum(e1) / len(e1))
    assert spread < 1e-5, f"E₁ зависит от ε: spread = {spread:.2e}"
    assert c["insensitive"]


def test_psi_node_amp_no_threshold_selection(audit):
    """Порог распада (синтез < 0.15) не выделяет PSI_NODE_AMP: маржа ≥ 1.5×
    на всём окне (0, 1], синтез монотонен по a (нет внутреннего экстремума)."""
    p = audit["psi_amp"]
    syns = p["synthesis"]
    for s in syns:
        assert s >= 1.5 * 0.15
    for s1, s2 in zip(syns, syns[1:]):
        assert s2 > s1
    assert p["no_selection"]


def test_entropy_pressure_rate_free(audit):
    """Плато вакуума (S ≈ 0.310, P4) не зависит от ENTROPY_PRESSURE (rate):
    spread < 5e-3 при EP = 0.005/0.02/0.08."""
    e = audit["entropy"]
    spread = max(e["S"]) - min(e["S"])
    assert spread < 5e-3, f"плато зависит от EP: spread = {spread:.2e}"
    assert all(abs(s - 0.310) < 0.03 for s in e["S"])  # эпоха P4: S ≈ 0.310
    assert e["rate_free"]


def test_field_gauge_debye_invariant(audit):
    """Закон Дебая 0.999^T держит hardcoded charge *= 0.999 (model._update_fields),
    не FIELD_GAUGE: dev < 1e-12 при FG = 0.01/0.1/0.5."""
    f = audit["field_gauge"]
    for d in f["debye_dev"]:
        assert d < 1e-12
    assert f["invariant"]


def test_coincidences_documented(audit):
    """Точные численные совпадения диалов задокументированы как нумерология
    (без механизма и связки в коде): FG = κ_E = 0.1; CS = ENTROPY_TARGET = 0.35;
    PNA = 2·CS = 0.7; PNA ≈ 1/√2 (1.005%)."""
    keys = {(c[0], c[1]) for c in audit["coincidences"]}
    assert ("FIELD_GAUGE", "kappa_E") in keys
    assert ("COULOMB_SOFT", "ENTROPY_TARGET") in keys
    assert ("PSI_NODE_AMP", "1/sqrt(2)") in keys
    for c in audit["coincidences"]:
        assert c[2] is False  # mechanism = False везде


def test_dials_remain_free(audit):
    """KILL G4: вывод не найден — все 5 диалов остаются свободными с ролью."""
    assert audit["conclusion"] == "free"
