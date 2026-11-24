# -*- coding: utf-8 -*-
"""Вектор 1 (G4+): вывод диалов из ДИНАМИКИ — аудит оставшихся механизмов.

История TDD: RED — `dial_vector1_audit()` не существует в wm/experiments
(ImportError). G4 (S6) закрыл кандидатов «структура + якоря» для всех 5
диалов (KILL). Вектор 1 (2026-11-24, timebox + kill-критерий) закрывает
остальные *динамические* кандидаты — условия, которые динамика самой
модели могла бы наложить на значения диалов. Условия зафиксированы ДО
вычисления чисел:

1. PSI_NODE_AMP — условие баланса рождения/распада узла: фиксированная
   точка популяции узлов зафиксировала бы амплитуду pin a. В модели
   (model._phase_transitions) узел распадается при syn < 0.15 ИЛИ
   возраст >= TYPE_LIFETIME; для ячейки узла syn(a) = (2/π + 1.2/1.3 + a
   + 0 + 0)/5, так что syn(a) >= 0.312 > 0.15 при любом a >= 0 — распад
   по синтезу недостижим, время жизни чисто возрастное (TYPE_LIFETIME,
   по частицам, не диал) → фиксированной точки для a нет → FREE.
   Проверено аналитически (сетка a) и live (Volume3D, узел "n").
2. FIELD_GAUGE — условие отслеживания поля: поле — фильтр 1-го порядка
   (model._update_fields): f_{t+1} = a_m·f_t + b·target_t, target_t =
   K_F·∇q_t, q_t *= 0.999 (Дебай, hardcoded). Ответ фильтра 1-го порядка
   (реальные коэффициенты, a_m > 0) на экспоненциальный источник —
   скалярный множитель источника: фазы и резонанса нет → «точное
   отслеживание» НЕ выбирает FG: любой FG отслеживает, амплитуда поля —
   свободный масштаб. Проверено численно (диполь, FG ∈ {0.01, 0.1, 0.5}).
3. SI-карта — «K = α·S0» — ТАУТОЛОГИЯ: S0 = dt·U/ħ, dt = h/c,
   h = α·ħ·c/(K·U) (конвенция S1) → S0 = α/K тождественно (K·S0 = α,
   dev < 1e-9). Независимые якоря SI-карты: M_P (→ U = M_P/1.2) и K
   (модельная кулоновская константа); вывод без якорей отсутствует.

Вывод (KILL): ни один из 5 диалов не выводится из динамики — G4
подтверждён, граница зафиксирована жёстче (проверены ВСЕ известные
кандидатные механизмы: 5 у G4 + 3 у Вектора 1 + структура якорей).

Эти тесты ЗАКРЕПЛЯЮТ отрицательный результат (регрессия kill): если
механизм появится (фиксированная точка / резонанс / нетаутологичная
связь) — тесты сломаются.
"""
import numpy as np
import pytest

from wm import experiments as X


@pytest.fixture(scope="module")
def audit():
    return X.dial_vector1_audit()


def test_v1_node_synthesis_threshold_unreachable(audit):
    """Узел: syn(a) >= 0.312 > 0.15 при любом a >= 0 — распад по синтезу недостижим."""
    n = audit["node"]
    assert n["threshold"] == 0.15
    assert n["syn_min_grid"] > 0.31
    assert n["unreachable"] is True


def test_v1_live_node_synthesis_always_above_threshold(audit):
    """Live (Volume3D, узел "n"): syn ячейки узла > 0.15 на всех шагах, узел жив."""
    live = audit["node"]["live"]
    assert live["n_steps"] == 60
    assert live["syn_min"] > 0.15
    assert live["above_threshold"] is True
    assert live["node_alive_after_60"] is True


def test_v1_field_tracking_no_resonance(audit):
    """Поле: отслеживание Дебай-источника без фазы/резонанса:
    отношение сходится (tail_conv < 1e-2), направление стабилизировано
    (dir_corr > 0.999), реальных осцилляций нет (sign_changes <= 1)."""
    f = audit["field"]
    for fg in f["FG"]:
        assert f["sign_changes"][fg] <= 1
        assert f["tail_conv_rel"][fg] < 1e-2
        assert f["dir_corr"][fg] > 0.999
    assert f["no_resonance"] is True


def test_v1_field_gauge_not_selected(audit):
    """Стационарное отношение ||f||/||q|| различно для разных FG — FG не выбирается."""
    f = audit["field"]
    assert f["limit_spread_rel"] > 1e-3
    assert f["not_selected"] is True


def test_v1_k_s0_is_tautology(audit):
    """SI: K·S0 = α тождественно (dev < 1e-9) — перефраз конвенции, не вывод."""
    si = audit["si"]
    assert si["dev"] < 1e-9
    assert si["tautology"] is True


def test_v1_conclusion_free(audit):
    """Вектор 1 KILL: все 5 диалов остаются свободными (G4 подтверждён)."""
    assert audit["conclusion"] == "free"
