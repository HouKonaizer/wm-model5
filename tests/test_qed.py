# -*- coding: utf-8 -*-
"""G5 (QED-прецизия, 2026-11-23): g-2 3-loop QED-ряд + верифицированные
SM-поправки R_π/R_K (RC-факторы) + верифицированный QED-ряд p-Ps.

Верифицированные источники (web_fetch 2026-11-23, НЕ из памяти):
  - C1 = 1/2 (Schwinger 1948), C2 = −0.32848, C3 = +1.184175:
    ru.wikipedia «Аномальный магнитный момент» (Physika Mikro Mira 1980).
  - R_π^SM = 1.2352(1)e-4 (e/μ), R_K^SM = 2.477(1)e-5 (e/μ):
    PRL 99, 231801 (2007), arXiv:0707.3439 (Cirigliano & Rosell) — SM = tree + RC.
  - R_K(KLOE) = 2.493(31)e-5 (e/μ): EPJC 64:627 (2009), arXiv:0907.3594.
  - p-Ps: Γ = Γ_LO·[1 − (5−π²/4)(α/π) + 2α²ln(1/α) + B_p(α/π)²
    − (3α³/2π)ln²(1/α) + ...], B_p = 5.1(3), Γ_SM = 7989.64(2) μs⁻¹
    → τ_SM = 125.163(4) пс: hep-ph/9910488 (Czarnecki–Melnikov–Yelkhovsky 1999).
  - 21 см: полный tree НЕ верифицирован онлайн → HOLD (только note).
"""
import math

from wm import experiments as X


def _checks(cat=None, sub=None):
    """Запустить нужные блоки и вернуть совпавшие чеки."""
    X.check_em()
    X.check_decays()
    out = []
    for c in X.CHECKS:
        if cat is not None and c.cat != cat:
            continue
        if sub is not None and sub not in c.name:
            continue
        out.append(c)
    return out


def _g2():
    cs = _checks(cat="ЭМ", sub="a_e")
    assert cs, "чек a_e не найден"
    return cs[-1]


def _r_pi():
    cs = _checks(cat="РАСП", sub="R_π")
    assert cs
    return cs[-1]


def _r_k():
    cs = _checks(cat="РАСП", sub="R_K = Γ")
    assert cs
    return cs[-1]


def _p_ps():
    cs = _checks(cat="РАСП", sub="парапозитрония")
    assert cs
    return cs[-1]


def _21cm():
    cs = _checks(cat="ЭМ", sub="21 см")
    assert cs
    return cs[-1]


# --- g-2 (3-loop) ---

def test_g2_3loop_value():
    """3-loop QED-ряд: a_e = (α/π)·[1/2 − 0.32848·(α/π) + 1.184175·(α/π)²]."""
    c = _g2()
    ap = X.ALPHA / math.pi
    ref = ap * (0.5 - 0.32848 * ap + 1.184175 * ap ** 2)
    assert abs(c.v_model - ref) / ref < 1e-12


def test_g2_3loop_dev_vs_exp():
    """Отклонение vs PDG (1.15965218059e-3) ~1e-7 (1 петля была +1.5e-3)."""
    c = _g2()
    assert c.kind == "prediction"
    assert abs(c.dev) < 1e-6


def test_g2_3loop_better_than_1loop():
    """3-loop хотя бы на 2 порядка точнее 1-петлённого α/(2π)."""
    c = _g2()
    a1 = X.ALPHA / (2.0 * math.pi)
    dev1 = abs(a1 - c.v_exp) / c.v_exp
    assert abs(c.dev) < dev1 / 100


def test_g2_not_1loop():
    """Значение чекa — НЕ 1-петлёное α/(2π) (G5 действительно применён)."""
    c = _g2()
    a1 = X.ALPHA / (2.0 * math.pi)
    assert abs(c.v_model - a1) / a1 > 1e-4


# --- R_π: SM = tree + верифицированный RC-фактор (PRL 99 231801) ---

def test_r_pi_sm_value():
    """v_model = SM 1.2352e-4 (e/μ) → 8095.9 (μ/e), НЕ tree 7792.2."""
    c = _r_pi()
    assert abs(c.v_model - 1.0 / 1.2352e-4) / (1.0 / 1.2352e-4) < 1e-9
    assert c.kind == "prediction"


def test_r_pi_sm_dev_vs_exp():
    """dev ≈ −0.18% (на tree было −3.93%)."""
    c = _r_pi()
    assert abs(c.dev) < 5e-3


def test_r_pi_verified_note():
    """note ссылается на верифицированный SM-источник и сохраняет tree."""
    c = _r_pi()
    n = c.note
    assert "1.2352" in n and "0707.3439" in n
    assert "7792" in n  # tree модели сохранён (честность)


# --- R_K: SM = tree + RC; эксп. якорь = KLOE 2009 ---

def test_r_k_sm_value():
    """v_model = SM 2.477e-5 (e/μ) → 40371.4 (μ/e), НЕ tree 38926.2."""
    c = _r_k()
    assert abs(c.v_model - 1.0 / 2.477e-5) / (1.0 / 2.477e-5) < 1e-9


def test_r_k_exp_anchor_kloe():
    """Эксп. якорь обновлён: KLOE R_K = 2.493e-5 (e/μ) (arXiv:0907.3594)."""
    c = _r_k()
    assert abs(c.v_exp - 1.0 / 2.493e-5) / (1.0 / 2.493e-5) < 1e-9


def test_r_k_sm_dev_vs_exp():
    """dev ≈ +0.65% (SM выше KLOE на ~0.5σ — согласовано)."""
    c = _r_k()
    assert 0.0 < c.dev < 0.01


def test_r_k_verified_note():
    """note ссылается на верифицированные SM и KLOE источники."""
    c = _r_k()
    n = c.note
    assert "2.477" in n and "0707.3439" in n and "KLOE" in n


# --- p-Ps: верифицированный QED-ряд O(α)...O(α³ ln²(1/α)) ---

def test_pps_sm_value():
    """v_model = τ_SM = 1/7989.64 μs⁻¹ = 125.162 пс (hep-ph/9910488)."""
    c = _p_ps()
    assert abs(c.v_model - 1.0 / 7989.64e6) / (1.0 / 7989.64e6) < 1e-9
    assert c.kind == "prediction"  # был correspondence → prediction (QED-ряд)


def test_pps_tension_documented():
    """SM (125.16 пс) ВЫШЕ экспа (124.35 пс) на +0.65% — задокументированное
    напряжение, НЕ сходство."""
    c = _p_ps()
    assert 0.005 < c.dev < 0.009
    n = c.note
    assert "9910488" in n and "7989.64" in n and "напряжение" in n.lower()


# --- 21 см: полный tree HOLD ---

def test_21cm_hold():
    """21 см: v_model = LO tree (без изменений); полный tree НЕ включён
    (HOLD — коэффициенты не верифицированы онлайн), честно в note."""
    c = _21cm()
    assert "HOLD" in c.note
    # LO tree: 1420.74 МГц (дipole) / 1421.16 МГц (модельная формула, константы
    # проекта) — в любом случае между 1420e6 и 1422e6, НЕ SM-цитата.
    assert 1420.0e6 < c.v_model < 1422.0e6
