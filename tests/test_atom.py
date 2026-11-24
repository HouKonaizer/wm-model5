# -*- coding: utf-8 -*-
"""Новая физика: параметры кварков, хадронизация, атом H (Coulomb-орбита),
молекула H2, проверка стабильности. Старый сценарий (фазы 1–8) не трогается —
его бит-идентичность закреплена в tests/test_dynamics.py."""
import math

import numpy as np
import pytest

from wm.model import (
    HADRON_RELEASE,
    TYPE_CODE,
    Volume3D,
)


# ============================================================
# 1. ПАРАМЕТРЫ КВАРКОВ
# ============================================================

def test_d_quark_types_registered():
    """d и d̄ — узловые типы (фермионы, спин 1/2); старые коды не сдвинулись."""
    assert "d" in TYPE_CODE
    assert "d̄" in TYPE_CODE
    assert TYPE_CODE["u"] == 7 and TYPE_CODE["ū"] == 8  # старые коды на месте
    v = Volume3D(size=8, seed=1)
    assert v.inject_particle((3, 3, 3), "d", spin=1)
    assert v.inject_particle((4, 4, 4), "d̄", spin=0)
    # фермион: Pauli-блок для 2-го d↑ в той же ячейке
    assert not v.inject_particle((3, 3, 3), "d", spin=1)


def test_hadron_release_calibrated_to_pdg():
    """Массовый избыток адрона (PDG) в единицах модели: (m_h − Σm_q)·S,
    S = 1.2/938.272 МэВ⁻¹. p: 99% массы = QCD-поле."""
    s = 1.2 / 938.27208816
    assert HADRON_RELEASE["p"] == pytest.approx((938.27208816 - 2 * 2.2 - 4.7) * s, rel=1e-9)
    assert HADRON_RELEASE["n"] == pytest.approx((939.56542052 - 2.2 - 2 * 4.7) * s, rel=1e-9)
    assert 1.1 < HADRON_RELEASE["p"] < 1.25  # ≈929 МэВ в единицах модели


# ============================================================
# 2. ХАДРОНИЗАЦИЯ (uud → p, udd → n)
# ============================================================

def _occupied_codes(v):
    return (set(v.node_t[v.node_t >= 0].tolist())
            | set(v.node_t2[v.node_t2 >= 0].tolist()))


def test_hadronization_uud_to_proton():
    v = Volume3D(size=20, seed=7)
    b = (5, 5, 5)
    v.inject_particle(b, "u", spin=1)
    v.inject_particle((b[0] + 1, b[1], b[2]), "u", spin=0)
    v.inject_particle((b[0], b[1] + 1, b[2]), "d", spin=1)
    e0, l0 = float(v.energy.sum()), v.energy_ledger
    assert v.hadronize()
    codes = _occupied_codes(v)
    assert TYPE_CODE["p"] in codes
    assert TYPE_CODE["u"] not in codes and TYPE_CODE["d"] not in codes
    # Депозит массового избытка: точный ledger
    assert v.energy_ledger - l0 == pytest.approx(HADRON_RELEASE["p"], rel=1e-12)
    assert float(v.energy.sum()) - e0 == pytest.approx(HADRON_RELEASE["p"], rel=1e-9)
    assert any(ev["type"] == "hadronization" and ev["to"] == "p"
               for ev in v.decay_events)


def test_hadronization_udd_to_neutron():
    v = Volume3D(size=20, seed=8)
    b = (6, 6, 6)
    v.inject_particle(b, "u", spin=1)
    v.inject_particle((b[0] + 1, b[1], b[2]), "d", spin=0)
    v.inject_particle((b[0], b[1] + 1, b[2]), "d", spin=1)
    l0 = v.energy_ledger
    assert v.hadronize()
    assert TYPE_CODE["n"] in _occupied_codes(v)
    assert v.energy_ledger - l0 == pytest.approx(HADRON_RELEASE["n"], rel=1e-12)


def test_hadronization_requires_cluster():
    """Кварки дальше HADRON_RADIUS → не цветовой синглет, хадронизации нет."""
    v = Volume3D(size=20, seed=9)
    v.inject_particle((3, 3, 3), "u", spin=1)
    v.inject_particle((10, 3, 3), "u", spin=0)
    v.inject_particle((3, 10, 3), "d", spin=1)
    assert not v.hadronize()
    assert TYPE_CODE["p"] not in _occupied_codes(v)
    assert not any(ev["type"] == "hadronization" for ev in v.decay_events)


def test_hadronization_wrong_flavor_no_event():
    """uud без d (uuu) — не протон: не хадронируется."""
    v = Volume3D(size=20, seed=10)
    b = (5, 5, 5)
    v.inject_particle(b, "u", spin=1)
    v.inject_particle((b[0] + 1, b[1], b[2]), "u", spin=0)
    v.inject_particle((b[0], b[1] + 1, b[2]), "u", spin=1)
    assert not v.hadronize()


# ============================================================
# 3. АТОМ ВОДОРОДА: COULOMB-ОРБИТА И СТАБИЛЬНОСТЬ
# ============================================================

def test_hydrogen_atom_stable():
    """e⁻ на круговой орбите вокруг p: за 250 шагов не падает и не улетает,
    оба узла живы."""
    v = Volume3D(size=20, seed=11)
    v.em_dynamics = True
    h = v.inject_hydrogen((10, 10, 10), r0=3)
    r0 = h["r0"]
    rmin, rmax = float("inf"), 0.0
    for _ in range(250):
        v.step()
        r = v.node_distance(h["p"], h["e"])
        rmin, rmax = min(rmin, r), max(rmax, r)
    assert v.find_node(h["p"]) is not None, "протон исчез"
    assert v.find_node(h["e"]) is not None, "электрон исчез"
    assert rmin >= 0.5 * r0, f"электрон упал в ядро: rmin={rmin:.2f} < {0.5*r0:.2f}"
    assert rmax <= 2.2 * r0, f"электрон улетел: rmax={rmax:.2f} > {2.2*r0:.2f}"


def test_hydrogen_orbit_radius_near_bohr_radius():
    """Средний радиус орбиты близок к r0 (не вырожденная эллипса)."""
    v = Volume3D(size=20, seed=12)
    v.em_dynamics = True
    h = v.inject_hydrogen((10, 10, 10), r0=3)
    rs = []
    for _ in range(120):
        v.step()
        rs.append(v.node_distance(h["p"], h["e"]))
    assert 0.7 * h["r0"] <= float(np.mean(rs)) <= 1.5 * h["r0"]


def test_em_dynamics_off_nodes_do_not_move():
    """Гейт: em_dynamics=False (старый сценарий) — заряженные узлы неподвижны."""
    v = Volume3D(size=20, seed=13)
    assert v.em_dynamics is False
    v.inject_particle((5, 5, 5), "p", spin=0)
    v.inject_particle((8, 5, 5), "e⁻", spin=1)
    for _ in range(50):
        v.step()
    assert v.node_t[ v._flat(5, 5, 5) ] == TYPE_CODE["p"] or \
        v.node_t2[v._flat(5, 5, 5)] == TYPE_CODE["p"]
    assert v.node_t[v._flat(8, 5, 5)] == TYPE_CODE["e⁻"] or \
        v.node_t2[v._flat(8, 5, 5)] == TYPE_CODE["e⁻"]


# ============================================================
# 4. МОЛЕКУЛА H2
# ============================================================

def test_h2_molecule_stable():
    """H2 (оба электрона в середине связи — общая связывающая орбиталь):
    связь p–p не рвётся (dmax ≤ 1.6·d0) и не сливает ядра в одну ячейку
    (dmin ≥ 0.45·d0; в этой модели связь осциллирует в 0.5–1.0·d0),
    электроны остаются в молекуле (≤ 2.5·r0), все 4 узла живы за 250 шагов."""
    v = Volume3D(size=20, seed=14)
    v.em_dynamics = True
    m = v.inject_h2((12, 6, 4), d0=2, r0=3)
    d0, r0 = m["d0"], m["r0"]
    dmin, dmax = float("inf"), 0.0
    er_max = 0.0
    for _ in range(250):
        v.step()
        p1, p2 = v.find_node(m["p1"]), v.find_node(m["p2"])
        e1, e2 = v.find_node(m["e1"]), v.find_node(m["e2"])
        assert None not in (p1, p2, e1, e2), "узел молекулы исчез"
        d = v.torus_distance(p1["pos"], p2["pos"])
        dmin, dmax = min(dmin, d), max(dmax, d)
        for e in (e1, e2):
            er = min(v.torus_distance(e["pos"], p1["pos"]),
                     v.torus_distance(e["pos"], p2["pos"]))
            er_max = max(er_max, er)
    assert dmin >= 0.45 * d0, f"ядра слились: dmin={dmin:.2f} < {0.45*d0:.2f}"
    assert dmax <= 1.6 * d0, f"связь p–p рванулась: dmax={dmax:.2f} > {1.6*d0:.2f}"
    assert er_max <= 2.5 * r0, f"электрон покинул молекулу: {er_max:.2f} > {2.5*r0:.2f}"


# ============================================================
# 5. КОНСЕРВАЦИЯ ЭНЕРГИИ С НОВЫМИ МЕХАНИЗМАМИ
# ============================================================

def test_energy_ledger_with_em_and_hadronization():
    """ΣE = const + явные источники: хадронизация (депозит) и EM-движение
    (не затрагивает слой L1) не нарушают тождество ΔΣE = ledger."""
    v = Volume3D(size=16, seed=19)
    v.em_dynamics = True
    b = (4, 4, 4)
    v.inject_particle(b, "u", spin=1)
    v.inject_particle((b[0] + 1, b[1], b[2]), "u", spin=0)
    v.inject_particle((b[0], b[1] + 1, b[2]), "d", spin=1)
    assert v.hadronize()
    v.inject_hydrogen((10, 6, 8), r0=2)
    e0, l0 = float(v.energy.sum()), v.energy_ledger
    for _ in range(80):
        v.step()
    dE = float(v.energy.sum()) - e0
    led = v.energy_ledger - l0
    scale = max(1.0, abs(e0), abs(led))
    assert abs(dE - led) < 1e-9 * scale, f"|ΔΣE − ledger| = {abs(dE-led):.2e}"
