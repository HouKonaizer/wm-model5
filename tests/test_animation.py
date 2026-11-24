# -*- coding: utf-8 -*-
"""Анимация лабораторных фаз: лёгкие кадры узлов (кварки → адроны → H → H2)."""
from wm.model import Volume3D, run_simulation

FULL_PHASES = 8
PH9 = 30    # фаза 9: кадр на каждый шаг (30)
PH10 = 125  # фаза 10: 250 шагов, каждый 2-й
PH11 = 125  # фаза 11: 250 шагов, каждый 2-й
TOTAL = FULL_PHASES + PH9 + PH10 + PH11


def test_animate_off_by_default():
    res = run_simulation(size=8, seed=7, quiet=True)
    assert len(res["frames"]) == 11
    assert all(not f.get("light") for f in res["frames"])


def test_animate_frame_stream():
    res = run_simulation(size=8, seed=7, quiet=True, animate=True)
    fr = res["frames"]
    assert len(fr) == TOTAL
    assert all(not f.get("light") for f in fr[:FULL_PHASES])
    assert all(f.get("light") for f in fr[FULL_PHASES:])
    assert all(isinstance(f["data"]["nodes"], list) for f in fr[FULL_PHASES:])


def test_animate_h2_final_nodes():
    res = run_simulation(size=8, seed=7, quiet=True, animate=True)
    last = res["frames"][-1]
    codes = sorted(n[3] for n in last["data"]["nodes"])
    assert codes == [1, 1, 2, 2]  # p, p, e⁻, e⁻
    for x, y, z in (n[:3] for n in last["data"]["nodes"]):
        assert 0 <= x < 8 and 0 <= y < 8 and 0 <= z < 8


def test_animate_h_orbit_moving():
    # Электрон атома H должен двигаться между кадрами (орбита, не статично)
    res = run_simulation(size=8, seed=7, quiet=True, animate=True)
    fr = res["frames"][FULL_PHASES + PH9: FULL_PHASES + PH9 + PH10]
    e_pos = set()
    for f in fr[::10]:
        for n in f["data"]["nodes"]:
            if n[3] == 2:
                e_pos.add(tuple(round(v, 1) for v in n[:3]))
    assert len(e_pos) >= 5, "позиция электрона не меняется между кадрами"


def test_snapshot_nodes_em_off():
    v = Volume3D(size=8, seed=3)
    v.inject_particle((1, 2, 3), "p", spin=0)
    v.inject_particle((5, 6, 7), "e⁻", spin=1)
    f = v.snapshot_nodes("тест")
    assert f["light"] is True
    assert f["t"] == 0
    bypos = {tuple(round(c, 3) for c in n[:3]): n[3] for n in f["data"]["nodes"]}
    assert bypos[(1, 2, 3)] == 1     # p — в центре своей ячейки
    assert bypos[(5, 6, 7)] == 2     # e⁻ — в центре своей ячейки
    assert f["metrics"]["matter"] == 2


def test_d_dbar_string_survives():
    """Струна d—d̄ (фаза 9) не должна рассеиваться на первом шаге
    (регрессия: _cell_is_quark принимал только u/ū)."""
    lab = Volume3D(20)
    assert lab.inject_quark_pair((10, 14, 6), dist=3, flavor="d")
    assert len(lab.strings) == 1
    lab.step()
    assert len(lab.strings) >= 1
