# -*- coding: utf-8 -*-
"""Тесты детерминизма: та же цепочка распадов при том же seed (регрессия)."""
from wm.model import run_simulation


def _events(world):
    return [(e["t"], e["type"]) for e in world.decay_events]


def test_decay_chain_known_events():
    world = run_simulation(size=20, seed=42, quiet=True)["world"]
    ev = _events(world)
    # известная цепочка (регрессия Option 4): t=312 аннигиляция u+ū,
    # 361 τ⁻→π±, 449 μ⁻→e⁻, 474 π±→μ⁻, 612 μ⁻→e⁻
    assert (312, "annihilation") in ev
    assert len(ev) >= 5, f"ожидалось ≥5 событий, получено {len(ev)}: {ev}"


def test_decay_chain_deterministic():
    w1 = run_simulation(size=20, seed=42, quiet=True)["world"]
    w2 = run_simulation(size=20, seed=42, quiet=True)["world"]
    assert _events(w1) == _events(w2)
    assert w1.matter_count == w2.matter_count
    assert abs(float(w1.energy.sum()) - float(w2.energy.sum())) < 1e-9
