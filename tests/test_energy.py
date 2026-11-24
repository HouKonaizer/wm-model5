# -*- coding: utf-8 -*-
"""Тесты сохранения энергии: ΔΣE(T) = ledger(T) (точная бухгалтерия, порт D9)."""
import pytest

from wm.model import Volume3D


def test_energy_ledger_identity():
    v = Volume3D(size=16, seed=19)
    v.inject_particle((4, 6, 4), "n")
    v.inject_particle((9, 6, 7), "n")
    v.inject_particle((3, 3, 3), "τ⁻")
    v.inject_quark_pair((8, 4, 8), dist=3)
    e0 = float(v.energy.sum())
    l0 = v.energy_ledger
    for t in range(60):
        if t == 50:
            v.apply_force((8, 8, 8), 0.3)
        if t == 10:
            v.step(external_injection={"pos": (5, 5, 5), "energy": 2.0, "charge": 0.5, "radius": 2})
            continue
        v.step()
    dE = float(v.energy.sum()) - e0
    led = v.energy_ledger - l0
    scale = max(1.0, abs(e0), abs(led))
    assert abs(dE - led) < 1e-9 * scale, f"|ΔΣE − ledger| = {abs(dE - led):.2e} (масштаб {scale:.0f})"


def test_energy_ledger_detail_sums_to_ledger():
    v = Volume3D(size=16, seed=19)
    v.inject_particle((4, 6, 4), "n")
    for t in range(40):
        if t == 10:
            v.step(external_injection={"pos": (5, 5, 5), "energy": 2.0, "charge": 0.0, "radius": 2})
            continue
        v.step()
    assert abs(sum(v.energy_ledger_detail.values()) - v.energy_ledger) < 1e-9
