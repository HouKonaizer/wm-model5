# -*- coding: utf-8 -*-
"""Тесты audit.html — офлайн-дашборд аудита (V2, полный срез: 4 таба).

audit.html генерируется `wm checks` рядом с verification.json:
данные вшиваются в шаблон audit_template.html (маркер /*__DATA__*/null),
4 таба: Audit / Predictions / P1–P5 / About.
"""
import json

import pytest

from wm.cli import main as cli


def _extract_data(html):
    """Достроить вшитый JSON из `const DATA = {...};`."""
    marker = "const DATA = "
    i = html.index(marker) + len(marker)
    # JSON заканчивается перед `;` на своей строке
    j = html.index(";\n", i)
    return json.loads(html[i:j])


def test_audit_generated_by_checks(tmp_path):
    assert cli(["checks", "--fast", "--out", str(tmp_path)]) == 0
    p = tmp_path / "audit.html"
    assert p.exists()
    html = p.read_text(encoding="utf-8")
    assert "/*__DATA__*/null" not in html  # маркер заменён
    d = _extract_data(html)
    assert d["n_checks"] == 68
    assert len(d["checks"]) == 68


def test_audit_header_and_tabs(tmp_path):
    assert cli(["checks", "--fast", "--out", str(tmp_path)]) == 0
    html = (tmp_path / "audit.html").read_text(encoding="utf-8")
    for s in ('id="tab-audit"', 'id="tab-pred"', 'id="tab-pred5"',
              'id="tab-about"', "бит-лок", "Аудит (все проверки)"):
        assert s in html
    # fast-режим: lattice-блок D7–D18 (вкл. κ D12) пропущен — κ в данных нет
    assert "0.4069113352735587" not in html


def test_write_audit_synthetic(tmp_path):
    from wm.experiments import write_audit_html

    v = {
        "generated_by": "synthetic", "n_checks": 2,
        "checks": [
            {"cat": "ЭМ", "name": "t1", "f_model": "a", "f_phys": "b",
             "v_model": 1.0, "v_exp": 1.0, "unit": "", "dev": 0.0,
             "kind": "prediction", "src": "s", "note": "n1"},
            {"cat": "ЭМ", "name": "t2", "f_model": "a", "f_phys": "b",
             "v_model": 2.0, "v_exp": 2.0, "unit": "", "dev": float("nan"),
             "kind": "discrete", "src": "s", "note": "n2"},
        ],
        "si_map": {"U_MeV_per_unit": 781.8934068, "h_m_per_cell": 5.1157e-15,
                   "dt_s_per_step": 1.7064e-23, "anchors": ["U: тест"]},
        "frozen_predictions": [],
    }
    (tmp_path / "verification.json").write_text(
        json.dumps(v, ensure_ascii=False), encoding="utf-8")
    write_audit_html(str(tmp_path))
    html = (tmp_path / "audit.html").read_text(encoding="utf-8")
    d = _extract_data(html)
    assert d["n_checks"] == 2
    assert d["checks"][1]["dev"] != d["checks"][1]["dev"]  # NaN сохранён


def test_write_audit_missing_json(tmp_path):
    from wm.experiments import write_audit_html

    with pytest.raises(FileNotFoundError):
        write_audit_html(str(tmp_path))
