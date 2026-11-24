# -*- coding: utf-8 -*-
"""Тесты CLI: wm version / run / checks (малые сетки, быстрый режим)."""
import json

from wm.cli import main as cli


def test_version(capsys):
    assert cli(["version"]) == 0
    out = capsys.readouterr().out
    assert "wm 0.1.0" in out


def test_run_small(tmp_path):
    assert cli(["run", "--size", "4", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "snapshots.json").exists()
    assert (tmp_path / "viewer.html").exists()
    sp = json.loads((tmp_path / "snapshots.json").read_text(encoding="utf-8"))
    assert sp["size"] == 4


def test_run_no_viewer(tmp_path):
    assert cli(["run", "--size", "4", "--out", str(tmp_path), "--no-viewer"]) == 0
    assert (tmp_path / "snapshots.json").exists()
    assert not (tmp_path / "viewer.html").exists()


def test_run_config_override(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[run]\nsize = 6\nseed = 5\n', encoding="utf-8")
    out = tmp_path / "out"
    assert cli(["run", "--config", str(cfg), "--out", str(out)]) == 0
    sp = json.loads((out / "snapshots.json").read_text(encoding="utf-8"))
    assert sp["size"] == 6


def test_checks_fast(tmp_path):
    assert cli(["checks", "--fast", "--out", str(tmp_path)]) == 0
    v = json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))
    assert v["n_checks"] == 68  # 76 без ЧИСЛ-блока D7–D14


def test_demo_writes_verification(tmp_path):
    assert cli(["demo", "--out", str(tmp_path)]) == 0
    v = json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))
    assert v["n_checks"] == 68


def test_demo_report_content(tmp_path, capsys):
    assert cli(["demo", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for s in ("68", "correspondence", "prediction", "discrete",
              "21 см", "R_π", "p-Ps", "e-06%"):
        assert s in out
    assert "+0.000%" not in out  # a_e (dev 7e-8) не скрывается за нулём


def test_demo_default_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli(["demo"]) == 0
    assert (tmp_path / "demo_out" / "verification.json").exists()
