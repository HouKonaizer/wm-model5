# -*- coding: utf-8 -*-
"""Тесты wm.config: дефолты, переопределение, валидация, поиск config.toml."""
import tomllib

import pytest

from wm.config import DEFAULTS, load_config

ROOT = __file__.rsplit("tests", 1)[0]


def test_empty_config_gives_defaults(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("", encoding="utf-8")
    assert load_config(str(p)) == DEFAULTS


def test_override(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[run]\nsize = 12\nseed = 5\noutput_dir = "out"\n', encoding="utf-8")
    cfg = load_config(str(p))
    assert cfg["run"]["size"] == 12
    assert cfg["run"]["seed"] == 5
    assert cfg["run"]["output_dir"] == "out"
    # не заданное — из дефолтов
    assert cfg["run"]["write_viewer"] is True


def test_explicit_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nope.toml"))


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text('[run]\nbogus = 1\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(p))


@pytest.mark.parametrize("body", ["size = 2", "size = -3", 'size = "x"', "size = true", "size = 2.5"])
def test_bad_size_raises(tmp_path, body):
    p = tmp_path / "config.toml"
    p.write_text(f"[run]\n{body}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(p))


def test_search_cwd_first(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text('[run]\nsize = 8\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert load_config(None)["run"]["size"] == 8


def test_search_falls_back_to_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # в tmp config.toml нет
    cfg = load_config(None)
    root_cfg = tomllib.load(open(ROOT + "config.toml", "rb"))
    assert cfg["run"]["size"] == root_cfg["run"]["size"]
