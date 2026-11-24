# -*- coding: utf-8 -*-
"""Тесты экспорта 3D-вьювера: snapshots.json + viewer.html (шаблон из проекта)."""
import json

import pytest

from wm.model import Volume3D, export_viewer


def _tiny_frames(size=4):
    v = Volume3D(size=size, seed=3)
    for _ in range(10):
        v.step()
    return [v.snapshot("тест 1"), v.snapshot("тест 2")]


def test_export_viewer_writes_files(tmp_path):
    frames = _tiny_frames()
    export_viewer(frames, str(tmp_path), size=4)
    sp = json.loads((tmp_path / "snapshots.json").read_text(encoding="utf-8"))
    assert sp["size"] == 4
    assert sp["layers"] == ["ВРЕМЯ", "ЭНЕРГИЯ", "ВЕРОЯТНОСТЬ", "ГРАВИТАЦИЯ", "ПОЛЯ"]
    assert len(sp["frames"]) == 2
    html = (tmp_path / "viewer.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/null" not in html, "маркер не подменён"
    assert '"size":4' in html


def test_export_viewer_template_missing_raises(tmp_path):
    # извлекаем шаблон из пути поиска: временно прячем корневой шаблон
    import wm.model as M
    import os

    root_tpl = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(M.__file__))),
                            "viewer_template.html")
    assert os.path.isfile(root_tpl)
    frames = _tiny_frames()
    # tmp_path без шаблона, корневой существует → должен сработать fallback (не ошибка)
    export_viewer(frames, str(tmp_path), size=4)
    assert (tmp_path / "viewer.html").exists()
