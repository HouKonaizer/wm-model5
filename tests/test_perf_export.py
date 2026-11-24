# -*- coding: utf-8 -*-
"""Оптимизации (TDD, сначала тесты):

1. Векторизация EM-двигателя (_update_em): аналитическая сила, torus-wrap,
   время шага на 1500 зарядах.
2. Бинарный формат экспорта: layout (10×float32 + 3×uint8 на D³), min/max,
   single-file (base64 в HTML) vs frames.bin + fetch, синтаксис JS (node --check).
3. Даунсемпл-фон: downsample (mean/max), bg_res, точные узлы в JSON.
"""
import base64
import functools
import http.server
import json
import re
import socketserver
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from wm.cli import main as cli
from wm.model import (
    COULOMB_K, COULOMB_SOFT2, EM_DT,
    _NODE_CHARGE, _NODE_MEFF,
    Volume3D,
    export_viewer,
)

# ---------- helpers ----------


def _tiny_frames(size=8, steps=10):
    """Два полных кадра на мелкой сетке."""
    v = Volume3D(size=size, seed=3)
    for _ in range(steps):
        v.step()
    return [v.snapshot("тест 1"), v.snapshot("тест 2")]


def _inline_js(html):
    m = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert m, "inline <script> в viewer.html не найден"
    return m[-1]


def _node_check(js, tmp_path, name="inline.js"):
    p = Path(tmp_path) / name
    p.write_text(js, encoding="utf-8")
    r = subprocess.run(["node", "--check", str(p)],
                       capture_output=True, text=True)
    assert r.returncode == 0, "node --check: " + r.stderr


def _bin_frame_size(D):
    """10 полей float32 + 3 поля uint8, каждое D³."""
    return (10 * 4 + 3) * D * D * D


# ---------- 1. EM-векторизация ----------

def test_em_force_analytic():
    """Один шаг _update_em: Δv двух зарядов = F/m·dt (Кулон + мягкое ядро)."""
    v = Volume3D(size=20, seed=1)
    v.em_dynamics = True
    ip = v._flat(10, 10, 10)
    ie = v._flat(13, 10, 10)
    assert v.inject_particle((10, 10, 10), "p", spin=0)
    assert v.inject_particle((13, 10, 10), "e⁻", spin=1)
    d2 = 9.0 + COULOMB_SOFT2
    k = COULOMB_K * _NODE_CHARGE["p"] * _NODE_CHARGE["e⁻"] / d2 ** 1.5
    v._update_em()
    # спин-маппинг: spin=0 → слот 2 (vel2_*), spin=1 → слот 1 (vel_*)
    dv_p = k * (-3.0) / _NODE_MEFF["p"] * EM_DT      # k < 0 → p в +X (притяжение)
    dv_e = k * (3.0) / _NODE_MEFF["e⁻"] * EM_DT      # e в −X
    assert v.vel2_x[ip] == pytest.approx(dv_p, rel=1e-9)
    assert v.vel_x[ie] == pytest.approx(dv_e, rel=1e-9)
    assert v.vel2_y[ip] == 0.0
    assert v.vel_z[ie] == 0.0
    # остальные узлы не двигались
    assert v.vel_x.sum() + v.vel2_x.sum() == pytest.approx(dv_p + dv_e)


def test_em_force_torus_wrap():
    """Заряды у противоположных граней: сила по кратчайшему (wrap) пути."""
    v = Volume3D(size=20, seed=2)
    v.em_dynamics = True
    ip = v._flat(1, 10, 10)
    ie = v._flat(18, 10, 10)
    assert v.inject_particle((1, 10, 10), "p", spin=0)
    assert v.inject_particle((18, 10, 10), "e⁻", spin=1)
    d2 = 9.0 + COULOMB_SOFT2  # кратчайшее расстояние = 3 (через границу)
    k = COULOMB_K * _NODE_CHARGE["p"] * _NODE_CHARGE["e⁻"] / d2 ** 1.5
    v._update_em()
    # P[p]−P[e] = (−17,0,0) → torus (3,0,0): p притягивается в −X (vel2: spin 0)
    assert v.vel2_x[ip] == pytest.approx(k * 3.0 / _NODE_MEFF["p"] * EM_DT,
                                         rel=1e-9)


def test_em_perf_1500():
    """1500 заряженных узлов: один шаг < 2 с (старый O(n²) — секунды)."""
    v = Volume3D(size=32, seed=5)
    v.em_dynamics = True
    placed = 0
    for idx in range(0, v.n, 5):
        if placed >= 1500:
            break
        x, y, z = idx // (v.size * v.size), (idx // v.size) % v.size, idx % v.size
        sym, spin = ("p", 0) if placed % 2 == 0 else ("e⁻", 1)
        assert v.inject_particle((x, y, z), sym, spin=spin)
        placed += 1
    assert placed == 1500
    t0 = time.perf_counter()
    v._update_em()
    dt = time.perf_counter() - t0
    assert dt < 2.0, f"медленно: {dt:.2f} c"


# ---------- 3. Даунсемпл-фон ----------

def test_downsample_mean_scalar():
    v = Volume3D(size=8, seed=1)
    a = np.arange(512, dtype=np.float64).reshape(8, 8, 8)
    b = v._downsample(a, 4)
    assert b.shape == (4, 4, 4)
    for x in range(4):
        for y in range(4):
            for z in range(4):
                block = a[2 * x:2 * x + 2, 2 * y:2 * y + 2, 2 * z:2 * z + 2]
                assert b[x, y, z] == pytest.approx(block.mean(), rel=1e-12)


def test_downsample_field_and_max():
    v = Volume3D(size=8, seed=1)
    f = np.arange(3 * 512, dtype=np.float64).reshape(3, 8, 8, 8)
    fb = v._downsample(f, 4)
    assert fb.shape == (3, 4, 4, 4)
    assert fb[1, 2, 3, 0] == pytest.approx(
        f[1, 4:6, 6:8, 0:2].mean(), rel=1e-12)
    m = np.zeros((8, 8, 8), dtype=np.int8)
    m[7, 7, 7] = 1
    mb = v._downsample(m, 4, mode="max")
    assert mb[3, 3, 3] == 1
    assert mb.sum() == 1
    a = np.zeros((8, 8, 8))
    with pytest.raises(ValueError):
        v._downsample(a, 3)  # 3 не делит 8


def test_auto_bg():
    from wm.model import _auto_bg
    assert _auto_bg(20) is None
    assert _auto_bg(80) == 40      # наибольший делитель 80 ≤ 64
    assert _auto_bg(100) == 50
    assert _auto_bg(128) == 64
    assert _auto_bg(1000) == 50


def test_snapshot_has_nodes():
    """snapshot(): полный кадр несёт точный список узлов (координаты + код + спин)."""
    v = Volume3D(size=8, seed=3)
    assert v.inject_particle((1, 2, 3), "p", spin=0)
    assert v.inject_particle((5, 6, 7), "e⁻", spin=1)
    d = v.snapshot("тест")["data"]
    assert [n for n in d["nodes"] if n[3] == 1] == [[1.0, 2.0, 3.0, 1, 0]]
    assert [n for n in d["nodes"] if n[3] == 2] == [[5.0, 6.0, 7.0, 2, 1]]


def test_snapshot_arrays():
    """snapshot_arrays: data — numpy-массивы; bg_res — даунсемпл на захвате."""
    v = Volume3D(size=16, seed=3)
    v.inject_particle((2, 3, 4), "p", spin=0)
    f = v.snapshot_arrays("тест")
    assert isinstance(f["data"]["energy"], np.ndarray)
    assert f["data"]["field_vec"].shape == (3, 16, 16, 16)
    assert [n for n in f["data"]["nodes"] if n[3] == 1] == [[2.0, 3.0, 4.0, 1, 0]]
    f4 = v.snapshot_arrays("тест", bg_res=4)
    assert f4["data"]["energy"].shape == (4, 4, 4)
    assert f4["data"]["field_vec"].shape == (3, 4, 4, 4)
    # фон даунсемплирован, узлы — точные
    src = v.energy.reshape(16, 16, 16)
    np.testing.assert_allclose(f4["data"]["energy"],
                               src.reshape(4, 4, 4, 4, 4, 4).mean(axis=(1, 3, 5)),
                               rtol=1e-12)


# ---------- 2. Бинарный экспорт ----------

def test_export_single_file(tmp_path):
    """Мелкий экспорт → один файл: base64 в HTML, frames.bin не нужен."""
    export_viewer(_tiny_frames(), str(tmp_path), size=8)
    sp = json.loads((tmp_path / "snapshots.json").read_text(encoding="utf-8"))
    assert sp["size"] == 8
    assert sp["bg"] is None  # мелкая сетка — фон полного разрешения
    html = (tmp_path / "viewer.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/null" not in html
    assert '"bin_b64":"' in html
    assert '"bin":"frames.bin"' not in html
    assert not (tmp_path / "frames.bin").exists()
    _node_check(_inline_js(html), tmp_path, "embed.js")


def test_export_file_mode(tmp_path):
    """single_file=False → frames.bin + snapshots.json + viewer.html (fetch)."""
    export_viewer(_tiny_frames(), str(tmp_path), size=8, single_file=False)
    assert (tmp_path / "frames.bin").exists()
    sp = json.loads((tmp_path / "snapshots.json").read_text(encoding="utf-8"))
    assert sp["bin"] == "frames.bin"
    assert "bin_b64" not in sp
    html = (tmp_path / "viewer.html").read_text(encoding="utf-8")
    assert '"bin":"frames.bin"' in html
    assert '"bin_b64":"' not in html
    _node_check(_inline_js(html), tmp_path, "filemode.js")


def test_export_bin_size(tmp_path):
    """Размер бинарника: 10×float32 + 3×uint8 на D³ за полный кадр."""
    export_viewer(_tiny_frames(), str(tmp_path), size=8, single_file=False)
    assert (tmp_path / "frames.bin").stat().st_size == 2 * _bin_frame_size(8)


def test_export_bin_layout(tmp_path):
    """Layout: float32-поля в фиксированном порядке; значения совпадают с кадром."""
    export_viewer(_tiny_frames(), str(tmp_path), size=8, single_file=False)
    buf = (tmp_path / "frames.bin").read_bytes()
    sp = json.loads((tmp_path / "snapshots.json").read_text(encoding="utf-8"))
    f0 = sp["frames"][0]
    ND = 8 ** 3
    o = f0["bin"][0]
    assert f0["bin"][1] == _bin_frame_size(8)
    tp = np.frombuffer(buf, dtype=np.float32, count=ND, offset=o)
    # сверка с исходными данными кадра
    v = Volume3D(size=8, seed=3)
    for _ in range(10):
        v.step()
    src = v.snapshot("тест 1")
    tp_expect = np.asarray(src["data"]["time_phase"], dtype=np.float32).ravel()
    np.testing.assert_allclose(tp, tp_expect, rtol=1e-6, atol=1e-7)
    # порядок BIN_F32: time_phase, time_rate, energy, ...
    en = np.frombuffer(buf, dtype=np.float32, count=ND, offset=o + 2 * ND * 4)
    en_expect = np.asarray(src["data"]["energy"], dtype=np.float32).ravel()
    np.testing.assert_allclose(en, en_expect, rtol=1e-6, atol=1e-7)
    # uint8-блок matter: смещение после 10 float32-полей
    om = o + 10 * ND * 4
    matter = np.frombuffer(buf, dtype=np.uint8, count=ND, offset=om)
    np.testing.assert_array_equal(
        matter, np.asarray(src["data"]["matter"], dtype=np.uint8).ravel())


def test_export_minmax(tmp_path):
    """min/max per field per frame — для легенды и shader-окраски."""
    export_viewer(_tiny_frames(), str(tmp_path), size=8, single_file=False)
    sp = json.loads((tmp_path / "snapshots.json").read_text(encoding="utf-8"))
    mm = sp["frames"][0]["minmax"]
    assert set(mm) == {"time_phase", "time_rate", "energy", "psi_abs",
                       "psi_phase", "grav", "field_mag", "fx", "fy", "fz"}
    for mn, mx in mm.values():
        assert mn <= mx
    # JSON не тащит большие поля
    assert "time_rate" not in sp["frames"][0]["data"]
    assert "nodes" in sp["frames"][0]["data"]


def test_export_bg_downsample(tmp_path):
    """bg_res=4: фон в бинарнике 4³, узлы — точные (полное разрешение)."""
    frames = _tiny_frames()
    export_viewer(frames, str(tmp_path), size=8, bg_res=4, single_file=False)
    sp = json.loads((tmp_path / "snapshots.json").read_text(encoding="utf-8"))
    assert sp["bg"] == 4
    assert (tmp_path / "frames.bin").stat().st_size == 2 * _bin_frame_size(4)
    buf = (tmp_path / "frames.bin").read_bytes()
    ND = 4 ** 3
    o = sp["frames"][0]["bin"][0]
    tp = np.frombuffer(buf, dtype=np.float32, count=ND, offset=o)
    src = np.asarray(frames[0]["data"]["time_phase"],
                     dtype=np.float64).reshape(8, 8, 8)
    expect = src.reshape(4, 2, 4, 2, 4, 2).mean(axis=(1, 3, 5)) \
        .astype(np.float32).ravel()
    np.testing.assert_allclose(tp, expect, rtol=1e-6, atol=1e-7)
    nodes = sp["frames"][0]["data"]["nodes"]
    assert isinstance(nodes, list)


def test_export_bg_mismatch(tmp_path):
    """bg_res, не делящий размер сетки → ясная ошибка."""
    with pytest.raises(ValueError):
        export_viewer(_tiny_frames(), str(tmp_path), size=8, bg_res=3,
                      single_file=False)


def test_run_bg_flag(tmp_path):
    """wm run --bg 4: снапшоты и экспорт на фоне 4³."""
    rc = cli(["run", "--size", "20", "--bg", "4", "--out", str(tmp_path)])
    assert rc == 0
    sp = json.loads((tmp_path / "snapshots.json").read_text(encoding="utf-8"))
    assert sp["size"] == 20
    assert sp["bg"] == 4
    n_full = sum(1 for f in sp["frames"] if "bin" in f)
    assert n_full == len(sp["frames"])  # все снапшоты полные
    if (tmp_path / "frames.bin").exists():
        assert (tmp_path / "frames.bin").stat().st_size \
            == n_full * _bin_frame_size(4)
    else:  # бинарник ≤ 32 МБ → встроён в HTML (base64)
        assert "bin_b64" in sp


def test_serve_static(tmp_path):
    """Файловый режим раздаётся статическим HTTP-сервером (wm serve)."""
    export_viewer(_tiny_frames(), str(tmp_path), size=8, single_file=False)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(tmp_path))

    class Srv(socketserver.TCPServer):
        allow_reuse_address = True

    srv = Srv(("127.0.0.1", 0), handler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        html = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/viewer.html", timeout=5).read()
        assert b"viewer" in html
        data = urllib.request.urlopen(
            f"http://127.0.0.1:{port}/frames.bin", timeout=5).read()
        assert data == (tmp_path / "frames.bin").read_bytes()
    finally:
        srv.shutdown()
        srv.server_close()


def test_serve_cli_registered():
    """wm serve зарегистрирован в CLI."""
    with pytest.raises(SystemExit) as ei:
        cli(["serve", "--help"])
    assert ei.value.code == 0
