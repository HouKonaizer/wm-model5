# -*- coding: utf-8 -*-
"""
wm.cli — командный интерфейс.

Команды:
    wm run [--config F] [--size N] [--seed S] [--out DIR] [--no-viewer] [--bg N]
        полный сценарий (фазы 1–8) → snapshots.json + viewer.html в DIR;
        --bg N — разрешение фона D³ (авто: полное для N<100, иначе делитель ≤64);
    wm checks [--config F] [--out DIR] [--fast]
        таблица сверок → verification.json в DIR (--fast: без ЧИСЛ D7–D14);
    wm demo [--out DIR]
        демо для внешнего читателя: fast-гейт (68 проверок) + отчёт (EN)
        → verification.json в DIR (по умолчанию demo_out/); физику не меняет;
    wm serve [--out DIR] [--port P] [--host H]
        HTTP-раздача каталога (viewer.html в файловом режиме + frames.bin);
    wm version
"""
import argparse
import contextlib
import io
import json
import os
import sys

from . import __version__
from .config import load_config


def _utf8_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def cmd_run(args):
    cfg = load_config(args.config)
    run = dict(cfg["run"])
    size = args.size if args.size is not None else run["size"]
    seed = args.seed if args.seed is not None else run["seed"]
    out_dir = args.out if args.out is not None else run["output_dir"]
    write_viewer = False if args.no_viewer else run["write_viewer"]
    os.makedirs(out_dir, exist_ok=True)

    from .model import main as model_main
    model_main(out_dir=out_dir, size=size, seed=seed,
               animate=args.animate, bg_res=args.bg)
    if not write_viewer:
        for name in ("viewer.html",):
            p = os.path.join(out_dir, name)
            if os.path.isfile(p):
                os.remove(p)
    return 0


def cmd_serve(args):
    """HTTP-раздача каталога: viewer.html в файловом режиме fetch-ит
    frames.bin из того же каталога (file:// в браузерах запрещает fetch)."""
    import functools
    import http.server
    import socketserver

    out_dir = args.out or "."
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"каталог не найден: {out_dir}")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=out_dir)

    class Srv(socketserver.TCPServer):
        allow_reuse_address = True

    srv = Srv((args.host, args.port), handler)
    print(f"wm serve: http://{args.host}:{srv.server_address[1]}/viewer.html "
          f"(Ctrl+C — стоп)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


def cmd_checks(args):
    cfg = load_config(args.config)
    out_dir = args.out if args.out is not None else cfg["checks"]["output_dir"]
    from .experiments import main as experiments_main
    experiments_main(out_dir=out_dir, fast=args.fast)
    return 0


def cmd_demo(args):
    """Демо для внешнего читателя: fast-гейт (68 проверок) + компактный отчёт (EN).

    Пишет verification.json в --out (по умолчанию demo_out/); физику не меняет.
    (Отчёт на английском — это витрина для внешнего читателя; остальной CLI — RU.)"""
    from .experiments import main as experiments_main

    def _dev_str(d):
        if d != d:  # NaN
            return "—"
        if abs(d) < 1e-5:  # мелкое отклонение — экспоненциально, не «+0.000%»
            return f"{d * 100:+.2e}%"
        return f"{d * 100:+.3f}%"

    out_dir = args.out or "demo_out"
    os.makedirs(out_dir, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        experiments_main(out_dir=out_dir, fast=True)
    with open(os.path.join(out_dir, "verification.json"),
              encoding="utf-8") as f:
        v = json.load(f)
    checks = v["checks"]
    kinds = {}
    for c in checks:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1

    print(f"wm {__version__} — self-consistent 5-subspace lattice model")
    print(f"fast verification gate: {v['n_checks']} checks "
          f"(correspondence={kinds.get('correspondence', 0)}, "
          f"prediction={kinds.get('prediction', 0)}, "
          f"discrete={kinds.get('discrete', 0)})")
    print()
    print("Predictions (computed from masses/anchors, no fitting):")
    for c in checks:
        if c["kind"] != "prediction":
            continue
        print(f"  {c['name'][:56]:<56s} {c['v_model']:>13.6g} "
              f"{c['v_exp']:>13.6g}  {_dev_str(c['dev']):>10s}")
    print()
    print("Full audit (incl. lattice block D7–D18):  python -m wm checks")
    print("Docs: README_EN.md (EN) | README.md, THEORY.md (RU) | PREPRINT.md")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="wm",
        description="5D-решётчатая модель: время, энергия, вероятность, гравитация, поля")
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("run", help="полный сценарий симуляции")
    pr.add_argument("--config", help="путь к config.toml (по умолчанию: cwd, корень проекта)")
    pr.add_argument("--size", type=int, help="размер сетки N³ (переопределяет конфиг)")
    pr.add_argument("--seed", type=int, help="сид RNG (переопределяет конфиг)")
    pr.add_argument("--out", help="каталог вывода (переопределяет конфиг)")
    pr.add_argument("--no-viewer", action="store_true", help="не писать viewer.html")
    pr.add_argument("--animate", action="store_true",
                    help="анимация фаз 9–11 (потоки кадров узлов: кварки → адроны → H → H2)")
    pr.add_argument("--bg", type=int,
                    help="разрешение фона D³ (D делит size; по умолчанию авто)")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("serve", help="HTTP-раздача каталога (viewer + frames.bin)")
    ps.add_argument("--out", help="каталог для раздачи (по умолчанию: cwd)")
    ps.add_argument("--port", type=int, default=8000, help="порт (по умолчанию 8000)")
    ps.add_argument("--host", default="127.0.0.1", help="хост (по умолчанию 127.0.0.1)")
    ps.set_defaults(func=cmd_serve)

    pd = sub.add_parser("demo",
                        help="демо для внешнего читателя: fast-гейт + отчёт (EN)")
    pd.add_argument("--out", help="каталог для verification.json (по умолчанию demo_out)")
    pd.set_defaults(func=cmd_demo)

    pc = sub.add_parser("checks", help="таблица сверок модель ↔ эксперимент")
    pc.add_argument("--config", help="путь к config.toml")
    pc.add_argument("--out", help="каталог для verification.json")
    pc.add_argument("--fast", action="store_true", help="без ЧИСЛ-блока D7–D14 (68 проверок)")
    pc.set_defaults(func=cmd_checks)

    pv = sub.add_parser("version", help="версия пакета")
    pv.set_defaults(func=lambda a: (print(f"wm {__version__}"), 0)[1])
    return p


def main(argv=None):
    _utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"wm: ошибка: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"wm: ошибка: {e}", file=sys.stderr)
        return 2
