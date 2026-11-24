# -*- coding: utf-8 -*-
"""
wm.config — конфигурация из TOML (Python 3.11+, tomllib).

Приоритет: явные аргументы CLI > config.toml > DEFAULTS.
Поиск config.toml (load_config(None)): 1) cwd, 2) корень проекта.

Формат::

    [run]
    size = 20        # размер сетки N³ (int, ≥ 4)
    seed = 42        # сид RNG (int)
    output_dir = "." # каталог для snapshots.json / viewer.html
    write_viewer = true

    [checks]
    output_dir = "." # каталог для verification.json
"""
import os
import tomllib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = {
    "run": {
        "size": 20,
        "seed": 42,
        "output_dir": ".",
        "write_viewer": True,
    },
    "checks": {
        "output_dir": ".",
    },
}


def _merge(defaults, override):
    out = {k: dict(v) for k, v in defaults.items()}
    for section, values in (override or {}).items():
        if section not in out:
            raise ValueError(f"неизвестный раздел конфигурации: [{section}]")
        for key, value in values.items():
            if key not in out[section]:
                raise ValueError(
                    f"неизвестный ключ в [{section}]: {key} "
                    f"(допустимы: {', '.join(out[section])})")
            out[section][key] = value
    return out


def _validate(cfg):
    size = cfg["run"]["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 4:
        raise ValueError(f"run.size: целое ≥ 4, получено {size!r}")
    seed = cfg["run"]["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"run.seed: целое, получено {seed!r}")
    for key in ("output_dir",):
        if not isinstance(cfg["run"][key], str):
            raise ValueError(f"run.{key}: строка, получено {cfg['run'][key]!r}")
    if not isinstance(cfg["run"]["write_viewer"], bool):
        raise ValueError(f"run.write_viewer: true/false, получено {cfg['run']['write_viewer']!r}")
    if not isinstance(cfg["checks"]["output_dir"], str):
        raise ValueError(f"checks.output_dir: строка, получено {cfg['checks']['output_dir']!r}")
    return cfg


def load_config(path=None):
    """Читает TOML-конфигурацию и навалидирует её.

    path=None — ищет config.toml в cwd, затем в корне проекта.
    Явно указанный отсутствующий path — FileNotFoundError.
    """
    if path is None:
        candidates = [os.path.join(os.getcwd(), "config.toml"),
                      os.path.join(PROJECT_ROOT, "config.toml")]
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            return _validate(_merge(DEFAULTS, {}))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"конфигурация не найдена: {path}")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return _validate(_merge(DEFAULTS, data))
