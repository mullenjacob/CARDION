"""Shared plumbing: config loading, logging, seeding, timestamped outputs, claims registry."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def timestamp() -> str:
    """Return a filesystem-safe local timestamp, e.g. ``20261022-153012``."""
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _parse_scalar(text: str) -> Any:
    """Parse a CLI override value as YAML (so ``0.1``, ``[1,2]`` and ``true`` work)."""
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` overrides to a nested config dict (returns a copy)."""
    out = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(value)
    return out


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> dict:
    """Load the YAML config and apply ``--set`` overrides."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8-sig") as fh:
        cfg = yaml.safe_load(fh)
    return apply_overrides(cfg, overrides or [])


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --config/--set/--seed/--device/--out-dir to a script's parser."""
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a config value (repeatable)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: config seed)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="compute device")
    parser.add_argument("--out-dir", default=None, help="output directory (default: config out_dir)")
    return parser


def resolve_cfg(args: argparse.Namespace) -> dict:
    """Merge config file, ``--set`` overrides and explicit CLI flags (CLI wins)."""
    cfg = load_config(args.config, args.overrides)
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = args.seed
    if getattr(args, "device", None) is not None:
        cfg["device"] = args.device
    if getattr(args, "out_dir", None) is not None:
        cfg["out_dir"] = args.out_dir
    return cfg


def setup_logging(name: str = "cardion", level: int = logging.INFO) -> logging.Logger:
    """Configure stdout logging with timestamps and return a logger."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
    root.setLevel(level)
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if present) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch is optional for the classical pipeline
        pass


def out_path(cfg: dict, stem: str, ext: str, tag: str | None = None, stamp: bool = True) -> Path:
    """Return ``<out_dir>/<stem>[_tag][_timestamp].<ext>`` and create the directory."""
    out_dir = Path(cfg.get("out_dir", "out"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [stem] + ([tag] if tag else []) + ([timestamp()] if stamp else [])
    return out_dir / ("_".join(parts) + "." + ext.lstrip("."))


def repo_path(rel: str | os.PathLike) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def save_json(obj: Any, path: str | os.PathLike) -> Path:
    """Write JSON with numpy support."""
    def default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serialisable: {type(o)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=default)
    return path


def register_claim(key: str, value: Any, artifact: str | os.PathLike, command: str,
                   note: str = "") -> None:
    """Record a paper claim (number -> artifact file -> producing command) in paper/claims.json."""
    path = REPO_ROOT / "paper" / "claims.json"
    claims = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    try:
        art = str(Path(artifact).resolve().relative_to(REPO_ROOT))
    except ValueError:
        art = str(artifact)
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    claims[key] = {"value": value, "artifact": art.replace("\\", "/"), "command": command, "note": note,
                   "recorded": timestamp()}
    save_json(claims, path)

