# Spec 01 — scaffold (`src/common.py`, `configs/default.yaml`)

Shared plumbing used by every script.

- `load_config(path=None, overrides=[])` -> dict. Overrides are `a.b.c=value` strings parsed as YAML (`--set dvs.threshold=0.05`).
- `add_common_args(parser)` adds `--config --set --seed --device --out-dir`; `resolve_cfg(args)` merges file < `--set` < explicit flags.
- `setup_logging()` stdout logging; `set_seed(seed)` seeds python/numpy/torch.
- `out_path(cfg, stem, ext, tag=None)` -> `out/<stem>[_tag]_<YYYYmmdd-HHMMSS>.<ext>` (directory created).
- `register_claim(key, value, artifact, command, note)` appends to `paper/claims.json` (every paper number must be registered).
- `save_json(obj, path)` handles numpy types. `repo_path(rel)` resolves config paths against the repo root.

Rules: every script is runnable as `python -m src.<module>` with `--help`; all outputs go to `out/`; seeds everywhere.
Acceptance: `python -m src.evaluate --help` runs; every module imports (`tests/test_imports.py`).
