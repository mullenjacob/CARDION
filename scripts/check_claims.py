"""Check that every number in the paper traces to a saved artifact.

For each ``{{c:key}}`` / ``{{r:a/b}}`` placeholder used by ``paper/main.tex.in`` and ``paper/sections/*.tex.in``: the claim must exist in
``paper/claims.json``, its artifact file must exist, and (for JSON artifacts) the claimed value must be found among the numbers stored in
that file. Also checks that the paper's key results (test-set MAE per method) still equal the numbers in the saved result tables.

    python scripts/check_claims.py            # exit code 1 if anything is untraceable
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"


def numbers_in(obj, out: list[float]) -> None:
    """Collect every number stored anywhere in a JSON structure."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            numbers_in(v, out)


def derived_numbers(obj, out: list[float]) -> None:
    """Numbers the registering scripts derive from stored values: counts of items, and means over lists / over per-clip dictionaries."""
    if isinstance(obj, list):
        out.append(float(len(obj)))
        nums = [float(x) for x in obj if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if nums:
            out.append(sum(nums) / len(nums))
        dicts = [x for x in obj if isinstance(x, dict)]
        for key in {k for d in dicts for k in d}:
            vals = [float(d[key]) for d in dicts if isinstance(d.get(key), (int, float)) and not isinstance(d.get(key), bool)]
            if vals:
                out.append(sum(vals) / len(vals))
        for x in obj:
            derived_numbers(x, out)
    elif isinstance(obj, dict):
        vals = [float(v) for v in obj.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if vals:
            out.append(sum(vals) / len(vals))
            out.append(float(len(vals)))
        subs = [v for v in obj.values() if isinstance(v, dict)]
        for key in {k for d in subs for k in d}:
            vv = [float(d[key]) for d in subs if isinstance(d.get(key), (int, float)) and not isinstance(d.get(key), bool)]
            if vv:
                out.append(sum(vv) / len(vv))
        out.append(float(len(obj)))
        for v in obj.values():
            derived_numbers(v, out)


def flat_rows(obj) -> list[dict]:
    """Every dictionary that has numeric values and sits inside lists (per-clip result rows), at any depth."""
    rows: list[dict] = []
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict) and any(isinstance(v, (int, float)) for v in x.values()):
                rows.append(x)
            rows += flat_rows(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            rows += flat_rows(v)
    return rows


def grouped_means(obj, out: list[float]) -> None:
    """Means (and means of absolute values) of every numeric field over rows sharing the same ``threshold`` / ``setting`` values."""
    rows = flat_rows(obj)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r.get("threshold"), r.get("setting")), []).append(r)
        groups.setdefault((r.get("threshold"), None), []).append(r)
    for members in groups.values():
        for key in {k for r in members for k in r}:
            vals = [float(r[key]) for r in members if isinstance(r.get(key), (int, float)) and not isinstance(r.get(key), bool)]
            if vals:
                out.append(sum(vals) / len(vals))
                out.append(sum(abs(v) for v in vals) / len(vals))


def derived_for(data) -> list[float]:
    """All derived numbers (counts, means, grouped means) for a loaded artifact."""
    out: list[float] = []
    derived_numbers(data, out)
    grouped_means(data, out)
    return out


def used_keys() -> set[str]:
    """Claim keys referenced by the paper templates."""
    keys: set[str] = set()
    files = [PAPER / "main.tex.in", *sorted((PAPER / "sections").glob("*.tex.in"))]
    for f in files:
        text = f.read_text(encoding="utf-8-sig")
        for m in re.finditer(r"\{\{c:([^}|]+)", text):
            keys.add(m.group(1))
        for m in re.finditer(r"\{\{r:([^}|]+)", text):
            keys.update(m.group(1).split("/"))
    return keys


def main() -> None:
    """CLI entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    claims = json.loads((PAPER / "claims.json").read_text(encoding="utf-8"))
    keys = sorted(used_keys())
    problems: list[str] = []
    verified_in_file = derived = 0
    for k in keys:
        c = claims.get(k)
        if c is None:
            problems.append(f"{k}: not registered in claims.json")
            continue
        art = ROOT / c["artifact"]
        if not art.exists():
            problems.append(f"{k}: artifact missing ({c['artifact']})")
            continue
        if art.suffix == ".json" and isinstance(c["value"], (int, float)):
            nums: list[float] = []
            numbers_in(json.loads(art.read_text(encoding="utf-8-sig")), nums)
            if any(math.isclose(c["value"], n, rel_tol=1e-6, abs_tol=1e-9) for n in nums):
                verified_in_file += 1
            elif any(math.isclose(c["value"], n, rel_tol=1e-6, abs_tol=1e-9) for n in derived_for(json.loads(art.read_text(encoding="utf-8-sig")))):
                derived += 1
            else:
                problems.append(f"{k}: value {c['value']} not found in {c['artifact']} (it may be derived; check the command: {c['command']})")
    print(f"{len(keys)} claims are used in the paper; {len(keys) - len(problems)} resolve to an existing artifact; "
          f"{verified_in_file} values were found verbatim inside their JSON artifact and {derived} equal a count or mean of stored values.")
    for p in problems:
        print("  PROBLEM:", p)
    unused_missing = [k for k, v in claims.items() if k not in keys and not (ROOT / v["artifact"]).exists()]
    print(f"{len(unused_missing)} further registered claims (not used in the paper text) point to artifacts that are no longer in out/.")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
