"""Register every number quoted from the literature (``paper/literature.json``) in ``paper/claims.json`` so external numbers are traceable too.

    python scripts/register_literature.py

Claim keys are ``lit.<entry>.<field>``; the artifact is ``paper/literature.json`` and the command names the verification method.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import register_claim  # noqa: E402


def main() -> None:
    """Register all numeric ``reported`` values."""
    lit = json.loads((ROOT / "paper" / "literature.json").read_text(encoding="utf-8"))
    n = 0
    for name, e in lit["entries"].items():
        for k, v in e.get("reported", {}).items():
            if isinstance(v, (int, float)):
                register_claim(f"lit.{name}.{k}", v, ROOT / "paper" / "literature.json",
                               f"external source {e.get('doi') or 'arXiv:' + e.get('arxiv', '?')}; read on {lit['verified_on']} (paper/literature.json)")
                n += 1
    print(f"registered {n} literature claims")


if __name__ == "__main__":
    main()
