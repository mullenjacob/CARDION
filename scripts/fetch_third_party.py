"""Fetch pinned third-party code into ``third_party/`` (not redistributed in this repo).

* rPPG-Toolbox reference subset (POS_WANG, CHROME_DEHAAN, post_process, utils) -> ``third_party/rppg_toolbox_ref/pkg``
  (custom licence upstream; read https://github.com/ubicomplab/rPPG-Toolbox/blob/main/LICENSE before reuse).
* v2e (MIT) source -> ``third_party/v2e_src/v2e-master`` (import path used by ``src.dvs_emulate``).

    python scripts/fetch_third_party.py [--skip-v2e]
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLBOX_SHA = "b7500b848f84ad7f86e277b4612563b69f4f88f9"
V2E_SHA = "cac638444557a224ba6ee5c29260916f152c00bd"
TOOLBOX_FILES = ["unsupervised_methods/utils.py", "unsupervised_methods/methods/POS_WANG.py",
                 "unsupervised_methods/methods/CHROME_DEHAAN.py", "evaluation/post_process.py"]


def fetch(url: str) -> bytes:
    """Download a URL and return its bytes."""
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def fetch_toolbox() -> None:
    """Download the Toolbox subset at the pinned commit and create empty package ``__init__`` files."""
    pkg = ROOT / "third_party" / "rppg_toolbox_ref" / "pkg"
    for rel in TOOLBOX_FILES:
        dest = pkg / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fetch(f"https://raw.githubusercontent.com/ubicomplab/rPPG-Toolbox/{TOOLBOX_SHA}/{rel}"))
        print("fetched", rel)
    for init in ("unsupervised_methods/__init__.py", "unsupervised_methods/methods/__init__.py", "evaluation/__init__.py"):
        (pkg / init).parent.mkdir(parents=True, exist_ok=True)
        (pkg / init).write_text("# empty: upstream __init__ star-imports methods we do not fetch\n")


def fetch_v2e() -> None:
    """Download and unpack v2e at the pinned commit as ``third_party/v2e_src/v2e-master``."""
    data = fetch(f"https://github.com/SensorsINI/v2e/archive/{V2E_SHA}.zip")
    out = ROOT / "third_party" / "v2e_src"
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        top = z.namelist()[0].split("/")[0]
        z.extractall(out)
    target = out / "v2e-master"
    if not target.exists():
        (out / top).rename(target)
    print("fetched v2e ->", target)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-v2e", action="store_true")
    args = ap.parse_args()
    fetch_toolbox()
    if not args.skip_v2e:
        fetch_v2e()


if __name__ == "__main__":
    main()
