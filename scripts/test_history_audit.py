"""Document exactly what happened to the protected test split, from the repository history and the saved outputs.

    python scripts/test_history_audit.py

Checks (all offline):
1. The frozen artefacts (model checkpoints, app models, reliability-gate rule) at the "FREEZE before final test" commit are byte-identical to the files used now.
2. The first and the second test evaluation contain the same window-level estimates for every method present in both; only the summary rows differ,
   and by which amount (the first table reported window-pooled MAE of the first training seed instead of the seed-averaged subject-level MAE).
3. The code changes between the freeze commit and the commit that contains the final test run touch only summary/reporting code (listed).
Output: ``out/test_history_audit_<time>.json`` and claims ``testhist.*``.
"""
from __future__ import annotations

import glob
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import register_claim, timestamp  # noqa: E402

GIT = str(ROOT / "tools" / "mingit" / "cmd" / "git.exe") if (ROOT / "tools" / "mingit" / "cmd" / "git.exe").exists() else (shutil.which("git") or "git")
FREEZE = "38ed504"
LATER = "c80aed3"
CMD = "python scripts/test_history_audit.py"


def git(*args: str) -> bytes:
    """Run git in the repository and return stdout bytes."""
    return subprocess.run([GIT, *args], cwd=ROOT, capture_output=True, check=True).stdout


def have_history() -> bool:
    """True only inside the original development repository (the public release has no such history)."""
    try:
        git("cat-file", "-e", f"{FREEZE}^{{commit}}")
        git("cat-file", "-e", f"{LATER}^{{commit}}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def main() -> None:
    if not have_history():
        print("MAINTAINER-ONLY SCRIPT: the development git history (commits %s and %s) is not available here, so the test-split history cannot be re-derived.\n"
              "The evidence it produced is in docs/TEST_SPLIT_HISTORY.md and out/test_history_audit_*.json (saved copy). Nothing was changed." % (FREEZE, LATER))
        raise SystemExit(0)
    res: dict = {"command": CMD, "freeze_commit": FREEZE, "later_commit": LATER}
    res["freeze_commit_subject"] = git("log", "-1", "--format=%s", FREEZE).decode().strip()
    res["freeze_commit_time"] = git("log", "-1", "--format=%ci", FREEZE).decode().strip()
    same = {}
    for f in ("models/snn.pt", "models/ann.pt", "models/models.json", "configs/quality_calibrated.json"):
        old = git("show", f"{FREEZE}:{f}")
        new = (ROOT / f).read_bytes()
        if f.endswith((".json", ".yaml")):                       # text files: ignore Windows/Unix line-ending differences
            old, new = old.replace(b"\r\n", b"\n"), new.replace(b"\r\n", b"\n")
        same[f] = {"identical_to_freeze": hashlib.sha256(old).hexdigest() == hashlib.sha256(new).hexdigest(), "sha256": hashlib.sha256(new).hexdigest()}
    res["frozen_files"] = same
    changed = git("diff", "--name-only", FREEZE, LATER, "--", "src", "scripts", "configs", "models").decode().split()
    res["files_changed_between_freeze_and_final_test_commit"] = changed
    first = json.loads(Path(sorted(glob.glob(str(ROOT / "out" / "superseded" / "results_mcd-test_final_*.json")))[0]).read_text())
    second = json.loads(Path(sorted(glob.glob(str(ROOT / "out" / "results_mcd-test_final_*.json")))[-1]).read_text())
    identical, n_cmp = True, 0
    for ca, cb in zip(first["per_clip"], second["per_clip"]):
        assert ca["spec"] == cb["spec"]
        for m, wb in cb["windows_data"].items():
            if m in ca["windows_data"]:
                n_cmp += 1
                identical &= (ca["windows_data"][m]["bpm"] == wb["bpm"] and ca["windows_data"][m]["ref"] == wb["ref"])
    res["window_level_estimates_identical"] = bool(identical)
    res["n_method_clip_pairs_compared"] = n_cmp
    ra, rb = {r["method"]: r["mae"] for r in first["rows"]}, {r["method"]: r["mae"] for r in second["rows"]}
    res["first_run_family_mae"], res["second_run_family_mae"] = ra, rb
    res["max_abs_family_mae_change"] = max(abs(ra[k] - rb[k]) for k in rb)
    res["files"] = {"first": sorted(glob.glob(str(ROOT / "out" / "superseded" / "results_mcd-test_final_*.json")))[0].split("\\")[-1],
                    "second": sorted(glob.glob(str(ROOT / "out" / "results_mcd-test_final_*.json")))[-1].split("\\")[-1]}
    jp = ROOT / "out" / f"test_history_audit_{timestamp()}.json"
    jp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    reg = lambda k, v: register_claim(k, v, jp, CMD)  # noqa: E731
    reg("testhist.window_level_identical", int(identical))
    reg("testhist.n_pairs_compared", n_cmp)
    reg("testhist.max_abs_family_mae_change", res["max_abs_family_mae_change"])
    for m in ("snn_th0.02", "snn_th0.05", "snn_th0.1", "ann_th0.05", "rgb_pos", "event_pos@0.02"):
        reg(f"testhist.first.{m}", ra[m])
        reg(f"testhist.second.{m}", rb[m])
    reg("testhist.frozen_files_identical", int(all(v["identical_to_freeze"] for v in same.values())))
    print(json.dumps({k: v for k, v in res.items() if k not in ("first_run_family_mae", "second_run_family_mae")}, indent=1))
    print("first vs second family MAE (SNN/ANN):", {m: (round(ra[m], 2), round(rb[m], 2)) for m in ra if m.startswith(("snn", "ann"))})


if __name__ == "__main__":
    main()
