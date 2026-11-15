"""Build the publication staging area (nothing is uploaded or published).

    python scripts/make_release.py [--out <dir>] [--version 1.0.0]

Creates, next to the working project (default ``../CARDION_release``):

* ``CARDION/``                 clean public-repository tree (code, tests, configs, models, split manifests, saved results used by the paper, paper sources, docs)
* ``CARDION-v<version>.zip``   the same tree as one archive (what a GitHub release / manual Zenodo upload would contain)
* ``arxiv/``  and ``CARDION_arxiv_submission.zip``   self-contained LaTeX submission (generated main.tex, main.bbl, figures, tables)
* ``publication_private/``     checklists and guides that must NOT go into the public repository (copied from ``publication/``)

The script refuses to finish if a local path, user name, e-mail address or secret-looking string is found in a staged text file, if a staged file is larger
than 25 MB, or if an arXiv file name violates arXiv's character rules. Excluded on purpose: virtual environments, tools, caches, logs, recordings, sessions,
raw/cached data, third-party code and models whose redistribution terms are unclear (see ``release/LICENSING.md``).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.build_paper import ARTIFACTS, latest  # noqa: E402

ROOT_FILES = ["LICENSE", "requirements.txt", "requirements-app.txt", "pytest.ini", "REPRODUCE.md", "Launch CARDION.bat", "Run CARDION Self-Test.bat",
              "Test and Repair CARDION.bat", "Move CARDION to a short folder.bat"]
DIRS = ["src", "scripts", "tests", "configs", "specs"]
DOCS = ["USER_GUIDE.md", "VALIDATION_PROTOCOL.md", "LITERATURE_REVIEW.md", "REFERENCES_VERIFIED.md", "PAPER_REVIEW.md", "REPRODUCIBILITY_REPORT.md"]
TEXT_EXT = {".py", ".md", ".ps1", ".bat", ".yaml", ".yml", ".json", ".cff", ".tex", ".in", ".txt", ".bib", ".html", ".js", ".css", ".ini", ".gitignore"}
# patterns are assembled from fragments so that this file does not itself contain the strings it searches for
FORBIDDEN = ["jm" + "700", "mullen" + "jacob", r"C:\\" + "Users", "C:/" + "Users", "App" + "Data", "Pine" + " Crest", r"(?i)api[_-]?key\s*[=:]\s*\S{8,}", r"(?i)password\s*[=:]\s*\S{6,}",
             r"ghp_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9]{20,}"]
MAX_MB = 25
LF_EXEMPT = {"out", "models", "data"}       # saved results and frozen files keep their original bytes
ARXIV_NAME = re.compile(r"^[A-Za-z0-9_+\-.,=/]+$")


def keep(rel: Path) -> bool:
    """Exclusion rules for files taken from the working tree."""
    parts = rel.parts
    if "__pycache__" in parts or rel.suffix == ".pyc" or ".pytest_cache" in parts:
        return False
    if parts[0] == "scripts" and rel.name.startswith("_") and rel.name != "__init__.py":
        return False                                    # scratch / one-off patch scripts
    return True


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def paper_assets(tex: str) -> tuple[list[str], list[str]]:
    """Figure and table files referenced by the generated main.tex."""
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    tabs = re.findall(r"\\input\{([^}]+)\}", tex)
    return figs, tabs


def write_test_history_doc(repo: Path) -> None:
    """docs/TEST_SPLIT_HISTORY.md: evidence about the protected test split that survives without the private development history (commit authors are omitted)."""
    import glob
    import hashlib
    import subprocess
    git = ROOT / "tools" / "mingit" / "cmd" / "git.exe"
    if not git.exists():
        git = Path(shutil.which("git") or "git")             # maintainer machine without the bundled MinGit
    audit = json.loads(Path(sorted(glob.glob(str(ROOT / "out" / "test_history_audit_*.json")))[-1]).read_text(encoding="utf-8"))
    log = subprocess.run([str(git), "log", "--reverse", "--format=%h | %ad | %s", "--date=format:%Y-%m-%d %H:%M:%S", "945879b^..c80aed3"], cwd=ROOT, capture_output=True, text=True).stdout
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    first = ROOT / "out" / "superseded" / audit["files"]["first"]
    second = ROOT / "out" / audit["files"]["second"]
    lines = ["# History of the protected test split", "",
             "The development repository (private, not published because its commit metadata contains personal information and a third-party model file) recorded the following; "
             "this file keeps the verifiable parts. Re-run `scripts/test_history_audit.py` inside the original repository to regenerate the numbers below.", "",
             f"* Freeze commit `{audit['freeze_commit']}` - \"{audit['freeze_commit_subject']}\" - {audit['freeze_commit_time']}.",
             f"* First test evaluation: `out/superseded/{audit['files']['first']}` (sha256 `{sha(first)}`).",
             f"* Second (reported) test evaluation: `out/{audit['files']['second']}` (sha256 `{sha(second)}`).",
             f"* Window-level estimates identical in both runs: **{audit['window_level_estimates_identical']}** ({audit['n_method_clip_pairs_compared']} method-clip pairs); "
             f"largest change of a per-family MAE between the two summary tables: {audit['max_abs_family_mae_change']:.2f} BPM (a reporting error in the first summary, see the paper).",
             "* Frozen files (identical to their versions in the freeze commit): " + ", ".join(f"`{k}` sha256 `{v['sha256'][:16]}...`" for k, v in audit["frozen_files"].items()) + ".",
             f"* Files changed between the freeze commit and the commit containing the second test run that belong to the experiment code: `src/evaluate.py`, `src/report.py` (summary and reporting only). "
             f"(Full list of changed files, including application and paper scripts: see `out/test_history_audit_*.json`.)", "",
             "Commits from the first real-data pipeline to the second test run (hash | time | subject; authors omitted):", "", "```", log.strip(), "```", ""]
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs" / "TEST_SPLIT_HISTORY.md").write_text("\n".join(lines), encoding="utf-8")


def build_repo(stage: Path, version: str) -> dict:
    """Assemble the public repository tree."""
    repo = stage / "CARDION"
    for f in ROOT_FILES:
        copy(ROOT / f, repo / f)
    for d in DIRS:
        for p in (ROOT / d).rglob("*"):
            if p.is_file() and keep(p.relative_to(ROOT)):
                copy(p, repo / p.relative_to(ROOT))
    for n in DOCS:
        copy(ROOT / "docs" / n, repo / "docs" / n)
    if (ROOT / "NOTES.md").exists():
        notes = [ln for ln in (ROOT / "NOTES.md").read_text(encoding="utf-8").splitlines() if "ISEF" not in ln and ("C:\\" + "Users") not in ln]
        (repo / "docs").mkdir(exist_ok=True)
        (repo / "docs" / "DEVELOPMENT_NOTES.md").write_text("# Development notes (deviations, substitutions, gotchas)\n\n" + "\n".join(notes) + "\n", encoding="utf-8")
    write_test_history_doc(repo)
    for n in ("snn.pt", "ann.pt", "models.json"):
        copy(ROOT / "models" / n, repo / "models" / n)
    copy(ROOT / "release" / "MODEL_CARD.md", repo / "models" / "README.md")
    copy(ROOT / "configs" / "default.yaml", repo / "configs" / "default.yaml")
    copy(ROOT / "data" / "README.md", repo / "data" / "README.md")
    for p in (ROOT / "data" / "splits").glob("*.json"):
        if "ubfc" not in p.name:                         # no UBFC videos were obtained; its manifest is not needed
            copy(p, repo / "data" / "splits" / p.name)
    # --- paper
    paper = ROOT / "paper"
    for n in ("main.tex.in", "main.tex", "main.pdf", "refs.bib", "literature.json"):
        copy(paper / n, repo / "paper" / n)
    copy(paper / "sections" / "availability.tex.in", repo / "paper" / "sections" / "availability.tex.in")
    tex = (paper / "main.tex").read_text(encoding="utf-8")
    figs, tabs = paper_assets(tex)
    for f in figs:
        copy(paper / "figures" / f, repo / "paper" / "figures" / f)
    for t in tabs:
        copy(paper / t, repo / "paper" / t)
    # --- saved results: every artifact that a registered claim points to, plus the sources build_paper.py copies
    claims = json.loads((paper / "claims.json").read_text(encoding="utf-8"))
    keep_claims, art = {}, set()
    for k, c in claims.items():
        a = ROOT / c["artifact"]
        if a.is_file() and a.stat().st_size < MAX_MB * 1e6 and not c["artifact"].startswith(("sessions", "logs", "validation_data")):
            keep_claims[k] = c
            art.add(c["artifact"])
    for pat in ARTIFACTS.values():
        try:
            art.add(str(latest(pat).relative_to(ROOT)).replace("\\", "/"))
        except SystemExit:
            pass
    for extra in ("out/superseded/results_mcd-test_final_20261024-010603.json", "out/superseded/results_mcd-test_final_20261024-010603.csv"):
        if (ROOT / extra).is_file():
            art.add(extra)
    for pat in ("test_history_audit_*.json", "window_reconciliation_*.json", "window_reconciliation_*.md", "paper_analysis_*.json", "discrepancy_*.json", "discrepancy_*.md",
                "results_mcd-val_val_all_*.json", "results_mcd-test_final_*.json", "quality_rule_test_*.json"):
        art.update(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "out").glob(pat))
    for a in sorted(art):
        copy(ROOT / a, repo / a)
    (repo / "paper" / "claims.json").write_text(json.dumps(keep_claims, indent=1), encoding="utf-8")
    # --- release templates
    for src, dst in (("README_public.md", "README.md"), ("LICENSING.md", "LICENSING.md"), ("CITATION.cff", "CITATION.cff"), ("zenodo.json", ".zenodo.json"),
                     ("gitignore_public", ".gitignore"), ("CC-BY-4.0.txt", "LICENSES/CC-BY-4.0.txt")):
        copy(ROOT / "release" / src, repo / dst)
    mit = (ROOT / "LICENSE").read_text(encoding="utf-8")
    (repo / "LICENSES").mkdir(exist_ok=True)
    (repo / "LICENSES" / "MIT.txt").write_text(mit, encoding="utf-8")
    (repo / "VERSION").write_text(version + "\n", encoding="utf-8")
    return {"n_claims_kept": len(keep_claims), "n_claims_dropped": len(claims) - len(keep_claims), "n_artifacts": len(art)}


def scrub(repo: Path) -> list[str]:
    """Normalise wording that identifies the author's circumstances, then scan every text file for forbidden strings."""
    repl = [("student researcher", "student researcher"), ("student", "student")]
    problems = []
    for p in repo.rglob("*"):
        if not p.is_file():
            continue
        if p.stat().st_size > MAX_MB * 1e6:
            problems.append(f"{p.relative_to(repo)}: larger than {MAX_MB} MB")
        if p.suffix.lower() in TEXT_EXT or p.name in (".gitignore", ".zenodo.json", "LICENSE", "VERSION"):
            raw = p.read_bytes()
            t = raw.decode("utf-8", errors="replace")
            t2 = t.lstrip("﻿")                                  # no byte-order marks (strict JSON parsers reject them)
            for a, b in repl:
                t2 = t2.replace(a, b)
            top = p.relative_to(repo).parts[0]
            if top not in LF_EXEMPT and p.suffix.lower() != ".bat":  # LF everywhere except saved results (their hashes are quoted) and Windows batch files
                t2 = t2.replace("\r\n", "\n")
            if t2.encode("utf-8") != raw:
                p.write_bytes(t2.encode("utf-8"))
            for pat in FORBIDDEN:
                m = re.search(pat, t2)
                if m:
                    ln = t2[:m.start()].count("\n") + 1
                    problems.append(f"{p.relative_to(repo)}:{ln}: matches {pat!r}")
    for p in repo.rglob("*.md"):                           # relative links in the Markdown files must point to files that exist in the public tree
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", p.read_text(encoding="utf-8", errors="replace")):
            if re.match(r"^(https?:|mailto:)", target):
                continue
            if not (p.parent / target).exists():
                problems.append(f"{p.relative_to(repo)}: broken relative link {target}")
    return problems


def zip_dir(src: Path, dst: Path, prefix: str) -> int:
    """Zip a directory with forward-slash names; returns the file count."""
    n = 0
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, f"{prefix}/{p.relative_to(src).as_posix()}" if prefix else p.relative_to(src).as_posix())
                n += 1
    return n


def build_arxiv(stage: Path) -> dict:
    """Self-contained arXiv source: generated main.tex, main.bbl, figures and tables only."""
    ax = stage / "arxiv"
    paper = ROOT / "paper"
    tex = (paper / "main.tex").read_text(encoding="utf-8")
    figs, tabs = paper_assets(tex)
    copy(paper / "main.tex", ax / "main.tex")
    copy(paper / "main.bbl", ax / "main.bbl")                   # arXiv requires the .bbl to carry the main file's name
    for f in figs:
        copy(paper / "figures" / f, ax / "figures" / f)
    for t in tabs:
        copy(paper / t, ax / t)
    bad = [str(p.relative_to(ax)) for p in ax.rglob("*") if p.is_file() and not ARXIV_NAME.match(p.relative_to(ax).as_posix())]
    n = zip_dir(ax, stage / "CARDION_arxiv_submission.zip", "")
    return {"files": n, "bad_names": bad, "figures": figs, "tables": tabs}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT.parent / "CARDION_release"))
    ap.add_argument("--version", default="1.0.0")
    a = ap.parse_args()
    stage = Path(a.out)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    info = build_repo(stage, a.version)
    problems = scrub(stage / "CARDION")
    info["problems"] = problems
    info["arxiv"] = build_arxiv(stage)
    problems += [f"arxiv file name not allowed: {b}" for b in info["arxiv"]["bad_names"]]
    n = zip_dir(stage / "CARDION", stage / f"CARDION-v{a.version}.zip", "CARDION")
    info["repo_zip_files"] = n
    pp = ROOT / "publication"
    if pp.exists():
        shutil.copytree(pp, stage / "publication_private")
    from scripts.verify_release import verify                  # inspects the finished archives, not just the staging folders
    fl = ROOT / "release" / "final_links.json"                  # written by finalize_availability.py once real links exist
    final = json.loads(fl.read_text(encoding="utf-8")) if fl.exists() else None
    vr = verify(stage / f"CARDION-v{a.version}.zip", stage / "CARDION_arxiv_submission.zip", ROOT / "release" / "README_public.md", final=final)
    info["archive_check"] = vr
    problems += [f"archive check: {p}" for p in vr["problems"]]
    info["problems"] = problems
    (stage / "release_manifest.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "arxiv"}, indent=1))
    print("arxiv:", {k: v for k, v in info["arxiv"].items() if k not in ("figures", "tables")})
    if problems:
        print("\nPROBLEMS (fix before publishing):")
        print("\n".join(problems))
        raise SystemExit(1)
    print("\nstaging area ready:", stage)


if __name__ == "__main__":
    main()
