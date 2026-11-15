"""Build paper/main.tex from paper/main.tex.in using ONLY numbers registered in paper/claims.json.

Placeholders: ``{{c:claim.key|.2f}}`` is replaced by the registered value (formatted); an unknown key is an error,
so no number can appear in the paper without a registered artifact + command. Latest table/figure artifacts are
copied from out/ into paper/. Then the paper is compiled with Tectonic when available.

    python scripts/build_paper.py [--no-compile]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"


def latest(pattern: str) -> Path:
    """Newest file in out/ matching a glob."""
    files = sorted(glob.glob(str(ROOT / "out" / pattern)))
    if not files:
        raise SystemExit(f"no artifact matches out/{pattern}")
    return Path(files[-1])


ARTIFACTS = {  # paper-relative destination -> glob in out/ (newest match wins)
    "figures/fig_pareto.png": "fig_pareto_mcd-test_final.png",
    "figures/fig_ablation_threshold.png": "fig_ablation_threshold_mcd-test_final.png",
    "figures/fig_bland_altman.png": "fig_bland_altman_mcd-test_final.png",
    "figures/fig_seeds.png": "fig_seeds_mcd-test_final.png",
    "figures/fig_subject_scatter.png": "fig_subject_scatter_mcd-test_final.png",
    "figures/fig_strata_state.png": "fig_strata_state_mcd-test_final.png",
    "figures/fig_diag_epoch_dynamics.png": "fig_diag_epoch_dynamics.png",
    "figures/fig_noise_study_amp0.2.png": "fig_noise_study_amp0.2.png",
    "tables/table_main.tex": "table_main_mcd-test_final.tex",
    "tables/table_efficiency.tex": "table_efficiency_mcd-test_final.tex",
    "tables/table_tests.tex": "table_tests_mcd-test_final.tex",
    "tables/table_strata_state.tex": "table_strata_state_mcd-test_final.tex",
    "tables/table_dataset.tex": "table_dataset.tex",
}


def fill(template: str, claims: dict) -> str:
    """Replace ``{{c:key|fmt}}`` / ``{{r:a/b|fmt}}`` placeholders; raise one KeyError listing every unregistered key."""
    missing: list[str] = []

    def val(key: str) -> float:
        if key not in claims:
            missing.append(key)
            return 1.0
        return claims[key]["value"]

    def sub(m: re.Match) -> str:
        key, _, fmt = m.group(1).partition("|")
        v = val(key)
        try:
            s = format(v, fmt) if fmt else str(v)
        except ValueError:                       # e.g. 'd' applied to a float claim: show it rounded instead
            s = format(float(v), ".0f")
        return s.replace("%", r"\%")             # a bare % would comment out the rest of the LaTeX line

    def ratio(m: re.Match) -> str:
        expr, _, fmt = m.group(1).partition("|")
        a, b = expr.split("/")
        return format(val(a) / val(b), fmt or ".2f")
    out = re.sub(r"\{\{r:([^}]+)\}\}", ratio, template)   # ratio of two registered claims: {{r:keyA/keyB|.1f}}
    out = re.sub(r"\{\{c:([^}]+)\}\}", sub, out)
    if missing:
        raise KeyError(f"{len(set(missing))} claims not registered in paper/claims.json: " + ", ".join(sorted(set(missing))))
    return out


LAY_SUMMARY = ("Can a heartbeat be read from webcam video after the video is turned into the sparse events of a neuromorphic camera? "
               "On real recordings, ordinary colour-based analysis was clearly more accurate than simulated events, and a spiking network gave no advantage. "
               "A system that shows a number only when confident looks far more accurate mostly because it skips hard moments.")


def plain(tex: str) -> str:
    """Rough LaTeX -> plain text for the Zenodo description."""
    tex = re.sub(r"\\emph\{([^}]*)\}", r"\1", tex)
    tex = tex.replace("\\ref{sec:setup}", "V").replace("\\%", "%")
    tex = tex.replace("\\,", " ").replace("~", " ").replace("\\_", "_").replace("$", "").replace("--", "-")
    return re.sub(r"\s+", " ", tex).strip()


def write_zenodo(text: str) -> None:
    """Write paper/README_zenodo.md (title, authors, abstract, keywords, licence) from the filled paper."""
    abstract = plain(re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S).group(1))
    title = plain(re.search(r"\\title\{(.*?)\}\n", text, re.S).group(1).replace("\\\\", " "))
    words = len(abstract.split())
    assert words < 200, f"abstract has {words} words"
    assert len(LAY_SUMMARY.split()) <= 60, "lay summary too long"
    md = (f"# {title}\n\n**Author:** Jacob Mullen. Preprint; not peer reviewed.\n\n"
          f"## Abstract ({words} words)\n\n{abstract}\n\n## Lay summary ({len(LAY_SUMMARY.split())} words)\n\n{LAY_SUMMARY}\n\n"
          "## Keywords\n\nremote photoplethysmography, event camera, spiking neural network, simulated events, reliability gating, reproducibility\n\n"
          "## License\n\nCode: MIT. Paper text and figures: CC BY 4.0 (recommended default; the author decides before upload).\n\n"
          "## AI-assistance disclosure\n\nSoftware, analysis and manuscript drafts were produced with AI assistance (Claude, Anthropic) under the author's direction; "
          "see the Declarations section of the paper. The author is responsible for the content.\n")
    (PAPER / "README_zenodo.md").write_text(md, encoding="utf-8")
    print(f"wrote paper/README_zenodo.md (abstract {words} words)")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()
    claims = json.loads((PAPER / "claims.json").read_text())
    for dest, pat in ARTIFACTS.items():
        try:
            src = latest(pat)
        except SystemExit as e:
            print("skip:", e)
            continue
        (PAPER / dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, PAPER / dest)
        print("copied", src.name, "->", dest)
    raw = (PAPER / "main.tex.in").read_text(encoding="utf-8-sig")
    raw = re.sub(r"@@([A-Z_]+)@@", lambda m: (PAPER / "sections" / f"{m.group(1).lower()}.tex.in").read_text(encoding="utf-8-sig"), raw)
    text = fill(raw, claims)
    (PAPER / "main.tex").write_text(text, encoding="utf-8")
    print("wrote paper/main.tex")
    write_zenodo(text)
    if args.no_compile:
        return
    ok = compile_latex()
    raise SystemExit(0 if ok else 1)


def find_tex() -> tuple[str, Path | None]:
    """Locate a LaTeX toolchain: portable MiKTeX in tools/, a system pdflatex, or Tectonic (in that order)."""
    miktex = ROOT / "tools" / "miktex" / "texmfs" / "install" / "miktex" / "bin" / "x64"
    if (miktex / "pdflatex.exe").exists():
        return "pdflatex", miktex
    if shutil.which("pdflatex"):
        return "pdflatex", None
    tect = ROOT / "tools" / "tectonic" / "tectonic.exe"
    return ("tectonic", tect.parent) if tect.exists() else ("none", None)


def compile_latex() -> bool:
    """pdflatex -> bibtex -> pdflatex x2 (or Tectonic); report undefined references/citations."""
    kind, d = find_tex()
    run = lambda *cmd: subprocess.run([str(c) for c in cmd], cwd=PAPER, capture_output=True, text=True)  # noqa: E731
    if kind == "none":
        print("no LaTeX toolchain found (install MiKTeX/TeX Live, or Tectonic); paper/main.tex is ready to compile")
        return False
    if kind == "tectonic":
        r = run(d / "tectonic.exe", "-X", "compile", "main.tex")
        log = r.stdout + r.stderr
    else:
        exe = (lambda n: d / f"{n}.exe") if d else (lambda n: n)
        for step in (("pdflatex", "-interaction=nonstopmode", "main.tex"), ("bibtex", "main"), ("pdflatex", "-interaction=nonstopmode", "main.tex"),
                     ("pdflatex", "-interaction=nonstopmode", "main.tex")):
            r = run(exe(step[0]), *step[1:])
        log = (PAPER / "main.log").read_text(encoding="utf-8", errors="ignore")
    problems = [l for l in log.splitlines() if re.search(r"undefined|Emergency stop|Fatal error|LaTeX Error", l)]
    pdf = PAPER / "main.pdf"
    print(f"toolchain: {kind}; PDF: {pdf.exists()} ({pdf.stat().st_size // 1024 if pdf.exists() else 0} KiB); problems: {problems if problems else 'none'}")
    return pdf.exists() and not problems



if __name__ == "__main__":
    main()

