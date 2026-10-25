"""Write the real repository / archive links into the manuscript's availability statement and rebuild the arXiv package.

    python scripts/finalize_availability.py --repo-url https://github.com/<account>/CARDION --zenodo-doi 10.5281/zenodo.<number>

Run this ONLY after the repository and the Zenodo record exist and you have checked that both links open. It
* validates the URL and DOI patterns and refuses empty or placeholder-looking values (``<``, ``>``, ``XXXX``, ``OWNER``, ``example``),
* checks that the links resolve (HTTP 200) unless ``--no-check`` is given,
* rewrites ``paper/sections/availability.tex.in`` (the link in the paper) and adds ``repository-code``/``url``/``doi`` to ``release/CITATION.cff``,
  remembering the links in ``release/final_links.json`` so that ``make_release.py`` and ``verify_release.py`` then REQUIRE them everywhere,
* rebuilds the paper (``scripts/build_paper.ps1``) and then the staging area (``scripts/make_release.py``), which re-checks the finished archives.
Nothing is uploaded. Maintainer-only: see ``publication/PUBLISH_ORDER.md`` for the order that keeps GitHub, Zenodo, the PDF and arXiv identical.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BAD = re.compile(r"[<>]|XXXX|OWNER|example\.|placeholder|TODO", re.I)


def reachable(url: str) -> bool:
    """True if the URL answers with HTTP 200 (GitHub and doi.org may redirect)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except Exception as e:  # noqa: BLE001
        print("not reachable:", url, e)
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-url", required=True)
    ap.add_argument("--zenodo-doi", required=True, help="e.g. 10.5281/zenodo.1234567 - use the CONCEPT DOI (it also covers the v1.0.1 release that carries this paper)")
    ap.add_argument("--version", default="1.0.1", help="version of the release that will carry the final paper (v1.0.0 was archived first to obtain the DOI)")
    ap.add_argument("--no-check", action="store_true", help="skip the HTTP reachability check")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    if BAD.search(a.repo_url) or BAD.search(a.zenodo_doi):
        raise SystemExit("refusing: the values look like placeholders")
    if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", a.repo_url):
        raise SystemExit("repo URL must look like https://github.com/<account>/<repository>")
    if not re.fullmatch(r"10\.5281/zenodo\.\d+", a.zenodo_doi):
        raise SystemExit("Zenodo DOI must look like 10.5281/zenodo.<digits>")
    if not a.no_check and not (reachable(a.repo_url) and reachable("https://doi.org/" + a.zenodo_doi)):
        raise SystemExit("a link did not resolve; make the repository and record public first (or pass --no-check if you are sure)")
    text = (r"MCD-rPPG~\cite{egorov2025mcd} is a third-party dataset that is not redistributed with this work; obtain it from its authors. "
            r"Subject identifiers, the split manifest and checksums, the source code, trained checkpoints, the claim registry, the literature record, "
            r"the saved window-level results and all analysis scripts are available at \url{" + a.repo_url + r"} and archived at \url{https://doi.org/" + a.zenodo_doi + r"}.")
    (ROOT / "paper" / "sections" / "availability.tex.in").write_bytes((text + "\n").encode("utf-8"))        # no byte-order mark
    cff_path = ROOT / "release" / "CITATION.cff"
    cff = [ln for ln in cff_path.read_text(encoding="utf-8-sig").splitlines() if not re.match(r"(repository-code|doi|url):", ln)]
    i = next(k for k, ln in enumerate(cff) if ln.startswith("license:"))
    cff[i + 1:i + 1] = [f'repository-code: "{a.repo_url}"', f'url: "{a.repo_url}"', f'doi: "{a.zenodo_doi}"']
    cff = [f'version: "{a.version}"' if ln.startswith("version:") else ln for ln in cff]
    cff_path.write_bytes(("\n".join(cff) + "\n").encode("utf-8"))
    zen_path = ROOT / "release" / "zenodo.json"
    zen = json.loads(zen_path.read_text(encoding="utf-8-sig"))
    zen["version"] = a.version
    zen_path.write_bytes((json.dumps(zen, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    (ROOT / "release" / "final_links.json").write_bytes(json.dumps({"repo_url": a.repo_url, "doi": a.zenodo_doi}, indent=1).encode("utf-8"))
    print("availability statement, CITATION.cff and release/final_links.json written")
    if a.no_build:
        return
    py = sys.executable
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "build_paper.ps1")], check=True)
    subprocess.run([py, str(ROOT / "scripts" / "make_release.py"), "--version", a.version], check=True)
    print("done: re-check arxiv/ and the PDF before uploading")


if __name__ == "__main__":
    main()
