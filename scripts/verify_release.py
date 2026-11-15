"""Inspect the FINISHED release archives (not the scripts that made them).

    python scripts/verify_release.py CARDION-v1.0.0.zip CARDION_arxiv_submission.zip [--readme release/README_public.md] [--compile]

Checks on the repository archive: every entry under ``CARDION/``, forward slashes, no byte-order marks in text files, ``.zenodo.json`` parses with a strict
JSON parser, ``CITATION.cff`` parses and (when ``cffconvert`` is installed) validates against the CFF 1.2.0 schema, author is exactly Jacob Mullen without an
affiliation, versions agree, README equals the release template, no scratch/private/third-party files, no personal strings or paths, Python files compile,
Markdown links resolve, no case-insensitive duplicate names.
Checks on the arXiv archive: files at the archive root, arXiv file-name rule, ``main.tex`` + ``main.bbl``, every figure/table resolves, no hidden or auxiliary
files, ``main.tex`` identical to the repository copy; with ``--compile`` the sources are built in a clean temporary directory (needs pdflatex) and must have
no undefined references; with ``pypdf`` installed their text must equal the repository's ``paper/main.pdf``.
Before publication no GitHub/DOI link may appear in the paper or ``CITATION.cff``; after publication pass ``--repo-url`` and ``--doi`` and every artefact
(arXiv ``main.tex``, repository ``paper/main.tex``/PDF, ``CITATION.cff``) must carry exactly those links.
Exit status 1 if anything fails.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEXT_EXT = {".py", ".md", ".ps1", ".bat", ".yaml", ".yml", ".json", ".cff", ".tex", ".in", ".txt", ".bib", ".html", ".js", ".css", ".ini", ".bbl"}
TEXT_NAMES = {".gitignore", ".zenodo.json", "VERSION", "LICENSE"}
# strings assembled from fragments so this file does not contain them itself
FORBIDDEN = ["jm" + "700", "mullen" + "jacob", r"C:\\" + "Users", "C:/" + "Users", "App" + "Data", "Pine" + " Crest", r"(?i)api[_-]?key\s*[=:]\s*\S{8,}",
             r"(?i)password\s*[=:]\s*\S{6,}", r"ghp_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9]{20,}", "@" + "gmail", "@" + "hotmail"]
BAD_PARTS = {"__pycache__", ".pytest_cache", ".venv", "venv", "tools", "logs", "sessions", "validation_data", "third_party", ".git"}
BAD_NAMES = {"CLAUDE.md", "HANDOFF.md", "NOTES.md", "face_landmarker.task", ".env", "Thumbs.db", ".DS_Store"}
ARXIV_NAME = re.compile(r"^[A-Za-z0-9_+\-.,=/]+$")
MAX_BYTES = 25_000_000


def is_text(name: str) -> bool:
    """True for files that should be UTF-8 text."""
    p = Path(name)
    return p.suffix.lower() in TEXT_EXT or p.name in TEXT_NAMES


def check_repo(zpath: Path, readme_template: Path | None, final: dict | None = None) -> tuple[list[str], dict]:
    """All checks on the repository archive."""
    problems: list[str] = []
    info: dict = {}
    z = zipfile.ZipFile(zpath)
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    info["files"] = len(names)
    info["bytes"] = sum(i.file_size for i in z.infolist())
    lower: dict[str, str] = {}
    for n in names:
        if "\\" in n or not n.startswith("CARDION/"):
            problems.append(f"entry not under CARDION/ with forward slashes: {n}")
        parts = n.split("/")[1:]
        if BAD_PARTS & set(parts[:-1]) or parts[-1] in BAD_NAMES or n.endswith(".pyc"):
            problems.append(f"file that must not be published: {n}")
        if len(parts) > 1 and parts[0] == "scripts" and parts[-1].startswith("_") and parts[-1] != "__init__.py":
            problems.append(f"scratch script published: {n}")
        if len(n) > 200:
            problems.append(f"path longer than 200 characters: {n}")
        if n.lower() in lower and lower[n.lower()] != n:
            problems.append(f"names differing only by case: {n} / {lower[n.lower()]}")
        lower[n.lower()] = n
    for i in z.infolist():
        if i.file_size > MAX_BYTES:
            problems.append(f"larger than 25 MB: {i.filename}")
    texts: dict[str, str] = {}
    for n in names:
        if not is_text(n):
            continue
        raw = z.read(n)
        if raw.startswith(b"\xef\xbb\xbf"):
            problems.append(f"UTF-8 byte-order mark: {n}")
        try:
            t = raw.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"not valid UTF-8: {n}")
            continue
        texts[n] = t
        for pat in FORBIDDEN:
            m = re.search(pat, t)
            if m:
                problems.append(f"{n}:{t[:m.start()].count(chr(10)) + 1}: personal/forbidden string {pat!r}")
    # --- metadata
    zen = texts.get("CARDION/.zenodo.json")
    cff = texts.get("CARDION/CITATION.cff")
    version = texts.get("CARDION/VERSION", "").strip()
    info["version_file"] = version
    if zen is None:
        problems.append("missing .zenodo.json")
    else:
        try:
            zd = json.loads(zen)                                        # strict: no utf-8-sig
            info["zenodo_keys"] = sorted(zd)
            for k in ("title", "upload_type", "description", "creators", "license", "version"):
                if k not in zd:
                    problems.append(f".zenodo.json lacks {k}")
            if [c.get("name") for c in zd.get("creators", [])] != ["Mullen, Jacob"]:
                problems.append(f".zenodo.json creators unexpected: {zd.get('creators')}")
            if any("affiliation" in c or "orcid" in c for c in zd.get("creators", [])):
                problems.append(".zenodo.json creator has an affiliation/ORCID entry (author has none)")
            if zd.get("version") != version:
                problems.append(f".zenodo.json version {zd.get('version')} != VERSION {version}")
            for k in ("related_identifiers", "doi", "prereserve_doi"):
                if k in zd:
                    problems.append(f".zenodo.json contains {k}: must not be filled before the real DOI exists")
        except json.JSONDecodeError as e:
            problems.append(f".zenodo.json is not strict JSON: {e}")
    if cff is None:
        problems.append("missing CITATION.cff")
    else:
        try:
            import yaml
            cd = yaml.safe_load(cff)
            for k in ("cff-version", "message", "title", "authors", "version", "license"):
                if k not in cd:
                    problems.append(f"CITATION.cff lacks {k}")
            a = cd.get("authors", [])
            if len(a) != 1 or a[0].get("family-names") != "Mullen" or a[0].get("given-names") != "Jacob" or "affiliation" in a[0]:
                problems.append(f"CITATION.cff authors unexpected: {a}")
            if str(cd.get("version")) != version:
                problems.append(f"CITATION.cff version {cd.get('version')} != VERSION {version}")
            if final is None:
                for k in ("doi", "repository-code", "url", "identifiers"):
                    if k in cd:
                        problems.append(f"CITATION.cff contains {k}: add only real URLs/DOIs after publication")
            else:
                if cd.get("repository-code") != final["repo_url"]:
                    problems.append(f"CITATION.cff repository-code {cd.get('repository-code')} != {final['repo_url']}")
                if cd.get("doi") != final["doi"]:
                    problems.append(f"CITATION.cff doi {cd.get('doi')} != {final['doi']}")
        except Exception as e:                                          # noqa: BLE001
            problems.append(f"CITATION.cff does not parse: {e}")
        try:
            from cffconvert.citation import Citation
            Citation(cff).validate()
            info["cff_schema"] = "validated with cffconvert"
        except ImportError:
            info["cff_schema"] = "cffconvert not installed (structure checked only)"
        except Exception as e:                                          # noqa: BLE001
            problems.append(f"CITATION.cff fails the CFF schema: {str(e)[:200]}")
    # --- README is the public one
    if readme_template is not None and "CARDION/README.md" in texts:
        want = readme_template.read_bytes().decode("utf-8").lstrip("\ufeff").replace("\r\n", "\n")
        if texts["CARDION/README.md"].replace("\r\n", "\n") != want.replace("licence", "licence"):
            problems.append("README.md differs from release/README_public.md")
    for must in ("README.md", "LICENSE", "LICENSING.md", "REPRODUCE.md", "paper/main.pdf", "paper/main.tex", "paper/claims.json", "docs/TEST_SPLIT_HISTORY.md",
                 "LICENSES/CC-BY-4.0.txt", "LICENSES/MIT.txt", "requirements.txt", "models/snn.pt", "models/ann.pt"):
        if "CARDION/" + must not in names:
            problems.append(f"missing {must}")
    # --- Python files compile
    for n, t in texts.items():
        if n.endswith(".py"):
            try:
                ast.parse(t, filename=n)
            except SyntaxError as e:
                problems.append(f"syntax error in {n}: {e}")
    # --- Markdown links
    nameset = set(names)
    for n, t in texts.items():
        if not n.endswith(".md"):
            continue
        base = Path(n).parent
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", t):
            if re.match(r"^(https?:|mailto:)", target):
                continue
            tgt = (base / target).as_posix()
            tgt = re.sub(r"[^/]+/\.\./", "", tgt)
            if tgt not in nameset and not any(x.startswith(tgt + "/") for x in nameset):
                problems.append(f"{n}: broken relative link {target}")
    info["texts"] = len(texts)
    return problems, info


def paper_assets(tex: str) -> list[str]:
    """Files referenced from main.tex."""
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    return figs + re.findall(r"\\input\{([^}]+)\}", tex)


def check_arxiv(apath: Path, repo_zip: Path, compile_it: bool, final: dict | None = None) -> tuple[list[str], dict]:
    """All checks on the arXiv archive."""
    problems: list[str] = []
    info: dict = {}
    z = zipfile.ZipFile(apath)
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    info["files"] = len(names)
    for n in names:
        if not ARXIV_NAME.match(n):
            problems.append(f"arXiv file name not allowed: {n}")
        if n.startswith(".") or "/." in n or n.endswith((".aux", ".log", ".out", ".blg", ".toc", ".pdf")):
            problems.append(f"hidden/auxiliary/compiled file in arXiv archive: {n}")
        if n.startswith("CARDION/") or "\\" in n:
            problems.append(f"arXiv archive must have files at its root: {n}")
    for must in ("main.tex", "main.bbl"):
        if must not in names:
            problems.append(f"arXiv archive lacks {must}")
    if "main.tex" in names:
        tex = z.read("main.tex")
        if tex.startswith(b"\xef\xbb\xbf"):
            problems.append("main.tex has a byte-order mark")
        t = tex.decode("utf-8")
        has_repo = bool(re.search(r"github\.com", t))
        if final is None and has_repo:
            problems.append("main.tex contains a GitHub link, but no final links were declared (use --repo-url/--doi after publication)")
        if final is not None:
            for needle in (final["repo_url"], "https://doi.org/" + final["doi"]):
                if needle not in t.replace("\\", ""):
                    problems.append(f"main.tex does not contain {needle}")
        gp = re.search(r"\\graphicspath\{\{([^}]*)\}", t)
        gdir = gp.group(1) if gp else ""
        used = set()
        for ref in paper_assets(t):
            cands = [ref, ref + ".tex", gdir + ref, gdir + ref + ".png"]
            hit = [c for c in cands if c in names]
            if not hit:
                problems.append(f"main.tex references {ref}, not in the archive")
            used.update(hit)
        for n in names:
            if n not in ("main.tex", "main.bbl") and n not in used and Path(n).with_suffix("").as_posix() not in used:
                problems.append(f"unused file in arXiv archive: {n}")
        for pat in FORBIDDEN:
            if re.search(pat, t) or (re.search(pat, z.read("main.bbl").decode("utf-8", "replace")) if "main.bbl" in names else False):
                problems.append(f"personal/forbidden string {pat!r} in arXiv sources")
        au = re.search(r"\\author\{(.*)\}\s*\n", t)
        info["author_line"] = au.group(1)[:80] if au else None
        if not au or not au.group(1).startswith("Jacob Mullen") or re.search(r"affil|School|University|Institute|Independent", au.group(1)):
            problems.append(f"author line unexpected: {info['author_line']}")
        bbl = z.read("main.bbl").decode("utf-8", "replace") if "main.bbl" in names else ""
        info["bbl_entries"] = bbl.count("\\bibitem")
        rz = zipfile.ZipFile(repo_zip)
        if "CARDION/paper/main.tex" in rz.namelist():
            if rz.read("CARDION/paper/main.tex").decode("utf-8").replace("\r\n", "\n") != t.replace("\r\n", "\n"):
                problems.append("arXiv main.tex differs from the repository's paper/main.tex")
    if compile_it:
        problems += compile_arxiv(z, zipfile.ZipFile(repo_zip))
    return problems, info


def pdf_text(data: bytes) -> list[str]:
    """Whitespace-normalised text of every page (needs the optional ``pypdf``)."""
    from pypdf import PdfReader
    return [re.sub(r"\s+", " ", pg.extract_text() or "").strip() for pg in PdfReader(io.BytesIO(data)).pages]


def compile_arxiv(z: zipfile.ZipFile, repo: zipfile.ZipFile | None = None) -> list[str]:
    """Build the sources in a clean temporary directory; report errors and undefined references; compare the text with the repository's paper/main.pdf."""
    from scripts.build_paper import find_tex
    kind, d = find_tex()
    if kind != "pdflatex":
        return ["--compile requested but no pdflatex found"]
    with tempfile.TemporaryDirectory(prefix="ax") as tmp:
        z.extractall(tmp)
        exe = (lambda n: str(d / f"{n}.exe")) if d else (lambda n: n)
        for _ in range(3):
            subprocess.run([exe("pdflatex"), "-interaction=nonstopmode", "main.tex"], cwd=tmp, capture_output=True, text=True)
        log = (Path(tmp) / "main.log").read_text(encoding="utf-8", errors="ignore")
        bad = [l for l in log.splitlines() if re.search(r"undefined|Emergency stop|Fatal error|LaTeX Error|Citation .* undefined", l)]
        pdf = Path(tmp) / "main.pdf"
        if not pdf.exists():
            return ["arXiv sources did not produce main.pdf"] + bad[:5]
        n_pages = len(re.findall(rb"/Type\s*/Page\b", pdf.read_bytes()))
        print(f"arXiv compile: main.pdf {pdf.stat().st_size // 1024} KiB, {n_pages} pages, problems: {bad or 'none'}")
        out = bad[:10]
        if repo is not None and "CARDION/paper/main.pdf" in repo.namelist():
            try:
                a, b = pdf_text(pdf.read_bytes()), pdf_text(repo.read("CARDION/paper/main.pdf"))
                same = a == b
                print(f"arXiv-built PDF vs repository paper/main.pdf: {len(a)} / {len(b)} pages, text identical: {same}")
                if not same:
                    out.append("PDF built from the arXiv sources differs in text from the repository's paper/main.pdf (rebuild with scripts/build_paper.ps1 and re-stage)")
            except ImportError:
                print("pypdf not installed: PDF text comparison skipped (pip install pypdf)")
        return out


def verify(repo_zip: Path, arxiv_zip: Path, readme_template: Path | None = None, compile_it: bool = False, final: dict | None = None) -> dict:
    """Run every check; returns ``{"problems": [...], "repo": {...}, "arxiv": {...}}``. ``final={"repo_url", "doi"}`` declares the real links (post-publication stage)."""
    p1, i1 = check_repo(repo_zip, readme_template, final)
    p2, i2 = check_arxiv(arxiv_zip, repo_zip, compile_it, final)
    return {"problems": p1 + p2, "repo": i1, "arxiv": i2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo_zip", type=Path)
    ap.add_argument("arxiv_zip", type=Path)
    ap.add_argument("--readme", type=Path, default=ROOT / "release" / "README_public.md")
    ap.add_argument("--compile", action="store_true", help="also compile the arXiv sources in a clean directory")
    ap.add_argument("--repo-url", help="real GitHub URL (post-publication stage): the paper and CITATION.cff must carry it")
    ap.add_argument("--doi", help="real Zenodo DOI, e.g. 10.5281/zenodo.123 (post-publication stage)")
    a = ap.parse_args()
    if bool(a.repo_url) != bool(a.doi):
        raise SystemExit("give --repo-url and --doi together, or neither")
    final = {"repo_url": a.repo_url, "doi": a.doi} if a.repo_url else None
    res = verify(a.repo_zip, a.arxiv_zip, a.readme if a.readme.exists() else None, a.compile, final)
    print(json.dumps({"repo": res["repo"], "arxiv": res["arxiv"]}, indent=1))
    if res["problems"]:
        print("\nFAILED:")
        print("\n".join(res["problems"]))
        raise SystemExit(1)
    print("\nrelease archives: all checks passed")


if __name__ == "__main__":
    main()
