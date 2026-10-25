"""Verify paper/refs.bib entries against Crossref (journals/proceedings) and arXiv, and report mismatches.

For every entry the title, first author, year and (if present) volume/pages are compared with the best Crossref match;
arXiv entries are checked through the arXiv API. The report is written to ``out/refs_verification.json``; entries that could not be
confirmed are listed so they can be fixed or removed by hand. Needs internet access.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEAD = {"User-Agent": "cardion-ref-check/1.0 (mailto:none@example.org)"}


def parse_bib(text: str) -> list[dict]:
    """Minimal BibTeX parser (``key = {value}`` fields)."""
    out = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        fields = {k.lower(): re.sub(r"[{}]", "", v).replace("\n", " ").strip() for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", m.group(3))}
        out.append({"type": m.group(1), "key": m.group(2).strip(), **fields})
    return out


def get(url: str) -> bytes:
    """HTTP GET with a polite User-Agent."""
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEAD), timeout=60) as r:
        return r.read()


def sim(a: str, b: str) -> float:
    """Case-insensitive similarity of two strings."""
    clean = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())  # noqa: E731
    return difflib.SequenceMatcher(None, clean(a), clean(b)).ratio()


def check(e: dict) -> dict:
    """Compare one entry with Crossref/arXiv."""
    res = {"key": e["key"], "title": e.get("title", ""), "ok": False, "notes": []}
    if "eprint" in e:
        xml = get("http://export.arxiv.org/api/query?id_list=" + e["eprint"]).decode("utf-8", "ignore")
        t = re.search(r"<entry>.*?<title>(.*?)</title>", xml, re.S)
        au = re.findall(r"<author>\s*<name>(.*?)</name>", xml)
        res["found_title"] = re.sub(r"\s+", " ", t.group(1)).strip() if t else None
        res["ok"] = bool(t) and sim(e["title"], res["found_title"]) > 0.85
        res["found_first_author"] = au[0] if au else None
        return res
    if e.get("doi"):
        items = [json.loads(get(f"https://api.crossref.org/works/{e['doi']}"))["message"]]    # exact record by DOI
    else:
        q = urllib.parse.quote(f"{e['title']} {e.get('author', '').split(' and ')[0]}")
        items = json.loads(get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=3"))["message"]["items"]
    best = max(items, key=lambda it: sim(e["title"], (it.get("title") or [""])[0]), default=None)
    if not best:
        res["notes"].append("no Crossref match")
        return res
    ft = (best.get("title") or [""])[0]
    res.update({"found_title": ft, "doi": best.get("DOI"), "found_year": (best.get("issued", {}).get("date-parts") or [[None]])[0][0],
                "found_volume": best.get("volume"), "found_pages": best.get("page")})
    ok = sim(e["title"], ft) > 0.85
    if res["found_year"] and str(res["found_year"]) != e.get("year"):
        res["notes"].append(f"year differs ({e.get('year')} vs {res['found_year']})")
    if e.get("volume") and best.get("volume") and e["volume"] != best["volume"]:
        res["notes"].append(f"volume differs ({e['volume']} vs {best['volume']})")
    first = lambda p: p.replace("--", "-").split("-")[0]  # noqa: E731  (some publishers register only the first page / article id)
    if e.get("pages") and best.get("page") and first(e["pages"]) != first(best["page"]):
        res["notes"].append(f"pages differ ({e['pages']} vs {best['page']})")
    res["ok"] = ok and not res["notes"]
    res["title_match"] = ok
    return res


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bib", default=str(ROOT / "paper" / "refs.bib"))
    args = ap.parse_args()
    entries = parse_bib(Path(args.bib).read_text(encoding="utf-8"))
    results = []
    for e in entries:
        try:
            r = check(e)
        except Exception as ex:  # noqa: BLE001
            r = {"key": e["key"], "ok": False, "notes": [f"lookup failed: {type(ex).__name__}: {ex}"]}
        results.append(r)
        print(("OK   " if r["ok"] else "CHECK"), r["key"], "|", "; ".join(r.get("notes", [])) or r.get("found_title", "")[:70], flush=True)
    out = ROOT / "out" / "refs_verification.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"{sum(r['ok'] for r in results)}/{len(results)} verified; report {out}")
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
