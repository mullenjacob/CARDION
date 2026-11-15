"""Fetch authoritative bibliographic metadata (Crossref by DOI, arXiv API by id) so reference entries are built from the registry, not from memory.

    python scripts/fetch_bib.py --doi 10.1109/TBME.2016.2609282 --arxiv 2505.09529
    python scripts/fetch_bib.py --search "Are SNNs really more energy-efficient than ANNs"

Prints one JSON object per item (title, authors, venue, volume, issue, pages, year, doi/arxiv, type, status) and saves everything to
``out/fetch_bib_<time>.json``. ``--search`` lists the top Crossref matches for a title so the right DOI can be chosen by a human.
Needs internet access.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import timestamp  # noqa: E402

HEAD = {"User-Agent": "cardion-bib/1.0 (research reference check)"}


def get(url: str) -> bytes:
    """HTTP GET with a polite user agent."""
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEAD), timeout=40) as r:
        return r.read()


def clean(s: str) -> str:
    """Strip tags/extra spaces from a title."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


def crossref_item(m: dict) -> dict:
    """Normalise one Crossref work."""
    return {"source": "crossref", "doi": m.get("DOI"), "type": m.get("type"), "title": clean((m.get("title") or [""])[0]),
            "authors": [f"{a.get('family', a.get('name', ''))}, {a.get('given', '')}".strip(", ") for a in m.get("author", [])],
            "venue": clean((m.get("container-title") or [""])[0]), "volume": m.get("volume"), "issue": m.get("issue"), "pages": m.get("page"),
            "article_number": m.get("article-number"), "year": (m.get("issued", {}).get("date-parts") or [[None]])[0][0], "publisher": m.get("publisher"),
            "url": m.get("URL")}


def by_doi(doi: str) -> dict:
    """Crossref record for a DOI."""
    return crossref_item(json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"])


def by_arxiv(aid: str) -> dict:
    """arXiv API record (title, authors, submission year, journal reference if the authors gave one)."""
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(get("http://export.arxiv.org/api/query?id_list=" + aid))
    e = root.find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        return {"source": "arxiv", "arxiv": aid, "error": "not found"}
    jr = e.find("x:journal_ref", ns)
    doi = e.find("x:doi", ns)
    return {"source": "arxiv", "arxiv": aid, "title": clean(e.find("a:title", ns).text), "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
            "year": int(e.find("a:published", ns).text[:4]), "published": e.find("a:published", ns).text[:10], "journal_ref": jr.text if jr is not None else None,
            "doi": doi.text if doi is not None else None}


def search(title: str, rows: int = 4) -> list[dict]:
    """Top Crossref matches for a free-text title."""
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": rows})
    return [crossref_item(m) for m in json.loads(get("https://api.crossref.org/works?" + q))["message"]["items"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doi", nargs="*", default=[])
    ap.add_argument("--arxiv", nargs="*", default=[])
    ap.add_argument("--search", nargs="*", default=[])
    a = ap.parse_args()
    out = []
    for d in a.doi:
        try:
            out.append(by_doi(d))
        except Exception as e:  # noqa: BLE001
            out.append({"source": "crossref", "doi": d, "error": str(e)})
    for x in a.arxiv:
        try:
            out.append(by_arxiv(x))
        except Exception as e:  # noqa: BLE001
            out.append({"source": "arxiv", "arxiv": x, "error": str(e)})
    for s in a.search:
        try:
            out.append({"search": s, "matches": search(s)})
        except Exception as e:  # noqa: BLE001
            out.append({"search": s, "error": str(e)})
    for o in out:
        print(json.dumps(o, ensure_ascii=False))
    p = ROOT / "out" / f"fetch_bib_{timestamp()}.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved", p)


if __name__ == "__main__":
    main()
