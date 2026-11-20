"""Build paper/refs.bib from verified records only.

Every entry is fetched from Crossref, arXiv or DataCite by DOI or title and
written from what the registry returns. Nothing is typed from memory, so a
reference that cannot be resolved is written as a loud PENDING entry rather
than quietly invented.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "refs.bib"
UA = {"User-Agent": "taq-bibliography-builder/1.0"}

# key -> DOI, or ("title", "<exact title>") to resolve by search
SOURCES: dict[str, object] = {}


def _get(url: str, tries: int = 3):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception:
            if attempt == tries - 1:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def by_doi(doi: str):
    return (_get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
            or {}).get("message")


def by_title(title: str):
    q = urllib.parse.quote(title)
    data = _get(f"https://api.crossref.org/works?query.bibliographic={q}&rows=5")
    if not data:
        return None
    want = re.sub(r"\W+", " ", title).strip().lower()
    for item in data["message"]["items"]:
        got = re.sub(r"\W+", " ", (item.get("title") or [""])[0]).strip().lower()
        if got == want:
            return item
    return None


def datacite(doi: str):
    """DataCite, for records that Crossref does not hold - datasets mostly."""
    d = _get(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}")
    if not d:
        return None
    a = d.get("data", {}).get("attributes", {})
    titles = a.get("titles") or []
    if not titles:
        return None
    return {"title": [titles[0].get("title", "")],
            "author": [{"literal": c.get("name", "")}
                       for c in (a.get("creators") or [])],
            "issued": {"date-parts": [[a.get("publicationYear")]]},
            "type": "dataset", "DOI": a.get("doi", doi),
            "container-title": [a.get("publisher", {}).get("name", "")
                                if isinstance(a.get("publisher"), dict)
                                else (a.get("publisher") or "")]}


def arxiv(arxiv_id: str):
    """arXiv's Atom feed, parsed just enough for a @misc entry."""
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=30) as r:
            xml = r.read().decode()
    except Exception:
        return None
    # the feed carries its own <title> and <id>; only the <entry> block is ours
    entry = re.search(r"<entry>(.*?)</entry>", xml, re.S)
    if not entry:
        return None
    body = entry.group(1)

    def tag(name):
        m = re.search(rf"<{name}>(.*?)</{name}>", body, re.S)
        return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
    authors = re.findall(r"<name>(.*?)</name>", body)
    if not tag("title"):
        return None
    return {"title": [tag("title")], "author": [{"literal": a} for a in authors],
            "issued": {"date-parts": [[int(tag("published")[:4] or 0)]]},
            "type": "posted-content", "DOI": f"10.48550/arXiv.{arxiv_id}",
            "container-title": [f"arXiv:{arxiv_id}"]}


def _authors(item) -> str:
    out = []
    for a in item.get("author", []) or []:
        if "literal" in a:
            out.append(a["literal"])
        else:
            out.append(f"{a.get('family','')}, {a.get('given','')}".strip(", "))
    return " and ".join(out) or "Unknown"


def _entry(key: str, item) -> str:
    if item is None:
        return (f"@misc{{{key},\n  title = {{PENDING: could not verify "
                f"'{key}' against any registry}},\n  note = {{PENDING}}\n}}\n")
    kind = {"journal-article": "article", "proceedings-article": "inproceedings",
            "book-chapter": "incollection", "book": "book",
            "posted-content": "misc", "dataset": "misc"}.get(
                item.get("type", ""), "misc")
    year = (item.get("issued", {}).get("date-parts") or [[None]])[0][0] or ""
    venue = (item.get("container-title") or [""])[0]
    fields = [f"  author = {{{_authors(item)}}}",
              f"  title = {{{(item.get('title') or [''])[0]}}}",
              f"  year = {{{year}}}"]
    if venue:
        fields.append(f"  {'journal' if kind == 'article' else 'booktitle'} = "
                      f"{{{venue}}}")
    for src, dst in (("volume", "volume"), ("page", "pages"), ("issue", "number")):
        if item.get(src):
            fields.append(f"  {dst} = {{{item[src]}}}")
    if item.get("DOI"):
        fields.append(f"  doi = {{{item['DOI']}}}")
    return "@%s{%s,\n%s\n}\n" % (kind, key, ",\n".join(fields))


def existing_entries(path: Path) -> dict[str, str]:
    """Keep what a previous run already verified; registries go down."""
    if not path.exists():
        return {}
    kept, text = {}, path.read_text()
    for block in text.split("\n@")[1:]:
        block = "@" + block
        key = re.search(r"@\w+\{([^,]+),", block)
        if key and "PENDING" not in block:
            kept[key.group(1)] = block.rstrip().rstrip("\n")
    return kept


def build(sources: dict, kept: dict[str, str] | None = None) -> str:
    kept = kept or {}
    out = ["% Generated by code/analysis/make_bib.py - every entry verified",
           "% against Crossref, DataCite or arXiv. Do not hand-edit.\n"]
    missing = []
    for key, spec in sources.items():
        if key in kept:
            out.append(kept[key] + "\n")
            print(f"  kept {key}", flush=True)
            continue
        if isinstance(spec, tuple) and spec[0] == "title":
            item = by_title(spec[1])
        elif isinstance(spec, tuple) and spec[0] == "arxiv":
            item = arxiv(spec[1])
        elif isinstance(spec, tuple) and spec[0] == "datacite":
            item = datacite(spec[1])
        else:
            item = by_doi(str(spec))
        if item is None:
            missing.append(key)
        out.append(_entry(key, item))
        print(("  ok   " if item else "  MISS ") + key, flush=True)
        time.sleep(0.4)
    if missing:
        print(f"\n{len(missing)} unresolved: {', '.join(missing)}")
    return "\n".join(out)


if __name__ == "__main__":
    spec_file = ROOT / "paper" / "sources.json"
    sources = json.loads(spec_file.read_text()) if spec_file.exists() else SOURCES
    sources = {k: (tuple(v) if isinstance(v, list) else v)
               for k, v in sources.items()}
    if not sources:
        sys.exit("no sources: write paper/sources.json first")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fresh = "--fresh" in sys.argv
    OUT.write_text(build(sources, {} if fresh else existing_entries(OUT)))
    print(f"wrote {OUT}")
