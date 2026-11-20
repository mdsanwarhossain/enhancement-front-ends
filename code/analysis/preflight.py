"""Check the manuscript is fit to submit.

Fails loudly on the things that are easy to miss and embarrassing to send: a
placeholder left in the text, a number with no result behind it, a citation
that never resolved, a figure referenced but never generated.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"

problems: list[str] = []
notes: list[str] = []


def check(condition, message, warn=False):
    if condition:
        return True
    (notes if warn else problems).append(message)
    return False


def pdf_text(pdf: Path) -> str:
    try:
        return subprocess.run(["pdftotext", str(pdf), "-"], check=True,
                              capture_output=True, text=True).stdout
    except Exception:
        return ""


def main() -> int:
    main_pdf = PAPER / "main.pdf"
    main_log = PAPER / "main.log"

    if not check(main_pdf.exists(), "paper/main.pdf has not been built"):
        return report()

    text = pdf_text(main_pdf)
    placeholders = sorted(set(re.findall(r"PENDING[-\w]*", text)))
    check(not placeholders,
          f"{len(placeholders)} placeholder(s) still in the PDF: "
          f"{', '.join(placeholders[:8])}")

    if main_log.exists():
        log = main_log.read_text(errors="ignore")
        n_ref = len(re.findall(r"LaTeX Warning: Reference", log))
        n_cite = len(re.findall(r"Citation `[^']+' (on page \d+ )?undefined", log))
        check(n_ref == 0, f"{n_ref} undefined cross-reference(s)")
        check(n_cite == 0, f"{n_cite} undefined citation(s)")
        bad = re.findall(r"^! (.+)$", log, re.M)
        check(not bad, f"{len(bad)} LaTeX error(s), first: "
                       f"{bad[0][:70] if bad else ''}")
        n_over = len(re.findall(r"Overfull \\hbox", log))
        check(n_over < 12, f"{n_over} overfull boxes - check the margins", warn=True)

    numbers = PAPER / "numbers.tex"
    if check(numbers.exists(), "paper/numbers.tex is missing"):
        pend = len(re.findall(r"\\PENDING", numbers.read_text())) - 1
        check(pend <= 0, f"{pend} macro(s) have no result behind them")

    bib = PAPER / "refs.bib"
    if check(bib.exists(), "paper/refs.bib is missing"):
        body = bib.read_text()
        unver = re.findall(r"@\w+\{([^,]+),[^@]*?PENDING", body)
        check(not unver,
              f"{len(unver)} unverified reference(s): {', '.join(unver[:6])}")
        keys = set(re.findall(r"@\w+\{([^,]+),", body))
        cited = set()
        for f in (PAPER / "sections").glob("*.tex"):
            for m in re.findall(r"\\cite\{([^}]+)\}", f.read_text()):
                cited.update(k.strip() for k in m.split(","))
        for m in re.findall(r"\\cite\{([^}]+)\}", (PAPER / "main.tex").read_text()):
            cited.update(k.strip() for k in m.split(","))
        check(cited <= keys,
              f"cited but not in refs.bib: {', '.join(sorted(cited - keys))}")
        unused = keys - cited
        check(not unused,
              f"{len(unused)} reference(s) in refs.bib are never cited: "
              f"{', '.join(sorted(unused))}", warn=True)

    figs = set()
    for f in list((PAPER / "sections").glob("*.tex")) + [PAPER / "main.tex"]:
        figs.update(re.findall(r"includegraphics\[[^]]*\]\{([^}]+)\}",
                               f.read_text()))
    for fig in sorted(figs):
        p = (PAPER / fig).resolve()
        check(p.with_suffix(".pdf").exists() or p.exists(),
              f"figure referenced but not generated: {fig}")

    for t in sorted((PAPER / "tables").glob("*.tex")):
        check(t.stat().st_size > 80, f"table looks empty: tables/{t.name}")

    pages = re.search(r"Pages:\s+(\d+)",
                      subprocess.run(["pdfinfo", str(main_pdf)],
                                     capture_output=True, text=True).stdout)
    if pages:
        notes.append(f"manuscript is {pages.group(1)} pages")
    abstract = re.search(r"Abstract\s+(.*?)\s*Keywords", text, re.S)
    if abstract:
        n = len(abstract.group(1).split())
        check(100 <= n <= 250, f"abstract is {n} words (Springer wants 150-250)",
              warn=not (100 <= n <= 250))
        notes.append(f"abstract is {n} words")

    return report()


def report() -> int:
    for n in notes:
        print(f"  note     {n}")
    for p in problems:
        print(f"  PROBLEM  {p}")
    if problems:
        print(f"\n{len(problems)} problem(s): not ready to submit")
        return 1
    print("\nready to submit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
