#!/usr/bin/env bash
# Build the three files the journal asks for, and refuse to do it if the
# manuscript is not actually ready.
set -e
cd "$(dirname "$0")/.."
V=../.venv/bin/python
OUT=submission
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"

echo "--- building the manuscript ---"
cd paper
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode supplementary.tex >/dev/null
pdflatex -interaction=nonstopmode supplementary.tex >/dev/null
cd ..

echo "--- checking ---"
$V code/analysis/preflight.py

echo "--- packing ---"
rm -rf "$OUT"
mkdir -p "$OUT/figures"
cp paper/main.pdf paper/supplementary.pdf "$OUT/"
cp paper/cover-letter.md paper/instructions.txt "$OUT/"
cp results/figures/*.pdf "$OUT/figures/"

# the LaTeX source, self-contained, with figures where the manuscript expects
rm -rf build && mkdir -p build/results/figures build/paper
cp -r paper/sections paper/tables build/paper/
cp paper/main.tex paper/numbers.tex paper/refs.bib \
   paper/sn-jnl.cls paper/sn-basic.bst paper/sn-mathphys.bst build/paper/
cp results/figures/*.pdf build/results/figures/
(cd build && zip -qr ../"$OUT"/latex-source.zip paper results)

# verify the zip compiles on its own, with no path back to this project
rm -rf /tmp/sub-check && mkdir -p /tmp/sub-check
(cd /tmp/sub-check && unzip -q "$OLDPWD/$OUT/latex-source.zip" && cd paper \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null \
  && bibtex main >/dev/null 2>&1 \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null)
pages=$(pdfinfo /tmp/sub-check/paper/main.pdf | awk '/^Pages/{print $2}')
echo "latex-source.zip compiles standalone: $pages pages"
rm -rf build /tmp/sub-check

echo
echo "ready in $OUT/:"
ls -lh "$OUT" | tail -n +2 | awk '{printf "  %-20s %s\n", $9, $5}'
