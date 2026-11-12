#!/bin/sh
# Build ms.pdf locally (TeX Live / MacTeX). Overleaf runs the equivalent latexmk sequence automatically.
#   python make_paper_inputs.py   # refresh numbers.tex, tables/, figures/ from ../results
#   python build_bib.py           # only if the DOI list changes (needs api.crossref.org)
set -e
cd "$(dirname "$0")"
[ -d /Library/TeX/texbin ] && PATH=/Library/TeX/texbin:$PATH
pdflatex -interaction=nonstopmode -halt-on-error ms.tex >/dev/null
bibtex ms >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error ms.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error ms.tex >/dev/null
grep -E "^! |undefined|Rerun" ms.log || true
grep "Output written" ms.log
