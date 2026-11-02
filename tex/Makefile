# Builds main.pdf. One target, nothing else (no clean, no watch) --
# intermediate pdflatex/bibtex artifacts are gitignored, not something
# this Makefile manages.
#
# Depends on figures/'s contents, so a regenerated figure triggers a
# rebuild even when its filename doesn't change. Figures and tables are
# regenerated separately, with `python -m nwpprec.plots` (see the README).

SOURCES := main.tex $(wildcard sections/*.tex) $(wildcard tables/*.tex) references.bib
FIGURES := $(wildcard ../figures/*)

main.pdf: $(SOURCES) $(FIGURES)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
