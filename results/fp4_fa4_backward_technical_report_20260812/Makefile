.PHONY: all clean

SECTIONS := $(wildcard sections/*.tex) $(wildcard appendices/*.tex)
STYLE_ASSET := ../fp4_fa4_technical_report_20260728/assets/graphcore-symbol.png

all: main.pdf

main.pdf: main.tex main.bib graphcore_bwd_report.sty $(STYLE_ASSET) $(SECTIONS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C
