.PHONY: all clean data

UNIFIED := ../fp4_fa4_unified_20260801
HAO_GRID := ../fp4_fa4_hao_table_gb200_20260802
B300 := ../fp4_fa4_b300_tuning_20260802
RECONSTRUCTION := ../fp4_fa4_reconstruction_20260805
SECTIONS := $(wildcard sections/*.tex) $(wildcard appendices/*.tex)
TABLES := $(UNIFIED)/tables/unified_macros.tex \
	$(UNIFIED)/tables/downstream_margin_macros.tex \
	$(UNIFIED)/tables/p_range_rows.tex \
	$(UNIFIED)/tables/format_rows.tex \
	$(UNIFIED)/tables/full_format_rows.tex \
	$(UNIFIED)/tables/cross_shape_rows.tex \
	$(UNIFIED)/tables/downstream_combined_rows.tex \
	$(UNIFIED)/tables/downstream_nvnv_failure_rows.tex \
	$(HAO_GRID)/tables/hao_grid_macros.tex \
	$(HAO_GRID)/tables/hao_grid_primary_rows.tex \
	$(B300)/tables/b300_tuning_macros.tex \
	$(B300)/tables/b300_tuning_rows.tex \
	$(B300)/tables/b300_d64_rows.tex \
	$(B300)/tables/primary_cross_generation_rows.tex \
	$(B300)/tables/accuracy_matched_rows.tex \
	$(RECONSTRUCTION)/tables/reconstruction_rows.tex
FIGURES := $(UNIFIED)/figures/headline_pareto.pdf \
	$(UNIFIED)/figures/cross_shape_speed_accuracy.pdf \
	$(RECONSTRUCTION)/nvmx_fast_report.png

all: main.pdf

main.pdf: main.tex main.bib graphcore_report.sty assets/graphcore-symbol.png $(SECTIONS) $(TABLES) $(FIGURES)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

data:
	python3 $(HAO_GRID)/build_summary.py
	python3 $(B300)/build_summary.py
	python3 $(UNIFIED)/build_summary.py
	python3 $(UNIFIED)/plot_summary.py
	python3 $(RECONSTRUCTION)/build_summary.py

clean:
	latexmk -C
