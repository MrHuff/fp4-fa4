.PHONY: all clean data data-from-raw

PYTHON ?= python3

# Freeze generated PDF metadata to the timestamp of the completed matched
# training receipt. Callers may override this for a later release.
SOURCE_DATE_EPOCH ?= 1788436743

UNIFIED := ../fp4_fa4_unified_20260801
HAO_GRID := ../fp4_fa4_hao_table_gb200_20260802
B300 := ../fp4_fa4_b300_tuning_20260802
RECONSTRUCTION := ../fp4_fa4_reconstruction_20260805
WAN := ../fp4_fa4_wan_cute_bf16_20260806
E2E_BATCH_SCALING := ../tk_fa4_8b_batch_scaling_20260901/e2e_batch_scaling_summary.json
SECTIONS := $(wildcard sections/*.tex) $(wildcard appendices/*.tex)
TABLES := $(UNIFIED)/tables/unified_macros.tex \
	$(UNIFIED)/tables/downstream_margin_macros.tex \
	$(UNIFIED)/tables/p_range_rows.tex \
	$(UNIFIED)/tables/full_format_rows.tex \
	$(UNIFIED)/tables/downstream_main_rows.tex \
	$(UNIFIED)/tables/downstream_nvnv_failure_rows.tex \
	$(HAO_GRID)/tables/hao_grid_macros.tex \
	$(B300)/tables/b300_tuning_macros.tex \
	$(B300)/tables/b300_tuning_rows.tex \
	$(B300)/tables/b300_d64_rows.tex \
	$(B300)/tables/primary_cross_generation_d128_rows.tex \
	$(B300)/tables/accuracy_matched_rows.tex \
	$(RECONSTRUCTION)/tables/reconstruction_rows.tex \
	$(WAN)/tables/wan_quality_speed_rows.tex
LOCAL_FIGURES := figures/causal_isolated_backward.pdf \
	figures/causal_combined_forward_backward.pdf \
	figures/llama8b_e2e_batch_scaling.pdf \
	figures/llama8b_training_curves.pdf \
	figures/llama8b_mxfp4_divergence.pdf \
	figures/llama8b_b4_matched_training_curves.pdf \
	figures/llama8b_b4_matched_throughput.pdf \
	figures/llama8b_b4_mxfp4_failure.pdf
RECEIPTS := $(wildcard receipts/*.json) $(E2E_BATCH_SCALING)
FIGURES := $(UNIFIED)/figures/headline_pareto.pdf \
	$(UNIFIED)/figures/cross_shape_speed_accuracy.pdf \
	$(RECONSTRUCTION)/nvmx_fast_report.png \
	$(LOCAL_FIGURES)

all: main.pdf

main.pdf: main.tex main.bib graphcore_report.sty assets/graphcore-symbol.png $(SECTIONS) $(TABLES) $(FIGURES) $(RECEIPTS)
	SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) FORCE_SOURCE_DATE=1 TZ=UTC \
		latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

data:
	$(PYTHON) $(HAO_GRID)/build_summary.py
	$(PYTHON) $(B300)/build_summary.py --from-summary
	$(PYTHON) $(UNIFIED)/build_summary.py
	$(PYTHON) $(UNIFIED)/plot_summary.py
	$(PYTHON) $(RECONSTRUCTION)/build_summary.py
	$(PYTHON) plot_causal_training.py

# Recompute the aggregate B300 summary when the separately distributed raw
# capture archive has been restored under $(B300)/artifacts/.
data-from-raw:
	$(PYTHON) $(B300)/build_summary.py

clean:
	latexmk -C
