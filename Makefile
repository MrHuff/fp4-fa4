PYTHON ?= python3
CMAKE ?= cmake
CUDA_HOME ?= /usr/local/cuda
CUDACXX ?= $(CUDA_HOME)/bin/nvcc
BUILD_DIR ?= build
JOBS ?= 4

.PHONY: help inventory inventory-check verify-source test paper list-measurements \
	configure-gemm gemm benchmark-gemm build-plan build-sm100

help:
	@echo "FA4 continuation workspace targets:"
	@echo "  make inventory          Regenerate the tracked source inventory"
	@echo "  make inventory-check    Check the committed source inventory"
	@echo "  make verify-source      Run inventory and release integrity checks"
	@echo "  make test               Run the CPU/unit contract suite"
	@echo "  make paper              Regenerate all offline paper artifacts"
	@echo "  make list-measurements  List fresh measurement families"
	@echo "  make gemm               Build the two standalone SM100 CUTLASS controls"
	@echo "  make benchmark-gemm     Build and run the standalone SM100 controls"
	@echo "  make build-plan         Print the SM100 clean-build plan"
	@echo "  make build-sm100        Build and manifest the SM100 route matrix"
	@echo ""
	@echo "build-plan/build-sm100 require absolute FA4_BUILD_ROOT, CUDA_HOME, and CUTLASS_DSL_ROOT."

inventory:
	$(PYTHON) tools/generate_fa4_source_inventory.py

inventory-check:
	$(PYTHON) tools/generate_fa4_source_inventory.py --check

verify-source: inventory-check
	$(PYTHON) tools/verify_fa4_release.py

test:
	$(PYTHON) -m pytest -q

paper:
	$(PYTHON) tools/reproduce_fa4_paper.py --run --offline all

list-measurements:
	$(PYTHON) tools/plan_fa4_measurements.py list

configure-gemm:
	CUDACXX="$(CUDACXX)" $(CMAKE) -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release

gemm: configure-gemm
	$(CMAKE) --build "$(BUILD_DIR)" --parallel "$(JOBS)" --target \
		bench_nvfp4_gemm bench_grouped_nvfp4_gemm

benchmark-gemm: gemm
	"$(BUILD_DIR)/bench_nvfp4_gemm"
	"$(BUILD_DIR)/bench_grouped_nvfp4_gemm"

build-plan:
	@test -n "$(FA4_BUILD_ROOT)" || (echo "FA4_BUILD_ROOT is required" >&2; exit 2)
	@test -n "$(CUDA_HOME)" || (echo "CUDA_HOME is required" >&2; exit 2)
	@test -n "$(CUTLASS_DSL_ROOT)" || (echo "CUTLASS_DSL_ROOT is required" >&2; exit 2)
	$(PYTHON) tools/build_fa4.py plan --build-root "$(FA4_BUILD_ROOT)" --cuda-home "$(CUDA_HOME)" --cutlass-dsl-root "$(CUTLASS_DSL_ROOT)"

build-sm100:
	@test -n "$(FA4_BUILD_ROOT)" || (echo "FA4_BUILD_ROOT is required" >&2; exit 2)
	@test -n "$(CUDA_HOME)" || (echo "CUDA_HOME is required" >&2; exit 2)
	@test -n "$(CUTLASS_DSL_ROOT)" || (echo "CUTLASS_DSL_ROOT is required" >&2; exit 2)
	$(PYTHON) tools/build_fa4.py build --build-root "$(FA4_BUILD_ROOT)" --cuda-home "$(CUDA_HOME)" --cutlass-dsl-root "$(CUTLASS_DSL_ROOT)"
