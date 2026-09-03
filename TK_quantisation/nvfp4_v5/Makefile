NVCC ?= /usr/local/cuda/bin/nvcc
OUT := _tk_quant_v5$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

NVCCFLAGS := -std=c++17 -O3
NVCCFLAGS += --expt-relaxed-constexpr
NVCCFLAGS += -forward-unknown-to-host-compiler
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCCFLAGS += -DNDEBUG -lineinfo
NVCCFLAGS += -shared -fPIC
NVCCFLAGS += -diag-suppress 3189,3288
NVCCFLAGS += -Xcudafe --diag_suppress=3288
NVCCFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_tk_quant_v5

# Architecture (GB200 / SM100)
NVCCFLAGS += -gencode arch=compute_100a,code=sm_100a

# Python/PyTorch
PYTHON_INCLUDES := $(shell python3 -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
PYBIND_INCLUDES := $(shell python3 -m pybind11 --includes)
PYTORCH_INCLUDES := $(shell python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
PYTORCH_LIBDIR := $(shell python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")

NVCCFLAGS += $(PYTHON_INCLUDES) $(PYBIND_INCLUDES) $(PYTORCH_INCLUDES) $(PYTORCH_LIBDIR)
NVCCFLAGS += -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10

KERNEL_HEADERS := core.cuh persistent_quantize.cuh persistent_group_quantize.cuh fused_amax_quantize.cuh
KERNEL_HEADERS += fused_group_amax_quantize.cuh amax_pipelined.cuh
KERNEL_HEADERS += quantize_transpose_tuned.cuh
KERNEL_HEADERS += group_quantize_transpose.cuh group_quantize_transpose_dim_1.cuh
KERNEL_HEADERS += fused_norm_quantize.cuh fused_silu_quantize.cuh persistent_silu_quantize.cuh fused_silu_deriv_quantize.cuh persistent_silu_deriv_quantize.cuh
KERNEL_HEADERS += persistent_norm_quantize.cuh

all: $(OUT)

$(OUT): tk_quantize.cu $(KERNEL_HEADERS)
	$(NVCC) tk_quantize.cu $(NVCCFLAGS) -o $(OUT)

clean:
	rm -f $(OUT)

.PHONY: all clean
