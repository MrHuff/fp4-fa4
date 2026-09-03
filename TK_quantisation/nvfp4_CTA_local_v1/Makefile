NVCC ?= /usr/local/cuda/bin/nvcc
OUT := _tk_quant_localcta$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

NVCCFLAGS := -std=c++17 -O3
NVCCFLAGS += --expt-relaxed-constexpr
NVCCFLAGS += -forward-unknown-to-host-compiler
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCCFLAGS += -DNDEBUG -lineinfo
NVCCFLAGS += -shared -fPIC
NVCCFLAGS += -diag-suppress 3189,3288
NVCCFLAGS += -Xcudafe --diag_suppress=3288
NVCCFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_tk_quant_localcta
NVCCFLAGS += -gencode arch=compute_100a,code=sm_100a

PYTHON_INCLUDES := $(shell python3 -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
PYBIND_INCLUDES := $(shell python3 -m pybind11 --includes)
PYTORCH_INCLUDES := $(shell python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
PYTORCH_LIBRARY_PATHS := $(shell python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join(library_paths()))")
PYTORCH_LIBDIR := $(foreach p,$(PYTORCH_LIBRARY_PATHS),-L$(p))
PYTORCH_RPATH := $(foreach p,$(PYTORCH_LIBRARY_PATHS),-Xlinker -rpath -Xlinker $(p))

NVCCFLAGS += $(PYTHON_INCLUDES) $(PYBIND_INCLUDES) $(PYTORCH_INCLUDES) $(PYTORCH_LIBDIR) $(PYTORCH_RPATH)
NVCCFLAGS += -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10

KERNEL_HEADERS := fused_localcta_quantize.cuh localcta_reconstruct.cuh
KERNEL_HEADERS += ../nvfp4_v5/silu_split_bf16.cuh

all: $(OUT)

$(OUT): tk_quantize.cu $(KERNEL_HEADERS)
	$(NVCC) tk_quantize.cu $(NVCCFLAGS) -o $(OUT)

clean:
	rm -f $(OUT)

.PHONY: all clean
