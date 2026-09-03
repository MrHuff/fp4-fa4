# Makefile for mxfp4_v2 persistent quantize kernel
NVCC := /usr/local/cuda/bin/nvcc

PYTHON := python3
PY_INC := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_LIB := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PY_VER := $(shell $(PYTHON) -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

PYBIND_INC := $(shell $(PYTHON) -m pybind11 --includes)
TORCH_INC := $(shell $(PYTHON) -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.include_paths()[0])")
TORCH_INC2 := $(shell $(PYTHON) -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.include_paths()[1])")
TORCH_LIB := $(shell $(PYTHON) -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.library_paths()[0])")

SRC := tk_quantize.cu
HEADERS := mxfp4_quantize.cuh mxfp4_group_quantize.cuh
OUT := mxfp4_quant_v2$(EXT_SUFFIX)

NVCCFLAGS := -std=c++20 -O3 \
    --expt-extended-lambda --expt-relaxed-constexpr \
    -forward-unknown-to-host-compiler \
    -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
    -DNDEBUG -lineinfo \
    -shared -fPIC \
    -diag-suppress 3189 -diag-suppress 177 \
    -D__CUDA_NO_HALF_OPERATORS__ \
    -D__CUDA_NO_HALF_CONVERSIONS__ \
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
    -D__CUDA_NO_HALF2_OPERATORS__ \
    -DTORCH_API_INCLUDE_EXTENSION_H \
    -DTORCH_EXTENSION_NAME=mxfp4_quant_v2 \
    -I$(PY_INC) \
    -I../nvfp4_v5 \
    $(PYBIND_INC) \
    -I$(TORCH_INC) -I$(TORCH_INC2) \
    -L$(TORCH_LIB) -L$(PY_LIB) \
    -l$(PY_VER) -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10 \
    -lcuda -lcudart_static -lcudadevrt -ldl \
    -gencode arch=compute_100a,code=sm_100a

all: $(OUT)

$(OUT): $(SRC) $(HEADERS)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(OUT)

clean:
	rm -f $(OUT)

.PHONY: all clean
