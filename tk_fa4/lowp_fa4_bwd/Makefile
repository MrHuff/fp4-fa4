GPU := B200
SRC := lowp_fa4_bwd.cu
EXT_SUFFIX := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT ?= ../_C_b300_lowp_bwd$(EXT_SUFFIX)
CMD := python3 -c "import tk_fa4._C_b300_lowp_bwd"
CONFIG := pytorch
include ../../ThunderKittens/kernels/common.mk

NVCCFLAGS += --threads 1 --split-compile=1 -I.. \
	-UTORCH_EXTENSION_NAME -DTORCH_EXTENSION_NAME=_C_b300_lowp_bwd

all: $(OUT)

$(OUT): $(SRC) \
	b300_bwd_cute16_candidate.cuh \
	b300_bwd_cute16_candidate_cute.cuh \
	b300_bwd_cute16_kernel_candidate.cuh \
	dq_projection_consumer.cuh \
	e4m3_to_mxfp4_v.cuh \
	gqa_d128_hierarchical_qkv_nvfp4_quantize.cuh \
	hierarchical_qkv_nvfp4_quantize.cuh \
	inverse_rope_nvfp4_quantize.cuh \
	projection_fp4_epilogue.cuh \
	rmsnorm_nvfp4_quantize.cuh \
	../../ThunderKittens/kernels/gemm/nvfp4_b200/nvfp4_quantize.cuh

.PHONY: all
