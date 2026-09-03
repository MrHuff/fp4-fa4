GPU := B200
SRC := fp4_fa4_bwd.cu
EXT_SUFFIX := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT := ../_C_b300_causal_fp4_bwd_experiments$(EXT_SUFFIX)
CMD := python3 -c "import tk_fa4._C_b300_causal_fp4_bwd_experiments"
CONFIG := pytorch
include ../../ThunderKittens/kernels/common.mk
NVCCFLAGS += -DTK_FA4_BACKWARD_ONLY_BUILD --threads 1 --split-compile=1 -UTORCH_EXTENSION_NAME -DTORCH_EXTENSION_NAME=_C_b300_causal_fp4_bwd_experiments

all: backward

backward: $(OUT)

$(OUT): $(SRC) \
	../b300_causal/bf16_b300_mha_causal_fp4.cu \
	../b300_bwd_fa4.cuh \
	bwd_experiment_common.inc \
	bwd_experiment_score_quant.inc \
	bwd_experiment_streaming_live.inc \
	bwd_experiment_transpose_delta.inc \
	bwd_shared_host_helpers.inc \
	bwd_full_mxfp4_live_tensorcore.inc \
	bwd_key_owned_dkdv_fused_probe.inc \
	bwd_key_owned_mxfp4_schedule.inc \
	bwd_host_dispatch.inc \
	bwd_pybind.inc

.PHONY: all backward
