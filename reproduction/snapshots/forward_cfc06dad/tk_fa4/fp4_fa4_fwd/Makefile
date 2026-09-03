GPU := B200
SRC := fp4_fa4_fwd_experiments.cu
EXT_SUFFIX := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT := ../_C_b300_causal_fp4_fwd_experiments$(EXT_SUFFIX)
CMD := python3 -c "import tk_fa4._C_b300_causal_fp4_fwd_experiments"
CONFIG := pytorch
MXFP4_FWD_TIMELINE ?= 0
MXFP4_FWD_PCHAIN_STAMPS ?= 0
PCHAIN_TARGET_IDX ?= 0
KPIPE_STAGE ?= 0
SCORE_REUSE_PIPE_STAGE ?= 0
KPIPE_SELECTIVE_POLICY ?= 0
HOTPLATE_SLOT_SCHED ?= 0
HOTPLATE_POLICY ?= 0
POLICY126_COUNTERS ?= 0
NVCC_THREADS ?= 4
NVCC_SPLIT_COMPILE ?= 4
include ../../ThunderKittens/kernels/common.mk
NVCCFLAGS += -DTK_FA4_FORWARD_ONLY_BUILD -DTK_FA4_ENABLE_MXFP4_FWD_TIMELINE=$(MXFP4_FWD_TIMELINE) -DTK_FA4_ENABLE_MXFP4_FWD_PCHAIN_STAMPS=$(MXFP4_FWD_PCHAIN_STAMPS) -DTK_FA4_MXFP4_FWD_PCHAIN_TARGET_IDX=$(PCHAIN_TARGET_IDX) -DTK_FA4_MXFP4_FWD_KPIPE_STAGE=$(KPIPE_STAGE) -DTK_FA4_MXFP4_FWD_SCORE_REUSE_PIPE_STAGE=$(SCORE_REUSE_PIPE_STAGE) -DTK_FA4_MXFP4_FWD_KPIPE_SELECTIVE_POLICY=$(KPIPE_SELECTIVE_POLICY) -DTK_FA4_MXFP4_FWD_HOTPLATE_SLOT_SCHED=$(HOTPLATE_SLOT_SCHED) -DTK_FA4_MXFP4_FWD_HOTPLATE_POLICY=$(HOTPLATE_POLICY) -DTK_FA4_MXFP4_FWD_POLICY126_COUNTERS=$(POLICY126_COUNTERS) --threads $(NVCC_THREADS) --split-compile=$(NVCC_SPLIT_COMPILE) -UTORCH_EXTENSION_NAME -DTORCH_EXTENSION_NAME=_C_b300_causal_fp4_fwd_experiments

all: forward

forward: $(OUT)

$(OUT): $(SRC) \
	fwd_bf16_baseline.inc \
	stage2_ex2_alu_helpers.cuh \
	fwd_configs.inc \
	fwd_device_helpers.inc \
	fwd_streaming_kernel.inc \
	fwd_fused_stream_kernel.inc \
	fwd_host_bindings.inc \
	fwd_host_dispatch.inc \
	fwd_pybind.inc \
	shared_host_helpers.inc

.PHONY: all forward
