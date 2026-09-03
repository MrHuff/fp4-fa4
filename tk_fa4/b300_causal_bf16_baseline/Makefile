GPU := B200
SRC := ../b300_causal/bf16_b300_mha_causal.cu
OUT := ../_C_b300_causal_bf16_baseline$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python3 -c "import tk_fa4._C_b300_causal_bf16_baseline"
CONFIG := pytorch
include ../../ThunderKittens/kernels/common.mk
NVCCFLAGS += -UTORCH_EXTENSION_NAME -DTORCH_EXTENSION_NAME=_C_b300_causal_bf16_baseline
