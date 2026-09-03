GPU := B200
SRC := ../native_gqa_tk_bwd/v382_d64_gqa_e4m3_hkv2_register_pd.cu
EXT_SUFFIX := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUT := ../_C_b300_gqa_tk_v382_d64_e4m3$(EXT_SUFFIX)
CMD := python3 -c "import tk_fa4._C_b300_gqa_tk_v382_d64_e4m3"
CONFIG := pytorch
COMMON_MK := ../../ThunderKittens/kernels/common.mk

include $(COMMON_MK)

NVCCFLAGS += --threads 1 --split-compile=1 -I.. \
	-UTORCH_EXTENSION_NAME \
	-DTORCH_EXTENSION_NAME=_C_b300_gqa_tk_v382_d64_e4m3

$(OUT): \
	Makefile \
	$(COMMON_MK) \
	$(SRC) \
	../native_gqa_tk_bwd/v382_d64_gqa_e4m3_hkv2_register_pd.cuh \
	../native_gqa_tk_bwd/v382_d64_gqa_e4m3_owner.cuh \
	../deprecated/fa4_common.cuh

.PHONY: all
