GPU := B200
SRC := tk_fa4.cu
OUT := _C_legacy$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python3 -c "from tk_fa4.deprecated import interface as _"
CONFIG := pytorch
NVCCFLAGS += -DTORCH_COMPILE
NVCCFLAGS += -I$(CURDIR)
include ../../ThunderKittens/kernels/common.mk
NVCCFLAGS += -UTORCH_EXTENSION_NAME -DTORCH_EXTENSION_NAME=_C_legacy
