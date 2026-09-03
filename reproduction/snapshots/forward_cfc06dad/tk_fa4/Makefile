GPU := B200
SRC := tk_fa4.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python3 -c "import tk_fa4"
CONFIG := pytorch
NVCCFLAGS += -DTORCH_COMPILE
NVCCFLAGS += -I$(CURDIR)
include ../ThunderKittens/kernels/common.mk

BENCH_WARMUP ?= 5
BENCH_ITERS ?= 20
BENCH_BACKEND ?= both
BENCH_PASS ?= forward
BENCH_RUNNER := python3 benchmark.py --warmup $(BENCH_WARMUP) --iters $(BENCH_ITERS) --backend $(BENCH_BACKEND) --pass $(BENCH_PASS)

all: forward-modules legacy

.PHONY: forward-modules legacy experiments clean-all bench-smoke bench-flagship bench-sweep

forward-modules:
	$(MAKE) -C b300_causal_bf16_baseline
	$(MAKE) -C b300_causal
	$(MAKE) -C b300_noncausal

legacy:
	$(MAKE) -C deprecated

experiments:
	$(MAKE) -C b300_causal_fp4_experiments

clean-all:
	$(MAKE) clean
	$(MAKE) -C b300_causal_bf16_baseline clean
	$(MAKE) -C b300_causal clean
	$(MAKE) -C b300_noncausal clean
	$(MAKE) -C b300_causal_fp4_experiments clean
	$(MAKE) -C deprecated clean

bench-smoke:
	$(BENCH_RUNNER) --batches 1 --seqlens 2048 --heads 16 --causal-values 0,1

bench-flagship:
	$(BENCH_RUNNER) --batches 1 --seqlens 2048,4096,8192,16384 --heads 16 --causal-values 0,1

bench-sweep:
	$(BENCH_RUNNER) --batches 1,4 --seqlens 2048,4096,8192,16384,32768,65536 --heads 16 --causal-values 0,1
