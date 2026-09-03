GPU := B200
all: forward backward

forward:
	$(MAKE) -C ../fp4_fa4_fwd forward

backward:
	$(MAKE) -C ../fp4_fa4_bwd backward

.PHONY: all forward backward
