# SageAttention3 SM100 Native Port Notes

SageAttention3's Blackwell native path is not a fair GB200 comparator yet. The
current `fp4attn_cuda` kernel instantiates SM120 register-fragment FP4 MMA:

- `SageAttention/sageattention3_blackwell/sageattn3/blackwell/cute_extension.h`
  defines `SM120_16x32x64_TN_VS_NVFP4` around
  `mma.sync.aligned.kind::mxf4nvf4.block_scale...`.
- `kernel_traits.h` wires that atom into both `TiledMmaQK` and `TiledMmaPV`.
- `mainloop_tma_ws.h` keeps QK score fragments in registers, runs fused online
  softmax/P quantization on those fragments, then immediately feeds register
  quantized P into the PV MMA.

On GB200/SM100, a forced native build fails in `ptxas` because the SM120
`mma.sync...block_scale` instruction is not supported for `sm_100a`. The SM100
equivalent is the `tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale...`
family, exposed by CuTe as `SM100_MMA_MXF4_SS`, but that atom writes
accumulators to TMEM rather than per-thread registers.

The native port is therefore a mainloop rewrite, not an atom rename:

1. Allocate TMEM for QK score accumulators and PV output accumulators.
2. Replace SA3's QK register-fragment `cute::gemm` loop with an SM100
   `tcgen05` QK issue path.
3. Load QK scores back from TMEM with `tcgen05.ld` into the layout expected by
   `SoftmaxFused`.
4. Preserve SA3's online softmax and dynamic P-scale quantization, but publish P
   in an SM100-friendly staging layout.
5. Replace SA3's PV register-fragment `cute::gemm` loop with SM100 `tcgen05`
   PV issue and TMEM accumulation.
6. Replace the current register-to-SMEM epilogue with a TMEM load/store
   epilogue, then store output.

There is also a shape mismatch for our local FP4-FA4 experiments: SA3's public
kernel assumes equal Q/K/V head dimensions of 64, 128, or 256, while the local
MXFP4 forward harness uses Q/K head dim 192 and V/O head dim 128. A fair native
SA3 comparator for this repo either needs a true SM100 port plus unequal
QK/V-head-dim support, or a separate SM100 tcgen05 attention kernel following
the SA3 algorithm.
