/*
 * Thin PyTorch C++ extension wrapping nvte_quantize_rmsnorm_silu
 * and nvte_quantize_rmsnorm from the modified Transformer Engine.
 *
 * Three modes of operation:
 *   1. fused_te_quantize_rmsnorm_silu  — amax provided externally (kernel-only)
 *   2. fused_te_quantize_rmsnorm       — amax provided externally, no activation
 *   3. fused_te_quantize_rmsnorm_silu_2pass — computes amax internally via Pass 1
 *
 * IMPORTANT FIX (IS_CACHED_ACT_OP):
 *   TE's nvfp4_transpose.cuh had IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS.
 *   This was wrong when RETURN_TRANSPOSE=false: the colwise pass (which populates
 *   the cache) is compiled out, so the rowwise pass read raw input.
 *   Fixed to: IS_CACHED_ACT_OP = COMPUTE_ACTIVATIONS && RETURN_TRANSPOSE.
 *   Now with RETURN_TRANSPOSE=false, the kernel takes the direct computation path
 *   (line 742 else branch) and computes activations inline. No dummy columnwise needed.
 *
 * Output scale_inv is in TE's padded layout:
 *   rowwise: (round_up(M, 128), round_up(K/16, 4))
 */
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/recipe.h>

#ifdef CUSTOM_QUANT_ENABLED
#include "custom_quantize.cuh"
#endif

using namespace transformer_engine;

// Thread-local flag set by the Python layer to enable custom quantisation path
static thread_local bool g_use_custom_quant = false;

// Dispatch helper: routes to custom or standard TE quantize
static inline void dispatch_nvte_quantize_v2(
    TensorWrapper &te_input, TensorWrapper &te_output,
    QuantizationConfigWrapper &quant_config, cudaStream_t stream) {
#ifdef CUSTOM_QUANT_ENABLED
  if (g_use_custom_quant) {
    custom_nvfp4_quantize_transpose(
        te_input.data(), te_output.data(), stream);
    return;
  }
#endif
  nvte_quantize_v2(te_input.data(), te_output.data(), quant_config, stream);
}

// RAII guard for setting g_use_custom_quant
struct CustomQuantGuard {
  bool prev;
  CustomQuantGuard(bool enable) : prev(g_use_custom_quant) { g_use_custom_quant = enable; }
  ~CustomQuantGuard() { g_use_custom_quant = prev; }
};

static inline int64_t round_up(int64_t val, int64_t mult) {
  return ((val + mult - 1) / mult) * mult;
}

// =========================================================================
// Helpers to build TE TensorWrappers
// =========================================================================
static TensorWrapper make_te_input(torch::Tensor input) {
  int64_t M = input.size(0), K = input.size(1);
  std::vector<size_t> in_shape = {(size_t)M, (size_t)K};
  return TensorWrapper(input.data_ptr(), in_shape, DType::kBFloat16);
}

// Rowwise-only output
static TensorWrapper make_te_output(
    torch::Tensor fp4_data, torch::Tensor scale_inv, torch::Tensor amax,
    int64_t M, int64_t K) {
  int64_t scale_rows = round_up(M, 128);
  int64_t scale_cols = round_up(K / 16, 4);

  std::vector<size_t> out_shape = {(size_t)M, (size_t)K};
  std::vector<size_t> si_shape = {(size_t)scale_rows, (size_t)scale_cols};

  TensorWrapper te_output(NVTE_NVFP4_1D_SCALING);
  te_output.set_rowwise_data(fp4_data.data_ptr(), DType::kFloat4E2M1, out_shape);
  te_output.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3, si_shape);
  te_output.set_amax(amax.data_ptr<float>(), DType::kFloat32, std::vector<size_t>{1});
  return te_output;
}

// Full rowwise + columnwise output (enables RETURN_TRANSPOSE=true in kernel)
static TensorWrapper make_te_output_full(
    torch::Tensor fp4_data, torch::Tensor scale_inv, torch::Tensor amax,
    torch::Tensor fp4_data_t, torch::Tensor scale_inv_t, torch::Tensor amax_t,
    int64_t M, int64_t K) {
  // Rowwise: shape (M, K), scales (round_up(M,128), round_up(K/16,4))
  int64_t r_scale_rows = round_up(M, 128);
  int64_t r_scale_cols = round_up(K / 16, 4);
  std::vector<size_t> r_shape = {(size_t)M, (size_t)K};
  std::vector<size_t> r_si_shape = {(size_t)r_scale_rows, (size_t)r_scale_cols};

  // Columnwise (transposed): shape (K, M), scales (round_up(K,128), round_up(M/16,4))
  int64_t c_scale_rows = round_up(K, 128);
  int64_t c_scale_cols = round_up(M / 16, 4);
  std::vector<size_t> c_shape = {(size_t)K, (size_t)M};
  std::vector<size_t> c_si_shape = {(size_t)c_scale_rows, (size_t)c_scale_cols};

  TensorWrapper te_output(NVTE_NVFP4_1D_SCALING);
  te_output.set_rowwise_data(fp4_data.data_ptr(), DType::kFloat4E2M1, r_shape);
  te_output.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3, r_si_shape);
  te_output.set_amax(amax.data_ptr<float>(), DType::kFloat32, std::vector<size_t>{1});
  te_output.set_columnwise_data(fp4_data_t.data_ptr(), DType::kFloat4E2M1, c_shape);
  te_output.set_columnwise_scale_inv(scale_inv_t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
  te_output.set_columnwise_amax(amax_t.data_ptr<float>(), DType::kFloat32, std::vector<size_t>{1});
  return te_output;
}

#if 0  // Disabled: nvte_quantize_rmsnorm / nvte_quantize_rmsnorm_silu not in current TE
// =========================================================================
// 1. quantize_rmsnorm_silu: fused RMSNorm + SiLU + NVFP4 quant (amax given)
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor>
fused_te_quantize_rmsnorm_silu(
    torch::Tensor input,        // (M, K) bf16
    torch::Tensor inv_rms,      // (M,) float32
    torch::Tensor norm_weight,  // (K,) bf16
    torch::Tensor amax          // (1,) float32 — precomputed
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  int64_t M = input.size(0), K = input.size(1);
  TORCH_CHECK(K % 16 == 0);

  auto stream = at::cuda::getCurrentCUDAStream();

  auto fp4_data = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(input.device()));
  int64_t scale_rows = round_up(M, 128);
  int64_t scale_cols = round_up(K / 16, 4);
  auto scale_inv = torch::empty({scale_rows, scale_cols},
                                torch::dtype(torch::kUInt8).device(input.device()));

  auto te_input = make_te_input(input);
  auto te_output = make_te_output(fp4_data, scale_inv, amax, M, K);
  QuantizationConfigWrapper quant_config;

  nvte_quantize_rmsnorm_silu(
      te_input.data(), te_output.data(), quant_config,
      inv_rms.data_ptr<float>(), norm_weight.data_ptr(), stream);

  return std::make_tuple(fp4_data, scale_inv);
}

// =========================================================================
// 2. quantize_rmsnorm: fused RMSNorm + NVFP4 quant (no activation, amax given)
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor>
fused_te_quantize_rmsnorm(
    torch::Tensor input, torch::Tensor inv_rms, torch::Tensor norm_weight,
    torch::Tensor amax
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  int64_t M = input.size(0), K = input.size(1);
  TORCH_CHECK(K % 16 == 0);

  auto stream = at::cuda::getCurrentCUDAStream();

  auto fp4_data = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(input.device()));
  int64_t scale_rows = round_up(M, 128);
  int64_t scale_cols = round_up(K / 16, 4);
  auto scale_inv = torch::empty({scale_rows, scale_cols},
                                torch::dtype(torch::kUInt8).device(input.device()));

  auto te_input = make_te_input(input);
  auto te_output = make_te_output(fp4_data, scale_inv, amax, M, K);
  QuantizationConfigWrapper quant_config;

  nvte_quantize_rmsnorm(
      te_input.data(), te_output.data(), quant_config,
      inv_rms.data_ptr<float>(), norm_weight.data_ptr(), stream);

  return std::make_tuple(fp4_data, scale_inv);
}

// =========================================================================
// 2b. quantize_rmsnorm_full: fused RMSNorm + NVFP4 quant (no activation)
//     Returns BOTH rowwise AND columnwise FP4 data (RETURN_TRANSPOSE=true)
//     Used for FFN w3 path where inv_rms + amax are reused from w1 Pass 1.
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_te_quantize_rmsnorm_full(
    torch::Tensor input,        // (M, K) bf16
    torch::Tensor inv_rms,      // (M,) float32
    torch::Tensor norm_weight,  // (K,) bf16
    torch::Tensor amax,         // (1,) float32 — precomputed (from w1 Pass 1)
    bool encode_centric = false
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  int64_t M = input.size(0), K = input.size(1);
  TORCH_CHECK(K % 16 == 0);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  // --- Rowwise output: (M, K) ---
  auto fp4_data = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(device));
  int64_t r_scale_rows = round_up(M, 128);
  int64_t r_scale_cols = round_up(K / 16, 4);
  auto scale_inv = torch::empty({r_scale_rows, r_scale_cols},
                                torch::dtype(torch::kUInt8).device(device));

  // --- Columnwise (transposed) output: (K, M) ---
  auto fp4_data_t = torch::empty({K, M / 2}, torch::dtype(torch::kUInt8).device(device));
  int64_t c_scale_rows = round_up(K, 128);
  int64_t c_scale_cols = round_up(M / 16, 4);
  auto scale_inv_t = torch::empty({c_scale_rows, c_scale_cols},
                                  torch::dtype(torch::kUInt8).device(device));

  // Copy amax for columnwise
  auto amax_t = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
  cudaMemcpyAsync(amax_t.data_ptr<float>(), amax.data_ptr<float>(),
                  sizeof(float), cudaMemcpyDeviceToDevice, stream);

  auto te_input = make_te_input(input);
  auto te_output = make_te_output_full(fp4_data, scale_inv, amax,
                                       fp4_data_t, scale_inv_t, amax_t,
                                       M, K);
  QuantizationConfigWrapper quant_config;
  // NOTE: set_encode_centric removed — not available in TE 2.14
  // if (encode_centric) {
  //   quant_config.set_encode_centric(true);
  // }

  nvte_quantize_rmsnorm(
      te_input.data(), te_output.data(), quant_config,
      inv_rms.data_ptr<float>(), norm_weight.data_ptr(), stream);

  return std::make_tuple(fp4_data, scale_inv, fp4_data_t, scale_inv_t);
}

// =========================================================================
// 3. quantize_rmsnorm_silu_2pass: compute amax internally (no Python overhead)
//
// Pass 1: Custom CUDA kernel computes inv_rms (per-row) + global_amax
//         by iterating data twice within one kernel launch:
//         - First loop: accumulate sum_sq → reduce → inv_rms
//         - Second loop: with inv_rms, compute rmsnorm+silu → accumulate amax
//         - atomicMax across blocks → global_amax
// Pass 2: TE kernel reads inv_rms + amax and produces NVFP4 output
// =========================================================================

// Forward declaration of the Pass 1 kernel launcher
// (defined in the .cu file compiled with nvcc below)
extern "C" void launch_fused_reduction_pass1(
    const void* x_ptr, const void* w_ptr,
    float epsilon, int rows, int cols,
    float* inv_rms_cache, float* global_amax_ptr,
    cudaStream_t stream);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_te_quantize_rmsnorm_silu_2pass(
    torch::Tensor input,        // (M, K) bf16
    torch::Tensor norm_weight,  // (K,) bf16
    float epsilon               // RMSNorm epsilon
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  int64_t M = input.size(0), K = input.size(1);
  TORCH_CHECK(K % 16 == 0);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  // Allocate outputs
  auto fp4_data = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(device));
  int64_t scale_rows = round_up(M, 128);
  int64_t scale_cols = round_up(K / 16, 4);
  auto scale_inv = torch::empty({scale_rows, scale_cols},
                                torch::dtype(torch::kUInt8).device(device));

  // Allocate intermediates (persistent for backward reuse)
  auto inv_rms = torch::empty({M}, torch::dtype(torch::kFloat32).device(device));
  auto amax = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

  // === Pass 1: Compute inv_rms + global_amax ===
  launch_fused_reduction_pass1(
      input.data_ptr(), norm_weight.data_ptr(),
      epsilon, (int)M, (int)K,
      inv_rms.data_ptr<float>(), amax.data_ptr<float>(),
      stream);

  // === Pass 2: TE kernel (RETURN_TRANSPOSE=false, direct path) ===
  auto te_input = make_te_input(input);
  auto te_output = make_te_output(fp4_data, scale_inv, amax, M, K);
  QuantizationConfigWrapper quant_config;

  nvte_quantize_rmsnorm_silu(
      te_input.data(), te_output.data(), quant_config,
      inv_rms.data_ptr<float>(), norm_weight.data_ptr(), stream);

  return std::make_tuple(fp4_data, scale_inv, inv_rms, amax);
}

// =========================================================================
// 3b. quantize_rmsnorm_silu_2pass_full: like 2pass but returns BOTH
//     rowwise AND columnwise FP4 data (enables RETURN_TRANSPOSE=true)
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
fused_te_quantize_rmsnorm_silu_2pass_full(
    torch::Tensor input,        // (M, K) bf16
    torch::Tensor norm_weight,  // (K,) bf16
    float epsilon,              // RMSNorm epsilon
    bool encode_centric         // encode-centric scaling?
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  int64_t M = input.size(0), K = input.size(1);
  TORCH_CHECK(K % 16 == 0);
  TORCH_CHECK(M % 32 == 0, "M must be multiple of 32 for TMA");
  TORCH_CHECK(K % 32 == 0, "K must be multiple of 32 for TMA");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  // --- Rowwise output: (M, K) ---
  auto fp4_data = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(device));
  int64_t r_scale_rows = round_up(M, 128);
  int64_t r_scale_cols = round_up(K / 16, 4);
  auto scale_inv = torch::empty({r_scale_rows, r_scale_cols},
                                torch::dtype(torch::kUInt8).device(device));

  // --- Columnwise (transposed) output: (K, M) ---
  auto fp4_data_t = torch::empty({K, M / 2}, torch::dtype(torch::kUInt8).device(device));
  int64_t c_scale_rows = round_up(K, 128);
  int64_t c_scale_cols = round_up(M / 16, 4);
  auto scale_inv_t = torch::empty({c_scale_rows, c_scale_cols},
                                  torch::dtype(torch::kUInt8).device(device));

  // Intermediates
  auto inv_rms = torch::empty({M}, torch::dtype(torch::kFloat32).device(device));
  auto amax = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
  auto amax_t = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

  // === Pass 1: Compute inv_rms + global_amax ===
  launch_fused_reduction_pass1(
      input.data_ptr(), norm_weight.data_ptr(),
      epsilon, (int)M, (int)K,
      inv_rms.data_ptr<float>(), amax.data_ptr<float>(),
      stream);

  // Use same amax for columnwise (kernel uses S_enc_rowwise when amax_colwise==nullptr
  // but here we explicitly provide it for correctness)
  // cudaMemcpyAsync to copy amax -> amax_t on device
  cudaMemcpyAsync(amax_t.data_ptr<float>(), amax.data_ptr<float>(),
                  sizeof(float), cudaMemcpyDeviceToDevice, stream);

  // === Pass 2: TE kernel with RETURN_TRANSPOSE=true ===
  auto te_input = make_te_input(input);
  auto te_output = make_te_output_full(fp4_data, scale_inv, amax,
                                       fp4_data_t, scale_inv_t, amax_t,
                                       M, K);

  QuantizationConfigWrapper quant_config;

  nvte_quantize_rmsnorm_silu(
      te_input.data(), te_output.data(), quant_config,
      inv_rms.data_ptr<float>(), norm_weight.data_ptr(), stream);

  return std::make_tuple(fp4_data, scale_inv, fp4_data_t, scale_inv_t,
                         inv_rms, amax, amax_t);
}

// =========================================================================
// 3c. quantize_rmsnorm_2pass_full: 2-pass NO-SILU variant
//     Pass 1: compute inv_rms + global_amax (no SiLU applied)
//     Pass 2: TE kernel with RETURN_TRANSPOSE=true (no activation)
//     Used for QKV forward in attention.
// =========================================================================

// Forward declaration of the no-SiLU Pass 1 kernel launcher
extern "C" void launch_fused_reduction_pass1_no_silu(
    const void* x_ptr, const void* w_ptr,
    float epsilon, int rows, int cols,
    float* inv_rms_cache, float* global_amax_ptr,
    cudaStream_t stream);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
fused_te_quantize_rmsnorm_2pass_full(
    torch::Tensor input,        // (M, K) bf16
    torch::Tensor norm_weight,  // (K,) bf16
    float epsilon,              // RMSNorm epsilon
    bool encode_centric         // encode-centric scaling?
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  int64_t M = input.size(0), K = input.size(1);
  TORCH_CHECK(K % 16 == 0);
  TORCH_CHECK(M % 32 == 0, "M must be multiple of 32 for TMA");
  TORCH_CHECK(K % 32 == 0, "K must be multiple of 32 for TMA");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  // --- Rowwise output: (M, K) ---
  auto fp4_data = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(device));
  int64_t r_scale_rows = round_up(M, 128);
  int64_t r_scale_cols = round_up(K / 16, 4);
  auto scale_inv = torch::empty({r_scale_rows, r_scale_cols},
                                torch::dtype(torch::kUInt8).device(device));

  // --- Columnwise (transposed) output: (K, M) ---
  auto fp4_data_t = torch::empty({K, M / 2}, torch::dtype(torch::kUInt8).device(device));
  int64_t c_scale_rows = round_up(K, 128);
  int64_t c_scale_cols = round_up(M / 16, 4);
  auto scale_inv_t = torch::empty({c_scale_rows, c_scale_cols},
                                  torch::dtype(torch::kUInt8).device(device));

  // Intermediates
  auto inv_rms = torch::empty({M}, torch::dtype(torch::kFloat32).device(device));
  auto amax = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
  auto amax_t = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

  // === Pass 1: Compute inv_rms + global_amax (NO SiLU) ===
  launch_fused_reduction_pass1_no_silu(
      input.data_ptr(), norm_weight.data_ptr(),
      epsilon, (int)M, (int)K,
      inv_rms.data_ptr<float>(), amax.data_ptr<float>(),
      stream);

  cudaMemcpyAsync(amax_t.data_ptr<float>(), amax.data_ptr<float>(),
                  sizeof(float), cudaMemcpyDeviceToDevice, stream);

  // === Pass 2: TE kernel (no activation) with RETURN_TRANSPOSE=true ===
  auto te_input = make_te_input(input);
  auto te_output = make_te_output_full(fp4_data, scale_inv, amax,
                                       fp4_data_t, scale_inv_t, amax_t,
                                       M, K);

  QuantizationConfigWrapper quant_config;

  nvte_quantize_rmsnorm(
      te_input.data(), te_output.data(), quant_config,
      inv_rms.data_ptr<float>(), norm_weight.data_ptr(), stream);

  return std::make_tuple(fp4_data, scale_inv, fp4_data_t, scale_inv_t,
                         inv_rms, amax, amax_t);
}
#endif  // Re-enabled TE rmsnorm functions


// Forward declaration of fused backward kernel (defined in fused_silu_rmsnorm_backward.cu)
extern "C" void launch_fused_silu_rmsnorm_backward(
    const void* grad_output, const void* d_add,
    const void* input,
    const void* weight, const float* cached_inv_rms,
    int rows, int cols,
    void* grad_input, float* dgamma,
    cudaStream_t stream);

// =========================================================================
// 4. fused_silu_rmsnorm_backward: SiLU' + RMSNorm backward + dgamma
//    Single fused CUDA kernel: computes grad_input + dgamma (via atomicAdd).
//    Key precision fix: expf (not __expf) for PyTorch-matching sigmoid.
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor>
fused_silu_rmsnorm_backward(
    torch::Tensor dx_proj,      // (M, K) bf16 — gradient from w1 dgrad GEMM
    torch::Tensor x_raw,        // (M, K) bf16 — raw input before RMSNorm
    torch::Tensor norm_weight,  // (K,) bf16 — gamma
    torch::Tensor inv_rms,      // (M,) float32 — cached from forward
    torch::Tensor d_add         // (M, K) bf16 — optional gradient from w3 path (empty = no add)
) {
  TORCH_CHECK(dx_proj.is_cuda() && dx_proj.is_contiguous());
  TORCH_CHECK(x_raw.is_cuda() && x_raw.is_contiguous());
  TORCH_CHECK(dx_proj.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(x_raw.scalar_type() == torch::kBFloat16);

  int64_t M = dx_proj.size(0), K = dx_proj.size(1);
  TORCH_CHECK(x_raw.size(0) == M && x_raw.size(1) == K);

  // d_add is optional — pass nullptr if empty
  const void* d_add_ptr = nullptr;
  if (d_add.numel() > 0) {
    TORCH_CHECK(d_add.is_cuda() && d_add.is_contiguous());
    TORCH_CHECK(d_add.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(d_add.size(0) == M && d_add.size(1) == K);
    d_add_ptr = d_add.data_ptr();
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = dx_proj.device();

  // Allocate outputs
  auto grad_input = torch::empty_like(x_raw);                                     // (M, K) bf16
  auto dgamma = torch::zeros({K}, torch::dtype(torch::kFloat32).device(device));   // (K,) f32, zeroed for atomicAdd

  launch_fused_silu_rmsnorm_backward(
      dx_proj.data_ptr(),
      d_add_ptr,
      x_raw.data_ptr(),
      norm_weight.data_ptr(),
      inv_rms.data_ptr<float>(),
      (int)M, (int)K,
      grad_input.data_ptr(),
      dgamma.data_ptr<float>(),
      stream);

  return std::make_tuple(grad_input, dgamma);
}

// Forward declaration of fused elementwise multiply + amax kernel
extern "C" void launch_elementwise_mul_amax(
    const void* h1, const void* h3, void* out,
    float* global_amax,
    int64_t numel, cudaStream_t stream);
extern void launch_scalar_scale(
    const float* src, float* dst, float scale, cudaStream_t stream);
extern void launch_dual_scalar_scale(
    const float* src1, float* dst1, const float* src2, float* dst2,
    float scale, cudaStream_t stream);
// Forward declaration of fused SiLU + multiply + amax kernel (corrected SwiGLU forward)
extern "C" void launch_fused_silu_mul_amax(
    const void* h1_raw, const void* h3, void* out,
    float* global_amax,
    int64_t numel, cudaStream_t stream);

// =========================================================================
// 5. fused_te_mul_quantize: h = h1 * h3 → NVFP4 quant (single C++ call)
//
// Eliminates Python overhead between elementwise mul and quantization.
// When swizzle_scales=true (TK path): returns TK-native tensors (fp4x2 dtype,
//   tiled FP8 scales) + precomputed sg = amax / 2688.
// Returns (fp4_data, scale_inv, fp4_data_t, scale_inv_t, amax, amax_t, sg)
// sg is empty when !swizzle_scales.
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
fused_te_mul_quantize(
    torch::Tensor h1,           // (M, H) bf16 — output of w1 GEMM (silu already applied)
    torch::Tensor h3,           // (M, H) bf16 — output of w3 GEMM
    bool encode_centric = false, // match NVFP4Quantizer default
    bool swizzle_scales = false  // write scales in TK-swizzled layout
) {
  TORCH_CHECK(h1.is_cuda() && h1.is_contiguous());
  TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
  TORCH_CHECK(h1.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(h1.sizes() == h3.sizes());

  int64_t M = h1.size(0), H = h1.size(1);
  TORCH_CHECK(H % 16 == 0);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = h1.device();

  // Step 1: Fused h = h1 * h3 + global amax in a single kernel pass
  auto h = torch::empty_like(h1);
  auto amax = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
  launch_elementwise_mul_amax(
      h1.data_ptr(), h3.data_ptr(), h.data_ptr(),
      amax.data_ptr<float>(),
      M * H, stream);

  // Step 2: Quantize h → NVFP4 (amax already computed in step 1)
  // Shared TE scale dimensions
  std::vector<size_t> r_shape = {(size_t)M, (size_t)H};
  std::vector<size_t> c_shape = {(size_t)H, (size_t)M};
  int64_t r_scale_rows = round_up(M, 128);
  int64_t r_scale_cols = round_up(H / 16, 4);
  int64_t c_scale_rows = round_up(H, 128);
  int64_t c_scale_cols = round_up(M / 16, 4);
  std::vector<size_t> r_si_shape = {(size_t)r_scale_rows, (size_t)r_scale_cols};
  std::vector<size_t> c_si_shape = {(size_t)c_scale_rows, (size_t)c_scale_cols};

  // Tile dims for TK-native layout
  int64_t ntm_r = M / 128, ntk_r = H / 64;   // rowwise
  int64_t ntm_c = H / 128, ntk_c = M / 64;   // columnwise

  torch::Tensor fp4_data, scale_inv, fp4_data_t, scale_inv_t;
  if (swizzle_scales) {
    fp4_data   = torch::empty({M, H / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    scale_inv  = torch::empty({ntm_r, ntk_r, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    fp4_data_t = torch::empty({H, M / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    scale_inv_t = torch::empty({ntm_c, ntk_c, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
  } else {
    fp4_data   = torch::empty({M, H / 2}, torch::dtype(torch::kUInt8).device(device));
    scale_inv  = torch::empty({r_scale_rows, r_scale_cols}, torch::dtype(torch::kUInt8).device(device));
    fp4_data_t = torch::empty({H, M / 2}, torch::dtype(torch::kUInt8).device(device));
    scale_inv_t = torch::empty({c_scale_rows, c_scale_cols}, torch::dtype(torch::kUInt8).device(device));
  }

  // Build TE output wrapper with SHARED amax for row + col
  TensorWrapper te_output(NVTE_NVFP4_1D_SCALING);
  float* amax_ptr = amax.data_ptr<float>();
  te_output.set_rowwise_data(fp4_data.data_ptr(), DType::kFloat4E2M1, r_shape);
  te_output.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3, r_si_shape);
  te_output.set_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
  te_output.set_columnwise_data(fp4_data_t.data_ptr(), DType::kFloat4E2M1, c_shape);
  te_output.set_columnwise_scale_inv(scale_inv_t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
  te_output.set_columnwise_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
  // REMOVED: if (swizzle_scales) te_output.set_with_gemm_swizzled_scales(true); // API removed in current TE

  // Quantize with pre-computed amax (no separate amax kernel needed)
  auto te_input = make_te_input(h);
  QuantizationConfigWrapper quant_config;
  dispatch_nvte_quantize_v2(te_input, te_output, quant_config, stream);

  // Precompute sc_global = amax / 2688 for TK (avoids Python aten::mul)
  torch::Tensor sg;
  if (swizzle_scales) {
    sg = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
    launch_scalar_scale(amax.data_ptr<float>(), sg.data_ptr<float>(),
                        1.0f / 2688.0f, stream);
  }

  return std::make_tuple(fp4_data, scale_inv, fp4_data_t, scale_inv_t,
                         amax, amax, sg);  // shared amax + sg
}


// Forward declaration of dual elementwise mul + amax kernel
extern "C" void launch_dual_elementwise_mul_amax(
    const void* dh, const void* h3, const void* h1,
    void* out1, void* out2,
    float* global_amax1, float* global_amax2,
    int64_t numel, cudaStream_t stream);
// Forward declaration of fused SiLU-derivative dual mul + amax kernel (corrected SwiGLU backward)
extern "C" void launch_fused_silu_deriv_dual_mul_amax(
    const void* dh, const void* h3, const void* h1_raw,
    void* out1, void* out2,
    float* global_amax1, float* global_amax2,
    int64_t numel, cudaStream_t stream);
// Forward declarations for strided SiLU kernels (read h1/h3 from h13 without .contiguous())
extern "C" void launch_fused_silu_mul_strided_amax(
    const void* h13, void* out,
    float* global_amax,
    int64_t M, int64_t H, cudaStream_t stream);
extern "C" void launch_fused_silu_deriv_dual_mul_strided_amax(
    const void* dh, const void* h13,
    void* out1, void* out2,
    float* global_amax1, float* global_amax2,
    int64_t M, int64_t H, cudaStream_t stream);
extern "C" void launch_fused_silu_deriv_dual_mul_strided_interleaved_amax(
    const void* dh, const void* h13,
    void* dh13,
    float* global_amax1, float* global_amax2,
    int64_t M, int64_t H, cudaStream_t stream);

// =========================================================================
// 5b. fused_te_dual_mul_quantize: dh1=dh*h3, dh3=dh*h1 → 2x NVFP4 quant
//
// Replaces 4 separate operations (2 element-wise mul + 2 quantize) with
// a single kernel for the element-wise multiply+amax, then 2 nvte_quantize_v2.
// When tk_swizzle=true: TK-native tensors + precomputed sg1, sg2.
//
// Returns (fp4_1, si_1, fp4_1t, si_1t, amax_1, amax_1t,
//          fp4_2, si_2, fp4_2t, si_2t, amax_2, amax_2t,
//          sg1, sg2)
// sg1/sg2 are empty when !tk_swizzle.
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
fused_te_dual_mul_quantize(
    torch::Tensor dh,           // (M, H) bf16 — gradient from w2
    torch::Tensor h3,           // (M, H) bf16 — cached h3 from forward
    torch::Tensor h1,           // (M, H) bf16 — cached h1 from forward
    bool encode_centric = false, // match NVFP4Quantizer default
    bool tk_swizzle = false      // write scales in TK-swizzled layout
) {
  TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
  TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
  TORCH_CHECK(h1.is_cuda() && h1.is_contiguous());
  TORCH_CHECK(dh.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1.sizes());

  int64_t M = dh.size(0), H = dh.size(1);
  TORCH_CHECK(H % 16 == 0);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = dh.device();

  // Step 1: Fused dual multiply + amax in a single kernel
  auto out1 = torch::empty_like(dh);  // dh * h3
  auto out2 = torch::empty_like(dh);  // dh * h1
  auto amax1 = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
  auto amax2 = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

  launch_dual_elementwise_mul_amax(
      dh.data_ptr(), h3.data_ptr(), h1.data_ptr(),
      out1.data_ptr(), out2.data_ptr(),
      amax1.data_ptr<float>(), amax2.data_ptr<float>(),
      M * H, stream);

  // Shared scale dimensions
  std::vector<size_t> r_shape = {(size_t)M, (size_t)H};
  std::vector<size_t> c_shape = {(size_t)H, (size_t)M};
  int64_t r_scale_rows = round_up(M, 128);
  int64_t r_scale_cols = round_up(H / 16, 4);
  int64_t c_scale_rows = round_up(H, 128);
  int64_t c_scale_cols = round_up(M / 16, 4);
  std::vector<size_t> r_si_shape = {(size_t)r_scale_rows, (size_t)r_scale_cols};
  std::vector<size_t> c_si_shape = {(size_t)c_scale_rows, (size_t)c_scale_cols};

  // Tile dims for TK-native layout
  int64_t ntm_r = M / 128, ntk_r = H / 64;
  int64_t ntm_c = H / 128, ntk_c = M / 64;

  // Step 2a: Quantize out1 (dh1 = dh * h3) → NVFP4
  torch::Tensor fp4_1, si_1, fp4_1t, si_1t;
  if (tk_swizzle) {
    fp4_1  = torch::empty({M, H / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    si_1   = torch::empty({ntm_r, ntk_r, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    fp4_1t = torch::empty({H, M / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    si_1t  = torch::empty({ntm_c, ntk_c, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
  } else {
    fp4_1  = torch::empty({M, H / 2}, torch::dtype(torch::kUInt8).device(device));
    si_1   = torch::empty({r_scale_rows, r_scale_cols}, torch::dtype(torch::kUInt8).device(device));
    fp4_1t = torch::empty({H, M / 2}, torch::dtype(torch::kUInt8).device(device));
    si_1t  = torch::empty({c_scale_rows, c_scale_cols}, torch::dtype(torch::kUInt8).device(device));
  }
  {
    TensorWrapper te_out1(NVTE_NVFP4_1D_SCALING);
    float* amax1_ptr = amax1.data_ptr<float>();
    te_out1.set_rowwise_data(fp4_1.data_ptr(), DType::kFloat4E2M1, r_shape);
    te_out1.set_rowwise_scale_inv(si_1.data_ptr(), DType::kFloat8E4M3, r_si_shape);
    te_out1.set_amax(amax1_ptr, DType::kFloat32, std::vector<size_t>{1});
    te_out1.set_columnwise_data(fp4_1t.data_ptr(), DType::kFloat4E2M1, c_shape);
    te_out1.set_columnwise_scale_inv(si_1t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
    te_out1.set_columnwise_amax(amax1_ptr, DType::kFloat32, std::vector<size_t>{1});
    // REMOVED: if (tk_swizzle) te_out1.set_with_gemm_swizzled_scales(true); // API removed in current TE

    auto te_in1 = make_te_input(out1);
    QuantizationConfigWrapper quant_config1;
    dispatch_nvte_quantize_v2(te_in1, te_out1, quant_config1, stream);
  }

  // Step 2b: Quantize out2 (dh3 = dh * h1) → NVFP4
  torch::Tensor fp4_2, si_2, fp4_2t, si_2t;
  if (tk_swizzle) {
    fp4_2  = torch::empty({M, H / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    si_2   = torch::empty({ntm_r, ntk_r, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    fp4_2t = torch::empty({H, M / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    si_2t  = torch::empty({ntm_c, ntk_c, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
  } else {
    fp4_2  = torch::empty({M, H / 2}, torch::dtype(torch::kUInt8).device(device));
    si_2   = torch::empty({r_scale_rows, r_scale_cols}, torch::dtype(torch::kUInt8).device(device));
    fp4_2t = torch::empty({H, M / 2}, torch::dtype(torch::kUInt8).device(device));
    si_2t  = torch::empty({c_scale_rows, c_scale_cols}, torch::dtype(torch::kUInt8).device(device));
  }
  {
    TensorWrapper te_out2(NVTE_NVFP4_1D_SCALING);
    float* amax2_ptr = amax2.data_ptr<float>();
    te_out2.set_rowwise_data(fp4_2.data_ptr(), DType::kFloat4E2M1, r_shape);
    te_out2.set_rowwise_scale_inv(si_2.data_ptr(), DType::kFloat8E4M3, r_si_shape);
    te_out2.set_amax(amax2_ptr, DType::kFloat32, std::vector<size_t>{1});
    te_out2.set_columnwise_data(fp4_2t.data_ptr(), DType::kFloat4E2M1, c_shape);
    te_out2.set_columnwise_scale_inv(si_2t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
    te_out2.set_columnwise_amax(amax2_ptr, DType::kFloat32, std::vector<size_t>{1});
    // REMOVED: if (tk_swizzle) te_out2.set_with_gemm_swizzled_scales(true); // API removed in current TE

    auto te_in2 = make_te_input(out2);
    QuantizationConfigWrapper quant_config2;
    dispatch_nvte_quantize_v2(te_in2, te_out2, quant_config2, stream);
  }

  // Precompute sc_global for TK
  torch::Tensor sg1, sg2;
  if (tk_swizzle) {
    sg1 = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
    sg2 = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
    launch_dual_scalar_scale(
        amax1.data_ptr<float>(), sg1.data_ptr<float>(),
        amax2.data_ptr<float>(), sg2.data_ptr<float>(),
        1.0f / 2688.0f, stream);
  }

  return std::make_tuple(
      fp4_1, si_1, fp4_1t, si_1t, amax1, amax1,   // shared amax
      fp4_2, si_2, fp4_2t, si_2t, amax2, amax2,   // shared amax
      sg1, sg2);
}

// =========================================================================
// 5c. fused_te_silu_mul_quantize: h = silu(h1_raw) * h3 → NVFP4 quant
//
// For the CORRECTED SwiGLU forward: SiLU is applied AFTER the W1 projection,
// not before. Uses the fused_silu_mul_amax CUDA kernel.
// Returns (fp4_data, scale_inv, fp4_data_t, scale_inv_t, amax, amax_t, sg)
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
fused_te_silu_mul_quantize(
    torch::Tensor h1_raw,       // (M, H) bf16 — raw W1 output (no activation)
    torch::Tensor h3,           // (M, H) bf16 — W3 output (gate)
    bool encode_centric = false,
    bool swizzle_scales = false
) {
  TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
  TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
  TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(h1_raw.sizes() == h3.sizes());

  int64_t M = h1_raw.size(0), H = h1_raw.size(1);
  TORCH_CHECK(H % 16 == 0);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = h1_raw.device();

  // Step 1: Fused h = silu(h1_raw) * h3 + global amax
  auto h = torch::empty_like(h1_raw);
  auto amax = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
  launch_fused_silu_mul_amax(
      h1_raw.data_ptr(), h3.data_ptr(), h.data_ptr(),
      amax.data_ptr<float>(),
      M * H, stream);

  // Step 2: Quantize h → NVFP4 (reuse same TE quantize path as fused_te_mul_quantize)
  std::vector<size_t> r_shape = {(size_t)M, (size_t)H};
  std::vector<size_t> c_shape = {(size_t)H, (size_t)M};
  int64_t r_scale_rows = round_up(M, 128);
  int64_t r_scale_cols = round_up(H / 16, 4);
  int64_t c_scale_rows = round_up(H, 128);
  int64_t c_scale_cols = round_up(M / 16, 4);
  std::vector<size_t> r_si_shape = {(size_t)r_scale_rows, (size_t)r_scale_cols};
  std::vector<size_t> c_si_shape = {(size_t)c_scale_rows, (size_t)c_scale_cols};

  int64_t ntm_r = M / 128, ntk_r = H / 64;
  int64_t ntm_c = H / 128, ntk_c = M / 64;

  torch::Tensor fp4_data, scale_inv, fp4_data_t, scale_inv_t;
  if (swizzle_scales) {
    fp4_data   = torch::empty({M, H / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    scale_inv  = torch::empty({ntm_r, ntk_r, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    fp4_data_t = torch::empty({H, M / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    scale_inv_t = torch::empty({ntm_c, ntk_c, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
  } else {
    fp4_data   = torch::empty({M, H / 2}, torch::dtype(torch::kUInt8).device(device));
    scale_inv  = torch::empty({r_scale_rows, r_scale_cols}, torch::dtype(torch::kUInt8).device(device));
    fp4_data_t = torch::empty({H, M / 2}, torch::dtype(torch::kUInt8).device(device));
    scale_inv_t = torch::empty({c_scale_rows, c_scale_cols}, torch::dtype(torch::kUInt8).device(device));
  }

  TensorWrapper te_output(NVTE_NVFP4_1D_SCALING);
  float* amax_ptr = amax.data_ptr<float>();
  te_output.set_rowwise_data(fp4_data.data_ptr(), DType::kFloat4E2M1, r_shape);
  te_output.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3, r_si_shape);
  te_output.set_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
  te_output.set_columnwise_data(fp4_data_t.data_ptr(), DType::kFloat4E2M1, c_shape);
  te_output.set_columnwise_scale_inv(scale_inv_t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
  te_output.set_columnwise_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
  // REMOVED: if (swizzle_scales) te_output.set_with_gemm_swizzled_scales(true); // API removed in current TE

  auto te_input = make_te_input(h);
  QuantizationConfigWrapper quant_config;
  dispatch_nvte_quantize_v2(te_input, te_output, quant_config, stream);

  torch::Tensor sg;
  if (swizzle_scales) {
    sg = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
    launch_scalar_scale(amax.data_ptr<float>(), sg.data_ptr<float>(),
                        1.0f / 2688.0f, stream);
  }

  return std::make_tuple(fp4_data, scale_inv, fp4_data_t, scale_inv_t,
                         amax, amax, sg);
}

// =========================================================================
// 5d. fused_te_silu_deriv_dual_mul_quantize: backward for corrected SwiGLU
//
// Given h = silu(h1_raw) * h3, computes:
//   dh1_raw = dh * h3 * silu'(h1_raw)  → NVFP4 quant
//   dh3     = dh * silu(h1_raw)        → NVFP4 quant
//
// Returns (fp4_1, si_1, fp4_1t, si_1t, amax_1, amax_1t,
//          fp4_2, si_2, fp4_2t, si_2t, amax_2, amax_2t,
//          sg1, sg2)
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
fused_te_silu_deriv_dual_mul_quantize(
    torch::Tensor dh,           // (M, H) bf16 — gradient from w2
    torch::Tensor h3,           // (M, H) bf16 — cached gate output from forward
    torch::Tensor h1_raw,       // (M, H) bf16 — cached raw W1 output (pre-silu)
    bool encode_centric = false,
    bool tk_swizzle = false
) {
  TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
  TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
  TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
  TORCH_CHECK(dh.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(dh.sizes() == h3.sizes() && dh.sizes() == h1_raw.sizes());

  int64_t M = dh.size(0), H = dh.size(1);
  TORCH_CHECK(H % 16 == 0);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = dh.device();

  // Step 1: Fused SiLU-derivative dual multiply + amax
  auto out1 = torch::empty_like(dh);  // dh * h3 * silu'(h1_raw)
  auto out2 = torch::empty_like(dh);  // dh * silu(h1_raw)
  auto amax1 = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
  auto amax2 = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

  launch_fused_silu_deriv_dual_mul_amax(
      dh.data_ptr(), h3.data_ptr(), h1_raw.data_ptr(),
      out1.data_ptr(), out2.data_ptr(),
      amax1.data_ptr<float>(), amax2.data_ptr<float>(),
      M * H, stream);

  // Step 2: Quantize both outputs → NVFP4 (same structure as fused_te_dual_mul_quantize)
  std::vector<size_t> r_shape = {(size_t)M, (size_t)H};
  std::vector<size_t> c_shape = {(size_t)H, (size_t)M};
  int64_t r_scale_rows = round_up(M, 128);
  int64_t r_scale_cols = round_up(H / 16, 4);
  int64_t c_scale_rows = round_up(H, 128);
  int64_t c_scale_cols = round_up(M / 16, 4);
  std::vector<size_t> r_si_shape = {(size_t)r_scale_rows, (size_t)r_scale_cols};
  std::vector<size_t> c_si_shape = {(size_t)c_scale_rows, (size_t)c_scale_cols};

  int64_t ntm_r = M / 128, ntk_r = H / 64;
  int64_t ntm_c = H / 128, ntk_c = M / 64;

  // Quantize out1 (dh1_raw = dh * h3 * silu'(h1_raw)) → NVFP4
  torch::Tensor fp4_1, si_1, fp4_1t, si_1t;
  if (tk_swizzle) {
    fp4_1  = torch::empty({M, H / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    si_1   = torch::empty({ntm_r, ntk_r, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    fp4_1t = torch::empty({H, M / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    si_1t  = torch::empty({ntm_c, ntk_c, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
  } else {
    fp4_1  = torch::empty({M, H / 2}, torch::dtype(torch::kUInt8).device(device));
    si_1   = torch::empty({r_scale_rows, r_scale_cols}, torch::dtype(torch::kUInt8).device(device));
    fp4_1t = torch::empty({H, M / 2}, torch::dtype(torch::kUInt8).device(device));
    si_1t  = torch::empty({c_scale_rows, c_scale_cols}, torch::dtype(torch::kUInt8).device(device));
  }
  {
    TensorWrapper te_out1(NVTE_NVFP4_1D_SCALING);
    float* amax1_ptr = amax1.data_ptr<float>();
    te_out1.set_rowwise_data(fp4_1.data_ptr(), DType::kFloat4E2M1, r_shape);
    te_out1.set_rowwise_scale_inv(si_1.data_ptr(), DType::kFloat8E4M3, r_si_shape);
    te_out1.set_amax(amax1_ptr, DType::kFloat32, std::vector<size_t>{1});
    te_out1.set_columnwise_data(fp4_1t.data_ptr(), DType::kFloat4E2M1, c_shape);
    te_out1.set_columnwise_scale_inv(si_1t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
    te_out1.set_columnwise_amax(amax1_ptr, DType::kFloat32, std::vector<size_t>{1});
    // REMOVED: if (tk_swizzle) te_out1.set_with_gemm_swizzled_scales(true); // API removed in current TE

    auto te_in1 = make_te_input(out1);
    QuantizationConfigWrapper quant_config1;
    dispatch_nvte_quantize_v2(te_in1, te_out1, quant_config1, stream);
  }

  // Quantize out2 (dh3 = dh * silu(h1_raw)) → NVFP4
  torch::Tensor fp4_2, si_2, fp4_2t, si_2t;
  if (tk_swizzle) {
    fp4_2  = torch::empty({M, H / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    si_2   = torch::empty({ntm_r, ntk_r, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    fp4_2t = torch::empty({H, M / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    si_2t  = torch::empty({ntm_c, ntk_c, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
  } else {
    fp4_2  = torch::empty({M, H / 2}, torch::dtype(torch::kUInt8).device(device));
    si_2   = torch::empty({r_scale_rows, r_scale_cols}, torch::dtype(torch::kUInt8).device(device));
    fp4_2t = torch::empty({H, M / 2}, torch::dtype(torch::kUInt8).device(device));
    si_2t  = torch::empty({c_scale_rows, c_scale_cols}, torch::dtype(torch::kUInt8).device(device));
  }
  {
    TensorWrapper te_out2(NVTE_NVFP4_1D_SCALING);
    float* amax2_ptr = amax2.data_ptr<float>();
    te_out2.set_rowwise_data(fp4_2.data_ptr(), DType::kFloat4E2M1, r_shape);
    te_out2.set_rowwise_scale_inv(si_2.data_ptr(), DType::kFloat8E4M3, r_si_shape);
    te_out2.set_amax(amax2_ptr, DType::kFloat32, std::vector<size_t>{1});
    te_out2.set_columnwise_data(fp4_2t.data_ptr(), DType::kFloat4E2M1, c_shape);
    te_out2.set_columnwise_scale_inv(si_2t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
    te_out2.set_columnwise_amax(amax2_ptr, DType::kFloat32, std::vector<size_t>{1});
    // REMOVED: if (tk_swizzle) te_out2.set_with_gemm_swizzled_scales(true); // API removed in current TE

    auto te_in2 = make_te_input(out2);
    QuantizationConfigWrapper quant_config2;
    dispatch_nvte_quantize_v2(te_in2, te_out2, quant_config2, stream);
  }

  // Precompute sc_global for TK
  torch::Tensor sg1, sg2;
  if (tk_swizzle) {
    sg1 = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
    sg2 = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
    launch_dual_scalar_scale(
        amax1.data_ptr<float>(), sg1.data_ptr<float>(),
        amax2.data_ptr<float>(), sg2.data_ptr<float>(),
        1.0f / 2688.0f, stream);
  }

  return std::make_tuple(
      fp4_1, si_1, fp4_1t, si_1t, amax1, amax1,
      fp4_2, si_2, fp4_2t, si_2t, amax2, amax2,
      sg1, sg2);
}


// Forward declaration of pure RMSNorm backward kernel (no SiLU)
extern "C" void launch_fused_rmsnorm_backward(
    const void* grad_normed, const void* input,
    const void* weight, const float* cached_inv_rms,
    int rows, int cols,
    void* grad_input, float* grad_weight_accum,
    cudaStream_t stream);

// =========================================================================
// 6. fused_rmsnorm_backward: pure RMSNorm backward (no SiLU)
//    Used for attention QKV paths where input is just normed, no activation.
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor>
fused_rmsnorm_backward(
    torch::Tensor d_normed,     // (M, K) bf16 — gradient w.r.t. RMSNorm output
    torch::Tensor x_raw,        // (M, K) bf16 — raw input before RMSNorm
    torch::Tensor norm_weight,  // (K,) bf16 — gamma
    torch::Tensor inv_rms       // (M,) float32 — cached from forward
) {
  TORCH_CHECK(d_normed.is_cuda() && d_normed.is_contiguous());
  TORCH_CHECK(x_raw.is_cuda() && x_raw.is_contiguous());
  TORCH_CHECK(d_normed.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(x_raw.scalar_type() == torch::kBFloat16);

  int64_t M = d_normed.size(0), K = d_normed.size(1);
  TORCH_CHECK(x_raw.size(0) == M && x_raw.size(1) == K);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = d_normed.device();

  auto grad_input = torch::empty_like(x_raw);
  auto dgamma = torch::zeros({K}, torch::dtype(torch::kFloat32).device(device));

  launch_fused_rmsnorm_backward(
      d_normed.data_ptr(),
      x_raw.data_ptr(),
      norm_weight.data_ptr(),
      inv_rms.data_ptr<float>(),
      (int)M, (int)K,
      grad_input.data_ptr(),
      dgamma.data_ptr<float>(),
      stream);

  return std::make_tuple(grad_input, dgamma);
}

// =========================================================================
// 6. fast_nvfp4_quantize: Direct bf16 → NVFP4 quant bypassing Python wrapper
//    Returns (fp4_data, scale_inv, fp4_data_t, scale_inv_t, amax, amax_t)
//    Eliminates: autograd Function overhead, NVFP4Tensor construction,
//    Quantizer Python dispatch — goes straight to nvte_quantize_v2.
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
fast_nvfp4_quantize(
    torch::Tensor input,        // (M, K) bf16, contiguous
    bool encode_centric         // encode-centric scaling mode
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(input.dim() == 2);
  int64_t M = input.size(0), K = input.size(1);
  TORCH_CHECK(M % 32 == 0, "M must be multiple of 32 for TMA");
  TORCH_CHECK(K % 32 == 0, "K must be multiple of 32 for TMA");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  // --- Rowwise output: (M, K/2) fp4 packed, scale_inv padded ---
  auto fp4_data = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(device));
  int64_t r_scale_rows = round_up(M, 128);
  int64_t r_scale_cols = round_up(K / 16, 4);
  auto scale_inv = torch::empty({r_scale_rows, r_scale_cols},
                                torch::dtype(torch::kUInt8).device(device));

  // --- Columnwise (transposed) output: (K, M/2) ---
  auto fp4_data_t = torch::empty({K, M / 2}, torch::dtype(torch::kUInt8).device(device));
  int64_t c_scale_rows = round_up(K, 128);
  int64_t c_scale_cols = round_up(M / 16, 4);
  auto scale_inv_t = torch::empty({c_scale_rows, c_scale_cols},
                                  torch::dtype(torch::kUInt8).device(device));

  // Single amax buffer — used for BOTH rowwise and colwise (same input data → same amax)
  // Eliminates the separate D2D memcpy that TE does to copy rowwise→colwise amax
  auto amax = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

  // Build TE input wrapper
  auto te_input = make_te_input(input);

  // Build output wrapper with SHARED amax pointer for both row and col
  // This avoids the D2D memcpy that TE's Python path does
  std::vector<size_t> r_shape = {(size_t)M, (size_t)K};
  std::vector<size_t> r_si_shape = {(size_t)r_scale_rows, (size_t)r_scale_cols};
  std::vector<size_t> c_shape = {(size_t)K, (size_t)M};
  std::vector<size_t> c_si_shape = {(size_t)c_scale_rows, (size_t)c_scale_cols};

  TensorWrapper te_output(NVTE_NVFP4_1D_SCALING);
  te_output.set_rowwise_data(fp4_data.data_ptr(), DType::kFloat4E2M1, r_shape);
  te_output.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3, r_si_shape);
  te_output.set_amax(amax.data_ptr<float>(), DType::kFloat32, std::vector<size_t>{1});
  te_output.set_columnwise_data(fp4_data_t.data_ptr(), DType::kFloat4E2M1, c_shape);
  te_output.set_columnwise_scale_inv(scale_inv_t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
  // SHARED amax: point colwise amax to SAME buffer as rowwise → elimiates D2D memcpy
  te_output.set_columnwise_amax(amax.data_ptr<float>(), DType::kFloat32, std::vector<size_t>{1});

  // Step 1: Compute global amax (2 kernels: zero_amax + amax_kernel)
  //         TE's public API, launches optimized vectorized bf16 reduction
  nvte_compute_amax(te_input.data(), te_output.data(), stream);

  // Step 2: Quantize (1 kernel: nvfp4_transpose_kernel with RETURN_TRANSPOSE=true)
  QuantizationConfigWrapper quant_config;
  dispatch_nvte_quantize_v2(te_input, te_output, quant_config, stream);

  // Return amax for both row and col (same value since same input)
  return std::make_tuple(fp4_data, scale_inv, fp4_data_t, scale_inv_t,
                         amax, amax);
}

// Forward declarations of custom fused amax kernel launchers
extern void launch_fused_amax_bf16(
    const void* input, float* amax, size_t N, cudaStream_t stream);
extern void launch_grouped_amax_bf16(
    const void* input, float* amaxes, const int64_t* split_elem_offsets,
    int num_splits, size_t total_elems, cudaStream_t stream);


// =========================================================================
// 7. fast_nvfp4_quantize_v2: Uses custom fused amax kernel instead of
//    TE's 2-kernel amax path (zero_amax + amax_kernel).
//    Total kernels: cudaMemsetAsync + fused_amax_bf16 + nvfp4_transpose = 2 launches
//    (memset is not a kernel launch, it's a DMA operation)
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
fast_nvfp4_quantize_v2(
    torch::Tensor input,        // (M, K) bf16, contiguous
    bool encode_centric         // encode-centric scaling mode
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(input.dim() == 2);
  int64_t M = input.size(0), K = input.size(1);
  TORCH_CHECK(M % 32 == 0, "M must be multiple of 32 for TMA");
  TORCH_CHECK(K % 32 == 0, "K must be multiple of 32 for TMA");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  // --- Rowwise output: (M, K/2) fp4 packed, scale_inv padded ---
  auto fp4_data = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(device));
  int64_t r_scale_rows = round_up(M, 128);
  int64_t r_scale_cols = round_up(K / 16, 4);
  auto scale_inv = torch::empty({r_scale_rows, r_scale_cols},
                                torch::dtype(torch::kUInt8).device(device));

  // --- Columnwise (transposed) output: (K, M/2) ---
  auto fp4_data_t = torch::empty({K, M / 2}, torch::dtype(torch::kUInt8).device(device));
  int64_t c_scale_rows = round_up(K, 128);
  int64_t c_scale_cols = round_up(M / 16, 4);
  auto scale_inv_t = torch::empty({c_scale_rows, c_scale_cols},
                                  torch::dtype(torch::kUInt8).device(device));

  // Amax buffer — allocated uninitialised, zeroed via cudaMemsetAsync (not a kernel!)
  auto amax = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
  float* amax_ptr = amax.data_ptr<float>();

  // Step 1: Zero + compute amax in 1 kernel (replaces TE's 2-kernel path)
  cudaMemsetAsync(amax_ptr, 0, sizeof(float), stream);
  launch_fused_amax_bf16(input.data_ptr(), amax_ptr, M * K, stream);

  // Build TE output wrapper with SHARED amax pointer for both row and col
  std::vector<size_t> r_shape = {(size_t)M, (size_t)K};
  std::vector<size_t> r_si_shape = {(size_t)r_scale_rows, (size_t)r_scale_cols};
  std::vector<size_t> c_shape = {(size_t)K, (size_t)M};
  std::vector<size_t> c_si_shape = {(size_t)c_scale_rows, (size_t)c_scale_cols};

  TensorWrapper te_output(NVTE_NVFP4_1D_SCALING);
  te_output.set_rowwise_data(fp4_data.data_ptr(), DType::kFloat4E2M1, r_shape);
  te_output.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3, r_si_shape);
  te_output.set_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
  te_output.set_columnwise_data(fp4_data_t.data_ptr(), DType::kFloat4E2M1, c_shape);
  te_output.set_columnwise_scale_inv(scale_inv_t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
  te_output.set_columnwise_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});

  // Step 2: Quantize (1 kernel: nvfp4_transpose_kernel)
  auto te_input = make_te_input(input);
  QuantizationConfigWrapper quant_config;
  dispatch_nvte_quantize_v2(te_input, te_output, quant_config, stream);

  return std::make_tuple(fp4_data, scale_inv, fp4_data_t, scale_inv_t,
                         amax, amax);
}

// =========================================================================
// 7a. fused_amax_quantize: amax + NVFP4 quant with optional TK-swizzle
//
//     Identical to fast_nvfp4_quantize_v2 but adds:
//       - tk_swizzle=true: outputs fp4x2 dtype, 3D fp8 scales, sg=amax/2688
//       - Eliminates tk_quantize_for_gemm's host-side overhead:
//         * No Python-side torch::empty with exotic dtypes
//         * No host-side TMA descriptor creation (TE kernel does it)
//         * No separate sg computation kernel
//     Returns: (fp4, si, fp4_t, si_t, sg, sg_t)
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
fused_amax_quantize(
    torch::Tensor input,        // (M, K) bf16, contiguous
    bool tk_swizzle = false     // write scales in TK-swizzled layout
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(input.dim() == 2);
  int64_t M = input.size(0), K = input.size(1);
  TORCH_CHECK(M % 32 == 0, "M must be multiple of 32 for TMA");
  TORCH_CHECK(K % 32 == 0, "K must be multiple of 32 for TMA");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  // TE scale dimensions (needed even for TK — kernel uses these internally)
  int64_t r_scale_rows = round_up(M, 128);
  int64_t r_scale_cols = round_up(K / 16, 4);
  int64_t c_scale_rows = round_up(K, 128);
  int64_t c_scale_cols = round_up(M / 16, 4);

  std::vector<size_t> r_shape = {(size_t)M, (size_t)K};
  std::vector<size_t> r_si_shape = {(size_t)r_scale_rows, (size_t)r_scale_cols};
  std::vector<size_t> c_shape = {(size_t)K, (size_t)M};
  std::vector<size_t> c_si_shape = {(size_t)c_scale_rows, (size_t)c_scale_cols};

  // Allocate outputs
  torch::Tensor fp4_data, scale_inv, fp4_data_t, scale_inv_t;
  if (tk_swizzle) {
    // TK-native format: fp4x2 dtype, 3D fp8 scales
    int64_t ntm_r = M / 128, ntk_r = K / 64;
    int64_t ntm_c = K / 128, ntk_c = M / 64;
    fp4_data   = torch::empty({M, K / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    scale_inv  = torch::empty({ntm_r, ntk_r, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    fp4_data_t = torch::empty({K, M / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    scale_inv_t = torch::empty({ntm_c, ntk_c, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
  } else {
    // TE-native format: uint8, 2D padded scales
    fp4_data   = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(device));
    scale_inv  = torch::empty({r_scale_rows, r_scale_cols}, torch::dtype(torch::kUInt8).device(device));
    fp4_data_t = torch::empty({K, M / 2}, torch::dtype(torch::kUInt8).device(device));
    scale_inv_t = torch::empty({c_scale_rows, c_scale_cols}, torch::dtype(torch::kUInt8).device(device));
  }

  // Amax + sg buffer: [amax, sg] contiguous
  auto amax_buf = torch::empty({2}, torch::dtype(torch::kFloat32).device(device));
  float* amax_ptr = amax_buf.data_ptr<float>();
  float* sg_ptr = amax_ptr + 1;

  // Step 1: Zero + compute amax
  cudaMemsetAsync(amax_ptr, 0, sizeof(float), stream);
  launch_fused_amax_bf16(input.data_ptr(), amax_ptr, M * K, stream);

  // Step 1b: Compute sg = amax / 2688 on device (avoids Python aten::div)
  if (tk_swizzle) {
    launch_scalar_scale(amax_ptr, sg_ptr, 1.0f / 2688.0f, stream);
  }

  // Step 2: Quantize with pre-computed amax
  TensorWrapper te_output(NVTE_NVFP4_1D_SCALING);
  te_output.set_rowwise_data(fp4_data.data_ptr(), DType::kFloat4E2M1, r_shape);
  te_output.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3, r_si_shape);
  te_output.set_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
  te_output.set_columnwise_data(fp4_data_t.data_ptr(), DType::kFloat4E2M1, c_shape);
  te_output.set_columnwise_scale_inv(scale_inv_t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
  te_output.set_columnwise_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
  // REMOVED: if (tk_swizzle) te_output.set_with_gemm_swizzled_scales(true); // API removed in current TE

  auto te_input = make_te_input(input);
  QuantizationConfigWrapper quant_config;
  dispatch_nvte_quantize_v2(te_input, te_output, quant_config, stream);

  // Return sg as 1-element tensor (shares storage with amax_buf)
  auto sg = amax_buf.narrow(0, 1, 1);
  auto amax = amax_buf.narrow(0, 0, 1);

  return std::make_tuple(fp4_data, scale_inv, fp4_data_t, scale_inv_t,
                         sg, sg);  // sg for both row and col
}

// Forward declaration of fused rmsnorm+silu dual kernel (no quantization)
extern "C" void launch_fused_rmsnorm_silu_dual(
    const void* x_ptr, const void* w_ptr,
    float epsilon, int rows, int cols,
    void* out_silu, void* out_normed,
    float* inv_rms_cache,
    cudaStream_t stream);

// Forward declaration of pure RMSNorm kernel (no SiLU, no amax, no quantization)
extern "C" void launch_fused_rmsnorm_only(
    const void* x_ptr, const void* w_ptr,
    float epsilon, int rows, int cols,
    void* out_normed, float* inv_rms_cache,
    cudaStream_t stream);

// Forward declaration of RMSNorm + amax kernel (no SiLU, amax for quantize)
extern "C" void launch_fused_rmsnorm_only_with_amax(
    const void* x_ptr, const void* w_ptr,
    float epsilon, int rows, int cols,
    void* out_normed, float* inv_rms_cache,
    float* global_amax,
    cudaStream_t stream);

// =========================================================================
// 7. fused_rmsnorm_silu_dual: RMSNorm + SiLU → bf16 (no quantization)
//    Returns silu(normed) for w1 path and normed for w3 path as bf16.
//    Use with _fast_quantize for exact TE quantization parity.
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_rmsnorm_silu_dual(
    torch::Tensor input,        // (M, K) bf16
    torch::Tensor norm_weight,  // (K,) bf16
    float epsilon               // RMSNorm epsilon
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  int64_t M = input.size(0), K = input.size(1);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  auto out_silu = torch::empty({M, K}, torch::dtype(torch::kBFloat16).device(device));
  auto out_normed = torch::empty({M, K}, torch::dtype(torch::kBFloat16).device(device));
  auto inv_rms = torch::empty({M}, torch::dtype(torch::kFloat32).device(device));

  launch_fused_rmsnorm_silu_dual(
      input.data_ptr(), norm_weight.data_ptr(),
      epsilon, (int)M, (int)K,
      out_silu.data_ptr(), out_normed.data_ptr(),
      inv_rms.data_ptr<float>(),
      stream);

  return std::make_tuple(out_silu, out_normed, inv_rms);
}


// =========================================================================
// 7b. fused_rmsnorm_only: RMSNorm ONLY → bf16 (no SiLU, no amax, no quant)
//     Dedicated kernel for QKV path. Returns normed + inv_rms.
//     Faster than fused_rmsnorm_silu_dual since it skips SiLU + 2nd output.
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor>
fused_rmsnorm_only(
    torch::Tensor input,        // (M, K) bf16
    torch::Tensor norm_weight,  // (K,) bf16
    float epsilon               // RMSNorm epsilon
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  int64_t M = input.size(0), K = input.size(1);

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  auto out_normed = torch::empty({M, K}, torch::dtype(torch::kBFloat16).device(device));
  auto inv_rms = torch::empty({M}, torch::dtype(torch::kFloat32).device(device));

  launch_fused_rmsnorm_only(
      input.data_ptr(), norm_weight.data_ptr(),
      epsilon, (int)M, (int)K,
      out_normed.data_ptr(),
      inv_rms.data_ptr<float>(),
      stream);

  return std::make_tuple(out_normed, inv_rms);
}


// =========================================================================
// 7c. fused_rmsnorm_quantize: RMSNorm + amax + NVFP4 quant (2 kernels total)
//
//     Single C++ dispatch replacing 4 separate operations:
//       OLD: rmsnorm_only (1) + memset (DMA) + fused_amax (1) + quantize (1) = 3 kernels + DMA
//       NEW: rmsnorm_with_amax (1) + quantize (1) = 2 kernels only
//
//     The RMSNorm kernel computes amax for free during Pass 2.
//     Returns: (fp4, si, fp4_t, si_t, amax, inv_rms)
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor>
fused_rmsnorm_quantize(
    torch::Tensor input,        // (M, K) bf16
    torch::Tensor norm_weight,  // (K,) bf16
    float epsilon,              // RMSNorm epsilon
    bool tk_swizzle = false     // write scales in TK-swizzled layout
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  int64_t M = input.size(0), K = input.size(1);
  TORCH_CHECK(M % 32 == 0, "M must be multiple of 32 for TMA");
  TORCH_CHECK(K % 32 == 0, "K must be multiple of 32 for TMA");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  // Intermediate bf16 normed (will be consumed by quantize in same stream)
  auto normed = torch::empty({M, K}, torch::dtype(torch::kBFloat16).device(device));
  auto inv_rms = torch::empty({M}, torch::dtype(torch::kFloat32).device(device));
  auto amax = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
  cudaMemsetAsync(amax.data_ptr<float>(), 0, sizeof(float), stream);

  // === Kernel 1: RMSNorm + amax (fused, 1 kernel) ===
  launch_fused_rmsnorm_only_with_amax(
      input.data_ptr(), norm_weight.data_ptr(),
      epsilon, (int)M, (int)K,
      normed.data_ptr(),
      inv_rms.data_ptr<float>(),
      amax.data_ptr<float>(),
      stream);

  // === Kernel 2: NVFP4 quantize with pre-computed amax (1 kernel) ===
  // Rowwise output
  auto fp4_data = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(device));
  int64_t r_scale_rows = round_up(M, 128);
  int64_t r_scale_cols = round_up(K / 16, 4);
  auto scale_inv = torch::empty({r_scale_rows, r_scale_cols},
                                torch::dtype(torch::kUInt8).device(device));

  // Columnwise (transposed) output
  auto fp4_data_t = torch::empty({K, M / 2}, torch::dtype(torch::kUInt8).device(device));
  int64_t c_scale_rows = round_up(K, 128);
  int64_t c_scale_cols = round_up(M / 16, 4);
  auto scale_inv_t = torch::empty({c_scale_rows, c_scale_cols},
                                  torch::dtype(torch::kUInt8).device(device));

  // Build TE output wrapper with SHARED amax for row + col
  std::vector<size_t> r_shape = {(size_t)M, (size_t)K};
  std::vector<size_t> r_si_shape = {(size_t)r_scale_rows, (size_t)r_scale_cols};
  std::vector<size_t> c_shape = {(size_t)K, (size_t)M};
  std::vector<size_t> c_si_shape = {(size_t)c_scale_rows, (size_t)c_scale_cols};

  TensorWrapper te_output(NVTE_NVFP4_1D_SCALING);
  float* amax_ptr = amax.data_ptr<float>();
  te_output.set_rowwise_data(fp4_data.data_ptr(), DType::kFloat4E2M1, r_shape);
  te_output.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3, r_si_shape);
  te_output.set_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
  te_output.set_columnwise_data(fp4_data_t.data_ptr(), DType::kFloat4E2M1, c_shape);
  te_output.set_columnwise_scale_inv(scale_inv_t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
  te_output.set_columnwise_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
  // REMOVED: if (tk_swizzle) te_output.set_with_gemm_swizzled_scales(true); // API removed in current TE

  auto te_input = make_te_input(normed);
  QuantizationConfigWrapper quant_config;
  dispatch_nvte_quantize_v2(te_input, te_output, quant_config, stream);

  return std::make_tuple(fp4_data, scale_inv, fp4_data_t, scale_inv_t,
                         amax, inv_rms);
}


// Forward declaration of new launcher with amax
extern "C" void launch_fused_rmsnorm_silu_dual_with_amax(
    const void* x_ptr, const void* w_ptr,
    float epsilon, int rows, int cols,
    void* out_silu, void* out_normed,
    float* inv_rms_cache,
    float* global_amax_silu, float* global_amax_norm,
    cudaStream_t stream);

// =========================================================================
// 8. fused_rmsnorm_silu_dual_quantize: RMSNorm+SiLU → BF16 → 2x NVFP4
//
// Single C++ dispatch (matches fast_nvfp4_quantize_v2 pattern):
//   1. Our custom kernel: silu(normed(x)) + normed(x) + inv_rms + amaxes (1 kernel)
//   2. nvte_quantize_v2 for silu path (1 kernel — uses pre-computed amax)
//   3. nvte_quantize_v2 for norm path (1 kernel — uses pre-computed amax)
//
// Returns (fp4_silu, si_silu, fp4_silu_t, si_silu_t, amax_silu,
//          fp4_norm, si_norm, fp4_norm_t, si_norm_t, amax_norm,
//          inv_rms)
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor, torch::Tensor>
fused_rmsnorm_silu_dual_quantize(
    torch::Tensor input,        // (M, K) bf16
    torch::Tensor norm_weight,  // (K,) bf16
    float epsilon,
    bool encode_centric = false,
    bool tk_swizzle = false     // write scales in TK-swizzled layout
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  int64_t M = input.size(0), K = input.size(1);
  TORCH_CHECK(K % 16 == 0);
  TORCH_CHECK(M % 32 == 0, "M must be multiple of 32");
  TORCH_CHECK(K % 32 == 0, "K must be multiple of 32");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  // BF16 intermediates (consumed by nvte_quantize_v2 in same stream)
  auto silu_normed = torch::empty({M, K}, torch::dtype(torch::kBFloat16).device(device));
  auto normed      = torch::empty({M, K}, torch::dtype(torch::kBFloat16).device(device));
  auto inv_rms     = torch::empty({M}, torch::dtype(torch::kFloat32).device(device));

  // Amaxes — single tensor each, SHARED for row+col (same as fast_nvfp4_quantize_v2)
  auto amax_silu = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
  auto amax_norm = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
  cudaMemsetAsync(amax_silu.data_ptr<float>(), 0, sizeof(float), stream);
  cudaMemsetAsync(amax_norm.data_ptr<float>(), 0, sizeof(float), stream);

  // When tk_swizzle=true, compute sc_global = amax / (6*448) for TK.
  // Also allocate amax_silu_sc and amax_norm_sc for precomputed values.
  torch::Tensor amax_silu_sc, amax_norm_sc;  // only used when tk_swizzle

  // Step 1: Our custom kernel → silu_normed + normed + inv_rms + amaxes (1 kernel)
  launch_fused_rmsnorm_silu_dual_with_amax(
      input.data_ptr(), norm_weight.data_ptr(),
      epsilon, (int)M, (int)K,
      silu_normed.data_ptr(), normed.data_ptr(),
      inv_rms.data_ptr<float>(),
      amax_silu.data_ptr<float>(), amax_norm.data_ptr<float>(),
      stream);

  // Shared scale dimensions
  int64_t r_rows = round_up(M, 128), r_cols = round_up(K / 16, 4);
  int64_t c_rows = round_up(K, 128), c_cols = round_up(M / 16, 4);
  std::vector<size_t> r_shape = {(size_t)M, (size_t)K};
  std::vector<size_t> c_shape = {(size_t)K, (size_t)M};
  std::vector<size_t> r_si_shape = {(size_t)r_rows, (size_t)r_cols};
  std::vector<size_t> c_si_shape = {(size_t)c_rows, (size_t)c_cols};

  QuantizationConfigWrapper quant_config;

  // Step 2: Quantize silu_normed → w1 path (1 kernel)
  // When tk_swizzle: fp4 as kFloat4_e2m1fn_x2, scales as kFloat8_e4m3fn in tiled shape
  // When !tk_swizzle: fp4 as kUInt8, scales as kUInt8 in flat shape (original)
  int64_t ntm_r = M / 128, ntk_r = K / 64;  // tile dims for rowwise
  int64_t ntm_c = K / 128, ntk_c = M / 64;  // tile dims for columnwise
  torch::Tensor fp4_silu, si_silu, fp4_silu_t, si_silu_t;
  if (tk_swizzle) {
    fp4_silu   = torch::empty({M, K / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    si_silu    = torch::empty({ntm_r, ntk_r, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    fp4_silu_t = torch::empty({K, M / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    si_silu_t  = torch::empty({ntm_c, ntk_c, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
  } else {
    fp4_silu   = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(device));
    si_silu    = torch::empty({r_rows, r_cols}, torch::dtype(torch::kUInt8).device(device));
    fp4_silu_t = torch::empty({K, M / 2}, torch::dtype(torch::kUInt8).device(device));
    si_silu_t  = torch::empty({c_rows, c_cols}, torch::dtype(torch::kUInt8).device(device));
  }
  {
    TensorWrapper te_out(NVTE_NVFP4_1D_SCALING);
    te_out.set_rowwise_data(fp4_silu.data_ptr(), DType::kFloat4E2M1, r_shape);
    te_out.set_rowwise_scale_inv(si_silu.data_ptr(), DType::kFloat8E4M3, r_si_shape);
    te_out.set_amax(amax_silu.data_ptr<float>(), DType::kFloat32, std::vector<size_t>{1});
    te_out.set_columnwise_data(fp4_silu_t.data_ptr(), DType::kFloat4E2M1, c_shape);
    te_out.set_columnwise_scale_inv(si_silu_t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
    te_out.set_columnwise_amax(amax_silu.data_ptr<float>(), DType::kFloat32, std::vector<size_t>{1});
    // REMOVED: if (tk_swizzle) te_out.set_with_gemm_swizzled_scales(true); // API removed in current TE

    auto te_in = make_te_input(silu_normed);
    dispatch_nvte_quantize_v2(te_in, te_out, quant_config, stream);
  }

  // Step 3: Quantize normed → w3 path (1 kernel)
  torch::Tensor fp4_norm, si_norm, fp4_norm_t, si_norm_t;
  if (tk_swizzle) {
    fp4_norm   = torch::empty({M, K / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    si_norm    = torch::empty({ntm_r, ntk_r, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    fp4_norm_t = torch::empty({K, M / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    si_norm_t  = torch::empty({ntm_c, ntk_c, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
  } else {
    fp4_norm   = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(device));
    si_norm    = torch::empty({r_rows, r_cols}, torch::dtype(torch::kUInt8).device(device));
    fp4_norm_t = torch::empty({K, M / 2}, torch::dtype(torch::kUInt8).device(device));
    si_norm_t  = torch::empty({c_rows, c_cols}, torch::dtype(torch::kUInt8).device(device));
  }
  {
    TensorWrapper te_out(NVTE_NVFP4_1D_SCALING);
    te_out.set_rowwise_data(fp4_norm.data_ptr(), DType::kFloat4E2M1, r_shape);
    te_out.set_rowwise_scale_inv(si_norm.data_ptr(), DType::kFloat8E4M3, r_si_shape);
    te_out.set_amax(amax_norm.data_ptr<float>(), DType::kFloat32, std::vector<size_t>{1});
    te_out.set_columnwise_data(fp4_norm_t.data_ptr(), DType::kFloat4E2M1, c_shape);
    te_out.set_columnwise_scale_inv(si_norm_t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
    te_out.set_columnwise_amax(amax_norm.data_ptr<float>(), DType::kFloat32, std::vector<size_t>{1});
    // REMOVED: if (tk_swizzle) te_out.set_with_gemm_swizzled_scales(true); // API removed in current TE

    auto te_in = make_te_input(normed);
    dispatch_nvte_quantize_v2(te_in, te_out, quant_config, stream);
  }

  // If tk_swizzle, precompute sc_global = amax / (6 * 448) = amax / 2688 on GPU
  // This avoids 4 Python aten::mul dispatcher calls
  if (tk_swizzle) {
    amax_silu_sc = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
    amax_norm_sc = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
    launch_dual_scalar_scale(
        amax_silu.data_ptr<float>(), amax_silu_sc.data_ptr<float>(),
        amax_norm.data_ptr<float>(), amax_norm_sc.data_ptr<float>(),
        1.0f / 2688.0f, stream);
  }

  return std::make_tuple(
      fp4_silu, si_silu, fp4_silu_t, si_silu_t, amax_silu,
      fp4_norm, si_norm, fp4_norm_t, si_norm_t, amax_norm,
      inv_rms,
      amax_silu_sc, amax_norm_sc);  // empty when !tk_swizzle
}

// =========================================================================
// 10. group_nvfp4_quantize: Grouped FP4 quantize with per-split amax.
//     Uses nvte_group_nvfp4_quantize_with_amax to quantize a stacked tensor
//     into per-split FP4 outputs, each with independent amax.
//     Single kernel launch for all splits (rowwise + columnwise).
//
//     IMPORTANT: The grouped kernel uses TMA to write ALL rowwise FP4 data to
//     a single contiguous buffer (tensor_map_output). Per-split rowwise data
//     and scales MUST be contiguous sub-regions of this buffer.
//     Columnwise data and scales use per-split GMEM stores (no TMA), so they
//     can be separately allocated.
// =========================================================================
std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor, torch::Tensor>>
group_nvfp4_quantize(
    torch::Tensor input,                // (total_rows, K) bf16, contiguous
    std::vector<int64_t> split_sections  // row counts per split, e.g. [q_dim, k_dim, v_dim]
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(input.dim() == 2);
  int64_t total_rows = input.size(0), K = input.size(1);
  TORCH_CHECK(total_rows % 32 == 0, "total_rows must be multiple of 32 for TMA");
  TORCH_CHECK(K % 32 == 0, "K must be multiple of 32 for TMA");

  int64_t sum_splits = 0;
  for (auto s : split_sections) sum_splits += s;
  TORCH_CHECK(sum_splits == total_rows, "split_sections must sum to total_rows");

  size_t num_tensors = split_sections.size();
  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  // Build TE input wrapper
  auto te_input = make_te_input(input);

  // -----------------------------------------------------------------------
  // Allocate CONTIGUOUS rowwise buffers — TMA writes ALL rows contiguously.
  // The grouped kernel builds a TMA tensor map from the first output tensor's
  // data pointer and writes rows 0..total_rows-1 to it. If each split has a
  // separate allocation, K and V data would overwrite random memory.
  // -----------------------------------------------------------------------
  auto fp4_data_all = torch::empty({total_rows, K / 2},
                                    torch::dtype(torch::kUInt8).device(device));
  int64_t r_scale_rows_all = round_up(total_rows, 128);
  int64_t r_scale_cols = round_up(K / 16, 4);
  auto scale_inv_all = torch::empty({r_scale_rows_all, r_scale_cols},
                                     torch::dtype(torch::kUInt8).device(device));

  // -----------------------------------------------------------------------
  // Per-split: columnwise (transposed) data + scales + amax (separately allocated,
  // because the kernel writes these with direct GMEM stores, not TMA).
  // -----------------------------------------------------------------------
  std::vector<size_t> c_split_sections(num_tensors);
  std::vector<torch::Tensor> fp4_data_ts(num_tensors);
  std::vector<torch::Tensor> scale_inv_ts(num_tensors);
  std::vector<torch::Tensor> scale_invs(num_tensors);  // rowwise scale_inv (per-split, GMEM stores)
  std::vector<torch::Tensor> amaxs(num_tensors);
  std::vector<TensorWrapper> te_outputs;
  te_outputs.reserve(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    te_outputs.emplace_back(NVTE_NVFP4_1D_SCALING);
  }

  int64_t row_offset = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    int64_t M_i = split_sections[i];
    c_split_sections[i] = (size_t)M_i;
    TORCH_CHECK(M_i % 32 == 0, "Each split must be a multiple of 32, got ", M_i);

    // Rowwise FP4 data: slice into the contiguous buffer
    auto fp4_data_i = fp4_data_all.narrow(0, row_offset, M_i);
    // Rowwise scales: per-split allocs (written via GMEM stores, not TMA)
    int64_t r_scale_rows_i = round_up(M_i, 128);
    scale_invs[i] = torch::empty({r_scale_rows_i, r_scale_cols},
                                  torch::dtype(torch::kUInt8).device(device));

    // Columnwise (transposed): separate per-split allocs (GMEM stores)
    fp4_data_ts[i] = torch::empty({K, M_i / 2}, torch::dtype(torch::kUInt8).device(device));
    int64_t c_scale_rows = round_up(K, 128);
    int64_t c_scale_cols = round_up(M_i / 16, 4);
    scale_inv_ts[i] = torch::empty({c_scale_rows, c_scale_cols},
                                    torch::dtype(torch::kUInt8).device(device));

    // Per-split amax
    amaxs[i] = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

    // Build TensorWrapper — rowwise data points into contiguous buffer
    std::vector<size_t> r_shape = {(size_t)M_i, (size_t)K};
    std::vector<size_t> r_si_shape = {(size_t)r_scale_rows_i, (size_t)r_scale_cols};
    std::vector<size_t> c_shape = {(size_t)K, (size_t)M_i};
    std::vector<size_t> c_si_shape = {(size_t)c_scale_rows, (size_t)c_scale_cols};

    te_outputs[i].set_rowwise_data(fp4_data_i.data_ptr(), DType::kFloat4E2M1, r_shape);
    te_outputs[i].set_rowwise_scale_inv(scale_invs[i].data_ptr(), DType::kFloat8E4M3, r_si_shape);
    te_outputs[i].set_amax(amaxs[i].data_ptr<float>(), DType::kFloat32, std::vector<size_t>{1});
    te_outputs[i].set_columnwise_data(fp4_data_ts[i].data_ptr(), DType::kFloat4E2M1, c_shape);
    te_outputs[i].set_columnwise_scale_inv(scale_inv_ts[i].data_ptr(), DType::kFloat8E4M3, c_si_shape);
    te_outputs[i].set_columnwise_amax(amaxs[i].data_ptr<float>(), DType::kFloat32, std::vector<size_t>{1});

    row_offset += M_i;
  }

  // Build NVTETensor* array for outputs
  std::vector<NVTETensor> output_ptrs(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    output_ptrs[i] = te_outputs[i].data();
  }

  // Step 1: Grouped amax — all per-split amaxes in 1 kernel launch.
  auto all_amaxes = torch::zeros({(int64_t)num_tensors}, torch::dtype(torch::kFloat32).device(device));
  std::vector<int64_t> h_elem_offsets(num_tensors + 1);
  h_elem_offsets[0] = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    h_elem_offsets[i + 1] = h_elem_offsets[i] + split_sections[i] * K;
  }
  // clone() prevents dangling pointer race — h_elem_offsets is stack-local
  auto d_elem_offsets = torch::from_blob(h_elem_offsets.data(), {(int64_t)(num_tensors + 1)},
                                          torch::kInt64).clone().to(device);
  launch_grouped_amax_bf16(input.data_ptr(), all_amaxes.data_ptr<float>(),
                           d_elem_offsets.data_ptr<int64_t>(),
                           (int)num_tensors, total_rows * K, stream);
  for (size_t i = 0; i < num_tensors; ++i) {
    cudaMemcpyAsync(amaxs[i].data_ptr(), all_amaxes.data_ptr<float>() + i,
                    sizeof(float), cudaMemcpyDeviceToDevice, stream);
  }

  // Step 2: Call grouped quantize kernel (1 kernel launch for all splits!)
  // NOTE: nvte_group_nvfp4_quantize_with_amax was removed in current TE version.
  // Use _tk_quant (USE_TK_QUANT=1) or the Python-level group quantization instead.
  throw std::runtime_error(
      "group_nvfp4_quantize: nvte_group_nvfp4_quantize_with_amax not available in this TE version. "
      "Use USE_TK_QUANT=1 for TK-based group quantization.");

  // Build return: list of (fp4, scale_inv, fp4_t, scale_inv_t, amax)
  // fp4 data = narrow views of contiguous buffer, scale_inv = per-split allocs
  std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor>> result;
  row_offset = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    int64_t M_i = split_sections[i];
    auto fp4_data_i = fp4_data_all.narrow(0, row_offset, M_i);
    result.emplace_back(fp4_data_i, scale_invs[i],
                        fp4_data_ts[i], scale_inv_ts[i],
                        amaxs[i]);
    row_offset += M_i;
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // NOTE: fused_te_quantize_rmsnorm_silu, _rmsnorm, _rmsnorm_full,
  //       _silu_2pass, _silu_2pass_full, _2pass_full are disabled
  //       (require nvte_quantize_rmsnorm_silu which isn't in current TE)
  m.def("fused_silu_rmsnorm_backward", &fused_silu_rmsnorm_backward,
        "Fused SiLU backward + RMSNorm backward + dgamma (single CUDA kernel)");
  m.def("fused_rmsnorm_backward", &fused_rmsnorm_backward,
        "Pure RMSNorm backward + dgamma (no SiLU, single CUDA kernel)");
  m.def("fused_te_mul_quantize", &fused_te_mul_quantize,
        "Fused h1*h3 + NVFP4 quantize (eliminates Python overhead)");
  m.def("fused_te_dual_mul_quantize", &fused_te_dual_mul_quantize,
        "Fused dh*h3 + dh*h1 → 2x NVFP4 quantize (backward grad optimization)");
  m.def("fused_te_silu_mul_quantize", &fused_te_silu_mul_quantize,
        "Fused silu(h1_raw)*h3 + NVFP4 quantize (corrected SwiGLU forward)");
  m.def("fused_te_silu_deriv_dual_mul_quantize", &fused_te_silu_deriv_dual_mul_quantize,
        "Fused dh*h3*silu'(h1_raw) + dh*silu(h1_raw) → 2x NVFP4 quantize (corrected SwiGLU backward)");
  m.def("fast_nvfp4_quantize", &fast_nvfp4_quantize,
        "Direct bf16 → NVFP4 row+col quantize (bypasses Python wrapper, uses TE amax)");
  m.def("fast_nvfp4_quantize_v2", &fast_nvfp4_quantize_v2,
        "Direct bf16 → NVFP4 row+col quantize (custom fused amax, 2 kernel launches)");
  m.def("fused_amax_quantize", &fused_amax_quantize,
        "Fused amax + NVFP4 quant with optional TK-swizzle (returns fp4, si, fp4_t, si_t, sg, sg)",
        py::arg("input"), py::arg("tk_swizzle") = false);
  m.def("fused_rmsnorm_silu_dual", &fused_rmsnorm_silu_dual,
        "Fused RMSNorm + SiLU → bf16 (no quant). Returns (silu_normed, normed, inv_rms)");
  m.def("fused_rmsnorm_only", &fused_rmsnorm_only,
        "Fused RMSNorm ONLY → bf16 (no SiLU, no quant). Returns (normed, inv_rms)");
  m.def("fused_rmsnorm_quantize", &fused_rmsnorm_quantize,
        "Fused RMSNorm + NVFP4 quant (2 kernels: norm+amax, quantize). Returns (fp4, si, fp4_t, si_t, amax, inv_rms)");
  m.def("fused_rmsnorm_silu_dual_quantize", &fused_rmsnorm_silu_dual_quantize,
        "Fused RMSNorm+SiLU → BF16 → 2x NVFP4 quant (single C++ dispatch, 3 CUDA kernels)");
#if 0  // Disabled: depends on nvte_quantize_rmsnorm / nvte_quantize_rmsnorm_silu
  m.def("fused_te_quantize_rmsnorm_2pass_full", &fused_te_quantize_rmsnorm_2pass_full,
        "2-pass RMSNorm+NVFP4 quant (no SiLU): pass1 amax+inv_rms, pass2 TE quant+transpose");
  m.def("fused_te_quantize_rmsnorm_silu_2pass_full", &fused_te_quantize_rmsnorm_silu_2pass_full,
        "2-pass RMSNorm+SiLU+NVFP4 quant: pass1 amax+inv_rms, pass2 TE quant+transpose");
  m.def("fused_te_quantize_rmsnorm_full", &fused_te_quantize_rmsnorm_full,
        "RMSNorm+NVFP4 quant with pre-computed inv_rms and amax (for w3 path)");
#endif
  m.def("group_nvfp4_quantize", &group_nvfp4_quantize,
        "Grouped FP4 quantize: stacked bf16 → per-split FP4 with independent amax (1 kernel)");
  m.def("set_custom_quant", [](bool enable) { g_use_custom_quant = enable; },
        "Enable/disable custom quantisation path (NVTE_CUSTOM_QUANT)",
        py::arg("enable"));
  m.def("get_custom_quant", []() { return g_use_custom_quant; },
        "Check if custom quantisation path is enabled");

  // ─── bf16-only fused elementwise (no quantize, for TK pipeline) ───
  m.def("fused_silu_mul_bf16", [](torch::Tensor h1_raw, torch::Tensor h3) {
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
    TORCH_CHECK(h1_raw.scalar_type() == torch::kBFloat16);
    int64_t M = h1_raw.size(0), H = h1_raw.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = h1_raw.device();
    auto h = torch::empty_like(h1_raw);
    auto amax = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    launch_fused_silu_mul_amax(
        h1_raw.data_ptr(), h3.data_ptr(), h.data_ptr(),
        amax.data_ptr<float>(), M * H, stream);
    return std::make_tuple(h, amax);
  }, "Fused silu(h1)*h3 → bf16 + amax (1 kernel, no quantize)",
     py::arg("h1_raw"), py::arg("h3"));

  m.def("fused_silu_deriv_dual_mul_bf16",
    [](torch::Tensor dh, torch::Tensor h3, torch::Tensor h1_raw) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h3.is_cuda() && h3.is_contiguous());
    TORCH_CHECK(h1_raw.is_cuda() && h1_raw.is_contiguous());
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16);
    int64_t numel = dh.numel();
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = dh.device();
    auto out1 = torch::empty_like(dh);
    auto out2 = torch::empty_like(dh);
    auto amax1 = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    auto amax2 = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    launch_fused_silu_deriv_dual_mul_amax(
        dh.data_ptr(), h3.data_ptr(), h1_raw.data_ptr(),
        out1.data_ptr(), out2.data_ptr(),
        amax1.data_ptr<float>(), amax2.data_ptr<float>(),
        numel, stream);
    return std::make_tuple(out1, out2, amax1, amax2);
  }, "Fused silu-derivative dual mul → 2x bf16 + 2x amax (1 kernel, no quantize)",
     py::arg("dh"), py::arg("h3"), py::arg("h1_raw"));

  // ─── Strided SiLU (reads h1/h3 from h13 without .contiguous()) ───
  m.def("fused_silu_mul_strided_bf16", [](torch::Tensor h13, int64_t H) {
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous());
    TORCH_CHECK(h13.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(h13.dim() == 2 && h13.size(1) == 2 * H);
    int64_t M = h13.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = h13.device();
    auto out = torch::empty({M, H}, torch::dtype(torch::kBFloat16).device(device));
    auto amax = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    launch_fused_silu_mul_strided_amax(
        h13.data_ptr(), out.data_ptr(),
        amax.data_ptr<float>(), M, H, stream);
    return std::make_tuple(out, amax);
  }, "Strided silu(h13[:,0:H])*h13[:,H:2H] → bf16 + amax (no .contiguous())",
     py::arg("h13"), py::arg("H"));

  m.def("fused_silu_deriv_dual_mul_strided_bf16",
    [](torch::Tensor dh, torch::Tensor h13) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous());
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16);
    int64_t M = dh.size(0), H = dh.size(1);
    TORCH_CHECK(h13.size(0) == M && h13.size(1) == 2 * H);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = dh.device();
    auto out1 = torch::empty_like(dh);
    auto out2 = torch::empty_like(dh);
    auto amax1 = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    auto amax2 = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    launch_fused_silu_deriv_dual_mul_strided_amax(
        dh.data_ptr(), h13.data_ptr(),
        out1.data_ptr(), out2.data_ptr(),
        amax1.data_ptr<float>(), amax2.data_ptr<float>(),
        M, H, stream);
    return std::make_tuple(out1, out2, amax1, amax2);
  }, "Strided silu-deriv dual mul (no .contiguous()) → 2x bf16 + 2x amax",
     py::arg("dh"), py::arg("h13"));

  m.def("fused_silu_deriv_dual_mul_strided_interleaved_bf16",
    [](torch::Tensor dh, torch::Tensor h13) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous());
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16);
    int64_t M = dh.size(0), H = dh.size(1);
    TORCH_CHECK(h13.size(0) == M && h13.size(1) == 2 * H);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = dh.device();
    auto dh13 = torch::empty({M, 2 * H}, torch::dtype(torch::kBFloat16).device(device));
    auto amax1 = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    auto amax2 = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));
    launch_fused_silu_deriv_dual_mul_strided_interleaved_amax(
        dh.data_ptr(), h13.data_ptr(),
        dh13.data_ptr(),
        amax1.data_ptr<float>(), amax2.data_ptr<float>(),
        M, H, stream);
    return std::make_tuple(dh13, amax1, amax2);
  }, "Strided silu-deriv → interleaved (M,2H) output (no torch.cat)",
     py::arg("dh"), py::arg("h13"));

  // ── CUDA graph-safe _out variants: write to pre-allocated buffers ──
  // These avoid graph-pool allocations by accepting output tensors
  // that were allocated outside the graph capture scope.

  m.def("fused_silu_deriv_dual_mul_strided_interleaved_bf16_out",
    [](torch::Tensor dh, torch::Tensor h13,
       torch::Tensor dh13_out, torch::Tensor amax1_out, torch::Tensor amax2_out) {
    TORCH_CHECK(dh.is_cuda() && dh.is_contiguous());
    TORCH_CHECK(h13.is_cuda() && h13.is_contiguous());
    TORCH_CHECK(dh.scalar_type() == torch::kBFloat16);
    int64_t M = dh.size(0), H = dh.size(1);
    TORCH_CHECK(h13.size(0) == M && h13.size(1) == 2 * H);
    TORCH_CHECK(dh13_out.is_cuda() && dh13_out.is_contiguous());
    TORCH_CHECK(dh13_out.size(0) == M && dh13_out.size(1) == 2 * H);
    // Zero amax buffers (equivalent to torch::zeros)
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(amax1_out.data_ptr(), 0, sizeof(float), stream);
    cudaMemsetAsync(amax2_out.data_ptr(), 0, sizeof(float), stream);
    launch_fused_silu_deriv_dual_mul_strided_interleaved_amax(
        dh.data_ptr(), h13.data_ptr(),
        dh13_out.data_ptr(),
        amax1_out.data_ptr<float>(), amax2_out.data_ptr<float>(),
        M, H, stream);
  }, "Graph-safe: write to pre-allocated output buffers",
     py::arg("dh"), py::arg("h13"),
     py::arg("dh13_out"), py::arg("amax1_out"), py::arg("amax2_out"));

  m.def("fused_rmsnorm_backward_out",
    [](torch::Tensor d_normed, torch::Tensor x_raw,
       torch::Tensor norm_weight, torch::Tensor inv_rms,
       torch::Tensor grad_input_out, torch::Tensor dgamma_out) {
    TORCH_CHECK(d_normed.is_cuda() && d_normed.is_contiguous());
    TORCH_CHECK(x_raw.is_cuda() && x_raw.is_contiguous());
    TORCH_CHECK(d_normed.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(x_raw.scalar_type() == torch::kBFloat16);
    int64_t M = d_normed.size(0), K = d_normed.size(1);
    TORCH_CHECK(x_raw.size(0) == M && x_raw.size(1) == K);
    TORCH_CHECK(grad_input_out.is_cuda() && grad_input_out.is_contiguous());
    TORCH_CHECK(grad_input_out.size(0) == M && grad_input_out.size(1) == K);
    TORCH_CHECK(dgamma_out.is_cuda() && dgamma_out.numel() == K);
    // Zero dgamma (equivalent to torch::zeros)
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(dgamma_out.data_ptr(), 0, K * sizeof(float), stream);
    launch_fused_rmsnorm_backward(
        d_normed.data_ptr(),
        x_raw.data_ptr(),
        norm_weight.data_ptr(),
        inv_rms.data_ptr<float>(),
        (int)M, (int)K,
        grad_input_out.data_ptr(),
        dgamma_out.data_ptr<float>(),
        stream);
  }, "Graph-safe: fused_rmsnorm_backward writing to pre-allocated buffers",
     py::arg("d_normed"), py::arg("x_raw"),
     py::arg("norm_weight"), py::arg("inv_rms"),
     py::arg("grad_input_out"), py::arg("dgamma_out"));
}
