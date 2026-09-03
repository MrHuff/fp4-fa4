/*
 * Minimal PyTorch C++ extension for FP4 quantization.
 * Contains only:
 *   1. fast_nvfp4_quantize_v2  — single tensor bf16 → NVFP4 (row+col)
 *   2. group_nvfp4_quantize    — grouped quantize for stacked QKV
 *
 * Stripped from te_fused_rmsnorm_ext.cpp, removing all broken fused functions.
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

static inline int64_t round_up(int64_t val, int64_t mult) {
  return ((val + mult - 1) / mult) * mult;
}

static TensorWrapper make_te_input(torch::Tensor input) {
  int64_t M = input.size(0), K = input.size(1);
  std::vector<size_t> in_shape = {(size_t)M, (size_t)K};
  return TensorWrapper(input.data_ptr(), in_shape, DType::kBFloat16);
}

// Forward declaration of the fused amax kernels
extern void launch_fused_amax_bf16(
    const void* input, float* amax, size_t N, cudaStream_t stream);
extern void launch_fused_amax_bf16_with_scale(
    const void* input, float* amax, float* sg_out, float scale,
    size_t N, cudaStream_t stream);
extern void launch_grouped_amax_bf16(
    const void* input, float* amaxes, const int64_t* split_elem_offsets,
    int num_splits, size_t total_elems, cudaStream_t stream);
extern void launch_scalar_scale(
    const float* src, float* dst, float scale, cudaStream_t stream);

// =========================================================================
// 1. fast_nvfp4_quantize_v2: bf16 → NVFP4 row+col (custom fused amax)
// When tk_swizzle=true: returns TK-native tensors (fp4x2 dtype, tiled FP8
//   scales) + precomputed sg = amax / 2688.
// Returns (fp4, si, fp4_t, si_t, amax, amax, sg)
// sg is empty when !tk_swizzle.
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fast_nvfp4_quantize_v2(torch::Tensor input, bool encode_centric,
                       bool tk_swizzle = false, bool custom_quant = false) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(input.dim() == 2);
  int64_t M = input.size(0), K = input.size(1);
  TORCH_CHECK(M % 32 == 0, "M must be multiple of 32 for TMA");
  TORCH_CHECK(K % 32 == 0, "K must be multiple of 32 for TMA");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  // Scale dimensions
  int64_t r_scale_rows = round_up(M, 128);
  int64_t r_scale_cols = round_up(K / 16, 4);
  int64_t c_scale_rows = round_up(K, 128);
  int64_t c_scale_cols = round_up(M / 16, 4);
  std::vector<size_t> r_shape = {(size_t)M, (size_t)K};
  std::vector<size_t> r_si_shape = {(size_t)r_scale_rows, (size_t)r_scale_cols};
  std::vector<size_t> c_shape = {(size_t)K, (size_t)M};
  std::vector<size_t> c_si_shape = {(size_t)c_scale_rows, (size_t)c_scale_cols};

  // Tile dims for TK-native layout
  int64_t ntm_r = M / 128, ntk_r = K / 64;
  int64_t ntm_c = K / 128, ntk_c = M / 64;

  torch::Tensor fp4_data, scale_inv, fp4_data_t, scale_inv_t;
  if (tk_swizzle) {
    fp4_data   = torch::empty({M, K / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    scale_inv  = torch::empty({ntm_r, ntk_r, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    fp4_data_t = torch::empty({K, M / 2}, torch::dtype(torch::kFloat4_e2m1fn_x2).device(device));
    scale_inv_t = torch::empty({ntm_c, ntk_c, 512}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
  } else {
    fp4_data   = torch::empty({M, K / 2}, torch::dtype(torch::kUInt8).device(device));
    scale_inv  = torch::empty({r_scale_rows, r_scale_cols}, torch::dtype(torch::kUInt8).device(device));
    fp4_data_t = torch::empty({K, M / 2}, torch::dtype(torch::kUInt8).device(device));
    scale_inv_t = torch::empty({c_scale_rows, c_scale_cols}, torch::dtype(torch::kUInt8).device(device));
  }
  // Allocate amax (+ sg if tk_swizzle) in one buffer to save a torch::empty call
  torch::Tensor amax_buf;
  float* amax_ptr;
  float* sg_ptr = nullptr;
  if (tk_swizzle) {
    amax_buf = torch::empty({2}, torch::dtype(torch::kFloat32).device(device));
    amax_ptr = amax_buf.data_ptr<float>();
    sg_ptr = amax_ptr + 1;
  } else {
    amax_buf = torch::empty({1}, torch::dtype(torch::kFloat32).device(device));
    amax_ptr = amax_buf.data_ptr<float>();
  }

  // Step 1: Zero + compute amax (+ scale if tk_swizzle) in fused host call
  cudaMemsetAsync(amax_ptr, 0, sizeof(float), stream);
  if (tk_swizzle) {
    launch_fused_amax_bf16_with_scale(
        input.data_ptr(), amax_ptr, sg_ptr, 1.0f / 2688.0f, M * K, stream);
  } else {
    launch_fused_amax_bf16(input.data_ptr(), amax_ptr, M * K, stream);
  }

  // Build TE output wrapper with SHARED amax for row + col
  TensorWrapper te_output(NVTE_NVFP4_1D_SCALING);
  te_output.set_rowwise_data(fp4_data.data_ptr(), DType::kFloat4E2M1, r_shape);
  te_output.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E4M3, r_si_shape);
  te_output.set_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
  te_output.set_columnwise_data(fp4_data_t.data_ptr(), DType::kFloat4E2M1, c_shape);
  te_output.set_columnwise_scale_inv(scale_inv_t.data_ptr(), DType::kFloat8E4M3, c_si_shape);
  te_output.set_columnwise_amax(amax_ptr, DType::kFloat32, std::vector<size_t>{1});
  if (tk_swizzle) {
    // REMOVED: TE API removed
  }

  // Step 2: Quantize — dispatch to custom or standard TE path
  auto te_input = make_te_input(input);
  QuantizationConfigWrapper quant_config;
#ifdef CUSTOM_QUANT_ENABLED
  if (custom_quant) {
    custom_nvfp4_quantize_transpose(
        te_input.data(), te_output.data(), stream);
  } else
#endif
  {
    nvte_quantize_v2(te_input.data(), te_output.data(), quant_config, stream);
  }

  // Views into amax_buf for return
  auto amax = amax_buf.narrow(0, 0, 1);
  torch::Tensor sg;
  if (tk_swizzle) {
    sg = amax_buf.narrow(0, 1, 1);
  }

  return std::make_tuple(fp4_data, scale_inv, fp4_data_t, scale_inv_t, amax, amax, sg);
}

// =========================================================================
// 2. group_nvfp4_quantize: grouped FP4 quantize with per-split amax
// =========================================================================
std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                       torch::Tensor, torch::Tensor, torch::Tensor>>
group_nvfp4_quantize(
    torch::Tensor input,
    std::vector<int64_t> split_sections,
    bool tk_swizzle = false,
    bool custom_quant = false
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(input.dim() == 2);
  int64_t total_rows = input.size(0), K = input.size(1);
  TORCH_CHECK(total_rows % 32 == 0);
  TORCH_CHECK(K % 32 == 0);

  int64_t sum_splits = 0;
  for (auto s : split_sections) sum_splits += s;
  TORCH_CHECK(sum_splits == total_rows);

  size_t num_tensors = split_sections.size();
  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();

  auto te_input = make_te_input(input);

  // ─── CRITICAL: Rowwise FP4 data must be ONE contiguous buffer ───
  // The TE kernel uses TMA (cp_async_bulk_tensor_2d_shared_to_global) for
  // rowwise data, which writes at global row offsets. It creates the TMA
  // tensor map over output_list[0]->data with total_rows. All per-split
  // rowwise outputs must therefore be contiguous views into one buffer.
  auto fp4_data_all = torch::empty({total_rows, K / 2},
                                    torch::dtype(torch::kUInt8).device(device));

  std::vector<size_t> c_split_sections(num_tensors);
  std::vector<torch::Tensor> fp4_datas(num_tensors);
  std::vector<torch::Tensor> scale_invs(num_tensors);
  std::vector<torch::Tensor> fp4_data_ts(num_tensors);
  std::vector<torch::Tensor> scale_inv_ts(num_tensors);
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

    // Rowwise FP4 data: narrow from contiguous buffer (required by TMA)
    fp4_datas[i] = fp4_data_all.narrow(0, row_offset, M_i);

    // Rowwise scales: separate allocations OK (kernel uses per-split pointers)
    int64_t r_scale_rows = round_up(M_i, 128);
    int64_t r_scale_cols = round_up(K / 16, 4);
    scale_invs[i] = torch::empty({r_scale_rows, r_scale_cols},
                                  torch::dtype(torch::kUInt8).device(device));

    // Columnwise data + scales: separate allocations OK (kernel uses direct GMEM stores)
    fp4_data_ts[i] = torch::empty({K, M_i / 2}, torch::dtype(torch::kUInt8).device(device));
    int64_t c_scale_rows = round_up(K, 128);
    int64_t c_scale_cols = round_up(M_i / 16, 4);
    scale_inv_ts[i] = torch::empty({c_scale_rows, c_scale_cols},
                                    torch::dtype(torch::kUInt8).device(device));

    amaxs[i] = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device));

    std::vector<size_t> r_shape = {(size_t)M_i, (size_t)K};
    std::vector<size_t> r_si_shape = {(size_t)r_scale_rows, (size_t)r_scale_cols};
    std::vector<size_t> c_shape = {(size_t)K, (size_t)M_i};
    std::vector<size_t> c_si_shape = {(size_t)c_scale_rows, (size_t)c_scale_cols};

    te_outputs[i].set_rowwise_data(fp4_datas[i].data_ptr(), DType::kFloat4E2M1, r_shape);
    te_outputs[i].set_rowwise_scale_inv(scale_invs[i].data_ptr(), DType::kFloat8E4M3, r_si_shape);
    te_outputs[i].set_amax(amaxs[i].data_ptr<float>(), DType::kFloat32, std::vector<size_t>{1});
    te_outputs[i].set_columnwise_data(fp4_data_ts[i].data_ptr(), DType::kFloat4E2M1, c_shape);
    te_outputs[i].set_columnwise_scale_inv(scale_inv_ts[i].data_ptr(), DType::kFloat8E4M3, c_si_shape);
    te_outputs[i].set_columnwise_amax(amaxs[i].data_ptr<float>(), DType::kFloat32, std::vector<size_t>{1});
    if (tk_swizzle) {
      // REMOVED: TE API removed
    }

    row_offset += M_i;
  }

  std::vector<NVTETensor> output_ptrs(num_tensors);
  for (size_t i = 0; i < num_tensors; ++i) {
    output_ptrs[i] = te_outputs[i].data();
  }

  // Build cumulative element offsets on device for the grouped kernel.
  // explicit cudaMemsetAsync on current stream to prevent Heisenbug race
  auto all_amaxes = torch::empty({(int64_t)num_tensors}, torch::dtype(torch::kFloat32).device(device));
  cudaMemsetAsync(all_amaxes.data_ptr<float>(), 0, num_tensors * sizeof(float), stream);
  std::vector<int64_t> h_elem_offsets(num_tensors + 1);
  h_elem_offsets[0] = 0;
  for (size_t i = 0; i < num_tensors; ++i) {
    h_elem_offsets[i + 1] = h_elem_offsets[i] + split_sections[i] * K;
  }
  
  auto pinned_options = torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true);
  auto pinned_offsets = torch::empty({(int64_t)(num_tensors + 1)}, pinned_options);
  std::memcpy(pinned_offsets.data_ptr(), h_elem_offsets.data(), (num_tensors + 1) * sizeof(int64_t));
  auto d_elem_offsets = pinned_offsets.to(device, /*non_blocking=*/true);
  // Store pinned_offsets in result or capture it to prevent destruction?
  // Actually, we can just use torch::zeros to prevent garbage memory issues.
  launch_grouped_amax_bf16(input.data_ptr(), all_amaxes.data_ptr<float>(),
                           d_elem_offsets.data_ptr<int64_t>(),
                           (int)num_tensors, total_rows * K, stream);
  // Copy per-split amaxes to each split's amax tensor
  for (size_t i = 0; i < num_tensors; ++i) {
    cudaMemcpyAsync(amaxs[i].data_ptr(), all_amaxes.data_ptr<float>() + i,
                    sizeof(float), cudaMemcpyDeviceToDevice, stream);
  }

  // Step 2: Dispatch quantize — custom or standard TE path
  QuantizationConfigWrapper quant_config;
#ifdef CUSTOM_QUANT_ENABLED
  if (custom_quant) {
    custom_nvfp4_group_quantize_transpose(
        te_input.data(), output_ptrs.data(),
        c_split_sections.data(), num_tensors, stream);
  } else
#endif
  {
    throw std::runtime_error(
        "group_nvfp4_quantize: nvte_group_nvfp4_quantize_with_amax not available in this TE version. "
        "Use USE_TK_QUANT=1 for TK-based group quantization.");  }

  std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                         torch::Tensor, torch::Tensor, torch::Tensor>> result;
  for (size_t i = 0; i < num_tensors; ++i) {
    // Return pinned_offsets to python so it stays alive until the stream finishes
    result.emplace_back(fp4_datas[i], scale_invs[i],
                        fp4_data_ts[i], scale_inv_ts[i], amaxs[i],
                        (i == 0) ? pinned_offsets : torch::Tensor());
  }
  return result;
}

// =========================================================================
// group_nvfp4_quantize_tk — sg + b_sg computed inside CUDA quantize kernel.
// Returns: (fp4_row, sc_row, fwd_b_sg, fp4_col, sc_col, dgrad_b_sg, sg)
// =========================================================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, torch::Tensor,
           torch::Tensor, torch::Tensor>
group_nvfp4_quantize_tk(
    torch::Tensor input,
    std::vector<int64_t> split_sections
) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous());
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16 && input.dim() == 2);
  TORCH_CHECK(input.size(0) % 128 == 0 && input.size(1) % 128 == 0);
  const int64_t total_rows = input.size(0), K = input.size(1);
  const size_t N = split_sections.size();
  auto stream = at::cuda::getCurrentCUDAStream();
  auto device = input.device();
  constexpr int64_t Nb = 256;
  const int64_t ntk_r = K / 64, ntm_c = K / 128;
  const int64_t r_sc_cols = round_up(K / 16, 4);
  const int64_t dgrad_tiles_per = K / Nb;

  // ── Pre-scan: compute sizes for contiguous allocations ──
  int64_t sum_splits = 0, total_ntm_r = 0, total_fwd_tiles = 0;
  int64_t total_sc_row_bytes = 0;
  int64_t total_col_fp4_bytes = 0, total_col_sc_bytes = 0;
  const int64_t sc_rows_c = round_up(K, (int64_t)128);

  // Stack-allocated arrays (N is always small, typically 3)
  int64_t h_elem_offsets[17];  // N+1, max 16 splits
  int64_t col_fp4_offsets[16]; // byte offsets into contiguous colwise fp4
  int64_t col_sc_offsets[16];  // byte offsets into contiguous colwise scales
  int64_t sc_row_offsets[16];
  TORCH_CHECK(N <= 16, "group_nvfp4_quantize_tk: max 16 splits");
  h_elem_offsets[0] = 0;

  for (size_t i = 0; i < N; ++i) {
    const int64_t M_i = split_sections[i];
    TORCH_CHECK(M_i % 128 == 0);
    sum_splits += M_i;
    total_ntm_r += M_i / 128;
    total_fwd_tiles += M_i / Nb;
    h_elem_offsets[i + 1] = h_elem_offsets[i] + M_i * K;

    // Colwise fp4: K × (M_i/2) bytes
    col_fp4_offsets[i] = total_col_fp4_bytes;
    total_col_fp4_bytes += K * (M_i / 2);

    // Colwise scales: sc_rows_c × round_up(M_i/16, 4) bytes
    const int64_t sc_cols_i = round_up(M_i / 16, (int64_t)4);
    col_sc_offsets[i] = total_col_sc_bytes;
    total_col_sc_bytes += sc_rows_c * sc_cols_i;

    // Rowwise scale offsets
    const int64_t r_sc_rows_i = round_up(M_i, (int64_t)128);
    sc_row_offsets[i] = total_sc_row_bytes;
    total_sc_row_bytes += r_sc_rows_i * r_sc_cols;
  }
  TORCH_CHECK(sum_splits == total_rows);
  const int64_t total_dgrad_tiles = (int64_t)N * dgrad_tiles_per;

  // ── MEGA-ALLOCATION: pack everything into ONE torch::empty ──
  // Layout: [wc_fp4_row | wc_sc_row | ZERO:{all_col_fp4 | all_col_sc | meta(4-aligned) | d_offsets(8-aligned)}]
  auto opts_u8 = torch::dtype(torch::kUInt8).device(device);

  const int64_t fp4_row_bytes = total_rows * (K / 2);
  const int64_t meta_bytes_raw = (int64_t)(N + N + total_fwd_tiles + total_dgrad_tiles) * (int64_t)sizeof(float);
  const int64_t meta_bytes = (meta_bytes_raw + 7) & ~(int64_t)7;  // 8-byte align for d_offsets after
  const int64_t offsets_bytes = (int64_t)(N + 1) * (int64_t)sizeof(int64_t);

  // Non-zero region
  const int64_t nonzero_bytes = fp4_row_bytes + total_sc_row_bytes;
  // Zero region (contiguous for single memset)
  const int64_t zero_bytes = total_col_fp4_bytes + total_col_sc_bytes + meta_bytes + offsets_bytes;
  const int64_t total_mega_bytes = nonzero_bytes + zero_bytes;

  auto mega_buf = torch::empty({total_mega_bytes}, opts_u8);
  char* base = (char*)mega_buf.data_ptr();

  // Carve out pointers
  char* wc_fp4_row_ptr     = base;
  char* wc_sc_row_ptr      = base + fp4_row_bytes;
  char* zero_start         = base + nonzero_bytes;
  char* all_col_fp4_ptr    = zero_start;
  char* all_col_sc_ptr     = zero_start + total_col_fp4_bytes;
  float* meta_ptr          = (float*)(zero_start + total_col_fp4_bytes + total_col_sc_bytes);
  int64_t* d_offsets_ptr   = (int64_t*)((char*)meta_ptr + meta_bytes);

  // Single memset for entire zero region
  cudaMemsetAsync(zero_start, 0, zero_bytes, stream);

  // Upload element offsets (within the zeroed region, overwrite after memset)
  cudaMemcpyAsync(d_offsets_ptr, h_elem_offsets, (N + 1) * sizeof(int64_t),
                  cudaMemcpyHostToDevice, stream);

  // Metadata pointers
  const int64_t meta_size = N + N + total_fwd_tiles + total_dgrad_tiles;
  float* all_amaxes_ptr  = meta_ptr;
  float* sg_cat_ptr      = meta_ptr + N;
  float* fwd_b_sg_ptr    = meta_ptr + 2 * N;
  float* dgrad_b_sg_ptr  = meta_ptr + 2 * N + total_fwd_tiles;

  // Create tensor views for the parts we return to Python (all share mega_buf storage)
  auto wc_fp4_row  = mega_buf.narrow(0, 0, fp4_row_bytes).reshape({total_rows, K / 2});
  auto wc_sc_row_flat = mega_buf.narrow(0, fp4_row_bytes, total_sc_row_bytes);
  auto all_col_fp4 = mega_buf.narrow(0, nonzero_bytes, total_col_fp4_bytes);
  auto all_col_sc  = mega_buf.narrow(0, nonzero_bytes + total_col_fp4_bytes, total_col_sc_bytes);

  // ── Build TE wrappers pointing into mega-buffer ──
  std::vector<TensorWrapper> te_outputs;
  te_outputs.reserve(N);
  std::vector<NVTETensor> output_ptrs(N);
  std::vector<size_t> c_split(N);
  int64_t row_offset = 0;

  for (size_t i = 0; i < N; ++i) {
    const int64_t M_i = split_sections[i];
    c_split[i] = (size_t)M_i;
    char* fp4_col_p = all_col_fp4_ptr + col_fp4_offsets[i];
    char* sc_col_p = all_col_sc_ptr + col_sc_offsets[i];
    const int64_t sc_cols_i = round_up(M_i / 16, (int64_t)4);

    te_outputs.emplace_back(NVTE_NVFP4_1D_SCALING);
    void* fp4_row_p = wc_fp4_row_ptr + row_offset * (K / 2);
    void* sc_row_p = wc_sc_row_ptr + sc_row_offsets[i];
    const int64_t r_sc_rows_i = round_up(M_i, (int64_t)128);

    te_outputs[i].set_rowwise_data(fp4_row_p, DType::kFloat4E2M1, std::vector<size_t>{(size_t)M_i, (size_t)K});
    te_outputs[i].set_rowwise_scale_inv(sc_row_p, DType::kFloat8E4M3, std::vector<size_t>{(size_t)r_sc_rows_i, (size_t)r_sc_cols});
    te_outputs[i].set_amax(all_amaxes_ptr + i, DType::kFloat32, std::vector<size_t>{1});
    te_outputs[i].set_columnwise_data(fp4_col_p, DType::kFloat4E2M1, std::vector<size_t>{(size_t)K, (size_t)M_i});
    te_outputs[i].set_columnwise_scale_inv(sc_col_p, DType::kFloat8E4M3,
        std::vector<size_t>{(size_t)sc_rows_c, (size_t)sc_cols_i});
    te_outputs[i].set_columnwise_amax(all_amaxes_ptr + i, DType::kFloat32, std::vector<size_t>{1});
    // REMOVED: TE API removed
    output_ptrs[i] = te_outputs[i].data();
    row_offset += M_i;
  }

  // ── Kernel launches ──
  launch_grouped_amax_bf16(input.data_ptr(), all_amaxes_ptr,
                           d_offsets_ptr,
                           (int)N, total_rows * K, stream);
#ifdef CUSTOM_QUANT_ENABLED
  auto te_input = make_te_input(input);
  custom_nvfp4_group_quantize_transpose_tk(
      te_input.data(), output_ptrs.data(), c_split.data(), N,
      sg_cat_ptr, fwd_b_sg_ptr, dgrad_b_sg_ptr,
      (int)Nb, stream);
#else
  TORCH_CHECK(false, "group_nvfp4_quantize_tk requires CUSTOM_QUANT_ENABLED");
#endif

  // ── Post-kernel: derive views (single loop, no allocations) ──
  auto wc_fp4_row_tk = wc_fp4_row.view(torch::kFloat4_e2m1fn_x2);
  auto wc_sc_row = wc_sc_row_flat.reshape({total_ntm_r, ntk_r, 512}).view(torch::kFloat8_e4m3fn);

  // Slice metadata — create tensor wrapping the meta region of mega_buf
  const int64_t meta_start = nonzero_bytes + total_col_fp4_bytes + total_col_sc_bytes;
  auto meta_view = mega_buf.narrow(0, meta_start, meta_size * (int64_t)sizeof(float));
  auto meta_f32 = meta_view.view(torch::kFloat32);
  auto sg_cat = meta_f32.narrow(0, N, N).clone();  // needs writable copy
  auto fwd_b_sg = meta_f32.narrow(0, 2 * N, total_fwd_tiles);
  auto dgrad_b_sg = meta_f32.narrow(0, 2 * N + total_fwd_tiles, total_dgrad_tiles);

  std::vector<torch::Tensor> fp4_col_tk(N), sc_col_views(N);
  int64_t fwd_tile_offset = 0;
  int64_t fp4_col_elem_offset = 0, sc_col_elem_offset = 0;
  for (size_t i = 0; i < N; ++i) {
    const int64_t M_i = split_sections[i];
    sg_cat[i] = fwd_b_sg[fwd_tile_offset];
    fwd_tile_offset += M_i / Nb;

    const int64_t fp4_elems_i = K * (M_i / 2);
    auto fp4_slice = all_col_fp4.narrow(0, fp4_col_elem_offset, fp4_elems_i)
                                .reshape({K, M_i / 2});
    fp4_col_tk[i] = fp4_slice.view(torch::kFloat4_e2m1fn_x2);
    fp4_col_elem_offset += fp4_elems_i;

    const int64_t ntk_c_i = M_i / 64;
    const int64_t sc_cols_i = round_up(M_i / 16, (int64_t)4);
    const int64_t sc_elems_i = sc_rows_c * sc_cols_i;
    auto sc_slice = all_col_sc.narrow(0, sc_col_elem_offset, sc_elems_i)
                              .reshape({sc_rows_c, sc_cols_i});
    sc_col_views[i] = sc_slice.reshape({ntm_c, ntk_c_i, 512}).view(torch::kFloat8_e4m3fn);
    sc_col_elem_offset += sc_elems_i;
  }

  return std::make_tuple(wc_fp4_row_tk, wc_sc_row, fwd_b_sg,
                         fp4_col_tk, sc_col_views, dgrad_b_sg, sg_cat, mega_buf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_nvfp4_quantize_v2", &fast_nvfp4_quantize_v2,
        "Direct bf16 → NVFP4 row+col quantize (custom fused amax)",
        py::arg("input"), py::arg("encode_centric") = false,
        py::arg("tk_swizzle") = false, py::arg("custom_quant") = false);
  m.def("group_nvfp4_quantize", &group_nvfp4_quantize,
        "Grouped FP4 quantize: stacked bf16 → per-split FP4 (1 kernel)",
        py::arg("input"), py::arg("split_sections"),
        py::arg("tk_swizzle") = false, py::arg("custom_quant") = false);
  m.def("group_nvfp4_quantize_tk", &group_nvfp4_quantize_tk,
        "Grouped FP4 quantize → TK-native output (sg+b_sg in CUDA kernel)",
        py::arg("input"), py::arg("split_sections"));
}
