/*************************************************************************
 * Grouped NVFP4 quantize+transpose kernel for dim=1 (column splits).
 *
 * Analogous to group_quantize_transpose.cuh but splits are along the
 * COLUMN dimension.  Each split (column-group) gets its own amax / sg
 * and separate scale + transpose output buffers.
 *
 * Requirement: every split_section must be a multiple of 128 so that
 *              128-wide chunks never straddle a group boundary.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_GROUP_QUANTIZE_TRANSPOSE_DIM1_NVFP4_CUH_
#define TRANSFORMER_ENGINE_GROUP_QUANTIZE_TRANSPOSE_DIM1_NVFP4_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include "util/cast_common.h"
#include "util/math.h"
#include "util/ptx.cuh"
#include "util/utils.cuh"
#include "core.cuh"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

namespace group_quantize_transpose_dim1_kernel {

using namespace quantization_and_transposition_SF;
using namespace core;
using namespace ptx;

#if FP4_TYPE_SUPPORTED

/* ─── Args ────────────────────────────────────────────────────────── */
constexpr int kMaxGroupsDim1 = 64;

struct Dim1GroupArgs {
  /* Per-group amax (host-zeroed before kernel; kernel does NOT write) */
  void *rowwise_amax_list[kMaxGroupsDim1];
  /* Per-group rowwise scale output buffers, each (rows, group_scale_stride) */
  void *output_rowwise_scale_inv_list[kMaxGroupsDim1];
  /* Per-group rowwise scale stride (= ((N_i/16)+3)/4*4 ) */
  int  output_rowwise_scale_stride[kMaxGroupsDim1];

  /* Per-group colwise (transpose) amax */
  void *colwise_amax_list[kMaxGroupsDim1];
  /* Per-group colwise fp4 data, each (N_i, rows/2) */
  void *output_colwise_data_list[kMaxGroupsDim1];
  /* Per-group colwise scale output */
  void *output_colwise_scale_inv_list[kMaxGroupsDim1];
  /* Per-group colwise scale stride (= ((rows/16)+3)/4*4 ) */
  int  output_colwise_scale_stride[kMaxGroupsDim1];

  /* Prefix sum of column counts: range[0]=0, range[i+1]=range[i]+N_i */
  int  col_split_range[kMaxGroupsDim1 + 1];
  int  num_groups;
  bool swizzle_scales;

  /* TK sg output: sg[i] = amax[i] / 2688 */
  float *sg_output;
};

/* ─── Helper to find which column-group a column belongs to ──────── */
__device__ __forceinline__ int GetColumnGroupId(
    const Dim1GroupArgs *args, int col_offset) {
  int gid = 0;
  while (args->col_split_range[gid + 1] <= col_offset) ++gid;
  return gid;
}

/* ─── Constants (same as dim-0 kernel) ───────────────────────────── */
constexpr size_t SCALE_DIM = 16;

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_NUM = 128;

constexpr size_t SCALES_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM;
constexpr size_t SCALES_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM;

constexpr size_t TILE_DIM_Y = 32;
constexpr size_t TILE_DIM_X = 128;

constexpr size_t TILES_Y = CHUNK_DIM_Y / TILE_DIM_Y;
constexpr size_t TILES_X = CHUNK_DIM_X / TILE_DIM_X;
constexpr size_t STAGES  = TILES_Y * TILES_X;

constexpr size_t BUFFS_NUM   = 2;
constexpr size_t BUFF_DIM_Y  = TILE_DIM_Y;
constexpr size_t BUFF_DIM_X  = TILE_DIM_X;
constexpr size_t BUFF_IN_SIZE  = BUFF_DIM_Y * BUFF_DIM_X;
constexpr size_t BUFF_OUT_SIZE = BUFF_DIM_Y * ((BUFF_DIM_X * 4) / 8);

constexpr size_t BUFF_OUT_T_DIM_Y = BUFF_DIM_X;
constexpr size_t BUFF_OUT_T_DIM_X = (BUFF_DIM_Y * 4) / 8;
constexpr size_t BUFF_OUT_T_SIZE  = BUFF_OUT_T_DIM_Y * BUFF_OUT_T_DIM_X;

constexpr size_t PACK_SIZE = 8;
constexpr size_t WAVES     = SCALE_DIM / PACK_SIZE;

constexpr size_t SCALING_FACTORS_PER_TILE_X = TILE_DIM_X / SCALE_DIM;
constexpr size_t THREADS_X_ROWWISE = SCALING_FACTORS_PER_TILE_X;
constexpr size_t THREADS_Y_ROWWISE = THREADS_NUM / THREADS_X_ROWWISE;

constexpr size_t ITERATIONS_NORMAL    = BUFF_DIM_Y / THREADS_Y_ROWWISE;
constexpr size_t ITERATIONS_TRANSPOSE = BUFF_DIM_Y / SCALE_DIM;  // BUFF_IN_DIM_Y
constexpr size_t BUFF_OUT_IT_OFFSET   = BUFF_OUT_T_DIM_X / ITERATIONS_TRANSPOSE;

constexpr size_t TOTAL_BANKS_WIDTH = (32 * 4 * 8) / 4;
constexpr size_t THREADS_PER_BANK  = TOTAL_BANKS_WIDTH / SCALE_DIM;

/* ─── Main kernel (bf16 only, no activation fusion) ──────────────── */
template <bool USE_STOCHASTIC_ROUNDING, bool RETURN_TRANSPOSE>
__global__ void __launch_bounds__(THREADS_NUM)
group_quantize_transpose_dim1_nvfp4_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    nvfp4_scale_t *const scales_ptr,   /* unused (per-group ptrs in args) */
    const float *noop,
    const size_t rows, const size_t cols,
    const size_t global_scale_stride,
    const size_t *rng_state,
    Dim1GroupArgs kernel_args) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using IType  = bf16;
  using IType2 = typename ptx::FPx2<IType>;

  if (noop != nullptr && noop[0] == 1.0f) return;

  /* ── RNG setup ─────────────────────────────────────────────────── */
  const size_t rng_sequence =
      threadIdx.x + blockIdx.x * THREADS_NUM + blockIdx.y * gridDim.x * THREADS_NUM;
  const size_t rng_seed   = rng_state ? rng_state[0] : 0;
  const size_t rng_offset = rng_state ? rng_state[1] : 0;
  transformer_engine::curanddx::detail::philox4x32_native_state<10> rng;
  rng.init(rng_seed, rng_sequence, rng_offset);
  uint4 random_uint4 = USE_STOCHASTIC_ROUNDING ? rng.generate4() : uint4{0,0,0,0};
  int rnd_idx = 0;

  /* ── Block / thread coordinates ────────────────────────────────── */
  const size_t block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t block_offset_X = blockIdx.x * CHUNK_DIM_X;
  const size_t chunk_rows = rows - block_offset_Y;

  const size_t tid_Y_rowwise = threadIdx.x / THREADS_X_ROWWISE;
  const size_t tid_X_rowwise = threadIdx.x % THREADS_X_ROWWISE;
  const size_t tid_X_colwise = threadIdx.x;

  const size_t thread_offset_Y_rowwise = tid_Y_rowwise;
  const size_t thread_offset_X_rowwise = tid_X_rowwise * SCALE_DIM;
  const size_t thread_offset_X_colwise = tid_X_colwise;

  const size_t row_base_rowwise = block_offset_Y + thread_offset_Y_rowwise;
  const size_t col_base_colwise = block_offset_X + thread_offset_X_colwise;

  const bool col_out_of_bounds_colwise = (col_base_colwise >= cols);

  const int thread_lane  = threadIdx.x % THREADS_PER_WARP;
  const int bank_group   = thread_lane / THREADS_PER_BANK;
  const bool is_master_thread = (threadIdx.x == 0);

  /* ── Determine column-group (constant for the entire chunk) ──── */
  const int group_id = GetColumnGroupId(&kernel_args, (int)block_offset_X);
  const int col_group_start = kernel_args.col_split_range[group_id];
  const int col_group_end   = kernel_args.col_split_range[group_id + 1];
  const int col_group_size  = col_group_end - col_group_start;

  /* Per-group pointers */
  float *amax_rowwise_ptr = reinterpret_cast<float*>(kernel_args.rowwise_amax_list[group_id]);
  nvfp4_scale_t *split_rowwise_scale_ptr =
      reinterpret_cast<nvfp4_scale_t*>(kernel_args.output_rowwise_scale_inv_list[group_id]);
  const int group_scale_stride = kernel_args.output_rowwise_scale_stride[group_id];

  float S_enc_rowwise = 1.0f, S_dec_rowwise = 1.0f;
  {
    float s = (amax_rowwise_ptr == nullptr) ? 1.0f
        : compute_global_encode_scaling_factor_FP4(*amax_rowwise_ptr);
    S_enc_rowwise = s;
    S_dec_rowwise = 1.0f / s;
  }

  /* Write sg once per group (first block touching this group) */
  if (kernel_args.sg_output != nullptr && is_master_thread &&
      blockIdx.y == 0 && block_offset_X == (size_t)col_group_start) {
    float amax_val = amax_rowwise_ptr ? *amax_rowwise_ptr : 0.0f;
    kernel_args.sg_output[group_id] = amax_val / 2688.0f;
  }

  /* Colwise (transpose) pointers */
  float S_enc_colwise = S_enc_rowwise, S_dec_colwise = S_dec_rowwise;
  float *amax_colwise_ptr = nullptr;
  fp4e2m1x2 *split_colwise_data_ptr = nullptr;
  nvfp4_scale_t *split_colwise_scale_ptr = nullptr;
  int split_colwise_scale_stride = 0;
  if constexpr (RETURN_TRANSPOSE) {
    amax_colwise_ptr = reinterpret_cast<float*>(kernel_args.colwise_amax_list[group_id]);
    split_colwise_data_ptr = reinterpret_cast<fp4e2m1x2*>(
        kernel_args.output_colwise_data_list[group_id]);
    split_colwise_scale_ptr = reinterpret_cast<nvfp4_scale_t*>(
        kernel_args.output_colwise_scale_inv_list[group_id]);
    split_colwise_scale_stride = kernel_args.output_colwise_scale_stride[group_id];
    if (amax_colwise_ptr) {
      float s = compute_global_encode_scaling_factor_FP4(*amax_colwise_ptr);
      S_enc_colwise = s;  S_dec_colwise = 1.0f / s;
    }
  }

  float thread_amax = 0.0f;

  /* ── Shared memory ─────────────────────────────────────────────── */
  constexpr size_t buff_elems       = BUFF_DIM_Y * BUFF_DIM_X;
  constexpr size_t buff_elems_total = BUFFS_NUM * buff_elems;

  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE((buff_elems_total * 4) / 8, TMA_SHMEM_ALIGNMENT);

  extern __shared__ char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  uintptr_t dshmem = (base_shmem_ptr + TMA_SHMEM_ALIGNMENT - 1) &
                     ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  IType      *in_sh            = reinterpret_cast<IType*>(dshmem);
  fp4e2m1x2  *out_data_sh      = reinterpret_cast<fp4e2m1x2*>(dshmem + buff_size_aligned_in);
  fp4e2m1x2  *out_t_data_sh    = reinterpret_cast<fp4e2m1x2*>(
      dshmem + buff_size_aligned_in + buff_size_aligned_out);
  nvfp4_scale_t *out_colwise_scales_sh = reinterpret_cast<nvfp4_scale_t*>(
      dshmem + buff_size_aligned_in + 2 * buff_size_aligned_out);

  constexpr size_t shmem_buff_size = buff_size_aligned_in / BUFFS_NUM;

  /* ── Barriers ──────────────────────────────────────────────────── */
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[STAGES];
  initialize_barriers<STAGES, THREADS_NUM>(mbar, is_master_thread);

  /* ── First TMA load ────────────────────────────────────────────── */
  copy_2d_to_shared(&in_sh[0], &tensor_map_input,
                     block_offset_X, block_offset_Y, shmem_buff_size,
                     &mbar[0], is_master_thread);

  /* ── Stage loop ────────────────────────────────────────────────── */
#pragma unroll
  for (size_t stage = 0; stage < STAGES; ++stage) {
    const size_t buff       = stage % BUFFS_NUM;
    const size_t next_stage = stage + 1;
    const size_t stage_offset_Y = stage * BUFF_DIM_Y;

    const size_t buff_offset_in    = buff * BUFF_IN_SIZE;
    const size_t buff_offset_out   = buff * BUFF_OUT_SIZE;
    const size_t buff_offset_out_t = buff * BUFF_OUT_T_SIZE;

    /* Prefetch next stage */
    if (next_stage < STAGES) {
      ptx::cp_async_bulk_wait_group_read<1>();
      const size_t next_buff = next_stage % BUFFS_NUM;
      copy_2d_to_shared(&in_sh[next_buff * BUFF_IN_SIZE], &tensor_map_input,
                         block_offset_X,
                         block_offset_Y + next_stage * BUFF_DIM_Y,
                         shmem_buff_size, &mbar[next_stage], is_master_thread);
    }

    ptx::fence_proxy_async_shared_cta();
    ptx::mbarrier_wait_parity(&mbar[stage], 0);

    float block_amax = 0.0f;

    /* ── COLWISE scaling (transpose) ─────────────────────────────── */
    if constexpr (RETURN_TRANSPOSE) {
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_TRANSPOSE; ++it) {
        const size_t in_off_Y = it * SCALE_DIM;
        const size_t shmem_base_in =
            buff_offset_in + in_off_Y * BUFF_DIM_X + thread_offset_X_colwise;
        const size_t shmem_base_out_t =
            buff_offset_out_t + thread_offset_X_colwise * BUFF_OUT_T_DIM_X + it * BUFF_OUT_IT_OFFSET;

        block_amax = 0.0f;
        IType in_colwise_IType[SCALE_DIM];

        /* BF16 fast path */
        IType block_amax_f16 = static_cast<IType>(0.0f);
#pragma unroll
        for (int i = 0; i < (int)SCALE_DIM; ++i) {
          in_colwise_IType[i] = in_sh[shmem_base_in + i * BUFF_DIM_X];
          block_amax_f16 = __hmax(block_amax_f16, __habs(in_colwise_IType[i]));
        }
        block_amax = static_cast<float>(block_amax_f16);

        const nvfp4_scale_t S_dec_b_fp8 =
            compute_decoding_scaling_factor(block_amax, S_enc_colwise);

        /* Store colwise scale to shmem */
        const size_t scale_idx_sh =
            tid_X_colwise * SCALES_PER_CHUNK_Y + stage * ITERATIONS_TRANSPOSE + it;
        out_colwise_scales_sh[scale_idx_sh] = S_dec_b_fp8;

        constexpr float float_max = detail::TypeExtrema<float>::max;
        const float block_scale_inverse = fminf(
            1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_colwise), float_max);
        const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

        fp4e2m1x4 regs[SCALE_DIM / 4];
#pragma unroll
        for (int e = 0; e < (int)(SCALE_DIM / 4); ++e) {
          const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);
          const uint64_t elts = *reinterpret_cast<uint64_t*>(&in_colwise_IType[4*e]);
          regs[e] = ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
              elts, block_scale_inverse_2x, rbits);
        }

        const int group = thread_lane / 16;
        uint32_t val[2];
        uint32_t *regs_4x = reinterpret_cast<uint32_t*>(regs);
        switch (group) {
          case 0: val[0]=regs_4x[0]; val[1]=regs_4x[1]; break;
          case 1: val[0]=regs_4x[1]; val[1]=regs_4x[0]; break;
        }
        uint32_t *out_t_u32 = reinterpret_cast<uint32_t*>(
            &out_t_data_sh[shmem_base_out_t]);
        out_t_u32[group]         = val[0];
        out_t_u32[(group+1)&1]   = val[1];
      }
    }

    /* ── ROWWISE scaling ─────────────────────────────────────────── */
    {
#pragma unroll
      for (size_t it = 0; it < ITERATIONS_NORMAL; ++it) {
        const size_t it_thread_offset_Y = thread_offset_Y_rowwise + it * THREADS_Y_ROWWISE;
        const size_t shmem_base_in  = buff_offset_in  + it_thread_offset_Y * BUFF_DIM_X;
        const size_t shmem_base_out = buff_offset_out + it_thread_offset_Y * ((BUFF_DIM_X*4)/8);
        const size_t it_offset_Y = stage_offset_Y + it * THREADS_Y_ROWWISE;

        block_amax = 0.0f;

        using IType2_t = typename ptx::FPx2<IType>;
        Vec<IType2_t, PACK_SIZE / 2> in_IType[WAVES];
        IType2_t thread_amax_2x = {static_cast<IType>(0.0f), static_cast<IType>(0.0f)};

#pragma unroll
        for (int w = 0; w < (int)WAVES; ++w) {
          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
          const size_t swizzled_thread_idx = thread_offset_X_rowwise + swizzled_group_idx;
          const size_t shmem_offset = shmem_base_in + swizzled_thread_idx;
          in_IType[w].load_from(&in_sh[shmem_offset]);
#pragma unroll
          for (int e = 0; e < (int)(PACK_SIZE/2); ++e)
            ptx::abs_max_2x(thread_amax_2x, thread_amax_2x, in_IType[w].data.elt[e]);
        }
        block_amax = static_cast<float>(
            __hmax(__habs(thread_amax_2x.x), __habs(thread_amax_2x.y)));

        /* Compute E4M3 block scale */
        const nvfp4_scale_t S_dec_b_fp8 =
            compute_decoding_scaling_factor(block_amax, S_enc_rowwise);

        /* Write rowwise scale to per-group buffer (dim-1 local col) */
        const size_t global_row = block_offset_Y + it_thread_offset_Y + stage_offset_Y * 0; /* already baked */
        const size_t abs_row = block_offset_Y + stage_offset_Y + it * THREADS_Y_ROWWISE + thread_offset_Y_rowwise;
        const size_t abs_col_in_16 = block_offset_X / SCALE_DIM + tid_X_rowwise;
        const size_t local_col_in_16 = abs_col_in_16 - (size_t)col_group_start / SCALE_DIM;
        const bool row_ok = (abs_row < rows);
        const bool col_ok = (abs_col_in_16 < cols / SCALE_DIM);

        if (row_ok && col_ok) {
          if (kernel_args.swizzle_scales) {
            const int tile_m = (int)abs_row / 128;
            const int row_in_tile = (int)abs_row % 128;
            const int tile_k = (int)local_col_in_16 / 4;
            const int k_byte = (int)local_col_in_16 % 4;
            const int j = row_in_tile % 32;
            const int grp = row_in_tile / 32;
            const int num_tiles_k = group_scale_stride / 4;
            const int tile_start = (tile_m * num_tiles_k + tile_k) * 512;
            const int byte_in_tile = j * 16 + grp * 4 + k_byte;
            reinterpret_cast<uint8_t*>(split_rowwise_scale_ptr)[tile_start + byte_in_tile] =
                reinterpret_cast<const uint8_t&>(S_dec_b_fp8);
          } else {
            const size_t idx = abs_row * group_scale_stride + local_col_in_16;
            split_rowwise_scale_ptr[idx] = S_dec_b_fp8;
          }
        }

        /* Quantize to FP4 */
        constexpr float float_max = detail::TypeExtrema<float>::max;
        const float block_scale_inverse = fminf(
            1.0f / (static_cast<float>(S_dec_b_fp8) * S_dec_rowwise), float_max);
        const float2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

#pragma unroll
        for (int w = 0; w < (int)WAVES; ++w) {
          Vec<fp4e2m1x4, PACK_SIZE / 4> out;
#pragma unroll
          for (int e = 0; e < (int)(PACK_SIZE/4); ++e) {
            const uint32_t rbits = get_rbits(rng, random_uint4, rnd_idx);
            const uint64_t elts = *reinterpret_cast<uint64_t*>(&in_IType[w].data.elt[2*e]);
            out.data.elt[e] = ptx::mul_cvt_bf16_to_fp4_4x<USE_STOCHASTIC_ROUNDING>(
                elts, block_scale_inverse_2x, rbits);
          }
          const size_t swizzled_group_idx = ((w + bank_group) * PACK_SIZE) % SCALE_DIM;
          const size_t swizzled_idx = swizzled_group_idx + thread_offset_X_rowwise;
          out.store_to(&out_data_sh[shmem_base_out + swizzled_idx / 2]);
        }
      }
    }

    __builtin_assume(thread_amax >= 0);
    thread_amax = fmaxf(thread_amax, block_amax);

    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    /* TMA store rowwise FP4 (contiguous output) */
    if (is_master_thread) {
      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t*>(&tensor_map_output),
          block_offset_X, block_offset_Y + stage_offset_Y,
          reinterpret_cast<uint64_t*>(&out_data_sh[buff_offset_out]));
      ptx::cp_async_bulk_commit_group();
    }

    /* Direct GMEM store for transposed FP4 data (per-group output) */
    if constexpr (RETURN_TRANSPOSE) {
      __syncthreads();
      /* Transpose maps: original (row, col) → transposed (col_local, row)
         col_local = col - col_group_start */
      const size_t t_row = block_offset_X - col_group_start + threadIdx.x;  // local col → row in transpose
      const size_t t_col_base = block_offset_Y + stage_offset_Y;            // global row → col in transpose
      const size_t group_rows_t = (size_t)col_group_size;                   // num rows in transpose = N_i
      if (t_row < (size_t)col_group_size && split_colwise_data_ptr != nullptr) {
        for (size_t dy = 0; dy < BUFF_DIM_Y; dy += 2) {
          size_t t_col = t_col_base + dy;
          if (t_col + 1 < rows) {
            size_t sh_idx = buff_offset_out_t + threadIdx.x * BUFF_OUT_T_DIM_X + dy / 2;
            size_t gm_idx = t_row * (rows / 2) + t_col / 2;
            split_colwise_data_ptr[gm_idx] = out_t_data_sh[sh_idx];
          }
        }
      }
    }
  } /* end stages */

  /* Colwise scale store */
  if constexpr (RETURN_TRANSPOSE) {
    const size_t tid_Y_t = tid_X_colwise;
    const size_t local_col_t = block_offset_X - col_group_start + tid_Y_t;
    const bool colwise_ok = (local_col_t < (size_t)col_group_size);
    if (colwise_ok && split_colwise_scale_ptr != nullptr) {
      const size_t scales_block_offset_X_t = block_offset_Y / SCALE_DIM;
      const size_t count =
          (chunk_rows >= CHUNK_DIM_Y) ? SCALES_PER_CHUNK_Y : (chunk_rows / SCALE_DIM);
      if (kernel_args.swizzle_scales) {
        const int local_scale_row = (int)local_col_t;
        const int tile_m = local_scale_row / 128;
        const int row_in_tile = local_scale_row % 128;
        const int j = row_in_tile % 32;
        const int grp_sw = row_in_tile / 32;
        const int num_tiles_k = split_colwise_scale_stride / 4;
        for (size_t s = 0; s < count; ++s) {
          const int sc_col = (int)(scales_block_offset_X_t + s);
          const int tile_k = sc_col / 4;
          const int k_byte = sc_col % 4;
          const int tile_start = (tile_m * num_tiles_k + tile_k) * 512;
          const int byte_in_tile = j * 16 + grp_sw * 4 + k_byte;
          const size_t sh_idx = tid_Y_t * SCALES_PER_CHUNK_Y + s;  // use stage ordering
          reinterpret_cast<uint8_t*>(split_colwise_scale_ptr)[tile_start + byte_in_tile] =
              reinterpret_cast<const uint8_t&>(out_colwise_scales_sh[sh_idx]);
        }
      } else {
        for (size_t s = 0; s < count; ++s) {
          const size_t sc_col = scales_block_offset_X_t + s;
          const size_t sh_idx = tid_Y_t * SCALES_PER_CHUNK_Y + s;
          split_colwise_scale_ptr[local_col_t * split_colwise_scale_stride + sc_col] =
              out_colwise_scales_sh[sh_idx];
        }
      }
    }
  }

  destroy_barriers<STAGES>(mbar, is_master_thread);
#else
  NVTE_DEVICE_ERROR("sm_100 or higher is required.");
#endif
}

#endif  // FP4_TYPE_SUPPORTED
}  // namespace group_quantize_transpose_dim1_kernel
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GROUP_QUANTIZE_TRANSPOSE_DIM1_NVFP4_CUH_
