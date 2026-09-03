#pragma once

// BWD-only fusion-shape probe for the key-owned dK+dV schedule.
//
// This intentionally reuses the BF16-staged full-tile scaffold for Q/K and
// row-major dO/V dequant. It validates the ownership boundary we want next:
// keep K/V resident, stream Q/dO, compute P, live dP, dS, and consume dS into
// dK before it ever becomes a global dS_col artifact.

namespace mxfp4_key_owned_dkdv_fused_probe {

using C = mxfp4_full_tile_staged::config;
using dqacc_chunk_tile = st_fl<tkfa4::kRefTileM, 64>;
using dqacc_chunk_gl = gl<float, -1, -1, -1, -1, dqacc_chunk_tile>;
using dk_chunk_tile = st_bf<tkfa4::kRefTileM, 64>;
using dq_bf_chunk_gl = gl<bf16, -1, -1, -1, -1, dk_chunk_tile>;
using dk_chunk_gl = gl<bf16, -1, -1, -1, -1, dk_chunk_tile>;
using dv_bf_gl = gl<bf16, -1, -1, -1, C::Dvo>;
static constexpr int kMxfp4RowScaleTile = 128;
static constexpr int kMxfp4ScalesPerTile = 32 * 16;

__device__ inline int row128_scale_offset(int row_in_128, int dim) {
    const int mx_block = dim >> 5;
    return ((row_in_128 & 31) * 16) + ((row_in_128 >> 5) * 4) + mx_block;
}

__device__ inline int64_t row_dvo_scale_idx(int b, int h, int row, int dim, int heads, int seqlen) {
    const int row_tiles = seqlen / kMxfp4RowScaleTile;
    return (((((static_cast<int64_t>(b) * heads + h) * row_tiles + (row / kMxfp4RowScaleTile)) *
              (C::Dvo / 128) + 0) * kMxfp4ScalesPerTile)) +
           row128_scale_offset(row & (kMxfp4RowScaleTile - 1), dim);
}

__device__ inline float e8m0_to_float(uint8_t byte) {
    if (byte == 0 || byte >= 0xFE) {
        return 0.0f;
    }
    return ldexpf(1.0f, static_cast<int>(byte) - 127);
}

__device__ inline float load_row_dvo_value(
    const uint8_t *__restrict__ payload,
    const uint8_t *__restrict__ scales,
    int b,
    int h,
    int row,
    int dim,
    int heads,
    int seqlen
) {
    constexpr int packed_dvo = C::Dvo / 2;
    const int64_t payload_idx =
        (((static_cast<int64_t>(b) * heads + h) * seqlen + row) * packed_dvo) + (dim >> 1);
    const int64_t scale_idx = row_dvo_scale_idx(b, h, row, dim, heads, seqlen);
    return mxfp4_full_direct::fp4_e2m1_to_float(
               mxfp4_full_direct::fp4_nibble(payload, payload_idx, dim)) *
           (e8m0_to_float(scales[scale_idx]) * (1.0f / 6.0f));
}

template <typename ST>
__device__ inline void fill_row_dvo_tile(
    ST &dst,
    const uint8_t *__restrict__ payload,
    const uint8_t *__restrict__ scales,
    int b,
    int h,
    int tile_idx,
    int heads,
    int seqlen
) {
    constexpr int rows = tkfa4::kRefTileM;
    constexpr int cols = C::Dvo;
    for (int idx = threadIdx.x; idx < rows * cols; idx += blockDim.x) {
        const int r = idx / cols;
        const int d = idx - r * cols;
        const float value = load_row_dvo_value(
            payload,
            scales,
            b,
            h,
            tile_idx * rows + r,
            d,
            heads,
            seqlen);
        dst[{r, d}] = __float2bfloat16(value);
    }
}

template <typename ST>
__device__ inline void fill_q_tile_pairwise(
    ST &dst,
    const uint8_t *__restrict__ q,
    const uint8_t *__restrict__ q_sc,
    const float *__restrict__ q_sg,
    int b,
    int h,
    int q_tile_idx,
    int heads,
    int seqlen
) {
    constexpr int rows = tkfa4::kRefTileM;
    constexpr int cols = C::Dqk;
    constexpr int packed_cols = cols / 2;
    const int q_depth = seqlen / 128;
    const float global_scale = q_sg[b * heads + h];
    for (int idx = threadIdx.x; idx < rows * packed_cols; idx += blockDim.x) {
        const int r = idx / packed_cols;
        const int packed_d = idx - r * packed_cols;
        const int d = packed_d * 2;
        const int row = q_tile_idx * rows + r;
        const int chunk = d >> 6;
        const int group = (d & 63) >> 4;
        const int sc_off = mxfp4_full_direct::prepared_scale_offset(row & 127, group);
        const int64_t sc_idx =
            (((static_cast<int64_t>(b) * q_depth + (row / 128)) * (heads * 3) + h * 3 + chunk) * 512) + sc_off;
        const float scale = mxfp4_full_direct::fp8_e4m3_to_float(q_sc[sc_idx]) * global_scale;
        const int64_t q_idx =
            (((static_cast<int64_t>(b) * heads + h) * seqlen + row) * packed_cols) + packed_d;
        const uint8_t byte = q[q_idx];
        dst[{r, d}] = __float2bfloat16(mxfp4_full_direct::fp4_e2m1_to_float(byte & 0x0F) * scale);
        dst[{r, d + 1}] = __float2bfloat16(mxfp4_full_direct::fp4_e2m1_to_float((byte >> 4) & 0x0F) * scale);
    }
}

template <typename ST>
__device__ inline void fill_k_tile_pairwise(
    ST &dst,
    const uint8_t *__restrict__ k,
    const uint8_t *__restrict__ k_sc,
    const float *__restrict__ k_sg,
    int b,
    int h,
    int k_tile_idx,
    int heads,
    int seqlen
) {
    constexpr int rows = tkfa4::kRefTileM;
    constexpr int cols = C::Dqk;
    constexpr int packed_cols = cols / 2;
    const int k_depth = seqlen / 64;
    const float global_scale = k_sg[b * heads + h];
    for (int idx = threadIdx.x; idx < rows * packed_cols; idx += blockDim.x) {
        const int r = idx / packed_cols;
        const int packed_d = idx - r * packed_cols;
        const int d = packed_d * 2;
        const int row = k_tile_idx * rows + r;
        const int chunk = d >> 6;
        const int group = (d & 63) >> 4;
        const int sc_off = mxfp4_full_direct::prepared_scale_offset(row & 127, group);
        const int64_t sc_idx =
            (((static_cast<int64_t>(b) * k_depth + (row / 64)) * (heads * 3) + h * 3 + chunk) * 512) + sc_off;
        const float scale = mxfp4_full_direct::fp8_e4m3_to_float(k_sc[sc_idx]) * global_scale;
        const int64_t k_idx =
            (((static_cast<int64_t>(b) * heads + h) * seqlen + row) * packed_cols) + packed_d;
        const uint8_t byte = k[k_idx];
        dst[{r, d}] = __float2bfloat16(mxfp4_full_direct::fp4_e2m1_to_float(byte & 0x0F) * scale);
        dst[{r, d + 1}] = __float2bfloat16(mxfp4_full_direct::fp4_e2m1_to_float((byte >> 4) & 0x0F) * scale);
    }
}

template <typename ST>
__device__ inline void fill_row_dvo_tile_pairwise(
    ST &dst,
    const uint8_t *__restrict__ payload,
    const uint8_t *__restrict__ scales,
    int b,
    int h,
    int tile_idx,
    int heads,
    int seqlen
) {
    constexpr int rows = tkfa4::kRefTileM;
    constexpr int cols = C::Dvo;
    constexpr int packed_cols = cols / 2;
    for (int idx = threadIdx.x; idx < rows * packed_cols; idx += blockDim.x) {
        const int r = idx / packed_cols;
        const int packed_d = idx - r * packed_cols;
        const int d = packed_d * 2;
        const int row = tile_idx * rows + r;
        const int64_t payload_idx =
            (((static_cast<int64_t>(b) * heads + h) * seqlen + row) * packed_cols) + packed_d;
        const int64_t scale_idx = row_dvo_scale_idx(b, h, row, d, heads, seqlen);
        const float scale = e8m0_to_float(scales[scale_idx]) * (1.0f / 6.0f);
        const uint8_t byte = payload[payload_idx];
        dst[{r, d}] = __float2bfloat16(mxfp4_full_direct::fp4_e2m1_to_float(byte & 0x0F) * scale);
        dst[{r, d + 1}] = __float2bfloat16(mxfp4_full_direct::fp4_e2m1_to_float((byte >> 4) & 0x0F) * scale);
    }
}

template <typename SV>
__device__ inline void fill_lse_delta_vec(
    SV &lse_dst,
    SV &delta_dst,
    const float *__restrict__ lse,
    const float *__restrict__ delta,
    int b,
    int h,
    int q_tile_idx,
    int heads,
    int seqlen
) {
    for (int r = threadIdx.x; r < tkfa4::kRefTileM; r += blockDim.x) {
        const int row = q_tile_idx * tkfa4::kRefTileM + r;
        const int64_t row_id = (static_cast<int64_t>(b) * heads + h) * seqlen + row;
        lse_dst[r] = lse[row_id] * tkfa4::kLog2E;
        delta_dst[r] = delta[row_id];
    }
}

template <int CHUNK>
__device__ inline void issue_dq_chunk_add(
    const rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN> &ds_bf,
    const rt_bf<tkfa4::kRefTileM, C::Dqk> &k_reg,
    const dqacc_chunk_gl &dq_accum,
    dqacc_chunk_tile *dq_smem,
    int warp,
    int batch_idx,
    int head_idx,
    int q_tile_idx,
    bool wait_before_store
) {
    rt_bf<tkfa4::kRefTileM, 64> k_chunk;
    rt_bf<tkfa4::kRefTileM, 64, ducks::rt_layout::col> k_col;
    rt_fl<tkfa4::kRefTileM, 64> dq_chunk;

    mxfp4_full_tile_staged::extract_chunk<CHUNK>(k_reg, k_chunk);
    warp::swap_layout(k_col, k_chunk);
    warp::zero(dq_chunk);
    warp::mma_AB(dq_chunk, ds_bf, k_col, dq_chunk);

    if (wait_before_store) {
        warp::tma::store_async_read_wait();
    }
    warp::store(dq_smem[warp], dq_chunk);
    __syncwarp();
    warp::tma::store_add_async(dq_accum, dq_smem[warp], {batch_idx, head_idx, q_tile_idx, CHUNK});
    warp::tma::store_commit_group();
}

__device__ inline void accumulate_dq_chunks_from_ds(
    rt_fl<tkfa4::kRefTileM, 64> &dq0_accum,
    rt_fl<tkfa4::kRefTileM, 64> &dq1_accum,
    rt_fl<tkfa4::kRefTileM, 64> &dq2_accum,
    const rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN> &ds_bf,
    const rt_bf<tkfa4::kRefTileM, C::Dqk> &k_reg
) {
    rt_bf<tkfa4::kRefTileM, 64> k_chunk;
    rt_bf<tkfa4::kRefTileM, 64, ducks::rt_layout::col> k_col;

    mxfp4_full_tile_staged::extract_chunk<0>(k_reg, k_chunk);
    warp::swap_layout(k_col, k_chunk);
    warp::mma_AB(dq0_accum, ds_bf, k_col, dq0_accum);

    mxfp4_full_tile_staged::extract_chunk<1>(k_reg, k_chunk);
    warp::swap_layout(k_col, k_chunk);
    warp::mma_AB(dq1_accum, ds_bf, k_col, dq1_accum);

    mxfp4_full_tile_staged::extract_chunk<2>(k_reg, k_chunk);
    warp::swap_layout(k_col, k_chunk);
    warp::mma_AB(dq2_accum, ds_bf, k_col, dq2_accum);
}

template <bool CAUSAL, bool DENSE_UNMASKED>
__device__ inline void tile_step_dqdkdv(
    rt_fl<tkfa4::kRefTileM, 64> &dk0_accum,
    rt_fl<tkfa4::kRefTileM, 64> &dk1_accum,
    rt_fl<tkfa4::kRefTileM, 64> &dk2_accum,
    rt_fl<tkfa4::kRefTileM, C::Dvo> &dv_accum,
    const rt_bf<tkfa4::kRefTileM, C::Dqk> &q_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dqk> &k_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dvo> &v_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dvo> &do_reg,
    const typename rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN>::col_vec &lse_log2_vec,
    const typename rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN>::col_vec &delta_vec,
    float scale,
    float scale_log2e,
    int q_tile_idx,
    int kv_subtile_idx,
    int actual_seq_len,
    const dqacc_chunk_gl &dq_accum,
    dqacc_chunk_tile *dq_smem,
    int warp,
    int batch_idx,
    int head_idx,
    bool wait_before_first_dq_store
) {
    rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN> p, ds;

    if constexpr (DENSE_UNMASKED) {
        tkfa4::bwd_fa4::detail::reconstruct_probability_tile_dense<C>(
            p,
            q_reg,
            k_reg,
            lse_log2_vec,
            scale_log2e);
    } else {
        tkfa4::bwd_fa4::detail::reconstruct_probability_tile<C>(
            p,
            q_reg,
            k_reg,
            lse_log2_vec,
            scale_log2e,
            q_tile_idx,
            kv_subtile_idx,
            actual_seq_len,
            CAUSAL);
    }

    {
        rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN> p_bf;
        rt_bf<tkfa4::kRefTileM, C::Dvo, ducks::rt_layout::col> do_col;
        rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN, ducks::rt_layout::col> p_col;
        warp::copy(p_bf, p);
        warp::swap_layout(p_col, p_bf);
        warp::swap_layout(do_col, do_reg);
        warp::mma_AtB(dv_accum, p_col, do_col, dv_accum);

        warp::zero(ds);
        warp::mma_ABt(ds, do_reg, v_reg, ds);
        warp::sub_row(ds, ds, delta_vec);
        warp::mul(ds, p, ds);
        warp::mul(ds, ds, scale);
    }

    rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN> ds_bf;
    warp::copy(ds_bf, ds);

    {
        rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN, ducks::rt_layout::col> ds_col;
        rt_bf<tkfa4::kRefTileM, 64> q_chunk;
        rt_bf<tkfa4::kRefTileM, 64, ducks::rt_layout::col> q_col;

        warp::swap_layout(ds_col, ds_bf);

        mxfp4_full_tile_staged::extract_chunk<0>(q_reg, q_chunk);
        warp::swap_layout(q_col, q_chunk);
        warp::mma_AtB(dk0_accum, ds_col, q_col, dk0_accum);

        mxfp4_full_tile_staged::extract_chunk<1>(q_reg, q_chunk);
        warp::swap_layout(q_col, q_chunk);
        warp::mma_AtB(dk1_accum, ds_col, q_col, dk1_accum);

        mxfp4_full_tile_staged::extract_chunk<2>(q_reg, q_chunk);
        warp::swap_layout(q_col, q_chunk);
        warp::mma_AtB(dk2_accum, ds_col, q_col, dk2_accum);
    }

    issue_dq_chunk_add<0>(
        ds_bf,
        k_reg,
        dq_accum,
        dq_smem,
        warp,
        batch_idx,
        head_idx,
        q_tile_idx,
        wait_before_first_dq_store);
    issue_dq_chunk_add<1>(
        ds_bf,
        k_reg,
        dq_accum,
        dq_smem,
        warp,
        batch_idx,
        head_idx,
        q_tile_idx,
        true);
    issue_dq_chunk_add<2>(
        ds_bf,
        k_reg,
        dq_accum,
        dq_smem,
        warp,
        batch_idx,
        head_idx,
        q_tile_idx,
        true);
}

template <bool CAUSAL, bool DENSE_UNMASKED>
__device__ inline void tile_step_dq_only(
    rt_fl<tkfa4::kRefTileM, 64> &dq0_accum,
    rt_fl<tkfa4::kRefTileM, 64> &dq1_accum,
    rt_fl<tkfa4::kRefTileM, 64> &dq2_accum,
    const rt_bf<tkfa4::kRefTileM, C::Dqk> &q_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dqk> &k_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dvo> &v_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dvo> &do_reg,
    const typename rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN>::col_vec &lse_log2_vec,
    const typename rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN>::col_vec &delta_vec,
    float scale,
    float scale_log2e,
    int q_tile_idx,
    int kv_subtile_idx,
    int actual_seq_len
) {
    rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN> p, ds;

    if constexpr (DENSE_UNMASKED) {
        tkfa4::bwd_fa4::detail::reconstruct_probability_tile_dense<C>(
            p,
            q_reg,
            k_reg,
            lse_log2_vec,
            scale_log2e);
    } else {
        tkfa4::bwd_fa4::detail::reconstruct_probability_tile<C>(
            p,
            q_reg,
            k_reg,
            lse_log2_vec,
            scale_log2e,
            q_tile_idx,
            kv_subtile_idx,
            actual_seq_len,
            CAUSAL);
    }

    warp::zero(ds);
    warp::mma_ABt(ds, do_reg, v_reg, ds);
    warp::sub_row(ds, ds, delta_vec);
    warp::mul(ds, p, ds);
    warp::mul(ds, ds, scale);

    rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN> ds_bf;
    warp::copy(ds_bf, ds);
    accumulate_dq_chunks_from_ds(dq0_accum, dq1_accum, dq2_accum, ds_bf, k_reg);
}

template <bool CAUSAL, bool DENSE_UNMASKED, bool COMPUTE_DK, bool COMPUTE_DV>
__device__ inline void tile_step_dkdv_only(
    rt_fl<tkfa4::kRefTileM, 64> &dk0_accum,
    rt_fl<tkfa4::kRefTileM, 64> &dk1_accum,
    rt_fl<tkfa4::kRefTileM, 64> &dk2_accum,
    rt_fl<tkfa4::kRefTileM, C::Dvo> &dv_accum,
    const rt_bf<tkfa4::kRefTileM, C::Dqk> &q_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dqk> &k_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dvo> &v_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dvo> &do_reg,
    const typename rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN>::col_vec &lse_log2_vec,
    const typename rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN>::col_vec &delta_vec,
    float scale,
    float scale_log2e,
    int q_tile_idx,
    int kv_subtile_idx,
    int actual_seq_len
) {
    rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN> p, ds;

    if constexpr (DENSE_UNMASKED) {
        tkfa4::bwd_fa4::detail::reconstruct_probability_tile_dense<C>(
            p,
            q_reg,
            k_reg,
            lse_log2_vec,
            scale_log2e);
    } else {
        tkfa4::bwd_fa4::detail::reconstruct_probability_tile<C>(
            p,
            q_reg,
            k_reg,
            lse_log2_vec,
            scale_log2e,
            q_tile_idx,
            kv_subtile_idx,
            actual_seq_len,
            CAUSAL);
    }

    if constexpr (COMPUTE_DV) {
        rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN> p_bf;
        rt_bf<tkfa4::kRefTileM, C::Dvo, ducks::rt_layout::col> do_col;
        rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN, ducks::rt_layout::col> p_col;
        warp::copy(p_bf, p);
        warp::swap_layout(p_col, p_bf);
        warp::swap_layout(do_col, do_reg);
        warp::mma_AtB(dv_accum, p_col, do_col, dv_accum);
    }

    if constexpr (COMPUTE_DK) {
        warp::zero(ds);
        warp::mma_ABt(ds, do_reg, v_reg, ds);
        warp::sub_row(ds, ds, delta_vec);
        warp::mul(ds, p, ds);
        warp::mul(ds, ds, scale);

        rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN> ds_bf;
        warp::copy(ds_bf, ds);

        rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN, ducks::rt_layout::col> ds_col;
        rt_bf<tkfa4::kRefTileM, 64> q_chunk;
        rt_bf<tkfa4::kRefTileM, 64, ducks::rt_layout::col> q_col;

        warp::swap_layout(ds_col, ds_bf);

        mxfp4_full_tile_staged::extract_chunk<0>(q_reg, q_chunk);
        warp::swap_layout(q_col, q_chunk);
        warp::mma_AtB(dk0_accum, ds_col, q_col, dk0_accum);

        mxfp4_full_tile_staged::extract_chunk<1>(q_reg, q_chunk);
        warp::swap_layout(q_col, q_chunk);
        warp::mma_AtB(dk1_accum, ds_col, q_col, dk1_accum);

        mxfp4_full_tile_staged::extract_chunk<2>(q_reg, q_chunk);
        warp::swap_layout(q_col, q_chunk);
        warp::mma_AtB(dk2_accum, ds_col, q_col, dk2_accum);
    }
}

__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void kernel_dqdkdv_single(
    const uint8_t *__restrict__ q,
    const uint8_t *__restrict__ q_sc,
    const float *__restrict__ q_sg,
    const uint8_t *__restrict__ k,
    const uint8_t *__restrict__ k_sc,
    const float *__restrict__ k_sg,
    const uint8_t *__restrict__ v_row,
    const uint8_t *__restrict__ v_row_sc,
    const uint8_t *__restrict__ dout_row,
    const uint8_t *__restrict__ dout_row_sc,
    const float *__restrict__ lse,
    const float *__restrict__ delta,
    const __grid_constant__ dqacc_chunk_gl dq_accum,
    const __grid_constant__ dk_chunk_gl dk,
    const __grid_constant__ dv_bf_gl dv,
    int heads,
    int seqlen,
    float softmax_scale
) {
    constexpr int q_tiles_buffered = 4;
    using qk_bf_tile = st_bf<tkfa4::kRefTileM, C::Dqk>;
    using v_bf_tile = st_bf<tkfa4::kRefTileM, C::Dvo>;
    using stats_smem_tile = col_vec<st_fl<tkfa4::kRefTileM, C::Dvo>>;
    using stats_vec = typename rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN>::col_vec;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) qk_bf_tile k_smem[C::WarpTiles];
    __shared__ alignas(1024) v_bf_tile v_smem[C::WarpTiles];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(1024) dqacc_chunk_tile dq_smem[C::WarpTiles];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile delta_smem[q_tiles_buffered];

    const int warp = threadIdx.x >> 5;
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int kv_block_idx = static_cast<int>(blockIdx.x);
    const int kv_tile_base = kv_block_idx * C::WarpTiles;
    const int kv_subtile_idx = kv_tile_base + warp;
    const int num_q_blocks = seqlen / (tkfa4::kRefTileM * q_tiles_buffered);
    const int q_block_start = kv_tile_base / q_tiles_buffered;

    for (int tile = 0; tile < C::WarpTiles; ++tile) {
        fill_k_tile_pairwise(
            k_smem[tile],
            k,
            k_sc,
            k_sg,
            batch_idx,
            head_idx,
            kv_tile_base + tile,
            heads,
            seqlen);
        fill_row_dvo_tile_pairwise(
            v_smem[tile],
            v_row,
            v_row_sc,
            batch_idx,
            head_idx,
            kv_tile_base + tile,
            heads,
            seqlen);
    }
    __syncthreads();

    rt_bf<tkfa4::kRefTileM, C::Dqk> k_reg;
    rt_bf<tkfa4::kRefTileM, C::Dvo> v_reg;
    rt_fl<tkfa4::kRefTileM, 64> dk0_accum, dk1_accum, dk2_accum;
    rt_fl<tkfa4::kRefTileM, C::Dvo> dv_accum;
    warp::load(k_reg, k_smem[warp]);
    warp::load(v_reg, v_smem[warp]);
    warp::zero(dk0_accum);
    warp::zero(dk1_accum);
    warp::zero(dk2_accum);
    warp::zero(dv_accum);

    for (int q_block_idx = q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        #pragma unroll
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            fill_q_tile_pairwise(
                q_smem[subtile],
                q,
                q_sc,
                q_sg,
                batch_idx,
                head_idx,
                q_tile_idx,
                heads,
                seqlen);
            fill_row_dvo_tile_pairwise(
                do_smem[subtile],
                dout_row,
                dout_row_sc,
                batch_idx,
                head_idx,
                q_tile_idx,
                heads,
                seqlen);
            fill_lse_delta_vec(
                lse_log2_smem[subtile],
                delta_smem[subtile],
                lse,
                delta,
                batch_idx,
                head_idx,
                q_tile_idx,
                heads,
                seqlen);
        }
        __syncthreads();

        const int first_active_subtile = ((kv_subtile_idx > q_tile_base) ? (kv_subtile_idx - q_tile_base) : 0);
        #pragma unroll 1
        for (int subtile = first_active_subtile; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<tkfa4::kRefTileM, C::Dqk> q_reg;
            rt_bf<tkfa4::kRefTileM, C::Dvo> do_reg;
            stats_vec lse_log2_vec, delta_vec;

            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(delta_vec, delta_smem[subtile]);

            if (q_tile_idx > kv_subtile_idx) {
                tile_step_dqdkdv<true, true>(
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    delta_vec,
                    softmax_scale,
                    softmax_scale * tkfa4::kLog2E,
                    q_tile_idx,
                    kv_subtile_idx,
                    seqlen,
                    dq_accum,
                    dq_smem,
                    warp,
                    batch_idx,
                    head_idx,
                    q_block_idx != q_block_start || subtile != first_active_subtile);
            } else {
                tile_step_dqdkdv<true, false>(
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    delta_vec,
                    softmax_scale,
                    softmax_scale * tkfa4::kLog2E,
                    q_tile_idx,
                    kv_subtile_idx,
                    seqlen,
                    dq_accum,
                    dq_smem,
                    warp,
                    batch_idx,
                    head_idx,
                    q_block_idx != q_block_start || subtile != first_active_subtile);
            }
        }
        warp::tma::store_async_wait();
        __syncthreads();
    }

    if (threadIdx.x < WARP_THREADS * C::WarpTiles) {
        rt_bf<tkfa4::kRefTileM, 64> dk_bf;
        warp::copy(dk_bf, dk0_accum);
        warp::store(dk, dk_bf, {batch_idx, head_idx, kv_subtile_idx, 0});
        warp::copy(dk_bf, dk1_accum);
        warp::store(dk, dk_bf, {batch_idx, head_idx, kv_subtile_idx, 1});
        warp::copy(dk_bf, dk2_accum);
        warp::store(dk, dk_bf, {batch_idx, head_idx, kv_subtile_idx, 2});

        rt_bf<tkfa4::kRefTileM, C::Dvo> dv_bf;
        warp::copy(dv_bf, dv_accum);
        warp::store<dim::DEPTH>(dv, dv_bf, {batch_idx, kv_subtile_idx, head_idx, 0});
    }
}

__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void kernel_qowned_dq_keyowned_dkdv_single(
    const uint8_t *__restrict__ q,
    const uint8_t *__restrict__ q_sc,
    const float *__restrict__ q_sg,
    const uint8_t *__restrict__ k,
    const uint8_t *__restrict__ k_sc,
    const float *__restrict__ k_sg,
    const uint8_t *__restrict__ v_row,
    const uint8_t *__restrict__ v_row_sc,
    const uint8_t *__restrict__ dout_row,
    const uint8_t *__restrict__ dout_row_sc,
    const float *__restrict__ lse,
    const float *__restrict__ delta,
    const __grid_constant__ dq_bf_chunk_gl dq,
    const __grid_constant__ dk_chunk_gl dk,
    const __grid_constant__ dv_bf_gl dv,
    int heads,
    int seqlen,
    float softmax_scale
) {
    constexpr int q_tiles_buffered = C::WarpTiles;
    using qk_bf_tile = st_bf<tkfa4::kRefTileM, C::Dqk>;
    using v_bf_tile = st_bf<tkfa4::kRefTileM, C::Dvo>;
    using stats_smem_tile = col_vec<st_fl<tkfa4::kRefTileM, C::Dvo>>;
    using stats_vec = typename rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN>::col_vec;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) qk_bf_tile k_smem[C::WarpTiles];
    __shared__ alignas(1024) v_bf_tile v_smem[C::WarpTiles];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile delta_smem[q_tiles_buffered];

    const int warp = threadIdx.x >> 5;
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int num_tile_blocks = seqlen / (tkfa4::kRefTileM * C::WarpTiles);
    const int role_block_idx = static_cast<int>(blockIdx.x);
    const bool q_owned_role = role_block_idx >= num_tile_blocks;
    const int tile_block_idx = q_owned_role ? (role_block_idx - num_tile_blocks) : role_block_idx;
    if (tile_block_idx >= num_tile_blocks) {
        return;
    }

    if (!q_owned_role) {
        const int kv_tile_base = tile_block_idx * C::WarpTiles;
        const int kv_subtile_idx = kv_tile_base + warp;
        const int num_q_blocks = num_tile_blocks;
        const int q_block_start = kv_tile_base / q_tiles_buffered;

        for (int tile = 0; tile < C::WarpTiles; ++tile) {
            fill_k_tile_pairwise(
                k_smem[tile],
                k,
                k_sc,
                k_sg,
                batch_idx,
                head_idx,
                kv_tile_base + tile,
                heads,
                seqlen);
            fill_row_dvo_tile_pairwise(
                v_smem[tile],
                v_row,
                v_row_sc,
                batch_idx,
                head_idx,
                kv_tile_base + tile,
                heads,
                seqlen);
        }
        __syncthreads();

        rt_bf<tkfa4::kRefTileM, C::Dqk> k_reg;
        rt_bf<tkfa4::kRefTileM, C::Dvo> v_reg;
        rt_fl<tkfa4::kRefTileM, 64> dk0_accum, dk1_accum, dk2_accum;
        rt_fl<tkfa4::kRefTileM, C::Dvo> dv_accum;
        warp::load(k_reg, k_smem[warp]);
        warp::load(v_reg, v_smem[warp]);
        warp::zero(dk0_accum);
        warp::zero(dk1_accum);
        warp::zero(dk2_accum);
        warp::zero(dv_accum);

        for (int q_block_idx = q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
            const int q_tile_base = q_block_idx * q_tiles_buffered;
            #pragma unroll
            for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
                const int q_tile_idx = q_tile_base + subtile;
                fill_q_tile_pairwise(
                    q_smem[subtile],
                    q,
                    q_sc,
                    q_sg,
                    batch_idx,
                    head_idx,
                    q_tile_idx,
                    heads,
                    seqlen);
                fill_row_dvo_tile_pairwise(
                    do_smem[subtile],
                    dout_row,
                    dout_row_sc,
                    batch_idx,
                    head_idx,
                    q_tile_idx,
                    heads,
                    seqlen);
                fill_lse_delta_vec(
                    lse_log2_smem[subtile],
                    delta_smem[subtile],
                    lse,
                    delta,
                    batch_idx,
                    head_idx,
                    q_tile_idx,
                    heads,
                    seqlen);
            }
            __syncthreads();

            const int first_active_subtile = ((kv_subtile_idx > q_tile_base) ? (kv_subtile_idx - q_tile_base) : 0);
            #pragma unroll 1
            for (int subtile = first_active_subtile; subtile < q_tiles_buffered; ++subtile) {
                const int q_tile_idx = q_tile_base + subtile;
                rt_bf<tkfa4::kRefTileM, C::Dqk> q_reg;
                rt_bf<tkfa4::kRefTileM, C::Dvo> do_reg;
                stats_vec lse_log2_vec, delta_vec;

                warp::load(q_reg, q_smem[subtile]);
                warp::load(do_reg, do_smem[subtile]);
                warp::load(lse_log2_vec, lse_log2_smem[subtile]);
                warp::load(delta_vec, delta_smem[subtile]);

                if (q_tile_idx > kv_subtile_idx) {
                    tile_step_dkdv_only<true, true, true, true>(
                        dk0_accum,
                        dk1_accum,
                        dk2_accum,
                        dv_accum,
                        q_reg,
                        k_reg,
                        v_reg,
                        do_reg,
                        lse_log2_vec,
                        delta_vec,
                        softmax_scale,
                        softmax_scale * tkfa4::kLog2E,
                        q_tile_idx,
                        kv_subtile_idx,
                        seqlen);
                } else {
                    tile_step_dkdv_only<true, false, true, true>(
                        dk0_accum,
                        dk1_accum,
                        dk2_accum,
                        dv_accum,
                        q_reg,
                        k_reg,
                        v_reg,
                        do_reg,
                        lse_log2_vec,
                        delta_vec,
                        softmax_scale,
                        softmax_scale * tkfa4::kLog2E,
                        q_tile_idx,
                        kv_subtile_idx,
                        seqlen);
                }
            }
            __syncthreads();
        }

        rt_bf<tkfa4::kRefTileM, 64> dk_bf;
        warp::copy(dk_bf, dk0_accum);
        warp::store(dk, dk_bf, {batch_idx, head_idx, kv_subtile_idx, 0});
        warp::copy(dk_bf, dk1_accum);
        warp::store(dk, dk_bf, {batch_idx, head_idx, kv_subtile_idx, 1});
        warp::copy(dk_bf, dk2_accum);
        warp::store(dk, dk_bf, {batch_idx, head_idx, kv_subtile_idx, 2});

        rt_bf<tkfa4::kRefTileM, C::Dvo> dv_bf;
        warp::copy(dv_bf, dv_accum);
        warp::store<dim::DEPTH>(dv, dv_bf, {batch_idx, kv_subtile_idx, head_idx, 0});
        return;
    }

    {
        const int q_tile_base = tile_block_idx * C::WarpTiles;
        const int q_tile_idx = q_tile_base + warp;

        #pragma unroll
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int load_q_tile_idx = q_tile_base + subtile;
            fill_q_tile_pairwise(
                q_smem[subtile],
                q,
                q_sc,
                q_sg,
                batch_idx,
                head_idx,
                load_q_tile_idx,
                heads,
                seqlen);
            fill_row_dvo_tile_pairwise(
                do_smem[subtile],
                dout_row,
                dout_row_sc,
                batch_idx,
                head_idx,
                load_q_tile_idx,
                heads,
                seqlen);
            fill_lse_delta_vec(
                lse_log2_smem[subtile],
                delta_smem[subtile],
                lse,
                delta,
                batch_idx,
                head_idx,
                load_q_tile_idx,
                heads,
                seqlen);
        }
        __syncthreads();

        rt_bf<tkfa4::kRefTileM, C::Dqk> q_reg;
        rt_bf<tkfa4::kRefTileM, C::Dvo> do_reg;
        stats_vec lse_log2_vec, delta_vec;
        warp::load(q_reg, q_smem[warp]);
        warp::load(do_reg, do_smem[warp]);
        warp::load(lse_log2_vec, lse_log2_smem[warp]);
        warp::load(delta_vec, delta_smem[warp]);

        rt_fl<tkfa4::kRefTileM, 64> dq0_accum, dq1_accum, dq2_accum;
        warp::zero(dq0_accum);
        warp::zero(dq1_accum);
        warp::zero(dq2_accum);

        for (int kv_block_idx = 0; kv_block_idx <= tile_block_idx; ++kv_block_idx) {
            const int kv_tile_base = kv_block_idx * C::WarpTiles;
            #pragma unroll
            for (int subtile = 0; subtile < C::WarpTiles; ++subtile) {
                fill_k_tile_pairwise(
                    k_smem[subtile],
                    k,
                    k_sc,
                    k_sg,
                    batch_idx,
                    head_idx,
                    kv_tile_base + subtile,
                    heads,
                    seqlen);
                fill_row_dvo_tile_pairwise(
                    v_smem[subtile],
                    v_row,
                    v_row_sc,
                    batch_idx,
                    head_idx,
                    kv_tile_base + subtile,
                    heads,
                    seqlen);
            }
            __syncthreads();

            #pragma unroll
            for (int subtile = 0; subtile < C::WarpTiles; ++subtile) {
                const int kv_subtile_idx = kv_tile_base + subtile;
                if (kv_subtile_idx > q_tile_idx) {
                    continue;
                }

                rt_bf<tkfa4::kRefTileM, C::Dqk> k_reg;
                rt_bf<tkfa4::kRefTileM, C::Dvo> v_reg;
                warp::load(k_reg, k_smem[subtile]);
                warp::load(v_reg, v_smem[subtile]);

                if (q_tile_idx > kv_subtile_idx) {
                    tile_step_dq_only<true, true>(
                        dq0_accum,
                        dq1_accum,
                        dq2_accum,
                        q_reg,
                        k_reg,
                        v_reg,
                        do_reg,
                        lse_log2_vec,
                        delta_vec,
                        softmax_scale,
                        softmax_scale * tkfa4::kLog2E,
                        q_tile_idx,
                        kv_subtile_idx,
                        seqlen);
                } else {
                    tile_step_dq_only<true, false>(
                        dq0_accum,
                        dq1_accum,
                        dq2_accum,
                        q_reg,
                        k_reg,
                        v_reg,
                        do_reg,
                        lse_log2_vec,
                        delta_vec,
                        softmax_scale,
                        softmax_scale * tkfa4::kLog2E,
                        q_tile_idx,
                        kv_subtile_idx,
                        seqlen);
                }
            }
            __syncthreads();
        }

        rt_bf<tkfa4::kRefTileM, 64> dq_bf;
        warp::copy(dq_bf, dq0_accum);
        warp::store(dq, dq_bf, {batch_idx, head_idx, q_tile_idx, 0});
        warp::copy(dq_bf, dq1_accum);
        warp::store(dq, dq_bf, {batch_idx, head_idx, q_tile_idx, 1});
        warp::copy(dq_bf, dq2_accum);
        warp::store(dq, dq_bf, {batch_idx, head_idx, q_tile_idx, 2});
    }
}

template <bool COMPUTE_DK, bool COMPUTE_DV>
__global__ __launch_bounds__(C::BlockThreads, C::MinBlocksPerSm)
void kernel(
    const uint8_t *__restrict__ q,
    const uint8_t *__restrict__ q_sc,
    const float *__restrict__ q_sg,
    const uint8_t *__restrict__ k,
    const uint8_t *__restrict__ k_sc,
    const float *__restrict__ k_sg,
    const uint8_t *__restrict__ v_row,
    const uint8_t *__restrict__ v_row_sc,
    const uint8_t *__restrict__ dout_row,
    const uint8_t *__restrict__ dout_row_sc,
    const float *__restrict__ lse,
    const float *__restrict__ delta,
    const __grid_constant__ dk_chunk_gl dk,
    const __grid_constant__ dv_bf_gl dv,
    int heads,
    int seqlen,
    float softmax_scale
) {
    constexpr int q_tiles_buffered = 4;
    using qk_bf_tile = st_bf<tkfa4::kRefTileM, C::Dqk>;
    using v_bf_tile = st_bf<tkfa4::kRefTileM, C::Dvo>;
    using stats_smem_tile = col_vec<st_fl<tkfa4::kRefTileM, C::Dvo>>;
    using stats_vec = typename rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN>::col_vec;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) qk_bf_tile k_smem[C::WarpTiles];
    __shared__ alignas(1024) v_bf_tile v_smem[C::WarpTiles];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile delta_smem[q_tiles_buffered];

    const int warp = threadIdx.x >> 5;
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int kv_block_idx = static_cast<int>(blockIdx.x);
    const int kv_tile_base = kv_block_idx * C::WarpTiles;
    const int kv_subtile_idx = kv_tile_base + warp;
    const int num_q_blocks = seqlen / (tkfa4::kRefTileM * q_tiles_buffered);
    const int q_block_start = kv_tile_base / q_tiles_buffered;

    for (int tile = 0; tile < C::WarpTiles; ++tile) {
        fill_k_tile_pairwise(
            k_smem[tile],
            k,
            k_sc,
            k_sg,
            batch_idx,
            head_idx,
            kv_tile_base + tile,
            heads,
            seqlen);
        fill_row_dvo_tile_pairwise(
            v_smem[tile],
            v_row,
            v_row_sc,
            batch_idx,
            head_idx,
            kv_tile_base + tile,
            heads,
            seqlen);
    }
    __syncthreads();

    rt_bf<tkfa4::kRefTileM, C::Dqk> k_reg;
    rt_bf<tkfa4::kRefTileM, C::Dvo> v_reg;
    rt_fl<tkfa4::kRefTileM, 64> dk0_accum, dk1_accum, dk2_accum;
    rt_fl<tkfa4::kRefTileM, C::Dvo> dv_accum;
    warp::load(k_reg, k_smem[warp]);
    warp::load(v_reg, v_smem[warp]);
    warp::zero(dk0_accum);
    warp::zero(dk1_accum);
    warp::zero(dk2_accum);
    warp::zero(dv_accum);

    for (int q_block_idx = q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        #pragma unroll
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            fill_q_tile_pairwise(
                q_smem[subtile],
                q,
                q_sc,
                q_sg,
                batch_idx,
                head_idx,
                q_tile_idx,
                heads,
                seqlen);
            fill_row_dvo_tile_pairwise(
                do_smem[subtile],
                dout_row,
                dout_row_sc,
                batch_idx,
                head_idx,
                q_tile_idx,
                heads,
                seqlen);
            fill_lse_delta_vec(
                lse_log2_smem[subtile],
                delta_smem[subtile],
                lse,
                delta,
                batch_idx,
                head_idx,
                q_tile_idx,
                heads,
                seqlen);
        }
        __syncthreads();

        const int first_active_subtile = ((kv_subtile_idx > q_tile_base) ? (kv_subtile_idx - q_tile_base) : 0);
        #pragma unroll 1
        for (int subtile = first_active_subtile; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<tkfa4::kRefTileM, C::Dqk> q_reg;
            rt_bf<tkfa4::kRefTileM, C::Dvo> do_reg;
            stats_vec lse_log2_vec, delta_vec;

            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(delta_vec, delta_smem[subtile]);

            if (q_tile_idx > kv_subtile_idx) {
                tile_step_dkdv_only<true, true, COMPUTE_DK, COMPUTE_DV>(
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    delta_vec,
                    softmax_scale,
                    softmax_scale * tkfa4::kLog2E,
                    q_tile_idx,
                    kv_subtile_idx,
                    seqlen);
            } else {
                tile_step_dkdv_only<true, false, COMPUTE_DK, COMPUTE_DV>(
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    delta_vec,
                    softmax_scale,
                    softmax_scale * tkfa4::kLog2E,
                    q_tile_idx,
                    kv_subtile_idx,
                    seqlen);
            }
        }
        __syncthreads();
    }

    if constexpr (COMPUTE_DK) {
        rt_bf<tkfa4::kRefTileM, 64> dk_bf;
        warp::copy(dk_bf, dk0_accum);
        warp::store(dk, dk_bf, {batch_idx, head_idx, kv_subtile_idx, 0});
        warp::copy(dk_bf, dk1_accum);
        warp::store(dk, dk_bf, {batch_idx, head_idx, kv_subtile_idx, 1});
        warp::copy(dk_bf, dk2_accum);
        warp::store(dk, dk_bf, {batch_idx, head_idx, kv_subtile_idx, 2});
    }

    if constexpr (COMPUTE_DV) {
        rt_bf<tkfa4::kRefTileM, C::Dvo> dv_bf;
        warp::copy(dv_bf, dv_accum);
        warp::store<dim::DEPTH>(dv, dv_bf, {batch_idx, kv_subtile_idx, head_idx, 0});
    }
}

__global__ void debug_dequant_row_dvo_kernel(
    const uint8_t *__restrict__ v_row,
    const uint8_t *__restrict__ v_row_sc,
    const uint8_t *__restrict__ dout_row,
    const uint8_t *__restrict__ dout_row_sc,
    at::BFloat16 *__restrict__ dk_pad,
    at::BFloat16 *__restrict__ dv_out,
    int batch,
    int heads,
    int seqlen,
    int mode
) {
    const int64_t total = static_cast<int64_t>(batch) * heads * seqlen * C::Dvo;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int d = static_cast<int>(idx % C::Dvo);
        const int s = static_cast<int>((idx / C::Dvo) % seqlen);
        const int h = static_cast<int>((idx / (C::Dvo * seqlen)) % heads);
        const int b = static_cast<int>(idx / (static_cast<int64_t>(C::Dvo) * seqlen * heads));
        float v_val = 0.0f;
        float do_val = 0.0f;
        if (mode == 2) {
            const int64_t scale_idx = row_dvo_scale_idx(b, h, s, d, heads, seqlen);
            v_val = static_cast<float>(v_row_sc[scale_idx]);
            do_val = static_cast<float>(dout_row_sc[scale_idx]);
        } else if (mode == 3) {
            const int64_t payload_idx =
                (((static_cast<int64_t>(b) * heads + h) * seqlen + s) * (C::Dvo / 2)) + (d >> 1);
            v_val = static_cast<float>(mxfp4_full_direct::fp4_nibble(v_row, payload_idx, d));
            do_val = static_cast<float>(mxfp4_full_direct::fp4_nibble(dout_row, payload_idx, d));
        } else {
            v_val = load_row_dvo_value(v_row, v_row_sc, b, h, s, d, heads, seqlen);
            do_val = load_row_dvo_value(dout_row, dout_row_sc, b, h, s, d, heads, seqlen);
        }
        const int64_t dk_off =
            (((static_cast<int64_t>(b) * heads + h) * seqlen + s) * 256) + d;
        const int64_t dv_off =
            (((static_cast<int64_t>(b) * seqlen + s) * heads + h) * C::Dvo) + d;
        reinterpret_cast<__nv_bfloat16 *>(dk_pad)[dk_off] = __float2bfloat16(v_val);
        reinterpret_cast<__nv_bfloat16 *>(dv_out)[dv_off] = __float2bfloat16(do_val);
    }
}

} // namespace mxfp4_key_owned_dkdv_fused_probe

std::vector<at::Tensor> dispatch_backward_key_owned_dkdv_fused_probe_prequantized(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor DOut_row_fp4,
    at::Tensor DOut_row_sc,
    at::Tensor V_row_fp4,
    at::Tensor V_row_sc,
    at::Tensor LSE,
    at::Tensor Delta,
    at::Tensor DK_pad_out,
    at::Tensor DV_out,
    double softmax_scale
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(DOut_row_fp4); CHECK_INPUT(DOut_row_sc);
    CHECK_INPUT(V_row_fp4); CHECK_INPUT(V_row_sc);
    CHECK_INPUT(LSE); CHECK_INPUT(Delta);
    CHECK_INPUT(DK_pad_out); CHECK_INPUT(DV_out);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(DOut_row_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "DOut_row_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(V_row_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "V_row_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(DOut_row_sc.dtype() == at::ScalarType::Byte, "DOut_row_sc must be byte");
    TORCH_CHECK(V_row_sc.dtype() == at::ScalarType::Byte, "V_row_sc must be byte");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(Delta.dtype() == at::ScalarType::Float, "Delta must be float32");
    TORCH_CHECK(DK_pad_out.dtype() == at::ScalarType::BFloat16, "DK_pad_out must be BF16");
    TORCH_CHECK(DV_out.dtype() == at::ScalarType::BFloat16, "DV_out must be BF16");

    using PC = mxfp4_key_owned_dkdv_fused_probe::C;
    const int batch = static_cast<int>(Q.size(0));
    const int heads = static_cast<int>(Q.size(1));
    const int seqlen = static_cast<int>(Q.size(2));
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q/K must have shape (B,H,S,96)");
    TORCH_CHECK(Q.size(3) == PC::Dqk / 2, "Q packed dim must be 96");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K must match Q shape");
    TORCH_CHECK(Q_sc.sizes() == at::IntArrayRef({batch, seqlen / 128, heads * 3, 512}), "Q_sc has wrong shape");
    TORCH_CHECK(K_sc.sizes() == at::IntArrayRef({batch, seqlen / 64, heads * 3, 512}), "K_sc has wrong shape");
    TORCH_CHECK(Q_sg.sizes() == at::IntArrayRef({batch, heads}), "Q_sg has wrong shape");
    TORCH_CHECK(K_sg.sizes() == at::IntArrayRef({batch, heads}), "K_sg has wrong shape");
    TORCH_CHECK(DOut_row_fp4.sizes() == at::IntArrayRef({batch, heads, seqlen, PC::Dvo / 2}), "DOut_row_fp4 has wrong shape");
    TORCH_CHECK(V_row_fp4.sizes() == DOut_row_fp4.sizes(), "V_row_fp4 has wrong shape");
    TORCH_CHECK(DOut_row_sc.sizes() == at::IntArrayRef({batch, heads, seqlen / 128, PC::Dvo / 128, 32, 16}), "DOut_row_sc has wrong shape");
    TORCH_CHECK(V_row_sc.sizes() == DOut_row_sc.sizes(), "V_row_sc has wrong shape");
    TORCH_CHECK(LSE.sizes() == at::IntArrayRef({batch, heads, 1, seqlen}), "LSE has wrong shape");
    TORCH_CHECK(Delta.sizes() == at::IntArrayRef({batch, heads, seqlen}), "Delta has wrong shape");
    TORCH_CHECK(DK_pad_out.sizes() == at::IntArrayRef({batch, heads, seqlen, 256}), "DK_pad_out has wrong shape");
    TORCH_CHECK(DV_out.sizes() == at::IntArrayRef({batch, seqlen, heads, PC::Dvo}), "DV_out has wrong shape");
    TORCH_CHECK(seqlen % (tkfa4::kRefTileM * PC::WarpTiles) == 0, "fused dK+dV probe requires seqlen divisible by 512");
    TORCH_CHECK(std::isfinite(static_cast<float>(softmax_scale)), "softmax_scale must be finite");

    if (const char *debug_dequant_env = std::getenv("TK_FA4_MXFP4_BWD_KEY_OWNED_FUSED_DKDV_DEBUG_DEQUANT")) {
        if (std::atoi(debug_dequant_env) != 0) {
            const int threads = 256;
            const int64_t total = static_cast<int64_t>(batch) * heads * seqlen * PC::Dvo;
            const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 4096));
            mxfp4_key_owned_dkdv_fused_probe::debug_dequant_row_dvo_kernel<<<blocks, threads>>>(
                reinterpret_cast<const uint8_t *>(V_row_fp4.data_ptr()),
                reinterpret_cast<const uint8_t *>(V_row_sc.data_ptr()),
                reinterpret_cast<const uint8_t *>(DOut_row_fp4.data_ptr()),
                reinterpret_cast<const uint8_t *>(DOut_row_sc.data_ptr()),
                DK_pad_out.data_ptr<at::BFloat16>(),
                DV_out.data_ptr<at::BFloat16>(),
                batch,
                heads,
                seqlen,
                std::atoi(debug_dequant_env));
            CUDACHECK(cudaGetLastError());
            return {DK_pad_out, DV_out};
        }
    }

    const int num_k_blocks = seqlen / (tkfa4::kRefTileM * PC::WarpTiles);
    using DKGL = mxfp4_key_owned_dkdv_fused_probe::dk_chunk_gl;
    using DVGL = mxfp4_key_owned_dkdv_fused_probe::dv_bf_gl;
    DKGL dk_gl = ::kittens::make_gl<DKGL>(
        reinterpret_cast<uint64_t>(DK_pad_out.data_ptr<at::BFloat16>()),
        batch,
        heads,
        seqlen,
        256);
    DVGL dv_gl = ::kittens::make_gl<DVGL>(
        reinterpret_cast<uint64_t>(DV_out.data_ptr<at::BFloat16>()),
        batch,
        seqlen,
        heads,
        PC::Dvo);

    dim3 grid(num_k_blocks, heads, batch);
    dim3 block(PC::BlockThreads, 1, 1);
    int fused_mode = 0;
    if (const char *mode_env = std::getenv("TK_FA4_MXFP4_BWD_KEY_OWNED_FUSED_DKDV_MODE")) {
        fused_mode = std::atoi(mode_env);
    }
    auto launch_probe_kernel = [&]<bool COMPUTE_DK, bool COMPUTE_DV>() {
        CUDACHECK(cudaFuncSetAttribute(
            mxfp4_key_owned_dkdv_fused_probe::kernel<COMPUTE_DK, COMPUTE_DV>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared));
        mxfp4_key_owned_dkdv_fused_probe::kernel<COMPUTE_DK, COMPUTE_DV><<<grid, block>>>(
            reinterpret_cast<const uint8_t *>(Q.data_ptr()),
            reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
            Q_sg.data_ptr<float>(),
            reinterpret_cast<const uint8_t *>(K.data_ptr()),
            reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
            K_sg.data_ptr<float>(),
            reinterpret_cast<const uint8_t *>(V_row_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_row_sc.data_ptr()),
            reinterpret_cast<const uint8_t *>(DOut_row_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(DOut_row_sc.data_ptr()),
            LSE.data_ptr<float>(),
            Delta.data_ptr<float>(),
            dk_gl,
            dv_gl,
            heads,
            seqlen,
            static_cast<float>(softmax_scale));
    };
    if (fused_mode == 1) {
        launch_probe_kernel.template operator()<true, false>();
    } else if (fused_mode == 2) {
        launch_probe_kernel.template operator()<false, true>();
    } else {
        launch_probe_kernel.template operator()<true, true>();
    }
    CUDACHECK(cudaGetLastError());

    return {DK_pad_out, DV_out};
}

std::vector<at::Tensor> dispatch_backward_key_owned_dqdkdv_single_kernel_probe_prequantized(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor DOut_row_fp4,
    at::Tensor DOut_row_sc,
    at::Tensor V_row_fp4,
    at::Tensor V_row_sc,
    at::Tensor LSE,
    at::Tensor Delta,
    at::Tensor DK_pad_out,
    at::Tensor DV_out,
    double softmax_scale
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(DOut_row_fp4); CHECK_INPUT(DOut_row_sc);
    CHECK_INPUT(V_row_fp4); CHECK_INPUT(V_row_sc);
    CHECK_INPUT(LSE); CHECK_INPUT(Delta);
    CHECK_INPUT(DK_pad_out); CHECK_INPUT(DV_out);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(DOut_row_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "DOut_row_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(V_row_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "V_row_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(DOut_row_sc.dtype() == at::ScalarType::Byte, "DOut_row_sc must be byte");
    TORCH_CHECK(V_row_sc.dtype() == at::ScalarType::Byte, "V_row_sc must be byte");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(Delta.dtype() == at::ScalarType::Float, "Delta must be float32");
    TORCH_CHECK(DK_pad_out.dtype() == at::ScalarType::BFloat16, "DK_pad_out must be BF16");
    TORCH_CHECK(DV_out.dtype() == at::ScalarType::BFloat16, "DV_out must be BF16");

    using PC = mxfp4_key_owned_dkdv_fused_probe::C;
    const int batch = static_cast<int>(Q.size(0));
    const int heads = static_cast<int>(Q.size(1));
    const int seqlen = static_cast<int>(Q.size(2));
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q/K must have shape (B,H,S,96)");
    TORCH_CHECK(Q.size(3) == PC::Dqk / 2, "Q packed dim must be 96");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K must match Q shape");
    TORCH_CHECK(Q_sc.sizes() == at::IntArrayRef({batch, seqlen / 128, heads * 3, 512}), "Q_sc has wrong shape");
    TORCH_CHECK(K_sc.sizes() == at::IntArrayRef({batch, seqlen / 64, heads * 3, 512}), "K_sc has wrong shape");
    TORCH_CHECK(Q_sg.sizes() == at::IntArrayRef({batch, heads}), "Q_sg has wrong shape");
    TORCH_CHECK(K_sg.sizes() == at::IntArrayRef({batch, heads}), "K_sg has wrong shape");
    TORCH_CHECK(DOut_row_fp4.sizes() == at::IntArrayRef({batch, heads, seqlen, PC::Dvo / 2}), "DOut_row_fp4 has wrong shape");
    TORCH_CHECK(V_row_fp4.sizes() == DOut_row_fp4.sizes(), "V_row_fp4 has wrong shape");
    TORCH_CHECK(DOut_row_sc.sizes() == at::IntArrayRef({batch, heads, seqlen / 128, PC::Dvo / 128, 32, 16}), "DOut_row_sc has wrong shape");
    TORCH_CHECK(V_row_sc.sizes() == DOut_row_sc.sizes(), "V_row_sc has wrong shape");
    TORCH_CHECK(LSE.sizes() == at::IntArrayRef({batch, heads, 1, seqlen}), "LSE has wrong shape");
    TORCH_CHECK(Delta.sizes() == at::IntArrayRef({batch, heads, seqlen}), "Delta has wrong shape");
    TORCH_CHECK(DK_pad_out.sizes() == at::IntArrayRef({batch, heads, seqlen, 256}), "DK_pad_out has wrong shape");
    TORCH_CHECK(DV_out.sizes() == at::IntArrayRef({batch, seqlen, heads, PC::Dvo}), "DV_out has wrong shape");
    TORCH_CHECK(seqlen % (tkfa4::kRefTileM * PC::WarpTiles) == 0, "single-kernel probe requires seqlen divisible by 512");
    TORCH_CHECK(std::isfinite(static_cast<float>(softmax_scale)), "softmax_scale must be finite");

    const int num_k_blocks = seqlen / (tkfa4::kRefTileM * PC::WarpTiles);
    using DKGL = mxfp4_key_owned_dkdv_fused_probe::dk_chunk_gl;
    using DVGL = mxfp4_key_owned_dkdv_fused_probe::dv_bf_gl;
    DKGL dk_gl = ::kittens::make_gl<DKGL>(
        reinterpret_cast<uint64_t>(DK_pad_out.data_ptr<at::BFloat16>()),
        batch,
        heads,
        seqlen,
        256);
    DVGL dv_gl = ::kittens::make_gl<DVGL>(
        reinterpret_cast<uint64_t>(DV_out.data_ptr<at::BFloat16>()),
        batch,
        seqlen,
        heads,
        PC::Dvo);

    dim3 grid(num_k_blocks, heads, batch);
    dim3 block(PC::BlockThreads, 1, 1);

    bool use_qowned_dq_single_kernel = false;
    if (const char *qowned_env = std::getenv("TK_FA4_MXFP4_BWD_QOWNED_DQ_SINGLE_KERNEL_PROBE")) {
        use_qowned_dq_single_kernel = std::atoi(qowned_env) != 0;
    }
    if (const char *legacy_env = std::getenv("TK_FA4_MXFP4_BWD_KEYOWNED_DQ_ACCUM_PROBE")) {
        if (std::atoi(legacy_env) != 0) {
            use_qowned_dq_single_kernel = false;
        }
    }
    if (use_qowned_dq_single_kernel) {
        at::Tensor DQ_pad_out = at::empty({batch, heads, seqlen, 256}, DK_pad_out.options());
        using DQBFGL = mxfp4_key_owned_dkdv_fused_probe::dq_bf_chunk_gl;
        DQBFGL dq_gl = ::kittens::make_gl<DQBFGL>(
            reinterpret_cast<uint64_t>(DQ_pad_out.data_ptr<at::BFloat16>()),
            batch,
            heads,
            seqlen,
            256);

        CUDACHECK(cudaFuncSetAttribute(
            mxfp4_key_owned_dkdv_fused_probe::kernel_qowned_dq_keyowned_dkdv_single,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared));
        dim3 split_grid(num_k_blocks * 2, heads, batch);
        mxfp4_key_owned_dkdv_fused_probe::kernel_qowned_dq_keyowned_dkdv_single<<<split_grid, block>>>(
            reinterpret_cast<const uint8_t *>(Q.data_ptr()),
            reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
            Q_sg.data_ptr<float>(),
            reinterpret_cast<const uint8_t *>(K.data_ptr()),
            reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
            K_sg.data_ptr<float>(),
            reinterpret_cast<const uint8_t *>(V_row_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(V_row_sc.data_ptr()),
            reinterpret_cast<const uint8_t *>(DOut_row_fp4.data_ptr()),
            reinterpret_cast<const uint8_t *>(DOut_row_sc.data_ptr()),
            LSE.data_ptr<float>(),
            Delta.data_ptr<float>(),
            dq_gl,
            dk_gl,
            dv_gl,
            heads,
            seqlen,
            static_cast<float>(softmax_scale));
        CUDACHECK(cudaGetLastError());

        return {DQ_pad_out, DK_pad_out, DV_out};
    }

    at::Tensor DQ_accum = at::zeros({batch, heads, seqlen, PC::Dqk}, DK_pad_out.options().dtype(at::ScalarType::Float));
    using DQGL = mxfp4_key_owned_dkdv_fused_probe::dqacc_chunk_gl;
    DQGL dq_gl = ::kittens::make_gl<DQGL>(
        reinterpret_cast<uint64_t>(DQ_accum.data_ptr<float>()),
        batch,
        heads,
        seqlen,
        PC::Dqk);

    CUDACHECK(cudaFuncSetAttribute(
        mxfp4_key_owned_dkdv_fused_probe::kernel_dqdkdv_single,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
    mxfp4_key_owned_dkdv_fused_probe::kernel_dqdkdv_single<<<grid, block>>>(
        reinterpret_cast<const uint8_t *>(Q.data_ptr()),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        Q_sg.data_ptr<float>(),
        reinterpret_cast<const uint8_t *>(K.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        K_sg.data_ptr<float>(),
        reinterpret_cast<const uint8_t *>(V_row_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_row_sc.data_ptr()),
        reinterpret_cast<const uint8_t *>(DOut_row_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(DOut_row_sc.data_ptr()),
        LSE.data_ptr<float>(),
        Delta.data_ptr<float>(),
        dq_gl,
        dk_gl,
        dv_gl,
        heads,
        seqlen,
        static_cast<float>(softmax_scale));
    CUDACHECK(cudaGetLastError());

    return {DQ_accum, DK_pad_out, DV_out};
}
