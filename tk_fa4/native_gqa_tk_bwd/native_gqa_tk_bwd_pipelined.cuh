#pragma once

#include "../deprecated/fa4_common.cuh"

// Project-owned D64 extraction of ThunderKittens' original pipelined MHA
// backward.  The async MMA completion points are retained verbatim; only the
// softmax scale and namespace/global ABI are adapted.  The submodule remains
// read-only.
namespace tkfa4::native_gqa_tk_bwd::pipelined {

constexpr int kDepth = 64;
constexpr int kTileHeight = 64;
constexpr int kConsumerWarpgroups = 2;
constexpr int kProducerWarpgroups = 1;
constexpr int kNumWarpgroups = kConsumerWarpgroups + kProducerWarpgroups;
constexpr int kNumWorkers = kNumWarpgroups * kittens::WARPGROUP_WARPS;
constexpr int kThreads = kNumWorkers * kittens::WARP_THREADS;
constexpr int kDynamicSmemBytes = 117760;

struct globals {
    static constexpr bool kE4m3Operands = false;
    static constexpr int kDeltaScale = 1;
    static constexpr int kStatsScale = 1;

    using q_tile = st_bf<kTileHeight, kDepth>;
    using k_tile = st_bf<kTileHeight, kDepth>;
    using v_tile = st_bf<kTileHeight, kDepth>;
    using dout_tile = st_bf<kTileHeight, kDepth>;
    using dq_tile = st_fl<kTileHeight, kDepth>;
    using dk_tile = st_fl<kTileHeight, kDepth>;
    using dv_tile = st_fl<kTileHeight, kDepth>;
    using l_tile = row_vec<st_fl<kTileHeight, kTileHeight>>;
    using delta_tile = row_vec<st_fl<kTileHeight, kTileHeight>>;
    using probability_tile = st_fp8e4m3<kTileHeight, kTileHeight>;
    using ds_tile = st_bf<kTileHeight, kTileHeight>;
    using lowp_register = rt_bf<16, kTileHeight>;

    using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
    using dout_gl = gl<bf16, -1, -1, -1, -1, dout_tile>;
    using dq_gl = gl<float, -1, -1, -1, -1, dq_tile>;
    using dk_gl = gl<float, -1, -1, -1, -1, dk_tile>;
    using dv_gl = gl<float, -1, -1, -1, -1, dv_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;
    using delta_gl = gl<float, -1, -1, -1, -1, delta_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    dout_gl dout;
    dq_gl dq;
    dk_gl dk;
    dv_gl dv;
    l_gl l_aux;
    delta_gl delta;
    float scale;
    float scale_log2e;
    int sequence;
    int head_ratio;
};

// First low-precision proof for the Llama D64/GQA shape.  Q/K/V/dO are
// represented as E4M3 values pre-scaled by 4.  Score and centered dP therefore
// carry a factor of 16; launch_e4m3 folds its reciprocal into the softmax scale
// and this kernel scales delta by 16 before subtraction.  P and dS are
// published to shared memory with a common 2^8 scale, allowing every gradient
// MMA to run as E4M3 while a single 2^-8 postscale returns raw-x4 gradients.
struct e4m3_globals {
    static constexpr bool kE4m3Operands = true;
    static constexpr int kDeltaScale = 16;
    static constexpr int kStatsScale = 16;

    using q_tile = st_fp8e4m3<kTileHeight, kDepth>;
    using k_tile = st_fp8e4m3<kTileHeight, kDepth>;
    using v_tile = st_fp8e4m3<kTileHeight, kDepth>;
    using dout_tile = st_fp8e4m3<kTileHeight, kDepth>;
    using dq_tile = st_fl<kTileHeight, kDepth>;
    using dk_tile = st_fl<kTileHeight, kDepth>;
    using dv_tile = st_fl<kTileHeight, kDepth>;
    using l_tile = row_vec<st_fl<kTileHeight, kTileHeight>>;
    using delta_tile = row_vec<st_fl<kTileHeight, kTileHeight>>;
    using probability_tile = st_fp8e4m3<kTileHeight, kTileHeight>;
    using ds_tile = st_fp8e4m3<kTileHeight, kTileHeight>;
    using lowp_register = rt_fp8e4m3<16, kTileHeight>;

    using q_gl = gl<fp8e4m3, -1, -1, -1, -1, q_tile>;
    using k_gl = gl<fp8e4m3, -1, -1, -1, -1, k_tile>;
    using v_gl = gl<fp8e4m3, -1, -1, -1, -1, v_tile>;
    using dout_gl = gl<fp8e4m3, -1, -1, -1, -1, dout_tile>;
    using dq_gl = gl<float, -1, -1, -1, -1, dq_tile>;
    using dk_gl = gl<float, -1, -1, -1, -1, dk_tile>;
    using dv_gl = gl<float, -1, -1, -1, -1, dv_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_tile>;
    using delta_gl = gl<float, -1, -1, -1, -1, delta_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    dout_gl dout;
    dq_gl dq;
    dk_gl dk;
    dv_gl dv;
    l_gl l_aux;
    delta_gl delta;
    float scale;
    float scale_log2e;
    int sequence;
    int head_ratio;
};

template <int SharedScale = 1>
__device__ inline void stream_tile(
    auto &register_tile,
    auto &shared_vector,
    int stage
) {
    static_assert(SharedScale == 1 || SharedScale == 16);
    const float2 shared_scale{
        static_cast<float>(SharedScale),
        static_cast<float>(SharedScale),
    };
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        const int base_column = 16 * index + 2 * (kittens::laneid() % 4);
        float2 stats_lo =
            *reinterpret_cast<float2 *>(&shared_vector[stage][base_column]);
        float2 stats_hi = *reinterpret_cast<float2 *>(
            &shared_vector[stage][base_column + 8]
        );
        if constexpr (SharedScale != 1) {
            stats_lo = base_ops::mul::template op<float2>(
                stats_lo,
                shared_scale
            );
            stats_hi = base_ops::mul::template op<float2>(
                stats_hi,
                shared_scale
            );
        }
        register_tile.tiles[0][index].data[0] = stats_lo;
        register_tile.tiles[0][index].data[1] = stats_lo;
        register_tile.tiles[0][index].data[2] = stats_hi;
        register_tile.tiles[0][index].data[3] = stats_hi;
    }
}

template <int SharedScale = 1>
__device__ inline void stream_sub_tile(
    auto &register_tile,
    auto &shared_vector,
    int stage
) {
    static_assert(SharedScale == 1 || SharedScale == 16);
    const float2 shared_scale{
        static_cast<float>(SharedScale),
        static_cast<float>(SharedScale),
    };
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        const int base_column = 16 * index + 2 * (kittens::laneid() % 4);
        float2 delta_lo = *reinterpret_cast<float2 *>(
            &shared_vector[stage][base_column]
        );
        float2 delta_hi = *reinterpret_cast<float2 *>(
            &shared_vector[stage][base_column + 8]
        );
        if constexpr (SharedScale != 1) {
            delta_lo = base_ops::mul::template op<float2>(
                delta_lo,
                shared_scale
            );
            delta_hi = base_ops::mul::template op<float2>(
                delta_hi,
                shared_scale
            );
        }
        register_tile.tiles[0][index].data[0] =
            base_ops::sub::template op<float2>(
                register_tile.tiles[0][index].data[0],
                delta_lo
            );
        register_tile.tiles[0][index].data[1] =
            base_ops::sub::template op<float2>(
                register_tile.tiles[0][index].data[1],
                delta_lo
            );
        register_tile.tiles[0][index].data[2] =
            base_ops::sub::template op<float2>(
                register_tile.tiles[0][index].data[2],
                delta_hi
            );
        register_tile.tiles[0][index].data[3] =
            base_ops::sub::template op<float2>(
                register_tile.tiles[0][index].data[3],
                delta_hi
            );
    }
}

__device__ inline void causal_mask(auto &register_tile, int query_block) {
    const int query_subtile_base = query_block * 4;
    const int key_subtile =
        static_cast<int>(blockIdx.x) * kConsumerWarpgroups * 4 +
        (kittens::warpid() / kittens::WARPGROUP_WARPS) * 4 +
        kittens::warpid() % kittens::WARPGROUP_WARPS;
#pragma unroll
    for (int column_tile = 0; column_tile < 4; ++column_tile) {
        const int query_subtile = query_subtile_base + column_tile;
        auto &attention_subtile = reinterpret_cast<rt_fl<16, 16> &>(
            register_tile.tiles[0][column_tile]
        );
        if (query_subtile < key_subtile) {
            warp::neg_infty(attention_subtile);
        } else if (query_subtile == key_subtile) {
            warp::make_causal_t(
                attention_subtile,
                attention_subtile,
                kittens::base_types::constants<float>::neg_infty()
            );
        }
    }
}

// Project-local workaround for ThunderKittens commit 278ff3de.  SM100 half
// TMEM tiles use 32-DP-lane warp partitions even though each warp materializes
// 16 logical rows.  The generic group<4> helper currently advances by 16 and
// corrupts alternating warp quadrants.  Keep the submodule read-only and use
// the pre-regression +{0,32,64,96} physical row mapping here.
template <ducks::rt::row_layout RegisterTile, ducks::tt::half TensorTile>
__device__ inline void load_half_tmem_async(
    RegisterTile &destination,
    const TensorTile &source
) {
    static_assert(RegisterTile::rows == 16);
    static_assert(RegisterTile::cols == TensorTile::cols);
    using warp_tensor_tile =
        tt<typename TensorTile::dtype, RegisterTile::rows, TensorTile::cols>;
    const warp_tensor_tile source_subtile{
        source.addr + ((32 * warpgroup::warpid()) << 16)
    };
    group<1>::load_async(destination, source_subtile);
}

template <ducks::rt::all RegisterTile, ducks::tt::half TensorTile>
__device__ inline void store_half_tmem_async(
    TensorTile &destination,
    const RegisterTile &source
) {
    static_assert(RegisterTile::rows == 16);
    static_assert(RegisterTile::cols == TensorTile::cols);
    using warp_tensor_tile =
        tt<typename TensorTile::dtype, RegisterTile::rows, TensorTile::cols>;
    warp_tensor_tile destination_subtile{
        destination.addr + ((32 * warpgroup::warpid()) << 16)
    };
    group<1>::store_async(destination_subtile, source);
}

template <uint32_t ExponentDelta = 0>
__device__ __forceinline__ uint16_t convert_f32_pair_to_e4m3(
    const float2 &values
) {
    const uint32_t source0 = std::bit_cast<uint32_t>(values.x);
    const uint32_t source1 = std::bit_cast<uint32_t>(values.y);
    uint32_t packed;
    asm volatile(
        "{\n"
        ".reg .b32 scaled0;\n"
        ".reg .b32 scaled1;\n"
        ".reg .b16 result;\n"
        "add.u32 scaled0, %1, %3;\n"
        "add.u32 scaled1, %2, %3;\n"
        "cvt.rn.satfinite.e4m3x2.f32 result, scaled1, scaled0;\n"
        "cvt.u32.u16 %0, result;\n"
        "}\n"
        : "=r"(packed)
        : "r"(source0), "r"(source1), "n"(ExponentDelta)
    );
    return static_cast<uint16_t>(packed);
}

// TK's generic FP32->FP8 tile conversion is convenient but has not been used
// by the retained FA4 backward path.  Preserve that path's verified ownership
// exchange exactly so this proof does not conflate an operand-layout bug with
// low-precision numerical error.
template <int Rows, int Cols>
__device__ __forceinline__ void convert_f32_to_e4m3(
    rt_fp8e4m3<Rows, Cols> &destination,
    const rt_fl<Rows, Cols> &source
) {
    const int lane = kittens::laneid();
    const int high_columns = (lane >> 1) & 1;
    const int source_pair = (lane & ~3) + 2 * (lane & 1);
    const int source_lane0 = source_pair + high_columns;
    const int source_lane1 = source_pair + 1 - high_columns;
    const uint32_t prmt_selector = high_columns ? 0x1054u : 0x5410u;
#pragma unroll
    for (int tile_row = 0; tile_row < destination.height; ++tile_row) {
#pragma unroll
        for (int tile_col = 0; tile_col < destination.width; ++tile_col) {
#pragma unroll
            for (
                int packed = 0;
                packed < destination.packed_per_tile;
                ++packed
            ) {
                const auto &source_tile = source.tiles[tile_row][
                    2 * tile_col + packed / 2
                ];
                const int source_index = packed & 1;
                const uint32_t converted_lo = convert_f32_pair_to_e4m3(
                    source_tile.data[source_index + 0]
                );
                const uint32_t converted_hi = convert_f32_pair_to_e4m3(
                    source_tile.data[source_index + 2]
                );
                const uint32_t adoption0 = (lane & 1) == 0
                    ? converted_lo
                    : converted_hi;
                const uint32_t adoption1 = (lane & 1) == 0
                    ? converted_hi
                    : converted_lo;
                const uint32_t shuffled0 = __shfl_sync(
                    MASK_ALL,
                    adoption0,
                    source_lane0
                );
                const uint32_t shuffled1 = __shfl_sync(
                    MASK_ALL,
                    adoption1,
                    source_lane1
                );
                uint32_t converted;
                asm volatile(
                    "prmt.b32 %0, %1, %2, %3;\n"
                    : "=r"(converted)
                    : "r"(shuffled0), "r"(shuffled1), "r"(prmt_selector)
                );
                destination.tiles[tile_row][tile_col].data[packed] =
                    std::bit_cast<fp8e4m3_4>(converted);
            }
        }
    }
}

// TK's generic SM100 shared descriptor advances an MN-major operand by one
// 16-row tile for every chunk index.  Dense FP8 MMA consumes K32, so its second
// command must advance two tiles.  The retained D192 TK backward carries the
// same project-local correction; without it rows 16..31 overlap and rows
// 48..63 are never consumed.
template <
    int Accumulate,
    ducks::tt::all Destination,
    ducks::st_descriptor::input A,
    ducks::st_descriptor::input B
>
__device__ __forceinline__ void fp8_mm_ab_corrected(
    Destination &destination,
    const A &a,
    const B &b
) {
    static_assert(Accumulate == 0 || Accumulate == 1);
    using a_tile = ducks::st_descriptor::detail::get_st<A>;
    using b_tile = ducks::st_descriptor::detail::get_st<B>;
    using input_type = typename a_tile::T;
    using output_type = typename Destination::T;
    static_assert(std::is_same_v<input_type, fp8e4m3>);
    static_assert(std::is_same_v<input_type, typename b_tile::T>);
    constexpr int kM = a_tile::rows;
    constexpr int kN = b_tile::cols;
    constexpr int kK = a_tile::cols;
    static_assert(kM == Destination::rows && kN == Destination::cols);
    static_assert(kK == b_tile::rows && kK % 32 == 0);
    constexpr uint32_t instruction =
        ::kittens::detail::tcgen05::instruction_descriptor<
            output_type,
            input_type,
            kM,
            kN,
            transpose::N,
            transpose::T,
            false
        >();
    if (warpgroup::laneid() == 0) {
        ::kittens::st_descriptor<a_tile, transpose::N> a_desc(a);
        ::kittens::st_descriptor<b_tile, transpose::T> b_desc(b);
        ::kittens::detail::tcgen05::template st_st<
            input_type,
            Accumulate,
            1
        >(
            destination.addr,
            a_desc.chunk_descriptor(0),
            b_desc.chunk_descriptor(0),
            instruction
        );
#pragma unroll
        for (int chunk = 1; chunk < kK / 32; ++chunk) {
            ::kittens::detail::tcgen05::template st_st<input_type, 1, 1>(
                destination.addr,
                a_desc.chunk_descriptor(chunk),
                b_desc.chunk_descriptor(2 * chunk),
                instruction
            );
        }
    }
}

template <
    int Accumulate,
    ducks::tt::all Destination,
    ducks::st_descriptor::input A,
    ducks::st_descriptor::input B
>
__device__ __forceinline__ void fp8_mm_atb_corrected(
    Destination &destination,
    const A &a,
    const B &b
) {
    static_assert(Accumulate == 0 || Accumulate == 1);
    using a_tile = ducks::st_descriptor::detail::get_st<A>;
    using b_tile = ducks::st_descriptor::detail::get_st<B>;
    using input_type = typename a_tile::T;
    using output_type = typename Destination::T;
    static_assert(std::is_same_v<input_type, fp8e4m3>);
    static_assert(std::is_same_v<input_type, typename b_tile::T>);
    constexpr int kM = a_tile::cols;
    constexpr int kN = b_tile::cols;
    constexpr int kK = a_tile::rows;
    static_assert(kM == Destination::rows && kN == Destination::cols);
    static_assert(kK == b_tile::rows && kK % 32 == 0);
    constexpr uint32_t instruction =
        ::kittens::detail::tcgen05::instruction_descriptor<
            output_type,
            input_type,
            kM,
            kN,
            transpose::T,
            transpose::T,
            false
        >();
    if (warpgroup::laneid() == 0) {
        ::kittens::st_descriptor<a_tile, transpose::T> a_desc(a);
        ::kittens::st_descriptor<b_tile, transpose::T> b_desc(b);
        ::kittens::detail::tcgen05::template st_st<
            input_type,
            Accumulate,
            1
        >(
            destination.addr,
            a_desc.chunk_descriptor(0),
            b_desc.chunk_descriptor(0),
            instruction
        );
#pragma unroll
        for (int chunk = 1; chunk < kK / 32; ++chunk) {
            ::kittens::detail::tcgen05::template st_st<input_type, 1, 1>(
                destination.addr,
                a_desc.chunk_descriptor(2 * chunk),
                b_desc.chunk_descriptor(2 * chunk),
                instruction
            );
        }
    }
}

template <
    typename KernelGlobals,
    typename AttentionTmem,
    typename GradientTmem,
    typename LowpRegister
>
__device__ inline void compute_loop(
    kittens::semaphore *stats_ready,
    kittens::semaphore *q_ready,
    kittens::semaphore *dout_ready,
    kittens::semaphore &score_ready,
    kittens::semaphore &dp_ready,
    kittens::semaphore &kv_step_done,
    rt_fl<16, 64> &score,
    rt_fl<16, 64> &dp,
    rt_fl<16, 64> &probability,
    rt_fl<16, 64> &ds,
    LowpRegister &probability_lowp,
    LowpRegister &ds_lowp,
    AttentionTmem &score_tmem,
    AttentionTmem &dp_tmem,
    GradientTmem &dk_tmem,
    GradientTmem &dv_tmem,
    auto &q_smem,
    auto &k_smem,
    auto &v_smem,
    auto &dout_smem,
    auto &probability_smem,
    auto &ds_smem,
    auto &l_smem,
    auto &delta_smem,
    int query_block,
    int query_start,
    int load_stage,
    const KernelGlobals &g
) {
    using probability_tmem = half_tt_bf<kTileHeight>;

    const int phase = ((query_block - query_start) / 2) % 2;
    const int consumer = kittens::warpid() / kittens::WARPGROUP_WARPS;
    wait(stats_ready[load_stage], phase);
    stream_tile<KernelGlobals::kStatsScale>(score, l_smem, load_stage);
    wait(q_ready[load_stage], phase);

    // SM100 TCGEN writes score and dP to TMEM.  The semaphore overload is the
    // Blackwell equivalent of Hopper's commit/wait pair and must complete
    // before the register post-processing consumes either tile.
    warpgroup::mm_ABt(
        score_tmem,
        k_smem[consumer],
        q_smem[load_stage],
        score_ready
    );
    wait(score_ready, phase);
    tensor_after_thread_sync();
    load_half_tmem_async(probability, score_tmem);
    tensor_load_wait();
    warp::add(score, score, probability);

    wait(dout_ready[load_stage], phase);
    warpgroup::mm_ABt(
        dp_tmem,
        v_smem[consumer],
        dout_smem[load_stage],
        dp_ready
    );
    wait(dp_ready, phase);
    tensor_after_thread_sync();
    load_half_tmem_async(dp, dp_tmem);
    tensor_load_wait();

    warp::mul(score, score, g.scale_log2e);
    causal_mask(score, query_block);
    warp::exp2(score, score);
    warp::copy(probability, score);
    if constexpr (KernelGlobals::kE4m3Operands) {
        warp::mul(score, probability, 256.0f);
        convert_f32_to_e4m3(probability_lowp, score);
    } else {
        warp::copy(probability_lowp, score);
    }
    stream_sub_tile<KernelGlobals::kDeltaScale>(
        dp,
        delta_smem,
        load_stage
    );
    warp::mul(ds, probability, dp);
    warp::mul(ds, ds, g.scale);
    if constexpr (KernelGlobals::kE4m3Operands) {
        warp::mul(ds, ds, 256.0f);
        convert_f32_to_e4m3(ds_lowp, ds);

        // Publish the packed register tiles directly.  The ownership exchange
        // in convert_f32_to_e4m3 matches the retained low-precision TK donor.
        warpgroup::store(probability_smem[consumer], probability_lowp);
        warpgroup::store(ds_smem[consumer], ds_lowp);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        warpgroup::sync(12 + consumer);

        if (query_block == query_start) {
            fp8_mm_ab_corrected<0>(
                dv_tmem,
                probability_smem[consumer],
                dout_smem[load_stage]
            );
            fp8_mm_ab_corrected<0>(
                dk_tmem,
                ds_smem[consumer],
                q_smem[load_stage]
            );
        } else {
            fp8_mm_ab_corrected<1>(
                dv_tmem,
                probability_smem[consumer],
                dout_smem[load_stage]
            );
            fp8_mm_ab_corrected<1>(
                dk_tmem,
                ds_smem[consumer],
                q_smem[load_stage]
            );
        }
    } else {
        warp::copy(ds_lowp, ds);

        // Reuse the completed score/dP columns for BF16 probability/dS
        // operands.  The project-local 32-DP-row store wrapper preserves the
        // SM100 half-TMEM warp mapping that the generic TK group<4> helper
        // currently regresses.
        probability_tmem probability_operand{score_tmem.addr};
        probability_tmem ds_operand{dp_tmem.addr};
        store_half_tmem_async(probability_operand, probability_lowp);
        store_half_tmem_async(ds_operand, ds_lowp);
        warpgroup::store(ds_smem[consumer], ds_lowp);
        tensor_store_wait();
        tensor_before_thread_sync();
        warpgroup::sync(12 + consumer);
        tensor_after_thread_sync();

        if (query_block == query_start) {
            warpgroup::mm_AB(
                dv_tmem,
                probability_operand,
                dout_smem[load_stage]
            );
            warpgroup::mm_AB(
                dk_tmem,
                ds_operand,
                q_smem[load_stage]
            );
        } else {
            warpgroup::mma_AB(
                dv_tmem,
                probability_operand,
                dout_smem[load_stage]
            );
            warpgroup::mma_AB(
                dk_tmem,
                ds_operand,
                q_smem[load_stage]
            );
        }
    }
    if (warpgroup::laneid() == 0) {
        tensor_commit<1>(kv_step_done);
    }
    wait(kv_step_done, phase);
    tensor_after_thread_sync();
    group<8>::sync(10);
}

template <typename KernelGlobals, typename DkTile, typename DvTile>
__device__ inline void store_kv(
    auto &dk_smem,
    auto &dk,
    auto &dv_smem,
    auto &dv,
    const KernelGlobals &g,
    kittens::semaphore &dq_ready,
    int kv_head,
    int store_phase
) {
    group<8>::sync(10);
    const int consumer = kittens::warpid() / kittens::WARPGROUP_WARPS;
    if constexpr (KernelGlobals::kE4m3Operands) {
        warp::mul(dk, dk, 1.0f / 256.0f);
        warp::mul(dv, dv, 1.0f / 256.0f);
    }
    warpgroup::store(dk_smem[consumer], dk);
    group<4>::sync(warpgroup::groupid() + 4);
    if (kittens::warpid() % 4 == 0) {
        coord<DkTile> coordinate = {
            static_cast<int>(blockIdx.z),
            kv_head,
            static_cast<int>(blockIdx.x) * kConsumerWarpgroups + consumer,
            0,
        };
        warp::tma::store_add_async(g.dk, dk_smem[consumer], coordinate);
    }

    wait(dq_ready, store_phase);
    warpgroup::store(dv_smem[consumer], dv);
    group<4>::sync(warpgroup::groupid() + 4);
    if (kittens::warpid() % 4 == 0) {
        coord<DvTile> coordinate = {
            static_cast<int>(blockIdx.z),
            kv_head,
            static_cast<int>(blockIdx.x) * kConsumerWarpgroups + consumer,
            0,
        };
        warp::tma::store_add_async(g.dv, dv_smem[consumer], coordinate);
    }
    warp::tma::store_async_wait();
}

template <typename KernelGlobals>
__global__ __launch_bounds__(kThreads, 1)
void main_kernel(const __grid_constant__ KernelGlobals g) {
    extern __shared__ int dynamic_shared[];
    tma_swizzle_allocator allocator(dynamic_shared);

    using k_tile = typename KernelGlobals::k_tile;
    using v_tile = typename KernelGlobals::v_tile;
    using q_tile = typename KernelGlobals::q_tile;
    using dout_tile = typename KernelGlobals::dout_tile;
    using dq_tile = typename KernelGlobals::dq_tile;
    using dk_tile = typename KernelGlobals::dk_tile;
    using dv_tile = typename KernelGlobals::dv_tile;
    using l_tile = typename KernelGlobals::l_tile;
    using delta_tile = typename KernelGlobals::delta_tile;
    using probability_tile = typename KernelGlobals::probability_tile;
    using ds_tile = typename KernelGlobals::ds_tile;
    using lowp_register = typename KernelGlobals::lowp_register;
    using attention_tmem = half_tt_fl<kTileHeight>;
    using gradient_tmem = half_tt_fl<kDepth>;

    k_tile (&k_smem)[kConsumerWarpgroups] =
        allocator.allocate<k_tile, kConsumerWarpgroups>();
    v_tile (&v_smem)[kConsumerWarpgroups] =
        allocator.allocate<v_tile, kConsumerWarpgroups>();
    q_tile (&q_smem)[2] = allocator.allocate<q_tile, 2>();
    dout_tile (&dout_smem)[2] = allocator.allocate<dout_tile, 2>();
    dq_tile (&dq_smem) = allocator.allocate<dq_tile>();
    l_tile (&l_smem)[2] = allocator.allocate<l_tile, 2>();
    delta_tile (&delta_smem)[2] = allocator.allocate<delta_tile, 2>();
    dk_tile *dk_smem = reinterpret_cast<dk_tile *>(&k_smem[0].data[0]);
    // Alias four disjoint FP32 epilogue tiles onto the dead input/staging
    // arena.  Using q_smem as the dV base only happened to work for BF16:
    // E4M3 halves each input tile and made dk_smem[1] == dv_smem[0].
    // dq_ready protects the latter 32 KiB until dQ has left shared memory.
    dv_tile *dv_smem = reinterpret_cast<dv_tile *>(
        dk_smem + kConsumerWarpgroups
    );
    ds_tile (&ds_smem)[kConsumerWarpgroups] =
        allocator.allocate<ds_tile, kConsumerWarpgroups>();
    probability_tile (&probability_smem)[kConsumerWarpgroups] =
        allocator.allocate<probability_tile, kConsumerWarpgroups>();

    tensor_allocator<1, 1> tmem_allocator{};
    attention_tmem score_tmem[kConsumerWarpgroups][2];
    attention_tmem dp_tmem[kConsumerWarpgroups][2];
    gradient_tmem dk_tmem[kConsumerWarpgroups];
    gradient_tmem dv_tmem[kConsumerWarpgroups];
    gradient_tmem dq_tmem[2];

#pragma unroll
    for (int consumer = 0; consumer < kConsumerWarpgroups; ++consumer) {
        score_tmem[consumer][0] =
            tmem_allocator.template allocate<attention_tmem>(consumer, 0);
        dp_tmem[consumer][0] = tmem_allocator.template allocate<attention_tmem>(
            consumer,
            kTileHeight
        );
        score_tmem[consumer][1] =
            tmem_allocator.template allocate<attention_tmem>(
                consumer,
                2 * kTileHeight
            );
        dp_tmem[consumer][1] = tmem_allocator.template allocate<attention_tmem>(
            consumer,
            3 * kTileHeight
        );
        dk_tmem[consumer] = tmem_allocator.template allocate<gradient_tmem>(
            consumer,
            4 * kTileHeight
        );
        dv_tmem[consumer] = tmem_allocator.template allocate<gradient_tmem>(
            consumer,
            4 * kTileHeight + kDepth
        );
    }
    // Keep dQ out of the score/probability columns while bringing up the
    // SM100 port.  D64 leaves exactly 128 columns after dK/dV, enough for two
    // stage-private dQ tiles and removes the Hopper-era TMEM alias hazard.
    dq_tmem[0] = tmem_allocator.template allocate<gradient_tmem>(
        0,
        4 * kTileHeight + 2 * kDepth
    );
    dq_tmem[1] = tmem_allocator.template allocate<gradient_tmem>(
        0,
        4 * kTileHeight + 3 * kDepth
    );

    const int warp = kittens::warpid();
    const int warpgroup = warp / kittens::WARPGROUP_WARPS;
    const int query_blocks = g.sequence / kTileHeight;
    const int kv_head = static_cast<int>(blockIdx.y) / g.head_ratio;
    const int query_start = static_cast<int>(blockIdx.x) * 2;
    int load_stage = 0;
    int next_stage = 1;

    __shared__ kittens::semaphore kv_ready;
    __shared__ kittens::semaphore q_ready[2];
    __shared__ kittens::semaphore dout_ready[2];
    __shared__ kittens::semaphore stats_ready[2];
    __shared__ kittens::semaphore compute_done[2];
    __shared__ kittens::semaphore dq_ready;
    __shared__ kittens::semaphore score_done[kConsumerWarpgroups][2];
    __shared__ kittens::semaphore dp_done[kConsumerWarpgroups][2];
    __shared__ kittens::semaphore kv_step_done[kConsumerWarpgroups][2];
    __shared__ kittens::semaphore dq_tmem_done[2];
    __shared__ kittens::semaphore kv_tmem_done[kConsumerWarpgroups];

    if (threadIdx.x == 0) {
        init_semaphore(kv_ready, 0, 1);
        init_semaphore(dq_ready, 1, 0);
#pragma unroll
        for (int stage = 0; stage < 2; ++stage) {
            init_semaphore(q_ready[stage], 0, 1);
            init_semaphore(dout_ready[stage], 0, 1);
            init_semaphore(stats_ready[stage], 0, 1);
            init_semaphore(compute_done[stage], 1, 0);
            init_semaphore(dq_tmem_done[stage], 0, 1);
#pragma unroll
            for (int consumer = 0; consumer < kConsumerWarpgroups; ++consumer) {
                init_semaphore(score_done[consumer][stage], 0, 1);
                init_semaphore(dp_done[consumer][stage], 0, 1);
                init_semaphore(kv_step_done[consumer][stage], 0, 1);
            }
        }
#pragma unroll
        for (int consumer = 0; consumer < kConsumerWarpgroups; ++consumer) {
            init_semaphore(kv_tmem_done[consumer], 0, 1);
        }

        tma::expect_bytes(
            kv_ready,
            (sizeof(k_smem[0]) + sizeof(v_smem[0])) * kConsumerWarpgroups
        );
#pragma unroll
        for (int consumer = 0; consumer < kConsumerWarpgroups; ++consumer) {
            coord<k_tile> coordinate = {
                static_cast<int>(blockIdx.z),
                kv_head,
                static_cast<int>(blockIdx.x) * kConsumerWarpgroups + consumer,
                0,
            };
            tma::load_async(k_smem[consumer], g.k, coordinate, kv_ready);
            tma::load_async(v_smem[consumer], g.v, coordinate, kv_ready);
        }

        coord<q_tile> q_coordinate = {
            static_cast<int>(blockIdx.z),
            static_cast<int>(blockIdx.y),
            query_start,
            0,
        };
        tma::expect_bytes(q_ready[load_stage], sizeof(q_smem[0]));
        tma::load_async(q_smem[load_stage], g.q, q_coordinate, q_ready[load_stage]);
        tma::expect_bytes(dout_ready[load_stage], sizeof(dout_smem[0]));
        tma::load_async(
            dout_smem[load_stage],
            g.dout,
            q_coordinate,
            dout_ready[load_stage]
        );
        coord<l_tile> stats_coordinate = {
            static_cast<int>(blockIdx.z),
            static_cast<int>(blockIdx.y),
            0,
            query_start,
        };
        tma::expect_bytes(
            stats_ready[load_stage],
            sizeof(l_smem[0]) + sizeof(delta_smem[0])
        );
        tma::load_async(
            l_smem[load_stage],
            g.l_aux,
            stats_coordinate,
            stats_ready[load_stage]
        );
        tma::load_async(
            delta_smem[load_stage],
            g.delta,
            stats_coordinate,
            stats_ready[load_stage]
        );
    }
    __syncthreads();

    if (warpgroup == kNumWarpgroups - 1) {
        warpgroup::decrease_registers<24>();
        if (warp % kittens::WARPGROUP_WARPS == 0) {
            for (
                int query_block = query_start;
                query_block < query_blocks;
                ++query_block, load_stage ^= 1, next_stage ^= 1
            ) {
                if (query_block + 1 < query_blocks) {
                    coord<q_tile> coordinate = {
                        static_cast<int>(blockIdx.z),
                        static_cast<int>(blockIdx.y),
                        query_block + 1,
                        0,
                    };
                    warp::tma::expect_bytes(
                        q_ready[next_stage],
                        sizeof(q_smem[0])
                    );
                    warp::tma::load_async(
                        q_smem[next_stage],
                        g.q,
                        coordinate,
                        q_ready[next_stage]
                    );
                    warp::tma::expect_bytes(
                        dout_ready[next_stage],
                        sizeof(dout_smem[0])
                    );
                    warp::tma::load_async(
                        dout_smem[next_stage],
                        g.dout,
                        coordinate,
                        dout_ready[next_stage]
                    );
                    coord<l_tile> stats_coordinate = {
                        static_cast<int>(blockIdx.z),
                        static_cast<int>(blockIdx.y),
                        0,
                        query_block + 1,
                    };
                    warp::tma::expect_bytes(
                        stats_ready[next_stage],
                        sizeof(l_smem[0]) + sizeof(delta_smem[0])
                    );
                    warp::tma::load_async(
                        l_smem[next_stage],
                        g.l_aux,
                        stats_coordinate,
                        stats_ready[next_stage]
                    );
                    warp::tma::load_async(
                        delta_smem[next_stage],
                        g.delta,
                        stats_coordinate,
                        stats_ready[next_stage]
                    );
                }
                wait(
                    compute_done[load_stage],
                    ((query_block - query_start) / 2) % 2
                );
            }
        } else if (warp % kittens::WARPGROUP_WARPS == 1) {
            for (
                int query_block = query_start;
                query_block < query_blocks;
                ++query_block, load_stage ^= 1, next_stage ^= 1
            ) {
                wait(
                    compute_done[load_stage],
                    ((query_block - query_start) / 2) % 2
                );
                coord<dq_tile> coordinate = {
                    static_cast<int>(blockIdx.z),
                    static_cast<int>(blockIdx.y),
                    query_block,
                    0,
                };
                warp::tma::store_add_async(g.dq, dq_smem, coordinate);
                warp::tma::store_async_wait();
                if (kittens::laneid() == 0) {
                    arrive(dq_ready);
                }
            }
        }
    } else {
        rt_fl<16, kDepth> dk;
        rt_fl<16, kDepth> dv;
        rt_fl<16, 64> score;
        rt_fl<16, 64> probability;
        rt_fl<16, 64> ds;
        rt_fl<16, 64> dp;
        lowp_register ds_lowp;
        lowp_register probability_lowp;

        if (warpgroup == 0) {
            warpgroup::increase_registers<256>();
            wait(kv_ready, 0);
            for (
                int query_block = query_start;
                query_block < query_blocks;
                ++query_block, load_stage ^= 1, next_stage ^= 1
            ) {
                compute_loop(
                    stats_ready,
                    q_ready,
                    dout_ready,
                    score_done[0][load_stage],
                    dp_done[0][load_stage],
                    kv_step_done[0][load_stage],
                    score,
                    dp,
                    probability,
                    ds,
                    probability_lowp,
                    ds_lowp,
                    score_tmem[0][load_stage],
                    dp_tmem[0][load_stage],
                    dk_tmem[0],
                    dv_tmem[0],
                    q_smem,
                    k_smem,
                    v_smem,
                    dout_smem,
                    probability_smem,
                    ds_smem,
                    l_smem,
                    delta_smem,
                    query_block,
                    query_start,
                    load_stage,
                    g
                );
                rt_fl<16, kDepth> dq;
                const int completion_phase =
                    ((query_block - query_start) / 2) % 2;
                if constexpr (KernelGlobals::kE4m3Operands) {
                    fp8_mm_atb_corrected<0>(
                        dq_tmem[load_stage],
                        ds_smem[0],
                        k_smem[0]
                    );
                    fp8_mm_atb_corrected<1>(
                        dq_tmem[load_stage],
                        ds_smem[1],
                        k_smem[1]
                    );
                } else {
                    warpgroup::mm_AtB(
                        dq_tmem[load_stage],
                        ds_smem[0],
                        k_smem[0]
                    );
                    warpgroup::mma_AtB(
                        dq_tmem[load_stage],
                        ds_smem[1],
                        k_smem[1]
                    );
                }
                if (warpgroup::laneid() == 0) {
                    tensor_commit<1>(dq_tmem_done[load_stage]);
                }
                wait(dq_tmem_done[load_stage], completion_phase);
                tensor_after_thread_sync();
                // WG1 produced ds_smem[1].  Hold both consumer warpgroups
                // here until the dQ TCGEN read is complete so WG1 cannot
                // overwrite that shared tile for the next query block.
                group<8>::sync(11);
                load_half_tmem_async(dq, dq_tmem[load_stage]);
                wait(dq_ready, next_stage);
                tensor_load_wait();
                if constexpr (KernelGlobals::kE4m3Operands) {
                    warp::mul(
                        dq,
                        dq,
                        1.0f / 256.0f
                    );
                }
                warpgroup::store(dq_smem, dq);
                group<4>::sync(warpgroup::groupid() + 4);
                if (warpgroup::laneid() == 0) {
                    arrive(compute_done[load_stage]);
                }
            }
            if (warpgroup::laneid() == 0) {
                tensor_commit<1>(kv_tmem_done[0]);
            }
            wait(kv_tmem_done[0], 0);
            tensor_after_thread_sync();
            load_half_tmem_async(dk, dk_tmem[0]);
            load_half_tmem_async(dv, dv_tmem[0]);
            tensor_load_wait();
            store_kv<KernelGlobals, dk_tile, dv_tile>(
                dk_smem,
                dk,
                dv_smem,
                dv,
                g,
                dq_ready,
                kv_head,
                next_stage
            );
        } else {
            warpgroup::increase_registers<224>();
            wait(kv_ready, 0);
            for (
                int query_block = query_start;
                query_block < query_blocks;
                ++query_block, load_stage ^= 1, next_stage ^= 1
            ) {
                compute_loop(
                    stats_ready,
                    q_ready,
                    dout_ready,
                    score_done[1][load_stage],
                    dp_done[1][load_stage],
                    kv_step_done[1][load_stage],
                    score,
                    dp,
                    probability,
                    ds,
                    probability_lowp,
                    ds_lowp,
                    score_tmem[1][load_stage],
                    dp_tmem[1][load_stage],
                    dk_tmem[1],
                    dv_tmem[1],
                    q_smem,
                    k_smem,
                    v_smem,
                    dout_smem,
                    probability_smem,
                    ds_smem,
                    l_smem,
                    delta_smem,
                    query_block,
                    query_start,
                    load_stage,
                    g
                );
                group<8>::sync(11);
            }
            if (warpgroup::laneid() == 0) {
                tensor_commit<1>(kv_tmem_done[1]);
            }
            wait(kv_tmem_done[1], 0);
            tensor_after_thread_sync();
            load_half_tmem_async(dk, dk_tmem[1]);
            load_half_tmem_async(dv, dv_tmem[1]);
            tensor_load_wait();
            store_kv<KernelGlobals, dk_tile, dv_tile>(
                dk_smem,
                dk,
                dv_smem,
                dv,
                g,
                dq_ready,
                kv_head,
                next_stage
            );
        }
    }
}

template <typename KernelGlobals>
inline void launch_impl(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &l_aux,
    at::Tensor &delta,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale,
    cudaStream_t stream
) {
    const KernelGlobals g{
        kittens::py::tensor_to_gl<typename KernelGlobals::q_gl>(q),
        kittens::py::tensor_to_gl<typename KernelGlobals::k_gl>(k),
        kittens::py::tensor_to_gl<typename KernelGlobals::v_gl>(v),
        kittens::py::tensor_to_gl<typename KernelGlobals::dout_gl>(dout),
        kittens::py::tensor_to_gl<typename KernelGlobals::dq_gl>(dq),
        kittens::py::tensor_to_gl<typename KernelGlobals::dk_gl>(dk),
        kittens::py::tensor_to_gl<typename KernelGlobals::dv_gl>(dv),
        kittens::py::tensor_to_gl<typename KernelGlobals::l_gl>(
            l_aux,
            q.size(0),
            q.size(1),
            1,
            q.size(2)
        ),
        kittens::py::tensor_to_gl<typename KernelGlobals::delta_gl>(
            delta,
            q.size(0),
            q.size(1),
            1,
            q.size(2)
        ),
        scale,
        scale * kLog2E,
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(1) / k.size(1)),
    };
    const dim3 grid(
        static_cast<unsigned int>(q.size(2) / 128),
        static_cast<unsigned int>(q.size(1)),
        static_cast<unsigned int>(q.size(0))
    );
    CUDACHECK(cudaFuncSetAttribute(
        main_kernel<KernelGlobals>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDynamicSmemBytes
    ));
    main_kernel<KernelGlobals><<<grid, kThreads, kDynamicSmemBytes, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

inline void launch(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &l_aux,
    at::Tensor &delta,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale,
    cudaStream_t stream
) {
    launch_impl<globals>(
        q,
        k,
        v,
        dout,
        l_aux,
        delta,
        dq,
        dk,
        dv,
        scale,
        stream
    );
}

inline void launch_e4m3(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &l_aux,
    at::Tensor &delta,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale,
    cudaStream_t stream
) {
    // Q/K/V/dO payloads carry a common raw x4 scale.  Their pair products are
    // x16, so fold the exact reciprocal into both score and dS derivatives.
    launch_impl<e4m3_globals>(
        q,
        k,
        v,
        dout,
        l_aux,
        delta,
        dq,
        dk,
        dv,
        scale / 16.0f,
        stream
    );
}

}  // namespace tkfa4::native_gqa_tk_bwd::pipelined
