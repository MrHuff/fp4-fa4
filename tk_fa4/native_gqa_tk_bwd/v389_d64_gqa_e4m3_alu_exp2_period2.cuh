#pragma once

#include "v386_d64_gqa_e4m3_k128q128_halfcols.cuh"

// Orthogonal v389 candidate: retain v387's role-specialized K128 x Q128
// backward pipeline and shared P/dS flow exactly, while replacing half of the
// native EX2 sites with the authenticated cd57 packed-f32x2 degree-1 ALU
// approximation.  The flattened per-thread pair cadence is period 2.
namespace tkfa4::native_gqa_tk_bwd::v389_d64_gqa_e4m3_alu_exp2_period2 {

namespace base =
    tkfa4::native_gqa_tk_bwd::v385_d64_gqa_e4m3_k128q128;
namespace half =
    tkfa4::native_gqa_tk_bwd::v386_d64_gqa_e4m3_k128q128_halfcols;

using base::globals;
using base::gradient_stage_tile;
using base::gradient_tmem_tile;
using base::kDepth;
using base::kHeadRatio;
using base::kKeyTile;
using base::kKvHeads;
using base::kOperandScale;
using base::kQueryHeads;
using base::kQueryTile;
using base::operand_tile;
using base::attention_tile;
using base::attention_tmem_tile;
using half::attention_e4m3_fragment;
using half::attention_fp32_fragment;
using half::attention_tmem_fragment;

constexpr int kColumnHalf = half::kColumnHalf;
constexpr int kThreads = 512;
constexpr int kComputeWarps = 8;
constexpr int kReduceWarpBase = 8;
constexpr int kReduceWarps = 4;
constexpr int kTensorIssueWarp = 12;
constexpr int kLoaderWarp = 13;
constexpr int kStatsWarp = 14;
constexpr int kStages = 2;

constexpr int kDpDqTmemOffset = 0;
constexpr int kDkTmemOffset = 128;
constexpr int kDvTmemOffset = 192;
constexpr int kScoreTmemOffset = 256;

struct shared_storage {
    operand_tile k;
    operand_tile v;
    operand_tile q[kStages];
    operand_tile dout[kStages];
    attention_tile probability_ds[kStages];
    gradient_stage_tile gradient;
    sv_fl<kQueryTile> lstat[kStages];
    sv_fl<kQueryTile> dstat[kStages];
};

static_assert(sizeof(shared_storage) <= 100352);
static_assert(sizeof(shared_storage) < 128 * 1024);

__device__ __forceinline__ int stage_phase(int iteration) {
    return (iteration >> 1) & 1;
}

__device__ __forceinline__ int previous_stage_phase(int iteration) {
    return ((iteration >> 1) - 1) & 1;
}

__device__ __forceinline__ attention_tmem_tile score_stage(
    const attention_tmem_tile &score_base,
    int stage
) {
    return attention_tmem_tile{score_base.addr + stage * kQueryTile};
}

// Exact CUDA-inline-PTX port of authenticated cd57's
// tk_exp2_alu_degree1_f32x2.  Keep this instruction sequence byte-for-byte in
// algorithmic content: only CUDA operand spelling differs from LLVM inline
// assembly.
__device__ __forceinline__ float2 exp2_alu_degree1_f32x2(float2 value) {
    uint32_t output_x;
    uint32_t output_y;
    asm(
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5;\n\t"
        ".reg .b64 l1, l2, l3, l4, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f5, 0f3F317218;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l4, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 %0, r7;\n\t"
        "mov.b32 %1, r8;\n\t"
        "}\n"
        : "=r"(output_x), "=r"(output_y)
        : "f"(value.x), "f"(value.y)
    );
    return {__uint_as_float(output_x), __uint_as_float(output_y)};
}

__device__ __forceinline__ float2 exp2_native_f32x2(float2 value) {
    float2 output;
    asm(
        "ex2.approx.ftz.f32 %0, %2;\n\t"
        "ex2.approx.ftz.f32 %1, %3;\n"
        : "=f"(output.x), "=f"(output.y)
        : "f"(value.x), "f"(value.y)
    );
    return output;
}

__device__ __forceinline__ float2 clamp_probability_log2(float2 value) {
    // Preserve v387/the native donor's public ABI: l_aux is unprelifted, so
    // device-side lstat is -LSE*log2(e).  The backward Q/K score is published
    // independently of the forward score; clamp its reconstructed log2(P)
    // to the mathematically valid unlifted ceiling of zero.
    float2 output;
    asm(
        "min.ftz.f32 %0, %2, 0f00000000;\n\t"
        "min.ftz.f32 %1, %3, 0f00000000;\n"
        : "=f"(output.x), "=f"(output.y)
        : "f"(value.x), "f"(value.y)
    );
    return output;
}

template <int PairIndex>
__device__ __forceinline__ void exp2_degree1_period2_pair(
    attention_fp32_fragment &score
) {
    static_assert(attention_fp32_fragment::height == 1);
    static_assert(attention_fp32_fragment::width == 4);
    static_assert(attention_fp32_fragment::packed_per_tile == 4);
    constexpr int kPairsPerTile =
        attention_fp32_fragment::packed_per_tile;
    constexpr int kPairCount =
        attention_fp32_fragment::height *
        attention_fp32_fragment::width *
        kPairsPerTile;
    constexpr int kTileColumn = PairIndex / kPairsPerTile;
    constexpr int kPayload = PairIndex % kPairsPerTile;

    const float2 clamped = clamp_probability_log2(
        score.tiles[0][kTileColumn].data[kPayload]
    );
    if constexpr ((PairIndex % 2) == 0) {
        score.tiles[0][kTileColumn].data[kPayload] =
            exp2_alu_degree1_f32x2(clamped);
    } else {
        score.tiles[0][kTileColumn].data[kPayload] =
            exp2_native_f32x2(clamped);
    }
    if constexpr (PairIndex + 1 < kPairCount) {
        exp2_degree1_period2_pair<PairIndex + 1>(score);
    }
}

__device__ __forceinline__ void exp2_degree1_period2(
    attention_fp32_fragment &score
) {
    // CuTe's flat loop uses (i / 2) % 2 == 0.  Each TK float2 payload is one
    // such pair; flattening [tile-column][payload] gives pair ordinals 0..15.
    exp2_degree1_period2_pair<0>(score);
}

__device__ __forceinline__ void make_probability_half(
    const attention_tmem_tile &score_tmem,
    shared_storage &storage,
    int output_subtile,
    int column_half,
    int stage,
    bool diagonal,
    float beta_log2e
) {
    attention_fp32_fragment score;
    attention_e4m3_fragment probability;
    const attention_tmem_fragment score_half =
        score_tmem.template subtile<attention_tmem_fragment>(
            0,
            column_half * kColumnHalf
        );
    group<8>::load_async(score, score_half);
    tensor_load_wait();
    warp::mul(score, score, beta_log2e);
    half::add_shared_row_vector_half(
        score,
        storage.lstat[stage],
        column_half
    );
    if (diagonal) {
        half::apply_diagonal_causal_mask(
            score,
            half::output_subtile_for_warp(warpid()),
            column_half
        );
    }
    // lstat is unprelifted at the native public boundary.  Apply exp2 first,
    // then retain v387's sole 2**8 lift for the E4M3 dV operand.
    exp2_degree1_period2(score);
    warp::mul(score, score, 256.0f);
    base::donor::convert_f32_to_e4m3(probability, score);
    auto destination =
        storage.probability_ds[stage].template subtile<16, kColumnHalf>(
            {output_subtile, column_half}
        );
    warp::store(destination, probability);
}

__device__ __forceinline__ void make_ds_half(
    const attention_tmem_tile &dp_tmem,
    shared_storage &storage,
    int output_subtile,
    int column_half,
    int stage,
    float beta
) {
    attention_fp32_fragment probability;
    attention_fp32_fragment dp;
    attention_e4m3_fragment probability_lowp;
    attention_e4m3_fragment ds_lowp;

    auto source =
        storage.probability_ds[stage].template subtile<16, kColumnHalf>(
            {output_subtile, column_half}
        );
    half::load_e4m3_half(probability_lowp, source);
    warp::copy(probability, probability_lowp);
    warp::mul(probability, probability, 1.0f / 256.0f);

    const attention_tmem_fragment dp_half =
        dp_tmem.template subtile<attention_tmem_fragment>(
            0,
            column_half * kColumnHalf
        );
    group<8>::load_async(dp, dp_half);
    tensor_load_wait();
    half::add_shared_row_vector_half(
        dp,
        storage.dstat[stage],
        column_half
    );
    warp::mul(dp, probability, dp);
    warp::mul(dp, dp, beta * 256.0f);
    base::donor::convert_f32_to_e4m3(ds_lowp, dp);
    auto destination =
        storage.probability_ds[stage].template subtile<16, kColumnHalf>(
            {output_subtile, column_half}
        );
    warp::store(destination, ds_lowp);
}

// group<4>::load_async uses physical warpid in its TMEM row mapping.  The
// publication role occupies physical warps 8..11, so address its logical
// 32-row slice explicitly and use a one-warp TMEM load.
__device__ __forceinline__ void drain_gradient_to_bf16(
    const gradient_tmem_tile &source,
    gradient_stage_tile &destination,
    int logical_warp
) {
    using warp_tmem_tile = tt_fl<32, kDepth>;
    rt_fl<32, kDepth> values_fp32;
    rt_bf<32, kDepth> values_bf16;
    const warp_tmem_tile source_slice =
        source.template subtile<warp_tmem_tile>(logical_warp * 32, 0);
    warp::load_async(values_fp32, source_slice);
    tensor_load_wait();
    warp::mul(values_fp32, values_fp32, 1.0f / 256.0f);
    warp::copy(values_bf16, values_fp32);
    auto destination_slice = destination.template subtile<32, kDepth>(
        {logical_warp, 0}
    );
    warp::store(destination_slice, values_bf16);
}

__device__ __forceinline__ void publish_gradient(
    const globals::gradient_gl &destination,
    gradient_stage_tile &stage,
    const coord<gradient_stage_tile> &coordinate,
    int logical_warp,
    barrier<kReduceWarps> staged,
    barrier<kReduceWarps> released
) {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    arrive_and_wait(staged);
    if (logical_warp == 0) {
        warp::tma::store_add_async(destination, stage, coordinate);
        warp::tma::store_async_wait();
    }
    arrive_and_wait(released);
}

__global__ __launch_bounds__(kThreads, 1)
void main_kernel(const __grid_constant__ globals g) {
    __shared__ alignas(1024) shared_storage storage;
    __shared__ alignas(16) semaphore persistent_ready;
    __shared__ alignas(16) semaphore query_ready[kStages];
    __shared__ alignas(16) semaphore stats_ready[kStages];
    __shared__ alignas(16) semaphore stats_consumed[kStages];
    __shared__ alignas(16) semaphore score_ready[kStages];
    __shared__ alignas(16) semaphore score_consumed[kStages];
    __shared__ alignas(16) semaphore probability_ready[kStages];
    __shared__ alignas(16) semaphore dp_ready[kStages];
    __shared__ alignas(16) semaphore dv_done[kStages];
    __shared__ alignas(16) semaphore ds_ready[kStages];
    __shared__ alignas(16) semaphore dk_done[kStages];
    __shared__ alignas(16) semaphore dq_ready[kStages];
    __shared__ alignas(16) semaphore dq_drained[kStages];
    __shared__ alignas(16) semaphore kernel_complete;

    const int physical_warp = warpid();
    const int lane = laneid();
    const int key_tile = static_cast<int>(blockIdx.x);
    const int query_head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int kv_head = query_head / kHeadRatio;
    const int iteration_count = g.sequence / kQueryTile - key_tile;

    // 8 compute warps retain the compiler's 128-register allocation.  The
    // publication WG borrows 24 registers/lane from the three control warps
    // and the idle warp: 8*128 + 4*152 + 4*96 = 64,512 registers/CTA.
    if (physical_warp >= kReduceWarpBase &&
        physical_warp < kTensorIssueWarp) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 152;" ::: "memory");
    } else if (physical_warp >= kTensorIssueWarp) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;" ::: "memory");
    }

    if (threadIdx.x == 0) {
        init_semaphore(persistent_ready, 0, 1);
        for (int stage = 0; stage < kStages; ++stage) {
            init_semaphore(query_ready[stage], 0, 1);
            init_semaphore(stats_ready[stage], 0, 1);
            init_semaphore(stats_consumed[stage], 0, kComputeWarps);
            init_semaphore(score_ready[stage], 0, 1);
            init_semaphore(score_consumed[stage], 0, kComputeWarps);
            init_semaphore(probability_ready[stage], 0, kComputeWarps);
            init_semaphore(dp_ready[stage], 0, 1);
            init_semaphore(dv_done[stage], 0, 1);
            init_semaphore(ds_ready[stage], 0, kComputeWarps);
            init_semaphore(dk_done[stage], 0, 1);
            init_semaphore(dq_ready[stage], 0, 1);
            init_semaphore(dq_drained[stage], 0, kReduceWarps);
        }
        init_semaphore(kernel_complete, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1, 1> tmem_allocator{};
    attention_tmem_tile dp_dq_tmem =
        tmem_allocator.template allocate<attention_tmem_tile>(kDpDqTmemOffset);
    gradient_tmem_tile dq_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(kDpDqTmemOffset);
    gradient_tmem_tile dk_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(kDkTmemOffset);
    gradient_tmem_tile dv_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(kDvTmemOffset);
    attention_tmem_tile score_tmem_base =
        tmem_allocator.template allocate<attention_tmem_tile>(
            kScoreTmemOffset
        );

    if (physical_warp == kLoaderWarp && lane == 0) {
        tma::expect_bytes(
            persistent_ready,
            sizeof(storage.k) + sizeof(storage.v)
        );
        tma::load_async(
            storage.k,
            g.k,
            coord<operand_tile>{batch, kv_head, key_tile, 0},
            persistent_ready
        );
        tma::load_async(
            storage.v,
            g.v,
            coord<operand_tile>{batch, kv_head, key_tile, 0},
            persistent_ready
        );

        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & 1;
            if (iteration >= kStages) {
                const int old_phase = previous_stage_phase(iteration);
                // Score and dK consume Q through independent tensor-core
                // operations.  Retire both readers before reusing Q.
                wait(score_ready[stage], old_phase);
                wait(dk_done[stage], old_phase);
                // dP and dV consume the same dO stage through independent
                // tensor-core operations.  dV completion alone does not
                // order the dP operand read, so both must retire before TMA
                // is allowed to overwrite this ping-pong slot.
                wait(dp_ready[stage], old_phase);
                wait(dv_done[stage], old_phase);
            }
            tma::expect_bytes(
                query_ready[stage],
                sizeof(storage.q[stage]) + sizeof(storage.dout[stage])
            );
            const int query_tile = key_tile + iteration;
            const coord<operand_tile> coordinate{
                batch,
                query_head,
                query_tile,
                0,
            };
            tma::load_async(
                storage.q[stage], g.q, coordinate, query_ready[stage]
            );
            tma::load_async(
                storage.dout[stage], g.dout, coordinate, query_ready[stage]
            );
        }
    } else if (physical_warp == kStatsWarp) {
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & 1;
            if (iteration >= kStages) {
                wait(
                    stats_consumed[stage],
                    previous_stage_phase(iteration)
                );
            }
            const int query_tile = key_tile + iteration;
            const int stats_base =
                (batch * kQueryHeads + query_head) * g.sequence +
                query_tile * kQueryTile;
            for (int column = lane; column < kQueryTile; column += 32) {
                storage.lstat[stage][column] =
                    g.l_aux[stats_base + column] * g.l_aux_scale;
                storage.dstat[stage][column] =
                    -16.0f * g.delta[stats_base + column];
            }
            __syncwarp();
            if (lane == 0) {
                arrive(stats_ready[stage]);
            }
        }
    } else if (physical_warp < kComputeWarps) {
        const int output_subtile =
            half::output_subtile_for_warp(physical_warp);
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & 1;
            const int phase = stage_phase(iteration);
            wait(score_ready[stage], phase);
            wait(stats_ready[stage], phase);
            tensor_after_thread_sync();
            if (iteration >= kStages) {
                const int old_phase = previous_stage_phase(iteration);
                wait(dk_done[stage], old_phase);
                wait(dq_ready[stage], old_phase);
            }
#pragma unroll 1
            for (int column_half = 0; column_half < 2; ++column_half) {
                make_probability_half(
                    score_stage(score_tmem_base, stage),
                    storage,
                    output_subtile,
                    column_half,
                    stage,
                    iteration == 0,
                    g.beta_log2e
                );
            }
            tensor_before_thread_sync();
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                arrive(score_consumed[stage]);
                arrive(probability_ready[stage]);
            }

            wait(dp_ready[stage], phase);
            wait(dv_done[stage], phase);
            tensor_after_thread_sync();
#pragma unroll 1
            for (int column_half = 0; column_half < 2; ++column_half) {
                make_ds_half(
                    dp_dq_tmem,
                    storage,
                    output_subtile,
                    column_half,
                    stage,
                    g.beta
                );
            }
            tensor_before_thread_sync();
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                arrive(stats_consumed[stage]);
                arrive(ds_ready[stage]);
            }
        }
        // tensor_allocator deprovisions TMEM from physical warp 0 in its
        // destructor.  Keep that warp alive until the publication WG has
        // drained and stored the final dV tile.
        if (physical_warp == 0) {
            wait(kernel_complete, 0);
            tensor_after_thread_sync();
        }
    } else if (physical_warp == kTensorIssueWarp && lane == 0) {
        wait(persistent_ready, 0);

        wait(query_ready[0], 0);
        base::issue_score_or_dp(
            score_tmem_base, storage.k, storage.q[0], score_ready[0]
        );
        wait(probability_ready[0], 0);
        base::issue_score_or_dp(
            dp_dq_tmem, storage.v, storage.dout[0], dp_ready[0]
        );
        base::issue_gradient_ab<0>(
            dv_tmem,
            storage.probability_ds[0],
            storage.dout[0],
            dv_done[0]
        );

        for (int iteration = 0; iteration + 1 < iteration_count; ++iteration) {
            const int stage = iteration & 1;
            const int phase = stage_phase(iteration);
            const int next_iteration = iteration + 1;
            const int next_stage = next_iteration & 1;
            const int next_phase = stage_phase(next_iteration);

            wait(query_ready[next_stage], next_phase);
            if (next_iteration >= kStages) {
                wait(
                    score_consumed[next_stage],
                    previous_stage_phase(next_iteration)
                );
                tensor_after_thread_sync();
            }
            attention_tmem_tile next_score_tmem =
                score_stage(score_tmem_base, next_stage);
            base::issue_score_or_dp(
                next_score_tmem,
                storage.k,
                storage.q[next_stage],
                score_ready[next_stage]
            );

            wait(ds_ready[stage], phase);
            tensor_after_thread_sync();
            if (iteration == 0) {
                base::issue_gradient_ab<0>(
                    dk_tmem,
                    storage.probability_ds[stage],
                    storage.q[stage],
                    dk_done[stage]
                );
            } else {
                base::issue_gradient_ab<1>(
                    dk_tmem,
                    storage.probability_ds[stage],
                    storage.q[stage],
                    dk_done[stage]
                );
            }
            base::issue_gradient_atb(
                dq_tmem,
                storage.probability_ds[stage],
                storage.k,
                dq_ready[stage]
            );

            wait(probability_ready[next_stage], next_phase);
            wait(dq_drained[stage], phase);
            tensor_after_thread_sync();
            base::issue_score_or_dp(
                dp_dq_tmem,
                storage.v,
                storage.dout[next_stage],
                dp_ready[next_stage]
            );
            base::issue_gradient_ab<1>(
                dv_tmem,
                storage.probability_ds[next_stage],
                storage.dout[next_stage],
                dv_done[next_stage]
            );
        }

        const int last_iteration = iteration_count - 1;
        const int last_stage = last_iteration & 1;
        const int last_phase = stage_phase(last_iteration);
        wait(ds_ready[last_stage], last_phase);
        tensor_after_thread_sync();
        if (last_iteration == 0) {
            base::issue_gradient_ab<0>(
                dk_tmem,
                storage.probability_ds[last_stage],
                storage.q[last_stage],
                dk_done[last_stage]
            );
        } else {
            base::issue_gradient_ab<1>(
                dk_tmem,
                storage.probability_ds[last_stage],
                storage.q[last_stage],
                dk_done[last_stage]
            );
        }
        base::issue_gradient_atb(
            dq_tmem,
            storage.probability_ds[last_stage],
            storage.k,
            dq_ready[last_stage]
        );
    } else if (
        physical_warp >= kReduceWarpBase &&
        physical_warp < kTensorIssueWarp
    ) {
        const int logical_warp = physical_warp - kReduceWarpBase;
        barrier<kReduceWarps> gradient_staged(1);
        barrier<kReduceWarps> gradient_released(2);

        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & 1;
            const int phase = stage_phase(iteration);
            wait(dq_ready[stage], phase);
            tensor_after_thread_sync();
            drain_gradient_to_bf16(
                dq_tmem, storage.gradient, logical_warp
            );
            tensor_before_thread_sync();
            __syncwarp();
            if (lane == 0) {
                arrive(dq_drained[stage]);
            }
            publish_gradient(
                g.dq,
                storage.gradient,
                coord<gradient_stage_tile>{
                    batch,
                    query_head,
                    key_tile + iteration,
                    0,
                },
                logical_warp,
                gradient_staged,
                gradient_released
            );
        }

        const int last_iteration = iteration_count - 1;
        const int last_stage = last_iteration & 1;
        const int last_phase = stage_phase(last_iteration);
        wait(dk_done[last_stage], last_phase);
        tensor_after_thread_sync();
        drain_gradient_to_bf16(dk_tmem, storage.gradient, logical_warp);
        publish_gradient(
            g.dk,
            storage.gradient,
            coord<gradient_stage_tile>{batch, kv_head, key_tile, 0},
            logical_warp,
            gradient_staged,
            gradient_released
        );

        wait(dv_done[last_stage], last_phase);
        tensor_after_thread_sync();
        drain_gradient_to_bf16(dv_tmem, storage.gradient, logical_warp);
        tensor_before_thread_sync();
        __syncwarp();
        publish_gradient(
            g.dv,
            storage.gradient,
            coord<gradient_stage_tile>{batch, kv_head, key_tile, 0},
            logical_warp,
            gradient_staged,
            gradient_released
        );
        if (logical_warp == 0 && lane == 0) {
            arrive(kernel_complete);
        }
    }
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
    float softmax_scale,
    cudaStream_t stream
) {
    const float beta = softmax_scale / 16.0f;
    const globals g{
        kittens::py::tensor_to_gl<globals::operand_gl>(q),
        kittens::py::tensor_to_gl<globals::operand_gl>(k),
        kittens::py::tensor_to_gl<globals::operand_gl>(v),
        kittens::py::tensor_to_gl<globals::operand_gl>(dout),
        kittens::py::tensor_to_gl<globals::gradient_gl>(dq),
        kittens::py::tensor_to_gl<globals::gradient_gl>(dk),
        kittens::py::tensor_to_gl<globals::gradient_gl>(dv),
        reinterpret_cast<const float *>(l_aux.data_ptr()),
        reinterpret_cast<const float *>(delta.data_ptr()),
        beta,
        beta * kLog2E,
        softmax_scale * kLog2E,
        static_cast<int>(q.size(2)),
    };
    const dim3 grid(
        static_cast<unsigned int>(q.size(2) / kKeyTile),
        static_cast<unsigned int>(q.size(1)),
        static_cast<unsigned int>(q.size(0))
    );
    v389_d64_gqa_e4m3_alu_exp2_period2::main_kernel<<<
        grid,
        kThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v389_d64_gqa_e4m3_alu_exp2_period2
