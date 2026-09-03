#pragma once

#include "v429_d128_gqa_e4m3_owner_x32_gradient_publication_production_bshd.cuh"

// D128 exact-native production-backward experiment derived from v429/v424.
// dQ alone is evacuated from its aliased TMEM page into reducer registers
// before dP(next) is released: D0 is packed to sixteen BF16x2 registers, while
// D1-D3 remain in three distinct 32-register FP32 payloads.  Publication then
// reuses v429's two D32 BF16 shared stages.  dK/dV retain v429's serial x32
// publication path exactly.
namespace tkfa4::native_gqa_tk_bwd::v430_d128_gqa_e4m3_register_dq_evacuation_production_bshd {

namespace prior =
    tkfa4::native_gqa_tk_bwd::v429_d128_gqa_e4m3_owner_x32_gradient_publication_production_bshd;
using namespace prior;

struct dq_register_payload {
    uint32_t packed_d0[kDepthChunk / 2];
    uint32_t raw_d1[kDepthChunk];
    uint32_t raw_d2[kDepthChunk];
    uint32_t raw_d3[kDepthChunk];
};

static_assert(kDepthChunk == 32);
static_assert(kDepthChunks == 4);

__device__ __forceinline__ uint32_t scale_pack_bf16_pair(
    uint32_t low,
    uint32_t high
) {
    constexpr float kOutputScale = 1.0f / 256.0f;
    const float2 value = make_float2(
        __uint_as_float(low),
        __uint_as_float(high)
    );
    const float2 scale = make_float2(kOutputScale, kOutputScale);
    const bf16_2 packed = __float22bfloat162_rn(
        __fmul2_rn(value, scale)
    );
    return *reinterpret_cast<const uint32_t *>(&packed);
}

__device__ __forceinline__ void pack_d0(
    uint32_t (&destination)[kDepthChunk / 2],
    const uint32_t (&source)[kDepthChunk]
) {
#pragma unroll
    for (int pair = 0; pair < kDepthChunk / 2; ++pair) {
        destination[pair] = scale_pack_bf16_pair(
            source[2 * pair],
            source[2 * pair + 1]
        );
    }
}

// The packed result aliases the low half of the raw array.  The increasing
// traversal is safe because destination[pair] never overwrites either source
// element for a future pair.
__device__ __forceinline__ void pack_raw_chunk_in_place(
    uint32_t (&values)[kDepthChunk]
) {
#pragma unroll
    for (int pair = 0; pair < kDepthChunk / 2; ++pair) {
        values[pair] = scale_pack_bf16_pair(
            values[2 * pair],
            values[2 * pair + 1]
        );
    }
}

// Four reducer warps own one 32-row stripe apiece.  D0 is loaded, waited, and
// packed before the other raw fragments become live.  D1-D3 are then issued to
// distinct register arrays and retired by one tcgen load wait.  No shared
// publication or TMA issue is legal before the collective dq_drained arrival.
__device__ __forceinline__ void evacuate_dq_to_registers(
    const gradient_tmem_tile &source,
    dq_register_payload &payload,
    semaphore &source_drained,
    int logical_warp,
    int lane
) {
    const uint32_t source_row =
        source.addr + (static_cast<uint32_t>(logical_warp * 32) << 16);
    {
        uint32_t raw_d0[kDepthChunk];
        prior::load_tmem_owner_x32(raw_d0, source_row);
        tensor_load_wait();
        pack_d0(payload.packed_d0, raw_d0);
    }

    prior::load_tmem_owner_x32(
        payload.raw_d1,
        source_row + 1 * kDepthChunk
    );
    prior::load_tmem_owner_x32(
        payload.raw_d2,
        source_row + 2 * kDepthChunk
    );
    prior::load_tmem_owner_x32(
        payload.raw_d3,
        source_row + 3 * kDepthChunk
    );
    tensor_load_wait();
    tensor_before_thread_sync();
    __syncwarp();
    if (lane == 0) {
        arrive(source_drained);
    }
}

__device__ __forceinline__ void stage_packed_dq_chunk(
    const uint32_t (&packed)[kDepthChunk / 2],
    gradient_chunk_tile &destination,
    int logical_warp,
    int lane
) {
    const int physical_row = logical_warp * 32 + lane;
    const uint32_t destination_base = static_cast<uint32_t>(
        __cvta_generic_to_shared(destination.data)
    );
#pragma unroll
    for (int packed_column = 0;
         packed_column < kDepthChunk / 2;
         packed_column += 4) {
        asm volatile(
            "st.shared.v4.b32 [%4], {%0, %1, %2, %3};\n"
            :
            : "r"(packed[packed_column + 0]),
              "r"(packed[packed_column + 1]),
              "r"(packed[packed_column + 2]),
              "r"(packed[packed_column + 3]),
              "r"(gradient_chunk_tile::idx(
                  destination_base,
                  {physical_row, 2 * packed_column}
              ))
            : "memory"
        );
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void publish_registered_dq_chunk(
    const uint32_t (&packed)[kDepthChunk / 2],
    const globals::gradient_gl &destination,
    gradient_chunk_tile (&publication)[kGradientPublicationStages],
    semaphore (&ready)[kGradientPublicationStages],
    semaphore (&reusable)[kGradientPublicationStages],
    int &publication_sequence,
    int batch,
    int sequence_tile,
    int head,
    int depth_chunk,
    int logical_warp,
    int lane
) {
    const int stage =
        publication_sequence & (kGradientPublicationStages - 1);
    const int ready_phase =
        (publication_sequence / kGradientPublicationStages) & 1;
    if (publication_sequence >= kGradientPublicationStages) {
        const int reuse_phase =
            ((publication_sequence - kGradientPublicationStages) /
             kGradientPublicationStages) & 1;
        if (logical_warp == 0 && lane == 0) {
            warp::tma::store_async_read_wait<1>();
            arrive(reusable[stage]);
        }
        wait(reusable[stage], reuse_phase);
    }

    stage_packed_dq_chunk(
        packed,
        publication[stage],
        logical_warp,
        lane
    );
    tensor_before_thread_sync();
    __syncwarp();
    if (lane == 0) {
        arrive(ready[stage]);
    }
    if (logical_warp == 0) {
        wait(ready[stage], ready_phase);
        if (lane == 0) {
            warp::tma::store_add_async<
                dim::DEPTH,
                cache_policy::NORMAL
            >(
                destination,
                publication[stage],
                coord<gradient_chunk_tile>{
                    batch,
                    sequence_tile,
                    head,
                    depth_chunk,
                }
            );
        }
    }
    ++publication_sequence;
}

__device__ __forceinline__ void publish_dq_after_register_evacuation(
    const gradient_tmem_tile &source,
    const globals::gradient_gl &destination,
    gradient_chunk_tile (&publication)[kGradientPublicationStages],
    semaphore (&ready)[kGradientPublicationStages],
    semaphore (&reusable)[kGradientPublicationStages],
    semaphore &source_drained,
    int &publication_sequence,
    int batch,
    int sequence_tile,
    int head,
    int logical_warp,
    int lane
) {
    dq_register_payload payload;
    evacuate_dq_to_registers(
        source,
        payload,
        source_drained,
        logical_warp,
        lane
    );

    publish_registered_dq_chunk(
        payload.packed_d0,
        destination,
        publication,
        ready,
        reusable,
        publication_sequence,
        batch,
        sequence_tile,
        head,
        0,
        logical_warp,
        lane
    );

    pack_raw_chunk_in_place(payload.raw_d1);
    publish_registered_dq_chunk(
        *reinterpret_cast<uint32_t (*)[kDepthChunk / 2]>(payload.raw_d1),
        destination,
        publication,
        ready,
        reusable,
        publication_sequence,
        batch,
        sequence_tile,
        head,
        1,
        logical_warp,
        lane
    );

    pack_raw_chunk_in_place(payload.raw_d2);
    publish_registered_dq_chunk(
        *reinterpret_cast<uint32_t (*)[kDepthChunk / 2]>(payload.raw_d2),
        destination,
        publication,
        ready,
        reusable,
        publication_sequence,
        batch,
        sequence_tile,
        head,
        2,
        logical_warp,
        lane
    );

    pack_raw_chunk_in_place(payload.raw_d3);
    publish_registered_dq_chunk(
        *reinterpret_cast<uint32_t (*)[kDepthChunk / 2]>(payload.raw_d3),
        destination,
        publication,
        ready,
        reusable,
        publication_sequence,
        batch,
        sequence_tile,
        head,
        3,
        logical_warp,
        lane
    );
}

__global__ __launch_bounds__(kThreads, 1)
void main_kernel(const __grid_constant__ globals g) {
    __shared__ alignas(1024) shared_storage storage;
    __shared__ alignas(16) semaphore persistent_ready;
    __shared__ alignas(16) semaphore query_ready[kInputStages];
    __shared__ alignas(16) semaphore stats_ready[kInputStages];
    __shared__ alignas(16) semaphore operand_consumed[kInputStages];
    __shared__ alignas(16) semaphore stats_consumed[kInputStages];
    __shared__ alignas(16) semaphore score_ready;
    __shared__ alignas(16) semaphore score_consumed;
    __shared__ alignas(16) semaphore probability_ready;
    __shared__ alignas(16) semaphore probability_consumed;
    __shared__ alignas(16) semaphore dp_ready;
    __shared__ alignas(16) semaphore dv_ready;
    __shared__ alignas(16) semaphore ds_ready;
    __shared__ alignas(16) semaphore dq_ready;
    __shared__ alignas(16) semaphore dk_ready;
    __shared__ alignas(16) semaphore dq_drained;
    __shared__ alignas(16) semaphore publication_ready[
        kGradientPublicationStages
    ];
    __shared__ alignas(16) semaphore publication_reusable[
        kGradientPublicationStages
    ];
    __shared__ alignas(16) semaphore kernel_complete;

    const int physical_warp = warpid();
    const int lane = laneid();
    const int key_tile = static_cast<int>(blockIdx.x);
    const int query_head = static_cast<int>(blockIdx.y);
    const int batch = static_cast<int>(blockIdx.z);
    const int kv_head = query_head / kHeadRatio;
    const int iteration_count = g.sequence / kQueryTile - key_tile;

    // 8 compute warps x 128 + 4 reducer warps x 152 +
    // 4 control/idle warps x 96 = 64,512 registers per CTA.
    if (physical_warp < kComputeWarps) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 128;" ::: "memory");
    } else if (physical_warp < kTensorIssueWarp) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 152;" ::: "memory");
    } else {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;" ::: "memory");
    }

    if (threadIdx.x == 0) {
        init_semaphore(persistent_ready, 0, 1);
        for (int stage = 0; stage < kInputStages; ++stage) {
            init_semaphore(query_ready[stage], 0, 1);
            init_semaphore(stats_ready[stage], 0, 1);
            init_semaphore(operand_consumed[stage], 0, 1);
            init_semaphore(
                stats_consumed[stage],
                0,
                kComputeWarps
            );
        }
        init_semaphore(score_ready, 0, 1);
        init_semaphore(score_consumed, 0, kComputeWarps);
        init_semaphore(probability_ready, 0, kComputeWarps);
        init_semaphore(probability_consumed, 0, kComputeWarps);
        init_semaphore(dp_ready, 0, 1);
        init_semaphore(dv_ready, 0, 1);
        init_semaphore(ds_ready, 0, kComputeWarps);
        init_semaphore(dq_ready, 0, 1);
        init_semaphore(dk_ready, 0, 1);
        init_semaphore(dq_drained, 0, kReduceWarps);
        for (int stage = 0; stage < kGradientPublicationStages; ++stage) {
            init_semaphore(publication_ready[stage], 0, kReduceWarps);
            init_semaphore(publication_reusable[stage], 0, 1);
        }
        init_semaphore(kernel_complete, 0, 1);
    }
    __syncthreads();

    tensor_allocator<1, 1> tmem_allocator{};
    attention_tmem_tile dp_tmem =
        tmem_allocator.template allocate<attention_tmem_tile>(
            kDpDqTmemOffset
        );
    gradient_tmem_tile dq_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(
            kDpDqTmemOffset
        );
    gradient_tmem_tile dk_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(
            kDkTmemOffset
        );
    gradient_tmem_tile dv_tmem =
        tmem_allocator.template allocate<gradient_tmem_tile>(
            kDvTmemOffset
        );
    attention_tmem_tile score_tmem =
        tmem_allocator.template allocate<attention_tmem_tile>(
            kScoreTmemOffset
        );

    if (physical_warp == kLoaderWarp && lane == 0) {
        tma::expect_bytes(
            persistent_ready,
            sizeof(storage.k) + sizeof(storage.v)
        );
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.k,
            g.k,
            coord<operand_tile>{batch, key_tile, kv_head, 0},
            persistent_ready
        );
        tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
            storage.v,
            g.v,
            coord<operand_tile>{batch, key_tile, kv_head, 0},
            persistent_ready
        );

        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & (kInputStages - 1);
            if (iteration >= kInputStages) {
                const int old_phase = previous_input_epoch_phase(iteration);
                wait(operand_consumed[stage], old_phase);
                wait(stats_consumed[stage], old_phase);
                tensor_after_thread_sync();
            }
            const int query_tile = key_tile + iteration;
            const coord<operand_tile> operand_coordinate{
                batch,
                query_tile,
                query_head,
                0,
            };
            tma::expect_bytes(
                query_ready[stage],
                sizeof(storage.q[stage]) + sizeof(storage.dout[stage])
            );
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                storage.q[stage],
                g.q,
                operand_coordinate,
                query_ready[stage]
            );
            tma::load_async<dim::DEPTH, cache_policy::NORMAL>(
                storage.dout[stage],
                g.dout,
                operand_coordinate,
                query_ready[stage]
            );

            const coord<stats_tile> stats_coordinate{
                batch,
                query_head,
                0,
                query_tile,
            };
            tma::expect_bytes(
                stats_ready[stage],
                sizeof(storage.lstat[stage]) +
                    sizeof(storage.dstat[stage])
            );
            tma::load_async(
                storage.lstat[stage],
                g.lstat,
                stats_coordinate,
                stats_ready[stage]
            );
            tma::load_async(
                storage.dstat[stage],
                g.dstat,
                stats_coordinate,
                stats_ready[stage]
            );
        }
    } else if (physical_warp < kComputeWarps) {
        const int output_subtile = output_subtile_for_warp(physical_warp);
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & (kInputStages - 1);
            const int phase = iteration_phase(iteration);
            const int input_phase = input_stage_epoch_phase(iteration);
            if (iteration > 0) {
                const int old_phase = iteration_phase(iteration - 1);
                wait(dv_ready, old_phase);
                wait(probability_consumed, old_phase);
            }
            wait(score_ready, phase);
            wait(stats_ready[stage], input_phase);
            tensor_after_thread_sync();
            make_probability_half(
                score_tmem,
                storage,
                output_subtile,
                0,
                stage,
                iteration == 0,
                g.beta_log2e,
                nullptr
            );
            make_probability_half(
                score_tmem,
                storage,
                output_subtile,
                1,
                stage,
                iteration == 0,
                g.beta_log2e,
                &score_consumed
            );
            tensor_before_thread_sync();
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                arrive(probability_ready);
            }

            wait(dp_ready, phase);
            if (iteration > 0) {
                const int old_phase = iteration_phase(iteration - 1);
                wait(dq_ready, old_phase);
                wait(dk_ready, old_phase);
            }
            tensor_after_thread_sync();
            make_ds_half(
                dp_tmem,
                storage,
                output_subtile,
                0,
                stage,
                g.beta
            );
            make_ds_half(
                dp_tmem,
                storage,
                output_subtile,
                1,
                stage,
                g.beta
            );
            tensor_before_thread_sync();
            __syncwarp();
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (lane == 0) {
                arrive(probability_consumed);
                arrive(stats_consumed[stage]);
                arrive(ds_ready);
            }
        }
        if (physical_warp == 0) {
            wait(kernel_complete, 0);
            tensor_after_thread_sync();
        }
    } else if (physical_warp == kTensorIssueWarp && lane == 0) {
        wait(persistent_ready, 0);

        // Prime score and dP for iteration zero.  They target disjoint TMEM
        // pages, and dP depends only on persistent V plus the staged dO.
        wait(query_ready[0], 0);
        core::issue_score_or_dp(
            score_tmem,
            storage.k,
            storage.q[0],
            score_ready
        );
        core::issue_score_or_dp(
            dp_tmem,
            storage.v,
            storage.dout[0],
            dp_ready
        );

        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int stage = iteration & (kInputStages - 1);
            const int phase = iteration_phase(iteration);
            const bool has_next = iteration + 1 < iteration_count;
            const int next_iteration = iteration + 1;
            const int next_stage = next_iteration & (kInputStages - 1);

            // Every compute warp has completed its second-half score load, but
            // continues native EX2/E4M3/shared-P work from that one live owner
            // fragment.  Reuse score TMEM now instead of waiting for P.
            if (has_next) {
                wait(score_consumed, phase);
                wait(
                    query_ready[next_stage],
                    input_stage_epoch_phase(next_iteration)
                );
                tensor_after_thread_sync();
                core::issue_score_or_dp(
                    score_tmem,
                    storage.k,
                    storage.q[next_stage],
                    score_ready
                );
            }

            wait(probability_ready, phase);
            tensor_after_thread_sync();
            if (iteration == 0) {
                core::issue_gradient_ab<0>(
                    dv_tmem,
                    storage.probability,
                    storage.dout[stage],
                    dv_ready
                );
            } else {
                core::issue_gradient_ab<1>(
                    dv_tmem,
                    storage.probability,
                    storage.dout[stage],
                    dv_ready
                );
            }

            wait(ds_ready, phase);
            tensor_after_thread_sync();
            core::issue_gradient_atb(
                dq_tmem,
                storage.ds,
                storage.k,
                dq_ready
            );
            if (iteration == 0) {
                core::issue_gradient_ab<0>(
                    dk_tmem,
                    storage.ds,
                    storage.q[stage],
                    dk_ready
                );
            } else {
                core::issue_gradient_ab<1>(
                    dk_tmem,
                    storage.ds,
                    storage.q[stage],
                    dk_ready
                );
            }

            // dP(next) is independent of next P/score, but aliases current dQ
            // in TMEM [0,128).  Launch it at the exact collective reducer
            // release point, before waiting on current dK/dV solely for input
            // stage recycling.  ds_ready already proves dP(current) was fully
            // consumed, so no redundant dp_ready wait belongs on this path.
            if (has_next) {
                wait(dq_drained, phase);
                tensor_after_thread_sync();
                core::issue_score_or_dp(
                    dp_tmem,
                    storage.v,
                    storage.dout[next_stage],
                    dp_ready
                );
            }

            // score_consumed retired Q's score reader, dK retires its remaining
            // Q reader, ds_ready retired dP's dO reader, and dV retires the
            // remaining dO reader.  Only dK/dV completion is still required.
            wait(dk_ready, phase);
            wait(dv_ready, phase);
            tensor_after_thread_sync();
            arrive(operand_consumed[stage]);
        }
    } else if (
        physical_warp >= kReduceWarpBase &&
        physical_warp < kReduceWarpBase + kReduceWarps
    ) {
        const int logical_warp = physical_warp - kReduceWarpBase;
        int publication_sequence = 0;
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            const int phase = iteration_phase(iteration);
            wait(dq_ready, phase);
            tensor_after_thread_sync();
            publish_dq_after_register_evacuation(
                dq_tmem,
                g.dq,
                storage.gradient,
                publication_ready,
                publication_reusable,
                dq_drained,
                publication_sequence,
                batch,
                key_tile + iteration,
                query_head,
                logical_warp,
                lane
            );
        }

        const int last_phase = iteration_phase(iteration_count - 1);
        wait(dk_ready, last_phase);
        tensor_after_thread_sync();
        publish_gradient_tile_owner_x32(
            dk_tmem,
            g.dk,
            storage.gradient,
            publication_ready,
            publication_reusable,
            nullptr,
            publication_sequence,
            batch,
            key_tile,
            kv_head,
            logical_warp,
            lane
        );

        wait(dv_ready, last_phase);
        tensor_after_thread_sync();
        publish_gradient_tile_owner_x32(
            dv_tmem,
            g.dv,
            storage.gradient,
            publication_ready,
            publication_reusable,
            nullptr,
            publication_sequence,
            batch,
            key_tile,
            kv_head,
            logical_warp,
            lane
        );
        if (logical_warp == 0 && lane == 0) {
            warp::tma::store_async_wait<0>();
            arrive(kernel_complete);
        }
    }
}

inline void launch(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lstat,
    at::Tensor &dstat,
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
        kittens::py::tensor_to_gl<globals::stats_gl>(
            lstat,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        kittens::py::tensor_to_gl<globals::stats_gl>(
            dstat,
            q.size(0),
            q.size(2),
            1,
            q.size(1)
        ),
        beta,
        beta * kLog2E,
        static_cast<int>(q.size(1)),
    };
    const dim3 grid(
        static_cast<unsigned int>(q.size(1) / kKeyTile),
        static_cast<unsigned int>(q.size(2)),
        static_cast<unsigned int>(q.size(0))
    );
    v430_d128_gqa_e4m3_register_dq_evacuation_production_bshd::main_kernel<<<
        grid,
        kThreads,
        0,
        stream
    >>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::v430_d128_gqa_e4m3_register_dq_evacuation_production_bshd
