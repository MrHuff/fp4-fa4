#pragma once

#ifndef TK_FA4_LOCAL_P_STAGE_LADDER_BUILD
#define TK_FA4_LOCAL_P_STAGE_LADDER_BUILD 0
#endif

#if TK_FA4_LOCAL_P_STAGE_LADDER_BUILD

#if !TK_FA4_CLUSTER_P_PIPELINE_PROBE_BUILD
#error "The local-P stage ladder requires CLUSTER_P_PIPELINE_PROBE=1"
#endif

#if TK_FA4_CLUSTER_P_PIPELINE_TRACE
#error "The local-P stage ladder requires CLUSTER_P_PIPELINE_TRACE=0"
#endif

// Fixed-work, compile-time-specialized local-P traffic ladder.  The existing
// kernel_mxfp4_cluster_p_local_control is the authoritative H implementation;
// it deliberately remains in fwd_cluster_p_pipeline.inc and is not modified.
enum local_p_stage_ladder_mode : int {
    LOCAL_P_STAGE_A = 0,
    LOCAL_P_STAGE_B = 1,
    LOCAL_P_STAGE_C = 2,
    LOCAL_P_STAGE_D = 3,
    LOCAL_P_STAGE_E = 4,
    LOCAL_P_STAGE_F = 5,
    LOCAL_P_STAGE_G = 6,
    LOCAL_P_STAGE_B_C_LIFECYCLE_RESIDENT = 7,
    LOCAL_P_STAGE_B_C_LIFECYCLE_PUBLISH = 8,
    LOCAL_P_STAGE_C_SERIAL = 9,
};

static constexpr int LOCAL_P_STAGE_EPOCHS = 32;

template <
    bool _FULL_OUTPUT_STORE,
    bool _LOAD_P_EACH_EPOCH,
    bool _PUBLISH_EACH_EPOCH,
    bool _LOAD_P_SCALE_EACH_EPOCH,
    bool _STAGE_P_SCALE_EACH_EPOCH,
    bool _LOAD_V_EACH_EPOCH,
    bool _LOAD_V_SCALE_EACH_EPOCH,
    bool _RESIDENT_ISSUE_PATH,
    bool _SERIAL_LIFECYCLE>
struct local_p_stage_mode_traits_base {
    static constexpr bool FULL_OUTPUT_STORE = _FULL_OUTPUT_STORE;
    static constexpr bool LOAD_P_EACH_EPOCH = _LOAD_P_EACH_EPOCH;
    static constexpr bool PUBLISH_EACH_EPOCH = _PUBLISH_EACH_EPOCH;
    static constexpr bool LOAD_P_SCALE_EACH_EPOCH =
        _LOAD_P_SCALE_EACH_EPOCH;
    static constexpr bool STAGE_P_SCALE_EACH_EPOCH =
        _STAGE_P_SCALE_EACH_EPOCH;
    static constexpr bool LOAD_V_EACH_EPOCH = _LOAD_V_EACH_EPOCH;
    static constexpr bool LOAD_V_SCALE_EACH_EPOCH =
        _LOAD_V_SCALE_EACH_EPOCH;
    static constexpr bool RESIDENT_ISSUE_PATH = _RESIDENT_ISSUE_PATH;
    static constexpr bool SERIAL_LIFECYCLE = _SERIAL_LIFECYCLE;
};

template <int MODE>
struct local_p_stage_mode_traits;

template <>
struct local_p_stage_mode_traits<LOCAL_P_STAGE_A>
    : local_p_stage_mode_traits_base<
          false, false, false, false, false, false, false, true, false> {};
template <>
struct local_p_stage_mode_traits<LOCAL_P_STAGE_B>
    : local_p_stage_mode_traits_base<
          true, false, false, false, false, false, false, true, false> {};
template <>
struct local_p_stage_mode_traits<LOCAL_P_STAGE_C>
    : local_p_stage_mode_traits_base<
          true, true, true, false, false, false, false, false, false> {};
template <>
struct local_p_stage_mode_traits<LOCAL_P_STAGE_D>
    : local_p_stage_mode_traits_base<
          true, true, true, true, false, false, false, false, false> {};
template <>
struct local_p_stage_mode_traits<LOCAL_P_STAGE_E>
    : local_p_stage_mode_traits_base<
          true, true, true, true, true, false, false, false, false> {};
template <>
struct local_p_stage_mode_traits<LOCAL_P_STAGE_F>
    : local_p_stage_mode_traits_base<
          true, true, true, true, true, true, false, false, false> {};
template <>
struct local_p_stage_mode_traits<LOCAL_P_STAGE_G>
    : local_p_stage_mode_traits_base<
          true, true, true, true, true, true, true, false, false> {};
template <>
struct local_p_stage_mode_traits<LOCAL_P_STAGE_B_C_LIFECYCLE_RESIDENT>
    : local_p_stage_mode_traits_base<
          true, false, false, false, false, false, false, false, false> {};
template <>
struct local_p_stage_mode_traits<LOCAL_P_STAGE_B_C_LIFECYCLE_PUBLISH>
    : local_p_stage_mode_traits_base<
          true, false, true, false, false, false, false, false, false> {};
template <>
struct local_p_stage_mode_traits<LOCAL_P_STAGE_C_SERIAL>
    : local_p_stage_mode_traits_base<
          true, true, true, false, false, false, false, false, true> {};

struct globals_local_p_stage_ladder {
    using p_tile = globals_mxfp4_cluster_p_fake::p_tile;
    using p_scale_tile = globals_mxfp4_cluster_p_fake::p_scale_tile;
    using v_tile = globals_mxfp4_cluster_p_fake::v_tile;
    using v_scale_tile = globals_mxfp4_cluster_p_fake::v_scale_tile;

    const uint8_t *p_fp4;
    const int32_t *p_scale_words;
    const uint8_t *v_fp4_hot;
    const int32_t *v_scale_words_hot;
    const uint8_t *v_fp4_perturbed;
    const int32_t *v_scale_words_perturbed;
    float *output;
    int v_bank;
    int blocks;

    static constexpr int dynamic_shared_memory_bytes() {
        return globals_mxfp4_cluster_p_fake::dynamic_shared_memory_bytes();
    }

    __host__ __inline__ dim3 grid() const { return dim3(blocks); }
    __host__ __inline__ dim3 block() const {
        return dim3(config_mxfp4_cluster_p_fake::NUM_THREADS);
    }
    __host__ __inline__ int dynamic_shared_memory() const {
        return dynamic_shared_memory_bytes();
    }
};

template <bool PERTURBED>
__device__ __forceinline__ const uint8_t *local_p_stage_v_source(
    const globals_local_p_stage_ladder &g,
    int epoch) {
    if constexpr (PERTURBED) {
        constexpr int BYTES_PER_TILE = 128 * 32;
        const int64_t block =
            static_cast<int64_t>(g.v_bank) * g.blocks + blockIdx.x;
        const int64_t tile = block * LOCAL_P_STAGE_EPOCHS + epoch;
        return g.v_fp4_perturbed + tile * BYTES_PER_TILE;
    } else {
        // The load itself is still executed every epoch in F/H+V; only the
        // global address is deliberately hot.
        return g.v_fp4_hot;
    }
}

template <bool PERTURBED>
__device__ __forceinline__ const int32_t *local_p_stage_v_scale_source(
    const globals_local_p_stage_ladder &g,
    int epoch) {
    if constexpr (PERTURBED) {
        constexpr int WORDS_PER_TILE = 32 * 4;
        const int64_t block =
            static_cast<int64_t>(g.v_bank) * g.blocks + blockIdx.x;
        const int64_t tile = block * LOCAL_P_STAGE_EPOCHS + epoch;
        return g.v_scale_words_perturbed + tile * WORDS_PER_TILE;
    } else {
        return g.v_scale_words_hot;
    }
}

template <typename OutputTT>
__device__ __forceinline__ void local_p_stage_store_sink(
    OutputTT &output,
    float *destination) {
    // A has no full output store, but this one-lane four-byte sink keeps a real
    // data dependency from the accumulator through a TMEM load to global memory.
    if (warpgroup::warpid() == 0) {
        float value;
        asm volatile(
            "{tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];}"
            : "=f"(value)
            : "r"(output.addr));
        tensor_load_wait();
        if (warp::elect_leader()) {
            *destination = value;
        }
    }
    warpgroup::sync(2);
}

template <int MODE, bool PERTURBED>
__launch_bounds__(config_mxfp4_cluster_p_fake::NUM_THREADS, 1)
__global__ void kernel_mxfp4_local_p_stage_ladder(
    const __grid_constant__ globals_local_p_stage_ladder g) {
    using C = config_mxfp4_cluster_p_fake;
    using G = globals_local_p_stage_ladder;
    using T = local_p_stage_mode_traits<MODE>;
    using score_tt = tt<float, 128, 64>;
    using output_tt = tt<float, 128, 128>;
    using scale_tt = full_tt_fp8e8m0<16>;

    static_assert(MODE >= LOCAL_P_STAGE_A && MODE <= LOCAL_P_STAGE_C_SERIAL);
    static_assert(C::NUM_THREADS == 256 && C::TOTAL_WGS == 2);
    static_assert(LOCAL_P_STAGE_EPOCHS == 32);
    static_assert(
        G::dynamic_shared_memory_bytes() ==
        globals_mxfp4_cluster_p_fake::dynamic_shared_memory_bytes());
    constexpr int LOGICAL_END = 240;
    static_assert(LOGICAL_END <= tensor_allocator<2, 1>::cols);

    constexpr bool FULL_OUTPUT_STORE = T::FULL_OUTPUT_STORE;
    constexpr bool LOAD_P_EACH_EPOCH = T::LOAD_P_EACH_EPOCH;
    constexpr bool PUBLISH_EACH_EPOCH = T::PUBLISH_EACH_EPOCH;
    constexpr bool LOAD_P_SCALE_EACH_EPOCH =
        T::LOAD_P_SCALE_EACH_EPOCH;
    constexpr bool STAGE_P_SCALE_EACH_EPOCH =
        T::STAGE_P_SCALE_EACH_EPOCH;
    constexpr bool LOAD_V_EACH_EPOCH = T::LOAD_V_EACH_EPOCH;
    constexpr bool LOAD_V_SCALE_EACH_EPOCH =
        T::LOAD_V_SCALE_EACH_EPOCH;
    constexpr bool RESIDENT_ISSUE_PATH = T::RESIDENT_ISSUE_PATH;
    constexpr bool SERIAL_LIFECYCLE = T::SERIAL_LIFECYCLE;
    constexpr bool RESIDENT_P_SCALE = !LOAD_P_SCALE_EACH_EPOCH;
    constexpr bool RESIDENT_V_SCALE = !LOAD_V_SCALE_EACH_EPOCH;

    // F/G have explicitly separate hot/perturbed entry points.  Other rungs
    // instantiate only PERTURBED=false.
    static_assert(!PERTURBED ||
                  MODE == LOCAL_P_STAGE_F || MODE == LOCAL_P_STAGE_G);

    tensor_allocator<2, 1> tm_alloc{};
    score_tt score = tm_alloc.template allocate<score_tt>(0);
    output_tt output = tm_alloc.template allocate<output_tt>(64);
    scale_tt p_scale[2] = {
        tm_alloc.template allocate<scale_tt>(192),
        tm_alloc.template allocate<scale_tt>(208)};
    scale_tt v_scale = tm_alloc.template allocate<scale_tt>(224);
    (void)score;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::p_tile (&p_smem)[2] =
        allocator.allocate<typename G::p_tile, 2>();
    typename G::p_scale_tile (&p_scale_smem)[2] =
        allocator.allocate<typename G::p_scale_tile, 2>();
    typename G::v_tile &v_smem = allocator.allocate<typename G::v_tile>();
    typename G::v_scale_tile &v_scale_smem =
        allocator.allocate<typename G::v_scale_tile>();

    __shared__ semaphore p_scale_done[2];
    __shared__ semaphore v_scale_done;
    __shared__ semaphore pv_src_done[2];
    __shared__ semaphore pv_done[2];
    __shared__ semaphore all_pv_done;

    const int group = warpgroup::groupid();
    const bool issue_warp = group == 0 && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            init_semaphore(p_scale_done[slot], 0, 1);
            init_semaphore(pv_src_done[slot], 0, 1);
            init_semaphore(pv_done[slot], 0, 1);
        }
        init_semaphore(v_scale_done, 0, 1);
        init_semaphore(all_pv_done, 0, 1);
    }

    // Resident preparation.  For a rung that moves an operand every epoch,
    // epoch zero is its first (and not an additional thirty-third) load.
    if constexpr (!LOAD_P_EACH_EPOCH || !LOAD_P_SCALE_EACH_EPOCH) {
        if (group == 0) {
            if constexpr (!LOAD_P_EACH_EPOCH) {
                cluster_p_load_payload_wg(p_smem[0], g.p_fp4);
            }
            if constexpr (!LOAD_P_SCALE_EACH_EPOCH) {
                if (warpgroup::warpid() == 0) {
                    cluster_p_load_scale_warp(
                        p_scale_smem[0], g.p_scale_words);
                }
            }
            warpgroup::sync(1);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(1);
        }
    }
    if constexpr (!LOAD_V_EACH_EPOCH || !LOAD_V_SCALE_EACH_EPOCH) {
        if (group == 1) {
            if constexpr (!LOAD_V_EACH_EPOCH) {
                cluster_p_load_payload_wg(v_smem, g.v_fp4_hot);
            }
            if constexpr (!LOAD_V_SCALE_EACH_EPOCH) {
                if (warpgroup::warpid() == 0) {
                    cluster_p_load_scale_warp(
                        v_scale_smem, g.v_scale_words_hot);
                }
            }
            warpgroup::sync(2);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(2);
        }
    }
    __syncthreads();
    if (issue_warp) {
        // Match H's acquire before its resident V-scale SMEM-to-TMEM load.
        tensor_after_thread_sync();
    }

    if (issue_lane) {
        if constexpr (RESIDENT_P_SCALE) {
            fp4pv_load_mxnv_scale_async_ranklocal(
                p_scale[0], p_scale_smem[0]);
        }
        if constexpr (RESIDENT_V_SCALE) {
            fp4pv_load_mxnv_scale_async_ranklocal(v_scale, v_scale_smem);
        }
        if constexpr (RESIDENT_P_SCALE && RESIDENT_V_SCALE) {
            detail::tcgen05::commit<1>(p_scale_done[0]);
        } else if constexpr (RESIDENT_P_SCALE) {
            detail::tcgen05::commit<1>(p_scale_done[0]);
        } else if constexpr (RESIDENT_V_SCALE) {
            // E uses the same resident V-scale completion sequence as H.
            detail::tcgen05::commit<1>(v_scale_done);
        }
    }
    if (issue_warp) {
        if constexpr (RESIDENT_P_SCALE && RESIDENT_V_SCALE) {
            wait(p_scale_done[0], 0);
            tensor_after_thread_sync();
        } else if constexpr (RESIDENT_P_SCALE) {
            wait(p_scale_done[0], 0);
            tensor_after_thread_sync();
        } else if constexpr (RESIDENT_V_SCALE) {
            wait(v_scale_done, 0);
            tensor_after_thread_sync();
        }
    }

    if constexpr (RESIDENT_ISSUE_PATH) {
        if (issue_warp) {
            tensor_before_thread_sync();
        }
        if (issue_lane) {
            #pragma unroll 1
            for (int epoch = 0; epoch < LOCAL_P_STAGE_EPOCHS; ++epoch) {
                if (epoch == 0) {
                    mm_ABt(
                        output, p_smem[0], v_smem,
                        p_scale[0], v_scale);
                } else {
                    mma_ABt(
                        output, p_smem[0], v_smem,
                        p_scale[0], v_scale);
                }
            }
            tensor_commit<1>(all_pv_done);
        }
    } else if (group == 0) {
        #pragma unroll 1
        for (int epoch = 0; epoch < LOCAL_P_STAGE_EPOCHS; ++epoch) {
            const int slot = SERIAL_LIFECYCLE ? 0 : (epoch & 1);
            const int phase = SERIAL_LIFECYCLE
                ? (epoch & 1)
                : ((epoch >> 1) & 1);

            if constexpr (!SERIAL_LIFECYCLE) {
                if (epoch >= 2 && issue_warp) {
                    wait(pv_src_done[slot], phase ^ 1);
                    wait(pv_done[slot], phase ^ 1);
                    tensor_after_thread_sync();
                }
            }
            // A repeated V payload load overwrites SMEM and needs source-done;
            // a repeated V-scale load overwrites its single TMEM page and must
            // wait for full completion of the preceding command.
            if constexpr (LOAD_V_EACH_EPOCH || LOAD_V_SCALE_EACH_EPOCH) {
                if (epoch >= 1 && issue_warp) {
                    const int previous = epoch - 1;
                    const int previous_slot = previous & 1;
                    const int previous_phase = (previous >> 1) & 1;
                    if constexpr (LOAD_V_EACH_EPOCH) {
                        wait(pv_src_done[previous_slot], previous_phase);
                    }
                    if constexpr (LOAD_V_SCALE_EACH_EPOCH) {
                        wait(pv_done[previous_slot], previous_phase);
                    }
                    tensor_after_thread_sync();
                }
            }
            warpgroup::sync(1);

            if constexpr (LOAD_P_EACH_EPOCH) {
                cluster_p_load_payload_wg(
                    p_smem[slot],
                    g.p_fp4 + static_cast<int64_t>(epoch) * 128 * 32);
            }
            if constexpr (LOAD_P_SCALE_EACH_EPOCH) {
                if (warpgroup::warpid() == 0) {
                    cluster_p_load_scale_warp(
                        p_scale_smem[slot],
                        g.p_scale_words +
                            static_cast<int64_t>(epoch) * 32 * 4);
                }
            }
            if constexpr (LOAD_V_EACH_EPOCH) {
                cluster_p_load_payload_wg(
                    v_smem, local_p_stage_v_source<PERTURBED>(g, epoch));
            }
            if constexpr (LOAD_V_SCALE_EACH_EPOCH) {
                if (warpgroup::warpid() == 0) {
                    cluster_p_load_scale_warp(
                        v_scale_smem,
                        local_p_stage_v_scale_source<PERTURBED>(g, epoch));
                }
            }
            warpgroup::sync(1);
            if constexpr (PUBLISH_EACH_EPOCH) {
                fp4pv_publish_shared_backing_proxy_only();
            }
            warpgroup::sync(1);
            if (issue_warp) {
                tensor_after_thread_sync();
            }

            // D performs this transfer once, after its epoch-zero global load.
            // E and later P-moving rungs perform and complete it every epoch.
            if (issue_lane) {
                if constexpr (LOAD_P_SCALE_EACH_EPOCH) {
                    if (STAGE_P_SCALE_EACH_EPOCH || epoch == 0) {
                        const int scale_slot =
                            STAGE_P_SCALE_EACH_EPOCH ? slot : 0;
                        fp4pv_load_mxnv_scale_async_ranklocal(
                            p_scale[scale_slot], p_scale_smem[slot]);
                        detail::tcgen05::commit<1>(
                            p_scale_done[scale_slot]);
                    }
                }
                if constexpr (LOAD_V_SCALE_EACH_EPOCH) {
                    fp4pv_load_mxnv_scale_async_ranklocal(
                        v_scale, v_scale_smem);
                    detail::tcgen05::commit<1>(v_scale_done);
                }
            }
            if (issue_warp) {
                if constexpr (LOAD_P_SCALE_EACH_EPOCH) {
                    if (STAGE_P_SCALE_EACH_EPOCH || epoch == 0) {
                        const int scale_slot =
                            STAGE_P_SCALE_EACH_EPOCH ? slot : 0;
                        const int scale_phase =
                            STAGE_P_SCALE_EACH_EPOCH ? phase : 0;
                        wait(p_scale_done[scale_slot], scale_phase);
                        tensor_after_thread_sync();
                    }
                }
                if constexpr (LOAD_V_SCALE_EACH_EPOCH) {
                    wait(v_scale_done, epoch & 1);
                    tensor_after_thread_sync();
                }
                tensor_before_thread_sync();
            }

            if (issue_lane) {
                const int scale_slot =
                    STAGE_P_SCALE_EACH_EPOCH ? slot : 0;
                const int p_operand_slot = LOAD_P_EACH_EPOCH ? slot : 0;
                if (epoch == 0) {
                    fp4pv_mm_p_stage_ranklocal_ABt<C>(
                        output, p_smem[p_operand_slot], v_smem,
                        p_scale[scale_slot], v_scale, pv_src_done[slot]);
                } else {
                    fp4pv_mma_p_stage_ranklocal_ABt<C>(
                        output, p_smem[p_operand_slot], v_smem,
                        p_scale[scale_slot], v_scale, pv_src_done[slot]);
                }
                tensor_commit<1>(pv_done[slot]);
            }
            if constexpr (SERIAL_LIFECYCLE) {
                if (issue_warp) {
                    wait(pv_src_done[slot], phase);
                    wait(pv_done[slot], phase);
                    tensor_after_thread_sync();
                }
            }
        }

        if (issue_warp) {
            if constexpr (!SERIAL_LIFECYCLE) {
                for (int epoch = LOCAL_P_STAGE_EPOCHS - 2;
                     epoch < LOCAL_P_STAGE_EPOCHS;
                     ++epoch) {
                    const int slot = epoch & 1;
                    const int phase = (epoch >> 1) & 1;
                    wait(pv_src_done[slot], phase);
                    wait(pv_done[slot], phase);
                    tensor_after_thread_sync();
                }
            }
            if (issue_lane) {
                arrive(all_pv_done);
            }
        }
    }

    if (group == 1) {
        wait(all_pv_done, 0);
        tensor_after_thread_sync();
        if constexpr (FULL_OUTPUT_STORE) {
            cluster_p_store_output(
                output,
                g.output + static_cast<int64_t>(blockIdx.x) * 128 * 128);
            warpgroup::sync(2);
        } else {
            local_p_stage_store_sink(
                output,
                g.output + static_cast<int64_t>(blockIdx.x) * 128 * 128);
        }
    }
    __syncthreads();
}

// Literal H-derived paired ablations.  This is intentionally separate from
// the A-G template: the unchanged H kernel remains the correctness anchor,
// while these two specializations add exactly one per-epoch V obligation to
// H's two-P-slot/source-done/full-done/final-drain sequence.
template <bool ADD_V, bool ADD_VSCALE, bool PERTURBED>
__launch_bounds__(config_mxfp4_cluster_p_fake::NUM_THREADS, 1)
__global__ void kernel_mxfp4_local_p_h_ablation(
    const __grid_constant__ globals_local_p_stage_ladder g) {
    using C = config_mxfp4_cluster_p_fake;
    using G = globals_local_p_stage_ladder;
    using score_tt = tt<float, 128, 64>;
    using output_tt = tt<float, 128, 128>;
    using scale_tt = full_tt_fp8e8m0<16>;

    static_assert(ADD_V != ADD_VSCALE,
                  "paired H ablations add V or V-scale, not both");
    static_assert(C::NUM_THREADS == 256 && C::TOTAL_WGS == 2);
    static_assert(LOCAL_P_STAGE_EPOCHS == 32);
    static_assert(
        G::dynamic_shared_memory_bytes() ==
        globals_mxfp4_cluster_p_fake::dynamic_shared_memory_bytes());
    constexpr int LOGICAL_END = 240;
    static_assert(LOGICAL_END <= tensor_allocator<2, 1>::cols);

    tensor_allocator<2, 1> tm_alloc{};
    score_tt score = tm_alloc.template allocate<score_tt>(0);
    output_tt output = tm_alloc.template allocate<output_tt>(64);
    scale_tt p_scale[2] = {
        tm_alloc.template allocate<scale_tt>(192),
        tm_alloc.template allocate<scale_tt>(208)};
    scale_tt v_scale = tm_alloc.template allocate<scale_tt>(224);
    (void)score;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::p_tile (&p_smem)[2] =
        allocator.allocate<typename G::p_tile, 2>();
    typename G::p_scale_tile (&p_scale_smem)[2] =
        allocator.allocate<typename G::p_scale_tile, 2>();
    typename G::v_tile &v_smem = allocator.allocate<typename G::v_tile>();
    typename G::v_scale_tile &v_scale_smem =
        allocator.allocate<typename G::v_scale_tile>();
    __shared__ semaphore p_scale_done[2];
    __shared__ semaphore v_scale_done;
    __shared__ semaphore pv_src_done[2];
    __shared__ semaphore pv_done[2];
    __shared__ semaphore all_pv_done;

    const int group = warpgroup::groupid();
    const bool issue_warp = group == 0 && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            init_semaphore(p_scale_done[slot], 0, 1);
            init_semaphore(pv_src_done[slot], 0, 1);
            init_semaphore(pv_done[slot], 0, 1);
        }
        init_semaphore(v_scale_done, 0, 1);
        init_semaphore(all_pv_done, 0, 1);
    }

    if (group == 1) {
        if constexpr (!ADD_V) {
            cluster_p_load_payload_wg(v_smem, g.v_fp4_hot);
        }
        if constexpr (!ADD_VSCALE) {
            if (warpgroup::warpid() == 0) {
                cluster_p_load_scale_warp(
                    v_scale_smem, g.v_scale_words_hot);
            }
        }
        warpgroup::sync(2);
        fp4pv_publish_shared_backing_proxy_only();
        warpgroup::sync(2);
    }
    __syncthreads();
    if (issue_warp) {
        tensor_after_thread_sync();
    }
    if constexpr (!ADD_VSCALE) {
        if (issue_lane) {
            fp4pv_load_mxnv_scale_async_ranklocal(v_scale, v_scale_smem);
            detail::tcgen05::commit<1>(v_scale_done);
        }
        if (issue_warp) {
            wait(v_scale_done, 0);
            tensor_after_thread_sync();
        }
    }

    if (group == 0) {
        #pragma unroll 1
        for (int epoch = 0; epoch < LOCAL_P_STAGE_EPOCHS; ++epoch) {
            const int slot = epoch & 1;
            const int phase = (epoch >> 1) & 1;
            if (epoch >= 2 && issue_warp) {
                wait(pv_src_done[slot], phase ^ 1);
                wait(pv_done[slot], phase ^ 1);
                tensor_after_thread_sync();
            }
            if (epoch >= 1 && issue_warp) {
                const int previous = epoch - 1;
                const int previous_slot = previous & 1;
                const int previous_phase = (previous >> 1) & 1;
                if constexpr (ADD_V) {
                    wait(pv_src_done[previous_slot], previous_phase);
                }
                if constexpr (ADD_VSCALE) {
                    wait(pv_done[previous_slot], previous_phase);
                }
                tensor_after_thread_sync();
            }
            warpgroup::sync(1);
            cluster_p_load_payload_wg(
                p_smem[slot],
                g.p_fp4 + static_cast<int64_t>(epoch) * 128 * 32);
            if (warpgroup::warpid() == 0) {
                cluster_p_load_scale_warp(
                    p_scale_smem[slot],
                    g.p_scale_words + static_cast<int64_t>(epoch) * 32 * 4);
            }
            if constexpr (ADD_V) {
                cluster_p_load_payload_wg(
                    v_smem, local_p_stage_v_source<PERTURBED>(g, epoch));
            }
            if constexpr (ADD_VSCALE) {
                if (warpgroup::warpid() == 0) {
                    cluster_p_load_scale_warp(
                        v_scale_smem,
                        local_p_stage_v_scale_source<PERTURBED>(g, epoch));
                }
            }
            warpgroup::sync(1);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(1);
            if (issue_warp) {
                tensor_after_thread_sync();
            }
            if (issue_lane) {
                fp4pv_load_mxnv_scale_async_ranklocal(
                    p_scale[slot], p_scale_smem[slot]);
                detail::tcgen05::commit<1>(p_scale_done[slot]);
                if constexpr (ADD_VSCALE) {
                    fp4pv_load_mxnv_scale_async_ranklocal(
                        v_scale, v_scale_smem);
                    detail::tcgen05::commit<1>(v_scale_done);
                }
            }
            if (issue_warp) {
                wait(p_scale_done[slot], phase);
                tensor_after_thread_sync();
                if constexpr (ADD_VSCALE) {
                    wait(v_scale_done, epoch & 1);
                    tensor_after_thread_sync();
                }
                tensor_before_thread_sync();
            }
            if (issue_lane) {
                if (epoch == 0) {
                    fp4pv_mm_p_stage_ranklocal_ABt<C>(
                        output, p_smem[slot], v_smem,
                        p_scale[slot], v_scale, pv_src_done[slot]);
                } else {
                    fp4pv_mma_p_stage_ranklocal_ABt<C>(
                        output, p_smem[slot], v_smem,
                        p_scale[slot], v_scale, pv_src_done[slot]);
                }
                tensor_commit<1>(pv_done[slot]);
            }
        }
        if (issue_warp) {
            const int drain_begin = LOCAL_P_STAGE_EPOCHS - 2;
            for (int epoch = drain_begin;
                 epoch < LOCAL_P_STAGE_EPOCHS;
                 ++epoch) {
                const int slot = epoch & 1;
                const int phase = (epoch >> 1) & 1;
                wait(pv_src_done[slot], phase);
                wait(pv_done[slot], phase);
                tensor_after_thread_sync();
            }
            if (issue_lane) {
                arrive(all_pv_done);
            }
        }
    }

    if (group == 1) {
        wait(all_pv_done, 0);
        tensor_after_thread_sync();
        cluster_p_store_output(
            output,
            g.output + static_cast<int64_t>(blockIdx.x) * 128 * 128);
        warpgroup::sync(2);
    }
    __syncthreads();
}

// Evidence-selected F optimizations.  Both retain F's exact global bytes,
// P/P-scale lifecycle, 32 scaled-K64 commands, 240 TMEM columns, and full
// output store.  The single-slot candidate moves V production to WG1 so it
// overlaps same-epoch P preparation.  PIPE2 adds a second V SMEM slot so WG1
// can also overlap the next V load with the current tensor-core command.
template <bool PIPE2>
static constexpr int
local_p_stage_v_move_candidate_dynamic_shared_memory_bytes() {
    using G = globals_local_p_stage_ladder;
    int bytes = 0;
    auto add = [&](int alloc_bytes) constexpr {
        bytes = align_up_constexpr(bytes, 1024) + alloc_bytes;
    };
    add(2 * (int)sizeof(typename G::p_tile));
    add(2 * (int)sizeof(typename G::p_scale_tile));
    add((PIPE2 ? 2 : 1) * (int)sizeof(typename G::v_tile));
    add((int)sizeof(typename G::v_scale_tile));
    return align_up_constexpr(bytes, 1024) + 1024;
}

template <bool PIPE2, bool PERTURBED>
__launch_bounds__(config_mxfp4_cluster_p_fake::NUM_THREADS, 1)
__global__ void kernel_mxfp4_local_p_v_move_candidate(
    const __grid_constant__ globals_local_p_stage_ladder g) {
    using C = config_mxfp4_cluster_p_fake;
    using G = globals_local_p_stage_ladder;
    using score_tt = tt<float, 128, 64>;
    using output_tt = tt<float, 128, 128>;
    using scale_tt = full_tt_fp8e8m0<16>;

    static_assert(C::NUM_THREADS == 256 && C::TOTAL_WGS == 2);
    static_assert(LOCAL_P_STAGE_EPOCHS == 32);
    static_assert(
        local_p_stage_v_move_candidate_dynamic_shared_memory_bytes<false>() ==
        G::dynamic_shared_memory_bytes());
    constexpr int V_SLOTS = PIPE2 ? 2 : 1;
    constexpr int LOGICAL_END = 240;
    static_assert(LOGICAL_END <= tensor_allocator<2, 1>::cols);

    tensor_allocator<2, 1> tm_alloc{};
    score_tt score = tm_alloc.template allocate<score_tt>(0);
    output_tt output = tm_alloc.template allocate<output_tt>(64);
    scale_tt p_scale[2] = {
        tm_alloc.template allocate<scale_tt>(192),
        tm_alloc.template allocate<scale_tt>(208)};
    scale_tt v_scale = tm_alloc.template allocate<scale_tt>(224);
    (void)score;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int *)&__shm[0]);
    typename G::p_tile (&p_smem)[2] =
        allocator.allocate<typename G::p_tile, 2>();
    typename G::p_scale_tile (&p_scale_smem)[2] =
        allocator.allocate<typename G::p_scale_tile, 2>();
    typename G::v_tile (&v_smem)[V_SLOTS] =
        allocator.allocate<typename G::v_tile, V_SLOTS>();
    typename G::v_scale_tile &v_scale_smem =
        allocator.allocate<typename G::v_scale_tile>();

    __shared__ semaphore p_scale_done[2];
    __shared__ semaphore v_scale_done;
    __shared__ semaphore v_ready[V_SLOTS];
    __shared__ semaphore pv_src_done[2];
    __shared__ semaphore pv_done[2];
    __shared__ semaphore all_pv_done;

    const int group = warpgroup::groupid();
    const bool issue_warp = group == 0 && warpgroup::warpid() == 0;
    const bool issue_lane = issue_warp && warp::elect_leader();
    const bool producer_warp = group == 1 && warpgroup::warpid() == 0;
    const bool producer_lane = producer_warp && warp::elect_leader();

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            init_semaphore(p_scale_done[slot], 0, 1);
            init_semaphore(pv_src_done[slot], 0, 1);
            init_semaphore(pv_done[slot], 0, 1);
        }
        #pragma unroll
        for (int slot = 0; slot < V_SLOTS; ++slot) {
            init_semaphore(v_ready[slot], 0, 1);
        }
        init_semaphore(v_scale_done, 0, 1);
        init_semaphore(all_pv_done, 0, 1);
    }

    // V scale is resident exactly as in F; only the V payload is produced per
    // epoch by WG1.  This setup is outside the measured 32-epoch movement.
    if (group == 1) {
        if (warpgroup::warpid() == 0) {
            cluster_p_load_scale_warp(v_scale_smem, g.v_scale_words_hot);
        }
        warpgroup::sync(2);
        fp4pv_publish_shared_backing_proxy_only();
        warpgroup::sync(2);
    }
    __syncthreads();
    if (issue_warp) {
        tensor_after_thread_sync();
    }
    if (issue_lane) {
        fp4pv_load_mxnv_scale_async_ranklocal(v_scale, v_scale_smem);
        detail::tcgen05::commit<1>(v_scale_done);
    }
    if (issue_warp) {
        wait(v_scale_done, 0);
        tensor_after_thread_sync();
    }

    if (group == 1) {
        #pragma unroll 1
        for (int epoch = 0; epoch < LOCAL_P_STAGE_EPOCHS; ++epoch) {
            const int v_slot = PIPE2 ? (epoch & 1) : 0;
            constexpr int V_REUSE_DISTANCE = PIPE2 ? 2 : 1;
            if (epoch >= V_REUSE_DISTANCE && producer_warp) {
                const int previous = epoch - V_REUSE_DISTANCE;
                const int previous_slot = previous & 1;
                const int previous_phase = (previous >> 1) & 1;
                wait(pv_src_done[previous_slot], previous_phase);
                tensor_after_thread_sync();
            }
            warpgroup::sync(2);
            cluster_p_load_payload_wg(
                v_smem[v_slot],
                local_p_stage_v_source<PERTURBED>(g, epoch));
            warpgroup::sync(2);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(2);
            if (producer_lane) {
                arrive(v_ready[v_slot]);
            }
        }
    } else {
        #pragma unroll 1
        for (int epoch = 0; epoch < LOCAL_P_STAGE_EPOCHS; ++epoch) {
            const int slot = epoch & 1;
            const int phase = (epoch >> 1) & 1;
            const int v_slot = PIPE2 ? slot : 0;
            const int v_phase = PIPE2 ? phase : (epoch & 1);

            if (epoch >= 2 && issue_warp) {
                wait(pv_src_done[slot], phase ^ 1);
                wait(pv_done[slot], phase ^ 1);
                tensor_after_thread_sync();
            }
            warpgroup::sync(1);
            cluster_p_load_payload_wg(
                p_smem[slot],
                g.p_fp4 + static_cast<int64_t>(epoch) * 128 * 32);
            if (warpgroup::warpid() == 0) {
                cluster_p_load_scale_warp(
                    p_scale_smem[slot],
                    g.p_scale_words +
                        static_cast<int64_t>(epoch) * 32 * 4);
            }
            warpgroup::sync(1);
            fp4pv_publish_shared_backing_proxy_only();
            warpgroup::sync(1);
            if (issue_warp) {
                wait(v_ready[v_slot], v_phase);
                tensor_after_thread_sync();
            }

            if (issue_lane) {
                fp4pv_load_mxnv_scale_async_ranklocal(
                    p_scale[slot], p_scale_smem[slot]);
                detail::tcgen05::commit<1>(p_scale_done[slot]);
            }
            if (issue_warp) {
                wait(p_scale_done[slot], phase);
                tensor_after_thread_sync();
                tensor_before_thread_sync();
            }
            if (issue_lane) {
                if (epoch == 0) {
                    fp4pv_mm_p_stage_ranklocal_ABt<C>(
                        output, p_smem[slot], v_smem[v_slot],
                        p_scale[slot], v_scale, pv_src_done[slot]);
                } else {
                    fp4pv_mma_p_stage_ranklocal_ABt<C>(
                        output, p_smem[slot], v_smem[v_slot],
                        p_scale[slot], v_scale, pv_src_done[slot]);
                }
                tensor_commit<1>(pv_done[slot]);
            }
        }

        if (issue_warp) {
            for (int epoch = LOCAL_P_STAGE_EPOCHS - 2;
                 epoch < LOCAL_P_STAGE_EPOCHS;
                 ++epoch) {
                const int slot = epoch & 1;
                const int phase = (epoch >> 1) & 1;
                wait(pv_src_done[slot], phase);
                wait(pv_done[slot], phase);
                tensor_after_thread_sync();
            }
            if (issue_lane) {
                arrive(all_pv_done);
            }
        }
    }

    if (group == 1) {
        wait(all_pv_done, 0);
        tensor_after_thread_sync();
        cluster_p_store_output(
            output,
            g.output + static_cast<int64_t>(blockIdx.x) * 128 * 128);
        warpgroup::sync(2);
    }
    __syncthreads();
}

static void check_local_p_stage_ladder_inputs(
    const at::Tensor &p_fp4,
    const at::Tensor &p_scale_words,
    const at::Tensor &v_fp4_hot,
    const at::Tensor &v_scale_words_hot,
    const at::Tensor &v_fp4_perturbed,
    const at::Tensor &v_scale_words_perturbed,
    const at::Tensor &output,
    int v_bank) {
    for (const at::Tensor *tensor : {
             &p_fp4, &p_scale_words, &v_fp4_hot,
             &v_scale_words_hot, &v_fp4_perturbed,
             &v_scale_words_perturbed, &output}) {
        TORCH_CHECK(tensor->device().is_cuda() && tensor->is_contiguous(),
                    "local-P ladder tensors must be contiguous CUDA tensors");
    }
    TORCH_CHECK(
        p_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
            p_fp4.sizes() == at::IntArrayRef(
                {LOCAL_P_STAGE_EPOCHS, 128, 32}),
        "P must be float4_e2m1fn_x2 (32,128,32)");
    TORCH_CHECK(
        p_scale_words.dtype() == at::ScalarType::Int &&
            p_scale_words.sizes() == at::IntArrayRef(
                {LOCAL_P_STAGE_EPOCHS, 32, 4}),
        "P scales must be int32 (32,32,4)");
    TORCH_CHECK(
        v_fp4_hot.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
            v_fp4_hot.sizes() == at::IntArrayRef({128, 32}),
        "hot V must be float4_e2m1fn_x2 (128,32)");
    TORCH_CHECK(
        v_scale_words_hot.dtype() == at::ScalarType::Int &&
            v_scale_words_hot.sizes() == at::IntArrayRef({32, 4}),
        "hot V scales must be int32 (32,4)");
    TORCH_CHECK(
        output.dtype() == at::ScalarType::Float && output.dim() == 3 &&
            output.size(1) == 128 && output.size(2) == 128,
        "output must be float32 (blocks,128,128)");
    const int64_t blocks = output.size(0);
    TORCH_CHECK(blocks == 256 || blocks == 304,
                "local-P ladder requires exactly 256 or 304 blocks");
    TORCH_CHECK(
        v_fp4_perturbed.dtype() == at::ScalarType::Float4_e2m1fn_x2 &&
            v_fp4_perturbed.dim() == 5 &&
            v_fp4_perturbed.size(1) == blocks &&
            v_fp4_perturbed.size(2) == LOCAL_P_STAGE_EPOCHS &&
            v_fp4_perturbed.size(3) == 128 &&
            v_fp4_perturbed.size(4) == 32,
        "perturbed V must be float4_e2m1fn_x2 "
        "(banks,blocks,32,128,32)");
    TORCH_CHECK(
        v_scale_words_perturbed.dtype() == at::ScalarType::Int &&
            v_scale_words_perturbed.dim() == 5 &&
            v_scale_words_perturbed.size(0) ==
                v_fp4_perturbed.size(0) &&
            v_scale_words_perturbed.size(1) == blocks &&
            v_scale_words_perturbed.size(2) == LOCAL_P_STAGE_EPOCHS &&
            v_scale_words_perturbed.size(3) == 32 &&
            v_scale_words_perturbed.size(4) == 4,
        "perturbed V scales must be int32 (banks,blocks,32,32,4)");
    TORCH_CHECK(v_fp4_perturbed.size(0) >= 6,
                "cache-perturbed backing requires at least six banks");
    TORCH_CHECK(v_bank >= 0 && v_bank < v_fp4_perturbed.size(0),
                "V backing bank index is out of range");
}

template <int MODE, bool PERTURBED>
static void launch_local_p_stage_ladder(
    at::Tensor p_fp4,
    at::Tensor p_scale_words,
    at::Tensor v_fp4_hot,
    at::Tensor v_scale_words_hot,
    at::Tensor v_fp4_perturbed,
    at::Tensor v_scale_words_perturbed,
    at::Tensor output,
    int v_bank) {
    check_local_p_stage_ladder_inputs(
        p_fp4, p_scale_words, v_fp4_hot, v_scale_words_hot,
        v_fp4_perturbed, v_scale_words_perturbed, output, v_bank);

    globals_local_p_stage_ladder g{
        reinterpret_cast<const uint8_t *>(p_fp4.data_ptr()),
        p_scale_words.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t *>(v_fp4_hot.data_ptr()),
        v_scale_words_hot.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t *>(v_fp4_perturbed.data_ptr()),
        v_scale_words_perturbed.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        v_bank,
        static_cast<int>(output.size(0)),
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_local_p_stage_ladder<MODE, PERTURBED>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), 1);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_mxfp4_local_p_stage_ladder<MODE, PERTURBED>,
        g));
}

template <bool ADD_V, bool ADD_VSCALE, bool PERTURBED>
static void launch_local_p_h_ablation(
    at::Tensor p_fp4,
    at::Tensor p_scale_words,
    at::Tensor v_fp4_hot,
    at::Tensor v_scale_words_hot,
    at::Tensor v_fp4_perturbed,
    at::Tensor v_scale_words_perturbed,
    at::Tensor output,
    int v_bank) {
    check_local_p_stage_ladder_inputs(
        p_fp4, p_scale_words, v_fp4_hot, v_scale_words_hot,
        v_fp4_perturbed, v_scale_words_perturbed, output, v_bank);

    globals_local_p_stage_ladder g{
        reinterpret_cast<const uint8_t *>(p_fp4.data_ptr()),
        p_scale_words.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t *>(v_fp4_hot.data_ptr()),
        v_scale_words_hot.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t *>(v_fp4_perturbed.data_ptr()),
        v_scale_words_perturbed.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        v_bank,
        static_cast<int>(output.size(0)),
    };
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_local_p_h_ablation<ADD_V, ADD_VSCALE, PERTURBED>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), g.dynamic_shared_memory(),
        at::cuda::getCurrentCUDAStream(), 1);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_mxfp4_local_p_h_ablation<ADD_V, ADD_VSCALE, PERTURBED>,
        g));
}

template <bool PIPE2, bool PERTURBED>
static void launch_local_p_v_move_candidate(
    at::Tensor p_fp4,
    at::Tensor p_scale_words,
    at::Tensor v_fp4_hot,
    at::Tensor v_scale_words_hot,
    at::Tensor v_fp4_perturbed,
    at::Tensor v_scale_words_perturbed,
    at::Tensor output,
    int v_bank) {
    check_local_p_stage_ladder_inputs(
        p_fp4, p_scale_words, v_fp4_hot, v_scale_words_hot,
        v_fp4_perturbed, v_scale_words_perturbed, output, v_bank);

    globals_local_p_stage_ladder g{
        reinterpret_cast<const uint8_t *>(p_fp4.data_ptr()),
        p_scale_words.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t *>(v_fp4_hot.data_ptr()),
        v_scale_words_hot.data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t *>(v_fp4_perturbed.data_ptr()),
        v_scale_words_perturbed.data_ptr<int32_t>(),
        output.data_ptr<float>(),
        v_bank,
        static_cast<int>(output.size(0)),
    };
    constexpr int DYNAMIC_SMEM =
        local_p_stage_v_move_candidate_dynamic_shared_memory_bytes<PIPE2>();
    CUDACHECK(cudaFuncSetAttribute(
        kernel_mxfp4_local_p_v_move_candidate<PIPE2, PERTURBED>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        DYNAMIC_SMEM));
    LaunchConfig<true, false> launch_config(
        g.grid(), g.block(), DYNAMIC_SMEM,
        at::cuda::getCurrentCUDAStream(), 1);
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_mxfp4_local_p_v_move_candidate<PIPE2, PERTURBED>,
        g));
}

#define LOCAL_P_STAGE_COMMON_ARGS                                             \
    at::Tensor p_fp4, at::Tensor p_scale_words, at::Tensor v_fp4_hot,       \
    at::Tensor v_scale_words_hot, at::Tensor v_fp4_perturbed,               \
    at::Tensor v_scale_words_perturbed, at::Tensor output, int v_bank

#define DEFINE_LOCAL_P_STAGE_DISPATCH(NAME, MODE, PERTURBED)                \
    void NAME(LOCAL_P_STAGE_COMMON_ARGS) {                                   \
        launch_local_p_stage_ladder<MODE, PERTURBED>(                        \
            p_fp4, p_scale_words, v_fp4_hot, v_scale_words_hot,              \
            v_fp4_perturbed, v_scale_words_perturbed, output, v_bank);       \
    }

DEFINE_LOCAL_P_STAGE_DISPATCH(
    dispatch_mxfp4_local_p_stage_a, LOCAL_P_STAGE_A, false)
DEFINE_LOCAL_P_STAGE_DISPATCH(
    dispatch_mxfp4_local_p_stage_b, LOCAL_P_STAGE_B, false)
DEFINE_LOCAL_P_STAGE_DISPATCH(
    dispatch_mxfp4_local_p_stage_c, LOCAL_P_STAGE_C, false)
DEFINE_LOCAL_P_STAGE_DISPATCH(
    dispatch_mxfp4_local_p_stage_d, LOCAL_P_STAGE_D, false)
DEFINE_LOCAL_P_STAGE_DISPATCH(
    dispatch_mxfp4_local_p_stage_e, LOCAL_P_STAGE_E, false)
DEFINE_LOCAL_P_STAGE_DISPATCH(
    dispatch_mxfp4_local_p_stage_f_hot, LOCAL_P_STAGE_F, false)
DEFINE_LOCAL_P_STAGE_DISPATCH(
    dispatch_mxfp4_local_p_stage_f_perturbed, LOCAL_P_STAGE_F, true)
DEFINE_LOCAL_P_STAGE_DISPATCH(
    dispatch_mxfp4_local_p_stage_g_hot, LOCAL_P_STAGE_G, false)
DEFINE_LOCAL_P_STAGE_DISPATCH(
    dispatch_mxfp4_local_p_stage_g_perturbed, LOCAL_P_STAGE_G, true)
DEFINE_LOCAL_P_STAGE_DISPATCH(
    dispatch_mxfp4_local_p_stage_b_c_lifecycle_resident,
    LOCAL_P_STAGE_B_C_LIFECYCLE_RESIDENT,
    false)
DEFINE_LOCAL_P_STAGE_DISPATCH(
    dispatch_mxfp4_local_p_stage_b_c_lifecycle_publish,
    LOCAL_P_STAGE_B_C_LIFECYCLE_PUBLISH,
    false)
DEFINE_LOCAL_P_STAGE_DISPATCH(
    dispatch_mxfp4_local_p_stage_c_serial, LOCAL_P_STAGE_C_SERIAL, false)

#define DEFINE_LOCAL_P_H_ABLATION_DISPATCH(                                  \
    NAME, ADD_V, ADD_VSCALE, PERTURBED)                                      \
    void NAME(LOCAL_P_STAGE_COMMON_ARGS) {                                   \
        launch_local_p_h_ablation<ADD_V, ADD_VSCALE, PERTURBED>(             \
            p_fp4, p_scale_words, v_fp4_hot, v_scale_words_hot,              \
            v_fp4_perturbed, v_scale_words_perturbed, output, v_bank);       \
    }

DEFINE_LOCAL_P_H_ABLATION_DISPATCH(
    dispatch_mxfp4_local_p_stage_h_plus_v_hot, true, false, false)
DEFINE_LOCAL_P_H_ABLATION_DISPATCH(
    dispatch_mxfp4_local_p_stage_h_plus_v_perturbed, true, false, true)
DEFINE_LOCAL_P_H_ABLATION_DISPATCH(
    dispatch_mxfp4_local_p_stage_h_plus_vscale_hot, false, true, false)
DEFINE_LOCAL_P_H_ABLATION_DISPATCH(
    dispatch_mxfp4_local_p_stage_h_plus_vscale_perturbed,
    false,
    true,
    true)

#define DEFINE_LOCAL_P_V_MOVE_CANDIDATE_DISPATCH(NAME, PIPE2, PERTURBED)     \
    void NAME(LOCAL_P_STAGE_COMMON_ARGS) {                                   \
        launch_local_p_v_move_candidate<PIPE2, PERTURBED>(                   \
            p_fp4, p_scale_words, v_fp4_hot, v_scale_words_hot,              \
            v_fp4_perturbed, v_scale_words_perturbed, output, v_bank);       \
    }

DEFINE_LOCAL_P_V_MOVE_CANDIDATE_DISPATCH(
    dispatch_mxfp4_local_p_stage_f_split_single_hot, false, false)
DEFINE_LOCAL_P_V_MOVE_CANDIDATE_DISPATCH(
    dispatch_mxfp4_local_p_stage_f_split_single_perturbed, false, true)
DEFINE_LOCAL_P_V_MOVE_CANDIDATE_DISPATCH(
    dispatch_mxfp4_local_p_stage_f_split_pipe2_hot, true, false)
DEFINE_LOCAL_P_V_MOVE_CANDIDATE_DISPATCH(
    dispatch_mxfp4_local_p_stage_f_split_pipe2_perturbed, true, true)

#undef DEFINE_LOCAL_P_V_MOVE_CANDIDATE_DISPATCH
#undef DEFINE_LOCAL_P_H_ABLATION_DISPATCH
#undef DEFINE_LOCAL_P_STAGE_DISPATCH
#undef LOCAL_P_STAGE_COMMON_ARGS

#endif
