#pragma once

#include <cstdio>
#include <cstdlib>

#include "b300_bwd_cute.cuh"
#include "b300_bwd_cute16.cuh"
#include "b300_bwd_cute16_kernel_candidate.cuh"

namespace tkfa4::bwd_cute16_candidate_cute {

namespace detail {

inline bool cute16_native_timing_enabled() {
    static const bool enabled = [] {
        const char *value = std::getenv("TK_FA4_CUTE16_NATIVE_TIMING");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

}  // namespace detail

template <typename C>
inline void launch_causal_backward(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale
) {
    static_assert(C::ClusterSize == 2, "CuTe-style private candidate wrapper expects ClusterSize=2");

    using CuTeConfig = bwd_cute::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;
    at::Tensor dqacc0 = at::zeros(
        {q.size(0), q.size(2), q.size(1) * CuTeConfig::ClusterSize, 64},
        lse_log2.options()
    );
    at::Tensor dqacc1 = at::zeros_like(dqacc0);
    at::Tensor dqacc2 = at::zeros_like(dqacc0);
    at::Tensor dq_semaphore;

    bwd_cute::launch_backward<CuTeConfig>(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq,
        dk,
        dv,
        dqacc0,
        dqacc1,
        dqacc2,
        dq_semaphore,
        true,
        scale,
        false
    );
}

template <typename C>
inline void launch_causal_backward_split_hybrid(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &dout,
    at::Tensor &lse_log2,
    at::Tensor &dpsum,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale
) {
    static_assert(C::ClusterSize == 2, "CuTe-style private split hybrid expects ClusterSize=2");

    using CuTeConfig = bwd_cute::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 2>;
    using Cluster1Config = tkfa4::bwd_cute16_kernel_candidate::config<C::Mb, C::Nb, C::Dqk, C::Dvo, 1>;

    at::Tensor dqacc0 = at::zeros(
        {q.size(0), q.size(2), q.size(1) * CuTeConfig::ClusterSize, 64},
        lse_log2.options()
    );
    at::Tensor dqacc1 = at::zeros_like(dqacc0);
    at::Tensor dqacc2 = at::zeros_like(dqacc0);
    at::Tensor dq_semaphore;
    at::Tensor dk_tmp = at::empty_like(dk);
    at::Tensor dv_tmp = at::empty_like(dv);

    bwd_cute::launch_backward<CuTeConfig>(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq,
        dk_tmp,
        dv_tmp,
        dqacc0,
        dqacc1,
        dqacc2,
        dq_semaphore,
        true,
        scale,
        false
    );
    tkfa4::bwd_cute16_kernel_candidate::launch_backward_dkdv_only<Cluster1Config>(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dk,
        dv,
        scale
    );
}

template <typename C>
inline void launch_causal_backward_cute16_native_exact(
    at::Tensor &q,
    at::Tensor &k,
    at::Tensor &v,
    at::Tensor &out,
    at::Tensor &lse,
    at::Tensor &dout,
    at::Tensor &dq,
    at::Tensor &dk,
    at::Tensor &dv,
    float scale
) {
    static_assert(C::ClusterSize == 2, "CuTe16-native private exact wrapper expects ClusterSize=2");

    using PreprocessConfig = tkfa4::bwd_cute16::preprocess_config<kB300VDim>;
    at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor dqacc_dummy = at::empty({1}, lse.options());

    cudaEvent_t total_start = nullptr;
    cudaEvent_t total_end = nullptr;
    cudaEvent_t preprocess_start = nullptr;
    cudaEvent_t preprocess_end = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_end = nullptr;
    const bool timing_enabled = detail::cute16_native_timing_enabled();
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (timing_enabled) {
        CUDACHECK(cudaEventCreate(&total_start));
        CUDACHECK(cudaEventCreate(&total_end));
        CUDACHECK(cudaEventCreate(&preprocess_start));
        CUDACHECK(cudaEventCreate(&preprocess_end));
        CUDACHECK(cudaEventCreate(&kernel_start));
        CUDACHECK(cudaEventCreate(&kernel_end));
        CUDACHECK(cudaEventRecord(total_start, stream));
        CUDACHECK(cudaEventRecord(preprocess_start, stream));
    }

    tkfa4::bwd_cute16::launch_preprocess<PreprocessConfig>(
        out,
        dout,
        lse,
        dpsum,
        lse_log2,
        dq
    );

    if (timing_enabled) {
        CUDACHECK(cudaEventRecord(preprocess_end, stream));
        CUDACHECK(cudaEventRecord(kernel_start, stream));
    }

    tkfa4::bwd_cute16_kernel::launch_backward<C>(
        q,
        k,
        v,
        dout,
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq,
        dk,
        dv,
        dqacc_dummy,
        dqacc_dummy,
        dqacc_dummy,
        nullptr,
        true,
        scale,
        false
    );

    if (timing_enabled) {
        float total_ms = 0.0f;
        float preprocess_ms = 0.0f;
        float kernel_ms = 0.0f;
        CUDACHECK(cudaEventRecord(kernel_end, stream));
        CUDACHECK(cudaEventRecord(total_end, stream));
        CUDACHECK(cudaEventSynchronize(total_end));
        CUDACHECK(cudaEventElapsedTime(&total_ms, total_start, total_end));
        CUDACHECK(cudaEventElapsedTime(&preprocess_ms, preprocess_start, preprocess_end));
        CUDACHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_end));
        std::fprintf(
            stderr,
            "cute16_native_timing_us preprocess=%.2f kernel=%.2f total=%.2f\n",
            preprocess_ms * 1000.0f,
            kernel_ms * 1000.0f,
            total_ms * 1000.0f
        );
        cudaEventDestroy(total_start);
        cudaEventDestroy(total_end);
        cudaEventDestroy(preprocess_start);
        cudaEventDestroy(preprocess_end);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_end);
    }
}

}  // namespace tkfa4::bwd_cute16_candidate_cute
