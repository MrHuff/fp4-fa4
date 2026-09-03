#include <c10/cuda/CUDAGuard.h>

#include "b300_bwd.cuh"
#include "b300_bwd_cute16_candidate.cuh"
#include "b300_bwd_cute16_candidate2.cuh"
#include "b300_bwd_cute16.cuh"
#include "b300_bwd_fa4.cuh"
#include "b300_bwd_fa4_postprocess.cuh"
#include "b300_bwd_fa4_preprocess.cuh"
#include "b300_bwd_postprocess.cuh"
#include "b300_bwd_preprocess.cuh"
#include "b300_common.cuh"

namespace {

void check_common_backward_inputs(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &out,
    const at::Tensor &l_aux,
    const at::Tensor &dout,
    float softmax_scale,
    int64_t actual_seq_len
) {
    tkfa4::check_bhsd(q, "q", at::kBFloat16);
    tkfa4::check_bhsd(k, "k", at::kBFloat16);
    tkfa4::check_bhsd(v, "v", at::kBFloat16);
    tkfa4::check_bhsd(out, "out", at::kBFloat16);
    tkfa4::check_bhsd(dout, "dout", at::kBFloat16);
    kittens::py::device_check(q, k, v, out, l_aux, dout);
    tkfa4::check_exact_b300_qkv_bhsd(q, k, v);
    tkfa4::check_exact_b300_l_aux(l_aux, q);

    TORCH_CHECK(tkfa4::is_sm100_device(), "tk_fa4 exact path requires GB200 / SM100");
    TORCH_CHECK(out.size(0) == v.size(0) && out.size(1) == v.size(1) && out.size(2) == v.size(2) && out.size(3) == v.size(3),
                "out must match v");
    TORCH_CHECK(dout.size(0) == v.size(0) && dout.size(1) == v.size(1) && dout.size(2) == v.size(2) && dout.size(3) == v.size(3),
                "dout must match v");
    TORCH_CHECK(q.size(2) % tkfa4::kForwardTileM == 0, "padded seqlen must be divisible by 128");
    TORCH_CHECK(actual_seq_len >= tkfa4::kB300MinSeqLen && actual_seq_len <= q.size(2), "invalid actual_seq_len");
    TORCH_CHECK(softmax_scale > 0.0f, "softmax_scale must be positive");
}

void check_dv_only_backward_inputs(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &dout,
    const at::Tensor &l_aux,
    float softmax_scale,
    int64_t actual_seq_len
) {
    tkfa4::check_bhsd(q, "q", at::kBFloat16);
    tkfa4::check_bhsd(k, "k", at::kBFloat16);
    tkfa4::check_bhsd(dout, "dout", at::kBFloat16);
    kittens::py::device_check(q, k, dout, l_aux);
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == dout.size(0), "batch sizes must match");
    TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == dout.size(1), "head counts must match");
    TORCH_CHECK(q.size(2) == k.size(2) && q.size(2) == dout.size(2), "sequence lengths must match");
    TORCH_CHECK(q.size(3) == tkfa4::kB300QKDim, "q head_dim must be 192");
    TORCH_CHECK(k.size(3) == tkfa4::kB300QKDim, "k head_dim must be 192");
    TORCH_CHECK(dout.size(3) == tkfa4::kB300VDim, "dout head_dim must be 128");
    tkfa4::check_exact_b300_l_aux(l_aux, q);

    TORCH_CHECK(tkfa4::is_sm100_device(), "tk_fa4 exact path requires GB200 / SM100");
    TORCH_CHECK(q.size(2) % tkfa4::kForwardTileM == 0, "padded seqlen must be divisible by 128");
    TORCH_CHECK(actual_seq_len >= tkfa4::kB300MinSeqLen && actual_seq_len <= q.size(2), "invalid actual_seq_len");
    TORCH_CHECK(softmax_scale > 0.0f, "softmax_scale must be positive");
}

void check_common_backward_inputs_fa4_style(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &out,
    const at::Tensor &lse,
    const at::Tensor &dout,
    float softmax_scale,
    int64_t actual_seq_len
) {
    tkfa4::check_bhsd(q, "q", at::kBFloat16);
    tkfa4::check_bhsd(k, "k", at::kBFloat16);
    tkfa4::check_bhsd(v, "v", at::kBFloat16);
    tkfa4::check_bhsd(out, "out", at::kBFloat16);
    tkfa4::check_bhsd(dout, "dout", at::kBFloat16);
    kittens::py::device_check(q, k, v, out, lse, dout);
    tkfa4::check_exact_b300_qkv_bhsd(q, k, v);
    tkfa4::check_exact_b300_lse(lse, q);

    TORCH_CHECK(tkfa4::is_sm100_device(), "tk_fa4 exact path requires GB200 / SM100");
    TORCH_CHECK(out.size(0) == v.size(0) && out.size(1) == v.size(1) && out.size(2) == v.size(2) && out.size(3) == v.size(3),
                "out must match v");
    TORCH_CHECK(dout.size(0) == v.size(0) && dout.size(1) == v.size(1) && dout.size(2) == v.size(2) && dout.size(3) == v.size(3),
                "dout must match v");
    TORCH_CHECK(q.size(2) % (tkfa4::kForwardTileM * 2) == 0, "experimental padded seqlen must be divisible by 256");
    TORCH_CHECK(actual_seq_len >= tkfa4::kB300MinSeqLen && actual_seq_len <= q.size(2), "invalid actual_seq_len");
    TORCH_CHECK(softmax_scale > 0.0f, "softmax_scale must be positive");
}

void check_common_backward_inputs_hot_public(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &out,
    const at::Tensor &lse,
    const at::Tensor &dout,
    float softmax_scale,
    int64_t actual_seq_len,
    int64_t min_seq_len = tkfa4::kB300MinSeqLen
) {
    tkfa4::check_bshd(q, "q", at::kBFloat16);
    tkfa4::check_bshd(k, "k", at::kBFloat16);
    tkfa4::check_bshd(v, "v", at::kBFloat16);
    tkfa4::check_bshd(out, "out", at::kBFloat16);
    tkfa4::check_bshd(dout, "dout", at::kBFloat16);
    kittens::py::device_check(q, k, v, out, lse, dout);
    tkfa4::check_exact_b300_qkv_bshd(q, k, v);
    tkfa4::check_exact_b300_lse_bsh(lse, q);

    TORCH_CHECK(tkfa4::is_sm100_device(), "tk_fa4 exact path requires GB200 / SM100");
    TORCH_CHECK(out.size(0) == v.size(0) && out.size(1) == v.size(1) && out.size(2) == v.size(2) && out.size(3) == v.size(3),
                "out must match v");
    TORCH_CHECK(dout.size(0) == v.size(0) && dout.size(1) == v.size(1) && dout.size(2) == v.size(2) && dout.size(3) == v.size(3),
                "dout must match v");
    TORCH_CHECK(q.size(1) % (tkfa4::kForwardTileM * 2) == 0, "hot padded seqlen must be divisible by 256");
    TORCH_CHECK(actual_seq_len >= min_seq_len && actual_seq_len <= q.size(1), "invalid actual_seq_len");
    TORCH_CHECK(softmax_scale > 0.0f, "softmax_scale must be positive");
}

void check_hot_backward_outputs(
    const at::Tensor &q,
    const at::Tensor &k,
    const at::Tensor &v,
    const at::Tensor &dq,
    const at::Tensor &dk,
    const at::Tensor &dv
) {
    kittens::py::device_check(q, k, v, dq, dk, dv);
    TORCH_CHECK(dq.sizes() == q.sizes(), "dq must match q");
    TORCH_CHECK(dk.sizes() == k.sizes(), "dk must match k");
    TORCH_CHECK(dv.sizes() == v.sizes(), "dv must match v");
    TORCH_CHECK(dq.scalar_type() == at::kFloat, "dq must be float32");
    TORCH_CHECK(dk.scalar_type() == at::kFloat, "dk must be float32");
    TORCH_CHECK(dv.scalar_type() == at::kFloat, "dv must be float32");
    TORCH_CHECK(dq.is_contiguous() && dk.is_contiguous() && dv.is_contiguous(), "output tensors must be contiguous");
}

}  // namespace

std::vector<at::Tensor> launch_b300_mha_bwd_hot_cute16_candidate(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    at::Tensor dq = at::empty({q.size(0), q.size(1), q.size(2), q.size(3)}, lse.options());
    at::Tensor dk = at::empty({k.size(0), k.size(1), k.size(2), k.size(3)}, lse.options());
    at::Tensor dv = at::empty({v.size(0), v.size(1), v.size(2), v.size(3)}, lse.options());
    tkfa4::bwd_cute16_candidate::launch_backward<HotConfig>(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        causal,
        softmax_scale,
        deterministic
    );
    return {dq, dk, dv};
}

std::vector<at::Tensor> launch_b300_mha_bwd_hot_cute16_candidate_bf16_dkdv(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    at::Tensor dq = at::empty({q.size(0), q.size(1), q.size(2), q.size(3)}, lse.options());
    at::Tensor dk = at::empty({k.size(0), k.size(1), k.size(2), k.size(3)}, k.options());
    at::Tensor dv = at::empty({v.size(0), v.size(1), v.size(2), v.size(3)}, v.options());
    tkfa4::bwd_cute16_candidate::launch_backward<HotConfig, kittens::bf16, float>(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        causal,
        softmax_scale,
        deterministic
    );
    return {dq, dk, dv};
}

std::vector<at::Tensor> launch_b300_mha_bwd_hot_cute16_candidate_2cta_dkdv(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    at::Tensor dq = at::empty({q.size(0), q.size(1), q.size(2), q.size(3)}, lse.options());
    at::Tensor dk = at::empty({k.size(0), k.size(1), k.size(2), k.size(3)}, lse.options());
    at::Tensor dv = at::empty({v.size(0), v.size(1), v.size(2), v.size(3)}, lse.options());
    tkfa4::bwd_cute16_candidate::launch_backward_cute_parity_2cta_dkdv<HotConfig>(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        causal,
        softmax_scale,
        deterministic
    );
    return {dq, dk, dv};
}

template <
    bool UseChunkedTmemDq = false,
    bool UsePipelinedTmemDq = false,
    bool DoubleBufferPipelinedInputs = false,
    bool EnqueuePipelinedDqEarly = true,
    int DenseSplitCount = 1,
    int DqReplaySplitCount = 1,
    bool UseTmemScoreDp = false,
    bool UseTmemFrontier = false,
    bool FuseDenseDq = false,
    bool AdaptiveLastQuarter = false,
    bool OverlapLoadAndDqReduce = false,
    bool SkipAdaptiveTailScratch = false,
    bool UseLdsmTransposeDs = false,
    bool DoubleBufferFusedDqTma = false,
    bool MaterializeDkdvBf16 = false,
    bool CacheDkdvFp32Base = false,
    bool ReleaseTmemOperandsEachIteration = false,
    bool SerializePipelinedDqBeforeDkdv = false,
    bool SerializeDenseFrontier = false
>
std::vector<at::Tensor> launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    static_assert(
        !CacheDkdvFp32Base || MaterializeDkdvBf16,
        "cached FP32 dK/dV bases require BF16 materialization"
    );
    const c10::cuda::CUDAGuard device_guard(q.device());
    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    at::Tensor dq = at::empty({q.size(0), q.size(1), q.size(2), q.size(3)}, lse.options());
    at::Tensor dk = at::empty(
        {k.size(0), k.size(1), k.size(2), k.size(3)},
        MaterializeDkdvBf16 ? k.options() : lse.options()
    );
    at::Tensor dv = at::empty(
        {v.size(0), v.size(1), v.size(2), v.size(3)},
        MaterializeDkdvBf16 ? v.options() : lse.options()
    );
    at::Tensor dk_acc = MaterializeDkdvBf16 && !CacheDkdvFp32Base
        ? at::empty({k.size(0), k.size(1), k.size(2), k.size(3)}, lse.options())
        : dk;
    at::Tensor dv_acc = MaterializeDkdvBf16 && !CacheDkdvFp32Base
        ? at::empty({v.size(0), v.size(1), v.size(2), v.size(3)}, lse.options())
        : dv;
    tkfa4::bwd_cute16_candidate::launch_backward_dense_tmem_frontier_dkdv<
        HotConfig,
        float,
        UseChunkedTmemDq,
        UsePipelinedTmemDq,
        DoubleBufferPipelinedInputs,
        EnqueuePipelinedDqEarly,
        DenseSplitCount,
        DqReplaySplitCount,
        UseTmemScoreDp,
        UseTmemFrontier,
        FuseDenseDq,
        AdaptiveLastQuarter,
        OverlapLoadAndDqReduce,
        SkipAdaptiveTailScratch,
        UseLdsmTransposeDs,
        DoubleBufferFusedDqTma,
        MaterializeDkdvBf16,
        CacheDkdvFp32Base,
        ReleaseTmemOperandsEachIteration,
        SerializePipelinedDqBeforeDkdv,
        SerializeDenseFrontier
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk_acc,
        dv_acc,
        causal,
        softmax_scale,
        deterministic,
        MaterializeDkdvBf16 ? &dk : nullptr,
        MaterializeDkdvBf16 ? &dv : nullptr
    );
    return {dq, dk, dv};
}

template <
    bool UseTmemDs = true,
    bool OverlapDsExchange = true,
    bool OverlapQWithDp = true,
    bool UseTmemP = false,
    bool OverlapDoWithDp = false,
    bool UseDpOperandReadyMbar = false,
    bool UseDqOperandReadyMbar = false,
    bool OverlapDvWithDs = false,
    bool PipelineNextScore = false,
    bool PreloadDqA = false,
    bool UseScoreOperandReadyMbar = false,
    bool UseDsWarpMulticastMbar = false,
    bool UseRoleSplit = false,
    bool RetainDsExchange = false,
    bool RetainDsLocal = false,
    bool UseNormalDoDv = false,
    bool UseTmaScoreK = false,
    bool DirectNextQdoDuringDqDrain = false,
    bool SingleOwnerCluster = false,
    bool UseFastExp2 = false,
    bool UseWarpStatsCache = false,
    bool PipelineLsePrefetch = false,
    bool UseDirectStatsLoads = false,
    bool SplitDvDkReady = false,
    bool StageDqAfterDv = false,
    bool StageDqPeerBeforeDv = false,
    bool UseWideDkN192 = false,
    bool DirectDsHalfStore = false,
    bool AsymmetricDvPublish = false
>
std::vector<at::Tensor> launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), lse.options());
    at::Tensor dv = at::empty(v.sizes(), lse.options());
    tkfa4::bwd_cute16_candidate::launch_backward_cta2_fused_dense<
        HotConfig,
        UseTmemDs,
        OverlapDsExchange,
        OverlapQWithDp,
        UseTmemP,
        OverlapDoWithDp,
        UseDpOperandReadyMbar,
        UseDqOperandReadyMbar,
        OverlapDvWithDs,
        PipelineNextScore,
        PreloadDqA,
        UseScoreOperandReadyMbar,
        UseDsWarpMulticastMbar,
        UseRoleSplit,
        RetainDsExchange,
        RetainDsLocal,
        UseNormalDoDv,
        UseTmaScoreK,
        DirectNextQdoDuringDqDrain,
        SingleOwnerCluster,
        UseFastExp2,
        UseWarpStatsCache,
        PipelineLsePrefetch,
        UseDirectStatsLoads,
        SplitDvDkReady,
        StageDqAfterDv,
        StageDqPeerBeforeDv,
        UseWideDkN192,
        DirectDsHalfStore,
        AsymmetricDvPublish
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        causal,
        softmax_scale,
        deterministic
    );
    return {dq, dk, dv};
}

std::vector<at::Tensor> launch_b300_mha_bwd_hot_cute16_candidate2(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    using HotConfig = tkfa4::bwd_cute16_candidate2::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    at::Tensor dq = at::empty({q.size(0), q.size(1), q.size(2), q.size(3)}, lse.options());
    at::Tensor dk = at::empty({k.size(0), k.size(1), k.size(2), k.size(3)}, lse.options());
    at::Tensor dv = at::empty({v.size(0), v.size(1), v.size(2), v.size(3)}, lse.options());
    tkfa4::bwd_cute16_candidate2::launch_backward<HotConfig>(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        causal,
        softmax_scale,
        deterministic
    );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor l_aux,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    using BwdConfig = tkfa4::bwd::config<tkfa4::kForwardTileM, tkfa4::kForwardTileN, tkfa4::kB300QKDim, tkfa4::kB300VDim, 1>;

    check_common_backward_inputs(q, k, v, out, l_aux, dout, softmax_scale, actual_seq_len);

    at::Tensor dq = at::empty({q.size(0), q.size(1), q.size(2), q.size(3)}, l_aux.options());
    at::Tensor dk = at::empty({k.size(0), k.size(1), k.size(2), k.size(3)}, l_aux.options());
    at::Tensor dv = at::empty({v.size(0), v.size(1), v.size(2), v.size(3)}, l_aux.options());
    at::Tensor delta = at::empty({q.size(0), q.size(1), 1, q.size(2)}, l_aux.options());
    at::Tensor dq_accum = at::zeros(
        {q.size(0), q.size(1), q.size(2) / tkfa4::kForwardTileM, 1, tkfa4::kForwardTileM, tkfa4::kB300QKDim},
        l_aux.options()
    );

    tkfa4::bwd::launch_preprocess<tkfa4::bwd::preprocess_config<tkfa4::kB300VDim>>(out, dout, delta);
    tkfa4::bwd::launch_backward<BwdConfig>(
        q,
        k,
        v,
        dout,
        l_aux,
        delta,
        dq,
        dk,
        dv,
        dq_accum,
        causal,
        softmax_scale,
        static_cast<int>(actual_seq_len),
        deterministic
    );
    tkfa4::bwd::postprocess_gradients(dq, dk, dv, deterministic);

    return {dq, dk, dv};
}

at::Tensor b300_mha_bwd_dv_only(
    at::Tensor q,
    at::Tensor k,
    at::Tensor dout,
    at::Tensor l_aux,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    using BwdConfig = tkfa4::bwd::config<tkfa4::kForwardTileM, tkfa4::kForwardTileN, tkfa4::kB300QKDim, tkfa4::kB300VDim, 1>;

    check_dv_only_backward_inputs(q, k, dout, l_aux, softmax_scale, actual_seq_len);

    at::Tensor dv = at::empty({k.size(0), k.size(1), k.size(2), tkfa4::kB300VDim}, l_aux.options());
    tkfa4::bwd::launch_backward_dv_only<BwdConfig>(
        q,
        k,
        dout,
        l_aux,
        dv,
        causal,
        softmax_scale,
        static_cast<int>(actual_seq_len),
        deterministic
    );

    return dv;
}

std::vector<at::Tensor> b300_mha_bwd_fa4_style(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    using BwdConfig = tkfa4::bwd_fa4::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    check_common_backward_inputs_fa4_style(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);

    at::Tensor dq = at::empty({q.size(0), q.size(1), q.size(2), q.size(3)}, lse.options());
    at::Tensor dk = at::empty({k.size(0), k.size(1), k.size(2), k.size(3)}, lse.options());
    at::Tensor dv = at::empty({v.size(0), v.size(1), v.size(2), v.size(3)}, lse.options());
    at::Tensor dpsum = at::empty({q.size(0), q.size(1), 1, q.size(2)}, lse.options());
    at::Tensor lse_log2 = at::empty({q.size(0), q.size(1), 1, q.size(2)}, lse.options());
    at::Tensor dq_accum = at::zeros(
        {q.size(0), q.size(1), q.size(2) / tkfa4::kForwardTileM, 2, tkfa4::kForwardTileM, tkfa4::kB300QKDim},
        lse.options()
    );
    at::Tensor dq_semaphore = at::empty(
        {q.size(0), q.size(1), q.size(2) / tkfa4::kRefTileM, 2},
        q.options().dtype(at::kInt)
    );

    tkfa4::bwd_fa4::launch_preprocess<tkfa4::bwd_fa4::preprocess_config<tkfa4::kB300VDim>>(
        out,
        dout,
        lse,
        dpsum,
        lse_log2,
        dq_accum,
        dq_semaphore
    );
    tkfa4::bwd_fa4::launch_backward<BwdConfig>(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dk,
        dv,
        dq_accum,
        dq_semaphore,
        causal,
        softmax_scale,
        static_cast<int>(actual_seq_len),
        deterministic
    );
    tkfa4::bwd_fa4::launch_postprocess<BwdConfig>(dq_accum, dq, dq_semaphore, deterministic);

    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_fa4_style_ref(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
    ) {
    return b300_mha_bwd_fa4_style(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        actual_seq_len,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
);

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
);

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate2_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
);

std::vector<at::Tensor> b300_mha_bwd_hot_trusted_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
);

std::vector<at::Tensor> b300_mha_bwd_hot_legacy_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len
);

std::vector<at::Tensor> b300_mha_bwd_hot(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(actual_seq_len == q.size(1), "hot backward does not support sequence padding");
    TORCH_CHECK(actual_seq_len % (tkfa4::kForwardTileM * 2) == 0, "hot backward requires seqlen divisible by 256");
    TORCH_CHECK(causal,
                "CuTe16 hot mode not implemented yet; current stage only supports causal=True");
    TORCH_CHECK(!deterministic,
                "CuTe16 hot mode not implemented yet; current stage only supports deterministic=False");

    return launch_b300_mha_bwd_hot_cute16_candidate(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    using HotConfig = tkfa4::bwd_cute16::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(actual_seq_len == q.size(1), "hot backward does not support sequence padding");
    TORCH_CHECK(actual_seq_len % (tkfa4::kForwardTileM * 2) == 0, "hot backward requires seqlen divisible by 256");

    at::Tensor dq = at::empty({q.size(0), q.size(1), q.size(2), q.size(3)}, lse.options());
    at::Tensor dk = at::empty({k.size(0), k.size(1), k.size(2), k.size(3)}, lse.options());
    at::Tensor dv = at::empty({v.size(0), v.size(1), v.size(2), v.size(3)}, lse.options());
    tkfa4::bwd_cute16::launch_backward<HotConfig>(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        causal,
        softmax_scale,
        deterministic
    );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_nopatch_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    using HotConfig = tkfa4::bwd_cute16::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(actual_seq_len == q.size(1), "hot backward does not support sequence padding");
    TORCH_CHECK(actual_seq_len % (tkfa4::kForwardTileM * 2) == 0, "hot backward requires seqlen divisible by 256");

    at::Tensor dq = at::empty({q.size(0), q.size(1), q.size(2), q.size(3)}, lse.options());
    at::Tensor dk = at::empty({k.size(0), k.size(1), k.size(2), k.size(3)}, lse.options());
    at::Tensor dv = at::empty({v.size(0), v.size(1), v.size(2), v.size(3)}, lse.options());
    tkfa4::bwd_cute16::launch_backward<HotConfig>(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        causal,
        softmax_scale,
        deterministic,
        false
    );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(actual_seq_len == q.size(1), "hot backward does not support sequence padding");
    TORCH_CHECK(actual_seq_len % (tkfa4::kForwardTileM * 2) == 0, "hot backward requires seqlen divisible by 256");

    return launch_b300_mha_bwd_hot_cute16_candidate(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_bf16_dkdv_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(actual_seq_len == q.size(1), "hot backward does not support sequence padding");
    TORCH_CHECK(actual_seq_len == 2048, "candidate BF16 DK/DV screen is currently only validated at seqlen 2048");
    TORCH_CHECK(actual_seq_len % (tkfa4::kForwardTileM * 2) == 0, "hot backward requires seqlen divisible by 256");

    return launch_b300_mha_bwd_hot_cute16_candidate_bf16_dkdv(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_2cta_dkdv_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(causal, "2CTA dK/dV candidate currently supports causal=True");
    TORCH_CHECK(!deterministic, "2CTA dK/dV candidate currently supports deterministic=False");
    TORCH_CHECK(actual_seq_len == q.size(1), "2CTA dK/dV candidate does not support sequence padding");
    TORCH_CHECK(actual_seq_len == 2048, "2CTA dK/dV candidate is currently only wired for seqlen 2048");

    return launch_b300_mha_bwd_hot_cute16_candidate_2cta_dkdv(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_2cta_dkdv_out_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(causal, "2CTA dK/dV candidate currently supports causal=True");
    TORCH_CHECK(!deterministic, "2CTA dK/dV candidate currently supports deterministic=False");
    TORCH_CHECK(actual_seq_len == q.size(1), "2CTA dK/dV candidate does not support sequence padding");
    TORCH_CHECK(actual_seq_len == 2048, "2CTA dK/dV candidate is currently only wired for seqlen 2048");
    check_hot_backward_outputs(q, k, v, dq, dk, dv);

    tkfa4::bwd_cute16_candidate::launch_backward_cute_parity_2cta_dkdv<HotConfig>(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        causal,
        softmax_scale,
        deterministic
    );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(causal, "dense TMEM frontier dK/dV candidate currently supports causal=True");
    TORCH_CHECK(!deterministic, "dense TMEM frontier dK/dV candidate currently supports deterministic=False");
    TORCH_CHECK(actual_seq_len == q.size(1), "dense TMEM frontier dK/dV candidate does not support sequence padding");
    TORCH_CHECK(actual_seq_len == 2048, "dense TMEM frontier dK/dV candidate is only wired for seqlen 2048");

    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_chunked_tmem_dq_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(causal, "dense TMEM frontier plus chunked TMEM dQ candidate currently supports causal=True");
    TORCH_CHECK(!deterministic, "dense TMEM frontier plus chunked TMEM dQ candidate currently supports deterministic=False");
    TORCH_CHECK(actual_seq_len == q.size(1), "dense TMEM frontier plus chunked TMEM dQ candidate does not support sequence padding");
    TORCH_CHECK(actual_seq_len == 2048, "dense TMEM frontier plus chunked TMEM dQ candidate is only wired for seqlen 2048");

    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<true>(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(causal, "dense TMEM frontier plus pipelined TMEM dQ candidate currently supports causal=True");
    TORCH_CHECK(!deterministic, "dense TMEM frontier plus pipelined TMEM dQ candidate currently supports deterministic=False");
    TORCH_CHECK(actual_seq_len == q.size(1), "dense TMEM frontier plus pipelined TMEM dQ candidate does not support sequence padding");
    TORCH_CHECK(actual_seq_len == 2048, "dense TMEM frontier plus pipelined TMEM dQ candidate is only wired for seqlen 2048");

    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<false, true>(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_dkdv_first_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(causal, "dK/dV-first double-buffered TMEM dQ candidate currently supports causal=True");
    TORCH_CHECK(!deterministic, "dK/dV-first double-buffered TMEM dQ candidate currently supports deterministic=False");
    TORCH_CHECK(actual_seq_len == q.size(1), "dK/dV-first double-buffered TMEM dQ candidate does not support sequence padding");
    TORCH_CHECK(actual_seq_len == 2048, "dK/dV-first double-buffered TMEM dQ candidate is only wired for seqlen 2048");

    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        false
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        512
    );
    TORCH_CHECK(causal, "dQ-first split-2 candidate currently supports causal=True");
    TORCH_CHECK(!deterministic, "dQ-first split-2 candidate currently supports deterministic=False");
    TORCH_CHECK(actual_seq_len == q.size(1), "dQ-first split-2 candidate does not support sequence padding");
    TORCH_CHECK(
        actual_seq_len >= 512 && actual_seq_len % 512 == 0,
        "dQ-first split-2 candidate requires seqlen >= 512 and divisible by 512"
    );

    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        512
    );
    TORCH_CHECK(causal, "BF16-materialized split-2 candidate requires causal=True");
    TORCH_CHECK(!deterministic, "BF16-materialized split-2 candidate requires deterministic=False");
    TORCH_CHECK(actual_seq_len == q.size(1), "BF16-materialized split-2 candidate does not support padding");
    TORCH_CHECK(
        q.size(0) == 1 &&
            ((actual_seq_len == 1024 && q.size(2) == 8) ||
             (actual_seq_len == 4096 && q.size(2) == 1)),
        "BF16-materialized split-2 candidate is private to B1 S1024/H8 or S4096/H1"
    );

    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2,
        1,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_cached_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        512
    );
    TORCH_CHECK(causal, "cached BF16-materialized split-2 candidate requires causal=True");
    TORCH_CHECK(!deterministic, "cached BF16-materialized split-2 candidate requires deterministic=False");
    TORCH_CHECK(actual_seq_len == q.size(1), "cached BF16-materialized split-2 candidate does not support padding");
    TORCH_CHECK(
        q.size(0) == 1 &&
            ((actual_seq_len == 1024 && q.size(2) == 8) ||
             (actual_seq_len == 4096 && q.size(2) == 1)),
        "cached BF16-materialized split-2 candidate is private to B1 S1024/H8 or S4096/H1"
    );

    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2,
        1,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

template <
    bool MaterializeDkdvBf16,
    bool CacheDkdvFp32Base,
    bool SerializePipelinedDqBeforeDkdv,
    bool SerializeDenseFrontier
>
std::vector<at::Tensor> launch_b300_mha_bwd_hot_cute16_candidate_split2_u19_operand_release(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    static_assert(!CacheDkdvFp32Base || MaterializeDkdvBf16);
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        512
    );
    TORCH_CHECK(causal, "U19 split-2 operand-release candidate requires causal=True");
    TORCH_CHECK(!deterministic, "U19 split-2 operand-release candidate requires deterministic=False");
    TORCH_CHECK(actual_seq_len == q.size(1), "U19 split-2 operand-release candidate does not support padding");

    const int64_t heads = q.size(2);
    const bool supported =
        (actual_seq_len == 512 && (heads == 1 || heads == 2 || heads == 4 || heads == 8)) ||
        (actual_seq_len == 1024 && (heads == 1 || heads == 2 || heads == 4)) ||
        (actual_seq_len == 2048 && (heads == 1 || heads == 2 || heads == 4 || heads == 8)) ||
        (actual_seq_len == 4096 && (heads == 2 || heads == 8)) ||
        (actual_seq_len == 8192 && (heads == 1 || heads == 2 || heads == 4 || heads == 16)) ||
        (actual_seq_len == 16384 && (heads == 4 || heads == 8));
    TORCH_CHECK(
        q.size(0) == 1 && supported,
        "U19 split-2 operand-release candidate only supports the exact private B1 U19 allowlist"
    );
    if (actual_seq_len == 8192 && heads == 1) {
        return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
            false,
            true,
            true,
            true,
            2,
            2,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            MaterializeDkdvBf16,
            CacheDkdvFp32Base,
            true,
            SerializePipelinedDqBeforeDkdv,
            SerializeDenseFrontier
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            causal,
            softmax_scale,
            deterministic
        );
    }

    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2,
        1,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        MaterializeDkdvBf16,
        CacheDkdvFp32Base,
        true,
        SerializePipelinedDqBeforeDkdv,
        SerializeDenseFrontier
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_operand_release_u19_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    return launch_b300_mha_bwd_hot_cute16_candidate_split2_u19_operand_release<
        false,
        false,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        actual_seq_len,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_cached_u19_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    return launch_b300_mha_bwd_hot_cute16_candidate_split2_u19_operand_release<
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        actual_seq_len,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_cached_operand_release_dq_overlap_u19_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    return launch_b300_mha_bwd_hot_cute16_candidate_split2_u19_operand_release<
        true,
        true,
        false,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        actual_seq_len,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_cached_operand_release_full_overlap_u19_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    return launch_b300_mha_bwd_hot_cute16_candidate_split2_u19_operand_release<
        true,
        true,
        false,
        false
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        actual_seq_len,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split3_dq_first_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        512
    );
    TORCH_CHECK(causal, "dQ-first split-3 candidate currently supports causal=True");
    TORCH_CHECK(!deterministic, "dQ-first split-3 candidate currently supports deterministic=False");
    TORCH_CHECK(actual_seq_len == q.size(1), "dQ-first split-3 candidate does not support sequence padding");
    TORCH_CHECK(
        actual_seq_len >= 512 && actual_seq_len % 512 == 0,
        "dQ-first split-3 candidate requires seqlen >= 512 and divisible by 512"
    );

    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        3
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_replay_split2_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "split-2 dQ replay candidate currently supports causal=True");
    TORCH_CHECK(!deterministic, "split-2 dQ replay candidate currently supports deterministic=False");
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "split-2 dQ replay candidate does not support sequence padding"
    );
    TORCH_CHECK(
        actual_seq_len == 8192 && q.size(2) == 1,
        "split-2 dQ replay candidate is restricted to seqlen 8192 and one head"
    );

    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2,
        2
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_adaptive_long_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "adaptive long candidate currently supports causal=True");
    TORCH_CHECK(!deterministic, "adaptive long candidate currently supports deterministic=False");
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "adaptive long candidate does not support sequence padding"
    );
    const int64_t heads = q.size(2);
    const bool supported_shape =
        (actual_seq_len == 4096 && (heads == 4 || heads == 8)) ||
        (actual_seq_len == 8192 && (heads == 1 || heads == 2 || heads == 4));
    TORCH_CHECK(
        supported_shape,
        "adaptive long candidate supports S4096 H4/H8 and S8192 H1/H2/H4"
    );

    if (actual_seq_len == 8192 && heads == 1) {
        return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
            false,
            true,
            true,
            true,
            2,
            2
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            causal,
            softmax_scale,
            deterministic
        );
    }
    if (actual_seq_len == 8192 && heads == 2) {
        return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
            false,
            true,
            true,
            true,
            3
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            causal,
            softmax_scale,
            deterministic
        );
    }
    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

template <bool UseTmemFrontier>
std::vector<at::Tensor> launch_b300_mha_bwd_hot_cute16_candidate_adaptive_long_tmem_score_dp(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "TMEM score/dP candidate supports causal=True only");
    TORCH_CHECK(!deterministic, "TMEM score/dP candidate supports deterministic=False only");
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "TMEM score/dP candidate does not support sequence padding"
    );
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && (heads == 4 || heads == 8)) ||
            (actual_seq_len == 8192 && (heads == 1 || heads == 2 || heads == 4)),
        "TMEM score/dP candidate supports S4096 H4/H8 and S8192 H1/H2/H4"
    );

    if (actual_seq_len == 8192 && heads == 1) {
        return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
            false,
            true,
            true,
            true,
            2,
            2,
            true,
            UseTmemFrontier
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            causal,
            softmax_scale,
            deterministic
        );
    }
    if (actual_seq_len == 8192 && heads == 2) {
        return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
            false,
            true,
            true,
            true,
            3,
            1,
            true,
            UseTmemFrontier
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            causal,
            softmax_scale,
            deterministic
        );
    }
    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2,
        1,
        true,
        UseTmemFrontier
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_adaptive_long_tmem_score_dp_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    return launch_b300_mha_bwd_hot_cute16_candidate_adaptive_long_tmem_score_dp<false>(
        q, k, v, out, lse, dout, causal, softmax_scale, actual_seq_len, deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_adaptive_long_tmem_score_dp_tmem_frontier_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    return launch_b300_mha_bwd_hot_cute16_candidate_adaptive_long_tmem_score_dp<true>(
        q, k, v, out, lse, dout, causal, softmax_scale, actual_seq_len, deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s4096h8_fused_dense_dq_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q, k, v, out, lse, dout, softmax_scale, actual_seq_len, 4096
    );
    TORCH_CHECK(causal, "fused dense dQ candidate supports causal=True only");
    TORCH_CHECK(!deterministic, "fused dense dQ candidate supports deterministic=False only");
    TORCH_CHECK(
        actual_seq_len == 4096 && actual_seq_len == q.size(1) && q.size(2) == 8,
        "fused dense dQ candidate is restricted to S4096 H8"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2,
        1,
        true,
        true,
        true
    >(
        q, k, v, out, lse, dout, causal, softmax_scale, deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s4096h8_fused_dense_dq_tail_quarter_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q, k, v, out, lse, dout, softmax_scale, actual_seq_len, 4096
    );
    TORCH_CHECK(causal, "tail-quarter fused dQ candidate supports causal=True only");
    TORCH_CHECK(!deterministic, "tail-quarter fused dQ candidate supports deterministic=False only");
    TORCH_CHECK(
        actual_seq_len == 4096 && actual_seq_len == q.size(1) && q.size(2) == 8,
        "tail-quarter fused dQ candidate is restricted to S4096 H8"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2,
        1,
        true,
        true,
        true,
        true
    >(
        q, k, v, out, lse, dout, causal, softmax_scale, deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s4096h8_fused_dense_dq_tail_quarter_load_reducer_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q, k, v, out, lse, dout, softmax_scale, actual_seq_len, 4096
    );
    TORCH_CHECK(causal, "load/reducer fused dQ candidate supports causal=True only");
    TORCH_CHECK(!deterministic, "load/reducer fused dQ candidate supports deterministic=False only");
    TORCH_CHECK(
        actual_seq_len == 4096 && actual_seq_len == q.size(1) && q.size(2) == 8,
        "load/reducer fused dQ candidate is restricted to S4096 H8"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2,
        1,
        true,
        true,
        true,
        true,
        true
    >(
        q, k, v, out, lse, dout, causal, softmax_scale, deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_fused_dense_dq_load_reducer_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q, k, v, out, lse, dout, softmax_scale, actual_seq_len, 4096
    );
    TORCH_CHECK(causal, "high-head fused dQ candidate supports causal=True only");
    TORCH_CHECK(!deterministic, "high-head fused dQ candidate supports deterministic=False only");
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "high-head fused dQ candidate does not support sequence padding"
    );
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "high-head fused dQ candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2,
        1,
        true,
        true,
        true,
        true,
        true
    >(
        q, k, v, out, lse, dout, causal, softmax_scale, deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_fused_dense_dq_skip_tail_scratch_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q, k, v, out, lse, dout, softmax_scale, actual_seq_len, 4096
    );
    TORCH_CHECK(causal, "skip-tail-scratch candidate supports causal=True only");
    TORCH_CHECK(!deterministic, "skip-tail-scratch candidate supports deterministic=False only");
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "skip-tail-scratch candidate does not support sequence padding"
    );
    TORCH_CHECK(q.size(0) == 1, "skip-tail-scratch candidate supports batch size 1 only");
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "skip-tail-scratch candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2,
        1,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q, k, v, out, lse, dout, causal, softmax_scale, deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_ldsm_ds_transpose_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q, k, v, out, lse, dout, softmax_scale, actual_seq_len, 4096
    );
    TORCH_CHECK(causal, "high-head LDSM-transpose candidate supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "high-head LDSM-transpose candidate supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "high-head LDSM-transpose candidate does not support sequence padding"
    );
    TORCH_CHECK(q.size(0) == 1, "high-head LDSM-transpose candidate supports batch size 1 only");
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "high-head LDSM-transpose candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2,
        1,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q, k, v, out, lse, dout, causal, softmax_scale, deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_double_buffer_dq_tma_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q, k, v, out, lse, dout, softmax_scale, actual_seq_len, 4096
    );
    TORCH_CHECK(causal, "double-buffered dQ TMA candidate supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "double-buffered dQ TMA candidate supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "double-buffered dQ TMA candidate does not support sequence padding"
    );
    TORCH_CHECK(q.size(0) == 1, "double-buffered dQ TMA candidate supports batch size 1 only");
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "double-buffered dQ TMA candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv<
        false,
        true,
        true,
        true,
        2,
        1,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q, k, v, out, lse, dout, causal, softmax_scale, deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_out_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(causal, "dense TMEM frontier dK/dV candidate currently supports causal=True");
    TORCH_CHECK(!deterministic, "dense TMEM frontier dK/dV candidate currently supports deterministic=False");
    TORCH_CHECK(actual_seq_len == q.size(1), "dense TMEM frontier dK/dV candidate does not support sequence padding");
    TORCH_CHECK(actual_seq_len == 2048, "dense TMEM frontier dK/dV candidate is only wired for seqlen 2048");
    check_hot_backward_outputs(q, k, v, dq, dk, dv);

    tkfa4::bwd_cute16_candidate::launch_backward_dense_tmem_frontier_dkdv<HotConfig>(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        causal,
        softmax_scale,
        deterministic
    );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA fused dense candidate supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA fused dense candidate supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA fused dense candidate does not support sequence padding"
    );
    TORCH_CHECK(q.size(0) == 1, "2-CTA fused dense candidate supports batch size 1 only");
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "2-CTA fused dense candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_role_split_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA role-split candidate supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA role-split candidate supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA role-split candidate does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1,
        "2-CTA role-split candidate supports batch size 1 only"
    );
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            ((actual_seq_len == 8192 || actual_seq_len == 16384) &&
             (heads == 4 || heads == 8 || heads == 16)),
        "2-CTA role-split candidate supports S4096 H8 and "
        "S8192/S16384 H4/H8/H16"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_retained_ds_exchange_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "retained dS exchange supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "retained dS exchange supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "retained dS exchange does not support sequence padding"
    );
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        q.size(0) == 1 &&
            ((actual_seq_len == 8192 &&
              (heads == 4 || heads == 8 || heads == 16)) ||
             (actual_seq_len == 16384 &&
              (heads == 8 || heads == 16))),
        "retained dS exchange supports B1 S8192 H4/H8/H16 and "
        "S16384 H8/H16 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_retained_ds_both_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "retained dS halves support causal=True only");
    TORCH_CHECK(
        !deterministic,
        "retained dS halves support deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "retained dS halves do not support sequence padding"
    );
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        q.size(0) == 1 &&
            ((actual_seq_len == 8192 &&
              (heads == 8 || heads == 16)) ||
             (actual_seq_len == 16384 &&
              (heads == 8 || heads == 16))),
        "retained dS halves support B1 S8192 H8/H16 and "
        "S16384 H8/H16 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_normal_dv_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "normal-dV route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "normal-dV route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "normal-dV route does not support sequence padding"
    );
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        q.size(0) == 1 &&
            ((actual_seq_len == 8192 && (heads == 8 || heads == 16)) ||
             (actual_seq_len == 16384 && (heads == 8 || heads == 16))),
        "normal-dV route supports B1 S8192/S16384 H8/H16 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_tma_score_k_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "TMA score-K route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "TMA score-K route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "TMA score-K route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            (actual_seq_len == 8192 || actual_seq_len == 16384) &&
            (q.size(2) == 8 || q.size(2) == 16),
        "TMA score-K route supports B1 S8192/S16384 H8/H16 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "direct-Q/dO route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "direct-Q/dO route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "direct-Q/dO route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            (actual_seq_len == 8192 || actual_seq_len == 16384) &&
            (q.size(2) == 8 || q.size(2) == 16),
        "direct-Q/dO route supports B1 S8192/S16384 H8/H16 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(causal, "paired early-dQ route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "paired early-dQ route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "paired early-dQ route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "paired early-dQ route supports B1 S16384 H16 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_direct_global_stats_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "paired direct-global-stats route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "paired direct-global-stats route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "paired direct-global-stats route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            ((actual_seq_len == 8192 && q.size(2) == 8) ||
             (actual_seq_len == 16384 && q.size(2) == 16)),
        "paired direct-global-stats route supports B1 S8192 H8 and "
        "S16384 H16 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_direct_global_stats_direct_ds_store_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "direct-dS-store route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "direct-dS-store route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "direct-dS-store route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            ((actual_seq_len == 8192 && q.size(2) == 8) ||
             (actual_seq_len == 16384 && q.size(2) == 16)),
        "direct-dS-store route supports B1 S8192 H8 and S16384 H16 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        false,
        false,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_direct_global_stats_direct_ds_store_fast_exp2_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "paired fast-exp2 direct-dS-store route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "paired fast-exp2 direct-dS-store route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "paired fast-exp2 direct-dS-store route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            ((actual_seq_len == 8192 && q.size(2) == 8) ||
             (actual_seq_len == 16384 && q.size(2) == 16)),
        "paired fast-exp2 direct-dS-store route supports B1 S8192 H8 and "
        "S16384 H16 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        false,
        true,
        false,
        false,
        true,
        true,
        true,
        false,
        false,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_direct_global_stats_direct_ds_store_fast_exp2_asymmetric_dv_publish_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "asymmetric dV publish route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "asymmetric dV publish route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "asymmetric dV publish route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "asymmetric dV publish route supports B1 S16384 H16 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        false,
        true,
        false,
        false,
        true,
        true,
        true,
        false,
        false,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "BF16 dK/dV route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "BF16 dK/dV route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "BF16 dK/dV route does not support sequence padding"
    );
    const bool supported_shape =
        q.size(0) == 1 &&
        ((actual_seq_len == 8192 && q.size(2) == 8) ||
         (actual_seq_len == 16384 && q.size(2) == 16));
    TORCH_CHECK(
        supported_shape,
        "BF16 dK/dV route supports B1 S8192 H8 or S16384 H16"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<HotConfig>(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "vectorized BF16 dK/dV route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "vectorized BF16 dK/dV route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "vectorized BF16 dK/dV route does not support sequence padding"
    );
    const bool supported_shape =
        q.size(0) == 1 &&
        ((actual_seq_len == 8192 && q.size(2) == 8) ||
         (actual_seq_len == 16384 && q.size(2) == 16));
    TORCH_CHECK(
        supported_shape,
        "vectorized BF16 dK/dV route supports B1 S8192 H8 or S16384 H16"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<HotConfig, true>(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_direct_ds_async_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "direct async dS BF16 route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "direct async dS BF16 route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "direct async dS BF16 route does not support sequence padding"
    );
    const bool supported_shape =
        q.size(0) == 1 &&
        ((actual_seq_len == 8192 && q.size(2) == 8) ||
         (actual_seq_len == 16384 && q.size(2) == 16));
    TORCH_CHECK(
        supported_shape,
        "direct async dS BF16 route supports B1 S8192 H8 or S16384 H16"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "producer bulk dS BF16 route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "producer bulk dS BF16 route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "producer bulk dS BF16 route does not support sequence padding"
    );
    const bool supported_shape =
        q.size(0) == 1 &&
        ((actual_seq_len == 8192 && q.size(2) == 8) ||
         (actual_seq_len == 16384 && q.size(2) == 16));
    TORCH_CHECK(
        supported_shape,
        "producer bulk dS BF16 route supports B1 S8192 H8 or S16384 H16"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "CTA-fence bulk dS BF16 route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "CTA-fence bulk dS BF16 route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "CTA-fence bulk dS BF16 route does not support sequence padding"
    );
    const bool supported_shape =
        q.size(0) == 1 &&
        ((actual_seq_len == 8192 && q.size(2) == 8) ||
         (actual_seq_len == 16384 && q.size(2) == 16));
    TORCH_CHECK(
        supported_shape,
        "CTA-fence bulk dS BF16 route supports B1 S8192 H8 or S16384 H16"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "dQ read-handoff BF16 route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "dQ read-handoff BF16 route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "dQ read-handoff BF16 route does not support sequence padding"
    );
    const bool supported_shape =
        q.size(0) == 1 &&
        ((actual_seq_len == 8192 && q.size(2) == 8) ||
         (actual_seq_len == 16384 && q.size(2) == 16));
    TORCH_CHECK(
        supported_shape,
        "dQ read-handoff BF16 route supports B1 S8192 H8 or S16384 H16"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "score-fanin BF16 route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "score-fanin BF16 route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "score-fanin BF16 route does not support sequence padding"
    );
    const bool supported_shape =
        q.size(0) == 1 &&
        ((actual_seq_len == 8192 && q.size(2) == 8) ||
         (actual_seq_len == 16384 && q.size(2) == 16));
    TORCH_CHECK(
        supported_shape,
        "score-fanin BF16 route supports B1 S8192 H8 or S16384 H16"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(causal, "timeout dQ wait route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "timeout dQ wait route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "timeout dQ wait route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "timeout dQ wait route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            false,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(causal, "wide direct-TMA dK route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "wide direct-TMA dK route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "wide direct-TMA dK route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "wide direct-TMA dK route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(causal, "all-timeout route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "all-timeout route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "all-timeout route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "all-timeout route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(causal, "named-dO-barrier route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "named-dO-barrier route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "named-dO-barrier route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "named-dO-barrier route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(causal, "score-fanout route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "score-fanout route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "score-fanout route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "score-fanout route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(causal, "runtime-accumulate route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "runtime-accumulate route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "runtime-accumulate route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "runtime-accumulate route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(causal, "H16 dQ-fanout route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "H16 dQ-fanout route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H16 dQ-fanout route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "H16 dQ-fanout route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(causal, "H16 dQ leader-arrive route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "H16 dQ leader-arrive route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H16 dQ leader-arrive route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "H16 dQ leader-arrive route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(causal, "H16 merged-dP-ready route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "H16 merged-dP-ready route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H16 merged-dP-ready route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "H16 merged-dP-ready route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(causal, "H16 wide-store route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "H16 wide-store route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H16 wide-store route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "H16 wide-store route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(causal, "H16 full-dS-bulk route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "H16 full-dS-bulk route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H16 full-dS-bulk route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "H16 full-dS-bulk route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_coalesced_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "H16 coalesced full-dS-bulk route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "H16 coalesced full-dS-bulk route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H16 coalesced full-dS-bulk route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "H16 coalesced full-dS-bulk route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_coalesced_wide_dq_k_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "H16 wide dQ-K load route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "H16 wide dQ-K load route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H16 wide dQ-K load route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "H16 wide dQ-K load route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_direct_tma_dk_q_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "direct-TMA-dK-Q BF16 route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "direct-TMA-dK-Q BF16 route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "direct-TMA-dK-Q BF16 route does not support sequence padding"
    );
    const bool supported_shape =
        q.size(0) == 1 &&
        actual_seq_len == 8192 &&
        q.size(2) == 8;
    TORCH_CHECK(
        supported_shape,
        "direct-TMA-dK-Q BF16 route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_coalesced_wide_dq_k_integrated_frontier_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "H16 integrated-frontier route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "H16 integrated-frontier route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H16 integrated-frontier route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "H16 integrated-frontier route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_coalesced_wide_dq_k_integrated_frontier_exact_h16_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 integrated-frontier route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 integrated-frontier route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 integrated-frontier route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 integrated-frontier route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_exact_ds_publish_fence_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 dS-publish-fence route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 dS-publish-fence route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 dS-publish-fence route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 dS-publish-fence route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_exact_named_dkdv_fanin_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 named dK/dV fan-in route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 named dK/dV fan-in route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 named dK/dV fan-in route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 named dK/dV fan-in route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_exact_leader_qdo_publish_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 leader-Q/dO-publish route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 leader-Q/dO-publish route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 leader-Q/dO-publish route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 leader-Q/dO-publish route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_exact_cached_qdo_ready_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 cached-qdo-ready route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 cached-qdo-ready route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 cached-qdo-ready route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 cached-qdo-ready route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_exact_grouped_qdo_tma_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 grouped-Q/dO-TMA route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 grouped-Q/dO-TMA route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 grouped-Q/dO-TMA route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 grouped-Q/dO-TMA route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_exact_grouped_qdo_elected_wide_dkq_tma_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 elected-wide-dK-Q-TMA route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 elected-wide-dK-Q-TMA route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 elected-wide-dK-Q-TMA route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 elected-wide-dK-Q-TMA route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_exact_grouped_qdo_elected_wide_dkq_elected_peer_do_tma_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 elected-peer-dO-TMA route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 elected-peer-dO-TMA route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 elected-peer-dO-TMA route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 elected-peer-dO-TMA route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_exact_cached_role_cluster_addresses_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 cached-role-address route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 cached-role-address route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 cached-role-address route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 cached-role-address route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_exact_cached_tensor_commit_addresses_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 cached-commit-address route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 cached-commit-address route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 cached-commit-address route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 cached-commit-address route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_exact_reducer_output_drain_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 reducer-output-drain route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 reducer-output-drain route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 reducer-output-drain route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 reducer-output-drain route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_exact_elected_score_k_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 elected-score-K route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 elected-score-K route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 elected-score-K route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 elected-score-K route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_exact_cluster_coordinates_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 cluster-coordinate route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 cluster-coordinate route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 cluster-coordinate route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 cluster-coordinate route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_split_dp_consumer_release_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 split dP-consumer route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 split dP-consumer route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 split dP-consumer route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 split dP-consumer route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_iteration_causal_mask_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 iteration-causal route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 iteration-causal route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 iteration-causal route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 iteration-causal route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 fused-TMEM P/dS route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 fused-TMEM P/dS route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 fused-TMEM P/dS route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 fused-TMEM P/dS route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_early_dq_a_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 early dQ-A publication route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 early dQ-A publication route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 early dQ-A publication route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 early dQ-A publication route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_dkdv_qdo_prefetch_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 post-dK Q/dO prefetch route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 post-dK Q/dO prefetch route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 post-dK Q/dO prefetch route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 post-dK Q/dO prefetch route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 owner-boundary Q/dO prefetch route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 owner-boundary Q/dO prefetch route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 owner-boundary Q/dO prefetch route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 owner-boundary Q/dO prefetch route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 TMEM-A runtime-accumulate route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 TMEM-A runtime-accumulate route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 TMEM-A runtime-accumulate route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 TMEM-A runtime-accumulate route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true   // UseFusedTmemRuntimeAccumulationPredicate
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 bitwise P-expansion route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 bitwise P-expansion route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 bitwise P-expansion route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 bitwise P-expansion route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true   // UseBitwisePExpansion
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 fused exp2-pack route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 fused exp2-pack route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 fused exp2-pack route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 fused exp2-pack route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true   // UseFusedExp2Pack
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_peeled_causal_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 peeled-causal route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 peeled-causal route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 peeled-causal route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 peeled-causal route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true,  // UseFusedExp2Pack
            true   // PeelCausalPrefix
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h16_branchless_do_source_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact H16 branchless-dO-source route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact H16 branchless-dO-source route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact H16 branchless-dO-source route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 16,
        "exact H16 branchless-dO-source route supports B1 S16384 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true,  // UseFusedExp2Pack
            true,  // PeelCausalPrefix
            true   // BranchlessDoSourceLoad
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

template <int ExpectedHeads>
std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s16384_branchless_do_base_select_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact S16384 branchless-dO-base route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S16384 branchless-dO-base route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S16384 branchless-dO-base route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384
            && q.size(2) == ExpectedHeads,
        "exact S16384 branchless-dO-base route supports B1 S16384 H",
        ExpectedHeads,
        " only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true,  // UseFusedExp2Pack
            true,  // PeelCausalPrefix
            true,  // BranchlessDoSourceLoad
            true,  // BranchlessDoSourceBaseSelect
            false  // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

template <int ExpectedHeads>
std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s16384h4h8_branchless_do_base_select_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact S16384 low-head branchless-dO-base route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S16384 low-head branchless-dO-base route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S16384 low-head branchless-dO-base route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384
            && q.size(2) == ExpectedHeads,
        "exact S16384 low-head branchless-dO-base route supports B1 S16384 H",
        ExpectedHeads,
        " only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true,  // UseFusedExp2Pack
            true,  // PeelCausalPrefix
            true,  // BranchlessDoSourceLoad
            true,  // BranchlessDoSourceBaseSelect
            false  // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s16384h32_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact S16384 H32 fused exp2-pack route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S16384 H32 fused exp2-pack route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S16384 H32 fused exp2-pack route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 32,
        "exact S16384 H32 fused exp2-pack route supports B1 S16384 H32 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true   // UseFusedExp2Pack
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s16384h64_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact S16384 H64 fused exp2-pack route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S16384 H64 fused exp2-pack route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S16384 H64 fused exp2-pack route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 64,
        "exact S16384 H64 fused exp2-pack route supports B1 S16384 H64 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true   // UseFusedExp2Pack
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s16384h128_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        16384
    );
    TORCH_CHECK(
        causal,
        "exact S16384 H128 fused exp2-pack route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S16384 H128 fused exp2-pack route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S16384 H128 fused exp2-pack route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 16384 && q.size(2) == 128,
        "exact S16384 H128 fused exp2-pack route supports B1 S16384 H128 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            128,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true   // UseFusedExp2Pack
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s32768h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        32768
    );
    TORCH_CHECK(
        causal,
        "exact S32768 H16 fused exp2-pack route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S32768 H16 fused exp2-pack route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S32768 H16 fused exp2-pack route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 32768 && q.size(2) == 16,
        "exact S32768 H16 fused exp2-pack route supports B1 S32768 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            256,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true   // UseFusedExp2Pack
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s32768h32_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        32768
    );
    TORCH_CHECK(
        causal,
        "exact S32768 H32 fused exp2-pack route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S32768 H32 fused exp2-pack route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S32768 H32 fused exp2-pack route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 32768 && q.size(2) == 32,
        "exact S32768 H32 fused exp2-pack route supports B1 S32768 H32 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            256,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true   // UseFusedExp2Pack
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s32768h64_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        32768
    );
    TORCH_CHECK(
        causal,
        "exact S32768 H64 fused exp2-pack route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S32768 H64 fused exp2-pack route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S32768 H64 fused exp2-pack route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 32768 && q.size(2) == 64,
        "exact S32768 H64 fused exp2-pack route supports B1 S32768 H64 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            256,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true   // UseFusedExp2Pack
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s32768h128_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        32768
    );
    TORCH_CHECK(
        causal,
        "exact S32768 H128 fused exp2-pack route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S32768 H128 fused exp2-pack route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S32768 H128 fused exp2-pack route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 32768 && q.size(2) == 128,
        "exact S32768 H128 fused exp2-pack route supports B1 S32768 H128 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            256,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true   // UseFusedExp2Pack
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s32768h128_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_peeled_causal_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        32768
    );
    TORCH_CHECK(
        causal,
        "exact S32768 H128 peeled-causal route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S32768 H128 peeled-causal route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S32768 H128 peeled-causal route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 32768 && q.size(2) == 128,
        "exact S32768 H128 peeled-causal route supports B1 S32768 H128 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            256,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true,  // UseFusedExp2Pack
            true   // PeelCausalPrefix
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

template <int ExpectedHeads>
std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s32768_branchless_do_base_select_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        32768
    );
    TORCH_CHECK(
        causal,
        "exact S32768 branchless-dO-base route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S32768 branchless-dO-base route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S32768 branchless-dO-base route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 32768
            && q.size(2) == ExpectedHeads,
        "exact S32768 branchless-dO-base route supports B1 S32768 H",
        ExpectedHeads,
        " only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            256,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true,  // UseFusedExp2Pack
            true,  // PeelCausalPrefix
            true,  // BranchlessDoSourceLoad
            true,  // BranchlessDoSourceBaseSelect
            false  // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s65536h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        65536
    );
    TORCH_CHECK(
        causal,
        "exact S65536 H16 fused exp2-pack route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S65536 H16 fused exp2-pack route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S65536 H16 fused exp2-pack route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 65536 && q.size(2) == 16,
        "exact S65536 H16 fused exp2-pack route supports B1 S65536 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            512,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true   // UseFusedExp2Pack
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s65536h32_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        65536
    );
    TORCH_CHECK(
        causal,
        "exact S65536 H32 fused exp2-pack route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S65536 H32 fused exp2-pack route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S65536 H32 fused exp2-pack route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 65536 && q.size(2) == 32,
        "exact S65536 H32 fused exp2-pack route supports B1 S65536 H32 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            512,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true   // UseFusedExp2Pack
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

template <int ExpectedHeads>
std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s65536_branchless_do_base_select_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        65536
    );
    TORCH_CHECK(
        causal,
        "exact S65536 branchless-dO-base route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S65536 branchless-dO-base route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S65536 branchless-dO-base route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 65536
            && q.size(2) == ExpectedHeads,
        "exact S65536 branchless-dO-base route supports B1 S65536 H",
        ExpectedHeads,
        " only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            512,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true,  // UseFusedExp2Pack
            false, // PeelCausalPrefix
            true,  // BranchlessDoSourceLoad
            true,  // BranchlessDoSourceBaseSelect
            false  // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s65536h64_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        65536
    );
    TORCH_CHECK(
        causal,
        "exact S65536 H64 fused exp2-pack route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S65536 H64 fused exp2-pack route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S65536 H64 fused exp2-pack route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 65536 && q.size(2) == 64,
        "exact S65536 H64 fused exp2-pack route supports B1 S65536 H64 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            512,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true   // UseFusedExp2Pack
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s65536h64_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_peeled_causal_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        65536
    );
    TORCH_CHECK(
        causal,
        "exact S65536 H64 peeled-causal route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S65536 H64 peeled-causal route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S65536 H64 peeled-causal route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 65536 && q.size(2) == 64,
        "exact S65536 H64 peeled-causal route supports B1 S65536 H64 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            512,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true,  // UseFusedExp2Pack
            true   // PeelCausalPrefix
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s65536h64_branchless_do_source_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        65536
    );
    TORCH_CHECK(
        causal,
        "exact S65536 H64 branchless-dO-source route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S65536 H64 branchless-dO-source route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S65536 H64 branchless-dO-source route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 65536 && q.size(2) == 64,
        "exact S65536 H64 branchless-dO-source route supports B1 S65536 H64 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            512,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true,  // UseFusedExp2Pack
            true,  // PeelCausalPrefix
            true   // BranchlessDoSourceLoad
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s65536h64_branchless_do_base_select_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        65536
    );
    TORCH_CHECK(
        causal,
        "exact S65536 H64 branchless-dO-base route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S65536 H64 branchless-dO-base route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S65536 H64 branchless-dO-base route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 65536 && q.size(2) == 64,
        "exact S65536 H64 branchless-dO-base route supports B1 S65536 H64 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            512,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true,  // UseFusedExp2Pack
            true,  // PeelCausalPrefix
            true,  // BranchlessDoSourceLoad
            true,  // BranchlessDoSourceBaseSelect
            false  // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s65536h64_bulk_do_dv_stage_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        65536
    );
    TORCH_CHECK(
        causal,
        "exact S65536 H64 bulk-dO-dV route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S65536 H64 bulk-dO-dV route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S65536 H64 bulk-dO-dV route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 65536 && q.size(2) == 64,
        "exact S65536 H64 bulk-dO-dV route supports B1 S65536 H64 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            512,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true,  // UseFusedExp2Pack
            true,  // PeelCausalPrefix
            true,  // BranchlessDoSourceLoad
            true,  // BranchlessDoSourceBaseSelect
            false, // PublishVOncePerOwner
            true   // BulkDoDvStage
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

template <
    bool FuseScoreScaleLse = false,
    bool RetainPackedP = false,
    bool SplitDirectDpsumAcrossDpDoneWait = false,
    bool FusedExp2Fragment4First = false,
    bool CarryDirectStatsOffset = false,
    bool AllBlockingReducerDqJoin = false,
    bool CarryAllRolePhases = false,
    bool UseExactDefaultScaleLog2e = false,
    bool ReverseDkTailTmemLoadIssue = false,
    bool PrearmNextQdoBeforeDkDone = false,
    bool UseX32TmemComputeLayout = false,
    bool UseLongSeqStatsCache = false,
    bool UseCompactScoreMma = false,
    bool UseCompactDpMma = false,
    bool UsePackedBf16DsProduct = false,
    bool SplitDqTmemAndSharedHandoff = false,
    bool DistributedDqSharedReadWait = false,
    bool BalancedSingleOwnerSchedule = false,
    bool UseSingleOwnerWarpStatsCache = false,
    bool CacheDqStageLanePointers = false,
    bool UseSlicedFp32PForDs = false,
    bool UseTmaVWithScoreK = false,
    bool UseStatsWarpScoreFanout = false,
    bool UseBatchedDqTmemLoads = false,
    bool UseDynamicDpReleaseBarrierId = false,
    bool PreissueFirstDpHalfBeforeQdoWait = false,
    bool OverlapSecondDpLoadWithReleaseBarrier = false,
    bool RelayDoDvCompletionViaExchangeWarp = false,
    bool OverlapDqPeerCopyWithDoDvCompletion = false,
    bool OverlapLocalDqStoreWithPeerCopy = false,
    bool UseNonblockingDqPublicationFollowers = false,
    bool SplitDqAliasLifetimeWithCuteTmemMap = false,
    bool DeferFirstDsTmemStoreWait = false,
    bool OverlapFinalDsTmemStoreWithPeerSharedStores = false,
    bool DelayScoreAliasReleaseUntilFirstDqTailLoad = false,
    bool InterleaveSteadyScoreExpPairs = false,
    bool ShiftOverlappingScoreHalfBeforeDpRelease = false,
    bool BuildCompactDpDescriptorsAfterWait = false,
    int LateTensorCommitAddressSharedMask = 0,
    bool CacheCompactDpDescriptorsInShared = false,
    bool OverlapFirstDpsumQuarterWithSecondPStore = false,
    bool HoistReducerDpReadyBeforeScoreWait = false,
    bool PipelineFirstDpQuarterLoads = false,
    bool PublishNextQdoAtDqAliasRelease = false,
    bool JoinNextQdoWithDqAliasRelease = false,
    bool PrecomputePostScoreFanoutAddresses = false,
    bool PrecomputeScoreIterationDeltaUnderFanout = false,
    int ExactQTileCount = 512,
    int ExpectedSeqLen = 65536,
    bool AllowQualifiedHeadSet = false
>
std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        ExpectedSeqLen
    );
    TORCH_CHECK(
        causal,
        "exact long-sequence loader-owned dK-Q route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact long-sequence loader-owned dK-Q route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact long-sequence loader-owned dK-Q route does not support sequence padding"
    );
    static_assert(
        (ExpectedSeqLen == 8192 && ExactQTileCount == 64) ||
        (ExpectedSeqLen == 16384 && ExactQTileCount == 128) ||
        (ExpectedSeqLen == 32768 && ExactQTileCount == 256) ||
        (ExpectedSeqLen == 65536 && ExactQTileCount == 512)
    );
    const bool supported_heads = !AllowQualifiedHeadSet
        ? q.size(2) == 64
        : ((ExpectedSeqLen == 8192 &&
            (q.size(2) == 8 || q.size(2) == 16)) ||
           (ExpectedSeqLen == 16384 &&
            (q.size(2) == 4 || q.size(2) == 8 || q.size(2) == 16 ||
             q.size(2) == 32 || q.size(2) == 64 || q.size(2) == 128)) ||
           ((ExpectedSeqLen == 32768 || ExpectedSeqLen == 65536) &&
            (q.size(2) == 16 || q.size(2) == 32 || q.size(2) == 64 ||
             q.size(2) == 128)));
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == ExpectedSeqLen &&
            supported_heads,
        "exact long-sequence loader-owned dK-Q route received an unsupported shape"
    );
    if constexpr (UseExactDefaultScaleLog2e) {
        TORCH_CHECK(
            softmax_scale == 0x1.279a74p-4f,
            "exact-default scale-log2e route requires softmax_scale=",
            0x1.279a74p-4f
        );
    }

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            ExactQTileCount,
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true,  // UseFusedExp2Pack
            true,  // PeelCausalPrefix
            true,  // BranchlessDoSourceLoad
            true,  // BranchlessDoSourceBaseSelect
            false, // PublishVOncePerOwner
            true,  // BulkDoDvStage
            true,  // LoaderOwnedDkQ
            FuseScoreScaleLse,
            RetainPackedP,
            SplitDirectDpsumAcrossDpDoneWait,
            FusedExp2Fragment4First,
            CarryDirectStatsOffset,
            !AllBlockingReducerDqJoin,
            CarryAllRolePhases,
            UseExactDefaultScaleLog2e,
            ReverseDkTailTmemLoadIssue,
            PrearmNextQdoBeforeDkDone,
            UseX32TmemComputeLayout,
            UseLongSeqStatsCache,
            UseCompactScoreMma,
            UseCompactDpMma,
            UsePackedBf16DsProduct,
            SplitDqTmemAndSharedHandoff,
            DistributedDqSharedReadWait,
            BalancedSingleOwnerSchedule,
            UseSingleOwnerWarpStatsCache,
            CacheDqStageLanePointers,
            UseSlicedFp32PForDs,
            UseTmaVWithScoreK,
            UseStatsWarpScoreFanout,
            UseBatchedDqTmemLoads,
            UseDynamicDpReleaseBarrierId,
            PreissueFirstDpHalfBeforeQdoWait,
            OverlapSecondDpLoadWithReleaseBarrier,
            RelayDoDvCompletionViaExchangeWarp,
            OverlapDqPeerCopyWithDoDvCompletion,
            OverlapLocalDqStoreWithPeerCopy,
            UseNonblockingDqPublicationFollowers,
            SplitDqAliasLifetimeWithCuteTmemMap,
            DeferFirstDsTmemStoreWait,
            OverlapFinalDsTmemStoreWithPeerSharedStores,
            DelayScoreAliasReleaseUntilFirstDqTailLoad,
            InterleaveSteadyScoreExpPairs,
            ShiftOverlappingScoreHalfBeforeDpRelease,
            BuildCompactDpDescriptorsAfterWait,
            LateTensorCommitAddressSharedMask,
            CacheCompactDpDescriptorsInShared,
            OverlapFirstDpsumQuarterWithSecondPStore,
            HoistReducerDpReadyBeforeScoreWait,
            PipelineFirstDpQuarterLoads,
            PublishNextQdoAtDqAliasRelease,
            JoinNextQdoWithDqAliasRelease,
            PrecomputePostScoreFanoutAddresses,
            PrecomputeScoreIterationDeltaUnderFanout
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
    );
    return {dq, dk, dv};
}

template <int ExactQTileCount, int ExpectedSeqLen>
std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_v382_advanced_long_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    return b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<
        true, true, true, true, true, false, true, true, true, true,
        false, false, true, false, true, true, true, true, true, true,
        true, true, true, true, true, true, true, true, true, true,
        true, true, true, true, false, false, true, false, 0, false,
        true, true, true, true, true, true, true,
        ExactQTileCount, ExpectedSeqLen, true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        actual_seq_len,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s65536h128_branchless_do_base_select_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        65536
    );
    TORCH_CHECK(
        causal,
        "exact S65536 H128 branchless-dO-base route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "exact S65536 H128 branchless-dO-base route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "exact S65536 H128 branchless-dO-base route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 65536 && q.size(2) == 128,
        "exact S65536 H128 branchless-dO-base route supports B1 S65536 H128 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_integrated_frontier<
            HotConfig,
            512,   // ExactQTileCount
            true,  // FenceDsBeforeDkdvReady
            true,  // UseNamedDkdvLocalFanIn
            true,  // LeaderOnlyQdoPublishFence
            true,  // CacheQdoReadyClusterAddress
            true,  // GroupQdoTmaLoads
            true,  // ElectedWideDkQTmaLoad
            true,  // ElectedPeerDoTmaLoad
            true,  // CacheRoleClusterAddresses
            true,  // CacheTensorCommitAddresses
            true,  // EnsureReducerOutputDrain
            true,  // ElectedScoreKTmaLoad
            true,  // UseExactClusterCoordinates
            true,  // EnforceDpTmemConsumerRelease
            true,  // SplitDpTmemConsumerRelease
            true,  // UseIterationCausalMask
            true,  // UseFusedTmemPAndDs
            true,  // OverlapFusedDqAPublication
            true,  // PrefetchNextQdoAfterDkdv
            true,  // PrefetchNextOwnerQdo
            true,  // UseFusedTmemRuntimeAccumulationPredicate
            true,  // UseBitwisePExpansion
            true,  // UseFusedExp2Pack
            true,  // PeelCausalPrefix
            true,  // BranchlessDoSourceLoad
            true,  // BranchlessDoSourceBaseSelect
            false  // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_direct_tma_dk_q_runtime_accumulate_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "direct-TMA runtime-accumulate BF16 route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "direct-TMA runtime-accumulate route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "direct-TMA runtime-accumulate route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "direct-TMA runtime-accumulate route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            false,
            false,
            false,
            false,
            false,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "H8 score-fanout route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "H8 score-fanout route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 score-fanout route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 score-fanout route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            false,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "H8 dQ-fanout route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "H8 dQ-fanout route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 dQ-fanout route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 dQ-fanout route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            false,
            true,
            true,
            true,
            true,
            true
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h8_exact_elected_peer_do_tma_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "H8 elected peer-dO route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "H8 elected peer-dO route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 elected peer-dO route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 elected peer-dO route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            false,  // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            false,  // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            false,  // BulkPeerDsFromFullTile
            false,  // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            false,  // IntegrateCausalFrontier
            0,      // ExactQTileCount
            false,  // FenceDsBeforeDkdvReady
            false,  // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true    // ElectedPeerDoTmaLoad
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h8_exact_elected_peer_do_integrated_frontier_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "H8 integrated frontier supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "H8 integrated frontier supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 integrated frontier does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 integrated frontier supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            false,  // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            false,  // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            false,  // BulkPeerDsFromFullTile
            false,  // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            false,  // FenceDsBeforeDkdvReady
            false,  // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true    // ElectedPeerDoTmaLoad
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "H8 named dK/dV fan-in supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "H8 named dK/dV fan-in supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 named dK/dV fan-in does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 named dK/dV fan-in supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            false,  // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            false,  // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            false,  // BulkPeerDsFromFullTile
            false,  // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true    // ElectedPeerDoTmaLoad
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "H8 merged dP-ready route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "H8 merged dP-ready route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 merged dP-ready route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 merged dP-ready route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            false,  // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            false,  // BulkPeerDsFromFullTile
            false,  // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true    // ElectedPeerDoTmaLoad
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "H8 fused-TMEM P/dS route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "H8 fused-TMEM P/dS route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 fused-TMEM P/dS route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 fused-TMEM P/dS route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true    // UseFusedTmemPAndDs
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_early_dq_a_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "H8 early-dQ-A route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "H8 early-dQ-A route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 early-dQ-A route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 early-dQ-A route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true    // OverlapFusedDqAPublication
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_early_dq_a_dkdv_qdo_prefetch_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "H8 post-dK Q/dO-prefetch route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "H8 post-dK Q/dO-prefetch route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 post-dK Q/dO-prefetch route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 post-dK Q/dO-prefetch route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true,   // OverlapFusedDqAPublication
            true    // PrefetchNextQdoAfterDkdv
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_early_dq_a_dkdv_qdo_prefetch_tmem_runtime_accumulate_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "H8 TMEM runtime-accumulate route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "H8 TMEM runtime-accumulate route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 TMEM runtime-accumulate route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 TMEM runtime-accumulate route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true,   // OverlapFusedDqAPublication
            true,   // PrefetchNextQdoAfterDkdv
            false,  // PrefetchNextOwnerQdo
            true    // UseFusedTmemRuntimeAccumulationPredicate
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_early_dq_a_dkdv_qdo_prefetch_tmem_runtime_accumulate_bit_expand_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "H8 bitwise P-expansion route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "H8 bitwise P-expansion route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 bitwise P-expansion route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 bitwise P-expansion route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true,   // OverlapFusedDqAPublication
            true,   // PrefetchNextQdoAfterDkdv
            false,  // PrefetchNextOwnerQdo
            true,   // UseFusedTmemRuntimeAccumulationPredicate
            true    // UseBitwisePExpansion
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h8_exact_persistent_v_stage_a_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "H8 persistent-V stage-A route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "H8 persistent-V stage-A route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 persistent-V stage-A route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 persistent-V stage-A route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true,   // OverlapFusedDqAPublication
            true,   // PrefetchNextQdoAfterDkdv
            false,  // PrefetchNextOwnerQdo
            true,   // UseFusedTmemRuntimeAccumulationPredicate
            true,   // UseBitwisePExpansion
            false,  // UseFusedExp2Pack
            false,  // PeelCausalPrefix
            false,  // BranchlessDoSourceLoad
            false,  // BranchlessDoSourceBaseSelect
            true    // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h8_exact_persistent_v_branchless_do_base_select_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "H8 branchless persistent-V route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "H8 branchless persistent-V route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 branchless persistent-V route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 branchless persistent-V route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true,   // OverlapFusedDqAPublication
            true,   // PrefetchNextQdoAfterDkdv
            false,  // PrefetchNextOwnerQdo
            true,   // UseFusedTmemRuntimeAccumulationPredicate
            true,   // UseBitwisePExpansion
            false,  // UseFusedExp2Pack
            false,  // PeelCausalPrefix
            true,   // BranchlessDoSourceLoad
            true,   // BranchlessDoSourceBaseSelect
            true    // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_h8_exact_persistent_v_branchless_wide_dq_k_tile_load_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "H8 wide dQ K-tile load route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "H8 wide dQ K-tile load route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "H8 wide dQ K-tile load route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 8,
        "H8 wide dQ K-tile load route supports B1 S8192 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            true,   // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true,   // OverlapFusedDqAPublication
            true,   // PrefetchNextQdoAfterDkdv
            false,  // PrefetchNextOwnerQdo
            true,   // UseFusedTmemRuntimeAccumulationPredicate
            true,   // UseBitwisePExpansion
            false,  // UseFusedExp2Pack
            false,  // PeelCausalPrefix
            true,   // BranchlessDoSourceLoad
            true,   // BranchlessDoSourceBaseSelect
            true    // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s8192h16_persistent_v_branchless_wide_dq_k_tile_load_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "S8192/H16 wide dQ K-tile load route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "S8192/H16 wide dQ K-tile load route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "S8192/H16 wide dQ K-tile load route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 16,
        "S8192/H16 wide dQ K-tile load route supports B1 S8192 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            true,   // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true,   // OverlapFusedDqAPublication
            true,   // PrefetchNextQdoAfterDkdv
            false,  // PrefetchNextOwnerQdo
            true,   // UseFusedTmemRuntimeAccumulationPredicate
            true,   // UseBitwisePExpansion
            false,  // UseFusedExp2Pack
            false,  // PeelCausalPrefix
            true,   // BranchlessDoSourceLoad
            true,   // BranchlessDoSourceBaseSelect
            true    // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s8192h2_persistent_v_branchless_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "S8192/H2 persistent-V branchless route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "S8192/H2 persistent-V branchless route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "S8192/H2 persistent-V branchless route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 2,
        "S8192/H2 persistent-V branchless route supports B1 S8192 H2 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true,   // OverlapFusedDqAPublication
            true,   // PrefetchNextQdoAfterDkdv
            false,  // PrefetchNextOwnerQdo
            true,   // UseFusedTmemRuntimeAccumulationPredicate
            true,   // UseBitwisePExpansion
            false,  // UseFusedExp2Pack
            false,  // PeelCausalPrefix
            true,   // BranchlessDoSourceLoad
            true,   // BranchlessDoSourceBaseSelect
            true    // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s8192h2_owner_q_split_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "S8192/H2 owner-Q split route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "S8192/H2 owner-Q split route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "S8192/H2 owner-Q split route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 2,
        "S8192/H2 owner-Q split route supports B1 S8192 H2 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_owner_q_split<HotConfig, 64, 2>(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s4096h4_persistent_v_branchless_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(
        causal,
        "S4096/H4 persistent-V branchless route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "S4096/H4 persistent-V branchless route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "S4096/H4 persistent-V branchless route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 4096 && q.size(2) == 4,
        "S4096/H4 persistent-V branchless route supports B1 S4096 H4 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            32,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true,   // OverlapFusedDqAPublication
            true,   // PrefetchNextQdoAfterDkdv
            false,  // PrefetchNextOwnerQdo
            true,   // UseFusedTmemRuntimeAccumulationPredicate
            true,   // UseBitwisePExpansion
            false,  // UseFusedExp2Pack
            false,  // PeelCausalPrefix
            true,   // BranchlessDoSourceLoad
            true,   // BranchlessDoSourceBaseSelect
            true    // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s4096h4_owner_q_split_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(
        causal,
        "S4096/H4 owner-Q split route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "S4096/H4 owner-Q split route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "S4096/H4 owner-Q split route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 4096 && q.size(2) == 4,
        "S4096/H4 owner-Q split route supports B1 S4096 H4 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_owner_q_split<HotConfig, 32, 4>(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s8192h4_persistent_v_branchless_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "S8192/H4 persistent-V branchless route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "S8192/H4 persistent-V branchless route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "S8192/H4 persistent-V branchless route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 4,
        "S8192/H4 persistent-V branchless route supports B1 S8192 H4 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true,   // OverlapFusedDqAPublication
            true,   // PrefetchNextQdoAfterDkdv
            false,  // PrefetchNextOwnerQdo
            true,   // UseFusedTmemRuntimeAccumulationPredicate
            true,   // UseBitwisePExpansion
            false,  // UseFusedExp2Pack
            false,  // PeelCausalPrefix
            true,   // BranchlessDoSourceLoad
            true,   // BranchlessDoSourceBaseSelect
            true    // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s8192h16_persistent_v_branchless_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(
        causal,
        "S8192/H16 persistent-V branchless route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "S8192/H16 persistent-V branchless route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "S8192/H16 persistent-V branchless route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 8192 && q.size(2) == 16,
        "S8192/H16 persistent-V branchless route supports B1 S8192 H16 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            64,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true,   // OverlapFusedDqAPublication
            true,   // PrefetchNextQdoAfterDkdv
            false,  // PrefetchNextOwnerQdo
            true,   // UseFusedTmemRuntimeAccumulationPredicate
            true,   // UseBitwisePExpansion
            false,  // UseFusedExp2Pack
            false,  // PeelCausalPrefix
            true,   // BranchlessDoSourceLoad
            true,   // BranchlessDoSourceBaseSelect
            true    // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s4096h8_persistent_v_branchless_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(
        causal,
        "S4096/H8 persistent-V branchless route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "S4096/H8 persistent-V branchless route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "S4096/H8 persistent-V branchless route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 4096 && q.size(2) == 8,
        "S4096/H8 persistent-V branchless route supports B1 S4096 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            32,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true,   // OverlapFusedDqAPublication
            true,   // PrefetchNextQdoAfterDkdv
            false,  // PrefetchNextOwnerQdo
            true,   // UseFusedTmemRuntimeAccumulationPredicate
            true,   // UseBitwisePExpansion
            false,  // UseFusedExp2Pack
            false,  // PeelCausalPrefix
            true,   // BranchlessDoSourceLoad
            true,   // BranchlessDoSourceBaseSelect
            true    // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s2048h8_persistent_v_branchless_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        2048
    );
    TORCH_CHECK(
        causal,
        "S2048/H8 persistent-V branchless route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "S2048/H8 persistent-V branchless route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "S2048/H8 persistent-V branchless route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 2048 && q.size(2) == 8,
        "S2048/H8 persistent-V branchless route supports B1 S2048 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_bf16_dkdv<
            HotConfig,
            true,   // CoalescedBf16Store
            false,  // DirectAsyncPeerDs
            true,   // ProducerBulkPeerDs
            true,   // ProducerBulkPeerDsCtaFenceOnly
            true,   // DqReadHandoffBeforeCompletion
            true,   // AggregateScoreConsumed
            true,   // DirectTmaDkQ
            true,   // TimeoutDqWait
            true,   // UseWideDkN192
            true,   // TimeoutAllRoleWaits
            true,   // UseNamedDoSourceBarrier
            true,   // UseComputeScoreFanout
            true,   // UseRuntimeAccumulationPredicate
            true,   // UseReducerDqFanout
            false,  // UseReducerDqLeaderArrive
            true,   // MergeScoreDpReady
            false,  // WideCoalescedBf16Store
            true,   // BulkPeerDsFromFullTile
            true,   // CoalescedPeerDsBulk
            false,  // WideDqKGlobalToShared
            true,   // IntegrateCausalFrontier
            16,     // ExactQTileCount
            true,   // FenceDsBeforeDkdvReady
            true,   // UseNamedDkdvLocalFanIn
            false,  // LeaderOnlyQdoPublishFence
            false,  // CacheQdoReadyClusterAddress
            false,  // GroupQdoTmaLoads
            false,  // ElectedWideDkQTmaLoad
            true,   // ElectedPeerDoTmaLoad
            false,  // CacheRoleClusterAddresses
            false,  // CacheTensorCommitAddresses
            false,  // EnsureReducerOutputDrain
            false,  // ElectedScoreKTmaLoad
            true,   // UseExactClusterCoordinates
            true,   // EnforceDpTmemConsumerRelease
            true,   // SplitDpTmemConsumerRelease
            true,   // UseIterationCausalMask
            true,   // UseFusedTmemPAndDs
            true,   // OverlapFusedDqAPublication
            true,   // PrefetchNextQdoAfterDkdv
            false,  // PrefetchNextOwnerQdo
            true,   // UseFusedTmemRuntimeAccumulationPredicate
            true,   // UseBitwisePExpansion
            false,  // UseFusedExp2Pack
            false,  // PeelCausalPrefix
            true,   // BranchlessDoSourceLoad
            true,   // BranchlessDoSourceBaseSelect
            true    // PublishVOncePerOwner
        >(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s2048h8_owner_q_split_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        2048
    );
    TORCH_CHECK(
        causal,
        "S2048/H8 owner-Q split route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "S2048/H8 owner-Q split route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "S2048/H8 owner-Q split route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 2048 && q.size(2) == 8,
        "S2048/H8 owner-Q split route supports B1 S2048 H8 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_owner_q_split<HotConfig, 16, 8>(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_s2048h4_owner_q_split_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        2048
    );
    TORCH_CHECK(
        causal,
        "S2048/H4 owner-Q split route supports causal=True only"
    );
    TORCH_CHECK(
        !deterministic,
        "S2048/H4 owner-Q split route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "S2048/H4 owner-Q split route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 && actual_seq_len == 2048 && q.size(2) == 4,
        "S2048/H4 owner-Q split route supports B1 S2048 H4 only"
    );

    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;
    at::Tensor dq = at::empty(q.sizes(), lse.options());
    at::Tensor dk = at::empty(k.sizes(), k.options());
    at::Tensor dv = at::empty(v.sizes(), v.options());
    tkfa4::bwd_cute16_candidate::
        launch_backward_cta2_fused_dense_owner_q_split<HotConfig, 16, 4>(
            q,
            k,
            v,
            out,
            lse,
            dout,
            dq,
            dk,
            dv,
            causal,
            softmax_scale,
            deterministic
        );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "single-owner route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "single-owner route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "single-owner route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            actual_seq_len == 16384 &&
            q.size(2) == 8,
        "single-owner route supports B1 S16384 H8 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "fast-exp2 route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "fast-exp2 route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "fast-exp2 route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            actual_seq_len == 16384 &&
            q.size(2) == 8,
        "fast-exp2 route supports B1 S16384 H8 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_warp_stats_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "warp-stats route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "warp-stats route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "warp-stats route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            actual_seq_len == 16384 &&
            q.size(2) == 8,
        "warp-stats route supports B1 S16384 H8 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_warp_stats_lse_pipeline_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "LSE-pipeline route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "LSE-pipeline route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "LSE-pipeline route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            actual_seq_len == 16384 &&
            q.size(2) == 8,
        "LSE-pipeline route supports B1 S16384 H8 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "direct-stats route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "direct-stats route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "direct-stats route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            actual_seq_len == 16384 &&
            q.size(2) == 8,
        "direct-stats route supports B1 S16384 H8 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_split_dv_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "split-dV route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "split-dV route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "split-dV route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            actual_seq_len == 16384 &&
            q.size(2) == 8,
        "split-dV route supports B1 S16384 H8 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_split_dv_early_dq_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "early-dQ route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "early-dQ route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "early-dQ route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            actual_seq_len == 16384 &&
            q.size(2) == 8,
        "early-dQ route supports B1 S16384 H8 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_split_dv_early_dq_peer_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "early-peer route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "early-peer route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "early-peer route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            actual_seq_len == 16384 &&
            q.size(2) == 8,
        "early-peer route supports B1 S16384 H8 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_split_dv_early_dq_peer_wide_dk_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        8192
    );
    TORCH_CHECK(causal, "wide-dK route supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "wide-dK route supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "wide-dK route does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1 &&
            actual_seq_len == 16384 &&
            q.size(2) == 8,
        "wide-dK route supports B1 S16384 H8 only"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_shared_ds_control_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA shared-dS control supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA shared-dS control supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA shared-dS control does not support sequence padding"
    );
    TORCH_CHECK(q.size(0) == 1, "2-CTA shared-dS control supports batch size 1 only");
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "2-CTA shared-dS control supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        false,
        false,
        false
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_serial_ds_control_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA serial-dS control supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA serial-dS control supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA serial-dS control does not support sequence padding"
    );
    TORCH_CHECK(q.size(0) == 1, "2-CTA serial-dS control supports batch size 1 only");
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "2-CTA serial-dS control supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        false,
        false
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_serial_q_control_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA serial-Q control supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA serial-Q control supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA serial-Q control does not support sequence padding"
    );
    TORCH_CHECK(q.size(0) == 1, "2-CTA serial-Q control supports batch size 1 only");
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "2-CTA serial-Q control supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        false
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_tmem_p_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA TMEM-P candidate supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA TMEM-P candidate supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA TMEM-P candidate does not support sequence padding"
    );
    TORCH_CHECK(q.size(0) == 1, "2-CTA TMEM-P candidate supports batch size 1 only");
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "2-CTA TMEM-P candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_tmem_p_overlap_do_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA dO-overlap candidate supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA dO-overlap candidate supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA dO-overlap candidate does not support sequence padding"
    );
    TORCH_CHECK(q.size(0) == 1, "2-CTA dO-overlap candidate supports batch size 1 only");
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "2-CTA dO-overlap candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_dp_ready_mbar_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA dP-ready candidate supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA dP-ready candidate supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA dP-ready candidate does not support sequence padding"
    );
    TORCH_CHECK(q.size(0) == 1, "2-CTA dP-ready candidate supports batch size 1 only");
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "2-CTA dP-ready candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_dq_ready_mbar_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA dQ-ready candidate supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA dQ-ready candidate supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA dQ-ready candidate does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1,
        "2-CTA dQ-ready candidate supports batch size 1 only"
    );
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "2-CTA dQ-ready candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_dv_overlap_ds_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA early-dV candidate supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA early-dV candidate supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA early-dV candidate does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1,
        "2-CTA early-dV candidate supports batch size 1 only"
    );
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "2-CTA early-dV candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_score_lookahead_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA score-lookahead candidate supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA score-lookahead candidate supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA score-lookahead candidate does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1,
        "2-CTA score-lookahead candidate supports batch size 1 only"
    );
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "2-CTA score-lookahead candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_dq_a_preload_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA dQ-A preload candidate supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA dQ-A preload candidate supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA dQ-A preload candidate does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1,
        "2-CTA dQ-A preload candidate supports batch size 1 only"
    );
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "2-CTA dQ-A preload candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_score_operand_mbar_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA score-mbar candidate supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA score-mbar candidate supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA score-mbar candidate does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1,
        "2-CTA score-mbar candidate supports batch size 1 only"
    );
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "2-CTA score-mbar candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_ds_warp_multicast_mbar_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(
        q,
        k,
        v,
        out,
        lse,
        dout,
        softmax_scale,
        actual_seq_len,
        4096
    );
    TORCH_CHECK(causal, "2-CTA dS multicast candidate supports causal=True only");
    TORCH_CHECK(
        !deterministic,
        "2-CTA dS multicast candidate supports deterministic=False only"
    );
    TORCH_CHECK(
        actual_seq_len == q.size(1),
        "2-CTA dS multicast candidate does not support sequence padding"
    );
    TORCH_CHECK(
        q.size(0) == 1,
        "2-CTA dS multicast candidate supports batch size 1 only"
    );
    const int64_t heads = q.size(2);
    TORCH_CHECK(
        (actual_seq_len == 4096 && heads == 8) ||
            (actual_seq_len == 8192 && heads == 4),
        "2-CTA dS multicast candidate supports S4096 H8 and S8192 H4"
    );
    return launch_b300_mha_bwd_hot_cute16_candidate_cta2_fused_dense<
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true
    >(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate_out_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    using HotConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(actual_seq_len == q.size(1), "hot backward does not support sequence padding");
    TORCH_CHECK(actual_seq_len % (tkfa4::kForwardTileM * 2) == 0, "hot backward requires seqlen divisible by 256");
    check_hot_backward_outputs(q, k, v, dq, dk, dv);

    tkfa4::bwd_cute16_candidate::launch_backward<HotConfig>(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        causal,
        softmax_scale,
        deterministic
    );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_trusted_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    static_cast<void>(actual_seq_len);
    return launch_b300_mha_bwd_hot_cute16_candidate(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate2_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(actual_seq_len == q.size(1), "hot backward does not support sequence padding");
    TORCH_CHECK(actual_seq_len % (tkfa4::kForwardTileM * 2) == 0, "hot backward requires seqlen divisible by 256");

    return launch_b300_mha_bwd_hot_cute16_candidate2(
        q,
        k,
        v,
        out,
        lse,
        dout,
        causal,
        softmax_scale,
        deterministic
    );
}

std::vector<at::Tensor> b300_mha_bwd_hot_dkdv_only_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    using DkdvConfig = tkfa4::bwd_cute16_candidate::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        1
    >;

    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(causal, "dK/dV-only hot backward currently supports causal=True");
    TORCH_CHECK(!deterministic, "dK/dV-only hot backward currently supports deterministic=False");
    TORCH_CHECK(actual_seq_len == q.size(1), "dK/dV-only hot backward does not support sequence padding");
    TORCH_CHECK(actual_seq_len >= tkfa4::kB300MinSeqLen && actual_seq_len <= 32768,
                "dK/dV-only hot backward supports seqlen in [2048, 32768]");
    TORCH_CHECK(actual_seq_len % (tkfa4::kForwardTileM * 2) == 0,
                "dK/dV-only hot backward requires seqlen divisible by 256");

    at::Tensor dk = at::empty({k.size(0), k.size(1), k.size(2), k.size(3)}, lse.options());
    at::Tensor dv = at::empty({v.size(0), v.size(1), v.size(2), v.size(3)}, lse.options());
    at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());

    tkfa4::bwd_cute16_candidate::launch_preprocess<tkfa4::bwd_cute16_candidate::preprocess_config<tkfa4::kB300VDim>>(
        out,
        dout,
        lse,
        dpsum,
        lse_log2
    );
    tkfa4::bwd_cute16_kernel_candidate::launch_backward_dkdv_only<DkdvConfig, float>(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dk,
        dv,
        softmax_scale
    );
    return {dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_cute16_candidate2_out_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    at::Tensor dq,
    at::Tensor dk,
    at::Tensor dv,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len,
    bool deterministic
) {
    const c10::cuda::CUDAGuard device_guard(q.device());
    using HotConfig = tkfa4::bwd_cute16_candidate2::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(actual_seq_len == q.size(1), "hot backward does not support sequence padding");
    TORCH_CHECK(actual_seq_len % (tkfa4::kForwardTileM * 2) == 0, "hot backward requires seqlen divisible by 256");
    check_hot_backward_outputs(q, k, v, dq, dk, dv);

    tkfa4::bwd_cute16_candidate2::launch_backward<HotConfig>(
        q,
        k,
        v,
        out,
        lse,
        dout,
        dq,
        dk,
        dv,
        causal,
        softmax_scale,
        deterministic
    );
    return {dq, dk, dv};
}

std::vector<at::Tensor> b300_mha_bwd_hot_legacy_internal(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor out,
    at::Tensor lse,
    at::Tensor dout,
    bool causal,
    float softmax_scale,
    int64_t actual_seq_len
) {
    using HotConfig = tkfa4::bwd_hot::config<
        tkfa4::kForwardTileM,
        tkfa4::kForwardTileN,
        tkfa4::kB300QKDim,
        tkfa4::kB300VDim,
        2
    >;

    check_common_backward_inputs_hot_public(q, k, v, out, lse, dout, softmax_scale, actual_seq_len);
    TORCH_CHECK(actual_seq_len == q.size(1), "legacy hot backward does not support sequence padding");
    TORCH_CHECK(actual_seq_len % (tkfa4::kForwardTileM * 2) == 0, "legacy hot backward requires seqlen divisible by 256");

    auto q_bhsd = q.permute({0, 2, 1, 3}).contiguous();
    auto k_bhsd = k.permute({0, 2, 1, 3}).contiguous();
    auto v_bhsd = v.permute({0, 2, 1, 3}).contiguous();
    auto out_bhsd = out.permute({0, 2, 1, 3}).contiguous();
    auto dout_bhsd = dout.permute({0, 2, 1, 3}).contiguous();

    at::Tensor dpsum = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor lse_log2 = at::empty({q.size(0), q.size(2), 1, q.size(1)}, lse.options());
    at::Tensor dq_bhsd = at::zeros({q.size(0), q.size(2), q.size(1), q.size(3)}, lse.options());
    at::Tensor dk_bhsd = at::zeros({k.size(0), k.size(2), k.size(1), k.size(3)}, lse.options());
    at::Tensor dv_bhsd = at::zeros({v.size(0), v.size(2), v.size(1), v.size(3)}, lse.options());
    at::Tensor dummy_semaphore = at::zeros({1, 1, 1, 1}, q.options().dtype(at::kInt));

    tkfa4::bwd_cute16::launch_preprocess<tkfa4::bwd_cute16::preprocess_config<tkfa4::kB300VDim>>(
        out,
        dout,
        lse,
        dpsum,
        lse_log2,
        dq_bhsd
    );
    tkfa4::bwd_hot::launch_backward<HotConfig>(
        q_bhsd,
        k_bhsd,
        v_bhsd,
        dout_bhsd,
        lse_log2,
        dpsum,
        dq_bhsd,
        dk_bhsd,
        dv_bhsd,
        causal,
        softmax_scale
    );
    return {
        dq_bhsd.permute({0, 2, 1, 3}).contiguous(),
        dk_bhsd.permute({0, 2, 1, 3}).contiguous(),
        dv_bhsd.permute({0, 2, 1, 3}).contiguous()
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("b300_mha_bwd", &b300_mha_bwd, "ThunderKittens exact B300 BF16 FlashAttention backward");
    m.def(
        "b300_mha_bwd_dv_only_internal",
        &b300_mha_bwd_dv_only,
        "ThunderKittens exact B300 BF16 FlashAttention dV-only backward stage"
    );
    m.def(
        "b300_mha_bwd_fa4_style",
        &b300_mha_bwd_fa4_style,
        "ThunderKittens experimental CuTe-style exact B300 BF16 FlashAttention backward"
    );
    m.def(
        "b300_mha_bwd_fa4_style_ref",
        &b300_mha_bwd_fa4_style_ref,
        "ThunderKittens experimental reference exact B300 BF16 FlashAttention backward"
    );
    m.def(
        "b300_mha_bwd_hot",
        &b300_mha_bwd_hot,
        "ThunderKittens experimental hot exact B300 BF16 FlashAttention backward for exact causal and noncausal shapes"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_internal",
        &b300_mha_bwd_hot_cute16_internal,
        "ThunderKittens private native CuTe16 bring-up exact B300 BF16 FlashAttention backward"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_nopatch_internal",
        &b300_mha_bwd_hot_cute16_nopatch_internal,
        "ThunderKittens private native CuTe16 exact B300 BF16 FlashAttention backward without causal patch kernels"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_internal",
        &b300_mha_bwd_hot_cute16_candidate_internal,
        "ThunderKittens private CuTe16 candidate exact B300 BF16 FlashAttention backward"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_bf16_dkdv_internal",
        &b300_mha_bwd_hot_cute16_candidate_bf16_dkdv_internal,
        "ThunderKittens private CuTe16 candidate B300 backward with BF16 DK/DV outputs"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_2cta_dkdv_internal",
        &b300_mha_bwd_hot_cute16_candidate_2cta_dkdv_internal,
        "ThunderKittens private CuTe16 candidate B300 backward with 2CTA seq2048 DK/DV route"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_2cta_dkdv_out_internal",
        &b300_mha_bwd_hot_cute16_candidate_2cta_dkdv_out_internal,
        "ThunderKittens private CuTe16 candidate B300 backward with 2CTA seq2048 DK/DV route into preallocated outputs"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_internal,
        "ThunderKittens private B300 backward with standalone dense TMEM plus exact frontier DK/DV route"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_chunked_tmem_dq_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_chunked_tmem_dq_internal,
        "ThunderKittens private B300 backward with standalone dense TMEM DK/DV and chunked TMEM DQ"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_internal,
        "ThunderKittens private B300 backward with standalone dense TMEM DK/DV and pipelined TMEM DQ"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_dkdv_first_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_dkdv_first_internal,
        "ThunderKittens private B300 backward with dK/dV-first standalone dense TMEM and double-buffered pipelined TMEM DQ"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_internal,
        "ThunderKittens private B300 backward with dQ-first split-2 dense dK/dV and double-buffered pipelined TMEM DQ"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_internal,
        "ThunderKittens private B300 backward with split-2 FP32 accumulation and fused BF16 dK/dV merge stores"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_cached_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_cached_internal,
        "ThunderKittens private B300 backward with cached split-2 FP32 bases and fused BF16 dK/dV merge stores"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_operand_release_u19_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_operand_release_u19_internal,
        "ThunderKittens private B300 backward U19 FP32 split-2 operand-release control"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_cached_u19_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_cached_u19_internal,
        "ThunderKittens private B300 backward U19 baseline with cached split-2 FP32 bases and fused BF16 dK/dV merge stores"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_cached_operand_release_dq_overlap_u19_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_cached_operand_release_dq_overlap_u19_internal,
        "ThunderKittens private B300 backward U19 operand-release child with clustered dQ overlap"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_cached_operand_release_full_overlap_u19_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_first_bf16_cached_operand_release_full_overlap_u19_internal,
        "ThunderKittens private B300 backward U19 operand-release child with dQ and frontier overlap"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split3_dq_first_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split3_dq_first_internal,
        "ThunderKittens private B300 backward with dQ-first split-3 dense dK/dV and double-buffered pipelined TMEM DQ"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_replay_split2_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_pipelined_tmem_dq_double_buffer_split2_dq_replay_split2_internal,
        "ThunderKittens private B300 backward with split-2 dense dK/dV and split-2 dQ replay for seqlen 8192, one head"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_adaptive_long_internal",
        &b300_mha_bwd_hot_cute16_candidate_adaptive_long_internal,
        "ThunderKittens private adaptive BF16 backward candidate for measured long-sequence regimes"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_adaptive_long_tmem_score_dp_internal",
        &b300_mha_bwd_hot_cute16_candidate_adaptive_long_tmem_score_dp_internal,
        "ThunderKittens private adaptive BF16 backward candidate using TMEM score and dP in dense dK/dV"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_adaptive_long_tmem_score_dp_tmem_frontier_internal",
        &b300_mha_bwd_hot_cute16_candidate_adaptive_long_tmem_score_dp_tmem_frontier_internal,
        "ThunderKittens private long BF16 backward candidate using standalone TMEM frontier dK/dV"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s4096h8_fused_dense_dq_internal",
        &b300_mha_bwd_hot_cute16_candidate_s4096h8_fused_dense_dq_internal,
        "ThunderKittens private S4096 H8 BF16 backward with on-chip dense dQ reduction"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s4096h8_fused_dense_dq_tail_quarter_internal",
        &b300_mha_bwd_hot_cute16_candidate_s4096h8_fused_dense_dq_tail_quarter_internal,
        "ThunderKittens private S4096 H8 BF16 backward with unsplit final causal quarter"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s4096h8_fused_dense_dq_tail_quarter_load_reducer_internal",
        &b300_mha_bwd_hot_cute16_candidate_s4096h8_fused_dense_dq_tail_quarter_load_reducer_internal,
        "ThunderKittens private S4096 H8 BF16 backward with load/reducer overlap"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_fused_dense_dq_load_reducer_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_fused_dense_dq_load_reducer_internal,
        "ThunderKittens private high-head BF16 backward with fused dQ and load/reducer overlap"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_fused_dense_dq_skip_tail_scratch_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_fused_dense_dq_skip_tail_scratch_internal,
        "ThunderKittens private high-head BF16 backward without unused tail split stores"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_ldsm_ds_transpose_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_ldsm_ds_transpose_internal,
        "ThunderKittens private high-head BF16 backward using LDSM dS transpose loads"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_double_buffer_dq_tma_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_double_buffer_dq_tma_internal,
        "ThunderKittens private high-head BF16 backward with two-stage dQ TMA drain"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_internal,
        "ThunderKittens private high-head BF16 backward with unsplit 2-CTA fused dense main"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_role_split_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_role_split_internal,
        "ThunderKittens private high-head BF16 backward with role-split 2-CTA dense main"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_retained_ds_exchange_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_retained_ds_exchange_internal,
        "ThunderKittens private high-head BF16 backward retaining the dS exchange half"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_retained_ds_both_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_retained_ds_both_internal,
        "ThunderKittens private high-head BF16 backward retaining both dS halves"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_normal_dv_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_normal_dv_internal,
        "ThunderKittens private high-head BF16 backward with normal-layout dV"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_tma_score_k_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_tma_score_k_internal,
        "ThunderKittens private high-head BF16 backward with TMA score K"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_internal,
        "ThunderKittens private high-head BF16 backward with direct next Q/dO"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_internal,
        "ThunderKittens private paired BF16 backward with early dQ staging"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_direct_global_stats_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_direct_global_stats_internal,
        "ThunderKittens private paired BF16 backward with direct global statistics loads"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_direct_global_stats_direct_ds_store_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_direct_global_stats_direct_ds_store_internal,
        "ThunderKittens private paired BF16 backward with direct dS half stores"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_direct_global_stats_direct_ds_store_fast_exp2_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_direct_global_stats_direct_ds_store_fast_exp2_internal,
        "ThunderKittens private paired BF16 backward with direct dS stores and approximate exp2"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_direct_global_stats_direct_ds_store_fast_exp2_asymmetric_dv_publish_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_direct_qdo_paired_early_dq_direct_global_stats_direct_ds_store_fast_exp2_asymmetric_dv_publish_internal,
        "ThunderKittens private paired BF16 backward with asymmetric dV publication"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_internal,
        "ThunderKittens private paired BF16 backward with BF16 dK/dV outputs"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_internal,
        "ThunderKittens private paired BF16 backward with coalesced BF16 dK/dV stores"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_direct_ds_async_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_direct_ds_async_internal,
        "ThunderKittens private paired BF16 backward with direct async peer dS stores"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_internal,
        "ThunderKittens private paired BF16 backward with producer-owned bulk dS stores"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_internal,
        "ThunderKittens private paired BF16 backward with CTA-fenced bulk dS stores"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_internal,
        "ThunderKittens private paired BF16 backward with dQ read-completion handoff"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_internal,
        "ThunderKittens private paired BF16 backward with aggregated score consumption"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_internal,
        "ThunderKittens private S16384 H16 BF16 backward with timeout dQ wait"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_internal,
        "ThunderKittens private S16384 H16 BF16 backward with wide direct-TMA dK Q"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_internal,
        "ThunderKittens private S16384 H16 BF16 backward with timeout role waits"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_internal,
        "ThunderKittens private S16384 H16 BF16 backward with named dO-source barrier"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_internal,
        "ThunderKittens private S16384 H16 BF16 backward with compute score fanout"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_internal,
        "ThunderKittens private S16384 H16 BF16 backward with runtime dV/dK accumulation predicate"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_internal,
        "ThunderKittens private S16384 H16 BF16 backward with reducer dQ fanout"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_internal,
        "ThunderKittens private S16384 H16 BF16 backward with nonblocking reducer leader fanout"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_internal,
        "ThunderKittens private S16384 H16 BF16 backward with merged score-consumed and V-ready handoff"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_internal,
        "ThunderKittens private S16384 H16 BF16 backward with aligned 128-bit dK/dV stores"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_internal,
        "ThunderKittens private S16384 H16 BF16 backward with peer dS bulk-copy sourced from the full dS tile"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_coalesced_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_coalesced_internal,
        "ThunderKittens private S16384 H16 BF16 backward with one 16 KiB peer dS bulk copy per CTA"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_coalesced_wide_dq_k_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_coalesced_wide_dq_k_internal,
        "ThunderKittens private S16384 H16 BF16 backward with 128-bit distributed dQ-K loads"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_coalesced_wide_dq_k_integrated_frontier_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_coalesced_wide_dq_k_integrated_frontier_internal,
        "ThunderKittens private S16384 H16 BF16 backward with integrated causal frontier"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_coalesced_wide_dq_k_integrated_frontier_exact_h16_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_dq_wait_timeout_wide_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_leader_arrive_merged_dp_ready_wide_store_full_ds_bulk_coalesced_wide_dq_k_integrated_frontier_exact_h16_internal,
        "ThunderKittens private S16384 H16 BF16 backward with exact tile count"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_exact_ds_publish_fence_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_exact_ds_publish_fence_internal,
        "ThunderKittens private S16384 H16 BF16 backward with a dS publish fence"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_exact_named_dkdv_fanin_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_exact_named_dkdv_fanin_internal,
        "ThunderKittens private S16384 H16 BF16 backward with named dK/dV fan-in"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_exact_leader_qdo_publish_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_exact_leader_qdo_publish_internal,
        "ThunderKittens private S16384 H16 BF16 backward with leader-owned Q/dO publication"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_exact_cached_qdo_ready_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_exact_cached_qdo_ready_internal,
        "ThunderKittens private S16384 H16 BF16 backward with cached Q/dO publication mapping"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_exact_grouped_qdo_tma_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_exact_grouped_qdo_tma_internal,
        "ThunderKittens private S16384 H16 BF16 backward with grouped Q/dO TMA loads"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_exact_grouped_qdo_elected_wide_dkq_tma_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_exact_grouped_qdo_elected_wide_dkq_tma_internal,
        "ThunderKittens private S16384 H16 BF16 backward with elected wide dK-Q TMA"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_exact_grouped_qdo_elected_wide_dkq_elected_peer_do_tma_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_exact_grouped_qdo_elected_wide_dkq_elected_peer_do_tma_internal,
        "ThunderKittens private S16384 H16 BF16 backward with elected peer dO TMA"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_exact_cached_role_cluster_addresses_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_exact_cached_role_cluster_addresses_internal,
        "ThunderKittens private S16384 H16 BF16 backward with cached role cluster addresses"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_exact_cached_tensor_commit_addresses_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_exact_cached_tensor_commit_addresses_internal,
        "ThunderKittens private S16384 H16 BF16 backward with cached tensor commit addresses"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_exact_reducer_output_drain_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_exact_reducer_output_drain_internal,
        "ThunderKittens private S16384 H16 BF16 backward with reducer output drain completion"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_exact_elected_score_k_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_exact_elected_score_k_internal,
        "ThunderKittens private S16384 H16 BF16 backward with elected score-K TMA"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_exact_cluster_coordinates_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_exact_cluster_coordinates_internal,
        "ThunderKittens private S16384 H16 BF16 backward with exact cluster coordinates"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_split_dp_consumer_release_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_split_dp_consumer_release_internal,
        "ThunderKittens private S16384 H16 BF16 backward with split dP-consumer release"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_iteration_causal_mask_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_iteration_causal_mask_internal,
        "ThunderKittens private S16384 H16 BF16 backward with iteration-relative causal masks"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_internal,
        "ThunderKittens private S16384 H16 BF16 backward with fused TMEM P and dS"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_early_dq_a_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_early_dq_a_internal,
        "ThunderKittens private S16384 H16 BF16 backward with early dQ-A publication"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_dkdv_qdo_prefetch_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_dkdv_qdo_prefetch_internal,
        "ThunderKittens private S16384 H16 BF16 backward with post-dK Q/dO prefetch"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_internal,
        "ThunderKittens private S16384 H16 BF16 backward with owner-boundary Q/dO prefetch"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_internal,
        "ThunderKittens private S16384 H16 BF16 backward with TMEM-A runtime accumulation predicates"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_internal,
        "ThunderKittens private S16384 H16 BF16 backward with exact bitwise P expansion"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal,
        "ThunderKittens private S16384 H16 BF16 backward with fused direct exp2-to-BF16 P packing"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_peeled_causal_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_peeled_causal_internal,
        "ThunderKittens private S16384 H16 BF16 backward with peeled causal score prefix"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_branchless_do_source_internal",
        &b300_mha_bwd_hot_cute16_candidate_h16_branchless_do_source_internal,
        "ThunderKittens private S16384 H16 BF16 backward with branchless dO source selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h16_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s16384_branchless_do_base_select_internal<16>,
        "ThunderKittens private S16384 H16 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s16384h4_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s16384h4h8_branchless_do_base_select_internal<4>,
        "ThunderKittens private S16384 H4 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s16384h8_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s16384h4h8_branchless_do_base_select_internal<8>,
        "ThunderKittens private S16384 H8 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s16384h32_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s16384_branchless_do_base_select_internal<32>,
        "ThunderKittens private S16384 H32 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s16384h64_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s16384_branchless_do_base_select_internal<64>,
        "ThunderKittens private S16384 H64 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s16384h128_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s16384_branchless_do_base_select_internal<128>,
        "ThunderKittens private S16384 H128 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s16384h32_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal",
        &b300_mha_bwd_hot_cute16_candidate_s16384h32_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal,
        "ThunderKittens private S16384 H32 BF16 backward with fused direct exp2-to-BF16 P packing"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s16384h64_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal",
        &b300_mha_bwd_hot_cute16_candidate_s16384h64_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal,
        "ThunderKittens private S16384 H64 BF16 backward with fused direct exp2-to-BF16 P packing"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s16384h128_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal",
        &b300_mha_bwd_hot_cute16_candidate_s16384h128_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal,
        "ThunderKittens private S16384 H128 BF16 backward with fused direct exp2-to-BF16 P packing"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s32768h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal",
        &b300_mha_bwd_hot_cute16_candidate_s32768h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal,
        "ThunderKittens private S32768 H16 BF16 backward with fused direct exp2-to-BF16 P packing"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s32768h32_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal",
        &b300_mha_bwd_hot_cute16_candidate_s32768h32_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal,
        "ThunderKittens private S32768 H32 BF16 backward with fused direct exp2-to-BF16 P packing"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s32768h64_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal",
        &b300_mha_bwd_hot_cute16_candidate_s32768h64_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal,
        "ThunderKittens private S32768 H64 BF16 backward with fused direct exp2-to-BF16 P packing"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s32768h128_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal",
        &b300_mha_bwd_hot_cute16_candidate_s32768h128_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal,
        "ThunderKittens private S32768 H128 BF16 backward with fused direct exp2-to-BF16 P packing"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s32768h128_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_peeled_causal_internal",
        &b300_mha_bwd_hot_cute16_candidate_s32768h128_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_peeled_causal_internal,
        "ThunderKittens private S32768 H128 BF16 backward with peeled causal score prefix"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s32768h128_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s32768_branchless_do_base_select_internal<128>,
        "ThunderKittens private S32768 H128 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s32768h16_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s32768_branchless_do_base_select_internal<16>,
        "ThunderKittens private S32768 H16 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s32768h32_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s32768_branchless_do_base_select_internal<32>,
        "ThunderKittens private S32768 H32 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s32768h64_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s32768_branchless_do_base_select_internal<64>,
        "ThunderKittens private S32768 H64 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h16_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal,
        "ThunderKittens private S65536 H16 BF16 backward with fused direct exp2-to-BF16 P packing"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h32_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h32_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal,
        "ThunderKittens private S65536 H32 BF16 backward with fused direct exp2-to-BF16 P packing"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h32_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536_branchless_do_base_select_internal<32>,
        "ThunderKittens private S65536 H32 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h16_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536_branchless_do_base_select_internal<16>,
        "ThunderKittens private S65536 H16 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_internal,
        "ThunderKittens private S65536 H64 BF16 backward with fused direct exp2-to-BF16 P packing"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_peeled_causal_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_fused_tmem_p_ds_owner_boundary_qdo_prefetch_tmem_runtime_accumulate_bit_expand_exp2_pack_peeled_causal_internal,
        "ThunderKittens private S65536 H64 BF16 backward with peeled causal score prefix"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_branchless_do_source_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_branchless_do_source_internal,
        "ThunderKittens private S65536 H64 BF16 backward with branchless dO source selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_branchless_do_base_select_internal,
        "ThunderKittens private S65536 H64 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_bulk_do_dv_stage_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_bulk_do_dv_stage_internal,
        "ThunderKittens private S65536 H64 BF16 backward with bulk dO-to-dV staging"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<false>,
        "ThunderKittens private S65536 H64 BF16 backward with loader-owned dK-Q publication"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, false>,
        "ThunderKittens private S65536 H64 BF16 backward with loader-owned dK-Q publication and fused score transform"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true>,
        "ThunderKittens private S65536 H64 BF16 backward retaining packed P through dS"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward splitting direct dPsum loads across dp_done"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with split dPsum loads and fused-exp2 fragment 4 first"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with a loop-carried direct-statistics offset"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with an all-blocking reducer dQ join"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with loop-carried phases in all active roles"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with exact-default score scale-log2e immediates"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_reverse_dk_tail_tmem_load_issue_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, false, true>,
        "ThunderKittens private S65536 H64 BF16 backward with reversed final dK-tail TMEM issue"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with exact-default score scale-log2e immediates and reversed final dK-tail TMEM issue"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with next-Q/dO transaction pre-arm"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_x32_tmem_compute_layout_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with x32 TMEM compute staging"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_piggyback_dpsum_tma_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with compact score issue and dPsum TMA piggybacked on Q/dO completion"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true>,
        "ThunderKittens private S65536 H64 BF16 backward with compact score MMA issue"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_packed_bf16_ds_product_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true>,
        "ThunderKittens private S65536 H64 BF16 backward with packed BF16 dS product"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_packed_bf16_ds_product_split_dq_tmem_shared_handoff_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with split dQ TMEM/shared handoff"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_packed_bf16_ds_product_split_dq_tmem_shared_handoff_distributed_wait_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with distributed dQ shared-read handoff wait"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_packed_bf16_ds_product_split_dq_tmem_shared_handoff_distributed_wait_balanced_single_owner_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with V51 scheduling and balanced single-owner clusters"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_packed_bf16_ds_product_split_dq_tmem_shared_handoff_distributed_wait_balanced_single_owner_warp_stats_cache_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with balanced single-owner clusters and warp-staged statistics"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_packed_bf16_ds_product_split_dq_tmem_shared_handoff_distributed_wait_balanced_single_owner_warp_stats_cache_cached_dq_stage_lane_pointers_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with cached even/odd reducer dQ shared-stage lane pointers"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_packed_bf16_ds_product_split_dq_tmem_shared_handoff_distributed_wait_balanced_single_owner_warp_stats_cache_cached_dq_stage_lane_pointers_sliced_fp32_p_ds_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward retaining FP32 P with sliced dP/dPsum consumption"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_packed_bf16_ds_product_split_dq_tmem_shared_handoff_distributed_wait_balanced_single_owner_warp_stats_cache_cached_dq_stage_lane_pointers_sliced_fp32_p_ds_tma_v_with_score_k_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward loading V by TMA on the score-K completion"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_packed_bf16_ds_product_split_dq_tmem_shared_handoff_distributed_wait_balanced_single_owner_warp_stats_cache_cached_dq_stage_lane_pointers_sliced_fp32_p_ds_tma_v_with_score_k_stats_warp_score_fanout_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward relaying score completion through the stats warp"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_packed_bf16_ds_product_split_dq_tmem_shared_handoff_distributed_wait_balanced_single_owner_warp_stats_cache_cached_dq_stage_lane_pointers_sliced_fp32_p_ds_tma_v_with_score_k_stats_warp_score_fanout_batched_dq_tmem_loads_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward loading all dQ TMEM fragments before reducer stores"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_packed_bf16_ds_product_split_dq_tmem_shared_handoff_distributed_wait_balanced_single_owner_warp_stats_cache_cached_dq_stage_lane_pointers_sliced_fp32_p_ds_tma_v_with_score_k_stats_warp_score_fanout_batched_dq_tmem_loads_dynamic_dp_release_barrier_id_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with a register-selected four-warp dP-release barrier"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_packed_bf16_ds_product_split_dq_tmem_shared_handoff_distributed_wait_balanced_single_owner_warp_stats_cache_cached_dq_stage_lane_pointers_sliced_fp32_p_ds_tma_v_with_score_k_stats_warp_score_fanout_batched_dq_tmem_loads_dynamic_dp_release_barrier_id_preissue_first_dp_half_before_qdo_wait_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward preissuing the first dP TMEM half across Q/dO readiness"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_mma_packed_bf16_ds_product_split_dq_tmem_shared_handoff_distributed_wait_balanced_single_owner_warp_stats_cache_cached_dq_stage_lane_pointers_sliced_fp32_p_ds_tma_v_with_score_k_stats_warp_score_fanout_batched_dq_tmem_loads_dynamic_dp_release_barrier_id_preissue_first_dp_half_before_qdo_wait_overlap_second_dp_load_with_release_barrier_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward overlapping the second dP TMEM load with the release join"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v174_exchange_warp_do_dv_completion_relay_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with exchange-warp dO-to-dV completion relay"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v195_overlap_dq_do_peer_copies_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward overlapping peer dQ and dO copies"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v199_overlap_local_dq_store_with_peer_copy_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private V195 backward overlapping local dQ stores with the peer copy"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v201_nonblocking_dq_publication_followers_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private V199 backward overlapping follower local dQ stores with the peer-source join"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v203_split_dq_alias_lifetime_cute_tmem_map_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private V201 backward splitting score and dP release across the CuTe dQ TMEM map"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v241_deferred_first_ds_tmem_store_wait_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private V203 backward deferring the first dS TMEM store wait until both half stores are issued"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v242_overlap_final_ds_tmem_store_with_peer_shared_stores_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private V241 backward overlapping the final dS TMEM store with peer shared-memory publication"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v248_delay_score_alias_release_until_first_dq_tail_load_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true>,
        "ThunderKittens private V242 backward issuing one dQ tail TMEM load before score-alias release"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v308_overlap_first_dpsum_quarter_with_second_p_store_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, 0, false, true>,
        "ThunderKittens private V248 backward overlapping the first dPsum quarter load with the second P half"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v318_shift_score_half_with_dpsum_overlap_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, true, false, 0, false, true>,
        "ThunderKittens private V308 backward releasing the overlapping score half before dP while retaining dPsum/P-store overlap"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v324_nonblocking_reducer_leader_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, false, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, true, false, 0, false, true>,
        "ThunderKittens private V318 backward allowing the reducer leader to leave the dQ fanout join after publishing it"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v339_hoist_reducer_dp_ready_before_score_wait_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, false, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, true, false, 0, false, true, true>,
        "ThunderKittens private V324 backward publishing reducer dP readiness before waiting for score completion"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v352_pipeline_first_dp_quarter_loads_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, false, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, true, false, 0, false, true, true, true>,
        "ThunderKittens private V339 backward pipelining the first dP half as two quarter loads"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v368_pipeline_next_qdo_and_peer_do_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, false, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, true, false, 0, false, true, true, true, true>,
        "ThunderKittens private V352 backward pipelining next Q/dO and peer dO before score-alias release"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v374_join_qdo_with_dq_alias_release_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, false, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, true, false, 0, false, true, true, true, true, true>,
        "ThunderKittens private V368 backward joining next Q/dO and dQ-alias completion"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v375_early_score_alias_release_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, false, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, true, false, 0, false, true, true, true, true, true>,
        "ThunderKittens private V374 backward releasing the score alias before the first dQ tail load"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v378_precompute_post_score_fanout_addresses_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, false, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, true, false, 0, false, true, true, true, true, true, true>,
        "ThunderKittens private V375 backward preparing post-score addresses under score fanout"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_v382_precompute_score_iteration_delta_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, false, true, true, true, true, false, false, true, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, true, false, 0, false, true, true, true, true, true, true, true>,
        "ThunderKittens private V378 backward computing the score iteration delta under fanout"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s8192_v382_advanced_long_internal",
        &b300_mha_bwd_hot_cute16_candidate_v382_advanced_long_internal<64, 8192>,
        "ThunderKittens private Exact64 V382 advanced long-sequence backward"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s16384_v382_advanced_long_internal",
        &b300_mha_bwd_hot_cute16_candidate_v382_advanced_long_internal<128, 16384>,
        "ThunderKittens private Exact128 V382 advanced long-sequence backward"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s32768_v382_advanced_long_internal",
        &b300_mha_bwd_hot_cute16_candidate_v382_advanced_long_internal<256, 32768>,
        "ThunderKittens private Exact256 V382 advanced long-sequence backward"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536_v382_advanced_long_internal",
        &b300_mha_bwd_hot_cute16_candidate_v382_advanced_long_internal<512, 65536>,
        "ThunderKittens private Exact512 V382 advanced long-sequence backward"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_fused_score_retain_packed_p_split_direct_dpsum_fused_exp2_fragment4_first_carry_stats_offset_all_blocking_reducer_dq_join_carry_all_role_phases_exact_default_scale_log2e_reverse_dk_tail_tmem_load_issue_prearm_next_qdo_compact_score_dp_mma_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h64_loader_owned_dk_q_internal<true, true, true, true, true, true, true, true, true, true, false, false, true, true>,
        "ThunderKittens private S65536 H64 BF16 backward with compact score and dP MMA issue"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s65536h128_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_s65536h128_branchless_do_base_select_internal,
        "ThunderKittens private S65536 H128 BF16 backward with branchless dO base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_direct_tma_dk_q_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_direct_tma_dk_q_internal,
        "ThunderKittens private paired BF16 backward with direct TMA dK Q operands"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_direct_tma_dk_q_runtime_accumulate_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_direct_tma_dk_q_runtime_accumulate_internal,
        "ThunderKittens private S8192 H8 direct-TMA backward with runtime accumulation predicates"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_internal,
        "ThunderKittens private S8192 H8 direct-TMA backward with H16 role handoffs"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_bf16_dkdv_coalesced_producer_bulk_ds_cta_fence_dq_read_handoff_score_fanin_direct_tma_dk_q_all_timeout_named_do_barrier_compute_score_fanout_runtime_accumulate_reducer_dq_fanout_internal,
        "ThunderKittens private S8192 H8 backward with reducer dQ fanout"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h8_exact_elected_peer_do_tma_internal",
        &b300_mha_bwd_hot_cute16_candidate_h8_exact_elected_peer_do_tma_internal,
        "ThunderKittens private S8192 H8 backward with elected peer dO TMA"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h8_exact_elected_peer_do_integrated_frontier_internal",
        &b300_mha_bwd_hot_cute16_candidate_h8_exact_elected_peer_do_integrated_frontier_internal,
        "ThunderKittens private S8192 H8 backward with integrated causal frontier"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_internal",
        &b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_internal,
        "ThunderKittens private S8192 H8 backward with named dK/dV fan-in"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_internal",
        &b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_internal,
        "ThunderKittens private S8192 H8 backward with merged dP-ready handoff"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_internal",
        &b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_internal,
        "ThunderKittens private S8192 H8 backward with fused TMEM P/dS"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_early_dq_a_internal",
        &b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_early_dq_a_internal,
        "ThunderKittens private S8192 H8 fused-TMEM backward with early dQ-A publication"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_early_dq_a_dkdv_qdo_prefetch_internal",
        &b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_early_dq_a_dkdv_qdo_prefetch_internal,
        "ThunderKittens private S8192 H8 fused-TMEM backward with post-dK Q/dO prefetch"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_early_dq_a_dkdv_qdo_prefetch_tmem_runtime_accumulate_internal",
        &b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_early_dq_a_dkdv_qdo_prefetch_tmem_runtime_accumulate_internal,
        "ThunderKittens private S8192 H8 fused-TMEM backward with runtime accumulation predicates"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_early_dq_a_dkdv_qdo_prefetch_tmem_runtime_accumulate_bit_expand_internal",
        &b300_mha_bwd_hot_cute16_candidate_h8_exact_integrated_frontier_named_dkdv_fanin_merged_dp_ready_fused_tmem_p_ds_early_dq_a_dkdv_qdo_prefetch_tmem_runtime_accumulate_bit_expand_internal,
        "ThunderKittens private S8192 H8 fused-TMEM backward with exact bitwise P expansion"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h8_exact_persistent_v_stage_a_internal",
        &b300_mha_bwd_hot_cute16_candidate_h8_exact_persistent_v_stage_a_internal,
        "ThunderKittens private S8192 H8 backward with one V publication per owner"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h8_exact_persistent_v_branchless_do_base_select_internal",
        &b300_mha_bwd_hot_cute16_candidate_h8_exact_persistent_v_branchless_do_base_select_internal,
        "ThunderKittens private S8192 H8 persistent-V backward with branchless dO source/base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_h8_exact_persistent_v_branchless_wide_dq_k_tile_load_internal",
        &b300_mha_bwd_hot_cute16_candidate_h8_exact_persistent_v_branchless_wide_dq_k_tile_load_internal,
        "ThunderKittens private S8192 H8 persistent-V backward with wide dQ K-tile load"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s8192h16_persistent_v_branchless_wide_dq_k_tile_load_internal",
        &b300_mha_bwd_hot_cute16_candidate_s8192h16_persistent_v_branchless_wide_dq_k_tile_load_internal,
        "ThunderKittens private S8192 H16 persistent-V backward with wide dQ K-tile load"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s8192h2_persistent_v_branchless_internal",
        &b300_mha_bwd_hot_cute16_candidate_s8192h2_persistent_v_branchless_internal,
        "ThunderKittens private S8192 H2 persistent-V backward with branchless dO source/base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s8192h2_owner_q_split_internal",
        &b300_mha_bwd_hot_cute16_candidate_s8192h2_owner_q_split_internal,
        "ThunderKittens private S8192 H2 backward with even/odd Q-work owner splits"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s4096h4_persistent_v_branchless_internal",
        &b300_mha_bwd_hot_cute16_candidate_s4096h4_persistent_v_branchless_internal,
        "ThunderKittens private S4096 H4 persistent-V branchless backward"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s4096h4_owner_q_split_internal",
        &b300_mha_bwd_hot_cute16_candidate_s4096h4_owner_q_split_internal,
        "ThunderKittens private S4096 H4 backward with even/odd Q-work owner splits"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s8192h4_persistent_v_branchless_internal",
        &b300_mha_bwd_hot_cute16_candidate_s8192h4_persistent_v_branchless_internal,
        "ThunderKittens private S8192 H4 persistent-V backward with branchless dO source/base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s8192h16_persistent_v_branchless_internal",
        &b300_mha_bwd_hot_cute16_candidate_s8192h16_persistent_v_branchless_internal,
        "ThunderKittens private S8192 H16 persistent-V backward with branchless dO source/base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s4096h8_persistent_v_branchless_internal",
        &b300_mha_bwd_hot_cute16_candidate_s4096h8_persistent_v_branchless_internal,
        "ThunderKittens private S4096 H8 persistent-V backward with branchless dO source/base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s2048h8_persistent_v_branchless_internal",
        &b300_mha_bwd_hot_cute16_candidate_s2048h8_persistent_v_branchless_internal,
        "ThunderKittens private S2048 H8 persistent-V backward with branchless dO source/base selection"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s2048h8_owner_q_split_internal",
        &b300_mha_bwd_hot_cute16_candidate_s2048h8_owner_q_split_internal,
        "ThunderKittens private S2048 H8 backward with even/odd Q-work owner splits"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_s2048h4_owner_q_split_internal",
        &b300_mha_bwd_hot_cute16_candidate_s2048h4_owner_q_split_internal,
        "ThunderKittens private S2048 H4 backward with even/odd Q-work owner splits"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_internal,
        "ThunderKittens private high-head BF16 backward with one owner per cluster"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_internal,
        "ThunderKittens private single-owner BF16 backward with approximate exp2"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_warp_stats_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_warp_stats_internal,
        "ThunderKittens private single-owner BF16 backward with warp-loaded stats"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_warp_stats_lse_pipeline_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_warp_stats_lse_pipeline_internal,
        "ThunderKittens private single-owner BF16 backward with pipelined LSE"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_internal,
        "ThunderKittens private single-owner BF16 backward with direct statistics loads"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_split_dv_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_split_dv_internal,
        "ThunderKittens private single-owner BF16 backward with split dV readiness"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_split_dv_early_dq_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_split_dv_early_dq_internal,
        "ThunderKittens private single-owner BF16 backward with early dQ staging"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_split_dv_early_dq_peer_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_split_dv_early_dq_peer_internal,
        "ThunderKittens private single-owner BF16 backward with early peer dQ staging"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_split_dv_early_dq_peer_wide_dk_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_single_owner_fast_exp2_direct_stats_split_dv_early_dq_peer_wide_dk_internal,
        "ThunderKittens private single-owner BF16 backward with N192 dK"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_shared_ds_control_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_shared_ds_control_internal,
        "ThunderKittens private 2-CTA fused dense shared-dS control"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_serial_ds_control_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_serial_ds_control_internal,
        "ThunderKittens private 2-CTA fused dense serial dS exchange control"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_serial_q_control_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_serial_q_control_internal,
        "ThunderKittens private 2-CTA fused dense serial Q preparation control"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_tmem_p_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_tmem_p_internal,
        "ThunderKittens private 2-CTA fused dense TMEM-P dV candidate"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_tmem_p_overlap_do_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_tmem_p_overlap_do_internal,
        "ThunderKittens private 2-CTA fused dense dO assembly overlap candidate"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_dp_ready_mbar_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_dp_ready_mbar_internal,
        "ThunderKittens private 2-CTA fused dense dP operand-ready mbarrier candidate"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_dq_ready_mbar_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_dq_ready_mbar_internal,
        "ThunderKittens private 2-CTA fused dense dQ operand-ready mbarrier candidate"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_dv_overlap_ds_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_dv_overlap_ds_internal,
        "ThunderKittens private 2-CTA fused dense early-dV phase-overlap candidate"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_score_lookahead_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_score_lookahead_internal,
        "ThunderKittens private 2-CTA fused dense score-lookahead candidate"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_dq_a_preload_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_dq_a_preload_internal,
        "ThunderKittens private 2-CTA fused dense dQ-A preload candidate"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_score_operand_mbar_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_score_operand_mbar_internal,
        "ThunderKittens private 2-CTA fused dense score operand mbarrier candidate"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_ds_warp_multicast_mbar_internal",
        &b300_mha_bwd_hot_cute16_candidate_high_head_cta2_fused_dense_ds_warp_multicast_mbar_internal,
        "ThunderKittens private 2-CTA fused dense per-warp multicast dS mbarrier candidate"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_out_internal",
        &b300_mha_bwd_hot_cute16_candidate_dense_tmem_frontier_dkdv_out_internal,
        "ThunderKittens private B300 backward with standalone dense TMEM plus exact frontier DK/DV route into preallocated outputs"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate_out_internal",
        &b300_mha_bwd_hot_cute16_candidate_out_internal,
        "ThunderKittens private CuTe16 candidate exact B300 BF16 FlashAttention backward into preallocated outputs"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate2_internal",
        &b300_mha_bwd_hot_cute16_candidate2_internal,
        "ThunderKittens private CuTe16 candidate2 exact B300 BF16 FlashAttention backward"
    );
    m.def(
        "b300_mha_bwd_hot_dkdv_only_internal",
        &b300_mha_bwd_hot_dkdv_only_internal,
        "ThunderKittens private dK/dV-only stage for hot B300 BF16 FlashAttention backward"
    );
    m.def(
        "b300_mha_bwd_hot_cute16_candidate2_out_internal",
        &b300_mha_bwd_hot_cute16_candidate2_out_internal,
        "ThunderKittens private CuTe16 candidate2 exact B300 BF16 FlashAttention backward into preallocated outputs"
    );
    m.def(
        "b300_mha_bwd_hot_trusted_internal",
        &b300_mha_bwd_hot_trusted_internal,
        "ThunderKittens private trusted hot exact B300 BF16 FlashAttention backward"
    );
    m.def(
        "b300_mha_bwd_hot_legacy_internal",
        &b300_mha_bwd_hot_legacy_internal,
        "ThunderKittens private legacy compact-hot exact B300 BF16 FlashAttention backward"
    );
}
