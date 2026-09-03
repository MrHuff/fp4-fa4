#pragma once

template <typename Tile>
__device__ inline void fp4pv_shape_zero_tile(Tile &tile) {
    const uint32_t base = __cvta_generic_to_shared(&tile.data[0]);
    for (int byte = int(threadIdx.x) * 4;
         byte < int(sizeof(Tile));
         byte += int(blockDim.x) * 4) {
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(base + byte), "r"(0u));
    }
}

template <typename ScaleTT>
__device__ __forceinline__ void
fp4pv_fill_mxfp4_scale_tmem_e8m0_issue_lane(
    ScaleTT &scale_tm,
    uint32_t packed_scale
) {
    static_assert(std::is_same_v<typename ScaleTT::T, fp8e8m0>);
    static_assert(ScaleTT::rows == 128);
    #pragma unroll
    for (int row = 0; row < 128; row += 32) {
        const uint32_t tmem_addr =
            scale_tm.addr + (static_cast<uint32_t>(row) << 16);
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x4.b32 "
            "[%0], {%1, %1, %1, %1};" ::
            "r"(tmem_addr), "r"(packed_scale));
    }
}

__device__ __forceinline__ void fp4pv_tmem_store_wait() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::);
}

template <typename TT, typename ST>
__device__ inline void
fp4pv_load_mxnv_scale_async_ranklocal(TT &dst, const ST &src) {
    load_mxnv_scale_async<1>(dst, src);
}

__device__ __forceinline__ void fp4pv_shared_async_proxy_fence_cta() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
