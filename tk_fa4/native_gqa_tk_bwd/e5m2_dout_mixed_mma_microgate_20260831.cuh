#pragma once

#include "../deprecated/fa4_common.cuh"

#include <cstdint>
#include <type_traits>

// Standalone, fail-closed proof for the only new tensor-core primitive needed
// by an E5M2-dO backward experiment.  This file is deliberately independent of
// every production backward kernel and dispatcher.
namespace tkfa4::native_gqa_tk_bwd::e5m2_dout_mixed_mma_microgate_20260831 {

constexpr int kRows = 128;
constexpr int kCols = 128;
constexpr int kReduction = 128;
constexpr int kReductionChunk = 32;
constexpr int kDrainWarps = 4;
constexpr int kTensorIssueWarp = 4;
constexpr int kLoaderWarp = 5;
constexpr int kThreads = 192;

using a_tile = st_fp8e4m3<kRows, kReduction>;
using b_tile = st_fp8e5m2<kRows, kReduction>;
using output_tile = st_fl<kRows, kCols>;
using output_tmem_tile = full_tt_fl<kCols>;
using output_chunk_tmem_tile = full_tt_fl<kReductionChunk>;

enum class operation : int {
    dp_abt = 0,
    dv_ab = 1,
};

// PTX 8-bit dense tcgen05 format fields are independent for A and B.  E4M3
// is 000 and E5M2 is 001.  Consequently changing only B from E4M3 to E5M2 is
// exactly bit 10 (0x400).  Keep this builder local so the proof does not need
// to weaken TK's useful same-type assertions.
constexpr uint32_t make_dense_fp8_instruction_descriptor(
    bool transpose_b,
    uint32_t a_format,
    uint32_t b_format
) {
    uint32_t descriptor = 0;
    descriptor |= 1u << 4;                    // FP32 accumulator/output.
    descriptor |= (a_format & 0x7u) << 7;    // A format.
    descriptor |= (b_format & 0x7u) << 10;   // B format.
    descriptor |= (transpose_b ? 1u : 0u) << 16;
    descriptor |= (static_cast<uint32_t>(kCols) >> 3) << 17;
    descriptor |= (static_cast<uint32_t>(kRows) >> 4) << 24;
    return descriptor;
}

constexpr uint32_t kE5m2BFormatMask = 0x400u;
constexpr uint32_t kDpE4m3E4m3Instruction =
    make_dense_fp8_instruction_descriptor(false, 0u, 0u);
constexpr uint32_t kDpE4m3E5m2Instruction =
    make_dense_fp8_instruction_descriptor(false, 0u, 1u);
constexpr uint32_t kDvE4m3E4m3Instruction =
    make_dense_fp8_instruction_descriptor(true, 0u, 0u);
constexpr uint32_t kDvE4m3E5m2Instruction =
    make_dense_fp8_instruction_descriptor(true, 0u, 1u);

static_assert(
    kDpE4m3E5m2Instruction ==
        (kDpE4m3E4m3Instruction | kE5m2BFormatMask)
);
static_assert(
    kDvE4m3E5m2Instruction ==
        (kDvE4m3E4m3Instruction | kE5m2BFormatMask)
);
static_assert(
    (kDpE4m3E5m2Instruction & (0x7u << 7)) == 0u &&
        (kDpE4m3E5m2Instruction & (0x7u << 10)) == (1u << 10)
);
static_assert(
    (kDvE4m3E5m2Instruction & (0x7u << 7)) == 0u &&
        (kDvE4m3E5m2Instruction & (0x7u << 10)) == (1u << 10)
);

struct globals {
    using a_gl = gl<
        fp8e4m3,
        1,
        1,
        -1,
        -1,
        tma::descriptor<a_tile, dim::ROW>
    >;
    using b_gl = gl<
        fp8e5m2,
        1,
        1,
        -1,
        -1,
        tma::descriptor<b_tile, dim::ROW>
    >;
    using output_gl = gl<
        float,
        1,
        1,
        -1,
        -1,
        tma::descriptor<output_tile, dim::ROW>
    >;

    a_gl a0;
    b_gl b0;
    a_gl a1;
    b_gl b1;
    output_gl output;
};

struct shared_storage {
    a_tile a;
    b_tile b;
    output_tile output;
};

static_assert(sizeof(a_tile) == 16 * 1024);
static_assert(sizeof(b_tile) == 16 * 1024);
static_assert(sizeof(output_tile) == 64 * 1024);
static_assert(sizeof(shared_storage) == 96 * 1024);

template <operation Operation>
__device__ __forceinline__ constexpr uint32_t mixed_instruction() {
    constexpr bool kTransposeB = Operation == operation::dv_ab;
    constexpr uint32_t kTkE4m3E4m3 =
        ::kittens::detail::tcgen05::instruction_descriptor<
            float,
            fp8e4m3,
            kRows,
            kCols,
            transpose::N,
            kTransposeB ? transpose::T : transpose::N,
            false
        >();
    constexpr uint32_t kMixed = kTkE4m3E4m3 | kE5m2BFormatMask;
    constexpr uint32_t kExpected = Operation == operation::dv_ab
        ? kDvE4m3E5m2Instruction
        : kDpE4m3E5m2Instruction;
    static_assert(kMixed == kExpected);
    static_assert((kMixed ^ kTkE4m3E4m3) == 0x400u);
    return kMixed;
}

// Issue one complete K128 product into a 128x128 FP32 TMEM accumulator.
// The instruction kind remains f8f6f4; only the descriptor's B-format bits
// differ.  dP consumes A @ B^T (both shared descriptors K-major).  dV
// consumes A @ B and carries the project-verified MN-major K32 correction:
// B advances by two descriptor chunks for each tensor-core K32 command.
template <operation Operation, int Accumulate>
__device__ __forceinline__ void issue_mixed_product(
    output_tmem_tile &destination,
    const a_tile &a,
    const b_tile &b
) {
    static_assert(Accumulate == 0 || Accumulate == 1);
    constexpr bool kDv = Operation == operation::dv_ab;
    constexpr int kBMajor = kDv ? transpose::T : transpose::N;
    constexpr uint32_t kInstruction = mixed_instruction<Operation>();

    if (warpgroup::laneid() == 0) {
        ::kittens::st_descriptor<a_tile, transpose::N> a_descriptor(a);
        ::kittens::st_descriptor<b_tile, kBMajor> b_descriptor(b);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

        ::kittens::detail::tcgen05::template st_st<
            fp8e4m3,
            Accumulate,
            1
        >(
            destination.addr,
            a_descriptor.chunk_descriptor(0),
            b_descriptor.chunk_descriptor(0),
            kInstruction
        );
#pragma unroll
        for (int chunk = 1; chunk < kReduction / kReductionChunk; ++chunk) {
            const int b_chunk = kDv ? 2 * chunk : chunk;
            ::kittens::detail::tcgen05::template st_st<fp8e4m3, 1, 1>(
                destination.addr,
                a_descriptor.chunk_descriptor(chunk),
                b_descriptor.chunk_descriptor(b_chunk),
                kInstruction
            );
        }
    }
}

__device__ __forceinline__ void drain_output(
    const output_tmem_tile &source,
    output_tile &destination,
    int physical_warp
) {
#pragma unroll
    for (int chunk = 0; chunk < kCols / kReductionChunk; ++chunk) {
        rt_fl<32, kReductionChunk> values;
        const output_chunk_tmem_tile source_chunk =
            source.template subtile<output_chunk_tmem_tile>(
                0,
                chunk * kReductionChunk
            );
        group<kDrainWarps>::load_async(values, source_chunk);
        tensor_load_wait();
        auto destination_slice =
            destination.template subtile<32, kReductionChunk>(
                {physical_warp, chunk}
            );
        warp::store(destination_slice, values);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

template <operation Operation, bool Accumulate>
__global__ __launch_bounds__(kThreads, 1)
void microgate_kernel(const __grid_constant__ globals g) {
    __shared__ alignas(1024) shared_storage storage;
    __shared__ alignas(16) semaphore input_ready;
    __shared__ alignas(16) semaphore mma_ready;

    const int physical_warp = warpid();
    const int lane = laneid();

    if (threadIdx.x == 0) {
        init_semaphore(input_ready, 0, 1);
        init_semaphore(mma_ready, 0, 1);
        g.a0.template prefetch_tma<a_tile, dim::ROW>();
        g.b0.template prefetch_tma<b_tile, dim::ROW>();
        if constexpr (Accumulate) {
            g.a1.template prefetch_tma<a_tile, dim::ROW>();
            g.b1.template prefetch_tma<b_tile, dim::ROW>();
        }
        g.output.template prefetch_tma<output_tile, dim::ROW>();
    }
    __syncthreads();

    tensor_allocator<1, 1> tmem_allocator{};
    output_tmem_tile accumulator =
        tmem_allocator.template allocate<output_tmem_tile>(0);

    if (physical_warp == kLoaderWarp && lane == 0) {
        tma::expect_bytes(input_ready, sizeof(a_tile) + sizeof(b_tile));
        tma::load_async<dim::ROW, cache_policy::NORMAL>(
            storage.a,
            g.a0,
            coord<a_tile>{0, 0, 0, 0},
            input_ready
        );
        tma::load_async<dim::ROW, cache_policy::NORMAL>(
            storage.b,
            g.b0,
            coord<b_tile>{0, 0, 0, 0},
            input_ready
        );
    }
    wait(input_ready, 0);
    __syncthreads();

    if (physical_warp == kTensorIssueWarp && lane == 0) {
        issue_mixed_product<Operation, 0>(
            accumulator,
            storage.a,
            storage.b
        );
        tensor_commit<1>(mma_ready);
    }
    wait(mma_ready, 0);
    tensor_after_thread_sync();
    __syncthreads();

    if constexpr (Accumulate) {
        if (physical_warp == kLoaderWarp && lane == 0) {
            tma::expect_bytes(input_ready, sizeof(a_tile) + sizeof(b_tile));
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                storage.a,
                g.a1,
                coord<a_tile>{0, 0, 0, 0},
                input_ready
            );
            tma::load_async<dim::ROW, cache_policy::NORMAL>(
                storage.b,
                g.b1,
                coord<b_tile>{0, 0, 0, 0},
                input_ready
            );
        }
        wait(input_ready, 1);
        __syncthreads();

        if (physical_warp == kTensorIssueWarp && lane == 0) {
            issue_mixed_product<Operation, 1>(
                accumulator,
                storage.a,
                storage.b
            );
            tensor_commit<1>(mma_ready);
        }
        wait(mma_ready, 1);
        tensor_after_thread_sync();
        __syncthreads();
    }

    if (physical_warp < kDrainWarps) {
        drain_output(accumulator, storage.output, physical_warp);
    }
    __syncthreads();

    if (physical_warp == kLoaderWarp && lane == 0) {
        warp::tma::store_async<dim::ROW, cache_policy::NORMAL>(
            g.output,
            storage.output,
            coord<output_tile>{0, 0, 0, 0}
        );
        warp::tma::store_async_wait<0>();
    }
    __syncthreads();
}

template <operation Operation, bool Accumulate>
inline void launch(
    at::Tensor &a0,
    at::Tensor &b0,
    at::Tensor &a1,
    at::Tensor &b1,
    at::Tensor &output,
    cudaStream_t stream
) {
    const globals g{
        kittens::py::tensor_to_gl<globals::a_gl>(a0, 1, 1, kRows, kReduction),
        kittens::py::tensor_to_gl<globals::b_gl>(b0, 1, 1, kRows, kReduction),
        kittens::py::tensor_to_gl<globals::a_gl>(a1, 1, 1, kRows, kReduction),
        kittens::py::tensor_to_gl<globals::b_gl>(b1, 1, 1, kRows, kReduction),
        kittens::py::tensor_to_gl<globals::output_gl>(output, 1, 1, kRows, kCols),
    };
    microgate_kernel<Operation, Accumulate><<<1, kThreads, 0, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

}  // namespace tkfa4::native_gqa_tk_bwd::e5m2_dout_mixed_mma_microgate_20260831
