#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "b300_common.cuh"
#include "projection_n_multicast.cuh"

namespace {

template <int PAIRS>
__global__ __launch_bounds__(128, 1) void tmem_pair_smoke_kernel(
    uint32_t *done
) {
    kittens::tensor_allocator<1, 2, false> tm_allocator;
    __shared__ uint32_t tmem_addr;
    kittens::everyone::tma::cluster::sync();
    const int rank = kittens::cluster_ctarank();
    if (kittens::warpid() == 0) {
        tm_allocator.provision(tmem_addr);
    }
    __syncthreads();
    tm_allocator.set_addr(tmem_addr);
    __syncthreads();
    if (kittens::warpid() == 0) {
        tm_allocator.deprovision();
    }
    if (threadIdx.x == 0) {
        done[rank] = 1u;
    }
}

template <int PAIRS>
at::Tensor launch_tmem_pair_smoke() {
    auto done = at::zeros(
        {2 * PAIRS},
        at::TensorOptions().device(at::kCUDA).dtype(at::kInt)
    );
    auto kernel = tmem_pair_smoke_kernel<PAIRS>;
    if constexpr (PAIRS > 4) {
        CUDACHECK(cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeNonPortableClusterSizeAllowed,
            1
        ));
    }
    kittens::LaunchConfig<true, false> launch_config(
        dim3(2, PAIRS, 1),
        dim3(128, 1, 1),
        0,
        at::cuda::getCurrentCUDAStream().stream(),
        dim3(2, PAIRS, 1)
    );
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel,
        reinterpret_cast<uint32_t *>(done.data_ptr())
    ));
    return done;
}

at::Tensor tmem_pair_smoke(int64_t cluster_size) {
    TORCH_CHECK(
        cluster_size == 2 || cluster_size == 4 ||
            cluster_size == 8 || cluster_size == 16,
        "cluster_size must be 2, 4, 8, or 16"
    );
    if (cluster_size == 2) {
        return launch_tmem_pair_smoke<1>();
    }
    if (cluster_size == 4) {
        return launch_tmem_pair_smoke<2>();
    }
    if (cluster_size == 8) {
        return launch_tmem_pair_smoke<4>();
    }
    return launch_tmem_pair_smoke<8>();
}

template <typename C>
at::Tensor launch_projection(
    const at::Tensor &input_fp4,
    const at::Tensor &input_scales,
    const at::Tensor &input_global_scale,
    const at::Tensor &weight_fp4,
    const at::Tensor &weight_scales,
    const at::Tensor &weight_global_scale,
    int cluster_cap
) {
    using G = tkfa4_projection_n_multicast::globals<C>;
    const int rows = static_cast<int>(input_fp4.size(0));
    const int reduction = static_cast<int>(input_fp4.size(1) * 2);
    const int output_width = static_cast<int>(weight_fp4.size(0));
    auto output = at::empty(
        {rows, output_width},
        input_fp4.options().dtype(at::kBFloat16)
    );
    G globals{
        .A = kittens::py::tensor_to_gl<typename G::A_gl>(
            input_fp4, 1, 1, rows, reduction / 2
        ),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            input_scales,
            1,
            input_scales.size(0),
            input_scales.size(1),
            256
        ),
        .A_scale = kittens::py::tensor_to_gl<typename G::scale_gl>(
            input_global_scale
        ),
        .B = kittens::py::tensor_to_gl<typename G::B_gl>(
            weight_fp4, 1, 1, output_width, reduction / 2
        ),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            weight_scales,
            1,
            weight_scales.size(0),
            weight_scales.size(1),
            256
        ),
        .B_scale = kittens::py::tensor_to_gl<typename G::scale_gl>(
            weight_global_scale
        ),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(
            output, 1, 1, rows, output_width
        ),
        .cluster_cap = cluster_cap,
    };
    tkfa4_projection_n_multicast::launch(
        globals,
        at::cuda::getCurrentCUDAStream().stream()
    );
    return output;
}

template <typename C>
at::Tensor launch_tma_multicast_smoke(
    const at::Tensor &input_fp4,
    const at::Tensor &input_scales,
    const at::Tensor &input_global_scale,
    const at::Tensor &weight_fp4,
    const at::Tensor &weight_scales,
    const at::Tensor &weight_global_scale
) {
    using G = tkfa4_projection_n_multicast::globals<C>;
    const int rows = static_cast<int>(input_fp4.size(0));
    const int reduction = static_cast<int>(input_fp4.size(1) * 2);
    const int output_width = static_cast<int>(weight_fp4.size(0));
    auto output = at::empty(
        {rows, output_width},
        input_fp4.options().dtype(at::kBFloat16)
    );
    auto done = at::zeros(
        {C::CLUSTER_SIZE},
        input_fp4.options().dtype(at::kInt)
    );
    G globals{
        .A = kittens::py::tensor_to_gl<typename G::A_gl>(
            input_fp4, 1, 1, rows, reduction / 2
        ),
        .A_sc = kittens::py::tensor_to_gl<typename G::A_sc_gl, false>(
            input_scales,
            1,
            input_scales.size(0),
            input_scales.size(1),
            256
        ),
        .A_scale = kittens::py::tensor_to_gl<typename G::scale_gl>(
            input_global_scale
        ),
        .B = kittens::py::tensor_to_gl<typename G::B_gl>(
            weight_fp4, 1, 1, output_width, reduction / 2
        ),
        .B_sc = kittens::py::tensor_to_gl<typename G::B_sc_gl, false>(
            weight_scales,
            1,
            weight_scales.size(0),
            weight_scales.size(1),
            256
        ),
        .B_scale = kittens::py::tensor_to_gl<typename G::scale_gl>(
            weight_global_scale
        ),
        .D = kittens::py::tensor_to_gl<typename G::D_gl>(
            output, 1, 1, rows, output_width
        ),
        .cluster_cap = 1,
    };
    tkfa4_projection_n_multicast::launch_tma_smoke(
        globals,
        reinterpret_cast<uint32_t *>(done.data_ptr()),
        at::cuda::getCurrentCUDAStream().stream()
    );
    return done;
}

at::Tensor tma_multicast_smoke(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor weight_fp4,
    at::Tensor weight_scales,
    at::Tensor weight_global_scale,
    int64_t cluster_size
) {
    if (cluster_size == 2) {
        return launch_tma_multicast_smoke<
            tkfa4_projection_n_multicast::config<2>
        >(
            input_fp4, input_scales, input_global_scale,
            weight_fp4, weight_scales, weight_global_scale
        );
    }
    if (cluster_size == 4) {
        return launch_tma_multicast_smoke<
            tkfa4_projection_n_multicast::config<4>
        >(
            input_fp4, input_scales, input_global_scale,
            weight_fp4, weight_scales, weight_global_scale
        );
    }
    if (cluster_size == 8) {
        return launch_tma_multicast_smoke<
            tkfa4_projection_n_multicast::config<8>
        >(
            input_fp4, input_scales, input_global_scale,
            weight_fp4, weight_scales, weight_global_scale
        );
    }
    TORCH_CHECK(cluster_size == 16, "cluster_size must be 2, 4, 8, or 16");
    return launch_tma_multicast_smoke<
        tkfa4_projection_n_multicast::config<16>
    >(
        input_fp4, input_scales, input_global_scale,
        weight_fp4, weight_scales, weight_global_scale
    );
}

at::Tensor project_nvfp4_n_multicast(
    at::Tensor input_fp4,
    at::Tensor input_scales,
    at::Tensor input_global_scale,
    at::Tensor weight_fp4,
    at::Tensor weight_scales,
    at::Tensor weight_global_scale,
    int64_t cluster_size,
    int64_t cluster_cap
) {
    TORCH_CHECK(
        input_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            weight_fp4.scalar_type() == at::kFloat4_e2m1fn_x2 &&
            input_fp4.is_cuda() && weight_fp4.is_cuda() &&
            input_fp4.is_contiguous() && weight_fp4.is_contiguous() &&
            input_fp4.dim() == 2 && weight_fp4.dim() == 2,
        "projection-N operands must be contiguous CUDA packed E2M1 matrices"
    );
    const int rows = static_cast<int>(input_fp4.size(0));
    const int reduction = static_cast<int>(input_fp4.size(1) * 2);
    const int output_width = static_cast<int>(weight_fp4.size(0));
    TORCH_CHECK(
        weight_fp4.size(1) == input_fp4.size(1) &&
            rows % 256 == 0 && reduction % 256 == 0 &&
            output_width % 256 == 0,
        "projection-N requires A=[M,K/2], B=[N,K/2], with M/K/N "
        "divisible by 256"
    );
    TORCH_CHECK(
        cluster_size == 2 || cluster_size == 4 ||
            cluster_size == 8 || cluster_size == 16,
        "cluster_size must be 2, 4, 8, or 16"
    );
    TORCH_CHECK(
        output_width % (cluster_size * 128) == 0,
        "output width must cover an integral projection-N supertile"
    );
    TORCH_CHECK(
        cluster_cap >= 0,
        "cluster_cap must be nonnegative; zero selects the occupancy cap"
    );
    TORCH_CHECK(
        input_scales.scalar_type() == at::kFloat8_e4m3fn &&
            weight_scales.scalar_type() == at::kFloat8_e4m3fn &&
            input_scales.is_cuda() && weight_scales.is_cuda() &&
            input_scales.is_contiguous() &&
            weight_scales.is_contiguous() &&
            input_scales.dim() == 3 && weight_scales.dim() == 3 &&
            input_scales.size(0) == rows / 128 &&
            input_scales.size(1) == reduction / 64 &&
            input_scales.size(2) == 512 &&
            weight_scales.size(0) == output_width / 128 &&
            weight_scales.size(1) == reduction / 64 &&
            weight_scales.size(2) == 512,
        "projection-N scales must use the canonical E4M3 NVFP4 pages"
    );
    TORCH_CHECK(
        input_global_scale.scalar_type() == at::kFloat &&
            weight_global_scale.scalar_type() == at::kFloat &&
            input_global_scale.is_cuda() && weight_global_scale.is_cuda() &&
            input_global_scale.numel() == 1 &&
            weight_global_scale.numel() == 1,
        "projection-N requires one float32 global scale per operand"
    );
    kittens::py::device_check(
        input_fp4,
        input_scales,
        input_global_scale,
        weight_fp4,
        weight_scales,
        weight_global_scale
    );
    const c10::cuda::CUDAGuard device_guard(input_fp4.device());
    const int cap = static_cast<int>(cluster_cap);
    if (cluster_size == 2) {
        return launch_projection<
            tkfa4_projection_n_multicast::config<2>
        >(
            input_fp4,
            input_scales,
            input_global_scale,
            weight_fp4,
            weight_scales,
            weight_global_scale,
            cap
        );
    }
    if (cluster_size == 4) {
        return launch_projection<
            tkfa4_projection_n_multicast::config<4>
        >(
            input_fp4,
            input_scales,
            input_global_scale,
            weight_fp4,
            weight_scales,
            weight_global_scale,
            cap
        );
    }
    if (cluster_size == 8) {
        return launch_projection<
            tkfa4_projection_n_multicast::config<8>
        >(
            input_fp4,
            input_scales,
            input_global_scale,
            weight_fp4,
            weight_scales,
            weight_global_scale,
            cap
        );
    }
    return launch_projection<
        tkfa4_projection_n_multicast::config<16>
    >(
        input_fp4,
        input_scales,
        input_global_scale,
        weight_fp4,
        weight_scales,
        weight_global_scale,
        cap
    );
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "tmem_pair_smoke",
        &tmem_pair_smoke,
        "Test independent cta_group::2 TMEM allocation in a larger cluster",
        pybind11::arg("cluster_size")
    );
    m.def(
        "tma_multicast_smoke",
        &tma_multicast_smoke,
        "Test A multicast and pair-local B loads in a larger cluster",
        pybind11::arg("input_fp4"),
        pybind11::arg("input_scales"),
        pybind11::arg("input_global_scale"),
        pybind11::arg("weight_fp4"),
        pybind11::arg("weight_scales"),
        pybind11::arg("weight_global_scale"),
        pybind11::arg("cluster_size")
    );
    m.def(
        "project_nvfp4_n_multicast",
        &project_nvfp4_n_multicast,
        "Query-owner projection-N NVFP4 multicast probe",
        pybind11::arg("input_fp4"),
        pybind11::arg("input_scales"),
        pybind11::arg("input_global_scale"),
        pybind11::arg("weight_fp4"),
        pybind11::arg("weight_scales"),
        pybind11::arg("weight_global_scale"),
        pybind11::arg("cluster_size") = 16,
        pybind11::arg("cluster_cap") = 0
    );
}
