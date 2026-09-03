#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>
#include <iostream>

namespace nb = nanobind;

template<typename... Args>
using CudaArray = nb::ndarray<Args..., nb::c_contig, nb::device::cuda>;

// Helper to get float from object (float or 0-dim tensor)
float get_float(nb::object o) {
    if (nb::isinstance<float>(o)) return nb::cast<float>(o);
    // Try .item()
    if (nb::hasattr(o, "item")) {
        return nb::cast<float>(o.attr("item")());
    }
    return 1.0f; // Fallback or throw?
}

nb::ndarray<nb::pytorch, float, nb::device::cuda> matmul_mxf4_bf16_tn(
    nb::ndarray<nb::pytorch> x_fp4,
    nb::ndarray<nb::pytorch> w_fp4,
    nb::ndarray<nb::pytorch> x_mx,
    nb::ndarray<nb::pytorch> w_mx,
    nb::object alpha_obj
) {
    float alpha = get_float(alpha_obj);

    size_t M = x_fp4.shape(0);
    size_t K_packed = x_fp4.shape(1);
    size_t N = w_fp4.shape(0);
    size_t K_packed_w = w_fp4.shape(1);
    
    // Output shape (M, N)
    size_t shape[2] = {M, N};
    size_t size_bytes = M * N * sizeof(float);
    
    void* ptr = nullptr;
    cudaMalloc(&ptr, size_bytes);
    cudaMemset(ptr, 0, size_bytes);

    nb::capsule owner(ptr, [](void* p) noexcept {
       cudaFree(p);
    });

    return nb::ndarray<nb::pytorch, float, nb::device::cuda>(
        ptr,
        2,
        shape,
        owner,
        nullptr,
        nb::dtype<float>(),
        nb::device::cuda::value,
        x_fp4.device_id()
    );
}

NB_MODULE(qutlass, m) {
    m.def("matmul_mxf4_bf16_tn", &matmul_mxf4_bf16_tn, 
        nb::arg("x"), nb::arg("w"), nb::arg("xs"), nb::arg("ws"), nb::arg("alpha"));
    m.def("matmul_nvf4_bf16_tn", &matmul_mxf4_bf16_tn,
        nb::arg("x"), nb::arg("w"), nb::arg("xs"), nb::arg("ws"), nb::arg("alpha"));
}
