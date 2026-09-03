#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <stdexcept>
#include "fused_pre_quant.h"
#include "fused_backward_quant.h"
#include "fused_backward.h"
#include "fused_rmsnorm_act_quant.h"
#include "fused_rmsnorm_act_quant_v2.h"
#include "fused_rmsnorm_act_quant_v3.h"
#include "fused_rmsnorm_act_quant_v4.h"
#include "fused_rmsnorm_act_quant_opt.h"
#include "fused_backward_opt.h"
#include "fused_multiact.h"
#include "fused_mxnorm.h"
#include "fused_quantize_host.h"
#include "fused_te_quant.h"

// NEW: 2-Pass Implementation (No Global Sync)
// We declare the external function first (normally would be in a header)
extern void launch_fused_rmsnorm_act_quant_2pass(
    const nv_bfloat16* x, const nv_bfloat16* w,
    float epsilon, int rows, int cols, float scale_override, bool use_four_six,
    __nv_fp4x4_e2m1* y, __nv_fp8_e4m3* scales, float* global_scale, float* inv_rms_cache
);

extern void launch_fused_backward_shmem(
    const nv_bfloat16* grad_output,
    const nv_bfloat16* input,
    const nv_bfloat16* weight,
    const float* cached_inv_rms,
    float epsilon,
    int rows, int cols,
    nv_bfloat16* grad_input
);

extern "C" void fused_gemm_fp4_bf16_sm100(
    void* D_ptr, const void* A_ptr, const void* B_ptr,
    const void* ScaleA_ptr, const void* ScaleB_ptr,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream
);

// Qutlass kernels
extern void matmul_host_nvf4_bf16_tn(
    void* D_ptr, const void* A_ptr, const void* B_ptr,
    const void* A_sf_ptr, const void* B_sf_ptr,
    const void* alpha_ptr,
    int M, int N, int K,
    cudaStream_t stream
);



namespace nb = nanobind;

#define CHECK(X) if(!(X)) throw std::runtime_error(#X " failed");

template<typename T>
static void check_eq(T a, T b, const char* msg = nullptr) {
    if (a != b) throw std::runtime_error("Assertion failed");
}
#define CHECK_EQ(A, B) check_eq(A, B, #A " == " #B)

template<typename... Args>
using CudaArray = nb::ndarray<Args..., nb::c_contig, nb::device::cuda>;

// -------------------------------------------------------------------------
// NEW: Fused RMSNorm + Activation + Quantization (Single Cooperative Kernel)
// -------------------------------------------------------------------------

void fused_rmsnorm_act_quant_binding(
    const CudaArray<>& out,                                    // FP4 output
    const CudaArray<>& scales,                                 // E4M3 micro-scales
    const CudaArray<float, nanobind::shape<>>& global_scale,   // Global scale output
    const CudaArray<float>& inv_rms_cache,                     // Cached inv_rms for backward
    const CudaArray<nb::ro>& inp,                              // Input (bf16)
    const CudaArray<nb::ro>& weight,                           // RMSNorm weight (bf16)
    float epsilon,
    float scale_override,
    bool use_four_six
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK_EQ(weight.dtype(), bf16_dt);
    CHECK_EQ(inp.shape(1), weight.shape(0));
    CHECK_EQ(inv_rms_cache.shape(0), inp.shape(0));
    
    launch_fused_rmsnorm_act_quant(
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        epsilon,
        inp.shape(0), inp.shape(1),
        scale_override,
        use_four_six,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data(),
        const_cast<float*>(inv_rms_cache.data())
    );
}

// -------------------------------------------------------------------------
// V2: Optimized kernel that computes block absmaxes during first pass
// -------------------------------------------------------------------------

void fused_rmsnorm_act_quant_v2_binding(
    const CudaArray<>& out,                                    // FP4 output
    const CudaArray<>& scales,                                 // E4M3 micro-scales
    const CudaArray<float, nanobind::shape<>>& global_scale,   // Global scale output
    const CudaArray<float>& inv_rms_cache,                     // Cached inv_rms for backward
    const CudaArray<nb::ro>& inp,                              // Input (bf16)
    const CudaArray<nb::ro>& weight,                           // RMSNorm weight (bf16)
    float epsilon,
    float scale_override,
    bool use_four_six
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK_EQ(weight.dtype(), bf16_dt);
    CHECK_EQ(inp.shape(1), weight.shape(0));
    CHECK_EQ(inv_rms_cache.shape(0), inp.shape(0));
    
    launch_fused_rmsnorm_act_quant_v2(
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        epsilon,
        inp.shape(0), inp.shape(1),
        scale_override,
        use_four_six,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data(),
        const_cast<float*>(inv_rms_cache.data())
    );
}

// -------------------------------------------------------------------------
// V3: Warp shuffles + inv_rms factored into quantization scale
// -------------------------------------------------------------------------

void fused_rmsnorm_act_quant_v3_binding(
    const CudaArray<>& out,
    const CudaArray<>& scales,
    const CudaArray<float, nanobind::shape<>>& global_scale,
    const CudaArray<float>& inv_rms_cache,
    const CudaArray<nb::ro>& inp,
    const CudaArray<nb::ro>& weight,
    float epsilon,
    float scale_override,
    bool use_four_six
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK_EQ(weight.dtype(), bf16_dt);
    CHECK_EQ(inp.shape(1), weight.shape(0));
    CHECK_EQ(inv_rms_cache.shape(0), inp.shape(0));
    
    launch_fused_rmsnorm_act_quant_v3(
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        epsilon,
        inp.shape(0), inp.shape(1),
        scale_override,
        use_four_six,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data(),
        const_cast<float*>(inv_rms_cache.data())
    );
}

// -------------------------------------------------------------------------
// V4: Lock-free with exclusive block ownership (no atomics)
// -------------------------------------------------------------------------

void fused_rmsnorm_act_quant_v4_binding(
    const CudaArray<>& out,
    const CudaArray<>& scales,
    const CudaArray<float, nanobind::shape<>>& global_scale,
    const CudaArray<float>& inv_rms_cache,
    const CudaArray<nb::ro>& inp,
    const CudaArray<nb::ro>& weight,
    float epsilon,
    float scale_override,
    bool use_four_six
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK_EQ(weight.dtype(), bf16_dt);
    CHECK_EQ(inp.shape(1), weight.shape(0));
    CHECK_EQ(inv_rms_cache.shape(0), inp.shape(0));
    
    launch_fused_rmsnorm_act_quant_v4(
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        epsilon,
        inp.shape(0), inp.shape(1),
        scale_override,
        use_four_six,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data(),
        const_cast<float*>(inv_rms_cache.data())
    );
}

// -------------------------------------------------------------------------
// V1-OPT: Optimized V1 with faster math intrinsics
// -------------------------------------------------------------------------

void fused_rmsnorm_act_quant_opt_binding(
    const CudaArray<>& out,
    const CudaArray<>& scales,
    const CudaArray<float, nanobind::shape<>>& global_scale,
    const CudaArray<float>& inv_rms_cache,
    const CudaArray<nb::ro>& inp,
    const CudaArray<nb::ro>& weight,
    float epsilon,
    float scale_override,
    bool use_four_six
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK_EQ(weight.dtype(), bf16_dt);
    CHECK_EQ(inp.shape(1), weight.shape(0));
    CHECK_EQ(inv_rms_cache.shape(0), inp.shape(0));
    
    launch_fused_rmsnorm_act_quant_opt(
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        epsilon,
        inp.shape(0), inp.shape(1),
        scale_override,
        use_four_six,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data(),
        const_cast<float*>(inv_rms_cache.data())
    );
}

// -------------------------------------------------------------------------
// Multi-Activation: Supports SiLU (0), ReLU² (1), GELU (2), ELU (3)
// -------------------------------------------------------------------------

void fused_rmsnorm_act_quant_multiact_binding(
    const CudaArray<>& out,
    const CudaArray<>& scales,
    const CudaArray<float, nanobind::shape<>>& global_scale,
    const CudaArray<float>& inv_rms_cache,
    const CudaArray<>& inp,
    const CudaArray<>& weight,
    float epsilon,
    float scale_override,
    bool use_four_six,
    int activation_type
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
    
    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK_EQ(weight.dtype(), bf16_dt);
    CHECK_EQ(inp.shape(1), weight.shape(0));
    CHECK_EQ(inv_rms_cache.shape(0), inp.shape(0));
    
    launch_fused_rmsnorm_act_quant_multiact_dispatch(
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        epsilon,
        inp.shape(0), inp.shape(1),
        scale_override,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data(),
        const_cast<float*>(inv_rms_cache.data()),
        use_four_six,
        activation_type
    );
}

// -------------------------------------------------------------------------
// MXNorm Binding
// -------------------------------------------------------------------------

void fused_mxnorm_binding(
    const CudaArray<>& out,
    const CudaArray<>& scales,
    const CudaArray<float, nanobind::shape<>>& global_scale,
    const CudaArray<float>& inv_rms_cache,
    const CudaArray<nb::ro>& inp,
    const CudaArray<nb::ro>& weight,
    float epsilon,
    float scale_override,
    bool use_four_six,
    int norm_mode
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
    
    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK_EQ(weight.dtype(), bf16_dt);
    CHECK_EQ(inp.shape(1), weight.shape(0));
    CHECK_EQ(inv_rms_cache.shape(0), inp.shape(0));
    
    launch_fused_mxnorm(
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        epsilon,
        inp.shape(0), inp.shape(1),
        scale_override,
        use_four_six,
        norm_mode,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data(),
        const_cast<float*>(inv_rms_cache.data())
    );
}

// -------------------------------------------------------------------------
// MXNorm Binding
// -------------------------------------------------------------------------



// Fast MXNorm Binding
void fused_mxnorm_fast_binding(
    const CudaArray<>& out,
    const CudaArray<>& scales,
    const CudaArray<float, nanobind::shape<>>& global_scale,
    const CudaArray<float>& inv_rms_cache,
    const CudaArray<nb::ro>& inp,
    const CudaArray<nb::ro>& weight,
    float epsilon,
    float scale_override,
    bool use_four_six
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
    
    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK_EQ(weight.dtype(), bf16_dt);
    CHECK_EQ(inp.shape(1), weight.shape(0));
    CHECK_EQ(inv_rms_cache.shape(0), inp.shape(0));
    
    launch_fused_mxnorm_fast(
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        epsilon,
        inp.shape(0), inp.shape(1),
        scale_override,
        use_four_six,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data(),
        const_cast<float*>(inv_rms_cache.data())
    );
}

// Fused Backward AbsMax Binding
void fused_backward_absmax_binding(
    const CudaArray<>& grad_input,  // Output: dx
    const CudaArray<nb::ro>& grad_output, // Input: dy
    const CudaArray<nb::ro>& input,       // Input: x
    const CudaArray<nb::ro>& weight,      // Input: w
    const CudaArray<float, nb::ro>& inv_rms_cache // Input: 1/s
) {
    launch_fused_backward_absmax(
        reinterpret_cast<const nv_bfloat16*>(grad_output.data()),
        reinterpret_cast<const nv_bfloat16*>(input.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        inv_rms_cache.data(),
        input.shape(0), input.shape(1),
        reinterpret_cast<nv_bfloat16*>(grad_input.data())
    );
}

// Fast MXNorm Block Binding
void fused_mxnorm_fast_block_binding(
    const CudaArray<>& out,
    const CudaArray<>& scales,
    const CudaArray<float, nanobind::shape<>>& global_scale,
    const CudaArray<float>& inv_rms_cache,
    const CudaArray<nb::ro>& inp,
    const CudaArray<nb::ro>& weight,
    float epsilon,
    float scale_override,
    bool use_four_six
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
    
    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK_EQ(weight.dtype(), bf16_dt);
    CHECK_EQ(inp.shape(1), weight.shape(0));
    CHECK_EQ(inv_rms_cache.shape(0), inp.shape(0));
    
    launch_fused_mxnorm_fast_block(
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        epsilon,
        inp.shape(0), inp.shape(1),
        scale_override,
        use_four_six,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data(),
        const_cast<float*>(inv_rms_cache.data())
    );
}

// -------------------------------------------------------------------------
// Legacy bindings (for backward compatibility)
// -------------------------------------------------------------------------

void fused_pre_quant_binding_v2(
    const CudaArray<>& out,
    const CudaArray<>& scales,
    const CudaArray<float, nanobind::shape<>>& global_scale,
    const CudaArray<float>& inv_rms_cache,
    const CudaArray<nb::ro>& inp,
    const CudaArray<nb::ro>& weight,
    float epsilon,
    const CudaArray<float, nanobind::shape<>>& global_amax,
    float scale_override
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK_EQ(weight.dtype(), bf16_dt);
    CHECK_EQ(inp.shape(1), weight.shape(0));
    CHECK_EQ(inv_rms_cache.shape(0), inp.shape(0));
    
    launch_fused_pre_quant(
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        epsilon,
        inp.shape(0), inp.shape(1),
        global_amax.data(),
        scale_override,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data(),
        const_cast<float*>(inv_rms_cache.data())
    );
}

void fused_pre_quant_binding(
    const CudaArray<>& out,
    const CudaArray<>& scales,
    const CudaArray<float, nanobind::shape<>>& global_scale,
    const CudaArray<nb::ro>& inp,
    const CudaArray<nb::ro>& weight,
    float epsilon,
    const CudaArray<float, nanobind::shape<>>& global_amax,
    float scale_override
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};

    CHECK_EQ(out.shape(0), inp.shape(0));
    CHECK_EQ(inp.dtype(), bf16_dt);
    CHECK_EQ(weight.dtype(), bf16_dt);
    CHECK_EQ(inp.shape(1), weight.shape(0));
    
    launch_fused_pre_quant(
        reinterpret_cast<const nv_bfloat16*>(inp.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        epsilon,
        inp.shape(0), inp.shape(1),
        global_amax.data(),
        scale_override,
        reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
        global_scale.data()
    );
}

// Backward bindings
void fused_backward_binding_v2(
    const CudaArray<>& grad_output,
    const CudaArray<>& input,
    const CudaArray<>& weight,
    const CudaArray<float>& cached_inv_rms,
    float epsilon,
    const CudaArray<>& grad_input
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
    CHECK_EQ(grad_output.dtype(), bf16_dt);
    CHECK_EQ(input.dtype(), bf16_dt);
    CHECK_EQ(grad_input.dtype(), bf16_dt);
    CHECK_EQ(input.shape(0), grad_input.shape(0));
    CHECK_EQ(cached_inv_rms.shape(0), input.shape(0));
    
    launch_fused_backward(
        reinterpret_cast<const nv_bfloat16*>(grad_output.data()),
        reinterpret_cast<const nv_bfloat16*>(input.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        cached_inv_rms.data(),
        epsilon,
        input.shape(0), input.shape(1),
        reinterpret_cast<nv_bfloat16*>(grad_input.data())
    );
}

// Optimized backward binding
void fused_backward_binding_opt(
    const CudaArray<>& grad_output,
    const CudaArray<>& input,
    const CudaArray<>& weight,
    const CudaArray<float>& cached_inv_rms,
    float epsilon,
    const CudaArray<>& grad_input
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
    CHECK_EQ(grad_output.dtype(), bf16_dt);
    CHECK_EQ(input.dtype(), bf16_dt);
    CHECK_EQ(grad_input.dtype(), bf16_dt);
    CHECK_EQ(input.shape(0), grad_input.shape(0));
    CHECK_EQ(cached_inv_rms.shape(0), input.shape(0));
    
    launch_fused_backward_opt(
        reinterpret_cast<const nv_bfloat16*>(grad_output.data()),
        reinterpret_cast<const nv_bfloat16*>(input.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        cached_inv_rms.data(),
        epsilon,
        input.shape(0), input.shape(1),
        reinterpret_cast<nv_bfloat16*>(grad_input.data())
    );
}

void fused_backward_binding(
    const CudaArray<>& grad_output,
    const CudaArray<>& input,
    const CudaArray<>& weight,
    float epsilon,
    const CudaArray<>& grad_input
) {
    nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
    CHECK_EQ(grad_output.dtype(), bf16_dt);
    CHECK_EQ(input.dtype(), bf16_dt);
    CHECK_EQ(grad_input.dtype(), bf16_dt);
    CHECK_EQ(input.shape(0), grad_input.shape(0));
    
    launch_fused_backward(
        reinterpret_cast<const nv_bfloat16*>(grad_output.data()),
        reinterpret_cast<const nv_bfloat16*>(input.data()),
        reinterpret_cast<const nv_bfloat16*>(weight.data()),
        epsilon,
        input.shape(0), input.shape(1),
        reinterpret_cast<nv_bfloat16*>(grad_input.data())
    );
}

NB_MODULE(_fused_ops, m) {
    // PRIMARY: New single-kernel fused implementation
    m.def("fused_rmsnorm_act_quant", &fused_rmsnorm_act_quant_binding, 
        nb::arg("out"), nb::arg("scales"), nb::arg("global_scale_out"), 
        nb::arg("inv_rms_cache"),
        nb::arg("input"), nb::arg("weight"), 
        nb::arg("epsilon"), nb::arg("scale_override"),
        nb::arg("use_four_six") = true,
        "Fused RMSNorm + SiLU + FP4 Quantization (single cooperative kernel)");

    // V2: Optimized - computes block absmaxes during first pass
    m.def("fused_rmsnorm_act_quant_v2", &fused_rmsnorm_act_quant_v2_binding, 
        nb::arg("out"), nb::arg("scales"), nb::arg("global_scale_out"), 
        nb::arg("inv_rms_cache"),
        nb::arg("input"), nb::arg("weight"), 
        nb::arg("epsilon"), nb::arg("scale_override"),
        nb::arg("use_four_six") = true,
        "Optimized Fused RMSNorm + SiLU + FP4 (block absmaxes computed in first pass)");

    // V3: Warp shuffles + inv_rms factored into quantization
    m.def("fused_rmsnorm_act_quant_v3", &fused_rmsnorm_act_quant_v3_binding, 
        nb::arg("out"), nb::arg("scales"), nb::arg("global_scale_out"), 
        nb::arg("inv_rms_cache"),
        nb::arg("input"), nb::arg("weight"), 
        nb::arg("epsilon"), nb::arg("scale_override"),
        nb::arg("use_four_six") = true,
        "V3: Warp shuffles + inv_rms factored into quant scale");

    // V4: Lock-free with exclusive block ownership (no atomics)
    m.def("fused_rmsnorm_act_quant_v4", &fused_rmsnorm_act_quant_v4_binding, 
        nb::arg("out"), nb::arg("scales"), nb::arg("global_scale_out"), 
        nb::arg("inv_rms_cache"),
        nb::arg("input"), nb::arg("weight"), 
        nb::arg("epsilon"), nb::arg("scale_override"),
        nb::arg("use_four_six") = true,
        "V4: Lock-free, exclusive block ownership (no atomics)");

    // V1-OPT: Optimized V1 with faster math intrinsics
    m.def("fused_rmsnorm_act_quant_opt", &fused_rmsnorm_act_quant_opt_binding, 
        nb::arg("out"), nb::arg("scales"), nb::arg("global_scale_out"), 
        nb::arg("inv_rms_cache"),
        nb::arg("input"), nb::arg("weight"), 
        nb::arg("epsilon"), nb::arg("scale_override"),
        nb::arg("use_four_six") = true,
        "V1-OPT: Fast math intrinsics (__expf)");

    // Legacy forward bindings
    m.def("fused_pre_quant_v2", &fused_pre_quant_binding_v2, 
        nb::arg("out"), nb::arg("scales"), nb::arg("global_scale_out"), 
        nb::arg("inv_rms_cache"),
        nb::arg("input"), nb::arg("weight"), nb::arg("epsilon"), 
        nb::arg("global_amax"), nb::arg("scale_override"),
        "Legacy fused pre-quant with inv_rms caching");

    m.def("fused_backward_v2", &fused_backward_binding_v2, 
        nb::arg("grad_output"), nb::arg("input"), nb::arg("weight"), 
        nb::arg("cached_inv_rms"), nb::arg("epsilon"), nb::arg("grad_input"),
        "Backward pass using cached inv_rms");

    m.def("fused_backward", &fused_backward_binding, 
        nb::arg("grad_output"), nb::arg("input"), nb::arg("weight"), 
        nb::arg("epsilon"), nb::arg("grad_input"),
        "Legacy backward pass (recomputes inv_rms)");

    m.def("fused_backward_opt", &fused_backward_binding_opt, 
        nb::arg("grad_output"), nb::arg("input"), nb::arg("weight"), 
        nb::arg("cached_inv_rms"), nb::arg("epsilon"), nb::arg("grad_input"),
        "Optimized backward pass (__expf)");

    m.def("fused_rmsnorm_act_quant_multiact", &fused_rmsnorm_act_quant_multiact_binding,
        nb::arg("out"), nb::arg("scales"), nb::arg("global_scale"),
        nb::arg("inv_rms_cache"),
        nb::arg("input"), nb::arg("weight"), nb::arg("epsilon"), nb::arg("scale_override"),
        nb::arg("use_four_six") = true,
        nb::arg("activation_type") = 0,
        "Multi-activation: 0=SiLU, 1=ReLU², 2=GELU, 3=ELU");

    m.def("fused_mxnorm", &fused_mxnorm_binding,
        nb::arg("out"), nb::arg("scales"), nb::arg("global_scale"),
        nb::arg("inv_rms_cache"),
        nb::arg("input"), nb::arg("weight"), nb::arg("epsilon"), nb::arg("scale_override"),
        nb::arg("use_four_six") = true,
        nb::arg("norm_mode") = 0,
        "MXNorm: 0=Average (RMS), 1=Max (AbsMax), 2=BlockMax");

    m.def("fused_mxnorm_fast", &fused_mxnorm_fast_binding,
        nb::arg("out"), nb::arg("scales"), nb::arg("global_scale"),
        nb::arg("inv_rms_cache"),
        nb::arg("input"), nb::arg("weight"), nb::arg("epsilon"), nb::arg("scale_override"),
        nb::arg("use_four_six") = true,
        "Fast MXNorm (AbsMax Only) - Specialized Kernel");

    m.def("fused_mxnorm_fast_block", &fused_mxnorm_fast_block_binding,
        nb::arg("out"), nb::arg("scales"), nb::arg("global_scale"),
        nb::arg("inv_rms_cache"),
        nb::arg("input"), nb::arg("weight"), nb::arg("epsilon"), nb::arg("scale_override"),
        nb::arg("use_four_six") = true,
        "Fast MXNorm (Block-Max Only) - Specialized Kernel");

    m.def("fused_backward_absmax", &fused_backward_absmax_binding,
        nb::arg("grad_input"), nb::arg("grad_output"), 
        nb::arg("input"), nb::arg("weight"), nb::arg("inv_rms_cache"),
        "Exact Backward Pass for AbsMax Norm (Sparse Update)");

    m.def("fused_rmsnorm_act_quant_2pass", 
        [](
            const CudaArray<>& out,
            const CudaArray<>& scales,
            const CudaArray<float, nanobind::shape<>>& global_scale,
            const CudaArray<float>& inv_rms_cache,
            const CudaArray<nb::ro>& inp,
            const CudaArray<nb::ro>& weight,
            float epsilon,
            float scale_override,
            bool use_four_six
        ) {
            nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
            CHECK_EQ(out.shape(0), inp.shape(0));
            CHECK_EQ(inp.dtype(), bf16_dt);
            CHECK_EQ(weight.dtype(), bf16_dt);
            CHECK_EQ(inp.shape(1), weight.shape(0));
            CHECK_EQ(inv_rms_cache.shape(0), inp.shape(0));
            
            launch_fused_rmsnorm_act_quant_2pass(
                reinterpret_cast<const nv_bfloat16*>(inp.data()),
                reinterpret_cast<const nv_bfloat16*>(weight.data()),
                epsilon, inp.shape(0), inp.shape(1), scale_override, use_four_six,
                reinterpret_cast<__nv_fp4x4_e2m1*>(out.data()),
                reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
                global_scale.data(),
                const_cast<float*>(inv_rms_cache.data())
            );
        },
        nb::arg("out"), nb::arg("scales"), nb::arg("global_scale_out"), 
        nb::arg("inv_rms_cache"),
        nb::arg("input"), nb::arg("weight"), 
        nb::arg("epsilon"), nb::arg("scale_override"),
        nb::arg("use_four_six") = true,
        "2-Pass Fused Kernel (No Global Sync) - Better for small shapes"
    );
    
    m.def("fused_backward_shmem", 
        [](
            const CudaArray<nb::ro>& grad_output,
            const CudaArray<nb::ro>& input,
            const CudaArray<nb::ro>& weight,
            const CudaArray<float>* cached_inv_rms,
            float epsilon,
            const CudaArray<>& grad_input
        ) {
             // ...
        },
        nb::arg("grad_output"), nb::arg("input"), nb::arg("weight"),
        nb::arg("cached_inv_rms").none() = nb::none(),
        nb::arg("epsilon"), nb::arg("grad_input"),
        "Single-Pass Shmem Fused Backward (Optimize for cols <= 8192)"
    );

    // FP4 GEMM Binding
    m.def("fused_gemm_fp4", [](
        nb::object D_obj,
        nb::object A_obj,
        nb::object B_obj,
        nb::object ScaleA_obj,
        nb::object ScaleB_obj,
        float alpha
    ) {
        // Cast to CudaArray<> (generic shape/dtype) or check manually
        // We just need data pointer and shape
        auto D = nb::cast<CudaArray<>>(D_obj);
        auto A = nb::cast<CudaArray<uint8_t>>(A_obj);
        // B, ScaleA, ScaleB are also uint8
        auto B = nb::cast<CudaArray<uint8_t>>(B_obj);
        auto ScaleA = nb::cast<CudaArray<uint8_t>>(ScaleA_obj);
        auto ScaleB = nb::cast<CudaArray<uint8_t>>(ScaleB_obj);

        int M = D.shape(0);
        int N = D.shape(1);
        int K = A.shape(1) * 2; 
        
        fused_gemm_fp4_bf16_sm100(
            D.data(),
            A.data(),
            B.data(),
            ScaleA.data(),
            ScaleB.data(),
            M, N, K,
            alpha, 0.0f,
            0 
        );
    }, nb::arg("D"), nb::arg("A"), nb::arg("B"), nb::arg("ScaleA"), nb::arg("ScaleB"), nb::arg("alpha"));

    // Qutlass GEMM Binding
    m.def("matmul_nvf4_bf16_tn", [](
        nb::object D_obj, nb::object A_obj, nb::object B_obj,
        nb::object ScaleA_obj, nb::object ScaleB_obj, float alpha
    ) {
        auto D = nb::cast<CudaArray<>>(D_obj);
        auto A = nb::cast<CudaArray<uint8_t>>(A_obj);
        auto B = nb::cast<CudaArray<uint8_t>>(B_obj);
        auto ScaleA = nb::cast<CudaArray<uint8_t>>(ScaleA_obj);
        auto ScaleB = nb::cast<CudaArray<uint8_t>>(ScaleB_obj);
        
        int M = D.shape(0);
        int N = D.shape(1);
        int K = A.shape(1) * 2;
        
        matmul_host_nvf4_bf16_tn(
            D.data(), A.data(), B.data(), ScaleA.data(), ScaleB.data(),
            &alpha, M, N, K, 0); 
    }, nb::arg("D"), nb::arg("A"), nb::arg("B"), nb::arg("ScaleA"), nb::arg("ScaleB"), nb::arg("alpha"));

    // Qutlass Quantize Binding
    m.def("fusedQuantizeNvAbsMax", [](
        nb::object D_obj, nb::object Scale_obj,
        nb::object A_obj, nb::object B_obj, nb::object GScale_obj
    ) {
        auto D = nb::cast<CudaArray<uint8_t>>(D_obj);
        auto Scale = nb::cast<CudaArray<uint8_t>>(Scale_obj);
        auto A = nb::cast<CudaArray<>>(A_obj);
        auto B = nb::cast<CudaArray<>>(B_obj);
        auto GScale = nb::cast<CudaArray<float>>(GScale_obj);

        QUTLASS::fusedQuantizeNvAbsMax_host_sm100(
            D.data(), Scale.data(), A.data(), B.data(), GScale.data(),
            A.size(), B.shape(1), 0);
    }, nb::arg("D"), nb::arg("Scale"), nb::arg("A"), nb::arg("B"), nb::arg("GScale"));

    // Fused Backward Quant: dequant → transpose → requant (no Hadamard)
    // TE-Compatible Fused Quant
    m.def("fused_te_quant", [](
        const CudaArray<>& out,
        const CudaArray<>& scales,
        const CudaArray<float, nanobind::shape<>>& global_scale,
        const CudaArray<float>& inv_rms_cache,
        const CudaArray<nb::ro>& inp,
        const CudaArray<nb::ro>& weight,
        float epsilon,
        int norm_mode,
        int act_mode
    ) {
        nb::dlpack::dtype bf16_dt{static_cast<std::uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
        CHECK_EQ(out.shape(0), inp.shape(0));
        CHECK_EQ(inp.dtype(), bf16_dt);
        CHECK_EQ(weight.dtype(), bf16_dt);
        CHECK_EQ(inp.shape(1), weight.shape(0));
        CHECK_EQ(inv_rms_cache.shape(0), inp.shape(0));

        launch_fused_te_quant(
            reinterpret_cast<const nv_bfloat16*>(inp.data()),
            reinterpret_cast<const nv_bfloat16*>(weight.data()),
            epsilon,
            inp.shape(0), inp.shape(1),
            norm_mode, act_mode,
            reinterpret_cast<unsigned char*>(out.data()),
            reinterpret_cast<__nv_fp8_e4m3*>(scales.data()),
            global_scale.data(),
            const_cast<float*>(inv_rms_cache.data())
        );
    },
    nb::arg("out"), nb::arg("scales"), nb::arg("global_scale"),
    nb::arg("inv_rms_cache"),
    nb::arg("input"), nb::arg("weight"),
    nb::arg("epsilon") = 1e-5f,
    nb::arg("norm_mode") = 0,
    nb::arg("act_mode") = 0,
    "TE-compatible fused RMSNorm+Act+NVFP4 quant. norm: 0=RMS,1=AbsMax,2=MXNorm. act: 0=SiLU,1=GeLU,2=Identity");

    m.def("dequant_transpose_quant", [](
        const CudaArray<>& out,           // FP4 output [K, N/2]
        const CudaArray<>& out_scales,    // FP8 scales [K, N/16]
        const CudaArray<float, nanobind::shape<>>& out_global_scale, // scalar
        const CudaArray<float, nanobind::shape<>>& scratch_amax,     // scratch float
        const CudaArray<nb::ro>& inp,     // FP4 input [N, K/2]
        const CudaArray<nb::ro>& inp_scales,  // FP8 scales [N, K/16]
        const CudaArray<float, nanobind::shape<>>& inp_global_scale, // scalar
        float scale_override
    ) {
        // Input is [N, K/2] packed FP4 → logical [N, K]
        int N = inp.shape(0);
        int K = inp.shape(1) * 2;  // packed FP4: K/2 bytes = K elements
        
        launch_dequant_transpose_quant(
            reinterpret_cast<__nv_fp4x2_storage_t*>(out.data()),
            reinterpret_cast<__nv_fp8_e4m3*>(out_scales.data()),
            out_global_scale.data(),
            reinterpret_cast<const __nv_fp4x2_storage_t*>(inp.data()),
            reinterpret_cast<const __nv_fp8_e4m3*>(inp_scales.data()),
            inp_global_scale.data(),
            scratch_amax.data(),
            N, K, scale_override
        );
    },
    nb::arg("out"), nb::arg("out_scales"), nb::arg("out_global_scale"),
    nb::arg("scratch_amax"),
    nb::arg("input"), nb::arg("input_scales"), nb::arg("input_global_scale"),
    nb::arg("scale_override") = 1.0f,
    "Fused dequant → transpose → requant (no Hadamard). Input [N,K/2] → Output [K,N/2].");
}
