// Minimal PyTorch C++ wrapper for fused_te_quant kernel
// Built via torch.utils.cpp_extension, no nanobind needed.

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

// Declared in fused_te_quant.cu
void launch_fused_te_quant(
    const nv_bfloat16* x, const nv_bfloat16* w,
    float epsilon, int rows, int cols,
    int norm_mode, int act_mode,
    unsigned char* y, __nv_fp8_e4m3* scales,
    float* global_scale, float* inv_rms_cache
);

torch::Tensor fused_te_quant_forward(
    torch::Tensor input,       // [M, K] bf16
    torch::Tensor weight,      // [K] bf16
    float epsilon,
    int norm_mode,             // 0=RMS, 1=AbsMax, 2=MXNorm
    int act_mode               // 0=SiLU, 1=GeLU, 2=Identity
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16);
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kBFloat16);
    TORCH_CHECK(input.dim() == 2);
    
    int M = input.size(0);
    int K = input.size(1);
    TORCH_CHECK(K % 16 == 0, "K must be multiple of 16");
    TORCH_CHECK(weight.size(0) == K);
    
    // Allocate outputs in TE-compatible format
    auto out_fp4 = torch::empty({M, K / 2}, input.options().dtype(torch::kUInt8));
    auto out_scales = torch::empty({M, K / 16}, input.options().dtype(torch::kUInt8));
    auto global_scale = torch::zeros({1}, input.options().dtype(torch::kFloat32));
    auto inv_rms = torch::empty({M}, input.options().dtype(torch::kFloat32));
    
    launch_fused_te_quant(
        reinterpret_cast<const nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(weight.data_ptr()),
        epsilon, M, K,
        norm_mode, act_mode,
        out_fp4.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(out_scales.data_ptr<uint8_t>()),
        global_scale.data_ptr<float>(),
        inv_rms.data_ptr<float>()
    );
    
    return out_fp4;  // Returns packed FP4; scales/global_scale accessible via closure
}

// Version that returns all outputs as a tuple
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
fused_te_quant_forward_full(
    torch::Tensor input,
    torch::Tensor weight,
    float epsilon,
    int norm_mode,
    int act_mode
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16);
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kBFloat16);
    TORCH_CHECK(input.dim() == 2);
    
    int M = input.size(0);
    int K = input.size(1);
    TORCH_CHECK(K % 16 == 0, "K must be multiple of 16");
    TORCH_CHECK(weight.size(0) == K);
    
    auto out_fp4 = torch::empty({M, K / 2}, input.options().dtype(torch::kUInt8));
    auto out_scales = torch::empty({M, K / 16}, input.options().dtype(torch::kUInt8));
    auto global_scale = torch::zeros({1}, input.options().dtype(torch::kFloat32));
    auto inv_rms = torch::empty({M}, input.options().dtype(torch::kFloat32));
    
    launch_fused_te_quant(
        reinterpret_cast<const nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(weight.data_ptr()),
        epsilon, M, K,
        norm_mode, act_mode,
        out_fp4.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(out_scales.data_ptr<uint8_t>()),
        global_scale.data_ptr<float>(),
        inv_rms.data_ptr<float>()
    );
    
    return std::make_tuple(out_fp4, out_scales, global_scale, inv_rms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_te_quant_forward, 
          "Fused RMSNorm+Act+NVFP4 (returns packed FP4)");
    m.def("forward_full", &fused_te_quant_forward_full,
          "Fused RMSNorm+Act+NVFP4 (returns fp4, scales, global_scale, inv_rms)");
}
