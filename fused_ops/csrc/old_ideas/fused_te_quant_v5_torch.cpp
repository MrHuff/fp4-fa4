// PyTorch C++ wrapper for fused_te_quant_v5 kernel (2D TMA + double buffering)

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

void launch_fused_te_quant_v5(
    const nv_bfloat16* x, const nv_bfloat16* w,
    float epsilon, int rows, int cols,
    int act_mode, int scale_mode,
    unsigned char* y, __nv_fp8_e4m3* scales,
    float* global_scale, float* inv_rms_cache,
    float* block_amax_scratch
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
fused_te_quant_v5_forward(
    torch::Tensor input,       // [M, K] bf16
    torch::Tensor weight,      // [K] bf16
    float epsilon,
    int act_mode,              // 0=SiLU, 1=GeLU, 2=Identity
    int scale_mode             // 0=decode-centric, 1=encode-centric
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16);
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kBFloat16);
    TORCH_CHECK(input.dim() == 2);
    
    int M = input.size(0);
    int K = input.size(1);
    TORCH_CHECK(K % 128 == 0, "K must be multiple of 128 (TILE_K)");
    TORCH_CHECK(weight.size(0) == K);
    
    auto opts = input.options();
    auto out_fp4 = torch::empty({M, K / 2}, opts.dtype(torch::kUInt8));
    auto out_scales = torch::empty({M, K / 16}, opts.dtype(torch::kUInt8));
    auto global_scale = torch::zeros({1}, opts.dtype(torch::kFloat32));
    auto inv_rms = torch::empty({M}, opts.dtype(torch::kFloat32));
    auto block_amax = torch::empty({M, K / 16}, opts.dtype(torch::kFloat32));
    
    launch_fused_te_quant_v5(
        reinterpret_cast<const nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(weight.data_ptr()),
        epsilon, M, K,
        act_mode, scale_mode,
        out_fp4.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(out_scales.data_ptr<uint8_t>()),
        global_scale.data_ptr<float>(),
        inv_rms.data_ptr<float>(),
        block_amax.data_ptr<float>()
    );
    
    return std::make_tuple(out_fp4, out_scales, global_scale, inv_rms);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_full", &fused_te_quant_v5_forward,
          "Fused RMSNorm+Act+NVFP4 V5 (2D TMA, double-buf, encode/decode)");
}
