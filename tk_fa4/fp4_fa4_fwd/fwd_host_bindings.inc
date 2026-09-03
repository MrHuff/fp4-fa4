inline void check_fp4pv_p_tensor(const at::Tensor &t, int batch, int heads, int seqlen) {
    CHECK_INPUT(t);
    TORCH_CHECK(t.dtype() == at::ScalarType::Float4_e2m1fn_x2, "P_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(t.dim() == 4, "P_fp4 must have shape (batch, heads, seqlen, seqlen/2)");
    TORCH_CHECK(t.size(0) == batch && t.size(1) == heads && t.size(2) == seqlen && t.size(3) == seqlen / 2,
                "P_fp4 must have shape (batch, heads, seqlen, seqlen/2)");
}


// Forward-only host dispatch and bindings live with this source.
#include "shared_host_helpers.inc"
#include "fwd_host_dispatch.inc"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#include "fwd_pybind.inc"
}
