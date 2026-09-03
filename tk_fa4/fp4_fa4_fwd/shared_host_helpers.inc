inline void check_fp4pv_p_scale_tensor(const at::Tensor &t, int batch, int heads, int seqlen) {
    CHECK_INPUT(t);
    TORCH_CHECK(t.dtype() == at::ScalarType::Float8_e4m3fn, "P_sc_prepared must be float8_e4m3fn");
    TORCH_CHECK(t.dim() == 5, "P_sc_prepared must have shape (batch, heads, seqlen/128, seqlen/64, 512)");
    TORCH_CHECK(t.size(0) == batch && t.size(1) == heads && t.size(2) == seqlen / 128 && t.size(3) == seqlen / 64 && t.size(4) == 512,
                "P_sc_prepared must have shape (batch, heads, seqlen/128, seqlen/64, 512)");
}

inline void check_fp4pv_mxfp4_v_scale_tensor(const at::Tensor &t, int batch, int heads, int seqlen, int dvo) {
    CHECK_INPUT(t);
    TORCH_CHECK(t.dtype() == at::ScalarType::Byte, "MXFP4 V_sc must be uint8 E8M0");
    TORCH_CHECK(t.dim() == 6, "MXFP4 V_sc must have shape (batch, heads, scale_rows, seqlen/128, 32, 16)");
    TORCH_CHECK(t.size(0) == batch && t.size(1) == heads && t.size(2) >= dvo / 128 &&
                t.size(3) == seqlen / 128 && t.size(4) == 32 && t.size(5) == 16,
                "MXFP4 V_sc must have shape (batch, heads, >=dvo/128, seqlen/128, 32, 16)");
}

inline void check_fp4pv_mxfp4_v_scale_half_contract(
    const at::Tensor &t, int batch, int heads, int seqlen) {
    check_fp4pv_mxfp4_v_scale_tensor(t, batch, heads, seqlen, 128);
    TORCH_CHECK(
        t.size(2) == 2,
        "MXFP4 forward V_sc must use canonical "
        "(batch,head,output_half,key_scale_tile,scale_row,scale_col) "
        "layout with exactly two records per D128 head");
}
