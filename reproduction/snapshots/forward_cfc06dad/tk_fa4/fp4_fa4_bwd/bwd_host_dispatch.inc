#include <c10/cuda/CUDAGuard.h>

#if !defined(TK_FA4_FORWARD_ONLY_BUILD) && !defined(TK_FA4_DQ_ONLY_BUILD)
void dispatch_backward_dv_streaming_live_localcta(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc_prepared,
    at::Tensor O,
    at::Tensor LSE,
    bool persistent_launch
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(O); CHECK_INPUT(LSE);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");
    TORCH_CHECK(O.dim() == 4, "O must be rank-4");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch && O.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads && O.size(2) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen && O.size(1) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(O.size(3) == 128, "O head_dim must be 128");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4 && LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");

    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, 128);
    check_fp4pv_v_scale_tensor(V_sc_prepared, batch, heads, seqlen, 128);

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_fp4pv<C, !FP4PV_DIAG_USE_DIRECT_ROW_UPDATE>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE)
    };

    const int tiles_m = seqlen / C::Mb;
    const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
    const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
    const int total_bids = batch * heads * num_block * C::CLUSTER_SIZE;
    const bool use_persistent = persistent_launch && (fp4pv_force_persistent_grid<C>::value || total_bids <= 2 * C::NUM_SM);
    const dim3 grid = use_persistent ? bwd_persistent_dim_fp4pv_exp<C>(batch, heads, seqlen) : bwd_fullgrid_dim_fp4pv_exp<C>(batch, heads, seqlen);
    const int dynamic_shared_memory = g.dynamic_shared_memory();
    LaunchConfig<true, false> launch_config(grid, g.block(), dynamic_shared_memory, at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaFuncSetAttribute(
        kernel_streaming_live_fp4pv<C, false, 3, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dynamic_shared_memory));
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_streaming_live_fp4pv<C, false, 3, true>,
        g,
        3,
        0,
        -1));
}
#endif
#ifndef TK_FA4_FORWARD_ONLY_BUILD
#ifndef TK_FA4_DQ_ONLY_BUILD
template <typename C, int MODE>
static inline void launch_backward_dv_streaming_live_mxfp4_config(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc,
    at::Tensor O,
    at::Tensor LSE,
    int batch,
    int heads,
    int seqlen,
    int k_sc_depth,
    double p_payload_scale,
    int p_quant_mode,
    bool persistent_launch
) {
    using G = globals_fp4pv_mxfp4_dv<C, false>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc.data_ptr()),
        kittens::py::tensor_to_gl<typename G::o_gl>(O),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        static_cast<float>(1.0 / p_payload_scale),
        p_quant_mode
    };

    const int tiles_m = seqlen / C::Mb;
    const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
    const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
    const int total_bids = batch * heads * num_block * C::CLUSTER_SIZE;
    const bool use_persistent = persistent_launch && (fp4pv_force_persistent_grid<C>::value || total_bids <= 2 * C::NUM_SM);
    const dim3 grid = use_persistent ? bwd_persistent_dim_fp4pv_exp<C>(batch, heads, seqlen) : bwd_fullgrid_dim_fp4pv_exp<C>(batch, heads, seqlen);
    const int dynamic_shared_memory =
        (MODE == 6 || MODE == 178) ? G::K256_DYNAMIC_SHARED_MEMORY_PADDED : g.dynamic_shared_memory();
    LaunchConfig<true, false> launch_config(grid, g.block(), dynamic_shared_memory, at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    static std::once_flag attr_once;
    std::call_once(attr_once, [&]() {
        CUDACHECK(cudaFuncSetAttribute(
            kernel_streaming_live_fp4pv<C, false, MODE, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            dynamic_shared_memory));
    });
    CUDACHECK(cudaLaunchKernelEx(
        launch_config,
        kernel_streaming_live_fp4pv<C, false, MODE, true, true>,
        g,
        MODE,
        0,
        -1));
}

#if !defined(TK_FA4_BACKWARD_ONLY_BUILD)
void dispatch_backward_dv_streaming_live_mxfp4(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc,
    at::Tensor O,
    at::Tensor LSE,
    double p_payload_scale,
    int p_quant_mode,
    int pv_issue_mode,
    bool persistent_launch
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(O); CHECK_INPUT(LSE);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");
    TORCH_CHECK(O.dim() == 4, "O must be rank-4");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch && O.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads && O.size(2) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen && O.size(1) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(O.size(3) == 128, "O head_dim must be 128");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4 && LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(p_payload_scale > 0.0 && std::isfinite(p_payload_scale), "p_payload_scale must be finite and positive");
    TORCH_CHECK(p_quant_mode >= 0 && p_quant_mode <= 2, "p_quant_mode must be 0 (rte), 1 (encode), or 2 (decode)");
    TORCH_CHECK(pv_issue_mode == 2 || pv_issue_mode == 4 || pv_issue_mode == 6 || pv_issue_mode == 7 ||
                    pv_issue_mode == 9 || pv_issue_mode == 10 || pv_issue_mode == 17 ||
                    pv_issue_mode == 19 || pv_issue_mode == 26 || pv_issue_mode == 31 ||
                    pv_issue_mode == 32 || pv_issue_mode == 38 ||
                    pv_issue_mode == 39 || pv_issue_mode == 40 || pv_issue_mode == 41 ||
                    pv_issue_mode == 42 || pv_issue_mode == 43 ||
                    pv_issue_mode == 44 || pv_issue_mode == 45 ||
                    pv_issue_mode == 46 || pv_issue_mode == 47 ||
                    pv_issue_mode == 48 || pv_issue_mode == 49 ||
                    pv_issue_mode == 50 || pv_issue_mode == 51 ||
                    pv_issue_mode == 52 || pv_issue_mode == 55 ||
                    pv_issue_mode == 56 || pv_issue_mode == 62 ||
                    pv_issue_mode == 63 || pv_issue_mode == 64 ||
                    pv_issue_mode == 65 || pv_issue_mode == 66 ||
                    pv_issue_mode == 68 || pv_issue_mode == 70 ||
                    pv_issue_mode == 71 || pv_issue_mode == 72 ||
                    pv_issue_mode == 75 || pv_issue_mode == 76 ||
                    pv_issue_mode == 77 || pv_issue_mode == 78 ||
                    pv_issue_mode == 80 || pv_issue_mode == 81 ||
                    pv_issue_mode == 82 || pv_issue_mode == 85 ||
                    pv_issue_mode == 86 || pv_issue_mode == 90 ||
                    pv_issue_mode == 91 || pv_issue_mode == 92 ||
                    pv_issue_mode == 93 || pv_issue_mode == 94 ||
                    pv_issue_mode == 95 || pv_issue_mode == 96 ||
                    pv_issue_mode == 97 || pv_issue_mode == 98 ||
                    pv_issue_mode == 99 || pv_issue_mode == 100 ||
                    pv_issue_mode == 101 || pv_issue_mode == 102 ||
                    pv_issue_mode == 103 || pv_issue_mode == 104 ||
                    pv_issue_mode == 105 || pv_issue_mode == 106 ||
                    pv_issue_mode == 107 || pv_issue_mode == 108 ||
                    pv_issue_mode == 109 || pv_issue_mode == 110 ||
                    pv_issue_mode == 111 || pv_issue_mode == 112 ||
                    pv_issue_mode == 113 || pv_issue_mode == 114 ||
                    pv_issue_mode == 115 || pv_issue_mode == 116 ||
                    pv_issue_mode == 117 || pv_issue_mode == 118 ||
                    pv_issue_mode == 119 || pv_issue_mode == 120 ||
                    pv_issue_mode == 123 || pv_issue_mode == 126 ||
                    pv_issue_mode == 127 || pv_issue_mode == 128 ||
                    pv_issue_mode == 129 || pv_issue_mode == 130 ||
                    pv_issue_mode == 131 || pv_issue_mode == 132 ||
                    pv_issue_mode == 133 || pv_issue_mode == 134 ||
                    pv_issue_mode == 135 || pv_issue_mode == 136 ||
                    pv_issue_mode == 137 || pv_issue_mode == 138 ||
                    pv_issue_mode == 139 || pv_issue_mode == 140 ||
                    pv_issue_mode == 141 || pv_issue_mode == 142 ||
                    pv_issue_mode == 143 || pv_issue_mode == 144 ||
                    pv_issue_mode == 145 || pv_issue_mode == 146 ||
                    pv_issue_mode == 147 || pv_issue_mode == 148 ||
                    pv_issue_mode == 149 || pv_issue_mode == 150 ||
                    pv_issue_mode == 151 || pv_issue_mode == 152 ||
                    pv_issue_mode == 153 || pv_issue_mode == 154 ||
                    pv_issue_mode == 155 || pv_issue_mode == 156 ||
                    pv_issue_mode == 157 || pv_issue_mode == 158 ||
                    pv_issue_mode == 159 || pv_issue_mode == 160 ||
                    pv_issue_mode == 161 || pv_issue_mode == 162 ||
                    pv_issue_mode == 163 || pv_issue_mode == 164 ||
                    pv_issue_mode == 165 || pv_issue_mode == 166 ||
                    pv_issue_mode == 167 || pv_issue_mode == 168 ||
                    pv_issue_mode == 169 || pv_issue_mode == 170 ||
                    pv_issue_mode == 171 || pv_issue_mode == 172 ||
                    pv_issue_mode == 173 || pv_issue_mode == 174 ||
                    pv_issue_mode == 176 || pv_issue_mode == 178 ||
                    pv_issue_mode == 180 || pv_issue_mode == 181 ||
                    pv_issue_mode == 182 || pv_issue_mode == 183 ||
                    pv_issue_mode == 184 || pv_issue_mode == 185 ||
                    pv_issue_mode == 186 || pv_issue_mode == 187 ||
                    pv_issue_mode == 188 || pv_issue_mode == 189 ||
                    pv_issue_mode == 190 || pv_issue_mode == 191 ||
                    pv_issue_mode == 192 || pv_issue_mode == 193 ||
                    pv_issue_mode == 194 || pv_issue_mode == 195 ||
                    pv_issue_mode == 196 || pv_issue_mode == 197 ||
                    pv_issue_mode == 198 || pv_issue_mode == 199 ||
                    pv_issue_mode == 200 || pv_issue_mode == 201 ||
                    pv_issue_mode == 202 || pv_issue_mode == 203 ||
                    pv_issue_mode == 204 || pv_issue_mode == 205 ||
                    pv_issue_mode == 206 || pv_issue_mode == 207 ||
                    pv_issue_mode == 208 || pv_issue_mode == 209 ||
                    pv_issue_mode == 210 || pv_issue_mode == 211 ||
                    pv_issue_mode == 212 || pv_issue_mode == 213 ||
                    pv_issue_mode == 214 || pv_issue_mode == 215 ||
                    pv_issue_mode == 216 || pv_issue_mode == 217 ||
                    pv_issue_mode == 218 || pv_issue_mode == 219 ||
                    pv_issue_mode == 220 || pv_issue_mode == 221 ||
                    pv_issue_mode == 222 || pv_issue_mode == 223 ||
                    pv_issue_mode == 224 || pv_issue_mode == 225 ||
                    pv_issue_mode == 226 || pv_issue_mode == 227 ||
                    pv_issue_mode == 228 || pv_issue_mode == 229 ||
                    pv_issue_mode == 230 || pv_issue_mode == 231 ||
                    pv_issue_mode == 232 || pv_issue_mode == 233 ||
                    pv_issue_mode == 234 || pv_issue_mode == 235 ||
                    pv_issue_mode == 236 || pv_issue_mode == 237 ||
                    pv_issue_mode == 238 || pv_issue_mode == 239 ||
                    pv_issue_mode == 240 || pv_issue_mode == 241 ||
                    pv_issue_mode == 242 || pv_issue_mode == 243 ||
                    pv_issue_mode == 244 || pv_issue_mode == 245 ||
                    pv_issue_mode == 246 || pv_issue_mode == 247 ||
                    pv_issue_mode == 248 || pv_issue_mode == 251 ||
                    pv_issue_mode == 252 || pv_issue_mode == 253 ||
                    pv_issue_mode == 254 || pv_issue_mode == 255 ||
                    pv_issue_mode == 256 || pv_issue_mode == 257 ||
                    pv_issue_mode == 258 || pv_issue_mode == 259 ||
                    pv_issue_mode == 260 || pv_issue_mode == 261 ||
                    pv_issue_mode == 268 || pv_issue_mode == 269 ||
                    pv_issue_mode == 308 ||
                    pv_issue_mode == 309 || pv_issue_mode == 310 ||
                    pv_issue_mode == 311 ||
                    pv_issue_mode == 312 || pv_issue_mode == 313 ||
                    pv_issue_mode == 316 || pv_issue_mode == 317 ||
                    pv_issue_mode == 318 || pv_issue_mode == 319 ||
                    pv_issue_mode == 320 || pv_issue_mode == 321 ||
                    pv_issue_mode == 322 || pv_issue_mode == 323 ||
                    pv_issue_mode == 324 || pv_issue_mode == 325 ||
                    pv_issue_mode == 326 || pv_issue_mode == 327 ||
                    pv_issue_mode == 328 || pv_issue_mode == 329 ||
                    pv_issue_mode == 330 || pv_issue_mode == 331 ||
                    pv_issue_mode == 332 || pv_issue_mode == 333 ||
                    pv_issue_mode == 336 || pv_issue_mode == 337 ||
                    pv_issue_mode == 344 || pv_issue_mode == 345 ||
                    pv_issue_mode == 346 || pv_issue_mode == 347 ||
                    pv_issue_mode == 348 || pv_issue_mode == 349 ||
                    pv_issue_mode == 350 || pv_issue_mode == 351 ||
                    pv_issue_mode == 352 || pv_issue_mode == 353 ||
                    pv_issue_mode == 354 || pv_issue_mode == 355 ||
                    pv_issue_mode == 356 || pv_issue_mode == 357 ||
                    pv_issue_mode == 358 || pv_issue_mode == 359 ||
                    pv_issue_mode == 360 || pv_issue_mode == 361 ||
                    pv_issue_mode == 362 || pv_issue_mode == 363 ||
                    pv_issue_mode == 364 || pv_issue_mode == 365 ||
                    pv_issue_mode == 366 || pv_issue_mode == 367 ||
                    pv_issue_mode == 368 || pv_issue_mode == 369 ||
                    pv_issue_mode == 370 || pv_issue_mode == 371 ||
                    pv_issue_mode == 372 || pv_issue_mode == 373 ||
                    pv_issue_mode == 380 || pv_issue_mode == 381 ||
                    pv_issue_mode == 382 || pv_issue_mode == 390 ||
                    pv_issue_mode == 391 || pv_issue_mode == 392 ||
                    pv_issue_mode == 393 || pv_issue_mode == 394 ||
                    pv_issue_mode == 395 || pv_issue_mode == 396 ||
                    pv_issue_mode == 397 || pv_issue_mode == 398 ||
                    pv_issue_mode == 399 || pv_issue_mode == 400 ||
                    pv_issue_mode == 401 || pv_issue_mode == 402 ||
                    pv_issue_mode == 403 ||
                    pv_issue_mode == 271 || pv_issue_mode == 272 ||
                    pv_issue_mode == 273 || pv_issue_mode == 274 ||
                    pv_issue_mode == 275 || pv_issue_mode == 276 ||
                    pv_issue_mode == 277 || pv_issue_mode == 278 ||
                    pv_issue_mode == 279 || pv_issue_mode == 280 ||
	                    pv_issue_mode == 281 || pv_issue_mode == 282 ||
	                    pv_issue_mode == 283 || pv_issue_mode == 284 ||
                    pv_issue_mode == 285 || pv_issue_mode == 286 ||
                    pv_issue_mode == 287 || pv_issue_mode == 288 ||
                    pv_issue_mode == 289 || pv_issue_mode == 290 ||
                    pv_issue_mode == 291 || pv_issue_mode == 292 ||
                    pv_issue_mode == 295 || pv_issue_mode == 296 ||
                    pv_issue_mode == 297 || pv_issue_mode == 298 ||
                    pv_issue_mode == 299 || pv_issue_mode == 300 ||
                    pv_issue_mode == 301 || pv_issue_mode == 302 ||
                    pv_issue_mode == 303 || pv_issue_mode == 304 ||
                    pv_issue_mode == 305 || pv_issue_mode == 306 ||
                    pv_issue_mode == 307,
		                "pv_issue_mode must be one of the compiled MXFP4 external-LSE dV modes");
    TORCH_CHECK((pv_issue_mode != 6 && pv_issue_mode != 178) || (seqlen % 256 == 0),
                "K=256 MXFP4 pv_issue_mode requires sequence length divisible by 256");

    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, 128);
    check_fp4fa4_bwd_mxfp4_v_scale_tensor(V_sc, batch, heads, seqlen, 128);

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    switch (pv_issue_mode) {
    case 2:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv<128, 128, 192, 128, 2>, 2>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 4:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv<128, 128, 192, 128, 2>, 4>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 6:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv<128, 128, 192, 128, 2>, 6>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 7:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv<128, 128, 192, 128, 2>, 7>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 9:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_5wg<128, 128, 192, 128, 2>, 9>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 10:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 10>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 17:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 17>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 19:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 19>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 26:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 26>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 31:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 31>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 32:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 32>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 38:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 38>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 39:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 39>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 40:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 40>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 41:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 41>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 42:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 42>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 43:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 43>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 44:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 44>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 45:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 45>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 46:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 46>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 47:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 47>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 48:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 48>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 49:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 49>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 50:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 50>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 51:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 51>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 52:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 52>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 55:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32<128, 128, 192, 128, 2>, 55>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 56:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_direct_sc<128, 128, 192, 128, 0, 2>, 56>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 62:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_direct_sc<128, 128, 192, 128, 0, 2>, 62>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 63:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 63>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 64:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 64>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 65:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg<128, 128, 192, 128, 2>, 65>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 66:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64<128, 128, 192, 128, 2>, 66>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 68:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64<128, 128, 192, 128, 2>, 68>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 70:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64<128, 128, 192, 128, 2>, 70>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 71:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64<128, 128, 192, 128, 2>, 71>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 72:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64<128, 128, 192, 128, 2>, 72>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 75:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1<128, 128, 192, 128, 2>, 75>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 76:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1<128, 128, 192, 128, 2>, 76>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 77:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1<128, 128, 192, 128, 2>, 77>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 78:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1<128, 128, 192, 128, 2>, 78>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 80:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1<128, 128, 192, 128, 2>, 80>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 81:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1<128, 128, 192, 128, 2>, 81>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 82:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1<128, 128, 192, 128, 2>, 82>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 85:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1<128, 128, 192, 128, 2>, 85>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 86:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1<128, 128, 192, 128, 2>, 86>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 90:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_5wg_n64_p32_persist1<128, 192, 128, 2>, 90>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 91:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1<128, 128, 192, 128, 2>, 91>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 92:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1<128, 128, 192, 128, 2>, 92>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 93:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1_pstage3<128, 128, 192, 128, 2>, 93>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 94:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1_pstage3<128, 128, 192, 128, 2>, 94>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 95:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1_pstage3<128, 128, 192, 128, 2>, 95>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 96:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1_pstage3<128, 128, 192, 128, 2>, 96>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 97:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1_pstage3<128, 128, 192, 128, 2>, 97>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 98:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1<128, 128, 192, 128, 2>, 98>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 99:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1_pstage3_force<128, 128, 192, 128, 2>, 99>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 100:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1_force<128, 128, 192, 128, 2>, 100>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 101:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_pstage3_force<128, 128, 192, 128, 2>, 101>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 102:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 102>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 103:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 103>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 104:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 104>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 105:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_pstage3_force<128, 128, 192, 128, 2>, 105>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 106:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 106>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 107:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_pstage3_force<128, 128, 192, 128, 2>, 107>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 108:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 108>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 109:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_pstage3_force<128, 128, 192, 128, 2>, 109>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 110:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 110>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 111:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_pstage3_force<128, 128, 192, 128, 2>, 111>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 112:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_pstage3_force<128, 128, 192, 128, 2>, 112>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 113:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 113>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 114:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 114>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 115:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 115>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 116:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 116>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 117:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 117>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 118:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 118>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 119:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 119>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 120:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 120>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 123:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_pstage3_force<128, 128, 192, 128, 2>, 123>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 126:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1<128, 128, 192, 128, 2>, 126>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 127:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_persist1_pstage3<128, 128, 192, 128, 2>, 127>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 128:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 128>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 129:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 129>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 130:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 130>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 131:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 131>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 132:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 132>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 133:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 133>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 134:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 134>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 135:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 135>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 136:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 136>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 137:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_pstage3_force<128, 128, 192, 128, 2>, 137>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 138:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 138>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 139:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 139>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 140:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 140>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 141:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 141>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 142:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 142>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 143:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 143>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 144:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 144>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 145:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 145>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 146:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 146>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 147:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 147>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 148:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 148>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 149:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 149>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 150:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 150>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 151:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 151>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 152:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 152>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 153:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 153>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 154:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 154>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 155:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 155>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 156:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 156>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 157:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 157>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 158:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 158>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 159:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 159>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 160:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 160>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 161:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 161>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 162:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 162>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 163:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 163>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 164:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 164>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 165:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 165>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 166:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 166>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 167:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 167>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 168:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 1>, 168>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 169:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 169>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 170:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 170>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 171:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 171>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 172:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 172>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 173:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_pstage3_force<128, 128, 192, 128, 2>, 173>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 174:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 174>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 176:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 176>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 178:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_5wg<128, 128, 192, 128, 2>, 178>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 180:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 180>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 181:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 181>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 182:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 182>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 183:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 183>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 184:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 184>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 185:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 185>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 186:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 186>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 187:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 187>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 188:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 188>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 189:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 189>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 190:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 190>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 191:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 191>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 192:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 192>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 193:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 193>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 194:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 194>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 195:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 195>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 196:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_pstage3_force<128, 128, 192, 128, 2>, 196>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 197:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 197>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 198:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 198>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 199:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 199>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 200:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 200>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 201:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 201>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 202:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 202>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 203:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 203>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 204:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 204>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 205:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 205>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 206:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 206>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 207:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 207>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 208:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 208>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 209:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 209>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 210:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 210>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 211:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 211>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 212:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 212>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 213:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 213>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 214:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 214>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 215:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 215>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 216:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 216>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 217:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 217>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 218:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 218>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 219:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 219>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 220:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 220>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 221:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 221>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 222:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 222>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 223:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 223>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 224:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 224>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 225:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 225>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 226:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 226>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 227:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 227>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 228:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 228>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 229:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 229>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 230:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 230>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 231:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 231>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 232:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 232>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 233:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 233>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 234:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 234>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 235:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 235>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 236:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 236>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 237:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 237>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 238:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_5wg_p128_force<128, 128, 192, 128, 2>, 238>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 239:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_5wg_p128_force<128, 128, 192, 128, 2>, 239>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 240:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_5wg_p128_force<128, 128, 192, 128, 2>, 240>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 241:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_5wg_p128_force<128, 128, 192, 128, 2>, 241>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 242:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_5wg_p128_force<128, 128, 192, 128, 2>, 242>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 243:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_5wg_p128_force<128, 128, 192, 128, 2>, 243>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 244:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 244>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 245:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 245>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 246:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_5wg_p128_pstage3_force<128, 128, 192, 128, 2>, 246>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 247:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 247>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 248:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_5wg_p128_pstage3_force<128, 128, 192, 128, 2>, 248>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 251:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 251>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 252:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 252>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 253:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 253>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 254:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 254>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 255:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>, 255>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 256:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 256>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 257:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 257>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 258:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 258>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 259:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 259>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 260:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 260>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 261:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 261>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 268:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 268>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 269:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 269>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 308:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 308>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 309:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 309>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 310:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 310>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 311:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 311>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 312:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 312>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 313:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 313>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 316:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 316>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 317:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 317>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 318:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 318>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 319:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 319>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 320:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 320>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 321:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 321>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 322:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 322>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 323:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 323>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 324:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 324>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 325:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 325>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 326:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 326>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 327:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 327>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 328:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 328>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 329:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 329>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 330:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 330>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 331:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 331>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 332:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 332>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 333:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 333>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 336:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 336>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 337:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 337>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 344:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 344>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 345:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 345>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 346:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 346>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 347:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 347>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 348:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 348>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 349:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 349>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 350:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 350>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 351:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 351>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 352:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 352>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 353:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 353>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 354:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 354>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 355:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 355>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 356:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 356>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 357:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 357>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 358:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 358>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 359:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 359>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 360:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 360>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 361:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 361>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 362:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 362>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 363:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 363>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 364:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 364>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 365:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 365>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 366:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 366>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 367:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 367>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 368:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 368>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 369:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 369>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 370:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 370>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 371:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 371>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 372:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 372>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 373:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 373>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 380:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 380>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 381:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 381>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 382:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 382>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 390:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 390>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 391:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 391>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 392:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 392>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 393:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 393>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 394:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 394>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 395:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 395>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 396:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 396>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 397:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 397>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 398:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 398>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 399:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 399>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 400:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 400>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 401:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 401>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 402:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 402>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 403:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 403>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 271:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 271>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 272:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 272>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 273:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 273>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 274:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 274>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 275:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 275>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 276:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 276>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 277:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 277>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 278:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 278>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 279:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 279>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 280:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 280>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 281:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 281>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 282:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 282>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 283:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 283>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 284:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 284>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
	    case 285:
	        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 285>(
	            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
	            p_payload_scale, p_quant_mode, persistent_launch);
	        break;
	    case 286:
	        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 286>(
	            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
	            p_payload_scale, p_quant_mode, persistent_launch);
	        break;
    case 287:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 287>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 288:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 288>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 289:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 289>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 290:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 290>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 291:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 291>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 292:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 292>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 295:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 295>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 296:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 296>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 297:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 297>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 298:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 298>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 299:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 299>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 300:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 300>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 301:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>, 301>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 302:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 302>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 303:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 303>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 304:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 304>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 305:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_force<128, 128, 192, 128, 2>, 305>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 306:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p128_pstage3_force<128, 128, 192, 128, 2>, 306>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 307:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>, 307>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    default:
        launch_backward_dv_streaming_live_mxfp4_config<config_fp4pv<128, 128, 192, 128, 2>, 2>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    }
}
#endif

void dispatch_backward_dv_streaming_live_mxfp4_rewrite(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc,
    at::Tensor O,
    at::Tensor LSE,
    double p_payload_scale,
    int p_quant_mode,
    bool persistent_launch
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(O); CHECK_INPUT(LSE);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");
    TORCH_CHECK(O.dim() == 4, "O must be rank-4");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch && O.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads && O.size(2) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen && O.size(1) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(O.size(3) == 128, "O head_dim must be 128");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4 && LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(p_payload_scale > 0.0 && std::isfinite(p_payload_scale), "p_payload_scale must be finite and positive");
    TORCH_CHECK(p_quant_mode >= 0 && p_quant_mode <= 2, "p_quant_mode must be 0 (rte), 1 (encode), or 2 (decode)");

    check_fp4pv_v_tensor(V_fp4, batch, heads, seqlen, 128);
    check_fp4fa4_bwd_mxfp4_v_scale_tensor(V_sc, batch, heads, seqlen, 128);

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    int pv_issue_mode = 163;
    if (seqlen <= 2048) {
        pv_issue_mode = 128;
    } else if (seqlen <= 4096) {
        // The key-owned backward schedule is not safe with mode 163 at 3K and
        // mode 390 is also a small win at 4K.
        pv_issue_mode = 390;
    } else if (seqlen <= 8192) {
        pv_issue_mode = 393;
    } else {
        pv_issue_mode = 247;
    }
    if (const char *mode_env = std::getenv("TK_FA4_MXFP4_BWD_KEY_OWNED_DV_MODE")) {
        pv_issue_mode = std::atoi(mode_env);
    }
    TORCH_CHECK(
        pv_issue_mode != 128 || seqlen <= 4096,
        "TK_FA4_MXFP4_BWD_KEY_OWNED_DV_MODE=128 is only validated up to seqlen 4096");

    if (pv_issue_mode == 163) {
        launch_backward_dv_streaming_live_mxfp4_config<
            config_fp4pv_6wg_p32_force<128, 128, 192, 128, 2>,
            163>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        return;
    }

    switch (pv_issue_mode) {
    case 128:
        launch_backward_dv_streaming_live_mxfp4_config<
            config_fp4pv_6wg_p64_force<128, 128, 192, 128, 2>,
            128>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 247:
        launch_backward_dv_streaming_live_mxfp4_config<
            config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>,
            247>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 390:
        launch_backward_dv_streaming_live_mxfp4_config<
            config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>,
            390>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    case 393:
        launch_backward_dv_streaming_live_mxfp4_config<
            config_fp4pv_6wg_p32_pstage3_force<128, 128, 192, 128, 2>,
            393>(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, V_fp4, V_sc, O, LSE, batch, heads, seqlen, k_sc_depth,
            p_payload_scale, p_quant_mode, persistent_launch);
        break;
    default:
        TORCH_CHECK(false, "TK_FA4_MXFP4_BWD_KEY_OWNED_DV_MODE must be one of 128, 163, 247, 390, or 393");
    }
}
#endif

#ifndef TK_FA4_DQ_ONLY_BUILD
pybind11::tuple dispatch_forward_streaming_live_qk_only(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto P_bf16_live = Q_sg.new_empty({batch, heads, seqlen, seqlen}, Q_sg.options().dtype(at::ScalarType::BFloat16));
    auto LSE = Q_sg.new_empty({batch, heads, 1, seqlen}, Q_sg.options().dtype(at::ScalarType::Float));

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(P_bf16_live.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        -1,
        0,
        seqlen / C::Nb,
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid_exact(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true, true>, g));

    return pybind11::make_tuple(P_bf16_live, LSE);
}

at::Tensor dispatch_forward_streaming_live_qk_only_saved_lse(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg); CHECK_INPUT(LSE);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4, "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto P_bf16_live = Q_sg.new_zeros({batch, heads, seqlen, seqlen}, Q_sg.options().dtype(at::ScalarType::BFloat16));

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    const int total_k_tiles = seqlen / C::Nb;
    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(P_bf16_live.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        -1,
        0,
        total_k_tiles,
    };
    g.disable_lse_store = 1;
    g.use_external_lse = 1;

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid_exact(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    constexpr int SAFE_K_TILE_WINDOW = 3;
    for (int k_base = 0; k_base < total_k_tiles; k_base += SAFE_K_TILE_WINDOW) {
        const int k_count = min(SAFE_K_TILE_WINDOW, total_k_tiles - k_base);
        g.max_k_iters = k_count;
        g.compute_k_iter_base = k_base;
        g.debug_stage_cut = DEBUG_STAGE_CUT_FRESH3_FULL;
        CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true, true>, g));
    }

    return P_bf16_live;
}

at::Tensor dispatch_backward_ds_streaming_live_qk_saved_lse(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    double softmax_scale
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(LSE); CHECK_INPUT(DP); CHECK_INPUT(Delta);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(DP.dtype() == at::ScalarType::BFloat16, "DP must be BF16");
    TORCH_CHECK(Delta.dtype() == at::ScalarType::Float, "Delta must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4, "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(DP.dim() == 4, "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(DP.size(0) == batch && DP.size(1) == heads && DP.size(2) == seqlen && DP.size(3) == seqlen,
                "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(Delta.dim() == 3, "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(Delta.size(0) == batch && Delta.size(1) == heads && Delta.size(2) == seqlen,
                "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto DS_bf16 = Q_sg.new_zeros({batch, heads, seqlen, seqlen}, Q_sg.options().dtype(at::ScalarType::BFloat16));

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    const int total_k_tiles = seqlen / C::Nb;
    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(DS_bf16.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        -1,
        0,
        total_k_tiles,
    };
    g.disable_lse_store = 1;
    g.use_external_lse = 1;
    g.backward_dp_bf16 = reinterpret_cast<bf16 *>(DP.data_ptr());
    g.backward_delta = reinterpret_cast<float *>(Delta.data_ptr());
    g.backward_softmax_scale = static_cast<float>(softmax_scale);
    g.emit_backward_ds = 1;

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid_exact(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    constexpr int SAFE_K_TILE_WINDOW = 3;
    for (int k_base = 0; k_base < total_k_tiles; k_base += SAFE_K_TILE_WINDOW) {
        const int k_count = min(SAFE_K_TILE_WINDOW, total_k_tiles - k_base);
        g.max_k_iters = k_count;
        g.compute_k_iter_base = k_base;
        g.debug_stage_cut = DEBUG_STAGE_CUT_FRESH3_FULL;
        CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true, true>, g));
    }

    return DS_bf16;
}

at::Tensor dispatch_backward_ds_streaming_live_qk_saved_lse_row_tile(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int row_tile,
    double softmax_scale
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(LSE); CHECK_INPUT(DP); CHECK_INPUT(Delta);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(DP.dtype() == at::ScalarType::BFloat16, "DP must be BF16");
    TORCH_CHECK(Delta.dtype() == at::ScalarType::Float, "Delta must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4, "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(DP.dim() == 4, "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(DP.size(0) == batch && DP.size(1) == heads && DP.size(2) == seqlen && DP.size(3) == seqlen,
                "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(Delta.dim() == 3, "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(Delta.size(0) == batch && Delta.size(1) == heads && Delta.size(2) == seqlen,
                "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    TORCH_CHECK(seqlen >= C::CLUSTER_SIZE * C::Mb, "fused MXFP4 dQ diagnostic requires seqlen >= 256 for CTA-pair synchronization");
    const int row_tiles = seqlen / C::Mb;
    TORCH_CHECK(row_tile >= 0 && row_tile < row_tiles, "row_tile must be within sequence row-tile range");

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    const int row_cluster_base = row_tile - (row_tile % C::CLUSTER_SIZE);
    const int causal_tiles = min(row_cluster_base + C::CLUSTER_SIZE, row_tiles);
    auto DS_tile = Q_sg.new_zeros({batch, heads, C::Mb, causal_tiles * C::Nb}, Q_sg.options().dtype(at::ScalarType::BFloat16));

    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(DS_tile.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        row_tile,
        0,
        causal_tiles,
    };
    g.disable_lse_store = 1;
    g.use_external_lse = 1;
    g.backward_dp_bf16 = reinterpret_cast<bf16 *>(DP.data_ptr());
    g.backward_delta = reinterpret_cast<float *>(Delta.data_ptr());
    g.backward_softmax_scale = static_cast<float>(softmax_scale);
    g.emit_backward_ds = 1;

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid_exact(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    constexpr int SAFE_K_TILE_WINDOW = 3;
    for (int k_base = 0; k_base < causal_tiles; k_base += SAFE_K_TILE_WINDOW) {
        const int k_count = min(SAFE_K_TILE_WINDOW, causal_tiles - k_base);
        g.max_k_iters = k_count;
        g.compute_k_iter_base = k_base;
        g.debug_stage_cut = DEBUG_STAGE_CUT_FRESH3_FULL;
        CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true, true>, g));
    }

    return DS_tile;
}

at::Tensor dispatch_backward_ds_streaming_live_qk_saved_lse_row_cluster(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int row_cluster_base,
    double softmax_scale
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(LSE); CHECK_INPUT(DP); CHECK_INPUT(Delta);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(DP.dtype() == at::ScalarType::BFloat16, "DP must be BF16");
    TORCH_CHECK(Delta.dtype() == at::ScalarType::Float, "Delta must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4, "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(DP.dim() == 4, "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(DP.size(0) == batch && DP.size(1) == heads && DP.size(2) == seqlen && DP.size(3) == seqlen,
                "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(Delta.dim() == 3, "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(Delta.size(0) == batch && Delta.size(1) == heads && Delta.size(2) == seqlen,
                "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    const int row_tiles = seqlen / C::Mb;
    TORCH_CHECK(row_cluster_base >= 0 && row_cluster_base < row_tiles, "row_cluster_base must be within sequence row-tile range");
    TORCH_CHECK(row_cluster_base % C::CLUSTER_SIZE == 0, "row_cluster_base must be aligned to the FP4PV row-cluster size");
    const int cluster_tiles = min(C::CLUSTER_SIZE, row_tiles - row_cluster_base);
    const int causal_tiles = min(row_cluster_base + cluster_tiles, row_tiles);

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto DS_cluster = Q_sg.new_zeros({batch, heads, cluster_tiles * C::Mb, causal_tiles * C::Nb}, Q_sg.options().dtype(at::ScalarType::BFloat16));

    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(DS_cluster.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        row_cluster_base,
        0,
        causal_tiles,
    };
    g.selected_row_tile_span = cluster_tiles;
    g.disable_lse_store = 1;
    g.use_external_lse = 1;
    g.backward_dp_bf16 = reinterpret_cast<bf16 *>(DP.data_ptr());
    g.backward_delta = reinterpret_cast<float *>(Delta.data_ptr());
    g.backward_softmax_scale = static_cast<float>(softmax_scale);
    g.emit_backward_ds = 1;

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid_exact(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    constexpr int SAFE_K_TILE_WINDOW = 3;
    for (int k_base = 0; k_base < causal_tiles; k_base += SAFE_K_TILE_WINDOW) {
        const int k_count = min(SAFE_K_TILE_WINDOW, causal_tiles - k_base);
        g.max_k_iters = k_count;
        g.compute_k_iter_base = k_base;
        g.debug_stage_cut = DEBUG_STAGE_CUT_FRESH3_FULL;
        CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true, true>, g));
    }

    return DS_cluster;
}

at::Tensor dispatch_backward_ds_streaming_live_qk_saved_lse_clustered_full(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    double softmax_scale
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(LSE); CHECK_INPUT(DP); CHECK_INPUT(Delta);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(DP.dtype() == at::ScalarType::BFloat16, "DP must be BF16");
    TORCH_CHECK(Delta.dtype() == at::ScalarType::Float, "Delta must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4, "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(DP.dim() == 4, "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(DP.size(0) == batch && DP.size(1) == heads && DP.size(2) == seqlen && DP.size(3) == seqlen,
                "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(Delta.dim() == 3, "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(Delta.size(0) == batch && Delta.size(1) == heads && Delta.size(2) == seqlen,
                "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    const int row_tiles = seqlen / C::Mb;
    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto DS_bf16 = Q_sg.new_zeros({batch, heads, seqlen, seqlen}, Q_sg.options().dtype(at::ScalarType::BFloat16));

    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(DS_bf16.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        0,
        0,
        row_tiles,
    };
    g.output_full_layout = 1;
    g.disable_lse_store = 1;
    g.use_external_lse = 1;
    g.backward_dp_bf16 = reinterpret_cast<bf16 *>(DP.data_ptr());
    g.backward_delta = reinterpret_cast<float *>(Delta.data_ptr());
    g.backward_softmax_scale = static_cast<float>(softmax_scale);
    g.emit_backward_ds = 1;

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid_exact(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    constexpr int SAFE_K_TILE_WINDOW = 3;
    for (int row_cluster_base = 0; row_cluster_base < row_tiles; row_cluster_base += C::CLUSTER_SIZE) {
        const int cluster_tiles = min(C::CLUSTER_SIZE, row_tiles - row_cluster_base);
        const int causal_tiles = min(row_cluster_base + cluster_tiles, row_tiles);
        g.selected_row_tile = row_cluster_base;
        g.selected_row_tile_span = cluster_tiles;
        g.tile_begin = 0;
        g.tiles_to_run = causal_tiles;
        for (int k_base = 0; k_base < causal_tiles; k_base += SAFE_K_TILE_WINDOW) {
            const int k_count = min(SAFE_K_TILE_WINDOW, causal_tiles - k_base);
            g.max_k_iters = k_count;
            g.compute_k_iter_base = k_base;
            g.debug_stage_cut = DEBUG_STAGE_CUT_FRESH3_FULL;
            CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true, true>, g));
        }
    }

    return DS_bf16;
}

std::vector<at::Tensor> dispatch_backward_ds_streaming_live_qk_saved_lse_clustered_full_row_mxfp4_impl(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int safe_k_tile_window,
    bool store_bf16_validation,
    bool emit_col_mxfp4,
    bool fused_stage_probe = false,
    bool emit_row_mxfp4 = true,
    bool fused_stage_transpose_probe = false
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(LSE); CHECK_INPUT(DP); CHECK_INPUT(Delta);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(DP.dtype() == at::ScalarType::BFloat16, "DP must be BF16");
    TORCH_CHECK(Delta.dtype() == at::ScalarType::Float, "Delta must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");
    TORCH_CHECK(p_quant_mode >= 0 && p_quant_mode <= 2, "p_quant_mode must be 0 (rte), 1 (encode), or 2 (decode)");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4, "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(DP.dim() == 4, "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(DP.size(0) == batch && DP.size(1) == heads && DP.size(2) == seqlen && DP.size(3) == seqlen,
                "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(Delta.dim() == 3, "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(Delta.size(0) == batch && Delta.size(1) == heads && Delta.size(2) == seqlen,
                "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    const int row_tiles = seqlen / C::Mb;
    TORCH_CHECK(safe_k_tile_window > 0 && safe_k_tile_window <= row_tiles,
                "safe_k_tile_window must be in [1, seqlen / 128]");
    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto DS_fp4_row = (fused_stage_probe || !emit_row_mxfp4)
        ? Q.new_empty({0}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2))
        : at::empty({batch, heads, seqlen, seqlen / 2}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2));
    auto DS_sc_row = (fused_stage_probe || !emit_row_mxfp4)
        ? Q_sg.new_empty({0}, Q_sg.options().dtype(at::ScalarType::Byte))
        : at::empty({batch, heads, row_tiles, seqlen / C::Nb, 32, 16}, Q.options().dtype(at::ScalarType::Byte));
    auto DS_fp4_col = (!fused_stage_probe && emit_col_mxfp4)
        ? at::empty({batch, heads, seqlen, seqlen / 2}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2))
        : Q.new_empty({0}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2));
    auto DS_sc_col = (!fused_stage_probe && emit_col_mxfp4)
        ? at::empty({batch, heads, row_tiles, seqlen / C::Nb, 32, 16}, Q.options().dtype(at::ScalarType::Byte))
        : Q_sg.new_empty({0}, Q_sg.options().dtype(at::ScalarType::Byte));
    auto DS_bf16 = (!fused_stage_probe && store_bf16_validation)
        ? Q_sg.new_zeros({batch, heads, seqlen, seqlen}, Q_sg.options().dtype(at::ScalarType::BFloat16))
        : Q_sg.new_empty({0}, Q_sg.options().dtype(at::ScalarType::BFloat16));
    auto fused_stage_marker = fused_stage_probe
        ? Q_sg.new_empty({batch, heads, seqlen, row_tiles}, Q_sg.options().dtype(at::ScalarType::Int))
        : Q_sg.new_empty({0}, Q_sg.options().dtype(at::ScalarType::Int));
    if (!fused_stage_probe && emit_row_mxfp4) {
        CUDACHECK(cudaMemset(DS_sc_row.data_ptr(), 0, DS_sc_row.nbytes()));
    }
    if (!fused_stage_probe && emit_col_mxfp4) {
        CUDACHECK(cudaMemset(DS_sc_col.data_ptr(), 0, DS_sc_col.nbytes()));
    }

    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        store_bf16_validation ? reinterpret_cast<bf16 *>(DS_bf16.data_ptr()) : nullptr,
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        0,
        0,
        row_tiles,
    };
    g.output_full_layout = 1;
    g.disable_lse_store = 1;
    g.use_external_lse = 1;
    g.backward_dp_bf16 = reinterpret_cast<bf16 *>(DP.data_ptr());
    g.backward_delta = reinterpret_cast<float *>(Delta.data_ptr());
    g.backward_softmax_scale = static_cast<float>(softmax_scale);
    g.emit_backward_ds = 1;
    if (fused_stage_probe) {
        g.backward_fused_stage_marker = reinterpret_cast<uint32_t *>(fused_stage_marker.data_ptr<int>());
        g.backward_fused_stage_transpose = fused_stage_transpose_probe ? 1 : 0;
    } else if (emit_row_mxfp4) {
        g.backward_ds_fp4_row = reinterpret_cast<uint8_t *>(DS_fp4_row.data_ptr());
        g.backward_ds_sc_row = reinterpret_cast<uint8_t *>(DS_sc_row.data_ptr());
        g.emit_backward_ds_row_mxfp4 = 1;
    }
    if (!fused_stage_probe && emit_col_mxfp4) {
        g.backward_ds_fp4_col = reinterpret_cast<uint8_t *>(DS_fp4_col.data_ptr());
        g.backward_ds_sc_col = reinterpret_cast<uint8_t *>(DS_sc_col.data_ptr());
        g.emit_backward_ds_col_mxfp4 = 1;
    }
    g.backward_ds_quant_mode = p_quant_mode;

    constexpr int SAFE_K_TILE_WINDOW = 3;
    auto launch_saved_lse_kernel = [&]<bool EMIT_COL_MXFP4, bool STORE_BF16, bool FUSED_STAGE_PROBE>() {
        CUDACHECK(cudaFuncSetAttribute(
            kernel_quantize_p_from_scores_debug<C, true, true, true, true, EMIT_COL_MXFP4, STORE_BF16, FUSED_STAGE_PROBE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            g.dynamic_shared_memory()));
        LaunchConfig<true, false> launch_config(g.grid_exact(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        for (int row_cluster_base = 0; row_cluster_base < row_tiles; row_cluster_base += C::CLUSTER_SIZE) {
            const int cluster_tiles = min(C::CLUSTER_SIZE, row_tiles - row_cluster_base);
            const int causal_tiles = min(row_cluster_base + cluster_tiles, row_tiles);
            g.selected_row_tile = row_cluster_base;
            g.selected_row_tile_span = cluster_tiles;
            g.tile_begin = 0;
            g.tiles_to_run = causal_tiles;
            for (int k_base = 0; k_base < causal_tiles; k_base += SAFE_K_TILE_WINDOW) {
                const int k_count = min(SAFE_K_TILE_WINDOW, causal_tiles - k_base);
                g.max_k_iters = k_count;
                g.compute_k_iter_base = k_base;
                g.debug_stage_cut = DEBUG_STAGE_CUT_FRESH3_FULL;
                CUDACHECK(cudaLaunchKernelEx(
                    launch_config,
                    kernel_quantize_p_from_scores_debug<C, true, true, true, true, EMIT_COL_MXFP4, STORE_BF16, FUSED_STAGE_PROBE>,
                    g));
            }
        }
    };
    if (emit_col_mxfp4) {
        if (store_bf16_validation) {
            launch_saved_lse_kernel.template operator()<true, true, false>();
        } else {
            launch_saved_lse_kernel.template operator()<true, false, false>();
        }
    } else {
        if (store_bf16_validation) {
            launch_saved_lse_kernel.template operator()<false, true, false>();
        } else {
            launch_saved_lse_kernel.template operator()<false, false, false>();
        }
    }

    if (emit_col_mxfp4 && !store_bf16_validation) {
        return {DS_fp4_row, DS_sc_row, DS_fp4_col, DS_sc_col};
    }
    if (emit_col_mxfp4) {
        return {DS_fp4_row, DS_sc_row, DS_fp4_col, DS_sc_col, DS_bf16};
    }
    if (!store_bf16_validation) {
        return {DS_fp4_row, DS_sc_row};
    }
    return {DS_fp4_row, DS_sc_row, DS_bf16};
}

std::vector<at::Tensor> dispatch_backward_ds_streaming_live_qk_saved_lse_clustered_full_row_mxfp4(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale
) {
    return dispatch_backward_ds_streaming_live_qk_saved_lse_clustered_full_row_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, LSE, DP, Delta, p_quant_mode, softmax_scale, 3, true, false);
}

std::vector<at::Tensor> dispatch_backward_ds_streaming_live_qk_saved_lse_clustered_full_row_mxfp4_no_bf16(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale
) {
    return dispatch_backward_ds_streaming_live_qk_saved_lse_clustered_full_row_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, LSE, DP, Delta, p_quant_mode, softmax_scale, 3, false, false);
}

std::vector<at::Tensor> dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4_impl(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int safe_k_tile_window,
    bool store_bf16_validation,
    bool emit_col_mxfp4,
    bool fused_stage_probe = false,
    bool emit_row_mxfp4 = true,
    bool fused_stage_transpose_probe = false
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(LSE); CHECK_INPUT(DP); CHECK_INPUT(Delta);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(DP.dtype() == at::ScalarType::BFloat16, "DP must be BF16");
    TORCH_CHECK(Delta.dtype() == at::ScalarType::Float, "Delta must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");
    TORCH_CHECK(p_quant_mode >= 0 && p_quant_mode <= 2, "p_quant_mode must be 0 (rte), 1 (encode), or 2 (decode)");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4, "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(DP.dim() == 4, "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(DP.size(0) == batch && DP.size(1) == heads && DP.size(2) == seqlen && DP.size(3) == seqlen,
                "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(Delta.dim() == 3, "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(Delta.size(0) == batch && Delta.size(1) == heads && Delta.size(2) == seqlen,
                "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    const int row_tiles = seqlen / C::Mb;
    TORCH_CHECK(safe_k_tile_window > 0 && safe_k_tile_window <= row_tiles,
                "safe_k_tile_window must be in [1, seqlen / 128]");
    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto DS_fp4_row = (fused_stage_probe || !emit_row_mxfp4)
        ? Q.new_empty({0}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2))
        : at::empty({batch, heads, seqlen, seqlen / 2}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2));
    auto DS_sc_row = (fused_stage_probe || !emit_row_mxfp4)
        ? Q_sg.new_empty({0}, Q_sg.options().dtype(at::ScalarType::Byte))
        : at::empty({batch, heads, row_tiles, seqlen / C::Nb, 32, 16}, Q.options().dtype(at::ScalarType::Byte));
    auto DS_fp4_col = (!fused_stage_probe && emit_col_mxfp4)
        ? at::empty({batch, heads, seqlen, seqlen / 2}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2))
        : Q.new_empty({0}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2));
    auto DS_sc_col = (!fused_stage_probe && emit_col_mxfp4)
        ? at::empty({batch, heads, row_tiles, seqlen / C::Nb, 32, 16}, Q.options().dtype(at::ScalarType::Byte))
        : Q_sg.new_empty({0}, Q_sg.options().dtype(at::ScalarType::Byte));
    auto DS_bf16 = (!fused_stage_probe && store_bf16_validation)
        ? Q_sg.new_zeros({batch, heads, seqlen, seqlen}, Q_sg.options().dtype(at::ScalarType::BFloat16))
        : Q_sg.new_empty({0}, Q_sg.options().dtype(at::ScalarType::BFloat16));
    auto fused_stage_marker = fused_stage_probe
        ? Q_sg.new_empty({batch, heads, seqlen, row_tiles}, Q_sg.options().dtype(at::ScalarType::Int))
        : Q_sg.new_empty({0}, Q_sg.options().dtype(at::ScalarType::Int));
    if (!fused_stage_probe && emit_row_mxfp4) {
        CUDACHECK(cudaMemset(DS_sc_row.data_ptr(), 0, DS_sc_row.nbytes()));
    }
    if (!fused_stage_probe && emit_col_mxfp4) {
        CUDACHECK(cudaMemset(DS_sc_col.data_ptr(), 0, DS_sc_col.nbytes()));
    }

    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        store_bf16_validation ? reinterpret_cast<bf16 *>(DS_bf16.data_ptr()) : nullptr,
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        -1,
        0,
        row_tiles,
    };
    g.output_full_layout = 1;
    g.disable_lse_store = 1;
    g.use_external_lse = 1;
    g.backward_dp_bf16 = reinterpret_cast<bf16 *>(DP.data_ptr());
    g.backward_delta = reinterpret_cast<float *>(Delta.data_ptr());
    g.backward_softmax_scale = static_cast<float>(softmax_scale);
    g.emit_backward_ds = 1;
    if (fused_stage_probe) {
        g.backward_fused_stage_marker = reinterpret_cast<uint32_t *>(fused_stage_marker.data_ptr<int>());
        g.backward_fused_stage_transpose = fused_stage_transpose_probe ? 1 : 0;
    } else if (emit_row_mxfp4) {
        g.backward_ds_fp4_row = reinterpret_cast<uint8_t *>(DS_fp4_row.data_ptr());
        g.backward_ds_sc_row = reinterpret_cast<uint8_t *>(DS_sc_row.data_ptr());
        g.emit_backward_ds_row_mxfp4 = 1;
    }
    if (!fused_stage_probe && emit_col_mxfp4) {
        g.backward_ds_fp4_col = reinterpret_cast<uint8_t *>(DS_fp4_col.data_ptr());
        g.backward_ds_sc_col = reinterpret_cast<uint8_t *>(DS_sc_col.data_ptr());
        g.emit_backward_ds_col_mxfp4 = 1;
    }
    g.backward_ds_quant_mode = p_quant_mode;

    auto launch_saved_lse_kernel = [&]<bool EMIT_ROW_MXFP4, bool EMIT_COL_MXFP4, bool STORE_BF16, bool FUSED_STAGE_PROBE>() {
        CUDACHECK(cudaFuncSetAttribute(
            kernel_quantize_p_from_scores_debug<C, true, true, true, EMIT_ROW_MXFP4, EMIT_COL_MXFP4, STORE_BF16, FUSED_STAGE_PROBE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            g.dynamic_shared_memory()));
        LaunchConfig<true, false> launch_config(g.grid_exact(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        for (int k_base = 0; k_base < row_tiles; k_base += safe_k_tile_window) {
            const int k_count = min(safe_k_tile_window, row_tiles - k_base);
            g.max_k_iters = k_count;
            g.compute_k_iter_base = k_base;
            g.debug_stage_cut = DEBUG_STAGE_CUT_FRESH3_FULL;
            CUDACHECK(cudaLaunchKernelEx(
                launch_config,
                kernel_quantize_p_from_scores_debug<C, true, true, true, EMIT_ROW_MXFP4, EMIT_COL_MXFP4, STORE_BF16, FUSED_STAGE_PROBE>,
                g));
        }
    };
    if (fused_stage_probe) {
        launch_saved_lse_kernel.template operator()<false, false, false, true>();
    } else if (emit_col_mxfp4 && emit_row_mxfp4) {
        if (store_bf16_validation) {
            launch_saved_lse_kernel.template operator()<true, true, true, false>();
        } else {
            launch_saved_lse_kernel.template operator()<true, true, false, false>();
        }
    } else if (emit_col_mxfp4) {
        if (store_bf16_validation) {
            launch_saved_lse_kernel.template operator()<false, true, true, false>();
        } else {
            launch_saved_lse_kernel.template operator()<false, true, false, false>();
        }
    } else {
        if (store_bf16_validation) {
            launch_saved_lse_kernel.template operator()<true, false, true, false>();
        } else {
            launch_saved_lse_kernel.template operator()<true, false, false, false>();
        }
    }

    if (fused_stage_probe) {
        return {fused_stage_marker};
    }
    if (emit_col_mxfp4 && !emit_row_mxfp4) {
        return {DS_fp4_col, DS_sc_col};
    }
    if (emit_col_mxfp4 && !store_bf16_validation) {
        return {DS_fp4_row, DS_sc_row, DS_fp4_col, DS_sc_col};
    }
    if (emit_col_mxfp4) {
        return {DS_fp4_row, DS_sc_row, DS_fp4_col, DS_sc_col, DS_bf16};
    }
    if (!store_bf16_validation) {
        return {DS_fp4_row, DS_sc_row};
    }
    return {DS_fp4_row, DS_sc_row, DS_bf16};
}

std::vector<at::Tensor> dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale
) {
    return dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, LSE, DP, Delta, p_quant_mode, softmax_scale, 3, true, false);
}

std::vector<at::Tensor> dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4_no_bf16(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale
) {
    return dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, LSE, DP, Delta, p_quant_mode, softmax_scale, 3, false, false);
}

std::vector<at::Tensor> dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_col_mxfp4_no_bf16(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale
) {
    return dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, LSE, DP, Delta, p_quant_mode, softmax_scale, 3, false, true);
}

std::vector<at::Tensor> dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_col_mxfp4_window_no_bf16(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int safe_k_tile_window
) {
    return dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, LSE, DP, Delta, p_quant_mode, softmax_scale, safe_k_tile_window, false, true);
}

std::vector<at::Tensor> dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_col_mxfp4_window_no_bf16(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int safe_k_tile_window
) {
    return dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, LSE, DP, Delta, p_quant_mode, softmax_scale, safe_k_tile_window, false, true, false, false);
}

std::vector<at::Tensor> dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_fused_stage_mxfp4_no_global(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int safe_k_tile_window
) {
    return dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, LSE, DP, Delta, p_quant_mode, softmax_scale, safe_k_tile_window, false, false, true);
}

std::vector<at::Tensor> dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_fused_col_stage_mxfp4_no_global(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int safe_k_tile_window
) {
    return dispatch_backward_ds_streaming_live_qk_saved_lse_fullgrid_row_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, LSE, DP, Delta, p_quant_mode, softmax_scale, safe_k_tile_window, false, false, true, true, true);
}
#endif

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut,
    bool emit_ds_row,
    bool emit_ds_col,
    bool direct_dq_bshd = false,
    bool zero_ds_col_sc = true,
    bool local_stage_transpose = false,
    bool zero_ds_row_sc = true,
    at::Tensor DQ_pad_out = at::Tensor(),
    at::Tensor DQ_bshd_out = at::Tensor(),
    at::Tensor DS_fp4_row_out = at::Tensor(),
    at::Tensor DS_sc_row_out = at::Tensor(),
    at::Tensor DS_fp4_col_out = at::Tensor(),
    at::Tensor DS_sc_col_out = at::Tensor(),
    bool live_dp = false,
    at::Tensor DOut_row_fp4 = at::Tensor(),
    at::Tensor DOut_row_sc = at::Tensor(),
    at::Tensor V_row_fp4 = at::Tensor(),
    at::Tensor V_row_sc = at::Tensor(),
    at::Tensor Pair_K_col_fp4 = at::Tensor(),
    at::Tensor Pair_K_col_sc = at::Tensor(),
    at::Tensor Pair_DQ_pad_out = at::Tensor()
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(K_col_fp4); CHECK_INPUT(K_col_sc);
    CHECK_INPUT(LSE); CHECK_INPUT(Delta);
    if (!live_dp) {
        CHECK_INPUT(DP);
    } else {
        if (DP.defined()) {
            CHECK_INPUT(DP);
        }
        CHECK_INPUT(DOut_row_fp4);
        CHECK_INPUT(DOut_row_sc);
        CHECK_INPUT(V_row_fp4);
        CHECK_INPUT(V_row_sc);
    }

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(K_col_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K_col_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(K_col_sc.dtype() == at::ScalarType::Byte, "K_col_sc must be uint8 E8M0 scales");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    if (!live_dp || DP.defined()) {
        TORCH_CHECK(DP.dtype() == at::ScalarType::BFloat16, "DP must be BF16");
    }
    TORCH_CHECK(Delta.dtype() == at::ScalarType::Float, "Delta must be float32");
    TORCH_CHECK(p_quant_mode >= 0 && p_quant_mode <= 2, "p_quant_mode must be 0 (rte), 1 (encode), or 2 (decode)");
    TORCH_CHECK(
        debug_stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL ||
            (debug_stage_cut >= DEBUG_STAGE_CUT_BWD_DQ_STAGE_ONLY &&
             debug_stage_cut <= DEBUG_STAGE_CUT_BWD_DQ_NOSEM_FULL),
        "debug_stage_cut must be DEBUG_STAGE_CUT_FRESH3_FULL or a fused-dQ debug cut"
    );

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    constexpr int DQ_PAD = 256;

    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");
    TORCH_CHECK(K.size(0) == batch && K.size(1) == heads && K.size(2) == seqlen, "K shape must match Q");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q/K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(K_col_fp4.dim() == 4, "K_col_fp4 must have shape (batch, heads, 256, seqlen/2)");
    TORCH_CHECK(K_col_fp4.size(0) == batch && K_col_fp4.size(1) == heads &&
                K_col_fp4.size(2) == DQ_PAD && K_col_fp4.size(3) == seqlen / 2,
                "K_col_fp4 must have shape (batch, heads, 256, seqlen/2)");
    TORCH_CHECK(K_col_sc.dim() == 6, "K_col_sc must have shape (batch, heads, 2, seqlen/128, 32, 16)");
    TORCH_CHECK(K_col_sc.size(0) == batch && K_col_sc.size(1) == heads &&
                K_col_sc.size(2) == DQ_PAD / 128 && K_col_sc.size(3) == seqlen / 128 &&
                K_col_sc.size(4) == 32 && K_col_sc.size(5) == 16,
                "K_col_sc must have shape (batch, heads, 2, seqlen/128, 32, 16)");
    TORCH_CHECK(LSE.dim() == 4 && LSE.size(0) == batch && LSE.size(1) == heads &&
                LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
    if (!live_dp || DP.defined()) {
        TORCH_CHECK(DP.dim() == 4 && DP.size(0) == batch && DP.size(1) == heads &&
                    DP.size(2) == seqlen && DP.size(3) == seqlen,
                    "DP must have shape (batch, heads, seqlen, seqlen)");
    }
    TORCH_CHECK(Delta.dim() == 3 && Delta.size(0) == batch && Delta.size(1) == heads &&
                Delta.size(2) == seqlen,
                "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    const int row_tiles = seqlen / C::Mb;
    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;
    const bool pair_single_kernel = Pair_K_col_fp4.defined() || Pair_K_col_sc.defined() || Pair_DQ_pad_out.defined();
    const bool live_dp_dq_row_path =
        debug_stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL &&
        emit_ds_row &&
        !emit_ds_col &&
        !local_stage_transpose;
    const bool live_dp_dq_col_path =
        debug_stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL &&
        !emit_ds_row &&
        emit_ds_col &&
        !direct_dq_bshd &&
        !local_stage_transpose;
    const bool live_dp_dq_no_ds_path =
        debug_stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL &&
        !emit_ds_row &&
        !emit_ds_col &&
        !direct_dq_bshd &&
        !local_stage_transpose;
    const bool live_dp_dk_transpose_path =
        debug_stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL &&
        !emit_ds_row &&
        !emit_ds_col &&
        !direct_dq_bshd &&
        local_stage_transpose;
    const bool live_dp_pair_no_ds_path =
        pair_single_kernel &&
        !emit_ds_row &&
        !emit_ds_col &&
        !direct_dq_bshd &&
        !local_stage_transpose;
    TORCH_CHECK(
        !live_dp || live_dp_dq_row_path || live_dp_dq_col_path || live_dp_dq_no_ds_path ||
            live_dp_dk_transpose_path || live_dp_pair_no_ds_path,
        "live dP is currently supported only for fused dQ, row/col-dS dQ, or transposed recompute-dK paths"
    );

    auto check_prealloc = [](const at::Tensor &t, at::IntArrayRef shape, at::ScalarType dtype, const char *name) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name, " must be a contiguous CUDA tensor");
        TORCH_CHECK(t.dtype() == dtype, name, " has wrong dtype");
        TORCH_CHECK(t.sizes() == shape, name, " has wrong shape");
    };

    if (DQ_pad_out.defined()) {
        check_prealloc(DQ_pad_out, {batch, heads, seqlen, DQ_PAD}, at::ScalarType::BFloat16, "DQ_pad_out");
    }
    if (DQ_bshd_out.defined()) {
        check_prealloc(DQ_bshd_out, {batch, seqlen, heads, 192}, at::ScalarType::BFloat16, "DQ_bshd_out");
    }
    if (DS_fp4_row_out.defined()) {
        check_prealloc(DS_fp4_row_out, {batch, heads, seqlen, seqlen / 2}, at::ScalarType::Float4_e2m1fn_x2, "DS_fp4_row_out");
    }
    if (DS_sc_row_out.defined()) {
        check_prealloc(DS_sc_row_out, {batch, heads, row_tiles, seqlen / C::Nb, 32, 16}, at::ScalarType::Byte, "DS_sc_row_out");
    }
    if (DS_fp4_col_out.defined()) {
        check_prealloc(DS_fp4_col_out, {batch, heads, seqlen, seqlen / 2}, at::ScalarType::Float4_e2m1fn_x2, "DS_fp4_col_out");
    }
    if (DS_sc_col_out.defined()) {
        check_prealloc(DS_sc_col_out, {batch, heads, row_tiles, seqlen / C::Nb, 32, 16}, at::ScalarType::Byte, "DS_sc_col_out");
    }
    if (live_dp) {
        check_prealloc(DOut_row_fp4, {batch, heads, seqlen, C::Dvo / 2}, at::ScalarType::Float4_e2m1fn_x2, "DOut_row_fp4");
        check_prealloc(DOut_row_sc, {batch, heads, row_tiles, C::Dvo / 128, 32, 16}, at::ScalarType::Byte, "DOut_row_sc");
        check_prealloc(V_row_fp4, {batch, heads, seqlen, C::Dvo / 2}, at::ScalarType::Float4_e2m1fn_x2, "V_row_fp4");
        check_prealloc(V_row_sc, {batch, heads, row_tiles, C::Dvo / 128, 32, 16}, at::ScalarType::Byte, "V_row_sc");
    }
    if (pair_single_kernel) {
        check_prealloc(Pair_K_col_fp4, {batch, heads, DQ_PAD, seqlen / 2}, at::ScalarType::Float4_e2m1fn_x2, "Pair_K_col_fp4");
        check_prealloc(Pair_K_col_sc, {batch, heads, DQ_PAD / C::Dvo, seqlen / C::Nb, 32, 16}, at::ScalarType::Byte, "Pair_K_col_sc");
        check_prealloc(Pair_DQ_pad_out, {batch, heads, seqlen, DQ_PAD}, at::ScalarType::BFloat16, "Pair_DQ_pad_out");
        TORCH_CHECK(live_dp && !direct_dq_bshd && !local_stage_transpose && !emit_ds_row && !emit_ds_col,
                    "pair single-kernel live-dP probe supports only pad-layout dQ+dK without dS emission");
    }
    TORCH_CHECK(!direct_dq_bshd || !DQ_pad_out.defined(), "DQ_pad_out is incompatible with direct_dq_bshd");
    TORCH_CHECK(direct_dq_bshd || !DQ_bshd_out.defined(), "DQ_bshd_out requires direct_dq_bshd");
    TORCH_CHECK(emit_ds_row || (!DS_fp4_row_out.defined() && !DS_sc_row_out.defined()), "row dS outputs require emit_ds_row");
    TORCH_CHECK(emit_ds_col || (!DS_fp4_col_out.defined() && !DS_sc_col_out.defined()), "col dS outputs require emit_ds_col");
    TORCH_CHECK(DS_fp4_row_out.defined() == DS_sc_row_out.defined(), "row dS preallocated payload/scale outputs must be provided together");
    TORCH_CHECK(DS_fp4_col_out.defined() == DS_sc_col_out.defined(), "col dS preallocated payload/scale outputs must be provided together");

    auto DQ_pad = direct_dq_bshd
        ? at::empty({0}, Q.options().dtype(at::ScalarType::BFloat16))
        : (DQ_pad_out.defined()
            ? DQ_pad_out
            : at::empty({batch, heads, seqlen, DQ_PAD}, Q.options().dtype(at::ScalarType::BFloat16)));
    auto DQ_bshd = direct_dq_bshd
        ? (DQ_bshd_out.defined()
            ? DQ_bshd_out
            : at::empty({batch, seqlen, heads, 192}, Q.options().dtype(at::ScalarType::BFloat16)))
        : at::empty({0}, Q.options().dtype(at::ScalarType::BFloat16));
    auto DS_fp4_row = emit_ds_row
        ? (DS_fp4_row_out.defined()
            ? DS_fp4_row_out
            : at::empty({batch, heads, seqlen, seqlen / 2}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2)))
        : Q.new_empty({0}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2));
    auto DS_sc_row = emit_ds_row
        ? (DS_sc_row_out.defined()
            ? DS_sc_row_out
            : at::empty({batch, heads, row_tiles, seqlen / C::Nb, 32, 16}, Q.options().dtype(at::ScalarType::Byte)))
        : Q_sg.new_empty({0}, Q_sg.options().dtype(at::ScalarType::Byte));
    auto DS_fp4_col = emit_ds_col
        ? (DS_fp4_col_out.defined()
            ? DS_fp4_col_out
            : at::empty({batch, heads, seqlen, seqlen / 2}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2)))
        : Q.new_empty({0}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2));
    auto DS_sc_col = emit_ds_col
        ? (DS_sc_col_out.defined()
            ? DS_sc_col_out
            : at::empty({batch, heads, row_tiles, seqlen / C::Nb, 32, 16}, Q.options().dtype(at::ScalarType::Byte)))
        : Q_sg.new_empty({0}, Q_sg.options().dtype(at::ScalarType::Byte));
    const bool row_future_sc_zeroed_in_kernel =
        emit_ds_row && zero_ds_row_sc && debug_stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL;
    if (emit_ds_row && zero_ds_row_sc && !row_future_sc_zeroed_in_kernel) {
        CUDACHECK(cudaMemset(DS_sc_row.data_ptr(), 0, DS_sc_row.nbytes()));
    }
    const bool col_future_sc_zeroed_in_kernel = emit_ds_col && zero_ds_col_sc;
    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        -1,
        0,
        row_tiles,
    };
    g.output_full_layout = 1;
    g.disable_lse_store = 1;
    g.use_external_lse = 1;
    g.backward_dp_bf16 = DP.defined() ? reinterpret_cast<bf16 *>(DP.data_ptr()) : nullptr;
    if (live_dp) {
        g.backward_dout_row_fp4 = reinterpret_cast<const uint8_t *>(DOut_row_fp4.data_ptr());
        g.backward_dout_row_sc = reinterpret_cast<const uint8_t *>(DOut_row_sc.data_ptr());
        g.backward_v_row_fp4 = reinterpret_cast<const uint8_t *>(V_row_fp4.data_ptr());
        g.backward_v_row_sc = reinterpret_cast<const uint8_t *>(V_row_sc.data_ptr());
    }
    g.backward_delta = reinterpret_cast<float *>(Delta.data_ptr());
    g.backward_softmax_scale = static_cast<float>(softmax_scale);
    g.emit_backward_ds = 1;
    g.backward_ds_quant_mode = p_quant_mode;
    if (emit_ds_row) {
        g.backward_ds_fp4_row = reinterpret_cast<uint8_t *>(DS_fp4_row.data_ptr());
        g.backward_ds_sc_row = reinterpret_cast<uint8_t *>(DS_sc_row.data_ptr());
        g.emit_backward_ds_row_mxfp4 = 1;
        g.backward_ds_row_skip_future_zero = row_future_sc_zeroed_in_kernel ? 0 : 1;
    }
    if (emit_ds_col) {
        g.backward_ds_fp4_col = reinterpret_cast<uint8_t *>(DS_fp4_col.data_ptr());
        g.backward_ds_sc_col = reinterpret_cast<uint8_t *>(DS_sc_col.data_ptr());
        g.emit_backward_ds_col_mxfp4 = 1;
        g.backward_ds_col_skip_future_zero = col_future_sc_zeroed_in_kernel ? 0 : 1;
    }
    g.backward_dq_k_col_fp4 = reinterpret_cast<const uint8_t *>(K_col_fp4.data_ptr());
    g.backward_dq_k_col_sc = reinterpret_cast<const uint8_t *>(K_col_sc.data_ptr());
    g.backward_dq_pad = direct_dq_bshd ? nullptr : reinterpret_cast<bf16 *>(DQ_pad.data_ptr());
    g.backward_dq_bshd = direct_dq_bshd ? reinterpret_cast<bf16 *>(DQ_bshd.data_ptr()) : nullptr;
    if (pair_single_kernel) {
        g.backward_pair_k_col_fp4 = reinterpret_cast<const uint8_t *>(Pair_K_col_fp4.data_ptr());
        g.backward_pair_k_col_sc = reinterpret_cast<const uint8_t *>(Pair_K_col_sc.data_ptr());
        g.backward_pair_dq_pad = reinterpret_cast<bf16 *>(Pair_DQ_pad_out.data_ptr());
    }
    g.backward_fused_stage_transpose = local_stage_transpose ? 1 : 0;
    g.max_k_iters = -1;
    g.compute_k_iter_base = 0;
    g.debug_stage_cut = debug_stage_cut;
    if (const char *skip_row_store_env = std::getenv("TK_FA4_MXFP4_BWD_SKIP_GLOBAL_DS_ROW_STORE")) {
        g.backward_skip_global_ds_row_store = std::atoi(skip_row_store_env) != 0 ? 1 : 0;
    }
	    if (live_dp) {
	        if (const char *debug_mode_env = std::getenv("TK_FA4_MXFP4_LIVE_DP_DEBUG_MODE")) {
	            g.live_dp_debug_mode = std::atoi(debug_mode_env);
	        }
	        if (const char *selected_head_env = std::getenv("TK_FA4_MXFP4_LIVE_DP_SELECTED_HEAD")) {
	            g.live_dp_selected_head = std::atoi(selected_head_env);
	        }
	        if (const char *selected_row_env = std::getenv("TK_FA4_MXFP4_LIVE_DP_SELECTED_ROW_TILE")) {
	            g.selected_row_tile = std::atoi(selected_row_env);
	        }
	        if (const char *max_k_iters_env = std::getenv("TK_FA4_MXFP4_LIVE_DP_MAX_K_ITERS")) {
	            g.max_k_iters = std::atoi(max_k_iters_env);
	        }
	    }

    auto launch_fused_dq_kernel = [&]<bool EMIT_DS_ROW, bool EMIT_DS_COL, bool DUAL_HALVES, int FIXED_CUT = -1, bool FIXED_PRODUCTION_PATH = false, bool LIVE_DP_KERNEL = false, bool LIVE_DP_MODE0_ONLY_KERNEL = false, bool LIVE_DP_MODE0_X8_KERNEL = false, bool BWD_PAIR_SINGLE_KERNEL = false>() {
        static std::once_flag attr_once;
        const int dynamic_smem = LIVE_DP_KERNEL ? g.dynamic_shared_memory_live_dp() : g.dynamic_shared_memory();
        std::call_once(attr_once, [&]() {
            CUDACHECK(cudaFuncSetAttribute(
                kernel_quantize_p_from_scores_debug<C, true, true, true, EMIT_DS_ROW, EMIT_DS_COL, false, true, DUAL_HALVES, FIXED_CUT, FIXED_PRODUCTION_PATH, LIVE_DP_KERNEL, LIVE_DP_MODE0_ONLY_KERNEL, LIVE_DP_MODE0_X8_KERNEL, BWD_PAIR_SINGLE_KERNEL>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                dynamic_smem));
        });
        LaunchConfig<true, false> launch_config(g.grid_exact(), g.block(), dynamic_smem, at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
        constexpr int dq_launches = DUAL_HALVES ? 1 : DQ_PAD / C::Dvo;
        int first_dq_half = 0;
        int last_dq_half = dq_launches;
        if constexpr (BWD_PAIR_SINGLE_KERNEL && !DUAL_HALVES) {
            if (const char *half_env = std::getenv("TK_FA4_MXFP4_BWD_LIVE_PAIR_SINGLE_HALF")) {
                const int requested_half = std::atoi(half_env);
                TORCH_CHECK(requested_half >= 0 && requested_half < dq_launches,
                            "TK_FA4_MXFP4_BWD_LIVE_PAIR_SINGLE_HALF is out of range");
                first_dq_half = requested_half;
                last_dq_half = requested_half + 1;
            }
        }
        for (int dq_half = first_dq_half; dq_half < last_dq_half; ++dq_half) {
            g.backward_dq_half = DUAL_HALVES ? -1 : dq_half;
            CUDACHECK(cudaLaunchKernelEx(
                launch_config,
                kernel_quantize_p_from_scores_debug<C, true, true, true, EMIT_DS_ROW, EMIT_DS_COL, false, true, DUAL_HALVES, FIXED_CUT, FIXED_PRODUCTION_PATH, LIVE_DP_KERNEL, LIVE_DP_MODE0_ONLY_KERNEL, LIVE_DP_MODE0_X8_KERNEL, BWD_PAIR_SINGLE_KERNEL>,
                g));
        }
    };
#ifdef TK_FA4_DQ_ONLY_BUILD
    TORCH_CHECK(live_dp && emit_ds_row && !emit_ds_col && !direct_dq_bshd && !local_stage_transpose,
                "DQ-only build supports only the live-dP row dS fused dQ contract");
    launch_fused_dq_kernel.template operator()<true, false, true, DEBUG_STAGE_CUT_FRESH3_FULL, true, true, true, false>();
#else
    auto launch_fused_dq_kernel_for_cut = [&]<bool EMIT_DS_ROW, bool EMIT_DS_COL, bool DUAL_HALVES>() {
        if (live_dp) {
            const bool live_dp_debug_requested = std::getenv("TK_FA4_MXFP4_LIVE_DP_DEBUG_MODE") != nullptr;
            const char *live_dp_static_x8_env = std::getenv("TK_FA4_MXFP4_LIVE_DP_STATIC_X8");
            const bool live_dp_static_x8_requested =
                live_dp_static_x8_env != nullptr && std::atoi(live_dp_static_x8_env) != 0;
            if (pair_single_kernel) {
                TORCH_CHECK(!live_dp_debug_requested && !live_dp_static_x8_requested,
                            "pair single-kernel live-dP path does not support live-dP debug or x8 modes");
                TORCH_CHECK(!EMIT_DS_ROW && !EMIT_DS_COL,
                            "pair single-kernel live-dP path supports only dQ+dK output without dS emission");
                launch_fused_dq_kernel.template operator()<false, false, false, DEBUG_STAGE_CUT_FRESH3_FULL, true, true, true, false, true>();
                return;
            }
            if constexpr (EMIT_DS_ROW && !EMIT_DS_COL) {
                if (live_dp_debug_requested) {
                    launch_fused_dq_kernel.template operator()<true, false, true, DEBUG_STAGE_CUT_FRESH3_FULL, true, true, false>();
                } else if (live_dp_static_x8_requested) {
                    launch_fused_dq_kernel.template operator()<true, false, true, DEBUG_STAGE_CUT_FRESH3_FULL, true, true, true, true>();
                } else {
                    launch_fused_dq_kernel.template operator()<true, false, true, DEBUG_STAGE_CUT_FRESH3_FULL, true, true, true>();
                }
            } else if constexpr (!EMIT_DS_ROW && EMIT_DS_COL) {
                if (live_dp_debug_requested) {
                    launch_fused_dq_kernel.template operator()<false, true, true, DEBUG_STAGE_CUT_FRESH3_FULL, true, true, false>();
                } else if (live_dp_static_x8_requested) {
                    launch_fused_dq_kernel.template operator()<false, true, true, DEBUG_STAGE_CUT_FRESH3_FULL, true, true, true, true>();
                } else {
                    launch_fused_dq_kernel.template operator()<false, true, true, DEBUG_STAGE_CUT_FRESH3_FULL, true, true, true>();
                }
            } else if constexpr (!EMIT_DS_ROW && !EMIT_DS_COL) {
                if (live_dp_debug_requested) {
                    launch_fused_dq_kernel.template operator()<false, false, true, DEBUG_STAGE_CUT_FRESH3_FULL, true, true, false>();
                } else if (live_dp_static_x8_requested) {
                    launch_fused_dq_kernel.template operator()<false, false, true, DEBUG_STAGE_CUT_FRESH3_FULL, true, true, true, true>();
                } else {
                    launch_fused_dq_kernel.template operator()<false, false, true, DEBUG_STAGE_CUT_FRESH3_FULL, true, true, true>();
                }
            } else {
                TORCH_CHECK(false, "live dP does not support simultaneous row and col dS emission");
            }
            return;
        }
        if constexpr (DUAL_HALVES) {
            if (debug_stage_cut == DEBUG_STAGE_CUT_FRESH3_FULL) {
                if constexpr ((EMIT_DS_COL && !EMIT_DS_ROW) || (EMIT_DS_ROW && !EMIT_DS_COL)) {
                    if (!direct_dq_bshd && !local_stage_transpose) {
                        launch_fused_dq_kernel.template operator()<EMIT_DS_ROW, EMIT_DS_COL, DUAL_HALVES, DEBUG_STAGE_CUT_FRESH3_FULL, true>();
                    } else {
                        launch_fused_dq_kernel.template operator()<EMIT_DS_ROW, EMIT_DS_COL, DUAL_HALVES, DEBUG_STAGE_CUT_FRESH3_FULL>();
                    }
                } else {
                    launch_fused_dq_kernel.template operator()<EMIT_DS_ROW, EMIT_DS_COL, DUAL_HALVES, DEBUG_STAGE_CUT_FRESH3_FULL>();
                }
            } else {
                launch_fused_dq_kernel.template operator()<EMIT_DS_ROW, EMIT_DS_COL, DUAL_HALVES>();
            }
        } else {
            launch_fused_dq_kernel.template operator()<EMIT_DS_ROW, EMIT_DS_COL, DUAL_HALVES>();
        }
    };
    if (emit_ds_row && emit_ds_col) {
        launch_fused_dq_kernel_for_cut.template operator()<true, true, true>();
    } else if (emit_ds_row) {
        launch_fused_dq_kernel_for_cut.template operator()<true, false, true>();
    } else if (emit_ds_col) {
        launch_fused_dq_kernel_for_cut.template operator()<false, true, true>();
    } else {
        launch_fused_dq_kernel_for_cut.template operator()<false, false, true>();
    }
#endif
    CUDACHECK(cudaGetLastError());
    auto DQ_result = direct_dq_bshd ? DQ_bshd : DQ_pad;
    if (emit_ds_row && emit_ds_col) {
        return {DQ_result, DS_fp4_row, DS_sc_row, DS_fp4_col, DS_sc_col};
    }
    if (emit_ds_row) {
        return {DQ_result, DS_fp4_row, DS_sc_row};
    }
    if (emit_ds_col) {
        return {DQ_result, DS_fp4_col, DS_sc_col};
    }
    return {DQ_result};
}

#ifndef TK_FA4_DQ_ONLY_BUILD
at::Tensor dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut, false, false)[0];
}

at::Tensor dispatch_backward_dk_streaming_live_qk_saved_lse_fused_mxfp4(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor Q_col_fp4,
    at::Tensor Q_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, Q_col_fp4, Q_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut, false, false, false, true, true)[0];
}

at::Tensor dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_out(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    at::Tensor DQ_pad_out,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        false, false, false, true, false,
        true,
        DQ_pad_out)[0];
}

at::Tensor dispatch_backward_dk_streaming_live_qk_saved_lse_fused_mxfp4_out(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor Q_col_fp4,
    at::Tensor Q_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    at::Tensor DK_pad_out,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, Q_col_fp4, Q_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        false, false, false, true, true,
        true,
        DK_pad_out)[0];
}

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut,
    bool zero_ds_row_sc = true
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        true, false, false, true, false, zero_ds_row_sc);
}

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_out(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    at::Tensor DQ_pad_out,
    at::Tensor DS_fp4_row_out,
    at::Tensor DS_sc_row_out,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut,
    bool zero_ds_row_sc = true
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        true, false, false, true, false, zero_ds_row_sc,
        DQ_pad_out, at::Tensor(), DS_fp4_row_out, DS_sc_row_out);
}
#endif

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_live_dp_out(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DOut_row_fp4,
    at::Tensor DOut_row_sc,
    at::Tensor V_row_fp4,
    at::Tensor V_row_sc,
    at::Tensor Delta,
    at::Tensor DQ_pad_out,
    at::Tensor DS_fp4_row_out,
    at::Tensor DS_sc_row_out,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut,
    bool zero_ds_row_sc = true
) {
    at::Tensor dummy_dp;
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, dummy_dp, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        true, false, false, true, false, zero_ds_row_sc,
        DQ_pad_out, at::Tensor(), DS_fp4_row_out, DS_sc_row_out, at::Tensor(), at::Tensor(),
        true, DOut_row_fp4, DOut_row_sc, V_row_fp4, V_row_sc);
}

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_live_dp_dq_bshd_out(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DOut_row_fp4,
    at::Tensor DOut_row_sc,
    at::Tensor V_row_fp4,
    at::Tensor V_row_sc,
    at::Tensor Delta,
    at::Tensor DQ_bshd_out,
    at::Tensor DS_fp4_row_out,
    at::Tensor DS_sc_row_out,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut,
    bool zero_ds_row_sc = true
) {
    at::Tensor dummy_dp;
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, dummy_dp, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        true, false, true, true, false, zero_ds_row_sc,
        at::Tensor(), DQ_bshd_out, DS_fp4_row_out, DS_sc_row_out, at::Tensor(), at::Tensor(),
        true, DOut_row_fp4, DOut_row_sc, V_row_fp4, V_row_sc);
}

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_live_dp_debug_out(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DOut_row_fp4,
    at::Tensor DOut_row_sc,
    at::Tensor V_row_fp4,
    at::Tensor V_row_sc,
    at::Tensor Delta,
    at::Tensor DQ_pad_out,
    at::Tensor DS_fp4_row_out,
    at::Tensor DS_sc_row_out,
    at::Tensor DP_live_debug_out,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut,
    bool zero_ds_row_sc = true
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, DP_live_debug_out, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        true, false, false, true, false, zero_ds_row_sc,
        DQ_pad_out, at::Tensor(), DS_fp4_row_out, DS_sc_row_out, at::Tensor(), at::Tensor(),
        true, DOut_row_fp4, DOut_row_sc, V_row_fp4, V_row_sc);
}

at::Tensor dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_live_dp_out(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DOut_row_fp4,
    at::Tensor DOut_row_sc,
    at::Tensor V_row_fp4,
    at::Tensor V_row_sc,
    at::Tensor Delta,
    at::Tensor DQ_pad_out,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut
) {
    at::Tensor dummy_dp;
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, dummy_dp, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        false, false, false, true, false, true,
        DQ_pad_out, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
        true, DOut_row_fp4, DOut_row_sc, V_row_fp4, V_row_sc)[0];
}

at::Tensor dispatch_backward_dk_streaming_live_qk_saved_lse_fused_mxfp4_live_dp_out(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor Q_col_fp4,
    at::Tensor Q_col_sc,
    at::Tensor LSE,
    at::Tensor DOut_row_fp4,
    at::Tensor DOut_row_sc,
    at::Tensor V_row_fp4,
    at::Tensor V_row_sc,
    at::Tensor Delta,
    at::Tensor DK_pad_out,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut
) {
    at::Tensor dummy_dp;
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, Q_col_fp4, Q_col_sc, LSE, dummy_dp, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        false, false, false, true, true, true,
        DK_pad_out, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
        true, DOut_row_fp4, DOut_row_sc, V_row_fp4, V_row_sc)[0];
}

std::vector<at::Tensor> dispatch_backward_dq_dk_streaming_live_qk_saved_lse_fused_mxfp4_live_dp_pair_out(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor Q_col_fp4,
    at::Tensor Q_col_sc,
    at::Tensor LSE,
    at::Tensor DOut_row_fp4,
    at::Tensor DOut_row_sc,
    at::Tensor V_row_fp4,
    at::Tensor V_row_sc,
    at::Tensor Delta,
    at::Tensor DQ_pad_out,
    at::Tensor DK_pad_out,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut
) {
    at::Tensor dummy_dp;
    at::Tensor dq;
    at::Tensor dk;
    const int batch = static_cast<int>(Q.size(0));
    const int heads = static_cast<int>(Q.size(1));
    const int seqlen = static_cast<int>(Q.size(2));
    const int tiles_m = seqlen / 128;
    const int num_block = (tiles_m + 1) / 2;
    const int pair_grid_blocks = batch * heads * num_block * 2;
    int live_pair_safe_exact_grid_blocks = 2048;
    if (const char *cap_env = std::getenv("TK_FA4_MXFP4_BWD_LIVE_PAIR_SAFE_EXACT_GRID_BLOCKS")) {
        live_pair_safe_exact_grid_blocks = std::atoi(cap_env);
    }
    bool single_kernel_pair = true;
    if (const char *single_env = std::getenv("TK_FA4_MXFP4_BWD_LIVE_PAIR_SINGLE_KERNEL")) {
        single_kernel_pair = std::atoi(single_env) != 0;
    } else if (live_pair_safe_exact_grid_blocks >= 0 &&
               pair_grid_blocks > live_pair_safe_exact_grid_blocks) {
        single_kernel_pair = false;
    }
    if (single_kernel_pair) {
        dq = dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, dummy_dp, Delta,
            p_quant_mode, softmax_scale, debug_stage_cut,
            false, false, false, true, false, true,
            DQ_pad_out, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            true, DOut_row_fp4, DOut_row_sc, V_row_fp4, V_row_sc,
            Q_col_fp4, Q_col_sc, DK_pad_out)[0];
        return {dq, DK_pad_out};
    }

    bool overlap_pair = true;
    if (const char *overlap_env = std::getenv("TK_FA4_MXFP4_BWD_LIVE_PAIR_OVERLAP")) {
        overlap_pair = std::atoi(overlap_env) != 0;
    } else {
        constexpr int LIVE_PAIR_OVERLAP_SAFE_EXACT_GRID_BLOCKS = 2048;
        if (pair_grid_blocks >= LIVE_PAIR_OVERLAP_SAFE_EXACT_GRID_BLOCKS) {
            overlap_pair = false;
        }
    }
    if (fp4fa4_bwd_env_enabled("TK_FA4_MXFP4_BWD_PROFILE_COMPONENTS")) {
        overlap_pair = false;
    }

    if (overlap_pair) {
        at::cuda::CUDAStream caller_stream = at::cuda::getCurrentCUDAStream(Q.device().index());
        at::cuda::CUDAStream dq_stream = at::cuda::getStreamFromPool(false, Q.device().index());
        at::cuda::CUDAStream dk_stream = at::cuda::getStreamFromPool(false, Q.device().index());
        cudaEvent_t ready_event = nullptr;
        cudaEvent_t dq_done_event = nullptr;
        cudaEvent_t dk_done_event = nullptr;
        CUDACHECK(cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(&dq_done_event, cudaEventDisableTiming));
        CUDACHECK(cudaEventCreateWithFlags(&dk_done_event, cudaEventDisableTiming));
        CUDACHECK(cudaEventRecord(ready_event, caller_stream.stream()));
        CUDACHECK(cudaStreamWaitEvent(dq_stream.stream(), ready_event, 0));
        CUDACHECK(cudaStreamWaitEvent(dk_stream.stream(), ready_event, 0));
        {
            c10::cuda::CUDAStreamGuard dq_guard(dq_stream);
            dq = dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
                Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, dummy_dp, Delta,
                p_quant_mode, softmax_scale, debug_stage_cut,
                false, false, false, true, false, true,
                DQ_pad_out, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                true, DOut_row_fp4, DOut_row_sc, V_row_fp4, V_row_sc)[0];
        }
        CUDACHECK(cudaEventRecord(dq_done_event, dq_stream.stream()));
        {
            c10::cuda::CUDAStreamGuard dk_guard(dk_stream);
            dk = dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
                Q, Q_sc, Q_sg, K, K_sc, K_sg, Q_col_fp4, Q_col_sc, LSE, dummy_dp, Delta,
                p_quant_mode, softmax_scale, debug_stage_cut,
                false, false, false, true, true, true,
                DK_pad_out, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                true, DOut_row_fp4, DOut_row_sc, V_row_fp4, V_row_sc)[0];
        }
        CUDACHECK(cudaEventRecord(dk_done_event, dk_stream.stream()));
        CUDACHECK(cudaStreamWaitEvent(caller_stream.stream(), dq_done_event, 0));
        CUDACHECK(cudaStreamWaitEvent(caller_stream.stream(), dk_done_event, 0));
        CUDACHECK(cudaEventDestroy(ready_event));
        CUDACHECK(cudaEventDestroy(dq_done_event));
        CUDACHECK(cudaEventDestroy(dk_done_event));
    } else {
        dq = dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, dummy_dp, Delta,
            p_quant_mode, softmax_scale, debug_stage_cut,
            false, false, false, true, false, true,
            DQ_pad_out, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            true, DOut_row_fp4, DOut_row_sc, V_row_fp4, V_row_sc)[0];
        dk = dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
            Q, Q_sc, Q_sg, K, K_sc, K_sg, Q_col_fp4, Q_col_sc, LSE, dummy_dp, Delta,
            p_quant_mode, softmax_scale, debug_stage_cut,
            false, false, false, true, true, true,
            DK_pad_out, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
            true, DOut_row_fp4, DOut_row_sc, V_row_fp4, V_row_sc)[0];
    }
    return {dq, dk};
}

#include "bwd_full_mxfp4_live_tensorcore.inc"
#include "bwd_key_owned_mxfp4_schedule.inc"

#ifndef TK_FA4_DQ_ONLY_BUILD
std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_dq_bshd(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut,
    bool zero_ds_row_sc = true
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        true, false, true, true, false, zero_ds_row_sc);
}

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_row_dq_bshd_out(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    at::Tensor DQ_bshd_out,
    at::Tensor DS_fp4_row_out,
    at::Tensor DS_sc_row_out,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut,
    bool zero_ds_row_sc = true
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        true, false, true, true, false, zero_ds_row_sc,
        at::Tensor(), DQ_bshd_out, DS_fp4_row_out, DS_sc_row_out);
}

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut, false, true);
}

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_nozero(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut, false, true, false, false);
}

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_out(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    at::Tensor DQ_pad_out,
    at::Tensor DS_fp4_col_out,
    at::Tensor DS_sc_col_out,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        false, true, false, true, false,
        true,
        DQ_pad_out, at::Tensor(), at::Tensor(), at::Tensor(), DS_fp4_col_out, DS_sc_col_out);
}

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_out_nozero(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    at::Tensor DQ_pad_out,
    at::Tensor DS_fp4_col_out,
    at::Tensor DS_sc_col_out,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        false, true, false, false, false,
        true,
        DQ_pad_out, at::Tensor(), at::Tensor(), at::Tensor(), DS_fp4_col_out, DS_sc_col_out);
}

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_live_dp_out(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DOut_row_fp4,
    at::Tensor DOut_row_sc,
    at::Tensor V_row_fp4,
    at::Tensor V_row_sc,
    at::Tensor Delta,
    at::Tensor DQ_pad_out,
    at::Tensor DS_fp4_col_out,
    at::Tensor DS_sc_col_out,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut,
    bool zero_ds_col_sc = true
) {
    at::Tensor dummy_dp;
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, dummy_dp, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut,
        false, true, false, zero_ds_col_sc, false,
        true, DQ_pad_out, at::Tensor(), at::Tensor(), at::Tensor(), DS_fp4_col_out, DS_sc_col_out,
        true, DOut_row_fp4, DOut_row_sc, V_row_fp4, V_row_sc);
}

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_dq_bshd(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut, false, true, true);
}

std::vector<at::Tensor> dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_with_ds_col_dq_bshd_nozero(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor K_col_fp4,
    at::Tensor K_col_sc,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int p_quant_mode,
    double softmax_scale,
    int debug_stage_cut
) {
    return dispatch_backward_dq_streaming_live_qk_saved_lse_fused_mxfp4_impl(
        Q, Q_sc, Q_sg, K, K_sc, K_sg, K_col_fp4, K_col_sc, LSE, DP, Delta,
        p_quant_mode, softmax_scale, debug_stage_cut, false, true, true, false);
}

__global__ void kernel_finalize_dq_dk_pad_bhsd_to_bshd(
    const at::BFloat16 *__restrict__ dq_pad,
    const at::BFloat16 *__restrict__ dk_pad,
    at::BFloat16 *__restrict__ dq,
    at::BFloat16 *__restrict__ dk,
    int batch,
    int heads,
    int seqlen
) {
    constexpr int D_OUT = 192;
    constexpr int D_PAD = 256;
    constexpr int BF16_PER_VEC = 8;
    constexpr int VECS_PER_ROW = D_OUT / BF16_PER_VEC;
    const int64_t total_vecs =
        static_cast<int64_t>(batch) * seqlen * heads * VECS_PER_ROW;
    auto dq_pad_vec = reinterpret_cast<const uint4 *>(dq_pad);
    auto dk_pad_vec = reinterpret_cast<const uint4 *>(dk_pad);
    auto dq_vec = reinterpret_cast<uint4 *>(dq);
    auto dk_vec = reinterpret_cast<uint4 *>(dk);
    for (int64_t vec_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         vec_idx < total_vecs;
         vec_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int d_vec = static_cast<int>(vec_idx % VECS_PER_ROW);
        int64_t tmp = vec_idx / VECS_PER_ROW;
        const int h = static_cast<int>(tmp % heads);
        tmp /= heads;
        const int s = static_cast<int>(tmp % seqlen);
        const int b = static_cast<int>(tmp / seqlen);
        const int64_t in_vec =
            (((static_cast<int64_t>(b) * heads + h) * seqlen + s) * (D_PAD / BF16_PER_VEC)) + d_vec;
        dq_vec[vec_idx] = dq_pad_vec[in_vec];
        dk_vec[vec_idx] = dk_pad_vec[in_vec];
    }
}

std::vector<at::Tensor> dispatch_finalize_dq_dk_pad_bhsd_to_bshd(
    at::Tensor DQ_pad,
    at::Tensor DK_pad
) {
    CHECK_INPUT(DQ_pad);
    CHECK_INPUT(DK_pad);
    TORCH_CHECK(DQ_pad.dtype() == at::ScalarType::BFloat16, "DQ_pad must be BF16");
    TORCH_CHECK(DK_pad.dtype() == at::ScalarType::BFloat16, "DK_pad must be BF16");
    TORCH_CHECK(DQ_pad.dim() == 4, "DQ_pad must have shape (B,H,S,256)");
    TORCH_CHECK(DK_pad.sizes() == DQ_pad.sizes(), "DK_pad must match DQ_pad shape");
    const int batch = static_cast<int>(DQ_pad.size(0));
    const int heads = static_cast<int>(DQ_pad.size(1));
    const int seqlen = static_cast<int>(DQ_pad.size(2));
    TORCH_CHECK(DQ_pad.size(3) == 256, "DQ_pad last dimension must be 256");
    TORCH_CHECK(seqlen > 0 && seqlen % 128 == 0, "seqlen must be positive and divisible by 128");

    auto DQ = at::empty({batch, seqlen, heads, 192}, DQ_pad.options());
    auto DK = at::empty({batch, seqlen, heads, 192}, DK_pad.options());
    const int threads = 256;
    const int64_t total_pairs = static_cast<int64_t>(batch) * seqlen * heads * (192 / 2);
    int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
    blocks = std::min(blocks, 65535);
    kernel_finalize_dq_dk_pad_bhsd_to_bshd<<<blocks, threads>>>(
        DQ_pad.data_ptr<at::BFloat16>(),
        DK_pad.data_ptr<at::BFloat16>(),
        DQ.data_ptr<at::BFloat16>(),
        DK.data_ptr<at::BFloat16>(),
        batch,
        heads,
        seqlen);
    CUDACHECK(cudaGetLastError());
    return {DQ, DK};
}

__global__ void kernel_finalize_dk_pad_bhsd_to_bshd(
    const at::BFloat16 *__restrict__ dk_pad,
    at::BFloat16 *__restrict__ dk,
    int batch,
    int heads,
    int seqlen
) {
    constexpr int D_OUT = 192;
    constexpr int D_PAD = 256;
    constexpr int BF16_PER_VEC = 8;
    constexpr int VECS_PER_ROW = D_OUT / BF16_PER_VEC;
    const int64_t total_vecs =
        static_cast<int64_t>(batch) * seqlen * heads * VECS_PER_ROW;
    auto dk_pad_vec = reinterpret_cast<const uint4 *>(dk_pad);
    auto dk_vec = reinterpret_cast<uint4 *>(dk);
    for (int64_t vec_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         vec_idx < total_vecs;
         vec_idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int d_vec = static_cast<int>(vec_idx % VECS_PER_ROW);
        int64_t tmp = vec_idx / VECS_PER_ROW;
        const int h = static_cast<int>(tmp % heads);
        tmp /= heads;
        const int s = static_cast<int>(tmp % seqlen);
        const int b = static_cast<int>(tmp / seqlen);
        const int64_t in_vec =
            (((static_cast<int64_t>(b) * heads + h) * seqlen + s) * (D_PAD / BF16_PER_VEC)) + d_vec;
        dk_vec[vec_idx] = dk_pad_vec[in_vec];
    }
}

at::Tensor dispatch_finalize_dk_pad_bhsd_to_bshd(
    at::Tensor DK_pad
) {
    CHECK_INPUT(DK_pad);
    TORCH_CHECK(DK_pad.dtype() == at::ScalarType::BFloat16, "DK_pad must be BF16");
    TORCH_CHECK(DK_pad.dim() == 4, "DK_pad must have shape (B,H,S,256)");
    const int batch = static_cast<int>(DK_pad.size(0));
    const int heads = static_cast<int>(DK_pad.size(1));
    const int seqlen = static_cast<int>(DK_pad.size(2));
    TORCH_CHECK(DK_pad.size(3) == 256, "DK_pad last dimension must be 256");
    TORCH_CHECK(seqlen > 0 && seqlen % 128 == 0, "seqlen must be positive and divisible by 128");

    auto DK = at::empty({batch, seqlen, heads, 192}, DK_pad.options());
    const int threads = 256;
    const int64_t total_pairs = static_cast<int64_t>(batch) * seqlen * heads * (192 / 2);
    int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
    blocks = std::min(blocks, 65535);
    kernel_finalize_dk_pad_bhsd_to_bshd<<<blocks, threads>>>(
        DK_pad.data_ptr<at::BFloat16>(),
        DK.data_ptr<at::BFloat16>(),
        batch,
        heads,
        seqlen);
    CUDACHECK(cudaGetLastError());
    return DK;
}

at::Tensor dispatch_backward_ds_streaming_live_qk_saved_lse_clustered_full_window(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int safe_k_tile_window,
    double softmax_scale
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(LSE); CHECK_INPUT(DP); CHECK_INPUT(Delta);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(DP.dtype() == at::ScalarType::BFloat16, "DP must be BF16");
    TORCH_CHECK(Delta.dtype() == at::ScalarType::Float, "Delta must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4, "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(DP.dim() == 4, "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(DP.size(0) == batch && DP.size(1) == heads && DP.size(2) == seqlen && DP.size(3) == seqlen,
                "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(Delta.dim() == 3, "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(Delta.size(0) == batch && Delta.size(1) == heads && Delta.size(2) == seqlen,
                "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");
    TORCH_CHECK(safe_k_tile_window >= 1 && safe_k_tile_window <= 3,
                "safe_k_tile_window must be in [1, 3]; larger windows deadlock this debug QK handoff");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    const int row_tiles = seqlen / C::Mb;
    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto DS_bf16 = Q_sg.new_zeros({batch, heads, seqlen, seqlen}, Q_sg.options().dtype(at::ScalarType::BFloat16));

    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(DS_bf16.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        0,
        0,
        row_tiles,
    };
    g.output_full_layout = 1;
    g.disable_lse_store = 1;
    g.use_external_lse = 1;
    g.backward_dp_bf16 = reinterpret_cast<bf16 *>(DP.data_ptr());
    g.backward_delta = reinterpret_cast<float *>(Delta.data_ptr());
    g.backward_softmax_scale = static_cast<float>(softmax_scale);
    g.emit_backward_ds = 1;

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(dim3(C::CLUSTER_SIZE), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    for (int row_cluster_base = 0; row_cluster_base < row_tiles; row_cluster_base += C::CLUSTER_SIZE) {
        const int cluster_tiles = min(C::CLUSTER_SIZE, row_tiles - row_cluster_base);
        const int causal_tiles = min(row_cluster_base + cluster_tiles, row_tiles);
        g.selected_row_tile = row_cluster_base;
        g.selected_row_tile_span = cluster_tiles;
        g.tile_begin = 0;
        g.tiles_to_run = causal_tiles;
        for (int k_base = 0; k_base < causal_tiles; k_base += safe_k_tile_window) {
            const int k_count = min(safe_k_tile_window, causal_tiles - k_base);
            g.max_k_iters = k_count;
            g.compute_k_iter_base = k_base;
            g.debug_stage_cut = DEBUG_STAGE_CUT_FRESH3_FULL;
            CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true, true>, g));
        }
    }

    return DS_bf16;
}

at::Tensor dispatch_backward_ds_streaming_live_qk_saved_lse_row_tile_maxk_debug(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor LSE,
    at::Tensor DP,
    at::Tensor Delta,
    int row_tile,
    int max_k_iters,
    int debug_stage_cut,
    double softmax_scale
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(LSE); CHECK_INPUT(DP); CHECK_INPUT(Delta);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");
    TORCH_CHECK(DP.dtype() == at::ScalarType::BFloat16, "DP must be BF16");
    TORCH_CHECK(Delta.dtype() == at::ScalarType::Float, "Delta must be float32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(LSE.dim() == 4, "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(LSE.size(0) == batch && LSE.size(1) == heads && LSE.size(2) == 1 && LSE.size(3) == seqlen,
                "LSE must have shape (batch, heads, 1, seqlen)");
    TORCH_CHECK(DP.dim() == 4, "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(DP.size(0) == batch && DP.size(1) == heads && DP.size(2) == seqlen && DP.size(3) == seqlen,
                "DP must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(Delta.dim() == 3, "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(Delta.size(0) == batch && Delta.size(1) == heads && Delta.size(2) == seqlen,
                "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    const int row_tiles = seqlen / C::Mb;
    TORCH_CHECK(row_tile >= 0 && row_tile < row_tiles, "row_tile must be within sequence row-tile range");
    const int row_cluster_base = row_tile - (row_tile % C::CLUSTER_SIZE);
    const int causal_tiles = min(row_cluster_base + C::CLUSTER_SIZE, row_tiles);
    TORCH_CHECK(max_k_iters > 0 && max_k_iters <= causal_tiles, "max_k_iters must be positive and within the selected row-cluster causal range");
    TORCH_CHECK(
        debug_stage_cut >= DEBUG_STAGE_CUT_FULL && debug_stage_cut <= DEBUG_STAGE_CUT_LEGACY_SINGLE_SLOT_FULL,
        "debug_stage_cut must be within the supported milestone range"
    );
    TORCH_CHECK(
        debug_stage_cut == DEBUG_STAGE_CUT_FULL ||
            debug_stage_cut == DEBUG_STAGE_CUT_DUAL_SLOT_FULL ||
            debug_stage_cut == DEBUG_STAGE_CUT_LEGACY_SINGLE_SLOT_FULL ||
            max_k_iters == 3,
        "non-full debug_stage_cut requires max_k_iters == 3"
    );

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto DS_tile = Q_sg.new_zeros({batch, heads, C::Mb, max_k_iters * C::Nb}, Q_sg.options().dtype(at::ScalarType::BFloat16));

    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(DS_tile.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        row_tile,
        0,
        max_k_iters,
    };
    g.max_k_iters = max_k_iters;
    g.disable_lse_store = 1;
    g.use_external_lse = 1;
    g.backward_dp_bf16 = reinterpret_cast<bf16 *>(DP.data_ptr());
    g.backward_delta = reinterpret_cast<float *>(Delta.data_ptr());
    g.backward_softmax_scale = static_cast<float>(softmax_scale);
    g.emit_backward_ds = 1;
    g.debug_stage_cut = debug_stage_cut;

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid_exact(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true, true>, g));

    return DS_tile;
}

__global__ void kernel_backward_ds_from_p_bf16(
    const bf16 *__restrict__ P,
    const bf16 *__restrict__ DP,
    const float *__restrict__ Delta,
    bf16 *__restrict__ DS,
    int64_t total_elems,
    int seqlen,
    float softmax_scale
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elems) {
        return;
    }
    const int64_t row_linear = idx / seqlen;
    const float p = __bfloat162float(P[idx]);
    const float dp = __bfloat162float(DP[idx]);
    const float delta = Delta[row_linear];
    DS[idx] = __float2bfloat16(p * (dp - delta) * softmax_scale);
}

at::Tensor dispatch_backward_ds_from_p_bf16(
    at::Tensor P,
    at::Tensor DP,
    at::Tensor Delta,
    double softmax_scale
) {
    CHECK_INPUT(P); CHECK_INPUT(DP); CHECK_INPUT(Delta);
    TORCH_CHECK(P.dtype() == at::ScalarType::BFloat16, "P must be BF16");
    TORCH_CHECK(DP.dtype() == at::ScalarType::BFloat16, "DP must be BF16");
    TORCH_CHECK(Delta.dtype() == at::ScalarType::Float, "Delta must be float32");
    TORCH_CHECK(P.dim() == 4, "P must have shape (batch, heads, seqlen, seqlen)");
    TORCH_CHECK(DP.dim() == 4, "DP must have shape (batch, heads, seqlen, seqlen)");
    const int batch = P.size(0);
    const int heads = P.size(1);
    const int seqlen = P.size(2);
    TORCH_CHECK(P.size(3) == seqlen, "P must be square in the last two dimensions");
    TORCH_CHECK(DP.size(0) == batch && DP.size(1) == heads && DP.size(2) == seqlen && DP.size(3) == seqlen,
                "DP must have the same shape as P");
    TORCH_CHECK(Delta.dim() == 3, "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(Delta.size(0) == batch && Delta.size(1) == heads && Delta.size(2) == seqlen,
                "Delta must have shape (batch, heads, seqlen)");
    TORCH_CHECK(std::isfinite(softmax_scale), "softmax_scale must be finite");

    auto DS = at::empty_like(P);
    const int64_t total_elems = P.numel();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_elems + threads - 1) / threads);
    kernel_backward_ds_from_p_bf16<<<blocks, threads, 0, 0>>>(
        reinterpret_cast<const bf16 *>(P.data_ptr()),
        reinterpret_cast<const bf16 *>(DP.data_ptr()),
        reinterpret_cast<const float *>(Delta.data_ptr()),
        reinterpret_cast<bf16 *>(DS.data_ptr()),
        total_elems,
        seqlen,
        static_cast<float>(softmax_scale)
    );
    CUDACHECK(cudaGetLastError());
    return DS;
}

at::Tensor dispatch_forward_streaming_live_qk_only_lse_only(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto LSE = Q_sg.new_empty({batch, heads, 1, seqlen}, Q_sg.options().dtype(at::ScalarType::Float));

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        -1,
        0,
        seqlen / C::Nb,
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid_exact(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true, true>, g));

    return LSE;
}

pybind11::tuple dispatch_quantize_p_from_scores_debug(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    bool return_debug
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto P_fp4 = Q.new_empty({batch, heads, seqlen, seqlen / 2}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2));
    // Zero-init the dumped prepared scales so partial-store bugs show up as deterministic zeros
    // instead of stale allocator contents when we debug the live quantization path.
    auto P_sc_prepared = Q_sc.new_zeros({batch, heads, seqlen / 128, seqlen / 64, 512}, Q_sc.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto LSE = Q_sg.new_empty({batch, heads, 1, seqlen}, Q_sg.options().dtype(at::ScalarType::Float));

    at::Tensor debug_row_max;
    at::Tensor debug_row_sum;
    at::Tensor debug_tile_amax;
    if (return_debug) {
        auto debug_opts = Q_sg.options().dtype(at::ScalarType::Float);
        debug_row_max = Q_sg.new_empty({batch, heads, seqlen, seqlen / 128}, debug_opts);
        debug_row_sum = Q_sg.new_empty({batch, heads, seqlen, seqlen / 128}, debug_opts);
        debug_tile_amax = Q_sg.new_empty({batch, heads, seqlen / 128, seqlen / 128}, debug_opts);
    }

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        reinterpret_cast<uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<uint8_t *>(P_sc_prepared.data_ptr()),
        nullptr,
        nullptr,
        nullptr,
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        return_debug ? reinterpret_cast<float *>(debug_row_max.data_ptr()) : nullptr,
        return_debug ? reinterpret_cast<float *>(debug_row_sum.data_ptr()) : nullptr,
        return_debug ? reinterpret_cast<float *>(debug_tile_amax.data_ptr()) : nullptr,
        0,
        seqlen / C::Nb,
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C>, g));

    if (!return_debug) {
        return pybind11::make_tuple(P_fp4, P_sc_prepared, LSE);
    }
    return pybind11::make_tuple(P_fp4, P_sc_prepared, LSE, debug_row_max, debug_row_sum, debug_tile_amax);
}

pybind11::tuple dispatch_quantize_p_from_scores_with_live_bf16_ref_debug(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto P_fp4 = Q.new_empty({batch, heads, seqlen, seqlen / 2}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2));
    auto P_sc_prepared = Q_sc.new_zeros({batch, heads, seqlen / 128, seqlen / 64, 512}, Q_sc.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto P_bf16_live = Q_sg.new_empty({batch, heads, seqlen, seqlen}, Q_sg.options().dtype(at::ScalarType::BFloat16));
    auto LSE = Q_sg.new_empty({batch, heads, 1, seqlen}, Q_sg.options().dtype(at::ScalarType::Float));
    auto debug_opts = Q_sg.options().dtype(at::ScalarType::Float);
    auto debug_row_max = Q_sg.new_empty({batch, heads, seqlen, seqlen / 128}, debug_opts);
    auto debug_row_sum = Q_sg.new_empty({batch, heads, seqlen, seqlen / 128}, debug_opts);
    auto debug_tile_amax = Q_sg.new_empty({batch, heads, seqlen / 128, seqlen / 128}, debug_opts);

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        reinterpret_cast<uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<uint8_t *>(P_sc_prepared.data_ptr()),
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(P_bf16_live.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        reinterpret_cast<float *>(debug_row_max.data_ptr()),
        reinterpret_cast<float *>(debug_row_sum.data_ptr()),
        reinterpret_cast<float *>(debug_tile_amax.data_ptr()),
        0,
        seqlen / C::Nb,
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true>, g));

    return pybind11::make_tuple(P_fp4, P_sc_prepared, P_bf16_live, LSE, debug_row_max, debug_row_sum, debug_tile_amax);
}

pybind11::tuple dispatch_quantize_p_from_scores_with_live_row_scales_and_bf16_ref_debug(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto P_fp4 = Q.new_empty({batch, heads, seqlen, seqlen / 2}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2));
    auto P_sc_prepared = Q_sc.new_zeros({batch, heads, seqlen / 128, seqlen / 64, 512}, Q_sc.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto P_sc_rowmajor_live = Q_sc.new_zeros({batch, heads, seqlen, seqlen / 16}, Q_sc.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto P_sc_rowmajor_scan = Q_sc.new_zeros({batch, heads, seqlen, seqlen / 16}, Q_sc.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto P_bf16_live = Q_sg.new_empty({batch, heads, seqlen, seqlen}, Q_sg.options().dtype(at::ScalarType::BFloat16));
    auto LSE = Q_sg.new_empty({batch, heads, 1, seqlen}, Q_sg.options().dtype(at::ScalarType::Float));
    auto debug_opts = Q_sg.options().dtype(at::ScalarType::Float);
    auto debug_row_max = Q_sg.new_empty({batch, heads, seqlen, seqlen / 128}, debug_opts);
    auto debug_row_sum = Q_sg.new_empty({batch, heads, seqlen, seqlen / 128}, debug_opts);
    auto debug_tile_amax = Q_sg.new_empty({batch, heads, seqlen / 128, seqlen / 128}, debug_opts);

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        reinterpret_cast<uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<uint8_t *>(P_sc_prepared.data_ptr()),
        reinterpret_cast<uint8_t *>(P_sc_rowmajor_live.data_ptr()),
        reinterpret_cast<uint8_t *>(P_sc_rowmajor_scan.data_ptr()),
        reinterpret_cast<bf16 *>(P_bf16_live.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        reinterpret_cast<float *>(debug_row_max.data_ptr()),
        reinterpret_cast<float *>(debug_row_sum.data_ptr()),
        reinterpret_cast<float *>(debug_tile_amax.data_ptr()),
        0,
        seqlen / C::Nb,
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true>, g));

    return pybind11::make_tuple(P_fp4, P_sc_prepared, P_sc_rowmajor_live, P_sc_rowmajor_scan, P_bf16_live, LSE, debug_row_max, debug_row_sum, debug_tile_amax);
}

pybind11::tuple dispatch_quantize_p_from_scores_with_live_bf16_ref_debug_direct(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto P_fp4 = Q.new_empty({batch, heads, seqlen, seqlen / 2}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2));
    auto P_sc_prepared = Q_sc.new_zeros({batch, heads, seqlen / 128, seqlen / 64, 512}, Q_sc.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto P_bf16_live = Q_sg.new_empty({batch, heads, seqlen, seqlen}, Q_sg.options().dtype(at::ScalarType::BFloat16));
    auto LSE = Q_sg.new_empty({batch, heads, 1, seqlen}, Q_sg.options().dtype(at::ScalarType::Float));
    auto debug_opts = Q_sg.options().dtype(at::ScalarType::Float);
    auto debug_row_max = Q_sg.new_empty({batch, heads, seqlen, seqlen / 128}, debug_opts);
    auto debug_row_sum = Q_sg.new_empty({batch, heads, seqlen, seqlen / 128}, debug_opts);
    auto debug_tile_amax = Q_sg.new_empty({batch, heads, seqlen / 128, seqlen / 128}, debug_opts);

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        reinterpret_cast<uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<uint8_t *>(P_sc_prepared.data_ptr()),
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(P_bf16_live.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        reinterpret_cast<float *>(debug_row_max.data_ptr()),
        reinterpret_cast<float *>(debug_row_sum.data_ptr()),
        reinterpret_cast<float *>(debug_tile_amax.data_ptr()),
        0,
        seqlen / C::Nb,
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, false>, g));

    return pybind11::make_tuple(P_fp4, P_sc_prepared, P_bf16_live, LSE, debug_row_max, debug_row_sum, debug_tile_amax);
}

pybind11::tuple dispatch_quantize_p_from_scores_tile_window_with_live_bf16_ref_debug(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    int tile_begin,
    int tiles_to_run,
    int debug_q_scale_mode,
    int debug_stage_cut,
    int debug_tmem_layout_mode,
    int debug_k_scale_depth_mode,
    int debug_qk_chunk_mask
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(tile_begin >= 0, "tile_begin must be non-negative");
    TORCH_CHECK(tiles_to_run > 0, "tiles_to_run must be positive");
    TORCH_CHECK(tile_begin + tiles_to_run <= seqlen / 128, "tile window must be within sequence tile range");
    TORCH_CHECK(
        debug_q_scale_mode >= DEBUG_Q_SCALE_MODE_TMEM &&
            debug_q_scale_mode <= DEBUG_Q_SCALE_MODE_SMEM_CHUNKED,
        "debug_q_scale_mode must be a supported q-scale staging mode"
    );
    TORCH_CHECK(
        debug_tmem_layout_mode >= DEBUG_TMEM_LAYOUT_COMPACT &&
            debug_tmem_layout_mode <= DEBUG_TMEM_LAYOUT_PRODUCTION_LIKE,
        "debug_tmem_layout_mode must be a supported TMEM layout mode"
    );
    TORCH_CHECK(
        debug_k_scale_depth_mode >= DEBUG_K_SCALE_DEPTH_MODE_TILE128 &&
            debug_k_scale_depth_mode <= DEBUG_K_SCALE_DEPTH_MODE_SLAB64,
        "debug_k_scale_depth_mode must be a supported K-scale depth mode"
    );
    constexpr int DEBUG_QK_CHUNK_COUNT = 192 / 64;
    TORCH_CHECK(
        debug_qk_chunk_mask > 0 &&
            debug_qk_chunk_mask < (1 << DEBUG_QK_CHUNK_COUNT),
        "debug_qk_chunk_mask must select at least one valid QK scale chunk"
    );
    TORCH_CHECK(
        debug_stage_cut == DEBUG_STAGE_CUT_FULL ||
            debug_stage_cut == DEBUG_STAGE_CUT_DUAL_SLOT_FULL ||
            debug_stage_cut == DEBUG_STAGE_CUT_LEGACY_SINGLE_SLOT_FULL,
        "tile-window live BF16 ref debug only supports full/dual-slot/legacy-single-slot stage cuts"
    );

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto P_fp4 = Q.new_empty({batch, heads, seqlen, tiles_to_run * (128 / 2)}, Q.options().dtype(at::ScalarType::Float4_e2m1fn_x2));
    auto P_sc_prepared = Q_sc.new_zeros({batch, heads, seqlen / 128, tiles_to_run * 2, 512}, Q_sc.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto P_bf16_live = Q_sg.new_empty({batch, heads, seqlen, tiles_to_run * 128}, Q_sg.options().dtype(at::ScalarType::BFloat16));
    auto Scores_bf16_live = Q_sg.new_empty({batch, heads, seqlen, tiles_to_run * 128}, Q_sg.options().dtype(at::ScalarType::BFloat16));
    auto Scores_bf16_scan_repacked = Q_sg.new_empty({batch, heads, seqlen, tiles_to_run * 128}, Q_sg.options().dtype(at::ScalarType::BFloat16));
    auto Scores_bf16_tmem_x16 = Q_sg.new_empty({batch, heads, seqlen, tiles_to_run * 128}, Q_sg.options().dtype(at::ScalarType::BFloat16));
    auto LSE = Q_sg.new_empty({batch, heads, 1, seqlen}, Q_sg.options().dtype(at::ScalarType::Float));
    auto debug_opts = Q_sg.options().dtype(at::ScalarType::Float);
    auto debug_row_max = Q_sg.new_empty({batch, heads, seqlen, tiles_to_run}, debug_opts);
    auto debug_row_sum = Q_sg.new_empty({batch, heads, seqlen, tiles_to_run}, debug_opts);
    auto debug_tile_amax = Q_sg.new_empty({batch, heads, seqlen / 128, tiles_to_run}, debug_opts);
    auto debug_tile_score_max = Q_sg.new_empty({batch, heads, seqlen, tiles_to_run}, debug_opts);
    auto debug_tile_exp_sum = Q_sg.new_empty({batch, heads, seqlen, tiles_to_run}, debug_opts);

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_quant_p_from_scores_debug<C>;
    auto debug_q_sc_loaded = Q_sc.new_zeros({batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, 512}, Q_sc.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto debug_k_sc_loaded = Q_sc.new_empty({batch, tiles_to_run, heads * C::QK_SCALE_CHUNKS, 512}, Q_sc.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto debug_q_sc_tmem_loaded = Q_sc.new_empty({0}, Q_sc.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto debug_k_sc_tmem_loaded = Q_sc.new_empty({0}, Q_sc.options().dtype(at::ScalarType::Float8_e4m3fn));
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        reinterpret_cast<uint8_t *>(P_fp4.data_ptr()),
        reinterpret_cast<uint8_t *>(P_sc_prepared.data_ptr()),
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(P_bf16_live.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        reinterpret_cast<float *>(debug_row_max.data_ptr()),
        reinterpret_cast<float *>(debug_row_sum.data_ptr()),
        reinterpret_cast<float *>(debug_tile_amax.data_ptr()),
        -1,
        tile_begin,
        tiles_to_run,
    };
    g.debug_q_scale_mode = debug_q_scale_mode;
    g.debug_stage_cut = debug_stage_cut;
    g.debug_tmem_layout_mode = debug_tmem_layout_mode;
    g.debug_k_scale_depth_mode = debug_k_scale_depth_mode;
    g.debug_qk_chunk_mask = debug_qk_chunk_mask;
    g.debug_tile_score_max = reinterpret_cast<float *>(debug_tile_score_max.data_ptr());
    g.debug_tile_exp_sum = reinterpret_cast<float *>(debug_tile_exp_sum.data_ptr());
    g.debug_scores_bf16_live = reinterpret_cast<bf16 *>(Scores_bf16_live.data_ptr());
    g.debug_scores_bf16_scan_repacked = reinterpret_cast<bf16 *>(Scores_bf16_scan_repacked.data_ptr());
    g.debug_scores_bf16_tmem_x16 = reinterpret_cast<bf16 *>(Scores_bf16_tmem_x16.data_ptr());
    g.debug_q_sc_loaded = reinterpret_cast<uint8_t *>(debug_q_sc_loaded.data_ptr());
    g.debug_k_sc_loaded = reinterpret_cast<uint8_t *>(debug_k_sc_loaded.data_ptr());
    g.debug_q_sc_tmem_loaded = nullptr;
    g.debug_k_sc_tmem_loaded = nullptr;

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true>, g));

    return pybind11::make_tuple(P_fp4, P_sc_prepared, P_bf16_live, Scores_bf16_live, Scores_bf16_scan_repacked, Scores_bf16_tmem_x16, LSE, debug_row_max, debug_row_sum, debug_tile_amax, debug_tile_score_max, debug_tile_exp_sum, debug_q_sc_loaded, debug_k_sc_loaded, debug_q_sc_tmem_loaded, debug_k_sc_tmem_loaded);
}

pybind11::tuple dispatch_quantize_p_from_scores_tile_window_live_bf16_only_debug(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    int tile_begin,
    int tiles_to_run
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(tile_begin >= 0, "tile_begin must be non-negative");
    TORCH_CHECK(tiles_to_run > 0, "tiles_to_run must be positive");
    TORCH_CHECK(tile_begin + tiles_to_run <= seqlen / 128, "tile window must be within sequence tile range");

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto P_bf16_live = Q_sg.new_empty({batch, heads, seqlen, tiles_to_run * 128}, Q_sg.options().dtype(at::ScalarType::BFloat16));
    auto LSE = Q_sg.new_empty({batch, heads, 1, seqlen}, Q_sg.options().dtype(at::ScalarType::Float));
    auto debug_opts = Q_sg.options().dtype(at::ScalarType::Float);
    auto debug_row_max = Q_sg.new_empty({batch, heads, seqlen, tiles_to_run}, debug_opts);
    auto debug_row_sum = Q_sg.new_empty({batch, heads, seqlen, tiles_to_run}, debug_opts);
    auto debug_tile_amax = Q_sg.new_empty({batch, heads, seqlen / 128, tiles_to_run}, debug_opts);

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(P_bf16_live.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        reinterpret_cast<float *>(debug_row_max.data_ptr()),
        reinterpret_cast<float *>(debug_row_sum.data_ptr()),
        reinterpret_cast<float *>(debug_tile_amax.data_ptr()),
        -1,
        tile_begin,
        tiles_to_run,
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true, true>, g));

    return pybind11::make_tuple(P_bf16_live, LSE, debug_row_max, debug_row_sum, debug_tile_amax);
}

pybind11::tuple dispatch_quantize_p_from_scores_tile_window_live_bf16_only_minimal_debug(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    int tile_begin,
    int tiles_to_run
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");
    TORCH_CHECK(tile_begin >= 0, "tile_begin must be non-negative");
    TORCH_CHECK(tiles_to_run > 0, "tiles_to_run must be positive");
    TORCH_CHECK(tile_begin + tiles_to_run <= seqlen / 128, "tile window must be within sequence tile range");

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto P_bf16_live = Q_sg.new_empty({batch, heads, seqlen, tiles_to_run * 128}, Q_sg.options().dtype(at::ScalarType::BFloat16));
    auto LSE = Q_sg.new_empty({batch, heads, 1, seqlen}, Q_sg.options().dtype(at::ScalarType::Float));

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(P_bf16_live.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        -1,
        tile_begin,
        tiles_to_run,
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid_exact(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true, true>, g));

    return pybind11::make_tuple(P_bf16_live, LSE);
}

pybind11::tuple dispatch_probe_loaded_qk_scale_chunks_debug(
    at::Tensor Q_sc_tile,
    at::Tensor K_sc_tile,
    int chunk_idx,
    int stage_mode
) {
    CHECK_INPUT(Q_sc_tile);
    CHECK_INPUT(K_sc_tile);

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    TORCH_CHECK(Q_sc_tile.dtype() == at::ScalarType::Float8_e4m3fn, "Q_sc_tile must be float8_e4m3fn");
    TORCH_CHECK(K_sc_tile.dtype() == at::ScalarType::Float8_e4m3fn, "K_sc_tile must be float8_e4m3fn");
    TORCH_CHECK(Q_sc_tile.dim() == 2, "Q_sc_tile must have shape (chunks, 512)");
    TORCH_CHECK(K_sc_tile.dim() == 2, "K_sc_tile must have shape (chunks, 512)");
    TORCH_CHECK(Q_sc_tile.size(0) == C::QK_SCALE_CHUNKS && Q_sc_tile.size(1) == 512, "Q_sc_tile must have shape (3, 512)");
    TORCH_CHECK(K_sc_tile.size(0) == C::QK_SCALE_CHUNKS && K_sc_tile.size(1) == 512, "K_sc_tile must have shape (3, 512)");
    TORCH_CHECK(Q_sc_tile.is_contiguous(), "Q_sc_tile must be contiguous");
    TORCH_CHECK(K_sc_tile.is_contiguous(), "K_sc_tile must be contiguous");
    TORCH_CHECK(Q_sc_tile.device() == K_sc_tile.device(), "Q_sc_tile and K_sc_tile must be on the same device");
    TORCH_CHECK(chunk_idx >= 0 && chunk_idx < C::QK_SCALE_CHUNKS, "chunk_idx must be within the valid QK chunk range");
    TORCH_CHECK(stage_mode >= DEBUG_SCALE_PROBE_STAGE_SOURCE_ONLY && stage_mode <= DEBUG_SCALE_PROBE_STAGE_FULL_DIRECT_DUMP, "stage_mode must be a valid debug_scale_probe_stage_mode value");

    auto q_sc_source_chunk = Q_sc_tile.new_empty({32, 16}, Q_sc_tile.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto k_sc_source_chunk = K_sc_tile.new_empty({32, 16}, K_sc_tile.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto q_sc_tmem_chunk = Q_sc_tile.new_empty({128, 16}, Q_sc_tile.options().dtype(at::ScalarType::BFloat16));
    auto k_sc_tmem_chunk = K_sc_tile.new_empty({128, 16}, K_sc_tile.options().dtype(at::ScalarType::BFloat16));
    auto q_sc_tmem_raw_words = Q_sc_tile.new_zeros({4, 32, 2, 4}, Q_sc_tile.options().dtype(at::ScalarType::Int));
    auto k_sc_tmem_raw_words = K_sc_tile.new_zeros({4, 32, 2, 4}, K_sc_tile.options().dtype(at::ScalarType::Int));

    globals_loaded_qk_scale_chunk_debug<C> g{
        reinterpret_cast<const uint8_t *>(Q_sc_tile.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc_tile.data_ptr()),
        reinterpret_cast<uint8_t *>(q_sc_source_chunk.data_ptr()),
        reinterpret_cast<uint8_t *>(k_sc_source_chunk.data_ptr()),
        reinterpret_cast<bf16 *>(q_sc_tmem_chunk.data_ptr()),
        reinterpret_cast<bf16 *>(k_sc_tmem_chunk.data_ptr()),
        reinterpret_cast<uint32_t *>(q_sc_tmem_raw_words.data_ptr()),
        reinterpret_cast<uint32_t *>(k_sc_tmem_raw_words.data_ptr()),
        chunk_idx,
        stage_mode,
    };

    CUDACHECK(cudaFuncSetAttribute(kernel_probe_loaded_qk_scale_chunks_debug<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_probe_loaded_qk_scale_chunks_debug<C>, g));

    return pybind11::make_tuple(q_sc_source_chunk, k_sc_source_chunk, q_sc_tmem_chunk, k_sc_tmem_chunk, q_sc_tmem_raw_words, k_sc_tmem_raw_words);
}

pybind11::tuple dispatch_quantize_p_from_scores_row_tile_window_live_bf16_only_minimal_debug(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    int row_tile,
    int max_k_iters,
    int tile_begin,
    int tiles_to_run,
    bool store_bf16,
    bool store_lse,
    int debug_stage_cut
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q and K must be rank-4 tensors in (batch, heads, seqlen, packed_dim) layout");

    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);

    TORCH_CHECK(K.size(0) == batch, "batch dimensions must match");
    TORCH_CHECK(K.size(1) == heads, "head dimensions must match");
    TORCH_CHECK(K.size(2) == seqlen, "sequence lengths must match");
    TORCH_CHECK(Q.size(3) == 192 / 2 && K.size(3) == 192 / 2, "Q and K packed head_dim must be 96");
    TORCH_CHECK(seqlen % 128 == 0, "sequence length must be divisible by 128");

    using C = config_fp4pv<128, 128, 192, 128, 2>;
    const int row_tiles = seqlen / C::Mb;
    TORCH_CHECK(row_tile >= 0 && row_tile < row_tiles, "row_tile must be within sequence row-tile range");
    TORCH_CHECK(tile_begin >= 0, "tile_begin must be non-negative");
    TORCH_CHECK(tiles_to_run > 0, "tiles_to_run must be positive");
    const int row_cluster_base = row_tile - (row_tile % C::CLUSTER_SIZE);
    const int full_row_tile_iters = min(row_cluster_base + C::CLUSTER_SIZE, row_tiles);
    TORCH_CHECK(max_k_iters == -1 || max_k_iters > 0, "max_k_iters must be -1 or positive");
    TORCH_CHECK(max_k_iters == -1 || max_k_iters <= full_row_tile_iters, "max_k_iters must not exceed the selected row tile causal iteration range");
    const int executed_row_tile_iters = (max_k_iters < 0) ? full_row_tile_iters : max_k_iters;
    TORCH_CHECK(tile_begin + tiles_to_run <= executed_row_tile_iters, "tile window must fit within the executed row tile causal iteration range");
    TORCH_CHECK(
        debug_stage_cut >= DEBUG_STAGE_CUT_FULL && debug_stage_cut <= DEBUG_STAGE_CUT_LEGACY_SINGLE_SLOT_FULL,
        "debug_stage_cut must be within the supported milestone range"
    );
    TORCH_CHECK(
        debug_stage_cut == DEBUG_STAGE_CUT_FULL ||
            debug_stage_cut == DEBUG_STAGE_CUT_DUAL_SLOT_FULL ||
            debug_stage_cut == DEBUG_STAGE_CUT_LEGACY_SINGLE_SLOT_FULL ||
            max_k_iters == 3,
        "debug_stage_cut requires max_k_iters == 3"
    );
    TORCH_CHECK(
        debug_stage_cut == DEBUG_STAGE_CUT_FULL ||
            debug_stage_cut == DEBUG_STAGE_CUT_DUAL_SLOT_FULL ||
            debug_stage_cut == DEBUG_STAGE_CUT_LEGACY_SINGLE_SLOT_FULL ||
            (!store_bf16 && !store_lse),
        "debug_stage_cut milestone probes require store_bf16=False and store_lse=False unless using a full-mode path"
    );

    constexpr int FP4PV_KSC_DEPTH_DIVISOR = 64;
    const int k_sc_depth = seqlen / FP4PV_KSC_DEPTH_DIVISOR;

    auto P_bf16_live = Q_sg.new_empty({batch, heads, C::Mb, tiles_to_run * C::Nb}, Q_sg.options().dtype(at::ScalarType::BFloat16));
    auto LSE = Q_sg.new_empty({batch, heads, 1, C::Mb}, Q_sg.options().dtype(at::ScalarType::Float));

    using G = globals_quant_p_from_scores_debug<C>;
    G g{
        kittens::py::tensor_to_gl<typename G::q_gl>(Q),
        tensor_to_fp4_scale_gl<typename G::q_sc_gl>(Q_sc, batch, seqlen / C::Mb, heads * C::QK_SCALE_CHUNKS, "Q_sc"),
        tensor_to_bh_scalar_gl<typename G::q_sg_gl>(Q_sg, batch, heads, "Q_sg"),
        kittens::py::tensor_to_gl<typename G::k_gl>(K),
        tensor_to_fp4_scale_gl<typename G::k_sc_gl>(K_sc, batch, k_sc_depth, heads * C::QK_SCALE_CHUNKS, "K_sc"),
        tensor_to_bh_scalar_gl<typename G::k_sg_gl>(K_sg, batch, heads, "K_sg"),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        reinterpret_cast<bf16 *>(P_bf16_live.data_ptr()),
        kittens::py::tensor_to_gl<typename G::lse_gl>(LSE),
        nullptr,
        nullptr,
        nullptr,
        row_tile,
        tile_begin,
        tiles_to_run,
    };
    g.max_k_iters = max_k_iters;
    if (!store_bf16) {
        g.p_bf16_live = nullptr;
    }
    g.disable_lse_store = store_lse ? 0 : 1;
    g.debug_stage_cut = debug_stage_cut;

    CUDACHECK(cudaFuncSetAttribute(kernel_quantize_p_from_scores_debug<C, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory()));
    LaunchConfig<true, false> launch_config(g.grid_exact(), g.block(), g.dynamic_shared_memory(), at::cuda::getCurrentCUDAStream(), C::CLUSTER_SIZE);
    CUDACHECK(cudaLaunchKernelEx(launch_config, kernel_quantize_p_from_scores_debug<C, true, true>, g));

    return pybind11::make_tuple(P_bf16_live, LSE);
}

#if 1
namespace mxfp4_full_direct {

static constexpr int DQK = 192;
static constexpr int DVO = 128;
static constexpr int QK_PACKED = DQK / 2;

__device__ inline float fp4_e2m1_to_float(uint8_t nibble) {
    switch (nibble & 0x0F) {
        case 0x0: return 0.0f;
        case 0x1: return 0.5f;
        case 0x2: return 1.0f;
        case 0x3: return 1.5f;
        case 0x4: return 2.0f;
        case 0x5: return 3.0f;
        case 0x6: return 4.0f;
        case 0x7: return 6.0f;
        case 0x8: return -0.0f;
        case 0x9: return -0.5f;
        case 0xA: return -1.0f;
        case 0xB: return -1.5f;
        case 0xC: return -2.0f;
        case 0xD: return -3.0f;
        case 0xE: return -4.0f;
        default: return -6.0f;
    }
}

__device__ inline float fp8_e4m3_to_float(uint8_t byte) {
    const int sign = (byte & 0x80) ? -1 : 1;
    const int exp = (byte >> 3) & 0x0F;
    const int mant = byte & 0x07;
    if (exp == 0) {
        return sign * ldexpf(static_cast<float>(mant), -9);
    }
    return sign * ldexpf(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
}

__device__ inline float bf16_to_float(const at::BFloat16 *ptr, int64_t idx) {
    const auto raw = reinterpret_cast<const __nv_bfloat16 *>(ptr);
    return __bfloat162float(raw[idx]);
}

__device__ inline int prepared_scale_offset(int row_in_128, int group_in_64) {
    return (((row_in_128 & 31) * 4 + (row_in_128 >> 5)) * 4) + group_in_64;
}

__device__ inline uint8_t fp4_nibble(const uint8_t *packed, int64_t packed_idx, int dim) {
    const uint8_t byte = packed[packed_idx];
    return (dim & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
}

__device__ inline float load_q_value(
    const uint8_t *q,
    const uint8_t *q_sc,
    const float *q_sg,
    int b,
    int h,
    int row,
    int dim,
    int heads,
    int seqlen
) {
    const int chunk = dim >> 6;
    const int group = (dim & 63) >> 4;
    const int64_t q_idx = (((static_cast<int64_t>(b) * heads + h) * seqlen + row) * QK_PACKED) + (dim >> 1);
    const int q_depth = seqlen / 128;
    const int sc_off = prepared_scale_offset(row & 127, group);
    const int64_t sc_idx =
        (((static_cast<int64_t>(b) * q_depth + (row / 128)) * (heads * 3) + h * 3 + chunk) * 512) + sc_off;
    return fp4_e2m1_to_float(fp4_nibble(q, q_idx, dim)) * fp8_e4m3_to_float(q_sc[sc_idx]) * q_sg[b * heads + h];
}

__device__ inline float load_k_value(
    const uint8_t *k,
    const uint8_t *k_sc,
    const float *k_sg,
    int b,
    int h,
    int row,
    int dim,
    int heads,
    int seqlen
) {
    const int chunk = dim >> 6;
    const int group = (dim & 63) >> 4;
    const int64_t k_idx = (((static_cast<int64_t>(b) * heads + h) * seqlen + row) * QK_PACKED) + (dim >> 1);
    const int k_depth = seqlen / 64;
    const int sc_off = prepared_scale_offset(row & 127, group);
    const int64_t sc_idx =
        (((static_cast<int64_t>(b) * k_depth + (row / 64)) * (heads * 3) + h * 3 + chunk) * 512) + sc_off;
    return fp4_e2m1_to_float(fp4_nibble(k, k_idx, dim)) * fp8_e4m3_to_float(k_sc[sc_idx]) * k_sg[b * heads + h];
}

__device__ inline float load_v_value(
    const uint8_t *v,
    const uint8_t *v_sc,
    int b,
    int h,
    int row,
    int dim,
    int heads,
    int seqlen
) {
    const int packed_seqlen = seqlen / 2;
    const int64_t v_idx = (((static_cast<int64_t>(b) * heads + h) * DVO + dim) * packed_seqlen) + (row >> 1);
    const int v_depth = seqlen / 64;
    const int group = (row & 63) >> 4;
    const int sc_off = prepared_scale_offset(dim & 127, group);
    const int64_t sc_idx = (((static_cast<int64_t>(b) * heads + h) * v_depth + (row / 64)) * 512) + sc_off;
    return fp4_e2m1_to_float(fp4_nibble(v, v_idx, row)) * fp8_e4m3_to_float(v_sc[sc_idx]);
}

__global__ void delta_kernel(
    const at::BFloat16 *__restrict__ out,
    const at::BFloat16 *__restrict__ dout,
    float *__restrict__ delta,
    int batch,
    int heads,
    int seqlen
) {
    const int64_t row_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(batch) * heads * seqlen;
    if (row_id >= total) {
        return;
    }
    const int s = row_id % seqlen;
    const int h = (row_id / seqlen) % heads;
    const int b = row_id / (seqlen * heads);
    float acc = 0.0f;
    #pragma unroll
    for (int d = 0; d < DVO; ++d) {
        const int64_t off = (((static_cast<int64_t>(b) * seqlen + s) * heads + h) * DVO) + d;
        acc += bf16_to_float(out, off) * bf16_to_float(dout, off);
    }
    delta[row_id] = acc;
}

__global__ void pair_kernel(
    const uint8_t *__restrict__ q,
    const uint8_t *__restrict__ q_sc,
    const float *__restrict__ q_sg,
    const uint8_t *__restrict__ k,
    const uint8_t *__restrict__ k_sc,
    const float *__restrict__ k_sg,
    const uint8_t *__restrict__ v,
    const uint8_t *__restrict__ v_sc,
    const at::BFloat16 *__restrict__ dout,
    const float *__restrict__ lse,
    const float *__restrict__ delta,
    float *__restrict__ dq,
    float *__restrict__ dk,
    float *__restrict__ dv,
    int batch,
    int heads,
    int seqlen,
    float softmax_scale
) {
    const int64_t pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t pairs_per_head = static_cast<int64_t>(seqlen) * seqlen;
    const int64_t total = static_cast<int64_t>(batch) * heads * pairs_per_head;
    if (pair_id >= total) {
        return;
    }
    const int j = pair_id % seqlen;
    const int i = (pair_id / seqlen) % seqlen;
    if (j > i) {
        return;
    }
    const int h = (pair_id / pairs_per_head) % heads;
    const int b = pair_id / (pairs_per_head * heads);

    float qk = 0.0f;
    #pragma unroll
    for (int d = 0; d < DQK; ++d) {
        qk += load_q_value(q, q_sc, q_sg, b, h, i, d, heads, seqlen) *
              load_k_value(k, k_sc, k_sg, b, h, j, d, heads, seqlen);
    }
    const int64_t row_id = (static_cast<int64_t>(b) * heads + h) * seqlen + i;
    const float p = expf(qk * softmax_scale - lse[row_id]);

    float dp = 0.0f;
    #pragma unroll
    for (int d = 0; d < DVO; ++d) {
        const int64_t do_off = (((static_cast<int64_t>(b) * seqlen + i) * heads + h) * DVO) + d;
        dp += bf16_to_float(dout, do_off) * load_v_value(v, v_sc, b, h, j, d, heads, seqlen);
    }
    const float ds = p * (dp - delta[row_id]) * softmax_scale;

    #pragma unroll
    for (int d = 0; d < DQK; ++d) {
        const float q_val = load_q_value(q, q_sc, q_sg, b, h, i, d, heads, seqlen);
        const float k_val = load_k_value(k, k_sc, k_sg, b, h, j, d, heads, seqlen);
        const int64_t dq_off = (((static_cast<int64_t>(b) * seqlen + i) * heads + h) * DQK) + d;
        const int64_t dk_off = (((static_cast<int64_t>(b) * seqlen + j) * heads + h) * DQK) + d;
        atomicAdd(dq + dq_off, ds * k_val);
        atomicAdd(dk + dk_off, ds * q_val);
    }
    #pragma unroll
    for (int d = 0; d < DVO; ++d) {
        const int64_t do_off = (((static_cast<int64_t>(b) * seqlen + i) * heads + h) * DVO) + d;
        const int64_t dv_off = (((static_cast<int64_t>(b) * seqlen + j) * heads + h) * DVO) + d;
        atomicAdd(dv + dv_off, p * bf16_to_float(dout, do_off));
    }
}

} // namespace mxfp4_full_direct

namespace mxfp4_full_tile_staged {

struct config {
    static constexpr int Mb = 128;
    static constexpr int Nb = 128;
    static constexpr int Dqk = 192;
    static constexpr int Dvo = 128;
    static constexpr int ClusterSize = 1;
    static constexpr int WarpTiles = 4;
    static constexpr int BlockThreads = WarpTiles * WARP_THREADS;
    static constexpr int MinBlocksPerSm = 1;
};

using dqacc_chunk_tile = st_fl<tkfa4::kRefTileM, 64>;
using dqacc_tile = st_fl<tkfa4::kRefTileM, config::Dqk>;
using dqacc_chunk_gl = gl<float, -1, -1, -1, -1, dqacc_chunk_tile>;
using dqacc_gl = gl<float, -1, -1, -1, -1, dqacc_tile>;
using dk_gl = gl<float, -1, -1, -1, config::Dqk>;
using dv_gl = gl<float, -1, -1, -1, config::Dvo>;

template <typename ST>
__device__ inline void fill_q_tile(
    ST &dst,
    const uint8_t *__restrict__ q,
    const uint8_t *__restrict__ q_sc,
    const float *__restrict__ q_sg,
    int b,
    int h,
    int q_tile_idx,
    int heads,
    int seqlen
) {
    constexpr int rows = tkfa4::kRefTileM;
    constexpr int cols = config::Dqk;
    for (int idx = threadIdx.x; idx < rows * cols; idx += blockDim.x) {
        const int r = idx / cols;
        const int d = idx - r * cols;
        const float value = mxfp4_full_direct::load_q_value(
            q,
            q_sc,
            q_sg,
            b,
            h,
            q_tile_idx * rows + r,
            d,
            heads,
            seqlen
        );
        dst[{r, d}] = __float2bfloat16(value);
    }
}

template <typename ST>
__device__ inline void fill_k_tile(
    ST &dst,
    const uint8_t *__restrict__ k,
    const uint8_t *__restrict__ k_sc,
    const float *__restrict__ k_sg,
    int b,
    int h,
    int k_tile_idx,
    int heads,
    int seqlen
) {
    constexpr int rows = tkfa4::kRefTileM;
    constexpr int cols = config::Dqk;
    for (int idx = threadIdx.x; idx < rows * cols; idx += blockDim.x) {
        const int r = idx / cols;
        const int d = idx - r * cols;
        const float value = mxfp4_full_direct::load_k_value(
            k,
            k_sc,
            k_sg,
            b,
            h,
            k_tile_idx * rows + r,
            d,
            heads,
            seqlen
        );
        dst[{r, d}] = __float2bfloat16(value);
    }
}

template <typename ST>
__device__ inline void fill_v_tile(
    ST &dst,
    const uint8_t *__restrict__ v,
    const uint8_t *__restrict__ v_sc,
    int b,
    int h,
    int v_tile_idx,
    int heads,
    int seqlen
) {
    constexpr int rows = tkfa4::kRefTileM;
    constexpr int cols = config::Dvo;
    for (int idx = threadIdx.x; idx < rows * cols; idx += blockDim.x) {
        const int r = idx / cols;
        const int d = idx - r * cols;
        const float value = mxfp4_full_direct::load_v_value(
            v,
            v_sc,
            b,
            h,
            v_tile_idx * rows + r,
            d,
            heads,
            seqlen
        );
        dst[{r, d}] = __float2bfloat16(value);
    }
}

template <typename ST>
__device__ inline void fill_dout_tile(
    ST &dst,
    const at::BFloat16 *__restrict__ dout,
    int b,
    int h,
    int q_tile_idx,
    int heads,
    int seqlen
) {
    constexpr int rows = tkfa4::kRefTileM;
    constexpr int cols = config::Dvo;
    const auto raw = reinterpret_cast<const __nv_bfloat16 *>(dout);
    for (int idx = threadIdx.x; idx < rows * cols; idx += blockDim.x) {
        const int r = idx / cols;
        const int d = idx - r * cols;
        const int row = q_tile_idx * rows + r;
        const int64_t off = (((static_cast<int64_t>(b) * seqlen + row) * heads + h) * cols) + d;
        dst[{r, d}] = raw[off];
    }
}

template <typename SV>
__device__ inline void fill_stats_vec(
    SV &lse_dst,
    SV &dpsum_dst,
    const at::BFloat16 *__restrict__ out,
    const at::BFloat16 *__restrict__ dout,
    const float *__restrict__ lse,
    int b,
    int h,
    int q_tile_idx,
    int heads,
    int seqlen
) {
    for (int r = threadIdx.x; r < tkfa4::kRefTileM; r += blockDim.x) {
        const int row = q_tile_idx * tkfa4::kRefTileM + r;
        const int64_t row_id = (static_cast<int64_t>(b) * heads + h) * seqlen + row;
        float delta = 0.0f;
        #pragma unroll
        for (int d = 0; d < config::Dvo; ++d) {
            const int64_t off = (((static_cast<int64_t>(b) * seqlen + row) * heads + h) * config::Dvo) + d;
            delta += mxfp4_full_direct::bf16_to_float(out, off) * mxfp4_full_direct::bf16_to_float(dout, off);
        }
        lse_dst[r] = lse[row_id] * tkfa4::kLog2E;
        dpsum_dst[r] = delta;
    }
}

template <typename ST>
__device__ inline void store_float_tiles(
    const ST *src,
    float *__restrict__ dst,
    int b,
    int h,
    int tile_base,
    int heads,
    int seqlen,
    int cols
) {
    constexpr int rows = tkfa4::kRefTileM;
    const int total = config::WarpTiles * rows * cols;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        const int warp_tile = idx / (rows * cols);
        const int rem = idx - warp_tile * rows * cols;
        const int r = rem / cols;
        const int d = rem - r * cols;
        const int row = (tile_base + warp_tile) * rows + r;
        const float value = src[warp_tile][{r, d}];
        const int64_t off = (((static_cast<int64_t>(b) * seqlen + row) * heads + h) * cols) + d;
        dst[off] = value;
    }
}

template <int ChunkIdx, typename FullTile, typename ChunkTile>
__device__ inline void extract_chunk(
    const FullTile &src,
    ChunkTile &dst
) {
    static_assert(FullTile::height == ChunkTile::height);
    static_assert(FullTile::width == ChunkTile::width * 3);
    #pragma unroll
    for (int i = 0; i < ChunkTile::height; ++i) {
        #pragma unroll
        for (int j = 0; j < ChunkTile::width; ++j) {
            dst.tiles[i][j] = src.tiles[i][j + ChunkIdx * ChunkTile::width];
        }
    }
}

template <typename FullTile, typename ChunkTile>
__device__ inline void stitch_three_chunks(
    FullTile &dst,
    const ChunkTile &chunk0,
    const ChunkTile &chunk1,
    const ChunkTile &chunk2
) {
    static_assert(FullTile::height == ChunkTile::height);
    static_assert(FullTile::width == ChunkTile::width * 3);
    #pragma unroll
    for (int i = 0; i < ChunkTile::height; ++i) {
        #pragma unroll
        for (int j = 0; j < ChunkTile::width; ++j) {
            dst.tiles[i][j] = chunk0.tiles[i][j];
            dst.tiles[i][j + ChunkTile::width] = chunk1.tiles[i][j];
            dst.tiles[i][j + 2 * ChunkTile::width] = chunk2.tiles[i][j];
        }
    }
}

template <typename C, bool CAUSAL, bool DENSE_UNMASKED>
__device__ inline void backward_tile_step_compact_fully_chunked(
    rt_fl<tkfa4::kRefTileM, 64> &dq0,
    rt_fl<tkfa4::kRefTileM, 64> &dq1,
    rt_fl<tkfa4::kRefTileM, 64> &dq2,
    rt_fl<tkfa4::kRefTileM, 64> &dk0_accum,
    rt_fl<tkfa4::kRefTileM, 64> &dk1_accum,
    rt_fl<tkfa4::kRefTileM, 64> &dk2_accum,
    rt_fl<tkfa4::kRefTileM, C::Dvo> &dv_accum,
    const rt_bf<tkfa4::kRefTileM, C::Dqk> &q_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dqk> &k_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dvo> &v_reg,
    const rt_bf<tkfa4::kRefTileM, C::Dvo> &do_reg,
    const typename rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN>::col_vec &lse_log2_vec,
    const typename rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN>::col_vec &dpsum_vec,
    float scale,
    float scale_log2e,
    int q_tile_idx,
    int kv_subtile_idx,
    int actual_seq_len
) {
    rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN> p, ds;

    if constexpr (DENSE_UNMASKED) {
        tkfa4::bwd_fa4::detail::reconstruct_probability_tile_dense<C>(
            p,
            q_reg,
            k_reg,
            lse_log2_vec,
            scale_log2e
        );
    } else {
        tkfa4::bwd_fa4::detail::reconstruct_probability_tile<C>(
            p,
            q_reg,
            k_reg,
            lse_log2_vec,
            scale_log2e,
            q_tile_idx,
            kv_subtile_idx,
            actual_seq_len,
            CAUSAL
        );
    }

    {
        rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN> p_bf;
        rt_bf<tkfa4::kRefTileM, C::Dvo, ducks::rt_layout::col> do_col;
        rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN, ducks::rt_layout::col> p_col;
        warp::copy(p_bf, p);
        warp::swap_layout(p_col, p_bf);
        warp::swap_layout(do_col, do_reg);
        warp::mma_AtB(dv_accum, p_col, do_col, dv_accum);

        warp::zero(ds);
        warp::mma_ABt(ds, do_reg, v_reg, ds);
        warp::sub_row(ds, ds, dpsum_vec);
        warp::mul(ds, p, ds);
        warp::mul(ds, ds, scale);
    }

    rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN> ds_bf;
    warp::copy(ds_bf, ds);

    {
        rt_bf<tkfa4::kRefTileM, tkfa4::kRefTileN, ducks::rt_layout::col> ds_col;
        rt_bf<tkfa4::kRefTileM, 64> q_chunk;
        rt_bf<tkfa4::kRefTileM, 64, ducks::rt_layout::col> q_col;

        warp::swap_layout(ds_col, ds_bf);

        extract_chunk<0>(q_reg, q_chunk);
        warp::swap_layout(q_col, q_chunk);
        warp::mma_AtB(dk0_accum, ds_col, q_col, dk0_accum);

        extract_chunk<1>(q_reg, q_chunk);
        warp::swap_layout(q_col, q_chunk);
        warp::mma_AtB(dk1_accum, ds_col, q_col, dk1_accum);

        extract_chunk<2>(q_reg, q_chunk);
        warp::swap_layout(q_col, q_chunk);
        warp::mma_AtB(dk2_accum, ds_col, q_col, dk2_accum);
    }

    {
        rt_bf<tkfa4::kRefTileM, 64> k_chunk;
        rt_bf<tkfa4::kRefTileM, 64, ducks::rt_layout::col> k_col;

        extract_chunk<0>(k_reg, k_chunk);
        warp::swap_layout(k_col, k_chunk);
        warp::zero(dq0);
        warp::mma_AB(dq0, ds_bf, k_col, dq0);

        extract_chunk<1>(k_reg, k_chunk);
        warp::swap_layout(k_col, k_chunk);
        warp::zero(dq1);
        warp::mma_AB(dq1, ds_bf, k_col, dq1);

        extract_chunk<2>(k_reg, k_chunk);
        warp::swap_layout(k_col, k_chunk);
        warp::zero(dq2);
        warp::mma_AB(dq2, ds_bf, k_col, dq2);
    }
}

__global__ __launch_bounds__(config::BlockThreads, config::MinBlocksPerSm)
void kernel(
    const uint8_t *__restrict__ q,
    const uint8_t *__restrict__ q_sc,
    const float *__restrict__ q_sg,
    const uint8_t *__restrict__ k,
    const uint8_t *__restrict__ k_sc,
    const float *__restrict__ k_sg,
    const uint8_t *__restrict__ v,
    const uint8_t *__restrict__ v_sc,
    const at::BFloat16 *__restrict__ out,
    const at::BFloat16 *__restrict__ dout,
    const float *__restrict__ lse,
    const __grid_constant__ dqacc_chunk_gl dq_accum,
    const __grid_constant__ dk_gl dk,
    const __grid_constant__ dv_gl dv,
    int heads,
    int seqlen,
    float softmax_scale
) {
    using C = config;
    constexpr int q_tiles_buffered = 4;
    using qk_bf_tile = st_bf<tkfa4::kRefTileM, C::Dqk>;
    using v_bf_tile = st_bf<tkfa4::kRefTileM, C::Dvo>;
    using stats_smem_tile = col_vec<st_fl<tkfa4::kRefTileM, C::Dvo>>;
    using stats_vec = typename rt_fl<tkfa4::kRefTileM, tkfa4::kRefTileN>::col_vec;

    __shared__ alignas(1024) qk_bf_tile q_smem[q_tiles_buffered];
    __shared__ alignas(1024) qk_bf_tile k_smem[C::WarpTiles];
    __shared__ alignas(1024) v_bf_tile v_smem[C::WarpTiles];
    __shared__ alignas(1024) v_bf_tile do_smem[q_tiles_buffered];
    __shared__ alignas(1024) dqacc_chunk_tile dq_smem[3][C::WarpTiles];
    __shared__ alignas(64) stats_smem_tile lse_log2_smem[q_tiles_buffered];
    __shared__ alignas(64) stats_smem_tile dpsum_smem[q_tiles_buffered];

    const int warp = threadIdx.x >> 5;
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int kv_block_idx = static_cast<int>(blockIdx.x);
    const int kv_tile_base = kv_block_idx * C::WarpTiles;
    const int kv_subtile_idx = kv_tile_base + warp;
    const int num_k_blocks = static_cast<int>(gridDim.x);
    const int num_q_blocks = seqlen / (tkfa4::kRefTileM * q_tiles_buffered);
    const int q_block_start = kv_tile_base / q_tiles_buffered;

    for (int tile = 0; tile < C::WarpTiles; ++tile) {
        fill_k_tile(k_smem[tile], k, k_sc, k_sg, batch_idx, head_idx, kv_tile_base + tile, heads, seqlen);
        fill_v_tile(v_smem[tile], v, v_sc, batch_idx, head_idx, kv_tile_base + tile, heads, seqlen);
    }
    __syncthreads();

    rt_bf<tkfa4::kRefTileM, C::Dqk> k_reg;
    rt_bf<tkfa4::kRefTileM, C::Dvo> v_reg;
    rt_fl<tkfa4::kRefTileM, 64> dk0_accum, dk1_accum, dk2_accum;
    rt_fl<tkfa4::kRefTileM, C::Dvo> dv_accum;
    warp::load(k_reg, k_smem[warp]);
    warp::load(v_reg, v_smem[warp]);
    warp::zero(dk0_accum);
    warp::zero(dk1_accum);
    warp::zero(dk2_accum);
    warp::zero(dv_accum);

    for (int q_block_idx = q_block_start; q_block_idx < num_q_blocks; ++q_block_idx) {
        const int q_tile_base = q_block_idx * q_tiles_buffered;
        #pragma unroll
        for (int subtile = 0; subtile < q_tiles_buffered; ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            fill_q_tile(q_smem[subtile], q, q_sc, q_sg, batch_idx, head_idx, q_tile_idx, heads, seqlen);
            fill_dout_tile(do_smem[subtile], dout, batch_idx, head_idx, q_tile_idx, heads, seqlen);
            fill_stats_vec(lse_log2_smem[subtile], dpsum_smem[subtile], out, dout, lse, batch_idx, head_idx, q_tile_idx, heads, seqlen);
        }
        __syncthreads();
        const int first_active_subtile = ((kv_subtile_idx > q_tile_base) ? (kv_subtile_idx - q_tile_base) : 0);
        #pragma unroll 1
        for (int subtile = first_active_subtile;
             subtile < q_tiles_buffered;
             ++subtile) {
            const int q_tile_idx = q_tile_base + subtile;
            rt_bf<tkfa4::kRefTileM, C::Dqk> q_reg;
            rt_bf<tkfa4::kRefTileM, C::Dvo> do_reg;
            rt_fl<tkfa4::kRefTileM, 64> dq0, dq1, dq2;
            stats_vec lse_log2_vec, dpsum_vec;

            warp::load(q_reg, q_smem[subtile]);
            warp::load(do_reg, do_smem[subtile]);
            warp::load(lse_log2_vec, lse_log2_smem[subtile]);
            warp::load(dpsum_vec, dpsum_smem[subtile]);

            if (q_tile_idx > kv_subtile_idx) {
                backward_tile_step_compact_fully_chunked<C, true, true>(
                    dq0,
                    dq1,
                    dq2,
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    softmax_scale,
                    softmax_scale * tkfa4::kLog2E,
                    q_tile_idx,
                    kv_subtile_idx,
                    seqlen
                );
            } else {
                backward_tile_step_compact_fully_chunked<C, true, false>(
                    dq0,
                    dq1,
                    dq2,
                    dk0_accum,
                    dk1_accum,
                    dk2_accum,
                    dv_accum,
                    q_reg,
                    k_reg,
                    v_reg,
                    do_reg,
                    lse_log2_vec,
                    dpsum_vec,
                    softmax_scale,
                    softmax_scale * tkfa4::kLog2E,
                    q_tile_idx,
                    kv_subtile_idx,
                    seqlen
                );
            }

            if (q_block_idx != q_block_start || subtile != first_active_subtile) {
                warp::tma::store_async_read_wait();
            }
            warp::store(dq_smem[0][warp], dq0);
            __syncwarp();
            if (num_k_blocks == 1) {
                warp::tma::store_async(dq_accum, dq_smem[0][warp], {batch_idx, head_idx, q_tile_idx, 0});
            } else {
                warp::tma::store_add_async(dq_accum, dq_smem[0][warp], {batch_idx, head_idx, q_tile_idx, 0});
            }
            warp::store(dq_smem[1][warp], dq1);
            __syncwarp();
            if (num_k_blocks == 1) {
                warp::tma::store_async(dq_accum, dq_smem[1][warp], {batch_idx, head_idx, q_tile_idx, 1});
            } else {
                warp::tma::store_add_async(dq_accum, dq_smem[1][warp], {batch_idx, head_idx, q_tile_idx, 1});
            }
            warp::store(dq_smem[2][warp], dq2);
            __syncwarp();
            if (num_k_blocks == 1) {
                warp::tma::store_async(dq_accum, dq_smem[2][warp], {batch_idx, head_idx, q_tile_idx, 2});
            } else {
                warp::tma::store_add_async(dq_accum, dq_smem[2][warp], {batch_idx, head_idx, q_tile_idx, 2});
            }
            warp::tma::store_commit_group();
        }
        warp::tma::store_async_wait();
    }

    rt_fl<tkfa4::kRefTileM, C::Dqk> dk_accum;
    stitch_three_chunks(dk_accum, dk0_accum, dk1_accum, dk2_accum);
    warp::store<dim::DEPTH>(dk, dk_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
    warp::store<dim::DEPTH>(dv, dv_accum, {batch_idx, kv_subtile_idx, head_idx, 0});
}

__global__ __launch_bounds__(WARP_THREADS * 3, 4)
void reduce_dq_kernel(
    const __grid_constant__ dqacc_chunk_gl dq_accum,
    float *__restrict__ dq,
    int heads,
    int seqlen
) {
    using C = config;
    __shared__ alignas(1024) dqacc_chunk_tile dq_smem[3];

    const int q_tile_idx = static_cast<int>(blockIdx.x);
    const int head_idx = static_cast<int>(blockIdx.y);
    const int batch_idx = static_cast<int>(blockIdx.z);
    const int chunk_idx = threadIdx.y;

    rt_fl<tkfa4::kRefTileM, 64> dq_reg;
    warp::load(dq_reg, dq_accum, {batch_idx, head_idx, q_tile_idx, chunk_idx});
    warp::store(dq_smem[chunk_idx], dq_reg);
    __syncwarp();

    constexpr int rows = tkfa4::kRefTileM;
    constexpr int cols = 64;
    constexpr int total = rows * cols;
    for (int idx = threadIdx.x; idx < total; idx += WARP_THREADS) {
        const int r = idx / cols;
        const int d = idx - r * cols;
        const int row = q_tile_idx * rows + r;
        const int out_d = chunk_idx * cols + d;
        const int64_t off = (((static_cast<int64_t>(batch_idx) * seqlen + row) * heads + head_idx) * C::Dqk) + out_d;
        dq[off] = dq_smem[chunk_idx][{r, d}];
    }
}

} // namespace mxfp4_full_tile_staged
#endif

#include "bwd_key_owned_dkdv_fused_probe.inc"

std::vector<at::Tensor> dispatch_backward_full_mxfp4_direct(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc_prepared,
    at::Tensor O,
    at::Tensor LSE,
    at::Tensor dO,
    double softmax_scale
) {
    TORCH_CHECK(false,
        "backward_full_mxfp4_direct has been removed from the benchmarkable path: "
        "it scalar-dequantized prequantized MXFP4 Q/K/V into BF16/float work and "
        "was not a true MXFP4 tensorcore FlashAttention backward.");
    return {};
#if 0
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(V_fp4); CHECK_INPUT(V_sc_prepared);
    CHECK_INPUT(O); CHECK_INPUT(LSE); CHECK_INPUT(dO);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "V_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(Q_sc.dtype() == at::ScalarType::Float8_e4m3fn, "Q_sc must be float8_e4m3fn");
    TORCH_CHECK(K_sc.dtype() == at::ScalarType::Float8_e4m3fn, "K_sc must be float8_e4m3fn");
    TORCH_CHECK(V_sc_prepared.dtype() == at::ScalarType::Float8_e4m3fn, "V_sc_prepared must be float8_e4m3fn");
    TORCH_CHECK(Q_sg.dtype() == at::ScalarType::Float, "Q_sg must be float32");
    TORCH_CHECK(K_sg.dtype() == at::ScalarType::Float, "K_sg must be float32");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(dO.dtype() == at::ScalarType::BFloat16, "dO must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");

    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q/K must have shape (B,H,S,96)");
    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    TORCH_CHECK(Q.size(3) == mxfp4_full_direct::QK_PACKED, "Q packed dim must be 96");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K must match Q shape");
    TORCH_CHECK(Q_sc.sizes() == at::IntArrayRef({batch, seqlen / 128, heads * 3, 512}), "Q_sc has wrong shape");
    TORCH_CHECK(K_sc.sizes() == at::IntArrayRef({batch, seqlen / 64, heads * 3, 512}), "K_sc has wrong shape");
    TORCH_CHECK(Q_sg.sizes() == at::IntArrayRef({batch, heads}), "Q_sg has wrong shape");
    TORCH_CHECK(K_sg.sizes() == at::IntArrayRef({batch, heads}), "K_sg has wrong shape");
    TORCH_CHECK(V_fp4.sizes() == at::IntArrayRef({batch, heads, 128, seqlen / 2}), "V_fp4 has wrong shape");
    TORCH_CHECK(V_sc_prepared.sizes() == at::IntArrayRef({batch, heads, 1, seqlen / 64, 512}), "V_sc_prepared has wrong shape");
    TORCH_CHECK(O.sizes() == at::IntArrayRef({batch, seqlen, heads, 128}), "O has wrong shape");
    TORCH_CHECK(dO.sizes() == O.sizes(), "dO must match O shape");
    TORCH_CHECK(LSE.sizes() == at::IntArrayRef({batch, heads, 1, seqlen}), "LSE must have shape (B,H,1,S)");
    TORCH_CHECK(seqlen % 128 == 0, "seqlen must be divisible by 128");

    auto opts = O.options().dtype(at::kFloat);
    at::Tensor dQ = O.new_zeros({batch, seqlen, heads, 192}, opts);
    at::Tensor dK = O.new_zeros({batch, seqlen, heads, 192}, opts);
    at::Tensor dV = O.new_zeros({batch, seqlen, heads, 128}, opts);
    at::Tensor delta = O.new_empty({batch, heads, seqlen}, opts);

    const int threads = 128;
    const int64_t rows = static_cast<int64_t>(batch) * heads * seqlen;
    mxfp4_full_direct::delta_kernel<<<static_cast<int>((rows + threads - 1) / threads), threads>>>(
        O.data_ptr<at::BFloat16>(),
        dO.data_ptr<at::BFloat16>(),
        delta.data_ptr<float>(),
        batch,
        heads,
        seqlen);
    CUDACHECK(cudaGetLastError());

    const int64_t pairs = static_cast<int64_t>(batch) * heads * seqlen * seqlen;
    mxfp4_full_direct::pair_kernel<<<static_cast<int>((pairs + threads - 1) / threads), threads>>>(
        reinterpret_cast<const uint8_t *>(Q.data_ptr()),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        Q_sg.data_ptr<float>(),
        reinterpret_cast<const uint8_t *>(K.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        K_sg.data_ptr<float>(),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
        dO.data_ptr<at::BFloat16>(),
        LSE.data_ptr<float>(),
        delta.data_ptr<float>(),
        dQ.data_ptr<float>(),
        dK.data_ptr<float>(),
        dV.data_ptr<float>(),
        batch,
        heads,
        seqlen,
        static_cast<float>(softmax_scale));
    CUDACHECK(cudaGetLastError());

    return {dQ, dK, dV};
#endif
}

std::vector<at::Tensor> dispatch_backward_full_mxfp4_tile_staged(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor V_fp4,
    at::Tensor V_sc_prepared,
    at::Tensor O,
    at::Tensor LSE,
    at::Tensor dO,
    double softmax_scale
) {
    TORCH_CHECK(false,
        "backward_full_mxfp4_tile_staged has been removed from the benchmarkable path: "
        "it scalar-dequantized prequantized MXFP4 Q/K/V into BF16 shared tiles and "
        "was not a true MXFP4 tensorcore FlashAttention backward.");
    return {};
#if 0
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(V_fp4); CHECK_INPUT(V_sc_prepared);
    CHECK_INPUT(O); CHECK_INPUT(LSE); CHECK_INPUT(dO);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be float4_e2m1fn_x2");
    TORCH_CHECK(V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "V_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(Q_sc.dtype() == at::ScalarType::Float8_e4m3fn, "Q_sc must be float8_e4m3fn");
    TORCH_CHECK(K_sc.dtype() == at::ScalarType::Float8_e4m3fn, "K_sc must be float8_e4m3fn");
    TORCH_CHECK(V_sc_prepared.dtype() == at::ScalarType::Float8_e4m3fn, "V_sc_prepared must be float8_e4m3fn");
    TORCH_CHECK(Q_sg.dtype() == at::ScalarType::Float, "Q_sg must be float32");
    TORCH_CHECK(K_sg.dtype() == at::ScalarType::Float, "K_sg must be float32");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(dO.dtype() == at::ScalarType::BFloat16, "dO must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be float32");

    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q/K must have shape (B,H,S,96)");
    const int batch = Q.size(0);
    const int heads = Q.size(1);
    const int seqlen = Q.size(2);
    TORCH_CHECK(Q.size(3) == mxfp4_full_direct::QK_PACKED, "Q packed dim must be 96");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K must match Q shape");
    TORCH_CHECK(Q_sc.sizes() == at::IntArrayRef({batch, seqlen / 128, heads * 3, 512}), "Q_sc has wrong shape");
    TORCH_CHECK(K_sc.sizes() == at::IntArrayRef({batch, seqlen / 64, heads * 3, 512}), "K_sc has wrong shape");
    TORCH_CHECK(Q_sg.sizes() == at::IntArrayRef({batch, heads}), "Q_sg has wrong shape");
    TORCH_CHECK(K_sg.sizes() == at::IntArrayRef({batch, heads}), "K_sg has wrong shape");
    TORCH_CHECK(V_fp4.sizes() == at::IntArrayRef({batch, heads, 128, seqlen / 2}), "V_fp4 has wrong shape");
    TORCH_CHECK(V_sc_prepared.sizes() == at::IntArrayRef({batch, heads, 1, seqlen / 64, 512}), "V_sc_prepared has wrong shape");
    TORCH_CHECK(O.sizes() == at::IntArrayRef({batch, seqlen, heads, 128}), "O has wrong shape");
    TORCH_CHECK(dO.sizes() == O.sizes(), "dO must match O shape");
    TORCH_CHECK(LSE.sizes() == at::IntArrayRef({batch, heads, 1, seqlen}), "LSE must have shape (B,H,1,S)");
    TORCH_CHECK(seqlen % 128 == 0, "seqlen must be divisible by 128");

    auto opts = O.options().dtype(at::kFloat);
    at::Tensor dQ = O.new_empty({batch, seqlen, heads, 192}, opts);
    at::Tensor dK = O.new_empty({batch, seqlen, heads, 192}, opts);
    at::Tensor dV = O.new_empty({batch, seqlen, heads, 128}, opts);
    at::Tensor dQAccum = O.new_zeros({batch, heads, seqlen, 192}, opts);
    at::Tensor delta = O.new_empty({batch, heads, seqlen}, opts);

    const int threads = 128;
    const int64_t rows = static_cast<int64_t>(batch) * heads * seqlen;
    mxfp4_full_direct::delta_kernel<<<static_cast<int>((rows + threads - 1) / threads), threads>>>(
        O.data_ptr<at::BFloat16>(),
        dO.data_ptr<at::BFloat16>(),
        delta.data_ptr<float>(),
        batch,
        heads,
        seqlen);
    CUDACHECK(cudaGetLastError());

    using C = mxfp4_full_tile_staged::config;
    using DQAccGL = mxfp4_full_tile_staged::dqacc_chunk_gl;
    using DKGL = mxfp4_full_tile_staged::dk_gl;
    using DVGL = mxfp4_full_tile_staged::dv_gl;
    DQAccGL dq_accum_gl = ::kittens::make_gl<DQAccGL>(
        reinterpret_cast<uint64_t>(dQAccum.data_ptr<float>()),
        batch,
        heads,
        seqlen,
        C::Dqk
    );
    DKGL dk_gl = ::kittens::make_gl<DKGL>(
        reinterpret_cast<uint64_t>(dK.data_ptr<float>()),
        batch,
        seqlen,
        heads,
        C::Dqk
    );
    DVGL dv_gl = ::kittens::make_gl<DVGL>(
        reinterpret_cast<uint64_t>(dV.data_ptr<float>()),
        batch,
        seqlen,
        heads,
        C::Dvo
    );
    dim3 grid(num_k_blocks, heads, batch);
    dim3 block(C::BlockThreads, 1, 1);
    CUDACHECK(cudaFuncSetAttribute(
        mxfp4_full_tile_staged::kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    ));
    mxfp4_full_tile_staged::kernel<<<grid, block>>>(
        reinterpret_cast<const uint8_t *>(Q.data_ptr()),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        Q_sg.data_ptr<float>(),
        reinterpret_cast<const uint8_t *>(K.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        K_sg.data_ptr<float>(),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
        dO.data_ptr<at::BFloat16>(),
        LSE.data_ptr<float>(),
        delta.data_ptr<float>(),
        dq_accum_gl,
        dk_gl,
        dv_gl,
        heads,
        seqlen,
        static_cast<float>(softmax_scale));
    CUDACHECK(cudaGetLastError());

    dim3 dq_grid(seqlen / tkfa4::kRefTileM, heads, batch);
    dim3 dq_block(WARP_THREADS, 3, 1);
    mxfp4_full_tile_staged::reduce_dq_kernel<<<dq_grid, dq_block>>>(
        dq_accum_gl,
        dQ.data_ptr<float>(),
        heads,
        seqlen);
    CUDACHECK(cudaGetLastError());

    return {dQ, dK, dV};
#endif
}

std::vector<at::Tensor> dispatch_backward_full_mxfp4_tensorcore(
    at::Tensor Q,
    at::Tensor Q_sc,
    at::Tensor Q_sg,
    at::Tensor K,
    at::Tensor K_sc,
    at::Tensor K_sg,
    at::Tensor Q_col,
    at::Tensor Q_col_sc,
    at::Tensor K_col,
    at::Tensor K_col_sc,
    at::Tensor V_fp4,
    at::Tensor V_sc_prepared,
    at::Tensor O,
    at::Tensor LSE,
    at::Tensor dO,
    double softmax_scale
) {
    CHECK_INPUT(Q); CHECK_INPUT(Q_sc); CHECK_INPUT(Q_sg);
    CHECK_INPUT(K); CHECK_INPUT(K_sc); CHECK_INPUT(K_sg);
    CHECK_INPUT(Q_col); CHECK_INPUT(Q_col_sc);
    CHECK_INPUT(K_col); CHECK_INPUT(K_col_sc);
    CHECK_INPUT(V_fp4); CHECK_INPUT(V_sc_prepared);
    CHECK_INPUT(O); CHECK_INPUT(LSE); CHECK_INPUT(dO);

    TORCH_CHECK(Q.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q must be prequantized float4_e2m1fn_x2");
    TORCH_CHECK(K.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K must be prequantized float4_e2m1fn_x2");
    TORCH_CHECK(Q_col.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Q_col must be saved prequantized float4_e2m1fn_x2");
    TORCH_CHECK(K_col.dtype() == at::ScalarType::Float4_e2m1fn_x2, "K_col must be saved prequantized float4_e2m1fn_x2");
    TORCH_CHECK(V_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "V_fp4 must be saved prequantized float4_e2m1fn_x2");
    TORCH_CHECK(Q_sg.dtype() == at::ScalarType::Float, "Q_sg must be float32");
    TORCH_CHECK(K_sg.dtype() == at::ScalarType::Float, "K_sg must be float32");
    TORCH_CHECK(O.dtype() == at::ScalarType::BFloat16, "O must be bfloat16");
    TORCH_CHECK(dO.dtype() == at::ScalarType::BFloat16, "dO must be bfloat16");
    TORCH_CHECK(LSE.dtype() == at::ScalarType::Float, "LSE must be forward-saved float32");

    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4, "Q/K must have shape (B,H,S,96)");
    const int batch = static_cast<int>(Q.size(0));
    const int heads = static_cast<int>(Q.size(1));
    const int seqlen = static_cast<int>(Q.size(2));
    TORCH_CHECK(Q.size(3) == 96, "Q/K packed dim must be 96 for Dqk=192");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K must match Q shape");
    TORCH_CHECK(Q_sc.sizes() == at::IntArrayRef({batch, seqlen / 128, heads * 3, 512}), "Q_sc has wrong B300 live-QK shape");
    TORCH_CHECK(K_sc.sizes() == at::IntArrayRef({batch, seqlen / 64, heads * 3, 512}), "K_sc has wrong B300 live-QK shape");
    TORCH_CHECK(Q_sg.sizes() == at::IntArrayRef({batch, heads}), "Q_sg has wrong shape");
    TORCH_CHECK(K_sg.sizes() == at::IntArrayRef({batch, heads}), "K_sg has wrong shape");
    TORCH_CHECK(V_fp4.sizes() == at::IntArrayRef({batch, heads, 128, seqlen / 2}), "V_fp4 has wrong saved prequantized shape");
    TORCH_CHECK(V_sc_prepared.sizes() == at::IntArrayRef({batch, heads, 1, seqlen / 64, 512}), "V_sc_prepared has wrong shape");
    TORCH_CHECK(O.sizes() == at::IntArrayRef({batch, seqlen, heads, 128}), "O has wrong shape");
    TORCH_CHECK(dO.sizes() == O.sizes(), "dO must match O shape");
    TORCH_CHECK(LSE.sizes() == at::IntArrayRef({batch, heads, 1, seqlen}), "LSE must have shape (B,H,1,S)");
    TORCH_CHECK(seqlen % 128 == 0, "seqlen must be divisible by 128");
    TORCH_CHECK(std::isfinite(static_cast<float>(softmax_scale)), "softmax_scale must be finite");

    const char *rewrite_scaffold_env = std::getenv("TK_FA4_MXFP4_BWD_FULL_REWRITE_SCAFFOLD");
    const bool use_rewrite_scaffold =
        rewrite_scaffold_env != nullptr &&
        std::strcmp(rewrite_scaffold_env, "0") != 0 &&
        std::strcmp(rewrite_scaffold_env, "false") != 0 &&
        std::strcmp(rewrite_scaffold_env, "False") != 0 &&
        std::strcmp(rewrite_scaffold_env, "no") != 0 &&
        std::strcmp(rewrite_scaffold_env, "off") != 0;
    TORCH_CHECK(use_rewrite_scaffold,
        "backward_full_mxfp4_tensorcore is the only valid full-MXFP4 backward contract, "
        "but the fused tensorcore kernel is not implemented yet. It must consume saved "
        "row+column prequantized Q/K, saved prequantized V, and forward-saved LSE directly; "
        "it must not scalar-dequantize Q/K/V or materialize a BF16 Q/K bridge. Set "
        "TK_FA4_MXFP4_BWD_FULL_REWRITE_SCAFFOLD=1 only to run the BF16-staged key-owned "
        "scheduler scaffold while replacing its operand staging with true MXFP4 tensorcore loads.");

    using C = mxfp4_full_tile_staged::config;
    const int num_k_blocks = seqlen / (tkfa4::kRefTileM * C::WarpTiles);
    TORCH_CHECK(
        num_k_blocks * tkfa4::kRefTileM * C::WarpTiles == seqlen,
        "TK_FA4_MXFP4_BWD_FULL_REWRITE_SCAFFOLD currently requires seqlen to be divisible by 512"
    );

    auto opts = O.options().dtype(at::kFloat);
    at::Tensor dQ = O.new_empty({batch, seqlen, heads, 192}, opts);
    at::Tensor dK = O.new_empty({batch, seqlen, heads, 192}, opts);
    at::Tensor dV = O.new_empty({batch, seqlen, heads, 128}, opts);
    at::Tensor dQAccum = (num_k_blocks == 1)
        ? O.new_empty({batch, heads, seqlen, 192}, opts)
        : O.new_zeros({batch, heads, seqlen, 192}, opts);

    using DQAccGL = mxfp4_full_tile_staged::dqacc_chunk_gl;
    using DKGL = mxfp4_full_tile_staged::dk_gl;
    using DVGL = mxfp4_full_tile_staged::dv_gl;
    DQAccGL dq_accum_gl = ::kittens::make_gl<DQAccGL>(
        reinterpret_cast<uint64_t>(dQAccum.data_ptr<float>()),
        batch,
        heads,
        seqlen,
        C::Dqk
    );
    DKGL dk_gl = ::kittens::make_gl<DKGL>(
        reinterpret_cast<uint64_t>(dK.data_ptr<float>()),
        batch,
        seqlen,
        heads,
        C::Dqk
    );
    DVGL dv_gl = ::kittens::make_gl<DVGL>(
        reinterpret_cast<uint64_t>(dV.data_ptr<float>()),
        batch,
        seqlen,
        heads,
        C::Dvo
    );
    dim3 grid(num_k_blocks, heads, batch);
    dim3 block(C::BlockThreads, 1, 1);
    CUDACHECK(cudaFuncSetAttribute(
        mxfp4_full_tile_staged::kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    ));
    mxfp4_full_tile_staged::kernel<<<grid, block>>>(
        reinterpret_cast<const uint8_t *>(Q.data_ptr()),
        reinterpret_cast<const uint8_t *>(Q_sc.data_ptr()),
        Q_sg.data_ptr<float>(),
        reinterpret_cast<const uint8_t *>(K.data_ptr()),
        reinterpret_cast<const uint8_t *>(K_sc.data_ptr()),
        K_sg.data_ptr<float>(),
        reinterpret_cast<const uint8_t *>(V_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(V_sc_prepared.data_ptr()),
        O.data_ptr<at::BFloat16>(),
        dO.data_ptr<at::BFloat16>(),
        LSE.data_ptr<float>(),
        dq_accum_gl,
        dk_gl,
        dv_gl,
        heads,
        seqlen,
        static_cast<float>(softmax_scale));
    CUDACHECK(cudaGetLastError());

    dim3 dq_grid(seqlen / tkfa4::kRefTileM, heads, batch);
    dim3 dq_block(WARP_THREADS, 3, 1);
    mxfp4_full_tile_staged::reduce_dq_kernel<<<dq_grid, dq_block>>>(
        dq_accum_gl,
        dQ.data_ptr<float>(),
        heads,
        seqlen);
    CUDACHECK(cudaGetLastError());

    return {dQ, dK, dV};
}
#endif
#endif
