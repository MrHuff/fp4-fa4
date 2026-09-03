__global__ void kernel_bf16_row_dot_delta_bshd(
    const bf16 *__restrict__ out,
    const bf16 *__restrict__ dout,
    float *__restrict__ delta,
    int batch,
    int heads,
    int seqlen
) {
    constexpr int DVO = 128;
    const int row_id = blockIdx.x;
    const int s = row_id % seqlen;
    const int h = (row_id / seqlen) % heads;
    const int b = row_id / (seqlen * heads);
    const int tid = threadIdx.x;
    float value = 0.0f;
    if (b < batch && tid < DVO) {
        const int64_t base = (((static_cast<int64_t>(b) * seqlen + s) * heads + h) * DVO);
        value = __bfloat162float(out[base + tid]) * __bfloat162float(dout[base + tid]);
    }
    value = fp4pv_warp_reduce_sum(value);
    __shared__ float warp_sums[4];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = (lane < 4) ? warp_sums[lane] : 0.0f;
        value = fp4pv_warp_reduce_sum(value);
        if (lane == 0 && b < batch) {
            delta[(static_cast<int64_t>(b) * heads + h) * seqlen + s] = value;
        }
    }
}

at::Tensor dispatch_bf16_row_dot_delta_bshd(at::Tensor Out, at::Tensor dOut) {
    CHECK_INPUT(Out); CHECK_INPUT(dOut);
    TORCH_CHECK(Out.dtype() == at::ScalarType::BFloat16, "Out must be BF16");
    TORCH_CHECK(dOut.dtype() == at::ScalarType::BFloat16, "dOut must be BF16");
    TORCH_CHECK(Out.dim() == 4, "Out must have shape (batch, seqlen, heads, 128)");
    TORCH_CHECK(dOut.sizes() == Out.sizes(), "dOut must match Out shape");
    const int batch = static_cast<int>(Out.size(0));
    const int seqlen = static_cast<int>(Out.size(1));
    const int heads = static_cast<int>(Out.size(2));
    TORCH_CHECK(Out.size(3) == 128, "Out/dOut last dimension must be 128");
    auto Delta = at::empty({batch, heads, seqlen}, Out.options().dtype(at::ScalarType::Float));
    const int total_rows = batch * heads * seqlen;
    kernel_bf16_row_dot_delta_bshd<<<total_rows, 128>>>(
        reinterpret_cast<const bf16 *>(Out.data_ptr()),
        reinterpret_cast<const bf16 *>(dOut.data_ptr()),
        reinterpret_cast<float *>(Delta.data_ptr()),
        batch,
        heads,
        seqlen);
    CUDACHECK(cudaGetLastError());
    return Delta;
}

__global__ void kernel_bf16_transpose_to_mxfp4(
    const bf16 *__restrict__ src,
    uint8_t *__restrict__ dst_fp4,
    uint8_t *__restrict__ dst_sc,
    int batch,
    int heads,
    int seqlen,
    int quant_mode
) {
    constexpr int TILE = 128;
    const int tiles = seqlen / TILE;
    const int tile_id = blockIdx.x;
    const int k_tile = tile_id % tiles;
    const int row_tile = (tile_id / tiles) % tiles;
    const int h = (tile_id / (tiles * tiles)) % heads;
    const int b = tile_id / (tiles * tiles * heads);
    const int local_row = threadIdx.x;
    if (b >= batch || local_row >= TILE) {
        return;
    }

    const int out_row = row_tile * TILE + local_row;
    const int packed_seqlen = seqlen / 2;
    const int heads_offset = b * heads + h;
    uint8_t *payload_row = &dst_fp4[((heads_offset * seqlen + out_row) * packed_seqlen) + k_tile * (TILE / 2)];
    uint8_t *scale_tile = &dst_sc[((heads_offset * tiles + row_tile) * tiles + k_tile) * 512];

    #pragma unroll
    for (int mx_block = 0; mx_block < 4; ++mx_block) {
        __align__(8) bf16_2 values[16];
        float block_amax = 0.0f;
        const int src_row_base = k_tile * TILE + mx_block * 32;
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const int src_row0 = src_row_base + jj * 2;
            const int64_t src_off0 =
                ((static_cast<int64_t>(heads_offset) * seqlen + src_row0) * seqlen) + out_row;
            values[jj].x = src[src_off0];
            values[jj].y = src[src_off0 + seqlen];
            block_amax = fmaxf(
                block_amax,
                fmaxf(fabsf(__bfloat162float(values[jj].x)), fabsf(__bfloat162float(values[jj].y))));
        }
        uint8_t e8m0 = fp4pv_float_to_e8m0(block_amax, quant_mode);
        uint32_t words[4];
        if (e8m0 == 0) {
            #pragma unroll
            for (int pack = 0; pack < 4; ++pack) {
                words[pack] = 0x77777777u;
            }
        } else {
            const float coeff = 6.0f * fp4pv_exp2f_rcp_e8m0(e8m0);
            #pragma unroll
            for (int pack = 0; pack < 4; ++pack) {
                const int off = pack * 4;
                const uint64_t e03 = *reinterpret_cast<const uint64_t *>(&values[off + 0]);
                const uint64_t e47 = *reinterpret_cast<const uint64_t *>(&values[off + 2]);
                words[pack] = transformer_engine::ptx::mul_cvt_bf16_to_fp4_8x_round_to_nearest<float>(e03, e47, coeff);
            }
        }
        uint8_t *payload_dst = payload_row + mx_block * 16;
        *reinterpret_cast<uint32_t *>(payload_dst + 0) = words[0];
        *reinterpret_cast<uint32_t *>(payload_dst + 4) = words[1];
        *reinterpret_cast<uint32_t *>(payload_dst + 8) = words[2];
        *reinterpret_cast<uint32_t *>(payload_dst + 12) = words[3];
        scale_tile[fp4pv_scale_swizzle_idx(local_row, mx_block)] = e8m0;
    }
}

std::vector<at::Tensor> dispatch_bf16_transpose_to_mxfp4(at::Tensor Src, int quant_mode) {
    CHECK_INPUT(Src);
    TORCH_CHECK(Src.dtype() == at::ScalarType::BFloat16, "Src must be BF16");
    TORCH_CHECK(Src.dim() == 4, "Src must have shape (batch, heads, seqlen, seqlen)");
    const int batch = static_cast<int>(Src.size(0));
    const int heads = static_cast<int>(Src.size(1));
    const int seqlen = static_cast<int>(Src.size(2));
    TORCH_CHECK(Src.size(3) == seqlen, "Src must be square in the last two dimensions");
    TORCH_CHECK(seqlen % 128 == 0, "Src seqlen must be divisible by 128");
    TORCH_CHECK(quant_mode >= 0 && quant_mode <= 2, "quant_mode must be 0 (rte), 1 (encode), or 2 (decode)");

    const int tiles = seqlen / 128;
    auto Dst_fp4 = at::empty({batch, heads, seqlen, seqlen / 2}, Src.options().dtype(at::ScalarType::Float4_e2m1fn_x2));
    auto Dst_sc = at::empty({batch, heads, tiles, tiles, 32, 16}, Src.options().dtype(at::ScalarType::Byte));
    const int total_tiles = batch * heads * tiles * tiles;
    kernel_bf16_transpose_to_mxfp4<<<total_tiles, 128>>>(
        reinterpret_cast<const bf16 *>(Src.data_ptr()),
        reinterpret_cast<uint8_t *>(Dst_fp4.data_ptr()),
        reinterpret_cast<uint8_t *>(Dst_sc.data_ptr()),
        batch,
        heads,
        seqlen,
        quant_mode);
    CUDACHECK(cudaGetLastError());
    return {Dst_fp4, Dst_sc};
}

__device__ __forceinline__ float fp4pv_decode_e2m1_nibble(uint8_t nibble) {
    switch (nibble & 0x0Fu) {
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

__device__ __forceinline__ float fp4pv_e8m0_to_float(uint8_t e8m0) {
    if (e8m0 == 0) {
        return 0.0f;
    }
    return __uint_as_float(static_cast<uint32_t>(e8m0) << 23);
}

__device__ __forceinline__ float fp4pv_load_mxfp4_row_value(
    const uint8_t *__restrict__ src_fp4,
    const uint8_t *__restrict__ src_sc,
    int heads_offset,
    int seqlen,
    int tiles,
    int row,
    int col
) {
    const int packed_seqlen = seqlen / 2;
    const int64_t payload_idx =
        ((static_cast<int64_t>(heads_offset) * seqlen + row) * packed_seqlen) + (col >> 1);
    const uint8_t packed = src_fp4[payload_idx];
    const uint8_t nibble = (col & 1) ? ((packed >> 4) & 0x0Fu) : (packed & 0x0Fu);
    const int scale_idx = fp4pv_scale_swizzle_idx(row & 127, (col & 127) >> 5);
    const int64_t sc_idx =
        (((static_cast<int64_t>(heads_offset) * tiles + (row >> 7)) * tiles + (col >> 7)) * 512) + scale_idx;
    const float scale = fp4pv_e8m0_to_float(src_sc[sc_idx]) * (1.0f / 6.0f);
    return fp4pv_decode_e2m1_nibble(nibble) * scale;
}

__global__ void kernel_row_mxfp4_transpose_to_mxfp4(
    const uint8_t *__restrict__ src_fp4,
    const uint8_t *__restrict__ src_sc,
    uint8_t *__restrict__ dst_fp4,
    uint8_t *__restrict__ dst_sc,
    int batch,
    int heads,
    int seqlen,
    int quant_mode
) {
    constexpr int TILE = 128;
    __shared__ bf16 tile_t[TILE][TILE];
    const int tiles = seqlen / TILE;
    const int tile_id = blockIdx.x;
    const int out_k_tile = tile_id % tiles;
    const int out_row_tile = (tile_id / tiles) % tiles;
    const int h = (tile_id / (tiles * tiles)) % heads;
    const int b = tile_id / (tiles * tiles * heads);
    const int local_row = threadIdx.x;
    if (b >= batch || local_row >= TILE) {
        return;
    }

    const int heads_offset = b * heads + h;
    const int out_row = out_row_tile * TILE + local_row;
    const int packed_seqlen = seqlen / 2;
    const int src_row = out_k_tile * TILE + local_row;
    const uint8_t *src_payload_row =
        &src_fp4[((static_cast<int64_t>(heads_offset) * seqlen + src_row) * packed_seqlen) + out_row_tile * (TILE / 2)];
    const uint8_t *src_scale_tile =
        &src_sc[((static_cast<int64_t>(heads_offset) * tiles + out_k_tile) * tiles + out_row_tile) * 512];

    #pragma unroll
    for (int mx_block = 0; mx_block < 4; ++mx_block) {
        const int scale_idx = fp4pv_scale_swizzle_idx(local_row, mx_block);
        const float scale = fp4pv_e8m0_to_float(src_scale_tile[scale_idx]) * (1.0f / 6.0f);
        const uint8_t *payload_src = src_payload_row + mx_block * 16;
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            const uint8_t packed = payload_src[jj];
            const int col0 = mx_block * 32 + jj * 2;
            tile_t[col0 + 0][local_row] = __float2bfloat16(fp4pv_decode_e2m1_nibble(packed & 0x0Fu) * scale);
            tile_t[col0 + 1][local_row] = __float2bfloat16(fp4pv_decode_e2m1_nibble((packed >> 4) & 0x0Fu) * scale);
        }
    }
    __syncthreads();

    uint8_t *payload_row =
        &dst_fp4[((static_cast<int64_t>(heads_offset) * seqlen + out_row) * packed_seqlen) + out_k_tile * (TILE / 2)];
    uint8_t *scale_tile =
        &dst_sc[((static_cast<int64_t>(heads_offset) * tiles + out_row_tile) * tiles + out_k_tile) * 512];

    #pragma unroll
    for (int mx_block = 0; mx_block < 4; ++mx_block) {
        float2 values[16];
        #pragma unroll
        for (int jj = 0; jj < 16; ++jj) {
            values[jj].x = __bfloat162float(tile_t[mx_block * 32 + jj * 2 + 0][local_row]);
            values[jj].y = __bfloat162float(tile_t[mx_block * 32 + jj * 2 + 1][local_row]);
        }
        uint32_t words[4];
        uint8_t e8m0 = 0;
        fp4pv_quantize_mxfp4_scores_group_bf16_amax(values, quant_mode, words, e8m0);
        uint8_t *payload_dst = payload_row + mx_block * 16;
        *reinterpret_cast<uint32_t *>(payload_dst + 0) = words[0];
        *reinterpret_cast<uint32_t *>(payload_dst + 4) = words[1];
        *reinterpret_cast<uint32_t *>(payload_dst + 8) = words[2];
        *reinterpret_cast<uint32_t *>(payload_dst + 12) = words[3];
        scale_tile[fp4pv_scale_swizzle_idx(local_row, mx_block)] = e8m0;
    }
}

std::vector<at::Tensor> dispatch_row_mxfp4_transpose_to_mxfp4(
    at::Tensor Src_fp4,
    at::Tensor Src_sc,
    int quant_mode
) {
    CHECK_INPUT(Src_fp4);
    CHECK_INPUT(Src_sc);
    TORCH_CHECK(Src_fp4.dtype() == at::ScalarType::Float4_e2m1fn_x2, "Src_fp4 must be float4_e2m1fn_x2");
    TORCH_CHECK(Src_sc.dtype() == at::ScalarType::Byte, "Src_sc must be uint8");
    TORCH_CHECK(Src_fp4.dim() == 4, "Src_fp4 must have shape (batch, heads, seqlen, seqlen/2)");
    const int batch = static_cast<int>(Src_fp4.size(0));
    const int heads = static_cast<int>(Src_fp4.size(1));
    const int seqlen = static_cast<int>(Src_fp4.size(2));
    TORCH_CHECK(Src_fp4.size(3) == seqlen / 2, "Src_fp4 last dimension must be seqlen/2");
    TORCH_CHECK(seqlen % 128 == 0, "Src_fp4 seqlen must be divisible by 128");
    const int tiles = seqlen / 128;
    TORCH_CHECK(Src_sc.dim() == 6, "Src_sc must have shape (batch, heads, seqlen/128, seqlen/128, 32, 16)");
    TORCH_CHECK(Src_sc.size(0) == batch && Src_sc.size(1) == heads && Src_sc.size(2) == tiles && Src_sc.size(3) == tiles &&
                Src_sc.size(4) == 32 && Src_sc.size(5) == 16,
                "Src_sc must have shape (batch, heads, seqlen/128, seqlen/128, 32, 16)");
    TORCH_CHECK(quant_mode >= 0 && quant_mode <= 2, "quant_mode must be 0 (rte), 1 (encode), or 2 (decode)");

    auto Dst_fp4 = at::empty_like(Src_fp4);
    auto Dst_sc = at::empty_like(Src_sc);
    const int total_tiles = batch * heads * tiles * tiles;
    kernel_row_mxfp4_transpose_to_mxfp4<<<total_tiles, 128>>>(
        reinterpret_cast<const uint8_t *>(Src_fp4.data_ptr()),
        reinterpret_cast<const uint8_t *>(Src_sc.data_ptr()),
        reinterpret_cast<uint8_t *>(Dst_fp4.data_ptr()),
        reinterpret_cast<uint8_t *>(Dst_sc.data_ptr()),
        batch,
        heads,
        seqlen,
        quant_mode);
    CUDACHECK(cudaGetLastError());
    return {Dst_fp4, Dst_sc};
}

// Keep host dispatch and bindings split by ownership so backward optimization
// work can live under tk_fa4/fp4_fa4_bwd without touching forward dispatch.
