// Backward-owned host helpers for fp4_fa4_bwd.cu.
// These intentionally use bwd-specific names so the full mixed module can
// include forward and backward dispatch files without helper redefinitions.

inline void check_fp4fa4_bwd_mxfp4_v_scale_tensor(const at::Tensor &t, int batch, int heads, int seqlen, int dvo) {
    CHECK_INPUT(t);
    TORCH_CHECK(t.dtype() == at::ScalarType::Byte, "MXFP4 V_sc must be uint8 E8M0");
    TORCH_CHECK(t.dim() == 6, "MXFP4 V_sc must have shape (batch, heads, scale_rows, seqlen/128, 32, 16)");
    TORCH_CHECK(t.size(0) == batch && t.size(1) == heads && t.size(2) >= dvo / 128 &&
                t.size(3) == seqlen / 128 && t.size(4) == 32 && t.size(5) == 16,
                "MXFP4 V_sc must have shape (batch, heads, >=dvo/128, seqlen/128, 32, 16)");
}

static inline bool fp4fa4_bwd_env_enabled(const char *name) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    return std::atoi(value) != 0;
}

template <typename C>
static inline dim3 bwd_fullgrid_dim_fp4pv_exp(int batch, int heads, int seqlen) {
    const int tiles_m = seqlen / C::Mb;
    const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
    const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
    const int total_bids = batch * heads * num_block * C::CLUSTER_SIZE;
    return dim3(total_bids);
}

template <typename C>
static inline dim3 bwd_persistent_dim_fp4pv_exp(int batch, int heads, int seqlen) {
    const int tiles_m = seqlen / C::Mb;
    const int tiles_per_cluster = C::NUM_SOFTMAXXERS * C::CLUSTER_SIZE;
    const int num_block = (tiles_m + tiles_per_cluster - 1) / tiles_per_cluster;
    const int total_bids = batch * heads * num_block * C::CLUSTER_SIZE;
    constexpr int PERSISTENT_GRID_CAP = fp4pv_persistent_grid_cap<C>();
    int grid_x = total_bids < PERSISTENT_GRID_CAP ? total_bids : PERSISTENT_GRID_CAP;
    if (grid_x < C::CLUSTER_SIZE) {
        grid_x = C::CLUSTER_SIZE;
    } else {
        grid_x = ((grid_x + C::CLUSTER_SIZE - 1) / C::CLUSTER_SIZE) * C::CLUSTER_SIZE;
    }
    return dim3(grid_x);
}
