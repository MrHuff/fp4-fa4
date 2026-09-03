#pragma once

#include "fa4_common.cuh"

namespace tkfa4::scheduler {

struct forward_tile_coord {
    int batch_idx;
    int q_head_idx;
    int kv_head_idx;
    int m_block;
};

struct forward_cluster_tile_coord {
    int batch_idx;
    int q_head_idx;
    int kv_head_idx;
    int m_block_base;
    int q_local;
};

struct forward_cluster_task_coord {
    int batch_idx;
    int q_head_idx;
    int kv_head_idx;
    int m_block_base;
    int q_local;
};

__host__ __device__ inline int q_head_to_kv_head(int q_head_idx, int head_ratio) {
    return q_head_idx / head_ratio;
}

__host__ __device__ inline int forward_total_tiles(
    int batch_size,
    int q_heads,
    int num_m_blocks
) {
    return batch_size * q_heads * num_m_blocks;
}

__host__ __device__ inline int forward_cluster_total_tiles(
    int batch_size,
    int kv_heads,
    int head_ratio,
    int num_m_blocks
) {
    return batch_size * kv_heads * head_ratio * tkfa4::ceil_div(num_m_blocks, 2);
}

__host__ __device__ inline int forward_cluster_task_total_tiles(
    int batch_size,
    int kv_heads,
    int head_ratio,
    int num_m_blocks,
    int q_stages,
    int cluster_size
) {
    (void)cluster_size;
    const int blocks_per_task = q_stages;
    return batch_size * kv_heads * head_ratio * tkfa4::ceil_div(num_m_blocks, blocks_per_task);
}

__host__ __device__ inline forward_tile_coord decode_forward_tile(
    int linear_tile,
    int batch_size,
    int q_heads,
    int kv_heads,
    int head_ratio,
    int num_m_blocks,
    bool causal
) {
    const int q_per_kv = head_ratio;
    const int tiles_per_batch = kv_heads * num_m_blocks * q_per_kv;
    const int batch_idx = linear_tile / tiles_per_batch;
    const int tile_in_batch = linear_tile % tiles_per_batch;
    const int tiles_per_kv = num_m_blocks * q_per_kv;
    const int kv_head_idx = tile_in_batch / tiles_per_kv;
    const int tile_in_kv = tile_in_batch % tiles_per_kv;
    const int m_block_asc = tile_in_kv / q_per_kv;
    const int q_local = tile_in_kv % q_per_kv;
    const int m_block = causal ? (num_m_blocks - 1 - m_block_asc) : m_block_asc;
    const int q_head_idx = kv_head_idx * q_per_kv + q_local;

    return {
        batch_idx,
        q_head_idx,
        kv_head_idx,
        m_block,
    };
}

__host__ __device__ inline forward_cluster_tile_coord decode_forward_cluster_tile(
    int linear_tile,
    int batch_size,
    int kv_heads,
    int head_ratio,
    int num_m_blocks,
    bool causal
) {
    const int q_per_kv = head_ratio;
    const int num_cluster_blocks = tkfa4::ceil_div(num_m_blocks, 2);
    const int tiles_per_batch = kv_heads * num_cluster_blocks * q_per_kv;
    const int batch_idx = linear_tile / tiles_per_batch;
    const int tile_in_batch = linear_tile % tiles_per_batch;
    const int tiles_per_kv = num_cluster_blocks * q_per_kv;
    const int kv_head_idx = tile_in_batch / tiles_per_kv;
    const int tile_in_kv = tile_in_batch % tiles_per_kv;
    const int m_cluster_asc = tile_in_kv / q_per_kv;
    const int q_local = tile_in_kv % q_per_kv;
    const int m_cluster = causal ? (num_cluster_blocks - 1 - m_cluster_asc) : m_cluster_asc;
    const int q_head_idx = kv_head_idx * q_per_kv + q_local;

    return {
        batch_idx,
        q_head_idx,
        kv_head_idx,
        m_cluster * 2,
        q_local,
    };
}

__host__ __device__ inline int forward_local_m_block(
    int m_block_base,
    int cta_rank,
    int num_m_blocks,
    bool causal
) {
    if (!causal) {
        return m_block_base + cta_rank;
    }
    const int hi = m_block_base + 1 < num_m_blocks ? m_block_base + 1 : (num_m_blocks - 1);
    return hi - cta_rank;
}

__host__ __device__ inline forward_cluster_task_coord decode_forward_cluster_task(
    int linear_tile,
    int batch_size,
    int kv_heads,
    int head_ratio,
    int num_m_blocks,
    int q_stages,
    int cluster_size,
    bool causal
) {
    const int q_per_kv = head_ratio;
    (void)cluster_size;
    const int blocks_per_task = q_stages;
    const int num_cluster_blocks = tkfa4::ceil_div(num_m_blocks, blocks_per_task);
    const int tiles_per_batch = kv_heads * num_cluster_blocks * q_per_kv;
    const int batch_idx = linear_tile / tiles_per_batch;
    const int tile_in_batch = linear_tile % tiles_per_batch;
    const int tiles_per_kv = num_cluster_blocks * q_per_kv;
    const int kv_head_idx = tile_in_batch / tiles_per_kv;
    const int tile_in_kv = tile_in_batch % tiles_per_kv;
    const int m_cluster_asc = tile_in_kv / q_per_kv;
    const int q_local = tile_in_kv % q_per_kv;
    const int m_cluster = causal ? (num_cluster_blocks - 1 - m_cluster_asc) : m_cluster_asc;
    const int q_head_idx = kv_head_idx * q_per_kv + q_local;

    return {
        batch_idx,
        q_head_idx,
        kv_head_idx,
        m_cluster * blocks_per_task,
        q_local,
    };
}

__host__ __device__ inline int forward_cluster_stage_m_block(
    int m_block_base,
    int stage,
    int cta_rank,
    int num_m_blocks,
    int q_stages,
    int cluster_size,
    bool causal
) {
    (void)cta_rank;
    (void)cluster_size;
    if (!causal) {
        return m_block_base + stage;
    }
    const int blocks_per_task = q_stages;
    const int hi_candidate = m_block_base + blocks_per_task - 1;
    const int hi = hi_candidate < num_m_blocks ? hi_candidate : (num_m_blocks - 1);
    return hi - stage;
}

__host__ __device__ inline int forward_cta_n_block_max(
    int m_block,
    int seqlen_q,
    int seqlen_k,
    bool causal
) {
    int n_block_max = tkfa4::ceil_div(seqlen_k, tkfa4::kForwardTileN);
    if (causal) {
        const int m_idx_max = (m_block + 1) * tkfa4::kForwardTileM;
        const int n_idx_right = m_idx_max + seqlen_k - seqlen_q;
        const int bounded = tkfa4::ceil_div(n_idx_right, tkfa4::kForwardTileN);
        n_block_max = bounded < n_block_max ? bounded : n_block_max;
    }
    return n_block_max > 0 ? n_block_max : 0;
}

__host__ __device__ inline int forward_subtile_n_block_max(
    int m_subtile,
    int seqlen_q,
    int seqlen_k,
    bool causal
) {
    int n_block_max = tkfa4::ceil_div(seqlen_k, tkfa4::kForwardTileN);
    if (causal) {
        const int m_idx_max = (m_subtile + 1) * tkfa4::kForwardSubtileM;
        const int n_idx_right = m_idx_max + seqlen_k - seqlen_q;
        const int bounded = tkfa4::ceil_div(n_idx_right, tkfa4::kForwardTileN);
        n_block_max = bounded < n_block_max ? bounded : n_block_max;
    }
    return n_block_max > 0 ? n_block_max : 0;
}

__host__ __device__ inline int forward_valid_cols(int n_block, int seqlen_k) {
    const int remaining = seqlen_k - n_block * tkfa4::kForwardTileN;
    if (remaining <= 0) {
        return 0;
    }
    return remaining < tkfa4::kForwardTileN ? remaining : tkfa4::kForwardTileN;
}

__host__ __device__ inline int forward_diag_n_block(int m_subtile) {
    return m_subtile / (tkfa4::kForwardTileN / tkfa4::kForwardSubtileM);
}

}  // namespace tkfa4::scheduler
