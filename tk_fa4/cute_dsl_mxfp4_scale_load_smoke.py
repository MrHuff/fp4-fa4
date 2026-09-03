from __future__ import annotations

from typing import Any

from cute_dsl_mxfp4_forward_d192_port import (
    Mxfp4D192PortGeometry,
    default_d192_port_geometry,
)
from cute_dsl_mxfp4_forward_scaffold import (
    load_reference_mixed_input_fmha_d256_module,
)


def build_mxfp4_k_scale_load_smoke_kernel_class() -> type[Any]:
    mixed_input_mod = load_reference_mixed_input_fmha_d256_module()
    import cutlass  # type: ignore
    import cutlass.cute as cute  # type: ignore
    import cutlass.cute.nvgpu.tcgen05 as tcgen05  # type: ignore
    import cutlass.pipeline as pipeline  # type: ignore
    import cutlass.utils as utils  # type: ignore
    import cutlass.utils.blackwell_helpers as sm100_utils  # type: ignore
    import cutlass.utils.blockscaled_layout as blockscaled_utils  # type: ignore
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait  # type: ignore

    class Mxfp4KScaleLoadSmoke:
        def __init__(self, geometry: Mxfp4D192PortGeometry | None = None):
            geometry = geometry or default_d192_port_geometry()
            self.geometry = geometry
            self.qk_cta_tiler = geometry.qk_cta_tiler
            self.qk_mma_tiler = geometry.qk_mma_tiler
            self.scale_granularity_qk = geometry.qk_scale_granularity
            self.qk_sf_vec_size = geometry.qk_sf_vec_size
            self.cluster_shape_mn = geometry.cluster_shape_mn
            self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
            self.scale_k_stage = self.qk_cta_tiler[2] // self.qk_mma_tiler[2]
            self.load_warp_id = 0
            self.consumer_warp_id = 1
            self.threads_per_warp = 32
            self.threads_per_cta = self.threads_per_warp * 2

        @cute.jit
        def __call__(
            self,
            k_iter: Any,
            scale_k_iter: Any,
            problem_shape: Any,
            stream: Any,
        ):
            b, s_k, h_q, h_k, d_qk = problem_shape
            h_r = h_q // h_k

            k_layout = cute.make_layout(
                (s_k, self.geometry.qk_head_dim_padded, ((h_r, h_k), b)),
                stride=(
                    self.geometry.qk_head_dim_padded,
                    1,
                    (
                        (
                            0,
                            self.geometry.qk_head_dim_padded * s_k,
                        ),
                        h_k * s_k * self.geometry.qk_head_dim_padded,
                    ),
                ),
            )
            k = cute.make_tensor(k_iter, k_layout)
            scale_k_d_r = self.qk_cta_tiler[2] // self.scale_granularity_qk
            scale_k_layout = cute.make_layout(
                (s_k * scale_k_d_r, ((h_r, h_k), b)),
                stride=(1, ((0, scale_k_d_r * s_k), s_k * scale_k_d_r * h_k)),
            )
            scale_k = cute.make_tensor(scale_k_iter, scale_k_layout)

            self.k_dtype = k.element_type
            self.scale_k_dtype = scale_k.element_type
            self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()

            qk_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.k_dtype,
                self.k_major_mode,
                self.k_major_mode,
                self.scale_k_dtype,
                self.qk_sf_vec_size,
                tcgen05.CtaGroup.TWO,
                self.qk_mma_tiler[:2],
            )
            cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (qk_tiled_mma.thr_id.shape,),
            )
            (
                k_scale_tma_layout,
                self.scale_k_tiler,
                _,
            ) = mixed_input_mod.prefill_utils.get_scale_smem_layout(
                self.scale_granularity_qk,
                self.qk_cta_tiler[2] // self.scale_granularity_qk,
                self.qk_mma_tiler,
                self.k_major_mode,
            )
            k_scale_smem_layout_staged = cute.append(
                k_scale_tma_layout,
                cute.make_layout(
                    (self.scale_k_stage),
                    stride=(cute.cosize(k_scale_tma_layout.outer)),
                ),
            )
            tma_load_kv_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(
                tcgen05.CtaGroup.ONE
            )
            tma_atom_scale_k, tma_tensor_scale_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_kv_op,
                scale_k,
                k_scale_tma_layout,
                (self.scale_k_tiler[0] // 2,),
            )
            self.tma_copy_scale_k_bytes = cute.size_in_bytes(
                self.scale_k_dtype, k_scale_tma_layout
            )

            @cute.struct
            class SharedStorage:
                scale_k_mbar_ptr: cute.struct.MemRange[
                    cute.Int64, self.scale_k_stage * 2
                ]

            self.shared_storage = SharedStorage
            self.kernel(
                qk_tiled_mma,
                tma_atom_scale_k,
                tma_tensor_scale_k,
                k_scale_smem_layout_staged,
            ).launch(
                grid=self.cluster_shape_mnk,
                block=[self.threads_per_cta, 1, 1],
                cluster=self.cluster_shape_mnk,
                stream=stream,
                min_blocks_per_mp=1,
            )

        @cute.kernel
        def kernel(
            self,
            qk_tiled_mma: cute.TiledMma,
            tma_atom_scale_k: cute.CopyAtom,
            mScaleK_kdl: cute.Tensor,
            k_scale_smem_layout_staged: cute.ComposedLayout,
        ):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            block_idx_m, _, _ = cute.arch.block_idx()
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (qk_tiled_mma.thr_id.shape,),
            )
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
                cta_rank_in_cluster
            )
            mma_tile_coord_v = block_idx_m % cute.size(qk_tiled_mma.thr_id.shape)

            smem = utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            scale_k_producer, scale_k_consumer = pipeline.PipelineTmaAsync.create(
                num_stages=self.scale_k_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.load_warp_id])
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.threads_per_warp
                ),
                tx_count=self.tma_copy_scale_k_bytes,
                barrier_storage=storage.scale_k_mbar_ptr.data_ptr(),
                tidx=0,
                defer_sync=True,
            ).make_participants()

            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
            pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

            sScaleK = smem.allocate_tensor(
                element_type=self.scale_k_dtype,
                layout=k_scale_smem_layout_staged.outer,
                swizzle=k_scale_smem_layout_staged.inner,
                byte_alignment=128,
            )
            kv_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )
            gScaleK_kdl = cute.flat_divide(mScaleK_kdl, self.scale_k_tiler)
            gScaleK_kdl_ = cute.logical_divide(
                gScaleK_kdl, (self.scale_k_tiler[0] // 2,)
            )[(None, mma_tile_coord_v), None, None]
            tKsScaleK, tKgScaleK_kdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_scale_k,
                block_in_cluster_coord_vmnk[1],
                kv_cta_layout,
                sScaleK,
                gScaleK_kdl_,
            )

            if warp_idx == self.load_warp_id:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_scale_k)
                tKgScaleK = tKgScaleK_kdl[None, None, 0]
                scale_k_handle = scale_k_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_scale_k,
                    tKgScaleK[None, 0],
                    tKsScaleK[None, scale_k_handle.index],
                    tma_bar_ptr=scale_k_handle.barrier,
                )

            if warp_idx == self.consumer_warp_id:
                scale_k_full = scale_k_consumer.wait_and_advance()
                scale_k_full.release()
            return

    return Mxfp4KScaleLoadSmoke


def build_mxfp4_k_scale_load_smoke_kernel(
    geometry: Mxfp4D192PortGeometry | None = None,
) -> Any:
    cls = build_mxfp4_k_scale_load_smoke_kernel_class()
    return cls(geometry=geometry)


def run_mxfp4_k_scale_load_smoke(
    *,
    batch_size: int = 1,
    seqlen_k: int = 128,
    heads_q: int = 12,
    heads_k: int = 12,
    device: str = "cuda:0",
    geometry: Mxfp4D192PortGeometry | None = None,
) -> dict[str, Any]:
    geometry = geometry or default_d192_port_geometry()

    import cutlass  # type: ignore
    import cutlass.cute as cute  # type: ignore
    import cutlass.torch as cutlass_torch  # type: ignore
    import torch
    from cutlass.cute.runtime import make_ptr  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the K-scale load smoke")
    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)

    k_numel = batch_size * heads_k * seqlen_k * geometry.qk_head_dim_padded
    k_scale_numel = (
        batch_size
        * heads_k
        * seqlen_k
        * (geometry.qk_head_dim_padded // geometry.qk_sf_vec_size)
    )
    k_torch = torch.empty((k_numel,), device=device, dtype=torch.float4_e2m1fn_x2)
    scale_k_torch = torch.empty(
        (k_scale_numel,), device=device, dtype=torch.float8_e8m0fnu
    )

    k_ptr = make_ptr(
        cutlass.Float4E2M1FN,
        k_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_k_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        scale_k_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )

    kernel = build_mxfp4_k_scale_load_smoke_kernel(geometry=geometry)
    current_stream = cutlass_torch.default_stream()
    problem_size = (
        batch_size,
        seqlen_k,
        heads_q,
        heads_k,
        geometry.qk_head_dim,
    )

    compiled = cute.compile(
        kernel,
        k_ptr,
        scale_k_ptr,
        problem_size,
        current_stream,
        options="--opt-level 2",
    )
    compiled(k_ptr, scale_k_ptr, problem_size, current_stream)
    torch.cuda.synchronize(device=device)

    return {
        "status": "ok",
        "device": str(device),
        "problem_size": problem_size,
        "k_numel": int(k_torch.numel()),
        "k_scale_numel": int(scale_k_torch.numel()),
    }


def build_mxfp4_q_scale_load_smoke_kernel_class() -> type[Any]:
    import cutlass  # type: ignore
    import cutlass.cute as cute  # type: ignore
    import cutlass.cute.nvgpu.tcgen05 as tcgen05  # type: ignore
    import cutlass.pipeline as pipeline  # type: ignore
    import cutlass.utils as utils  # type: ignore
    import cutlass.utils.blackwell_helpers as sm100_utils  # type: ignore
    import cutlass.utils.blockscaled_layout as blockscaled_utils  # type: ignore
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait  # type: ignore

    class Mxfp4QScaleLoadSmoke:
        def __init__(self, geometry: Mxfp4D192PortGeometry | None = None):
            geometry = geometry or default_d192_port_geometry()
            self.geometry = geometry
            self.qk_cta_tiler = geometry.qk_cta_tiler
            self.qk_mma_tiler = geometry.qk_mma_tiler
            self.scale_granularity_qk = geometry.qk_scale_granularity
            self.qk_sf_vec_size = geometry.qk_sf_vec_size
            self.cluster_shape_mn = geometry.cluster_shape_mn
            self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
            self.scale_q_stage = self.qk_cta_tiler[2] // self.qk_mma_tiler[2]
            self.load_warp_id = 0
            self.consumer_warp_id = 1
            self.threads_per_warp = 32
            self.threads_per_cta = self.threads_per_warp * 2

        @cute.jit
        def __call__(
            self,
            q_iter: Any,
            scale_q_iter: Any,
            problem_shape: Any,
            stream: Any,
        ):
            b, s_q, h_q, h_k, d_qk = problem_shape
            h_r = h_q // h_k

            q_layout = cute.make_layout(
                (s_q, self.geometry.qk_head_dim_padded, ((h_r, h_k), b)),
                stride=(
                    self.geometry.qk_head_dim_padded,
                    1,
                    (
                        (
                            self.geometry.qk_head_dim_padded * s_q,
                            self.geometry.qk_head_dim_padded * s_q * h_r,
                        ),
                        h_r * h_k * s_q * self.geometry.qk_head_dim_padded,
                    ),
                ),
            )
            q = cute.make_tensor(q_iter, q_layout)

            q_scale_d_r = self.qk_cta_tiler[2] // self.scale_granularity_qk
            q_scale_layout = cute.make_layout(
                (s_q * q_scale_d_r, ((h_r, h_k), b)),
                stride=(
                    1,
                    (
                        (
                            q_scale_d_r * s_q,
                            q_scale_d_r * s_q * h_r,
                        ),
                        h_r * h_k * s_q * q_scale_d_r,
                    ),
                ),
            )
            scale_q = cute.make_tensor(scale_q_iter, q_scale_layout)

            self.q_dtype = q.element_type
            self.scale_q_dtype = scale_q.element_type
            self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()

            qk_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.q_dtype,
                self.q_major_mode,
                self.q_major_mode,
                self.scale_q_dtype,
                self.qk_sf_vec_size,
                tcgen05.CtaGroup.TWO,
                self.qk_mma_tiler[:2],
            )
            cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (qk_tiled_mma.thr_id.shape,),
            )

            q_scale_size_m = self.qk_mma_tiler[0] // cute.size(
                qk_tiled_mma.thr_id.shape
            )
            q_scale_tma_layout = cute.make_composed_layout(
                cute.make_swizzle(0, 4, 3),
                0,
                cute.make_layout((q_scale_size_m * q_scale_d_r,)),
            )
            q_scale_tma_layout_staged = cute.append(
                q_scale_tma_layout,
                cute.make_layout(
                    (self.scale_q_stage),
                    stride=(cute.cosize(q_scale_tma_layout.outer)),
                ),
            )
            self.scale_q_tiler = (q_scale_size_m * q_scale_d_r,)
            tma_load_q_scale_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(
                tcgen05.CtaGroup.ONE
            )
            tma_atom_scale_q, tma_tensor_scale_q = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_q_scale_op,
                scale_q,
                q_scale_tma_layout,
                self.scale_q_tiler,
            )
            self.tma_copy_scale_q_bytes = cute.size_in_bytes(
                self.scale_q_dtype, q_scale_tma_layout
            )

            @cute.struct
            class SharedStorage:
                scale_q_mbar_ptr: cute.struct.MemRange[
                    cute.Int64, self.scale_q_stage * 2
                ]

            self.shared_storage = SharedStorage
            self.kernel(
                qk_tiled_mma,
                tma_atom_scale_q,
                tma_tensor_scale_q,
                q_scale_tma_layout_staged,
            ).launch(
                grid=self.cluster_shape_mnk,
                block=[self.threads_per_cta, 1, 1],
                cluster=self.cluster_shape_mnk,
                stream=stream,
                min_blocks_per_mp=1,
            )

        @cute.kernel
        def kernel(
            self,
            qk_tiled_mma: cute.TiledMma,
            tma_atom_scale_q: cute.CopyAtom,
            mScaleQ_qdl: cute.Tensor,
            q_scale_tma_layout_staged: cute.ComposedLayout,
        ):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (qk_tiled_mma.thr_id.shape,),
            )
            block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
                cta_rank_in_cluster
            )

            smem = utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            scale_q_producer, scale_q_consumer = pipeline.PipelineTmaAsync.create(
                num_stages=self.scale_q_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.load_warp_id])
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.threads_per_warp
                ),
                tx_count=self.tma_copy_scale_q_bytes,
                barrier_storage=storage.scale_q_mbar_ptr.data_ptr(),
                tidx=0,
                defer_sync=True,
            ).make_participants()

            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
            pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

            sScaleQ = smem.allocate_tensor(
                element_type=self.scale_q_dtype,
                layout=q_scale_tma_layout_staged.outer,
                swizzle=q_scale_tma_layout_staged.inner,
                byte_alignment=128,
            )
            q_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            gScaleQ_qdl = cute.flat_divide(mScaleQ_qdl, self.scale_q_tiler)
            tQsScaleQ, tQgScaleQ_qdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_scale_q,
                block_in_cluster_coord_vmnk[2],
                q_cta_layout,
                sScaleQ,
                gScaleQ_qdl,
            )

            if warp_idx == self.load_warp_id:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_scale_q)
                tQgScaleQ = tQgScaleQ_qdl[None, 0, 0]
                scale_q_handle = scale_q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_scale_q,
                    tQgScaleQ[None],
                    tQsScaleQ[None, scale_q_handle.index],
                    tma_bar_ptr=scale_q_handle.barrier,
                )

            if warp_idx == self.consumer_warp_id:
                scale_q_full = scale_q_consumer.wait_and_advance()
                scale_q_full.release()
            return

    return Mxfp4QScaleLoadSmoke


def build_mxfp4_q_scale_load_smoke_kernel(
    geometry: Mxfp4D192PortGeometry | None = None,
) -> Any:
    cls = build_mxfp4_q_scale_load_smoke_kernel_class()
    return cls(geometry=geometry)


def run_mxfp4_q_scale_load_smoke(
    *,
    batch_size: int = 1,
    seqlen_q: int = 128,
    heads_q: int = 12,
    heads_k: int = 12,
    device: str = "cuda:0",
    geometry: Mxfp4D192PortGeometry | None = None,
) -> dict[str, Any]:
    geometry = geometry or default_d192_port_geometry()

    import cutlass  # type: ignore
    import cutlass.cute as cute  # type: ignore
    import cutlass.torch as cutlass_torch  # type: ignore
    import torch
    from cutlass.cute.runtime import make_ptr  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Q-scale load smoke")
    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)

    q_numel = batch_size * heads_q * seqlen_q * geometry.qk_head_dim_padded
    q_scale_numel = (
        batch_size
        * heads_q
        * seqlen_q
        * (geometry.qk_head_dim_padded // geometry.qk_sf_vec_size)
    )
    q_torch = torch.empty((q_numel,), device=device, dtype=torch.float4_e2m1fn_x2)
    scale_q_torch = torch.empty(
        (q_scale_numel,), device=device, dtype=torch.float8_e8m0fnu
    )

    q_ptr = make_ptr(
        cutlass.Float4E2M1FN,
        q_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_q_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        scale_q_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )

    kernel = build_mxfp4_q_scale_load_smoke_kernel(geometry=geometry)
    current_stream = cutlass_torch.default_stream()
    problem_size = (
        batch_size,
        seqlen_q,
        heads_q,
        heads_k,
        geometry.qk_head_dim,
    )

    compiled = cute.compile(
        kernel,
        q_ptr,
        scale_q_ptr,
        problem_size,
        current_stream,
        options="--opt-level 2",
    )
    compiled(q_ptr, scale_q_ptr, problem_size, current_stream)
    torch.cuda.synchronize(device=device)

    return {
        "status": "ok",
        "device": str(device),
        "problem_size": problem_size,
        "q_numel": int(q_torch.numel()),
        "q_scale_numel": int(scale_q_torch.numel()),
    }
