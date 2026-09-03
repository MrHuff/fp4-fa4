from __future__ import annotations

from typing import Any

from cute_dsl_mxfp4_forward_d192_port import (
    Mxfp4D192PortGeometry,
    default_d192_port_geometry,
)
def build_mxfp4_qk_load_smoke_kernel_class() -> type[Any]:
    import cutlass  # type: ignore
    import cutlass.cute as cute  # type: ignore
    import cutlass.cute.nvgpu.tcgen05 as tcgen05  # type: ignore
    import cutlass.pipeline as pipeline  # type: ignore
    import cutlass.utils as utils  # type: ignore
    import cutlass.utils.blackwell_helpers as sm100_utils  # type: ignore
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait  # type: ignore

    class Mxfp4QkLoadSmoke:
        def __init__(self, geometry: Mxfp4D192PortGeometry | None = None):
            geometry = geometry or default_d192_port_geometry()
            self.geometry = geometry
            self.qk_cta_tiler = geometry.qk_cta_tiler
            self.qk_mma_tiler = (128, 128, 64)
            self.cluster_shape_mn = (1, 1)
            self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
            self.q_stage = self.qk_cta_tiler[2] // self.qk_mma_tiler[2]
            self.load_warp_id = 0
            self.consumer_warp_id = 1
            self.threads_per_warp = 32
            self.threads_per_cta = self.threads_per_warp * 2

        @cute.jit
        def __call__(
            self,
            q_iter: Any,
            k_iter: Any,
            problem_shape: Any,
            stream: Any,
        ):
            b, s_q, s_k, h_q, h_k, d_qk = problem_shape
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

            q = cute.make_tensor(q_iter, q_layout)
            k = cute.make_tensor(k_iter, k_layout)
            self.q_dtype = q.element_type
            self.k_dtype = k.element_type

            self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
            self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()

            qk_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.q_dtype,
                self.q_major_mode,
                self.k_major_mode,
                cutlass.Float8E8M0FNU,
                self.geometry.qk_sf_vec_size,
                tcgen05.CtaGroup.ONE,
                self.qk_mma_tiler[:2],
            )
            self.cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (qk_tiled_mma.thr_id.shape,),
            )

            q_smem_layout_staged = sm100_utils.make_smem_layout_a(
                qk_tiled_mma,
                self.qk_mma_tiler,
                self.q_dtype,
                self.q_stage,
            )
            k_smem_layout_staged = sm100_utils.make_smem_layout_b(
                qk_tiled_mma,
                self.qk_mma_tiler,
                self.k_dtype,
                self.q_stage,
            )
            q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
            k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])

            q_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, qk_tiled_mma.thr_id
            )
            k_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, qk_tiled_mma.thr_id
            )
            tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
                q_op,
                q,
                q_smem_layout,
                self.qk_mma_tiler,
                qk_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
                k_op,
                k,
                k_smem_layout,
                self.qk_mma_tiler,
                qk_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )

            self.tma_copy_q_bytes = cute.size_in_bytes(
                self.q_dtype, q_smem_layout
            ) * cute.size(qk_tiled_mma.thr_id.shape)
            self.tma_copy_k_bytes = cute.size_in_bytes(self.k_dtype, k_smem_layout)

            @cute.struct
            class SharedStorage:
                q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
                k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]

            self.shared_storage = SharedStorage

            grid = (1, 1, 1)
            self.kernel(
                qk_tiled_mma,
                tma_atom_q,
                tma_tensor_q,
                tma_atom_k,
                tma_tensor_k,
                q_smem_layout_staged,
                k_smem_layout_staged,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=self.cluster_shape_mnk,
                stream=stream,
                min_blocks_per_mp=1,
            )

        @cute.kernel
        def kernel(
            self,
            qk_tiled_mma: cute.TiledMma,
            tma_atom_q: cute.CopyAtom,
            mQ_qdl: cute.Tensor,
            tma_atom_k: cute.CopyAtom,
            mK_kdl: cute.Tensor,
            q_smem_layout_staged: cute.ComposedLayout,
            k_smem_layout_staged: cute.ComposedLayout,
        ):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (qk_tiled_mma.thr_id.shape,),
            )

            smem = utils.SmemAllocator()
            storage = smem.allocate(self.shared_storage)
            q_producer, q_consumer = pipeline.PipelineTmaUmma.create(
                num_stages=self.q_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.load_warp_id])
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.consumer_warp_id])
                ),
                tx_count=self.tma_copy_q_bytes,
                barrier_storage=storage.q_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            ).make_participants()
            k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
                num_stages=self.q_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.load_warp_id])
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.consumer_warp_id])
                ),
                tx_count=self.tma_copy_k_bytes,
                barrier_storage=storage.k_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            ).make_participants()

            pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
            pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

            sQ = smem.allocate_tensor(
                element_type=self.q_dtype,
                layout=q_smem_layout_staged.outer,
                swizzle=q_smem_layout_staged.inner,
                byte_alignment=128,
            )
            sK = smem.allocate_tensor(
                element_type=self.k_dtype,
                layout=k_smem_layout_staged.outer,
                swizzle=k_smem_layout_staged.inner,
                byte_alignment=128,
            )

            qk_thr_mma = qk_tiled_mma.get_slice(0)
            q_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            gQ_qdl = cute.flat_divide(
                mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2])
            )
            tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
            tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                cta_rank_in_cluster,
                q_cta_layout,
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tSgQ_qdl, 0, 3),
            )

            k_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )
            gK_kdl = cute.flat_divide(
                mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2])
            )
            tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
            tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_k,
                cta_rank_in_cluster,
                k_cta_layout,
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK_kdl, 0, 3),
            )

            if warp_idx == self.load_warp_id:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
                tQgQ = tQgQ_qdl[None, 0, None, 0]
                tKgK = tKgK_kdl[None, None, None, 0]

                q_handle = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tQgQ[None, 0],
                    tQsQ[None, q_handle.index],
                    tma_bar_ptr=q_handle.barrier,
                )
                k_handle = k_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_k,
                    tKgK[None, 0, 0],
                    tKsK[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                )

            if warp_idx == self.consumer_warp_id:
                q_full = q_consumer.wait_and_advance()
                q_full.release()
                k_full = k_consumer.wait_and_advance()
                k_full.release()
            return

    return Mxfp4QkLoadSmoke


def build_mxfp4_qk_load_smoke_kernel(
    geometry: Mxfp4D192PortGeometry | None = None,
) -> Any:
    cls = build_mxfp4_qk_load_smoke_kernel_class()
    return cls(geometry=geometry)


def run_mxfp4_qk_load_smoke(
    *,
    batch_size: int = 1,
    seqlen_q: int = 128,
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
        raise RuntimeError("CUDA is required for the QK load smoke")
    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)

    q_numel = batch_size * heads_q * seqlen_q * geometry.qk_head_dim_padded
    k_numel = batch_size * heads_k * seqlen_k * geometry.qk_head_dim_padded
    q_torch = torch.empty((q_numel,), device=device, dtype=torch.float4_e2m1fn_x2)
    k_torch = torch.empty((k_numel,), device=device, dtype=torch.float4_e2m1fn_x2)

    q_ptr = make_ptr(
        cutlass.Float4E2M1FN,
        q_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    k_ptr = make_ptr(
        cutlass.Float4E2M1FN,
        k_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    kernel = build_mxfp4_qk_load_smoke_kernel(geometry=geometry)
    current_stream = cutlass_torch.default_stream()
    problem_size = (
        batch_size,
        seqlen_q,
        seqlen_k,
        heads_q,
        heads_k,
        geometry.qk_head_dim,
    )

    compiled = cute.compile(
        kernel,
        q_ptr,
        k_ptr,
        problem_size,
        current_stream,
        options="--opt-level 2",
    )
    compiled(
        q_ptr,
        k_ptr,
        problem_size,
        current_stream,
    )
    torch.cuda.synchronize(device=device)

    return {
        "status": "ok",
        "device": str(device),
        "problem_size": problem_size,
        "q_numel": int(q_torch.numel()),
        "k_numel": int(k_torch.numel()),
    }
