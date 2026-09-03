from __future__ import annotations

from typing import Any

from cute_dsl_mxfp4_forward_d192_port import (
    Mxfp4D192PortGeometry,
    default_d192_port_geometry,
)


def build_mxfp4_qk_mma_tile_smoke_kernel_class() -> type[Any]:
    import cutlass  # type: ignore
    import cutlass.cute as cute  # type: ignore
    import cutlass.cute.nvgpu.tcgen05 as tcgen05  # type: ignore
    import cutlass.pipeline as pipeline  # type: ignore
    import cutlass.utils as utils  # type: ignore
    import cutlass.utils.blackwell_helpers as sm100_utils  # type: ignore
    import cutlass.utils.blockscaled_layout as blockscaled_utils  # type: ignore
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait  # type: ignore

    class Mxfp4QkMmaTileSmoke:
        def __init__(
            self,
            geometry: Mxfp4D192PortGeometry | None = None,
            a_source_tmem: bool = False,
        ):
            geometry = geometry or default_d192_port_geometry()
            self.geometry = geometry
            self.a_source_tmem = a_source_tmem
            self.qk_cta_tiler = geometry.qk_cta_tiler
            self.qk_mma_tiler = (128, 128, geometry.qk_head_dim_padded)
            self.scale_granularity_qk = geometry.qk_scale_granularity
            self.qk_sf_vec_size = geometry.qk_sf_vec_size
            self.acc_dtype = cutlass.Float32
            self.cluster_shape_mn = (1, 1)
            self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
            self.q_stage = self.qk_cta_tiler[2] // self.qk_mma_tiler[2]
            self.acc_stage = 1
            self.load_warp_id = 0
            self.mma_warp_id = 1
            self.epilog_warp_ids = (2, 3, 4, 5)
            self.threads_per_warp = 32
            self.threads_per_cta = self.threads_per_warp * (
                2 + len(self.epilog_warp_ids)
            )
            self.tmem_alloc_barrier = pipeline.NamedBarrier(
                barrier_id=1,
                num_threads=self.threads_per_cta,
            )
            self.q_smem_ready_barrier = pipeline.NamedBarrier(
                barrier_id=2,
                num_threads=self.threads_per_warp * (1 + len(self.epilog_warp_ids)),
            )
            self.q_tmem_transform_barrier = pipeline.NamedBarrier(
                barrier_id=3,
                num_threads=self.threads_per_warp * len(self.epilog_warp_ids),
            )
            self.q_tmem_ready_barrier = pipeline.NamedBarrier(
                barrier_id=4,
                num_threads=self.threads_per_warp * (1 + len(self.epilog_warp_ids)),
            )

        @cute.jit
        def __call__(
            self,
            q_iter: Any,
            k_iter: Any,
            scale_q_iter: Any,
            scale_k_iter: Any,
            scores_iter: Any,
            problem_shape: Any,
            stream: Any,
            store_accumulator: cutlass.Constexpr = False,
        ):
            b, s_q, s_k, h_q, h_k, d_qk = problem_shape
            q_layout = cute.make_layout(
                (s_q, self.geometry.qk_head_dim_padded, h_q * b),
                stride=(
                    self.geometry.qk_head_dim_padded,
                    1,
                    self.geometry.qk_head_dim_padded * s_q,
                ),
            )
            k_layout = cute.make_layout(
                (s_k, self.geometry.qk_head_dim_padded, h_k * b),
                stride=(
                    self.geometry.qk_head_dim_padded,
                    1,
                    self.geometry.qk_head_dim_padded * s_k,
                ),
            )
            q = cute.make_tensor(q_iter, q_layout)
            k = cute.make_tensor(k_iter, k_layout)
            scores_layout = cute.make_ordered_layout(
                (
                    cute.assume(s_q, divby=self.qk_mma_tiler[0]),
                    cute.assume(s_k, divby=self.qk_mma_tiler[1]),
                    h_q * b,
                ),
                order=(0, 1, 2),
            )
            scores = cute.make_tensor(scores_iter, scores_layout)

            scale_q_layout = blockscaled_utils.tile_atom_to_shape_SF(
                q.shape, self.qk_sf_vec_size
            )
            scale_k_layout = blockscaled_utils.tile_atom_to_shape_SF(
                k.shape, self.qk_sf_vec_size
            )
            scale_q = cute.make_tensor(scale_q_iter, scale_q_layout)
            scale_k = cute.make_tensor(scale_k_iter, scale_k_layout)

            self.q_dtype = q.element_type
            self.k_dtype = k.element_type
            self.sf_dtype = scale_q.element_type
            self.c_dtype = scores.element_type
            self.c_layout = utils.LayoutEnum.from_tensor(scores)
            self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
                self.qk_mma_tiler,
                False,
                self.c_layout,
                self.c_dtype,
            )
            self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
            self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
            a_source = tcgen05.OperandSource.SMEM
            if cutlass.const_expr(self.a_source_tmem):
                a_source = tcgen05.OperandSource.TMEM
            qk_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.q_dtype,
                self.q_major_mode,
                self.k_major_mode,
                self.sf_dtype,
                self.qk_sf_vec_size,
                tcgen05.CtaGroup.ONE,
                self.qk_mma_tiler[:2],
                a_source,
            )
            cluster_layout_vmnk = cute.tiled_divide(
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
            scale_q_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
                qk_tiled_mma,
                self.qk_mma_tiler,
                self.qk_sf_vec_size,
                self.q_stage,
            )
            scale_k_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                qk_tiled_mma,
                self.qk_mma_tiler,
                self.qk_sf_vec_size,
                self.q_stage,
            )
            q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
            k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
            scale_q_smem_layout = cute.slice_(
                scale_q_smem_layout_staged, (None, None, None, 0)
            )
            scale_k_smem_layout = cute.slice_(
                scale_k_smem_layout_staged, (None, None, None, 0)
            )

            q_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, qk_tiled_mma.thr_id
            )
            k_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, qk_tiled_mma.thr_id
            )
            scale_q_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, qk_tiled_mma.thr_id
            )
            scale_k_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, qk_tiled_mma.thr_id
            )

            tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
                q_op,
                q,
                q_smem_layout,
                self.qk_mma_tiler,
                qk_tiled_mma,
                cluster_layout_vmnk.shape,
            )
            tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
                k_op,
                k,
                k_smem_layout,
                self.qk_mma_tiler,
                qk_tiled_mma,
                cluster_layout_vmnk.shape,
            )
            tma_atom_scale_q, tma_tensor_scale_q = cute.nvgpu.make_tiled_tma_atom_A(
                scale_q_op,
                scale_q,
                scale_q_smem_layout,
                self.qk_mma_tiler,
                qk_tiled_mma,
                cluster_layout_vmnk.shape,
                internal_type=cute.Int16,
            )
            tma_atom_scale_k, tma_tensor_scale_k = cute.nvgpu.make_tiled_tma_atom_B(
                scale_k_op,
                scale_k,
                scale_k_smem_layout,
                self.qk_mma_tiler,
                qk_tiled_mma,
                cluster_layout_vmnk.shape,
                internal_type=cute.Int16,
            )

            self.tma_copy_q_bytes = cute.size_in_bytes(
                self.q_dtype, q_smem_layout
            ) * cute.size(qk_tiled_mma.thr_id.shape)
            self.tma_copy_k_bytes = cute.size_in_bytes(self.k_dtype, k_smem_layout)
            self.tma_copy_scale_q_bytes = cute.size_in_bytes(
                self.sf_dtype, scale_q_smem_layout
            ) * cute.size(qk_tiled_mma.thr_id.shape)
            self.tma_copy_scale_k_bytes = cute.size_in_bytes(
                self.sf_dtype, scale_k_smem_layout
            )
            sf_atom_mn = 32
            mma_inst_tile_k = 4
            self.num_sfa_tmem_cols = (
                self.qk_mma_tiler[0] // sf_atom_mn
            ) * mma_inst_tile_k
            self.num_sfb_tmem_cols = (
                self.qk_mma_tiler[1] // sf_atom_mn
            ) * mma_inst_tile_k
            self.num_accumulator_tmem_cols = self.qk_mma_tiler[1] * self.acc_stage

            @cute.struct
            class SharedStorage:
                q_mbar_ptr: cute.struct.MemRange[cute.Int64, self.q_stage * 2]
                k_mbar_ptr: cute.struct.MemRange[cute.Int64, self.q_stage * 2]
                scale_q_mbar_ptr: cute.struct.MemRange[cute.Int64, self.q_stage * 2]
                scale_k_mbar_ptr: cute.struct.MemRange[cute.Int64, self.q_stage * 2]
                acc_mbar_ptr: cute.struct.MemRange[cute.Int64, self.acc_stage * 2]
                tmem_holding_buf: cute.Int32

            self.shared_storage = SharedStorage
            qk_grid = (1, 1, h_q * b)
            self.kernel(
                qk_tiled_mma,
                tma_atom_q,
                tma_tensor_q,
                tma_atom_k,
                tma_tensor_k,
                tma_atom_scale_q,
                tma_tensor_scale_q,
                tma_atom_scale_k,
                tma_tensor_scale_k,
                scores,
                self.epi_tile,
                q_smem_layout_staged,
                k_smem_layout_staged,
                scale_q_smem_layout_staged,
                scale_k_smem_layout_staged,
                store_accumulator,
            ).launch(
                grid=qk_grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=self.cluster_shape_mnk,
                stream=stream,
                min_blocks_per_mp=1,
            )

        @cute.jit
        def s2t_copy_and_partition(
            self,
            sSF: cute.Tensor,
            tSF: cute.Tensor,
            copy_dtype: type[cutlass.Numeric],
        ):
            tCsSF_compact = cute.filter_zeros(sSF)
            tCtSF_compact = cute.filter_zeros(tSF)
            copy_atom_s2t = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
                copy_dtype,
            )
            tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
            thr_copy_s2t = tiled_copy_s2t.get_slice(0)
            tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
            tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t, tCsSF_compact_s2t_
            )
            tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
            return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

        @cute.jit
        def epilog_tmem_copy_and_partition(
            self,
            tidx: Any,
            tAcc: cute.Tensor,
            gScores_mnl: cute.Tensor,
            epi_tile: cute.Tile,
        ):
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.qk_mma_tiler,
                self.c_layout,
                self.c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            )
            tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            gScores_epi = cute.flat_divide(
                gScores_mnl[((None, None), 0, 0, None, None, None)],
                epi_tile,
            )
            tTR_gScores = thr_copy_t2r.partition_D(gScores_epi)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_gScores[(None, None, None, 0, 0, 0, 0, 0)].shape,
                self.acc_dtype,
            )
            return tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_gScores

        @cute.jit
        def epilogue_debug_store(
            self,
            tidx: Any,
            tile_coord_mnl: Any,
            tCtAcc: cute.Tensor,
            gScores_mnl: cute.Tensor,
            epi_tile: cute.Tile,
        ):
            tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_gScores = (
                self.epilog_tmem_copy_and_partition(
                    tidx,
                    tCtAcc,
                    gScores_mnl,
                    epi_tile,
                )
            )
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_gScores = tTR_gScores[
                (None, None, None, None, None, *tile_coord_mnl)
            ]
            tTR_gScores = cute.group_modes(tTR_gScores, 3, cute.rank(tTR_gScores))
            tTR_rScores = cute.make_rmem_tensor(
                tTR_gScores[(None, None, None, 0)].shape,
                self.c_dtype,
            )
            simt_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.c_dtype,
            )
            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
            for subtile_idx in cutlass.range(subtile_cnt):
                tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                tTR_rScores.store(tTR_rAcc.load().to(self.c_dtype))
                cute.copy(
                    simt_atom,
                    tTR_rScores,
                    tTR_gScores[(None, None, None, subtile_idx)],
                )

        @cute.kernel
        def kernel(
            self,
            qk_tiled_mma: cute.TiledMma,
            tma_atom_q: cute.CopyAtom,
            mQ_qdl: cute.Tensor,
            tma_atom_k: cute.CopyAtom,
            mK_kdl: cute.Tensor,
            tma_atom_scale_q: cute.CopyAtom,
            mScaleQ_qdl: cute.Tensor,
            tma_atom_scale_k: cute.CopyAtom,
            mScaleK_kdl: cute.Tensor,
            mScores_mnl: cute.Tensor,
            epi_tile: cute.Tile,
            q_smem_layout_staged: cute.ComposedLayout,
            k_smem_layout_staged: cute.ComposedLayout,
            scale_q_smem_layout_staged: cute.Layout,
            scale_k_smem_layout_staged: cute.Layout,
            store_accumulator: cutlass.Constexpr,
        ):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            _, _, l_tile_idx = cute.arch.block_idx()
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
            q_producer, q_consumer = pipeline.PipelineTmaUmma.create(
                num_stages=self.q_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.load_warp_id])
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.mma_warp_id])
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
                    pipeline.Agent.Thread, len([self.mma_warp_id])
                ),
                tx_count=self.tma_copy_k_bytes,
                barrier_storage=storage.k_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            ).make_participants()
            scale_q_producer, scale_q_consumer = pipeline.PipelineTmaUmma.create(
                num_stages=self.q_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.load_warp_id])
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.mma_warp_id])
                ),
                tx_count=self.tma_copy_scale_q_bytes,
                barrier_storage=storage.scale_q_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            ).make_participants()
            scale_k_producer, scale_k_consumer = pipeline.PipelineTmaUmma.create(
                num_stages=self.q_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.load_warp_id])
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.mma_warp_id])
                ),
                tx_count=self.tma_copy_scale_k_bytes,
                barrier_storage=storage.scale_k_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            ).make_participants()
            acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
                num_stages=self.acc_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp * len(self.epilog_warp_ids),
                ),
                barrier_storage=storage.acc_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf,
                barrier_for_retrieve=self.tmem_alloc_barrier,
            )
            tmem.allocate(512)

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
            sScaleQ = smem.allocate_tensor(
                element_type=self.sf_dtype,
                layout=scale_q_smem_layout_staged,
                byte_alignment=128,
            )
            sScaleK = smem.allocate_tensor(
                element_type=self.sf_dtype,
                layout=scale_k_smem_layout_staged,
                byte_alignment=128,
            )

            qk_thr_mma = qk_tiled_mma.get_slice(0)
            q_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            kv_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )
            gQ_qdl = cute.flat_divide(
                mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2])
            )
            gK_kdl = cute.flat_divide(
                mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2])
            )
            gScaleQ_qdl = cute.local_tile(
                mScaleQ_qdl,
                cute.slice_(self.qk_mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            gScaleK_kdl = cute.local_tile(
                mScaleK_kdl,
                cute.slice_(self.qk_mma_tiler, (0, None, None)),
                (None, None, None),
            )
            tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
            tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
            tSgScaleQ_qdl = qk_thr_mma.partition_A(gScaleQ_qdl)
            tSgScaleK_kdl = qk_thr_mma.partition_B(gScaleK_kdl)
            gScores_mnl = cute.local_tile(
                mScores_mnl,
                cute.slice_(self.qk_mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            tCgScores_mnl = qk_thr_mma.partition_C(gScores_mnl)

            tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                block_in_cluster_coord_vmnk[2],
                q_cta_layout,
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(tSgQ_qdl, 0, 3),
            )
            tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_k,
                block_in_cluster_coord_vmnk[1],
                kv_cta_layout,
                cute.group_modes(sK, 0, 3),
                cute.group_modes(tSgK_kdl, 0, 3),
            )
            tQsScaleQ, tQgScaleQ_qdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_scale_q,
                block_in_cluster_coord_vmnk[2],
                q_cta_layout,
                cute.group_modes(sScaleQ, 0, 3),
                cute.group_modes(tSgScaleQ_qdl, 0, 3),
            )
            tKsScaleK, tKgScaleK_kdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_scale_k,
                block_in_cluster_coord_vmnk[1],
                kv_cta_layout,
                cute.group_modes(sScaleK, 0, 3),
                cute.group_modes(tSgScaleK_kdl, 0, 3),
            )
            tQsScaleQ = cute.filter_zeros(tQsScaleQ)
            tQgScaleQ_qdl = cute.filter_zeros(tQgScaleQ_qdl)
            tKsScaleK = cute.filter_zeros(tKsScaleK)
            tKgScaleK_kdl = cute.filter_zeros(tKgScaleK_kdl)

            if cutlass.const_expr(not self.a_source_tmem):
                tCrQ = qk_tiled_mma.make_fragment_A(sQ)
            tCrK = qk_tiled_mma.make_fragment_B(sK)
            acc_shape = qk_tiled_mma.partition_shape_C(self.qk_mma_tiler[:2])
            tCtAcc_fake = qk_tiled_mma.make_fragment_C(acc_shape)

            tmem.wait_for_alloc()
            base_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            a_tmem_cols = 0
            if cutlass.const_expr(self.a_source_tmem):
                a_tmem_cols = self.qk_mma_tiler[2]
                q_tmem_ptr = cute.recast_ptr(base_tmem_ptr, dtype=self.q_dtype)
                tCrQ = cute.make_tensor(
                    q_tmem_ptr,
                    qk_tiled_mma.make_fragment_A(q_smem_layout_staged.outer).layout,
                )
            acc_tmem_ptr = base_tmem_ptr + a_tmem_cols
            tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            scale_q_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtScaleQ_layout = blockscaled_utils.make_tmem_layout_sfa(
                qk_tiled_mma,
                self.qk_mma_tiler,
                self.qk_sf_vec_size,
                cute.slice_(scale_q_smem_layout_staged, (None, None, None, 0)),
            )
            tCtScaleQ = cute.make_tensor(scale_q_tmem_ptr, tCtScaleQ_layout)
            scale_k_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + self.num_accumulator_tmem_cols
                + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtScaleK_layout = blockscaled_utils.make_tmem_layout_sfb(
                qk_tiled_mma,
                self.qk_mma_tiler,
                self.qk_sf_vec_size,
                cute.slice_(scale_k_smem_layout_staged, (None, None, None, 0)),
            )
            tCtScaleK = cute.make_tensor(scale_k_tmem_ptr, tCtScaleK_layout)
            (
                tiled_copy_s2t_scale_q,
                tCsScaleQ_compact_s2t,
                tCtScaleQ_compact_s2t,
            ) = self.s2t_copy_and_partition(sScaleQ, tCtScaleQ, self.sf_dtype)
            (
                tiled_copy_s2t_scale_k,
                tCsScaleK_compact_s2t,
                tCtScaleK_compact_s2t,
            ) = self.s2t_copy_and_partition(sScaleK, tCtScaleK, self.sf_dtype)
            if warp_idx == self.load_warp_id:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_scale_q)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_scale_k)
                q_handle = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tQgQ_qdl[None, 0, None, l_tile_idx][None, 0],
                    tQsQ[None, q_handle.index],
                    tma_bar_ptr=q_handle.barrier,
                )
                k_handle = k_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_k,
                    tKgK_kdl[None, None, None, l_tile_idx][None, 0, 0],
                    tKsK[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                )
                scale_q_handle = scale_q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_scale_q,
                    tQgScaleQ_qdl[None, 0, None, l_tile_idx][None, 0],
                    tQsScaleQ[None, scale_q_handle.index],
                    tma_bar_ptr=scale_q_handle.barrier,
                )
                scale_k_handle = scale_k_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_scale_k,
                    tKgScaleK_kdl[None, None, None, l_tile_idx][None, 0, 0],
                    tKsScaleK[None, scale_k_handle.index],
                    tma_bar_ptr=scale_k_handle.barrier,
                )

            if cutlass.const_expr(self.a_source_tmem):
                if (warp_idx >= self.epilog_warp_ids[0]) and (
                    warp_idx <= self.epilog_warp_ids[-1]
                ):
                    self.q_smem_ready_barrier.arrive_and_wait()
                    q_r2t_local_tidx = (
                        tidx - self.threads_per_warp * self.epilog_warp_ids[0]
                    )
                    copy_atom_r2t_q = cute.make_copy_atom(
                        tcgen05.St32x32bOp(
                            tcgen05.Repetition(8),
                            tcgen05.Unpack.NONE,
                        ),
                        self.q_dtype,
                    )
                    tiled_copy_r2t_q = tcgen05.make_tmem_copy(
                        copy_atom_r2t_q,
                        tCrQ[(None, None, None, 0)],
                    )
                    thr_copy_r2t_q = tiled_copy_r2t_q.get_slice(q_r2t_local_tidx)
                    tCsQ_r2t = thr_copy_r2t_q.partition_S(sQ)
                    tCtQ_r2t = thr_copy_r2t_q.partition_D(tCrQ)
                    tCrQ_r2t = cute.make_rmem_tensor(
                        tCsQ_r2t[(None, None, None, None, 0)].shape,
                        self.q_dtype,
                    )
                    cute.autovec_copy(
                        tCsQ_r2t[(None, None, None, None, 0)],
                        tCrQ_r2t,
                    )
                    cute.copy(
                        tiled_copy_r2t_q,
                        tCrQ_r2t,
                        tCtQ_r2t[(None, None, None, None, 0)],
                    )
                    self.q_tmem_transform_barrier.arrive_and_wait()
                    cute.arch.fence_view_async_tmem_store()
                    self.q_tmem_ready_barrier.arrive_and_wait()

            if warp_idx == self.mma_warp_id:
                acc_producer.acquire_and_advance()
                q_full = q_consumer.wait_and_advance()
                k_full = k_consumer.wait_and_advance()
                scale_q_full = scale_q_consumer.wait_and_advance()
                scale_k_full = scale_k_consumer.wait_and_advance()
                s2t_stage_coord = (None, None, None, None, 0)
                if cutlass.const_expr(self.a_source_tmem):
                    self.q_smem_ready_barrier.arrive_and_wait()
                    self.q_tmem_ready_barrier.arrive_and_wait()
                cute.copy(
                    tiled_copy_s2t_scale_q,
                    tCsScaleQ_compact_s2t[s2t_stage_coord],
                    tCtScaleQ_compact_s2t,
                )
                cute.copy(
                    tiled_copy_s2t_scale_k,
                    tCsScaleK_compact_s2t[s2t_stage_coord],
                    tCtScaleK_compact_s2t,
                )
                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                num_kblocks = cute.size(tCrQ, mode=[2])
                for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                    kblock_coord = (None, None, kblock_idx, 0)
                    q_kblock_coord = kblock_coord
                    sf_kblock_coord = (None, None, kblock_idx)
                    qk_tiled_mma.set(
                        tcgen05.Field.SFA,
                        tCtScaleQ[sf_kblock_coord].iterator,
                    )
                    qk_tiled_mma.set(
                        tcgen05.Field.SFB,
                        tCtScaleK[sf_kblock_coord].iterator,
                    )
                    cute.gemm(
                        qk_tiled_mma,
                        tCtAcc,
                        tCrQ[q_kblock_coord],
                        tCrK[kblock_coord],
                        tCtAcc,
                    )
                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                q_full.release()
                k_full.release()
                scale_q_full.release()
                scale_k_full.release()
                acc_producer.commit()

            if (warp_idx >= self.epilog_warp_ids[0]) and (
                warp_idx <= self.epilog_warp_ids[-1]
            ):
                acc_full = acc_consumer.wait_and_advance()
                if store_accumulator:
                    self.epilogue_debug_store(
                        tidx - self.epilog_warp_ids[0] * self.threads_per_warp,
                        (0, 0, l_tile_idx),
                        tCtAcc,
                        tCgScores_mnl,
                        epi_tile,
                    )
                acc_full.release()

            pipeline.sync(barrier_id=1)
            tmem.free(base_tmem_ptr)
            return

    return Mxfp4QkMmaTileSmoke


def build_mxfp4_qk_mma_tile_smoke_kernel(
    geometry: Mxfp4D192PortGeometry | None = None,
    a_source_tmem: bool = False,
) -> Any:
    cls = build_mxfp4_qk_mma_tile_smoke_kernel_class()
    return cls(geometry=geometry, a_source_tmem=a_source_tmem)


def run_mxfp4_qk_mma_tile_smoke(
    *,
    batch_size: int = 1,
    seqlen_q: int = 128,
    seqlen_k: int = 128,
    heads_q: int = 12,
    heads_k: int = 12,
    device: str = "cuda:0",
    geometry: Mxfp4D192PortGeometry | None = None,
    store_accumulator: bool = False,
    zero_inputs: bool = False,
    constant_ones: bool = False,
    constant_ones_unpadded: bool = False,
    constant_ones_unpadded_packed: bool = False,
    a_source_tmem: bool = False,
) -> dict[str, Any]:
    geometry = geometry or default_d192_port_geometry()
    if heads_q != heads_k:
        raise ValueError(
            "The standalone QK MMA tile smoke currently uses a flat GEMM L mode "
            "and requires heads_q == heads_k; add GQA stride mapping after this "
            "base MMA path is validated."
        )
    selected_raw_modes = sum(
        int(flag)
        for flag in (
            zero_inputs,
            constant_ones,
            constant_ones_unpadded,
            constant_ones_unpadded_packed,
        )
    )
    if selected_raw_modes > 1:
        raise ValueError("Select at most one deterministic raw-input mode")

    import cutlass  # type: ignore
    import cutlass.cute as cute  # type: ignore
    import cutlass.torch as cutlass_torch  # type: ignore
    import torch
    from cutlass.cute.runtime import make_ptr  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the QK MMA tile smoke")
    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)

    q_numel = batch_size * heads_q * seqlen_q * geometry.qk_head_dim_padded
    k_numel = batch_size * heads_k * seqlen_k * geometry.qk_head_dim_padded
    q_scale_numel = (
        batch_size
        * heads_q
        * seqlen_q
        * (geometry.qk_head_dim_padded // geometry.qk_sf_vec_size)
    )
    k_scale_numel = (
        batch_size
        * heads_k
        * seqlen_k
        * (geometry.qk_head_dim_padded // geometry.qk_sf_vec_size)
    )
    scores_numel = (
        batch_size * heads_q * seqlen_q * seqlen_k
        if store_accumulator
        else 1
    )
    raw_input_storage = (
        zero_inputs
        or constant_ones
        or constant_ones_unpadded
        or constant_ones_unpadded_packed
    )
    q_storage_dtype = torch.uint8 if raw_input_storage else torch.float4_e2m1fn_x2
    k_storage_dtype = torch.uint8 if raw_input_storage else torch.float4_e2m1fn_x2
    scale_storage_dtype = torch.uint8 if raw_input_storage else torch.float8_e8m0fnu
    q_torch = torch.empty((q_numel,), device=device, dtype=q_storage_dtype)
    k_torch = torch.empty((k_numel,), device=device, dtype=k_storage_dtype)
    scale_q_torch = torch.empty((q_scale_numel,), device=device, dtype=scale_storage_dtype)
    scale_k_torch = torch.empty((k_scale_numel,), device=device, dtype=scale_storage_dtype)
    scores_torch = torch.empty((scores_numel,), device=device, dtype=torch.float32)
    if zero_inputs:
        q_torch.zero_()
        k_torch.zero_()
        scale_q_torch.zero_()
        scale_k_torch.zero_()
        scores_torch.fill_(float("nan"))
    if constant_ones:
        q_torch.fill_(0x22)
        k_torch.fill_(0x22)
        scale_q_torch.fill_(0x7F)
        scale_k_torch.fill_(0x7F)
        scores_torch.fill_(float("nan"))
    if constant_ones_unpadded:
        q_torch.fill_(0x22)
        k_torch.fill_(0x22)
        scale_q_torch.fill_(0x7F)
        scale_k_torch.fill_(0x7F)
        q_rows = batch_size * heads_q * seqlen_q
        k_rows = batch_size * heads_k * seqlen_k
        qk_bytes_per_row = geometry.qk_head_dim_padded
        qk_active_bytes_per_row = geometry.qk_head_dim
        q_torch.view(q_rows, qk_bytes_per_row)[
            :, qk_active_bytes_per_row:
        ].zero_()
        k_torch.view(k_rows, qk_bytes_per_row)[
            :, qk_active_bytes_per_row:
        ].zero_()
        scores_torch.fill_(float("nan"))
    if constant_ones_unpadded_packed:
        q_torch.zero_()
        k_torch.zero_()
        scale_q_torch.fill_(0x7F)
        scale_k_torch.fill_(0x7F)
        # Float4 pointer strides are logical lanes; the physical row is packed.
        q_rows = batch_size * heads_q * seqlen_q
        k_rows = batch_size * heads_k * seqlen_k
        qk_packed_bytes_per_row = geometry.qk_head_dim_padded // 2
        qk_active_packed_bytes_per_row = geometry.qk_head_dim // 2
        q_packed = q_torch[: q_rows * qk_packed_bytes_per_row].view(
            q_rows,
            qk_packed_bytes_per_row,
        )
        k_packed = k_torch[: k_rows * qk_packed_bytes_per_row].view(
            k_rows,
            qk_packed_bytes_per_row,
        )
        q_packed[:, :qk_active_packed_bytes_per_row].fill_(0x22)
        k_packed[:, :qk_active_packed_bytes_per_row].fill_(0x22)
        scores_torch.fill_(float("nan"))
    if raw_input_storage:
        torch.cuda.synchronize(device=device)

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
    scale_q_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        scale_q_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    scale_k_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        scale_k_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    scores_ptr = make_ptr(
        cutlass.Float32,
        scores_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    kernel = build_mxfp4_qk_mma_tile_smoke_kernel(
        geometry=geometry,
        a_source_tmem=a_source_tmem,
    )
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
        scale_q_ptr,
        scale_k_ptr,
        scores_ptr,
        problem_size,
        current_stream,
        store_accumulator,
        options="--opt-level 2",
    )
    compiled(
        q_ptr,
        k_ptr,
        scale_q_ptr,
        scale_k_ptr,
        scores_ptr,
        problem_size,
        current_stream,
        store_accumulator,
    )
    torch.cuda.synchronize(device=device)

    result = {
        "status": "ok",
        "device": str(device),
        "problem_size": problem_size,
        "store_accumulator": bool(store_accumulator),
        "a_source_tmem": bool(a_source_tmem),
        "zero_inputs": bool(zero_inputs),
        "constant_ones": bool(constant_ones),
        "constant_ones_unpadded": bool(constant_ones_unpadded),
        "constant_ones_unpadded_packed": bool(constant_ones_unpadded_packed),
        "q_numel": int(q_torch.numel()),
        "k_numel": int(k_torch.numel()),
        "q_scale_numel": int(scale_q_torch.numel()),
        "k_scale_numel": int(scale_k_torch.numel()),
        "scores_numel": int(scores_torch.numel()),
    }
    if raw_input_storage:
        result["q_nonzero_bytes"] = int((q_torch != 0).sum().item())
        result["k_nonzero_bytes"] = int((k_torch != 0).sum().item())
        result["scale_q_nonzero_bytes"] = int((scale_q_torch != 0).sum().item())
        result["scale_k_nonzero_bytes"] = int((scale_k_torch != 0).sum().item())
    if store_accumulator:
        finite_scores = torch.nan_to_num(scores_torch, nan=0.0)
        nan_indices = torch.isnan(scores_torch).nonzero().flatten()[:16]
        result["scores_max_abs"] = float(finite_scores.abs().max().item())
        result["scores_min"] = float(finite_scores.min().item())
        result["scores_max"] = float(finite_scores.max().item())
        result["scores_nan_count"] = int(torch.isnan(scores_torch).sum().item())
        result["scores_nan_indices_head"] = [
            int(index) for index in nan_indices.detach().cpu().tolist()
        ]
        if constant_ones or constant_ones_unpadded or constant_ones_unpadded_packed:
            expected_score = float(
                geometry.qk_head_dim
                if constant_ones_unpadded or constant_ones_unpadded_packed
                else geometry.qk_head_dim_padded
            )
            finite_mask = ~torch.isnan(scores_torch)
            result["expected_score"] = expected_score
            result["scores_expected_max_abs_diff"] = float(
                (scores_torch[finite_mask] - expected_score).abs().max().item()
            )
    return result
