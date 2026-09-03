from __future__ import annotations

from typing import Any

from cute_dsl_mxfp4_forward_d192_port import (
    Mxfp4D192PortGeometry,
    default_d192_port_geometry,
    d192_port_patch_points,
)
from cute_dsl_mxfp4_forward_scaffold import (
    load_reference_mixed_input_fmha_d256_module,
)


def build_mxfp4_fmha_d192_kernel_class() -> type[Any]:
    mixed_input_mod = load_reference_mixed_input_fmha_d256_module()
    import cutlass  # type: ignore
    import cutlass.cute as cute  # type: ignore
    import cutlass.cute.nvgpu.tcgen05 as tcgen05  # type: ignore
    import cutlass.pipeline as pipeline  # type: ignore
    import cutlass.utils as utils  # type: ignore
    import cutlass.utils.blackwell_helpers as sm100_utils  # type: ignore
    import cutlass.utils.blockscaled_layout as blockscaled_utils  # type: ignore
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait  # type: ignore

    class Mxfp4FusedMultiHeadAttentionD192(
        mixed_input_mod.MixedInputFusedMultiHeadAttentionPrefillD256
    ):
        """Concrete D192 MXFP4 forward port stub.

        This is intentionally not runnable yet. It exists to hold the actual target geometry
        and kernel-local field layout while we replace the dense/int8 path inherited from the
        mixed-input D256 reference with:
        - blockscaled FP4 QK
        - online MXFP4 P quantization
        - blockscaled FP4 PV
        """

        def __init__(
            self,
            geometry: Mxfp4D192PortGeometry | None = None,
            store_qk_accumulator: bool = False,
            store_qk_softmax: bool = False,
            store_qk_mxfp4_quant: bool = False,
            store_qk_mxfp4_scale_debug: bool = False,
            store_qk_mxfp4_payload_debug: bool = False,
            store_mxfp4_pv_accumulator: bool = False,
        ):
            geometry = geometry or default_d192_port_geometry()
            import cutlass  # type: ignore

            self.port_geometry = geometry
            self.store_qk_accumulator_static = store_qk_accumulator
            self.store_qk_softmax_static = store_qk_softmax
            self.store_qk_mxfp4_quant_static = store_qk_mxfp4_quant
            self.store_qk_mxfp4_scale_debug_static = store_qk_mxfp4_scale_debug
            self.store_qk_mxfp4_payload_debug_static = store_qk_mxfp4_payload_debug
            self.store_mxfp4_pv_accumulator_static = store_mxfp4_pv_accumulator
            self.qk_acc_dtype = cutlass.Float32
            self.pv_acc_dtype = cutlass.Float32

            self.cta_tiler = geometry.qk_cta_tiler
            self.qk_cta_tiler = geometry.qk_cta_tiler
            self.qk_mma_tiler = geometry.qk_mma_tiler
            self.qk_producer_mma_tiler = (
                128,
                128,
                geometry.qk_head_dim_padded,
            )
            self.qk_producer_cluster_shape_mn = (1, 1)
            self.pv_cta_tiler = geometry.pv_cta_tiler
            self.pv_mma_tiler = geometry.pv_mma_tiler
            self.pv_producer_mma_tiler = geometry.pv_cta_tiler
            self.pv_block_tiler = geometry.pv_cta_tiler
            self.scale_granularity_qk = geometry.qk_scale_granularity
            self.scale_granularity_v = geometry.v_scale_granularity
            self.qk_sf_vec_size = geometry.qk_sf_vec_size
            self.pv_sf_vec_size = geometry.pv_sf_vec_size
            self.cluster_shape_mn = geometry.cluster_shape_mn
            self.tmem_warp_shape_mn = (4, 1)
            self.is_persistent = geometry.is_persistent
            self.mask_type = mixed_input_mod.fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
            self.transform_warp_ids = (0, 1, 2, 3, 4, 5, 6, 7)
            self.softmax_warp_ids = (8, 9, 10, 11)
            self.correction_warp_ids = (12, 13, 14, 15)
            self.mma_warp_id = 16
            self.load_warp_id = 17
            self.producer_consumer_warp_id = 18
            self.empty_warp_ids = (18, 19)
            self.num_tmem_alloc_cols = 512
            self.tmem_alloc_sync_bar_id = 1
            self.tmem_s_offset = 256
            self.tmem_p_offset = self.tmem_s_offset
            self.tmem_o_offset = 0
            self.num_regs_softmax = 256
            self.num_regs_correction = 112
            self.num_regs_other = 32
            self.num_regs_transform = 40
            self.buffer_align_bytes = 1024
            self.threads_per_warp = 32
            self.threads_per_cta = self.threads_per_warp * len(
                (
                    *self.transform_warp_ids,
                    *self.softmax_warp_ids,
                    *self.correction_warp_ids,
                    self.load_warp_id,
                    self.mma_warp_id,
                    *self.empty_warp_ids,
                )
            )
            self.tmem_alloc_barrier = pipeline.NamedBarrier(
                barrier_id=self.tmem_alloc_sync_bar_id,
                num_threads=self.threads_per_cta,
            )
            self.p_tmem_transform_barrier = pipeline.NamedBarrier(
                barrier_id=2,
                num_threads=self.threads_per_warp * len(self.softmax_warp_ids),
            )
            self.p_tmem_ready_barrier = pipeline.NamedBarrier(
                barrier_id=3,
                num_threads=self.threads_per_warp
                * (1 + len(self.softmax_warp_ids)),
            )

        def _setup_attributes(self):
            self.q_stage = self.qk_cta_tiler[2] // self.qk_producer_mma_tiler[2]
            self.kv_stage = 4
            self.scale_q_stage = self.q_stage
            self.scale_k_stage = self.q_stage
            self.scale_v_stage = self.kv_stage
            self.qk_acc_stage = 2
            self.qk_issue_acc_stage = 1
            self.pv_acc_stage = 1
            self.kv_trans_stage = 2

        def port_patch_points(self) -> tuple[str, ...]:
            return d192_port_patch_points()

        @cute.jit
        def __call__(
            self,
            q_iter: Any,
            k_iter: Any,
            v_iter: Any,
            o_iter: Any,
            scale_q_iter: Any,
            scale_k_iter: Any,
            scale_v_iter: Any,
            scores_iter: Any,
            problem_shape: Any,
            scale_softmax_log2: Any,
            scale_output: Any,
            window_size_left: Any,
            window_size_right: Any,
            stream: Any,
            store_qk_accumulator: cutlass.Constexpr = False,
            store_qk_softmax: cutlass.Constexpr = False,
            store_qk_mxfp4_quant: cutlass.Constexpr = False,
            store_qk_mxfp4_scale_debug: cutlass.Constexpr = False,
            store_qk_mxfp4_payload_debug: cutlass.Constexpr = False,
            store_mxfp4_pv_accumulator: cutlass.Constexpr = False,
        ):
            self._setup_attributes()
            store_qk_accumulator = self.store_qk_accumulator_static
            store_qk_softmax = self.store_qk_softmax_static
            store_qk_mxfp4_quant = self.store_qk_mxfp4_quant_static
            store_qk_mxfp4_scale_debug = self.store_qk_mxfp4_scale_debug_static
            store_qk_mxfp4_payload_debug = self.store_qk_mxfp4_payload_debug_static
            store_mxfp4_pv_accumulator = self.store_mxfp4_pv_accumulator_static
            b, s_q, s_k, h_q, h_k, d_qk, d_v = problem_shape
            h_r = h_q // h_k

            q_layout = cute.make_layout(
                (s_q, self.port_geometry.qk_head_dim_padded, ((h_r, h_k), b)),
                stride=(
                    self.port_geometry.qk_head_dim_padded,
                    1,
                    (
                        (
                            self.port_geometry.qk_head_dim_padded * s_q,
                            self.port_geometry.qk_head_dim_padded * s_q * h_r,
                        ),
                        h_r * h_k * s_q * self.port_geometry.qk_head_dim_padded,
                    ),
                ),
            )
            k_layout = cute.make_layout(
                (s_k, self.port_geometry.qk_head_dim_padded, ((h_r, h_k), b)),
                stride=(
                    self.port_geometry.qk_head_dim_padded,
                    1,
                    (
                        (
                            0,
                            self.port_geometry.qk_head_dim_padded * s_k,
                        ),
                        h_k * s_k * self.port_geometry.qk_head_dim_padded,
                    ),
                ),
            )
            v_layout = cute.make_layout(
                (d_v, s_k, ((h_r, h_k), b)),
                stride=(1, d_v, ((0, d_v * s_k), h_k * s_k * d_v)),
            )
            o_layout = cute.make_layout(
                (s_q, d_v, ((h_r, h_k), b)),
                stride=(
                    cute.assume(d_v, divby=self.port_geometry.v_head_dim),
                    1,
                    (
                        (
                            cute.assume(d_v * s_q, divby=self.port_geometry.v_head_dim),
                            cute.assume(
                                d_v * s_q * h_r,
                                divby=self.port_geometry.v_head_dim,
                            ),
                        ),
                        cute.assume(
                            h_r * h_k * s_q * d_v,
                            divby=self.port_geometry.v_head_dim,
                        ),
                    ),
                ),
            )

            q = cute.make_tensor(q_iter, q_layout)
            k = cute.make_tensor(k_iter, k_layout)
            v = cute.make_tensor(v_iter, v_layout)
            o = cute.make_tensor(o_iter, o_layout)
            scores_layout = cute.make_ordered_layout(
                (
                    cute.assume(s_q, divby=self.qk_producer_mma_tiler[0]),
                    cute.assume(s_k, divby=self.qk_producer_mma_tiler[1]),
                    h_q * b,
                ),
                order=(0, 1, 2),
            )
            scores = cute.make_tensor(scores_iter, scores_layout)

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
            k_scale_layout = cute.make_layout(
                (s_k * q_scale_d_r, ((h_r, h_k), b)),
                stride=(1, ((0, q_scale_d_r * s_k), s_k * q_scale_d_r * h_k)),
            )
            v_scale_layout = blockscaled_utils.tile_atom_to_shape_SF(
                v.shape, self.pv_sf_vec_size
            )
            scale_q = cute.make_tensor(scale_q_iter, q_scale_layout)
            scale_k = cute.make_tensor(scale_k_iter, k_scale_layout)
            scale_v = cute.make_tensor(scale_v_iter, v_scale_layout)

            q_mma_layout = cute.make_layout(
                (s_q, self.port_geometry.qk_head_dim_padded, h_q * b),
                stride=(
                    self.port_geometry.qk_head_dim_padded,
                    1,
                    self.port_geometry.qk_head_dim_padded * s_q,
                ),
            )
            k_mma_layout = cute.make_layout(
                (s_k, self.port_geometry.qk_head_dim_padded, h_k * b),
                stride=(
                    self.port_geometry.qk_head_dim_padded,
                    1,
                    self.port_geometry.qk_head_dim_padded * s_k,
                ),
            )
            q_mma = cute.make_tensor(q_iter, q_mma_layout)
            k_mma = cute.make_tensor(k_iter, k_mma_layout)
            scale_q_mma = cute.make_tensor(
                scale_q_iter,
                blockscaled_utils.tile_atom_to_shape_SF(
                    q_mma.shape, self.qk_sf_vec_size
                ),
            )
            scale_k_mma = cute.make_tensor(
                scale_k_iter,
                blockscaled_utils.tile_atom_to_shape_SF(
                    k_mma.shape, self.qk_sf_vec_size
                ),
            )

            self.q_dtype = q.element_type
            self.k_dtype = k.element_type
            self.v_dtype = v.element_type
            self.o_dtype = o.element_type
            self.scale_q_dtype = scale_q.element_type
            self.scale_k_dtype = scale_k.element_type
            self.scale_v_dtype = scale_v.element_type
            self.sf_dtype = scale_q_mma.element_type
            self.p_dtype = self.q_dtype
            self.c_dtype = scores.element_type
            self.c_layout = utils.LayoutEnum.from_tensor(scores)
            self.qk_epi_tile = sm100_utils.compute_epilogue_tile_shape(
                self.qk_producer_mma_tiler,
                False,
                self.c_layout,
                self.c_dtype,
            )

            qk_producer_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                q_mma.element_type,
                utils.LayoutEnum.from_tensor(q_mma).mma_major_mode(),
                utils.LayoutEnum.from_tensor(k_mma).mma_major_mode(),
                scale_q_mma.element_type,
                self.qk_sf_vec_size,
                tcgen05.CtaGroup.ONE,
                self.qk_producer_mma_tiler[:2],
            )
            pv_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.q_dtype,
                tcgen05.OperandMajorMode.K,
                utils.LayoutEnum.from_tensor(v).mma_major_mode(),
                self.scale_v_dtype,
                self.pv_sf_vec_size,
                tcgen05.CtaGroup.ONE,
                self.pv_producer_mma_tiler[:2],
                tcgen05.OperandSource.TMEM,
            )
            qk_producer_cluster_shape_mnk = (*self.qk_producer_cluster_shape_mn, 1)
            qk_producer_cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout(qk_producer_cluster_shape_mnk),
                (qk_producer_tiled_mma.thr_id.shape,),
            )

            q_producer_smem_layout_staged = sm100_utils.make_smem_layout_a(
                qk_producer_tiled_mma,
                self.qk_producer_mma_tiler,
                self.q_dtype,
                self.q_stage,
            )
            k_producer_smem_layout_staged = sm100_utils.make_smem_layout_b(
                qk_producer_tiled_mma,
                self.qk_producer_mma_tiler,
                self.k_dtype,
                self.q_stage,
            )
            v_producer_smem_layout_staged = sm100_utils.make_smem_layout_b(
                pv_tiled_mma,
                self.pv_producer_mma_tiler,
                self.v_dtype,
                self.kv_stage,
            )
            p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
                pv_tiled_mma,
                self.pv_producer_mma_tiler,
                self.p_dtype,
                self.qk_issue_acc_stage,
            )
            q_producer_smem_layout = cute.select(
                q_producer_smem_layout_staged, mode=[0, 1, 2]
            )
            k_producer_smem_layout = cute.select(
                k_producer_smem_layout_staged, mode=[0, 1, 2]
            )
            v_producer_smem_layout = cute.select(
                v_producer_smem_layout_staged, mode=[0, 1, 2]
            )
            q_producer_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.qk_producer_cluster_shape_mn,
                qk_producer_tiled_mma.thr_id,
            )
            k_producer_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.qk_producer_cluster_shape_mn,
                qk_producer_tiled_mma.thr_id,
            )
            v_producer_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.qk_producer_cluster_shape_mn,
                pv_tiled_mma.thr_id,
            )
            tma_atom_q_producer, tma_tensor_q_producer = (
                cute.nvgpu.make_tiled_tma_atom_A(
                    q_producer_op,
                    q_mma,
                    q_producer_smem_layout,
                    self.qk_producer_mma_tiler,
                    qk_producer_tiled_mma,
                    qk_producer_cluster_layout_vmnk.shape,
                )
            )
            tma_atom_k_producer, tma_tensor_k_producer = (
                cute.nvgpu.make_tiled_tma_atom_B(
                    k_producer_op,
                    k_mma,
                    k_producer_smem_layout,
                    self.qk_producer_mma_tiler,
                    qk_producer_tiled_mma,
                    qk_producer_cluster_layout_vmnk.shape,
                )
            )
            v_mma_layout = cute.make_layout(
                (self.port_geometry.v_head_dim, s_k, h_q * b),
                stride=(
                    1,
                    self.port_geometry.v_head_dim,
                    self.port_geometry.v_head_dim * s_k,
                ),
            )
            v_mma = cute.make_tensor(v_iter, v_mma_layout)
            scale_v_mma = cute.make_tensor(
                scale_v_iter,
                blockscaled_utils.tile_atom_to_shape_SF(
                    v_mma.shape, self.pv_sf_vec_size
                ),
            )
            tma_atom_v_producer, tma_tensor_v_producer = (
                cute.nvgpu.make_tiled_tma_atom_B(
                    v_producer_op,
                    v_mma,
                    v_producer_smem_layout,
                    self.pv_producer_mma_tiler,
                    pv_tiled_mma,
                    qk_producer_cluster_layout_vmnk.shape,
                )
            )

            q_scale_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
                qk_producer_tiled_mma,
                self.qk_producer_mma_tiler,
                self.qk_sf_vec_size,
                self.scale_q_stage,
            )
            k_scale_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
                qk_producer_tiled_mma,
                self.qk_producer_mma_tiler,
                self.qk_sf_vec_size,
                self.scale_k_stage,
            )
            q_scale_smem_layout = cute.slice_(
                q_scale_smem_layout_staged, (None, None, None, 0)
            )
            k_scale_smem_layout = cute.slice_(
                k_scale_smem_layout_staged, (None, None, None, 0)
            )
            q_scale_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.qk_producer_cluster_shape_mn,
                qk_producer_tiled_mma.thr_id,
            )
            k_scale_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
                self.qk_producer_cluster_shape_mn,
                qk_producer_tiled_mma.thr_id,
            )
            tma_atom_scale_q_producer, tma_tensor_scale_q_producer = (
                cute.nvgpu.make_tiled_tma_atom_A(
                    q_scale_op,
                    scale_q_mma,
                    q_scale_smem_layout,
                    self.qk_producer_mma_tiler,
                    qk_producer_tiled_mma,
                    qk_producer_cluster_layout_vmnk.shape,
                    internal_type=cute.Int16,
                )
            )
            tma_atom_scale_k_producer, tma_tensor_scale_k_producer = (
                cute.nvgpu.make_tiled_tma_atom_B(
                    k_scale_op,
                    scale_k_mma,
                    k_scale_smem_layout,
                    self.qk_producer_mma_tiler,
                    qk_producer_tiled_mma,
                    qk_producer_cluster_layout_vmnk.shape,
                    internal_type=cute.Int16,
                )
            )

            self.tma_copy_q_bytes = cute.size_in_bytes(
                self.q_dtype, q_producer_smem_layout
            ) * cute.size(qk_producer_tiled_mma.thr_id.shape)
            self.tma_copy_k_bytes = cute.size_in_bytes(
                self.k_dtype, k_producer_smem_layout
            )
            self.tma_copy_v_bytes = cute.size_in_bytes(
                self.v_dtype, v_producer_smem_layout
            )
            self.tma_copy_scale_q_bytes = cute.size_in_bytes(
                self.sf_dtype, q_scale_smem_layout
            ) * cute.size(qk_producer_tiled_mma.thr_id.shape)
            self.tma_copy_scale_k_bytes = cute.size_in_bytes(
                self.sf_dtype, k_scale_smem_layout
            )
            sf_atom_mn = 32
            mma_inst_tile_k = 4
            self.num_sfa_tmem_cols = (
                self.qk_producer_mma_tiler[0] // sf_atom_mn
            ) * mma_inst_tile_k
            self.num_sfb_tmem_cols = (
                self.qk_producer_mma_tiler[1] // sf_atom_mn
            ) * mma_inst_tile_k
            self.num_accumulator_tmem_cols = (
                self.qk_producer_mma_tiler[1] * self.qk_issue_acc_stage
            )

            @cute.struct
            class SharedStorage:
                q_mbar_ptr: cute.struct.MemRange[cute.Int64, self.q_stage * 2]
                k_mbar_ptr: cute.struct.MemRange[cute.Int64, self.q_stage * 2]
                v_mbar_ptr: cute.struct.MemRange[cute.Int64, self.kv_stage * 2]
                scale_q_mbar_ptr: cute.struct.MemRange[
                    cute.Int64, self.scale_q_stage * 2
                ]
                scale_k_mbar_ptr: cute.struct.MemRange[
                    cute.Int64, self.scale_k_stage * 2
                ]
                acc_mbar_ptr: cute.struct.MemRange[
                    cute.Int64, self.qk_issue_acc_stage * 2
                ]
                tmem_holding_buf: cute.Int32

            self.shared_storage = SharedStorage
            qk_issue_grid = (
                cute.ceil_div(s_q, self.qk_producer_mma_tiler[0]),
                cute.ceil_div(s_k, self.qk_producer_mma_tiler[1]),
                h_q * b,
            )
            self.qk_producer_smoke_kernel(
                qk_producer_tiled_mma,
                pv_tiled_mma,
                tma_atom_q_producer,
                tma_tensor_q_producer,
                tma_atom_k_producer,
                tma_tensor_k_producer,
                tma_atom_v_producer,
                tma_tensor_v_producer,
                tma_atom_scale_q_producer,
                tma_tensor_scale_q_producer,
                tma_atom_scale_k_producer,
                tma_tensor_scale_k_producer,
                scores,
                self.qk_epi_tile,
                q_producer_smem_layout_staged,
                k_producer_smem_layout_staged,
                v_producer_smem_layout_staged,
                p_tmem_layout_staged,
                q_scale_smem_layout_staged,
                k_scale_smem_layout_staged,
                store_qk_accumulator,
                store_qk_softmax,
                store_qk_mxfp4_quant,
                store_qk_mxfp4_scale_debug,
                store_qk_mxfp4_payload_debug,
                store_mxfp4_pv_accumulator,
                scale_softmax_log2,
            ).launch(
                grid=qk_issue_grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=qk_producer_cluster_shape_mnk,
                stream=stream,
                min_blocks_per_mp=1,
            )
            return

            self.tile_sched_params, grid = mixed_input_mod.fmha_utils.compute_grid(
                o.shape,
                self.qk_cta_tiler,
                self.is_persistent,
            )

            self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
            self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
            self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
            self.o_layout = utils.LayoutEnum.from_tensor(o)
            cta_group = tcgen05.CtaGroup.TWO
            p_major_mode = tcgen05.OperandMajorMode.K
            p_source = tcgen05.OperandSource.TMEM

            qk_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.q_dtype,
                self.q_major_mode,
                self.k_major_mode,
                self.scale_q_dtype,
                self.qk_sf_vec_size,
                cta_group,
                self.qk_mma_tiler[:2],
            )
            pv_tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
                self.q_dtype,
                p_major_mode,
                self.v_major_mode,
                self.scale_v_dtype,
                self.pv_sf_vec_size,
                cta_group,
                self.pv_mma_tiler[:2],
                p_source,
            )

            self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
            self.cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout(self.cluster_shape_mnk),
                (qk_tiled_mma.thr_id.shape,),
            )
            self.epi_tile = self.pv_cta_tiler[:2]

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
                self.kv_stage,
            )
            v_smem_layout_staged = sm100_utils.make_smem_layout_b(
                pv_tiled_mma,
                self.pv_mma_tiler,
                self.v_dtype,
                self.kv_stage,
            )
            q_scale_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
                qk_tiled_mma,
                self.qk_cta_tiler,
                self.qk_sf_vec_size,
                self.scale_q_stage,
            )
            (
                k_scale_tma_layout,
                self.scale_k_tiler,
                k_scale_s2r_view_layout,
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
            (
                v_scale_tma_layout,
                self.scale_v_tiler,
                v_scale_s2r_view_layout,
            ) = mixed_input_mod.prefill_utils.get_scale_smem_layout(
                self.scale_granularity_v,
                self.pv_cta_tiler[2] // self.scale_granularity_v,
                self.pv_mma_tiler,
                self.v_major_mode,
            )
            v_scale_smem_layout_staged = cute.append(
                v_scale_tma_layout,
                cute.make_layout(
                    (self.scale_v_stage),
                    stride=(cute.cosize(v_scale_tma_layout.outer)),
                ),
            )
            p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
                pv_tiled_mma,
                self.pv_mma_tiler,
                self.p_dtype,
                self.qk_acc_stage,
            )

            q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
            k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
            v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
            q_scale_smem_layout = cute.slice_(
                q_scale_smem_layout_staged, (None, None, None, 0)
            )
            p_tmem_layout = cute.select(p_tmem_layout_staged, mode=[0, 1, 2])

            q_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, qk_tiled_mma.thr_id
            )
            k_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, qk_tiled_mma.thr_id
            )
            v_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, pv_tiled_mma.thr_id
            )
            tma_atom_q, _ = cute.nvgpu.make_tiled_tma_atom_A(
                q_op,
                q,
                q_smem_layout,
                self.qk_mma_tiler,
                qk_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_k, _ = cute.nvgpu.make_tiled_tma_atom_B(
                k_op,
                k,
                k_smem_layout,
                self.qk_mma_tiler,
                qk_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            tma_atom_v, _ = cute.nvgpu.make_tiled_tma_atom_B(
                v_op,
                v,
                v_smem_layout,
                self.pv_mma_tiler,
                pv_tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
            q_scale_size_m = self.qk_mma_tiler[0] // cute.size(qk_tiled_mma.thr_id.shape)
            q_scale_tma_layout = cute.make_composed_layout(
                cute.make_swizzle(0, 4, 3),
                0,
                cute.make_layout((q_scale_size_m * q_scale_d_r,)),
            )
            tma_load_kv_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(
                tcgen05.CtaGroup.ONE
            )
            tma_load_q_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)
            self.scale_q_tiler = (q_scale_size_m * q_scale_d_r,)
            tma_atom_scale_q, _ = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_q_op,
                scale_q,
                q_scale_tma_layout,
                self.scale_q_tiler,
            )
            tma_atom_scale_k, _ = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_kv_op,
                scale_k,
                k_scale_tma_layout,
                (self.scale_k_tiler[0] // 2,),
            )
            tma_atom_scale_v, _ = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_load_kv_op,
                scale_v,
                v_scale_tma_layout,
                self.scale_v_tiler,
            )

            grid = cute.round_up(grid, self.cluster_shape_mnk)
            self.setup_smoke_kernel(
                tma_atom_q,
                tma_atom_k,
                tma_atom_v,
                tma_atom_scale_q,
                tma_atom_scale_k,
                tma_atom_scale_v,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                cluster=self.cluster_shape_mnk,
                stream=stream,
                min_blocks_per_mp=1,
            )

        @cute.jit
        def s2t_copy_and_partition(self, sSF: cute.Tensor, tSF: cute.Tensor):
            tCsSF_compact = cute.filter_zeros(sSF)
            tCtSF_compact = cute.filter_zeros(tSF)
            copy_atom_s2t = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
                self.sf_dtype,
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
        def qk_epilog_tmem_copy_and_partition(
            self,
            tidx: Any,
            tAcc: cute.Tensor,
            gScores_mnl: cute.Tensor,
            epi_tile: cute.Tile,
        ):
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.qk_producer_mma_tiler,
                self.c_layout,
                self.c_dtype,
                self.qk_acc_dtype,
                epi_tile,
                False,
            )
            tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r,
                tAcc_epi[(None, None, 0, 0)],
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
                self.qk_acc_dtype,
            )
            return tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_gScores

        @cute.jit
        def qk_epilogue_debug_store(
            self,
            tidx: Any,
            tile_coord_mnl: Any,
            tCtAcc: cute.Tensor,
            gScores_mnl: cute.Tensor,
            epi_tile: cute.Tile,
        ):
            tiled_copy_t2r, tTR_tAcc, tTR_rAcc, tTR_gScores = (
                self.qk_epilog_tmem_copy_and_partition(
                    tidx,
                    tCtAcc,
                    gScores_mnl,
                    epi_tile,
                )
            )
            tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
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

        @cute.jit
        def positive_e2m1_dequant(self, value: Any):
            quantized = 0.0
            if value >= 0.25:
                quantized = 0.5
            if value >= 0.75:
                quantized = 1.0
            if value >= 1.25:
                quantized = 1.5
            if value >= 1.75:
                quantized = 2.0
            if value >= 2.5:
                quantized = 3.0
            if value >= 3.5:
                quantized = 4.0
            if value >= 5.0:
                quantized = 6.0
            return quantized

        @cute.jit
        def positive_e8m0_rte_scale(self, value: Any):
            scale = 1.0
            if value <= 1.0e-38:
                scale = 1.0
            threshold = 0.7071067811865476
            candidate = 0.5
            for _ in range(32):
                if value < threshold:
                    scale = candidate
                threshold = threshold * 0.5
                candidate = candidate * 0.5
            return scale

        @cute.jit
        def qk_softmax_p_debug_store(
            self,
            tidx: Any,
            tile_coord_mnl: Any,
            tCtAcc: cute.Tensor,
            tScS: cute.Tensor,
            gScores_mnl: cute.Tensor,
            epi_tile: cute.Tile,
            scale_softmax_log2: Any,
            quantize_mxfp4: cutlass.Constexpr = False,
            debug_mxfp4_scale: cutlass.Constexpr = False,
            debug_mxfp4_payload: cutlass.Constexpr = False,
        ):
            tStS_slice = tCtAcc[(None, None), 0, 0]
            tScS_slice = tScS[(None, None), 0, 0]
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(32)),
                self.qk_acc_dtype,
            )
            tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
            thr_load = tmem_tiled_load.get_slice(tidx)
            tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
            tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
            tTMEM_LOADrS = cute.make_rmem_tensor(
                tTMEM_LOADcS.shape,
                self.qk_acc_dtype,
            )
            cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()

            row_max = -cutlass.Float32.inf
            row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
            row_max_safe = row_max
            if row_max == -cutlass.Float32.inf:
                row_max_safe = 0.0
            minus_row_max_scale = (0.0 - row_max_safe) * scale_softmax_log2
            for k in cutlass.range(cute.size(tTMEM_LOADrS), vectorize=True):
                tTMEM_LOADrS[k] = (
                    tTMEM_LOADrS[k] * scale_softmax_log2 + minus_row_max_scale
                )
                tTMEM_LOADrS[k] = cute.math.exp2(tTMEM_LOADrS[k], fastmath=True)
            if quantize_mxfp4 or debug_mxfp4_scale or debug_mxfp4_payload:
                frg_cnt = 4
                frg_tile = cute.size(tTMEM_LOADrS) // frg_cnt
                tTMEM_LOADrS_frg = cute.logical_divide(
                    tTMEM_LOADrS, cute.make_layout(frg_tile)
                )
                for j in range(frg_cnt):
                    p_amax = 0.0
                    p_amax = tTMEM_LOADrS_frg[None, j].load().reduce(
                        cute.ReductionOp.MAX, p_amax, 0
                    )
                    p_scale = self.positive_e8m0_rte_scale(p_amax)
                    if debug_mxfp4_scale:
                        for k in range(cute.size(tTMEM_LOADrS_frg, mode=[0])):
                            tTMEM_LOADrS_frg[k, j] = p_scale
                    else:
                        for k in range(cute.size(tTMEM_LOADrS_frg, mode=[0])):
                            payload = tTMEM_LOADrS_frg[k, j] * 6.0 / p_scale
                            payload_quant = self.positive_e2m1_dequant(payload)
                            if debug_mxfp4_payload:
                                tTMEM_LOADrS_frg[k, j] = payload_quant
                            else:
                                tTMEM_LOADrS_frg[k, j] = (
                                    payload_quant * p_scale * (1.0 / 6.0)
                                )

            tmem_store_atom = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(32)),
                self.qk_acc_dtype,
            )
            tmem_tiled_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_slice)
            thr_store = tmem_tiled_store.get_slice(tidx)
            tTMEM_STOREcS = thr_store.partition_S(tScS_slice)
            tTMEM_STOREtP = thr_store.partition_D(tStS_slice)
            tTMEM_STORErP = cute.make_rmem_tensor(
                tTMEM_STOREcS.shape,
                self.qk_acc_dtype,
            )
            for k in cutlass.range(cute.size(tTMEM_STORErP), vectorize=True):
                tTMEM_STORErP[k] = tTMEM_LOADrS[k]
            cute.copy(tmem_tiled_store, tTMEM_STORErP, tTMEM_STOREtP)
            cute.arch.fence_view_async_tmem_store()

            self.qk_epilogue_debug_store(
                tidx,
                tile_coord_mnl,
                tCtAcc,
                gScores_mnl,
                epi_tile,
            )

        @cute.jit
        def qk_softmax_p_to_tmem_payload(
            self,
            tidx: Any,
            tCtAcc: cute.Tensor,
            tScS: cute.Tensor,
            sP: cute.Tensor,
            tOrP: cute.Tensor,
            scale_softmax_log2: Any,
        ):
            tStS_slice = tCtAcc[(None, None), 0, 0]
            tScS_slice = tScS[(None, None), 0, 0]
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition(32)),
                self.qk_acc_dtype,
            )
            tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
            thr_load = tmem_tiled_load.get_slice(tidx)
            tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
            tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
            tTMEM_LOADrS = cute.make_rmem_tensor(
                tTMEM_LOADcS.shape,
                self.qk_acc_dtype,
            )
            cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)
            cute.arch.fence_view_async_tmem_load()

            row_max = -cutlass.Float32.inf
            row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
            row_max_safe = row_max
            if row_max == -cutlass.Float32.inf:
                row_max_safe = 0.0
            minus_row_max_scale = (0.0 - row_max_safe) * scale_softmax_log2
            for k in cutlass.range(cute.size(tTMEM_LOADrS), vectorize=True):
                tTMEM_LOADrS[k] = (
                    tTMEM_LOADrS[k] * scale_softmax_log2 + minus_row_max_scale
                )
                tTMEM_LOADrS[k] = cute.math.exp2(tTMEM_LOADrS[k], fastmath=True)

            tmem_store_atom = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(8)),
                self.p_dtype,
            )
            tmem_tiled_store = tcgen05.make_tmem_copy(
                tmem_store_atom,
                tOrP[(None, None, None, 0)],
            )
            thr_store = tmem_tiled_store.get_slice(tidx)
            tTMEM_STOREcP = thr_store.partition_S(sP)
            tTMEM_STOREtP = thr_store.partition_D(tOrP)
            tTMEM_STORErP = cute.make_rmem_tensor(
                tTMEM_STOREcP[(None, None, None, None, 0)].shape,
                self.p_dtype,
            )
            for k in cutlass.range(cute.size(tTMEM_LOADrS), vectorize=True):
                tTMEM_LOADrS[k] = tTMEM_LOADrS[k] * 6.0
            s_vec = tTMEM_LOADrS.load()
            tTMEM_STORErP.store(s_vec.to(self.p_dtype))
            cute.copy(
                tmem_tiled_store,
                tTMEM_STORErP,
                tTMEM_STOREtP[(None, None, None, None, 0)],
            )

        @cute.kernel
        def qk_producer_smoke_kernel(
            self,
            qk_tiled_mma: cute.TiledMma,
            pv_tiled_mma: cute.TiledMma,
            tma_atom_q: cute.CopyAtom,
            mQ_qdl: cute.Tensor,
            tma_atom_k: cute.CopyAtom,
            mK_kdl: cute.Tensor,
            tma_atom_v: cute.CopyAtom,
            mV_kdl: cute.Tensor,
            tma_atom_scale_q: cute.CopyAtom,
            mScaleQ_qdl: cute.Tensor,
            tma_atom_scale_k: cute.CopyAtom,
            mScaleK_kdl: cute.Tensor,
            mScores_mnl: cute.Tensor,
            epi_tile: cute.Tile,
            q_smem_layout_staged: cute.ComposedLayout,
            k_smem_layout_staged: cute.ComposedLayout,
            v_smem_layout_staged: cute.ComposedLayout,
            p_tmem_layout_staged: cute.ComposedLayout,
            q_scale_smem_layout_staged: cute.Layout,
            k_scale_smem_layout_staged: cute.Layout,
            store_qk_accumulator: cutlass.Constexpr,
            store_qk_softmax: cutlass.Constexpr,
            store_qk_mxfp4_quant: cutlass.Constexpr,
            store_qk_mxfp4_scale_debug: cutlass.Constexpr,
            store_qk_mxfp4_payload_debug: cutlass.Constexpr,
            store_mxfp4_pv_accumulator: cutlass.Constexpr,
            scale_softmax_log2: Any,
        ):
            store_qk_accumulator = self.store_qk_accumulator_static
            store_qk_softmax = self.store_qk_softmax_static
            store_qk_mxfp4_quant = self.store_qk_mxfp4_quant_static
            store_qk_mxfp4_scale_debug = self.store_qk_mxfp4_scale_debug_static
            store_qk_mxfp4_payload_debug = self.store_qk_mxfp4_payload_debug_static
            store_mxfp4_pv_accumulator = self.store_mxfp4_pv_accumulator_static
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            q_tile_idx, k_tile_idx, l_tile_idx = cute.arch.block_idx()
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            cluster_shape_mnk = (*self.qk_producer_cluster_shape_mn, 1)
            cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout(cluster_shape_mnk),
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
            v_producer, v_consumer = pipeline.PipelineTmaUmma.create(
                num_stages=self.kv_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.load_warp_id])
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len([self.mma_warp_id])
                ),
                tx_count=self.tma_copy_v_bytes,
                barrier_storage=storage.v_mbar_ptr.data_ptr(),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            ).make_participants()
            scale_q_producer, scale_q_consumer = pipeline.PipelineTmaUmma.create(
                num_stages=self.scale_q_stage,
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
                num_stages=self.scale_k_stage,
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
                num_stages=self.qk_issue_acc_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.threads_per_warp * len(self.softmax_warp_ids),
                ),
                barrier_storage=storage.acc_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
            tmem = utils.TmemAllocator(
                storage.tmem_holding_buf,
                barrier_for_retrieve=self.tmem_alloc_barrier,
            )
            tmem.allocate(512)

            pipeline_init_wait(cluster_shape_mn=self.qk_producer_cluster_shape_mn)
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
            sV = smem.allocate_tensor(
                element_type=self.v_dtype,
                layout=v_smem_layout_staged.outer,
                swizzle=v_smem_layout_staged.inner,
                byte_alignment=128,
            )
            sP = smem.allocate_tensor(
                element_type=self.p_dtype,
                layout=p_tmem_layout_staged.outer,
                swizzle=p_tmem_layout_staged.inner,
                byte_alignment=128,
            )
            sScaleQ = smem.allocate_tensor(
                element_type=self.sf_dtype,
                layout=q_scale_smem_layout_staged,
                byte_alignment=128,
            )
            sScaleK = smem.allocate_tensor(
                element_type=self.sf_dtype,
                layout=k_scale_smem_layout_staged,
                byte_alignment=128,
            )

            qk_thr_mma = qk_tiled_mma.get_slice(0)
            pv_thr_mma = pv_tiled_mma.get_slice(0)
            q_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            kv_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )
            gQ_qdl = cute.flat_divide(
                mQ_qdl, cute.select(self.qk_producer_mma_tiler, mode=[0, 2])
            )
            gK_kdl = cute.flat_divide(
                mK_kdl, cute.select(self.qk_producer_mma_tiler, mode=[1, 2])
            )
            gV_kdl = cute.flat_divide(
                mV_kdl, cute.select(self.pv_producer_mma_tiler, mode=[1, 2])
            )
            gScaleQ_qdl = cute.local_tile(
                mScaleQ_qdl,
                cute.slice_(self.qk_producer_mma_tiler, (None, 0, None)),
                (None, None, None),
            )
            gScaleK_kdl = cute.local_tile(
                mScaleK_kdl,
                cute.slice_(self.qk_producer_mma_tiler, (0, None, None)),
                (None, None, None),
            )
            tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
            tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
            tOgV_kdl = pv_thr_mma.partition_B(gV_kdl)
            tSgScaleQ_qdl = qk_thr_mma.partition_A(gScaleQ_qdl)
            tSgScaleK_kdl = qk_thr_mma.partition_B(gScaleK_kdl)
            gScores_mnl = cute.local_tile(
                mScores_mnl,
                cute.slice_(self.qk_producer_mma_tiler, (None, None, 0)),
                (None, None, None),
            )
            tCgScores_mnl = qk_thr_mma.partition_C(gScores_mnl)
            cS = cute.make_identity_tensor(
                (
                    self.qk_producer_mma_tiler[0],
                    self.qk_producer_mma_tiler[1],
                )
            )
            tScS = qk_thr_mma.partition_C(cS)

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
            tVsV, tVgV_kdl = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v,
                block_in_cluster_coord_vmnk[1],
                kv_cta_layout,
                cute.group_modes(sV, 0, 3),
                cute.group_modes(tOgV_kdl, 0, 3),
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

            tCrQ = qk_tiled_mma.make_fragment_A(sQ)
            tCrK = qk_tiled_mma.make_fragment_B(sK)
            tOrV = pv_tiled_mma.make_fragment_B(sV)
            acc_shape = qk_tiled_mma.partition_shape_C(
                self.qk_producer_mma_tiler[:2]
            )
            tCtAcc_fake = qk_tiled_mma.make_fragment_C(acc_shape)
            pv_acc_shape = pv_tiled_mma.partition_shape_C(
                self.pv_producer_mma_tiler[:2]
            )
            tOtAcc_fake = pv_tiled_mma.make_fragment_C(pv_acc_shape)

            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            tOtAcc = cute.make_tensor(acc_tmem_ptr, tOtAcc_fake.layout)
            p_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + 256, dtype=self.p_dtype)
            tOrP = cute.make_tensor(
                p_tmem_ptr,
                pv_tiled_mma.make_fragment_A(p_tmem_layout_staged.outer).layout,
            )
            scale_q_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtScaleQ_layout = blockscaled_utils.make_tmem_layout_sfa(
                qk_tiled_mma,
                self.qk_producer_mma_tiler,
                self.qk_sf_vec_size,
                cute.slice_(q_scale_smem_layout_staged, (None, None, None, 0)),
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
                self.qk_producer_mma_tiler,
                self.qk_sf_vec_size,
                cute.slice_(k_scale_smem_layout_staged, (None, None, None, 0)),
            )
            tCtScaleK = cute.make_tensor(scale_k_tmem_ptr, tCtScaleK_layout)
            (
                tiled_copy_s2t_scale_q,
                tCsScaleQ_compact_s2t,
                tCtScaleQ_compact_s2t,
            ) = self.s2t_copy_and_partition(sScaleQ, tCtScaleQ)
            (
                tiled_copy_s2t_scale_k,
                tCsScaleK_compact_s2t,
                tCtScaleK_compact_s2t,
            ) = self.s2t_copy_and_partition(sScaleK, tCtScaleK)

            if warp_idx == self.load_warp_id:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_scale_q)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_scale_k)
                q_handle = q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tQgQ_qdl[None, q_tile_idx, None, l_tile_idx][None, 0],
                    tQsQ[None, q_handle.index],
                    tma_bar_ptr=q_handle.barrier,
                )
                k_handle = k_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_k,
                    tKgK_kdl[None, None, None, l_tile_idx][None, k_tile_idx, 0],
                    tKsK[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                )
                if cutlass.const_expr(store_mxfp4_pv_accumulator):
                    v_handle = v_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_v,
                        tVgV_kdl[None, None, None, l_tile_idx][None, k_tile_idx, 0],
                        tVsV[None, v_handle.index],
                        tma_bar_ptr=v_handle.barrier,
                    )
                scale_q_handle = scale_q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_scale_q,
                    tQgScaleQ_qdl[None, q_tile_idx, None, l_tile_idx][None, 0],
                    tQsScaleQ[None, scale_q_handle.index],
                    tma_bar_ptr=scale_q_handle.barrier,
                )
                scale_k_handle = scale_k_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_scale_k,
                    tKgScaleK_kdl[None, None, None, l_tile_idx][None, k_tile_idx, 0],
                    tKsScaleK[None, scale_k_handle.index],
                    tma_bar_ptr=scale_k_handle.barrier,
                )

            if warp_idx == self.mma_warp_id:
                acc_producer.acquire_and_advance()
                q_full = q_consumer.wait_and_advance()
                k_full = k_consumer.wait_and_advance()
                scale_q_full = scale_q_consumer.wait_and_advance()
                scale_k_full = scale_k_consumer.wait_and_advance()
                s2t_stage_coord = (None, None, None, None, 0)
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
                        tCrQ[kblock_coord],
                        tCrK[kblock_coord],
                        tCtAcc,
                    )
                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                q_full.release()
                k_full.release()
                scale_q_full.release()
                scale_k_full.release()
                acc_producer.commit()
                if cutlass.const_expr(store_mxfp4_pv_accumulator):
                    v_full = v_consumer.wait_and_advance()
                    self.p_tmem_ready_barrier.arrive_and_wait()
                    acc_producer.acquire_and_advance()
                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    num_pv_kblocks = cute.size(tOrP, mode=[2])
                    for kblock_idx in cutlass.range(
                        num_pv_kblocks,
                        unroll_full=True,
                    ):
                        pv_kblock_coord = (None, None, kblock_idx, 0)
                        sf_kblock_coord = (None, None, kblock_idx)
                        pv_tiled_mma.set(
                            tcgen05.Field.SFA,
                            tCtScaleQ[sf_kblock_coord].iterator,
                        )
                        pv_tiled_mma.set(
                            tcgen05.Field.SFB,
                            tCtScaleK[sf_kblock_coord].iterator,
                        )
                        cute.gemm(
                            pv_tiled_mma,
                            tOtAcc,
                            tOrP[pv_kblock_coord],
                            tOrV[pv_kblock_coord],
                            tOtAcc,
                        )
                        pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    v_full.release()
                    acc_producer.commit()

            if (warp_idx >= self.softmax_warp_ids[0]) and (
                warp_idx <= self.softmax_warp_ids[-1]
            ):
                acc_full = acc_consumer.wait_and_advance()
                if cutlass.const_expr(store_mxfp4_pv_accumulator):
                    self.qk_softmax_p_to_tmem_payload(
                        tidx - self.softmax_warp_ids[0] * self.threads_per_warp,
                        tCtAcc,
                        tScS,
                        sP,
                        tOrP,
                        scale_softmax_log2,
                    )
                    self.p_tmem_transform_barrier.arrive_and_wait()
                    cute.arch.fence_view_async_tmem_store()
                    self.p_tmem_ready_barrier.arrive_and_wait()
                    acc_full.release()
                    pv_full = acc_consumer.wait_and_advance()
                    self.qk_epilogue_debug_store(
                        tidx - self.softmax_warp_ids[0] * self.threads_per_warp,
                        (q_tile_idx, k_tile_idx, l_tile_idx),
                        tOtAcc,
                        tCgScores_mnl,
                        epi_tile,
                    )
                    pv_full.release()
                elif cutlass.const_expr(
                    store_qk_softmax
                    or store_qk_mxfp4_quant
                    or store_qk_mxfp4_scale_debug
                    or store_qk_mxfp4_payload_debug
                ):
                    self.qk_softmax_p_debug_store(
                        tidx - self.softmax_warp_ids[0] * self.threads_per_warp,
                        (q_tile_idx, k_tile_idx, l_tile_idx),
                        tCtAcc,
                        tScS,
                        tCgScores_mnl,
                        epi_tile,
                        scale_softmax_log2,
                        store_qk_mxfp4_quant,
                        store_qk_mxfp4_scale_debug,
                        store_qk_mxfp4_payload_debug,
                    )
                    acc_full.release()
                elif cutlass.const_expr(store_qk_accumulator):
                    self.qk_epilogue_debug_store(
                        tidx - self.softmax_warp_ids[0] * self.threads_per_warp,
                        (q_tile_idx, k_tile_idx, l_tile_idx),
                        tCtAcc,
                        tCgScores_mnl,
                        epi_tile,
                    )
                    acc_full.release()
                else:
                    acc_full.release()

            pipeline.sync(barrier_id=1)
            tmem.free(acc_tmem_ptr)
            return

        @cute.kernel
        def setup_smoke_kernel(
            self,
            tma_atom_q: cute.CopyAtom,
            tma_atom_k: cute.CopyAtom,
            tma_atom_v: cute.CopyAtom,
            tma_atom_scale_q: cute.CopyAtom,
            tma_atom_scale_k: cute.CopyAtom,
            tma_atom_scale_v: cute.CopyAtom,
        ):
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == self.load_warp_id:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_scale_q)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_scale_k)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_scale_v)
            return

    return Mxfp4FusedMultiHeadAttentionD192


def build_mxfp4_fmha_d192_kernel(
    geometry: Mxfp4D192PortGeometry | None = None,
    store_qk_accumulator: bool = False,
    store_qk_softmax: bool = False,
    store_qk_mxfp4_quant: bool = False,
    store_qk_mxfp4_scale_debug: bool = False,
    store_qk_mxfp4_payload_debug: bool = False,
    store_mxfp4_pv_accumulator: bool = False,
) -> Any:
    cls = build_mxfp4_fmha_d192_kernel_class()
    return cls(
        geometry=geometry,
        store_qk_accumulator=store_qk_accumulator,
        store_qk_softmax=store_qk_softmax,
        store_qk_mxfp4_quant=store_qk_mxfp4_quant,
        store_qk_mxfp4_scale_debug=store_qk_mxfp4_scale_debug,
        store_qk_mxfp4_payload_debug=store_qk_mxfp4_payload_debug,
        store_mxfp4_pv_accumulator=store_mxfp4_pv_accumulator,
    )


def run_mxfp4_fmha_d192_setup_smoke(
    *,
    batch_size: int = 1,
    seqlen_q: int = 128,
    seqlen_k: int = 128,
    heads_q: int = 12,
    heads_k: int = 12,
    device: str = "cuda:0",
    geometry: Mxfp4D192PortGeometry | None = None,
    warmup_iterations: int = 0,
    iterations: int = 1,
    store_qk_accumulator: bool = False,
    store_qk_softmax: bool = False,
    store_qk_mxfp4_quant: bool = False,
    store_qk_mxfp4_scale_debug: bool = False,
    store_qk_mxfp4_payload_debug: bool = False,
    store_mxfp4_pv_accumulator: bool = False,
    scale_softmax_log2_value: float = 1.0,
    zero_inputs: bool = False,
    constant_ones: bool = False,
    constant_ones_unpadded_packed: bool = False,
    q_ones_k_half_zero_packed: bool = False,
    q_ones_k_alternating_zero_packed: bool = False,
) -> dict[str, Any]:
    geometry = geometry or default_d192_port_geometry()

    def mxfp4_e8m0_rte_scale(value: float) -> float:
        if value <= 1.0e-38:
            return 1.0
        scale = 1.0
        threshold = 0.7071067811865476
        candidate = 0.5
        for _ in range(32):
            if value < threshold:
                scale = candidate
            threshold *= 0.5
            candidate *= 0.5
        return scale

    def positive_e2m1_dequant(value: float) -> float:
        if value < 0.25:
            return 0.0
        if value < 0.75:
            return 0.5
        if value < 1.25:
            return 1.0
        if value < 1.75:
            return 1.5
        if value < 2.5:
            return 2.0
        if value < 3.5:
            return 3.0
        if value < 5.0:
            return 4.0
        return 6.0

    def mxfp4_rte_reconstruct(value: float) -> float:
        scale = mxfp4_e8m0_rte_scale(value)
        payload = value * 6.0 / scale
        return positive_e2m1_dequant(payload) * scale / 6.0

    if heads_q != heads_k:
        raise ValueError(
            "The fused D192 QK MMA smoke currently uses flat GEMM L mode and "
            "requires heads_q == heads_k. Reintroduce GQA K-head replication "
            "after the base issue-only path is validated."
        )
    selected_raw_modes = sum(
        int(flag)
        for flag in (
            zero_inputs,
            constant_ones,
            constant_ones_unpadded_packed,
            q_ones_k_half_zero_packed,
            q_ones_k_alternating_zero_packed,
        )
    )
    if selected_raw_modes > 1:
        raise ValueError("Select at most one deterministic raw-input mode")
    selected_store_modes = sum(
        int(flag)
        for flag in (
            store_qk_accumulator,
            store_qk_softmax,
            store_qk_mxfp4_quant,
            store_qk_mxfp4_scale_debug,
            store_qk_mxfp4_payload_debug,
            store_mxfp4_pv_accumulator,
        )
    )
    if selected_store_modes > 1:
        raise ValueError(
            "store_qk_accumulator, store_qk_softmax, store_qk_mxfp4_quant, "
            "store_qk_mxfp4_scale_debug, store_qk_mxfp4_payload_debug, and "
            "store_mxfp4_pv_accumulator are exclusive"
        )

    import cutlass  # type: ignore
    import cutlass.cute as cute  # type: ignore
    import cutlass.torch as cutlass_torch  # type: ignore
    import torch
    from cutlass.cute.runtime import make_ptr  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the D192 MXFP4 smoke run")
    torch_device = torch.device(device)
    torch.cuda.set_device(torch_device)

    h_r = heads_q // heads_k
    qk_padded = geometry.qk_head_dim_padded
    d_v = geometry.v_head_dim

    q_numel = batch_size * heads_q * seqlen_q * qk_padded
    k_numel = batch_size * heads_k * seqlen_k * qk_padded
    v_numel = batch_size * heads_k * seqlen_k * d_v
    o_numel = batch_size * heads_q * seqlen_q * d_v
    q_scale_numel = batch_size * heads_q * seqlen_q * (qk_padded // geometry.qk_sf_vec_size)
    k_scale_numel = batch_size * heads_k * seqlen_k * (qk_padded // geometry.qk_sf_vec_size)
    v_scale_numel = batch_size * heads_k * seqlen_k * (d_v // geometry.pv_sf_vec_size)
    scores_numel = (
        batch_size * heads_q * seqlen_q * seqlen_k
        if (
            store_qk_accumulator
            or store_qk_softmax
            or store_qk_mxfp4_quant
            or store_qk_mxfp4_scale_debug
            or store_qk_mxfp4_payload_debug
            or store_mxfp4_pv_accumulator
        )
        else 1
    )
    qk_issue_grid = (
        (seqlen_q + 127) // 128,
        (seqlen_k + 127) // 128,
        heads_q * batch_size,
    )

    def alloc_flat(numel: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty((numel,), device=device, dtype=dtype)

    raw_input_storage = (
        zero_inputs
        or constant_ones
        or constant_ones_unpadded_packed
        or q_ones_k_half_zero_packed
        or q_ones_k_alternating_zero_packed
    )
    q_storage_dtype = torch.uint8 if raw_input_storage else torch.float4_e2m1fn_x2
    k_storage_dtype = torch.uint8 if raw_input_storage else torch.float4_e2m1fn_x2
    v_storage_dtype = torch.uint8 if raw_input_storage else torch.float4_e2m1fn_x2
    scale_storage_dtype = torch.uint8 if raw_input_storage else torch.float8_e8m0fnu
    q_torch = alloc_flat(q_numel, q_storage_dtype)
    k_torch = alloc_flat(k_numel, k_storage_dtype)
    v_torch = alloc_flat(v_numel, v_storage_dtype)
    o_torch = alloc_flat(o_numel, torch.bfloat16)
    scale_q_torch = alloc_flat(q_scale_numel, scale_storage_dtype)
    scale_k_torch = alloc_flat(k_scale_numel, scale_storage_dtype)
    scale_v_torch = alloc_flat(v_scale_numel, torch.float8_e8m0fnu)
    scores_torch = alloc_flat(scores_numel, torch.float32)

    if zero_inputs:
        q_torch.zero_()
        k_torch.zero_()
        v_torch.zero_()
        scale_q_torch.zero_()
        scale_k_torch.zero_()
        scale_v_torch.zero_()
        scores_torch.fill_(float("nan"))
    if constant_ones:
        q_torch.fill_(0x22)
        k_torch.fill_(0x22)
        v_torch.fill_(0x22)
        scale_q_torch.fill_(0x7F)
        scale_k_torch.fill_(0x7F)
        scale_v_torch.fill_(0x7F)
        scores_torch.fill_(float("nan"))
    if constant_ones_unpadded_packed:
        q_torch.zero_()
        k_torch.zero_()
        v_torch.fill_(0x22)
        scale_q_torch.fill_(0x7F)
        scale_k_torch.fill_(0x7F)
        scale_v_torch.fill_(0x7F)
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
    if q_ones_k_half_zero_packed:
        q_torch.zero_()
        k_torch.zero_()
        v_torch.fill_(0x22)
        scale_q_torch.fill_(0x7F)
        scale_k_torch.fill_(0x7F)
        scale_v_torch.fill_(0x7F)
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
        k_packed_by_head = k_packed.view(
            batch_size * heads_k,
            seqlen_k,
            qk_packed_bytes_per_row,
        )
        k_packed_by_head[
            :,
            : seqlen_k // 2,
            :qk_active_packed_bytes_per_row,
        ].fill_(0x22)
        scores_torch.fill_(float("nan"))
    if q_ones_k_alternating_zero_packed:
        q_torch.zero_()
        k_torch.zero_()
        v_torch.fill_(0x22)
        scale_q_torch.fill_(0x7F)
        scale_k_torch.fill_(0x7F)
        scale_v_torch.fill_(0x7F)
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
        k_packed_by_head = k_packed.view(
            batch_size * heads_k,
            seqlen_k,
            qk_packed_bytes_per_row,
        )
        k_packed_by_head[
            :,
            0::2,
            :qk_active_packed_bytes_per_row,
        ].fill_(0x22)
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
    v_ptr = make_ptr(
        cutlass.Float4E2M1FN,
        v_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    o_ptr = make_ptr(
        cutlass.BFloat16,
        o_torch.data_ptr(),
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
    scale_v_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        scale_v_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    scores_ptr = make_ptr(
        cutlass.Float32,
        scores_torch.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    fmha = build_mxfp4_fmha_d192_kernel(
        geometry=geometry,
        store_qk_accumulator=store_qk_accumulator,
        store_qk_softmax=store_qk_softmax,
        store_qk_mxfp4_quant=store_qk_mxfp4_quant,
        store_qk_mxfp4_scale_debug=store_qk_mxfp4_scale_debug,
        store_qk_mxfp4_payload_debug=store_qk_mxfp4_payload_debug,
        store_mxfp4_pv_accumulator=store_mxfp4_pv_accumulator,
    )
    current_stream = cutlass_torch.default_stream()
    problem_size = (
        batch_size,
        seqlen_q,
        seqlen_k,
        heads_q,
        heads_k,
        geometry.qk_head_dim,
        geometry.v_head_dim,
    )

    compiled = cute.compile(
        fmha,
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        scale_q_ptr,
        scale_k_ptr,
        scale_v_ptr,
        scores_ptr,
        problem_size,
        scale_softmax_log2_value,
        1.0,
        None,
        None,
        current_stream,
        store_qk_accumulator,
        store_qk_softmax,
        store_qk_mxfp4_quant,
        store_qk_mxfp4_scale_debug,
        store_qk_mxfp4_payload_debug,
        store_mxfp4_pv_accumulator,
        options="--opt-level 2",
    )
    for _ in range(warmup_iterations):
        compiled(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            scale_q_ptr,
            scale_k_ptr,
            scale_v_ptr,
            scores_ptr,
            problem_size,
            scale_softmax_log2_value,
            1.0,
            None,
            None,
            current_stream,
            store_qk_accumulator,
            store_qk_softmax,
            store_qk_mxfp4_quant,
            store_qk_mxfp4_scale_debug,
            store_qk_mxfp4_payload_debug,
            store_mxfp4_pv_accumulator,
        )
    torch.cuda.synchronize(device=device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iterations):
        compiled(
            q_ptr,
            k_ptr,
            v_ptr,
            o_ptr,
            scale_q_ptr,
            scale_k_ptr,
            scale_v_ptr,
            scores_ptr,
            problem_size,
            scale_softmax_log2_value,
            1.0,
            None,
            None,
            current_stream,
            store_qk_accumulator,
            store_qk_softmax,
            store_qk_mxfp4_quant,
            store_qk_mxfp4_scale_debug,
            store_qk_mxfp4_payload_debug,
            store_mxfp4_pv_accumulator,
        )
    end_event.record()
    torch.cuda.synchronize(device=device)
    exec_time_ms = start_event.elapsed_time(end_event) / max(iterations, 1)

    result = {
        "status": "ok",
        "mode": "qk_mma_issue",
        "device": str(device),
        "problem_size": problem_size,
        "qk_issue_grid": qk_issue_grid,
        "warmup_iterations": int(warmup_iterations),
        "iterations": int(iterations),
        "exec_time_ms": float(exec_time_ms),
        "exec_time_us": float(exec_time_ms) * 1000.0,
        "store_qk_accumulator": bool(store_qk_accumulator),
        "store_qk_softmax": bool(store_qk_softmax),
        "store_qk_mxfp4_quant": bool(store_qk_mxfp4_quant),
        "store_qk_mxfp4_scale_debug": bool(store_qk_mxfp4_scale_debug),
        "store_qk_mxfp4_payload_debug": bool(store_qk_mxfp4_payload_debug),
        "store_mxfp4_pv_accumulator": bool(store_mxfp4_pv_accumulator),
        "scale_softmax_log2_value": float(scale_softmax_log2_value),
        "zero_inputs": bool(zero_inputs),
        "constant_ones": bool(constant_ones),
        "constant_ones_unpadded_packed": bool(constant_ones_unpadded_packed),
        "q_ones_k_half_zero_packed": bool(q_ones_k_half_zero_packed),
        "q_ones_k_alternating_zero_packed": bool(
            q_ones_k_alternating_zero_packed
        ),
        "q_numel": int(q_torch.numel()),
        "k_numel": int(k_torch.numel()),
        "v_numel": int(v_torch.numel()),
        "o_numel": int(o_torch.numel()),
        "scale_q_numel": int(scale_q_torch.numel()),
        "scale_k_numel": int(scale_k_torch.numel()),
        "scale_v_numel": int(scale_v_torch.numel()),
        "scores_numel": int(scores_torch.numel()),
    }
    if raw_input_storage:
        result["q_nonzero_bytes"] = int((q_torch != 0).sum().item())
        result["k_nonzero_bytes"] = int((k_torch != 0).sum().item())
        result["v_nonzero_bytes"] = int((v_torch != 0).sum().item())
        result["scale_q_nonzero_bytes"] = int((scale_q_torch != 0).sum().item())
        result["scale_k_nonzero_bytes"] = int((scale_k_torch != 0).sum().item())
        result["scale_v_nonzero_bytes"] = int((scale_v_torch != 0).sum().item())
    if (
        store_qk_accumulator
        or store_qk_softmax
        or store_qk_mxfp4_quant
        or store_qk_mxfp4_scale_debug
        or store_qk_mxfp4_payload_debug
        or store_mxfp4_pv_accumulator
    ):
        finite_scores = torch.nan_to_num(scores_torch, nan=0.0)
        nan_indices = torch.isnan(scores_torch).nonzero().flatten()[:16]
        result["scores_max_abs"] = float(finite_scores.abs().max().item())
        result["scores_min"] = float(finite_scores.min().item())
        result["scores_max"] = float(finite_scores.max().item())
        result["scores_nan_count"] = int(torch.isnan(scores_torch).sum().item())
        result["scores_nan_indices_head"] = [
            int(index) for index in nan_indices.detach().cpu().tolist()
        ]
        if store_mxfp4_pv_accumulator:
            expected_score = float(seqlen_k * 6.0)
            if q_ones_k_half_zero_packed or q_ones_k_alternating_zero_packed:
                expected_low_p = 2.0 ** (
                    -float(geometry.qk_head_dim) * float(scale_softmax_log2_value)
                )
                if expected_low_p < 1.0e-37:
                    expected_low_p = 0.0
                expected_low_payload = positive_e2m1_dequant(
                    float(expected_low_p) * 6.0
                )
                expected_score = float(
                    (seqlen_k // 2) * 6.0
                    + (seqlen_k - seqlen_k // 2) * expected_low_payload
                )
            finite_mask = ~torch.isnan(scores_torch)
            result["expected_score"] = expected_score
            result["scores_expected_max_abs_diff"] = float(
                (scores_torch[finite_mask] - expected_score).abs().max().item()
            )
        elif (
            store_qk_softmax
            or store_qk_mxfp4_quant
            or store_qk_mxfp4_scale_debug
            or store_qk_mxfp4_payload_debug
        ):
            if q_ones_k_half_zero_packed or q_ones_k_alternating_zero_packed:
                expected_low_p = 2.0 ** (
                    -float(geometry.qk_head_dim) * float(scale_softmax_log2_value)
                )
                if expected_low_p < 1.0e-37:
                    expected_low_p = 0.0
                expected_min = expected_low_p
                expected_max = 1.0
                if store_qk_mxfp4_quant:
                    if q_ones_k_alternating_zero_packed:
                        expected_min = (
                            positive_e2m1_dequant(float(expected_low_p) * 6.0) / 6.0
                        )
                    else:
                        expected_min = mxfp4_rte_reconstruct(float(expected_low_p))
                elif store_qk_mxfp4_scale_debug:
                    if q_ones_k_alternating_zero_packed:
                        expected_min = 1.0
                    else:
                        expected_min = mxfp4_e8m0_rte_scale(float(expected_low_p))
                    expected_max = 1.0
                elif store_qk_mxfp4_payload_debug:
                    if q_ones_k_alternating_zero_packed:
                        expected_min = positive_e2m1_dequant(
                            float(expected_low_p) * 6.0
                        )
                    else:
                        expected_min = 6.0
                    expected_max = 6.0
                result["expected_score_min"] = float(expected_min)
                result["expected_score_max"] = float(expected_max)
                result["scores_expected_min_abs_diff"] = abs(
                    result["scores_min"] - result["expected_score_min"]
                )
                result["scores_expected_max_abs_diff"] = abs(
                    result["scores_max"] - result["expected_score_max"]
                )
            else:
                expected_score = 6.0 if store_qk_mxfp4_payload_debug else 1.0
                finite_mask = ~torch.isnan(scores_torch)
                result["expected_score"] = expected_score
                result["scores_expected_max_abs_diff"] = float(
                    (scores_torch[finite_mask] - expected_score).abs().max().item()
                )
        elif store_qk_accumulator and (
            q_ones_k_half_zero_packed or q_ones_k_alternating_zero_packed
        ):
            result["expected_score_min"] = 0.0
            result["expected_score_max"] = float(geometry.qk_head_dim)
            result["scores_expected_min_abs_diff"] = abs(
                result["scores_min"] - result["expected_score_min"]
            )
            result["scores_expected_max_abs_diff"] = abs(
                result["scores_max"] - result["expected_score_max"]
            )
        elif store_qk_accumulator and (constant_ones or constant_ones_unpadded_packed):
            expected_score = float(
                geometry.qk_head_dim
                if constant_ones_unpadded_packed
                else geometry.qk_head_dim_padded
            )
            finite_mask = ~torch.isnan(scores_torch)
            result["expected_score"] = expected_score
            result["scores_expected_max_abs_diff"] = float(
                (scores_torch[finite_mask] - expected_score).abs().max().item()
            )
        elif store_qk_accumulator and zero_inputs:
            expected_score = 0.0
            finite_mask = ~torch.isnan(scores_torch)
            result["expected_score"] = expected_score
            result["scores_expected_max_abs_diff"] = float(
                (scores_torch[finite_mask] - expected_score).abs().max().item()
            )
    return result
