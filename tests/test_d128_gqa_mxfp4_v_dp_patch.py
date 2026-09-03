from __future__ import annotations

import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOWP = ROOT / "tk_fa4" / "lowp_fa4_bwd"
CONTROL = (
    ROOT
    / "qutlass"
    / "third_party"
    / "cutlass"
    / "examples"
    / "python"
    / "CuTeDSL"
    / "blackwell"
    / "fmha_bwd.py"
)
MX_DP_PATCH = LOWP / "d128_gqa_mxfp4_v_dp.patch"

# Default ``_load_control`` composition.  Optional owner-fused, direct-TMA,
# detached-TMEM, and precomposed controls intentionally are not part of this
# test: the MX dP spike is based on the ordinary shared-P control requested by
# the experiment contract.
DEFAULT_PATCH_CHAIN = (
    "d64_gqa_cute.patch",
    "d64_gqa_tile_ready.patch",
    "d64_gqa_owner_quantize.patch",
    "d64_gqa_fp8_p_lift.patch",
    "d64_gqa_owner_full_operand.patch",
    "d64_gqa_owner_kv_quantize.patch",
    "d64_gqa_owner_kv_no_materialize.patch",
    "d64_gqa_reverse_query.patch",
    "d64_gqa_head_fast_raster.patch",
    "d64_gqa_fp8_p_layout.patch",
    "d64_gqa_fused_probability_lift.patch",
    "d64_gqa_owner_d64.patch",
    "d64_gqa_forward_mx_probability_replay.patch",
)


def _apply_patch(source: str, patch: Path, work: Path, index: int) -> str:
    source_path = work / f"control_{index:02d}.py"
    source_path.write_text(source)
    result = subprocess.run(
        [
            "patch",
            "--silent",
            "--output=-",
            str(source_path),
            str(patch),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _compose_default_control(tmp_path: Path) -> str:
    source = CONTROL.read_text()
    for index, patch_name in enumerate(DEFAULT_PATCH_CHAIN, start=1):
        source = _apply_patch(source, LOWP / patch_name, tmp_path, index)
    return source


def _backward_class(source: str) -> ast.ClassDef:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "BlackwellFusedMultiHeadAttentionBackward"
    ]
    assert len(matches) == 1
    return matches[0]


def _method(node: ast.ClassDef, name: str) -> ast.FunctionDef:
    matches = [
        item
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _name_path(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _name_path(node.value)
        return f"{owner}.{node.attr}" if owner is not None else None
    return None


def _statement_call(statement: ast.stmt, path: str) -> ast.Call | None:
    if (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and _name_path(statement.value.func) == path
    ):
        return statement.value
    return None


def _calls(node: ast.AST, path: str) -> list[ast.Call]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call) and _name_path(item.func) == path
    ]


def _const_expr_flag(node: ast.If, *, negated: bool = False) -> bool:
    test = node.test
    if (
        not isinstance(test, ast.Call)
        or _name_path(test.func) != "cutlass.const_expr"
        or len(test.args) != 1
        or test.keywords
    ):
        return False
    flag = test.args[0]
    if negated:
        if not isinstance(flag, ast.UnaryOp) or not isinstance(flag.op, ast.Not):
            return False
        flag = flag.operand
    return _name_path(flag) == "self.use_d128_mxfp4_v_dp"


def _owner_block(
    root: ast.AST, target: ast.stmt
) -> tuple[ast.AST, list[ast.stmt], int]:
    for owner in ast.walk(root):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(owner, field, None)
            if isinstance(statements, list) and target in statements:
                return owner, statements, statements.index(target)
    raise AssertionError(f"statement at line {target.lineno} has no owner block")


def _attribute_assignment(
    method: ast.FunctionDef, attribute: str
) -> ast.Assign:
    matches = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and _name_path(node.targets[0]) == f"self.{attribute}"
    ]
    assert len(matches) == 1
    return matches[0]


def _name_assignment(method: ast.FunctionDef, name: str) -> ast.Assign:
    matches = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    assert len(matches) == 1
    return matches[0]


def _same_expression(node: ast.AST, expression: str) -> bool:
    expected = ast.parse(expression, mode="eval").body
    return ast.dump(node, include_attributes=False) == ast.dump(
        expected, include_attributes=False
    )


def test_patch_applies_to_fully_composed_default_control(tmp_path: Path) -> None:
    control = _compose_default_control(tmp_path)
    patched = _apply_patch(control, MX_DP_PATCH, tmp_path, 99)

    # Parsing catches indentation/signature drift without importing CUDA or
    # CUTLASS and therefore cannot initialize a GPU.
    ast.parse(patched)

    old_call = _method(_backward_class(control), "__call__")
    new_call = _method(_backward_class(patched), "__call__")
    old_args = [arg.arg for arg in old_call.args.args]
    new_args = [arg.arg for arg in new_call.args.args]

    # Every old positional argument keeps its index.  The physical scale
    # iterator is an optional final argument, so old callers retain behavior.
    assert new_args[:-1] == old_args
    assert new_args[-1] == "mxfp4_v_scale_iter"
    assert len(new_call.args.defaults) == 1
    assert isinstance(new_call.args.defaults[0], ast.Constant)
    assert new_call.args.defaults[0].value is None

    init = _method(_backward_class(patched), "__init__")
    flag_assignments = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "use_d128_mxfp4_v_dp"
            for target in node.targets
        )
    ]
    assert len(flag_assignments) == 1
    assert isinstance(flag_assignments[0].value, ast.Constant)
    assert flag_assignments[0].value.value is False


def test_spike_encodes_direct_gqa_layout_and_fails_closed(tmp_path: Path) -> None:
    control = _compose_default_control(tmp_path)
    patched = _apply_patch(control, MX_DP_PATCH, tmp_path, 99)
    patch_text = MX_DP_PATCH.read_text()

    # The payload broadcasts G.  Scale pages stay in the producer's exact
    # [B, K/128, Hkv, 512] order and therefore have no G mode to replicate.
    assert "((0, d), k_seq * d * h_k)" in patched
    assert "h_k * TK_D128_MXFP4_DP_SCALE_PAGE_BYTES" in patched
    assert "pages * h_k * TK_D128_MXFP4_DP_SCALE_PAGE_BYTES" in patched
    assert ".repeat(" not in patch_text
    assert ".expand(" not in patch_text

    # MXFP4 V has the width-six endpoint while dO remains the projection's
    # resident E4M3(x4).  Normalize the mixed raw dP to the retained x16
    # contract and center it in one packed FMA, without a global SxS pass.
    compute_anchor = patched.index("# Compute dS = dsoftmax(P, dP, sum_OdO)")
    fused_center = patched.index("cute.arch.fma_packed_f32x2", compute_anchor)
    rescale = patched.index(
        "(TK_D128_MXFP4_DP_RAW_TO_X16,) * 2", fused_center
    )
    retained_center = patched.index("cute.arch.add_packed_f32x2", rescale)
    assert compute_anchor < fused_center < rescale < retained_center
    assert "TK_D128_MXFP4_DP_RAW_TO_X16 = 2.0 / 3.0" in patched

    # The complete implementation remains guarded until the generated SM100
    # specialization is compiled; default callers can never enter it.
    assert "self.d128_mxfp4_v_dp_compiled = False" in patched
    assert "and not self.d128_mxfp4_v_dp_compiled" in patched
    assert "fail-closed" in patched


def test_spike_wires_native_blockscaled_dp_and_on_chip_do(
    tmp_path: Path,
) -> None:
    control = _compose_default_control(tmp_path)
    patched = _apply_patch(control, MX_DP_PATCH, tmp_path, 99)

    # Build the non-materializable S2T copy objects inside the elected MMA
    # scope.  Passing them through ``bwd`` makes the DSL treat them as
    # loop-carried values and leaves unresolved materializations during MLIR
    # legalization; ordinary shared tensors are safe function arguments.
    assert "mx_s2t_sfa" not in patched
    assert "mx_s2t_sfb" not in patched
    backward = _backward_class(patched)
    mma = _method(backward, "mma")
    mma_annotations = {
        arg.arg: ast.unparse(arg.annotation)
        for arg in mma.args.args
        if arg.annotation is not None
    }
    assert mma_annotations["sMxVScale"] == "cute.Tensor"
    assert mma_annotations["sMxdOScale"] == "cute.Tensor"
    s2t_calls = [
        node
        for node in ast.walk(mma)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "d128_mx_s2t_copy_and_partition"
    ]
    assert len(s2t_calls) == 2
    dynamic_loops = [node for node in ast.walk(mma) if isinstance(node, ast.While)]
    assert dynamic_loops
    assert max(call.lineno for call in s2t_calls) < min(
        loop.lineno for loop in dynamic_loops
    )

    # dP is a native mixed FP4-by-E4M3 blockscaled MMA.  The separate A/B dtype
    # arguments are required by CUTLASS DSL 4.5; the legacy single-dtype form
    # would silently describe the removed pure-MX route.
    assert "make_blockscaled_trivial_tiled_mma" in patched
    mixed_atom = (
        "mx_VdO_tiled_mma = "
        "sm100_utils.make_blockscaled_trivial_tiled_mma(\n"
        "            Float4E2M1FN,\n"
        "            Float8E4M3FN,"
    )
    assert mixed_atom in patched
    assert "Float8E8M0FNU" in patched
    assert "make_smem_layout_sfa" in patched
    assert "make_smem_layout_sfb" in patched
    assert "make_tmem_layout_sfa" in patched
    assert "make_tmem_layout_sfb" in patched
    assert "tcgen05.Field.SFA" in patched
    assert "tcgen05.Field.SFB" in patched
    assert "tcgen05.Cp4x32x128bOp" in patched

    # Packed Float4 GMEM is expanded into SM100's byte-addressed TMA-unpack
    # representation.  SMEM layout, allocation, and descriptor stay Uint8;
    # the MMA atom—not a pointer recast—declares logical E2M1.
    assert "v_smem_layout_dtype = Uint8" in patched
    assert "v_smem_dtype = Uint8" in patched
    assert "internal_type=Uint8" in patched
    assert "dP_operand_dtype, V_smem_layout" in patched
    assert "cute.recast_ptr(sV.iterator, dtype=Float4E2M1FN)" not in patched
    assert "tdPTrV = dP_tiled_mma.make_fragment_A(sV)" in patched

    # A blockscaled TMA descriptor cannot represent the already-swizzled
    # producer page without changing its top-level shape.  The correctness
    # path copies the 512 physical bytes once, as 128 four-byte vectors.
    assert "tma_atom_V_scale" not in patched
    assert "tma_copy_V_scale" not in patched
    assert "gMxVScalePage = V_scale_in[" in patched
    assert "cute.make_layout(4)" in patched
    assert "if tidx < 128:" in patched
    assert "gMxVScaleVec[None, tidx].load()" in patched
    assert "self.cta_sync_barrier.arrive_and_wait()" in patched

    # The resident two-stage E4M3 dO feeds mixed dP directly and remains intact
    # for dense dV.  SFB is a one-time 512-byte E8M0 unity page; there is no
    # second dO payload, quantizer, or reduce-to-MMA readiness pipeline.
    assert "tdPTrdO = dP_tiled_mma.make_fragment_B(sdO)" in patched
    assert "sD128MxScales" in patched
    assert "TK_D128_MXFP4_DP_SHARED_SCALE_BYTES = 2 * 512" in patched
    assert "tdVrdOT[None, None, k_block, load_mma_dO_consumer_state.index]" in patched
    assert "sMxdOScaleBytes[tidx * 4 + byte] = Uint8(127)" in patched
    assert "sD128MxdO" not in patched
    assert "def quantize_d128_mx_dO_tile(" not in patched
    assert "d128_mx_dO_ready" not in patched



def test_score_release_is_uniform_exclusive_and_before_divergence(
    tmp_path: Path,
) -> None:
    control = _compose_default_control(tmp_path)
    patched = _apply_patch(control, MX_DP_PATCH, tmp_path, 99)
    compute = _method(_backward_class(patched), "compute")

    # There is exactly one score-fragment T2R copy in the compute loop.
    score_copy_statements = [
        node
        for node in ast.walk(compute)
        if isinstance(node, ast.Expr)
        and (call := _statement_call(node, "cute.copy")) is not None
        and [_name_path(arg) for arg in call.args]
        == ["tiled_t2r", "tTR_tST", "tTR_rST"]
    ]
    assert len(score_copy_statements) == 1
    score_copy = score_copy_statements[0]

    releases = _calls(compute, "mma_compute_S_pipeline.consumer_release")
    release_advances = _calls(compute, "mma_compute_S_consumer_state.advance")
    assert len(releases) == 2
    assert len(release_advances) == 2

    # The MX route's branch is compile-time-uniform.  Its three direct
    # statements implement CUTLASS's required per-consumer sequence, and the
    # branch is adjacent to the T2R copy and precedes the first runtime mask
    # divergence.  Thus every dispatched compute thread arrives once.
    early_branches = [
        node
        for node in ast.walk(compute)
        if isinstance(node, ast.If)
        and _const_expr_flag(node)
        and any(call in releases for call in ast.walk(node))
    ]
    assert len(early_branches) == 1
    early = early_branches[0]
    assert len(early.body) == 3
    early_fence = _statement_call(
        early.body[0], "cute.arch.fence_view_async_tmem_load"
    )
    early_release = _statement_call(
        early.body[1], "mma_compute_S_pipeline.consumer_release"
    )
    early_advance = _statement_call(
        early.body[2], "mma_compute_S_consumer_state.advance"
    )
    assert early_fence is not None and not early_fence.args
    assert early_release is not None
    assert [_name_path(arg) for arg in early_release.args] == [
        "mma_compute_S_consumer_state"
    ]
    assert early_advance is not None and not early_advance.args

    early_owner, early_block, early_index = _owner_block(compute, early)
    assert early_block[early_index - 1] is score_copy
    assert isinstance(early_block[early_index + 1], ast.If)
    assert _name_path(early_block[early_index + 1].test) == "is_masked_tile"
    score_waits = _calls(compute, "mma_compute_S_pipeline.consumer_wait")
    assert len(score_waits) == 1
    score_wait_statements = [
        statement
        for statement in early_block[: early_block.index(score_copy)]
        if _statement_call(
            statement, "mma_compute_S_pipeline.consumer_wait"
        )
        is score_waits[0]
    ]
    assert len(score_wait_statements) == 1
    assert [_name_path(arg) for arg in score_waits[0].args] == [
        "mma_compute_S_consumer_state"
    ]
    assert not early.orelse

    # The ordinary route retains exactly one mutually exclusive late release,
    # directly after P publication.  No unguarded third release can silently
    # double-arrive on the score pipeline's empty barrier.
    late_branches = [
        node
        for node in ast.walk(compute)
        if isinstance(node, ast.If)
        and _const_expr_flag(node, negated=True)
        and any(call in releases for call in ast.walk(node))
    ]
    assert len(late_branches) == 1
    late = late_branches[0]
    assert len(late.body) == 2
    assert not late.orelse
    late_release = _statement_call(
        late.body[0], "mma_compute_S_pipeline.consumer_release"
    )
    late_advance = _statement_call(
        late.body[1], "mma_compute_S_consumer_state.advance"
    )
    assert late_release is not None and late_advance is not None
    assert {id(call) for call in releases} == {
        id(early_release),
        id(late_release),
    }
    assert {id(call) for call in release_advances} == {
        id(early_advance),
        id(late_advance),
    }

    late_owner, late_block, late_index = _owner_block(compute, late)
    assert late_owner is early_owner
    assert late_block is early_block
    assert _statement_call(
        late_block[late_index - 2], "compute_mma_P_pipeline.producer_commit"
    ) is not None
    assert _statement_call(
        late_block[late_index - 1], "compute_mma_P_producer_state.advance"
    ) is not None


def test_score_pipeline_covers_the_full_compute_warpgroup(
    tmp_path: Path,
) -> None:
    control = _compose_default_control(tmp_path)
    patched = _apply_patch(control, MX_DP_PATCH, tmp_path, 99)
    backward = _backward_class(patched)
    init = _method(backward, "__init__")
    setup = _method(backward, "_setup_attributes")

    compute_warp_ids = ast.literal_eval(
        _attribute_assignment(init, "compute_warp_id").value
    )
    num_compute_warps = ast.literal_eval(
        _attribute_assignment(init, "num_compute_warps").value
    )
    threads_per_warp = ast.literal_eval(
        _attribute_assignment(init, "threads_per_warp").value
    )
    score_stages = ast.literal_eval(
        _attribute_assignment(setup, "mma_compute_S_stage").value
    )
    assert compute_warp_ids == tuple(
        range(compute_warp_ids[0], compute_warp_ids[-1] + 1)
    )
    assert len(compute_warp_ids) == num_compute_warps == 8
    assert threads_per_warp == 32
    assert num_compute_warps * threads_per_warp == 256
    assert score_stages == 1

    # bwd dispatches compute for the inclusive range above, so the set of
    # threads executing the uniform release branch matches the pipeline's
    # consumer-arrival count rather than merely naming the same constants.
    bwd = _method(backward, "bwd")
    compute_calls = [
        node
        for node in ast.walk(bwd)
        if isinstance(node, ast.Expr)
        and _statement_call(node, "self.compute") is not None
    ]
    assert len(compute_calls) == 1
    dispatch, _, _ = _owner_block(bwd, compute_calls[0])
    assert isinstance(dispatch, ast.If)
    assert _same_expression(
        dispatch.test,
        "warp_idx >= self.compute_warp_id[0] "
        "and warp_idx <= self.compute_warp_id[-1]",
    )

    make_pipeline = _method(
        backward, "make_and_init_mma_compute_S_pipeline"
    )
    consumer_assignment = _name_assignment(
        make_pipeline, "mma_compute_S_consumer_group"
    )
    assert isinstance(consumer_assignment.value, ast.Call)
    consumer_call = consumer_assignment.value
    assert _name_path(consumer_call.func) == "pipeline.CooperativeGroup"
    assert len(consumer_call.args) == 2 and not consumer_call.keywords
    assert _name_path(consumer_call.args[0]) == "pipeline.Agent.Thread"
    assert _same_expression(
        consumer_call.args[1],
        "self.num_compute_warps * self.threads_per_warp",
    )

    returns = [
        node
        for node in ast.walk(make_pipeline)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
    ]
    assert len(returns) == 1
    create_call = returns[0].value
    assert _name_path(create_call.func) == "pipeline.PipelineUmmaAsync.create"
    keywords = {keyword.arg: keyword.value for keyword in create_call.keywords}
    assert _name_path(keywords["barrier_storage"]) == "mma_compute_S_mbar_ptr"
    assert _name_path(keywords["num_stages"]) == "self.mma_compute_S_stage"
    assert _name_path(keywords["consumer_group"]) == (
        "mma_compute_S_consumer_group"
    )


def test_aliased_scale_publication_and_next_score_reuse_are_ordered(
    tmp_path: Path,
) -> None:
    control = _compose_default_control(tmp_path)
    patched = _apply_patch(control, MX_DP_PATCH, tmp_path, 99)
    mma = _method(_backward_class(patched), "mma")

    loops = [
        node
        for node in ast.walk(mma)
        if isinstance(node, ast.While)
        and _same_expression(node.test, "iter_count > 0")
    ]
    assert len(loops) == 1
    loop = loops[0]

    # Exactly two syntactic S2T publication sites cover the prologue and the
    # steady-state loop.  Both are guarded by the same uniform MX predicate.
    def publication_operands(branch: ast.If) -> list[str | None]:
        return [
            _name_path(call.args[0])
            for statement in branch.body
            if (call := _statement_call(statement, "cute.copy")) is not None
        ]

    publications = [
        node
        for node in ast.walk(mma)
        if isinstance(node, ast.If)
        and _const_expr_flag(node)
        and publication_operands(node) == ["s2t_sfa", "s2t_sfb"]
    ]
    assert len(publications) == 2
    assert len(
        [
            call
            for call in _calls(mma, "cute.copy")
            if call.args and _name_path(call.args[0]) in {"s2t_sfa", "s2t_sfb"}
        ]
    ) == 4

    publication_owners = {
        id(branch): _owner_block(mma, branch) for branch in publications
    }
    initial_publications = [
        branch
        for branch in publications
        if publication_owners[id(branch)][0] is mma
    ]
    loop_publications = [
        branch
        for branch in publications
        if publication_owners[id(branch)][0] is loop
    ]
    assert len(initial_publications) == len(loop_publications) == 1
    initial_publication = initial_publications[0]
    loop_publication = loop_publications[0]
    assert len(initial_publication.body) == 2

    # Prologue: score commit/phase advance precede a uniform score-pipeline
    # reacquire, which in turn dominates the first aliased S2T publication.
    initial_acquires = [
        node
        for node in mma.body
        if isinstance(node, ast.If)
        and _const_expr_flag(node)
        and len(node.body) == 1
        and _statement_call(
            node.body[0], "mma_compute_S_pipeline.producer_acquire"
        )
        is not None
    ]
    assert len(initial_acquires) == 1
    initial_acquire = initial_acquires[0]
    initial_acquire_call = _statement_call(
        initial_acquire.body[0], "mma_compute_S_pipeline.producer_acquire"
    )
    assert initial_acquire_call is not None
    assert [_name_path(arg) for arg in initial_acquire_call.args] == [
        "mma_compute_S_producer_state"
    ]
    initial_commit_indices = [
        index
        for index, statement in enumerate(mma.body)
        if _statement_call(
            statement, "mma_compute_S_pipeline.producer_commit"
        )
        is not None
    ]
    assert len(initial_commit_indices) == 1
    initial_commit_index = initial_commit_indices[0]
    initial_acquire_index = mma.body.index(initial_acquire)
    initial_publication_index = mma.body.index(initial_publication)
    assert initial_commit_index < initial_acquire_index < initial_publication_index
    assert _statement_call(
        mma.body[initial_commit_index + 1],
        "mma_compute_S_producer_state.advance",
    ) is not None

    # Steady state: after the next score commit, the MX publication branch
    # reacquires the same score state before either scale S2T copy.
    assert len(loop_publication.body) == 3
    loop_acquire = _statement_call(
        loop_publication.body[0],
        "mma_compute_S_pipeline.producer_acquire",
    )
    assert loop_acquire is not None
    assert [_name_path(arg) for arg in loop_acquire.args] == [
        "mma_compute_S_producer_state"
    ]
    score_loops = [
        statement
        for statement in loop.body
        if isinstance(statement, ast.For)
        and any(
            isinstance(call, ast.Call)
            and _name_path(call.func) == "cute.gemm"
            and call.args
            and _name_path(call.args[0]) == "KQ_tiled_mma"
            for call in ast.walk(statement)
        )
    ]
    assert len(score_loops) == 1
    score_loop = score_loops[0]
    loop_commit_indices = [
        index
        for index, statement in enumerate(loop.body)
        if _statement_call(
            statement, "mma_compute_S_pipeline.producer_commit"
        )
        is not None
    ]
    assert len(loop_commit_indices) == 1
    loop_commit_index = loop_commit_indices[0]
    assert (
        loop.body.index(score_loop)
        < loop_commit_index
        < loop.body.index(loop_publication)
    )
    assert _statement_call(
        loop.body[loop_commit_index + 1],
        "mma_compute_S_producer_state.advance",
    ) is not None

    # Conversely, the previous iteration's dP must finish reading the aliased
    # scale columns before the next KQ score MMA overwrites them.  The retained
    # route has its own mutually exclusive score acquire at the same handoff.
    shadow_waits = [
        node
        for node in loop.body
        if isinstance(node, ast.If)
        and _const_expr_flag(node)
        and len(node.body) == 2
        and _statement_call(
            node.body[0], "mma_compute_dP_pipeline.consumer_wait"
        )
        is not None
        and _statement_call(
            node.body[1], "d128_mx_dP_shadow_consumer_state.advance"
        )
        is not None
    ]
    assert len(shadow_waits) == 1
    shadow_wait = shadow_waits[0]
    shadow_wait_call = _statement_call(
        shadow_wait.body[0], "mma_compute_dP_pipeline.consumer_wait"
    )
    assert shadow_wait_call is not None
    assert [_name_path(arg) for arg in shadow_wait_call.args] == [
        "d128_mx_dP_shadow_consumer_state"
    ]
    assert loop.body.index(shadow_wait) < loop.body.index(score_loop)

    retained_acquires = [
        node
        for node in loop.body
        if isinstance(node, ast.If)
        and _const_expr_flag(node, negated=True)
        and len(node.body) == 1
        and _statement_call(
            node.body[0], "mma_compute_S_pipeline.producer_acquire"
        )
        is not None
    ]
    assert len(retained_acquires) == 1
    assert loop.body.index(retained_acquires[0]) < loop.body.index(score_loop)

    shadow_state = _name_assignment(
        mma, "d128_mx_dP_shadow_consumer_state"
    ).value
    assert isinstance(shadow_state, ast.Call)
    assert _name_path(shadow_state.func) == "pipeline.make_pipeline_state"
    assert [_name_path(arg) for arg in shadow_state.args] == [
        "pipeline.PipelineUserType.Consumer",
        "self.mma_compute_dP_stage",
    ]


def test_mixed_route_preserves_dense_do_and_cutlass_45_proxy_contract(
    tmp_path: Path,
) -> None:
    control = _compose_default_control(tmp_path)
    patched = _apply_patch(control, MX_DP_PATCH, tmp_path, 99)

    # Dense dO is still loaded through its original E4M3 TMA atom and remains
    # the operand of PdO/dV; mixed dP only changes V's A descriptor.
    assert "tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B" in patched
    assert "dO_smem_layout_staged = sm100_utils.make_smem_layout_b" in patched
    assert "tdVrdOT = PdO_tiled_mma.make_fragment_B(sdOT)" in patched
    assert "d128_mx_word_amax" not in patched
    assert "d128_mx_pack_e4m3x8_to_e2m1x8" not in patched

    # CUTLASS DSL 4.5 removed the enum-valued proxy-fence spelling.  Every
    # inherited async-shared fence in the composed route uses the string API.
    assert "cute.arch.ProxyKind" not in patched
    assert "cute.arch.SharedSpace" not in patched
    assert patched.count('cute.arch.fence_proxy("async.shared", space="cta")') == 4


def test_physical_scale_stride_matches_producer_page_formula() -> None:
    pages = 3
    hkv = 8
    for batch in (0, 1):
        for head in (0, 3, 7):
            for row in (0, 31, 32, 63, 64, 95, 96, 127, 128, 255, 383):
                for k_block in range(4):
                    page = row // 128
                    physical = (
                        batch * hkv * pages * 512
                        + page * hkv * 512
                        + head * 512
                        + (row % 32) * 16
                        + ((row // 32) % 4) * 4
                        + k_block
                    )
                    flat_page_stride = (
                        batch * hkv * pages * 512
                        + page * hkv * 512
                        + head * 512
                        + (row % 32) * 16
                        + ((row % 128) // 32) * 4
                        + k_block
                    )
                    assert flat_page_stride == physical
