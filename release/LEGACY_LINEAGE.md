# Native backward development lineage

The route catalog describes the decision-relevant method families: controls,
paper routes, promoted implementations, important numerical ablations, and
fail-closed experiments. It is not a claim that every intermediate source
revision is a distinct scientific method.

The causal source export also retains the complete native backward development
lineage. At this release state, `tk_fa4/native_gqa_tk_bwd/` contains 80
versioned `Makefile.v*` entry points together with their CUDA translation
units, headers, validators, and notes. The exact expected Makefile set is in
`release/legacy_backward_makefiles.txt`; the release verifier compares that
list with the checkout so an intermediate revision cannot disappear silently.

Most of these versions are exploratory schedule or ABI revisions. They were
not all measured under a common protocol, and many have no surviving result
receipt. Their presence proves source continuity, not correctness, speed, or
fitness for training. Start from `release/routes.json` when reproducing a
reported or decision-relevant result. Consult an uncatalogued version only
when investigating its source history, and promote it to a named route with a
build target, tests, evidence boundary, and promotion gate before using it in
a new claim.

The earlier `cfc06dad` epoch is preserved separately under
`reproduction/snapshots/forward_cfc06dad/tk_fa4/`. It includes that epoch's
complete source/development tree, including its 12-file `fp4_fa4_bwd/`
prototype family. Do not combine those files with the later causal tree by
copying them in place.
