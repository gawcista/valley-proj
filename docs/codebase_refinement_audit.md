# ValleyScope Codebase Refinement Audit

Date: 2026-06-12 | Status: Read-only audit (no code changes)

This document identifies concrete refinement candidates across the four
ValleyScope physics layers and the debug/diagnostics layer.  For each
candidate: physical safety, risk, existing test coverage, and protected
contracts.

## Baseline Metrics

| Category | Count |
|----------|-------|
| Production Python modules | 58 |
| Production LOC | ~10,700 |
| Test files | 27 |
| Tests collected (pytest) | 709 |
| Test LOC | ~11,150 |
| Test/production LOC ratio | ~1.04 |

## Layer 1 — Valley Projection + Diagnostics

**Files**: `projection/`, `symmetry/`, `analysis/projector_symmetry.py`,
`analysis/target_subspace_closure.py`, `analysis/hsp_star*.py`,
`analysis/decision_tree.py`, `analysis/layer_pairing_diagnostic.py`,
`subspace/`.

**Physical contracts protected**:
- Seed projector symmetry: D_g P_a^0 D_g^† ≈ P_{π_g(a)}
- HSP little group: G_k = {g | gk = k + G_M}
- Valley mapping: π_g(a)
- Target-subspace closure: D_raw unitarity and group-relation checks

### Candidate 1.1: Remove or merge `decision_tree.py` (384 lines)

**Why safe**: The `derive_derived_score`, `derive_polarization_score`,
`derive_valley_status` functions are used only inside `analyze_hsp.py`
and `summary_report.py` to produce human-readable status strings (`clean`,
`approx`, `mixed`).  The physics is a layer of display logic, not a
readiness gate.

**Risk**: Low.  The status strings are user-facing; test assertions in
`test_decision_tree.py` (465 lines) cover the mapping from numeric scores
to status labels.  Merging into `analysis/` would not change behavior.

**Coverage**: 465 lines in `tests/test_decision_tree.py`.

**Recommendation**: Merge thresholds logic into `analysis/` status helpers
used by the workflow and summary; keep tests as regression anchors for
user-facing status labels.

### Candidate 1.2: `layer_pairing_diagnostic.py` (281 lines)

**Why safe**: Layer-pairing diagnostics are a model-specific micro-tool
unused in the main workflow.  This module is imported but its outputs are
debug-only.

**Risk**: Low.  Tests in `tests/test_layer_pairing_diagnostic.py` exist
but the module has no coupling to readiness gates.

**Recommendation**: Leave as-is or move to a `debug/` subpackage.

### Candidate 1.3: `analyze_hsp.py` internal helpers (32 functions, 2137 lines)

**Why safe**: Most `_build_*`, `_infer_*`, `_apply_*_gate` functions are
physics logic but live in the workflow orchestrator rather than in their
respective analysis modules.  Relocating them does not change behavior.

**Risk**: Medium.  Call sites inside the main `analyze_hsp` function would
need updating.  Tests that mock internal helpers (`monkeypatch`) would
need import-path updates.

**Coverage**: Tests scattered across `test_io_and_workflow.py` (5107 lines,
~130 tests), `test_symmetry_adapted_valley_report.py`, etc.

**Recommendation**: Extract `_symprec_scan_summary`, `_resolve_qcut`,
`_partition_valley_orbits`, `_build_sampled_k_coverage`, and
`_warn_fixed_center_distance` into their respective analysis modules.
Keep the main `analyze_hsp` as a thin orchestrator (~500 lines).

## Layer 2 — Trusted Valley-Preserving Irreps / EBR Input

**Files**: `analysis/irrep_workflow_decision.py`, `analysis/valley_irrep_matching.py`,
`analysis/valley_little_group.py`, `analysis/symmetry_adapted_*.py`,
`analysis/ebr_input_candidates.py`, `analysis/ebr_problem_instances.py`,
`analysis/ebr_export_bundle.py`.

**Physical contracts protected**:
- Readiness: only `trusted` rows produce `ready_for_ebr_input=true`
- HSP completeness: `expected_hsps` policy table, no inference
- Irrep matching: exact phase-to-label for spinful C3/C2

### Candidate 2.1: `valley_little_group.py` (1241 lines)

**Why safe**: The module is physically essential (valley-preserving subgroup
construction, irrep matching integration, valley-orbit partitioning) but
is very large.  The logic is correct; the size is from accumulated helper
functions that could be split into focused modules.

**Risk**: Medium.  Any split must preserve the `G_k^(a)` construction
and the per-valley operation inventory that feeds irrep matching.

**Coverage**: `test_valley_irrep_matching.py` (375 lines), plus
scattered workflow tests.

**Recommendation**: Split into `valley_preserving_subgroup.py` (core
G_k^(a) construction) and `valley_little_group_inventory.py` (inventory
and reporting).

### Candidate 2.2: `symmetry_adapted_valley_report.py` (705 lines)

**Why safe**: Outputs to a debug-level JSON file in `debug` profile.
Physics is correct; size is from report construction, not from
computation.

**Risk**: Low.  The formal physics (projector construction, character
diagnostics) is in separate modules.

**Recommendation**: Move report construction logic to `reports/` and
keep the computational core in `analysis/`.

### Candidate 2.3: `symmetry_adapted_character_diagnostics.py` (341 lines)

**Why safe**: Character diagnostics are embedded in the SA report, not
a standalone output.  The computational logic is correct.

**Risk**: Low.  Covered by SA report tests.

**Recommendation**: Merge into `symmetry_adapted_valley_report.py` or
keep as a separate helper if the SA report module is split.

## Layer 3 — External Reduced EBR Mapping / Package-Data

**Files**: `analysis/reduced_ebr_mapping.py`, `analysis/reduced_ebr_solver.py`,
`analysis/irrep_runtime_reducer.py`, `analysis/irrep_data_normalizer.py`,
`analysis/irreptables_runtime_table_builder.py`,
`analysis/irrep_availability_probe.py`,
`analysis/reduced_ebr_source_basis_inspector.py`,
`analysis/reduced_ebr_spec_template_validator.py`,
`analysis/database_ingestion_record.py`.

**Physical contracts protected**:
- Exact integer decomposition (Smith normal form, no OR-Tools)
- Basis compatibility gate (expected_hsps match)
- External-table validation (load_reduced_ebr_table)
- No built-in tables, no heuristic fitting

### Candidate 3.1: `reduced_ebr_mapping.py` + `reduced_ebr_solver.py` (365 + 348 = 713 lines)

**Why safe**: The solver was already extracted from the mapper in a
previous task.  Both are now focused and testable independently.
Further splitting is not needed.

**Risk**: Very low.  Both are well-tested (112 tests).

**Recommendation**: Keep as-is.  The solver/mapper separation is clean.

### Candidate 3.2: Normalizer/reducer/builder/inspector/validator chain (9 modules, ~1500 lines)

**Why safe**: These are all offline authoring tools, not wired into
`analyze_hsp`.  They form a coherent Layer 3 toolchain.  Consolidation
into fewer modules would reduce import surface.

**Risk**: Low.  Each module has focused tests.

**Current modules**:
- `irrep_runtime_reducer.py` (220 lines)
- `irrep_data_normalizer.py` (200 lines)
- `irreptables_runtime_table_builder.py` (170+ lines load path only)
- `irrep_availability_probe.py` (193 lines)
- `reduced_ebr_source_basis_inspector.py` (170 lines)
- `reduced_ebr_spec_template_validator.py` (217 lines)
- `database_ingestion_record.py` (224 lines)

**Recommendation**: Merge inspector + normalizer + reducer into a single
`reduced_ebr_toolchain.py` (~500 lines).  Keep template-builder separate
as an authoring utility.  Keep probe and ingestion-record separate
(different concerns).

### Candidate 3.3: `cli.py` (460 lines)

**Why safe**: The CLI is a thin dispatcher.  Each subcommand delegates
to a single function.  The module is growing linearly with each new
subcommand.

**Risk**: Low.

**Recommendation**: Accept the linear growth.  Do not split until the
CLI exceeds ~800 lines or subcommands need shared logic.

## Debug-Only Diagnostics

**Files**: `reports/h5_report.py`, `reports/csv_report.py`,
`analysis/symmetry_adapted_valley_report.py`,
`analysis/subspace_representation_quality.py`,
`analysis/hsp_star_conjugation.py`,
`analysis/hsp_star_derived_characters.py`,
`analysis/target_subspace_closure.py` (dual-purpose: physics + debug output).

**Physical contracts protected**:
- Debug outputs are behind `output.profile: debug`
- No debug file affects readiness gates

### Candidate 4.1: Debug JSON output files (10+ files)

**Why safe**: All are written only in `debug` profile.  Internal
computations for readiness gates run regardless; only file emission
is gated.

**Risk**: Low.  Verified by output profile contract tests.

**Recommendation**: No reduction needed.  The profile gate is sufficient.

## Test Surface

### Summary

| Test File | LOC | Tests | Category |
|-----------|-----|-------|----------|
| `test_io_and_workflow.py` | 5107 | ~130 | Catch-all: config, workflow, output, summary, EBR, ingestion, benchmarks |
| `test_reduced_ebr_mapping.py` | 1539 | 112 | Table validation, solver, basis gate, classifier, CLI, package data, doc contracts |
| `test_irreptables_table_builder.py` | 920 | ~30 | Builder, spec, preflight, inspector, C3 audit contracts |
| `test_symmetry.py` | 1808 | ~50 | Space group, little group, valley preservation, rotation eigenvalues |
| `test_symmetry_adapted_valley_report.py` | 844 | ~20 | SA report structure and diagnostics |
| Remaining 22 files | ~3000 | ~367 | Physics-focused unit tests |

### Candidate T.1: Split `test_io_and_workflow.py` (5107 lines)

**Why safe**: The file mixes at least 8 distinct test categories.  Splitting
along category boundaries preserves all assertions and coverage.

**Risk**: Low.  Pure test-file reorganisation.

**Proposed split**:
- `test_config.py` — config parsing tests (~18 tests)
- `test_analyze_hsp_outputs.py` — workflow output tests (~12 tests)
- `test_output_profile.py` — standard/debug profile tests (~10 tests)
- `test_ebr_pipeline.py` — EBR input/instance/bundle tests (~12 tests)
- `test_reduced_ebr_smoke.py` — E2E smoke tests (~5 tests)
- `test_database_ingestion.py` — ingestion record tests (~10 tests)
- `test_benchmark_docs.py` — benchmark doc contract tests (~8 tests)
- `test_provenance.py` — provenance propagation tests (~5 tests)
- `test_phase_tables.py` — phase table data contract tests (~5 tests)

**Recommendation**: Split after Layer 3 C3 convention decision resolves.

### Candidate T.2: `test_reduced_ebr_mapping.py` duplicate assertions

**Why safe**: Many table-validation tests check individual error messages
(`vector length`, `nonnegative`, `unique`).  A previous slimming pass
reduced from 247 to 222 tests; further parameterization could reduce by
another ~15 tests.

**Risk**: Low.

**Recommendation**: Parameterize the remaining table-validation negative
cases into a single data-driven test.

## Non-Goals And Blockers

| Item | Status |
|------|--------|
| C3 convention review | BLOCKED — external/manual decision needed before any C3 table data |
| `irrep` runtime adapter | BLOCKED — depends on C3 convention resolution |
| Reduced EBR package-data tables | BLOCKED — no reviewed external table exists |
| OR-Tools / `irrep.ebrs` integration | NEVER — explicit non-goal |
| `irrep2` as dependency | NEVER — private reference-only |

## Verification

Commands run on `cc/codebase-refinement-audit` after the docs-only audit
change.  No production code or tests were modified during this audit:

```bash
$ python -m pytest tests/test_irreptables_table_builder.py tests/test_irrep_data_normalizer.py -q
...............................................................          [100%]
64 passed in 4.75s

$ python -m pytest -q
........................................................................ [ 91%]
............................................................             [100%]
709 passed in 9.88s

$ git diff --check HEAD
(clean — no output)

$ python scripts/check_agent_protocol.py
agent protocol check passed
```
