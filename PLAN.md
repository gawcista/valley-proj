# PLAN.md

This roadmap reflects the current ValleyScope methodology.

## Current Code Baseline (updated 2026-06)

The repo already has:

**Layer 1 — valley projection + diagnostics:**
* q-cut valley projection and seed projector construction;
* multi-valley basis diagnostics using projected seed matrices;
* valley orbit and valley mapping logic;
* moire HSP symmetry representation matrices D_raw;
* seed projector symmetry-consistency diagnostic (`projector_symmetry_report.json`);
* target-subspace closure provenance diagnostics (`target_subspace_closure.json`);
* subspace representation quality diagnostics (embedded in formal reports;
  standalone `subspace_representation_quality.json` is optional/default-off).

**Layer 2 — gated irrep workflow:**
* irrep workflow decision layer (`direct_qcut` / `symmetry_adapted` / `blocked`);
* formal symmetry-adapted valley analysis (P_a^sym, integrated);
* valley-preserving character/eigenphase diagnostics;
* minimal valley-preserving irrep matching (spinful C3, C2 with internal tables);
* HSP-star conjugation and derived character layer;
* EBR input candidates, problem instances, and export bundle.

**Layer 3 — external solver interface:**
* default-off reduced EBR mapping interface (exact integer, external table only);
* ValleyScope-native reduced EBR solver API
  (`valleyscope.analysis.reduced_ebr_solver`): Smith normal form integer-span
  check plus bounded nonnegative search, with no OR-Tools dependency;
* runtime source normalizer/reducer for package-style 3D EBR data to
  ValleyScope sampled-HSP, valley-preserving reduced tables;
* offline `valleyscope inspect-ebr-source` source-basis inspector for
  mapping-spec authoring; it reports source irrep labels/degeneracies only and
  does not infer moire HSPs or ValleyScope valley-preserving irrep keys;
* offline `valleyscope scaffold-spec` / `valleyscope validate-spec`
  authoring aids for non-buildable mapping-spec templates and preflight
  source-basis coverage checks; they do not infer HSPs or ValleyScope
  valley-preserving irrep keys;
* offline `irreptables` table builder with explicit source-irrep/HSP/key maps
  and provenance, plus `valleyscope build-reduced-ebr-table` for canonical
  mapping-spec driven table generation; optional `--source-basis` preflight
  validates the human-authored mapping before build; no raw 3D decomposition
  call and no `analyze-hsp` wiring;
* material-independent C3-like reduced EBR authoring audit
  (`docs/reduced_ebr_c3_authoring_audit.md`) documenting the source 3D irrep
  labels, HSP little group / valley-preserving subgroup boundary, valley
  sewing-matrix boundary, and pending human review decisions; no table data is
  shipped; the phase-convention audit shows public
  `irreptables.ebrs.load_ebr_data` does not expose C3 eigenphase/character
  data, and the source-irrep restriction feasibility audit shows public
  `irreptables.irreps` has no SG 150 character-table data while
  `irrep.spacegroup_irreps` requires explicit structure/symmetry inputs and
  manual valley-preserving C3 subgroup restriction; source labels are not
  review-ready from package data alone; the human C3 convention decision
  packet records the required review formulas, per-label evidence needs, and
  provenance requirements;
* OR-Tools / `irrep.ebrs` raw 3D decomposition is not required by the portable
  ValleyScope core path; unsafe native optional probes are opt-in only;
* `valley_reduced_ebr_mapping.json` (only when `analysis.reduced_ebr.enabled`).

**High-throughput database interface:**
* explicit offline database ingestion record collector
  (`valleyscope collect-database-record`);
* `database_ingestion_record.json` is built from public outputs only and is not
  a default `analyze-hsp` output.

**Output and summary:**
* `valley_summary.txt` / `valley_summary.json` (main user entry);
* `valley_ebr_export_bundle.json` (downstream EBR entry);
* `valley_reduced_ebr_mapping.json` (default-off, when analysis.reduced_ebr.enabled);
* `database_ingestion_record.json` (explicit offline collector output for
  high-throughput indexing; not controlled by `output.profile`);
* `valley_weights.csv` (quick-scan file, standard profile);
* Output controlled by `output.profile: standard | debug` (standard = public only,
  debug = full diagnostics). Legacy `output.write_detailed_files` is deprecated.
* Debug/detail outputs (debug profile only): diagnostics.h5, valley_subspace.json,
  symmetry_report.json, symmetry_eigenvalues.csv, valley_basis_transform.h5,
  projector_symmetry_report.json, symmetry_adapted_valley_analysis.json,
  target_subspace_closure.json, hsp_star_conjugation.json,
  hsp_star_derived_characters.json, subspace_representation_quality.json,
  irrep_workflow_decisions.json, valley_irrep_matching.json,
  valley_ebr_input_candidates.json, valley_ebr_problem_instances.json,
  folded_center_report.json, sampled_k_coverage.json.

## Methodology

For each moire HSP k, define the target DFT band subspace

```math
H_k = \mathrm{span}\{|\psi_{n,k}\rangle\}.
```

Inside this subspace:

1. `P_a^0` is the q-cut valley seed projector.
2. `D_g` is the moire HSP symmetry representation matrix.
3. `pi_g(a)` is the operation-induced valley mapping.
4. The seed diagnostic checks

   ```math
   D_g P_a^0 D_g^\dagger \approx P_{\pi_g(a)}^0 .
   ```

5. `P_a^sym` is the symmetry-adapted valley projector, built from seed
   averaging and spectral purification. It is used when the seed basis fails
   the projector symmetry check but P_a^sym construction is feasible (the
   `symmetry_adapted` workflow path).

6. The `direct_qcut` path bypasses P_a^sym when the q-cut seed basis already
   passes all readiness gates (tMoTe2-like clean systems).

### Projector modes

Two projector-center modes (`projection.projector_mode`). This is
momentum-space parent-valley projection, not full Bloch-state unfolding.

- **`fixed_center`** (default): Local fixed-valley-point diagnostic.
  `q = k_M + G_M` is compared to fixed monolayer valley centers `Q_a`.
- **`k_resolved_parent_valley`**: Momentum-space parent-valley diagnostic.
  `Q_a` is folded into the moire BZ to obtain `k_a^fold`; dynamic centers
  `Q_a(k_M) = Q_a + (k_M - k_a^fold)` are used for each sampled moire k_M.
  Deprecated alias: `folded_family`.

**Readiness boundary**: `k_resolved_parent_valley` is weight/report-only.
Seed projector matrices and all irrep/EBR readiness gates use fixed-center
projectors regardless of `projector_mode`.

Do not treat high q-cut valley purity as irrep readiness. Do not test
valley-changing operations as invariance of a single label operator.

## Phase Completion Status

| Phase | Description | Status |
|-------|------------|--------|
| Phase 0 | Freeze old interpretation (q-cut seed = diagnostic) | Done |
| Phase 1 | Seed projector symmetry-consistency diagnostics | Done |
| Phase 2 | Symmetry-adapted valley projector prototype | Done — integrated into production workflow |
| Phase 3 | Symmetry-adapted valley-irrep workflow | Done — with direct_qcut gated path |
| Phase 4 | Real benchmarks (tMoTe2, tZrSe2) | Smoke tests pass; blockers documented |
| Phase 5 | EBR input/candidate/problem/export pipeline | Done |
| Phase 6 | Reduced EBR mapping interface | Default-off external-table solver exists; no built-in tables |
| Phase 7 | Parent-valley / k-resolved projector diagnostics | Done — merged to main |
| Phase 8 | Output contract cleanup (standard/debug profiles) | Done — merged to main |
| Phase 9 | High-throughput single-run ingestion record | Done — explicit offline collector, public-output only |
| Phase 10 | Valley-irrep phase table data contract | Done — spinful C3/C2 phase tables are validated package data |
| Phase 11 | Runtime EBR source reduction adapter | Done — normalizer/reducer plus offline `irreptables` builder, external tables only |
| Phase 12 | ValleyScope-native reduced EBR solver API | Done — extracted pure Python / SymPy solver, no OR-Tools |

## Real Benchmarks

**tMoTe2 K/Kp** (P321):
- Clean C3 valley-preserving subspaces with `direct_qcut` / `trusted` readiness.
- GammaM and KM produce trusted C3-like irrep labels.
- MM has only identity in the HSP little group → blocked.

**tZrSe2 M-star** (P312):
- P2-like single-valley subspaces identified at space-group level.
- Current blockers: `spinor_convention_unverified`, low seed overlap at GammaM,
  D_raw target-subspace closure at ~1.9e-2 (usable_with_caution).
- M1/M2 non-identity C2 characters require HSP-star derivation from MM M3
  source. Source is diagnostic_only until closure/seed quality improves.
- No trusted C2 EBR instances exist yet.

## Open Work

* ~~Freeze public schema~~ — Done. `docs/schema.md` is the authoritative frozen
  public schema. `output.profile: standard` (default) emits only public
  user-facing outputs; debug/detail files require `output.profile: debug`.
* ~~Build benchmark matrix~~ — Done. `docs/benchmarks/benchmark_matrix.md`
  records tMoTe2 and tZrSe2 fixture status. tPdSe2/PdSe2 is deferred until
  evidence exists.
* **Benchmark matrix as regression anchor**: keep `docs/benchmarks/benchmark_matrix.md`
  up to date with current fixture status so that output-contract regressions
  are caught by standard-profile smoke tests.
* Spinor convention benchmark verification for tZrSe2-like fixtures
  (blocker B1 in `docs/benchmarks/tzrse2_blocker_evidence.md`).
* Expanded-band HDF5 to test D_raw closure sensitivity for tZrSe2 M3 C2
  (blocker B2).
* ~~Design `irrep2`-like reduced-dimensional irrep/EBR data model~~ — Done.
  `docs/reduced_dimensional_irrep_ebr_data_model.md` covers physical objects,
  label conventions, package-data layout, and validation rules.  The public
  Python package `irrep` may be used as a runtime 3D irrep/EBR data source,
  but final ValleyScope output must be reduced-dimensional and
  valley-preserving, not raw 3D `irrep` decomposition.
* ~~`irrep` / `irreptables` runtime source adapter boundary~~ — Done.  The
  implemented path normalizes package-style 3D EBR data, applies explicit
  sampled-HSP and valley-preserving key maps, filters zero reduced EBR vectors
  with provenance, and builds ValleyScope external reduced tables from
  canonical mapping specs via `valleyscope build-reduced-ebr-table`, with
  optional `--source-basis` preflight before table construction.
  `valleyscope inspect-ebr-source`, `valleyscope scaffold-spec`, and
  `valleyscope validate-spec` provide deterministic authoring aids for the
  public source basis and mapping specs, but do not infer HSPs or ValleyScope
  irrep keys.  It remains offline/library-only and is not wired into
  `analyze-hsp`.  The adapter must use the `irreptables.ebrs.load_ebr_data`
  data path, not `irrep.ebrs` raw 3D decomposition or OR-Tools.
* First reviewed C3-like external table: pending human review of the
  source-irrep/HSP/ValleyScope-key mapping spec documented in
  `docs/reduced_ebr_c3_authoring_audit.md`; `-GM5` and `-K5` currently have
  independent ValleyScope evidence but remain `needs_human_review`, while
  degenerate source labels such as `-K6` are blocked until the source irrep is
  restricted/decomposed into the valley-preserving C3 subgroup. Do not ship
  package-data table until that mapping, restriction convention, and
  provenance are reviewed. The next physics step is an external/manual C3
  character-table convention review using the decision packet in
  `docs/reduced_ebr_c3_authoring_audit.md`, not more automatic table
  generation.
* ~~Package-data skeleton~~ — Done. `valleyscope/data/reduced_ebr/` exists
  with empty manifest, README, and catalog module. No table data shipped.
* ~~Loader integration~~ — Done. `catalog.py` validates manifests, routes
  `load_reviewed_reduced_ebr_table()` through `load_reduced_ebr_table()`,
  and rejects path traversal. No built-in tables shipped.
* High-throughput database pipeline beyond single-run ingestion:
  benchmark-generated ingestion-record regression anchors, then multi-run
  manifest/index collection if needed.
