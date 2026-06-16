# Generic Valley-Projected Irrep Backend Reset — Phase 1

Date: 2026-06-16 | Status: Phase 1 implementation

## Classification Of Relevant Modules

| Module | Classification | Rationale |
|--------|---------------|-----------|
| `valley_little_group.py` | **core** | HSP little group, valley-preserving subgroup, standard-group matching — physical primitives |
| `symmetry_adapted_valley_report.py` | **core** (with prototype_legacy fields) | Orbit-level and subspace-level symmetry-adapted analysis; `subspace_group_candidate` inference is proto_legacy |
| `valley_irrep_matching.py` | **prototype_legacy** | C2/C3 phase-table matching is prototype; kept as `legacy_phase_table` fallback |
| `ebr_input_candidates.py` | **core** | Aggregation of trusted irreps into EBR candidates — generic filter, no Cn dependency |
| `ebr_problem_instances.py` | **core** (with proto_legacy policy removed) | Problem instance grouping; hard-coded `_EXPECTED_HSP` policy demoted to explicit legacy wrapper |
| `ebr_export_bundle.py` | **core** | Export packaging — generic, no Cn dependency |
| `irrep_runtime_reducer.py` | **core** | Multiplicity-aware reduction of 3D EBR source to sampled-HSP basis |
| `irreptables_runtime_table_builder.py` | **core** | Bilbao-source reduced-table builder and mapping-spec driver |
| `reduced_ebr_mapping.py` | **core** | Layer 3 exact-integer reduced EBR mapping interface |
| `reduced_ebr_solver.py` | **core** | Smith normal form + bounded nonnegative search |
| `valleyscope/data/valley_irreps/*` | **prototype_legacy** | Spinful C2/C3 phase tables — validated legacy data, kept as fallback |
| `valleyscope/data/reduced_ebr/*` | **core** | Reviewed package-data reduced EBR table catalog |

## C2/C3 Legacy Dependencies Status

### Kept As Prototype/Fallback Validation

- `spinful_C3_phase_v1.json` / `spinful_C2_phase_v1.json` phase tables
- `match_valley_irrep()` — C2/C3 phase matching, renamed to `legacy_phase_table_matching`
- `_candidate_matches_order()` — C3_like/C2_like ordering policy
- All existing C2/C3 phase table tests — regression fixtures

### Production Dependencies Removed Or Demoted

- `ebr_problem_instances._EXPECTED_HSP`: hard-coded C3_like/C2_like HSP policy → explicit legacy wrapper, new default no_policy
- `symmetry_adapted_valley_report._build_subspace_group()`: `subspace_group_candidate` → kept as `legacy_subspace_group_candidate`
- `valley_irrep_matching.build_valley_irrep_matching_report()`: `tables_implemented` → now `legacy_tables_implemented`
- `summary_report._render_*()` functions: `subspace_group_candidate` display → now shows `subspace_space_group` as primary, `legacy_subspace_group_candidate` as hint

## New Primitive

`valleyscope/analysis/valley_projected_representation.py` provides the
generic `build_valley_projected_representation_report()` that extracts
already-computed physical objects (space group, HSP little group,
valley-preserving subgroup, characters, eigenphases, readiness) into a
serializable report with `subspace_space_group` as the primary identifier.

C2_like/C3_like hints are preserved under `legacy_subspace_group_candidate`
and must not be treated as final physical labels in public text.
