# C2 MM/M3 External Dry-Run Smoke

Date: 2026-06-15 | Status: Dry-run smoke (not package data; not a decomposition report)

## Scope

This document records a dry-run external mapping spec build and validation
for the single candidate C2 row at MM/M3.  It is a builder-exercise smoke,
not a reviewed package-data table and not a reduced EBR decomposition
claim for any material.

## Dry-Run Spec

`docs/reduced_ebr_c2_mm_m3_external_mapping_spec_dry_run.json`:

- source data: `irreptables` / Bilbao SG 149 (P312) spinful
- sampled HSP: `["MM"]`
- allowed keys: `MM:C2_spinor_phase_-1/4`, `MM:C2_spinor_phase_+1/4`
- source labels: only `-M3` and `-M4` mapped to MM with multiplicity 1
- non-sampled labels (GammaM, A, H, KM, L) have HSP entries but omit
  multiplicity maps (filtered before reduced basis)

## Build Result

The spec passes preflight validation against the real SG 149 source basis
and builds a valid reduced external table with 2 irrep keys, 18 EBR
vectors of length 2.

Verified by `test_c2_mm_m3_dry_run_spec_and_build` in
`tests/test_irreptables_table_builder.py`.

## Non-Goals

- Not a reviewed C2 package-data table.
- Not wired into `analyze-hsp`.
- Does not claim a reduced EBR decomposition for any validation fixture.
- Material names absent from spec and test.

## Mapping E2E Smoke

`test_c2_mm_m3_dry_run_mapping_e2e_solved_exact` builds the temporary
dry-run table, constructs a synthetic ready export bundle with vector
`[1, 1]` (one copy of each C2 spinor phase key at MM), and calls
`build_reduced_ebr_mapping()`:

- `mapping_status`: `solved_exact`
- `classification`: `atomic-compatible-candidate`
- `integer_span_status`: `in_integer_span`
- `nonnegative_solution_status`: `solved_exact`

The synthetic bundle does not correspond to any real material.

## CLI Build/Map E2E Smoke

`test_c2_mm_m3_dry_run_cli_build_and_map_e2e` exercises the full
external user CLI path:

```text
inspect-ebr-source --space-group-number 149 --spinful
  -> build-reduced-ebr-table --source-basis (dry-run spec)
  -> map-reduced-ebr (synthetic bundle + generated table)
```

Result: `mapping_status: solved_exact`, `classification:
atomic-compatible-candidate`, `irrep_vector: [1, 1]`.
