# tMoTe2 P321 C3 Reduced EBR Mapping Validation

Date: 2026-06-14 | Status: Validation fixture (not a production benchmark claim)

## Scope

This document records a local validation run of the reviewed C3-like
reduced EBR package-data table
`P321_C3_like_GammaM_KM_spinful_v1` against the real tMoTe2 P321
fixture.  tMoTe2 is a validation fixture only — no material-specific
production logic or schema name is used.

## Fixture

- Config: `real_tests/tMoTe2/analyze.yaml` (temporary copy with
  `analysis.reduced_ebr.table_name` added)
- Output: untracked temporary directory
- No fixture files committed

## Configuration Delta From Default

```yaml
analysis:
  reduced_ebr:
    enabled: true
    table_name: P321_C3_like_GammaM_KM_spinful_v1
```

All other parameters (k-points, iband, valley_centers, projector,
symmetry, tol) are the fixture defaults.

## Validation Result

### Export Bundle

- Status: `ready_for_external_solver`
- Bundle count: 2
- Both bundles ready for external solver

| Bundle ID | Valley | Subspace Group | Ready |
|-----------|--------|---------------|-------|
| `bundle_ebr_instance_001` | K_valley | C3_like | true |
| `bundle_ebr_instance_002` | Kp_valley | C3_like | true |

### Reduced EBR Mapping

- Status: `solved_exact`
- Table status: `loaded` (package-data table)
- Excluded bundles: 0

| Bundle | Status | Classification | Integer Span | Decomposition |
|--------|--------|--------------|-------------|---------------|
| `bundle_ebr_instance_001` (K_valley) | `solved_exact` | `atomic-compatible-candidate` | `in_integer_span` | `-E↑G(2) x 1` |
| `bundle_ebr_instance_002` (Kp_valley) | `solved_exact` | `atomic-compatible-candidate` | `in_integer_span` | `-E↑G(2) x 1` |

Both K and K' valleys produce identical atomic-compatible decompositions
with one copy of the `2d(3,3)` EBR `-E↑G(2)`.

### Output Profile

Run used default `output.profile: standard`.  Only public files written:

- `valley_summary.txt`
- `valley_summary.json`
- `valley_weights.csv`
- `valley_ebr_export_bundle.json`
- `valley_reduced_ebr_mapping.json`

No debug/detail files were emitted.

## Caveats

- This is a local fixture validation, not a committed regression test.
- tMoTe2 is used as a P321 C3-like benchmark only; the result is
  recorded as a physically meaningful outcome (both valleys produce
  `atomic-compatible-candidate`), not as a pass/fail gate.
- The `solved_exact` result depends on the tMoTe2 fixture's specific
  band selection and symmetry tolerance; it does not guarantee that
  all P321 systems will produce `solved_exact`.
- The decomposition `-E↑G(2) x 1` for each valley is the minimal
  EBR representation using the reviewed C3 table.  The result is
  physically consistent with the valley-preserving C3 subgroup
  reduction.
