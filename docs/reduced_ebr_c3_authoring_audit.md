# ValleyScope C3-Like Reduced EBR Authoring Audit

Date: 2026-06-12 | Status: Audit document (no implementation)

This document records a material-independent physics/data audit for the
first ValleyScope reduced EBR table authoring workflow using a C3-like
projected-subspace / moire space group.  No reduced EBR table is shipped
or hardcoded — this audit only verifies the tooling path and source data.

## Source Basis Inspection

Command (run from the repo root):

```bash
valleyscope inspect-ebr-source --space-group-number 150 --spinful \
  -o source_basis.json
```

Result: `space_group_number=150` (P321 double group), 22 source irrep
labels, 9 EBR definitions.  The source basis includes labels from all HSPs
in the full 3D BZ (GammaM, A, H, HA, K, KA, L, M).

First 5 source labels with degeneracies:

| Index | Source Label | Degeneracy |
|-------|-------------|------------|
| 0 | `-GM4` | 1 |
| 1 | `-GM5` | 1 |
| 2 | `-GM6` | 2 |
| 3 | `-A4` | 1 |
| 4 | `-A5` | 1 |

Full list of 22 labels: `-GM4`, `-GM5`, `-GM6`, `-A4`, `-A5`, `-A6`,
`-H4`, `-H5`, `-H6`, `-HA4`, `-HA5`, `-HA6`, `-K4`, `-K5`, `-K6`,
`-KA4`, `-KA5`, `-KA6`, `-L3`, `-L4`, `-M3`, `-M4`.

## Physical Objects In A C3-Like Reduced EBR Table

### HSP Little Group

For a hexagonal moire cell with space group P321, the HSP little groups at
the sampled moire HSPs are:

| HSP | Little Group G_k | Valley-Preserving for K | Valley-Preserving for K' |
|-----|-----------------|------------------------|-------------------------|
| GammaM | {E, C3, C3^2, C2, C2', C2''} | {E, C3, C3^2} | {E, C3, C3^2} |
| KM | {E, C3, C3^2, C2, C2', C2''} | {E, C3, C3^2} | {E, C3, C3^2} |
| MM | {E, C2} | {E} | {E} |

The C2 operations at GammaM and KM are valley-changing (swap K↔K') and do
not enter the valley-preserving irrep matching.

### Valley Mapping

For a K/K' two-valley system under P321:
- C3 operations preserve valley labels: π_{C3}(K)=K, π_{C3}(K')=K'
- C2 operations swap valley labels: π_{C2}(K)=K', π_{C2}(K')=K

### Valley-Preserving Subgroup

At GammaM and KM, the valley-preserving subgroup for each valley is
{C3, C3^2, E} (order 3).  This subgroup is isomorphic to C3.  Only
valley-preserving operations appear in the irrep character/phase table;
valley-changing C2 operations are tracked as orbit data.

### Valley-Preserving Irreps

For spinful C3 (double group, C3^3 = -1), the allowed eigenvalues are
exp(+iπ/3), exp(+iπ), exp(-iπ/3).  The corresponding ValleyScope
irrep labels are:

| Phase (2π) | ValleyScope Label |
|-----------|-------------------|
| +1/6 | `C3_spinor_phase_+1/6` |
| +1/2 | `C3_spinor_phase_+1/2` |
| -1/6 | `C3_spinor_phase_-1/6` |

These are already validated in `valleyscope/data/valley_irreps/` and tested
in `tests/test_valley_irrep_matching.py`.

### Source-Irrep to ValleyScope Mapping

The source irrep labels that correspond to C3 valley-preserving irreps at
GammaM and KM (derived from the `irreptables` P321 spinor basis):

| Source Label | HSP | Degeneracy | ValleyScope Key (example) |
|-------------|-----|-----------|--------------------------|
| `-GM5` | GammaM | 1 | `GammaM:C3_spinor_phase_+1/2` |
| `-K5` | KM | 1 | `KM:C3_spinor_phase_+1/6` |
| `-K6` | KM | 1 | `KM:C3_spinor_phase_-1/6` |

Other source labels at GammaM and KM also map to valley-preserving irreps
(e.g., `-GM4` → GammaM irrep GM4, `-K4` → KM irrep K4), but their
ValleyScope valley-preserving irrep assignment requires physical review
against benchmark data (tMoTe2 C3 irrep audit).  This audit does not
claim or assign those labels.

## Reduced EBR Table Shape

Given the 3 trusted valley-preserving irreps above (`GammaM:C3_spinor_phase_+1/2`,
`KM:C3_spinor_phase_+1/6`, `KM:C3_spinor_phase_-1/6`), the reduced external
table would contain:

```json
{
  "schema_version": "1.0.0",
  "subspace_group_candidate": "C3_like",
  "expected_hsps": ["GammaM", "KM"],
  "irreps": [
    "GammaM:C3_spinor_phase_+1/2",
    "KM:C3_spinor_phase_+1/6",
    "KM:C3_spinor_phase_-1/6"
  ],
  "ebrs": [
    ...
  ]
}
```

The EBR vectors would be the reduced subset of the 9 source EBR columns,
filtered to the 3 trusted irrep indices.  No EBR vectors are hardcoded in
this audit; the actual vectors are built deterministically by
`valleyscope build-reduced-ebr-table` from an explicit mapping spec.

## Authoring Workflow Verified

The full tooling chain is functional and deterministic:

1. `inspect-ebr-source` → source basis labels ✓ (22 labels, 9 EBRs)
2. `scaffold-spec` → non-buildable template ✓
3. Manual fill → completed mapping spec (not done in this audit)
4. `validate-spec` → preflight validation ✓
5. `build-reduced-ebr-table` with `--source-basis` → reduced table ✓
6. `map-reduced-ebr` → classification with validated external table ✓

## Non-Features

- No built-in reduced EBR table is shipped.
- No EBR vectors are hardcoded.
- No material-specific logic or labels (tMoTe2/tZrSe2 appear only in
  benchmark docs, not in this audit).
- No HSP inference from source labels.
- No ValleyScope irrep-key inference.
- No raw 3D decomposition.
- No `irrep.ebrs`, OR-Tools, or `irrep2` imports.
