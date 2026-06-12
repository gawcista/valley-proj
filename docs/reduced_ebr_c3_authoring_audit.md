# ValleyScope C3-Like Reduced EBR Authoring Audit

Date: 2026-06-12 | Status: Audit document (no implementation)

This document records a material-independent physics/data audit for the
first ValleyScope reduced EBR table authoring workflow using a C3-like
projected-subspace / moire space group.  No reduced EBR table is shipped
or hardcoded — this audit only verifies the tooling path and source data.

## Source 3D Irrep Labels From `irreptables`

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

For a C3-like reduced EBR table, the HSP little group must be taken from the
ValleyScope HSP little-group inventory for the sampled moire HSPs.  The
reduced EBR vector basis does not use the full valley orbit; it uses only the
valley-preserving subgroup for each valley.

Current C3-like authoring target:

| HSP | Role In Reduced Table | Valley-Preserving Subgroup |
|-----|-----------------------|----------------------------|
| GammaM | required sampled HSP | {E, C3, C3^2} |
| KM | required sampled HSP | {E, C3, C3^2} |
| MM | optional / not in C3 reduced basis | identity-only |

MM identity-only rows are not part of the required C3 reduced EBR vector
basis.

Any C2 operation that appears in a run's HSP little-group inventory and maps
K to K' is valley-changing sewing data.  It must not enter the
valley-preserving C3 irrep basis.

### Valley Mapping

For a K/K' two-valley system under P321:
- C3 operations preserve valley labels: π_{C3}(K)=K, π_{C3}(K')=K'
- C2 operations swap valley labels: π_{C2}(K)=K', π_{C2}(K')=K

### Valley-Preserving Subgroup

At GammaM and KM, the valley-preserving subgroup for each valley is
{C3, C3^2, E} (order 3).  This subgroup is isomorphic to C3.  Only
valley-preserving operations appear in the irrep character/phase table;
valley-changing operations are tracked as orbit data.

### Valley-Changing Operation And Valley Sewing Matrix

A valley-changing operation belongs to the HSP little group for a sampled
HSP but maps one valley label to another.  For example, a C2 operation can
map K to K'.  This operation does not contribute a valley-preserving
character for a single valley.

The corresponding object is the valley sewing matrix:

```text
S_g: H_{k,a} -> H_{k,pi_g(a)}
```

It records how the valley-a subspace is connected to the valley-pi_g(a)
subspace.  The valley sewing matrix is required for a full valley-orbit
representation, but it is not an entry of the single-valley reduced EBR
vector basis.

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

### Candidate Source-Irrep to ValleyScope Mapping Decisions

The source irrep labels below are candidate rows for a C3-like mapping spec.
They are not a reviewed reduced EBR table and do not, by themselves, justify
shipping built-in data.  A human review must verify the source convention,
the HSP little group convention, and the spinful C3 eigenphase convention
before these rows can be used as packaged data.

| Source Label | HSP | Degeneracy | ValleyScope Key (example) |
|-------------|-----|-----------|--------------------------|
| `-GM5` | GammaM | 1 | `GammaM:C3_spinor_phase_+1/2` |
| `-K5` | KM | 1 | `KM:C3_spinor_phase_+1/6` |
| `-K6` | KM | 1 | `KM:C3_spinor_phase_-1/6` |

Other source labels at GammaM and KM also belong to the source 3D irrep
basis (for example, `-GM4` and `-K4`), but their ValleyScope
valley-preserving irrep assignment requires independent physical review.
This audit does not claim or assign those labels.

## Reduced EBR Vector Basis

If a reviewed mapping spec selects the three candidate ValleyScope keys above
(`GammaM:C3_spinor_phase_+1/2`, `KM:C3_spinor_phase_+1/6`,
`KM:C3_spinor_phase_-1/6`), the reduced external table would contain:

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
filtered to the reviewed irrep indices.  No EBR vectors are hardcoded in
this audit; the actual vectors are built deterministically by
`valleyscope build-reduced-ebr-table` from an explicit mapping spec after
that spec passes source-basis preflight.

## Authoring Workflow Status

The current status of the material-independent authoring workflow is:

1. `inspect-ebr-source` → source basis labels ✓ (22 labels, 9 EBRs)
2. `scaffold-spec` → non-buildable template available
3. Manual fill → pending human review of HSP and ValleyScope irrep-key maps
4. `validate-spec` → pending completed reviewed spec
5. `build-reduced-ebr-table --source-basis` → pending reviewed spec
6. `map-reduced-ebr` → pending validated external table

## C3 Phase Convention Evidence

### Available Data From Public `irreptables` API

The irreptables `load_ebr_data(150, True)` API provides:

- `basis.irrep_labels`: 22 source irrep label strings (e.g. `-GM5`)
- `basis.degeneracies`: integer degeneracy per label
- `ebrs[].ebr_name`: EBR label
- `ebrs[].wyckoff_position`: Wyckoff position
- `ebrs[].irrep_list`: which source irreps each EBR contributes to
- `ebrs[].vector`: integer multiplicity vector aligned to `irrep_labels`
- `smith_form.u/r/v`: precomputed Smith normal form matrices

**No C3 eigenphase/character data is exposed.** The irreptables API does
not provide irrep character tables, eigenphase decompositions, or C3
rotation eigenvalue data.  Source irrep labels like `-GM5` are opaque
string identifiers with no attached phase convention documentation.

### GammaM Source Labels

| Source Label | Deg | HSP Little Group | Valley-Preserving Subgroup | Known C3 Phase? | Status |
|-------------|-----|-----------------|---------------------------|-----------------|--------|
| `-GM4` | 1 | {E,C3,C3^2} | {E,C3,C3^2} | No | `blocked` — no phase convention evidence from irreptables API |
| `-GM5` | 1 | {E,C3,C3^2} | {E,C3,C3^2} | Yes (ValleyScope irrep matching benchmark: P321 GammaM C3 eigenphase = +1/2 matches `C3_spinor_phase_+1/2`) | `review_ready` |
| `-GM6` | 2 | {E,C3,C3^2} | {E,C3,C3^2} | No | `blocked` — degeneracy 2; not a 1D C3 irrep |

### KM Source Labels

| Source Label | Deg | HSP Little Group | Valley-Preserving Subgroup | Known C3 Phase? | Status |
|-------------|-----|-----------------|---------------------------|-----------------|--------|
| `-K4` | 1 | {E,C3,C3^2} | {E,C3,C3^2} | No | `blocked` — no phase convention evidence from irreptables API |
| `-K5` | 1 | {E,C3,C3^2} | {E,C3,C3^2} | Yes (phase +1/6 matches `C3_spinor_phase_+1/6`) | `review_ready` |
| `-K6` | 2 | {E,C3,C3^2} | {E,C3,C3^2} | No | `blocked` — degeneracy 2; not a 1D C3 irrep |

### KA Source Labels (Excluded From C3 Reduced Basis)

| Source Label | Deg | HSP Resolution | Status |
|-------------|-----|---------------|--------|
| `-KA4` | 1 | Maps to HSP KA, not in {GammaM, KM} | `blocked` — not in sampled HSP set for C3 reduced basis |
| `-KA5` | 1 | Same | `blocked` |
| `-KA6` | 2 | Same | `blocked` |

### Blocker Summary

The public irreptables API does not expose C3 eigenphase or character
decomposition data for individual source irrep labels.  The following
blocks prevent fully automated C3 phase mapping:

1. **No eigenphase data**: `irreptables.load_ebr_data` returns combinatorial
   EBR data (which irreps belong to which EBR) but not the C3 rotation
   eigenvalues of each irrep.  Source labels like `-GM5` cannot be mapped
   to `C3_spinor_phase_+1/2` from irreptables data alone.

2. **Degeneracy > 1**: `-GM6` and `-K6` have degeneracy 2, which violates
   the one-dimensional constraint of the ValleyScope spinful C3 phase
   table.  These are not 1D C3 irreps and cannot be used for single-valley
   valley-preserving EBR input.

3. **Opaque label convention**: Source irrep labels like `-GM4`, `-K4` have
   no documented phase convention in the irreptables API.  The `-4`/`-5`/`-6`
   numbering convention is not a published C3 eigenphase key.

4. **No irrep character table in API**: The irreptables public API provides
   only EBR combinatorial data and Smith normal form; it does not expose
   `SpaceGroupIrreps` or character tables that could be queried for C3
   eigenvalues.

### What CAN Be Mapped (With Independent Evidence)

| Source | ValleyScope Key | Evidence |
|--------|----------------|----------|
| `-GM5` (deg 1) | `GammaM:C3_spinor_phase_+1/2` | ValleyScope irrep-matching benchmark: P321 GammaM C3 eigenphase = +1/2, matches C3_spinor_phase_+1/2.  This is a per-run ValleyScope diagnostic, not an irreptables API guarantee. |
| `-K5` (deg 1) | `KM:C3_spinor_phase_+1/6` | Same benchmark: KM C3 eigenphase = +1/6. |
| `-K6` (deg 2) | Not mappable to 1D C3 irrep | Degeneracy 2; need direct-sum decomposition beyond current matching scope. |

### Conclusion

Only `-GM5` and `-K5` have sufficient independent evidence (via the
ValleyScope P321 irrep matching benchmark) to be mapped to ValleyScope spinful
C3 phase labels.  All other GammaM/KM source labels are blocked by a
combination of missing phase convention data, degeneracy > 1, or
undocumented label conventions in the irreptables API.  A reviewed C3-like
reduced EBR table authoring requires resolving these blockers — either
through an external irrep character table source or through explicit
human review of the `irrep.spacegroup_irreps` module (which imports
cleanly but was not called in this audit).

## Non-Features

- No built-in reduced EBR table is shipped.
- No EBR vectors are hardcoded.
- No material-specific logic or labels.
- No HSP inference from source labels.
- No ValleyScope irrep-key inference.
- No raw 3D decomposition.
- No `irrep.ebrs`, OR-Tools, or `irrep2` imports.
