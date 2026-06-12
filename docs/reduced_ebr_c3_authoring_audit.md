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

| Source Label | HSP | Degeneracy | Candidate ValleyScope Key | Status |
|-------------|-----|-----------|---------------------------|--------|
| `-GM5` | GammaM | 1 | `GammaM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-K5` | KM | 1 | `KM:C3_spinor_phase_+1/6` | `needs_human_review` |
| `-K6` | KM | 2 | none yet | `blocked` — requires restriction/decomposition of a degenerate source irrep before it can contribute to a 1D C3 phase basis |

Other source labels at GammaM and KM also belong to the source 3D irrep
basis (for example, `-GM4` and `-K4`), but their ValleyScope
valley-preserving irrep assignment requires independent physical review.
This audit does not claim or assign those labels.

## Reduced EBR Vector Basis

No reviewed reduced EBR vector basis is selected by this audit.  The intended
C3-like ValleyScope basis still has the following target keys, but a buildable
mapping spec requires reviewed source-irrep restrictions before these keys can
be populated from public `irreptables` labels:

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
filtered to reviewed source-irrep restriction data.  No EBR vectors are
hardcoded in this audit; the actual vectors are built deterministically by
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

The public `irreptables.ebrs.load_ebr_data(150, True)` API provides:

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
| `-GM5` | 1 | {E,C3,C3^2} | {E,C3,C3^2} | Independent run evidence suggests +1/2, but not from public irreptables phase data | `needs_human_review` |
| `-GM6` | 2 | {E,C3,C3^2} | {E,C3,C3^2} | No | `blocked` — degeneracy 2; not a 1D C3 irrep |

### KM Source Labels

| Source Label | Deg | HSP Little Group | Valley-Preserving Subgroup | Known C3 Phase? | Status |
|-------------|-----|-----------------|---------------------------|-----------------|--------|
| `-K4` | 1 | {E,C3,C3^2} | {E,C3,C3^2} | No | `blocked` — no phase convention evidence from irreptables API |
| `-K5` | 1 | {E,C3,C3^2} | {E,C3,C3^2} | Independent run evidence suggests +1/6, but not from public irreptables phase data | `needs_human_review` |
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

1. **No eigenphase data**: `irreptables.ebrs.load_ebr_data` returns
   combinatorial EBR data (which irreps belong to which EBR) but not the C3
   rotation eigenvalues of each irrep.  Source labels like `-GM5` cannot be
   mapped to `C3_spinor_phase_+1/2` from irreptables data alone.

2. **Degeneracy > 1**: `-GM6` and `-K6` have degeneracy 2 in the source
   3D irrep basis. They cannot be used directly as one-dimensional
   ValleyScope spinful C3 phase entries without first restricting and
   decomposing the source irrep into the valley-preserving C3 subgroup.

3. **Opaque label convention**: Source irrep labels like `-GM4`, `-K4` have
   no documented phase convention in the irreptables API.  The `-4`/`-5`/`-6`
   numbering convention is not a published C3 eigenphase key.

4. **No irrep character table in API**: The irreptables public API provides
   only EBR combinatorial data and Smith normal form; it does not expose
   `SpaceGroupIrreps` or character tables that could be queried for C3
   eigenvalues.

### Candidate Rows With Independent Evidence

| Source | Candidate ValleyScope Key | Evidence | Status |
|--------|--------------------------|----------|--------|
| `-GM5` (deg 1) | `GammaM:C3_spinor_phase_+1/2` | Independent ValleyScope C3 eigenphase evidence. This is per-run diagnostic evidence, not an irreptables API guarantee. | `needs_human_review` |
| `-K5` (deg 1) | `KM:C3_spinor_phase_+1/6` | Independent ValleyScope C3 eigenphase evidence. | `needs_human_review` |
| `-K6` (deg 2) | none yet | Degeneracy 2; requires source-irrep restriction/decomposition before it can contribute to the 1D C3 phase basis. | `blocked` |

### Conclusion

No source label is review-ready from public `irreptables` EBR data alone.
`-GM5` and `-K5` are candidate rows with independent ValleyScope evidence,
so they are `needs_human_review`, not final table data.  `-K6` is blocked
until the degenerate source irrep is restricted/decomposed into the
valley-preserving C3 basis.  A reviewed C3-like reduced EBR table authoring
requires resolving these blockers through an external irrep character table
source or explicit human review of the relevant source-irrep convention.

## Source-Irrep Restriction Feasibility Audit

### APIs Inspected

| Package | Module | Key Classes | Can Query SG 150 C3 Characters? |
|---------|--------|-------------|-------------------------------|
| `irrep` v2.6.3 | `irrep.spacegroup_irreps` | `SpaceGroupIrreps`, `SpaceGroup`, `IrrepTable` | Partial — requires explicit lattice + symmetry operations + spinor rotations; no `from_sg_number` factory |
| `irreptables` | `irreptables.irreps` | `IrrepTable`, `Irrep`, `KPoint` | No — `IrrepTable(SGnumber, spinor)` parses data files; no data files for SG 150 in `irreptables/data/` |
| `irreptables` | `irreptables.ebrs` | `load_ebr_data` | No — returns combinatorial EBR data only; no character/eigenphase information |

### Inspection Commands

```python
# irrep.spacegroup_irreps: requires lattice + rotations + spinor_rotations
from irrep.spacegroup_irreps import SpaceGroupIrreps
# No simple from_sg_number factory available.

# irreptables.irreps: IrrepTable parser, no SG 150 data files
import irreptables.irreps as ir
# ir.IrrepTable(150, True) -> AssertionError (no data file)

# irreptables.ebrs: EBR-only, no character data
from irreptables.ebrs import load_ebr_data
# load_ebr_data(150, True) -> basis + EBR vectors + Smith form only
```

### Findings

1. **No `from_sg_number` query endpoint**: neither `irrep.spacegroup_irreps`
   nor `irreptables.irreps` provides a simple query-by-space-group-number
   API for C3 eigenphase/character data.

2. **`irreptables.irreps` cannot load SG 150**: the `IrrepTable` constructor
   parses irrep output data files, but `irreptables/data/` does not contain
   SG 150 data files.

3. **`irrep.spacegroup_irreps` is computationally accessible but heavy**:
   it can be constructed from a lattice + spglib symmetry operations, but:
   - requires explicit crystal structure (not just SG number)
   - requires spinor rotation matching via `match_spinor_rotations`
   - outputs full-space-group irreps, not per-HSP valley-preserving restrictions
   - no built-in C3 subgroup restriction function

4. **Character/restriction decomposition not automated**: the API provides
   full space-group irreps, not the C3 subgroup restriction that would
   tell us that `-GM5` decomposes as `C3_spinor_phase_+1/2`.

### Updated Per-Label Status

| Source Label | Deg | Status | Reason |
|-------------|-----|--------|--------|
| `-GM4` | 1 | `blocked` | No public API gives C3 character for this label |
| `-GM5` | 1 | `needs_human_review` | ValleyScope run evidence suggests +1/2; cannot be verified from public package APIs alone |
| `-GM6` | 2 | `blocked` | Degeneracy 2; requires C3 restriction decomposition; no public API provides this |
| `-K4` | 1 | `blocked` | No public API gives C3 character for this label |
| `-K5` | 1 | `needs_human_review` | ValleyScope run evidence suggests +1/6; cannot be verified from public package APIs alone |
| `-K6` | 2 | `blocked` | Degeneracy 2; requires C3 restriction decomposition |
| `-KA4/-KA5/-KA6` | 1/1/2 | `blocked` | Not in sampled HSP set {GammaM, KM} |

### Conclusion

The public `irrep` / `irreptables` APIs do not provide a simple, queryable
C3 character/eigenphase table for SG 150 spinful source irreps.  Source-irrep
to ValleyScope C3 phase mapping requires:

1. Explicit crystal structure + lattice for `SpaceGroupIrreps` construction,
   then manual C3 subgroup restriction — computationally feasible but not
   automated.
2. Or external physics reference (literature-reviewed C3 character tables).

The `irrep.spacegroup_irreps` module is the closest candidate for a future
runtime adapter, but it requires:
- a moire/projected-subspace crystal structure (not just a space group
  number);
- spinor rotation matching (already built into the module);
- explicit C3 subgroup restriction logic (not provided by the module).

This blocker should be resolved before a reviewed C3-like reduced EBR
table can be packaged.

## Non-Features

- No built-in reduced EBR table is shipped.
- No EBR vectors are hardcoded.
- No material-specific logic or labels.
- No HSP inference from source labels.
- No ValleyScope irrep-key inference.
- No raw 3D decomposition.
- No `irrep.ebrs`, OR-Tools, or `irrep2` imports.
