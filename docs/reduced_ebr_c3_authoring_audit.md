# ValleyScope C3-Like Reduced EBR Authoring Audit

Date: 2026-06-12 | Status: Audit document (no implementation)

This document records a material-independent physics/data audit for the
first ValleyScope reduced EBR table authoring workflow using a C3-like
projected-subspace / moire space group.  No reduced EBR table is shipped
or hardcoded — this audit only verifies the tooling path and source data.

## C3 Convention Readiness Summary

No source irrep label is `review_ready` from public package data alone.
Every label is either `needs_human_review` (machine-readable Bilbao-derived
C3 character evidence exists but the ValleyScope reduction convention has not
been signed off) or `blocked_by_missing_restriction_data` (the label requires
decomposition of a degenerate source irrep before it can enter the 1D C3
phase basis).

| Source Label | Deg | HSP | Candidate ValleyScope Key | Status |
|-------------|-----|-----|---------------------------|--------|
| `-GM4` | 1 | GammaM | `GammaM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-GM5` | 1 | GammaM | `GammaM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-K4` | 1 | KM | `KM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-K5` | 1 | KM | `KM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-GM6` | 2 | GammaM | none — degeneracy 2; requires C3 restriction decomposition | `blocked_by_missing_restriction_data` |
| `-K6` | 2 | KM | none — degeneracy 2; requires C3 restriction decomposition | `blocked_by_missing_restriction_data` |
| `-KA4`, `-KA5`, `-KA6` | 1/1/2 | KA | none — not in sampled HSP set {GammaM, KM} | `blocked_by_missing_restriction_data` |

The ValleyScope irrep basis keys for a C3-like reduced table are:

| Phase (2π) | ValleyScope Key | Available In Phase Tables? |
|-----------|-----------------|---------------------------|
| +1/6 | `C3_spinor_phase_+1/6` | Yes — `spinful_C3_phase_v1.json` |
| +1/2 | `C3_spinor_phase_+1/2` | Yes — `spinful_C3_phase_v1.json` |
| -1/6 | `C3_spinor_phase_-1/6` | Yes — `spinful_C3_phase_v1.json` |

The HSP set for a C3-like reduced table is `{GammaM, KM}`.  MM is
identity-only in the C3 valley-preserving subgroup and is not part of the
required C3 reduced EBR vector basis.

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
| `-GM4` | GammaM | 1 | `GammaM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-GM5` | GammaM | 1 | `GammaM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-K4` | KM | 1 | `KM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-K5` | KM | 1 | `KM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-GM6` | GammaM | 2 | none yet | `blocked_by_missing_restriction_data` — requires restriction/decomposition of a degenerate source irrep before it can contribute to a 1D C3 phase basis |
| `-K6` | KM | 2 | none yet | `blocked_by_missing_restriction_data` — requires restriction/decomposition of a degenerate source irrep before it can contribute to a 1D C3 phase basis |

The `-4`/`-5` distinction at GammaM and KM is in C2 characters, not C3
characters.  Those C2 operations are valley sewing data for a single-valley
C3 reduced basis and must not be used as reduced-basis irrep labels.

## Reduced EBR Vector Basis

No reviewed reduced EBR vector basis is selected by this audit.  The C3-like
ValleyScope candidate key universe is shown below, but a buildable mapping
spec requires reviewed source-irrep restrictions before these keys can be
populated from public `irreptables` labels:

```json
{
  "schema_version": "1.0.0",
  "subspace_group_candidate": "C3_like",
  "expected_hsps": ["GammaM", "KM"],
  "irreps": [
    "GammaM:C3_spinor_phase_+1/6",
    "GammaM:C3_spinor_phase_+1/2",
    "GammaM:C3_spinor_phase_-1/6",
    "KM:C3_spinor_phase_+1/6",
    "KM:C3_spinor_phase_+1/2",
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

This EBR-vector API does not expose C3 eigenphase/character data.  The
character data is available through the separate Bilbao-derived irrep-table
API:

```python
from irreptables.irreps import IrrepTable
table = IrrepTable("150", True)  # string SG identifier is required
```

`IrrepTable("150", True)` exposes SG 150 spinful source labels, k-point
labels, symmetry-operation matrices, and complex characters keyed by
1-indexed Bilbao operation IDs.  `IrrepTable(150, True)` with an integer
space-group number fails because this API compares the input to the string
identifier stored in the table file.

### GammaM Source Labels

| Source Label | Deg | HSP Little Group | Valley-Preserving Subgroup | Known C3 Phase? | Status |
|-------------|-----|-----------------|---------------------------|-----------------|--------|
| `-GM4` | 1 | {E,C3,C3^2} | {E,C3,C3^2} | `+1/2` from `IrrepTable("150", True)` | `needs_human_review` |
| `-GM5` | 1 | {E,C3,C3^2} | {E,C3,C3^2} | `+1/2` from `IrrepTable("150", True)` | `needs_human_review` |
| `-GM6` | 2 | {E,C3,C3^2} | {E,C3,C3^2} | op2/op3 chars `(+1,+1)`; needs 2D restriction decomposition | `blocked_by_missing_restriction_data` — degeneracy 2; not a 1D C3 irrep |

### KM Source Labels

| Source Label | Deg | HSP Little Group | Valley-Preserving Subgroup | Known C3 Phase? | Status |
|-------------|-----|-----------------|---------------------------|-----------------|--------|
| `-K4` | 1 | {E,C3,C3^2} | {E,C3,C3^2} | `+1/2` from `IrrepTable("150", True)` | `needs_human_review` |
| `-K5` | 1 | {E,C3,C3^2} | {E,C3,C3^2} | `+1/2` from `IrrepTable("150", True)` | `needs_human_review` |
| `-K6` | 2 | {E,C3,C3^2} | {E,C3,C3^2} | op2/op3 chars `(+1,+1)`; needs 2D restriction decomposition | `blocked_by_missing_restriction_data` — degeneracy 2; not a 1D C3 irrep |

### KA Source Labels (Excluded From C3 Reduced Basis)

| Source Label | Deg | HSP Resolution | Status |
|-------------|-----|---------------|--------|
| `-KA4` | 1 | Maps to HSP KA, not in {GammaM, KM} | `blocked_by_missing_restriction_data` — not in sampled HSP set for C3 reduced basis |
| `-KA5` | 1 | Same | `blocked_by_missing_restriction_data` |
| `-KA6` | 2 | Same | `blocked_by_missing_restriction_data` |

### Blocker Summary

The C3 characters are machine-readable from `IrrepTable("150", True)`, but
the following blocks still prevent shipping a reviewed reduced EBR table:

1. **Degeneracy > 1**: `-GM6` and `-K6` have degeneracy 2 in the source
   3D irrep basis. They cannot be used directly as one-dimensional
   ValleyScope spinful C3 phase entries without first restricting and
   decomposing the source irrep into the valley-preserving C3 subgroup.

2. **Reduction convention review**: Source irrep labels like `-GM4`, `-GM5`,
   `-K4`, and `-K5` now have machine-readable C3 characters, but the
   ValleyScope reduced table still needs a signed-off mapping from source
   label to sampled HSP and ValleyScope irrep key.

3. **Valley-preserving boundary**: Bilbao operation IDs 2 and 3 are the C3
   operations for SG 150.  Operations 4-6 are C2 operations in the full HSP
   little group and must be treated as valley sewing data, not as labels in
   the single-valley C3 reduced basis.

### Candidate Rows With Bilbao C3 Character Evidence

| Source | C3 Character Evidence | Candidate C3 Phase | ValleyScope Key | Status |
|--------|-----------------------|---------------------|-----------------|--------|
| `-GM4` (deg 1) | op2=-1, op3=-1 | +1/2 from op2 | `GammaM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-GM5` (deg 1) | op2=-1, op3=-1 | +1/2 from op2 | `GammaM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-GM6` (deg 2) | op2=+1, op3=+1 | unresolved | none yet | `blocked_by_missing_restriction_data` |
| `-K4` (deg 1) | op2=-1, op3=-1 | +1/2 from op2 | `KM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-K5` (deg 1) | op2=-1, op3=-1 | +1/2 from op2 | `KM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-K6` (deg 2) | op2=+1, op3=+1 | unresolved | none yet | `blocked_by_missing_restriction_data` |

**Note corrected 2026-06-13**: The earlier audit incorrectly mapped `-K5`
to `KM:C3_spinor_phase_+1/6` based on independent ValleyScope eigenphase
runs.  The Bilbao-derived `irreptables.irreps.IrrepTable("150", True)`
character table shows `-K5` has op2 C3 character -1, giving the candidate
phase **+1/2**, not +1/6.  No +1/6 assignment is supported by a 1D source
label in this audit.

### Conclusion

With the `irreptables.irreps` Bilbao character data, the C3 characters for
all six in-scope source labels at GammaM and KM are now machine-checkable via
`IrrepTable("150", True)`.  The key remaining blocker is the C3 subgroup
restriction decomposition for the degenerate labels (`-GM6`, `-K6`), which
requires explicit use of both C3 operation characters and the Bilbao
double-group lift convention before assigning any 1D spinful C3
multiplicities.  A reviewed C3-like reduced EBR table still requires human
review of the mapping convention and provenance before it can be shipped.

## Source-Irrep Restriction Feasibility Audit

### APIs Inspected

| Package | Module | Key Classes | Can Query SG 150 C3 Characters? |
|---------|--------|-------------|-------------------------------|
| `irrep` v2.6.3 | `irrep.spacegroup_irreps` | `SpaceGroupIrreps`, `SpaceGroup`, `IrrepTable` | Partial — requires explicit lattice + symmetry operations + spinor rotations; no `from_sg_number` factory |
| `irreptables` | `irreptables.irreps` | `IrrepTable`, `Irrep`, `SymopTable` | **Yes** — `IrrepTable("150", True)` (string SG identifier) loads Bilbao-derived SG 150 spinful irrep character-table data; 16 irreps with per-operation complex characters |
| `irreptables` | `irreptables.ebrs` | `load_ebr_data` | No — returns combinatorial EBR data only; no character/eigenphase information |

### Inspection Commands

```python
# irrep.spacegroup_irreps: requires lattice + rotations + spinor_rotations
from irrep.spacegroup_irreps import SpaceGroupIrreps
# No simple from_sg_number factory available.

# irreptables.irreps: Bilbao-derived irrep character-table data — VERIFIED
import irreptables.irreps as ir
# ir.IrrepTable("150", True)  -> loads 16 spinful irreps with characters
# NOTE: IrrepTable(150, True) (int) fails — must use string "150"

# irreptables.ebrs: EBR-only, no character data
from irreptables.ebrs import load_ebr_data
# load_ebr_data(150, True) -> basis + EBR vectors + Smith form only
```

### Findings

1. **`irreptables.irreps.IrrepTable("150", True)` loads SG 150 spinful irrep
   character data.**  This is a Bilbao-derived data source with 16 irreps,
   6 symmetry operations, and per-operation complex characters.  The SG
   identifier must be passed as a string — `IrrepTable(150, True)` (int)
   fails with `AssertionError`.  This was the key correction to the earlier
   audit, which incorrectly claimed `irreptables.irreps` could not load
   SG 150.

2. **`irrep.spacegroup_irreps` is computationally accessible but heavy**:
   it can be constructed from a lattice + spglib symmetry operations, but:
   - requires explicit crystal structure (not just SG number)
   - requires spinor rotation matching via `match_spinor_rotations`
   - outputs full-space-group irreps, not per-HSP valley-preserving restrictions
   - no built-in C3 subgroup restriction function

3. **Character/restriction decomposition not automated**: the API provides
   full space-group irreps, not the reviewed C3 subgroup restriction that
   would by itself justify a ValleyScope mapping such as
   `-GM5` -> `C3_spinor_phase_+1/2`.

4. **`irreptables.ebrs.load_ebr_data` provides only EBR combinatorial data**
   (labels, degeneracies, vectors, Smith form) — no character information.

### Updated Per-Label Status (With Bilbao C3 Characters)

| Source Label | Deg | C3 Character Evidence | Candidate C3 Phase (2π) | ValleyScope Key | Status |
|-------------|-----|-----------------------|--------------------------|-----------------|--------|
| `-GM4` | 1 | op2=-1, op3=-1 | +1/2 from op2 | `GammaM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-GM5` | 1 | op2=-1, op3=-1 | +1/2 from op2 | `GammaM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-GM6` | 2 | op2=+1, op3=+1 | unresolved | none yet | `blocked_by_missing_restriction_data` — needs C3 restriction decomposition |
| `-K4` | 1 | op2=-1, op3=-1 | +1/2 from op2 | `KM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-K5` | 1 | op2=-1, op3=-1 | +1/2 from op2 | `KM:C3_spinor_phase_+1/2` | `needs_human_review` |
| `-K6` | 2 | op2=+1, op3=+1 | unresolved | none yet | `blocked_by_missing_restriction_data` — needs C3 restriction decomposition |
| `-KA4/-KA5/-KA6` | 1/1/2 | n/a | n/a | out of scope | `blocked_by_missing_restriction_data` — not in sampled HSP set {GammaM, KM} |

**Key finding corrected 2026-06-13**: The distinction between `-GM4`/`-GM5`
(and `-K4`/`-K5`) is in their **C2 characters**, not their **C3 characters**.
All four 1D labels at GammaM and KM have op2 C3 character = -1 (candidate
phase +1/2).
The earlier audit incorrectly mapped `-K5` to phase +1/6 based on
independent ValleyScope eigenphase runs; the Bilbao character table
shows `-K5` has op2 C3 character -1, giving candidate phase +1/2.  The
+1/6 and -1/6 phases do not come from any 1D source label identified here;
whether they appear in the degenerate-label restrictions remains an explicit
blocker.

### Conclusion

The `irreptables.irreps.IrrepTable("150", True)` path provides the
machine-readable Bilbao character table needed for SG 150 C3 character
evidence.  It does not by itself produce a ValleyScope reduced EBR table:
ValleyScope must still apply the sampled-HSP contract, valley mapping,
valley-preserving subgroup restriction, and human-reviewed provenance before
any package-data table can be marked reviewed.

## Human C3 Convention Review Packet

This section is a structured decision packet for human physics review.
It does not claim any mapping, ship any table data, or implement any
automatic convention inference.  The reviewer must decide each row
explicitly.

### Physics Decision Needed

Choose, for each SG 150 spinful source label below, whether it can be
mapped to a ValleyScope spinful C3 phase label.  The mapping is:

```
source_label  ->  ValleyScope C3 phase label
                    (C3_spinor_phase_+1/6, C3_spinor_phase_+1/2, C3_spinor_phase_-1/6)
```

A decision of "none" means the label is blocked and cannot enter a
reviewed C3-like reduced EBR table.

### Review Formulas

The valley-preserving subgroup for the C3 valley at GammaM and KM is:

```
G_k^{(a)} = { g in G_k | pi_g(a) = a }   (C3-preserving only)
```

For spinful C3 (double group, C3^3 = -1), the allowed 1D eigenvalues are:

```
exp(+i*pi/3)  ->  phase +1/6  ->  C3_spinor_phase_+1/6
exp(+i*pi)    ->  phase +1/2  ->  C3_spinor_phase_+1/2
exp(-i*pi/3)  ->  phase -1/6  ->  C3_spinor_phase_-1/6
```

For a source representation restricted to the C3 valley-preserving
subgroup, the characters `chi(E)`, `chi(C3)`, `chi(C3^2)` decompose
the representation into 1D C3 spinful irreps:

```
chi(E) = d            (total dimension)
chi(C3)  = sum_i n_i * exp(2*pi*i*phase_i)
chi(C3^2) = sum_i n_i * exp(4*pi*i*phase_i)
```

where `phase_i` is measured in units of `2*pi`, and `n_i` is the
multiplicity of each 1D C3 irrep.  For a 1D source label (degeneracy 1),
`d=1` and `chi(C3)` directly gives the C3 phase.

For a degenerate source label (`-GM6`, `-K6` with degeneracy 2), the
character `chi(C3)` is a sum of two phase factors that must be resolved
into individual 1D C3 irreps through character decomposition.

### Decision Table

| Source Label | Deg | Current Status | Needed Evidence | Acceptable Provenance | Can Enter Table Without Review? |
|-------------|-----|---------------|-----------------|----------------------|-------------------------------|
| `-GM4` | 1 | `needs_human_review` | Confirm op2 `chi(C3)=-1` (=candidate phase +1/2) from `IrrepTable("150", True)` and approve the source-label convention | Bilbao-derived `irreptables.irreps` plus optional literature cross-check | No — review required |
| `-GM5` | 1 | `needs_human_review` | Confirm op2 `chi(C3)=-1` (=candidate phase +1/2) from `IrrepTable("150", True)` and approve the source-label convention | Bilbao-derived `irreptables.irreps` plus optional literature cross-check | No — review required |
| `-GM6` | 2 | `blocked_by_missing_restriction_data` | C3 restriction decomposition: use `chi(C3)` and `chi(C3^2)` for this 2D irrep; determine whether it decomposes into 1D C3 irreps under the reviewed double-group convention | Same as above; requires explicit character decomposition | No — requires restriction + decomposition |
| `-K4` | 1 | `needs_human_review` | Confirm op2 `chi(C3)=-1` (=candidate phase +1/2) from `IrrepTable("150", True)` and approve the source-label convention | Bilbao-derived `irreptables.irreps` plus optional literature cross-check | No — review required |
| `-K5` | 1 | `needs_human_review` | Confirm op2 `chi(C3)=-1` (=candidate phase +1/2) from `IrrepTable("150", True)` and approve the source-label convention | Bilbao-derived `irreptables.irreps` plus optional literature cross-check | No — review required |
| `-K6` | 2 | `blocked_by_missing_restriction_data` | C3 restriction decomposition: use `chi(C3)` and `chi(C3^2)` for this 2D irrep | Same as `-GM6`; requires explicit character decomposition | No — requires restriction + decomposition |

### Provenance Requirements For Reviewed External Table

If any row is promoted to a reviewed status by the human reviewer, the
accompanying provenance must record:

1. **Source**: space group number, spinor convention, data source package
   name/version.
2. **Review method**: how the C3 character/eigenphase evidence was obtained
   (literature reference, explicit computation, benchmark run).
3. **Reviewer**: initials and date.
4. **Schema version**: `1.0.0`.
5. **Reduction contract**: expected HSPs ({GammaM, KM}), valley-preserving
   subgroup (C3), allowed ValleyScope irrep keys, and the explicit
   `source_hsp_by_irrep` / `valleyscope_key_by_source_irrep` maps.

Until at least one row has this provenance, no C3-like reduced EBR table
may be claimed as reviewed.

## Machine-Checked vs. External Evidence

### Evidence Already Machine-Checked By Existing Tests

| Evidence | Test File(s) | What Is Checked |
|----------|-------------|-----------------|
| Spinful C3 phase table data contract | `tests/test_phase_tables.py` | 3 labels, phases in canonical range, no EBR vectors, no material names |
| Source basis inspector returns canonical payload | `tests/test_irreptables_table_builder.py` | 22 labels, 9 EBRs from `irreptables.ebrs.load_ebr_data(150, True)` |
| SG 150 spinful Bilbao character data | `tests/test_irreptables_table_builder.py` | `IrrepTable("150", True)`, operation IDs 2/3 as C3/C3^2, and C3 characters for `-GM4/-GM5/-GM6/-K4/-K5/-K6` |
| Table builder produces loadable reduced tables | `tests/test_irreptables_table_builder.py` | Valid mapping spec → buildable table with provenance |
| Spec template/validator preflight checks | `tests/test_irreptables_table_builder.py` | Placeholder rejection, HSP/key consistency, source-basis coverage |
| Reduced EBR solver API | `tests/test_reduced_ebr_smoke.py` | Smith normal form, integer-span check, three-way classification |
| C3 audit doc contract | `tests/test_irreptables_table_builder.py` | Physical object coverage, no material names, no `review_ready`, C2 kept out of C3 basis |
| No forbidden imports in builder/inspector/validator | `tests/test_irreptables_table_builder.py` | No `irrep2`, no OR-Tools, no `irrep.ebrs` |

### Evidence Requiring External / Manual Review

| Evidence Gap | Why Still Requires Review | Required Resolution |
|-------------|--------------------------|-------------------|
| C3 restriction decomposition for `-GM6` (deg 2) | Characters `chi(op2)=+1` and `chi(op3)=+1` are known, but the 2D source irrep must be decomposed into 1D spinful C3 irreps under a reviewed double-group convention | Explicit subgroup-character decomposition and reviewer sign-off |
| C3 restriction decomposition for `-K6` (deg 2) | Same | Same |
| Source-irrep label convention documentation (`-4`/`-5`/`-6` numbering) | Character data are machine-readable, but the reduced-table mapping still needs provenance and convention sign-off | Record `irreptables` package/version, operation IDs, phase convention, reviewer, and date |

## Human Decisions Still Required

Before a reviewed C3-like reduced EBR table can be packaged as ValleyScope
data, the following explicit human decisions are required:

1. **Confirm the 1D labels with Bilbao character evidence.**
   `IrrepTable("150", True)` gives `chi(C3)=-1` for `-GM4`, `-GM5`,
   `-K4`, and `-K5`, corresponding to phase +1/2.  The reviewer must
   approve the source-label convention and record provenance.

2. **Resolve `-GM6` and `-K6` C3 restriction decomposition.**
   These are 2D source irreps.  Before either can contribute to the 1D
   C3 phase basis, its op2/op3 C3 characters and double-group lift
   convention must be decomposed into individual 1D C3 spinful irreps.

3. **Decide which labels enter the first reduced basis.**
   The reviewer may include all confirmed 1D labels, a minimal subset, or
   defer labels whose role in the reduced-dimensional table is not yet clear.

4. **Sign off on provenance record.** Every reviewed row must carry:
   source (SG number, spinor convention, data source package/version),
   review method, reviewer initials and date, and schema version.

Until decisions 1–4 are complete, no C3-like reduced EBR table may be
claimed as reviewed, and no such table may be shipped as ValleyScope
package data.

## C3 Character Evidence Feasibility Assessment

Date: 2026-06-13 | Status: Feasibility audit (no implementation)

This section assesses whether reproducible, machine-checkable C3
character/eigenphase evidence exists for each SG 150 (P321) spinful
source irrep label in scope.  It does not generate or ship any table
data.  It is written for a human reviewer who must decide which
evidence source to accept before a reviewed C3-like package-data
reduced EBR table can be shipped.

### Target Physical Mapping Chain

```
source SG 150 spinful irrep label
  -> sampled moire HSP (GammaM or KM)
  -> HSP little group G_k
  -> valley mapping pi_g(a)
  -> valley-preserving subgroup G_k^{(a)} = {C3, C3^2, E}
  -> ValleyScope spinful C3 irrep phase key
       (C3_spinor_phase_+1/6, +1/2, -1/6)
```

The evidence required at each step:

1. **HSP assignment**: which 3D BZ HSP the source label belongs to
   (GammaM, K, KA, etc.).
2. **C3 eigenphase/character**: for 1D labels, the C3 rotation eigenvalue
   `chi(C3)` = `exp(2*pi*i*phase)` with `phase in (-0.5, 0.5]`.
3. **C3 restriction decomposition** (degenerate labels only): for labels
   with degeneracy > 1, the decomposition of the source irrep under the
   C3 valley-preserving subgroup into 1D spinful C3 irreps, giving
   multiplicities `n_{+1/6}`, `n_{+1/2}`, `n_{-1/6}`.

### Per-Label Evidence Requirements

| Source Label | Deg | Sampled HSP | Evidence Type | Can ValleyScope Key Be Assigned Without External Evidence? |
|-------------|-----|-------------|--------------|----------------------------------------------------------|
| `-GM4` | 1 | GammaM | C3 eigenphase `exp(2*pi*i*phase)` | Machine-readable via `IrrepTable("150", True)`; needs human convention review |
| `-GM5` | 1 | GammaM | C3 eigenphase `exp(2*pi*i*phase)` | Machine-readable via `IrrepTable("150", True)`; needs human convention review |
| `-GM6` | 2 | GammaM | C3 restriction decomposition → two 1D phases | No — requires character decomposition of 2D irrep |
| `-K4` | 1 | KM | C3 eigenphase `exp(2*pi*i*phase)` | Machine-readable via `IrrepTable("150", True)`; needs human convention review |
| `-K5` | 1 | KM | C3 eigenphase `exp(2*pi*i*phase)` | Machine-readable via `IrrepTable("150", True)`; needs human convention review |
| `-K6` | 2 | KM | C3 restriction decomposition → two 1D phases | No — requires character decomposition of 2D irrep |
| `-KA4/-KA5/-KA6` | 1/1/2 | KA | (out of scope) | Out of scope — KA not in sampled `{GammaM, KM}` set |

### API Feasibility Audit

#### Path 1: `irreptables.ebrs.load_ebr_data(150, True)`

**Status: INSUFFICIENT.**  Provides EBR combinatorial data only:
source irrep labels, degeneracies, EBR names, Wyckoff positions,
EBR vectors, and precomputed Smith normal form.  No C3
character/eigenphase data is exposed.  This is already machine-checked
by `tests/test_irreptables_table_builder.py`.

#### Path 2: `irreptables.irreps.IrrepTable("150", True)`

**Status: AVAILABLE — BILBAO-DERIVED CHARACTER TABLE.**  This loads
SG 150 (P321) spinful irrep data from Bilbao-derived data files
shipped with the `irreptables` package.  The API provides:
- 16 irreps with names, dimensions, k-point labels, and complex
  per-operation characters keyed by 1-indexed Bilbao operation IDs.
- 6 symmetry operations (rotations + translations) with order
  information: identity (op 1), two C3 operations (ops 2-3), three
  C2 operations (ops 4-6).

Critical usage note: the SG identifier must be a **string**
(`IrrepTable("150", True)`), not an integer.  `IrrepTable(150, True)`
raises `AssertionError` because the API compares the input to a
string SG identifier.

This is the recommended path for machine-checkable C3 character
evidence.  The key observations (verified by
`tests/test_irreptables_table_builder.py`):

1. All four 1D labels at GammaM and KM (`-GM4`, `-GM5`, `-K4`, `-K5`)
   have C3 character = -1 = `exp(+i*pi)` = phase +1/2.
2. The degenerate labels (`-GM6`, `-K6`) have op2/op3 character evidence
   `(+1, +1)`.  This is not yet a reviewed 1D C3 phase decomposition;
   it is the input to a separate subgroup-restriction step.
3. The 1D labels are distinguished by their **C2** characters (ops
   4-6), which are valley-changing and not relevant to the C3
   valley-preserving reduced basis.

#### Path 3: `irrep.spacegroup_irreps.SpaceGroupIrreps`

**Status: FEASIBLE BUT HEAVY — NOT NEEDED FOR BASIC C3 CHARACTERS.**
The `irreptables.irreps` Bilbao data (Path 2) already provides C3
characters.  `SpaceGroupIrreps` is an alternative heavy path that

- Explicit hexagonal lattice vectors.
- Symmetry operations (rotations + translations) from spglib
  `get_symmetry_from_database(hall_number=458)` for P321.
- The constructor handles spinor rotation computation internally
  via `SymmetryOperation` when `spinor=True`.
- Time-reversal symmetry must be provided explicitly (standard for
  non-magnetic systems: `[True] * len(rotations)`).

Caveats:
- No `from_sg_number(150, spinful=True)` factory exists.
- After construction, HSP irreps are accessible through the object's
  internal structures, but extracting per-label C3 characters requires
  navigating the API's irrep/kpoint/character hierarchy.
- The `SpaceGroupIrreps` object produces **full 3D space-group irreps**,
  not valley-preserving C3-restricted irreps.  A manual C3 subgroup
  restriction step is required to extract individual C3 eigenphases
  from a 2D irrep (for `-GM6`, `-K6`).
- The computation is reproducible given the same lattice + spglib
  symmetry operations, but the output format is API-dependent and
  not a simple lookup table.

#### Path 4: External Literature / Physics Reference

**Status: REQUIRED FOR HUMAN REVIEW.**  Published C3 character tables
for P321 spinful double group irreps from standard references
(e.g., Bradley & Cracknell, Bilbao Crystallographic Server).  This
is the standard physics approach and is independent of any Python
package API.  A human reviewer would:
1. Look up the C3 character for each source irrep label at GammaM
   and K in the published table.
2. Verify the phase convention matches ValleyScope's
   `phase in (-0.5, 0.5]`.
3. For degenerate labels, perform the C3 subgroup restriction
   decomposition.
4. Record the mapping and provenance.

#### Path 5: Independent ValleyScope Benchmark Run (P321-like clean system)

**Status: SUPPORTING EVIDENCE ONLY.**  A ValleyScope `analyze-hsp` run
on a trusted P321 benchmark system produces
valley-preserving C3 eigenphases at GammaM and KM.  These are
per-sample eigenphases, not source irrep label assignments.  They
can corroborate a mapping but cannot establish it independently — a
ValleyScope run does not know which source irrep label `-GM5`
corresponds to without an external convention reference.

### Per-Label Feasibility Summary

| Source Label | Deg | Public API Evidence | External Evidence Needed | Feasibility Verdict |
|-------------|-----|--------------------|--------------------------|--------------------|
| `-GM4` | 1 | `IrrepTable("150", True)`: C3 character -1 | Human convention/provenance sign-off | Feasible; maps to phase +1/2 after review |
| `-GM5` | 1 | `IrrepTable("150", True)`: C3 character -1 | Human convention/provenance sign-off | Feasible; maps to phase +1/2 after review |
| `-GM6` | 2 | `IrrepTable("150", True)`: op2/op3 C3 characters +1/+1 | C3 subgroup-character decomposition + sign-off | Feasible but requires restriction decomposition |
| `-K4` | 1 | `IrrepTable("150", True)`: C3 character -1 | Human convention/provenance sign-off | Feasible; maps to phase +1/2 after review |
| `-K5` | 1 | `IrrepTable("150", True)`: C3 character -1 | Human convention/provenance sign-off | Feasible; maps to phase +1/2 after review |
| `-K6` | 2 | `IrrepTable("150", True)`: op2/op3 C3 characters +1/+1 | C3 subgroup-character decomposition + sign-off | Feasible but requires restriction decomposition |
| `-KA4/-KA5/-KA6` | 1/1/2 | None | (out of scope) | Out of scope — KA not in C3 reduced basis |

Key: `IrrepTable("150", True)` is the machine-readable Bilbao-derived raw
character table.  It supplies character evidence, not a reviewed ValleyScope
reduced EBR table.

### Machine-Checkable Facts

These facts are verified by existing tests and require no human review:

1. **irreptables EBR data**: 22 source labels, 9 EBRs, degeneracies
   match `irreptables.ebrs.load_ebr_data(150, True)` output.
   Tested: `test_irreptables_table_builder.py`.
2. **ValleyScope C3 phase table**: 3 spinful C3 phase keys with
   phases in `(-0.5, 0.5]` canonical range.
   Tested: `test_phase_tables.py`.
3. **irreptables SG 150 spinful character data**:
   `IrrepTable("150", True)` loads 16 irreps and 6 symmetry operations;
   op 2/3 have spatial C3/C3^2 rotations; in-scope 1D labels have op2
   C3 character -1.
   Tested: `test_irreptables_table_builder.py`.
4. **No `from_sg_number` factory in public irrep API**.
   Verified by source inspection of `irrep.spacegroup_irreps`.
5. **`irrep.spacegroup_irreps.SpaceGroupIrreps` constructor accepts
   lattice + spglib rotations + translations**: requires explicit
   inputs; spinor rotations handled internally; no simple lookup API.
   Verified by source inspection.

### External / Human-Required Facts

These facts cannot be promoted to reviewed package data without human
convention review:

1. **C3 restriction decomposition for each 2D source irrep**
   (`-GM6`, `-K6`): requires explicit subgroup-character decomposition.
2. **Phase convention alignment**: the Bilbao-derived phase convention must be
   recorded against ValleyScope's `exp(2*pi*i*phase)` with
   `phase in (-0.5, 0.5]`.
3. **HSP label correspondence**: source labels like `-GM5` refer to specific
   irreps at the GammaM 3D HSP; this correspondence must be recorded in the
   reviewed mapping spec.
4. **Reduced-basis selection**: a reviewer must decide which source labels
   enter the first C3-like reduced table.

### Human Decision Checklist

Before any C3-like reduced EBR table may enter ValleyScope
package data as reviewed, the human reviewer must complete each item:

- [ ] **1. Select evidence source.**
  Options: (a) published literature character table for P321 spinful;
  (b) explicit `SpaceGroupIrreps` computation with documented
  lattice + symmetry inputs; (c) both, with cross-verification.
  Source: _______________

- [ ] **2. Confirm the 1D C3 eigenphases.**
  `-GM4`, `-GM5`, `-K4`, and `-K5` have `chi(C3)=-1`, giving phase
  +1/2 = ValleyScope
  `C3_spinor_phase_+1/2`.
  Verified source/version: _______________

- [ ] **3. Resolve `-K6` C3 restriction.**
  2D source irrep at KM; decompose under C3 into 1D spinful phases using
  both op2/op3 characters and the reviewed double-group lift convention.
  Decomposition: _______________
  Source: _______________

- [ ] **4. Resolve `-GM6` C3 restriction (if needed).**
  Decomposition: _______________ Source: _______________

- [ ] **5. Decide the first C3 reduced-basis source labels.**
  Decision: [ ] include all reviewed 1D labels / [ ] minimal subset /
  [ ] defer selected labels
  Rationale: _______________

- [ ] **6. Verify phase convention.**
  Confirm the evidence source uses `exp(2*pi*i*phase)` with
  `phase in (-0.5, 0.5]`.
  [ ] Verified

- [ ] **7. Sign off.**
  Reviewer: _______________ Date: _______________

Items 2, 3, 5, 6, and 7 are the minimum required before a reviewed
C3-like reduced EBR table can be shipped.  Item 4 is required if the
reviewer wants to include `-GM6`.

### Feasibility Conclusion (Revised 2026-06-13)

**C3 character evidence is now machine-checkable.** The
`irreptables.irreps.IrrepTable("150", True)` Bilbao-derived data
provides reproducible, per-operation character values for all six
in-scope source labels at GammaM and KM.  This was the key missing
piece identified in the earlier audit and is now verified by
`tests/test_irreptables_table_builder.py`.

The 1D labels (`-GM4`, `-GM5`, `-K4`, `-K5`) all have op2 C3 character = -1,
giving candidate phase +1/2 after convention review.  The degenerate labels
(`-GM6`, `-K6`) have op2/op3 character evidence `(+1, +1)`, but this audit
does not assign them 1D C3 phase multiplicities.  The `-4`/`-5` distinction
at each HSP is in the C2 characters, not C3; those C2 characters are valley
sewing data for the single-valley reduced C3 basis.

**Remaining blockers:**

1. **C3 restriction decomposition** for `-GM6` and `-K6`: the Bilbao
   data gives op2/op3 C3 characters (+1,+1), but extracting individual
   1D C3 spinful eigenphases requires explicit subgroup restriction and a
   reviewed double-group lift convention.  No `{+1/6, -1/6}` multiplicity
   assignment is claimed by this audit.

2. **Human review of the mapping convention**: the C3 character values
   are now machine-verified, but the mapping from source labels to
   ValleyScope keys still requires a human to confirm the convention
   and sign off on the provenance record per the checklist above.

3. **Operation identification**: the Bilbao convention uses 1-indexed
   operation IDs.  Operations 2 and 3 are C3/C3^2 (valley-preserving);
   operations 4-6 are C2 (valley-changing at K/K').  All four 1D
   labels at each HSP have identical C3 characters — the `-4`/`-5`
   distinction is in C2 characters only.

The most practical next step is to complete the Human Decision Checklist
above using the Bilbao character evidence, then populate a reviewed
mapping spec.  An external literature character table is still
recommended for cross-verification but is no longer the only viable
evidence source.

Until human review is complete, no C3-like reduced EBR table may be
shipped as reviewed ValleyScope package data.

## C3 Double-Group Lift Convention Audit

Date: 2026-06-13 | Status: Audit (evidence verified by
`tests/test_irreptables_table_builder.py`)

This section audits the double-group (spinor) lift convention used by
`irreptables.irreps.IrrepTable("150", True)` for SG 150 (P321) and
determines whether the Bilbao character-table operation indexing maps
directly to ValleyScope's C3 valley-preserving subgroup phase formula.

### Bilbao Convention: Independent Spinor Lifts Per Spatial Operation

The Bilbao data file (`irreps-SG=150-spin.dat`) encodes each symmetry
operation with:

1. A spatial rotation matrix `R` (3×3 integer).
2. A translation vector `t`.
3. An SU(2) spinor rotation matrix `S` (2×2 complex), computed from
   four real absolute values and four phase/π values via
   `S_ij = abs_ij * exp(i*π*phase_ij)`.

For SG 150 (P321), the relevant valley-preserving operations are:

| Op ID | Spatial R | SU(2) S | Double-Group Role |
|-------|----------|---------|-------------------|
| 1 | Identity | I | E |
| 2 | C3 | diag(exp(+iπ/3), exp(-iπ/3)) | C3 generator g |
| 3 | C3^2 | diag(exp(-iπ/3), exp(+iπ/3)) | NOT g^2 — see below |

Spatially, `R3 = R2 @ R2` (C3^2 = C3²).  This is verified.

### Key Finding: op3 ≠ op2² in the Double Group

The double-group product of op2 with itself is:

```text
S2 @ S2 = diag(exp(+2iπ/3), exp(-2iπ/3))
S3      = diag(exp(-iπ/3), exp(+iπ/3))
```

`S2 @ S2 ≠ S3` and also `S2 @ S2 ≠ -S3`.  The two spinor matrices
differ by more than a sign — they represent **different double-group
elements**.

The double-group relation `g³ = -E` is correctly satisfied:
`S2³ = diag(-1, -1) = -I`.

### Character-Level Consequence

For all four 1D irreps at GammaM and KM (`-GM4`, `-GM5`, `-K4`, `-K5`):

| Value | Bilbao Table | Group-Theoretic | Match? |
|-------|------------|----------------|--------|
| chi(C3) = chi(op2) | -1 | -1 = exp(+iπ) | ✓ |
| chi(op3) from table | -1 | — | — |
| chi(C3²) = chi(op2)² | — | +1 = exp(+2iπ) | **✗ chi(op3) ≠ chi(op2)²** |

Bilbao's `chi(op3) = -1` is NOT the group-theoretic `chi(C3²)`.
The raw Bilbao op3 character must not be used in ValleyScope's
`chi(C3²)` phase formula for 1D irreps.

For the 2D irreps (`-GM6`, `-K6`):

| Value | Bilbao Table | Group-Theoretic (from eigenvalues) | Match? |
|-------|------------|-----------------------------------|--------|
| chi(op2) | +1 | +1 = exp(+iπ/3) + exp(-iπ/3) | ✓ |
| chi(op3) | +1 | −1 = exp(+2iπ/3) + exp(-2iπ/3) | **✗** |

### ValleyScope Phase Convention Implications

ValleyScope's spinful C3 phase formula requires **group-theoretic**
characters for the valley-preserving subgroup `{E, g, g²}` where `g`
is the C3 generator and `g²` is its double-group square:

```text
chi(g)  = Σ_i n_i * exp(2πi * phase_i)       (from Bilbao op2)
chi(g²) = Σ_i n_i * exp(4πi * phase_i)       (NOT Bilbao op3!)
```

For 1D irreps: `chi(g²) = chi(g)²` (group-theoretic square).

For degenerate irreps: `chi(g²)` must be computed from the C3 eigenvalue
decomposition, not read from the Bilbao op3 character.

### Verified Convention Mapping

The Bilbao `IrrepTable("150", True)` characters can be used for
ValleyScope's C3 valley-preserving subgroup as follows:

1. **`chi(C3) = chi(op2)`** from the Bilbao table.  This is the C3
   generator character.  ✓ Direct read.

2. **`chi(C3²) = chi(C3)²`** for 1D irreps.  Compute from op2 alone.
   Do NOT use `chi(op3)` from the Bilbao table.

3. **For degenerate irreps**, decompose `chi(op2)` into spinful C3
   eigenphases, then compute `chi(C3²)` from the eigenvalue expansion
   `Σ_i n_i * exp(4πi * phase_i)`.  The Bilbao op3 character is not
   usable for this calculation.

4. **The C2 characters (op4-6)** are valley-changing sewing data and
   must not enter the C3 valley-preserving reduced EBR vector basis.

This convention is machine-verified by
`tests/test_irreptables_table_builder.py`.

### Remaining Ambiguity

The mapping `chi(C3²) = chi(C3)²` is mathematically correct for 1D
representations but should be cross-verified with an explicit P321
spinful double-group multiplication table (e.g., from Bradley &
Cracknell or the Bilbao Crystallographic Server).  The independent
spinor lifts in the Bilbao convention are a specific gauge choice;
a different gauge would give different op3 characters but the same
group-theoretic `chi(C3²)`.

## Non-Features

- No built-in reduced EBR table is shipped.
- No EBR vectors are hardcoded.
- No material-specific logic or labels.
- No HSP inference from source labels.
- No ValleyScope irrep-key inference.
- No raw 3D decomposition.
- No `irrep.ebrs`, OR-Tools, or `irrep2` imports.
