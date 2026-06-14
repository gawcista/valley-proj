# ValleyScope C2-Like Reduced EBR Mapping Signoff Packet

Date: 2026-06-14 | Status: Signoff packet (no C2 table shipped)

## Scope

This document records the Bilbao-derived source evidence and the
unresolved convention questions for a C2-like reduced EBR table
authoring workflow using SG 149 (P312) spinful irreptables data.
It is a **signoff packet**, not a C2 reduced EBR table, not a
decomposition report, and not `analyze-hsp` workflow wiring.

A future reviewed C2 mapping spec or reduced EBR table may be built
from this packet after the signoff checklist is completed.

### Non-Goals

- This packet does not ship any C2 reduced EBR table JSON.
- It does not wire into `analyze-hsp` or `valleyscope/data/reduced_ebr/`.
- It does not claim a tZrSe2 reduced EBR decomposition.
- It does not use OR-Tools, `irrep.ebrs`, or `irrep2`.

## 1. Accepted Source Data

| Field | Value |
|-------|-------|
| Data source | `irreptables.irreps.IrrepTable("149", True)` |
| Data origin | Bilbao Crystallographic Server (via irreptables) |
| Space group | 149 (P312) |
| Spinor type | Spinful (double group, SOC) |
| Number of symmetry operations | 6 (identity, 2×C3, 3×order-2) |
| Number of irreps | 16 |
| Character encoding | Complex Bilbao/irreptables character values per operation |
| Operation indexing | 1-indexed Bilbao convention |

Machine-verified by the C2 authoring audit (`docs/reduced_ebr_c2_authoring_audit.md`).

### Source Label Evidence vs. ValleyScope Reduced-Basis Labels

The Bilbao character table provides **source character evidence** for 16
SG 149 spinful irreps across all 3D BZ HSPs.  These are NOT ValleyScope
reduced-basis labels.  The distinction:

- **Source character evidence**: Bilbao irrep label + dimension + operation
  characters at a 3D HSP — this is the canonical input data.
- **ValleyScope reduced-basis label**: `{HSP}:{phase_key}` format key
  after explicit sampled-HSP, valley mapping, valley-preserving subgroup,
  and central-sign convention reduction.

No source label is directly mappable to a ValleyScope C2 reduced-basis
label without passing through the convention steps documented below.

## 2. Source HSP Little Group Evidence

From the Bilbao SG 149 spinful table, the order-2 source-character
availability is:

| Source HSP | Exposed little-group ops | Order-2 source ops |
|------------|--------------------------|--------------------|
| GM | ops 1-6 | ops 4, 5, 6 |
| A | ops 1-6 | ops 4, 5, 6 |
| M | ops 1, 6 | op 6 |
| L | ops 1, 6 | op 6 |
| K | ops 1, 2, 3 | none in the source irrep characters |
| H | ops 1, 2, 3 | none in the source irrep characters |

Globally, ops 4-6 are spatial order-2 rotations (trace = -1,
determinant = +1).  For the spinful double group, S^2 = -I is verified
for all three.  However, a source irrep row only carries characters for
the operations in its own HSP little group.  K/H source rows are C3-like
in this table and must not be used as C2 source-character evidence.

This is source HSP little group data.  The ValleyScope sampled-HSP set
for a C2-like reduced table depends on the valley family (e.g., tZrSe2
M-star `{GammaM, KM, MM}`) and is not determined by the Bilbao table alone.

## 3. Valley Mapping (tZrSe2 P312 M-Star Example)

For the tZrSe2 M-star three-valley system under P312:

```
Valleys: M1, M2, M3 (threefold M-star at K)
C3 ops:  M1 → M2 → M3 → M1  (valley-changing)
C2 ops:  GammaM has M1/M2/M3 HSP-local C2-like rows
         MM has an explicit M3 C2-like source row
         KM and MM M1/M2 need HSP-star derivation for non-identity C2 data
```

- C3 operations are valley-changing for all M-star valleys.
- GammaM has candidate HSP-local C2-like rows for M1, M2, and M3.  In
  the current fixture these rows remain diagnostic-only because of spinor
  convention, seed-overlap, and representation-quality blockers.
- MM has the explicit HSP-local C2-like source row for M3, also
  diagnostic-only until the C2 convention and representation-quality blockers
  are resolved.
- KM, and MM for M1/M2, need HSP-star derivation for non-identity C2
  data rather than direct sampled-HSP-local C2 rows.

This valley mapping is fixture-specific for tZrSe2.  A different C2
valley system under P312 could have different valley-preserving
subgroup structure.

## 4. Valley-Preserving Subgroup

For a local C2-like row such as M3 at GammaM or MM (tZrSe2 M-star):

```text
G_k^{(M3)} = {g ∈ G_k | π_g(M3) = M3} = {E, C2_M3}
```

The valley-preserving subgroup is order 2, isomorphic to C2.
At GammaM, M1 and M2 have analogous local C2-like candidate rows with their
own preserving C2 representatives.  These are source/readiness evidence, not
trusted reduced-basis assignments.

At KM, and at MM for M1/M2, the sampled-HSP-local preserving subgroup is
identity-only in the current fixture.  Non-identity C2 characters for those
rows would need trusted HSP-star derivation from an explicit source row such
as MM/M3.

## 5. Valley-Changing Operations and Valley Sewing Matrix

- **C3 operations** (op2, op3): valley-changing for all M-star labels.
- **C2 operations between M1 and M2**: valley sewing data mapping
  M1↔M2 at certain HSPs.  These are valley-changing for a single-valley
  C2 reduced basis.
- **C2 operations from M3** to other labels at specific HSPs:
  HSP-star-dependent valley sewing data.

Valley-changing operation characters are **valley sewing matrix** data.
They must NOT enter the single-valley C2 reduced EBR vector basis.
A full M-star three-valley orbit representation would use sewing matrices
to capture the orbital relationship, but that is outside the scope of
this single-valley C2 signoff.

## 6. C2 Source-Op / Spinor-Lift Convention (Unresolved — Blocker For C2 Table Shipment)

### Bilbao Source Evidence

The SG 149 spinful Bilbao table provides the following order-2 source
characters for relevant 1D rows:

| Source Label | Deg | kpname | Order-2 op | Character | Phase (2π) |
|--------------|-----|--------|------------|-----------|------------|
| `-GM4` | 1 | GM | op4/op5/op6 | `-i` | -1/4 |
| `-GM5` | 1 | GM | op4/op5/op6 | `+i` | +1/4 |
| `-GM6` | 2 | GM | op4/op5/op6 | 0 | not 1D |
| `-M3` | 1 | M | op6 | `-i` | -1/4 |
| `-M4` | 1 | M | op6 | `+i` | +1/4 |

The 3D source table also has A/L analogues at kz=1/2.  Those are full
3D source rows and are only relevant if a future reduced mapping spec
explicitly keeps them.  K/H source rows do not expose order-2
characters in `IrrepTable("149", True)`.

### ValleyScope Spinful C2 Phase Labels

`valleyscope/data/valley_irreps/spinful_C2_phase_v1.json`:

| Phase (2π) | ValleyScope Label |
|-----------|-------------------|
| +1/4 | `C2_spinor_phase_+1/4` |
| -1/4 | `C2_spinor_phase_-1/4` |

These correspond to double-group C2 eigenvalues ±i (since C2^2 = -E
for spinors, giving eigenvalue equation λ^2 = -1 → λ = ±i).

### Convention Gap

The relevant GM and M 1D source rows already carry spinful C2-like
characters `±i`, matching the allowed ValleyScope spinful C2 phase set
`±1/4` at the level of source-character phases.

The unresolved convention gap is therefore not a fake `-1 -> ±i`
conversion.  It is the review of which Bilbao source HSP and operation
represent the ValleyScope valley-preserving C2 generator for each
sampled moire HSP / valley label, and whether the local ValleyScope
generator orientation matches the source phase or requires a sign/lift
conversion.

### Mapping Hypotheses (Not Reviewed)

For 1D source rows with Bilbao C2-like character `±i`, the ValleyScope
C2 eigenphase could be:

- **Hypothesis A**: direct phase mapping after source-op assignment:
  `chi(op_j)=+i -> C2_spinor_phase_+1/4` and
  `chi(op_j)=-i -> C2_spinor_phase_-1/4`.
  This requires identifying which Bilbao op (`4`, `5`, or `6`, or only
  `6` for M/L source rows) corresponds to the valley-preserving C2
  generator for each ValleyScope valley label.

- **Hypothesis B**: the local ValleyScope C2 generator is opposite in
  orientation or differs by a central lift from the chosen Bilbao source
  representative, so the phase sign or central factor must be converted
  before assigning a ValleyScope key.

All hypotheses are `not_reviewed` / `blocked_pending_spinor_lift_reference`
until a physicist resolves the convention.

### C3 Analog

The C3 convention was resolved as:
```text
chi(C3)  = chi(op2)           (Bilbao C3 generator)
chi(C3²) = -chi(op3)          (central-negative correction)
```

The C2 convention requires an analogous resolution answering:
- Which Bilbao op is the valley-preserving C2 generator?
- Does the selected generator orientation match the source character phase?
- Does the selected Bilbao C2-like source character map directly to the
  ValleyScope C2 spinful phase ±1/4, or is a sign/lift conversion needed?

## 7. Signoff Checklist

Before a C2-like reduced EBR table can be shipped as reviewed
ValleyScope package data:

- [ ] **C2 source-op / spinor-lift convention resolved.**  Explicit mapping from
  Bilbao SG 149 C2-like source characters (`±i` in the relevant 1D
  source rows) to ValleyScope C2 spinful eigenphases (±1/4), with
  spinor-lift and generator-orientation justification.
  Convention statement: _______________

- [ ] **Valley-preserving C2 operation identified.**  Which Bilbao op
  (4, 5, or 6) corresponds to the valley-preserving C2 generator
  for each valley label at each HSP.
  Assignment: _______________

- [ ] **Source HSP set confirmed.**  Which Bilbao HSPs (GM, K, H, ...)
  are sampled moire HSPs for the target valley family.
  HSP set: _______________

- [ ] **C2 valley sewing data excluded.**  Non-preserving order-2
  operations and C3 operations confirmed as valley-changing and
  excluded from the reduced EBR basis.

- [ ] **Spinor benchmark reference selected.**  A trusted reference
  system for C2 spinor convention verification (literature P312
  compound or explicit `SpaceGroupIrreps` computation).
  Reference: _______________

- [ ] **Reviewer signoff.**
  Reviewer: _______________ Date: _______________

Until this checklist is completed, no C2-like reduced EBR table may
be shipped as reviewed ValleyScope package data.

This packet records the source evidence and convention questions.
Signing it does not generate a table — it authorizes a future
`valleyscope build-reduced-ebr-table` run to produce a reviewed C2
table from a mapping spec that conforms to the resolved convention.
