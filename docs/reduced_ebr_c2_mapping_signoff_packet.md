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
| Character encoding | Complex: `ch = \|ch\| * exp(iπ * phase/π)` per operation |
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

From the Bilbao SG 149 spinful table, the little-group structure at
each relevant HSP:

| HSP | k-coordinate | Little-Group Order-2 Ops |
|-----|-------------|-------------------------|
| GammaM (GM) | (0,0,0) | ops 4, 5, 6 |
| K | (1/3, 1/3, 0) | ops 4, 5, 6 |
| H | (1/3, 1/3, 1/2) | ops 4, 5, 6 |

Ops 4-6 are spatial order-2 rotations (trace = -1, determinant = +1).
For the spinful double group, S^2 = -I is verified for all three.

This is source HSP little group data.  The ValleyScope sampled-HSP set
for a C2-like reduced table depends on the valley family (e.g., tZrSe2
M-star `{GammaM, KM, MM}`) and is not determined by the Bilbao table alone.

## 3. Valley Mapping (tZrSe2 P312 M-Star Example)

For the tZrSe2 M-star three-valley system under P312:

```
Valleys: M1, M2, M3 (threefold M-star at K)
C3 ops:  M1 → M2 → M3 → M1  (valley-changing)
C2 ops:  M3 preserved at GammaM, MM (valley-preserving)
         M1 ↔ M2 swapped       (valley-changing)
```

- C3 operations are valley-changing for all M-star valleys.
- One C2 operation preserves M3 at GammaM and MM — this is the
  valley-preserving C2 operation for valley M3.
- C2 operations between M1 and M2 are valley-changing sewing data.

This valley mapping is fixture-specific for tZrSe2.  A different C2
valley system under P312 could have different valley-preserving
subgroup structure.

## 4. Valley-Preserving Subgroup

For the M3 valley at GammaM (tZrSe2 M-star):

```text
G_k^{(M3)} = {g ∈ G_k | π_g(M3) = M3} = {E, C2_M3}
```

The valley-preserving subgroup is order 2, isomorphic to C2.
For M1 and M2 valleys at GammaM, KM, and MM, the sampled-HSP-local
preserving subgroup is identity-only (C2 ops are valley-changing
between M1↔M2 at these HSPs).

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

## 6. Central-Sign Convention (Unresolved — Blocker For C2 Table Shipment)

### Bilbao Source Evidence

The SG 149 spinful Bilbao table provides order-2 characters for 1D
irreps at GammaM and K:

| Source Label | Deg | kpname | Bilbao chi(op4) | Phase (2π) |
|-------------|-----|--------|----------------|-----------|
| `-GM4` | 1 | GM | -1 | +0.500 |
| `-GM5` | 1 | GM | -1 | +0.500 |
| `-K4` | 1 | K | -1 | +0.500 |
| `-K5` | 1 | K | exp(-iπ/3) | -0.167 |
| `-K6` | 1 | K | exp(+iπ/3) | +0.167 |

### ValleyScope Spinful C2 Phase Labels

`valleyscope/data/valley_irreps/spinful_C2_phase_v1.json`:

| Phase (2π) | ValleyScope Label |
|-----------|-------------------|
| +1/4 | `C2_spinor_phase_+1/4` |
| -1/4 | `C2_spinor_phase_-1/4` |

These correspond to double-group C2 eigenvalues ±i (since C2^2 = -E
for spinors, giving eigenvalue equation λ^2 = -1 → λ = ±i).

### Convention Gap

The Bilbao chi(op4) = -1 (exp(+iπ)) for `-GM4`/`-GM5`/`-K4` does NOT
equal exp(±iπ/2) = ±i.  The ValleyScope C2 spinful phase labels ±1/4
are NOT direct identities of the Bilbao order-2 characters.

This is the same class of issue resolved for C3: the Bilbao character
table uses independent spinor lifts per spatial operation, and the
central-sign convention must be established before the source
characters can be mapped to ValleyScope reduced-basis labels.

### Mapping Hypotheses (Not Reviewed)

For 1D irreps with Bilbao chi(op4) = -1 (phase +1/2), the ValleyScope
C2 eigenphase could be:

- **Hypothesis A**: C2 eigenphase = ±1/4 directly, with a sign
  convention relating Bilbao chi(op4) to the C2 generator eigenvalue.
  This would require identifying which Bilbao op (4, 5, or 6)
  corresponds to the valley-preserving C2 for each valley label.

- **Hypothesis B**: The Bilbao order-2 character requires a
  central-sign correction analogous to the C3
  `chi(C3²) = -chi(op3)` convention.

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
- What is the central-sign convention for the C2 eigenphase?
- How does chi(C2) from Bilbao relate to the ValleyScope C2 spinful phase ±1/4?

## 7. Signoff Checklist

Before a C2-like reduced EBR table can be shipped as reviewed
ValleyScope package data:

- [ ] **C2 central-sign convention resolved.**  Explicit mapping from
  Bilbao SG 149 order-2 characters to ValleyScope C2 spinful
  eigenphases (±1/4), with spinor-lift justification.
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
