# ValleyScope C2-Like Reduced EBR Unblock Audit

Date: 2026-06-14 | Status: Audit document (no C2 table shipped)

## Scope

This document audits the C2-like valley-preserving reduced EBR basis
for SG 149 (P312) spinful irreps, with the tZrSe2 M-star P312 fixture
as the target validation system.  It is a **signoff-preparation audit**,
not a C2 package-data table and not a reduced EBR decomposition report.

No C2-like reduced EBR table is shipped in this audit.

## Source Data Availability

### irreptables Bilbao Character Table

`irreptables.irreps.IrrepTable("149", True)` loads SG 149 (P312) spinful
irrep data:

- 16 irreps (same structure as SG 150)
- 6 symmetry operations: op1 (E), op2-3 (C3), op4-6 (C2, order 2)
- S2^2 = -I verified for all three C2 ops (double-group relation satisfied)
- 16 EBR source labels, 18 EBR definitions

The data source is verified machine-readable.

Machine-verified by: `python -c "import irreptables.irreps as ir;
ir.IrrepTable('149', True)"` — loads successfully.

### SG 149 Order-2 Character Evidence (Bilbao Table)

The order-2 source-character evidence exposed by
`IrrepTable("149", True)` is HSP-local:

| Source HSP | Exposed little-group ops | Order-2 source ops |
|------------|--------------------------|--------------------|
| GM | ops 1-6 | ops 4, 5, 6 |
| A | ops 1-6 | ops 4, 5, 6 |
| M | ops 1, 6 | op 6 |
| L | ops 1, 6 | op 6 |
| K | ops 1, 2, 3 | none in the source irrep characters |
| H | ops 1, 2, 3 | none in the source irrep characters |

Relevant 1D C2-like source rows carry spinful phases `±1/4` directly:

| Source Label | Deg | kpname | Order-2 op | Character | Phase (2π) |
|--------------|-----|--------|------------|-----------|------------|
| `-GM4` | 1 | GM | op4/op5/op6 | `-i` | -1/4 |
| `-GM5` | 1 | GM | op4/op5/op6 | `+i` | +1/4 |
| `-GM6` | 2 | GM | op4/op5/op6 | 0 | not 1D |
| `-M3` | 1 | M | op6 | `-i` | -1/4 |
| `-M4` | 1 | M | op6 | `+i` | +1/4 |

K/H rows are C3-like in this source table and must not be cited as C2
source-character rows.  The remaining convention review is not a
`-1 -> ±i` conversion; it is the assignment of the correct Bilbao source
HSP/op to each ValleyScope valley-preserving C2 generator and any required
generator-orientation or spinor-lift conversion.

### ValleyScope C2 Phase Table

`valleyscope/data/valley_irreps/spinful_C2_phase_v1.json` provides:

| Phase (2π) | ValleyScope Label |
|-----------|-------------------|
| +1/4 | `C2_spinor_phase_+1/4` |
| -1/4 | `C2_spinor_phase_-1/4` |

These are validated by `tests/test_phase_tables.py`.

## Physical Objects For C2-Like Reduced EBR Table

### HSP Little Group

For P312 source irreps, order-2 character availability from the Bilbao table is:

| HSP | k-coordinate | Little-Group C2 Ops | Notes |
|-----|-------------|-------------------|-------|
| GammaM (GM) | (0,0,0) | ops 4,5,6 | Full C2 little group |
| M | see source table | op6 | C2-like source row |
| K | (1/3,1/3,0) | none in source characters | C3-like source row |
| H | (1/3,1/3,1/2) | none in source characters | C3-like source row |

The tZrSe2 M-star fixture is a moire/valley-projected validation context,
not a direct copy of the raw Bilbao source-label list:
- GammaM: M1, M2, and M3 each have an HSP-local C2-like row, but all are
  diagnostic-only because of spinor-convention, seed-overlap, and for M3
  representation-unitarity blockers.
- KM: M1/M2/M3 are identity-only in the configured sampled HSP; non-identity
  C2 data would have to come from HSP-star derivation.
- MM: M1/M2 are identity-only at the configured sampled HSP; M3 has the
  explicit HSP-local C2-like source row, but it is diagnostic-only because
  of representation-unitarity and spinor-convention blockers.

### Valley Mapping For tZrSe2 P312 M-Star

```
Valley labels: M1, M2, M3 (threefold M-star at K)
C3 operations: cycle M1 -> M2 -> M3 -> M1 (valley-changing at K)
C2 operations: preserve M3 at GammaM and MM; map M1 <-> M2 at
  certain HSPs (HSP-star dependent)
```

For the C2-like valley-preserving subgroup, GammaM has candidate local
C2-like rows for M1/M2/M3, while MM has an explicit C2-like source row for
M3.  KM and MM M1/M2 require HSP-star derivation rather than direct
single-HSP local C2 data.

### Valley-Preserving Subgroup

At GammaM for valleys M1/M2/M3: `G_k^{(a)} = {E, C2_a}` (order 2),
with different C2 representatives for different M-star labels.

At MM for valley M3: `G_k^{(M3)} = {E, C2}` for the explicit sampled
HSP source row.

At KM, and at MM for M1/M2, the sampled HSP-local preserving subgroup is
identity-only; non-identity C2 characters would need trusted HSP-star
derivation from an explicit source such as MM/M3.

### Valley-Changing Operations

C3 operations (ops 2-3): cycle M1→M2→M3→M1 — valley-changing
C2 ops between M1/M2: valley-changing (sewing M1↔M2)

These must not enter the single-valley C2 reduced EBR vector basis.

### C2 Source-Op / Phase Convention: Blocker

The Bilbao SG 149 spinful table gives relevant C2-like 1D source
characters `±i`, consistent with the allowed ValleyScope C2 spinful
phase set `±1/4` at the level of source-character phases.

The unresolved convention issue is therefore more specific: ValleyScope
must identify which Bilbao source HSP and order-2 op represent the
valley-preserving C2 generator for each sampled moire HSP / valley label,
and whether the local generator orientation matches the source phase or
requires a sign/lift conversion.

**This source-op and spinor-lift convention resolution must be completed
before any C2-like reduced EBR mapping can be reviewed.**

## tZrSe2 Fixture Blocker Status

Local fixture probe (from `real_tests/tZrSe2/analyze.yaml`, default config):

| Blocker | Status | Detail |
|---------|--------|--------|
| Export bundle | `no_bundles` (0) | No EBR problem instances exist |
| EBR input candidates | none | No candidate passes readiness gates |
| B1: spinor convention | **Blocked** | `spinor_convention_verified: False` on all rows; no reference benchmark (`spinor_benchmark: None`) for C2 spinor cross-verification |
| B2: D_raw closure | **Blocked** | GammaM/MM op=5 closure is `usable_with_caution`, but local representation unitarity remains above trusted EBR-readiness thresholds |
| All eigenphase rows | `diagnostic_only: True` | Cannot be used for reduced EBR input |
| `spinor_rotation_applied` | `True` | Spinor rotation is mechanically applied; the issue is convention verification, not missing rotation |

### Blocker Root Cause Analysis

1. **B1 (spinor convention)**: This is the primary blocker. Without a
   verified C2 source-op and spinor-lift convention, no eigenphase row can be
   trusted for reduced EBR input.  The ValleyScope C2 phase convention
   (±1/4) needs explicit physics signoff mapping the selected Bilbao C2-like
   source characters to the local valley-preserving C2 generator.

2. **B2 (D_raw closure)**: Target-subspace closure for op=5 at GammaM/MM is
   `usable_with_caution`, but that is not a trusted EBR-input pass.  The
   resulting local representation unitarity remains above the EBR-readiness
   threshold.  This is a numerical/DFT target-subspace quality issue, not a
   code-design issue.

3. **Seed overlap**: Low seed overlap values at GammaM prevent candidate
   rows from reaching `ready_for_ebr_input` status.

All three blockers are physical/numerical, not code-design blockers.
The code path for C2-like reduced EBR mapping exists and works (tested
with C3 table on tMoTe2).  The blockers are in the physics input quality
and convention verification, not in the software.

## Signoff-Preparation Checklist

Before a C2-like reduced EBR table can be shipped as reviewed
ValleyScope package data, these items must be completed:

### Physics Convention Signoff (User Required)

- [ ] **C2 source-op / spinor-lift convention resolved.** Map the selected
  Bilbao SG 149 C2-like little-group characters to ValleyScope C2 spinful
  eigenphases (±1/4).  This is the analog of the C3
  `chi(C3)=chi(op2), chi(C3^2)=-chi(op3)` convention.  The C2 convention
  must address:
  - Which Bilbao source HSP and op index correspond to the C2 generator
  - Whether the selected source character maps directly to the ValleyScope
    phase key or needs generator-orientation / spinor-lift conversion
  - How to treat valley-changing C2 operations as valley sewing matrix data

- [ ] **Spinor benchmark for C2 convention verification.** A trusted
  reference system with known C2 spinful eigenphases (e.g., a
  literature-verified P312 compound or explicit `SpaceGroupIrreps`
  computation).

### Fixture Quality (User Required)

- [ ] **tZrSe2 seed projector quality improved** or acceptable substitute
  fixture identified.  Current GammaM seed overlap is below the
  `topology_input_ready` threshold.

- [ ] **D_raw target-subspace closure at GammaM** improved to usable
  level, or expanded-band HDF5 tested for closure convergence.

### Implementation Tasks (After Signoff)

After the above physics signoffs are complete:

1. Define `source_hsp_by_irrep` for all SG 149 source labels.
2. Define `valleyscope_irrep_multiplicity_by_source_irrep` for
   C2-preserving sampled HSPs (identity of which HSPs depends on tZrSe2
   M-star HSP-star analysis).
3. Pin real-source C2 reduced EBR vectors (same methodology as C3).
4. Produce reviewed C2-like mapping spec and package-data table.

## Conclusion

**A C2-like reduced EBR table is physically definable** — the SG 149
Bilbao source data is available and the ValleyScope v1.1 multiplicity-aware
builder supports C2-like group labels.  However, **the C2 source-op /
spinor-lift convention (which Bilbao C2-like source character maps to each
ValleyScope valley-preserving C2 phase key) must be resolved by a physicist
before any reviewed C2 table can be authored.**

**tZrSe2 remains blocked** as a validation fixture primarily by the
spinor convention verification gap (B1) and secondarily by numerical
quality issues (B2, seed overlap).  These are physical/numerical
blockers, not ValleyScope code-design blockers.

**No C2-like package-data table should be shipped until the C2 source-op /
spinor-lift convention is signed off,** analogous to the C3 signoff process
documented in `docs/reduced_ebr_c3_mapping_signoff_packet.md`.

No C2 package-data table is shipped by this audit.
