# ValleyScope MM/M3 C2 Mapping-Spec Skeleton

Date: 2026-06-15 | Status: review-only skeleton; not buildable; no C2 table shipped

## Scope

This document records the candidate row-level mapping for the first
C2-like valley-preserving reduced-basis entry: MM/M3.  It is a
**review-only skeleton**, not a buildable mapping spec, not a reduced
EBR table, and not `analyze-hsp` wiring.

A future reviewed C2 mapping spec may be built from this skeleton after
the spinor convention and source-op assignment are signed off by a
physicist.

## Candidate Row-Level Mapping

### Source Side

| Field | Value |
|-------|-------|
| Source data | `irreptables.irreps.IrrepTable("149", True)` |
| Space group | 149 (P312) |
| Spinor type | Spinful (double group) |
| Source HSP | M (Bilbao) |
| Source operation | op6 (order-2, S_eig = ±i) |
| Source labels | `-M3`, `-M4` |
| Source characters | `-M3`: -i (phase -1/4), `-M4`: +i (phase +1/4) |

Machine-verified by `tests/test_irreptables_table_builder.py`
(double-group tests) and `docs/reduced_ebr_c2_source_op_mapping_audit.md`.

### ValleyScope Side

| Field | Value |
|-------|-------|
| Validation fixture | tZrSe2 P312 M-star (local only, not committed) |
| Sampled moire HSP | MM |
| Valley label | M3_valley |
| ValleyScope operation ID | op5 |
| Eigenvalue | ~±0.99i (near-unity C2 spinful) |
| Readiness | `diagnostic_only: True`, `spinor_convention_verified: False` |

### Candidate ValleyScope Irrep Keys

| Bilbao Source Label | Source Character | ValleyScope Key | Multiplicity |
|--------------------|-----------------|-----------------|-------------|
| `-M3` | -i | `MM:C2_spinor_phase_-1/4` | 1 |
| `-M4` | +i | `MM:C2_spinor_phase_+1/4` | 1 |

The candidate phase mapping is direct: -i → -1/4, +i → +1/4, conditional
on the source-op assignment and local generator orientation that identify
Bilbao op6 with the valley-preserving C2 generator for M3 at MM.

### Non-Buildable Pseudo-Spec

```json
{
  "non_buildable_example": true,
  "_comment": "NON-BUILDABLE SKELETON — for review discussion only",
  "build_failure_reason": "draft schema_version plus review-only provenance; do not pass to build-reduced-ebr-table",
  "schema_version": "draft-c2-mm-m3-0.1",
  "status": "review_only_not_buildable",
  "data_source": "irreptables",
  "space_group_number": "149",
  "spinful": true,
  "subspace_group_candidate": "C2_like",
  "expected_hsps": ["MM"],
  "allowed_irrep_keys": [
    "MM:C2_spinor_phase_-1/4",
    "MM:C2_spinor_phase_+1/4"
  ],
  "source_hsp_by_irrep": {
    "-M3": "MM",
    "-M4": "MM"
  },
  "source_label_context": {
    "-M3": {"bilbao_hsp": "M", "valleyscope_sampled_hsp": "MM"},
    "-M4": {"bilbao_hsp": "M", "valleyscope_sampled_hsp": "MM"}
  },
  "valleyscope_irrep_multiplicity_by_source_irrep": {
    "-M3": {"MM:C2_spinor_phase_-1/4": 1},
    "-M4": {"MM:C2_spinor_phase_+1/4": 1}
  },
  "provenance": {
    "_comment": "NOT a reviewed provenance block — placeholder only",
    "review_status": "not_reviewed",
    "candidate_source_op": "Bilbao M op6",
    "candidate_valleyscope_op": "ValleyScope MM op5",
    "source_op_spinor_lift_convention": "direct phase mapping +i->+1/4, -i->-1/4 after source-op and generator orientation are accepted (candidate, not reviewed)"
  }
}
```

## Excluded / Blocked Rows

### GammaM M1/M2/M3

- **Blocker**: `blocked_by_coordinate_convention`.  Three Bilbao C2 ops
  (4,5,6) at GM all give identical source characters for a given irrep,
  so the Bilbao source character alone cannot distinguish which op is
  the valley-preserving generator for each M-star label.
- **Blocker**: eigenvalue quality is poor (~0.34i-0.74i, not close to
  the clean spinful C2 values ±i).
- **Required**: physicist signoff on valley-mapping orientation
  convention (which C2 axis preserves each M-star label at GammaM)
  plus improved D_raw closure / seed overlap quality.

### KM M1/M2/M3

- **Blocker**: `blocked_by_hsp_star_derivation`.  Bilbao K-source rows
  expose only C3-like ops (1,2,3) — no C2 characters.  C2 data for KM
  valleys must be derived by HSP-star conjugation from MM or GammaM
  source rows.

### MM M1/M2

- **Blocker**: `blocked_by_hsp_star_derivation`.  Bilbao M-source rows
  expose only op6.  M1/M2 C2 characters at MM would need HSP-star
  derivation, not direct Bilbao lookup.

### 3D Source Rows (A, L, etc.)

- **Out of scope**.  A/L source rows at kz=1/2 are full 3D HSPs.  They
  are only relevant if a future C2 reduced spec explicitly includes
  those 3D HSPs in the sampled set.

## Why This Cannot Yet Produce A Reviewed Table

1. **Incomplete sampled-HSP reduced basis**: only MM/M3 has a candidate
   source-op assignment.  GammaM and KM rows are blocked.  A reviewed
   C2 reduced EBR table needs at least one complete sampled HSP with
   all valley-preserving source labels mapped.

2. **Untrusted tZrSe2 readiness**: all eigenphase rows are
   `diagnostic_only: True` with `spinor_convention_verified: False`.
   No topology input is trusted for any C2 row.

3. **No package-data review record**: the candidate mapping is not
   reviewed.  Even if the convention were signed off, the candidate
   must pass through the reviewed package-data provenance gate before
   any C2 table can be shipped.

## Next Steps After Skeleton Review

1. Physicist signoff on the C2 source-op / spinor-lift convention
   (candidate: direct mapping +i→+1/4, -i→-1/4 after source-op and
   local generator orientation are accepted).
2. Physicist signoff on Bilbao op6 → ValleyScope MM op5 generator
   assignment for M3 at MM.
3. Resolve GammaM orientation convention (which Bilbao C2 op → which
   M-star valley) and improve eigenvalue quality.
4. After (1) and (2): extend skeleton to a reviewed mapping spec with
   provenance signoff.
5. After all blocks cleared: build reviewed C2 package-data table via
   `valleyscope build-reduced-ebr-table`.

No C2 package-data table is shipped by this skeleton.
