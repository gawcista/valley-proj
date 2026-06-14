# ValleyScope C2 Source-Op Mapping Audit

Date: 2026-06-14 | Status: Audit document (no C2 table shipped)

## Scope

This audit compares Bilbao SG 149 (P312) spinful source operations
and characters against the ValleyScope C2-like valley-preserving rows
observed in the tZrSe2 M-star P312 validation fixture.  It assigns
candidate source-HSP/op mappings where evidence is sufficient and marks
blocked assignments explicitly.

No C2-like reduced EBR table is shipped by this audit.

## 1. Bilbao SG149/P312 Source Operations

From `irreptables.irreps.IrrepTable("149", True)`:

| Bilbao Op | Spatial Order | R-matrix row 0 | SU(2) S eigenvalues | C2-like? |
|-----------|-------------|----------------|--------------------|----------|
| op1 | 1 (E) | [1,0,0] | (+1, +1) | No |
| op2 | 3 (C3) | [0,-1,0] | exp(±iπ/3) | No |
| op3 | 3 (C3²) | [-1,1,0] | exp(∓iπ/3) | No |
| op4 | 2 | [0,-1,0] | **(+i, -i)** | **Yes** |
| op5 | 2 | [-1,1,0] | **(+i, +i)** | **Yes** |
| op6 | 2 | [1,0,0] | **(+i, -i)** | **Yes** |

Ops 4-6 are spatial order-2 rotations (trace = -1, det = +1) with
SU(2) spinor eigenvalues ±i.  S² = -I is verified for all three.
These are the Bilbao source C2-like operations.

### Bilbao Source HSP Availability

| Source HSP | Exposed Ops | C2 Ops | 1D C2 Characters |
|-----------|------------|--------|-----------------|
| GM (GammaM) | 1-6 | 4,5,6 | `-GM4`: all -i (phase -1/4); `-GM5`: all +i (phase +1/4) |
| M | 1,6 | 6 | `-M3`: -i (phase -1/4); `-M4`: +i (phase +1/4) |
| A | 1-6 | 4,5,6 | C2 chars available (not in C2 sampled HSP scope) |
| L | 1,6 | 6 | C2 chars available (not in C2 sampled HSP scope) |
| K | 1,2,3 | none | C3-like only — NOT C2 source evidence |
| H | 1,2,3 | none | C3-like only — NOT C2 source evidence |

K and H source rows are C3-like only in this Bilbao table and must
not be used as C2 source-character evidence.

## 2. tZrSe2 M-Star ValleyScope C2-Like Rows

From debug-profile run of `real_tests/tZrSe2/analyze.yaml` (local fixture,
not committed):

| kpoint | ValleyScope Op ID | Target Valley | Eigenvalue (approx.) | Bilbao Candidate |
|--------|------------------|--------------|---------------------|-----------------|
| GammaM | 3 | M2_valley | ~±0.34i | op4, op5, or op6 at GM |
| GammaM | 4 | M1_valley | ~-0.68i, ~+0.33i | op4, op5, or op6 at GM |
| GammaM | 5 | M3_valley | ~+0.74i, ~+0.07i | op4, op5, or op6 at GM |
| MM | 5 | M3_valley | ~±0.99i | op6 at M (M-source rows) |

All rows are `diagnostic_only: True`, `spinor_convention_verified: False`.

The Bilbao table has three distinct C2 ops (4,5,6) at GM.  ValleyScope
assigns specific operation IDs (3,4,5) to specific M-star valleys.
The mapping from Bilbao op {4,5,6} to ValleyScope op {3,4,5} depends
on the valley-preserving subgroup generator selection, which is a
coordinate/orientation convention.

## 3. Candidate Source-HSP/Op Assignments

### 3a. MM M3 — Evidence Sufficient For Candidate

Bilbao M-source rows (`-M3`, `-M4`) expose only op6 as the C2-like
operation.  The tZrSe2 MM M3 row uses ValleyScope op=5 with eigenvalue
~±0.99i — clean and close to the spinful C2 values ±i.

Candidate assignment:
- Source HSP: M (Bilbao) ↔ MM (ValleyScope moire HSP)
- Source op: Bilbao op6
- Source character: `-M3` → -i (phase -1/4), `-M4` → +i (phase +1/4)
- ValleyScope row: MM/M3 has near-unity C2 eigenvalue ~+i at op5
- Candidate phase mapping: +i → `C2_spinor_phase_+1/4`, -i → `C2_spinor_phase_-1/4`

**Status: candidate_mapping_identified.**  The eigenvalue ~±0.99i at
MM M3 is the closest to a clean C2 spinful value.  The mapping
Bilbao op6 → ValleyScope op5 is a coordinate/orientation convention.
The phase mapping +i → +1/4 appears direct (no sign conversion needed).

Blocker: `spinor_convention_verified: False` for the MM M3 row.
The candidate mapping is physically consistent but not signed off.

### 3b. GammaM M1/M2/M3 — Evidence Insufficient For Single-Op Mapping

The Bilbao GM source rows have three C2 ops (4,5,6) all giving the
same character (±i) for each 1D irrep.  Any of the three C2 ops
could be the valley-preserving generator for a given valley label.
The assignment of a specific Bilbao op to a specific valley is a
generator-orientation convention that the Bilbao table alone cannot
resolve.

| ValleyScope Row | Valley | Eigenvalue Quality | Candidate Bilbao Source |
|----------------|--------|-------------------|------------------------|
| GammaM op=3 | M2 | ~±0.34i (not clean) | Bilbao GM `-GM4`/`-GM5` via op4/5/6 |
| GammaM op=4 | M1 | ~-0.68i / ~+0.33i (not clean) | Bilbao GM `-GM4`/`-GM5` via op4/5/6 |
| GammaM op=5 | M3 | ~+0.74i / ~+0.07i (not clean) | Bilbao GM `-GM4`/`-GM5` via op4/5/6 |

All three GammaM rows have eigenvalues that deviate significantly from
the clean spinful C2 values ±i.  The MM M3 row is the only one with
near-unity C2 character.

**Status: blocked_by_coordinate_convention.**  The Bilbao GM C2 ops
(4,5,6) produce identical characters for a given source irrep, so the
source character alone cannot distinguish which op is the
valley-preserving generator for a particular M-star label.  This
assignment requires an explicit valley-mapping orientation convention:
which spatial C2 axis preserves each M-star valley label at GammaM.

Additionally, the GammaM eigenvalue quality is poor — the deviation
from ±i indicates that the ValleyScope C2 eigenphase extraction at
GammaM is affected by seed-overlap and D_raw closure quality, not
just convention.

## 4. Candidate Phase Mapping

For the rows where candidate assignments exist, the phase mapping
appears direct:

| Bilbao Source Character | Bilbao Phase | ValleyScope Eigenphase | ValleyScope Key |
|------------------------|-------------|----------------------|-----------------|
| -i | -1/4 | ≈ -i | `C2_spinor_phase_-1/4` |
| +i | +1/4 | ≈ +i | `C2_spinor_phase_+1/4` |

For the clean near-unity eigenvalue at MM M3 (~+0.99i), the candidate
would be `C2_spinor_phase_+1/4`.  The Bilbao source character is +i
(from `-M4`), and the phase mapping is direct (+i → +1/4) with no
sign conversion needed.

This is a CANDIDATE mapping, not a reviewed one.  The mapping is marked
`blocked_pending_spinor_convention_review` until the central-sign
convention for C2 is signed off.

## 5. Per-Row Status Summary

| ValleyScope Row | Bilbao Source Candidate | Evidence Quality | Blocker |
|----------------|------------------------|-----------------|---------|
| MM M3 op=5 | M-source op6 / `-M3` or `-M4` | Good (~±0.99i) | `spinor_convention_verified: False` |
| GammaM M1 op=4 | GM-source op4/5/6 / `-GM4` or `-GM5` | Poor (~0.68i/0.33i) | `blocked_by_coordinate_convention` + eigenvalue quality |
| GammaM M2 op=3 | GM-source op4/5/6 / `-GM4` or `-GM5` | Poor (~0.34i) | `blocked_by_coordinate_convention` + eigenvalue quality |
| GammaM M3 op=5 | GM-source op4/5/6 / `-GM4` or `-GM5` | Poor (~0.74i/0.07i) | `blocked_by_coordinate_convention` + eigenvalue quality |
| KM M1/M2/M3 | none directly (K/H have no C2 chars) | — | `blocked_by_hsp_star_derivation` (needs nonlocal derivation from MM/GammaM) |
| MM M1/M2 | none directly (M-source has only op6) | — | `blocked_by_hsp_star_derivation` |

## 6. Conclusion

### Evidence-Based

- **Bilbao SG149 GM and M source rows provide C2 characters ±i directly.**
  The source phase convention matches the ValleyScope C2 spinful phase
  set ±1/4 at the level of eigenvalue values.  No fake `-1 → ±i`
  conversion is needed.

- **MM M3 is the most promising candidate** for a first C2 reduced-basis
  row: eigenvalue ~±0.99i, clear Bilbao source (`-M3`/`-M4` via op6),
  and a direct phase mapping (+i → +1/4, -i → -1/4).

### Blocked

- **GammaM M1/M2/M3 source-op assignment is blocked** by the
  coordinate/orientation convention: three Bilbao C2 ops (4,5,6) at GM
  all give identical source characters for a given irrep, so the
  assignment of which op preserves which valley label is convention-
  dependent and requires explicit physicist signoff.

- **GammaM eigenvalue quality is insufficient** for reduced EBR input:
  deviations from ±i are O(0.3) or larger, well above trusted
  thresholds.

- **KM and nonlocal MM rows require HSP-star derivation**, not direct
  Bilbao source-op lookup.  This is a methodology step, not a source-
  data gap.

- **Spinor convention remains unverified** for all rows.  This is the
  primary blocker for any C2 reduced EBR table.

### Next Steps

1. Physicist signoff on C2 central-sign convention (candidate: direct
   mapping +i → +1/4, -i → -1/4).
2. Physicist signoff on valley-preserving C2 generator orientation
   for each M-star label at each HSP.
3. Improved tZrSe2 GammaM D_raw closure / seed overlap quality, or
   identification of an alternative C2 validation fixture with cleaner
   eigenphase extraction.

No C2 package-data table is shipped by this audit.
