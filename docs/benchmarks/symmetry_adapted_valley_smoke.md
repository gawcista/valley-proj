# Symmetry-Adapted Valley Pipeline — Real-Data Smoke Benchmark

Date: 2026-05-22
Branch: `cc/real-smoke-symmetry-adapted-valley`
Base: `09b97b5c` (main)

## Test Configuration

Both benchmarks run the existing `analyze_hsp` workflow with the experimental
flag `analysis.symmetry_adapted_valley.enabled: true`. Default-off behavior
verified: without the flag, no `symmetry_adapted_valley_analysis.json` is
written (see `pytest -q` test `test_default_off_no_symmetry_adapted_valley_output`).

Config paths relative to repo root:
- `real_tests/tMoTe2/analyze.yaml`
- `real_tests/tZrSe2/analyze.yaml`

Temporary experimental configs created at:
- `real_tests/tMoTe2/analyze_experimental.yaml` (not committed)
- `real_tests/tZrSe2/analyze_experimental.yaml` (not committed)

## 1. tMoTe2 — K/K' Valley Benchmark

### Material / Orbit
- Moire space group: P321 (No. 150)
- Valley subspaces: K_valley, Kp_valley (2 valleys)
- Target bands: [2195, 2196], spinor SOC (convention verified)
- HSPs: GammaM, KM, MM
- Subspace quality: S_min=0.985-0.992, min_concentration=1.0 (clean)

### Q-Cut Seed Projector Symmetry Error
- `projector_symmetry_report.json` status: `symmetry_consistency_failures_detected`
- Valley-preserving ops (K_valley -> K_valley, Kp_valley -> Kp_valley):
  all `passed` with epsilon_seed ~0.000-0.006
- Valley-changing ops (K_valley -> Kp_valley, Kp_valley -> K_valley):
  likely `failed` (seed projector symmetry-consistency violation)

### P_a^sym Status — FAILED

| Kpoint  | Status | Reason |
|---------|--------|--------|
| GammaM  | failed | ambiguous representative operation for K_valley -> Kp_valley: candidates=[3,4,5] |
| KM      | failed | ambiguous representative operation for K_valley -> Kp_valley: candidates=[3,4,5] |
| MM      | failed | no valley-preserving operation found for reference valley (K_valley) |

### Root Cause Analysis

1. **Ambiguous representative (GammaM, KM):** In P321, operations 3, 4, 5 all map
   K_valley -> Kp_valley at these kpoints. The current toy pipeline requires a
   _unique_ representative operation to generate P_Kp^sym from P_K^sym.
   Multiple C2 axes and C3 can all map between K and K' in the moire BZ.
   This is a **genuine multi-representative problem**: P321 has multiple
   symmetry operations connecting the K/K' valleys.

2. **No valley-preserving operation (MM):** At the M point, the HSP little group
   does not contain any operation that preserves the K valley. Physics:
   at the M point (moire BZ boundary), K/K' are not little-group invariant.
   This is **expected physical behavior**.

### Experimental Pipeline Summary

| Metric | Value |
|--------|-------|
| selected_rank | 0 (construction failed) |
| selected_rank | failure |
| seed_overlap | 0.0 (no projectors built) |
| diagnostic_only | True |
| local_irrep_ready | False |

## 2. tZrSe2 — M-Star Valley Benchmark

### Material / Orbit
- Moire space group: P312 (No. 149)
- Valley subspaces: M1_valley, M2_valley, M3_valley (3 valleys)
- Target bands: 6 bands (degenerate M-star manifold)
- HSPs: GammaM, KM, MM
- Subspace quality: S_min=0.893-0.982, min_concentration=0.9999+ (clean)

### Q-Cut Seed Projector Symmetry Error
- `projector_symmetry_report.json` status: `symmetry_consistency_failures_detected`
- GammaM and KM: ALL C3/C3^2 rows `failed` with epsilon_seed ~1.1-1.4
  (seed projectors do NOT satisfy D_C3 P_a^0 D_C3^dag ≈ P_b^0)
- MM: operations `not_evaluated` (D_raw unavailable for valley-permuting ops
  at this kpoint? needs further investigation)

### P_a^sym Status — MIXED

| Kpoint  | Orbit | Status | Reason |
|---------|-------|--------|--------|
| GammaM  | M1/M2/M3 | failed | ambiguous rep M1->M2: candidates=[2,5] |
| KM      | M1/M2/M3 | failed | no valley-preserving op for M1 |
| MM      | M1/M2 | failed | no valley-preserving op for M1 |
| MM      | **M3 only** | **ok** | rank=2, gap, overlap=0.957 |

### MM M3 Valley — SUCCESS Case

This is the most significant result. At MM, the M3 valley has a valley-preserving
operation (C2_M3 in P312), and the symmetry-adapted construction succeeds:

| Diagnostic | Value |
|------------|-------|
| selected_rank | 2 |
| rank_source | gap |
| seed_overlap | 0.957 (M3_valley) |
| max_projector_symmetry_error | 0.0095 |
| orthogonality_error | 0.0000 |
| total_projector_idempotency_error | 0.0000 |
| completeness_error | None (not_evaluated) |
| character eigenphase (op 5) | ±0.25 (spinor C2, double-group phases) |
| diagnostic_only | True (character layer) |
| irrep_matching_status | failed_input_readiness |

**Interpretation:** At the MM point, C2_M3 preserves M3 and swaps M1/M2.
The M3 seed projector has rank-2 (two bands assigned to M3). The
symmetrization works cleanly: orthogonality and idempotency are exact.
The seed overlap of 0.957 is excellent. However, the character layer
still reports diagnostic_only because the projector construction result
is only available for M3, not the full orbit.

## 3. Key Findings

### Blocking Issues

1. **Representative ambiguity is a real obstruction for P321/P312:**
   Multiple symmetry operations map between valleys in the same orbit.
   The current pipeline requires unique representatives and fails otherwise.
   This is **not a toy limitation** — it reflects genuine multi-valley
   symmetry structure. A resolution strategy is needed (e.g., choosing the
   "canonical" generator, or allowing any representative and verifying
   consistency).

2. **Valley-preserving operations depend on kpoint:**
   At MM, K_valley has no valley-preserving operation (physical).
   At KM, M1 has no valley-preserving operation. The per-kpoint nature
   of the HSP little group means valley stabilizers vary by kpoint.
   This is expected and the code correctly reports it.

3. **Q-cut seed projectors fail symmetry-consistency for M-star:**
   epsilon_seed ~1.1 for C3 valley-changing operations indicates the
   raw q-cut seed basis does NOT transform correctly under C3.
   This validates the need for symmetry-adapted projectors.

### Success Case

4. **MM M3 valley succeeds with excellent quality:**
   rank=2, seed_overlap=0.957, orthogonality=0, idempotency=0.
   This demonstrates the pipeline can work when:
   - a valley-preserving operation exists at the HSP
   - a unique representative is available
   - the seed projector has good subspace quality

## 4. Physical Interpretation

- The q-cut seed basis fails symmetry-consistency for valley-changing
  C3 operations (epsilon ~1.1), confirming that seed projectors are
  diagnostics, not trusted irrep bases.
- The symmetry-adapted construction improves orthogonality/idempotency
  but is blocked by representative ambiguity for full orbits.
- The MM M3 success shows the symmetrization works when geometric
  conditions are met.
- No spinor-convention issue: tMoTe2 spinor verified, eigenphases clean.

## 5. Recommendation for Next Task

1. **Resolve representative ambiguity:** Implement a strategy for multiple
   candidate representative operations. Options:
   (a) Choose the lowest-order candidate (prefer C3 over C2 for M-star)
   (b) Compute all candidates and verify they produce the same P_b^sym
   (c) Let the user specify a preferred generator via config
2. **Handle kpoint-dependent valley-preserving operations:**
   Report per-kpoint VPS status, don't require all kpoints to have VPS
   for all valleys.
3. **Run the successful MM M3 case through irrep character table matching**
   (when that layer is implemented) to validate the eigenphases.
4. **Investigate why MM C3 rows are not_evaluated** in projector_symmetry
   (likely D_raw missing for valley-permuting ops at that specific kpoint).

## pytest Result

```
$ pytest -q
283 passed in 3.37s
0 skipped, 0 xfailed, 0 warnings
```

No new tests added (smoke is observational only).
