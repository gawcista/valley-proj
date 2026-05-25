# Symmetry-Adapted Valley Pipeline — Real-Data Smoke Benchmark

Date: 2026-05-25
Branch: `cc/real-smoke-representative-diagnostics`
Base: `a260cd8e` (main)
Previous run: `cc/real-smoke-symmetry-adapted-valley` at `09b97b5c`

## Changes Since Previous Run

The representative ambiguity resolution introduced in `cc/representative-ambiguity-resolution`
(now on main) replaces the old "fail on first ambiguous candidate" behavior with
candidate equivalence checking. When multiple operations map a0->a, the pipeline
constructs P_a^sym for each candidate and compares pairwise by ||P(c_i)-P(c_j)||_F.
If all candidates produce equivalent projectors within tolerance, the ambiguity is
resolved (representative_resolution="equivalent_candidates"). Otherwise it fails
with "ambiguous_inequivalent_candidates" and the max projector difference.

This benchmark re-runs both materials with the new diagnostics to determine
whether the ambiguity is resolved by equivalence or remains genuinely ambiguous.

## Test Configuration

Config paths relative to repo root:
- `real_tests/tMoTe2/analyze.yaml` + `analysis.symmetry_adapted_valley.enabled: true`
- `real_tests/tZrSe2/analyze.yaml` + `analysis.symmetry_adapted_valley.enabled: true`
- Temporary experimental configs not committed; output dirs gitignored.

## 1. tMoTe2 — K/K' Valley Benchmark (P321, No. 150)

### Summary

| Kpoint  | Status | Reason |
|---------|--------|--------|
| GammaM  | diagnostic_only | ambiguous_inequivalent_candidates K->Kp: [3,4,5] max_diff=1.00 |
| KM      | diagnostic_only | ambiguous_inequivalent_candidates K->Kp: [3,4,5] max_diff=1.00 |
| MM      | diagnostic_only | no valley-preserving operation for K_valley |

### Representative Ambiguity — NOT RESOLVED

At GammaM and KM, operations 3, 4, 5 all map K_valley -> Kp_valley.
Candidate equivalence check: max projector difference = 1.00 (O(1) — genuinely different).
These are distinct C2 axes in P321.
The different C2 operations produce substantially different target projectors
(difference ~1), confirming they represent physically distinct valley mappings.

Machine-readable diagnostics:
- representative_resolution: ambiguous_inequivalent_candidates
- representative_resolution_by_valley: {"Kp_valley": "ambiguous_inequivalent_candidates"}
- representative_candidates_by_valley: {"Kp_valley": [3, 4, 5]}
- representative_projector_difference_by_valley: {"Kp_valley": 0.9999...}
- max_projector_difference: 1.0000e+00

### MM — No Valley-Preserving Operation

At the M point, K_valley has no operation preserving it in the HSP little group.
Physics: at the moire BZ M-point, K/K' reciprocal momenta are not invariant
under the M-point little group. Expected behavior.

## 2. tZrSe2 — M-Star Valley Benchmark (P312, No. 149)

### Summary

| Kpoint  | Orbit | Status | Reason |
|---------|-------|--------|--------|
| GammaM  | M1/M2/M3 | diagnostic_only | ambiguous_inequivalent M1->M2: [2,5] max_diff=1.75 |
| KM      | M1/M2/M3 | diagnostic_only | no valley-preserving operation for M1 |
| MM      | M1/M2 | diagnostic_only | no valley-preserving operation for M1 |
| MM      | **M3** | **diagnostic_only** | projector OK (rank=2, overlap=0.957) |

### MM M3 Valley — Projector Construction SUCCEEDS

This is the only case where the full pipeline reaches the character layer:

| Diagnostic | Value |
|------------|-------|
| proj_status | ok |
| selected_rank | 2 |
| rank_source | gap |
| purification_gap | 0.945 |
| seed_overlap (M3) | 0.957 |
| orthogonality_error | 0.0000 |
| idempotency_error | 5.6e-16 |
| max_projector_symmetry_error | 0.0095 |
| representative_resolution | unique |
| sewing_status | failed (no valley-changing ops with D_raw at MM) |
| character eigenphase (op 5, C2_M3) | ±0.25 (spinor double-group) |
| irrep_matching_status | failed_input_readiness |

The valley-preserving representation D_M3(C2_M3) = U_M3^dag D_{C2} U_M3 is a 2x2
matrix with eigenphases ±0.25 (double-group C2 spinor phases). This is physically
consistent with a spinor C2 rotation having eigenvalues ±i.

### Representative Ambiguity — NOT RESOLVED

At GammaM, operations 2 (C3^2) and 5 (C2_M3) both map M1 -> M2.
Candidate equivalence check: max projector difference = 1.75 (O(1) — genuinely different).
C3^2 and C2_M3 are fundamentally different operations (3-fold vs 2-fold rotation).
The produced target projectors differ by O(1), confirming these are not equivalent.

Machine-readable diagnostics:
- representative_resolution: ambiguous_inequivalent_candidates
- representative_resolution_by_valley: {"M2_valley": "ambiguous_inequivalent_candidates"}
- representative_candidates_by_valley: {"M2_valley": [2, 5]}
- representative_projector_difference_by_valley: {"M2_valley": 1.7530}
- max_projector_difference: 1.7530e+00

### KM — No Valley-Preserving Operation

At the K point, M1 has no operation preserving it. Physics: the M valleys
are not little-group invariant at the K point.

## 3. MM Projector Symmetry "Not Evaluated" — Resolved

Previous benchmark noted MM C3 rows in projector_symmetry_report.json as
"not_evaluated". Investigation result:

All operations 1-4 (C3, C3^2, and two C2 axes) are **not in the little group**
at the MM point. Operation 5 (C2_M3) IS in the little group at MM.
The "not_evaluated" with reason "not in little group" is correct behavior:
`build_raw_representations_for_kpoint` only computes D_raw for little-group
operations, so C3 D_raw matrices are never built for MM.

This is expected physics, not a code bug.

## 4. Failure Classification

| Failure type | tMoTe2 | tZrSe2 |
|-------------|--------|--------|
| equivalent representatives accepted | 0 | 0 |
| ambiguous inequivalent candidates | GammaM, KM | GammaM |
| missing representative operation | — | — |
| no valley-preserving operation | MM | KM, MM (M1/M2 orbit) |
| failed rank selection | — | — |
| low seed overlap | — | — |
| failed projector symmetry-consistency | — | — |
| failed sewing unitarity | — | — |
| not_evaluated (missing input/mapping) | — | — |
| successful projector w/ character | — | MM M3 (partial) |

## 5. Physical Interpretation

1. **Representative ambiguity is a fundamental physics problem, not a numerical artifact:**
   The inequivalent candidates produce projector differences of O(1), confirming
   they represent genuinely different symmetry operations. An explicit policy is
   needed to select among candidates.

2. **The MM M3 case validates the pipeline:** When a unique valley-preserving
   operation exists (C2_M3 at MM) and representative ops are unique, the
   construction succeeds with excellent quality (rank=2, overlap=0.957,
   orthogonality exact). The spinor C2 eigenphases of ±0.25 are physically
   correct.

3. **K-point-dependent valley-preserving subgroups are expected:** Different
   HSP little groups give different valley-preserving operations at each kpoint.

## 6. Recommendation

Do not auto-select among inequivalent representatives yet. The next task should
keep the run diagnostic-only and expose enough candidate-wise data to design an
auditable policy. Candidate policies to evaluate later:
(a) prefer an orbit-transport generator specified by valley-star convention
    (for example, C3 for a C3-related M-star, not an unrelated C2)
(b) allow the user to specify a preferred generator via config
(c) continue reporting all candidates and mark diagnostic_only without
    auto-selecting

Option (c) is the safest default until option (a) has a reviewed group-theory
definition for each valley-star type.

## 7. pytest Result

```
$ pytest -q
288 passed
0 skipped, 0 xfailed
```

## 8. Public Terminology Check

Verified in both `projector_symmetry_report.json` and
`symmetry_adapted_valley_analysis.json`: generated JSON uses the current
projector symmetry-consistency and valley-preserving-subgroup terminology.
