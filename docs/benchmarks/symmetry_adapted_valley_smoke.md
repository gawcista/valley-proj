# Symmetry-Adapted Valley Pipeline — Real-Data Smoke Benchmark

Date: 2026-05-25
Branch: `fix/zrse2-local-p2-subspaces`
Base: local `main` at `93d39f3`
Previous run: `cc/real-smoke-representative-diagnostics` at `c57e63b`

## Changes Since Previous Run

This run keeps the workflow experimental and default-off, but makes the
experimental report more audit-ready:

- representative candidate comparisons are now stored as machine-readable
  per-valley pairwise projector differences;
- accepted representatives record `selected_representative_by_valley`;
- inequivalent representatives record `selection_policy=none` and
  `representative_auto_selected=false`;
- the experimental workflow injects an identity representation when `D_raw`
  omits the order-1 operation, because every valley-preserving subgroup contains
  identity;
- downstream representation / character diagnostics are skipped when projector
  construction has already failed;
- `irrep_matching_input_ready`, `irrep_matching_input_status`, and
  `irrep_matching_input_reason` define the gate that future table matching must
  use;
- the experimental report now also includes
  `valley_preserving_subspaces`, a singleton per-valley view that restricts
  each report to operations preserving that valley inside the HSP little group;
- the workflow passes configured `rotation.unitarity_tol` and
  `rotation.root_deviation_tol` into the experimental representation and
  character diagnostics.

No automatic representative selection policy is implemented.

## Test Configuration

Config paths relative to repo root:

- `real_tests/tMoTe2/analyze.yaml` plus
  `analysis.symmetry_adapted_valley.enabled: true`
- `real_tests/tZrSe2/analyze.yaml` plus
  `analysis.symmetry_adapted_valley.enabled: true`
- temporary experimental configs and output directories are gitignored.

Commands:

```bash
python -m valleyscope.cli analyze-hsp real_tests/tMoTe2/analyze_experimental.yaml
python -m valleyscope.cli analyze-hsp real_tests/tZrSe2/analyze_experimental.yaml
pytest -q
```

## 1. tMoTe2 — K/K' Valley Benchmark (P321, No. 150)

### Summary

| Kpoint | Status | Main reason |
|---|---|---|
| GammaM full K/K' orbit | diagnostic_only | ambiguous inequivalent K_valley -> Kp_valley representatives |
| KM full K/K' orbit | diagnostic_only | ambiguous inequivalent K_valley -> Kp_valley representatives |
| MM full K/K' orbit | diagnostic_only | unique representative exists, but completeness fails |
| GammaM/KM singleton K and K' preserving subspaces | ok | C3-preserving local reports ready |
| MM singleton K and K' preserving subspaces | ok | identity-only local reports ready |

### GammaM / KM Representative Diagnostics

Operations 3, 4, and 5 all map `K_valley -> Kp_valley`.

Machine-readable fields:

- `representative_resolution_by_valley["Kp_valley"] =
  "ambiguous_inequivalent_candidates"`
- `representative_selection_policy_by_valley["Kp_valley"] = "none"`
- `selected_representative_by_valley["Kp_valley"] = null`
- `representative_auto_selected_by_valley["Kp_valley"] = false`

Pairwise projector differences:

| Kpoint | candidate pair | projector difference |
|---|---:|---:|
| GammaM | 3 vs 4 | 8.62e-6 |
| GammaM | 3 vs 5 | 0.999996 |
| GammaM | 4 vs 5 | 0.999987 |
| KM | 3 vs 4 | 5.83e-7 |
| KM | 3 vs 5 | 1.000000 |
| KM | 4 vs 5 | 0.999999 |

Interpretation: operations 3 and 4 are nearly equivalent for the selected
target subspace, but operation 5 produces an O(1)-different projector. The
candidate set is therefore not equivalent as a whole, and the workflow correctly
keeps the orbit diagnostic-only instead of auto-selecting.

### MM Projector Diagnostics

After identity is included, the MM K/K' orbit no longer fails because of a
missing valley-preserving operation. It now reaches a more physical diagnostic:

| Quantity | Value |
|---|---:|
| selected_rank | 1 |
| rank_source | gap |
| representative K -> Kp | operation 5 |
| seed_overlap(K_valley) | 0.9918 |
| seed_overlap(Kp_valley) | 7.3e-5 |
| completeness_error | 0.5000 |

This means the unique operation-5 propagation from the K seed does not recover
the q-cut K' seed in the target subspace. The result remains diagnostic-only.

## 2. tZrSe2 — M-Star Valley Benchmark (P312, No. 149)

### Summary

| Kpoint | Orbit | Status | Main reason |
|---|---|---|---|
| GammaM | M1/M2/M3 full orbit | diagnostic_only | rank gap insufficient after identity + C2_M1 symmetrization |
| GammaM | M1, M2, M3 singleton preserving subspaces | warn | three local rank-2 P2-like subspaces found |
| KM | M1/M2/M3 | diagnostic_only | unique representatives, but propagated projectors fail quality checks |
| MM | M1/M2 | diagnostic_only | unique representative, but orthogonality/idempotency fail |
| MM | M3 full orbit | ok | singleton projector and C2 character diagnostics pass configured tolerances |

### GammaM

Including identity changes the reference-valley symmetrization. The run now
fails before representative selection:

| Quantity | Value |
|---|---:|
| rank_source | gap_insufficient |
| max rank gap | 0.3244 |
| rank_tol | 0.5 |

This remains the correct result for the full M-star orbit construction: the
reference projector purification itself is not robust enough under automatic
gap rank selection.

The new singleton preserving-subspace view separately reports the three local
P2-like subspaces expected from the three C2 operations:

| Valley | preserving ops | selected_rank | status | C2 eigenphases | seed overlap |
|---|---:|---:|---|---|---:|
| M1 | [0, 4] | 2 | warn | -0.25, +0.25 | 0.739 |
| M2 | [0, 3] | 2 | warn | -0.25, +0.25 | 0.659 |
| M3 | [0, 5] | 2 | warn | +0.25, +0.25 | 0.711 |

The warnings are physical diagnostics: the q-cut seed projectors are not
already symmetry-consistent, so the C2-preserving projected subspaces deviate
from the raw q-cut seed subspaces. They are still local symmetry-adapted
subspaces under the configured real-data tolerances.

### KM

The orbit has unique representatives:

- `M1_valley -> M2_valley`: operation 2
- `M1_valley -> M3_valley`: operation 1

Projector construction still fails:

| Quantity | Value |
|---|---:|
| selected_rank | 2 |
| rank_source | gap |
| seed_overlap(M1) | 0.901 |
| seed_overlap(M2) | 0.0739 |
| seed_overlap(M3) | 0.0502 |
| orthogonality_error | 0.0725 |
| total_projector_idempotency_error | 0.4593 |

The propagated projectors do not remain a clean valley-star decomposition.

### MM

The M1/M2 orbit has a unique representative:

- `M1_valley -> M2_valley`: operation 5

It still fails projector quality:

| Quantity | Value |
|---|---:|
| selected_rank | 2 |
| rank_source | gap |
| seed_overlap(M1) | 0.917 |
| seed_overlap(M2) | 0.838 |
| orthogonality_error | 0.0831 |
| total_projector_idempotency_error | 0.3064 |

The singleton M3 orbit remains the cleanest nontrivial MM projector diagnostic:

| Quantity | Value |
|---|---:|
| projector status | ok |
| selected_rank | 2 |
| rank_source | gap |
| seed_overlap(M3) | 0.957 |
| C2_M3 eigenphases | -0.25, +0.25 |
| max valley-preserving unitarity error | 1.82e-2 |
| max eigenvalue modulus deviation | 6.46e-3 |

The representation and character layers pass the configured real-data
tolerances. `irrep_matching_input_ready` remains false because the spinor
convention is not benchmark-verified.

## 3. Projector Symmetry-Consistency Context

The q-cut seed projectors still fail the seed projector symmetry-consistency
check for the important valley-changing rows:

- tMoTe2: operation 5 gives epsilon near 1 at GammaM/KM/MM.
- tZrSe2: all GammaM M-star rows fail with epsilon about 1.08-1.33.
- tZrSe2 MM operation 5 fails for M1/M2 exchange with epsilon about 0.408 and
  warns for M3 preservation with epsilon about 0.013.

This confirms the methodology reset: q-cut projectors are seed diagnostics, not
trusted valley irrep bases.

## 4. Failure Classification

| Failure type | tMoTe2 | tZrSe2 |
|---|---|---|
| equivalent representatives accepted | none | none |
| ambiguous inequivalent candidates | GammaM, KM | none in current gated run |
| missing representative operation | none | none |
| failed rank selection | none | GammaM full orbit |
| failed completeness | MM | none |
| failed orthogonality/idempotency | none | KM, MM M1/M2 |
| representation / character readiness failure | none after projector-failure gate | none under configured tolerances |
| successful local preserving-subspace diagnostic | GammaM/KM K/K' C3, MM K/K' identity-only | GammaM M1/M2/M3 C2, KM identity-only, MM M1/M2 identity-only, MM M3 C2 |

## 5. Readiness for Irrep Matching

The experimental workflow is now ready for the next implementation step:
table-based valley-preserving irrep matching can be added behind the strict
`irrep_matching_input_ready` gate.

The real smoke cases remain blocked:

| Material | Kpoint/orbit | `irrep_matching_input_status` | Reason |
|---|---|---|---|
| tMoTe2 | full K/K' orbits | blocked | projector construction failed |
| tMoTe2 | singleton K/K' preserving subspaces | ready | spinor convention verified; local preserving reports pass |
| tZrSe2 | GammaM full M-star orbit | blocked | projector construction failed |
| tZrSe2 | GammaM local M1/M2/M3 P2 subspaces | blocked | spinor convention unverified |
| tZrSe2 | KM M-star | blocked | projector construction failed |
| tZrSe2 | MM M1/M2 | blocked | projector construction failed |
| tZrSe2 | MM M3 | blocked | spinor convention unverified |

Matching implementation requirements:

1. Keep `ambiguous_inequivalent_candidates` diagnostic-only with no
   auto-selection.
2. Keep projector failures as hard gates: downstream representations and
   characters are not evaluated from zero or invalid bases.
3. Only match orbit reports with `irrep_matching_input_ready=true`.
4. Require projector status `ok`.
5. Require representation and sewing unitarity within reviewed tolerance.
6. Require spinor convention verified for spinful labels.
7. Record `irrep_matching_status` separately from projector construction.

Do not loosen real-data representation tolerances silently to make the smoke
cases pass.

## 6. pytest Result

```bash
pytest -q
# 295 passed
```

## 7. Public Terminology Check

Generated JSON uses projector symmetry-consistency and
valley-preserving-subgroup terminology. The experimental report does not use
old projector or subgroup terminology as public schema.
