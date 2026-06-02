# tZrSe2 Blocker Evidence Protocol

Date: 2026-06-02

This document records the current blocker status of tZrSe2 in the ValleyScope
pipeline. Each blocker is classified as a physics/convention blocker, a
numerical/provenance blocker, or a code-design blocker. The checklist for
removing each blocker is stated explicitly. No current data is upgraded to
trusted.

## Current Overall Status

- `valley_ebr_problem_instances.status = no_instances`
- `valley_ebr_input_candidates.status = no_candidates`
- No trusted EBR input exists for any tZrSe2 valley.

## Blocker Inventory

### B1: spinor_convention_unverified

- **Classification**: physics/convention blocker
- **Source file**: `real_tests/tZrSe2/analyze.yaml` line 73-75
- **Current value**: `spinor.convention_verified: false`
- **Effect**: All valley-preserving character diagnostics and irrep matching
  results are downgraded to `diagnostic_only`. `irrep_matching_input_ready`
  is false at every kpoint/valley. The `direct_qcut` path cannot reach
  `trusted` readiness.
- **Why not yet verified**: The VASP spinor convention (`vasp_up_down_saxis_z`)
  determines the SU(2) rotation matrix used in `spin_rotation_matrix()`. If
  the convention in the code does not match the actual VASP calculation, the
  eigenphases of spinful C2/C3 operations would be shifted, producing wrong
  irrep labels.
- **Evidence that the convention may be correct** (diagnostic only): tZrSe2
  MM/M3/op=5 produces C2 eigenphases [-0.25, +0.25], which are the expected
  spinful C2 values. tMoTe2 uses the same convention and passes its benchmark
  (`tMoTe2_VBM_C3_literature`). This is suggestive but not proof.
- **Checklist to mark `convention_verified = true`**:
  1. Identify a known benchmark material with published spinful C2/C3 irrep
     labels that has been calculated with the same VASP version and spinor
     settings as tZrSe2.
  2. Run `analyze-hsp` on that benchmark and confirm the matched irrep labels
     agree with the published reference.
  3. Document the benchmark material, reference, and confirmation in this file.
  4. Only then set `convention_verified: true` in `analyze.yaml`.
- **Alternative if no benchmark exists**: compare `D_a` phases from two
  independent band subsets with the same irrep label; consistency across
  subsets provides evidence (but is weaker than an external benchmark).

### B2: D_raw target-subspace closure (op=5, C2_M3)

- **Classification**: numerical/provenance blocker
- **Source file**: `real_tests/tZrSe2/output/valley_analysis/target_subspace_closure.json`
- **Current values**:

  | Kpoint | op | closure_quality | raw_unitarity | max_residual | D_raw SV range | gram_error |
  |--------|----|-----------------|---------------|-------------|----------------|------------|
  | GammaM | 5 | usable_with_caution | 2.77e-2 | 1.16e-2 | [0.993, 0.997] | 1.18e-5 |
  | MM | 5 | usable_with_caution | 1.93e-2 | 1.29e-2 | [0.994, 0.998] | 7.31e-5 |

- **Effect**: The `usable_with_caution` quality is a target-subspace closure
  classification, not an EBR readiness pass. The M3 local representation remains
  `diagnostic_only` because valley-preserving representation unitarity is
  2.19e-2 at GammaM/M3 and 1.82e-2 at MM/M3, both larger than
  `representation_unitarity_fail_tol=1e-2`. The derived C2 characters for M1/M2
  from M3 source are blocked because the source is `diagnostic_only`.
- **Physics**: The 6-band target subspace (iband 2929-2934) captures 99.3-99.8%
  of the norm under C2 rotation. The lost 0.6-1.3% goes to bands outside the
  target window. This is a DFT plane-wave truncation effect, not a code bug.
  The singular values (0.993-0.998) are very close to 1, indicating mild
  non-closure.
- **Checklist to resolve**:
  1. Generate expanded-band HDF5 (e.g. iband 2925-2938). If D_raw closure and
     the resulting valley-preserving representation unitarity drop below
     `representation_unitarity_fail_tol=1e-2` with expanded bands, the
     local-representation diagnostic-only blocker is resolved. For EBR mapping
     readiness, the stricter `ebr_unitarity_max=1e-3` gate must also be checked.
  2. If expanded-band HDF5 is not available, consider implementing diagnostic
     polar-reunitarization reporting. This would NOT change the D_raw used for
     character computation but would provide evidence about whether
     reunitarization would close the gap.
  3. Do NOT loosen `representation_unitarity_fail_tol` beyond current 1e-2 or
     `ebr_unitarity_max` beyond default 1e-3 to make tZrSe2 pass.

### B3: Low seed overlap at GammaM

- **Classification**: numerical/provenance blocker (q-cut quality)
- **Source file**: `real_tests/tZrSe2/output/valley_analysis/symmetry_adapted_valley_analysis.json`
- **Current values**:

  | Valley | Kpoint | seed_overlap | ebr_seed_overlap_min | Pass? |
  |--------|--------|-------------|---------------------|-------|
  | M1 | GammaM | 0.739 | 0.8 | No |
  | M2 | GammaM | 0.659 | 0.8 | No |
  | M3 | GammaM | 0.711 | 0.8 | No |
  | M1 | MM | 0.917 | 0.8 | Yes |
  | M2 | MM | 0.917 | 0.8 | Yes |
  | M3 | MM | 0.957 | 0.8 | Yes |

- **Effect**: GammaM q-cut seed projectors do not overlap well with the
  symmetry-adapted projectors. This blocks the `symmetry_adapted` path from
  reaching `trusted` readiness at GammaM. MM has much better seed overlap
  (0.92-0.96).
- **Physics**: The q-cut at GammaM (fraction=0.15 of min valley distance)
  produces projector windows that do not isolate M valleys as cleanly as at
  MM. This is a geometric effect: at Gamma, all three M-valley q-vectors are
  close together in the shell.
- **Checklist to resolve**:
  1. Scan q-cut fraction at GammaM to find optimal value maximizing seed
     overlap while maintaining valley separability.
  2. If no single q-cut works for all kpoints, consider per-kpoint q-cut
     configuration (requires config schema change).
  3. Do NOT lower `ebr_seed_overlap_min` (currently 0.8) to make tZrSe2 pass.

### B4: M3 source trust cascade

- **Classification**: code-design blocker (derived from B2+B3)
- **Effect**: MM/M3_valley is the only explicit HSP with a non-identity
  valley-preserving C2 operation (op=5). The HSP-star conjugation graph
  correctly identifies that M1/M2 C2 characters can be derived from M3 source
  via space-group conjugation. However, M3's character is `diagnostic_only`
  due to B2: the M3 valley-preserving representation unitarity remains above
  `representation_unitarity_fail_tol=1e-2`. Therefore:
  - `valley_irrep_matching` status = `diagnostic_only` for M3 C2
  - HSP-star derived characters for M1/M2 C2 are `diagnostic_only`
  - `valley_ebr_input_candidates` = `no_candidates`
  - `valley_ebr_problem_instances` = `no_instances`
- **This is not a bug**: the code correctly propagates source trust through
  the derivation chain. Once B2 (D_raw closure) is resolved, the cascade
  automatically clears.
- **Checklist to resolve**: Resolve B1 and B2. No separate action needed for
  B4 — it is a consequence, not an independent cause.

### B5: KM and MM M1/M2 have no valley-preserving ops beyond identity

- **Classification**: physics/numerical blocker (not a code issue)
- **Source file**: `real_tests/tZrSe2/output/valley_analysis/irrep_workflow_decisions.json`
- **Current values**:

  | Kpoint | Valley | HSP ops | Workflow |
  |--------|--------|---------|----------|
  | KM | M1 | [0] | blocked |
  | KM | M2 | [0] | blocked |
  | KM | M3 | [0] | blocked |
  | MM | M1 | [0] | blocked |
  | MM | M2 | [0] | blocked |
  | MM | M3 | [0,5] | symmetry_adapted/usable_with_caution |

- **Effect**: These valleys have no non-identity valley-preserving HSP-little-group
  operations at their respective kpoints. The only path to obtain characters for
  them is HSP-star derivation from the MM/M3 source (B4). KM has no D_raw at all
  for C2 ops (they are not in the KM little group), so no direct computation is
  possible.
- **This is physically correct**: at the KM HSP, the C2 operations map KM to
  another member of the HSP star. They should NOT be forced into the KM local
  representation.
- **Checklist**: Resolve B4. These valleys will be unblocked when HSP-star
  derived characters become trusted.

## Summary

| # | Blocker | Type | Resolved By |
|---|---------|------|-------------|
| B1 | spinor_convention_unverified | Physics/convention | External benchmark comparison |
| B2 | D_raw target-subspace closure | Numerical/provenance | Expanded-band HDF5 or reunitarization evidence |
| B3 | Low seed overlap at GammaM | Numerical/provenance | q-cut optimization or per-kpoint config |
| B4 | M3 source trust cascade | Code-design (consequential) | Resolving B1+B2 automatically clears this |
| B5 | KM/MM M1/M2 no VP ops beyond identity | Physics (not a code issue) | Resolving B4 via HSP-star derivation |

## Explicit Non-Actions

The following must NOT be done to make tZrSe2 appear "ready":
- Loosen `representation_unitarity_fail_tol` beyond current 1e-2
- Loosen `ebr_unitarity_max` beyond default 1e-3
- Lower `ebr_seed_overlap_min` below current 0.8
- Mark spinor convention verified without an external benchmark
- Implement polar reunitarization for the official character computation path
- Use `allow_caution=True` to promote `usable_with_caution` rows to `trusted`
- Add tZrSe2-specific if/else logic to any readiness gate
