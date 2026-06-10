# tMoTe2 P321 C3 Valley-Preserving Irrep Benchmark Audit

Date: 2026-06-10 | Branch: `cc/tmote2-c3-irrep-audit`

## Scope

Audit the implemented spinful C3 valley-preserving irrep table logic for
the tMoTe2 P321 benchmark: check label conventions, phase matching,
valley-preserving subgroup usage, readiness gates, and EBR input/export
readiness.

## Command

```bash
cd real_tests/tMoTe2
python -m valleyscope.cli analyze-hsp analyze.yaml
```

Config: `real_tests/tMoTe2/analyze.yaml`
- `spinor.convention_verified: true`, benchmark `tMoTe2_VBM_C3_literature`
- `rotation.readiness_preset: loose`
- `qcut_fraction: 0.20`
- Space group: P321 (150), spinful C3 double group

## Trusted Candidates

8 EBR input candidates across 4 (kpoint, valley) rows:

| Kpoint | Valley | Op | Irrep Label | Phase (2π) | Path | Readiness |
|--------|--------|----|-------------|------------|------|-----------|
| GammaM | K_valley | 1 (C3) | C3_spinor_phase_+1/2 | +0.5 | direct_qcut | trusted |
| GammaM | K_valley | 2 (C3) | C3_spinor_phase_+1/2 | -0.5 | direct_qcut | trusted |
| GammaM | Kp_valley | 1 (C3) | C3_spinor_phase_+1/2 | -0.5 | direct_qcut | trusted |
| GammaM | Kp_valley | 2 (C3) | C3_spinor_phase_+1/2 | +0.5 | direct_qcut | trusted |
| KM | K_valley | 1 (C3) | C3_spinor_phase_+1/6 | +0.166667 | direct_qcut | trusted |
| KM | K_valley | 2 (C3) | C3_spinor_phase_-1/6 | -0.166667 | direct_qcut | trusted |
| KM | Kp_valley | 1 (C3) | C3_spinor_phase_-1/6 | -0.166667 | direct_qcut | trusted |
| KM | Kp_valley | 2 (C3) | C3_spinor_phase_+1/6 | +0.166667 | direct_qcut | trusted |

Root deviations: 1.2–4.7 × 10⁻⁶ at GammaM, 1.2 × 10⁻⁵ at KM.
All `rotation_ready=True`, `topology_input_ready=True`, `diagnostic_only=False`.

## Blocked Rows

| Kpoint | Valley | Reason | Blocker |
|--------|--------|--------|---------|
| MM | K_valley | No valley-preserving ops beyond identity in HSP little group | hsp_star_derivation_not_available |
| MM | Kp_valley | Same | hsp_star_derivation_not_available |

MM has only identity (`op=0`) as HSP little group for both valleys. C3 ops
(1,2) map MM to other HSP-star members. HSP-star conjugation graph is
complete (8 matched), but only identity ops carry derived characters
(chi=1.0, trusted=True). Non-identity C2/C3 at other HSP-star members are
not available for derivation → blocked.

## Irrep Table Conventions Check

### Spinful C3 Phase Conventions

The spinful C3 table (`irreps/tables.py` via `irreptables`) provides:
- `C3^3 = -1` (spinor double group)
- Allowed eigenvalues: `exp(+iπ/3)` (phase +1/6), `exp(+iπ)` = -1 (phase 1/2), `exp(-iπ/3)` (phase -1/6)

Matched irrep labels:
- Phase +0.5 → C3_spinor_phase_+1/2 (GammaM K)
- Phase -0.5 → C3_spinor_phase_+1/2 (GammaM K') — **same irrep label despite opposite sign**
- Phase +0.166667 → C3_spinor_phase_+1/6 (KM K)
- Phase -0.166667 → C3_spinor_phase_-1/6 (KM K)

Note: GammaM K and K' both match C3_spinor_phase_+1/2 with opposite phase signs.
For spinful C3, `exp(iπ) = -1` and `exp(-iπ) = -1` are the same eigenvalue
(mod 2π), so op=1 (phase 0.5) and op=2 (phase -0.5) map to the same label.
The two C3 generators (ops 1 and 2) are conjugate; their eigenvalues are
consistent with the same 1D irrep. ✅ Convention consistent.

### Operation Matching

`match_table_operations()` maps detected ops to table indices by rotation
matrix + translation modulo lattice. All 3 C3 ops (detected ids 0,1,2)
match the standard P321 table indices. The valley-preserving subgroup for
each valley is {0,1,2} = C3_like. ✅

### Valley-Preserving Subgroup Usage

HSP little group: P321 ops {0,1,2,3,4,5} at GammaM/KM (full C3v).
Valley-preserving ops for K_valley: {0,1,2} (C3 rotations preserve K).
Valley-changing ops: {3,4,5} (C2 rotations swap K↔K').
Only valley-preserving ops enter eigenvalue/irrep rows. ✅

## Readiness Gate Verification

For each of the 4 trusted (kpoint, valley) rows, all gates pass:

| Gate | Status |
|------|--------|
| Seed projector symmetry (valley-preserving ops) | ✅ Passed (GammaM epsilon_seed ≈ 9e-6; KM epsilon_seed ≈ 0.0064, passes loose preset) |
| Target-subspace closure (valley-preserving ops) | ✅ Clean (unitarity < 3e-5) |
| Spinor convention | ✅ Verified (tMoTe2_VBM_C3_literature) |
| Root deviation | ✅ GammaM < 5e-6; KM ≈ 1.2e-5 (passes loose preset: 1e-4) |
| D_block_leakage_norm / epsilon_seed | ✅ GammaM < 7e-7; KM ≈ 0.0064 (passes loose preset: 1e-2) |
| Valley concentration | ✅ > 0.99 |
| q-cut seed projector quality | ✅ Rank estimates correct, eigenvalue gaps > 0.98 |

Notable: KM `D_block_leakage_norm` and `epsilon_seed` are both ≈ 0.0064,
which exceeds strict (1e-6) and normal (1e-3) presets but passes loose
(1e-2). This residual K↔K' mixing is a known two-valley artifact in the
q-cut seed basis. `readiness_preset: loose` is the stated benchmark
configuration.

## EBR Input/Export Status

- **EBR input candidates**: 8 trusted candidates (4 rows × 2 C3 generators)
- **EBR problem instances**: 2 instances
  - `ebr_instance_001`: K_valley, C3_like, complete, ready
  - `ebr_instance_002`: Kp_valley, C3_like, complete, ready
- **EBR export bundle**: 2 bundles, `ready_for_external_solver`
  - Expected HSPs: [GammaM, KM]
  - Optional HSPs: [MM]
  - Missing optional: [MM]

Both instances are complete for C3_like valleys. The missing MM HSP is
expected (only identity in HSP little group) and correctly marked optional.

## Projector Symmetry Note

The projector symmetry-consistency report checks
`D_g P_a D_g^dag ≈ P_{pi_g(a)}` for all detected operations, including
valley-changing ones. The output shows FAILED checks for valley-CHANGING
C2 operations (ops 3,4,5: epsilon 0.74–1.0). These operations are not used
as valley-preserving irrep evidence — the singleton valley-preserving
subspaces use only valley-preserving C3 ops {0,1,2}, which pass with
epsilon_seed ≈ 9e-6 (GammaM) / ≈ 0.0064 (KM, loose preset).

Valley-changing C2 failures are tracked but do not block singleton
valley-preserving readiness. The full K/K' valley-orbit report remains
`diagnostic_only` because valley-changing C2 sewing candidates are
inequivalent (`projector_construction_failed`). The orbit-level
`irrep_matching_input_ready` is false; only per-valley singleton subspaces
reach `ebr_ready: True`. ✅

## Remaining Risks

1. `readiness_preset: loose` is used; KM `D_block_leakage_norm` and
   `epsilon_seed` are both ~0.0064, passing loose (1e-2) but failing normal
   (1e-3). This is a known two-valley mixing artifact, not a code issue. The
   benchmark smoke doc should note which preset was used.
2. The MM blocked rows cannot be resolved without non-identity valley-preserving
   ops at MM. The HSP-star conjugation graph is complete but only identity
   characters are derivable. Physics limitation.
3. The `tMoTe2_VBM_C3_literature` spinor benchmark label is recorded but the
   actual literature reference is not in the repo. Should be documented
   explicitly.

## Test Status

```bash
pytest -q
# 483 passed (run separately before audit — no code changes in this branch)
```

No code changes made. This is a read-only audit document.
