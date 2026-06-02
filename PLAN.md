# PLAN.md

This roadmap reflects the current ValleyScope methodology.

## Current Code Baseline (updated 2026-06)

The repo already has:

**Layer 1 — valley projection + diagnostics:**
* q-cut valley projection and seed projector construction;
* multi-valley basis diagnostics using projected seed matrices;
* valley orbit and valley mapping logic;
* moire HSP symmetry representation matrices D_raw;
* seed projector symmetry-consistency diagnostic (`projector_symmetry_report.json`);
* target-subspace closure provenance diagnostics (`target_subspace_closure.json`);
* subspace representation quality diagnostics (embedded in formal reports;
  standalone `subspace_representation_quality.json` is optional/default-off).

**Layer 2 — gated irrep workflow:**
* irrep workflow decision layer (`direct_qcut` / `symmetry_adapted` / `blocked`);
* formal symmetry-adapted valley analysis (P_a^sym, integrated);
* valley-preserving character/eigenphase diagnostics;
* minimal valley-preserving irrep matching (spinful C3, C2 with internal tables);
* HSP-star conjugation and derived character layer;
* EBR input candidates, problem instances, and export bundle.

**Layer 3 — external solver interface:**
* default-off reduced EBR mapping interface (exact integer, external table only);
* `valley_reduced_ebr_mapping.json` (only when `analysis.reduced_ebr.enabled`).

**Output and summary:**
* `valley_summary.txt` / `valley_summary.json` (main user entry);
* `valley_ebr_export_bundle.json` (downstream EBR entry);
* debug/detail outputs: diagnostics.h5, optional/default-off
  subspace_representation_quality.json, HSP-star conjugation/derived character
  JSONs, raw matrices, qcut scans.

## Methodology

For each moire HSP k, define the target DFT band subspace

```math
H_k = \mathrm{span}\{|\psi_{n,k}\rangle\}.
```

Inside this subspace:

1. `P_a^0` is the q-cut valley seed projector.
2. `D_g` is the moire HSP symmetry representation matrix.
3. `pi_g(a)` is the operation-induced valley mapping.
4. The seed diagnostic checks

   ```math
   D_g P_a^0 D_g^\dagger \approx P_{\pi_g(a)}^0 .
   ```

5. `P_a^sym` is the symmetry-adapted valley projector, built from seed
   averaging and spectral purification. It is used when the seed basis fails
   the projector symmetry check but P_a^sym construction is feasible (the
   `symmetry_adapted` workflow path).

6. The `direct_qcut` path bypasses P_a^sym when the q-cut seed basis already
   passes all readiness gates (tMoTe2-like clean systems).

Do not treat high q-cut valley purity as irrep readiness. Do not test
valley-changing operations as invariance of a single label operator.

## Phase Completion Status

| Phase | Description | Status |
|-------|------------|--------|
| Phase 0 | Freeze old interpretation (q-cut seed = diagnostic) | Done |
| Phase 1 | Seed projector symmetry-consistency diagnostics | Done |
| Phase 2 | Symmetry-adapted valley projector prototype | Done — integrated into production workflow |
| Phase 3 | Symmetry-adapted valley-irrep workflow | Done — with direct_qcut gated path |
| Phase 4 | Real benchmarks (tMoTe2, tZrSe2) | Smoke tests pass; blockers documented |
| Phase 5 | EBR input/candidate/problem/export pipeline | Done |
| Phase 6 | Reduced EBR mapping interface | Default-off external-table solver exists; no built-in tables |

## Real Benchmarks

**tMoTe2 K/Kp** (P321):
- Clean C3 valley-preserving subspaces with `direct_qcut` / `trusted` readiness.
- GammaM and KM produce trusted C3-like irrep labels.
- MM has only identity in the HSP little group → blocked.

**tZrSe2 M-star** (P312):
- P2-like single-valley subspaces identified at space-group level.
- Current blockers: `spinor_convention_unverified`, low seed overlap at GammaM,
  D_raw target-subspace closure at ~1.9e-2 (usable_with_caution).
- M1/M2 non-identity C2 characters require HSP-star derivation from MM M3
  source. Source is diagnostic_only until closure/seed quality improves.
- No trusted C2 EBR instances exist yet.

## Open Work

* Spinor convention benchmark verification for tZrSe2.
* Expanded-band HDF5 to test D_raw closure sensitivity.
* Design reviewed EBR table ingestion for trusted C3/C2 instances.
* High-throughput database pipeline.
