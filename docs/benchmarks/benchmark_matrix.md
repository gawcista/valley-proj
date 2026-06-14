# ValleyScope Benchmark Matrix

Date: 2026-06-10

This document records real-material validation fixtures and their current
status. Real materials are regression evidence only — they must not appear
in `valleyscope/` production logic, output strings, config semantics, or
schema names. See individual benchmark docs under `docs/benchmarks/` for
detailed audit and blocker evidence.

## Fixture Status

| Fixture | Valley Family | HSP Context | Validation Role | Output Profile | Irrep/EBR Readiness |
|---------|--------------|-------------|-----------------|----------------|---------------------|
| tMoTe2 P321 | K/K' (C3-preserving) | GammaM, KM, MM | Clean `direct_qcut` trusted path; spinful C3 irrep benchmark | Standard (4 files) | 2 EBR bundles `ready_for_external_solver` |
| tZrSe2 P312 | M-star (C2-preserving) | GammaM, KM, MM | `symmetry_adapted` path; blocker evidence protocol | Standard (4 files) | diagnostic-only; export bundle `no_bundles` because problem instances are `no_instances` |
| tPdSe2 / PdSe2 | — | — | Deferred | — | No fixtures or evidence yet |

## tMoTe2 P321 — K/K' Valley

- **Valley family**: Two-valley K/K' (C3-preserving, `C3_like` EBR table label)
- **HSP context**: GammaM, KM with full C3 little group; MM identity-only
- **Validation role**: Clean-system benchmark for the `direct_qcut` trusted path
  and spinful C3 irrep table matching
- **Irrep readiness**: 8 trusted EBR input candidates at GammaM and KM;
  MM blocked (no non-identity valley-preserving ops at HSP)
- **EBR readiness**: 2 export bundles, `ready_for_external_solver`
  (K_valley and Kp_valley, `C3_like` EBR table label)
- **Config notes**: `spinor.convention_verified: true`,
  `readiness_preset: loose`; KM D_block_leakage ~0.0064 passes loose (1e-2)
  but not normal (1e-3)
- **Expected standard-profile output files**:
  `valley_summary.txt`, `valley_summary.json`, `valley_weights.csv`,
  `valley_ebr_export_bundle.json`
- **Detailed audit**: `docs/benchmarks/tmote2_c3_irrep_audit.md`
- **Smoke walkthrough**: `docs/benchmarks/symmetry_adapted_valley_smoke.md` §1
- **Reviewed package-table validation**: `docs/benchmarks/tmote2_reduced_ebr_mapping_validation.md`
  — Local fixture run with `P321_C3_like_GammaM_KM_spinful_v1`; both valleys
  `solved_exact`, `atomic-compatible-candidate`.

## tZrSe2 P312 — M-Star Valley

- **Valley family**: Three-valley M-star (C2-preserving, `C2_like` EBR table label)
- **HSP context**: GammaM M1/M2/M3 with full C2 HSP little group;
  KM and MM M1/M2 identity-only (C2 maps to other HSP-star members);
  MM M3 has C2 HSP-little-group op
- **Validation role**: Real-system benchmark for `symmetry_adapted` path,
  HSP-star conjugation, derived characters, and blocker cascade logic
- **Irrep readiness**: `diagnostic_only` everywhere —
  `spinor_convention_unverified` downgrades all rows
- **EBR readiness**: public export bundle status is `no_bundles` because the
  problem-instance layer is `no_instances`; 5 blockers are documented in
  `docs/benchmarks/tzrse2_blocker_evidence.md`:
  B1 (spinor convention), B2 (D_raw closure at MM/GammaM op=5), B3 (low
  GammaM seed overlap), B4 (M3 source trust cascade, consequential), B5
  (KM/MM M1/M2 no VP ops beyond identity, physics limitation)
- **Expected standard-profile output files**:
  `valley_summary.txt`, `valley_summary.json`, `valley_weights.csv`,
  `valley_ebr_export_bundle.json`
  (export bundle has `status: no_bundles`, `bundle_count: 0`)
- **Detailed blocker evidence**: `docs/benchmarks/tzrse2_blocker_evidence.md`
- **Smoke walkthrough**: `docs/benchmarks/symmetry_adapted_valley_smoke.md` §2
- **C2-like reduced EBR unblock audit**: `docs/reduced_ebr_c2_authoring_audit.md`
  — SG149/P312 source data is available, but C2 source-op / spinor-lift
  convention signoff and tZrSe2 fixture-quality blockers remain before any
  C2 table can be reviewed.
- **C2 convention signoff packet**:
  `docs/reduced_ebr_c2_mapping_signoff_packet.md` — records the unresolved
  Bilbao source-HSP/op assignment, generator-orientation, and spinor-lift
  questions; no C2 table or tZrSe2 reduced EBR decomposition is claimed.

## tPdSe2 / PdSe2

- **Status**: Deferred. No fixtures, HDF5 data, or evidence documents exist.
  This row is a placeholder and must not be inferred.
- **When evidence becomes available**: add the fixture row above with the
  same format, link to detailed audit/blocker docs, and record the
  standard-profile output file list.

## Database Ingestion Record Anchors

`valleyscope collect-database-record` is an explicit offline collector
that builds a compact ingestion record from existing public outputs. It is
not a default `analyze-hsp` output.

Real-fixture smoke anchors (see `docs/benchmarks/database_ingestion_record_smoke.md`):

| Fixture | record_status | ready_bundles | trusted irrep records | space group |
|---------|--------------|---------------|----------------------|-------------|
| tMoTe2 P321 | `has_ready_ebr_bundles` | 2 | 8 | P321 (150) |
| tZrSe2 P312 | `no_ready_ebr_bundles` | 0 | 0 | P312 (149) |

tZrSe2 trusted irrep count is zero because all rows are blocked by
physical/readiness blockers (`spinor_convention_unverified`, D_raw closure,
low seed overlap, HSP-star derivation), not database-ingestion errors.

## Standard Output Contract

All real-material fixtures use `output.profile: standard` (or the default,
which is `standard`). The public output set for each fixture is:

1. `valley_summary.txt` — human-readable physics summary
2. `valley_summary.json` — machine-readable summary
3. `valley_weights.csv` — per-(kpoint, band) quick-scan weights
4. `valley_ebr_export_bundle.json` — downstream EBR entry (always written;
   carries `status: no_bundles` when no trusted EBR instances exist)
5. `valley_reduced_ebr_mapping.json` — only when
   `analysis.reduced_ebr.enabled: true` and a user-supplied validated table
   is provided

Debug/detail standalone files (diagnostics.h5, valley_subspace.json,
symmetry_report.json, hsp_star_conjugation.json, etc.) are **not** written
in the standard profile. Internal computations for readiness gates, blocker
explanations, and summary status may still run — the profile controls only
standalone file emission, not internal data flow.

Set `output.profile: debug` to restore the full diagnostic file set.
