# Symmetry-Adapted Valley Pipeline — Real-Data Smoke Benchmark

Date: 2026-05-26
Branch: `cc/subspace-group-ebr-readiness`
Base: `fix/zrse2-local-p2-subspaces` at `74cc238`

## Convention Guard

`_projector_matrix()` stores P[i,j] = <psi_i|P|psi_j> (standard convention).
The correct symmetry action is `D @ P @ D.conj().T` = `D P D^dag`.
The formula `D.T @ P @ D.conj()` is **incorrect** (gives O(1) error for C3).
Tests added to lock this convention.

## Config

```yaml
analysis:
  symmetry_adapted_valley:
    enabled: false  # default off
    seed_overlap_warn_tol: 0.8
    seed_overlap_fail_tol: 0.5
    projector_symmetry_warn_tol: 1e-2
    projector_symmetry_fail_tol: 1e-1
    representation_unitarity_warn_tol: 1e-3
    representation_unitarity_fail_tol: 1e-2
    ebr_seed_overlap_min: 0.8
    ebr_unitarity_max: 1e-3
```

## 1. tMoTe2 — K/K' Valley Benchmark (P321)

### Subspace Group / EBR Readiness

| Kpoint | Scope | Candidate | EBR Ready | Blockers |
|--------|-------|-----------|-----------|----------|
| GammaM | full K/K' orbit | — | false | projector failed (inequivalent candidates) |
| KM | full K/K' orbit | — | false | projector failed (inequivalent candidates) |
| MM | full K/K' orbit | — | false | completeness_error=0.5 |
| GammaM | local K / K' | C3_like | true | none |
| KM | local K / K' | C3_like | true | none |
| MM | local K / K' | C1 | false | subspace_group_candidate_missing |

Representative ambiguity NOT resolved: C2 ops 3,4,5 all map K→Kp with O(1) projector differences.
The local valley-preserving K/K' subspaces at GammaM and KM are now valid
inputs for the next character-table matching step; the full K/K' orbit remains
diagnostic-only because valley-changing C2 representatives are inequivalent.

## 2. tZrSe2 — M-Star Valley Benchmark (P312)

### GammaM — Local Singleton Valley-Preserving Subspaces

Three singleton reports, each for one valley's preserving subgroup {E, C2_Mi}:

| Valley | VP Ops | Rank | Seed Overlap | C2 Eigenphases | Subspace Group | EBR Ready | Blockers |
|--------|--------|------|-------------|----------------|----------------|-----------|----------|
| M1 | [0, 4] | 2 | 0.739 | -0.25, +0.25 | C2_like (blocked) | false | spinor_unverified, low_seed_overlap=0.739 |
| M2 | [0, 3] | 2 | 0.659 | -0.25, +0.25 | C2_like (blocked) | false | spinor_unverified, low_seed_overlap=0.659 |
| M3 | [0, 5] | 2 | 0.711 | +0.25, +0.25 | C2_like (blocked) | false | spinor_unverified, low_seed_overlap=0.711, representation_unitarity=2.2e-2 |

All three valleys correctly identify C2-preserving singleton subspaces with rank 2.
Spinor C2 eigenphases (±0.25) are physically correct for double-group C2.
Seed overlap 0.66-0.74 indicates the q-cut seed basis is not well-aligned with
the symmetry-adapted basis at GammaM.

### MM — Partial HSP-Star

The current MM kpoint at frac=[0.5, 0, 0] only covers ONE of the three moire M points.
The three monolayer M valleys fold to different moire M points:
- M1 → (0.5, 0.5, 0) [NOT in current config]
- M2 → (0.5, 0.0, 0) [= current MM]
- M3 → (0.0, 0.5, 0) [NOT in current config]

At the current MM, only ops 4 and 5 are in the little group. Op 3 is NOT in the
MM little group (it belongs to the M point at (0, 0.5, 0)).

| Valley | VP Ops | Rank | Seed Overlap | Subspace Group | EBR Ready | Blockers |
|--------|--------|------|-------------|----------------|-----------|----------|
| M1 | [0] | 2 | 0.917 | C1 (blocked) | false | spinor_unverified, subspace_group_candidate_missing |
| M2 | [0] | 2 | 0.917 | C1 (blocked) | false | spinor_unverified, subspace_group_candidate_missing |
| M3 | [0, 5] | 2 | 0.957 | C2_like (blocked) | false | spinor_unverified, representation_unitarity=1.8e-2 |

Note: M1's C2 (op 4) is NOT in the current MM little group — it belongs to
(0.5, 0.5, 0). To analyze all three M valleys with their valley-preserving C2
operations, three separate moire M points are needed.

### Subspace Group Summary

All three GammaM valley-preserving subspaces identify as **C2_like** candidates
(P2-like local symmetry). At the single current MM kpoint, only M3 has a
nontrivial C2-like preserving subgroup; M1 and M2 are C1 at this HSP.
The ZrSe2 subspaces are **blocked** from EBR mapping:
- `spinor_convention_unverified` (primary blocker for all)
- `low_seed_overlap_min` (GammaM: 0.659-0.739)
- `representation_unitarity` (M3 at GammaM: 2.2e-2; M3 at MM: 1.8e-2)

## 3. EBR Readiness Gate

The `ebr_mapping_input` schema requires ALL of:
1. local_irrep_ready=true
2. diagnostic_only=false
3. spinor_convention_verified=true
4. seed_overlap >= ebr_seed_overlap_min (0.8)
5. representation unitarity <= ebr_unitarity_max (1e-3)

tMoTe2 local K/K' C3-like subspaces at GammaM and KM pass these gates and are
ready as inputs to the next character-table matching step. ZrSe2 remains
blocked by spinor convention, low GammaM seed overlap, and M3 unitarity issues.

## 4. pytest Result

```
$ pytest -q
301 passed
```

## 5. Recommendation

1. Verify spinor convention for tZrSe2 to clear the primary blocker.
2. Add MM2 and MM3 kpoints to the config to cover the full M-star.
3. Character table matching for C2_like subspaces (P2/C2 irrep tables).
