# Symmetry-Adapted Valley Pipeline - Real-Data Smoke Benchmark

Date: 2026-05-27

This benchmark records the current real-data behavior of the formal
`symmetry_adapted_valley_analysis.json` output. Large HDF5/POSCAR files and
generated output directories stay local under `real_tests/` and are not tracked.

## Convention Guard

`_projector_matrix()` stores `P[i,j] = <psi_i|P|psi_j>` in the standard
convention. The correct symmetry action is `D @ P @ D.conj().T`, i.e.
`D P D^dag`. The transposed action `D.T @ P @ D.conj()` is incorrect and is
guarded by tests.

## Config

The symmetry-adapted valley analysis is now a formal default-on output. It can
be disabled only for legacy comparison runs:

```yaml
analysis:
  symmetry_adapted_valley:
    enabled: false
```

The report separates two layers:

- `subspace_space_group`: full-space-group valley-preserving operation content
  for the valley subspace, e.g. `P2` or `P3`.
- `subspace_group`: HSP-local preserving representation available at the
  current k point, e.g. `C2_like`, `C3_like`, or `C1`.

This distinction matters whenever a valley-preserving operation maps the
current HSP to another member of the HSP star.

## New diagnostics (2026-05-27)

### target_subspace_closure

Per-(kpoint, operation) check whether D_raw is unitary and closed under
group relations (e.g. C2 spinful: ||D^2 + I||_F).  Only valley-preserving
operation failures block EBR readiness; valley-changing operations
naturally have non-unitary D_raw in the target subspace.

Schema: `target_subspace_closure.json`

### hsp_star_conjugation

Maps valley-preserving operations across HSP-star members via space-group
conjugation h = r g r^{-1}.  Reports matched, missing_operation_product,
antiunitary_not_implemented, diagnostic_only, and valley_mapping_mismatch.
TRS derivations are schema-recognised but not implemented.

Schema: `hsp_star_conjugation.json`

### hsp_star_derived_characters

Derives characters chi_{k1,a1}(h) = chi_{k0,a0}(g) from unitary space-group
conjugation.  Source character must be internally consistent (not diagnostic_only).
Blocked by target_subspace_closure_failed at source.

Schema: `hsp_star_derived_characters.json`

### EBR readiness blockers (updated)

New blocker keys replace the old `hsp_local_preserving_character_missing`:

- `hsp_star_character_derived` — character available via symmetry derivation
- `hsp_star_derivation_not_available` — no derivation available for this valley
- `target_subspace_closure_failed` — D_raw non-unitary for valley-preserving op
- `antiunitary_derivation_not_implemented` — TRS conjugation needed but not ready

## 1. tMoTe2 - K/K' Valley Benchmark (P321)

Command:

```bash
python -m valleyscope.cli analyze-hsp real_tests/tMoTe2/analyze.yaml
```

### Valley-Preserving Subspaces

| Kpoint | Valley | Subspace SG | SG ops | HSP ops | HSP-local group | EBR input ready | Blockers |
|--------|--------|-------------|--------|---------|-----------------|-----------------|----------|
| GammaM | K | P3 | [0,1,2] | [0,1,2] | C3_like | true | none |
| GammaM | K' | P3 | [0,1,2] | [0,1,2] | C3_like | true | none |
| KM | K | P3 | [0,1,2] | [0,1,2] | C3_like | true | none |
| KM | K' | P3 | [0,1,2] | [0,1,2] | C3_like | true | none |
| MM | K | P3 | [0,1,2] | [0] | C1 | false | hsp_star_derivation_not_available |
| MM | K' | P3 | [0,1,2] | [0] | C1 | false | hsp_star_derivation_not_available |

The full K/K' orbit reports remain diagnostic-only because the K->K' C2
representatives are inequivalent. The singleton valley-preserving subspaces at
GammaM and KM are ready inputs for the next character-table matching step.

## 2. tZrSe2 - M-Star Valley Benchmark (P312)

Command:

```bash
python -m valleyscope.cli analyze-hsp real_tests/tZrSe2/analyze.yaml
```

### GammaM

At GammaM, all three M-valley C2 operations belong to the HSP little group, so
the full-space-group subspace candidate and the HSP-local preserving
representation both show the same C2 content.

| Valley | Subspace SG | SG ops | HSP ops | HSP-local group | Rank | Seed overlap | C2 eigenphases | EBR blockers |
|--------|-------------|--------|---------|-----------------|------|--------------|----------------|--------------|
| M1 | P2 | [0,4] | [0,4] | C2_like | 2 | 0.739 | -0.25, +0.25 | spinor_unverified, low_seed_overlap |
| M2 | P2 | [0,3] | [0,3] | C2_like | 2 | 0.659 | -0.25, +0.25 | spinor_unverified, low_seed_overlap |
| M3 | P2 | [0,5] | [0,5] | C2_like | 2 | 0.711 | +0.25, +0.25 | spinor_unverified, low_seed_overlap, representation_unitarity |

### KM And MM

At KM and the single configured MM HSP, the M1/M2 C2 operations map the HSP to
another star member. `hsp_ops=[0]` for M1/M2 is correct. The EBR blocker is
`hsp_star_derivation_not_available` (not `hsp_local_preserving_character_missing`).
Derived characters from the conjugation graph can provide the M1/M2 C2 character
at the correct HSP star member without additional DFT.

| Kpoint | Valley | Subspace SG | SG ops | HSP ops | HSP-local group | EBR blockers |
|--------|--------|-------------|--------|---------|-----------------|--------------|
| KM | M1 | P2 | [0,4] | [0] | C1 | spinor_unverified, hsp_star_derivation_not_available |
| KM | M2 | P2 | [0,3] | [0] | C1 | spinor_unverified, hsp_star_derivation_not_available |
| KM | M3 | P2 | [0,5] | [0] | C1 | spinor_unverified, hsp_star_derivation_not_available |
| MM | M1 | P2 | [0,4] | [0] | C1 | spinor_unverified, hsp_star_derivation_not_available |
| MM | M2 | P2 | [0,3] | [0] | C1 | spinor_unverified, hsp_star_derivation_not_available |
| MM | M3 | P2 | [0,5] | [0,5] | C2_like | spinor_unverified, representation_unitarity |

The `hsp_star_derivation_not_available` status is correct for the current state:
the conjugation graph successfully identifies the mapping from source (MM M3 C2)
to target H-h=HSP-star members, but the source character at MM M3 is itself
diagnostic_only (unitarity error 1.8e-2), so derived characters are also
diagnostic_only and not trusted for EBR input.

## 3. EBR Readiness Gate

`ebr_mapping_input.ready=true` requires all of:

1. `local_irrep_ready=true`
2. `diagnostic_only=false`
3. verified spinor convention for spinor wavefunctions
4. seed overlap above `ebr_seed_overlap_min`
5. representation unitarity below `ebr_unitarity_max`
6. target subspace closure passed for all valley-preserving operations
7. required character available either explicit or symmetry-derived
8. no diagnostic-only source for derived characters

## 4. pytest Result

```bash
pytest -q
# 338 passed
```

## 5. Next Steps

1. Verify the tZrSe2 spinor convention against a known benchmark so
   `spinor_convention_unverified` is cleared.
2. Improve the ZrSe2 q-cut seed projector quality (currently seed overlaps
   ~0.66-0.74 at GammaM) to get internally consistent source characters
   that can feed the derived character layer.
3. Start character-table matching for the ready tMoTe2 `P3` singleton
   subspaces.
4. Extend to tZrSe2 `P2` after spinor and projector quality issues are
   resolved.
