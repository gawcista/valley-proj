# Symmetry-Adapted Valley Pipeline - Real-Data Smoke Benchmark

Date: 2026-05-26

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
| MM | K | P3 | [0,1,2] | [0] | C1 | false | hsp_local_preserving_character_missing |
| MM | K' | P3 | [0,1,2] | [0] | C1 | false | hsp_local_preserving_character_missing |

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

The GammaM q-cut seed is not ideal for ZrSe2: seed overlaps are only
approximately 0.66-0.74, and the seed projector symmetry-consistency errors
are O(1). The symmetry-adapted singleton reports are therefore useful
diagnostics but are not yet EBR-ready.

### KM And MM

At the single configured MM HSP, the current HSP little group contains only the
M3-preserving C2. The M1 and M2 C2 operations preserve their monolayer valley
labels under the full space group but map this HSP to another member of the HSP
star. Therefore the correct output is `subspace_space_group=P2` for all three
M valleys, while `hsp_ops` is nontrivial only for the valley whose C2 belongs
to the current HSP little group.

| Kpoint | Valley | Subspace SG | SG ops | HSP ops | HSP-local group | EBR blockers |
|--------|--------|-------------|--------|---------|-----------------|--------------|
| KM | M1 | P2 | [0,4] | [0] | C1 | spinor_unverified, hsp_local_preserving_character_missing |
| KM | M2 | P2 | [0,3] | [0] | C1 | spinor_unverified, hsp_local_preserving_character_missing |
| KM | M3 | P2 | [0,5] | [0] | C1 | spinor_unverified, hsp_local_preserving_character_missing |
| MM | M1 | P2 | [0,4] | [0] | C1 | spinor_unverified, hsp_local_preserving_character_missing |
| MM | M2 | P2 | [0,3] | [0] | C1 | spinor_unverified, hsp_local_preserving_character_missing |
| MM | M3 | P2 | [0,5] | [0,5] | C2_like | spinor_unverified, representation_unitarity |

This resolves the previous apparent M1/M2 failure: the code must not force a
non-HSP-little-group C2 into the current HSP-local representation, but it should
still report the full-space-group subspace candidate `P2`. The current output
does exactly that.

## 3. EBR Readiness Gate

`ebr_mapping_input.ready=true` requires all of:

1. `local_irrep_ready=true`
2. `diagnostic_only=false`
3. verified spinor convention for spinor wavefunctions
4. seed overlap above `ebr_seed_overlap_min`
5. representation unitarity below `ebr_unitarity_max`
6. a current-HSP valley-preserving character when table matching needs one

When `subspace_space_group=P2/P3` exists but the current HSP only has identity
for that valley, the blocker is
`hsp_local_preserving_character_missing`. This means the subspace space group
candidate exists, but the local character must be taken from the appropriate
HSP-star member before subspace EBR matching.

## 4. pytest Result

```bash
pytest -q
# 312 passed
```

## 5. Next Steps

1. Construct symmetry-derived HSP-star members so each ZrSe2 M-valley `P2`
   subspace has its C2 character at the correct HSP without requiring
   additional DFT.
2. Verify the ZrSe2 spinor convention against a known benchmark.
3. Start character-table matching for the ready tMoTe2 `P3` singleton
   subspaces, then extend to ZrSe2 `P2` after spinor and HSP-star issues are
   resolved.
