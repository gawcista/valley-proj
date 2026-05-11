# valley-proj

HSP-only valley projection and rotation-eigenvalue diagnostics for VASP-derived moire wavefunctions.

## Scope

V1 reads an HDF5 intermediate format for analysis. A limited `WAVECAR` extractor is provided to generate that HDF5 from selected k-points and bands. V1 does not compute Berry curvature or Wilson loops, and does not output a complete integer Chern number. It reports valley weights, valley-goodness diagnostics, degenerate-subspace basis transforms, symmetry checks, and rotation eigenvalue data needed for separate `C mod n` analysis.

## HDF5 Input

The analyzer expects one group per k point because `nG` may differ by k point:

```text
/metadata/lattice/direct_cart              [3,3]
/metadata/lattice/reciprocal_cart          [3,3]
/metadata/spinor                           bool
/metadata/source                           string
/metadata/vasp_band_index_base             int

/kpoints/0/name                            string
/kpoints/0/frac                            [3]
/kpoints/0/cart                            [3]
/kpoints/0/g_vectors_frac                  [nG,3]
/kpoints/0/g_vectors_cart                  [nG,3]
/kpoints/0/coefficients                    [nb,nspinor,nG]
/kpoints/0/energies_eV                     [nb]
/kpoints/0/band_indices_vasp               [nb]
```

Band indices in YAML use VASP 1-based indices. The code handles internal Python indexing.

## YAML Configuration

Use `examples/config_template.yaml` as the starting point. The monolayer reciprocal lattice must come from either:

- explicit `monolayer_lattices.*.reciprocal_cart`; or
- user-provided `monolayer_poscars` plus `layer_transforms`.

The moire POSCAR is used for symmetry diagnostics, not as the default source of monolayer reciprocal lattice.

## Running HSP Projection

If starting from a traditional VASP `WAVECAR`, first extract selected wavefunctions:

```bash
valley-proj extract-wavecar extract.yaml
```

Use `examples/extract_wavecar_template.yaml` as a template. The extractor writes `selected_wavefunctions.h5`, then the analyzer reads that file:

```bash
valley-proj analyze-hsp config.yaml
```

or from the source tree:

```bash
python -m valley_proj.cli analyze-hsp config.yaml
```

Outputs are written under `output.directory`:

```text
valley_weights.csv
valley_subspace.json
rotation_eigenvalues.csv
symmetry_report.json
valley_basis_transform.h5
diagnostics.h5
```

`valley_weights.csv` contains one row per target k point and band. `diagnostics.h5` stores sector masks, center masks, ambiguous masks, and q-cut metadata for later inspection.

## WAVECAR Extractor

The extractor is deliberately narrow and validation-heavy. It supports standard record-based `WAVECAR` files with RTAG values `45200`, `45210`, `53300`, or `53310`. It extracts selected VASP 1-based k-point and band indices into the V1 HDF5 schema.

For collinear `ISPIN=2`, set `extract.spin_index` and extract one spin channel at a time. For noncollinear/SOC records, the extractor detects `[2,nG]` spinor coefficients when the coefficient record length is `2*nplane`.

The extractor reconstructs the VASP G-list from the WAVECAR lattice, k-point, and `ENCUT`. If the generated G-list length does not match the `nplane` value stored in WAVECAR, it stops with an error instead of writing a potentially wrong HDF5 file.

After extraction, verify:

```text
/kpoints/N/coefficients shape is [nb,nspinor,nG]
/kpoints/N/g_vectors_cart length matches nG
/kpoints/N/norms are close to 1
/kpoints/N/energies_eV match the target VASP bands
```

## Diagnostics

`ambiguous_cross_sector: warn_exclude` is the default. If one plane-wave component falls into multiple valley sectors, it is excluded from all sector weights and counted as `ambiguous_weight`. This prevents double counting from creating artificial valley purity.

`qcut_scan` can be used to check whether `W_val`, `P_v`, and `ambiguous_weight` are stable against the projection radius.

## V1 Exclusions

V1 does not implement full mBZ mesh diagnostics, Berry curvature, Wilson loop, automatic full Chern inference, true layer-resolved projection, automatic monolayer valley search, or all possible `WAVECAR` variants such as gamma-only layouts.
