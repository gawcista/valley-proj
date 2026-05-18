# ValleyScope

[中文文档](README.zh.md)

ValleyScope is a VASP post-processing workflow for momentum-space valley projection, valley-adapted wavefunction construction, and valley-resolved symmetry representation analysis in moire supercells. The long-term goal is valley-resolved little-group irrep extraction and reduced EBR / topology diagnostics — not a black-box valley-Chern calculator.

V1 is an HSP-only diagnostic: it analyzes selected high-symmetry points, constructs monolayer-valley projectors in the plane-wave Hilbert space, fixes the valley gauge of near-degenerate target subspaces, and reports rotation eigenvalues as a first step toward full valley-resolved symmetry representations. A strict valley-resolved topology statement still requires later full-mBZ validation.

## Physical Motivation

In a moire calculation, the Bloch momentum belongs to the moire Brillouin zone. The valley label of interest, however, is usually inherited from the monolayer Brillouin zone. A state at a moire high-symmetry point is therefore not "a K-valley state" just because the moire momentum is at a moire $K_M$ point. ValleyScope treats valley as a monolayer-valley sector resolved through the plane-wave momenta contained in the moire wavefunction.

For twisted bilayers, top and bottom layer momenta associated with the same monolayer valley are allowed to hybridize. Such top/bottom hybridization does not by itself destroy the valley quantum number. The relevant question is whether the target state or target subspace remains confined to a chosen valley subspace, or whether there is appreciable intervalley mixing or weight outside the selected valley windows.

The V1 workflow answers that question at selected HSPs. It also checks whether a candidate symmetry operation belongs to the little group of the HSP and whether it preserves the target valley sector before reporting rotation eigenvalues. These eigenvalues are useful symmetry diagnostics, but by themselves they do not prove a complete integer Chern number.

## Physical Definitions

### Momentum-Space Valley Projection

For a VASP moire-supercell Bloch state at moire momentum $\mathbf k_M$, each plane-wave component carries the physical momentum

```math
\mathbf q=\mathbf k_M+\mathbf G_M ,
```
where $\mathbf G_M$ is a moire reciprocal lattice vector. ValleyScope uses only the in-plane component $\mathbf q_\parallel$ for valley assignment.

A monolayer valley center $a$ is specified in a monolayer reciprocal coordinate system and mapped into the moire frame through the layer transform. Its projector window is defined by the minimum distance to the monolayer reciprocal-lattice star:

```math
d_a(\mathbf q)=
\min_{\mathbf G_{\rm mono}}
\lvert
\mathbf q_\parallel-
(\mathbf Q_a+\mathbf G_{\rm mono})
\rvert ,
```
```math
\Omega_a(q_{\rm cut})=
\{\,\mathbf q\mid d_a(\mathbf q)<q_{\rm cut}\,\}.
```
A valley-sector projector is built from a union of such windows. For a two-valley twisted bilayer example,

```math
\Omega_K=
\Omega_{{\rm top},K}\cup\Omega_{{\rm bottom},K},
\qquad
\Omega_{K'}=
\Omega_{{\rm top},K'}\cup\Omega_{{\rm bottom},K'} .
```
The labels used in YAML, such as `K_sector`, are only user-chosen names for these physical sectors. They should be read as "K-valley sector" or "K'-valley sector," not as special code-defined physical objects.

### Weights and Intervalley-Mixing Diagnostics

Let $P_i$ be the valley-sector projector for sector $i$. With the default policy, plane-wave components that fall into projector windows belonging to more than one sector are removed from all sector projectors and collected in the overlap projector $P_\times$:

```math
\Omega_\times=
\{\,\mathbf q\mid
\mathbf q\in\Omega_i\cap\Omega_j
\ \text{for some}\ i\ne j\,\},
```
```math
P_i=
\sum_{\mathbf q\in\Omega_i\setminus\Omega_\times,\ s}
|\mathbf q,s\rangle\langle\mathbf q,s|,
\qquad
P_\times=
\sum_{\mathbf q\in\Omega_\times,\ s}
|\mathbf q,s\rangle\langle\mathbf q,s| .
```
For a normalized state $|\psi\rangle$, the monolayer-valley-resolved weight in sector $i$ is

```math
W_i=\langle\psi|P_i|\psi\rangle .
```
The target-valley-subspace weight is

```math
W_{\rm val}=\sum_i W_i .
```
It measures how much of the state lies in the user-defined target valley subspace. It depends on the valley centers and projection radius; it is not a topological invariant.

The valley purity is

```math
P_v=\frac{\max_i W_i}{W_{\rm val}},
\qquad W_{\rm val}>0 .
```
It measures whether the assigned valley weight is concentrated in one valley sector or distributed among several sectors. In the two-sector case, `P_v` and `|eta|` are redundant: `P_v = (1 + |eta|) / 2`. ValleyScope uses `|eta|` as the valley concentration score for two-sector diagnostics. For three or more sectors, `|eta|` is undefined and `P_v` is used directly as the concentration score.

The concentration score is classified into three tiers:
- **clean** (`raw_valley_clean` / `valley_separable_subspace`): concentration above the clean threshold (default `P_v=0.95`, equivalent to `|eta|=0.90`). Suitable as input for valley-resolved symmetry diagnostics.
- **approximate** (`raw_valley_approx` / `valley_approximately_separable_subspace`): concentration between the approx and clean thresholds (default `P_v=0.85~0.95`, `|eta|=0.70~0.90`). Usable with caution; rotation eigenvalues in this regime are diagnostic-only.
- **mixed** (`raw_valley_mixed` / `valley_mixed_subspace`): concentration below the approx threshold. Valley sectors are not well separated; single-valley diagnostics are not reliable.

For a two-valley subspace ordered as K-valley followed by K'-valley, the signed valley polarization is

```math
\eta=
\frac{W_K-W_{K'}}{W_K+W_{K'}}
=
\frac{W_K-W_{K'}}{W_{\rm val}} .
```
The projector-window overlap weight is

```math
W_{\rm overlap}=\langle\psi|P_\times|\psi\rangle .
```
It measures the weight in plane-wave components that fall into more than one valley-sector window. With the default exclusion policy, this weight is not assigned to any valley sector.

The residual weight outside both the target valley subspace and the overlap region is

```math
W_{\rm res}=1-W_{\rm val}-W_{\rm overlap}.
```
For a normalized state, V1 therefore reports the decomposition $W_{\rm val}+W_{\rm overlap}+W_{\rm res}=1$. In CSV and JSON output, `W_overlap` stores $W_{\rm overlap}$ and `W_res` stores the out-of-valley residual weight. A large overlap weight is a warning about the chosen windows or cutoff, not evidence for a new physical valley.

### Gauge Fixing in a Near-Degenerate Subspace

For an isolated nondegenerate band, the single-state quantities above can be read directly. For a near-degenerate target subspace, raw VASP eigenvectors are gauge-dependent: any unitary rotation inside the degenerate subspace gives an equally valid set of eigenvectors. Row-by-row projection of individual VASP bands is therefore not the main physical diagnostic.

V1 instead projects the target subspace and constructs the projected valley operator. For a two-valley subspace spanned by raw VASP states $\{|\psi_m\rangle\}$,

```math
S_{mn}=
\langle\psi_m|(P_K+P_{K'})|\psi_n\rangle,
\qquad
V_{mn}=
\langle\psi_m|(P_K-P_{K'})|\psi_n\rangle .
```
The matrix $S$ tests whether the target subspace lies in the chosen valley subspace. The matrix $V$ fixes a valley-adapted basis inside the target subspace. In practice, the `valley_subspace.json` and `valley_basis_transform.h5` files are the primary outputs for near-degenerate states.

### Symmetry Diagnostics

ValleyScope performs symmetry-operation detection from the moire or bilayer structure file. By default, candidate symmetry operations are detected from the moire structure file using spglib, with user-controlled symprec and optional symprec_scan. A rotation eigenvalue is reported only after two checks:

1. The operation belongs to the little group of the analyzed HSP.
2. The operation preserves the target valley sector.

The resulting matrix is a little-group representation in the valley-adapted subspace. Its eigenvalues are rotation eigenvalue diagnostics. They can constrain topology in symmetry-based formulas, but V1 does not infer a full integer Chern number.

## Workflow

### Install

```bash
git clone https://github.com/gawcista/valley-proj.git
cd valley-proj
pip install -e .
```

From a source checkout:

```bash
python -m valleyscope.cli --help
```

After installation:

```bash
valleyscope --help
```

### Start Here

The normal workflow has two steps.

First, extract a small HDF5 file from `WAVECAR`:

```bash
valleyscope extract-wavecar extract.yaml
```

Then analyze the selected HSP wavefunctions:

```bash
valleyscope analyze-hsp analyze.yaml
```

The analyzer reads the HDF5 file, not the full `WAVECAR`. This keeps the physics workflow independent from repeated large binary reads and makes it easier to debug the selected k points and bands.

### Minimal Example

A minimal `analyze.yaml` — only the required fields. All others use defaults (`qcut_mode: moire_shell`, symmetry detection skipped, etc.):

```yaml
input:
  wavefunction_h5: ./wave.h5

analysis:
  kpoints: [GammaM, KM, MM]
  iband: [2195, 2196]

valley_centers:
  coordinate_mode: cart
  centers:
    - name: K
      cart: [0.5, 0.0, 0.0]
    - name: Kp
      cart: [-0.5, 0.0, 0.0]

valley_manifolds:
  - name: K_valley
    centers: [K]
  - name: Kp_valley
    centers: [Kp]

output:
  directory: ./valley_analysis
```

Current analysis configs use `analysis.iband` for VASP band indices and
`valley_manifolds` for user-defined monolayer-valley subspaces. The older
input keys `analysis.target_bands_vasp` and `valley_sectors` have been removed
from the public schema.

### Extract From WAVECAR

Write an `extract.yaml` like this:

```yaml
input:
  wavecar: ./WAVECAR

extract:
  kpoints:
    - name: GammaM
      vasp_index: 1
    - name: KM
      vasp_index: 2
    - name: MM
      vasp_index: 3

  bands_vasp: [2195, 2196]

  # For collinear ISPIN=2, extract one spin channel at a time.
  # For SOC/noncollinear WAVECAR files, the extractor detects spinor records.
  spin_index: 1

output:
  wavefunction_h5: ./wave.h5
```

Run:

```bash
valleyscope extract-wavecar extract.yaml
```

After extraction, check that the HDF5 file contains the intended k points and bands. The most important datasets are:

```text
/metadata/g_vector_order
/kpoints/N/name
/kpoints/N/frac
/kpoints/N/g_vectors_frac
/kpoints/N/coefficients       [nb, nspinor, nG]
/kpoints/N/energies_eV
/kpoints/N/band_indices_vasp
```

If the extractor reports a G-vector count mismatch, stop there. That usually means the current WAVECAR variant or G-list convention is not supported yet.

### Analyze HSP Wavefunctions (Full Reference)

For twisted bilayers, the most important part of `analyze.yaml` is the layer transform. If the structure was generated from integer commensurate matrices, prefer `supercell_matrix` over a hand-written `rotation_deg`. The commensurate cell may include an overall basis rotation, so the twist angle alone is often not enough.

For a `generate_hexagonal_210(9, 5, ...)` style cell:

```yaml
input:
  wavefunction_h5: ./wave.h5

  # Monolayer POSCARs for layer reciprocal lattices and valley centers.
  # With these files set, reciprocal_cart usually does not need to be written by hand.
  monolayer_poscars:
    top: ./2dm-5370.vasp
    bottom: ./2dm-5370.vasp

layer_transforms:
  top:
    # Row-vector convention: M_moire = P^T A_layer.
    supercell_matrix:
      - [9, -5, 0]
      - [5, 4, 0]
      - [0, 0, 1]
  bottom:
    supercell_matrix:
      - [9, -4, 0]
      - [4, 5, 0]
      - [0, 0, 1]

analysis:
  kpoints: [GammaM, KM, MM]
  iband: [2195, 2196]
  subspace_energy_tol_meV: 1.0

valley_centers:
  coordinate_mode: layer_frac
  centers:
    - name: top_K
      layer: top
      frac: [0.333333333333, 0.333333333333, 0.0]
    - name: bottom_K
      layer: bottom
      frac: [0.333333333333, 0.333333333333, 0.0]
    - name: top_Kp
      layer: top
      frac: [-0.333333333333, -0.333333333333, 0.0]
    - name: bottom_Kp
      layer: bottom
      frac: [-0.333333333333, -0.333333333333, 0.0]

valley_manifolds:
  - name: K_valley
    centers: [top_K, bottom_K]
  - name: Kp_valley
    centers: [top_Kp, bottom_Kp]

projection:
  qcut_fraction: 0.20
  qcut_scan: [0.15, 0.20, 0.25, 0.30]
  thresholds:
    W_val_min: 0.8
    P_v_clean: 0.95
    P_v_approx: 0.85

symmetry:
  operations:
    structure_file: ./2dm-5370-7.34.vasp
  tolerance:
    symprec: 1.0e-3
    angle_tolerance: -1.0
    symprec_scan: [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3]
  filters:
    rotation_order: auto

spinor:
  convention: vasp_up_down_saxis_z
  convention_verified: true
  benchmark: tMoTe2_VBM_C3_literature

rotation:
  unitarity_tol: 1.0e-4
  root_deviation_tol: 1.0e-6
  D_valley_offdiag_tol: 1.0e-6

output:
  directory: ./valley_analysis
```

Run:

```bash
valleyscope analyze-hsp analyze.yaml
```

### Reading the Screen Output

`analyze-hsp` first prints a compact physics summary. This is the first layer of diagnosis; the CSV, JSON, and HDF5 files are for reproducibility and debugging.

The screen summary is organized as:

```text
Input
Valley manifolds
Valley projection summary
Valley-adapted subspace
Symmetry diagnostics
Allowed valley-preserving rotations
Rotation eigenvalues
Warnings
Output files
```

Use the `Valley projection summary` table to check `derived`, `pol`, `W_overlap`, `W_res`, and `valley_status` band by band. Use `Valley-adapted subspace` to decide whether the near-degenerate target subspace is a good two-valley subspace. In the rotation table, `topology_input_ready` only means that the HSP rotation eigenvalue is suitable as an input to a later symmetry-based topology analysis; it does not validate full-mBZ valley-resolved topology. The legacy `topology_ready` column is kept as a backward-compatible alias of `topology_input_ready`.

For SOC/spinor wavefunctions, ValleyScope applies the SU(2) spin rotation in the plane-wave representation. By default, VASP spinor phase conventions are treated as unverified, so spinful rows are reported with `spinor_rotation_applied=True`, `spinor_convention_verified=False`, and `diagnostic_only=True`. After an explicit benchmark, set `spinor.convention_verified: true` and record the benchmark name. For the tMoTe2 VBM C3 check used here, the benchmark is the literature pattern $\Gamma_M: -1$ and $K_M: e^{\pm i\pi/3}$ for the two spin-valley branches; this still does not validate full-mBZ valley-resolved topology.

Example screen summary:

```text
Input
-----
wavefunction_h5: ./wave.h5
operation structure: ./2dm-5370-7.34.vasp
operation-detection backend: spglib
spinor convention: vasp_up_down_saxis_z (verified=True, benchmark=tMoTe2_VBM_C3_literature)
target k-points: GammaM, KM, MM
target bands (VASP): 2195, 2196
qcut mode: relative_min_sector_distance
qcut value: 0.034 A^-1

Valley manifolds
----------------
label     centers
--------  ----------------
K_valley  top_K, bottom_K
Kp_valley top_Kp, bottom_Kp

Valley projection summary
-------------------------
kpoint  band  derived  pol   W_overlap  W_res  valley_status
------  ----  -------  ----  ---------  -----  --------------
GammaM  2195  0.98     0.98  0          0.02   raw_valley_clean
```

### Structure Files

These inputs serve different purposes:

```yaml
input:
  wavefunction_h5: ./wave.h5
  monolayer_poscars:
    top: ./2dm-5370.vasp
    bottom: ./2dm-5370.vasp

symmetry:
  operations:
    structure_file: ./2dm-5370-7.34.vasp
```

`input.wavefunction_h5` is the extracted moire wavefunction HDF5 file. It contains the selected HSP plane-wave coefficients and the moire reciprocal lattice used by the projection workflow.

`input.monolayer_poscars` are monolayer structures. They are used to build layer reciprocal lattices and interpret `layer_frac` valley centers. Once they are set, you usually do not need to write `monolayer_lattices.*.reciprocal_cart` manually.

`symmetry.operations.structure_file` is the moire or bilayer POSCAR/CONTCAR used for candidate real-space symmetry-operation detection. If it is missing or the path is wrong, `symmetry_report.json` will show that detection was skipped.

A monolayer POSCAR cannot replace the moire POSCAR for symmetry-operation detection. The moire POSCAR also should not be silently reused as a monolayer reciprocal lattice.

`symmetry.filters.rotation_order` controls which rotation eigenvalue is attempted:

- `auto`: infer the target order from the detected moire space group and candidate rotations. For example, `P321` / `P312` resolves to `C3`, and `P422` resolves to `C4`.
- integer `n`: analyze only `C_n`, currently with `n = 2, 3, 4, 6`.
- `None` / `none`: detect and report symmetry operations, but skip rotation-eigenvalue extraction.

By default, rotation eigenvalues are computed only for one generator of each selected cyclic rotation subgroup. For example, if the detected point group contains both `C3` and `C3^2`, ValleyScope reports the generator and keeps the other power in `symmetry_report.json` as a detected operation rather than computing a redundant eigenvalue table.

## Reading the Outputs

The analyzer writes results under `output.directory`:

```text
valley_summary.txt          ← read first (human-readable)
valley_summary.json
valley_weights.csv          ← quick scan
valley_subspace.json        ← primary result for near-degenerate states
rotation_eigenvalues.csv    ← rotation eigenvalues
symmetry_report.json        ← symmetry operation details
valley_basis_transform.h5   ← basis transform for later calculations
diagnostics.h5              ← projector masks and q-cut scan data (debug)
```

### `valley_weights.csv`

This CSV contains one row per raw VASP band:

```text
kpoint, band_vasp, energy_eV, K_sector, Kp_sector, W_val, P_v, eta, W_overlap, W_res
```

The sector columns are YAML labels. `W_val`, `P_v`, and `eta` store the target-valley-subspace weight, valley purity, and signed valley polarization. `W_overlap` stores the cross-sector projector-window overlap weight. `W_res` stores the out-of-valley residual weight.

Use this file for a first scan. For near-degenerate target states, the raw band rows are gauge-dependent and should not be the final interpretation.

### `valley_subspace.json`

This is the primary summary for near-degenerate states. It records the projected valley operator, the valley-adapted basis diagnostics, and the eigenvalues indicating how well the target subspace lies in the selected valley subspace.

### `rotation_eigenvalues.csv`

Rows are written only when an operation passes all V1 filters:

1. It is a proper rotation with order in `[2, 3, 4, 6]`.
2. It belongs to the little group of the target HSP.
3. It preserves the target valley sector.

An empty file with only a header is not automatically an error. It often means no operation passed these checks for the selected HSP and valley sector.

The readiness columns are deliberately conservative:

```text
rotation_ready, topology_input_ready, topology_ready, spinor_rotation_applied,
spinor_convention_verified, diagnostic_only, D_valley_offdiag_norm
```

`rotation_ready` checks whether the representation matrix was constructed without missing plane-wave mappings and with small unitarity deviation. `topology_input_ready` additionally requires a valid two-sector valley-adapted basis, small root-of-unity deviation, and small two-sector `D_valley_offdiag_norm`. It does not claim a full valley Chern number. For compatibility, `topology_ready` stores the same value. `D_valley_offdiag_norm` is only a two-sector valley-adapted diagnostic; it is not a general multi-valley or multidimensional-irrep criterion.

Spinor rows remain diagnostic-only unless the VASP spinor convention is benchmark-verified. If `spinor.convention_verified: true` is used, the benchmark label is written to `spinor_benchmark` so the result remains auditable.

The main columns mean:

- `kpoint`: the analyzed moire HSP.
- `operation_id`: the detected symmetry-operation id in `symmetry_report.json`; by default this is one generator of the selected cyclic rotation subgroup.
- `order`: rotation order, for example `3` for `C3`.
- `basis`: whether the representation was diagonalized in the `valley_adapted` basis or only as a raw diagnostic.
- `state_index`: index inside the chosen target subspace after diagonalizing the rotation matrix.
- `eigenvalue_real`, `eigenvalue_imag`: Cartesian components of the complex rotation eigenvalue.
- `phase_2pi`: eigenvalue phase divided by `2π`; for spinful `C3`, phases such as `1/6` and `5/6` are expected double-group phases.
- `nearest_root_of_unity` and `root_deviation`: nearest allowed root and its complex-plane distance from the computed eigenvalue.
- `unitarity_deviation`: deviation of the representation matrix from a unitary matrix within the selected plane-wave/band subspace.
- `rotation_ready`: the representation matrix passed the V1 numerical construction checks.
- `topology_input_ready`: conservative flag for later symmetry-based topology analysis input; it is not a topology result.
- `diagnostic_only`: the row should be read as a diagnostic rather than a validated topology input.
- `D_valley_offdiag_norm`: for two-sector valley-adapted diagnostics only, the off-diagonal norm of the rotation matrix in the valley-adapted basis.
- `valley_eta`: signed valley polarization of the valley-adapted state, when available.

### `valley_basis_transform.h5`

This file stores the transformation from the raw VASP band basis to the valley-adapted basis:

```text
/GammaM/transform
/GammaM/eta
/GammaM/s_matrix
/GammaM/v_matrix
/GammaM/band_indices_vasp
/GammaM/sectors
```

Use this transform for any later representation calculation in the valley-adapted subspace.

### `symmetry_report.json`

This records the detected symmetry operations, including operation type, rotation matrix, translation, candidate rotation status, operation-detection backend, and little-group / valley-preservation diagnostics.

If the report says:

```json
"status": "skipped"
```

the usual cause is that `symmetry.operations.structure_file` is missing or points to the wrong file. This path must be the moire or bilayer POSCAR/CONTCAR.

### `diagnostics.h5`

This file stores projector masks and q-cut scan data:

```text
/projectors/<kpoint>/sector_masks
/projectors/<kpoint>/center_masks
/projectors/<kpoint>/overlap_mask
/qcut_scan/<kpoint>/qcuts
/qcut_scan/<kpoint>/w_val
/qcut_scan/<kpoint>/purity
/qcut_scan/<kpoint>/eta
/qcut_scan/<kpoint>/W_overlap
/qcut_scan/<kpoint>/overlap_count
/rotation/<kpoint>/<operation_id>/D_raw
/rotation/<kpoint>/<operation_id>/D_valley
/rotation/<kpoint>/<operation_id>/eigenvalues
/rotation/<kpoint>/<operation_id>/root_deviation
/rotation/<kpoint>/<operation_id>/rotation_cart
/rotation/<kpoint>/<operation_id>/translation_cart
/rotation/<kpoint>/<operation_id>/rotation_ready
/rotation/<kpoint>/<operation_id>/topology_input_ready
/rotation/<kpoint>/<operation_id>/spinor_rotation_applied
/rotation/<kpoint>/<operation_id>/spinor_convention_verified
/rotation/<kpoint>/<operation_id>/diagnostic_only
/rotation/<kpoint>/<operation_id>/D_valley_offdiag_norm
```

Use it when tuning `qcut_fraction`, checking whether the projector windows select the expected momenta, or debugging the rotation representation. `D_valley` is present only when a valid valley-adapted basis was available.

## Roadmap

### V1.1: Little-group eigenvalue diagnostics

For each target HSP, compute representation matrices, eigenvalues, and characters for ALL valley-preserving little-group symmetry operations (not just the rotation generator). Output as transparent symmetry-eigenvalue diagnostics.

- Enumerate all proper-rotation operations that satisfy little-group AND valley-preservation checks.
- Compute `D_raw` and `D_valley` (valley-adapted) representation matrices.
- Report eigenvalues, phases, characters, and readiness flags per operation.
- Output: `little_group_eigenvalues.csv` and `little_group_representations.json`.

**Irrep label matching is deferred.** Valley-resolved single states are not automatically matched to full double space group irrep labels. This requires defining the valley-preserving subgroup restriction and parent full irrep relationship — theoretical work for a later stage.

### V2: Valley-resolved reduced EBR / topology diagnostics

- Define the valley-preserving subgroup `G_tau`.
- Construct or import reduced EBR tables for `G_tau`.
- Perform valley-resolved reduced compatibility and EBR decomposition.
- Classify results as atomic-limit compatible, reduced-EBR obstruction, or semimetal candidate.

Valley-resolved reduced EBR analysis is not the same as full physical-space-group EBR decomposition.

## V1 Scope

Current V1 does not include:

- full-mBZ valley-goodness validation;
- Berry curvature;
- Wilson loops;
- automatic integer Chern-number inference;
- true layer-resolved projection;
- automatic monolayer VBM/CBM valley search;
- complete support for every WAVECAR variant.

The priority is to make HSP valley projection, valley-adapted subspaces, and symmetry diagnostics reproducible and inspectable.
