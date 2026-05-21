# ValleyScope

[中文文档](README.zh.md)

ValleyScope is a VASP post-processing workflow for momentum-space valley projection and valley-resolved symmetry analysis in moire supercells. It starts from selected moire wavefunctions, measures their monolayer-valley character, fixes the valley gauge in near-degenerate target subspaces, and evaluates little-group representations for valley-preserving symmetry operations.

The code is designed for transparent symmetry analysis rather than automatic topology labeling. Symmetry eigenvalues and representation matrices are useful inputs for later topology diagnostics, but a strict valley-resolved topology statement requires additional validation beyond a few high-symmetry points.

## Physical Motivation

In a moire calculation, the Bloch momentum belongs to the moire Brillouin zone. The valley label of interest, however, is usually inherited from the monolayer Brillouin zone. A state at a moire high-symmetry point is therefore not "a K-valley state" just because the moire momentum is at a moire $K_M$ point. ValleyScope treats valley as a monolayer-valley label resolved through the plane-wave momenta contained in the moire wavefunction.

For twisted bilayers, top and bottom layer momenta associated with the same monolayer valley are allowed to hybridize. Such top/bottom hybridization does not by itself destroy the valley quantum number. The relevant question is whether the target state or target subspace remains confined to a chosen valley subspace, or whether there is appreciable intervalley mixing or weight outside the selected valley windows.

The workflow answers that question at selected high-symmetry points. It also checks whether a candidate symmetry operation belongs to the little group of the point and whether it preserves the selected valley before reporting symmetry eigenvalues.

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
A valley projector is built from a union of such windows. For a two-valley twisted bilayer example,

```math
\Omega_K=
\Omega_{{\rm top},K}\cup\Omega_{{\rm bottom},K},
\qquad
\Omega_{K'}=
\Omega_{{\rm top},K'}\cup\Omega_{{\rm bottom},K'} .
```
The labels used in YAML, such as `K_valley`, are user-chosen names for these valley subspaces.

### Weights and Intervalley-Mixing Diagnostics

Let $P_i$ be the projector for valley $i$. With the default policy, plane-wave components that fall into projector windows belonging to more than one valley are removed from all valley projectors and collected in the overlap projector $P_\times$:

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
For a normalized state $|\psi\rangle$, the monolayer-valley-resolved weight in valley $i$ is

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
It measures whether the assigned valley weight is concentrated in one valley or distributed among several valleys. For two valleys, `P_v` and `|eta|` are related: `P_v = (1 + |eta|) / 2`. For three or more valleys (e.g., the M-valley star in a hexagonal lattice), `|eta|` is not a general metric; the concentration score `P_v` (or `valley_concentration_alpha` in the adapted basis) is used instead.

The concentration score is classified into three public categories:
- **clean**: concentration above `valley_concentration_clean` (default 0.95, equivalent to `|eta|=0.90` in a two-valley analysis). Legacy alias: `P_v_clean`.
- **approx**: concentration between `valley_concentration_approx` (default 0.85) and the clean threshold. Legacy alias: `P_v_approx`.
- **mixed**: concentration below the approximate threshold; valley-preserving symmetry data should be treated as diagnostic-only.

The public summary status vocabulary also includes gate and availability states:

- **not_derived**: the target state or target subspace does not have enough weight in the user-defined valley subspace.
- **unreliable**: projector windows or residual weight fail the configured reliability checks.
- **n/a**: the subspace diagnostic is not applicable, for example for a single nondegenerate band.

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
It measures the weight in plane-wave components that fall into more than one valley window. With the default exclusion policy, this weight is not assigned to any valley.

The residual weight outside both the target valley subspace and the overlap region is

```math
W_{\rm res}=1-W_{\rm val}-W_{\rm overlap}.
```
For a normalized state, ValleyScope reports the decomposition $W_{\rm val}+W_{\rm overlap}+W_{\rm res}=1$. In CSV and JSON output, `W_overlap` stores $W_{\rm overlap}$ and `W_res` stores the out-of-valley residual weight. A large overlap weight is a warning about the chosen windows or cutoff, not evidence for a new physical valley.

### Gauge Fixing in a Near-Degenerate Subspace

For an isolated nondegenerate band, the single-state quantities above can be read directly. For a near-degenerate target subspace, raw VASP eigenvectors are gauge-dependent: any unitary rotation inside the degenerate subspace gives an equally valid set of eigenvectors. Row-by-row projection of individual VASP bands is therefore not the main physical diagnostic.

ValleyScope instead projects the target subspace and constructs the projected valley operators. Given $N_v$ valley projectors $P_a$ and raw VASP states $\{|\psi_i\rangle\}$, the projected valley matrices are

```math
\left(P_a^{\rm sub}\right)_{ij}=
\langle\psi_i|P_a|\psi_j\rangle,
\qquad
S=\sum_a P_a^{\rm sub}.
```

The matrix $S$ tests whether the target subspace lies in the chosen valley subspace (generalization of $W_{\rm val}$ to the subspace level). $s_{\rm min} = \min\,{\rm eig}(S)$ is the subspace-level derived score.

To fix a valley-adapted basis, ValleyScope constructs a label operator

```math
L = \sum_a \lambda_a P_a^{\rm sub},
```

where $\lambda_a$ are distinct real numbers (e.g., $-1, 1$ for two valleys, or $0,1,2,\dots$ for more). Diagonalizing $L$ yields the valley-adapted basis $|\phi_\alpha\rangle = \sum_i |\psi_i\rangle U_{i\alpha}$. In this basis,

```math
w_{\alpha a} = \langle\phi_\alpha|P_a|\phi_\alpha\rangle,
\qquad
{\rm assigned\_valley}_\alpha = \arg\max_a w_{\alpha a},
\qquad
{\rm concentration}_\alpha = \frac{\max_a w_{\alpha a}}{\sum_a w_{\alpha a}}.
```

For the special case of exactly two valleys, the signed polarization $\eta_\alpha = (w_{\alpha 1} - w_{\alpha 2})/(w_{\alpha 1} + w_{\alpha 2})$ is provided as a compatibility diagnostic, and $P_{v,\alpha} = (1 + |\eta_\alpha|)/2$. The old $V = P_K - P_{K'}$ is a special case, not the general multi-valley definition.

For three or more valleys, $\eta$ is undefined; use `valley_weights_adapted`, `assigned_valleys`, and `valley_concentration` instead. The `valley_subspace.json` and `valley_basis_transform.h5` files are the primary diagnostics for near-degenerate states.

### Symmetry Diagnostics

ValleyScope performs symmetry-operation detection from the moire or bilayer structure file. By default, candidate symmetry operations are detected from the moire structure file using spglib, with user-controlled symprec and optional symprec_scan. A symmetry eigenvalue is reported only after two checks:

1. The operation belongs to the HSP little group $G_k$ of the analyzed HSP.
2. The operation preserves the **specific valley being analyzed**, not all selected valleys.

The resulting matrix is a little-group representation in the valley-adapted subspace, restricted to the current valley block. Its eigenvalues are symmetry-analysis data. They can constrain topology in symmetry-based formulas, but ValleyScope does not infer a full integer Chern number from high-symmetry-point data alone.

Valley-preserving subgroups are reported separately for each valley. Valley orbits and valley mappings are included in the subgroup report. The current V1.1 reports valley orbits and operation mappings; strict minimal coset representatives and induced representation relations are not yet fully automated. The all-valley intersection (operations preserving every selected valley) is retained as a debug field but is not used for valley-preserving irrep matching.

### Single-Valley Irrep Interpretation

A valley-preserving irrep should be compared with irreps of the valley-preserving subgroup inside the HSP little group:

```math
G_k^{(a)}=\{\,g\in G_k\mid \pi_g(a)=a\,\},
```

where $G_k$ is the HSP little group and $\pi_g(a)$ is the valley mapping induced by operation $g$. It should not be interpreted as an irrep of the full moire space group unless the operation preserves that valley subspace.

With SOC, the comparison must use double-valued irreps. A spinor wavefunction changes sign under a $2\pi$ rotation, so spinful $C_3$ satisfies $C_3^3=-1$ and allows eigenvalues such as $\exp(+i\pi/3)$, $-1$, and $\exp(-i\pi/3)$.

After valley-preservation filtering, ValleyScope reports $G_k^{(a)}$ for each selected valley, not the all-valley intersection. Operations that move valley $a$ to another selected valley are reported as valley-changing orbit data and are not used as valley-preserving eigenvalue rows for $a$. When operation-to-table mapping is complete, `irrep_matching.irrep_results_by_kpoint` reports representation-level `irrep_multiplicities` from character decomposition. When every ready state independently selects a unique one-dimensional irrep, `state_irrep_results` records per-state irrep labels. These are valley-preserving character matching results, not reduced EBR decompositions or topology conclusions.

### M-Star Valley Orbit and Valley-Preserving Subgroups

For a three-valley M-star in a hexagonal moire cell (e.g., $P321$ / $P312$), the three M points form a valley orbit:

```math
O_M = \{M_1, M_2, M_3\}
```

Typical valley mappings are:

```math
C_3: M_1 \to M_2 \to M_3 \to M_1,
\qquad
C_2^{(M_1)}: M_1 \to M_1,\; M_2 \leftrightarrow M_3 .
```

The HSP little group is:

```math
G_k = \{ g \in G \mid g k = k + \mathbf G_M \}.
```

The valley-preserving subgroup for $M_1$ inside the HSP little group is:

```math
G_k^{(M_1)} = \{ g \in G_k \mid \pi_g(M_1) = M_1 \}.
```

Similarly, $G_k^{(M_2)}=\{g\in G_k\mid\pi_g(M_2)=M_2\}$ and $G_k^{(M_3)}=\{g\in G_k\mid\pi_g(M_3)=M_3\}$. When the corresponding valley orbit is $C_3$-related, these subgroups are related by conjugation:

```math
G_k^{(M_2)} = C_3 G_k^{(M_1)} C_3^{-1}, \qquad
G_k^{(M_3)} = C_3^2 G_k^{(M_1)} C_3^{-2}.
```

Single-valley irreps should be matched to the corresponding valley-preserving subgroup $G_k^{(M_i)}$, while the full M-star representation also requires valley-changing operations and valley sewing matrices.

**V1.1 scope:** ValleyScope currently reports valley orbits and operation mappings. Strict minimal coset representatives, induced representation decomposition, and reduced EBR decomposition are deferred to later work. Full-group irreps describe the entire M-star manifold; valley-preserving irreps describe $G_k^{(M_i)}$. If both are presented, the orbit, mapping, and induction-subduction relations must also be stated, otherwise the information is incomplete.

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

valley_subspaces:
  - name: K_valley
    centers: [K]
  - name: Kp_valley
    centers: [Kp]

output:
  directory: ./valley_analysis
```

Current analysis configs use `analysis.iband` for VASP band indices and
`valley_subspaces` for user-defined monolayer-valley subspaces.

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

  # Optional: max |delta_Ecut| in eV for automatic G-list reconstruction
  # cutoff adjustment (default 0.0 = strict exact-match only).
  # Start from 0.005 eV or 0.01 eV.  Values near 0.05 eV or larger are
  # suspicious -- check WAVECAR variant, lattice convention, G-list ordering,
  # and k-point convention before raising the tolerance.
  # ecut_adjust_tol does NOT modify the DFT ENCUT; it only adjusts the
  # post-processing G-list reconstruction effective cutoff tolerance.
  # ecut_adjust_tol: 0.0

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
/metadata/g_list_reconstruction_mode   (exact or ecut_adjusted)
/metadata/original_encut_eV
/metadata/reconstruction_encut_eV
/metadata/ecut_adjust_tol_eV
/metadata/ecut_adjust_delta_eV
/kpoints/N/name
/kpoints/N/frac
/kpoints/N/g_vectors_frac
/kpoints/N/coefficients       [nb, nspinor, nG]
/kpoints/N/energies_eV
/kpoints/N/band_indices_vasp
/kpoints/N/nplane_record
/kpoints/N/target_g_count
/kpoints/N/generated_g_count_at_header_encut
/kpoints/N/generated_g_count_final
/kpoints/N/ecut_adjust_delta_eV
```

If the extractor reports a G-vector count mismatch, first try adding a small `ecut_adjust_tol` (start from 0.005 eV or 0.01 eV) to the config. WAVECAR files from large SOC/noncollinear moire supercells can show small differences between VASP's internal G-list cutoff and the reconstructed cutoff. Values near 0.05 eV or larger are suspicious — check the WAVECAR variant, lattice convention, G-list ordering, and k-point convention. `ecut_adjust_tol` is a post-processing G-list reconstruction tolerance, not a modification of the DFT ENCUT. Strict mode (`ecut_adjust_tol: 0.0`) remains the default.

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

valley_subspaces:
  - name: K_valley
    centers: [top_K, bottom_K]
  - name: Kp_valley
    centers: [top_Kp, bottom_Kp]

projection:
  qcut_fraction: 0.20
  qcut_scan: [0.15, 0.20, 0.25, 0.30]
  thresholds:
    W_val_min: 0.8
    valley_concentration_clean: 0.95
    valley_concentration_approx: 0.85

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
  convention_verified: true
  benchmark: spinor_C3_reference_check

rotation:
  readiness_preset: strict
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
Valley subspaces
Valley projection summary
Valley subspace analysis
Symmetry analysis
Symmetry eigenvalues
Warnings
Output files
```

Use `Valley projection summary` for raw-band quantities:

```text
W_val:      valley-subspace weight
P_v:        valley purity
W_overlap: projector-window overlap weight
W_res:      residual weight
```

The screen `status` is intentionally compact but keeps gate failures distinct: `clean`, `approx`, `mixed`, `not_derived`, `unreliable`, or `n/a`.

Use `Valley subspace analysis` for the target-band subspace as a whole. Raw VASP eigenvectors inside a near-degenerate subspace are gauge-dependent, so the per-band projection is not the final diagnostic. ValleyScope constructs general projected valley matrices $P_a^{\rm sub}$ and a label operator $L = \sum_a \lambda_a P_a^{\rm sub}$:

```text
S = sum_a P_a^sub:  target-valley-subspace projector in the selected target bands
S_min:              minimum target-valley-subspace weight
S_max:              maximum target-valley-subspace weight
min_concentration:  minimum valley concentration in the valley-adapted basis
assigned_valleys:   valley assignment per adapted state
eta_adapted:        signed valley polarization (two-valley compatibility only)
```

`S` checks whether the selected target bands are well described by the chosen valley subspace. The label operator `L` fixes the valley-adapted basis for any number of valleys. For two valleys, `eta_adapted` is provided as a compatibility diagnostic. For three or more valleys (e.g., M-valley star), `eta` is absent; use `valley_weights_adapted`, `assigned_valleys`, and `valley_concentration` instead. The two-valley $V = P_K - P_{K'}$ is a special case of this general framework.

The screen `status` values are `clean`, `approx`, `mixed`, `not_derived`, `unreliable`, and `n/a`.

`Symmetry analysis` first reports the detected space group and symmetry operations, then lists which operations belong to each target HSP little group and which of those preserve each selected valley. Operations that map one selected valley to another are reported as `valley-changing`. The JSON summary also contains `symmetry_analysis.valley_preserving_subgroup_report`: it reports valley orbits, valley-preserving subgroups, and the all-valley intersection as a debug-only field. If `irreptables` table mapping and character matching are complete, `irrep_matching.irrep_results_by_kpoint` records HSP irrep multiplicities such as `{"-K5": 1, "-K6": 1}`. When each ready state separately matches a unique one-dimensional irrep, the same result includes `state_irrep_results` with per-state labels. `Symmetry eigenvalues` reports the representation eigenvalues that were actually computed. `symmetry_characters` aggregates $\chi^{a}_k(g)=\mathrm{Tr}\,D^{a}_k(g)$ for computed operations that pass the HSP-little-group and valley-preserving checks; it is the character input layer for irrep matching. `topology_input_ready` only means that the HSP symmetry eigenvalue is suitable as an input to a later symmetry-based topology analysis; it does not validate full-mBZ valley-resolved topology.

For SOC/spinor wavefunctions, ValleyScope applies the SU(2) spin rotation in the plane-wave representation. By default, VASP spinor phase conventions are treated as unverified, so spinful rows are reported with `spinor_rotation_applied=True`, `spinor_convention_verified=False`, and `diagnostic_only=True`. After an explicit spinor-convention check, set `spinor.convention_verified: true` and record the check name in `spinor.benchmark`; this still does not validate full-mBZ valley-resolved topology. `spinor.convention` is optional in YAML because only the default VASP up/down convention is currently supported.

Example screen summary:

```text
Input
-----
wavefunction_h5: ./wave.h5
operation structure: ./2dm-5370-7.34.vasp
operation-detection backend: spglib
spinor convention: vasp_up_down_saxis_z (verified=True, benchmark=spinor_C3_reference_check)
target k-points: GammaM, KM, MM
iband (VASP): 2195, 2196
qcut mode: relative_min_valley_distance
qcut value: 0.034 A^-1

Valley subspaces
----------------
label     centers
--------  ----------------
K_valley  top_K, bottom_K
Kp_valley top_Kp, bottom_Kp

Valley projection summary
-------------------------
W_val:      valley-subspace weight
P_v:        valley purity
W_overlap: projector-window overlap weight
W_res:      residual weight
kpoint  band  W_val  P_v   W_overlap  W_res  status
------  ----  -----  ----  ---------  -----  ------
GammaM  2195  0.98   0.99  0          0.02   clean
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

`symmetry.filters.rotation_order` is a legacy summary/highlight field. In V1.1 it does **not** control which operations enter symmetry analysis. All detected proper HSP-little-group operations (order `2`, `3`, `4`, `6`) are enumerated and filtered by valley-preserving subgroup membership. `rotation_order` records the requested or automatically resolved rotation order for summary compatibility:

- `auto`: infer the target order from the detected moire space group (e.g., `P321` / `P312` → `C3`, `P422` → `C4`).
- integer `n`: select `C_n` as the highlighted cyclic rotation order (`n = 2, 3, 4, 6`).
- `None` / `none`: detect and report symmetry operations, but skip symmetry-eigenvalue extraction.

The valley-preserving gate is:

```math
\text{little\_group\_passed}(g, k) \land \text{mapped\_valley}[v_a] = v_a
```

not the old all-valley intersection. An operation that preserves one valley while exchanging others is a valid valley-preserving operation for the preserved valley. Valley-changing operations are reported as valley-orbit data but do not enter the source valley's valley-preserving eigenvalue rows.

`root_deviation_tol`, `D_valley_offdiag_tol`, and `irrep_weight_tol` are numerical readiness thresholds, not universal physical constants. `root_deviation_tol` checks how close a computed symmetry eigenvalue is to the nearest allowed root of unity. `D_valley_offdiag_tol` checks the two-valley `D_valley` off-diagonal norm in the current valley-adapted benchmark. `irrep_weight_tol` checks whether character-decomposition weights are close enough to integers for irrep labeling. `readiness_preset` accepts `strict`, `normal`, and `loose`: `strict` keeps `1.0e-6`, `1.0e-6`, and `1.0e-5`; `normal` uses `1.0e-5`, `1.0e-3`, and `5.0e-5`; `loose` uses `1.0e-4`, `1.0e-2`, and `1.0e-4`. Explicit threshold values override the preset. Interpret all of them together with qcut stability, `W_val`, `P_v`, `S_min`, spinor benchmark status, plane-wave mapping quality, and symmetry tolerance. Do not loosen them only to obtain `topology_input_ready=True` or an irrep label.

## Reading the Outputs

The analyzer writes results under `output.directory`:

```text
valley_summary.txt          ← read first (human-readable)
valley_summary.json
valley_weights.csv          ← quick scan
valley_subspace.json        ← multi-valley subspace data
symmetry_eigenvalues.csv    ← symmetry eigenvalues
symmetry_report.json        ← symmetry analysis
valley_basis_transform.h5   ← basis transform for later calculations
diagnostics.h5              ← projector masks and q-cut scan data (debug)
```

### `valley_weights.csv`

This CSV contains one row per raw VASP band:

```text
kpoint, band_vasp, energy_eV, K_valley, Kp_valley, W_val, P_v, eta, W_overlap, W_res
```

The valley columns are YAML labels. `W_val`, `P_v`, and `eta` store the target-valley-subspace weight, valley purity, and signed valley polarization. `W_overlap` stores the projector-window overlap weight. `W_res` stores the out-of-valley residual weight.

Use this file for a first scan. For near-degenerate target states, the raw band rows are gauge-dependent and should not be the final interpretation.

### `valley_subspace.json`

This is the primary summary for near-degenerate states. It records the projected valley matrices $P_a^{\rm sub}$, the label operator $L$, the valley-adapted basis diagnostics, and S-matrix eigenvalues. For general multi-valley analysis, the key fields are:

- `valley_labels`: list of valley names
- `s_matrix` / `s_eigenvalues` / `s_min` / `s_max`: subspace-level diagnostics
- `valley_matrices`: projected valley matrices $P_a^{\rm sub}$
- `label_operator`: the valley label operator $L$
- `valley_weights_adapted`: per-state valley weights in the adapted basis
- `assigned_valleys`: valley assignment per adapted state
- `valley_concentration` / `min_valley_concentration`: concentration scores
- `eta_adapted`: signed polarization (only for exactly two valleys)
- `commutator_norm_max` / `idempotency_deviation_max`: numerical diagnostics
- `stably_separable` / `reason`: stability verdict

Note: `valley_weights` and `sector_weights` hold the same data in `valley_subspace.json`. The `sector_weights` key is retained as a legacy alias. New code should prefer `valley_weights`. Similarly, `diagnostics.h5` stores masks under both `valley_masks` (preferred) and `sector_masks` (legacy). Internal class names such as `SectorProjectors` may still use "sector" for historical reasons.

### `symmetry_eigenvalues.csv`

Rows are written only when an operation passes all representation filters:

1. It is a proper rotation with order in `[2, 3, 4, 6]`.
2. It belongs to the little group of the target HSP.
3. It preserves the selected valley.

An empty file with only a header is not automatically an error. It often means no operation passed these checks for the selected HSP and valley.

The readiness columns are deliberately conservative:

```text
rotation_ready, topology_input_ready, topology_ready, spinor_rotation_applied,
spinor_convention_verified, diagnostic_only, D_valley_offdiag_norm
```

`rotation_ready` checks whether the representation matrix was constructed without missing plane-wave mappings and with small unitarity deviation. `topology_input_ready` additionally requires a valid valley-adapted basis, small root-of-unity deviation, and small `D_valley_offdiag_norm`. It does not claim a full valley Chern number. For compatibility, `topology_ready` stores the same value. `D_valley_offdiag_norm` is a two-valley quality measure; it is not a general multi-valley criterion.

Spinor rows remain diagnostic-only unless the VASP spinor convention is benchmark-verified. If `spinor.convention_verified: true` is used, the benchmark label is written to `spinor_benchmark` so the result remains auditable.

The main columns mean:

- `kpoint`: the analyzed moire HSP.
- `operation_id`: the detected symmetry-operation id in `symmetry_report.json`.
- `order`: rotation order, for example `3` for `C3`.
- `state_index`: index inside the chosen target subspace after diagonalizing the rotation matrix.
- `eigenvalue_real`, `eigenvalue_imag`: Cartesian components of the complex symmetry eigenvalue.
- `phase_2pi`: eigenvalue phase divided by `2π`; for spinful `C3`, phases such as `1/6` and `5/6` are expected double-group phases.
- `nearest_root_of_unity` and `root_deviation`: nearest allowed root and its complex-plane distance from the computed eigenvalue.
- `unitarity_deviation`: deviation of the representation matrix from a unitary matrix within the selected plane-wave/band subspace.
- `rotation_ready`: the representation matrix passed the numerical construction checks.
- `topology_input_ready`: conservative flag for later symmetry-based topology analysis input; it is not a topology result.
- `diagnostic_only`: the row should be read as a diagnostic rather than a validated topology input.
- `D_valley_offdiag_norm`: for two-valley quality checks only, the off-diagonal norm of the representation matrix in the valley basis.
- `valley_eta`: signed valley polarization of the valley-adapted state, when available.

### `symmetry_characters` in `valley_summary.json`

The summary JSON contains a first-class `symmetry_characters` list. Each row is grouped by `(kpoint, target_valley, operation_id)` and records `character_raw`, `character_valley`, readiness flags, and whether the operation was accepted for that valley-preserving representation. Rows are included only after the HSP-little-group and valley-preserving checks pass. This is the character input layer for later valley-preserving irrep matching; no character table or reduced EBR decomposition is applied here.

### `valley_basis_transform.h5`

This file stores the transformation from the raw VASP band basis to the valley-adapted basis:

```text
/GammaM/transform
/GammaM/eta                 (two-valley only)
/GammaM/s_matrix
/GammaM/label_operator
/GammaM/s_eigenvalues
/GammaM/valley_weights_adapted
/GammaM/assigned_valleys
/GammaM/valley_concentration
/GammaM/band_indices_vasp
/GammaM/valid_valley_subspace
```

For exactly two valleys, `eta` and `v_matrix` are included as compatibility fields. For three or more valleys, use `valley_weights_adapted` and `assigned_valleys`. Use this transform for any later representation calculation in the valley-adapted subspace.

### `symmetry_report.json`

This records the detected symmetry operations, including operation type, rotation matrix, translation, candidate rotation status, operation-detection backend, and little-group / valley-preservation diagnostics.

The summary JSON additionally exposes `symmetry_analysis.valley_preserving_subgroup_report`. This result lists valley orbits, valley-preserving operation ids, closure status, and any standard subgroup match obtained from the detected preserving operations. The all-valley intersection is kept only as a debug field and is not used for irrep matching. `irrep_matching` records table mapping and, when enough ready characters are available, `irrep_results_by_kpoint` with representation-level `irrep_multiplicities` and clean one-dimensional `state_irrep_results`.

If the report says:

```json
"status": "skipped"
```

the usual cause is that `symmetry.operations.structure_file` is missing or points to the wrong file. This path must be the moire or bilayer POSCAR/CONTCAR.

### `diagnostics.h5`

This file stores projector masks, q-cut scan data, and symmetry representation matrices:

```text
/projectors/<kpoint>/center_masks
/projectors/<kpoint>/overlap_mask
/qcut_scan/<kpoint>/qcuts
/qcut_scan/<kpoint>/w_val
/qcut_scan/<kpoint>/purity
/qcut_scan/<kpoint>/eta
/qcut_scan/<kpoint>/W_overlap
/qcut_scan/<kpoint>/overlap_count
/symmetry_representations/<kpoint>/<operation_id>/D_raw
/symmetry_representations/<kpoint>/<operation_id>/D_valley
/symmetry_representations/<kpoint>/<operation_id>/eigenvalues
/symmetry_representations/<kpoint>/<operation_id>/root_deviation
/symmetry_representations/<kpoint>/<operation_id>/rotation_cart
/symmetry_representations/<kpoint>/<operation_id>/translation_cart
/symmetry_representations/<kpoint>/<operation_id>/rotation_ready
/symmetry_representations/<kpoint>/<operation_id>/topology_input_ready
/symmetry_representations/<kpoint>/<operation_id>/spinor_rotation_applied
/symmetry_representations/<kpoint>/<operation_id>/spinor_convention_verified
/symmetry_representations/<kpoint>/<operation_id>/diagnostic_only
/symmetry_representations/<kpoint>/<operation_id>/D_valley_offdiag_norm
```

Use it when tuning `qcut_fraction`, checking whether the projector windows select the expected momenta, or debugging the rotation representation. `D_valley` is present only when a valid valley-adapted basis was available.
