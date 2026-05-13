<a id="english"></a>
# ValleyScope

**Language:** English | [中文](#zh-cn)

ValleyScope is a VASP post-processing workflow for momentum-space valley projection and symmetry diagnostics in moire supercells. It is intended for situations where the relevant low-energy states inherit valley character from an underlying monolayer Brillouin zone, but the actual first-principles calculation is performed in a large moire supercell.

The code is not a black-box Chern-number calculator. V1 is an HSP-only diagnostic: it analyzes selected high-symmetry points, constructs monolayer-valley projectors in the plane-wave Hilbert space, fixes the valley gauge of near-degenerate target subspaces, and reports the symmetry data needed for rotation-eigenvalue diagnostics. A strict valley-resolved topology statement still requires later full-mBZ validation.

## Physical Motivation

In a moire calculation, the Bloch momentum belongs to the moire Brillouin zone. The valley label of interest, however, is usually inherited from the monolayer Brillouin zone. A state at a moire high-symmetry point is therefore not “a K-valley state” just because the moire momentum is at a moire $K_M$ point. ValleyScope treats valley as a monolayer-valley sector resolved through the plane-wave momenta contained in the moire wavefunction.

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
The labels used in YAML, such as `K_sector`, are only user-chosen names for these physical sectors. They should be read as “K-valley sector” or “K'-valley sector,” not as special code-defined physical objects.

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
It measures whether the assigned valley weight is concentrated in one valley sector or distributed among several sectors.

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

ValleyScope performs symmetry-operation detection from the moire or bilayer structure file. By default, candidate symmetry operations are detected from the moiré structure file using spglib, with user-controlled symprec and optional symprec_scan. A rotation eigenvalue is reported only after two checks:

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

### Analyze HSP Wavefunctions

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
  target_bands_vasp: [2195, 2196]
  degeneracy_tol_meV: 1.0

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

valley_sectors:
  - name: K_sector
    centers: [top_K, bottom_K]
  - name: Kp_sector
    centers: [top_Kp, bottom_Kp]

projection:
  use_2d_momentum_only: true
  qcut_mode: relative_min_sector_distance
  qcut_fraction: 0.30
  qcut_scan: [0.20, 0.25, 0.30, 0.35]
  overlap_cross_sector: warn_exclude
  thresholds:
    W_val_min: 0.8
    P_v_clean: 0.95
    P_v_approx: 0.85

symmetry:
  operations:
    mode: auto
    structure_file: ./2dm-5370-7.34.vasp
    backend: spglib
  tolerance:
    symprec: 1.0e-3
    angle_tolerance: -1.0
    symprec_scan: [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3]
  filters:
    proper_rotations_only: true
    allowed_orders: [2, 3, 4, 6]

output:
  directory: ./valley_analysis
  summary_stdout: true
  write_summary_txt: true
  write_summary_json: true
  write_detailed_files: true
  write_json: true
  write_csv: true
  write_hdf5_basis_transform: true
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

Use the `Valley projection summary` table to check `W_val`, `P_v`, `W_overlap`, and `W_res` band by band. Use `Valley-adapted subspace` to decide whether the near-degenerate target subspace is a good two-valley subspace. In the rotation table, `topology_input_ready` only means that the HSP rotation eigenvalue is suitable as an input to a later symmetry-based topology analysis; it does not validate full-mBZ valley-resolved topology. The legacy `topology_ready` column is kept as a backward-compatible alias of `topology_input_ready`.

For SOC/spinor wavefunctions, ValleyScope applies the SU(2) spin rotation in the plane-wave representation, but VASP spinor phase conventions have not yet been benchmark-verified. Such rows are reported with `spinor_rotation_applied=True`, `spinor_convention_verified=False`, and `diagnostic_only=True`; they are not marked `topology_input_ready`.

Example screen summary:

```text
Input
-----
wavefunction_h5: ./wave.h5
operation structure: ./2dm-5370-7.34.vasp
operation-detection backend: spglib
target k-points: GammaM, KM, MM
target bands (VASP): 2195, 2196
qcut mode: relative_min_sector_distance
qcut value: 0.034 A^-1

Valley manifolds
----------------
label      centers
---------  ----------------
K_sector   top_K, bottom_K
Kp_sector  top_Kp, bottom_Kp

Valley projection summary
-------------------------
kpoint  band  W_val  P_v   W_overlap  W_res  status
------  ----  -----  ----  ---------  -----  ------------
GammaM  2195  0.98   0.99  0          0.02   valley-clean
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

## Reading the Outputs

The analyzer writes results under `output.directory`:

```text
valley_summary.txt
valley_summary.json
valley_weights.csv
valley_subspace.json
valley_basis_transform.h5
symmetry_report.json
rotation_eigenvalues.csv
diagnostics.h5
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

Spinor rows remain diagnostic-only unless the VASP spinor convention is benchmark-verified.

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

---

<a id="zh-cn"></a>
# ValleyScope 中文说明

**语言:** [English](#english) | 中文

ValleyScope 是一套面向 VASP moire 超胞波函数的动量空间谷投影（momentum-space valley projection）与对称性诊断（symmetry diagnostics）后处理流程。它适用于这样一类问题：低能态的谷特征（valley character）来自底层单层布里渊区，但实际第一性原理计算是在大型 moire 超胞中完成的。

它不是黑箱陈数（Chern number）程序。V1 是只针对高对称点的诊断（HSP-only diagnostic）：它只分析用户选定的高对称点，在平面波希尔伯特空间（plane-wave Hilbert space）中构造单层谷投影算符（monolayer-valley projectors），对近简并目标子空间进行谷规范固定（valley gauge fixing），并输出旋转本征值诊断（rotation-eigenvalue diagnostics）所需的对称性信息。严格的谷分辨拓扑判断仍然需要后续全 moire 布里渊区验证（full-mBZ validation）。

## 物理动机

在 moire 计算中，布洛赫动量（Bloch momentum）属于 moire 布里渊区。但我们关心的谷标记（valley label）通常继承自单层布里渊区（monolayer Brillouin zone）。因此，某个态位于 moire 高对称点并不意味着它就是某个单层谷态。ValleyScope 把谷定义为通过 moire 波函数中的平面波动量（plane-wave momenta）解析出的单层谷扇区（monolayer-valley sector）。

对转角双层体系（twisted bilayer），同一个单层谷中的上下层动量可以发生杂化。这种层间杂化（top/bottom layer hybridization）本身不破坏谷量子数。真正需要诊断的是目标态或目标子空间是否仍然局限在选定的谷子空间（valley subspace）中，还是出现了明显的谷间混合（intervalley mixing）或谷外剩余权重（out-of-valley residual weight）。

V1 工作流在选定高对称点上回答这个问题。它还会检查候选对称操作是否属于该高对称点的小群（little group），以及是否保持目标谷扇区（preserve target valley sector），然后才报告旋转本征值（rotation eigenvalue）。此类本征值是有用的对称性诊断，但单独并不能证明完整整数陈数。

## 物理定义

### 动量空间谷投影（Momentum-Space Valley Projection）

对 moire 超胞中动量为 $\mathbf k_M$ 的 VASP 布洛赫态，每个平面波分量的物理动量为

```math
\mathbf q=\mathbf k_M+\mathbf G_M ,
```
其中 $\mathbf G_M$ 是 moire 超胞倒格矢。ValleyScope 只使用面内分量 $\mathbf q_\parallel$ 进行谷归属（valley assignment）。

单层谷中心（monolayer valley center）$a$ 在单层倒空间坐标中给出，并通过层变换（layer transform）映射到 moire 坐标系。它对应的投影窗口（projector window）由到单层倒格矢星的最小距离定义：

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
谷扇区投影算符（valley-sector projector）由一组窗口的并集构造。以含两个谷的转角双层为例，

```math
\Omega_K=
\Omega_{{\rm top},K}\cup\Omega_{{\rm bottom},K},
\qquad
\Omega_{K'}=
\Omega_{{\rm top},K'}\cup\Omega_{{\rm bottom},K'} .
```
YAML 中的 `K_sector` 这类名称只是用户定义的标签。物理上应理解为 K 谷扇区（K-valley sector）或 K' 谷扇区（K'-valley sector），而不是代码内置的特殊物理对象。

### 权重与谷间混合诊断（Weights and Intervalley-Mixing Diagnostics）

设 $P_i$ 是第 $i$ 个谷扇区的投影算符。默认策略下，如果某个平面波分量同时落入多个扇区的投影窗口，它会从所有谷扇区投影算符中移除，并放入重叠投影算符（overlap projector）$P_\times$：

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
对归一化态 $|\psi\rangle$，第 $i$ 个扇区的单层谷分辨权重（monolayer-valley-resolved weight）为

```math
W_i=\langle\psi|P_i|\psi\rangle .
```
目标谷子空间权重（target-valley-subspace weight）为

```math
W_{\rm val}=\sum_i W_i .
```
它衡量该态有多少权重落在用户定义的目标谷子空间中。它依赖谷中心和投影半径，不是拓扑不变量。

谷纯度（valley purity）定义为

```math
P_v=\frac{\max_i W_i}{W_{\rm val}},
\qquad W_{\rm val}>0 .
```
它衡量已经分配到谷子空间的权重是否集中在单一谷扇区中。

对按 K 谷、K' 谷排列的两谷子空间，有符号谷极化（signed valley polarization）为

```math
\eta=
\frac{W_K-W_{K'}}{W_K+W_{K'}}
=
\frac{W_K-W_{K'}}{W_{\rm val}} .
```
投影窗口重叠权重（projector-window overlap weight）为

```math
W_{\rm overlap}=\langle\psi|P_\times|\psi\rangle .
```
它衡量落入多个谷扇区投影窗口的平面波分量权重。在默认排除策略下，这部分权重不会分配给任何谷扇区。

谷外剩余权重（out-of-valley residual weight）定义为目标谷子空间和重叠区域之外的剩余部分：

```math
W_{\rm res}=1-W_{\rm val}-W_{\rm overlap}.
```
对归一化态，V1 因而报告分解 $W_{\rm val}+W_{\rm overlap}+W_{\rm res}=1$。在 CSV 和 JSON 输出中，`W_overlap` 保存 $W_{\rm overlap}$，`W_res` 保存谷外剩余权重。大的重叠权重通常说明投影窗口或截断半径需要检查，而不是出现了新的物理谷。

### 近简并子空间中的规范固定（Gauge Fixing）

对孤立非简并能带，可以直接读取上述单态诊断。对近简并目标子空间，VASP 原始本征矢是规范依赖的（gauge-dependent）：子空间内部任意酉变换都会给出同样合法的本征矢集合。因此，逐条 VASP 能带投影不是主要物理诊断。

V1 会投影整个目标子空间并构造投影谷算符（projected valley operator）。对由 VASP 原始态 $\{|\psi_m\rangle\}$ 张成的两谷子空间，

```math
S_{mn}=
\langle\psi_m|(P_K+P_{K'})|\psi_n\rangle,
\qquad
V_{mn}=
\langle\psi_m|(P_K-P_{K'})|\psi_n\rangle .
```
矩阵 $S$ 检查目标子空间是否落在选定谷子空间中。矩阵 $V$ 在目标子空间内选出谷适配基（valley-adapted basis）。实际读取近简并态时，应主要看 `valley_subspace.json` 和 `valley_basis_transform.h5`。

### 对称性诊断（Symmetry Diagnostics）

ValleyScope 从 moire 或双层结构文件中自动识别候选实空间对称操作。默认情况下，程序用 spglib 作为候选对称操作识别后端，并由用户控制 `symprec` 和可选的 `symprec_scan`。只有通过两项检查后，程序才报告旋转本征值：

1. 该操作属于目标高对称点的小群。
2. 该操作保持目标谷扇区。

得到的矩阵是谷适配子空间中的小群表示（little-group representation）。其本征值是旋转本征值诊断。它们可用于基于对称性的拓扑公式中的约束，但 V1 不自动推断完整整数陈数。

## 运行流程

### 安装

```bash
git clone https://github.com/gawcista/valley-proj.git
cd valley-proj
pip install -e .
```

源码目录中可以直接运行：

```bash
python -m valleyscope.cli --help
```

安装后命令行为：

```bash
valleyscope --help
```

### 从这里开始

推荐流程只有两步。

第一步，从 `WAVECAR` 抽取一个较小的 HDF5 文件：

```bash
valleyscope extract-wavecar extract.yaml
```

第二步，分析选定的高对称点（HSP）波函数：

```bash
valleyscope analyze-hsp analyze.yaml
```

主分析流程读取 HDF5，而不是反复直接读取完整 `WAVECAR`。这样可以把物理分析和大型二进制解析分开，也更方便检查具体抽取了哪些 k 点和能带。

### 从 WAVECAR 抽取

先写 `extract.yaml`：

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

运行：

```bash
valleyscope extract-wavecar extract.yaml
```

抽取后，应检查 HDF5 中的 k 点、能带和数组形状。最重要的数据集是：

```text
/metadata/g_vector_order
/kpoints/N/name
/kpoints/N/frac
/kpoints/N/g_vectors_frac
/kpoints/N/coefficients       [nb, nspinor, nG]
/kpoints/N/energies_eV
/kpoints/N/band_indices_vasp
```

如果抽取器报 G-vector 数量不匹配，不要强行继续。这通常表示当前版本还不支持该 `WAVECAR` 变体或 G-vector 列表约定。

### 分析高对称点（HSP）波函数

对转角双层体系，`analyze.yaml` 里最容易出错的是层变换（layer transform）。如果结构来自整数公度矩阵，优先使用 `supercell_matrix`，不要只根据转角手写 `rotation_deg`。实际 moire 超胞可能带有整体基矢旋转，单独的转角往往不够。

以 `generate_hexagonal_210(9, 5, ...)` 这类结构为例：

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
  target_bands_vasp: [2195, 2196]
  degeneracy_tol_meV: 1.0

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

valley_sectors:
  - name: K_sector
    centers: [top_K, bottom_K]
  - name: Kp_sector
    centers: [top_Kp, bottom_Kp]

projection:
  use_2d_momentum_only: true
  qcut_mode: relative_min_sector_distance
  qcut_fraction: 0.30
  qcut_scan: [0.20, 0.25, 0.30, 0.35]
  overlap_cross_sector: warn_exclude
  thresholds:
    W_val_min: 0.8
    P_v_clean: 0.95
    P_v_approx: 0.85

symmetry:
  operations:
    mode: auto
    structure_file: ./2dm-5370-7.34.vasp
    backend: spglib
  tolerance:
    symprec: 1.0e-3
    angle_tolerance: -1.0
    symprec_scan: [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3]
  filters:
    proper_rotations_only: true
    allowed_orders: [2, 3, 4, 6]

output:
  directory: ./valley_analysis
  summary_stdout: true
  write_summary_txt: true
  write_summary_json: true
  write_detailed_files: true
  write_json: true
  write_csv: true
  write_hdf5_basis_transform: true
```

运行：

```bash
valleyscope analyze-hsp analyze.yaml
```

### 如何阅读屏幕输出

`analyze-hsp` 会先在终端打印一份紧凑的物理诊断摘要。它是第一层判断依据；CSV、JSON 和 HDF5 文件主要用于复现、检查和调试。

屏幕摘要按以下部分组织：

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

先看 `Valley projection summary` 表中的 `W_val`、`P_v`、`W_overlap` 和 `W_res`，判断每条能带是否主要来自目标谷子空间。再看 `Valley-adapted subspace`，判断近简并目标子空间是否构成良好的双谷子空间。在旋转本征值表中，`topology_input_ready` 只表示这个高对称点旋转本征值适合作为后续基于对称性的拓扑分析输入；它不验证整个 moire 布里渊区上的谷分辨拓扑。旧字段 `topology_ready` 保留为 `topology_input_ready` 的兼容别名。

对于含自旋轨道耦合的 spinor 波函数，ValleyScope 会在平面波表示中施加 SU(2) 自旋旋转，但 VASP spinor 相位约定尚未经过基准验证。因此这些行会标记为 `spinor_rotation_applied=True`、`spinor_convention_verified=False` 和 `diagnostic_only=True`，不会标记为 `topology_input_ready`。

屏幕摘要示例：

```text
Input
-----
wavefunction_h5: ./wave.h5
operation structure: ./2dm-5370-7.34.vasp
operation-detection backend: spglib
target k-points: GammaM, KM, MM
target bands (VASP): 2195, 2196
qcut mode: relative_min_sector_distance
qcut value: 0.034 A^-1

Valley manifolds
----------------
label      centers
---------  ----------------
K_sector   top_K, bottom_K
Kp_sector  top_Kp, bottom_Kp

Valley projection summary
-------------------------
kpoint  band  W_val  P_v   W_overlap  W_res  status
------  ----  -----  ----  ---------  -----  ------------
GammaM  2195  0.98   0.99  0          0.02   valley-clean
```

### 结构文件怎么选

这些输入的用途不同：

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

`input.wavefunction_h5` 是抽取后的 moire 波函数 HDF5 文件，包含选定高对称点的平面波系数，以及投影流程使用的 moire 倒格矢。

`input.monolayer_poscars` 是单层结构，用于构造各层倒格矢，并解释 `layer_frac` 谷中心。设置了单层 POSCAR 后，通常不需要手写 `monolayer_lattices.*.reciprocal_cart`。

`symmetry.operations.structure_file` 是用于识别候选实空间对称操作的 moire 或双层 POSCAR/CONTCAR。如果缺少它或路径写错，`symmetry_report.json` 会显示对称操作识别被跳过。

单层 POSCAR 不能替代 moire POSCAR 做对称操作识别。moire POSCAR 也不应该被程序默认拿来当成单层倒格矢来源。

## 读取输出

分析结果写到 `output.directory`：

```text
valley_summary.txt
valley_summary.json
valley_weights.csv
valley_subspace.json
valley_basis_transform.h5
symmetry_report.json
rotation_eigenvalues.csv
diagnostics.h5
```

### `valley_weights.csv`

这个 CSV 每行对应一个原始 VASP 能带：

```text
kpoint, band_vasp, energy_eV, K_sector, Kp_sector, W_val, P_v, eta, W_overlap, W_res
```

扇区列名是 YAML 标签。`W_val`、`P_v` 和 `eta` 分别保存目标谷子空间权重、谷纯度和有符号谷极化。`W_overlap` 保存跨扇区投影窗口重叠权重。`W_res` 保存谷外剩余权重。

这个文件适合做第一眼检查。对近简并目标态，逐条原始能带结果是规范依赖的，不能作为最终物理解释。

### `valley_subspace.json`

这是近简并态的主要摘要文件。它记录投影谷算符、谷适配基诊断，以及目标子空间落在所选谷子空间中的程度。

### `valley_basis_transform.h5`

这里保存从原始 VASP 能带基到谷适配基的变换矩阵：

```text
/GammaM/transform
/GammaM/eta
/GammaM/s_matrix
/GammaM/v_matrix
/GammaM/band_indices_vasp
/GammaM/sectors
```

后续要在谷适配子空间中计算表示（representation）时，应使用这里的变换矩阵。

### `symmetry_report.json`

这里记录检测到的对称操作，包括操作类型、旋转矩阵、平移、是否是候选旋转、操作识别后端，以及小群与谷保持检查结果。

如果看到：

```json
"status": "skipped"
```

通常说明 `symmetry.operations.structure_file` 没有设置或路径错误。这个路径必须指向 moire 或双层 POSCAR/CONTCAR。

### `rotation_eigenvalues.csv`

只有操作同时通过 V1 的三个过滤条件时才会写入：

1. 是真旋转（proper rotation），阶数在 `[2, 3, 4, 6]`。
2. 属于目标高对称点的小群。
3. 保持目标谷扇区。

如果这个文件只有表头，不一定是错误。它往往表示当前高对称点和谷扇区下没有操作通过这些检查。

readiness 相关列采用保守定义：

```text
rotation_ready, topology_input_ready, topology_ready, spinor_rotation_applied,
spinor_convention_verified, diagnostic_only, D_valley_offdiag_norm
```

`rotation_ready` 检查表示矩阵是否没有缺失的平面波映射，并且幺正性偏差较小。`topology_input_ready` 进一步要求有效的双谷适配基、较小的单位根偏差，以及较小的双谷 `D_valley_offdiag_norm`。它不声明完整 valley Chern number。为了兼容旧结果，`topology_ready` 保存同一个值。`D_valley_offdiag_norm` 只是双谷谷适配诊断，不是通用多谷或多维不可约表示判据。

spinor 行在 VASP spinor 约定完成基准验证前都只作为诊断结果。

### `diagnostics.h5`

这里保存投影掩码（projector masks）与 q-cut 扫描数据：

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

调 `qcut_fraction`、检查投影窗口是否选中了预期动量，或调试旋转表示时，优先看这个文件。只有存在有效谷适配基时，才会写入 `D_valley`。

## V1 边界

当前 V1 不包含：

- 全 moire 布里渊区谷有效性验证（full-mBZ valley-goodness validation）；
- Berry 曲率；
- Wilson 回路（Wilson loops）；
- 自动完整整数陈数判断；
- 真正的层分辨投影（layer-resolved projection）；
- 自动单层 VBM/CBM 谷搜索；
- 对所有 WAVECAR variant 的完整支持。

V1 的优先级是把高对称点谷投影、谷适配子空间和对称性诊断做到可复现、可检查。
