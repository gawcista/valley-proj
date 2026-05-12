<a id="english"></a>
# ValleyScope

**Language:** English | [中文](#zh-cn)

ValleyScope is a VASP post-processing workflow for momentum-space valley projection and symmetry diagnostics in moire supercells. It is intended for situations where the relevant low-energy states inherit valley character from an underlying monolayer Brillouin zone, but the actual first-principles calculation is performed in a large moire supercell.

The code is not a black-box Chern-number calculator. V1 is an HSP-only diagnostic: it analyzes selected high-symmetry points, constructs monolayer-valley projectors in the plane-wave Hilbert space, fixes the valley gauge of near-degenerate target subspaces, and reports the symmetry data needed for rotation-eigenvalue diagnostics. A strict valley-resolved topology statement still requires later full-mBZ validation.

## Physical Motivation

In a moire calculation, the Bloch momentum belongs to the moire Brillouin zone. The valley label of interest, however, is usually inherited from the monolayer Brillouin zone. A state at a moire high-symmetry point is therefore not “a K-valley state” just because the moire momentum is at a moire $K_M$ point. ValleyScope treats valley as a monolayer-valley sector resolved through the plane-wave momenta contained in the moire wavefunction.

For twisted bilayers, top and bottom layer momenta associated with the same monolayer valley are allowed to hybridize. Such top/bottom hybridization does not by itself destroy the valley quantum number. The relevant question is whether the target state or target subspace remains confined to a chosen valley manifold, or whether there is appreciable intervalley mixing or weight outside the selected valley windows.

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
\left|
\mathbf q_\parallel-
\left(\mathbf Q_a+\mathbf G_{\rm mono}\right)
\right|,
```
```math
\Omega_a(q_{\rm cut})=
\left\{
\mathbf q:\ d_a(\mathbf q)<q_{\rm cut}
\right\}.
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
\left\{
\mathbf q:\ \mathbf q\in\Omega_i\cap\Omega_j
\ {\rm for\ some}\ i\ne j
\right\},
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
The target-valley-manifold weight is

```math
W_{\rm val}=\sum_i W_i .
```
It measures how much of the state lies in the user-defined target valley manifold. It depends on the valley centers and projection radius; it is not a topological invariant.

The valley purity is

```math
P_v=\frac{\max_i W_i}{W_{\rm val}},
\qquad W_{\rm val}>0 .
```
It measures whether the assigned valley weight is concentrated in one valley sector or distributed among several sectors.

For a two-valley manifold ordered as K-valley followed by K'-valley, the signed valley polarization is

```math
\eta=
\frac{W_K-W_{K'}}{W_K+W_{K'}}
=
\frac{W_K-W_{K'}}{W_{\rm val}} .
```
The out-of-valley residual weight is

```math
L_{\rm out}=1-W_{\rm val}-W_\times,
\qquad
W_\times=\langle\psi|P_\times|\psi\rangle .
```
In the CSV output, `leakage` stores $L_{\rm out}$. The column `ambiguous_weight` stores $W_\times$, the cross-sector projector-window overlap weight. A large overlap weight is a warning about the chosen windows or cutoff, not evidence for a new physical valley.

### Gauge Fixing in a Near-Degenerate Subspace

For an isolated nondegenerate band, the single-state quantities above can be read directly. For a near-degenerate target subspace, raw VASP eigenvectors are gauge-dependent: any unitary rotation inside the degenerate subspace gives an equally valid set of eigenvectors. Row-by-row projection of individual VASP bands is therefore not the main physical diagnostic.

V1 instead projects the target subspace and constructs the projected valley operator. For a two-valley manifold spanned by raw VASP states $\{|\psi_m\rangle\}$,

```math
S_{mn}=
\langle\psi_m|(P_K+P_{K'})|\psi_n\rangle,
\qquad
V_{mn}=
\langle\psi_m|(P_K-P_{K'})|\psi_n\rangle .
```
The matrix $S$ tests whether the target subspace lies in the chosen valley manifold. The matrix $V$ fixes a valley-adapted basis inside the target subspace. In practice, the `valley_subspace.json` and `valley_basis_transform.h5` files are the primary outputs for near-degenerate states.

### Symmetry Diagnostics

ValleyScope obtains candidate symmetry operations from `spglib` using the moire POSCAR/CONTCAR. A rotation eigenvalue is reported only after two checks:

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

  # Moire/bilayer POSCAR for spglib symmetry.
  # This is not the monolayer POSCAR.
  poscar: ./2dm-5370-7.34.vasp

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
  ambiguous_cross_sector: warn_exclude
  thresholds:
    W_val_min: 0.8
    P_v_clean: 0.95
    P_v_approx: 0.85

symmetry:
  source: spglib
  symprec: 1.0e-3
  symprec_scan: [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3]
  angle_tolerance: -1.0
  allowed_orders: [2, 3, 4, 6]
  proper_rotations_only: true
  little_group_check: true
  valley_preservation_check: true

output:
  directory: ./valley_analysis
  write_json: true
  write_csv: true
  write_hdf5_basis_transform: true
```

Run:

```bash
valleyscope analyze-hsp analyze.yaml
```

### POSCAR Files

These two inputs serve different purposes:

```yaml
input:
  poscar: ./2dm-5370-7.34.vasp
  monolayer_poscars:
    top: ./2dm-5370.vasp
    bottom: ./2dm-5370.vasp
```

`input.poscar` is the moire or bilayer POSCAR/CONTCAR. It is used by `spglib` to find symmetry operations. If it is missing or the path is wrong, `symmetry_report.json` will show that symmetry analysis was skipped.

`input.monolayer_poscars` are monolayer structures. They are used to build layer reciprocal lattices and interpret `layer_frac` valley centers. Once they are set, you usually do not need to write `monolayer_lattices.*.reciprocal_cart` manually.

A monolayer POSCAR cannot replace the moire POSCAR for symmetry. The moire POSCAR also should not be silently reused as a monolayer reciprocal lattice.

## Reading the Outputs

The analyzer writes results under `output.directory`:

```text
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
kpoint, band_vasp, energy_eV, K_sector, Kp_sector, W_val, P_v, eta, leakage, ambiguous_weight
```

The sector columns are YAML labels. `W_val`, `P_v`, and `eta` store the target-valley-manifold weight, valley purity, and signed valley polarization. `leakage` stores the out-of-valley residual weight. `ambiguous_weight` stores the cross-sector projector-window overlap weight.

Use this file for a first scan. For near-degenerate target states, the raw band rows are gauge-dependent and should not be the final interpretation.

### `valley_subspace.json`

This is the primary summary for near-degenerate states. It records the projected valley operator, the valley-adapted basis diagnostics, and the eigenvalues indicating how well the target subspace lies in the selected valley manifold.

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

This records the `spglib` operations obtained from `input.poscar`, including operation type, rotation matrix, translation, candidate rotation status, and little-group / valley-preservation diagnostics.

If the report says:

```json
"status": "skipped"
```

the usual cause is that `input.poscar` is missing or points to the wrong file. This path must be the moire or bilayer POSCAR.

### `rotation_eigenvalues.csv`

Rows are written only when an operation passes all V1 filters:

1. It is a proper rotation with order in `[2, 3, 4, 6]`.
2. It belongs to the little group of the target HSP.
3. It preserves the target valley sector.

An empty file with only a header is not automatically an error. It often means no operation passed these checks for the selected HSP and valley sector.

### `diagnostics.h5`

This file stores projector masks and q-cut scan data:

```text
/projectors/<kpoint>/sector_masks
/projectors/<kpoint>/center_masks
/projectors/<kpoint>/ambiguous_mask
/qcut_scan/<kpoint>/qcuts
/qcut_scan/<kpoint>/w_val
/qcut_scan/<kpoint>/purity
/qcut_scan/<kpoint>/eta
/qcut_scan/<kpoint>/ambiguous_weight
```

Use it when tuning `qcut_fraction` or checking whether the projector windows select the expected momenta.

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

ValleyScope 是一套面向 VASP moire 超胞波函数的 momentum-space valley projection 与 symmetry diagnostics 后处理流程。它适用于这样一类问题：低能态的 valley character 来自底层单层布里渊区，但实际第一性原理计算是在大型 moire 超胞中完成的。

它不是黑箱 Chern number 程序。V1 是 HSP-only diagnostic：它只分析用户选定的高对称点，在 plane-wave Hilbert space 中构造 monolayer-valley projectors，对近简并目标子空间进行 valley gauge fixing，并输出 rotation-eigenvalue diagnostics 所需的对称性信息。严格的 valley-resolved topology 判断仍然需要后续 full-mBZ validation。

## 物理动机

在 moire 计算中，Bloch momentum 属于 moire Brillouin zone。但我们关心的 valley label 通常继承自 monolayer Brillouin zone。因此，某个态位于 moire 高对称点并不意味着它就是某个 monolayer valley 态。ValleyScope 把 valley 定义为通过 moire 波函数中的 plane-wave momenta 解析出的 monolayer-valley sector。

对 twisted bilayer，同一个 monolayer valley 中的 top 与 bottom layer momenta 可以发生杂化。这种 top/bottom layer hybridization 本身不破坏 valley quantum number。真正需要诊断的是目标态或目标子空间是否仍然局限在选定的 valley manifold 中，还是出现了明显 intervalley mixing 或 out-of-valley residual weight。

V1 工作流在选定 HSP 上回答这个问题。它还会检查候选对称操作是否属于该 HSP 的 little group，以及是否 preserve target valley sector，然后才报告 rotation eigenvalue。此类本征值是有用的 symmetry diagnostic，但单独并不能证明完整整数 Chern number。

## 物理定义

### Momentum-Space Valley Projection

对 moire 超胞中动量为 $\mathbf k_M$ 的 VASP Bloch state，每个 plane-wave component 的物理动量为

```math
\mathbf q=\mathbf k_M+\mathbf G_M ,
```
其中 $\mathbf G_M$ 是 moire 超胞倒格矢。ValleyScope 只使用面内分量 $\mathbf q_\parallel$ 进行 valley assignment。

monolayer valley center $a$ 在单层倒空间坐标中给出，并通过 layer transform 映射到 moire frame。它对应的 projector window 由到单层倒格矢星的最小距离定义：

```math
d_a(\mathbf q)=
\min_{\mathbf G_{\rm mono}}
\left|
\mathbf q_\parallel-
\left(\mathbf Q_a+\mathbf G_{\rm mono}\right)
\right|,
```
```math
\Omega_a(q_{\rm cut})=
\left\{
\mathbf q:\ d_a(\mathbf q)<q_{\rm cut}
\right\}.
```
valley-sector projector 由一组 windows 的并集构造。以两 valley 的 twisted bilayer 为例，

```math
\Omega_K=
\Omega_{{\rm top},K}\cup\Omega_{{\rm bottom},K},
\qquad
\Omega_{K'}=
\Omega_{{\rm top},K'}\cup\Omega_{{\rm bottom},K'} .
```
YAML 中的 `K_sector` 这类名称只是用户定义的标签。物理上应理解为 K-valley sector 或 K'-valley sector，而不是代码内置的特殊物理对象。

### Weights 与 Intervalley-Mixing Diagnostics

设 $P_i$ 是第 $i$ 个 valley sector 的投影算符。默认策略下，如果某个 plane-wave component 同时落入多个 sector 的 projector windows，它会从所有 sector projectors 中移除，并放入 overlap projector $P_\times$：

```math
\Omega_\times=
\left\{
\mathbf q:\ \mathbf q\in\Omega_i\cap\Omega_j
\ {\rm for\ some}\ i\ne j
\right\},
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
对归一化态 $|\psi\rangle$，第 $i$ 个 sector 的 monolayer-valley-resolved weight 为

```math
W_i=\langle\psi|P_i|\psi\rangle .
```
target-valley-manifold weight 为

```math
W_{\rm val}=\sum_i W_i .
```
它衡量该态有多少权重落在用户定义的 target valley manifold 中。它依赖 valley centers 和 projection radius，不是拓扑不变量。

valley purity 定义为

```math
P_v=\frac{\max_i W_i}{W_{\rm val}},
\qquad W_{\rm val}>0 .
```
它衡量已经分配到 valley manifold 的权重是否集中在单一 valley sector 中。

对按 K-valley、K'-valley 排列的两 valley manifold，signed valley polarization 为

```math
\eta=
\frac{W_K-W_{K'}}{W_K+W_{K'}}
=
\frac{W_K-W_{K'}}{W_{\rm val}} .
```
out-of-valley residual weight 为

```math
L_{\rm out}=1-W_{\rm val}-W_\times,
\qquad
W_\times=\langle\psi|P_\times|\psi\rangle .
```
在 CSV 输出中，`leakage` 保存 $L_{\rm out}$。`ambiguous_weight` 保存 $W_\times$，即 cross-sector projector-window overlap weight。大的 overlap weight 通常说明 projector windows 或 cutoff 选择需要检查，而不是出现了新的物理 valley。

### 近简并子空间中的 Gauge Fixing

对孤立非简并 band，可以直接读取上述单态诊断。对近简并目标子空间，raw VASP eigenvectors 是 gauge-dependent 的：子空间内部任意酉变换都会给出同样合法的本征矢集合。因此，逐条 VASP band projection 不是主要物理诊断。

V1 会投影整个目标子空间并构造 projected valley operator。对由 VASP 原始态 $\{|\psi_m\rangle\}$ 张成的两 valley manifold，

```math
S_{mn}=
\langle\psi_m|(P_K+P_{K'})|\psi_n\rangle,
\qquad
V_{mn}=
\langle\psi_m|(P_K-P_{K'})|\psi_n\rangle .
```
矩阵 $S$ 检查目标子空间是否落在选定 valley manifold 中。矩阵 $V$ 在目标子空间内选出 valley-adapted basis。实际读取近简并态时，应主要看 `valley_subspace.json` 和 `valley_basis_transform.h5`。

### Symmetry Diagnostics

ValleyScope 使用 moire POSCAR/CONTCAR 调用 `spglib` 得到候选对称操作。只有通过两项检查后，程序才报告 rotation eigenvalue：

1. 该操作属于目标 HSP 的 little group。
2. 该操作 preserve target valley sector。

得到的矩阵是 valley-adapted subspace 中的 little-group representation。其本征值是 rotation eigenvalue diagnostic。它们可用于 symmetry-based topology 公式中的约束，但 V1 不自动推断完整整数 Chern number。

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

第二步，分析选定的 HSP 波函数：

```bash
valleyscope analyze-hsp analyze.yaml
```

主分析流程读取 HDF5，而不是反复直接读取完整 `WAVECAR`。这样可以把物理分析和大型二进制解析分开，也更方便检查具体抽取了哪些 k 点和 band。

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

抽取后，应检查 HDF5 中的 k 点、band 和数组形状。最重要的数据集是：

```text
/metadata/g_vector_order
/kpoints/N/name
/kpoints/N/frac
/kpoints/N/g_vectors_frac
/kpoints/N/coefficients       [nb, nspinor, nG]
/kpoints/N/energies_eV
/kpoints/N/band_indices_vasp
```

如果 extractor 报 G-vector 数量不匹配，不要强行继续。这通常表示当前版本还不支持该 WAVECAR variant 或 G-list convention。

### 分析 HSP 波函数

对 twisted bilayer，`analyze.yaml` 里最容易出错的是 layer transform。如果结构来自整数公度矩阵，优先使用 `supercell_matrix`，不要只根据转角手写 `rotation_deg`。实际 moire cell 可能带有整体 basis rotation，单独的 twist angle 往往不够。

以 `generate_hexagonal_210(9, 5, ...)` 这类结构为例：

```yaml
input:
  wavefunction_h5: ./wave.h5

  # Moire/bilayer POSCAR for spglib symmetry.
  # This is not the monolayer POSCAR.
  poscar: ./2dm-5370-7.34.vasp

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
  ambiguous_cross_sector: warn_exclude
  thresholds:
    W_val_min: 0.8
    P_v_clean: 0.95
    P_v_approx: 0.85

symmetry:
  source: spglib
  symprec: 1.0e-3
  symprec_scan: [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3]
  angle_tolerance: -1.0
  allowed_orders: [2, 3, 4, 6]
  proper_rotations_only: true
  little_group_check: true
  valley_preservation_check: true

output:
  directory: ./valley_analysis
  write_json: true
  write_csv: true
  write_hdf5_basis_transform: true
```

运行：

```bash
valleyscope analyze-hsp analyze.yaml
```

### POSCAR 文件怎么选

这两个输入的用途不同：

```yaml
input:
  poscar: ./2dm-5370-7.34.vasp
  monolayer_poscars:
    top: ./2dm-5370.vasp
    bottom: ./2dm-5370.vasp
```

`input.poscar` 是 moire 或 bilayer 的 POSCAR/CONTCAR，用于 `spglib` 识别 symmetry operations。如果缺少它或路径写错，`symmetry_report.json` 会显示 symmetry analysis 被跳过。

`input.monolayer_poscars` 是单层结构，用于构造各层 reciprocal lattice，并解释 `layer_frac` valley centers。设置了单层 POSCAR 后，通常不需要手写 `monolayer_lattices.*.reciprocal_cart`。

单层 POSCAR 不能替代 moire POSCAR 做 symmetry。moire POSCAR 也不应该被程序默认拿来当成单层 reciprocal lattice。

## 读取输出

分析结果写到 `output.directory`：

```text
valley_weights.csv
valley_subspace.json
valley_basis_transform.h5
symmetry_report.json
rotation_eigenvalues.csv
diagnostics.h5
```

### `valley_weights.csv`

这个 CSV 每行对应一个原始 VASP band：

```text
kpoint, band_vasp, energy_eV, K_sector, Kp_sector, W_val, P_v, eta, leakage, ambiguous_weight
```

sector 列名是 YAML 标签。`W_val`、`P_v` 和 `eta` 分别保存 target-valley-manifold weight、valley purity 和 signed valley polarization。`leakage` 保存 out-of-valley residual weight。`ambiguous_weight` 保存 cross-sector projector-window overlap weight。

这个文件适合做第一眼检查。对近简并目标态，逐条 raw band 结果是 gauge-dependent 的，不能作为最终物理解释。

### `valley_subspace.json`

这是近简并态的主要摘要文件。它记录 projected valley operator、valley-adapted basis diagnostics，以及目标子空间落在所选 valley manifold 中的程度。

### `valley_basis_transform.h5`

这里保存从原始 VASP band basis 到 valley-adapted basis 的变换矩阵：

```text
/GammaM/transform
/GammaM/eta
/GammaM/s_matrix
/GammaM/v_matrix
/GammaM/band_indices_vasp
/GammaM/sectors
```

后续要在 valley-adapted subspace 中计算 representation 时，应使用这里的 transform。

### `symmetry_report.json`

这里记录从 `input.poscar` 得到的 `spglib` operations，包括操作类型、rotation matrix、translation、是否是候选 rotation，以及 little-group / valley-preservation diagnostics。

如果看到：

```json
"status": "skipped"
```

通常说明 `input.poscar` 没有设置或路径错误。这个路径必须指向 moire 或 bilayer POSCAR。

### `rotation_eigenvalues.csv`

只有操作同时通过 V1 的三个过滤条件时才会写入：

1. 是 proper rotation，阶数在 `[2, 3, 4, 6]`。
2. 属于目标 HSP 的 little group。
3. preserve target valley sector。

如果这个文件只有表头，不一定是错误。它往往表示当前 HSP 和 valley sector 下没有操作通过这些检查。

### `diagnostics.h5`

这里保存 projector masks 与 q-cut scan data：

```text
/projectors/<kpoint>/sector_masks
/projectors/<kpoint>/center_masks
/projectors/<kpoint>/ambiguous_mask
/qcut_scan/<kpoint>/qcuts
/qcut_scan/<kpoint>/w_val
/qcut_scan/<kpoint>/purity
/qcut_scan/<kpoint>/eta
/qcut_scan/<kpoint>/ambiguous_weight
```

调 `qcut_fraction` 或检查 projector windows 是否选中了预期 momenta 时，优先看这个文件。

## V1 边界

当前 V1 不包含：

- full-mBZ valley-goodness validation；
- Berry curvature；
- Wilson loops；
- 自动完整整数 Chern number 判断；
- 真正 layer-resolved projection；
- 自动 monolayer VBM/CBM valley 搜索；
- 对所有 WAVECAR variant 的完整支持。

V1 的优先级是把 HSP valley projection、valley-adapted subspaces 和 symmetry diagnostics 做到可复现、可检查。
