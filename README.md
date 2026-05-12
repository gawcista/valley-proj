<a id="english"></a>
# ValleyScope

**Language:** English | [中文](#zh-cn)

ValleyScope is a VASP post-processing tool for valley projection and symmetry diagnostics in moire supercells. The current version focuses on one practical workflow: take selected high-symmetry-point wavefunction coefficients, project them onto user-defined monolayer valley sectors, build a valley-adapted basis for near-degenerate bands, and write the diagnostics needed before any rotation-eigenvalue analysis.

It is not a black-box Chern-number code. V1 deliberately writes intermediate quantities that can be checked: valley weights, leakage, cross-sector overlap, valley-adapted basis transforms, spglib operations, little-group checks, and valley-preservation checks.

## Install

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

## Start Here

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

## Extract From WAVECAR

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

## Analyze HSP Wavefunctions

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

## POSCAR Files

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

## Output Files

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

One row per original VASP band:

```text
kpoint, band_vasp, energy_eV, K_sector, Kp_sector, W_val, P_v, eta, leakage, ambiguous_weight
```

This is useful for a quick scan. For near-degenerate target bands, do not interpret a single original VASP eigenvector too literally: the vectors can be mixed by an arbitrary unitary rotation inside the degenerate subspace.

### `valley_subspace.json`

This is usually the most useful physics summary. For each k point, it includes the original band weights and the valley-adapted subspace. In a two-valley case:

```text
S = P_K + P_Kp
V = P_K - P_Kp
```

`s_eigenvalues` measure how well the target subspace lies inside the chosen valley manifold. `eta` gives the signed valley polarization in the valley-adapted basis.

For example:

```text
GammaM eta = +0.9999, -0.9999
GammaM S   = 0.9967, 0.9967
```

This means the two-dimensional target subspace at `GammaM` can be cleanly separated into K-like and Kp-like valley-adapted states.

### `valley_basis_transform.h5`

This stores the transformation from original VASP band basis to valley-adapted basis:

```text
/GammaM/transform
/GammaM/eta
/GammaM/s_matrix
/GammaM/v_matrix
/GammaM/band_indices_vasp
/GammaM/sectors
```

Use this transform for later valley-adapted representation calculations. Do not compute rotation eigenvalues by assigning physical meaning to the raw VASP eigenvectors in a near-degenerate subspace.

### `symmetry_report.json`

This records the `spglib` symmetry operations from `input.poscar`, including operation type, rotation matrix, translation, candidate rotation status, and diagnostic information for little-group and valley-preservation checks.

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

An empty file with only a header is not automatically an error. It often means no operation passed these checks for the selected k point and valley sector.

### `diagnostics.h5`

Low-level projector and scan data:

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

Use this file when tuning `qcut_fraction` or checking whether a projector mask is selecting the expected momentum components.

## Physical Picture

For each moire Bloch state, ValleyScope looks at the actual plane-wave momentum:

```text
q = k_moire + G_moire
```

Only the in-plane part is used for valley projection. The code folds `q_parallel` with the corresponding monolayer reciprocal lattice and checks whether it lies inside a window around a valley center.

A valley sector can contain multiple centers. For twisted bilayer MoTe2, a typical definition is:

```text
K_sector  = [top_K, bottom_K]
Kp_sector = [top_Kp, bottom_Kp]
```

This is intentional: top/bottom hybridization inside the same monolayer valley does not destroy the valley quantum number. A layer is not a valley, and a moire-BZ K point is not automatically the monolayer K valley.

## Common Questions

### What is `ambiguous_weight`?

If a plane-wave component falls into windows belonging to different valley sectors, it is a cross-sector overlap. With the default `ambiguous_cross_sector: warn_exclude`, that component is excluded from all sector weights and written to `ambiguous_weight`.

Small values are usually a cutoff-boundary effect. Large values mean you should check the valley centers, layer transforms, and `qcut_fraction`.

### Why are the raw bands mixed but the subspace is clean?

Near-degenerate VASP eigenvectors have gauge freedom. A pair of raw eigenvectors can be arbitrary mixtures of K and Kp. In that case, read the `valley_adapted_subspace` section in `valley_subspace.json`, especially `eta` and `s_eigenvalues`.

### When should I write `reciprocal_cart` manually?

Usually never. Prefer:

```yaml
input:
  monolayer_poscars:
    top: ./monolayer.vasp
    bottom: ./monolayer.vasp
```

Write `monolayer_lattices.*.reciprocal_cart` only when the monolayer POSCAR is unavailable or when you intentionally want to override the lattice read from the POSCAR.

## V1 Scope

Current V1 does not include:

- full mBZ mesh valley-goodness diagnostics;
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

ValleyScope 是一个面向 VASP moire 超胞波函数的 valley projection 与 symmetry diagnostics 工具。当前版本主要解决一个实际问题：从已选出的高对称点波函数系数出发，投影到用户定义的 monolayer valley sectors，在近简并目标能带中构造 valley-adapted basis，并输出 rotation-eigenvalue 分析前必须检查的诊断量。

它不是黑箱 Chern number 程序。V1 会把中间量明确写出来，包括 valley weights、leakage、cross-sector overlap、valley-adapted basis transform、spglib operations、little-group check 和 valley-preservation check。

## 安装

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

## 从这里开始

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

## 从 WAVECAR 抽取

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

## 分析 HSP 波函数

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

## POSCAR 文件怎么选

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

## 输出文件

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

每一行对应一个原始 VASP band：

```text
kpoint, band_vasp, energy_eV, K_sector, Kp_sector, W_val, P_v, eta, leakage, ambiguous_weight
```

这个文件适合快速浏览。对近简并目标 bands，不要过度解释单个原始 VASP 本征矢，因为简并子空间内的本征矢可以被任意酉变换混合。

### `valley_subspace.json`

这通常是最重要的物理摘要。每个 k point 下包含原始 band weights 和 valley-adapted subspace。两 valley 情形下程序构造：

```text
S = P_K + P_Kp
V = P_K - P_Kp
```

`s_eigenvalues` 衡量目标子空间有多少落在用户定义的 valley manifold 中。`eta` 是 valley-adapted basis 中的 signed valley polarization。

例如：

```text
GammaM eta = +0.9999, -0.9999
GammaM S   = 0.9967, 0.9967
```

这表示 `GammaM` 的二维目标子空间可以很干净地分成 K-like 和 Kp-like valley-adapted states。

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

后续要在 valley-adapted basis 中计算 representation 时，应使用这里的 `transform`。不要在近简并子空间里直接把原始 VASP eigenvectors 当作有固定物理 valley character 的态。

### `symmetry_report.json`

这里记录从 `input.poscar` 得到的 `spglib` symmetry operations，包括操作类型、rotation matrix、translation、是否是候选 rotation，以及 little-group 和 valley-preservation 的诊断信息。

如果看到：

```json
"status": "skipped"
```

通常说明 `input.poscar` 没有设置或路径错误。这个路径必须指向 moire 或 bilayer POSCAR。

### `rotation_eigenvalues.csv`

只有操作同时通过 V1 的三个过滤条件时才会写入：

1. 是 proper rotation，阶数在 `[2, 3, 4, 6]`。
2. 属于目标 HSP 的 little group。
3. preserve 目标 valley sector。

如果这个文件只有表头，不一定是错误。它往往表示当前 k 点和 valley sector 下没有操作通过这些检查。

### `diagnostics.h5`

这里保存低层 projector 和扫描数据：

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

调 `qcut_fraction` 或检查 projector mask 是否合理时，优先看这个文件。

## 物理图像

对每个 moire Bloch state，ValleyScope 使用实际 plane-wave momentum：

```text
q = k_moire + G_moire
```

valley projection 只看面内分量。程序把 `q_parallel` 按对应层的 monolayer reciprocal lattice 折回，然后判断它是否落在某个 valley center 附近的 window 中。

一个 valley sector 可以包含多个 centers。对 twisted bilayer MoTe2，常见定义是：

```text
K_sector  = [top_K, bottom_K]
Kp_sector = [top_Kp, bottom_Kp]
```

这是有意这样定义的：同一个 monolayer valley 内的 top/bottom hybridization 不破坏 valley quantum number。layer 不是 valley，moire BZ 的 K 点也不自动等于 monolayer K valley。

## 常见问题

### `ambiguous_weight` 是什么？

如果某个 plane-wave component 同时落入不同 valley sectors 的窗口，它就是 cross-sector overlap。默认 `ambiguous_cross_sector: warn_exclude` 会把这个 component 从所有 sector weights 里排除，并把权重写入 `ambiguous_weight`。

小的 `ambiguous_weight` 往往只是 cutoff 边界效应。如果它很大，应检查 valley centers、layer transforms 和 `qcut_fraction`。

### 为什么原始 bands 混合，但子空间很干净？

近简并 VASP eigenvectors 有 gauge freedom。两个原始本征矢可以是 K 和 Kp 的任意线性组合。这时应看 `valley_subspace.json` 中的 `valley_adapted_subspace`，尤其是 `eta` 和 `s_eigenvalues`。

### 什么时候需要手写 `reciprocal_cart`？

通常不需要。优先提供：

```yaml
input:
  monolayer_poscars:
    top: ./monolayer.vasp
    bottom: ./monolayer.vasp
```

只有在没有单层 POSCAR，或者你想有意覆盖 POSCAR 中读取出的 lattice 时，才写 `monolayer_lattices.*.reciprocal_cart`。

## V1 边界

当前 V1 不包含：

- full mBZ mesh valley-goodness diagnostics；
- Berry curvature；
- Wilson loops；
- 自动完整整数 Chern number 判断；
- 真正 layer-resolved projection；
- 自动 monolayer VBM/CBM valley 搜索；
- 对所有 WAVECAR variant 的完整支持。

V1 的优先级是把 HSP valley projection、valley-adapted subspaces 和 symmetry diagnostics 做到可复现、可检查。
