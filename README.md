# ValleyScope

ValleyScope 是一个面向 VASP moire 超胞波函数的 valley projection 与 symmetry diagnostics 工具。它现在主要做一件事：从已经选出的 HSP wavefunction coefficients 出发，判断目标能带或近简并子空间是否可以稳定地分成 monolayer valley sectors，并输出后续 rotation-eigenvalue 分析需要的中间量。

它不是黑箱 Chern number 程序。V1 不做 Berry curvature、Wilson loop、完整 mBZ mesh diagnostic，也不会自动声称整数 Chern number。它的价值在于把每一步可检查的量写出来：valley weights、leakage、cross-sector overlap、valley-adapted basis、spglib symmetry operations、little-group 与 valley-preservation checks。

## 安装

```bash
git clone https://github.com/gawcista/valley-proj.git
cd valley-proj
pip install -e .
```

开发环境下可直接运行：

```bash
python -m valleyscope.cli --help
```

安装后命令行为：

```bash
valleyscope --help
```

## 推荐工作流

### 1. 从 WAVECAR 抽取 HDF5

分析核心读取 HDF5，不直接在主流程里反复读完整 `WAVECAR`。先写一个 `extract.yaml`：

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

  # Collinear ISPIN=2 时一次抽一个 spin channel。
  # SOC/noncollinear WAVECAR 会从 coefficient record length 识别 spinor。
  spin_index: 1

output:
  wavefunction_h5: ./wave.h5
```

运行：

```bash
valleyscope extract-wavecar extract.yaml
```

成功后检查 `wave.h5` 中应有：

```text
/metadata/g_vector_order = vasp_z_y_x
/kpoints/N/coefficients shape = [nb, nspinor, nG]
/kpoints/N/band_indices_vasp = 你在 YAML 里指定的 bands
```

如果 extractor 报 G-vector 数量不匹配，不要强行继续。那通常表示 WAVECAR variant 或 G-list convention 还没有被当前版本支持。

### 2. 准备 analyze YAML

对 twisted bilayer，最容易出错的是 layer transform。不要只凭“转角是 7.34 度”手写一个 `rotation_deg`，因为实际 moire cell 可能还包含整体 basis rotation。若结构由 `twist_generator.generate_hexagonal_210(m, n, ...)` 生成，推荐直接使用 generator 的整数矩阵。

例如 `generate_hexagonal_210(9, 5, ...)` 对应：

```yaml
input:
  wavefunction_h5: ./wave.h5

  # moire/bilayer POSCAR，用于 spglib symmetry。
  # 注意：这不是单层 POSCAR。
  poscar: ./2dm-5370-7.34.vasp

  # 单层 POSCAR，用于 monolayer reciprocal lattice / valley centers。
  # 设置了这个以后通常不需要手写 reciprocal_cart。
  monolayer_poscars:
    top: ./2dm-5370.vasp
    bottom: ./2dm-5370.vasp

layer_transforms:
  top:
    # row-vector convention: M_moire = P^T A_layer
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

源码树中也可以用：

```bash
python -m valleyscope.cli analyze-hsp analyze.yaml
```

## `poscar` 和 `monolayer_poscars` 的区别

这两个字段经常被混淆：

```yaml
input:
  poscar: ./2dm-5370-7.34.vasp
  monolayer_poscars:
    top: ./2dm-5370.vasp
    bottom: ./2dm-5370.vasp
```

`input.poscar` 是 moire/bilayer 的 POSCAR 或 CONTCAR。它用于 `spglib` 识别超胞的 symmetry operations。若缺少它，`symmetry_report.json` 会显示 symmetry analysis 被跳过。

`input.monolayer_poscars` 是单层 POSCAR。它用于计算单层 reciprocal lattice，并定义 monolayer valley centers。设置了它以后，一般不需要在 YAML 中手写 `monolayer_lattices.*.reciprocal_cart`。

单层 POSCAR 不能替代 moire POSCAR 做 symmetry；moire POSCAR 也不应被默认拿来当单层 reciprocal lattice。

## 如何读输出

分析结果写入 `output.directory`。典型文件如下：

```text
valley_weights.csv
valley_subspace.json
valley_basis_transform.h5
symmetry_report.json
rotation_eigenvalues.csv
diagnostics.h5
```

### `valley_weights.csv`

逐个原始 VASP band 输出 valley weights：

```text
kpoint, band_vasp, energy_eV, K_sector, Kp_sector, W_val, P_v, eta, leakage, ambiguous_weight
```

这个文件适合快速扫一眼，但不要在近简并情况下只看单条 band。近简并子空间内，VASP 输出的两个本征矢可以被任意酉变换混合；这时单条 band 的 `eta` 可能看起来不纯，但子空间本身仍然是很好的 valley 子空间。

### `valley_subspace.json`

这是最重要的物理摘要。每个 k point 下包含：

- `weights`: 原始 VASP band 的 valley weights。
- `valley_adapted_subspace`: 对近简并目标 band 子空间构造的 valley-adapted basis。
- `eta`: valley-adapted basis 的 signed valley polarization。两 valley 情形下应接近 `+1/-1`。
- `s_eigenvalues`: 目标子空间落在用户定义 valley manifold 内的权重。越接近 1，说明该目标子空间越被 valley windows 完整覆盖。

例如：

```text
GammaM eta = +0.9999, -0.9999
GammaM S   = 0.9967, 0.9967
```

这表示 GammaM 的两维近简并子空间可以非常稳定地分解成 K/K' valley-adapted states。

### `valley_basis_transform.h5`

保存从原始 VASP band basis 到 valley-adapted basis 的变换矩阵。每个 k point 通常有：

```text
/GammaM/transform
/GammaM/eta
/GammaM/s_matrix
/GammaM/v_matrix
/GammaM/band_indices_vasp
/GammaM/sectors
```

后续如果要在 valley-adapted basis 中计算 rotation representation，应使用这里的 `transform`，而不是直接解释原始 VASP 本征矢。

### `symmetry_report.json`

`spglib` 从 `input.poscar` 识别出的 symmetry operations。这里会记录：

- space group。
- 每个操作的 fractional rotation/translation。
- `kind`，例如 `C3`、`C2`、mirror、inversion。
- 是否是 V1 rotation workflow 的候选操作。
- little-group 与 valley-preservation 相关信息。

如果看到：

```json
"status": "skipped"
```

通常说明 `input.poscar` 没有设置，或路径不存在。注意这里需要 moire/bilayer POSCAR。

### `rotation_eigenvalues.csv`

只有同时通过 little-group check 和 valley-preservation check 的操作才会写入 rotation eigenvalues。这个文件只有表头并不一定是 bug；它可能表示当前 HSP / symmetry operation / valley sector 组合没有通过 V1 的 single-valley rotation workflow。

### `diagnostics.h5`

低层调试数据，主要用于检查 projector 是否合理：

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

如果你在调 `qcut_fraction`，这个文件比 CSV 更方便。

## 物理定义简述

对 moire Bloch state，程序使用每个 plane-wave component 的实际动量：

```text
q = k_moire + G_moire
```

然后只看面内分量 `q_parallel`，把它按对应层的 monolayer reciprocal lattice 折回，判断是否落在某个 valley center 周围的 window 中。

一个 valley sector 可以包含多个 centers。对 twisted bilayer MoTe2，通常是：

```text
K_sector  = [top_K, bottom_K]
Kp_sector = [top_Kp, bottom_Kp]
```

这反映的是：同一个 valley 的 top/bottom hybridization 不破坏 valley quantum number。不要把 layer 当成 valley，也不要把 moire BZ 的 K 点直接等同于 monolayer K valley。

对两 valley sector，程序在目标子空间中构造：

```text
S = P_K + P_Kp
V = P_K - P_Kp
```

`S` 的本征值告诉你目标子空间是否被 valley manifold 覆盖；`V/S` 给出 valley polarization `eta`。

## 常见问题

### 为什么有 `ambiguous_weight`？

如果一个 plane-wave component 同时落入不同 valley sectors 的 window，它会被标记为 cross-sector overlap。默认策略 `warn_exclude` 会把它从所有 sector weights 中排除，并把权重写入 `ambiguous_weight`。这可以避免同一个 component 被重复计数而虚假提高 valley purity。

少量 `ambiguous_weight` 通常只是 cutoff 边界效应。若它很大，应检查 valley centers、layer transforms 和 `qcut_fraction`。

### 为什么原始 band 的 valley 不纯，但子空间很纯？

近简并时，VASP 输出的单个本征矢有 gauge 自由度。它可以是 K/K' 的任意线性组合。此时应该读 `valley_subspace.json` 中 `valley_adapted_subspace` 的 `eta` 和 `s_eigenvalues`，而不是只看 `valley_weights.csv` 中单行的 `eta`。

### `rotation_eigenvalues.csv` 为空是不是错误？

不一定。V1 只在操作同时满足以下条件时输出 single-valley rotation eigenvalue：

1. 是 proper rotation，阶数在 `[2, 3, 4, 6]`。
2. 属于该 HSP 的 little group。
3. preserve 目标 valley sector。

任一条件不满足，操作会留在 `symmetry_report.json` 中作为诊断，但不会进入 `rotation_eigenvalues.csv`。

### 什么时候需要手写 `reciprocal_cart`？

一般不需要。优先提供：

```yaml
input:
  monolayer_poscars:
    top: ./monolayer.vasp
    bottom: ./monolayer.vasp
```

只有当你没有单层 POSCAR，或想覆盖程序从 POSCAR 读出的 lattice 时，才使用 `monolayer_lattices.*.reciprocal_cart`。

## V1 边界

当前版本不包含：

- full mBZ mesh valley-goodness diagnostic。
- Berry curvature。
- Wilson loop。
- 自动完整整数 Chern number 判断。
- 真正 layer-resolved projection。
- 自动 monolayer VBM/CBM valley 搜索。
- 所有 WAVECAR variant 的完整支持。

这些功能可以在后续版本中加入，但 V1 的优先级是把 HSP valley projection、简并子空间 valley basis 和 symmetry diagnostics 做成可检查、可复现的 workflow。
