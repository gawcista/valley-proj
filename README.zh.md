# ValleyScope 中文说明

**语言:** 中文 | [English](README.md)

ValleyScope 是一套面向 VASP moire 超胞波函数的动量空间谷投影（momentum-space valley projection）与对称性分析（symmetry analysis）后处理流程。它适用于低能态的谷特征（valley character）来自底层单层布里渊区，但实际第一性原理计算在大型 moire 超胞中完成的情况。

它不是黑箱陈数（Chern number）程序。V1 是只针对高对称点的诊断（HSP-only diagnostic）：分析选定高对称点，在平面波希尔伯特空间中构造单层谷投影算符，对近简并目标子空间进行谷规范固定，并输出谷保持对称操作的本征值诊断。严格的谷分辨拓扑判断需要后续全 mBZ 验证。

## 物理动机

在 moire 计算中，布洛赫动量属于 moire 布里渊区。但谷标记通常继承自单层布里渊区。因此，某个态位于 moire 高对称点并不意味着它就是某个单层谷态。ValleyScope 把谷定义为通过 moire 波函数中的平面波动量解析出的单层谷扇区。

对转角双层体系，同一个单层谷中的上下层动量可以杂化。这种层间杂化本身不破坏谷量子数。真正需要诊断的是目标态是否仍然局限在选定的谷子空间中，还是出现了谷间混合或谷外剩余权重。

V1 工作流在选定高对称点上回答这个问题。它还会检查候选对称操作是否属于该高对称点的小群（little group），以及是否保持目标谷扇区，然后才报告对称本征值。

## 物理定义

### 动量空间谷投影

对 moire 超胞中动量为 $\mathbf k_M$ 的 VASP 布洛赫态，每个平面波分量的物理动量为

```math
\mathbf q=\mathbf k_M+\mathbf G_M ,
```
其中 $\mathbf G_M$ 是 moire 超胞倒格矢。ValleyScope 只使用面内分量 $\mathbf q_\parallel$ 进行谷归属。

单层谷中心 $a$ 在单层倒空间坐标中给出，通过层变换映射到 moire 坐标系。其投影窗口由到单层倒格矢星的最小距离定义：

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
谷扇区投影算符由一组窗口的并集构造。以含两个谷的转角双层为例，

```math
\Omega_K=
\Omega_{{\rm top},K}\cup\Omega_{{\rm bottom},K},
\qquad
\Omega_{K'}=
\Omega_{{\rm top},K'}\cup\Omega_{{\rm bottom},K'} .
```
YAML 中的 `K_sector` 等名称只是用户定义的标签。

### 权重与谷间混合诊断

设 $P_i$ 是第 $i$ 个谷扇区的投影算符。默认策略下，落入多个扇区投影窗口的平面波分量会从所有谷扇区中移除，放入重叠投影算符 $P_\times$：

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
对归一化态 $|\psi\rangle$，扇区权重为

```math
W_i=\langle\psi|P_i|\psi\rangle .
```
目标谷子空间权重为

```math
W_{\rm val}=\sum_i W_i .
```
谷纯度为

```math
P_v=\frac{\max_i W_i}{W_{\rm val}},\qquad W_{\rm val}>0 .
```
对于两扇区情形，`P_v` 与 `|eta|` 存在冗余关系：`P_v = (1 + |eta|) / 2`。ValleyScope 在双扇区诊断中使用 `|eta|` 作为谷集中度分数；三扇区及以上 `|eta|` 无定义，直接使用 `P_v`。

谷集中度分数分为三档：
- **clean**（`raw_valley_clean` / `valley_separable_subspace`）：集中度高于 clean 阈值（默认 `P_v=0.95`，即 `|eta|=0.90`）。适合作为谷分辨对称性诊断的输入。
- **approximate**（`raw_valley_approx` / `valley_approximately_separable_subspace`）：集中度介于 approx 和 clean 阈值之间（默认 `P_v=0.85~0.95`，`|eta|=0.70~0.90`）。可谨慎使用；此范围内的对称本征值仅供诊断参考。
- **mixed**（`raw_valley_mixed` / `valley_mixed_subspace`）：集中度低于 approx 阈值。谷扇区未有效分离，单谷诊断不可靠。

对两谷子空间，有符号谷极化为

```math
\eta=
\frac{W_K-W_{K'}}{W_K+W_{K'}}
=
\frac{W_K-W_{K'}}{W_{\rm val}} .
```
投影窗口重叠权重为

```math
W_{\rm overlap}=\langle\psi|P_\times|\psi\rangle .
```
谷外剩余权重为

```math
W_{\rm res}=1-W_{\rm val}-W_{\rm overlap}.
```
V1 报告分解 $W_{\rm val}+W_{\rm overlap}+W_{\rm res}=1$。在 CSV/JSON 输出中，`W_overlap` 保存 $W_{\rm overlap}$，`W_res` 保存谷外剩余权重。

### 近简并子空间中的规范固定

对孤立非简并能带，可以直接读取上述单态诊断。对近简并目标子空间，VASP 原始本征矢是规范依赖的：子空间内部任意酉变换都合法。

V1 投影整个目标子空间并构造投影谷算符。对两谷子空间，

```math
S_{mn}=
\langle\psi_m|(P_K+P_{K'})|\psi_n\rangle,
\qquad
V_{mn}=
\langle\psi_m|(P_K-P_{K'})|\psi_n\rangle .
```
矩阵 $S$ 检查目标子空间是否落在选定谷子空间中。矩阵 $V$ 在目标子空间内选出谷适配基。实际应主要看 `valley_subspace.json` 和 `valley_basis_transform.h5`。

### 对称性诊断

ValleyScope 用 spglib 从 moire/bilayer 结构文件中自动识别对称操作，支持 `symprec` 和 `symprec_scan`。对称本征值仅在两项检查通过后才报告：

1. 该操作属于目标高对称点的小群。
2. 该操作保持目标谷扇区。

得到的矩阵是谷适配子空间中的小群表示。其本征值是对称性诊断，可用于基于对称性的拓扑公式约束，但 V1 不自动推断完整整数陈数。

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

### 快速开始

推荐流程只有两步。

第一步，从 `WAVECAR` 抽取 HDF5：

```bash
valleyscope extract-wavecar extract.yaml
```

第二步，分析高对称点波函数：

```bash
valleyscope analyze-hsp analyze.yaml
```

主分析流程读取 HDF5，不反复读完整 `WAVECAR`。

### 最小可运行配置

一个只含必要字段的 `analyze.yaml`：

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

其余字段使用默认值（`qcut_mode: moire_shell`, `qcut_shell: 3`, symmetry detection 跳过等）。
当前分析配置使用 `analysis.iband` 表示 VASP 能带编号，使用
`valley_manifolds` 表示用户定义的单层谷子空间。旧输入字段
`analysis.target_bands_vasp` 和 `valley_sectors` 已从公开 schema 中移除。

### 从 WAVECAR 抽取

`extract.yaml`：

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

  # 共线 ISPIN=2 时每次抽取一个自旋通道
  # SOC/非共线 WAVECAR 由抽取器自动检测 spinor
  spin_index: 1

output:
  wavefunction_h5: ./wave.h5
```

```bash
valleyscope extract-wavecar extract.yaml
```

检查 HDF5 内容：

```text
/metadata/g_vector_order
/kpoints/N/name
/kpoints/N/frac
/kpoints/N/g_vectors_frac
/kpoints/N/coefficients       [nb, nspinor, nG]
/kpoints/N/energies_eV
/kpoints/N/band_indices_vasp
```

如果抽取器报 G-vector 数量不匹配，说明当前 WAVECAR 变体或 G-list 约定尚不支持。

### 分析高对称点波函数

对转角双层，层变换是最容易出错的地方。结构来自整数公度矩阵时，优先用 `supercell_matrix`。

完整配置示例（`analyze.yaml`）：

```yaml
input:
  wavefunction_h5: ./wave.h5

  # 单层 POSCAR 用于构造层倒格矢和解释 layer_frac 谷中心
  monolayer_poscars:
    top: ./2dm-5370.vasp
    bottom: ./2dm-5370.vasp

layer_transforms:
  top:
    # Row-vector convention: M_moire = P^T A_layer
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

```bash
valleyscope analyze-hsp analyze.yaml
```

### 如何阅读屏幕输出

`analyze-hsp` 的终端摘要按以下部分组织：

```text
Input
Valley manifolds
Valley projection summary
Two-valley subspace
Symmetry analysis
Symmetry eigenvalues
Warnings
Output files
```

先看 `Valley projection summary`，它只给出逐条原始 VASP 能带的投影量：

```text
W_val:      谷子空间权重
P_v:        谷纯度
W_overlap: 投影窗口重叠权重
W_res:      剩余权重
```

屏幕上的 `status` 只保留三类：`clean`、`approx`、`mixed`。

再看 `Two-valley subspace`。近简并子空间中的原始 VASP 本征矢依赖规范选择，逐条能带投影不是最终诊断。ValleyScope 在目标子空间中构造

```math
S=P_K+P_{K'}, \qquad V=P_K-P_{K'} .
```

`S` 判断整个目标子空间是否主要位于所选双谷子空间中；`V` 在该子空间内固定谷适配基。这里输出的 `S_min`、`S_max`、`P_v_min` 和 `eta_adapted` 是子空间诊断，不是 raw-band 表的重复。

`Symmetry analysis` 先列出空间群和检测到的对称操作，再按高对称点列出 little-group 操作和 valley-preserving 操作。`Symmetry eigenvalues` 只列出实际构造了表示矩阵并求得本征值的操作。`topology_input_ready` 只表示该高对称点对称本征值可作为后续基于对称性的拓扑分析输入，不验证整个 mBZ 的谷分辨拓扑。

对于 SOC spinor 波函数，ValleyScope 在平面波表示中施加 SU(2) 自旋旋转。默认 VASP 自旋约定未验证时标记为 `diagnostic_only=True`。完成基准测试后可设置 `spinor.convention_verified: true`。

屏幕摘要示例：

```text
Input
-----
wavefunction_h5: ./wave.h5
operation structure: ./2dm-5370-7.34.vasp
operation-detection backend: spglib
spinor convention: vasp_up_down_saxis_z (verified=True, benchmark=tMoTe2_VBM_C3_literature)
target k-points: GammaM, KM, MM
iband (VASP): 2195, 2196
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
W_val:      谷子空间权重
P_v:        谷纯度
W_overlap: 投影窗口重叠权重
W_res:      剩余权重
kpoint  band  W_val  P_v   W_overlap  W_res  status
------  ----  -----  ----  ---------  -----  ------
GammaM  2195  0.98   0.99  0          0.02   clean
```

### 结构文件

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

- `input.wavefunction_h5`：抽取后的 moire 波函数 HDF5，含高对称点平面波系数和 moire 倒格矢。
- `input.monolayer_poscars`：单层结构，用于构造各层倒格矢和解释 `layer_frac` 谷中心。
- `symmetry.operations.structure_file`：用于识别对称操作的 moire/bilayer POSCAR/CONTCAR。缺失时 symmetry detection 被跳过。

单层 POSCAR 不能替代 moire POSCAR 做对称操作识别。

`symmetry.filters.rotation_order` 选择用于对称本征值提取的 proper-rotation 阶数：
- `auto`：从空间群推断（如 `P321`/`P312` → `C3`，`P422` → `C4`）
- 整数 `n`：只分析 `C_n`
- `None`/`none`：跳过对称本征值提取

默认只计算每个循环旋转子群的一个生成元。

## 读取输出

分析结果写到 `output.directory`：

```text
valley_summary.txt          ← 最先看
valley_summary.json
valley_weights.csv          ← 第一眼检查
valley_subspace.json        ← 双谷子空间数据
symmetry_eigenvalues.csv    ← 对称本征值
symmetry_report.json        ← 对称性分析
valley_basis_transform.h5   ← 谷适配基变换
diagnostics.h5              ← 调试用
```

### valley_weights.csv

每行对应一个原始 VASP 能带：

```text
kpoint, band_vasp, energy_eV, K_sector, Kp_sector, W_val, P_v, eta, W_overlap, W_res
```

扇区列名来自 YAML 标签。`W_val`、`P_v`、`eta` 分别为目标谷子空间权重、谷纯度、有符号谷极化。近简并目标态的逐条结果是规范依赖的。

### valley_subspace.json

近简并态的主要摘要文件。记录投影谷算符、谷适配基诊断，以及目标子空间落在所选谷子空间中的程度。

### symmetry_eigenvalues.csv

只有操作通过 V1 表示矩阵检查后才写入：真旋转且阶数在 `[2,3,4,6]`、属于 HSP 小群、保持目标谷扇区。

主要列：

- `kpoint`：moiré 高对称点
- `operation_id`：`symmetry_report.json` 中的对称操作编号
- `order`：旋转阶数
- `basis`：`valley_adapted` 或 `raw_diagnostic`
- `phase_2pi`：本征值相位 / 2π
- `nearest_root_of_unity` / `root_deviation`：最近单位根及偏差
- `rotation_ready`：表示矩阵通过 V1 数值检查
- `topology_input_ready`：适合作为后续拓扑分析输入（保守标志）
- `diagnostic_only`：仅供诊断
- `D_valley_offdiag_norm`：双谷适配基中表示矩阵非对角块范数
- `valley_eta`：谷适配态的有符号谷极化

### valley_basis_transform.h5

```text
/GammaM/transform
/GammaM/eta
/GammaM/s_matrix
/GammaM/v_matrix
/GammaM/band_indices_vasp
/GammaM/sectors
```

### symmetry_report.json

记录检测到的对称操作（类型、旋转矩阵、平移、候选旋转状态、后端、小群/谷保持检查结果）。

`"status": "skipped"` 通常表示 `symmetry.operations.structure_file` 未设置或路径错误。

### diagnostics.h5

投影掩码与 q-cut 扫描数据，用于调参和调试：

```text
/projectors/<kpoint>/sector_masks
/projectors/<kpoint>/center_masks
/projectors/<kpoint>/overlap_mask
/qcut_scan/<kpoint>/qcuts
/qcut_scan/<kpoint>/w_val
...
/symmetry_representations/<kpoint>/<operation_id>/D_raw
/symmetry_representations/<kpoint>/<operation_id>/D_valley
...
```

`D_valley` 仅在存在有效谷适配基时写入。

## V1.1：对称本征值诊断

对每个目标高对称点，计算满足小群条件和谷保持条件的 proper-rotation 对称操作的表示矩阵、本征值和特征标。

- 枚举通过小群检查 + 谷保持检查的全部 proper rotation。
- 计算 `D_raw` 和 `D_valley`（谷适配基）表示矩阵。
- 输出本征值、相位、特征标和 readiness 标志。
- 输出：`symmetry_eigenvalues.csv`、`symmetry_report.json` 和 `diagnostics.h5`。

**Irrep label matching 暂缓。** 谷分辨的单态不会自动匹配到完整 double space group irrep label（如 -K4/-K5/-K6）。这需要先定义 valley-preserving subgroup restriction 和 parent full irrep 的关系，留待后续理论工作。

## V1 边界

当前 V1 不包含：

- 全 mBZ 谷有效性验证
- Berry 曲率
- Wilson 回路
- 自动完整整数陈数判断
- 真正的层分辨投影
- 自动单层 VBM/CBM 谷搜索
- 对所有 WAVECAR variant 的完整支持

V1 的优先级是把高对称点谷投影、谷适配子空间和对称性诊断做到可复现、可检查。
