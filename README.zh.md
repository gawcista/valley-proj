# ValleyScope 中文说明

**语言:** 中文 | [English](README.md)

ValleyScope 是一套面向 VASP moire 超胞波函数的动量空间谷投影（momentum-space valley projection）与谷分辨对称性分析（valley-resolved symmetry analysis）后处理流程。它从选定的 moire 波函数出发，诊断其单层谷特征，在近简并目标子空间中固定谷规范，并计算谷保持对称操作的小群表示。

这些对称本征值和表示矩阵是后续拓扑分析的输入，而不是自动拓扑标签。严格的谷分辨拓扑判断仍需要高对称点之外的额外验证。

## 物理动机

在 moire 计算中，布洛赫动量属于 moire 布里渊区。但谷标记通常继承自单层布里渊区。因此，某个态位于 moire 高对称点并不意味着它就是某个单层谷态。ValleyScope 把谷定义为通过 moire 波函数中的平面波动量解析出的单层谷。

对转角双层体系，同一个单层谷中的上下层动量可以杂化。这种层间杂化本身不破坏谷量子数。真正需要诊断的是目标态是否仍然局限在选定的谷子空间中，还是出现了谷间混合或谷外剩余权重。

工作流在选定高对称点上回答这个问题。它还会检查候选对称操作是否属于该高对称点的小群（little group），以及是否保持所选谷，然后才报告对称本征值。

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
谷投影算符由一组窗口的并集构造。以含两个谷的转角双层为例，

```math
\Omega_K=
\Omega_{{\rm top},K}\cup\Omega_{{\rm bottom},K},
\qquad
\Omega_{K'}=
\Omega_{{\rm top},K'}\cup\Omega_{{\rm bottom},K'} .
```
YAML 中的 `K_valley` 等名称只是用户定义的谷子空间标签。

### 权重与谷间混合诊断

设 $P_i$ 是第 $i$ 个谷的投影算符。默认策略下，落入多个谷投影窗口的平面波分量会从所有谷投影算符中移除，放入重叠投影算符 $P_\times$：

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
对归一化态 $|\psi\rangle$，第 $i$ 个谷的权重为

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
对于双谷情形，`P_v` 与 `|eta|` 存在冗余关系：`P_v = (1 + |eta|) / 2`。三谷及以上 `|eta|` 无定义，直接使用 `P_v` 作为谷集中度分数。

谷集中度分数分为三档：
- **clean**：高于 clean 阈值（默认 `P_v=0.95`；在双谷分析中等价于 `|eta|=0.90`）。
- **approx**：介于 approximate 和 clean 阈值之间。
- **mixed**：低于 approximate 阈值；单谷对称性数据只应作为诊断参考。

公开摘要中的 `status` 还包括：

- **not_derived**：目标态或目标子空间在用户定义谷子空间中的权重不足。
- **unreliable**：投影窗口或剩余权重没有通过当前可靠性检查。
- **n/a**：该子空间诊断不适用，例如单条非简并能带。

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
ValleyScope 报告分解 $W_{\rm val}+W_{\rm overlap}+W_{\rm res}=1$。在 CSV/JSON 输出中，`W_overlap` 保存 $W_{\rm overlap}$，`W_res` 保存谷外剩余权重。

### 近简并子空间中的规范固定

对孤立非简并能带，可以直接读取上述单态诊断。对近简并目标子空间，VASP 原始本征矢是规范依赖的：子空间内部任意酉变换都合法。

ValleyScope 投影整个目标子空间并构造投影谷矩阵。给定 $N_v$ 个谷投影算符 $P_a$ 和目标态 $\{|\psi_i\rangle\}$，

```math
P_a^{\rm sub}_{ij}=
\langle\psi_i|P_a|\psi_j\rangle,
\qquad
S=\sum_a P_a^{\rm sub},
\qquad
L = \sum_a \lambda_a P_a^{\rm sub}.
```

矩阵 $S$ 检查目标子空间是否落在选定谷子空间中（$W_{\rm val}$ 的子空间推广）。谷标签算符 $L$ 在任意谷数下固定谷适配基。对角化 $L$ 得到适配基 $|\phi_\alpha\rangle$：

```math
w_{\alpha a} = \langle\phi_\alpha|P_a|\phi_\alpha\rangle,
\qquad
{\rm assigned\_valley}_\alpha = \arg\max_a w_{\alpha a},
\qquad
{\rm concentration}_\alpha = \frac{\max_a w_{\alpha a}}{\sum_a w_{\alpha a}}.
```

旧两谷 $V = P_K - P_{K'}$ 只是 $N_v=2$ 时的特例。对三谷及以上（如六角晶格 M-valley star），不应使用 $\eta$；应使用 `valley_weights_adapted`、`assigned_valleys`、`valley_concentration`。实际应主要看 `valley_subspace.json` 和 `valley_basis_transform.h5`。

### 对称性诊断

ValleyScope 用 spglib 从 moire/bilayer 结构文件中自动识别对称操作，支持 `symprec` 和 `symprec_scan`。对称本征值仅在两项检查通过后才报告：

1. 该操作属于目标高对称点的小群。
2. 该操作保持所选谷。

得到的矩阵是谷适配子空间中的小群表示。其本征值是对称性分析数据，可用于基于对称性的拓扑公式约束，但 ValleyScope 不从少数高对称点自动推断完整整数陈数。

### 单谷不可约表示的解释

single-valley irrep 应当与谷保持子群（valley-preserving subgroup）的不可约表示比较：

```math
G_\tau=\{\,g\in G\mid gV_\tau=V_\tau\,\},
```

其中 $V_\tau$ 是选定的单层谷子空间。除非某个操作保持该谷子空间，否则单谷表示不应解释为完整 moire space group 的不可约表示。

含 SOC 时必须使用 double-valued irreps。spinor 波函数在 $2\pi$ 旋转下变号，因此 spinful $C_3$ 满足 $C_3^3=-1$，允许 $\exp(+i\pi/3)$、$-1$、$\exp(-i\pi/3)$ 等本征值。

经过 valley-preservation 过滤后，ValleyScope 会报告全局 `G_tau` 操作集合；在操作数据足够时，会用 `spglib.get_spacegroup_type_from_symmetry` 尝试识别标准空间群。每个 HSP 的条目记为 `G_tau,k`，表示用于 character 输入的谷保持小群。当 operation-to-table mapping 完成时，`irrep_matching.irrep_results_by_kpoint` 会通过 character decomposition 给出 representation-level `irrep_multiplicities`。per-state label 只应在一维结果足够 clean 时再解释，不会贴给 raw VASP states。

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

valley_subspaces:
  - name: K_valley
    centers: [K]
  - name: Kp_valley
    centers: [Kp]

output:
  directory: ./valley_analysis
```

其余字段使用默认值（`qcut_mode: moire_shell`, `qcut_shell: 3`, symmetry detection 跳过等）。
当前分析配置使用 `analysis.iband` 表示 VASP 能带编号，使用
`valley_subspaces` 表示用户定义的单层谷子空间。

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

  # 可选：自动 G-list 重建 cutoff 微调的最大 |delta_Ecut|，单位 eV。
  # 对大型 SOC/非共线 moire WAVECAR，设置一个小值（如 0.1 eV）
  # 可解决 VASP 内部 G-list 与 header ENCUT 重建之间的 cutoff 边界差异。
  # 默认 0.0 = 严格精确匹配模式。
  # ecut_adjust_tol: 0.0

output:
  wavefunction_h5: ./wave.h5
```

```bash
valleyscope extract-wavecar extract.yaml
```

检查 HDF5 内容：

```text
/metadata/g_vector_order
/metadata/g_list_reconstruction_mode   (exact 或 ecut_adjusted)
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

如果抽取器报 G-vector 数量不匹配，先尝试添加一个小 `ecut_adjust_tol`（如 0.1 eV）。大型 SOC/非共线 moire 超胞的 WAVECAR 可能因 cutoff 边界约定出现小差异。如果需要较大 adjustment，说明 WAVECAR 变体、晶格约定或 G-list 排序可能有更根本的不匹配。

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
  benchmark: spinor_C3_reference_check

rotation:
  readiness_preset: strict
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
Valley subspaces
Valley projection summary
Valley subspace analysis
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

屏幕上的 `status` 保持紧凑，但会区分前置失败：`clean`、`approx`、`mixed`、`not_derived`、`unreliable` 或 `n/a`。

再看 `Valley subspace analysis`。近简并子空间中的原始 VASP 本征矢依赖规范选择，逐条能带投影不是最终诊断。ValleyScope 在目标子空间中构造通用投影谷矩阵 $P_a^{\rm sub}$ 和标签算符 $L = \sum_a \lambda_a P_a^{\rm sub}$：

```math
S=\sum_a P_a^{\rm sub}, \qquad
L=\sum_a \lambda_a P_a^{\rm sub}.
```

```text
S = sum_a P_a^sub:  目标能带中的目标谷子空间投影算符
S_min:              目标谷子空间权重下界
S_max:              目标谷子空间权重上界
min_concentration:  谷适配基中的最低谷集中度
assigned_valleys:   每个适配态的谷归属
eta_adapted:        有符号谷极化（仅双谷兼容字段）
```

`S` 判断所选目标能带是否能被选定的谷子空间良好描述；标签算符 `L` 在任意谷数下固定谷适配基。对双谷，`eta_adapted` 作为兼容字段继续输出；对三谷及以上，应使用 `valley_weights_adapted`、`assigned_valleys` 和 `valley_concentration`。双谷 $V=P_K-P_{K'}$ 是这个通用框架的特例。这里输出的是对子空间整体的诊断，不是逐条原始能带投影表的重复。

屏幕摘要的 `status` 取值为 `clean`、`approx`、`mixed`、`not_derived`、`unreliable` 和 `n/a`。前三者描述谷集中度；`not_derived` 表示目标谷子空间权重不足，`unreliable` 表示投影窗口或剩余权重检查失败，`n/a` 表示该子空间诊断不适用。

`Symmetry analysis` 先列出空间群和检测到的对称操作，再按高对称点列出 little-group 操作和 valley-preserving 操作。把一个谷映射到另一个谷的操作会标记为 `valley-exchanging`。JSON 摘要还包含 `symmetry_analysis.valley_preserving_subgroup_report`：它报告全局谷保持 `G_tau` 操作集合，在操作数据足够时尝试识别标准空间群，并记录每个 HSP 的 `G_tau,k` 小群操作集合。如果 `irreptables` table mapping 和 character matching 完成，`irrep_matching.irrep_results_by_kpoint` 会记录 HSP irrep multiplicities，例如 `{"-K5": 1, "-K6": 1}`。`Symmetry eigenvalues` 只列出实际构造了表示矩阵并求得本征值的操作。`symmetry_characters` 聚合通过 little-group 和 valley-preservation 检查的 $\chi^\tau_k(g)=\mathrm{Tr}\,D^\tau_k(g)$；它是 irrep matching 的 character 输入层。`topology_input_ready` 只表示该高对称点对称本征值可作为后续基于对称性的拓扑分析输入，不验证整个 mBZ 的谷分辨拓扑。

对于 SOC spinor 波函数，ValleyScope 在平面波表示中施加 SU(2) 自旋旋转。默认 VASP 自旋约定未验证时标记为 `diagnostic_only=True`。完成基准测试后可设置 `spinor.convention_verified: true`。

屏幕摘要示例：

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

`symmetry.filters.rotation_order` 控制是否启用对称本征值提取，并记录 candidate-rotation summary 中突出显示的循环旋转阶数：
- `auto`：从空间群推断（如 `P321`/`P312` → `C3`，`P422` → `C4`）
- 整数 `n`：请求以 `C_n` 作为循环旋转阶数
- `None`/`none`：跳过对称本征值提取

当前对称分析会对当前支持的所有 proper valley-preserving little-group operations 尝试构造表示和本征值诊断，目前支持阶数为 `2`、`3`、`4`、`6`。non-valley-preserving operations 只作为诊断保留，不作为 single-valley eigenvalues。`rotation_order` 不再表示只对一个生成元对角化，而是作为 summary 兼容字段记录用户请求或自动解析出的旋转阶数。

`root_deviation_tol` 和 `D_valley_offdiag_tol` 是 numerical readiness thresholds，不是普适物理常数。`root_deviation_tol` 检查计算得到的对称本征值是否足够接近允许的 root of unity。`D_valley_offdiag_tol` 检查当前双谷 benchmark 中 `D_valley` 的非对角范数。`readiness_preset` 支持 `strict`、`normal`、`loose`：默认 `strict` 对 root deviation 和双谷非对角检查都使用 `1.0e-6`；`normal` 使用 `1.0e-5` 与 `1.0e-3`；`loose` 使用 `1.0e-4` 与 `1.0e-2`。显式设置 `root_deviation_tol` 或 `D_valley_offdiag_tol` 会覆盖 preset。解释它们时必须同时看 qcut 稳定性、`W_val`、`P_v`、`S_min`、spinor benchmark、plane-wave mapping 质量和 symmetry tolerance。不要为了得到 `topology_input_ready=True` 而随意放宽。

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
kpoint, band_vasp, energy_eV, K_valley, Kp_valley, W_val, P_v, eta, W_overlap, W_res
```

谷列名来自 YAML 标签。`W_val`、`P_v`、`eta` 分别为目标谷子空间权重、谷纯度、有符号谷极化。近简并目标态的逐条结果是规范依赖的。

### valley_subspace.json

近简并态的主要摘要文件。记录投影谷算符、谷适配基诊断，以及目标子空间落在所选谷子空间中的程度。

### symmetry_eigenvalues.csv

只有操作通过表示矩阵检查后才写入：真旋转且阶数在 `[2,3,4,6]`、属于 HSP 小群、保持所选谷。

只有 header-only 的空文件不一定是错误；它通常表示当前 HSP 与谷设置下没有操作通过这些检查。

主要列：

- `kpoint`：moiré 高对称点
- `operation_id`：`symmetry_report.json` 中的对称操作编号
- `order`：旋转阶数
- `phase_2pi`：本征值相位 / 2π
- `nearest_root_of_unity` / `root_deviation`：最近单位根及偏差
- `rotation_ready`：表示矩阵通过数值检查
- `topology_input_ready`：适合作为后续拓扑分析输入（保守标志）
- `diagnostic_only`：仅供诊断
- `D_valley_offdiag_norm`：双谷基中表示矩阵非对角块范数
- `valley_eta`：谷适配态的有符号谷极化

### valley_summary.json 中的 symmetry_characters

summary JSON 包含一等字段 `symmetry_characters`。每一行按 `(kpoint, operation_id)` 聚合，记录 `character_raw`、`character_valley`、readiness flags，以及该操作是否可作为 single-valley representation。只有通过 little-group 和 valley-preservation 检查的操作会进入这里。这是为后续 `G_tau` irrep matching 准备的 character 输入层；这里不使用 character table，也不做 reduced EBR decomposition。

### valley_basis_transform.h5

```text
/GammaM/transform
/GammaM/eta                 （仅双谷）
/GammaM/s_matrix
/GammaM/label_operator
/GammaM/s_eigenvalues
/GammaM/valley_weights_adapted
/GammaM/assigned_valleys
/GammaM/valley_concentration
/GammaM/band_indices_vasp
/GammaM/valid_valley_subspace
```

对正好两个谷，`eta` 和 `v_matrix` 会作为兼容字段继续写出。对三谷及以上，应使用 `valley_weights_adapted` 和 `assigned_valleys`。后续在谷适配子空间中构造表示矩阵时使用这里的 `transform`。

### symmetry_report.json

记录检测到的对称操作（类型、旋转矩阵、平移、候选旋转状态、后端、小群/谷保持检查结果）。

summary JSON 额外暴露 `symmetry_analysis.valley_preserving_subgroup_report`。该结果列出全局 `G_tau` 操作、valley-exchanging 操作编号、闭合状态，以及从检测到的谷保持操作得到的标准空间群匹配。每个 HSP 的 `G_tau,k` 条目列出作为 character 输入的小群操作。`irrep_matching` 记录 table mapping；当有足够的 ready characters 时，`irrep_results_by_kpoint` 给出 representation-level `irrep_multiplicities`。

`"status": "skipped"` 通常表示 `symmetry.operations.structure_file` 未设置或路径错误。

### diagnostics.h5

投影掩码与 q-cut 扫描数据，用于调参和调试：

```text
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
