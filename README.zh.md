# ValleyScope 中文说明

[English](README.md)

ValleyScope 是面向**二维莫尔材料高通量后处理**的 valley projection、
valley-projected irreps 与 reduced EBR workflow。它从选定的 VASP 波函数
提取 parent-layer 动量 valley 信息，识别各 valley-projected subspace 的
对称性，并依据经过审查的 Bilbao/irreptables 约定构造或求解降维 EBR 问题。

ValleyScope 输出物理证据和明确 blocker，不会把高 valley 权重或少量对称
本征值自动解释为拓扑标签。

## 支持范围

目前主要支持并经过验证的计算范围为：

- VASP 平面波波函数，输入为 `WAVECAR` 或 ValleyScope HDF5 中间文件；
- 非磁、含自旋轨道耦合、具有 parent time-reversal symmetry（TRS）且使用
  默认 VASP Cartesian spin frame `SAXIS=[0,0,1]` 的二维莫尔体系；
- 选定的莫尔高对称点（HSP）、目标能带、parent-layer valley center，以及
  用于对称性识别的莫尔或双层结构。

工作流对晶格、空间群、HSP 集合、valley orbit 和 valley-projected subspace
space group 保持通用。真实材料只用于验证，不会选择生产算法。

上述计算条件无法仅从 `WAVECAR` 或紧凑 HDF5 中完整恢复，用户必须确认源
计算属于该范围。磁性体系、spin-space-group 处理、无 SOC 非共线计算和任意
自旋轴目前不作为已验证的生产输入。

## 物理工作流

```text
VASP WAVECAR / ValleyScope HDF5
-> q-cut momentum-valley projection
-> valley mapping 与 valley-projected subspace symmetry
-> HSP little group 与 valley-preserving subgroup
-> 限制在 valley-preserving subgroup 上的对称表示
-> valley-preserving irreps
-> Bilbao/irreptables 约定下的 reduced-dimensional EBR data
-> exact-integer reduced EBR decomposition
```

### 动量 Valley 投影

对莫尔动量为 \(\mathbf k_M\) 的布洛赫态，每个平面波分量的动量为

```math
\mathbf q = \mathbf k_M + \mathbf G_M .
```

ValleyScope 将 \(\mathbf q\) 的面内分量与配置的 parent-layer valley center
比较，并对相应单层倒格矢取模。q-cut 窗口为 valley \(a\) 定义 seed
projector \(P_a^0\)。对归一化态，

```math
W_a = \langle\psi|P_a^0|\psi\rangle,
\qquad
W_{\rm val} = \sum_a W_a .
```

这是动量空间 parent-valley projection，不是完整的单层 Bloch-state
unfolding。`W_val` 依赖真实的 valley center 和 q-cut，不是拓扑不变量。

近简并目标能带中的单条 VASP 本征矢依赖规范选择，因此 ValleyScope 分析整个
目标子空间，包括投影 valley 矩阵和 valley-adapted basis。详细的子空间权重、
归属和 projector-quality 证据由 debug profile 保留：

```text
S_min:              目标谷子空间权重下界
min_concentration:  valley-adapted basis 中的最低 valley 集中度
assigned_valleys:   每个适配态的 valley 归属
valley_weights_adapted: 适配基中的 valley 权重
```

可选的 projector mode 为：

- `fixed_center`（默认）：使用固定 parent-layer valley center；这些 seed
  projector 参与对称性与 irrep readiness。
- `k_resolved_parent_valley`：使用动态 center 报告整个莫尔布里渊区中的
  parent-valley 权重；它不会替代 irrep 或 EBR readiness 中的 fixed-center
  projector。

### Valley-Preserving Symmetry

对 HSP \(k\)，HSP little group 为

```math
G_k = \{g \mid gk = k + G_M\}.
```

若 \(\pi_g(a)\) 是操作 \(g\) 诱导的 valley mapping，则 valley \(a\) 的
valley-preserving subgroup（谷保持子群）为

```math
G_k^{(a)} = \{g \in G_k \mid \pi_g(a)=a\}.
```

seed-projector covariance 条件为

```math
D_g P_a^0 D_g^\dagger \approx P_{\pi_g(a)}^0 .
```

把 \(a\) 映射到另一 valley 的操作是 **valley-changing operation**。它提供
valley orbit 和 **valley sewing matrix（谷缝合矩阵）** 数据，不能被强行
纳入 \(G_k^{(a)}\) 的单 valley irrep。

ValleyScope 把对称表示限制在真实的 valley-preserving operation set 上；
SOC 波函数使用 double-valued irreps。匹配依赖经过验证的
standard-setting certificate，它把计算得到的仿射空间群操作和倒空间坐标映射
到 Bilbao/irreptables 约定。仅有旋转矩阵或用户标签不足以证明该约定。

### 时间反演

Parent TRS **不意味着** one-valley subspace 保持 TRS。一般情况下，时间反演
联系一对 time-reversed valleys：

- 单 valley irrep 是其 valley-preserving subgroup 的酉不可约表示；
- time-reversal completion 是独立的 valley 间构造；
- one-valley 结果不会默认匹配到 grey group；
- 只有所需反酉与 valley 间证据存在并通过验证时，才使用联合 grey-group
  数据。

仅当输入确定来自 parent-TRS 计算时才启用
`analysis.time_reversal.enabled`。缺失或不一致的 sewing evidence 会保留为
明确 blocker。

### Reduced EBR 分析

原始三维 EBR 数据不是 ValleyScope 的答案。ValleyScope 使用经过审查的
Bilbao/irreptables 源约定，并把数据约化到与 valley 计算相同的物理基：

```text
source EBR data
-> 已认证的 valley-projected subspace space group
-> sampled source-HSP basis
-> 限制到 valley-preserving subgroup
-> matched valley-preserving irrep basis 中的重数
-> reduced EBR matrix
```

得到的整数向量由纯 Python/SymPy exact integer solver 求解。输出区分：精确
非负 EBR 组合、属于整数张成但无非负 witness、不属于整数张成、有界搜索尚不
确定，以及被物理证据阻塞。任何一种分类本身都不是陈数结论。

### 信任条件

高 valley purity 有参考价值，但不足以得到可信 irrep 或 EBR。进入可信结果还
要求相关证据通过，包括：

- 目标子空间闭合性和表示酉性；
- seed projector 在完整 valley mapping 下的 covariance；
- 定义明确的 valley-projected subspace 与 \(G_k^{(a)}\)；
- 完整且无歧义的 operation mapping 和 source-HSP mapping；
- 经过验证的 standard setting 与受审查的 irrep/EBR provenance；
- 保守的 spinful 证据，以及启用时间反演时的 antiunitary sewing evidence。

干净的 fixed-center seed basis 可以直接使用；否则 ValleyScope 可构造
symmetry-adapted valley basis。失败或不完整的证据保持 diagnostic-only 或
blocked，不应仅为获得标签而放宽 tolerance。

## 安装

ValleyScope 要求 Python 3.10 或更高版本。

```bash
git clone https://github.com/gawcista/valley-proj.git
cd valley-proj
python -m pip install -e .
```

检查已安装命令：

```bash
valleyscope --help
valleyscope analyze-hsp --help
```

在源码目录中，`python -m valleyscope.cli --help` 等价。

## 快速开始

标准流程只需抽取一次紧凑 HDF5，随后分析该文件。

### 1. 抽取选定的 WAVECAR 数据

创建 `extract.yaml`：

```yaml
input:
  wavecar: ./WAVECAR

extract:
  kpoints:
    - name: GammaM
      vasp_index: 1
    - name: KM
      vasp_index: 2
  bands_vasp: [101, 102]

output:
  wavefunction_h5: ./wavefunctions.h5
```

`vasp_index` 和 `bands_vasp` 均使用从 1 开始的 VASP 编号。

```bash
valleyscope extract-wavecar extract.yaml
```

### 2. 分析 HSP 波函数

创建 `analyze.yaml`：

```yaml
input:
  wavefunction_h5: ./wavefunctions.h5
  monolayer_poscars:
    parent: ./monolayer.vasp

analysis:
  kpoints: [GammaM, KM]
  iband: [101, 102]
  time_reversal:
    enabled: true

valley_centers:
  coordinate_mode: layer_frac
  centers:
    - name: valley_a
      layer: parent
      frac: [0.333333333333, 0.333333333333, 0.0]
    - name: valley_b
      layer: parent
      frac: [-0.333333333333, -0.333333333333, 0.0]

valley_subspaces:
  - name: valley_a
    centers: [valley_a]
  - name: valley_b
    centers: [valley_b]

projection:
  projector_mode: fixed_center
  qcut_fraction: 0.20

symmetry:
  operations:
    structure_file: ./moire.vasp

output:
  directory: ./valley_analysis
  profile: standard
```

必须用实际体系的 HSP、能带、valley center、结构和 q-cut 替换示例值。对多层
体系，应一致定义各层倒空间坐标和变换；对公度结构，整数 supercell transform
通常比只给转角更可靠。

```bash
valleyscope analyze-hsp analyze.yaml
```

### 终端摘要示例

standard profile 在终端打印与 `valley_summary.txt` 完全相同的结果优先摘要。
主要部分为：

```text
Run and projection context
Valley projection by sampled state
Valley-projected subspace space group and trusted HSP irreps
Authoritative reduced EBR results
Readiness blockers and warnings
Public output files
```

运行上下文包含 `qcut mode:` 和实际 q-cut。standard
`valley_summary.json` 保留紧凑的 `valley_projection_summary`、canonical
`valley_resolved_irreps`、`reduced_ebr_summary` 和
`readiness_blocker_summary`。常见投影状态包括
`fixed_center_not_captured`、`not_derived` 和 `unreliable`。

## 输入与配置

一次分析需要：

- ValleyScope HDF5，包含选定的平面波系数、倒格矢、k 点、能量和 VASP
  能带编号；
- 单层倒格信息和物理定义的 valley center，必要时包含 layer transform；
- HDF5 中实际存在的 HSP label 与 `analysis.iband`；
- 在 `symmetry.operations.structure_file` 指定供 spglib 识别操作的莫尔或
  双层 POSCAR/CONTCAR。

单层结构定义 parent-layer 倒空间坐标；莫尔或双层结构定义对称操作，二者不能
互换。

日常计算使用 `output.profile: standard`；需要详细 projector、表示、闭合性、
HSP-star 或 sewing 证据时使用 `output.profile: debug`。这里有意省略高级
source-table 与 standard-setting override；准备受审查输入时，应以当前 CLI
help 和 `valleyscope/io/config.py` parser 为准。

## 输出

standard profile 突出以下公开入口：

| 文件 | 用途 |
| --- | --- |
| `valley_summary.txt` | 结果优先的人类可读摘要，应首先查看 |
| `valley_summary.json` | 紧凑机器记录，包含投影、canonical irrep、权威 reduced-EBR 与 blocker 摘要 |
| `valley_weights.csv` | 快速扫描每个 (kpoint, VASP band) 的原始 valley 权重 |
| `valley_ebr_export_bundle.json` | 仅当至少存在一个 ready bundle 时写出 |
| `valley_reduced_ebr_mapping.json` | 仅当 reduced EBR mapping 已启用且实际完成评估时写出 |

`valley_resolved_irreps` 对每个采样 `(kpoint, valley)` 只保留一个 canonical
记录，包括 valley-projected subspace space group、HSP little group、
valley-preserving operation IDs、readiness、blocker 和
`irrep_multiplicities`。

`valley_weights.csv` 的 raw row 适合快速筛查，但近简并能带子空间内的单条
结果依赖规范选择，应结合摘要中的子空间与 readiness 信息解释。

debug profile 保留 `Valley subspaces`、`Valley subspace analysis`、
`diagnostics.h5`、对称性报告、限制后的表示数据及 irrep/EBR provenance
等详细证据。这些是调试入口，不是每位用户都必须逐个检查的主输出。若没有行
满足相应物理作用域，某些 debug 表可能只有 header-only；这本身不表示运行
失败。
数据库 ingestion 从 standalone 公开 bundle/mapping 文件读取完整 EBR
证据；紧凑 summary 不再复制这些完整 payload。

## 执行 Reduced EBR Mapping

`analysis.reduced_ebr.enabled` 默认关闭。对 ready canonical bundle，分析器可从
已安装的 `irreptables` 数据构造受审查的 reduced table，也可读取用户提供并
验证过的 reduced table 或 reviewed mapping specification。所有路径都必须先
通过 group、setting、spin、HSP、irrep 和 provenance 检查，才能进入 exact
solver。

权威评估在 `analyze-hsp` 内完成，因为此时仍可用生产者拥有的表示证据重新计算。
导出的 JSON bundle 保留 identity link，便于审计和下游传递，但这些 link 本身
不能重新建立数值可信性。因此，仅向 standalone 命令提供序列化 JSON 时，它
执行的是 fail-closed 兼容性审计，不会把 identity-only bundle 提升为权威解：

```bash
valleyscope map-reduced-ebr \
  valley_ebr_export_bundle.json \
  validated_reduced_ebr_table.json \
  --output valley_reduced_ebr_mapping.json
```

ValleyScope 不提供临时拼接或未经审查的生产 EBR 表。

## 限制与非目标

ValleyScope 当前不提供：

- 把原始三维 EBR decomposition 作为 valley-resolved 结果；
- 内置未经审查的 EBR 表或启发式浮点 EBR fitting；
- compatibility relations；
- Berry curvature、Wilson loop 或 Chern number；
- 自动的 full-moire-Brillouin-zone valley-goodness 验证；
- 仅依据 HSP 数据给出的拓扑结论。

## 开发

安装测试依赖并运行测试：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

公开命令定义在 `valleyscope/cli.py`，配置解析定义在
`valleyscope/io/config.py`，公开输出选择定义在
`valleyscope/reports/analysis_outputs.py`。
