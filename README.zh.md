# ValleyScope 中文说明

[English](README.md)

ValleyScope 用于分析二维莫尔材料中 VASP 波函数的能谷成分与对称性表示。
它从选定的能带子空间中分辨母层各能谷（valley）的贡献，确定保持每个谷的
对称操作，并求取莫尔高对称点（HSP）处的不可约表示（irrep）。

这些不可约表示可进一步与约化到同一谷子空间和高对称点基底的基本能带表示
（elementary band representation，EBR）比较。所得结果描述的是选定态的
对称性内容，不能单独给出整个莫尔布里渊区上的拓扑分类。

## 支持范围

当前实现面向以下计算条件：

- VASP 平面波波函数，输入为 `WAVECAR` 或 ValleyScope HDF5 中间文件；
- 非磁、含自旋轨道耦合（SOC）、完整体系具有时间反演对称性（TRS），且使用
  默认 VASP 笛卡尔自旋坐标系 `SAXIS=[0,0,1]` 的二维莫尔体系；
- 选定的莫尔高对称点和目标能带，以及明确指定的母层谷中心和用于确定空间
  对称性的莫尔或双层结构。

计算使用输入体系本身的晶格与对称操作，不按材料名称选择算法，也不预设谷的
数目。现有数值检验覆盖若干具体情形，并不代表所有空间群或能带子空间均已验证。

上述计算条件无法仅从 `WAVECAR` 或紧凑 HDF5 中完整恢复，用户必须确认源
计算属于该范围。磁性体系、自旋空间群、无 SOC 的非共线计算和任意自旋轴
尚不在已验证的适用范围内。

## 物理方法

```text
平面波系数
  → 母层能谷投影
  → 各谷投影子空间的对称性
  → 高对称点小群表示
  → 与约化 EBR 比较
```

### 动量空间的能谷投影

对莫尔动量为 \(\mathbf k_M\) 的布洛赫态，每个平面波分量的动量为

```math
\mathbf q = \mathbf k_M + \mathbf G_M .
```

ValleyScope 将 \(\mathbf q\) 的面内分量与指定的母层谷中心比较，并考虑所有
单层倒格矢平移下的最短笛卡尔距离；非正交晶格也使用这一距离定义。
动量截断（q-cut）窗口为谷 \(a\) 定义初始投影算符 \(P_a^0\)。对归一化态，

```math
W_a = \langle\psi|P_a^0|\psi\rangle,
\qquad
W_{\rm val} = \sum_a W_a .
```

这是按母层能谷划分的动量空间投影，不是完整的单层布洛赫态能带反折叠。
`W_val` 取决于谷中心和动量截断半径，不是拓扑不变量。

近简并能带中的单条 VASP 本征矢依赖规范选择，因此 ValleyScope 分析整个
目标子空间，包括其中的谷投影矩阵和按谷分辨的基底。设置
`output.profile: debug` 可查看子空间权重、各态的谷归属，以及投影算符的
正交性和协变性残差。

投影方式有两种：

- `fixed_center`（默认）：使用固定母层谷中心，以相应的初始投影算符检验
  子空间对称性并确定不可约表示。
- `k_resolved_parent_valley`：在采样的莫尔动量处，使用随动量变化的中心
  计算母层谷权重。对称性与 EBR 分析仍从固定中心的初始投影算符出发；这一选项也
  不等于验证了整个布里渊区内的谷特征。

### 保持单谷的对称性

高对称点 \(k\) 处的小群为

```math
G_k = \{g \mid gk = k + G_M\}.
```

若 \(\pi_g(a)\) 表示对称操作 \(g\) 对谷的映射，则谷 \(a\) 的谷保持子群
（valley-preserving subgroup）为

```math
G_k^{(a)} = \{g \in G_k \mid \pi_g(a)=a\}.
```

初始投影算符应满足协变性条件

```math
D_g P_a^0 D_g^\dagger \approx P_{\pi_g(a)}^0 .
```

把 \(a\) 映射到另一谷的操作称为换谷操作（valley-changing operation）。
它通过谷缝合矩阵（valley sewing matrix）联系两个子空间，不属于
\(G_k^{(a)}\) 的单谷表示。若波函数已验证这一幺正关系，可据此确定对称性
相关的未采样点处的表示。变换后的子空间仍须与目标小群的不可约表示匹配，
并与直接采样得到的态分开标明来源。

ValleyScope 在完整的谷保持子群上匹配不可约表示（valley-preserving irrep），
对 SOC 波函数使用双值表示（double-valued irreps）。这包括多维不可约表示，
不局限于某个旋转生成元的本征值。与 Bilbao/irreptables 比较时，基矢、原点、
仿射操作和完整平移晶格必须采用一致的晶体学约定。仅有相同的旋转矩阵或
空间群名称并不足够。

### 时间反演

完整体系具有时间反演对称性，不意味着单谷子空间也保持这一对称性。当时间
反演交换两个谷时，应先用各自谷保持子群的幺正不可约表示描述每个谷。

对已知保持时间反演的计算，可利用已验证的源不可约表示和表示表中的时间
反演配对关系，推断未采样的时间反演伙伴。这一代数推断既不同于幺正谷缝合，
也不同于直接利用波函数检验反幺正缝合关系。输出会区分直接采样与推断的表示。

包含时间反演的联合描述，即灰群共表示，还需要额外的反幺正及谷间关系检验。
这些检验不通过，并不自动否定已经独立验证的单谷幺正表示。

仅当输入确定来自保持时间反演对称性的计算时，才启用
`analysis.time_reversal.enabled`。若所需信息缺失或不一致，相应的对称关系
将保留为未确定，而不是强行赋予表示标签。

### 约化 EBR 分析

完整体系三维空间群的 EBR 不能直接与单谷不可约表示比较。ValleyScope 使用
谷投影子空间空间群对应的、经过核对的 Bilbao/irreptables 数据，将其限制到
与波函数计算相同的高对称点及谷保持不可约表示基底：

```text
源 EBR 数据
→ 标准晶体学约定已验证的谷投影子空间空间群
→ 所用的源高对称点基底
→ 限制到谷保持子群
→ 谷保持不可约表示的重数
→ 约化 EBR 矩阵
```

不可约表示的重数组成整数向量，各约化 EBR 向量组成整数矩阵的列。
ValleyScope 使用 Python/SymPy 的精确算术求取展开系数，并区分非负整数
EBR 组合、属于整数张成但无非负组合、不属于整数张成、有界搜索未定，以及
物理信息不足等情形。

存在非负组合，说明所用高对称点上的表示与约化 EBR 相容，不代表已证明存在
全局等价的局域 Wannier 函数。反之，不属于整数张成的结论也取决于所选谷
子空间、高对称点基底和经过核对的 EBR 生成元；它不是陈数计算。

### 不可约表示的适用条件

高谷纯度不足以保证能带子空间承载所需的对称性表示。计算还会检验：

- 目标子空间在对称操作下的闭合性与表示的幺正性；
- 初始投影算符在完整谷映射下的协变性；
- 谷投影子空间及 \(G_k^{(a)}\) 是否定义明确；
- 平面波、对称操作和源高对称点之间的映射是否完整；
- 晶体学约定及不可约表示、EBR 源表是否一致；
- 双群乘法关系，以及需要时的反幺正缝合关系。

固定中心的初始基底若满足这些条件，可直接使用；否则 ValleyScope 尝试构造
对称性适配的谷基底。若两者均不满足条件，就不赋予相应的不可约表示或 EBR
结果，并保留残差与原因供检查。输出详略不会改变这些物理要求。

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

### 2. 分析高对称点波函数

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

请用实际体系的高对称点、能带、谷中心、结构和动量截断半径替换示例值。对多层
体系，各层倒空间坐标与坐标变换应采用一致约定；对公度结构，整数超胞变换
通常比只给转角更可靠。

```bash
valleyscope analyze-hsp analyze.yaml
```

### 终端摘要示例

终端与 `valley_summary.txt` 显示相同的摘要：

```text
Run and projection context
Valley projection by sampled state
Valley-projected subspace space group and trusted HSP irreps
Authoritative reduced EBR results
Readiness blockers and warnings
Public output files
```

阅读谷权重时，应结合子空间对称性和不可约表示结果。`qcut mode:` 说明动量
窗口的定义方式。JSON 中的 `valley_projection_summary` 对应谷投影摘要；
不可约表示、EBR 结果和未满足的条件分别记录在 `valley_resolved_irreps`、
`reduced_ebr_summary` 和 `readiness_blocker_summary` 中。

投影状态标签包括 `fixed_center_not_captured`、`not_derived` 和 `unreliable`。
其中，固定中心窗口内权重低，并不能证明该态不来自相应母层能谷。
详细输出中的 `Valley subspaces` 和 `Valley subspace analysis` 包括：

```text
S_min:              目标谷子空间权重下界
min_concentration:  谷适配基中最低的谷集中度
assigned_valleys:   每个适配态的谷归属
valley_weights_adapted: 适配基中的谷权重
```

## 输入与配置

一次分析需要：

- ValleyScope HDF5，包含选定的平面波系数、倒格矢、k 点、能量和 VASP
  能带编号；
- 单层倒格信息和有物理依据的谷中心，必要时包含层间坐标变换；
- HDF5 中实际存在的高对称点标签与 `analysis.iband`；
- 在 `symmetry.operations.structure_file` 指定供 spglib 识别操作的莫尔或
  双层 POSCAR/CONTCAR。

单层结构定义母层的倒空间坐标；莫尔或双层结构定义对称操作，二者不能
互换。

日常计算使用 `output.profile: standard`；需要检查投影残差、表示矩阵、
子空间闭合性、高对称点星或缝合矩阵时，使用 `output.profile: debug`。
源表与标准晶体学约定的高级选项可查阅命令帮助和
[`valleyscope/io/config.py`](valleyscope/io/config.py)。这些选项不能代替对
输入物理约定的核对。

## 输出

标准输出包括：

| 文件 | 用途 |
| --- | --- |
| `valley_summary.txt` | 谷权重、子空间对称性、不可约表示、EBR 结果和未满足的条件 |
| `valley_summary.json` | 便于后续分析的结构化结果 |
| `valley_weights.csv` | 各 k 点、各 VASP 能带的原始谷权重 |
| `valley_ebr_export_bundle.json` | 不可约表示向量与相应对称性数据；至少一组 EBR 输入满足导出条件时写出 |
| `valley_reduced_ebr_mapping.json` | 启用并实际评估约化 EBR 匹配时写出 |

`valley_resolved_irreps` 为每个采样的 `(kpoint, valley)` 保留一条记录，
包含谷投影子空间的空间群、高对称点小群、谷保持操作、不可约表示重数
（`irrep_multiplicities`），以及允许或阻止表示赋值的条件。

`valley_weights.csv` 便于快速筛查，但近简并子空间内单条能带的权重依赖
规范选择，应结合子空间分析解释，不能将其视为不随规范改变的能带标签。

详细输出还保留 `diagnostics.h5`、对称性报告、限制后的表示矩阵及不可约表示
和 EBR 的来源信息，便于检查为何未能确定某个表示。若没有态满足相应的物理
范围，某些表可能只有表头（header-only）；这本身不表示计算失败。
离线汇总除了摘要，还会读取单独的 EBR 导出文件和匹配文件，因此应将这些
文件保存在一起。

## 进行约化 EBR 匹配

约化 EBR 分析默认关闭。在 `analyze.yaml` 的 `analysis` 下加入
`reduced_ebr: {enabled: true}` 即可请求这一分析。程序可以从已安装的
`irreptables` 构造约化表，也可以读取用户提供且经过核对的约化表或映射说明。
无论采用哪种方式，群、晶体学设置、自旋约定、高对称点基底、不可约表示及
表格来源都必须与波函数计算一致。

这一分析应在 `analyze-hsp` 中完成，此时仍可利用波函数和数值表示进行检验。
下面的独立命令用于检查导出数据的相容性。JSON 中的标识本身无法重现波函数
层面的检验，因此不足以让该命令给出经过物理验证的 EBR 结果：

```bash
valleyscope map-reduced-ebr \
  valley_ebr_export_bundle.json \
  validated_reduced_ebr_table.json \
  --output valley_reduced_ebr_mapping.json
```

ValleyScope 不提供临时拼接或未经审查的 EBR 表。

## 汇总多组计算结果

已完成的计算可汇总为单组 JSON 记录或多组计算的索引，便于比较：

```bash
valleyscope collect-database-record ./valley_analysis \
  --output ./database_ingestion_record.json

valleyscope collect-database-index ./run_a/valley_analysis ./run_b/valley_analysis \
  --output ./database_index.json
```

汇总时会区分最终 EBR 结果、不完整输入和被排除的结果，并检查它们与当前计算
摘要是否一致，但不会重新从波函数计算对称性矩阵。输入路径需要逐一指定。
目前实现的是离线结果汇总，不是数据库服务，也不负责自动调度计算。

## 数值检验与当前结果

小规模构造的旋量波函数用于检验从谷投影、不可约表示赋值到精确约化 EBR
比较的完整数值计算。[`P3` 样例](tests/test_portable_numerical_chain.py)
检验一对时间反演相关的谷；[`P4mm` 样例](tests/test_noncommuting_numerical_chain.py)
检验 \(\Gamma\)、\(X\)、\(M\) 点处非对易的二维双值表示。
两套样例都包含归一化且谷纯度高、但不满足空间对称性闭合的态；程序不会为
这些态给出可信的表示或 EBR 分解。
P4mm 样例只有一个谷，且属于对称型空间群；它不检验换谷镜面操作或
非对称型空间群中的分数平移相位。

代码版本 `49cc142` 的本地材料回归计算得到：

| 计算体系 | 选定能带子空间的谷分辨结果 |
| --- | --- |
| tMoTe₂ | \(K\) 与 \(K'\) 两个谷的不可约表示向量均不属于所用约化 EBR 的整数张成（`outside_integer_span`） |
| tZrSe₂ | 三个 \(M\) 谷各自存在精确的非负整数约化 EBR 组合（`solved_exact`） |

这些结果对应特定的能带选择和高对称点，并非对两种材料的普遍断言。
tMoTe₂ 的结果不是直接的陈数计算，tZrSe₂ 的结果也不证明全局 Wannier
可构造性。部分可选的时间反演或联合灰群结果仍未确定。大型材料波函数及其
输出不随仓库发布；上述小规模构造样例不依赖这些文件。

## 限制与非目标

ValleyScope 当前不提供：

- 将原始三维 EBR 分解作为谷分辨结果；
- 内置未经审查的 EBR 表，或用启发式浮点拟合代替整数求解；
- 能带表示的相容关系分析；
- 贝里曲率、Wilson 环路或陈数计算；
- 整个莫尔布里渊区内谷特征的自动检验；
- 仅依据高对称点数据作出的拓扑结论。

## 开发

安装测试依赖并运行测试：

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

若只运行小规模旋量波函数样例：

```bash
python -m pytest -q tests/test_portable_numerical_chain.py tests/test_noncommuting_numerical_chain.py
```

测试会自行生成所需的小波函数文件。本地开发笔记的检查默认跳过，公开仓库的
测试不要求提供这些笔记。以下命令还可对已提交版本构建独立安装包并验证：

```bash
python -m pip install build
python scripts/release_gate.py --checkout .
```

请在没有未提交改动的源码目录中运行；这一检查可能下载依赖。命令定义见
[`valleyscope/cli.py`](valleyscope/cli.py)，配置解析见
[`valleyscope/io/config.py`](valleyscope/io/config.py)，输出选择见
[`valleyscope/reports/analysis_outputs.py`](valleyscope/reports/analysis_outputs.py)。
