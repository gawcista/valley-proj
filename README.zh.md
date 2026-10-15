# ValleyScope 中文说明

[English](README.md)

ValleyScope 用于分析二维莫尔材料中 VASP 波函数的能谷成分与对称性表示。
它从选定的能带子空间中分辨母层各能谷（valley）的贡献，确定保持每个谷的
对称操作，并求取莫尔高对称点（HSP）处的不可约表示（irrep）。
若存在保持单谷的旋转对称性，还可给出该子空间陈数模旋转阶数的条件性约束。

这些不可约表示可进一步与约化到同一谷子空间和高对称点基底的基本能带表示
（elementary band representation，EBR）比较，描述选定态在这些动量处的
对称性内容。

## 支持范围

独立的 `project-bands` 工作流支持非磁标量 VASP 态（`ISPIN=1`）及已支持的
SOC 旋量，可沿明确给定的采样路径计算母层谷权重。标量权重逐个空间本征态
计算，不乘自旋简并因子。投影输入为选定能带的 HDF5 系数、母层几何与配置的
谷窗口；空间对称操作、不可约表示表和表示证书用于独立的 HSP 分析。
独立投影的适用范围限于上述标量态和 SOC 旋量，`ISPIN=2` 共线态与非 SOC
非共线态需另行支持。

对称性、不可约表示与 EBR 工作流 `analyze-hsp` 面向以下计算条件：

- 从 VASP `WAVECAR` 抽取并保存为 ValleyScope HDF5 的平面波波函数；
- 非磁、含自旋轨道耦合（SOC）、完整体系具有时间反演对称性（TRS），且使用
  默认 VASP 笛卡尔自旋坐标系 `SAXIS=[0,0,1]` 的二维莫尔体系；
- 选定的莫尔高对称点和目标能带，以及明确指定的母层谷中心和用于确定空间
  对称性的莫尔或双层结构。

计算依据输入体系的晶格、对称操作和谷中心进行。下文的数值样例列出目前
检验过的情形。使用前应核对源计算的磁性、SOC、时间反演对称性和自旋坐标系
是否符合上述条件。

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

`W_val` 表示所选母层能谷动量窗口捕获的权重，其数值取决于谷中心和
动量截断半径。

近简并能带中的单条 VASP 本征矢依赖规范选择，因此 ValleyScope 分析整个
目标子空间，包括其中的谷投影矩阵和按谷分辨的基底。设置
`output.profile: debug` 可查看子空间权重、各态的谷归属，以及投影算符的
正交性和协变性残差。

投影方式有两种：

- `fixed_center`（默认）：使用固定母层谷中心，以相应的初始投影算符检验
  子空间对称性并确定不可约表示。
- `k_resolved_parent_valley`：在采样的莫尔动量处，使用随动量变化的中心
  计算母层谷权重。对称性与 EBR 分析仍从固定中心的初始投影算符出发。

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
它通过谷缝合矩阵（valley sewing matrix）联系两个子空间。
若波函数已验证这一幺正关系，可据此确定对称性
相关的未采样点处的表示。变换后的子空间仍须与目标小群的不可约表示匹配，
并与直接采样得到的态分开标明来源。

ValleyScope 在完整的谷保持子群上匹配不可约表示（valley-preserving irrep），
对 SOC 波函数使用双值表示（double-valued irreps），包括多维不可约表示。
与 Bilbao/irreptables 比较时，基矢、原点、仿射操作和完整平移晶格必须采用
一致的晶体学约定。

### 时间反演

当时间反演交换两个谷时，应先用各自谷保持子群的幺正不可约表示描述每个谷。

对已知保持时间反演的计算，可利用已验证的源不可约表示和表示表中的时间
反演配对关系，推断未采样的时间反演伙伴。这一代数推断既不同于幺正谷缝合，
也不同于直接利用波函数检验反幺正缝合关系。输出会区分直接采样与推断的表示。

包含时间反演的联合描述，即灰群共表示，还需要额外的反幺正及谷间关系检验。
这些检验与单谷幺正表示分别评估并记录结果。

仅当输入确定来自保持时间反演对称性的计算时，才启用
`analysis.time_reversal.enabled`。若所需信息缺失或不一致，相应的对称关系
将保留为未确定。

### 条件性的谷陈数余数

旋转本征值可以约束陈数模 \(n\) 的余数，理论依据见
[Fang、Gilbert 与 Bernevig，Phys. Rev. B 86, 115112 (2012)](https://doi.org/10.1103/PhysRevB.86.115112)。
ValleyScope 从实际保持单谷的绕 \(+z\) 旋转中，自动选择
\(n=2,3,4,6\) 中的最高阶，并组合所需旋转不变动量处的子空间旋转矩阵
行列式。所得余数对应**整个选定的谷子空间**，所用操作应是绕层外轴的
有限阶谷保持旋转。
计算包含自旋旋转相位；若使用经过验证的时间反演推导，会与直接采样的证据
分别标明来源。

例如，`C = 1 (mod 3)` 的前提是整个莫尔布里渊区上存在光滑、恒秩的谷子空间，
且满足一致的布洛赫边界衔接与所用旋转对称性。结果给出同余类
\(C=r+mn\)，其中 \(m\) 为整数；`conditional` 状态记录这些全局前提。

这一分析使用已有波函数与对称性输入，独立于约化 EBR 分析。若没有保持单谷
的旋转，状态为 `not_applicable`；若缺少必需的高对称点或可信表示证据，
状态为 `blocked`，记录原因，余数为 `null`。能够计算的结果始终标为
`conditional`，而 `global_valley_subspace_status` 保持 `not_evaluated`。

### 约化 EBR 分析

ValleyScope 使用谷投影子空间空间群对应的、经过核对的 Bilbao/irreptables
数据，将其限制到与波函数计算相同的高对称点及谷保持不可约表示基底：

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

存在非负组合，说明所用高对称点上的表示与约化 EBR 相容。整数张成的判定
对应所选谷子空间、高对称点基底和经过核对的 EBR 生成元。建立全局 Wannier
描述还需要整个布里渊区内的布洛赫子空间信息。

### 不可约表示的适用条件

为谷子空间赋予对称性表示，需要检验：

- 目标子空间在对称操作下的闭合性与表示的幺正性；
- 初始投影算符在完整谷映射下的协变性；
- 谷投影子空间及 \(G_k^{(a)}\) 是否定义明确；
- 平面波、对称操作和源高对称点之间的映射是否完整；
- 晶体学约定及不可约表示、EBR 源表是否一致；
- 双群乘法关系，以及需要时的反幺正缝合关系。

固定中心的初始基底若满足这些条件，可直接使用；否则 ValleyScope 尝试构造
对称性适配的谷基底。若两种构造均未通过，相应的不可约表示或 EBR 结果
记录为未确定。两种输出模式采用相同的物理要求，并保留残差与原因供检查。

## 安装

ValleyScope 要求 Python 3.10 或更高版本，editable 安装使用 pip 21.3 或更高版本。
在选好的 Python 环境中，克隆源码并安装：

```bash
git clone https://github.com/gawcista/valley-proj.git
cd valley-proj
python -m pip install -e .
```

若同一台机器上的其他计算需要不同版本的包，建议使用独立的 Conda 或虚拟环境。
安装使用当前激活的 Python 解释器及其配置的软件源。

### 依赖环境

程序的直接运行依赖及其物理用途如下：

| 包 | 版本要求 | 分析中的用途 |
| --- | --- | --- |
| NumPy | 由 pip 安装 | 平面波系数、投影算符与表示矩阵 |
| h5py | 由 pip 安装 | 选定能带波函数的 HDF5 读写 |
| PyYAML | 由 pip 安装 | 晶格、谷中心与分析参数 |
| spglib | 由 pip 安装 | 输入结构的空间对称操作 |
| SymPy | 由 pip 安装 | 约化 EBR 的精确整数分解 |
| irreptables | `3.1.0` | 标准小群不可约表示与源 EBR 数据 |
| irrep | Python 3.10/3.11：`>=3.1.1,<3.3`；Python 3.12+：`>=3.1.1,<3.4` | irreptables 调用的辅助函数 |

`pip` 同时安装这些包的依赖，包括 irrep 所需的 SciPy。测试额外依赖 pytest，
并用 setuptools 检查包元数据；构建 ValleyScope 的 wheel 或源码分发包使用
`build` 和 `setuptools>=68`。

通过完整测试及安装后检查的 Python 3.10 参照环境为：Python 3.10.20、
NumPy 2.2.6、h5py 3.16.0、PyYAML 6.0.3、spglib 2.7.0、SymPy 1.14.0、
SciPy 1.15.3、irrep 3.2.0、irreptables 3.1.0。其余包的安装版本由
pip 根据所选 Python 解释器解析。
本轮检查使用 Linux x86_64。额外检查确认，预装的兼容版本 irrep 3.1.1
会在安装后保留；pip 21.3.1 也可在已有源码目录中完成 editable 安装。

安装时，Python 环境标记会自动选择兼容的官方 irrep 版本。
Python 3.10/3.11 使用较早的版本系列；Python 3.12+ 还可使用采用新版
f-string 语法的 3.3 系列。环境中已经安装且符合要求的版本可以继续使用。
允许的版本范围用于依赖选择，数值检查确定具体受测组合。
源码测试与安装包 CI 覆盖 Python 3.10、3.11、3.12、3.13，并实际执行
表示表和 EBR 分析。

受审查的不可约表示和 EBR 源数据保持为 irreptables 3.1.0。
为复现具体计算，可保存其安装包版本，例如运行 `python -m pip freeze`。
也可用 `python -m pip show irrep irreptables` 查看版本。
离线集群需要通过配置好的本地 wheel 目录或软件镜像提供所需依赖。

检查已安装命令：

```bash
valleyscope --help
valleyscope analyze-hsp --help
```

在源码目录中，`python -m valleyscope.cli --help` 等价。

需要生成图片时，安装可选绘图依赖：

```bash
python -m pip install -e ".[plot]"
valleyscope project-bands --help
valleyscope plot-valley-bands --help
```

投影和 HSP 分析仅在请求绘图时导入 Matplotlib。

## 快速开始

莫尔超胞包含大量原子，相应的能带数量也很大，而具体研究通常只关注目标
能量范围内的几条能带。对这一选定子空间，ValleyScope 先从庞大的 `WAVECAR`
中抽取指定能带及所需 k 点，保存这些态的完整平面波系数（包括 SOC 态的
旋量分量）和相关数据，生成紧凑的 HDF5 文件。能带选择应包含完整的目标
子空间及其近简并伙伴；后续的对称性闭合检验会判断其是否适合表示分析。
后续的谷投影、对称性和约化 EBR 分析都读取这个 HDF5 文件。
一次抽取即可重复分析同一批波函数，并减少存储量和跨计算机传输的数据量。

HDF5 还连接了波函数抽取与后续通用分析方法：其中保存平面波系数、晶格、
倒格矢、k 点、能量和能带编号。未来可为其他 DFT 程序编写相应的抽取方法，
例如将 Quantum ESPRESSO（QE）的波函数转换为同一格式，复用后续分析方法。
扩展对称性分析时，还需衔接并验证该程序的坐标、布洛赫相位、归一化和旋量约定。
当前已有的抽取方法读取 VASP `WAVECAR`。

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

该命令按串行方式将选定能带和 k 点保存为 `wavefunctions.h5`。下面的分析配置通过
`input.wavefunction_h5` 读取这个文件。

抽取过程将选定能带的系数逐步写入 HDF5，并分块重建平面波 G 列表。
默认串行路径采用这一节省内存的流程，保留完整平面波基底、旋量分量、
归一化计算次序，以及指定的能带和 k 点顺序，系数使用 `complex128` 保存。
完成的 HDF5 文件通过原子替换发布；抽取失败时保留已有输出。

可在同一个 `extract` 配置下加入执行选项：

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `memory_gb` | 检测到的额度 | 可选的有限正数内存预算，单位 GiB，并受检测到的可用内存限制 |
| `g_chunk_size` | `65536` | 每块候选 G 向量数量，取正整数；内存规划可进一步减小它 |

例如，可在 `extract.yaml` 中设置候选 G 向量分块大小：

```yaml
extract:
  g_chunk_size: 65536
```

使用整节点分配时，通过命令行显式启用自动并行抽取：

```bash
valleyscope extract-wavecar extract.yaml --workers=auto
```

命令行通过 `--workers=auto` 选择并行，默认串行执行。旧配置中的
`extract.workers` YAML 项会被忽略，并在运行时提示。

CPU affinity 与 `SLURM_CPUS_PER_TASK` 共同限制进程数。主机可用内存、
cgroup 和 Slurm 内存额度约束执行预算，`memory_gb` 可进一步降低预算。
资源规划采用保守的工作内存估计，实际 RSS 可能超过估计值。
无法检测可用内存时，并行抽取需要显式设置 `memory_gb`。
并行效果取决于所选态与存储吞吐量。

例如，在已激活 ValleyScope 安装环境的条件下，提交一个任务、64 个 CPU
的 Slurm 作业：

```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
srun valleyscope extract-wavecar extract.yaml --workers=auto
```

内存申请量应根据输入和集群条件选择。并行抽取使用一个本地进程池，分阶段处理
G 列表重建与波函数系数，由单一写入进程生成 HDF5；执行依赖 Python
多进程机制和已有依赖包。计时输出分别报告 G 列表重建，以及合并的系数
读取、归一化与写入阶段耗时。

### 2. 沿采样路径投影能带

将[投影模板](examples/projection_bands_template.yaml)复制为 `projection.yaml`，
填写 HDF5 路径、物理母层晶格与层变换、谷中心、k 点名称和原始 VASP 能带
编号。各选定态保留完整平面波系数及适用的旋量分量。先运行投影，再绘制
序列化结果：

```bash
valleyscope project-bands projection.yaml
valleyscope plot-valley-bands valley_projection/valley_projection.json
valleyscope plot-valley-bands valley_projection/valley_projection.json \
  --output-directory figures --energy-reference-eV 0.0 --label-bands
```

投影写出 `valley_projection.json`、`valley_projection.txt` 和
`valley_weights.csv`。独立 JSON 文档使用 `document_kind: valley_projection`
和 `schema_version: 1.0.0`，记录各路径出现位置、原始能带编号、能量、范数及
原始/归一化权重；子空间、irrep、EBR 与陈数阶段均为 `not_requested`。
绘图命令读取这一 JSON，写出 `valley_fat_bands.png`、`valley_fat_bands.pdf`
与 `valley_fat_bands.metadata.json`。重新绘图只需保留 JSON。

各配置谷分别占一个面板，最后一个面板展示残余和重叠诊断。灰色能带线使
零谷权重的态仍然可见。全部面板共用能量与标记面积标尺：标记面积表示窗口
绝对概率除以完整态范数 `N = sum(abs(C)**2)`，并对适用的旋量分量求和。
各谷保留相对于完整态的覆盖率；配置谷之间不会再归一化到总和为一。
同一谷的多个中心按并集计算。`warn_exclude` 从单谷权重排除跨谷重叠，并以
`W_overlap` 单独记录；`error` 拒绝重叠窗口。权重满足
`W_val + W_overlap + W_res = N`。按中心记录的窗口属于可相互重叠的动量
诊断，其物理含义限于相应窗口的捕获权重。

`fixed_center` 使用最短笛卡尔母层周期距离，将 `k + G_M` 与固定物理母层
中心比较，严格采用 `distance < qcut`。可选比较模式
`k_resolved_parent_valley` 使用 `Q_a(k) = Q_a + (k - kfold_a)` 标记选定的
折叠动量族，输出该动量窗口的权重。参考布洛赫态跟踪和能带展开需要额外
物理数据。动态窗口重合时，各配置谷仍分别保留。HSP 就绪性继续采用固定
中心的初始投影算符。

`analysis.kpoints` 决定路径的出现顺序。`path.break_before` 使用从零开始的
出现位置索引，表示新分段的起点；单独重复端点仍属于当前分段。
`path.ticks` 使用 `{index, label}` 条目。距离由笛卡尔倒空间坐标计算，单位
为 Å⁻¹。通过 `plot.energy_reference_eV` 设置有限 eV 参考值（默认 `0.0`），
或用 `--energy-reference-eV` 覆盖；绘图元数据记录从全部能量扣除的参考值。
`--label-bands` 显示原始 VASP 能带编号。近简并本征态保留原始能量与本征矢；
简并块权重的迹提供与块内基底选择无关的解释。

### 3. 分析高对称点波函数

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
Valley Chern residues from rotation eigenvalues
Authoritative reduced EBR results
Readiness blockers and warnings
Public output files
```

阅读谷权重时，应结合子空间对称性和不可约表示结果。`qcut mode:` 说明动量
窗口的定义方式。JSON 中的 `valley_projection_summary` 对应谷投影摘要；
不可约表示、EBR 结果和未满足的条件分别记录在 `valley_resolved_irreps`、
`reduced_ebr_summary` 和 `readiness_blocker_summary` 中。
独立的 `valley_chern_mod` 报告记录条件性旋转余数及其适用性或缺失证据。

投影状态标签包括 `fixed_center_not_captured`、`not_derived` 和 `unreliable`。
固定中心窗口内权重低，表示所选窗口在该采样动量处捕获的态权重较小。
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

单层结构定义母层的倒空间坐标；莫尔或双层结构定义对称操作。

日常计算使用 `output.profile: standard`；需要检查投影残差、表示矩阵、
子空间闭合性、高对称点星或缝合矩阵时，使用 `output.profile: debug`。
源表与标准晶体学约定的高级选项可查阅命令帮助和
[`valleyscope/io/config.py`](valleyscope/io/config.py)，选择时应与输入的物理约定一致。

## 输出

`analyze-hsp` 的标准输出包括：

| 文件 | 用途 |
| --- | --- |
| `valley_summary.txt` | 谷权重、子空间对称性、不可约表示、条件性陈数余数、EBR 结果和未满足的条件 |
| `valley_summary.json` | 便于后续分析的结构化结果 |
| `valley_weights.csv` | 各 k 点、各 VASP 能带的原始谷权重 |
| `valley_ebr_export_bundle.json` | 不可约表示向量与相应对称性数据；至少一组 EBR 输入满足导出条件时写出 |
| `valley_reduced_ebr_mapping.json` | 启用并实际评估约化 EBR 匹配时写出 |

`valley_resolved_irreps` 为每个采样的 `(kpoint, valley)` 保留一条记录，
包含谷投影子空间的空间群、高对称点小群、谷保持操作、不可约表示重数
（`irrep_multiplicities`），以及允许或阻止表示赋值的条件。

摘要 schema `2.2.0` 在两种输出模式下都包含 `valley_chern_mod`。每条谷记录
保留状态、模数、余数、子空间秩、旋转操作，以及本征值证据或阻断原因。
缺失余数使用 `null`。EBR 导出、匹配和 ingestion 各自保留独立的 schema
与物理适用条件。

`valley_weights.csv` 便于快速筛查，但近简并子空间内单条能带的权重依赖
规范选择，应结合子空间分析解释。

详细输出还保留 `diagnostics.h5`、对称性报告、限制后的表示矩阵及不可约表示
和 EBR 的来源信息，便于检查为何未能确定某个表示。若没有态满足相应的物理
范围，某些表可能只有表头（header-only），相应结果记录在摘要中。
离线汇总除了摘要，还会读取单独的 EBR 导出文件和匹配文件，因此应将这些
文件保存在一起。

## 进行约化 EBR 匹配

约化 EBR 分析默认关闭。在 `analyze.yaml` 的 `analysis` 下加入
`reduced_ebr: {enabled: true}` 即可请求这一分析。程序可以从已安装的
`irreptables` 构造约化表，也可以读取用户提供且经过核对的约化表或映射说明。
无论采用哪种方式，群、晶体学设置、自旋约定、高对称点基底、不可约表示及
表格来源都必须与波函数计算一致。

这一分析应在 `analyze-hsp` 中完成，此时仍可利用波函数和数值表示进行检验。
下面的独立命令用于检查导出数据的相容性；物理验证使用 `analyze-hsp` 中的
波函数和数值表示：

```bash
valleyscope map-reduced-ebr \
  valley_ebr_export_bundle.json \
  validated_reduced_ebr_table.json \
  --output valley_reduced_ebr_mapping.json
```

约化表保留经过核对的来源和晶体学约定。

## 汇总多组计算结果

已完成的计算可汇总为单组 JSON 记录或多组计算的索引，便于比较：

```bash
valleyscope collect-database-record ./valley_analysis \
  --output ./database_ingestion_record.json

valleyscope collect-database-index ./run_a/valley_analysis ./run_b/valley_analysis \
  --output ./database_index.json
```

汇总时会区分最终 EBR 结果、不完整输入和被排除的结果，并检查它们与当前计算
摘要是否一致，使用的是已有计算保存的对称性结果。输入路径需要逐一指定。
数据库持久化和集群调度属于后续扩展，其方案将结合波函数的计算位置与
已完成计算的收集方式确定。

## 数值检验与当前结果

小规模构造的旋量波函数用于检验从谷投影、不可约表示赋值到精确约化 EBR
比较的完整数值计算。[`P3` 样例](tests/test_portable_numerical_chain.py)
检验一对时间反演相关的谷；[`P4mm` 样例](tests/test_noncommuting_numerical_chain.py)
检验 \(\Gamma\)、\(X\)、\(M\) 点处非对易的二维双值表示。
两套样例都包含归一化且谷纯度高、但不满足空间对称性闭合的态，相应结果
记录为被阻断。
P4mm 样例覆盖对称型空间群中的单谷子空间，包括非对易小群操作的作用。

这两套样例中的二维谷子空间分别给出模 3（P3）和模 4（P4mm）的条件性
零余数。破坏空间对称性闭合的对照则返回 `blocked`，余数为 `null`。
这些检查也会在安装后的程序上执行。另有行列式算术测试覆盖各个允许旋转阶数
及有、无自旋时的非零余数，用于检验旋转公式。

代码版本 `49cc142` 的本地材料回归计算得到：

| 计算体系 | 选定能带子空间的谷分辨结果 |
| --- | --- |
| tMoTe₂ | \(K\) 与 \(K'\) 两个谷的不可约表示向量均不属于所用约化 EBR 的整数张成（`outside_integer_span`） |
| tZrSe₂ | 三个 \(M\) 谷各自存在精确的非负整数约化 EBR 组合（`solved_exact`） |

对同一批保留的 tMoTe₂ 波函数及选定能带，后续旋转分析得到所选一维谷子空间的
\(C_K=1\pmod 3\)、\(C_{K'}=2\pmod 3\)，两者均为条件性结果。
其中一个所需高对称点的表示来自另一谷的已核对时间反演推断。
允许的整数分别为 \(C_K=1+3m\)、\(C_{K'}=2+3m'\)，其中 \(m,m'\) 为整数。

保留的 tMoTe₂ 波函数还用于错误 valley 投影对照：将 \(K/K'\) 中心替换为
\(\Gamma\) 或三个 \(M\) 中心。在三处采样高对称点及原有 q-cut 扫描中，
\(K/K'\) 窗口捕获选定态的大部分权重，而替代窗口均低于未改动的捕获阈值。
\(\Gamma\) 权重接近零；在原配置 q-cut 下，\(M\) 窗口的非零权重主要来自
与另一层周期性 \(K/K'\) 窗口的交集，表征这些动量窗口内捕获的权重。
这一检查同时考察单带权重和投影算符限制在完整目标子空间中的本征值谱。
[投影对照测试](tests/test_projection_controls.py) 包含一个小型构造样例，
用于检验这种跨层窗口重叠。

这些材料结果对应所述能带选择、高对称点和投影窗口。部分可选的时间反演
或联合灰群结果仍未确定。公开测试自行构造小型波函数，较大的材料计算及
其输出保存在本地。

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

测试会自行生成所需的小波函数文件，本地开发笔记的检查可显式启用。
以下命令还可对已提交版本构建独立安装包并验证：

```bash
python -m pip install build
python scripts/release_gate.py --checkout .
```

请在没有未提交改动的源码目录中运行；这一检查安装声明的官方依赖，
并可能下载这些包。安装后的 P3
和 P4mm 旋量对样例也检查整谷子空间的条件性旋转余数。空间对称性闭合失败时，
返回 `blocked`，余数为 `null`。这些检查考察采样点的小群表示及其旋转行列式。

命令定义见
[`valleyscope/cli.py`](valleyscope/cli.py)，配置解析见
[`valleyscope/io/config.py`](valleyscope/io/config.py)，输出选择见
[`valleyscope/reports/analysis_outputs.py`](valleyscope/reports/analysis_outputs.py)。
