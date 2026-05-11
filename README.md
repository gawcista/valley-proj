# ValleyScope

**Language:** [English overview](#overview) | [中文说明](#zh-cn)

## Overview

ValleyScope provides valley-resolved wavefunction, subspace, and symmetry diagnostics for first-principles band structures. Its current workflow analyzes VASP-derived moire wavefunctions at selected high-symmetry points, constructs momentum-space projectors onto user-defined monolayer valley sectors, diagnoses whether a target state or subspace is valley-resolved, and computes symmetry representation data needed for rotation-eigenvalue analysis.

V1 is intentionally not a black-box Chern-number code. It reports checkable intermediate quantities: valley weights, valley purity, leakage, cross-sector overlap diagnostics, valley-adapted basis transforms, little-group checks, valley-preservation checks, and rotation eigenvalues. Berry curvature, Wilson loops, full mBZ diagnostics, automatic monolayer valley search, true layer-resolved projection, and automatic integer Chern inference are outside the V1 scope.

### Physical Definitions

For a moire Bloch state at moire momentum $\mathbf k_M$, the plane-wave expansion is

```math
\psi_{n\mathbf k_M}(\mathbf r) = \sum_{\mathbf G_M,s} c_{n,\mathbf k_M+\mathbf G_M,s} e^{i(\mathbf k_M+\mathbf G_M)\cdot \mathbf r} |s\rangle .
```

Here $\mathbf G_M$ is a moire reciprocal lattice vector, $s$ is the spinor index, and

```math
\mathbf q = \mathbf k_M+\mathbf G_M
```

is the actual plane-wave momentum. Valley projection uses the in-plane component $\mathbf q_\parallel$. The valley label refers to a monolayer valley sector, not to a high-symmetry point of the moire Brillouin zone.

A valley center $\mathbf Q_a$ is a monolayer valley momentum after the appropriate layer rotation and coordinate conversion. The distance from a plane-wave component to that center is

```math
d_a(\mathbf q) = \min_{\mathbf G_{\rm mono}} \left| \mathbf q_\parallel - (\mathbf Q_a+\mathbf G_{\rm mono}) \right|.
```

The corresponding momentum window is

```math
\Omega_a(q_{\rm cut})=\{\mathbf q: d_a(\mathbf q)<q_{\rm cut}\}.
```

For a valley sector $\mathcal V$ containing several centers, for example top and bottom layer centers belonging to the same monolayer valley, the sector window is the union

```math
\Omega_{\mathcal V} = \bigcup_{a\in\mathcal V}\Omega_a .
```

The sector projector is therefore

```math
P_{\mathcal V} = \sum_{\mathbf q\in\Omega_{\mathcal V},s} |\mathbf q,s\rangle\langle \mathbf q,s|.
```

The union is important: if two centers inside the same sector overlap, a plane-wave component is counted once. If a component lies in windows belonging to different sectors, V1 treats it as a cross-sector overlap. With the default `ambiguous_cross_sector: warn_exclude`, that component is excluded from all sector weights and reported separately.

For a normalized state $|\psi\rangle$, the sector weight is

```math
W_{\mathcal V_i} = \langle\psi|P_{\mathcal V_i}|\psi\rangle .
```

The total weight inside the user-defined valley manifold is

```math
W_{\rm val} = \sum_i W_{\mathcal V_i},
```

and the multi-sector valley purity is

```math
P_v = \frac{\max_i W_{\mathcal V_i}}{\sum_i W_{\mathcal V_i}}, \qquad \sum_i W_{\mathcal V_i}>0 .
```

For two sectors $K$ and $K'$, the signed valley polarization is also reported:

```math
\eta = \frac{W_K-W_{K'}}{W_K+W_{K'}} .
```

If cross-sector overlaps are excluded, the output field `ambiguous_weight` should be read as the cross-sector overlap weight

```math
W_\times = \langle\psi|P_\times|\psi\rangle,
```

where $P_\times$ projects onto plane-wave components selected by more than one valley sector under the chosen $q_{\rm cut}$. The reported `leakage` is the weight outside the selected valley windows after separating this overlap:

```math
L_{\rm out}=1-W_{\rm val}-W_\times .
```

The total weight not assigned to any sector is therefore $1-W_{\rm val}=L_{\rm out}+W_\times$. Thus $W_\times$ is a numerical diagnostic of the valley-window definition, not a new condensed-matter observable. In prose, use "cross-sector overlap weight"; `ambiguous_weight` is the current serialized output key.

### Degenerate Subspaces

For an isolated nondegenerate eigenstate, the original VASP eigenvector can be assigned a valley character when $W_{\rm val}$ and $P_v$ are sufficiently large. For a degenerate or nearly degenerate subspace, individual VASP eigenvectors are gauge-dependent and should not be interpreted directly.

For a two-sector subspace spanned by $\{|\psi_i\rangle\}_{i=1}^N$, V1 constructs

```math
S_{ij} = \langle\psi_i|P_K+P_{K'}|\psi_j\rangle ,
```

```math
V_{ij} = \langle\psi_i|P_K-P_{K'}|\psi_j\rangle .
```

Diagonalizing $V$, or the corresponding $S$-orthogonalized problem when needed, gives valley-adapted combinations

```math
|\phi_\alpha\rangle = \sum_i |\psi_i\rangle U_{i\alpha}.
```

Their valley-manifold weight and signed polarization are

```math
W_\alpha = \langle\phi_\alpha|S|\phi_\alpha\rangle , \qquad \eta_\alpha = \frac{\langle\phi_\alpha|V|\phi_\alpha\rangle} {\langle\phi_\alpha|S|\phi_\alpha\rangle}.
```

For more than two sectors, V1 uses the projected sector matrices

```math
M_{\mathcal V_i} = P_{\rm tar}P_{\mathcal V_i}P_{\rm tar},
```

where $P_{\rm tar}$ is the projector onto the target band subspace.

### Symmetry and Rotation Eigenvalues

Symmetry operations are read from `spglib` using the moire POSCAR/CONTCAR. In direct fractional coordinates, `spglib` uses

```math
\mathbf x' = W\mathbf x+\mathbf w .
```

For reciprocal fractional momenta, V1 applies

```math
\mathbf k' = W^{-T}\mathbf k .
```

An operation belongs to the little group of a target high-symmetry point $\mathbf k$ only if

```math
W^{-T}\mathbf k-\mathbf k \in \mathbb Z^3
```

within tolerance. Before using a rotation for a single-valley eigenvalue, V1 also checks valley preservation:

```math
gP_{\mathcal V}g^{-1}\approx P_{\mathcal V}.
```

Operations that map the target sector to another sector are reported but rejected for single-valley rotation-eigenvalue analysis.

For valley-adapted states, the rotation representation matrix is

```math
D^{(\tau)}_{mn}(g;\mathbf k) = \langle \phi_m^\tau(g\mathbf k) | \hat g | \phi_n^\tau(\mathbf k) \rangle .
```

At a high-symmetry point where $g\mathbf k=\mathbf k+\mathbf G_M$, the eigenvalues of $D^{(\tau)}$ are the reported rotation eigenvalues. These data can constrain $C_\tau \bmod n$ for a compatible $C_n$ formula, but they do not determine a full integer Chern number by themselves.

### HDF5 Input

The analyzer reads an HDF5 intermediate file with one group per k point, because the number of plane waves can vary across k points:

```text
/metadata/lattice/direct_cart              [3,3]
/metadata/lattice/reciprocal_cart          [3,3]
/metadata/spinor                           bool
/metadata/source                           string
/metadata/vasp_band_index_base             int

/kpoints/0/name                            string
/kpoints/0/frac                            [3]
/kpoints/0/cart                            [3]
/kpoints/0/g_vectors_frac                  [nG,3]
/kpoints/0/g_vectors_cart                  [nG,3]
/kpoints/0/coefficients                    [nb,nspinor,nG]
/kpoints/0/energies_eV                     [nb]
/kpoints/0/band_indices_vasp               [nb]
```

Band indices in YAML use VASP 1-based convention. The code converts to Python indexing internally.

### YAML Configuration

Use `examples/config_template.yaml` as a starting point. The monolayer reciprocal lattice must come from either explicit `monolayer_lattices.*.reciprocal_cart` entries or user-provided `monolayer_poscars` plus `layer_transforms`. The moire POSCAR is used for symmetry diagnostics; it is not the default source of monolayer reciprocal lattice vectors.

Key sections are:

```yaml
input:
  wavefunction_h5: ./selected_wavefunctions.h5
  poscar: ./CONTCAR

analysis:
  kpoints: [GammaM, KM, MM]
  target_bands_vasp: [101, 102]
  degeneracy_tol_meV: 1.0

valley_centers:
  coordinate_mode: cart
  centers:
    - name: top_K
      cart: [...]

valley_sectors:
  - name: K_sector
    centers: [top_K, bottom_K]

projection:
  qcut_mode: moire_shell
  qcut_shell: 3
  ambiguous_cross_sector: warn_exclude

symmetry:
  source: spglib
  poscar: ./CONTCAR
  symprec: 1.0e-3
  symprec_scan: [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3]
  proper_rotations_only: true
  allowed_orders: [2, 3, 4, 6]
```

### Running the Workflow

If starting from a VASP `WAVECAR`, first extract selected k points and bands:

```bash
valleyscope extract-wavecar extract.yaml
```

Then run the HSP analysis:

```bash
valleyscope analyze-hsp config.yaml
```

From the source tree, the equivalent command is:

```bash
python -m valleyscope.cli analyze-hsp config.yaml
```

Outputs are written under `output.directory`:

```text
valley_weights.csv
valley_subspace.json
rotation_eigenvalues.csv
symmetry_report.json
valley_basis_transform.h5
diagnostics.h5
```

`valley_weights.csv` contains one row per target k point and band. `diagnostics.h5` stores sector masks, center masks, cross-sector overlap masks, and q-cut metadata for later inspection. `qcut_scan` can be used to check whether $W_{\rm val}$, $P_v$, and $W_\times$ are stable against the projection radius.

### WAVECAR Extractor

The extractor is deliberately narrow and validation-heavy. It supports standard record-based `WAVECAR` files with RTAG values `45200`, `45210`, `53300`, or `53310`. It extracts selected VASP 1-based k-point and band indices into the V1 HDF5 schema.

For collinear `ISPIN=2`, set `extract.spin_index` and extract one spin channel at a time. For noncollinear/SOC records, the extractor detects `[2,nG]` spinor coefficients when the coefficient record length is `2*nplane`.

After extraction, verify that the coefficient shape is `[nb,nspinor,nG]`, the G-vector arrays have length `nG`, the stored norms are close to 1, and the energies match the intended VASP bands.

<a id="zh-cn"></a>

## 中文说明

ValleyScope 面向第一性原理能带结构，提供 valley-resolved wavefunction、subspace 与 symmetry diagnostics。当前工作流用于分析 VASP 计算得到的 moire 超胞波函数：在给定高对称点处构造面向 monolayer valley sectors 的动量空间投影算符，诊断目标态或目标子空间是否可以被可靠地分辨为 valley 态，并输出旋转本征值分析所需的对称性表示矩阵。

V1 不是黑箱 Chern number 计算器。它输出可检查的中间物理量：valley weights、valley purity、leakage、跨 sector 重叠诊断、valley-adapted basis 变换、little-group 检查、valley-preservation 检查和 rotation eigenvalues。Berry curvature、Wilson loop、完整 mBZ 诊断、自动 monolayer valley 搜索、真正 layer-resolved projection，以及自动完整整数 Chern number 推断均不属于 V1 范围。

### 物理定义

moire 超胞 Bloch 态在 $\mathbf k_M$ 处的平面波展开写为

```math
\psi_{n\mathbf k_M}(\mathbf r) = \sum_{\mathbf G_M,s} c_{n,\mathbf k_M+\mathbf G_M,s} e^{i(\mathbf k_M+\mathbf G_M)\cdot \mathbf r} |s\rangle .
```

其中 $\mathbf G_M$ 是 moire 倒格矢，$s$ 是 spinor index，

```math
\mathbf q = \mathbf k_M+\mathbf G_M
```

是实际平面波动量。valley projection 只使用面内动量 $\mathbf q_\parallel$。这里的 valley 指 monolayer valley sector，不是 moire Brillouin zone 中的高对称点。

valley center $\mathbf Q_a$ 是经过层旋转和坐标转换后的单层 valley 动量。平面波分量到该 center 的距离定义为

```math
d_a(\mathbf q) = \min_{\mathbf G_{\rm mono}} \left| \mathbf q_\parallel - (\mathbf Q_a+\mathbf G_{\rm mono}) \right|.
```

对应的 valley window 为

```math
\Omega_a(q_{\rm cut})=\{\mathbf q: d_a(\mathbf q)<q_{\rm cut}\}.
```

一个 valley sector $\mathcal V$ 可以包含多个 centers，例如同一 monolayer valley 的 top/bottom layer centers。该 sector 的窗口取并集：

```math
\Omega_{\mathcal V} = \bigcup_{a\in\mathcal V}\Omega_a .
```

sector projector 定义为

```math
P_{\mathcal V} = \sum_{\mathbf q\in\Omega_{\mathcal V},s} |\mathbf q,s\rangle\langle \mathbf q,s|.
```

并集计数可以避免同一 sector 内的重叠窗口重复统计。如果某个 plane-wave component 同时落入不同 valley sectors 的窗口，V1 将其视为跨 sector 重叠。默认 `ambiguous_cross_sector: warn_exclude` 会把该分量从所有 sector weights 中排除，并单独报告。

对归一化态 $|\psi\rangle$，sector weight 为

```math
W_{\mathcal V_i} = \langle\psi|P_{\mathcal V_i}|\psi\rangle .
```

用户定义的 valley manifold 中的总权重为

```math
W_{\rm val} = \sum_i W_{\mathcal V_i},
```

多 sector 情形下的 valley purity 定义为

```math
P_v = \frac{\max_i W_{\mathcal V_i}}{\sum_i W_{\mathcal V_i}}, \qquad \sum_i W_{\mathcal V_i}>0 .
```

两 valley 情形还输出有符号 valley polarization：

```math
\eta = \frac{W_K-W_{K'}}{W_K+W_{K'}} .
```

如果跨 sector 重叠被排除，输出字段 `ambiguous_weight` 应理解为 cross-sector overlap weight：

```math
W_\times = \langle\psi|P_\times|\psi\rangle,
```

其中 $P_\times$ 投影到在当前 $q_{\rm cut}$ 下被多个 valley sectors 同时选中的 plane-wave components。此时报告中的 `leakage` 是分离出该重叠后、位于所选 valley windows 之外的权重：

```math
L_{\rm out}=1-W_{\rm val}-W_\times .
```

因此，总的未分配到任何 sector 的权重为 $1-W_{\rm val}=L_{\rm out}+W_\times$。$W_\times$ 是对 valley-window 选择的数值诊断，不是新的凝聚态物理可观测量。自然语言中建议称为“cross-sector overlap weight”或“跨 sector 重叠权重”；`ambiguous_weight` 只是当前输出文件中的字段名。

### 简并子空间

对孤立非简并态，只要 $W_{\rm val}$ 和 $P_v$ 足够高，就可以直接给原始 VASP 本征矢赋予 valley character。对严格简并或近简并子空间，单个 VASP 本征矢依赖任意酉变换 gauge，不能逐个直接解释。

设两 valley sector 子空间由 $\{|\psi_i\rangle\}_{i=1}^N$ 张成，V1 构造

```math
S_{ij} = \langle\psi_i|P_K+P_{K'}|\psi_j\rangle ,
```

```math
V_{ij} = \langle\psi_i|P_K-P_{K'}|\psi_j\rangle .
```

对 $V$ 或相应的 $S$-正交化问题对角化，得到 valley-adapted basis：

```math
|\phi_\alpha\rangle = \sum_i |\psi_i\rangle U_{i\alpha}.
```

每个新基矢的 valley-manifold weight 和有符号 polarization 为

```math
W_\alpha = \langle\phi_\alpha|S|\phi_\alpha\rangle , \qquad \eta_\alpha = \frac{\langle\phi_\alpha|V|\phi_\alpha\rangle} {\langle\phi_\alpha|S|\phi_\alpha\rangle}.
```

多 valley sector 情形使用 projected sector matrices：

```math
M_{\mathcal V_i} = P_{\rm tar}P_{\mathcal V_i}P_{\rm tar},
```

其中 $P_{\rm tar}$ 是目标 band 子空间的投影算符。

### 对称性与旋转本征值

对称操作由 `spglib` 从 moire POSCAR/CONTCAR 识别。在 direct fractional coordinates 中，`spglib` 的约定是

```math
\mathbf x' = W\mathbf x+\mathbf w .
```

对 reciprocal fractional momenta，V1 使用

```math
\mathbf k' = W^{-T}\mathbf k .
```

一个操作属于目标高对称点 $\mathbf k$ 的 little group，当且仅当

```math
W^{-T}\mathbf k-\mathbf k \in \mathbb Z^3
```

在给定容差内成立。进一步，在使用某个旋转计算 single-valley eigenvalue 前，还必须检查 valley preservation：

```math
gP_{\mathcal V}g^{-1}\approx P_{\mathcal V}.
```

如果该操作把目标 valley sector 映射到另一个 sector，程序会报告该映射，但不会用它计算 single-valley rotation eigenvalue。

对 valley-adapted states，旋转表示矩阵定义为

```math
D^{(\tau)}_{mn}(g;\mathbf k) = \langle \phi_m^\tau(g\mathbf k) | \hat g | \phi_n^\tau(\mathbf k) \rangle .
```

在满足 $g\mathbf k=\mathbf k+\mathbf G_M$ 的高对称点处，$D^{(\tau)}$ 的本征值就是输出的 rotation eigenvalues。这些本征值可以在适用的 $C_n$ 公式中约束 $C_\tau \bmod n$，但不能单独给出完整整数 Chern number。

### HDF5 输入

分析核心读取 HDF5 中间格式。由于不同 k 点的 plane-wave 数量可能不同，文件按 k point 分组：

```text
/metadata/lattice/direct_cart              [3,3]
/metadata/lattice/reciprocal_cart          [3,3]
/metadata/spinor                           bool
/metadata/source                           string
/metadata/vasp_band_index_base             int

/kpoints/0/name                            string
/kpoints/0/frac                            [3]
/kpoints/0/cart                            [3]
/kpoints/0/g_vectors_frac                  [nG,3]
/kpoints/0/g_vectors_cart                  [nG,3]
/kpoints/0/coefficients                    [nb,nspinor,nG]
/kpoints/0/energies_eV                     [nb]
/kpoints/0/band_indices_vasp               [nb]
```

YAML 配置中的 band index 使用 VASP 1-based convention，代码内部再转换为 Python index。

### YAML 配置

建议从 `examples/config_template.yaml` 开始。monolayer reciprocal lattice 必须来自显式的 `monolayer_lattices.*.reciprocal_cart`，或来自用户提供的 `monolayer_poscars` 与 `layer_transforms`。moire POSCAR 用于 symmetry diagnostics，不默认作为 monolayer reciprocal lattice 的来源。

关键配置段如下：

```yaml
input:
  wavefunction_h5: ./selected_wavefunctions.h5
  poscar: ./CONTCAR

analysis:
  kpoints: [GammaM, KM, MM]
  target_bands_vasp: [101, 102]
  degeneracy_tol_meV: 1.0

valley_centers:
  coordinate_mode: cart
  centers:
    - name: top_K
      cart: [...]

valley_sectors:
  - name: K_sector
    centers: [top_K, bottom_K]

projection:
  qcut_mode: moire_shell
  qcut_shell: 3
  ambiguous_cross_sector: warn_exclude

symmetry:
  source: spglib
  poscar: ./CONTCAR
  symprec: 1.0e-3
  symprec_scan: [1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3]
  proper_rotations_only: true
  allowed_orders: [2, 3, 4, 6]
```

### 运行流程

如果输入来自 VASP `WAVECAR`，先抽取指定 k points 和 bands：

```bash
valleyscope extract-wavecar extract.yaml
```

随后运行 HSP 分析：

```bash
valleyscope analyze-hsp config.yaml
```

在源码树中也可以使用：

```bash
python -m valleyscope.cli analyze-hsp config.yaml
```

输出写入 `output.directory`：

```text
valley_weights.csv
valley_subspace.json
rotation_eigenvalues.csv
symmetry_report.json
valley_basis_transform.h5
diagnostics.h5
```

`valley_weights.csv` 每行对应一个目标 k point 和 band。`diagnostics.h5` 保存 sector masks、center masks、cross-sector overlap masks 和 q-cut metadata，便于后续检查。`qcut_scan` 可用于检查 $W_{\rm val}$、$P_v$ 和 $W_\times$ 是否随 projection radius 稳定。

### WAVECAR Extractor

extractor 的范围刻意保持较窄，并进行严格校验。它支持标准 record-based `WAVECAR`，RTAG 可为 `45200`、`45210`、`53300` 或 `53310`。它将指定的 VASP 1-based k-point 和 band indices 抽取到 V1 HDF5 schema。

对 collinear `ISPIN=2`，需要设置 `extract.spin_index` 并一次抽取一个 spin channel。对 noncollinear/SOC records，如果 coefficient record length 为 `2*nplane`，extractor 会识别 `[2,nG]` spinor coefficients。

抽取后应检查：`coefficients` 的 shape 为 `[nb,nspinor,nG]`，G-vector 数组长度等于 `nG`，保存的 norms 接近 1，并且 energies 对应目标 VASP bands。
