# ValleyScope 公共 Schema 与 Benchmark Readiness 审计报告

日期: 2026-06-02
审计范围: 只读，当前 `main` 上的 post-merge 状态；parent-valley
projection 合并 commit 为 `8742fe8`。

## 审计问题 1: 主用户输出是否仅限于 valley_summary.txt/json？

**结论: 基本是，但有改进空间。**

`valley_summary.txt` 和 `valley_summary.json` 是明确标注的"主用户入口"。它们由 `summary_report.py` 的 `build_summary_payload()` 构建，包含所有分析层的聚合信息。

然而，在 detailed output 默认开启且相关分析层有数据时，输出目录会写出多种
intermediate/debug 文件（数量随配置和启用的分析层变化）。用户在 `README.md`
中看到的文档区分了三类输出：

| 类别 | 文件 | 状态 |
|------|------|------|
| 主入口 | `valley_summary.txt`, `valley_summary.json` | 明确 |
| 下游 EBR | `valley_ebr_export_bundle.json` | 明确 |
| 可选 EBR | `valley_reduced_ebr_mapping.json` (需 `analysis.reduced_ebr.enabled`) | 明确 |
| Formal Analysis | `symmetry_adapted_valley_analysis.json`, `valley_irrep_matching.json`, `irrep_workflow_decisions.json`, `projector_symmetry_report.json`, `target_subspace_closure.json` | 应该默认写但归类为中间产物 |
| Debug/Detail | `diagnostics.h5`, `valley_weights.csv`, `valley_subspace.json`, `symmetry_report.json`, `symmetry_eigenvalues.csv`, `valley_basis_transform.h5`, `hsp_star_conjugation.json`, `hsp_star_derived_characters.json`, `subspace_representation_quality.json` (可选), `folded_center_report.json`, `sampled_k_coverage.json` | 大部分正确归类 |

**建议**: `AGENTS.md` 的 Public Output Schema 已清楚分层。不需要改代码，
但建议在 `README.md` 中加强"主入口 vs 详细输出"的视觉区分。

---

## 审计问题 2: 下游 EBR 输出是否明确限制？

**结论: 是，严格限制。**

`analysis_outputs.py` 第 171-200 行明确控制 EBR 输出链:

1. **`valley_ebr_input_candidates.json`** — 仅收集 `readiness=trusted` + `path!=blocked` + `match_status=matched` + `!diagnostic_only` + `matched_irrep` 存在的行 (`ebr_input_candidates.py:68-81`)
2. **`valley_ebr_problem_instances.json`** — 仅打包 `ready_for_ebr_decomposition=True` + `status=complete` 的实例
3. **`valley_ebr_export_bundle.json`** — 模式版本 `1.0.0`，仅包含 trusted 实例
4. **`valley_reduced_ebr_mapping.json`** — 默认关闭 (`analysis.reduced_ebr.enabled: false`)；需要用户提供的外部表文件；使用精确整数分解

不包含任何内置 EBR 表、compatibility relations、或 heuristic decomposition。符合 `AGENTS.md` Hard Rules。

---

## 审计问题 3: Debug/Detail 输出是否正确归类？

**结论: 大多数正确。`folded_center_report.json` 和 `sampled_k_coverage.json` 需要确认归类。**

`AGENTS.md` 第 87-101 行列出了 debug/detail 输出。新增的 `folded_center_report.json` 和 `sampled_k_coverage.json` 已加入该列表。

| 文件 | 当前归类 | 正确性 |
|------|---------|--------|
| `diagnostics.h5` | Debug/Detail | ✅ 原始矩阵 dump |
| `valley_weights.csv` | Debug/Detail | ✅ 快速扫描 |
| `valley_subspace.json` | Debug/Detail | ✅ 子空间数据 |
| `symmetry_report.json` | Debug/Detail | ✅ 对称操作 |
| `symmetry_eigenvalues.csv` | Debug/Detail | ✅ 本征值 |
| `valley_basis_transform.h5` | Debug/Detail | ✅ 基变换 |
| `hsp_star_conjugation.json` | Debug/Detail | ✅ 共轭图 |
| `hsp_star_derived_characters.json` | Debug/Detail | ✅ 衍生特征标 |
| `subspace_representation_quality.json` | Debug/Detail (可选) | ✅ 默认关闭 |
| `target_subspace_closure.json` | Debug/Detail | ✅ 闭包诊断 |
| `folded_center_report.json` | Debug/Detail | ✅ 折叠中心 |
| `sampled_k_coverage.json` | Debug/Detail | ✅ k 点覆盖 |
| `projector_symmetry_report.json` | Debug/Detail | ✅ 投影仪对称性 |

**注意**: `projector_symmetry_report.json`、`target_subspace_closure.json`、`hsp_star_conjugation.json`、`hsp_star_derived_characters.json` 虽然在 debug 列表中，但它们是 EBR readiness gate 的关键输入。建议在文档中将它们标记为 "EBR diagnostic (debug/detail output)" 而非纯粹的 "debug"。

---

## 审计问题 4: Parent-valley projection 是否影响 irrep readiness？

**当前结论 (b31dc7a 实施后)**:

`k_resolved_parent_valley` 是严格的 **weight/report-only diagnostic**。

代码分离 (`analyze_hsp.py`):
- `reporting_projectors`: 使用 mode-adjusted centers → 仅用于 weights,
  center_weights, CSV, summary status
- `seed_projectors`: 始终使用 `fixed_center` → 用于 seed matrices, projector
  symmetry-consistency, symmetry-adapted diagnostics, irrep workflow
  decisions, EBR pipeline

`fixed_center_not_captured` 状态仅影响 `valley_subspace.json` 和 summary
中的显示文本（`_short_valley_status()` 映射），不改变任何 readiness gate。

<details>
<summary>审计历史: 原始审计发现及修正路径</summary>

原始审计时 (`3d9f03d`) 发现 `projector_mode` 通过以下路径影响 readiness:
`projector_mode → effective_centers → seed_matrices → projector_symmetry →
irrep_workflow`。具体来说:
1. `projector_mode` → `adjust_centers_for_parent_valley()` 调整 center 位置
2. → `build_sector_projectors()` 产生 masks
3. → `_add_valley_subspace_diagnostic()` 使用 masks 生成 seed matrices
4. → `build_projector_symmetry_report()` 检查种子投影仪对称性
5. → `build_irrep_workflow_decisions()` 读取 seed symmetry status

提交 `b31dc7a` 通过拆分 reporting/seed projectors 切断了路径 3-5:
readiness 路径始终使用 fixed-center seed projectors。

</details>

---

## 审计问题 5: 最小 Benchmark Matrix

### tMoTe2 (P321, K/K' valleys)

| Kpoint | Valley | Workflow Path | Readiness | EBR Ready | Blocker | 预期冻结 |
|--------|--------|---------------|-----------|-----------|---------|----------|
| GammaM | K | direct_qcut | trusted | true | none | ✅ trusted |
| GammaM | K' | direct_qcut | trusted | true | none | ✅ trusted |
| KM | K | direct_qcut | trusted | true | none | ✅ trusted |
| KM | K' | direct_qcut | trusted | true | none | ✅ trusted |
| MM | K | symmetry_adapted | blocked | false | hsp_star_derivation_not_available | ⚠️ blocked (物理: 仅 identity) |
| MM | K' | symmetry_adapted | blocked | false | hsp_star_derivation_not_available | ⚠️ blocked (物理: 仅 identity) |

### tZrSe2 (P312, M-star valleys)

| Kpoint | Valley | Workflow Path | Readiness | EBR Ready | Blocker(s) | 预期冻结 |
|--------|--------|---------------|-----------|-----------|------------|----------|
| GammaM | M1 | symmetry_adapted | diagnostic_only | false | B1(spinor), B3(low_seed=0.74) | 🔴 blocked |
| GammaM | M2 | symmetry_adapted | diagnostic_only | false | B1(spinor), B3(low_seed=0.66) | 🔴 blocked |
| GammaM | M3 | symmetry_adapted | diagnostic_only | false | B1(spinor), B3(low_seed=0.71), B2(closure) | 🔴 blocked |
| KM | M1 | symmetry_adapted | blocked | false | B1, B5(no VP ops), B4(cascade) | 🔴 blocked |
| KM | M2 | symmetry_adapted | blocked | false | B1, B5(no VP ops), B4(cascade) | 🔴 blocked |
| KM | M3 | symmetry_adapted | blocked | false | B1, B5(no VP ops), B4(cascade) | 🔴 blocked |
| MM | M1 | symmetry_adapted | blocked | false | B1, B5(no VP ops), B4(cascade) | 🔴 blocked |
| MM | M2 | symmetry_adapted | blocked | false | B1, B5(no VP ops), B4(cascade) | 🔴 blocked |
| MM | M3 | symmetry_adapted | diagnostic_only | false | B1(spinor), B2(closure=1.9e-2) | 🔴 blocked |

Blocker 优先级: B1 (spinor) > B2 (closure) > B3 (seed) > B4 (cascade) > B5 (physics)

### PdSe2 / tPdSe2 (88.5° NHSP, V0/V0p valleys)

| Kpoint | Valley | fixed_center W_val | k_resolved_parent_valley W_val | 预期冻结 |
|--------|--------|---------------------|-------------------------------|----------|
| Gamma | V0/V0p | 0 (fixed_center_not_captured) | 0.07-0.18 (P_v=0.5) | 🟡 diagnostic_only |
| kp1 | V0/V0p | 0.34-0.74 | 0.09-0.24 | 🟡 diagnostic_only |
| kp2 | V0/V0p | ~0.70-0.85 | ~0.10-0.25 | 🟡 diagnostic_only |
| kp3 | V0/V0p | ~0.70-0.85 | ~0.10-0.25 | 🟡 diagnostic_only |

**注意**: PdSe2 目前仅用于 smoke test，没有 symmetry analysis 配置。不应加入正式的 benchmark matrix 直到:
- 配置 symmetry.operations.structure_file
- 完成自旋约定验证
- 确定 q-cut 参数

---

## 审计问题 6: 下一步安全行动

### 优先级排序

| 优先级 | 行动 | 理由 | 风险 |
|--------|------|------|------|
| **P1** | Schema 冻结文档 + 回归测试 | 合并后 schema 需要正式文档化；readiness 边界需要测试锁定 | 低 |
| **P2** | tMoTe2 C3 irrep table 审查 | 已有 4 个 trusted candidates (GammaM/KM × K/K')，需要审查后发布 | 低 (已有数据) |
| **P3** | tZrSe2 spinor 约定验证 | B1 是所有 tZrSe2 路径的根阻塞器 | 需要外部 benchmark |
| **P4** | tZrSe2 expanded-band HDF5 | B2 (closure) 可能由截断效应引起 | 需要额外 DFT 计算 |
| **P5** | tZrSe2 GammaM q-cut 优化 | B3 (seed overlap) 可通过 q-cut 扫描改善 | 低 (参数扫描) |
| **P6** | Reviewed EBR table 摄入 | 仅当有审核过的外部表时才安全 | 高 (无表则无输出) |

注: P0 (parent-valley readiness 边界) 已于 `b31dc7a` 解决。

### 推荐下一轮 cc 任务

```text
1. 添加 readiness-boundary 回归测试: 验证 k_resolved_parent_valley 模式
   改变 reporting weights 但不改变 seed matrices / projector symmetry /
   irrep-EBR readiness。

2. 更新 AGENTS.md/README.md/PLAN.md: 明确 parent-valley mode 是
   reporting-only；readiness seed projectors 始终使用 fixed_center。

3. 冻结 tMoTe2 C3 benchmark: 审查 irreps/tables.py 中的自旋 C3 表，
   运行完整 tMoTe2 benchmark 并记录精确 irrep 标签。更新
```

### 不应做的事

- ❌ 不添加 EBR tables (内置未审核表)
- ❌ 不添加 compatibility relations
- ❌ 不添加 heuristic decomposition
- ❌ 不添加材料特例逻辑
- ❌ 不放宽容差掩盖 tZrSe2 blockers
- ❌ 不将 PdSe2 加入正式 benchmark matrix（缺少 symmetry config）

---

## 验证命令

```bash
pytest -q
# 481 passed

pytest -q tests/test_projection.py
# 22 passed

pytest -q tests/test_io_and_workflow.py
# 66 passed

git diff --check HEAD
# clean
```

## 文件引用索引

| 目的 | 文件 |
|------|------|
| 输出 schema 定义 | `AGENTS.md:79-101` |
| 输出编排器 | `valleyscope/reports/analysis_outputs.py:22-313` |
| 摘要构建 | `valleyscope/reports/summary_report.py:15-117` |
| EBR 输入门控 | `valleyscope/analysis/ebr_input_candidates.py:68-81` |
| Irrep 工作流决策 | `valleyscope/analysis/irrep_workflow_decision.py:25-180` |
| tMoTe2 benchmark | `docs/benchmarks/symmetry_adapted_valley_smoke.md:74-96` |
| tZrSe2 benchmark | `docs/benchmarks/symmetry_adapted_valley_smoke.md:97-138` |
| tZrSe2 blocker 证据 | `docs/benchmarks/tzrse2_blocker_evidence.md:1-178` |
| Parent-valley 投影 | `valleyscope/projection/sector_projectors.py:101-155` |
| 投影模式配置 | `valleyscope/io/config.py:31-50` |
