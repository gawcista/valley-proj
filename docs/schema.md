# ValleyScope Public Output Schema

Version: 1.0.0 | Date: 2026-06-02

This document defines the **public output schema** for ValleyScope. Files not
listed here are debug/detail or intermediate diagnostics and may change
without a schema version bump.

## Public User Entry

### `valley_summary.txt`

Human-readable plain-text summary. Always written. Content mirrors
`valley_summary.json` in text form.

### `valley_summary.json`

Machine-readable summary. Always written.

Top-level fields (always present):

| Field | Type | Description |
|-------|------|-------------|
| `input` | object | Input configuration summary |
| `input.wavefunction_h5` | string | Path to wavefunction HDF5 |
| `input.operation_structure_file` | string\|null | Path to structure file for symmetry detection |
| `input.operation_detection_backend` | string | `"spglib"` |
| `input.spinor_convention` | string | `"vasp_up_down_saxis_z"` |
| `input.spinor_convention_verified` | bool | Whether spinor convention is benchmark-verified |
| `input.spinor_benchmark` | string\|null | Benchmark label if verified |
| `target_kpoints` | list[string] | Analyzed HSP labels |
| `iband` | list[int] | VASP band indices |
| `valley_subspaces` | list[object] | Each: `{"label": string, "centers": [string]}` |
| `qcut` | object | Projection parameters |
| `qcut.projector_mode` | string | `"fixed_center"` or `"k_resolved_parent_valley"` |
| `qcut.mode` | string | Radius mode: `"absolute"`, `"moire_shell"`, or `"relative_min_valley_distance"` |
| `qcut.value_Ainv` | number | Effective qcut radius in Å⁻¹ |
| `qcut.scan` | list[number] | qcut scan fractions or absolute values |
| `qcut.fraction` | number | Only when `mode == "relative_min_valley_distance"` |
| `valley_projection_summary` | list[object] | Per-(kpoint, band) projection summary |
| `valley_projection_summary[].kpoint` | string | HSP label |
| `valley_projection_summary[].band_vasp` | int | VASP band index |
| `valley_projection_summary[].W_val` | number | Valley-subspace weight |
| `valley_projection_summary[].P_v` | number | Valley purity |
| `valley_projection_summary[].eta` | number\|null | Signed valley polarization (two-valley only) |
| `valley_projection_summary[].W_overlap` | number | Projector-window overlap weight |
| `valley_projection_summary[].W_res` | number | Residual (out-of-valley) weight |
| `valley_projection_summary[].status` | string | `clean`, `approx`, `mixed`, `not_derived`, `fixed_center_not_captured`, `unreliable`, `n/a` |
| `valley_subspace_analysis` | list[object] | Per-kpoint subspace diagnostics |
| `valley_subspace_analysis[].kpoint` | string | HSP label |
| `valley_subspace_analysis[].basis_status` | string | Subspace classification |
| `valley_subspace_analysis[].S_min` | number | Minimum target-valley-subspace weight |
| `valley_subspace_analysis[].S_max` | number | Maximum target-valley-subspace weight |
| `valley_subspace_analysis[].min_concentration` | number | Minimum valley concentration in adapted basis |
| `valley_subspace_analysis[].assigned_valleys` | list[string] | Valley assignment per adapted state |
| `valley_subspace_analysis[].status` | string | `clean`, `approx`, `mixed`, `not_derived`, `unreliable`, `n/a` |
| `valley_projector_quality` | list[object] | Per-kpoint q-cut seed projector quality |
| `symmetry_analysis` | object | Detected symmetry operations, little groups, valley-preserving subgroups |
| `symmetry_eigenvalues` | list[object] | Per-(kpoint, op, valley, state) eigenvalue rows |
| `symmetry_characters` | list[object] | Per-(kpoint, valley, op) character rows |
| `rotation_readiness_thresholds` | object | Active rotation readiness thresholds |
| `warnings` | list[string] | Collected diagnostic warnings |
| `output_files` | object | Map of output key to file path |
| `legend` | object | Human-readable descriptions of key fields |

Conditionally present top-level fields:

| Field | Condition |
|-------|-----------|
| `symmetry_eigenvalue_summary` | Symmetry detection enabled |
| `projector_symmetry` | Projector symmetry report available |
| `symmetry_adapted_valley_analysis` | Symmetry-adapted valley analysis enabled (default) |
| `target_subspace_closure` | D_raw closure report available |
| `hsp_star_conjugation` | HSP-star conjugation report available |
| `hsp_star_derived_characters` | Derived character report available |
| `irrep_workflow_decisions` | Irrep workflow decision layer active |
| `valley_irrep_matching` | Irrep matching complete |
| `valley_ebr_input_candidates` | EBR input candidates collected |
| `valley_ebr_problem_instances` | EBR problem instances built |
| `valley_ebr_export_bundle` | EBR export bundle built |
| `valley_reduced_ebr_mapping` | Reduced EBR mapping enabled with valid table |
| `folded_center_report` | Always (when moire lattice available) |
| `sampled_k_coverage` | Always (when moire lattice available) |

---

## Downstream EBR Entry

### `valley_ebr_export_bundle.json`

The public EBR input for external solvers. Schema version `"1.0.0"`.

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"no_bundles"`, `"ready_for_external_solver"`, `"partial_export"` |
| `bundle_count` | int | Number of exported bundles |
| `excluded_count` | int | Number of excluded instances |
| `schema_version` | string | `"1.0.0"` |
| `reduced_ebr_decomposition_status` | string | `"not_implemented"` |
| `interpretation` | string | Human-readable status explanation |
| `bundles` | list[object] | Trusted EBR input bundles |
| `excluded_instances` | list[object] | Instances excluded from export |

Each bundle:

| Field | Type | Description |
|-------|------|-------------|
| `bundle_id` | string | Unique bundle identifier |
| `source_instance_id` | string | Source problem instance ID |
| `valley` | string | Valley label |
| `subspace_group_candidate` | string | Subspace space-group candidate (e.g. `"P3"`, `"P2"`) |
| `workflow_path` | string | `"direct_qcut"` or `"symmetry_adapted"` |
| `readiness_level` | string | `"trusted"` |
| `irreps_by_kpoint` | object | Per-kpoint irrep labels |
| `operations_by_kpoint` | object | Per-kpoint operation IDs |
| `expected_hsps` | list[string] | Required HSP labels for this group |
| `optional_hsps` | list[string] | Optional HSP labels |
| `missing_optional_hsps` | list[string] | Optional HSPs not in input data |
| `ready_for_external_solver` | bool | Always `true` for exported bundles |

Inclusion gate: `ready_for_ebr_decomposition == true` AND `status == "complete"`
AND `readiness_level == "trusted"`.

### `valley_reduced_ebr_mapping.json`

Exact-integer reduced EBR decomposition. **Default-off**; requires
`analysis.reduced_ebr.enabled: true` and a user-supplied validated table file.

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"not_evaluated"`, `"solved_exact"`, `"no_exact_solution"` |
| `solutions` | list[object] | Per-bundle solution entries |
| `excluded_bundles` | list[object] | Bundles excluded from solving |
| `solver` | string | `"brute_force_exact_integer"` |
| `table_status` | string | `"loaded"` or `"not_provided"` |
| `interpretation` | string | Human-readable status |

Each solution (when `status == "solved_exact"`):

| Field | Type | Description |
|-------|------|-------------|
| `bundle_id` | string | Source bundle ID |
| `valley` | string | Valley label |
| `subspace_group_candidate` | string | Group candidate |
| `ebr_decomposition` | list[object] | EBR labels with integer coefficients |
| `ebr_decomposition[].label` | string | EBR irrep label from user table |
| `ebr_decomposition[].coefficient` | int | Non-negative integer multiplicity |

---

## Debug / Detail Outputs

The following files are debug/detail diagnostics. They may change without a
schema version bump and are not part of the frozen public schema.

| File | Description |
|------|-------------|
| `valley_weights.csv` | Per-(kpoint, band) raw valley weights with center-resolved columns |
| `valley_subspace.json` | Per-kpoint valley subspace diagnostics (seed matrices, adapted basis) |
| `symmetry_report.json` | Detected symmetry operations, valley mappings, little groups |
| `symmetry_eigenvalues.csv` | Per-state symmetry eigenvalues and readiness flags |
| `diagnostics.h5` | Projector masks, qcut scan data, symmetry representation matrices, center weights |
| `valley_basis_transform.h5` | Valley-adapted basis transform matrices |
| `projector_symmetry_report.json` | Seed projector symmetry-consistency diagnostics |
| `symmetry_adapted_valley_analysis.json` | Full symmetry-adapted valley analysis |
| `target_subspace_closure.json` | D_raw target-subspace closure diagnostics |
| `hsp_star_conjugation.json` | HSP-star conjugation graphs |
| `hsp_star_derived_characters.json` | Symmetry-derived HSP-star characters |
| `subspace_representation_quality.json` | Subspace representation quality decomposition (default-off) |
| `irrep_workflow_decisions.json` | Per-valley irrep workflow path decisions |
| `valley_irrep_matching.json` | Valley-preserving irrep matching results |
| `valley_ebr_input_candidates.json` | EBR input candidate collection (intermediate) |
| `valley_ebr_problem_instances.json` | EBR problem instances (intermediate) |
| `folded_center_report.json` | Valley-center folding into moire BZ, k-center distances |
| `sampled_k_coverage.json` | Sampled k-point branch coverage diagnostic |

---

## Methodology Layers

1. **Layer 1 — valley projection diagnostics**: `valley_summary.*`,
   `valley_weights.csv`, `valley_subspace.json`, `diagnostics.h5`,
   `folded_center_report.json`, `sampled_k_coverage.json`
2. **Layer 2 — trusted irreps / EBR input**: `valley_ebr_export_bundle.json`,
   `valley_irrep_matching.json`, `irrep_workflow_decisions.json`,
   `symmetry_adapted_valley_analysis.json`, `projector_symmetry_report.json`,
   `target_subspace_closure.json`, `hsp_star_*`
3. **Layer 3 — optional external-table reduced EBR mapping**:
   `valley_reduced_ebr_mapping.json` (default-off)
