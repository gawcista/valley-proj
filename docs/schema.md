# ValleyScope Public Output Schema

Version: 1.1.0 | Date: 2026-06-10

This document defines the **public output schema** for ValleyScope. Files not
listed here are debug/detail or intermediate diagnostics and may change
without a schema version bump.

## Output Profile

ValleyScope uses `output.profile` to control which files are written:

| Profile | Behavior |
|---------|----------|
| `standard` (default) | Public user-facing outputs only: summary, EBR export bundle, reduced EBR mapping, valley weights CSV |
| `debug` | Full diagnostic file set: all standard outputs plus detailed JSON/HDF5 diagnostics |

The legacy `output.write_detailed_files` boolean is deprecated and mapped
to `output.profile`: `false` → `standard`, `true` → `debug`.
Explicit `output.profile` takes precedence.

To enable full diagnostics in a config:

```yaml
output:
  directory: ./valley_analysis
  profile: debug
```

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
| `valley_reduced_ebr_mapping` | Reduced EBR mapping enabled (`analysis.reduced_ebr.enabled`) |
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
| `subspace_group_candidate` | string | Effective EBR table label (e.g. `"C3_like"`, `"C2_like"`) — built from `C{max_valley_preserving_order}_like`. External reduced EBR tables must use this label, not crystallographic notation. |
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

**Label convention**: `subspace_group_candidate` uses the effective EBR table
label form `C{max_order}_like` (e.g. `C3_like`, `C2_like`), built from the
maximum valley-preserving operation order. External reduced EBR tables must
use this label. The crystallographic notation (`P3`, `P2`) appears in
`subspace_space_group.candidate_space_group_symbol` within the
`symmetry_adapted_valley_analysis.json` report but is not the EBR table key.
The two conventions are related — a `C3_like` valley-preserving subgroup
inside a `P3`-family space group, a `C2_like` subgroup inside a
`P2`-family space group — but the EBR contract uses the `C{order}_like`
form exclusively.

### `valley_reduced_ebr_mapping.json`

Exact-integer reduced EBR decomposition. **Default-off**; produced only when
`analysis.reduced_ebr.enabled: true`. A user-supplied validated table is
required for `solved_exact` / `no_exact_solution` decomposition attempts;
without a table the output carries `status: missing_table`.

Top-level fields (always present):

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"not_evaluated"`, `"missing_table"`, `"solved_exact"`, `"no_exact_solution"` |
| `mapping_status` | string | Same as `status` |
| `reduced_ebr_decomposition_status` | string | Same as `status` |
| `table_status` | string | `"not_applicable"` (no export bundle available), `"not_provided"` (enabled but no table file), `"loaded"` (table loaded and decomposition attempted) |
| `solutions` | list[object] | Per-bundle solution entries (empty list when no bundles to decompose) |
| `excluded_bundles` | list[object] | Bundles excluded from solving |
| `solver` | string | `"brute_force_exact_integer"` |
| `interpretation` | string | Human-readable status message |

Conditional fields:

| Field | Type | Condition |
|-------|------|-----------|
| `max_coefficient` | int | Present when a table is loaded and `analysis.reduced_ebr.enabled` |

Each solution entry (common fields, always present):

| Field | Type | Description |
|-------|------|-------------|
| `bundle_id` | string | Source bundle ID |
| `valley` | string | Valley label |
| `subspace_group_candidate` | string | Group candidate |
| `irrep_vector` | list[int] | Integer count vector aligned to table irrep order |
| `status` | string | `"solved_exact"` or `"no_exact_solution"` |

When `status == "solved_exact"`, additionally:

| Field | Type | Description |
|-------|------|-------------|
| `ebr_decomposition` | list[object] | EBR labels with integer coefficients (non-zero only) |
| `ebr_decomposition[].label` | string | EBR irrep label from user table |
| `ebr_decomposition[].coefficient` | int | Non-negative integer multiplicity |

When `status == "no_exact_solution"`, `ebr_decomposition` is absent.

Each excluded bundle:

| Field | Type | Description |
|-------|------|-------------|
| `bundle_id` | string | Source bundle ID |
| `reason` | string | `"missing_table"`, `"not ready for external solver"`, `"table group X != bundle group Y"`, or `"could not resolve irrep keys to table irreps"` |

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

---

## Compact Example Shapes

### `valley_summary.json`

```json
{
  "input": {
    "wavefunction_h5": "wave.h5",
    "operation_structure_file": "POSCAR",
    "operation_detection_backend": "spglib",
    "spinor_convention": "vasp_up_down_saxis_z",
    "spinor_convention_verified": false,
    "spinor_benchmark": null
  },
  "target_kpoints": ["GammaM", "KM"],
  "iband": [2195, 2196],
  "valley_subspaces": [
    {"label": "K_valley", "centers": ["top_K", "bottom_K"]}
  ],
  "qcut": {
    "projector_mode": "fixed_center",
    "mode": "relative_min_valley_distance",
    "value_Ainv": 0.034,
    "fraction": 0.2,
    "scan": [0.15, 0.20, 0.25]
  },
  "valley_projection_summary": [
    {
      "kpoint": "GammaM", "band_vasp": 2195,
      "W_val": 0.98, "P_v": 0.99, "eta": 0.96,
      "W_overlap": 0.0, "W_res": 0.02,
      "status": "clean"
    }
  ],
  "valley_subspace_analysis": [
    {
      "kpoint": "GammaM", "basis_status": "valley_separable",
      "S_min": 0.97, "S_max": 0.99,
      "min_concentration": 0.98,
      "assigned_valleys": ["K_valley", "Kp_valley"],
      "status": "clean"
    }
  ],
  "warnings": [],
  "legend": { "W_val": "valley-subspace weight", "P_v": "valley purity" }
}
```

### `valley_ebr_export_bundle.json`

```json
{
  "status": "ready_for_external_solver",
  "bundle_count": 1,
  "excluded_count": 0,
  "schema_version": "1.0.0",
  "reduced_ebr_decomposition_status": "not_implemented",
  "interpretation": "1 bundle(s) ready for external reduced EBR decomposition",
  "bundles": [
    {
      "bundle_id": "bundle_K_valley",
      "source_instance_id": "K_valley",
      "valley": "K_valley",
      "subspace_group_candidate": "C3_like",
      "workflow_path": "direct_qcut",
      "readiness_level": "trusted",
      "irreps_by_kpoint": {"GammaM": ["C3_spinor_phase_+1/2"], "KM": ["C3_spinor_phase_+1/6", "C3_spinor_phase_-1/6"]},
      "operations_by_kpoint": {"GammaM": [0, 1, 2], "KM": [0, 1, 2]},
      "expected_hsps": ["GammaM", "KM"],
      "optional_hsps": ["MM"],
      "missing_optional_hsps": ["MM"],
      "ready_for_external_solver": true
    }
  ],
  "excluded_instances": []
}
```

### `valley_reduced_ebr_mapping.json` (with table)

```json
{
  "status": "solved_exact",
  "mapping_status": "solved_exact",
  "reduced_ebr_decomposition_status": "solved_exact",
  "table_status": "loaded",
  "solutions": [
    {
      "bundle_id": "bundle_K_valley",
      "valley": "K_valley",
      "subspace_group_candidate": "C3_like",
      "irrep_vector": [1, 1],
      "status": "solved_exact",
      "ebr_decomposition": [
        {"label": "(A)1a", "coefficient": 1},
        {"label": "(B)1a", "coefficient": 1}
      ]
    }
  ],
  "excluded_bundles": [],
  "solver": "brute_force_exact_integer",
  "max_coefficient": 6,
  "interpretation": "Exact integer linear combination of EBR vectors ..."
}
```

### `valley_reduced_ebr_mapping.json` (missing table)

```json
{
  "status": "missing_table",
  "mapping_status": "missing_table",
  "reduced_ebr_decomposition_status": "missing_table",
  "table_status": "not_provided",
  "solutions": [],
  "excluded_bundles": [
    {"bundle_id": "bundle_K_valley", "reason": "missing_table"}
  ],
  "solver": "brute_force_exact_integer",
  "interpretation": "no reduced EBR table provided"
}
```
