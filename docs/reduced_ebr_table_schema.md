# Reduced EBR Table Schema

Version: 1.0.0 | Date: 2026-06-10

This document defines the external table format for the ValleyScope
default-off exact-integer reduced EBR mapping interface. The mapping is
performed by `valleyscope/analysis/reduced_ebr_mapping.py`.

No built-in EBR tables, compatibility relations, heuristic fits, or
floating-point decomposition are provided. Reduced EBR decomposition
requires a user-supplied validated table in the format below.
Without a table, the interface reports `status: missing_table`.

External user tables are validated by
`valleyscope.analysis.reduced_ebr_mapping.load_reduced_ebr_table()` and do
not need package-data review metadata. They are selected with
`analysis.reduced_ebr.table_file`. Reviewed package-data tables are a
stricter path selected with `analysis.reduced_ebr.table_name` and loaded by
`valleyscope.data.reduced_ebr.catalog.load_reviewed_reduced_ebr_table()`;
that path requires reviewed metadata in both `manifest.json` and the table's
top-level `provenance` object, including `review_status: "reviewed"`,
`reviewer`, `review_date`, `review_method`, `source_reference`, and
`valleyscope_reduction: "sampled_hsp_valley_preserving"`.
Reviewed package-data tables also require physical identity provenance in the
table `provenance` block: `data_source`, `space_group_number`, `spinful`,
`subspace_group_candidate`, `expected_hsps`, and
`central_sign_convention`. The `subspace_group_candidate` and `expected_hsps`
provenance values must match the table top-level fields exactly, so the table
identity stays tied to the sampled HSP little group and valley-preserving
subgroup reduction rather than to a material name. Valley-changing operation
and valley sewing matrix data must remain outside the reduced EBR vector basis
unless a future reviewed table explicitly defines a different valley mapping
problem.
`analysis.reduced_ebr.table_file` and `analysis.reduced_ebr.table_name` are
mutually exclusive.

## Required Table Keys

| Key | Type | Description |
|-----|------|-------------|
| `schema_version` | string | Table schema version (e.g. `"1.0.0"`) |
| `subspace_group_candidate` | string | Physical valley-projected subspace-space-group symbol matching export bundle `subspace_group_candidate` (e.g. `"P3"`, `"P4"`, `"P2"`). This is the primary EBR table key. |
| `expected_hsps` | list[string] | Unique non-empty HSP labels covered by this table (e.g. `["GammaM", "KM"]`) |
| `irreps` | list[string] | Unique non-empty irrep keys in the format `<kpoint>:<irrep_label>` with optional `:op<N>` suffix |
| `ebrs` | list[object] | EBR definitions, each with `label` (unique non-empty string) and `vector` (non-empty list of nonnegative integers aligned to `irreps`) |

## Allowed Status Values

`valley_reduced_ebr_mapping.json` reports these top-level status values:

| Status | Meaning |
|--------|---------|
| `not_evaluated` | No export bundle available |
| `missing_table` | `analysis.reduced_ebr.enabled: true` but no `table_file` or `table_name` provided |
| `solved_exact` | All bundles classified as `atomic-compatible-candidate` |
| `no_exact_solution` | At least one bundle classified as fragile or stable |

Per-solution classification uses exact integer algebra (Smith normal form
over ZZ):

| Classification | Integer Span | Nonnegative Solution | Meaning |
|---------------|-------------|---------------------|---------|
| `atomic-compatible-candidate` | in span | solved_exact | Nonnegative exact integer EBR combination exists |
| `fragile-topology-candidate` | in span | no_nonnegative_solution | Integer combination exists but needs negative coefficients; `integer_solution` witness provided |
| `stable-topology-candidate` | outside span | no_nonnegative_solution | Target irrep vector is outside the integer EBR span |

EBR vectors must have at least one positive entry (zero vectors are
rejected at table-load time).  `max_coefficient` is a safety cap; if it
truncates a derived physical bound, `search_status:
truncated_by_max_coefficient` is set.

## Irrep Key Format

Each key in `irreps` must match:

```
<kpoint>:<irrep_label>[:op<N>]
```

where:
- `<kpoint>` is a non-empty alphanumeric/underscore HSP label (e.g. `GammaM`, `KM`, `MM`)
- `<irrep_label>` is the valley-preserving irrep label (e.g. `C3_spinor_phase_+1/2`, `C3_spinor_phase_-1/6`)
- `:op<N>` is an optional operation suffix for disambiguation when the same irrep label appears for multiple operations at the same kpoint

Example valid keys:
```
GammaM:C3_spinor_phase_+1/2
KM:C3_spinor_phase_+1/6
KM:C3_spinor_phase_-1/6
GammaM:C3_spinor_phase_+1/2:op1
GammaM:C3_spinor_phase_+1/2:op2
```

The `:op<N>` suffix is used ONLY when disambiguation is needed. Both table
and bundle irrep keys may include or omit the suffix; the matcher resolves
exact matches first, then unique suffix-stripped matches. Ambiguous
suffix-stripped matches are rejected.

## Table-Label Matching Rules

1. The table `subspace_group_candidate` must exactly match the bundle
   `subspace_group_candidate`. Mismatched groups are excluded with reason
   `"table group X != bundle group Y"`.

2. The table `expected_hsps` is the reduced-dimensional EBR basis contract.
   For trusted decomposition, it must match the bundle `expected_hsps` set
   and the HSP-key set of `bundle.irreps_by_kpoint`. Legacy bundles without
   declared `expected_hsps` derive the basis only from `irreps_by_kpoint`
   keys. Missing, extra, malformed, or inferred HSP data is not accepted.

3. Bundle irrep labels are resolved to table irrep indices via the irrep
   key format above. Exact match is preferred; unique suffix-stripped match
   is a fallback. Ambiguous matches are rejected.

4. The resolved bundle irrep count vector is then decomposed against the
   EBR vectors using brute-force exact integer search up to
   `max_coefficient`. Only exact matches are reported; no heuristic or
   floating-point fitting is performed.

## Toy Table Example

```json
{
  "schema_version": "1.0.0",
  "subspace_group_candidate": "P3",
  "expected_hsps": ["GammaM", "KM"],
  "irreps": [
    "GammaM:C3_spinor_phase_+1/2",
    "KM:C3_spinor_phase_+1/6",
    "KM:C3_spinor_phase_-1/6"
  ],
  "ebrs": [
    {"label": "EBR_A", "vector": [1, 0, 1]},
    {"label": "EBR_B", "vector": [1, 1, 0]}
  ]
}
```

This table defines a `P3` reduced-basis toy problem with 3 irreps across
2 HSPs and 2 EBR basis vectors.  It is a toy example for testing; it does
not represent a reviewed physical EBR table for any real material.

## Toy Decomposition Example

Given the toy table above and a bundle with irrep count vector `[1, 1, 1]`:

- `1 * EBR_A + 0 * EBR_B = [1, 0, 1] ≠ [1, 1, 1]` → no match
- `0 * EBR_A + 1 * EBR_B = [1, 1, 0] ≠ [1, 1, 1]` → no match
- `1 * EBR_A + 1 * EBR_B = [2, 1, 1] ≠ [1, 1, 1]` → no match
- Result: `status: no_exact_solution`

Given a bundle with irrep count vector `[2, 1, 1]`:

- `1 * EBR_A + 1 * EBR_B = [2, 1, 1] = [2, 1, 1]` → exact match
- Result: `status: solved_exact`, decomposition:
  ```json
  {
    "bundle_id": "bundle_001",
    "valley": "K_valley",
    "subspace_group_candidate": "P3",
    "irrep_vector": [2, 1, 1],
    "status": "solved_exact",
    "ebr_decomposition": [
      {"label": "EBR_A", "coefficient": 1},
      {"label": "EBR_B", "coefficient": 1}
    ]
  }
  ```

## Offline CLI

A standalone CLI entry performs exact-integer reduced EBR mapping from an
existing `valley_ebr_export_bundle.json` plus either a user-supplied validated
external table or a reviewed package-data table name. This separates Layer 2
export-bundle generation from Layer 3 table-dependent mapping for
high-throughput workflows.

```bash
valleyscope map-reduced-ebr \
  valley_ebr_export_bundle.json \
  external_reduced_ebr_table.json \
  --output valley_reduced_ebr_mapping.json \
  --max-coefficient 6

valleyscope map-reduced-ebr \
  valley_ebr_export_bundle.json \
  --table-name reviewed_table_name \
  --output valley_reduced_ebr_mapping.json
```

Arguments:
- `bundle` — path to `valley_ebr_export_bundle.json` (required)
- `table` — path to external reduced EBR table JSON
- `--table-name` — name of a reviewed package-data table loaded through
  `load_reviewed_reduced_ebr_table`
- `--output`, `-o` — output path (default: `valley_reduced_ebr_mapping.json`)
- `--max-coefficient` — max coefficient per EBR in brute-force search (default: 6)

Exactly one of positional `table` or `--table-name` is required; they are
mutually exclusive. No default table name is implied.

Stdout summary example:
```
status:              solved_exact
total bundles:       1
solved (exact):      1
no exact solution:   0
excluded:            0
reduced EBR mapping: valley_reduced_ebr_mapping.json
```

The CLI:
- is Layer 3 and requires a user-supplied validated table;
- uses exact-integer brute-force matching only;
- the table physical subspace-space-group symbol (for example `P3`, `P4`,
  or `P2`) must match the export bundle `subspace_group_candidate`;
- no decomposition claim is made without a validated external table.

For offline construction of an external reduced EBR table from public
package-style source data, see
[`build_reduced_ebr_table_spec.md`](build_reduced_ebr_table_spec.md).  That
mapping spec performs ValleyScope's sampled-HSP, valley-preserving reduction
and is separate from reduced EBR decomposition.

## Explicit Non-Features

- No built-in EBR tables are provided.
- No heuristic or floating-point EBR fitting is performed.
- No compatibility relations are implemented.
- No decomposition claim is made without a user-supplied validated table.
- The toy examples above are for schema illustration only and do not
  represent reviewed physical EBR tables.
