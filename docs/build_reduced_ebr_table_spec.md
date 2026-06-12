# ValleyScope `build-reduced-ebr-table` Canonical Mapping Spec

Version: 1.0.0 | Date: 2026-06-12

The `valleyscope build-reduced-ebr-table SPEC.json --output TABLE.json` CLI
builds a ValleyScope reduced-dimensional external EBR table from a JSON
mapping spec and public `irreptables` data.

This document defines the canonical spec format.  Every spec must pass
`valleyscope.analysis.irreptables_runtime_table_builder.build_reduced_table_from_spec_file()`.
The helper validates the table through `load_reduced_ebr_table()` before
returning it, and the CLI validates the written JSON before reporting success.

## Spec Schema

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `schema_version` | string | Yes | Must be `"1.0.0"` |
| `data_source` | string | Yes | Must be `"irreptables"` |
| `space_group_number` | int or string | Yes | Projected-subspace / moire space group number for `irreptables.ebrs.load_ebr_data` (e.g., `150` for P321) |
| `spinful` | bool | Yes | Double-valued (spinor) irreps (`true` for SOC systems) |
| `source_hsp_by_irrep` | dict | Yes | Map from every source irrep label to a sampled moire HSP label. All source labels must have entries. |
| `valleyscope_key_by_source_irrep` | dict | Yes | Map from every source irrep label to a ValleyScope valley-preserving irrep key (`kpoint:label[:opN]`). |
| `expected_hsps` | list[str] | Yes | Sampled moire HSP labels in the reduced basis. Only irreps whose HSP is in this set appear in the output. |
| `allowed_irrep_keys` | list[str] | Yes | Trusted valley-preserving irrep keys; order defines the reduced output `irreps` basis. Must be a subset of `valleyscope_key_by_source_irrep` values. |
| `subspace_group_candidate` | str | Yes | `C{order}_like` label (e.g., `"C3_like"`, `"C2_like"`). |
| `provenance` | dict | No | Optional extra provenance fields attached to the output. |

## Example

```json
{
  "schema_version": "1.0.0",
  "data_source": "irreptables",
  "space_group_number": 150,
  "spinful": true,
  "source_hsp_by_irrep": {
    "-GM5": "GammaM",
    "-K5": "KM",
    "-K6": "KM"
  },
  "valleyscope_key_by_source_irrep": {
    "-GM5": "GammaM:C3_spinor_phase_+1/2",
    "-K5": "KM:C3_spinor_phase_+1/6",
    "-K6": "KM:C3_spinor_phase_-1/6"
  },
  "expected_hsps": ["GammaM", "KM"],
  "allowed_irrep_keys": [
    "GammaM:C3_spinor_phase_+1/2",
    "KM:C3_spinor_phase_+1/6",
    "KM:C3_spinor_phase_-1/6"
  ],
  "subspace_group_candidate": "C3_like"
}
```

## Rules

1. **Explicit mappings only**: `source_hsp_by_irrep` and
   `valleyscope_key_by_source_irrep` must cover every source irrep label.
   Missing entries raise `ValueError`.
2. **No implicit HSP inference**: `GM` → `GammaM` is not automatic; the
   mapping must state it explicitly.
3. **No implicit irrep-key inference**: irrep labels from the source package
   are not assumed to be ValleyScope valley-preserving irrep labels.
4. **Reduction is deterministic**: given the same spec + source package
   version, the output is identical.
5. **Output validated**: the output table passes `load_reduced_ebr_table()`;
   the CLI also validates the file after writing it.

## Non-Features

- No built-in unreviewed EBR tables.
- No heuristic or floating-point EBR fitting.
- No compatibility relations.
- No orbit-level or full-group irreps — valley-preserving irreps only.
- No `irrep.ebrs`, OR-Tools, or private `irrep2` imports.
- No raw 3D decomposition.
