# ValleyScope `build-reduced-ebr-table` Canonical Mapping Spec

Version: 1.1.0 | Date: 2026-06-13

The `valleyscope build-reduced-ebr-table SPEC.json --output TABLE.json` CLI
builds a ValleyScope reduced-dimensional external EBR table from a JSON
mapping spec and public `irreptables` data.

This document defines the canonical spec formats.  Every spec must pass
`valleyscope.analysis.irreptables_runtime_table_builder.build_reduced_table_from_spec_file()`.
The helper validates the table through `load_reduced_ebr_table()` before
returning it, and the CLI validates the written JSON before reporting success.

## Spec Schema

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `schema_version` | string | Yes | Must be `"1.0.0"` for legacy one-to-one specs or `"1.1.0"` for multiplicity-aware specs |
| `data_source` | string | Yes | Must be `"irreptables"` |
| `space_group_number` | int or string | Yes | Projected-subspace / moire space group number for `irreptables.ebrs.load_ebr_data` (e.g., `150` for P321) |
| `spinful` | bool | Yes | Double-valued (spinor) irreps (`true` for SOC systems) |
| `source_hsp_by_irrep` | dict | Yes | Map from every source irrep label to a sampled moire HSP label. All source labels must have entries. |
| `valleyscope_key_by_source_irrep` | dict | Yes for `1.0.0` | Legacy map from every source irrep label to one ValleyScope valley-preserving irrep key (`kpoint:label[:opN]`). |
| `valleyscope_irrep_multiplicity_by_source_irrep` | dict | Yes for `1.1.0` | Multiplicity map from source irrep label to `{ValleyScope irrep key: positive integer}` for sampled-HSP, valley-preserving reduction. |
| `expected_hsps` | list[str] | Yes | Sampled moire HSP labels in the reduced basis. Only irreps whose HSP is in this set appear in the output. |
| `allowed_irrep_keys` | list[str] | Yes | Trusted valley-preserving irrep keys; order defines the reduced output `irreps` basis. In `1.0.0`, keys come from `valleyscope_key_by_source_irrep`; in `1.1.0`, keys come from `valleyscope_irrep_multiplicity_by_source_irrep` entries. |
| `subspace_group_candidate` | str | Yes | Physical valley-projected subspace-space-group symbol (e.g., `"P3"`, `"P4"`, `"P2"`). |
| `provenance` | dict | No | Optional extra provenance fields attached to the output. |

## Legacy 1.0.0 Example

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
  "subspace_group_candidate": "P3"
}
```

## Multiplicity-Aware 1.1.0 Example

Use `schema_version: "1.1.0"` when the source package irrep basis must be
restricted to a valley-preserving subgroup before forming ValleyScope's reduced
EBR vector basis.

```json
{
  "schema_version": "1.1.0",
  "data_source": "irreptables",
  "space_group_number": 150,
  "spinful": true,
  "source_hsp_by_irrep": {
    "-GM4": "GammaM",
    "-GM5": "GammaM",
    "-GM6": "GammaM",
    "-K4": "KM",
    "-K5": "KM",
    "-K6": "KM",
    "-A5": "A"
  },
  "valleyscope_irrep_multiplicity_by_source_irrep": {
    "-GM4": {"GammaM:C3_spinor_phase_+1/2": 1},
    "-GM5": {"GammaM:C3_spinor_phase_+1/2": 1},
    "-GM6": {
      "GammaM:C3_spinor_phase_+1/6": 1,
      "GammaM:C3_spinor_phase_-1/6": 1
    },
    "-K4": {"KM:C3_spinor_phase_+1/2": 1},
    "-K5": {"KM:C3_spinor_phase_+1/2": 1},
    "-K6": {
      "KM:C3_spinor_phase_+1/6": 1,
      "KM:C3_spinor_phase_-1/6": 1
    }
  },
  "expected_hsps": ["GammaM", "KM"],
  "allowed_irrep_keys": [
    "GammaM:C3_spinor_phase_+1/6",
    "GammaM:C3_spinor_phase_+1/2",
    "GammaM:C3_spinor_phase_-1/6",
    "KM:C3_spinor_phase_+1/6",
    "KM:C3_spinor_phase_+1/2",
    "KM:C3_spinor_phase_-1/6"
  ],
  "subspace_group_candidate": "P3"
}
```

Here `source_hsp_by_irrep` may cover source labels outside `expected_hsps`.
Those non-sampled HSP labels, such as `-A5` in the example above, do not need
entries in `valleyscope_irrep_multiplicity_by_source_irrep` because they are
filtered before the sampled-HSP reduced basis is formed.

## Rules

1. **Explicit mappings only**: `source_hsp_by_irrep` and
   the schema-version-specific irrep map must be human-authored.  In `1.0.0`,
   `valleyscope_key_by_source_irrep` must cover every source irrep label.  In
   `1.1.0`, `valleyscope_irrep_multiplicity_by_source_irrep` must cover every
   source irrep label whose HSP is in `expected_hsps`; source labels outside the
   sampled HSP set are filtered out.
2. **No implicit HSP inference**: `GM` → `GammaM` is not automatic; the
   mapping must state it explicitly.
3. **No implicit irrep-key inference**: irrep labels from the source package
   are not assumed to be ValleyScope valley-preserving irrep labels.
4. **Multiplicity semantics**: for `1.1.0`,
   `valleyscope_irrep_multiplicity_by_source_irrep[source_label][key] = m`
   means that the source irrep contributes multiplicity `m` to the
   ValleyScope valley-preserving irrep key after sampled-HSP,
   valley-preserving subgroup reduction.  This supports many-to-one source
   irrep aggregation and one-to-many degenerate-source decomposition.
5. **Reduction is deterministic**: given the same spec + source package
   version, the output is identical.
6. **Output validated**: the output table passes `load_reduced_ebr_table()`;
   the CLI also validates the file after writing it.

## Authoring Workflow

1. **Inspect the source basis**:
   ```bash
   valleyscope inspect-ebr-source --space-group-number 150 -o source_basis.json
   ```

2. **Scaffold a mapping spec template**:
   ```bash
   valleyscope scaffold-spec source_basis.json -o spec_template.json
   ```
   This produces a non-buildable template with every source label and
   `REQUIRED_FILL_BY_HUMAN` placeholders.  No HSP or ValleyScope irrep
   keys are inferred.

3. **Manually fill the template**: map every source irrep label to a
   sampled moire HSP.  For `1.0.0`, map every source irrep label to one
   ValleyScope valley-preserving irrep key.  For `1.1.0`, fill the
   multiplicity map for every source label at the sampled HSPs.
   Fill `expected_hsps`, `allowed_irrep_keys`, and
   `subspace_group_candidate`.

4. **Validate the completed spec**:
   ```bash
   valleyscope validate-spec spec.json source_basis.json
   ```
   Checks source label coverage, placeholder completion, canonical fields,
   and HSP/key mapping consistency.  Returns nonzero on validation failure.

5. **Build the reduced table** (with optional preflight):
   ```bash
   valleyscope build-reduced-ebr-table spec.json --source-basis source_basis.json -o table.json
   ```
   The `--source-basis` flag runs the preflight validator before building.
   Omit it to build without validation.

6. **Map the reduced EBR**:
   ```bash
   valleyscope map-reduced-ebr bundle.json table.json -o mapping.json
   ```

## Non-Features

- No built-in unreviewed EBR tables.
- No heuristic or floating-point EBR fitting.
- No compatibility relations.
- No orbit-level or full-group irreps — valley-preserving irreps only.
- No `irrep.ebrs`, OR-Tools, or private `irrep2` imports.
- No raw 3D decomposition.
