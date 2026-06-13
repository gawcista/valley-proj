# C3 Real-Source v1.1 Workflow Smoke

Date: 2026-06-13 | Status: Smoke audit (no table data shipped)

## Scope

This document records a workflow smoke test that exercises the full v1.1
multiplicity-aware authoring path — scaffold, validate, and build — using
real public `irreptables` SG 150 (P321) spinful source data.  It is a
**workflow smoke/audit**, not package data, not a reduced EBR
decomposition report, and not `analyze-hsp` wiring.

No reduced EBR table JSON is committed or shipped.

## Real Source Basis

From `irreptables.ebrs.load_ebr_data(150, True)` via
`inspect_irreptables_source_basis()`:

- 22 source irrep labels covering all 3D BZ HSPs
- 9 EBR definitions with integer vectors

Exact label set (machine-recorded by
`test_c3_real_source_v1_1_workflow_smoke`):

```text
-GM4, -GM5, -GM6,
-A4, -A5, -A6,
-H4, -H5, -H6,
-HA4, -HA5, -HA6,
-K4, -K5, -K6,
-KA4, -KA5, -KA6,
-L3, -L4,
-M3, -M4
```

## HSP Assignments

Each source label is assigned to its 3D BZ HSP by explicit
human-authored `source_hsp_by_irrep`:

| HSP | Source Labels | In C3 Sampled Set? |
|-----|-------------|-------------------|
| GammaM | `-GM4`, `-GM5`, `-GM6` | Yes |
| A | `-A4`, `-A5`, `-A6` | No |
| H | `-H4`, `-H5`, `-H6` | No |
| HA | `-HA4`, `-HA5`, `-HA6` | No |
| KM | `-K4`, `-K5`, `-K6` | Yes |
| KA | `-KA4`, `-KA5`, `-KA6` | No |
| L | `-L3`, `-L4` | No |
| M | `-M3`, `-M4` | No |

`expected_hsps = ["GammaM", "KM"]`.  MM is identity-only in the C3
valley-preserving subgroup and is excluded.  KA is excluded as out of
sampled scope.

## Valley-Preserving Multiplicities (Sampled HSPs Only)

Following the C3 mapping signoff packet
(`docs/reduced_ebr_c3_mapping_signoff_packet.md`):

| Source Label | Deg | HSP | ValleyScope Irrep Key(s) | Multiplicity |
|-------------|-----|-----|--------------------------|-------------|
| `-GM4` | 1 | GammaM | `GammaM:C3_spinor_phase_+1/2` | 1 |
| `-GM5` | 1 | GammaM | `GammaM:C3_spinor_phase_+1/2` | 1 |
| `-GM6` | 2 | GammaM | `GammaM:C3_spinor_phase_+1/6` | 1 |
| | | | `GammaM:C3_spinor_phase_-1/6` | 1 |
| `-K4` | 1 | KM | `KM:C3_spinor_phase_+1/2` | 1 |
| `-K5` | 1 | KM | `KM:C3_spinor_phase_+1/2` | 1 |
| `-K6` | 2 | KM | `KM:C3_spinor_phase_+1/6` | 1 |
| | | | `KM:C3_spinor_phase_-1/6` | 1 |

Non-sampled-HSP labels (A, H, HA, KA, L, M) omit multiplicity entries
per the v1.1 validator's sampled-HSP rule.

## Valley-Preserving Subgroup And Exclusion

The physical objects are separated as:

- **HSP little group** G_k: the full little group at each sampled HSP,
  including both C3 and C2 operations for P321.
- **Valley mapping** π_g(a): C3 operations preserve valley labels;
  C2 operations swap K↔K'.
- **Valley-preserving subgroup** G_k^{(a)} = `{E, C3, C3²}` ≅ C3.
  Only valley-preserving operations contribute to the single-valley
  reduced EBR irrep basis.
- **Valley-changing operations** (C2, op4-6): map K↔K'.  Their
  characters are **valley sewing matrix** data and must not enter the
  C3 reduced EBR vector basis.
- The central-sign convention `chi(C3)=chi(op2)`, `chi(C3²)=-chi(op3)`
  is used per the signoff packet.

## Workflow Smoke Result

The test `test_c3_real_source_v1_1_workflow_smoke` in
`tests/test_irreptables_table_builder.py`:

1. Inspects the real SG 150 spinful source basis.
2. Builds a v1.1 spec with explicit HSP and multiplicity maps.
3. Validates the spec against the source basis (preflight passes).
4. Builds a reduced external table via `build_reduced_table_from_spec_file()`.
5. Asserts the table has 6 irrep keys, HSPs `["GammaM", "KM"]`,
   C3-like group, nonnegative integer vectors of length 6, and zero
   C2/valley sewing entries.

The generated table is written to a temporary path only and is never
committed.

## Pinned C3 Reduced EBR Vector Reference

From `irreptables` SG 150 spinful EBR data + v1.1 multiplicity mapping.
Vectors are in the canonical key order:

```text
[GammaM:C3_spinor_phase_+1/6,
 GammaM:C3_spinor_phase_+1/2,
 GammaM:C3_spinor_phase_-1/6,
 KM:C3_spinor_phase_+1/6,
 KM:C3_spinor_phase_+1/2,
 KM:C3_spinor_phase_-1/6]
```

| Wyckoff | Reduced EBR Vector | Test Pinned? |
|---------|-------------------|-------------|
| 1a(32,32) | `[0, 1, 0, 0, 1, 0]` | Yes |
| 1a(32,32) | `[0, 1, 0, 0, 1, 0]` | Yes |
| 1a(32,32) | `[1, 0, 1, 1, 0, 1]` | Yes |
| 1b(32,32) | `[0, 1, 0, 0, 1, 0]` | Yes |
| 1b(32,32) | `[0, 1, 0, 0, 1, 0]` | Yes |
| 1b(32,32) | `[1, 0, 1, 1, 0, 1]` | Yes |
| — | `[1, 0, 1, 1, 0, 1]` | Yes |
| — | `[1, 0, 1, 0, 2, 0]` | Yes |
| — | `[0, 2, 0, 1, 0, 1]` | Yes |

These vectors are pinned by `test_c3_real_source_pins_vector_reference`
in `tests/test_irreptables_table_builder.py`.  A change in the public
`irreptables` EBR data or the ValleyScope multiplicity mapping will
cause the reference test to fail, providing explicit drift detection.

## Mapping E2E Smoke

The test `test_c3_real_source_mapping_e2e_solved_exact` builds a
temporary C3 reduced table from real source data, constructs a ready
export-bundle payload matching one EBR vector (`[1, 0, 1, 1, 0, 1]`),
and calls `build_reduced_ebr_mapping()`.  The result asserts:

- `mapping_status == "solved_exact"`
- `classification == "atomic-compatible-candidate"`
- `integer_span_status == "in_integer_span"`
- `nonnegative_solution_status == "solved_exact"`
- Nonnegative integer EBR decomposition present
- No C2/valley sewing keys in the reduced irrep basis

This exercises the Layer 3 exact-integer solver path (`smith_normal_form_plus_bounded_nonnegative_search`)
with a real-source temporary table.  It is not a shipped reduced EBR
decomposition report for any material.

## Non-Delivery

- This smoke test does not ship a reduced EBR table JSON.
- No data is written under `valleyscope/data/reduced_ebr/`.
- No `analyze-hsp` wiring.
- The table generated by the smoke test is temporary and is discarded
  after the test completes.
- Human provenance signoff per `docs/reduced_ebr_c3_mapping_signoff_packet.md`
  is still required before any reviewed table can be shipped.
