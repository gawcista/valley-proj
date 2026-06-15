# Database Ingestion Record — Real-Fixture Smoke Anchors

Date: 2026-06-11

This document records the key ingestion-record fields produced by
`valleyscope collect-database-record` from the two real validation
fixtures.  The ingestion record is an **explicit offline collector
artifact**, not a default `analyze-hsp` output.

Run commands (from fixture directories so relative config paths resolve):

```bash
tmpdir=$(mktemp -d)
(cd real_tests/tMoTe2 && python -m valleyscope.cli analyze-hsp analyze.yaml)
(cd real_tests/tMoTe2 && python -m valleyscope.cli collect-database-record \
  output/valley_analysis_wave --output "$tmpdir/tmote2_ingestion_record.json")
(cd real_tests/tZrSe2 && python -m valleyscope.cli analyze-hsp analyze.yaml)
(cd real_tests/tZrSe2 && python -m valleyscope.cli collect-database-record \
  output/valley_analysis --output "$tmpdir/tzrse2_ingestion_record.json")
```

## tMoTe2 P321 — Clean C3 `direct_qcut` Fixture

```
record status:          has_ready_ebr_bundles
ready bundles:          2
trusted irrep records:  8
ingestion record:       <tmpdir>/tmote2_ingestion_record.json
```

| Field | Value |
|-------|-------|
| `record_status` | `has_ready_ebr_bundles` |
| `ready_bundle_count` | 2 |
| `excluded_bundle_count` | 0 |
| `valley_irrep_records` count | 8 |
| `reduced_ebr_mapping_status` | `not_available` |
| `space_group_international` | `P321` |
| `space_group_number` | 150 |
| `spinor_convention_verified` | `true` |
| `target_kpoints` | `["GammaM", "KM", "MM"]` |
| `output_profile` | `standard` |

Each of the 8 trusted irrep records corresponds to a C3 valley-preserving
operation at GammaM (2 valleys × 2 C3 generators = 4 records) or KM
(2 valleys × 2 C3 generators = 4 records).  All have `workflow_path:
direct_qcut`, `readiness_level: trusted`, and carry `matched_irrep`,
`operation_id`, `operation_order`, `eigenphases`, `character`, and
`source` provenance.

## tZrSe2 P312 — Blocked C2/M-Star Fixture

```
record status:          no_ready_ebr_bundles
ready bundles:          0
trusted irrep records:  0
ingestion record:       <tmpdir>/tzrse2_ingestion_record.json
```

| Field | Value |
|-------|-------|
| `record_status` | `no_ready_ebr_bundles` |
| `ready_bundle_count` | 0 |
| `excluded_bundle_count` | 0 |
| `valley_irrep_records` count | 0 |
| `reduced_ebr_mapping_status` | `not_available` |
| `space_group_international` | `P312` |
| `space_group_number` | 149 |
| `spinor_convention_verified` | `false` |
| `output_profile` | `standard` |

Zero trusted irrep records: all rows are blocked by
`spinor_convention_unverified`, low seed overlap, D_raw target-subspace
closure failures, and HSP-star derivation unavailability.  These are
physical/readiness blockers, not database-ingestion errors.  The ingestion
record correctly reports `no_ready_ebr_bundles` without hiding the blocked
status.

## Physics Anchors Confirmed

- Clean C3 fixture: 2 ready bundles, 8 trusted valley-preserving irrep
  records, space group P321 (150), spinor convention verified.
- Blocked C2/M-star fixture: 0 ready bundles, 0 trusted irrep records,
  space group P312 (149), spinor convention NOT verified.
- `database_ingestion_record.json` is an explicit offline collector
  artifact; it is NOT written by default `analyze-hsp` runs.
- No tolerances or readiness gates were changed to produce these results.

## Reduced-EBR-Enabled Ingestion Anchor (2026-06-15)

With `analysis.reduced_ebr.table_name:
P321_C3_like_GammaM_KM_spinful_v1` and `output.profile: standard`:

| Field | Value |
|-------|-------|
| `record_status` | `has_ready_ebr_bundles` |
| `ready_bundle_count` | 2 |
| `valley_irrep_records` count | 8 |
| `reduced_ebr_mapping_status` | `solved_exact` |
| `reduced_ebr_table_status` | `loaded` |
| `reduced_ebr_classification_counts.atomic_compatible` | 2 |
| `reduced_ebr_classification_counts.fragile_topology` | 0 |
| `reduced_ebr_classification_counts.stable_topology` | 0 |
| Public output files | 5 (standard profile) |

Verified by `test_ingestion_record_from_public_outputs_with_reduced_ebr_mapping`
in `tests/test_database_ingestion.py` and by local tMoTe2 fixture run.
The reduced EBR mapping data is recorded in the ingestion record
alongside the existing valley-preserving irrep data, consuming only
public output files.
