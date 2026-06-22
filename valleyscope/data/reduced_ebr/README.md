# ValleyScope Reduced EBR Package Data

This directory is the package-data catalog for reviewed, provenance-tracked
reduced-dimensional EBR tables.

## Current Status

**Two manifest entries ship one reviewed legacy validation fixture.** The
catalog (`manifest.json`) lists:

- `P3_GammaM_KM_spinful_v1` (preferred): Physical P3 subspace-space-group
  identity for the sampled `GammaM`/`KM` valley-preserving C3 subgroup.  This
  is a reviewed alias of the existing SG150/P321 parent-to-P3 reduced table,
  not an independently reviewed SG143 source table.
- `P321_C3_like_GammaM_KM_spinful_v1` (legacy compatibility name): Points to
  the same reviewed table file.  The `C3_like` label in the manifest name is
  legacy only; the table's `subspace_group_candidate` is `"P3"`.

Both entries are marked with
`convention_source: "irreptables_reduced_legacy_phase_label"`.  They remain
loadable for regression and backward compatibility, but they are not the
production convention source.  Production reduced EBR tables must use
Bilbao/irreptables labels and are discovered through
`list_production_reduced_ebr_tables()`.  At this stage that production list is
empty until an irreptables-derived reduced table is reviewed.

Additional tables will be added only after:

1. Literature review and physical validation.
2. Benchmark verification against known trusted irreps.
3. Documented provenance (source, reference, reviewer, date).
4. Schema contract and consistency checks pass.

## Adding A Reviewed Table

When a table is ready for shipment:

1. Add the table JSON file to this directory.
2. Add an entry to `manifest.json` under `tables` with the filename and a
   brief description.
3. Add reviewed-table metadata to the manifest entry:
   `review_status: "reviewed"`, `reviewer`, `review_date`, `review_method`,
   and `source_reference`.
4. Add a top-level table `provenance` object with the same reviewed-table
   fields and `valleyscope_reduction: "sampled_hsp_valley_preserving"`.
5. Add physical identity provenance to the table `provenance` object:
   `data_source`, `space_group_number`, `spinful`,
   `subspace_group_candidate`, `expected_hsps`, and
   `central_sign_convention`.  `subspace_group_candidate` and
   `expected_hsps` must match the table top-level fields exactly.
6. Run the package-data validation tests.
7. The table must satisfy the schema defined in
   `docs/reduced_ebr_table_schema.md`.

Reviewed package-data tables are loaded with
`load_reviewed_reduced_ebr_table()`.  This catalog path enforces both the
manifest review metadata and the table `provenance` gate. The physical
identity provenance records the HSP little group, valley mapping, and
valley-preserving subgroup context for the reduced-dimensional table.
Valley-changing operation information remains valley sewing matrix data unless
a future reviewed table explicitly defines a different reduced basis.

External reduced EBR tables supplied by users through
`analysis.reduced_ebr.table_file` are loaded with `load_reduced_ebr_table()`.
That external path validates the table schema and exact-integer vector
contract, but it does not require reviewed package-data provenance.

## Schema

See `docs/reduced_ebr_table_schema.md` for the external table format and
`docs/reduced_dimensional_irrep_ebr_data_model.md` for the data model
design and validation rules.

## Non-Features

- No built-in unreviewed or material-specific tables are provided.
- No compatibility relations.
- No heuristic or floating-point EBR fitting.
- No material-specific identifiers.
- The tables here are for valley-resolved reduced-dimensional irreps/EBRs,
  not full 3D space-group EBR tables from the Python package `irrep`.
