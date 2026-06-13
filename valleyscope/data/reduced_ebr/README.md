# ValleyScope Reduced EBR Package Data

This directory is the package-data catalog for reviewed, provenance-tracked
reduced-dimensional EBR tables.

## Current Status

**No reviewed tables are currently shipped.** The catalog (`manifest.json`)
contains an empty table list.  Tables will be added only after:

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
5. Run the package-data validation tests.
6. The table must satisfy the schema defined in
   `docs/reduced_ebr_table_schema.md`.

Reviewed package-data tables are loaded with
`load_reviewed_reduced_ebr_table()`.  This catalog path enforces both the
manifest review metadata and the table `provenance` gate.

External reduced EBR tables supplied by users through
`analysis.reduced_ebr.table_file` are loaded with `load_reduced_ebr_table()`.
That external path validates the table schema and exact-integer vector
contract, but it does not require reviewed package-data provenance.

## Schema

See `docs/reduced_ebr_table_schema.md` for the external table format and
`docs/reduced_dimensional_irrep_ebr_data_model.md` for the data model
design and validation rules.

## Non-Features

- No built-in tables are provided.
- No compatibility relations.
- No heuristic or floating-point EBR fitting.
- No material-specific identifiers.
- The tables here are for valley-resolved reduced-dimensional irreps/EBRs,
  not full 3D space-group EBR tables from the Python package `irrep`.
