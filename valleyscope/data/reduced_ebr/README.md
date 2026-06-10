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
3. Run the package-data validation tests.
4. The table must satisfy the schema defined in
   `docs/reduced_ebr_table_schema.md`.

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
