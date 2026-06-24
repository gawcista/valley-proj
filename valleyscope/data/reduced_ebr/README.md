# ValleyScope Reduced EBR Package Data

This directory contains the package-data catalog for reduced-dimensional EBR
tables. No static reviewed physical reduced EBR tables are shipped here.

## Current Status

**No static reduced EBR tables are shipped.** The manifest is empty.
Production reduced EBR data comes from:

- `valleyscope/irreps/ebr_data_adapter.py` → loads Bilbao/irreptables source EBR data
- `valleyscope/analysis/irreptables_runtime_table_builder.py` → applies
  ValleyScope's sampled-HSP valley-preserving reduction
- external user-supplied table files loaded through
  `valleyscope.analysis.reduced_ebr_mapping.load_reduced_ebr_table()`

Static JSON payloads under `tests/fixtures/reduced_ebr/` are test fixtures only
and are not production data.

## Schema

See `docs/reduced_ebr_table_schema.md` for the external table format.
