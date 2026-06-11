# Valley-Preserving Irrep Phase Tables

This directory contains versioned, validated package-data phase tables for
minimal valley-preserving irrep matching.  Currently supports spinful C3
and C2 one-dimensional irrep phase labels only.

These are irrep matching data, not reduced EBR tables.  No EBR vectors,
compatibility relations, or 3D space-group irrep tables from the Python
package `irrep` are shipped here.

## Available Tables

| Name | Order | Spinful | Labels |
|------|-------|---------|--------|
| `spinful_C3_phase_v1` | 3 | true | C3_spinor_phase_+1/6, C3_spinor_phase_+1/2, C3_spinor_phase_-1/6 |
| `spinful_C2_phase_v1` | 2 | true | C2_spinor_phase_+1/4, C2_spinor_phase_-1/4 |

## Schema

Each table is a JSON file with:
- `schema_version`, `name`, `spinful` (true), `operation_order`;
- `subspace_group_candidates` — labels this table applies to;
- `phase_convention` — documentation of the phase convention;
- `irreps` — list of `{"label": ..., "phases": [...]}`.

Phases use the convention `eigenvalue = exp(2*pi*i*phase)` with
`phase in (-0.5, 0.5]`.  All irrep labels are one-dimensional.

## Usage

```python
from valleyscope.data.valley_irreps.catalog import get_irrep_phase_list
c3_table = get_irrep_phase_list("spinful_C3_phase_v1")
```
