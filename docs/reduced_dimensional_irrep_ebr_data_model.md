# Reduced-Dimensional Irrep/EBR Data Model Design

Date: 2026-06-10 | Status: Package-data skeleton implemented (no table data)

This document proposes a future ValleyScope reviewed/provenance-tracked
package-data layer for valley-preserving irreps and reduced-dimensional
EBR vectors.  It is a design/audit document, not a specification for
immediate implementation.  No data loader, built-in table, or new
dependency is added by this document.

## 1. Motivation

Current ValleyScope irrep matching uses two tiny versioned package data
phase tables in `valleyscope/data/valley_irreps/` for spinful C3 and C2
phase-to-label matching only.  These are valley-preserving irrep matching
tables, not reduced EBR tables, and they contain no EBR vectors.  The
external-table path
(`valleyscope/analysis/reduced_ebr_mapping.py`) requires user-supplied
validated tables with no built-in data.

For high-throughput database use, a structured, versioned, provenance-tracked
package-data layer is needed — not an accumulation of ad hoc tables, and not
raw 3D EBR decomposition reported as ValleyScope output.  This is no direct reuse
as final output.  The public Python package `irrep` may be used as a runtime data source
for 3D space-group irrep/EBR data.  ValleyScope then performs the required
**valley-resolved reduced-dimensional reduction**: sampled moire HSP set, HSP
little group, valley mapping, valley-preserving subgroup, and valley-specific
operations reduce the full 3D space-group irrep/EBR basis to a 2D
valley-preserving subset.

## 2. Design Inspirations

### 2.1 `irrep2` — reduced-dimensional decomposition

Local repository: `/home/gawcista/Working/database/irrep2` (available for
inspection during this design task).  This repository is not public and must
remain reference-only: do not add it as a dependency, import path, vendored
source, or required runtime asset.

Key patterns:
- **Reduced basis**: the full 3D space-group irrep set is reduced to the
  subset of irreps whose HSP labels match the actually-sampled k-points
  (`_basis_indices = [index for index, label in enumerate(basis_labels)
  if extract_hsp_label(label) in actual_hsps]`).  Only this reduced subset
  enters the EBR decomposition.
- **Structured data models**: Python dataclasses (`BandManifoldRecord`,
  `ReducedEBRResult`, `ReducedEBRSolution`) with `to_dict()` serialization.
- **Smith normal form**: exact integer decomposition via sympy
  `smith_normal_decomp` for existence checking.
- **Nonnegative search**: recursive enumeration for physically meaningful
  nonnegative EBR coefficient vectors.
- **Status taxonomy**: `atomic-compatible-candidate`,
  `fragile-topology-candidate`, `stable-topology-candidate` — physically
  meaningful categories, not pass/fail flags.

### 2.2 `irrep` Python package — packaged-data architecture

Python package: `irrep` v2.6.3 (installed locally; inspected for data-model
patterns only).  Not to be confused with ValleyScope's physical irrep objects
or `valley_irrep_matching.py`.

Current direction: `irrep` can become a runtime backend/data source for 3D
space-group irreps and EBR vectors, but ValleyScope must own the reduction
from that 3D basis to sampled-HSP valley-preserving reduced EBR matrices.  Do
not call or report `irrep`'s raw 3D EBR decomposition as the final
valley-resolved reduced EBR result.

Key patterns:
- **Versioned package data**: data files shipped inside the package's `data/`
  directory, versioned alongside the code.
- **EBR data structure**: each EBR table entry has `ebr_name`, `wyckoff_position`,
  `vector` (integer list aligned to irrep basis), and precomputed `smith_form`
  with `u`, `r`, `v` matrices.
- **Dataclass-based API**: `get_ebr_matrix()`, `get_smith_form()`,
  `get_ebr_names_and_positions()` — well-typed accessors, not raw dict
  traversal.
- **Validation at load time**: table loading includes structure checks before
  computation begins.

### 2.3 Synthesis for ValleyScope

| Aspect | `irrep2` pattern | `irrep` pattern | ValleyScope adaptation |
|--------|-----------------|-----------------|----------------------|
| Data model | `dataclass(slots=True)` | `dataclass` + accessor functions | `dataclass(slots=True)` with `to_dict()` |
| Data storage | CSV in `irdata/` | JSON in `data/` | Versioned JSON in `valleyscope/data/` |
| Versioning | Implicit (git) | Package version | Explicit `schema_version` + git hash provenance |
| Basis reduction | HSP-filtered indices | Static full 3D table | Valley-preserving subgroup + HSP-filtered |
| Decomposition | Smith normal form | Precomputed Smith form | Exact integer solve; public `or-tools` may be used if justified |
| Validation | Record-level status checks | Load-time structure checks | Load-time + schema contract tests |

## 3. Physical Objects

The data model must distinguish these physical concepts.  Every packaged table
must document which physical object it represents.

### 3.1 HSP Little Group

```
G_k = { g in G | g k = k + G_M }
```

The subgroup of the moire space group that leaves the HSP k-point invariant
(modulo reciprocal lattice).  This is the local symmetry group at each
sampled momentum.  An operation may belong to `G_k` at one k-point but not
at another (e.g., C2 at MM vs KM in a hexagonal cell).

### 3.2 Valley Mapping

```
pi_g(a) = the valley label that operation g maps valley a to
```

The operation-induced permutation of valley labels.  Defined from the valley
center positions and the operation's real-space action on monolayer
coordinates.  A valley belongs to a valley orbit under the space group.

### 3.3 Valley-Preserving Subgroup

```
G_k^{(a)} = { g in G_k | pi_g(a) = a }
```

The subgroup of the HSP little group that preserves a specific valley label.
This is the relevant symmetry group for valley-resolved irrep matching:
operations that change the valley label are tracked as valley-changing data
but do not enter the valley-preserving eigenvalue/character table.

Valley-preserving subgroups for different members of a valley orbit are
related by conjugation:
```
G_k^{(M2)} = C3 G_k^{(M1)} C3^{-1}
```

### 3.4 Valley-Preserving Irrep

A representation of `G_k^{(a)}` in the valley-adapted target subspace.  The
character `chi_a(g) = Tr D_a(g)` is computed for each valley-preserving
operation.  For spinful systems, double-valued irreps are required
(``C3^3 = -1``, ``C2^2 = -1``).

Label format in current production: `C{order}_spinor_phase_{sign}{fraction}`,
e.g. `C3_spinor_phase_+1/2`, `C2_spinor_phase_+1/4`.

### 3.5 Valley Sewing Matrix

For valley-changing operations (``pi_g(a) != a``), the representation data is
not a character but a sewing matrix `S_g` that maps the valley-a subspace to
the valley-``pi_g(a)`` subspace.  This is essential for full valley-orbit
representations but is **not** required for single-valley reduced EBR
decomposition; single-valley EBR only needs valley-preserving irreps.

### 3.6 Reduced-Dimensional EBR Vector

An integer vector `c_i` of EBR multiplicities that, when projected onto the
reduced irrep basis (the subset of irreps at actually-sampled HSPs for a
specific valley-preserving subgroup), matches the bundle's irrep count vector:

```
sum_i c_i * EBR_i[reduced_basis] = bundle_irrep_counts
```

The "reduced-dimensional" qualifier means:
- Only irreps at the sampled HSPs are in the basis (not all HSPs in the 3D BZ).
- Only valley-preserving irreps enter (valley-changing operations are excluded).
- The EBR vectors are the full vectors, but only the reduced subset of columns
  (those matching the sampled HSPs + valley-preserving irreps) participates in
  the decomposition.

## 4. Label Conventions

### 4.1 Crystallographic vs Effective vs Package-Data Labels

| Context | Notation | Example | Source |
|---------|----------|---------|--------|
| Crystallographic (full space group) | Hermann-Mauguin symbol | `P3`, `P2`, `P321`, `P312` | spglib detection |
| `subspace_space_group` | Crystallographic candidate | `P3`, `P2` | `_build_subspace_space_group_for_valley()` |
| `subspace_group_candidate` (EBR table label) | `C{max_order}_like` | `C3_like`, `C2_like`, `C1` | `_subspace_group_candidate_from_orders()` |
| Future package-data identifier | TBD — must include valley-preserving subgroup + irreps | e.g. `P321_C3_like_GammaM_KM` | Proposed |

External reduced EBR tables must use the `C{order}_like` form for
`subspace_group_candidate`.  The crystallographic notation (`P3`, `P2`)
appears in `subspace_space_group.candidate_space_group_symbol` and is
informational, not the EBR table key.

A future reviewed package-data layer should define **unambiguous canonical
identifiers** that encode: space group + valley-preserving subgroup +
sampled HSP set.  Example pattern:
```
<space_group>_<subgroup_candidate>_<hsp_list>
P321_C3_like_GammaM_KM
P312_C2_like_GammaM_KM_MM
```

This avoids the ambiguity of `C3_like` alone (which doesn't tell you which
space group or HSP set the table covers).

## 5. Proposed Package-Data Layout

### 5.1 Directory Structure

```
valleyscope/
  data/                              # New top-level package-data directory
    __init__.py                      # (empty)
    README.md                        # Data provenance and usage notes
    valley_irreps/                   # Minimal valley-preserving irrep phase tables
      manifest.json
      spinful_C3_phase_v1.json
      spinful_C2_phase_v1.json
    reduced_ebr/                     # Reduced-dimensional EBR tables
      P321_C3_like_GammaM_KM.json
      P312_C2_like_GammaM_KM_MM.json
      ...
    tests/                           # Package-data validation tests
      test_data_consistency.py
```

### 5.2 File Format (JSON)

Each table file is a versioned JSON document.  Proposed schema:

```json
{
  "schema_version": "1.0.0",
  "provenance": {
    "source": "reviewed_literature",
    "reference": "Phys. Rev. B XX, YYYYYY (20XX)",
    "reviewer": "initials",
    "review_date": "2026-XX-XX",
    "git_commit_at_review": "abc1234"
  },
  "subspace_group_candidate": "C3_like",
  "space_group_international": "P321",
  "space_group_number": 150,
  "spinor": true,
  "expected_hsps": ["GammaM", "KM"],
  "optional_hsps": ["MM"],
  "valley_preserving_subgroup": {
    "description": "C3 valley-preserving subgroup of P321 at GammaM and KM",
    "generators": ["C3"],
    "operation_ids_in_standard_orientation": [0, 1, 2]
  },
  "irreps": [
    "GammaM:C3_spinor_phase_+1/2",
    "KM:C3_spinor_phase_+1/6",
    "KM:C3_spinor_phase_-1/6"
  ],
  "irrep_metadata": {
    "GammaM:C3_spinor_phase_+1/2": {
      "character": "exp(i*pi) = -1",
      "dimension": 1,
      "degeneracy": 1
    },
    "KM:C3_spinor_phase_+1/6": {
      "character": "exp(i*pi/3)",
      "dimension": 1,
      "degeneracy": 1
    },
    "KM:C3_spinor_phase_-1/6": {
      "character": "exp(-i*pi/3)",
      "dimension": 1,
      "degeneracy": 1
    }
  },
  "ebrs": [
    {
      "label": "EBR_(A)1a",
      "wyckoff_position": "1a",
      "site_symmetry": "P3",
      "vector": [1, 0, 1],
      "notes": "toy example only — not a reviewed physical EBR"
    },
    {
      "label": "EBR_(B)1a",
      "wyckoff_position": "1a",
      "site_symmetry": "P3",
      "vector": [1, 1, 0],
      "notes": "toy example only — not a reviewed physical EBR"
    }
  ],
  "validation": {
    "checksum_sha256": "<sha256 of irrep + ebr sections>",
    "integer_vector_consistency_checked": true,
    "character_table_consistency_checked": false,
    "reviewed_by_human": false
  }
}
```

### 5.3 Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Data format version |
| `provenance` | object | Source, reference, reviewer, date, git commit, and physical identity provenance |
| `provenance.source` | string | `"reviewed_literature"`, `"benchmark_verified"`, etc. |
| `provenance.data_source` | string | Source data package or table family |
| `provenance.space_group_number` | int or string | Source space-group number |
| `provenance.spinful` | bool | Whether double-group irreps are used |
| `provenance.subspace_group_candidate` | string | Must match table `subspace_group_candidate` |
| `provenance.expected_hsps` | list[string] | Must match table `expected_hsps` |
| `provenance.central_sign_convention` | string | Spinor lift / central-sign convention, or `"not_applicable"` |
| `subspace_group_candidate` | string | `C{order}_like` label matching export bundles |
| `space_group_international` | string | Hermann-Mauguin symbol |
| `space_group_number` | int | International Tables number |
| `spinor` | bool | Whether double-group irreps are used |
| `expected_hsps` | list[string] | HSP labels covered by this table |
| `irreps` | list[string] | Irrep keys in `kpoint:label[:opN]` format |
| `ebrs` | list[object] | EBR definitions with `label`, `vector` |

The physical identity provenance is part of the reviewed package-data gate.
It records the HSP little group, valley mapping, and valley-preserving subgroup
context that defines the reduced-dimensional table. It also records when
valley-changing operation information is excluded as valley sewing matrix data
rather than included in the reduced EBR basis.

### 5.4 Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `optional_hsps` | list[string] | HSPs not required for completeness |
| `valley_preserving_subgroup` | object | Generators, operation IDs |
| `irrep_metadata` | object | Per-irrep character, dimension, degeneracy |
| `ebrs[].wyckoff_position` | string | Wyckoff position label |
| `ebrs[].site_symmetry` | string | Site-symmetry group |
| `ebrs[].notes` | string | Reviewer notes |
| `validation` | object | Checksum and consistency flags |

## 6. Integration With Existing Paths

### 6.1 External-Table CLI (`valleyscope map-reduced-ebr`)

The CLI already accepts user-supplied JSON tables.  A future package-data
layer would **not change the CLI interface**.  Instead, the table files
shipped in `valleyscope/data/reduced_ebr/` would be validated, reviewed
versions that can be passed to the CLI:

```bash
valleyscope map-reduced-ebr \
  valley_ebr_export_bundle.json \
  $(python -c "import valleyscope; print(valleyscope.__path__[0])")/data/reduced_ebr/P321_C3_like_GammaM_KM.json
```

Or, with a convenience lookup (future):
```bash
valleyscope map-reduced-ebr \
  valley_ebr_export_bundle.json \
  --builtin P321_C3_like_GammaM_KM     # future convenience, not in this design
```

### 6.2 `analysis.reduced_ebr` Config Path

The existing `analysis.reduced_ebr.enabled` + `table_file` config path
is the external user-table path.  No static reviewed package-data tables are
shipped, and no config key selects built-in reduced EBR package data by name.
Bilbao/irreptables-derived reduced tables are built through the runtime
reducer and may be supplied explicitly through
`analysis.reduced_ebr.table_file`.


### 6.3 Public Schema Preservation

No change to `docs/schema.md` or the public `valley_reduced_ebr_mapping.json`
output schema.  The package-data files are **inputs** to the decomposition,
not outputs.

## 7. Validation Rules Before Data Can Be Trusted

Before any packaged table is shipped as reviewed ValleyScope data, all of
the following must pass:

1. **Exact label matching**: `subspace_group_candidate` in the table must
   match the export bundle.  No fuzzy or order-only matching.
2. **No HSP-only fallback**: irrep keys use full `kpoint:label[:opN]` format.
   HSP-only key resolution (without kpoint prefix) is not accepted for
   trusted data.
3. **Integer vector consistency**: every EBR vector is aligned to the
   `irreps` list, contains nonnegative integers, and the set of labels is
   unique.
4. **Character table consistency**: if `irrep_metadata` includes character
   data, it must be consistent with the irrep labels and the valley-preserving
   subgroup structure.
5. **Documented convention and source**: `provenance.source`,
   `provenance.reference`, and `provenance.reviewer` must be non-empty.
6. **Schema version match**: the loader must check `schema_version` and
   reject unknown versions.
7. **Checksum integrity** (optional but recommended): a SHA-256 checksum
   of the irrep + EBR sections guards against accidental corruption.

## 8. Non-Goals

- **No raw 3D `irrep` decomposition as final output.** The public Python
  package `irrep` may be a runtime source for 3D irreps/EBR vectors, but
  ValleyScope's target is valley-resolved reduced-dimensional irreps/EBRs
  after sampled-HSP and valley-preserving reduction.
- **No `irrep2` runtime dependency.**  The local `irrep2` repository is useful
  as a reduced-dimensional reference implementation, but it is not public and
  must not be imported, vendored, or required by ValleyScope.
- **No unreviewed built-in tables.**  Every shipped table must pass the
  validation rules above and have documented provenance.
- **No compatibility relations yet.**  Reduced compatibility relations are a
  separate physical problem (see `irrep2` for the approach).
- **No heuristic or floating-point EBR fitting.**  Exact integer matching
  only, via Smith normal form or brute-force search.
- **No material-specific program branches.**  The data model is generic;
  material names appear only in provenance/reference metadata, not in
  program logic.

## 9. Inspection Notes

### 9.1 `/home/gawcista/Working/database/irrep2`

Available for inspection.  Key observations:
- `ebr_reduced.py`: Smith normal form integer decomposition, nonnegative
  search, duplicate-column grouping, HSP-reduced basis filtering.
- `models.py`: `dataclass(slots=True)` with `to_dict()` — clean, typed,
  serializable.
- `normalize.py`: `load_ebr_data()` returns structured EBR tables from
  package data in `irdata/`.
- `classifier.py`: combines compatibility + reduced EBR into final
  classification.

### 9.2 Python package `irrep` v2.6.3

Installed locally.  Key observations:
- `ebrs.py`: `get_ebr_matrix()`, `get_smith_form()`,
  `get_ebr_names_and_positions()` — typed accessors over dict-based EBR data.
- EBR data structure: dict with `ebrs` (list of `{ebr_name, wyckoff_position,
  vector}`), `basis` (irrep_labels, degeneracies), precomputed `smith_form`.
- OR-Tools SAT solver for nonnegative decomposition — ValleyScope should
  prefer Smith normal form + recursive search to avoid heavy dependencies.

### 9.3 Package name clarification

The Python package is named `irrep` (singular).  This design document uses
`irrep` to refer to the external Python package and reserves "irrep" /
"valley-preserving irrep" / "irrep matching" for ValleyScope's physical
concepts.  Previous versions of this handoff incorrectly used `irreps`
(plural); those occurrences have been corrected in `AGENTS.md` and `PLAN.md`.

## 10. Implementation Roadmap

1. ~~**This design document**~~ — reviewed and approved (commit `12135c4`).
2. ~~**Stub data directory**~~ — `valleyscope/data/reduced_ebr/` skeleton
   created with empty manifest, README, and catalog module. No table data.
3. **Runtime/external loader** — use
   `valleyscope.analysis.reduced_ebr_mapping.load_reduced_ebr_table()` for
   explicit external table files and the irreptables runtime reducer for
   Bilbao-derived reduced tables.
4. ~~**Static reviewed package-data tables**~~ — removed from the current
   production path. Static JSON payloads remain only as test fixtures.
5. ~~**Convenience CLI builtin selector**~~ — no CLI selector is provided for
   static package data.
6. ~~**Config integration**~~ — `analysis.reduced_ebr.table_name` was
   removed. Use external `table_file` or the irreptables runtime reducer.
