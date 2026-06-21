# PLAN.md

This roadmap resets the ValleyScope implementation direction as of
2026-06-16.  The project target is not a collection of C2/C3 examples.  The
target is a general high-throughput workflow for valley-projected irreps and
valley-resolved reduced EBR decomposition in moire materials.

Stage-specific task details belong in `.codex_cc_handoff.md`; this file records
the physical method and implementation direction.

## Physical Target

For each sampled moire HSP `k` and valley label `a`, ValleyScope must first
identify the symmetry of the valley-projected subspace.  The final physical
object is the **valley-projected subspace space group** (a valley-preserving
subgroup/subspace group of the moire space group), together with its HSP
little-group representations.  It is not an effective `C2_like`/`C3_like`
label.

At each sampled HSP, the relevant local representation is the representation
of the **valley-preserving HSP little subgroup**

```math
G_k^{(a)} = \{g \in G_k \mid \pi_g(a)=a\},
\qquad
G_k = \{g \in G \mid gk = k + G_M\}.
```

The output object is the valley-projected representation

```math
D_a(g), \quad g \in G_k^{(a)},
```

with closure/unitarity/projector-symmetry diagnostics.  Irrep labels are then
matched against Bilbao/irreptables source conventions for the same restricted
operation set.  Valley-changing operations are not failures; they are
valley sewing matrix data and must not be forced into a single-valley irrep
label.

Reduced EBR decomposition is downstream:

```text
valley-projected irreps at sampled HSPs
-> Bilbao/irreptables source EBR data
-> ValleyScope reduction to sampled HSPs and G_k^(a) irreps
-> exact integer reduced EBR decomposition
```

This workflow must be space-group based and group-agnostic.  C2, C3, C4, C6,
mirrors, and products of generators are all operation content inside the
subspace space group / HSP little group, not separate production strategies or
final output categories.

## Keep As Core

These existing layers are physically necessary and should remain:

* HDF5/WAVECAR HSP input and target-band subspace handling.
* q-cut valley seed projectors `P_a^0`.
* seed projector symmetry-consistency diagnostics:

  ```math
  D_g P_a^0 D_g^\dagger \approx P_{\pi_g(a)}^0 .
  ```

* target-subspace closure diagnostics.
* valley orbit and valley mapping logic.
* HSP little group inventory.
* per-valley valley-preserving subgroup construction.
* direct q-cut trusted path when the fixed-center seed basis passes all
  readiness gates.
* symmetry-adapted projector path when the seed basis is not directly trusted
  but a physically valid valley-adapted subspace can be constructed.
* standard/debug output profile separation.
* exact-integer reduced EBR solver based on ValleyScope reduced tables.
* offline database ingestion/index collectors built from public outputs.

## Downgrade To Prototype Or Debug

The following are useful validation/prototype assets but must not define the
production architecture:

* `spinful_C2_phase_v1` and `spinful_C3_phase_v1` phase tables:
  keep as regression fixtures and fallback prototype data, not as the main
  irrep backend.
* legacy C2/C3-specific `subspace_group_candidate` policy:
  treat as an unfinished prototype hint.  It must not appear as the final
  subspace definition in public user-facing output.
* `P321_C3_like_GammaM_KM_spinful_v1`:
  keep as one legacy reviewed reduced-table validation fixture while the
  generic subspace-space-group reduction is being built.  It is not the model
  for all future systems.
* tMoTe2/tZrSe2 benchmark logic:
  keep under `real_tests/` and `docs/benchmarks/`; never use material names or
  material-specific branches in production code.
* HSP-star derived character reports:
  keep as debug/provenance until the generic representation matching path
  explicitly consumes them with a documented physical rule.
* Large database plumbing tasks:
  pause new expansion until the generic valley irrep representation backend is
  in place.

## Remove Or Replace Candidates

These are not to be deleted blindly.  They should be removed or replaced only
with tests proving that the generic path covers the behavior:

* Hard-coded C2/C3 dispatch in `valley_irrep_matching.py`.
* `_EXPECTED_HSP` policy table in `ebr_problem_instances.py`.
* Any public schema or summary text that presents `C2_like`/`C3_like` as the
  subspace definition or final output.  These names are legacy prototype hints,
  not physical space-group labels.
* Tests whose only purpose is to preserve C2/C3 special-case behavior instead
  of preserving physical group-agnostic behavior.
* Handoff tasks that only add more C2/C3 smoke coverage without advancing the
  generic representation and Bilbao-reduction backend.

## Correct General Workflow

### Layer 1 — Valley Projection And Readiness

Input:

```text
VASP WAVECAR / HDF5 HSP wavefunctions
space-group operations
target bands
valley centers
```

Output:

```text
P_a^0 seed projectors
valley weights
seed projector symmetry-consistency
target-subspace closure
HSP little group G_k
valley mapping pi_g(a)
valley-preserving subgroup G_k^(a)
```

Readiness gates must be physical:

* low valley weight alone is not an irrep failure;
* high valley purity alone is not irrep readiness;
* valley-changing operations test covariance under `pi_g`, not invariance;
* failed closure/projector-symmetry rows are diagnostic-only.

### Layer 2 — Generic Valley-Preserving Representation

Build a group-agnostic representation object for each `(k, valley)`:

```text
subspace_space_group_number / symbol / operation_ids
space_group_number
space_group_symbol
kpoint
valley
hsp_little_group_operation_ids
valley_preserving_operation_ids
valley_changing_operation_ids
operation_matrices D_a(g)
characters chi_a(g)
eigenvalues/eigenphases
closure/unitarity/projector diagnostics
readiness status
```

The core matching unit is the full character/eigenvalue data on
`G_k^(a)`, not a single Cn generator.  A cyclic subgroup can still be matched
from one generator when mathematically sufficient, but the implementation
should treat that as a special case of the generic representation object.

The public output must identify the subspace space group and HSP little-group
irreps.  It must not report `C2_like`, `C3_like`, or any future `C{n}_like`
string as the final physical irrep/EBR object.

### Layer 3 — Bilbao/irreptables Irrep Matching

Use Bilbao-derived `irreptables` source data as the convention standard.
The matching path should:

1. identify the source space-group irreps at the relevant HSP;
2. map source operations to ValleyScope operation IDs;
3. restrict source characters to `G_k^(a)`;
4. compare the ValleyScope representation character against restricted source
   irreps using an explicit tolerance and convention record;
5. support reducible representations by returning multiplicities, not only
   one-dimensional phase labels;
6. report unmatched or ambiguous rows as diagnostic-only.

The existing C2/C3 phase tables remain useful tests for the cyclic, rank-1
case, but they should be bypassable by the generic Bilbao character matcher.

### Layer 4 — Reduced EBR Table Construction

Reduced EBR tables must be generated by applying the same physical reduction
to Bilbao/irreptables EBR data:

```text
full source EBR vector
-> sampled moire HSP set
-> source-HSP operation map
-> valley-preserving subgroup restriction
-> ValleyScope irrep basis multiplicities
-> reduced EBR matrix
```

No raw 3D decomposition is a ValleyScope final result.  Raw source data is
input evidence only.

### Layer 5 — Reduced EBR Decomposition And Database Output

Use the existing pure Python / SymPy exact solver on the reduced EBR matrix.
Database ingestion should consume public outputs only:

```text
valley_summary.json
valley_ebr_export_bundle.json
valley_reduced_ebr_mapping.json
```

Further database index work should pause unless it directly supports the
generic irrep/EBR representation record.

## Revised Phase Status

| Phase | Status | Meaning |
|-------|--------|---------|
| A. Valley projection/readiness | Core | Physically necessary and mostly implemented |
| B. Valley-preserving subspace space-group inventory | Core | `subspace_space_group` is the primary public identifier; `valley_projected_representations.representation_records` is the group-agnostic output layer |
| C. Generic representation object | Implemented | `build_valley_projected_representation_report()` produces per-`(kpoint, valley)` grouped records with `subspace_space_group`, HSP little group ops, VP/VC ops, characters/eigenphases, readiness, and irrep matching status |
| D. Generic Bilbao irrep matcher | Implemented | `match_restricted_characters()` in `generic_irrep_matching.py` (Bilbao restricted-character matching over full `G_k^(a)`); strategy boundary in `build_valley_irrep_matching_report()` enforces generic/legacy mode separation |
| E. Generic reduced EBR builder | Implemented | `build_reduced_table_from_runtime_source()` and `build_reduced_table_from_irreptables()` (with injectable `source_loader`); both use physical `subspace_group_candidate` |
| F. Exact reduced EBR solver | Core | Pure Python / SymPy; synthetic P4 E2E contract passes through the full pipeline |
| G. Database ingestion/index | Core, paused expansion | Ingestion record consumes public outputs; pause new expansion until real-material validation advances |

## Completed Work (2026-06-16 through 2026-06-21)

1. **Generic representation record contract.**
   `build_valley_projected_representation_report()` extended with
   `valley_irrep_matching` parameter and `representation_records` output.
   Each `(kpoint, valley)` record carries the physical `subspace_space_group`
   as the primary identifier, with `legacy_subspace_group_candidate` for
   provenance only.  P4/order-4 test proves group-agnostic behavior.

2. **Generic irrep matching strategy boundary.**
   `build_valley_irrep_matching_report()` enforces a clean `matching_mode`
   (`"generic"` / `"legacy"`).  In generic mode, legacy `by_kpoint` phase-table
   entries for `(kpoint, valley)` pairs with generic coverage are suppressed.
   `ebr_input_candidates()` gates legacy promotion on `matching_mode`.

3. **Generic public output contract.**
   `subspace_group_candidate` in `generic_matches_by_kpoint` entries now uses
   the physical `candidate_space_group_symbol` when available, via
   `_generic_group_identity()`.  Public outputs consistently use physical
   subspace-space-group symbols; `C{n}_like` labels appear only as
   `legacy_subspace_group_candidate` provenance.

4. **Reduced EBR E2E contracts.**
   Synthetic P4 E2E test through `build_reduced_table_from_runtime_source()`
   and `build_reduced_table_from_irreptables()` (with fake `source_loader`).
   Both paths validate generic provenance: `data_source`, `package`,
   `space_group_number`, `spinful`, `expected_hsps`, `subspace_group_candidate`.

5. **Public output contract test.**
   `test_p4_public_output_contract` validates all standard outputs use
   physical subspace-space-group symbols and forbids Cn-like promotion.
   `test_generic_irrep_positive_analyze_hsp_workflow_e2e` strengthened
   with representation_record contract assertions.

## Immediate Work Order

1. **Remove remaining legacy C2/C3 dependencies.**
   `match_valley_irrep()` and `_irrep_table_for_order()` in
   `valley_irrep_matching.py` are only used in legacy mode.  Once the
   generic source-payload adapter is proven for all supported space groups
   in real end-to-end runs, the legacy phase-table code can be quarantined
   or removed.

2. **Real-material validation with P321/P312 fixtures.**
   tMoTe2 and tZrSe2 benchmarks in `real_tests/` and `docs/benchmarks/`
   can validate the generic pipeline but must NOT drive production logic.
   The next step is to produce a trusted P3 generic irrep match for the
   tMoTe2 P321 K-valley fixture.

3. **Reduced EBR table provenance for reviewed package-data tables.**
   The reviewed `P321_C3_like_GammaM_KM_spinful_v1` table still uses
   `C3_like` as its provenance identity.  When the tMoTe2 P3 generic
   match is trusted, update the reviewed table to `P3`.

## Validation Fixtures

Real materials are validation fixtures only:

* tMoTe2: clean P321/K-valley fixture for direct q-cut readiness and
  valley-preserving HSP little-group behavior.
* tZrSe2: P312/M-star fixture for blocked or difficult valley-preserving
  subgroup behavior.
* Future C4/C6/mirror systems should be added as fixtures when available, but
  never as production branches.

Fixture outcomes may validate the generic workflow.  They must not become
case-specific implementation rules.

## Success Criteria

The next credible milestone is not "more C3/C2 tests".  It is:

```text
For an arbitrary sampled HSP and valley label, ValleyScope emits a
valley-projected subspace-space-group record plus an HSP little-group
representation record for G_k^(a), including operation IDs,
matrices/characters/eigenphases, readiness, and matching status.
```

The following milestone is:

```text
The representation record is matched against Bilbao/irreptables restricted
characters without hard-coded C2/C3 logic.
```

Only then should we claim general valley-resolved irreps.  Reduced EBR
decomposition follows after the same restricted-basis convention is used to
build reduced EBR matrices.
