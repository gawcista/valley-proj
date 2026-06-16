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
| A. Valley projection/readiness | Keep | Physically necessary and mostly implemented |
| B. Valley-preserving subspace space-group inventory | Keep/refine | Implemented pieces exist; should become the central public object |
| C. Generic representation object | Missing | Needs implementation before more EBR expansion |
| D. Generic Bilbao irrep matcher | Missing | Current C2/C3 matcher is prototype only |
| E. Generic reduced EBR builder | Partial | Reducer exists, but still relies on explicit mapping specs/tables |
| F. Exact reduced EBR solver | Keep | Implemented and should stay pure Python / SymPy |
| G. Database ingestion/index | Keep, pause expansion | Adequate for now; do not add more plumbing until C/D advance |

## Immediate Work Order

1. **Stop Cn-specific expansion.**
   Cancel the previous C3 database-contract task.  It is not wrong, but it is
   lower priority than fixing the generic irrep backend.

2. **Code audit and classification.**
   Produce a code-level inventory with each module marked:
   `core`, `prototype`, `debug`, `fixture`, or `delete-candidate`.

3. **Introduce a generic representation schema.**
   Add a small internal/public data object for the valley-projected subspace
   space group and its `(k, valley, G_k^(a))` little-group representation data.
   Existing outputs should be adapted to populate this object rather than
   emitting legacy `C{n}_like` hints as if they were the final result.

4. **Replace hard-coded C2/C3 matching with a strategy boundary.**
   Keep the phase tables as a fallback prototype strategy, but make the public
   API group-agnostic and ready for a Bilbao character matcher.

5. **Remove expected-HSP special policy from the production path.**
   EBR problem instances should derive required HSP/irrep basis from a reduced
   table or reviewed source reduction, not from a hand-coded legacy
   `C{n}_like` / `Pn` policy table.

6. **Implement the first generic Bilbao character-matching skeleton.**
   This can initially cover the same cyclic cases as the old C2/C3 matcher,
   but the API and tests must prove the logic is group-agnostic.

7. **Only after generic irrep matching works, return to reduced EBR table
   generation and database end-to-end tests.**

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
