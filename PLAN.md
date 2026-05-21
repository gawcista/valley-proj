# PLAN.md

This roadmap reflects the current ValleyScope methodology. It separates
q-cut valley seed diagnostics from the later construction of
symmetry-adapted valley projectors.

## Current Code Baseline

The repo already has:

* WAVECAR extraction with `ecut_adjust_tol`;
* HDF5 intermediate wavefunction input;
* q-cut valley projection;
* multi-valley basis diagnostics using projected seed matrices;
* valley orbit and valley mapping logic;
* moire HSP symmetry representation matrices;
* `symmetry_eigenvalues.csv`;
* `diagnostics.h5`;
* prototype irrep matching;
* seed projector symmetry-consistency diagnostic output.

These pieces should be retained and re-layered. The q-cut basis remains a seed
and diagnostic layer. Trusted valley-preserving irreps should eventually come
from symmetry-adapted valley projectors `P_a^sym`.

## Methodology Reset

For each moire HSP k, define the target DFT band subspace

```math
H_k = \mathrm{span}\{|\psi_{n,k}\rangle\}.
```

Inside this subspace:

1. `P_a^0` is the q-cut valley seed projector.
2. `D_g` is the moire HSP symmetry representation matrix.
3. `pi_g(a)` is the operation-induced valley mapping.
4. The seed diagnostic checks

   ```math
   D_g P_a^0 D_g^\dagger \approx P_{\pi_g(a)}^0 .
   ```

5. Future trusted valley irrep extraction requires `P_a^sym`, a
   symmetry-adapted valley projector.

Do not treat high q-cut valley purity as irrep readiness. Do not test
valley-changing operations as invariance of a single label operator.

## Required Quality Metrics

Per material, HSP, operation, and valley orbit, the project should eventually
record:

* `seed_projector_symmetry_error`;
* future symmetry-adapted projector symmetry error;
* seed overlap between `P_a^sym` and `P_a^0`;
* rank selection diagnostics;
* orthogonality error;
* completeness error;
* valley sewing matrices;
* sewing unitarity error;
* valley-preserving representation;
* irrep matching status.

Status values should distinguish `matched`, `diagnostic_only`,
`failed_symmetry_consistency`, `failed_rank_selection`, `failed_seed_overlap`,
`failed_table_mapping`, `spinor_convention_unverified`,
`insufficient_target_subspace`, and explicit other reasons.

## Phase 0: Freeze Old Interpretation

Goal: prevent q-cut seed basis from being interpreted as a trusted irrep basis.

Deliverables:

* README / README.zh / AGENTS / PLAN state that q-cut seed projectors are
  diagnostic.
* Existing q-cut irrep-like outputs are marked diagnostic-only when readiness
  gates fail.
* tZrSe2 M-star failure mode is documented as a regression benchmark.

cc implementation prompt:

```text
Do not implement new physics. Audit docs and summaries so q-cut valley seed
projectors P_a^0 are described only as seed diagnostics. Ensure failed
readiness checks cannot be presented as trusted valley-preserving irreps.
```

Codex review checklist:

* no tolerance loosening;
* no wording that equates q-cut purity with irrep readiness;
* no public schema using old projector or subgroup terminology.

Tests:

* targeted summary/schema tests;
* no large WAVECAR data.

Blockers:

* unresolved public schema migration requirements.

Non-goals:

* no `P_a^sym`;
* no reduced EBR;
* no Berry curvature, Wilson loop, or Chern number.

## Phase 1: Projector Symmetry-Consistency Diagnostics

Goal: report whether q-cut valley seed projectors satisfy

```math
D_g P_a^0 D_g^\dagger \approx P_{\pi_g(a)}^0 .
```

Deliverables:

* `projector_symmetry_report.json`;
* `valley_summary.json.projector_symmetry`;
* `seed_projector_symmetry_error` per `(kpoint, operation, valley)`;
* readiness demotion when the check fails;
* warning text that affected q-cut seed-basis labels are diagnostic-only.

cc implementation prompt:

```text
Implement or maintain the seed projector symmetry-consistency diagnostic.
Use D_raw in the target DFT subspace and valley mapping pi_g(a). Output
epsilon_seed as seed_projector_symmetry_error. Do not build P_a^sym and do not
modify the irrep basis.
```

Codex review checklist:

* `D_g P_a D_g^\dagger` is compared with `P_{pi_g(a)}`;
* valley-changing operations are included in the diagnostic;
* failure demotes q-cut seed-basis rows to diagnostic-only;
* schema uses `projector_symmetry`, not old projector terms.

Tests:

* exact identity case;
* exact valley swap case;
* C3 three-valley cyclic mapping;
* C2 fixes one valley and swaps two;
* deliberately non-symmetry-consistent seed;
* missing mapping not evaluated;
* workflow writes `projector_symmetry_report.json`;
* compact summary exists;
* failed seed check sets diagnostic-only / readiness false.

Blockers:

* no seed matrix when target bands are single-band or not near-degenerate.

Non-goals:

* no symmetry-adapted projectors;
* no reduced EBR.

## Phase 2: Symmetry-Adapted Valley Projector Prototype

Goal: implement a toy-only prototype for `P_a^sym` construction.

Recommended route:

1. Choose a reference valley in each orbit.
2. Identify the valley-preserving subgroup

   ```math
   G_k^{(a)} = \{g \in G_k \mid \pi_g(a)=a\}.
   ```

3. Average the reference seed over this subgroup.
4. Hermitian symmetrize.
5. Purify by spectral decomposition with rank diagnostics.
6. Generate the other valley projectors using audited orbit mappings.
7. Check symmetry-consistency, orthogonality, completeness, seed overlap, and
   valley sewing unitarity.

Deliverables:

* toy-only module, not connected to `analyze_hsp`;
* rank selection report;
* projector quality report;
* exact synthetic tests.

cc implementation prompt:

```text
Build a toy-only prototype for symmetry-adapted valley projectors P_a^sym.
Do not connect it to the production workflow. Fail explicitly if orbit mapping
or rank selection is ambiguous. Output diagnostics for rank, seed overlap,
orthogonality, completeness, projector symmetry-consistency, and sewing
unitarity.
```

Codex review checklist:

* no fake strict coset decomposition;
* failures are explicit;
* no production workflow integration;
* exact toy answers are checked.

Tests:

* three-valley C3 orbit;
* C2 fixes one valley and swaps two;
* non-symmetry-consistent seed fails or reports poor diagnostics;
* rank gap ambiguity is reported.

Blockers:

* deciding robust rank inference for high-throughput real data.

Non-goals:

* no real-material production use;
* no reduced EBR.

## Phase 3: Symmetry-Adapted Valley-Irrep Workflow

Goal: use `P_a^sym` basis for trusted valley-preserving irrep matching after
Phase 2 is reviewed.

Deliverables:

* guarded workflow path using `P_a^sym`;
* q-cut seed basis retained as diagnostic;
* valley-preserving representation in `P_a^sym H_k`;
* valley sewing matrices for valley-changing operations;
* full HSP little-group data kept separate from valley-preserving irreps.

cc implementation prompt:

```text
Integrate reviewed P_a^sym artifacts into the irrep workflow behind an
explicit readiness gate. Trusted valley-preserving irreps must come from the
symmetry-adapted basis. Keep q-cut seed-basis rows diagnostic-only when the
seed projector symmetry-consistency check fails.
```

Codex review checklist:

* full-group irrep, HSP little group, valley orbit, valley-preserving subgroup,
  valley-preserving irrep, and valley sewing matrix are separated;
* state labels are emitted only when all readiness checks pass;
* old q-cut labels remain diagnostic-only when appropriate.

Tests:

* exact local representation toy;
* K/Kp P3 toy;
* three-valley M-star toy;
* schema and summary tests.

Blockers:

* reviewed design for storing `P_a^sym` artifacts in JSON/HDF5.

Non-goals:

* no reduced EBR implementation.

## Phase 4: Real Benchmarks

Goal: validate the workflow against local real examples without adding large
data to the repository.

Benchmarks:

* tMoTe2 K/Kp: P321, valley-preserving P3-like behavior, spinful C3 phases.
* tZrSe2 M-star: P312, M1/M2/M3 orbit, C2 operations that fix one valley and
  exchange the other two.

Expected record:

* old q-cut seed basis may fail projector symmetry-consistency;
* new `P_a^sym` should reduce the relevant leakage if the method works;
* large deviation between `P_a^sym` and `P_a^0` is a meaningful failure mode,
  not something to hide with tolerances.

cc implementation prompt:

```text
Run local real-data smoke checks only. Do not commit large WAVECAR/HDF5/output
artifacts. Record metrics and failure modes in a small text or JSON summary.
```

Tests:

* no large files in repo;
* toy tests still pass.

Non-goals:

* no high-throughput database run yet.

## Phase 5: Database Pipeline

Goal: prepare high-throughput summary contracts after the projector and irrep
workflow is reliable.

Deliverables:

* config schema for database runs;
* machine-readable quality flags;
* summary tables for material / HSP / valley orbit;
* clear failure categories.

Non-goals:

* no reduced EBR until valley-resolved irreps are reliable.

## Phase 6: Valley-Resolved Reduced EBR Planning

Goal: design reduced EBR decomposition only after trusted valley-resolved irreps
exist.

Deliverables:

* design note connecting full-group irreps, valley-preserving irreps, valley
  sewing matrices, and reduced EBR tables;
* review questions for conventions and table sources.

Non-goals:

* no implementation before design review.

## First cc Task Prompts

### Task 1: Projector Symmetry-Consistency Diagnostics

Branch: `cc/projector-symmetry-diagnostics`

Context:

q-cut valley seed projectors `P_a^0` are diagnostics. They are not trusted irrep
bases unless they pass the projector symmetry-consistency check.

Files to inspect:

* `valleyscope/workflows/analyze_hsp.py`
* `valleyscope/analysis/projector_symmetry.py`
* `valleyscope/analysis/symmetry_eigenvalue_diagnostic.py`
* `valleyscope/reports/analysis_outputs.py`
* `valleyscope/reports/summary_report.py`
* `tests/test_projector_symmetry.py`
* `tests/test_io_and_workflow.py`

Requirements:

* Use `D_raw` before per-valley filtering.
* Compute `seed_projector_symmetry_error`.
* Write `projector_symmetry_report.json`.
* Add `projector_symmetry` to `valley_summary.json`.
* Demote affected q-cut rows to diagnostic-only on failure.
* Do not implement `P_a^sym`.

Formula:

```math
\epsilon_{\rm seed}(g,a)
=
\|D_g P_a^0 D_g^\dagger - P_{\pi_g(a)}^0\|_F
/
\max(\|P_a^0\|_F, {\rm small}).
```

Tests:

* exact identity;
* valley swap;
* C3 cyclic;
* C2 fixed-valley plus swapped pair;
* non-symmetry-consistent seed;
* missing mapping;
* workflow report writing;
* readiness demotion.

Non-goals:

* no symmetry-adapted projector;
* no reduced EBR, Berry curvature, Wilson loop, or Chern number.

Final handoff:

* branch name;
* commit hash;
* changed files;
* formulas;
* schema additions;
* pytest result;
* known limitations.

### Task 2: Symmetry-Adapted Projector Toy Prototype

Branch: `cc/symmetry-adapted-projector-toy-prototype`

Context:

Trusted valley-resolved irreps require `P_a^sym`, but the first implementation
must be toy-only and isolated from the production workflow.

Files to inspect:

* `valleyscope/analysis/projector_symmetry.py`
* the internal valley-preserving subgroup helper module
* `tests/test_projector_symmetry.py`
* `tests/test_symmetry.py`

Requirements:

* Implement a toy-only projector construction module.
* Use valley-preserving subgroup averaging for a reference valley.
* Purify with rank diagnostics.
* Generate orbit-related projectors only from audited mappings.
* Report seed overlap, orthogonality, completeness, projector
  symmetry-consistency, and sewing unitarity.
* Fail explicitly on ambiguous rank or mapping.

Non-goals:

* no production workflow integration;
* no real-data benchmark;
* no reduced EBR.

Final handoff:

* branch name;
* commit hash;
* changed files;
* algorithm summary;
* tests and pytest result;
* failure modes.

### Task 3: Symmetry-Adapted Irrep Workflow Draft

Branch: `cc/symmetry-adapted-irrep-workflow-draft`

Context:

This task starts only after Task 1 and Task 2 pass Codex review.

Requirements:

* Use `P_a^sym` basis for trusted valley-preserving irrep matching.
* Keep q-cut seed basis as diagnostic.
* Preserve full HSP little-group data separately.
* Output valley mapping and valley sewing matrices.
* Keep readiness flags conservative.

Non-goals:

* no reduced EBR;
* no Chern number;
* no hidden tolerance relaxation.

Final handoff:

* branch name;
* commit hash;
* changed files;
* schema changes;
* readiness logic;
* tests and pytest result;
* open physics questions.
