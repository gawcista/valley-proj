# AGENTS.md

This file contains long-lived project rules for agents working on ValleyScope.
Stage-specific plans, open tasks, and temporary checklists belong in `PLAN.md`.

## Project Goal

ValleyScope is a VASP post-processing workflow for high-throughput extraction
of valley-resolved irreps in moire materials. The long-term target is
valley-resolved reduced EBR decomposition for a moire database.

The current physical layering is:

```text
VASP WAVECAR / HDF5
-> q-cut valley seed projectors P_a^0
-> seed projector diagnostics
-> moire HSP symmetry representations D_g
-> projector symmetry-consistency checks
-> future symmetry-adapted valley projectors P_a^sym
-> valley-preserving irreps plus valley sewing data
-> future valley-resolved reduced EBR decomposition
```

The q-cut valley seed projector is a momentum-valley seed and diagnostic. It
does not automatically define a trusted valley irrep basis.

## Roles

Codex acts as project architect, methodology reviewer, and task planner. Codex
may edit small fixes, documentation, schema wiring, tests, and planning files,
but should not take over large implementation work unless the user explicitly
asks for it. New implementation tasks should be handed to cc with clear branch
names, files to inspect, formulas, tests, non-goals, and handoff requirements.

cc is the main implementation agent. cc should work on an independent branch,
self-test, self-review, and provide a Codex review handoff. cc must not merge
to `main` unless the user explicitly authorizes it.

All agents must prioritize physical consistency over speed. If a convention is
unclear after reading the repo and refs, ask the user in Chinese instead of
inventing a rule.

## Communication And Code Style

* Natural-language communication with the user must be in Chinese.
* Code, variable names, filenames, CLI options, and necessary comments use
  English.
* Comments should be short and useful.
* User-facing terminology should use `valley`, not `sector`.
* Legacy/internal names such as `sector_mapping` may remain for compatibility,
  but new public schema, summaries, and docs should use valley terminology.
* Use `HSP little group`, `valley mapping`, `valley-preserving subgroup`,
  `valley-preserving operation`, `valley-changing operation`, and
  `valley sewing matrix` in public-facing text.

## Physical Methodology

The projector symmetry-consistency condition is

```math
D_g P_a D_g^\dagger \approx P_{\pi_g(a)} .
```

Here $D_g$ is the moire HSP symmetry representation in the target DFT band
subspace, and $\pi_g(a)$ is the operation-induced mapping of valley labels.
This is not an invariance condition. For valley-changing operations, do not
require $D_g P_a D_g^\dagger \approx P_a$.

For q-cut seed projectors, the diagnostic is

```math
\epsilon_{\rm seed}(g,a)
=
\frac{\|D_g P_a^0 D_g^\dagger - P_{\pi_g(a)}^0\|_F}
{\max(\|P_a^0\|_F, {\rm small})}.
```

`epsilon_seed` is the seed projector symmetry error.

The HSP little group is

```math
G_k = \{g \in G \mid gk = k + G_M\}.
```

For a valley label $a$, the valley-preserving subgroup inside the HSP little
group is

```math
G_k^{(a)} = \{g \in G_k \mid \pi_g(a)=a\}.
```

Valley-preserving irreps are not full-group irreps. Any final database output
must distinguish full-group irrep, HSP little group, valley orbit, valley
mapping, valley-preserving subgroup, valley-preserving irrep, valley-changing
operation, and valley sewing matrix.

## Current Stage Boundaries

Current work may include:

* HDF5 intermediate input for VASP HSP wavefunctions.
* q-cut valley seed projector construction.
* Valley weight, qcut, and target-subspace diagnostics.
* Moire HSP symmetry representations.
* Valley orbit and valley mapping reports.
* Seed projector symmetry-consistency diagnostics.
* Diagnostic-only q-cut-basis symmetry eigenvalues.
* Prototype irrep matching when all readiness checks pass.

Current work must not implement:

* symmetry-adapted valley projectors `P_a^sym` unless assigned as a reviewed
  task;
* reduced EBR decomposition;
* Berry curvature;
* Wilson loop;
* Chern number;
* full-mBZ valley-goodness validation;
* unreviewed character-table, subgroup, or compatibility-relation logic.

## Hard Rules

* Do not loosen tolerances to hide O(1) block leakage or failed
  symmetry-consistency checks.
* Do not infer irrep readiness from high valley purity alone.
* Do not treat `[D_g, L] != 0` as a failure for valley-changing operations.
  The correct condition is projector symmetry-consistency under $\pi_g$.
* Do not use moire reciprocal lattice vectors to redefine monolayer valleys.
* Do not report q-cut seed-basis irrep labels as trusted when the seed projector
  symmetry-consistency check fails.
* Do not make the two-valley special case the main framework.
* Do not treat `ecut_adjust_tol` as the physical VASP `ENCUT`.
* Do not restore removed public fields such as `valley_sectors` or
  `target_bands_vasp`.
* Do not add large WAVECAR files or large real-material outputs to the repo.

## Readiness And Output Rules

If seed projector symmetry-consistency fails for a relevant
`(kpoint, operation, valley)` row, any q-cut seed-basis valley-preserving irrep
or symmetry eigenvalue label for that row must be marked diagnostic-only. The
row should expose clear readiness information such as:

```text
diagnostic_only = true
topology_input_ready = false
local_irrep_ready = false
reason includes "seed projector symmetry-consistency failed"
```

Public schema should prefer:

* `projector_symmetry`
* `seed_projector_symmetry`
* `seed_projector_symmetry_error`
* `seed_projector_symmetry_status`
* `hsp_little_group`
* `valley_mapping`
* `valley_preserving_subgroup`
* `valley_preserving_operations`
* `valley_changing_operations`
* `valley_preserving_irrep`
* `valley_sewing_matrix`

Avoid old projector or subgroup terms as public field names, summary titles, or
new function names.

## Review Checklist

When reviewing cc output, Codex must check:

* q-cut seed and symmetry-adapted projector roles are clearly separated;
* valley-changing operations use
  `D_g P_a D_g^\dagger ≈ P_{pi_g(a)}`;
* seed projector symmetry error is normalized sensibly;
* rank selection diagnostics are present whenever a purified projector is
  introduced;
* orthogonality, completeness, unitarity, and target-subspace closure are
  tested where relevant;
* spinor convention readiness remains conservative;
* public schema uses the current terminology;
* old schema is only retained as an explicit legacy alias when necessary;
* toy tests do not depend on large WAVECAR files;
* `pytest -q` or the relevant targeted tests are reported.

