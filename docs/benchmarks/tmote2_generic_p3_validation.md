# tMoTe2 P321 Generic P3 Irrep Validation — Blocked By Restricted-Character Ambiguity

Date: 2026-06-21 | Branch: `cc/tmote2-p321-generic-p3-validation`

## Scope

Validate the generic Bilbao/irreptables P3 valley-preserving irrep path on the
real tMoTe2 P321 fixture. tMoTe2 is a validation fixture only — no
material-specific production logic.

## Command

```bash
python -m valleyscope.cli analyze-hsp /tmp/tmote2_generic_p3.yaml
```

Config delta from `real_tests/tMoTe2/analyze.yaml`:

```yaml
analysis:
  generic_irrep_source:
    enabled: true
    spacegroup_number: 150      # P321
    spinor: true
    source_hsp_labels:
      GammaM: {K_valley: GM, Kp_valley: GM}
      KM:     {K_valley: K,  Kp_valley: K}
      MM:     {K_valley: M,  Kp_valley: M}
```

All other config (qcut, spinor convention, tolerances, iband, valley centers,
valley subspaces, symprec) identical to the production `analyze.yaml`.

## Result: Generic Path Blocked

All generic restricted-character matches are blocked. Zero EBR input candidates
are produced ("no candidates"), zero EBR problem instances ("no_instances"),
zero export bundles ("no_bundles").

## Blocker Analysis

### GammaM and KM HSPs: Ambiguous Restricted Source Irreps

P321 spinful irreps at GM (source HSP `GM`) and K (source HSP `K`) have three
irreps each:

| Source Irrep | Dim | Characters (ops 1,2,3) |
|-------------|-----|------------------------|
| -GM4 / -K4  | 1   | {1: 1, 2: -1, 3: -1}  |
| -GM5 / -K5  | 1   | {1: 1, 2: -1, 3: -1}  |
| -GM6 / -K6  | 2   | {1: 2, 2:  1, 3:  1}  |

When restricted to the valley-preserving C3 subgroup `G_k^(a) = {E, C3, C3²}`
(ValleyScope ops [0, 1, 2] → source ops [1, 2, 3]), the 1D irreps -GM4/-GM5
(and -K4/-K5) have **identical** restricted characters: both give {1: 1, 2: -1,
3: -1} on the valley-preserving operation set.

The generic restricted-character matcher detects this and reports:

```
ambiguous_restricted_source_irreps: source irreps are not distinguishable
on the restricted operation set: {((1.0, 0.0), (-1.0, 0.0), (-1.0, 0.0)):
['-GM4', '-GM5']}
```

This is a **genuine physical ambiguity**: the valley-preserving subgroup is a
proper subgroup of the HSP little group, and restricting full-GM/full-K source
irreps to the valley-preserving ops makes some source irreps degenerate.

The legacy C3 phase-table matching sidesteps this by operating on a single C3
generator with eigenphase, not on the full restricted character profile. But
that approach is less physically complete — it ignores the full `G_k^(a)`
character data.

### MM HSP: Missing Source Irrep Characters

At source HSP `M`, the P321 table has only two irreps (-M3, -M4) with
characters defined only on ops {1, 6}. The ValleyScope VP ops [0, 1, 2] map
to source ops [1, 2, 3] (identity + C3 + C3²), but ops 2 and 3 are NOT in
the M HSP little group. The adapter reports:

```
missing_source_irrep_characters: source irreps are missing mapped operation
characters: {'-M3': [2, 3], '-M4': [2, 3]}
```

This is consistent with the legacy benchmark finding: MM has only identity as
valley-preserving in the HSP little group. C3 ops map MM to other HSP star
members, so the `G_k^(a)` at MM is just {identity}.

## Production Code Status

- `subspace_space_group.candidate_space_group_symbol`: `"P3"` for all trusted
  rows — correct.
- `legacy_subspace_group_candidate`: `"C3_like"` for all rows — correct
  provenance.
- `matching_mode`: `"generic"` — correct.
- `matching_strategy`: `"bilbao_restricted_character"` — correct.
- `irrep_matching.matching_status`: `"blocked"` — correct, not a silent
  legacy fallback.
- No tMoTe2-specific production logic was added. The blocker is in the
  physical data-flow: restricted-character irreps of a proper subgroup.

## Physical Conclusion

The generic P3 restricted-character path **does not silently fall back** to
legacy C3 phase-table matching. It correctly reports blocked/diagnostic rows
with explicit reasons. The blocker is a physical one: restricting full HSP
little-group source irreps to a proper valley-preserving subgroup produces
character degeneracies that the generic matcher correctly identifies.

Possible resolutions (NOT for this task):

1. Augment the generic matcher with compatibility relations or additional
   physical constraints (e.g., selection rules from the full HSP little group)
   to break degeneracies.
2. Use the full HSP little-group characters for matching first, then project
   to the valley-preserving subgroup.
3. Accept ambiguity at the full-character level and use a minimal-eigenvalue
   heuristic (like the legacy phase table) only for the cyclic generator,
   with explicit provenance that this is a generator-only match.

## tMoTe2 As A Fixture

tMoTe2 is a validation fixture only. The blocker above is a generic physical
problem that any material with a proper valley-preserving subgroup will face.
This benchmark validates that the generic path is physically honest about the
ambiguity rather than producing a trusted match from incomplete data.
