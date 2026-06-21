# tMoTe2 P321 Generic P3 Irrep Validation

Date: 2026-06-21 | Branch: `cc/tmote2-p321-generic-p3-validation`

## Scope

Validate the generic Bilbao/irreptables valley-preserving irrep path on the
real tMoTe2 P321 fixture. tMoTe2 is a validation fixture only; no
material-specific production logic is used.

The physical point is that the full moire space group is P321, but a
single-valley projected subspace is not a representation of the full P321
little group. The relevant one-valley symmetry is the valley-preserving
subspace space group. For the trusted K/K' rows in this fixture, this is P3:
identity plus the C3 rotations. The valley-changing C2 operations are sewing
data between valleys, not single-valley irrep operations.

## Correct Generic Validation Command

```bash
python -m valleyscope.cli analyze-hsp /tmp/tmp.8G9ssDjAY6/tmote2_generic_p3_sg143.yaml
```

Config delta from `real_tests/tMoTe2/analyze.yaml`:

```yaml
analysis:
  generic_irrep_source:
    enabled: true
    spacegroup_number: 143      # P3 subspace space group, not full P321
    spinor: true
    source_hsp_labels:
      GammaM: {K_valley: GM, Kp_valley: GM}
      KM:     {K_valley: K,  Kp_valley: K}
      MM:     {K_valley: M,  Kp_valley: M}
```

All other config values were inherited from `real_tests/tMoTe2/analyze.yaml`
with absolute input paths and a temporary output directory.

## Result

The generic P3 source path succeeds for the trusted direct-qcut rows at
GammaM and KM:

```text
mode generic status ok
GammaM K_valley  matched P3 C3_like {'-GM4': 1} {'0': 1, '1': 2, '2': 3}
GammaM Kp_valley matched P3 C3_like {'-GM4': 1} {'0': 1, '1': 2, '2': 3}
KM     K_valley  matched P3 C3_like {'-K6': 1}  {'0': 1, '1': 2, '2': 3}
KM     Kp_valley matched P3 C3_like {'-K5': 1}  {'0': 1, '1': 2, '2': 3}
MM     K_valley  blocked None None {} {} missing_source_irrep_characters
MM     Kp_valley blocked None None {} {} missing_source_irrep_characters
candidates 4
instances 2
bundles 2
```

Interpretation:

- `subspace_group_candidate` is the physical P3 symbol.
- `legacy_subspace_group_candidate` remains `C3_like` as provenance only.
- The four trusted GammaM/KM rows are matched by the generic
  `bilbao_restricted_character` path.
- The pipeline produces 4 EBR input candidates, 2 problem instances, and
  2 export bundles.
- No legacy C3 phase-table fallback is needed for these trusted rows.

This validates the intended direction: use Bilbao/irreptables conventions for
the valley-projected subspace space group, not the full parent moire space
group when valley-changing operations are outside the one-valley irrep.

## Negative Control: Full P321 Source Table

Using `spacegroup_number: 150` (P321) is the wrong source table for the
single-valley projected subspace. It tests a different mathematical object:
full P321 HSP irreps restricted to the C3 valley-preserving subgroup.

That negative-control run blocks at GammaM and KM because pairs such as
`-GM4` / `-GM5` and `-K4` / `-K5` become indistinguishable after restriction
to the C3 operations. This ambiguity is real for the full P321-to-C3
restriction, but it is not a blocker for the correct P3 subspace-irrep
workflow.

## MM Rows

MM is correctly identity-only in `G_k^(a)` after the
`cc/generic-source-subspace-little-group` fix:

- `hsp_preserving_operation_ids` at MM is `[0]` (only identity).
- `valley_preserving_operation_ids` from the subspace space group is `[0, 1, 2]`.
- `G_k^(a)` = `[0] ∩ [0, 1, 2]` = `[0]` (identity-only).
- Generic source preflight now uses this correct per-HSP set, not the
  full subspace space-group VP list.

With identity-only `G_k^(a)`, the matcher uses only the identity character
and does not request C3 characters that are absent from the source M HSP.
MM matches succeed with the trivial identity irrep when source data is
available for that HSP-label mapping.

## Physical Conclusion

The tMoTe2 fixture supports the user's physical interpretation:

- full moire symmetry: P321;
- one-valley projected subspace symmetry for trusted GammaM/KM rows: P3;
- valley-changing C2 operations are sewing operations, not one-valley irrep
  generators;
- generic P3 restricted-character matching works for the trusted rows when
  the source table is the P3 subspace space group.

The reported P321 restricted-character ambiguity is useful as a negative
control, but it should not be treated as a physical blocker of the
valley-projected P3 workflow.

## Resolution

The operation-set fix was implemented in `cc/generic-source-subspace-little-group`
(commit range `main..cc/generic-source-subspace-little-group`). The generic
source preflight now computes `G_k^(a)` as the intersection of HSP
little-group operations and valley-preserving operations, rather than using
the full subspace-space-group VP list at every HSP. This is group-agnostic
and uses no tMoTe2-specific production logic.
