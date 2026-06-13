# ValleyScope C3-Like Reduced EBR Mapping Signoff Packet

Date: 2026-06-13 | Status: Signoff packet (no table data shipped)

## Scope

This document records the reviewed physical conventions and mapping
decisions for a C3-like reduced EBR table authoring workflow using SG
150 (P321) spinful Bilbao-derived irrep data.  It is a **signoff
packet**, not a reduced EBR table, not a decomposition report, and not
`analyze-hsp` workflow wiring.

A future reviewed mapping spec or reduced EBR table may be built from
this packet after the checklist in section 9 is completed and signed.

### Non-Goals

- This packet does not ship any reduced EBR table JSON.
- It does not wire into `analyze-hsp` or `valleyscope/data/reduced_ebr/`.
- It does not report reduced EBR decomposition.
- It does not use OR-Tools, `irrep.ebrs`, or `irrep2`.
- It does not include real material names or material-specific logic.

## 1. Accepted Source Data

| Field | Value |
|-------|-------|
| Data source | `irreptables.irreps.IrrepTable("150", True)` |
| Data origin | Bilbao Crystallographic Server (via irreptables) |
| Space group | 150 (P321) |
| Spinor type | Spinful (double group, SOC) |
| Number of symmetry operations | 6 (identity, 2×C3, 3×C2) |
| Number of irreps | 16 (covering all 3D BZ HSPs) |
| Character encoding | Complex: `ch = |ch| * exp(iπ * phase/π)` per operation |
| Operation indexing | 1-indexed Bilbao convention |

Machine-verified by `tests/test_irreptables_table_builder.py`.

## 2. Accepted HSP Set

| HSP | Role | Valley-Preserving Subgroup Order | Source Irrep Labels |
|-----|------|--------------------------------|--------------------|
| GammaM | Required sampled HSP | 3 ({E, C3, C3²}) | `-GM4`, `-GM5`, `-GM6` |
| KM | Required sampled HSP | 3 ({E, C3, C3²}) | `-K4`, `-K5`, `-K6` |
| MM | Not in C3 reduced basis | 1 (identity only) | — |
| KA | Not in sampled HSP set | — | `-KA4`, `-KA5`, `-KA6` (out of scope) |

## 3. Accepted Valley-Preserving Subgroup

For a single valley at GammaM or KM under P321:

```text
G_k^{(a)} = {g ∈ G_k | π_g(a) = a} = {E, C3, C3²} ≅ C3
```

- C3 operations preserve valley labels: π_{C3}(K)=K, π_{C3}(K')=K'
- C2 operations (op4, op5, op6) map K↔K' and are valley-changing
- Only valley-preserving operations contribute to the single-valley
  reduced EBR irrep basis

## 4. Central-Sign Convention

The Bilbao character table uses independent SU(2) spinor lifts for
each spatial operation.  The C3 generator (op2) and the C3² spatial
operation (op3) satisfy:

```text
S3 = -S2²    (verified numerically)
op3 = -op2²   (double-group relation)
```

Therefore:

```text
chi(C3)  = chi(op2)           ← read directly from Bilbao table
chi(C3²) = -chi(op3)          ← double-group square with corrected sign
```

This convention is consistent for both 1D and 2D irreps:

| Verification | chi(op2) | chi(op3) Bilbao | chi(C3) | chi(C3²) = -chi(op3) | chi(C3)² |
|-------------|---------|----------------|---------|----------------------|----------|
| 1D (`-GM5`) | -1 | -1 | -1 | +1 | +1 |
| 2D (`-GM6`) | +1 | +1 | +1 | -1 | (N/A — 2D) |

Machine-verified by `tests/test_irreptables_table_builder.py` (double-group
lift convention tests).

## 5. Source-Label Mapping Table

All six in-scope source labels are included in the first C3 reduced
basis.  ValleyScope irrep keys follow the `{HSP}:{phase_key}` format
with the phase convention documented in
`valleyscope/data/valley_irreps/`.

| Source Label | Deg | HSP | Bilbao chi(C3) | C3 Phase (2π) | ValleyScope Key |
|-------------|-----|-----|---------------|--------------|-----------------|
| `-GM4` | 1 | GammaM | -1 (op2) | +1/2 | `GammaM:C3_spinor_phase_+1/2` |
| `-GM5` | 1 | GammaM | -1 (op2) | +1/2 | `GammaM:C3_spinor_phase_+1/2` |
| `-GM6` | 2 | GammaM | +1 (op2) | — | multiplicity `{+1/6: 1, -1/6: 1}` at GammaM |
| `-K4` | 1 | KM | -1 (op2) | +1/2 | `KM:C3_spinor_phase_+1/2` |
| `-K5` | 1 | KM | -1 (op2) | +1/2 | `KM:C3_spinor_phase_+1/2` |
| `-K6` | 2 | KM | +1 (op2) | — | multiplicity `{+1/6: 1, -1/6: 1}` at KM |

For degenerate labels (`-GM6`, `-K6`), the multiplicity encodes the
valley-preserving C3 subgroup restriction: each contributes one state
with phase +1/6 and one state with phase -1/6.

The `-4`/`-5` distinction at each HSP is in C2 characters (valley
sewing data) and does not affect the C3 valley-preserving phase
assignment.

## 6. C2/op4-6 Valley Sewing Data Exclusion

Operations op4, op5, op6 are C2 rotations (order 2, spatial rotation
trace = -1).  In the P321 K/K' valley system, these operations are
**valley-changing**:

```text
π_{C2}(K) = K',   π_{C2}(K') = K
```

Their characters are valley sewing data.  They must NOT enter the
single-valley C3 reduced EBR vector basis.  A future full valley-orbit
representation may use C2 characters for sewing-matrix construction,
but that is outside this signoff packet's scope.

## 7. Provenance Fields For Future Mapping Spec / Table

When a reviewed mapping spec or reduced EBR table is built from this
packet, the following provenance fields must carry non-empty,
reviewer-verified values.  They include both fields currently enforced
by the package-data catalog gate and additional reviewer-required fields
needed to keep the C3 reduction physically auditable.

| Field | Required Value | Example |
|-------|---------------|---------|
| `data_source` | `"irreptables"` | — |
| `space_group_number` | `150` (int or string `"150"`) | — |
| `spinful` | `true` | — |
| `subspace_group_candidate` | `"C3_like"` | — |
| `expected_hsps` | `["GammaM", "KM"]` | — |
| `valleyscope_reduction` | `"sampled_hsp_valley_preserving"` | — |
| `review_status` | `"reviewed"` | — |
| `reviewer` | (reviewer initials) | `"JD"` |
| `review_date` | (ISO date) | `"2026-06-13"` |
| `review_method` | how evidence was obtained | `"Bilbao-derived irreptables.irreps.IrrepTable character table"` |
| `source_reference` | external reference for character data | `"Bilbao Crystallographic Server, SG 150 spinful irreps (via irreptables)"` |
| `central_sign_convention` | `"chi(C3)=chi(op2), chi(C3^2)=-chi(op3)"` | — |

Catalog-enforced provenance fields in
`valleyscope.data.reduced_ebr.catalog` are `review_status`, `reviewer`,
`review_date`, `review_method`, `source_reference`, and
`valleyscope_reduction` for the reviewed package-data table path.

Reviewer-required signoff fields in this packet additionally include
`data_source`, `space_group_number`, `spinful`, `subspace_group_candidate`,
`expected_hsps`, and `central_sign_convention`.  A future mapping spec should
carry these fields even if the current catalog gate does not enforce each one
individually.

## 8. Statement Of Non-Delivery

This packet:

- Is NOT a reduced EBR table.
- Is NOT a reduced EBR decomposition report.
- Is NOT `analyze-hsp` workflow wiring.
- Is NOT a replacement for the C3 authoring audit
  (`docs/reduced_ebr_c3_authoring_audit.md`), which remains the
  authoritative audit document.
- Does NOT ship any JSON under `valleyscope/data/reduced_ebr/`.

No reviewed reduced EBR table may be claimed or shipped until the
checklist in section 9 is completed by a qualified reviewer and the
provenance fields in section 7 are populated.

## 9. Signoff Checklist

Before a C3-like reduced EBR table can be built and shipped as reviewed
ValleyScope package data, the following items must be completed:

- [ ] **Source data accepted.** `irreptables.irreps.IrrepTable("150", True)` as Bilbao-derived SG150 spinful character source.
- [ ] **HSP set confirmed.** `{GammaM, KM}` for C3 reduced basis; MM excluded; KA out of scope.
- [ ] **Valley-preserving subgroup confirmed.** `{E, C3, C3²}` ≅ C3.
- [ ] **Central-sign convention confirmed.** `chi(C3)=chi(op2)`, `chi(C3²)=-chi(op3)`.
- [ ] **Six source labels mapped.** Table in section 5 verified.
- [ ] **C2 valley sewing data excluded.** op4/5/6 C2 characters not in C3 reduced basis.
- [ ] **Provenance fields populated.** All fields in section 7 filled with non-empty, verified values.
- [ ] **Reviewer signoff.**
  Reviewer: _______________ Date: _______________

This checklist records the agreed mapping decisions.  Signing it does
not generate a table — it authorizes a future `valleyscope
build-reduced-ebr-table` run to produce a reviewed table from a mapping
spec that conforms to this packet.
