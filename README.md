# ValleyScope

[中文说明](README.zh.md)

ValleyScope is a high-throughput post-processing workflow for **valley
projection, valley-projected irreps, and reduced EBR analysis in
two-dimensional moire materials**. It extracts selected VASP wavefunctions,
resolves their parent-layer momentum-valley content, identifies the symmetry
of each valley-projected subspace, and prepares or solves reduced-dimensional
EBR problems using reviewed Bilbao/irreptables conventions.

ValleyScope reports physical evidence and explicit blockers. It does not turn
high valley weight or a few symmetry eigenvalues into an automatic topology
label.

## Supported Scope

The primary supported and validated calculation domain is:

- VASP plane-wave wavefunctions, read from `WAVECAR` or a ValleyScope HDF5
  intermediate;
- nonmagnetic, spin-orbit-coupled two-dimensional moire systems with parent
  time-reversal symmetry (TRS) and the default VASP Cartesian spin frame
  `SAXIS=[0,0,1]`;
- selected moire high-symmetry points (HSPs), target bands, parent-layer
  valley centers, and a moire or bilayer structure for symmetry detection.

The workflow is general in lattice, space group, HSP set, valley orbit, and
valley-projected subspace space group. Real materials are validation fixtures,
not branches in the production method.

The calculation assumptions above are not fully recoverable from a
`WAVECAR` or compact HDF5 file alone. Users must ensure that the source
calculation belongs to this domain. Magnetic systems, spin-space-group
treatments, non-SOC noncollinear calculations, and arbitrary spin axes are not
currently claimed as validated production inputs.

## Physical Workflow

```text
VASP WAVECAR / ValleyScope HDF5
-> q-cut momentum-valley projection
-> valley mapping and valley-projected subspace symmetry
-> HSP little group and valley-preserving subgroup
-> symmetry representations restricted to the valley-preserving subgroup
-> valley-preserving irreps
-> reduced-dimensional EBR data in Bilbao/irreptables conventions
-> exact-integer reduced EBR decomposition
```

### Momentum-Valley Projection

For a moire Bloch state at momentum \(\mathbf k_M\), each plane-wave component
has momentum

```math
\mathbf q = \mathbf k_M + \mathbf G_M .
```

ValleyScope compares the in-plane component of \(\mathbf q\) with configured
parent-layer valley centers modulo the corresponding monolayer reciprocal
lattice. A q-cut window defines a seed projector \(P_a^0\) for valley \(a\).
For a normalized state,

```math
W_a = \langle\psi|P_a^0|\psi\rangle,
\qquad
W_{\rm val} = \sum_a W_a .
```

This is momentum-space parent-valley projection, not full monolayer Bloch-state
unfolding. `W_val` depends on the physical valley centers and q-cut, and is not
a topological invariant.

For near-degenerate target bands, individual VASP eigenvectors are
gauge-dependent. ValleyScope therefore analyzes the whole target subspace,
including its projected valley matrices and a valley-adapted basis. Detailed
subspace weights, assignments, and projector-quality evidence are retained by
the debug profile.

Two projector modes are available:

- `fixed_center` (default) uses fixed parent-layer valley centers. These seed
  projectors enter symmetry and irrep readiness.
- `k_resolved_parent_valley` uses dynamic centers for parent-valley weight
  reporting across the moire Brillouin zone. It does not replace the
  fixed-center projectors in irrep or EBR readiness.

### Valley-Preserving Symmetry

For an HSP \(k\), the HSP little group is

```math
G_k = \{g \mid gk = k + G_M\}.
```

If \(\pi_g(a)\) is the valley mapping induced by operation \(g\), the
valley-preserving subgroup for valley \(a\) is

```math
G_k^{(a)} = \{g \in G_k \mid \pi_g(a)=a\}.
```

The seed-projector covariance test is

```math
D_g P_a^0 D_g^\dagger \approx P_{\pi_g(a)}^0 .
```

An operation that maps \(a\) to another valley is a **valley-changing
operation**. It contributes valley-orbit and **valley sewing matrix** data; it
must not be forced into the single-valley irrep of \(G_k^{(a)}\).

ValleyScope restricts each symmetry representation to the actual
valley-preserving operation set and matches double-valued irreps for SOC
wavefunctions. Matching uses a validated standard-setting certificate that
relates the computed affine space-group operations and reciprocal coordinates
to the Bilbao/irreptables convention. Rotation matrices or user labels alone
do not establish this convention.

### Time Reversal

Parent TRS does **not** imply that a one-valley subspace is TR-invariant. In
general, time reversal relates distinct time-reversed valleys:

- a single-valley irrep is a unitary irrep of its valley-preserving subgroup;
- time-reversal completion is a separate inter-valley construction;
- a one-valley result is not matched to a grey group by default;
- joint grey-group data are used only when the required antiunitary and
  inter-valley evidence is present and validated.

Enable `analysis.time_reversal.enabled` only for input known to come from a
parent-TRS calculation. Missing or inconsistent sewing evidence remains an
explicit blocker.

### Reduced EBR Analysis

Raw three-dimensional EBR data are not a ValleyScope answer. ValleyScope uses
reviewed Bilbao/irreptables source conventions and reduces them to the same
physical basis used by the valley calculation:

```text
source EBR data
-> certified valley-projected subspace space group
-> sampled source-HSP basis
-> restriction to the valley-preserving subgroup
-> multiplicities in the matched valley-preserving irrep basis
-> reduced EBR matrix
```

The resulting integer vector is solved with the pure Python/SymPy exact
integer solver. Results distinguish an exact nonnegative EBR combination,
membership in the integer span without a nonnegative witness, exclusion from
the integer span, an indeterminate bounded search, and a blocked calculation.
None of these classifications alone is a Chern-number statement.

### Trust Criteria

High valley purity is useful but not sufficient for a trusted irrep or EBR
result. Promotion also requires the relevant evidence to pass, including:

- target-subspace closure and representation unitarity;
- seed-projector covariance under the full valley mapping;
- a well-defined valley-projected subspace and \(G_k^{(a)}\);
- complete, unambiguous operation and source-HSP mappings;
- a validated standard setting and reviewed irrep/EBR provenance;
- conservative spinful and, when used, antiunitary sewing evidence.

A clean fixed-center seed basis may be used directly. Otherwise ValleyScope
may construct a symmetry-adapted valley basis. Failed or incomplete evidence
stays diagnostic-only or blocked; tolerances should not be relaxed merely to
obtain a label.

## Installation

ValleyScope requires Python 3.10 or newer.

```bash
git clone https://github.com/gawcista/valley-proj.git
cd valley-proj
python -m pip install -e .
```

Check the installed commands:

```bash
valleyscope --help
valleyscope analyze-hsp --help
```

From a source checkout, `python -m valleyscope.cli --help` is equivalent.

## Quick Start

The normal workflow extracts a compact HDF5 file once, then analyzes that
file.

### 1. Extract Selected WAVECAR Data

Create `extract.yaml`:

```yaml
input:
  wavecar: ./WAVECAR

extract:
  kpoints:
    - name: GammaM
      vasp_index: 1
    - name: KM
      vasp_index: 2
  bands_vasp: [101, 102]

output:
  wavefunction_h5: ./wavefunctions.h5
```

`vasp_index` and `bands_vasp` are one-based VASP indices.

```bash
valleyscope extract-wavecar extract.yaml
```

### 2. Analyze the HSP Wavefunctions

Create `analyze.yaml`:

```yaml
input:
  wavefunction_h5: ./wavefunctions.h5
  monolayer_poscars:
    parent: ./monolayer.vasp

analysis:
  kpoints: [GammaM, KM]
  iband: [101, 102]
  time_reversal:
    enabled: true

valley_centers:
  coordinate_mode: layer_frac
  centers:
    - name: valley_a
      layer: parent
      frac: [0.333333333333, 0.333333333333, 0.0]
    - name: valley_b
      layer: parent
      frac: [-0.333333333333, -0.333333333333, 0.0]

valley_subspaces:
  - name: valley_a
    centers: [valley_a]
  - name: valley_b
    centers: [valley_b]

projection:
  projector_mode: fixed_center
  qcut_fraction: 0.20

symmetry:
  operations:
    structure_file: ./moire.vasp

output:
  directory: ./valley_analysis
  profile: standard
```

Replace the HSPs, bands, valley centers, structures, and q-cut with values
derived from the actual system. For multilayers, define each layer's
reciprocal frame and transform consistently; for commensurate structures,
integer supercell transforms are preferable to a twist angle alone.

```bash
valleyscope analyze-hsp analyze.yaml
```

### Example screen summary

The standard profile prints the same result-first text written to
`valley_summary.txt`. Its main sections are:

```text
Run and projection context
Valley projection by sampled state
Valley-projected subspace space group and trusted HSP irreps
Authoritative reduced EBR results
Readiness blockers and warnings
Public output files
```

The run context includes lines such as `qcut mode:` and the effective q-cut.
The standard `Valley projection summary` is the compact
`valley_projection_summary`; the same record also keeps canonical
`valley_resolved_irreps`, `reduced_ebr_summary`, and
`readiness_blocker_summary`. Common projection states include
`fixed_center_not_captured`, `not_derived`, and `unreliable`. The debug profile
retains `Valley subspace analysis`; `S_min` is the
minimum target-valley-subspace weight, alongside `min_concentration`,
`assigned_valleys`, and
`valley_weights_adapted`.

## Inputs and Configuration

An analysis needs:

- a ValleyScope HDF5 file containing selected coefficients, reciprocal
  vectors, k points, energies, and VASP band indices;
- monolayer reciprocal-lattice information and physically defined valley
  centers, including layer transforms when needed;
- target HSP labels and `analysis.iband` values that exist in the HDF5 file;
- a moire or bilayer POSCAR/CONTCAR at
  `symmetry.operations.structure_file` for spglib operation detection.

The monolayer structure defines parent-layer reciprocal coordinates; the
moire or bilayer structure defines the symmetry operations. They are not
interchangeable.

Use `output.profile: standard` for routine runs and `output.profile: debug`
when detailed projector, representation, closure, HSP-star, or sewing evidence
is needed. Advanced source-table and standard-setting overrides are
intentionally omitted here: use the current CLI help and parser in
`valleyscope/io/config.py` when preparing reviewed inputs.

## Outputs

The standard profile emphasizes these public surfaces:

| File | Purpose |
| --- | --- |
| `valley_summary.txt` | Result-first human-readable summary; read this first |
| `valley_summary.json` | Compact machine-readable record with projection, canonical irrep, authoritative reduced-EBR, and blocker summaries |
| `valley_weights.csv` | Quick scan of raw per-(kpoint, VASP band) valley weights |
| `valley_ebr_export_bundle.json` | Written only when at least one ready bundle exists |
| `valley_reduced_ebr_mapping.json` | Written only when reduced EBR mapping is enabled and evaluated |

`valley_resolved_irreps` contains one canonical record per sampled
`(kpoint, valley)`, including the valley-projected subspace space group, HSP
little group, valley-preserving operation IDs, readiness, blockers, and
`irrep_multiplicities`.

Raw rows in `valley_weights.csv` are useful for screening, but individual rows
inside a near-degenerate band subspace are gauge-dependent. Interpret them
together with the subspace and readiness information in the summary.

The debug profile retains detailed JSON/HDF5 evidence such as
`diagnostics.h5`, symmetry reports, restricted representation data, and
irrep/EBR provenance. These are debugging surfaces, not files that every user
must inspect.
Database ingestion reads complete EBR bundle and mapping evidence from the
standalone public files; the compact summary does not duplicate those payloads.
Some debug tables may be header-only when no row satisfies their physical
scope; that alone is not a run failure.

## Running Reduced EBR Mapping

`analysis.reduced_ebr.enabled` is off by default. For ready canonical bundles,
the analyzer can build reviewed reduced tables from the installed
`irreptables` source, or consume a user-supplied validated reduced table or
reviewed mapping specification. All paths must pass the same group, setting,
spin, HSP, irrep, and provenance checks before the exact solver runs.

Authoritative evaluation runs inside `analyze-hsp`, where the producer-owned
representation evidence is available for recomputation. An exported JSON
bundle retains identity links for audit and downstream transport, but those
links alone cannot re-establish numerical trust. Therefore the standalone
command is a fail-closed compatibility audit when given only serialized JSON;
it does not promote an identity-only bundle to an authoritative solution:

```bash
valleyscope map-reduced-ebr \
  valley_ebr_export_bundle.json \
  validated_reduced_ebr_table.json \
  --output valley_reduced_ebr_mapping.json
```

ValleyScope does not ship ad hoc or unreviewed production EBR tables.

## Limits and Non-Goals

ValleyScope currently does not provide:

- raw three-dimensional EBR decomposition as a valley-resolved result;
- built-in unreviewed EBR tables or heuristic floating-point EBR fitting;
- compatibility relations;
- Berry curvature, Wilson loops, or Chern numbers;
- automatic full-moire-Brillouin-zone valley-goodness validation;
- a topology conclusion from HSP data alone.

## Development

Install the test dependency and run the suite:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The public command surface is defined in `valleyscope/cli.py`, configuration
parsing in `valleyscope/io/config.py`, and public output selection in
`valleyscope/reports/analysis_outputs.py`.
