# ValleyScope

[中文说明](README.zh.md)

ValleyScope analyzes the valley character and symmetry representations of
VASP wavefunctions in two-dimensional moiré materials. It resolves the
contribution of parent-layer valleys to a selected band subspace, determines
the symmetries that preserve each valley, and assigns irreducible
representations (irreps) at moiré high-symmetry points (HSPs).
When a valley-preserving rotation is present, it also reports a conditional
constraint on the subspace Chern number modulo the rotation order.

These irreps can then be compared with elementary band representations
(EBRs), reduced to the same valley subspace and HSP basis. The result describes
the symmetry content of the selected states. It is not, by itself, a
classification of their topology over the full moiré Brillouin zone.

## Supported Scope

The present implementation is intended for:

- VASP plane-wave wavefunctions, read from `WAVECAR` or a ValleyScope HDF5
  intermediate;
- nonmagnetic, spin-orbit-coupled two-dimensional moiré systems with parent
  time-reversal symmetry (TRS) and the default VASP Cartesian spin frame
  `SAXIS=[0,0,1]`;
- selected moiré HSPs and target bands, with specified parent-layer valley
  centers and a moiré or bilayer structure for determining spatial symmetries.

The construction uses the lattice and symmetry operations of the supplied
system, without material-specific rules or a prescribed number of valleys.
Numerical tests cover particular cases, not every space group or band manifold.

The calculation assumptions above are not fully recoverable from a
`WAVECAR` or compact HDF5 file alone. Users must ensure that the source
calculation belongs to this domain. Magnetic systems, spin-space-group
treatments, non-SOC noncollinear calculations, and arbitrary spin axes are not
within the validated scope.

## Physical Method

```text
Plane-wave coefficients
  → parent-valley projection
  → symmetry of each valley-projected subspace
  → HSP little-group representations
  → comparison with reduced EBRs
```

### Momentum-Valley Projection

For a moiré Bloch state at momentum \(\mathbf k_M\), each plane-wave component
has momentum

```math
\mathbf q = \mathbf k_M + \mathbf G_M .
```

ValleyScope compares the in-plane component of \(\mathbf q\) with configured
parent-layer valley centers modulo the corresponding monolayer reciprocal
lattice. The distance is the shortest Cartesian distance over reciprocal
lattice translations, including for nonorthogonal lattices. A q-cut window
defines a seed projector \(P_a^0\) for valley \(a\).
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
subspace weights, valley assignments, and deviations from projector
orthogonality and covariance are available with `output.profile: debug`.

Two projector modes are available:

- `fixed_center` (default) uses fixed parent-layer valley centers. These seed
  projectors are used to test subspace symmetry and assign irreps.
- `k_resolved_parent_valley` uses dynamic centers for parent-valley weight
  reporting at the sampled moiré momenta. Symmetry and EBR analysis still
  starts from fixed-center seed projectors. This reporting option does not
  validate valley character throughout the Brillouin zone.

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

An operation that maps \(a\) to another valley is a valley-changing operation.
Its action relates the two subspaces through a valley sewing matrix; it is
not part of the single-valley representation of \(G_k^{(a)}\). Where this
unitary relation has been verified from the wavefunctions, it can determine
the representation at a symmetry-related, unsampled point. The transported
subspace is checked against the target little-group irreps, with its origin
recorded separately from directly sampled states.

ValleyScope matches the representation on the full valley-preserving subgroup,
using double-valued irreps for SOC wavefunctions. This includes multidimensional
irreps, not only eigenvalues of a rotation generator. Comparison with
Bilbao/irreptables requires a consistent crystallographic setting: basis,
origin, affine operations, and the complete translation lattice. Matching
rotation matrices or space-group names alone is insufficient.

### Time Reversal

Parent TRS does not imply that a one-valley subspace is TR-invariant. When
time reversal exchanges valleys, each valley is first described by the
unitary irreps of its own valley-preserving subgroup.

For a parent-TRS calculation, an unsampled time-reversed irrep may be inferred
from a verified source irrep and the reviewed time-reversal pairing in the
source tables. This algebraic inference is distinct from both unitary valley
sewing and a numerical test of antiunitary sewing between wavefunctions.
Directly sampled and inferred irreps remain distinguishable in the results.

A joint description including time reversal (a grey-group corepresentation)
requires additional antiunitary and inter-valley checks. Failure of those
checks does not by itself invalidate a separately verified single-valley
unitary representation.

Enable `analysis.time_reversal.enabled` only for input known to come from a
parent-TRS calculation. A symmetry relation is left unresolved when the
information needed to establish it is missing or inconsistent.

### Conditional Valley Chern Residues

Rotation eigenvalues can constrain a Chern number modulo \(n\), as derived by
[Fang, Gilbert and Bernevig, Phys. Rev. B 86, 115112 (2012)](https://doi.org/10.1103/PhysRevB.86.115112).
ValleyScope automatically selects the highest-order actual valley-preserving
rotation about \(+z\) among \(n=2,3,4,6\), and combines the determinants of
the subspace rotation matrices at the required invariant momenta. The result
is for the **whole selected valley subspace**, not its individual bands.
In-plane twofold axes, valley-changing rotations and screw operations do not
supply this \(C_{nz}\) constraint.
Spinful rotation phases and any validated time-reversal inference are included;
inferred evidence is identified separately from sampled evidence.

A result such as `C = 1 (mod 3)` is conditional on the existence of a smooth,
constant-rank valley subspace throughout the mBZ, with consistent Bloch boundary
identifications and the stated rotation symmetry. The HSP calculation does not
establish these global conditions, and cannot distinguish integers that differ
by \(n\). No full integer Chern number is assigned.

This analysis runs independently of the reduced EBR option and introduces no
required inputs. Without a valley-preserving rotation it is `not_applicable`;
missing required HSPs or unverified representation evidence give `blocked`,
with reasons and no numerical residue. Available numerical results are always
`conditional`, while `global_valley_subspace_status` remains `not_evaluated`.

### Reduced EBR Analysis

EBRs of the parent three-dimensional space group cannot be compared directly
with single-valley irreps. ValleyScope uses reviewed Bilbao/irreptables data
for the valley-projected subspace space group and restricts them to the same
HSP and valley-preserving irrep basis as the wavefunction calculation:

```text
source EBR data
-> valley-projected subspace space group in a verified standard setting
-> sampled source-HSP basis
-> restriction to the valley-preserving subgroup
-> multiplicities in the matched valley-preserving irrep basis
-> reduced EBR matrix
```

The irrep multiplicities form an integer vector, and the reduced EBR vectors
form the columns of an integer matrix. ValleyScope solves for their
coefficients using exact arithmetic in Python/SymPy. It distinguishes a
nonnegative integer EBR combination, integer-span membership without a
nonnegative combination, exclusion from the integer span, an inconclusive
bounded search, and insufficient physical information.

A nonnegative combination establishes compatibility with the reduced EBRs
at the HSPs considered. It does not establish a globally equivalent set of
localized Wannier functions. Conversely, exclusion from the integer span is
relative to the chosen subspace, HSP basis, and reviewed EBR generators;
it is not a Chern-number calculation.

### Conditions for Assigning Irreps

High valley purity does not ensure that a band subspace carries a symmetry
representation. The calculation also checks:

- target-subspace closure and representation unitarity;
- seed-projector covariance under the full valley mapping;
- a well-defined valley-projected subspace and \(G_k^{(a)}\);
- complete mappings of plane waves, symmetry operations, and source HSPs;
- agreement with the crystallographic setting and irrep/EBR source tables;
- the double-group multiplication law and, when used, antiunitary sewing.

A fixed-center seed basis can be used directly if it satisfies these conditions.
Otherwise, ValleyScope attempts a symmetry-adapted valley basis. If neither
construction satisfies the conditions, the affected irrep or EBR result is
not assigned. The residuals and reasons remain available for inspection;
changing the output profile does not change these requirements.

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

Extract the selected wavefunctions into a compact HDF5 file once, then use
that file for subsequent analyses.

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

The terminal and `valley_summary.txt` show the same summary:

```text
Run and projection context
Valley projection by sampled state
Valley-projected subspace space group and trusted HSP irreps
Valley Chern residues from rotation eigenvalues
Authoritative reduced EBR results
Readiness blockers and warnings
Public output files
```

Read the valley weights together with the subspace symmetry and irrep results.
`qcut mode:` records how the momentum window was defined. In JSON,
`Valley projection summary` corresponds to `valley_projection_summary`; irreps, EBR
results, and unresolved conditions appear in `valley_resolved_irreps`,
`reduced_ebr_summary`, and `readiness_blocker_summary`. The separate
`valley_chern_mod` report records conditional rotation residues and their
applicability or missing evidence.

Projection status labels include `fixed_center_not_captured`, `not_derived`,
and `unreliable`. In particular, a low weight in a fixed-center window does
not establish that a state has no parent-valley origin. The debug sections
`Valley subspaces` and `Valley subspace analysis` include `S_min`
(minimum target-valley-subspace weight), `min_concentration`, `assigned_valleys`, and
`valley_weights_adapted`.

## Inputs and Configuration

An analysis needs:

- a ValleyScope HDF5 file containing selected coefficients, reciprocal
  vectors, k points, energies, and VASP band indices;
- monolayer reciprocal-lattice information and physically defined valley
  centers, including layer transforms when needed;
- target HSP labels and `analysis.iband` values that exist in the HDF5 file;
- a moiré or bilayer POSCAR/CONTCAR at
  `symmetry.operations.structure_file` for spglib operation detection.

The monolayer structure defines parent-layer reciprocal coordinates; the
moiré or bilayer structure defines the symmetry operations. They are not
interchangeable.

Use `output.profile: standard` for routine runs and `output.profile: debug`
to inspect projector residuals, representation matrices, subspace closure,
HSP stars, or sewing matrices. Advanced source-table and standard-setting
options are described by the CLI help and the configuration parser in
[`valleyscope/io/config.py`](valleyscope/io/config.py). They do not remove the
need to verify the physical conventions of the input.

## Outputs

The standard output contains:

| File | Purpose |
| --- | --- |
| `valley_summary.txt` | Valley weights, subspace symmetries, irreps, conditional Chern residues, EBR results, and unresolved conditions |
| `valley_summary.json` | The corresponding structured results for further analysis |
| `valley_weights.csv` | Quick scan of raw per-(kpoint, VASP band) valley weights |
| `valley_ebr_export_bundle.json` | Irrep vectors and supporting symmetry data; written when at least one EBR input satisfies the export conditions |
| `valley_reduced_ebr_mapping.json` | Written only when reduced EBR mapping is enabled and evaluated |

`valley_resolved_irreps` contains one record per sampled
`(kpoint, valley)`, including the valley-projected subspace space group, HSP
little group, valley-preserving operations, irrep multiplicities
(`irrep_multiplicities`), and the conditions that permit or prevent assignment.

Summary schema `2.2.0` includes `valley_chern_mod` in both output profiles.
Each valley row records its status, modulus, residue, subspace rank, rotation,
and supporting eigenvalue evidence or blocking reasons. A missing residue is
`null`, not zero. This addition does not change the EBR export, mapping, or
ingestion schemas, and does not promote a blocked EBR result.

Raw rows in `valley_weights.csv` are useful for screening, but individual rows
inside a near-degenerate band subspace are gauge-dependent. Interpret them
together with the subspace analysis, rather than as invariant band labels.

The debug profile retains detailed JSON/HDF5 evidence such as
`diagnostics.h5`, symmetry reports, restricted representation data, and
irrep/EBR source information. They are useful when a representation cannot be
assigned. Some tables are header-only when no state falls within the relevant
physical scope; this alone does not mean that the calculation failed.
Offline result collection reads the separate EBR export and mapping files as
well as the summary, so keep these files together.

## Running Reduced EBR Mapping

Reduced EBR analysis is off by default. To request it, add
`reduced_ebr: {enabled: true}` under `analysis` in `analyze.yaml`.
The analyzer can construct reduced tables from the installed `irreptables`
source, or read a reviewed reduced table or mapping specification supplied by
the user. In each case, the group, setting, spin convention, HSP basis,
irreps, and table origin must agree with the wavefunction calculation.

Run this analysis within `analyze-hsp`, while the wavefunctions and numerical
representations are available for checking. The separate command below checks
compatibility of exported data. JSON identifiers alone cannot reproduce the
wavefunction-level checks, so they are not sufficient for it to assign a
physically validated EBR result:

```bash
valleyscope map-reduced-ebr \
  valley_ebr_export_bundle.json \
  validated_reduced_ebr_table.json \
  --output valley_reduced_ebr_mapping.json
```

ValleyScope does not supply ad hoc or unreviewed EBR tables.

## Collecting Results Across Calculations

Completed calculations can be collected into a JSON record or a multi-run
index for comparison:

```bash
valleyscope collect-database-record ./valley_analysis \
  --output ./database_ingestion_record.json

valleyscope collect-database-index ./run_a/valley_analysis ./run_b/valley_analysis \
  --output ./database_index.json
```

The collector keeps final EBR results separate from incomplete inputs and
excluded results. It checks consistency with the current calculation summary,
but does not recompute symmetry matrices from the wavefunctions. Inputs must
be listed explicitly. This is offline result collection, not a database
service or an automated calculation scheduler.

## Numerical Checks and Present Results

Small generated spinor wavefunctions test the numerical calculation from
projection through irrep assignment and exact reduced EBR comparison. The
[`P3` example](tests/test_portable_numerical_chain.py) checks two
time-reversed valleys. The [`P4mm` example](tests/test_noncommuting_numerical_chain.py)
checks noncommuting, two-dimensional double-valued representations at
\(\Gamma\), \(X\), and \(M\). In both examples, normalized valley-pure
states that fail spatial-symmetry closure are rejected.
The P4mm example has one valley and a symmorphic space group; it does not
test valley-changing mirrors or nonsymmorphic translation phases.

Local material regressions at code revision `49cc142` gave:

| Calculation | Valley-resolved result for the selected band subspace |
| --- | --- |
| tMoTe₂ | Both \(K\) and \(K'\) irrep vectors lie outside the integer span of the reviewed reduced EBRs (`outside_integer_span`) |
| tZrSe₂ | Each of the three \(M\) valleys admits a nonnegative exact reduced EBR combination (`solved_exact`) |

These are results for specific band selections and HSPs, not universal
statements about either material. The tMoTe₂ result is not a direct Chern
calculation; the tZrSe₂ result does not establish global Wannierizability.
Some optional time-reversal or joint grey-group results remain unresolved.
Large material wavefunctions and their outputs are not distributed with the
repository; the generated tests can be run without them.

## Limits and Non-Goals

ValleyScope currently does not provide:

- raw three-dimensional EBR decomposition as a valley-resolved result;
- built-in unreviewed EBR tables or heuristic floating-point EBR fitting;
- compatibility relations;
- Berry curvature, Wilson loops, or full integer Chern numbers;
- automatic validation of valley character throughout the moiré Brillouin zone;
- an unconditional topology conclusion from HSP data alone.

## Development

Install the test dependency and run the suite:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

To run the small spinful numerical examples alone:

```bash
python -m pytest -q tests/test_portable_numerical_chain.py tests/test_noncommuting_numerical_chain.py
```

The tests generate their own small wavefunction files. Tests for local
development notes are skipped by default and do not require those notes in a
public checkout. An additional installed-package check builds and tests an
isolated copy of the committed revision:

```bash
python -m pip install build
python scripts/release_gate.py --checkout .
```

Run it from a clean checkout; it may download dependencies. The installed P3
and P4mm spin-pair examples also check the
conditional whole-subspace rotation residues. States that fail spatial-symmetry
closure must retain an unavailable residue, not zero. These are local
representation checks, not validation of a valley subspace throughout the mBZ.

Commands are defined in
[`valleyscope/cli.py`](valleyscope/cli.py), configuration parsing in
[`valleyscope/io/config.py`](valleyscope/io/config.py), and output selection in
[`valleyscope/reports/analysis_outputs.py`](valleyscope/reports/analysis_outputs.py).
