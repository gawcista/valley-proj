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
(EBRs), reduced to the same valley subspace and HSP basis. The comparison
describes the symmetry content of the selected states at these momenta.

## Supported Scope

The present implementation is intended for:

- VASP plane-wave wavefunctions extracted from `WAVECAR` into a ValleyScope
  HDF5 file for analysis;
- nonmagnetic, spin-orbit-coupled two-dimensional moiré systems with parent
  time-reversal symmetry (TRS) and the default VASP Cartesian spin frame
  `SAXIS=[0,0,1]`;
- selected moiré HSPs and target bands, with specified parent-layer valley
  centers and a moiré or bilayer structure for determining spatial symmetries.

The construction follows the lattice, symmetry operations, and valley centers
of the supplied system. The numerical examples below give the cases tested
so far. Users should check the source calculation's magnetic state, SOC,
time-reversal symmetry, and spin frame against the conditions above.

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

`W_val` measures the weight captured by the chosen parent-valley momentum
windows. Its value depends on the valley centers and q-cut radius.

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
  starts from fixed-center seed projectors.

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
Its action relates the two subspaces through a valley sewing matrix. Where this
unitary relation has been verified from the wavefunctions, it can determine
the representation at a symmetry-related, unsampled point. The transported
subspace is checked against the target little-group irreps, with its origin
recorded separately from directly sampled states.

ValleyScope matches the representation on the full valley-preserving subgroup,
using double-valued irreps for SOC wavefunctions, including multidimensional
irreps. Comparison with Bilbao/irreptables requires a consistent
crystallographic setting: basis, origin, affine operations, and the complete
translation lattice.

### Time Reversal

When time reversal exchanges valleys, each valley is first described by the
unitary irreps of its own valley-preserving subgroup.

For a parent-TRS calculation, an unsampled time-reversed irrep may be inferred
from a verified source irrep and the reviewed time-reversal pairing in the
source tables. This algebraic inference is distinct from both unitary valley
sewing and a numerical test of antiunitary sewing between wavefunctions.
Directly sampled and inferred irreps remain distinguishable in the results.

A joint description including time reversal (a grey-group corepresentation)
requires additional antiunitary and inter-valley checks. Their results are
recorded alongside the independently assessed single-valley unitary representations.

Enable `analysis.time_reversal.enabled` only for input known to come from a
parent-TRS calculation. A symmetry relation is left unresolved when the
information needed to establish it is missing or inconsistent.

### Conditional Valley Chern Residues

Rotation eigenvalues can constrain a Chern number modulo \(n\), as derived by
[Fang, Gilbert and Bernevig, Phys. Rev. B 86, 115112 (2012)](https://doi.org/10.1103/PhysRevB.86.115112).
ValleyScope automatically selects the highest-order actual valley-preserving
rotation about \(+z\) among \(n=2,3,4,6\), and combines the determinants of
the subspace rotation matrices at the required invariant momenta. The result
is for the **whole selected valley subspace**. The selected operation must be
a finite-order, valley-preserving rotation about the out-of-plane axis.
Spinful rotation phases and any validated time-reversal inference are included;
inferred evidence is identified separately from sampled evidence.

A result such as `C = 1 (mod 3)` is conditional on the existence of a smooth,
constant-rank valley subspace throughout the mBZ, with consistent Bloch boundary
identifications and the stated rotation symmetry. The result specifies the
congruence class \(C=r+mn\), with integer \(m\). Its `conditional` status
records these global assumptions.

This analysis uses the existing wavefunction and symmetry inputs and runs
independently of the reduced EBR option. Without a valley-preserving rotation
it is `not_applicable`; missing required HSPs or unverified representation
evidence give `blocked`,
with reasons and a `null` residue. Available numerical results are always
`conditional`, while `global_valley_subspace_status` remains `not_evaluated`.

### Reduced EBR Analysis

ValleyScope uses reviewed Bilbao/irreptables data
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
at the HSPs considered. Integer-span membership is defined relative to the
chosen valley subspace, HSP basis, and reviewed EBR generators. Establishing
a global Wannier description additionally requires information about the
Bloch subspace throughout the Brillouin zone.

### Conditions for Assigning Irreps

Assigning a symmetry representation to a valley subspace requires:

- target-subspace closure and representation unitarity;
- seed-projector covariance under the full valley mapping;
- a well-defined valley-projected subspace and \(G_k^{(a)}\);
- complete mappings of plane waves, symmetry operations, and source HSPs;
- agreement with the crystallographic setting and irrep/EBR source tables;
- the double-group multiplication law and, when used, antiunitary sewing.

A fixed-center seed basis can be used directly if it satisfies these conditions.
Otherwise, ValleyScope attempts a symmetry-adapted valley basis. If both
constructions fail, the affected irrep or EBR result is reported as unresolved.
Both output profiles apply the same physical requirements and retain the
residuals and reasons for unresolved results.

## Installation

ValleyScope requires Python 3.10 or newer. Use a dedicated environment for
the wavefunction analysis:

```bash
git clone https://github.com/gawcista/valley-proj.git
cd valley-proj
python -m venv .venv
source .venv/bin/activate
python scripts/build_irrep_compat_wheel.py --output-dir dist/compat
python -m pip install --find-links dist/compat -e .
```

### Dependencies

The direct runtime dependencies and their roles are:

| Package | Requirement | Use in the analysis |
| --- | --- | --- |
| NumPy | Installed by pip | Plane-wave coefficients, projectors, and representation matrices |
| h5py | Installed by pip | Selected-band wavefunctions in HDF5 |
| PyYAML | Installed by pip | Lattice, valley, and analysis parameters |
| spglib | Installed by pip | Spatial symmetry operations of the supplied structure |
| SymPy | Installed by pip | Exact-integer reduced EBR decomposition |
| irreptables | `3.1.0` | Standard little-group irreps and source EBR data |
| irrep | `3.3.0+valleyscope.1` | Utility functions imported by irreptables |

`pip` also installs the dependencies of these packages, including SciPy
through irrep. The test extra adds pytest; building a ValleyScope wheel or
source distribution uses `build` and `setuptools>=68`.

The helper above downloads the official irrep 3.3.0 wheel, verifies its fixed
SHA256, and builds the local revision `3.3.0+valleyscope.1`. Its source change
is a single f-string quote correction in `Storage.generate_name`, allowing
the package to import on Python 3.10. The representation tables, EBR data,
and numerical methods retain their upstream contents. The wheel preserves
the upstream license and records the source URLs, hashes, and change in
`valleyscope_patch.json` in its distribution metadata. Package versions can
be inspected with `python -m pip show irrep irreptables`.

For an offline build of this dependency, provide the official wheel with
`--source-wheel /path/to/irrep-3.3.0-py3-none-any.whl`; the same SHA256 check
applies. Keep the generated wheel available through `--find-links` when
installing ValleyScope. A fully offline installation also needs wheels for
the remaining dependencies. Source and installed-package CI cover Python
3.10 and 3.13.

Check the installed commands:

```bash
valleyscope --help
valleyscope analyze-hsp --help
```

From a source checkout, `python -m valleyscope.cli --help` is equivalent.

## Quick Start

Moiré supercells contain many atoms and consequently many bands, while a
study often focuses on just a few bands near the energy range of interest.
For that selected subspace, ValleyScope extracts the specified bands at the
required k points from the large `WAVECAR` and saves their complete plane-wave
coefficients, including the spinor components for SOC states, and associated
data in a compact HDF5 file. Select the complete target band manifold,
including near-degenerate partners; the later symmetry-closure checks assess
its suitability for representation analysis. Subsequent valley
projection, symmetry, and reduced EBR analyses read this HDF5 file. Extracting
it once reduces the data to store and transfer between computers and allows
the same wavefunctions to be reused across analyses.

The HDF5 file also connects wavefunction extraction with the common analysis
methods. It stores the coefficients, lattice, reciprocal vectors, k points,
energies, and band indices used downstream. This organization provides a
route to other DFT programs: a future extractor for Quantum ESPRESSO (QE),
for example, could supply the corresponding data in the same format. Extending
the symmetry analysis also requires matching and validating that program's
coordinate, Bloch-phase, normalization, and spinor conventions. The current
extractor reads VASP `WAVECAR` files.

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

This writes `wavefunctions.h5` with the selected bands and k points. The
analysis configuration below uses it as `input.wavefunction_h5`.

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
and `unreliable`. A low fixed-center weight means that the chosen window
captures little of the state at that sampled momentum. The debug sections
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
moiré or bilayer structure defines the symmetry operations.

Use `output.profile: standard` for routine runs and `output.profile: debug`
to inspect projector residuals, representation matrices, subspace closure,
HSP stars, or sewing matrices. Advanced source-table and standard-setting
options are described by the CLI help and the configuration parser in
[`valleyscope/io/config.py`](valleyscope/io/config.py). Choose them consistently
with the physical conventions of the input.

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
`null`. EBR export, mapping, and ingestion retain their own schemas and
physical acceptance conditions.

Raw rows in `valley_weights.csv` are useful for screening, but individual rows
inside a near-degenerate band subspace are gauge-dependent. Interpret them
together with the subspace analysis.

The debug profile retains detailed JSON/HDF5 evidence such as
`diagnostics.h5`, symmetry reports, restricted representation data, and
irrep/EBR source information. They help explain unresolved representation
assignments. Some tables are header-only when no state falls within the relevant
physical scope; the accompanying summary records the outcome.
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
compatibility of exported data; physical validation uses the wavefunctions
and numerical representations available in `analyze-hsp`:

```bash
valleyscope map-reduced-ebr \
  valley_ebr_export_bundle.json \
  validated_reduced_ebr_table.json \
  --output valley_reduced_ebr_mapping.json
```

The reduced tables retain their reviewed source and crystallographic setting.

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
using the stored symmetry results. Inputs are listed explicitly. Database
persistence and cluster scheduling are future extensions whose design will
depend on where the wavefunctions are computed and how completed calculations
are collected.

## Numerical Checks and Present Results

Small generated spinor wavefunctions test the numerical calculation from
projection through irrep assignment and exact reduced EBR comparison. The
[`P3` example](tests/test_portable_numerical_chain.py) checks two
time-reversed valleys. The [`P4mm` example](tests/test_noncommuting_numerical_chain.py)
checks noncommuting, two-dimensional double-valued representations at
\(\Gamma\), \(X\), and \(M\). In both examples, normalized valley-pure
states that fail spatial-symmetry closure are rejected.
The P4mm example covers one valley in a symmorphic space group, including
the action of noncommuting little-group operations.

For the rank-two valley subspaces in these examples, the rotation analysis
gives conditional residues of zero modulo 3 (P3) and modulo 4 (P4mm).
The symmetry-breaking controls give `blocked` with a `null` residue.
These checks also run against the installed package. Separate determinant
tests cover nonzero residues for each allowed rotation order, with and
without spin. These determinant tests check the rotation formula.

Local material regressions at code revision `49cc142` gave:

| Calculation | Valley-resolved result for the selected band subspace |
| --- | --- |
| tMoTe₂ | Both \(K\) and \(K'\) irrep vectors lie outside the integer span of the reviewed reduced EBRs (`outside_integer_span`) |
| tZrSe₂ | Each of the three \(M\) valleys admits a nonnegative exact reduced EBR combination (`solved_exact`) |

For the same retained tMoTe₂ wavefunctions and selected bands, a subsequent
rotation analysis gives \(C_K=1\pmod 3\) and \(C_{K'}=2\pmod 3\) for the
selected rank-one valley subspaces. Both results are conditional. A required
unsampled HSP is supplied
by reviewed time-reversal inference from the opposite valley. The allowed
integers are \(C_K=1+3m\) and \(C_{K'}=2+3m'\), with integers \(m,m'\).

A projection control on the retained tMoTe₂ wavefunctions replaces the
\(K/K'\) centers with \(\Gamma\) or the three \(M\) centers. Across three
sampled HSPs and the existing q-cut scan, the \(K/K'\) windows capture most
of the selected states, while the alternative windows remain below the
unchanged capture threshold. The \(\Gamma\) weight is nearly zero. The
nonzero \(M\)-window weight at the configured q-cut comes mainly from
intersections with the other layer's periodic \(K/K'\) windows. It measures
capture within these momentum windows. The check includes both individual-band
weights and the spectrum of the projector restricted to the full target
subspace. A small generated example of this cross-layer window overlap is
included in [the projection controls](tests/test_projection_controls.py).

These material results refer to the stated band selections, HSPs, and projection
windows. Some optional time-reversal or joint grey-group results remain
unresolved. The public tests generate small wavefunctions; the larger material
calculations and their outputs are retained locally.

## Development

Install the test dependency and run the suite:

```bash
python scripts/build_irrep_compat_wheel.py --output-dir dist/compat
python -m pip install --find-links dist/compat -e ".[test]"
python -m pytest -q
```

To run the small spinful numerical examples alone:

```bash
python -m pytest -q tests/test_portable_numerical_chain.py tests/test_noncommuting_numerical_chain.py
```

The tests generate their own small wavefunction files. Tests for local
development notes run when explicitly enabled. An additional installed-package
check builds and tests an isolated copy of the committed revision:

```bash
python -m pip install build
python scripts/release_gate.py --checkout .
```

Run it from a clean checkout; it builds the same irrep compatibility wheel
and may download dependencies. The installed P3
and P4mm spin-pair examples also check the
conditional whole-subspace rotation residues. States that fail spatial-symmetry
closure return `blocked` with a `null` residue. These checks exercise the
sampled little-group representations and their rotation determinants.

Commands are defined in
[`valleyscope/cli.py`](valleyscope/cli.py), configuration parsing in
[`valleyscope/io/config.py`](valleyscope/io/config.py), and output selection in
[`valleyscope/reports/analysis_outputs.py`](valleyscope/reports/analysis_outputs.py).
