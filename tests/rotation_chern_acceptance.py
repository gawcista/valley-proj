"""Installed-gate checks for generated rank-two spin-pair Chern controls.

The P3/P4mm orbitals are trivial and their spin-pair rotation determinants
are one. Thus their whole-subspace residues are independently zero, with
no assertion of a smooth global valley bundle or a full integer Chern number.
This module validates real workflow output; it produces no trust evidence.
"""

from __future__ import annotations

import cmath


def run_rotation_chern_arithmetic_acceptance() -> dict[str, object]:
    """Nonzero formula controls only; no wavefunction or trust certification."""
    from valleyscope.analysis.rotation_chern_math import rotation_chern_residue

    # Products are exp(2*pi*i/n), yielding residue one without a spin factor.
    scalar_inputs = {
        2: [-1, 1, 1, 1],
        3: [cmath.exp(2j * cmath.pi / 3), 1, 1],
        4: [1j, 1, 1],
        6: [cmath.exp(1j * cmath.pi / 3), 1, 1],
    }
    # Rank-one spinful eigenvalues obey their local rotation orders. For
    # n=3,4,6 the raw product is -exp(2*pi*i/n), so the odd-rank factor
    # changes it to residue one. C2 has no extra spin factor.
    spinful_inputs = {
        2: [1j, 1j, 1j, -1j],
        3: [cmath.exp(1j * cmath.pi / 3), cmath.exp(1j * cmath.pi / 3), -1],
        4: [cmath.exp(1j * cmath.pi / 4), cmath.exp(-1j * cmath.pi / 4), -1j],
        6: [cmath.exp(1j * cmath.pi / 6), cmath.exp(-1j * cmath.pi / 3), -1j],
    }
    rows = []
    for spinful, rank, inputs in ((False, 2, scalar_inputs), (True, 1, spinful_inputs)):
        for order, determinants in inputs.items():
            residue = rotation_chern_residue(order, determinants, rank, spinful)
            assert type(residue) is int and residue == 1, (order, rank, spinful, residue)
            rows.append({"modulus": order, "residue": residue,
                         "spinful": spinful, "subspace_rank": rank})
    return {"scope": "arithmetic_only_not_numerical_trust", "rows": rows}


def assert_rotation_chern_summary(
    summary: dict[str, object], *, valleys: set[str], modulus: int,
    expected_hsp_powers: dict[str, int], blocked: bool = False,
) -> dict[str, object]:
    """Check exact fixture expectations and return the checked log fields."""
    assert summary["schema_version"] == "2.2.0"
    report = summary["valley_chern_mod"]
    expected_status = "blocked" if blocked else "conditional"
    assert report["status"] == expected_status, report
    assert report["global_valley_subspace_status"] == "not_evaluated", report
    rows = report["rows"]
    assert isinstance(rows, list) and len(rows) == len(valleys), rows
    assert {row["valley"] for row in rows} == valleys, rows
    for row in rows:
        assert type(row["modulus"]) is int and row["modulus"] == modulus, row
        assert row["status"] == expected_status, row
        if blocked:
            assert row["residue"] is None, row
            assert isinstance(row["blocking_reasons"], list) and row["blocking_reasons"], row
            assert all(isinstance(reason, str) and reason for reason in row["blocking_reasons"]), row
            continue
        assert type(row["residue"]) is int and row["residue"] == 0, row
        assert type(row["subspace_rank"]) is int and row["subspace_rank"] == 2, row
        assert row["blocking_reasons"] == [], row
        evidence = row["eigenvalue_evidence"]
        assert len(evidence) == len(expected_hsp_powers), row
        assert {entry["sampled_kpoint"]: entry["rotation_power"] for entry in evidence} == expected_hsp_powers, row
        for entry in evidence:
            assert entry["completion_kind"] == "observed_at_sampled_kpoint", entry
            assert entry["source_hsp_label"] == entry["sampled_kpoint"], entry
            assert type(entry["subspace_rank"]) is int and entry["subspace_rank"] == 2, entry
            assert isinstance(entry["scoped_representation_evidence_identity"], str), entry
            assert entry["scoped_representation_evidence_identity"], entry
            # Trivial spatial orbital times the complete spin pair has
            # det(exp(-i theta sigma_z / 2)) = 1 at every required HSP.
            assert abs(complex(**entry["determinant"]) - 1) < 1e-12, entry
    return {
        "schema_version": summary["schema_version"],
        "status": report["status"],
        "global_valley_subspace_status": report["global_valley_subspace_status"],
        "rows": [
            {key: row[key] for key in ("valley", "modulus", "residue", "status", "subspace_rank")}
            for row in sorted(rows, key=lambda row: row["valley"])
        ],
    }
