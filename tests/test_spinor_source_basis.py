from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from valleyscope.io.h5_reader import read_wavefunction_h5
from valleyscope.io.spinor_source_basis import (
    build_spinor_source_basis_certificate,
    validate_spinor_source_basis_record,
)
from valleyscope.io.wavefunction_convention import (
    H5_LAYOUT_IDENTITY,
    WAVECAR_H5_EXTRACTOR_IDENTITY,
)


def _write_wavefunction_h5(path: Path, *, nspinor: int) -> None:
    with h5py.File(path, "w") as h5:
        metadata = h5.create_group("metadata")
        lattice = metadata.create_group("lattice")
        lattice["direct_cart"] = np.eye(3)
        lattice["reciprocal_cart"] = np.eye(3)
        metadata["spinor"] = nspinor == 2
        metadata["source"] = "WAVECAR:test"
        metadata["vasp_band_index_base"] = 1
        kpoint = h5.create_group("kpoints").create_group("0")
        kpoint["name"] = "G"
        kpoint["frac"] = np.zeros(3)
        kpoint["cart"] = np.zeros(3)
        kpoint["g_vectors_frac"] = np.zeros((1, 3), dtype=int)
        kpoint["g_vectors_cart"] = np.zeros((1, 3))
        coefficients = np.zeros((1, nspinor, 1), dtype=np.complex128)
        coefficients[0, 0, 0] = 1.0
        kpoint["coefficients"] = coefficients
        kpoint["energies_eV"] = np.zeros(1)
        kpoint["band_indices_vasp"] = np.ones(1, dtype=int)


def test_spinor_source_basis_uses_fixed_v1_scope_without_user_input(tmp_path):
    path = tmp_path / "wave.h5"
    _write_wavefunction_h5(path, nspinor=2)

    wavefunction = read_wavefunction_h5(path)
    record = build_spinor_source_basis_certificate(wavefunction).to_record()

    assert record["schema_version"] == "1.1.0"
    assert record["applicability"] == "applicable"
    assert record["status"] == "passed"
    assert record["reason_codes"] == []
    assert record["profile_identity"] == "vasp_nonmagnetic_soc_default_saxis_v1"
    assert record["profile_assumptions"] == {
        "nonmagnetic": True,
        "soc": True,
        "time_reversal": True,
        "saxis_cart": [0.0, 0.0, 1.0],
    }
    assert record["evidence_origin"] == "workflow_scope_contract"
    assert record["source_claims_parsed"] is False
    assert record["coefficient_layout"]["shape_order"] == [
        "band",
        "spinor_component",
        "reciprocal_grid",
    ]
    assert record["coefficient_layout"]["component_order"] == [
        "vasp_spinor_component_0",
        "vasp_spinor_component_1",
    ]
    assert record["coefficient_layout"]["nspinor"] == 2
    assert record["parser_identity"] == "valleyscope_h5_reader_v1"
    assert record["hdf5_layout_identity"] == H5_LAYOUT_IDENTITY
    assert record["extractor_provenance"] is None
    assert "extractor_identity" not in record
    assert record["extracted_wavefunction_payload_identity"].startswith("sha256:")
    assert record["certificate_identity"].startswith("sha256:")
    assert validate_spinor_source_basis_record(record).status == "passed"


def test_spinor_source_basis_preserves_actual_optional_extractor_provenance(
    tmp_path,
):
    path = tmp_path / "wave.h5"
    _write_wavefunction_h5(path, nspinor=2)
    with h5py.File(path, "r+") as h5:
        h5["metadata/extractor_identity"] = WAVECAR_H5_EXTRACTOR_IDENTITY

    wavefunction = read_wavefunction_h5(path)
    record = build_spinor_source_basis_certificate(wavefunction).to_record()

    assert wavefunction.metadata.hdf5_layout_identity == H5_LAYOUT_IDENTITY
    assert (
        wavefunction.metadata.extractor_provenance
        == WAVECAR_H5_EXTRACTOR_IDENTITY
    )
    assert record["hdf5_layout_identity"] == H5_LAYOUT_IDENTITY
    assert record["extractor_provenance"] == WAVECAR_H5_EXTRACTOR_IDENTITY
    assert validate_spinor_source_basis_record(record).status == "passed"


def test_spinor_source_basis_marks_scalar_input_not_applicable(tmp_path):
    path = tmp_path / "wave.h5"
    _write_wavefunction_h5(path, nspinor=1)

    record = build_spinor_source_basis_certificate(
        read_wavefunction_h5(path)
    ).to_record()

    assert record["applicability"] == "not_applicable"
    assert record["status"] == "not_applicable"
    assert record["reason_codes"] == ["scalar_wavefunction_outside_v1_spinor_profile"]
    assert validate_spinor_source_basis_record(record).status == "not_applicable"


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (
            ("profile_assumptions", "soc"),
            False,
            "profile_assumptions_mismatch",
        ),
        (
            ("profile_assumptions", "saxis_cart"),
            [1.0, 0.0, 0.0],
            "profile_assumptions_mismatch",
        ),
        (
            ("coefficient_layout", "component_order"),
            ["vasp_spinor_component_1", "vasp_spinor_component_0"],
            "coefficient_layout_mismatch",
        ),
        (
            ("extracted_wavefunction_payload_identity",),
            "sha256:" + "0" * 64,
            "certificate_identity_mismatch",
        ),
        (
            ("hdf5_layout_identity",),
            "unreviewed_layout",
            "hdf5_layout_identity_mismatch",
        ),
        (
            ("parser_identity",),
            "unreviewed_parser",
            "parser_identity_mismatch",
        ),
        (
            ("status",),
            "blocked",
            "derived_status_mismatch",
        ),
        (
            ("certificate_identity",),
            "sha256:" + "f" * 64,
            "certificate_identity_mismatch",
        ),
    ],
)
def test_spinor_source_basis_rejects_tampered_serialized_records(
    tmp_path,
    path,
    value,
    reason,
):
    h5_path = tmp_path / "wave.h5"
    _write_wavefunction_h5(h5_path, nspinor=2)
    record = build_spinor_source_basis_certificate(
        read_wavefunction_h5(h5_path)
    ).to_record()
    tampered = deepcopy(record)
    target = tampered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    validation = validate_spinor_source_basis_record(tampered)

    assert validation.status == "blocked"
    assert reason in validation.reason_codes


def test_hdf5_payload_identity_changes_when_coefficient_payload_changes(tmp_path):
    path = tmp_path / "wave.h5"
    _write_wavefunction_h5(path, nspinor=2)
    before = read_wavefunction_h5(path).metadata.hdf5_payload_identity

    with h5py.File(path, "r+") as h5:
        h5["kpoints/0/coefficients"][0, 0, 0] = 1.0j

    after = read_wavefunction_h5(path).metadata.hdf5_payload_identity
    assert before != after


def test_hdf5_reader_rejects_spinor_metadata_shape_conflict(tmp_path):
    path = tmp_path / "wave.h5"
    _write_wavefunction_h5(path, nspinor=2)
    with h5py.File(path, "r+") as h5:
        h5["metadata/spinor"][()] = False

    with pytest.raises(
        ValueError,
        match="metadata/spinor conflicts with coefficient nspinor",
    ):
        read_wavefunction_h5(path)
