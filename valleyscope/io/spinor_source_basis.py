from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from valleyscope.io.h5_reader import WavefunctionData
from valleyscope.io.wavefunction_convention import (
    COEFFICIENT_SHAPE_ORDER,
    H5_PARSER_IDENTITY,
    V1_EVIDENCE_ORIGIN,
    V1_PROFILE_ASSUMPTIONS,
    V1_PROFILE_IDENTITY,
    WAVECAR_H5_EXTRACTOR_IDENTITY,
    canonical_identity,
    spinor_component_order,
    valid_sha256_identity,
)


SOURCE_BASIS_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class SourceBasisValidation:
    status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class SpinorSourceBasisCertificate:
    extracted_wavefunction_payload_identity: str
    nspinor: int
    parser_identity: str
    extractor_identity: str

    @property
    def applicability(self) -> str:
        return "applicable" if self.nspinor == 2 else "not_applicable"

    @property
    def reason_codes(self) -> tuple[str, ...]:
        if self.nspinor == 2:
            return ()
        return ("scalar_wavefunction_outside_v1_spinor_profile",)

    @property
    def status(self) -> str:
        return "passed" if self.nspinor == 2 else "not_applicable"

    def _identity_content(self) -> dict[str, object]:
        return {
            "schema_version": SOURCE_BASIS_SCHEMA_VERSION,
            "applicability": self.applicability,
            "profile_identity": V1_PROFILE_IDENTITY,
            "profile_assumptions": _profile_assumptions(),
            "evidence_origin": V1_EVIDENCE_ORIGIN,
            "source_claims_parsed": False,
            "coefficient_layout": {
                "shape_order": list(COEFFICIENT_SHAPE_ORDER),
                "component_order": list(spinor_component_order(self.nspinor)),
                "nspinor": self.nspinor,
            },
            "parser_identity": self.parser_identity,
            "extractor_identity": self.extractor_identity,
            "extracted_wavefunction_payload_identity": (
                self.extracted_wavefunction_payload_identity
            ),
        }

    @property
    def certificate_identity(self) -> str:
        return canonical_identity(self._identity_content())

    def to_record(self) -> dict[str, object]:
        record = self._identity_content()
        record.update(
            {
                "status": self.status,
                "reason_codes": list(self.reason_codes),
                "certificate_identity": self.certificate_identity,
            }
        )
        return record


def build_spinor_source_basis_certificate(
    wavefunction: WavefunctionData,
) -> SpinorSourceBasisCertificate:
    metadata = wavefunction.metadata
    return SpinorSourceBasisCertificate(
        extracted_wavefunction_payload_identity=metadata.hdf5_payload_identity,
        nspinor=metadata.nspinor,
        parser_identity=metadata.parser_identity,
        extractor_identity=metadata.extractor_identity,
    )


def validate_spinor_source_basis_record(
    record: Mapping[str, object],
) -> SourceBasisValidation:
    reasons: list[str] = []
    if not isinstance(record, Mapping):
        return SourceBasisValidation("blocked", ("record_malformed",))

    nspinor = _record_nspinor(record)
    if nspinor not in (1, 2):
        reasons.append("coefficient_layout_mismatch")
        nspinor = 2

    if record.get("schema_version") != SOURCE_BASIS_SCHEMA_VERSION:
        reasons.append("schema_version_mismatch")
    if record.get("profile_identity") != V1_PROFILE_IDENTITY:
        reasons.append("profile_identity_mismatch")
    if record.get("profile_assumptions") != _profile_assumptions():
        reasons.append("profile_assumptions_mismatch")
    if record.get("evidence_origin") != V1_EVIDENCE_ORIGIN:
        reasons.append("evidence_origin_mismatch")
    if record.get("source_claims_parsed") is not False:
        reasons.append("source_claims_origin_mismatch")

    expected_layout = {
        "shape_order": list(COEFFICIENT_SHAPE_ORDER),
        "component_order": list(spinor_component_order(nspinor)),
        "nspinor": nspinor,
    }
    if record.get("coefficient_layout") != expected_layout:
        reasons.append("coefficient_layout_mismatch")
    if record.get("parser_identity") != H5_PARSER_IDENTITY:
        reasons.append("parser_identity_mismatch")
    if record.get("extractor_identity") != WAVECAR_H5_EXTRACTOR_IDENTITY:
        reasons.append("extractor_identity_mismatch")

    payload_identity = record.get("extracted_wavefunction_payload_identity")
    if not valid_sha256_identity(payload_identity):
        reasons.append("payload_identity_malformed")

    expected_applicability = (
        "applicable" if nspinor == 2 else "not_applicable"
    )
    expected_status = "passed" if nspinor == 2 else "not_applicable"
    expected_reasons = (
        [] if nspinor == 2 else ["scalar_wavefunction_outside_v1_spinor_profile"]
    )
    if record.get("applicability") != expected_applicability:
        reasons.append("derived_applicability_mismatch")
    if record.get("status") != expected_status:
        reasons.append("derived_status_mismatch")
    if record.get("reason_codes") != expected_reasons:
        reasons.append("derived_reason_codes_mismatch")

    identity_content = {
        key: record.get(key)
        for key in (
            "schema_version",
            "applicability",
            "profile_identity",
            "profile_assumptions",
            "evidence_origin",
            "source_claims_parsed",
            "coefficient_layout",
            "parser_identity",
            "extractor_identity",
            "extracted_wavefunction_payload_identity",
        )
    }
    try:
        expected_identity = canonical_identity(identity_content)
    except (TypeError, ValueError):
        expected_identity = None
    if (
        expected_identity is None
        or record.get("certificate_identity") != expected_identity
        or not valid_sha256_identity(record.get("certificate_identity"))
    ):
        reasons.append("certificate_identity_mismatch")

    reasons = _unique(reasons)
    if reasons:
        return SourceBasisValidation("blocked", tuple(reasons))
    return SourceBasisValidation(expected_status, tuple(expected_reasons))


def _record_nspinor(record: Mapping[str, object]) -> int | None:
    layout = record.get("coefficient_layout")
    if not isinstance(layout, Mapping):
        return None
    value = layout.get("nspinor")
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return value


def _profile_assumptions() -> dict[str, Any]:
    return {
        "nonmagnetic": True,
        "soc": True,
        "time_reversal": True,
        "saxis_cart": [0.0, 0.0, 1.0],
    }


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
