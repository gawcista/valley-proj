"""Readiness composition from revalidated scoped representation evidence."""

from __future__ import annotations

from collections.abc import Mapping

from valleyscope.analysis.scoped_representation_evidence import (
    validate_scoped_representation_evidence_record,
)


def compose_representation_readiness(
    scoped_evidence_record: Mapping[str, object],
    **raw_inputs: object,
) -> dict[str, object]:
    """Revalidate producer evidence and derive consumer-specific readiness."""
    validation = validate_scoped_representation_evidence_record(
        scoped_evidence_record,
        **raw_inputs,
    )
    scope = scoped_evidence_record.get("scope")
    scope_kind = scope.get("scope_kind") if isinstance(scope, Mapping) else None
    evidence_valid = validation.status == "passed"
    blockers: list[str] = []
    if not evidence_valid:
        blockers.append("scoped_representation_evidence_invalid")
        blockers.extend(validation.reason_codes)

    return {
        "scoped_representation_evidence_identity": (
            scoped_evidence_record.get("evidence_identity")
        ),
        "scope_kind": scope_kind,
        "local_irrep_ready": evidence_valid and scope_kind == "local_irrep",
        "valley_sewing_ready": evidence_valid and scope_kind == "valley_sewing",
        "diagnostic_only": not evidence_valid,
        "blockers": list(dict.fromkeys(blockers)),
    }
