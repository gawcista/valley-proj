"""Reduced EBR mapping: exact integer decomposition from export bundles.

Loads a user-supplied reduced-EBR table, validates it, and performs exact
integer matching via Smith normal form plus bounded nonnegative search.  No
built-in tables are provided; real-material EBR claims require an explicit
table file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from valleyscope.irreps.magnetic_groups import derive_type_ii_bns_number
from valleyscope.irreps.time_reversal_geometry import (
    centered_k_equivalent,
    normalize_centering_vectors,
)
from valleyscope.analysis.time_reversal_sewing import (
    validate_time_reversal_sewing_report,
)
from valleyscope.analysis.promotion_identity import (
    build_promotion_input_identity,
    merge_table_input_provenance,
)
from valleyscope.analysis.unitary_provenance import (
    unitary_bundle_claims_time_reversal_completion,
    validate_direct_unitary_bundle,
    validate_tr_completed_unitary_bundle,
)
from valleyscope.io.wavefunction_convention import valid_sha256_identity
from valleyscope.analysis.scoped_representation_evidence import (
    validate_scoped_representation_evidence_record,
)

_REQUIRED_TABLE_KEYS = {"schema_version", "subspace_group_candidate",
                         "expected_hsps", "irreps", "ebrs"}
_OUTPUT_SCHEMA_VERSION = "2.0.0"
_SOLVER_NAME = "smith_normal_form_plus_bounded_nonnegative_search"

# Standard-setting certificate convention — REAL producer vocabulary.
# These values are emitted by StandardSettingCertificate (standard_setting_kmap)
# and serialized by ebr_problem_instances._certificate_identity().  The
# validator consumes them verbatim; it must not invent a parallel vocabulary.
_TRUSTED_DATA_SOURCE = "irreptables"
_TRUSTED_PACKAGE = "irreptables"
_TRUSTED_REDUCTION = "sampled_hsp_valley_preserving"
_PRIMITIVE_DIRECT_RELATION = "direct_coordinate_match"
_CENTERED_RELATIONS = frozenset({
    "explicit_transform", "operation_basis_reconstruction",
})
_OP_MAPPING_PASSED = "operation_basis_verification_passed"
_AFFINE_PASSED = "passed"
_CENTERED_TYPES = frozenset({"A", "B", "C", "I", "F", "R"})


def _blocker(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


# ---------------------------------------------------------------------------
# Independent crystallographic evidence (spglib database only, never derived
# from the bundle being validated).  No SG-specific lookup tables.
# ---------------------------------------------------------------------------
import functools


@functools.lru_cache(maxsize=None)
def _spglib_type(hall_number: int):
    try:
        import spglib
        return spglib.get_spacegroup_type(int(hall_number))
    except Exception:
        return None


def _hall_belongs_to_sg(hall_number: object, sg_number: object) -> bool:
    if not _is_positive_int(hall_number) or not _is_positive_int(sg_number):
        return False
    t = _spglib_type(int(hall_number))
    if t is None:
        return False
    try:
        return int(t.number) == int(sg_number)
    except Exception:
        return False


def _centering_from_hall_symbol(hall_symbol: object) -> str:
    token = str(hall_symbol or "").strip()
    if token.startswith("-"):
        token = token[1:].lstrip()
    return token[:1].upper()


@functools.lru_cache(maxsize=None)
def _derive_table_standard_setting(
    sg_number: int,
    spinful: bool = False,
) -> dict | None:
    """Derive the canonical Hall setting from irreptables source operations.

    This evidence is independent of the bundle certificate.  All Hall choices
    for the same SG are checked; zero or multiple affine matches fail closed.
    """
    try:
        from valleyscope.analysis.standard_setting_kmap import (
            derive_irreptables_standard_setting_identity,
        )
        from valleyscope.irreps.tables import load_standard_irrep_table
        table = load_standard_irrep_table(sg_number, spinor=spinful)
        identity = derive_irreptables_standard_setting_identity(table, sg_number)
    except Exception:
        return None
    if identity.get("status") != "unique_match":
        return None
    return {
        "hall_number": int(identity["hall_number"]),
        "hall_symbol": str(identity["hall_symbol"]),
        "centering_type": str(identity["centering_type"]),
        "space_group_number": int(identity["space_group_number"]),
        "space_group_symbol": str(identity["space_group_symbol"]),
        "canonical_setting_status": str(identity["status"]),
        "canonical_setting_source": str(identity["source"]),
        "candidate_hall_numbers": list(identity["candidate_hall_numbers"]),
        "affine_matching_hall_numbers": list(
            identity["affine_matching_hall_numbers"]
        ),
        "primitive_conventional_index": int(
            identity["primitive_conventional_index"]
        ),
        "centering_cosets": [
            list(vector) for vector in identity["centering_cosets"]
        ],
        "standard_setting_operation_count": int(
            identity["standard_setting_operation_count"]
        ),
        "standard_operation_closure_validated": bool(
            identity["standard_operation_closure_validated"]
        ),
    }


# ---------------------------------------------------------------------------
# Production bundle promotion: validation candidate → validated table/bundle pair
# ---------------------------------------------------------------------------

def promote_bundle_for_solve(
    *,
    bundle: dict,
    table: dict,
    cprime_validation_context: dict[str, object] | None = None,
) -> dict:
    """Validate a bundle against a reduced EBR table and promote if compatible.

    Fail-closed convention trust chain.  Promotion requires that the bundle's
    standard-setting certificate (produced by the real
    ``StandardSettingCertificate`` serializer) is validated AND internally
    consistent, and that it agrees with an *independently derived* table
    standard setting (spglib), the table provenance, and the spin convention.
    There is no bypass and no injected fictitious evidence: the table setting
    is never copied from the bundle.

    Every disagreement or absence blocks with a structured ``{code, detail}``
    reason.  Only an all-passed report promotes to ``validated_basis``.
    """
    blockers: list[dict[str, str]] = []
    report: dict[str, object] = {
        "table_provenance_check": "not_attempted",
        "table_setting_check": "not_attempted",
        "table_serialized_setting_check": "not_attempted",
        "sg_symbol_check": "not_attempted",
        "sg_number_check": "not_attempted",
        "certificate_check": "not_attempted",
        "cprime_identity_check": "not_attempted",
        "certificate_consistency_check": "not_attempted",
        "cert_sg_consistency_check": "not_attempted",
        "affine_setting_check": "not_attempted",
        "hall_setting_check": "not_attempted",
        "spin_convention_check": "not_attempted",
        "problem_kind_check": "not_attempted",
        "hsp_basis_check": "not_attempted",
        "irrep_basis_check": "not_attempted",
    }

    # ---- A. Table provenance basics ----
    prov = table.get("provenance", {})
    if not isinstance(prov, dict):
        prov = {}
    data_source = str(prov.get("data_source", ""))
    package = str(prov.get("package", ""))
    package_version = str(prov.get("package_version", ""))
    table_spinful = prov.get("spinful")
    table_sg_num = prov.get("space_group_number")
    valleyscope_reduction = str(prov.get("valleyscope_reduction", ""))

    prov_ok = True
    if data_source != _TRUSTED_DATA_SOURCE:
        blockers.append(_blocker("table_data_source_invalid",
            f"table data_source must be '{_TRUSTED_DATA_SOURCE}', got "
            f"{data_source!r}"))
        prov_ok = False
    if package != _TRUSTED_PACKAGE or not package_version:
        blockers.append(_blocker("table_package_invalid",
            f"table package must be '{_TRUSTED_PACKAGE}' with non-empty "
            f"version, got package={package!r} version={package_version!r}"))
        prov_ok = False
    if not _is_positive_int(table_sg_num):
        blockers.append(_blocker("table_sg_number_missing",
            "table provenance missing positive space_group_number"))
        prov_ok = False
    if not isinstance(table_spinful, bool):
        blockers.append(_blocker("table_spinful_missing",
            "table provenance missing boolean spinful"))
        prov_ok = False
    if valleyscope_reduction != _TRUSTED_REDUCTION:
        blockers.append(_blocker("table_reduction_provenance_invalid",
            f"table valleyscope_reduction must be '{_TRUSTED_REDUCTION}', got "
            f"{valleyscope_reduction!r}"))
        prov_ok = False
    report["table_provenance_check"] = "passed" if prov_ok else "failed"

    _validate_problem_kind_compatibility(
        bundle=bundle,
        provenance=prov,
        table_space_group_number=table_sg_num,
        table_spinful=table_spinful,
        blockers=blockers,
        report=report,
    )

    # ---- A2. Independent table standard-setting evidence (spglib) ----
    table_setting = (
        _derive_table_standard_setting(int(table_sg_num), table_spinful)
        if _is_positive_int(table_sg_num) and isinstance(table_spinful, bool)
        else None
    )
    if table_setting is None:
        blockers.append(_blocker("table_standard_setting_unresolved",
            "could not independently derive a unique Bilbao/irreptables "
            f"standard setting for SG {table_sg_num!r}; irreptables does not "
            "expose a unique Hall/centering choice"))
        report["table_setting_check"] = "blocked"
        table_hall = None
        table_hall_symbol = ""
        table_centering = ""
    else:
        report["table_setting_check"] = "passed"
        table_hall = table_setting["hall_number"]
        table_hall_symbol = table_setting["hall_symbol"]
        table_centering = table_setting["centering_type"]

    serialized_table_setting = prov.get("standard_setting_identity")
    if serialized_table_setting is None:
        report["table_serialized_setting_check"] = "not_provided"
    elif not isinstance(serialized_table_setting, dict) or table_setting is None:
        blockers.append(_blocker(
            "table_serialized_setting_invalid",
            "table provenance standard_setting_identity is malformed or "
            "cannot be independently validated",
        ))
        report["table_serialized_setting_check"] = "failed"
    else:
        serialized_ok = (
            serialized_table_setting.get("status") == "unique_match"
            and serialized_table_setting.get("hall_number") == table_hall
            and serialized_table_setting.get("hall_symbol") == table_hall_symbol
            and serialized_table_setting.get("centering_type") == table_centering
            and serialized_table_setting.get("affine_matching_hall_numbers")
            == [table_hall]
            and serialized_table_setting.get("primitive_conventional_index")
            == table_setting["primitive_conventional_index"]
            and serialized_table_setting.get("centering_cosets")
            == table_setting["centering_cosets"]
            and serialized_table_setting.get(
                "standard_setting_operation_count"
            ) == table_setting["standard_setting_operation_count"]
            and serialized_table_setting.get(
                "standard_operation_closure_validated"
            ) is True
        )
        if serialized_ok:
            report["table_serialized_setting_check"] = "passed"
        else:
            blockers.append(_blocker(
                "table_serialized_setting_mismatch",
                "serialized irreptables standard-setting identity does not "
                "match an independent source-table derivation",
            ))
            report["table_serialized_setting_check"] = "failed"

    # ---- B. SG number + symbol: bundle vs table ----
    bundle_sg = str(bundle.get("subspace_group_candidate", ""))
    table_sg = str(table.get("subspace_group_candidate", ""))
    if bundle_sg and bundle_sg == table_sg:
        report["sg_symbol_check"] = "passed"
    else:
        blockers.append(_blocker("sg_symbol_mismatch",
            f"SG symbol mismatch: bundle {bundle_sg!r} != table {table_sg!r}"))
        report["sg_symbol_check"] = "failed"

    bundle_sg_num = bundle.get("subspace_sg_number")
    if not _is_positive_int(bundle_sg_num):
        blockers.append(_blocker("sg_number_missing",
            "bundle missing positive subspace_sg_number"))
        report["sg_number_check"] = "failed"
    elif not _is_positive_int(table_sg_num):
        report["sg_number_check"] = "failed"
    elif int(bundle_sg_num) != int(table_sg_num):
        blockers.append(_blocker("sg_number_mismatch",
            f"SG number mismatch: bundle {bundle_sg_num} != table "
            f"{table_sg_num}"))
        report["sg_number_check"] = "failed"
    else:
        report["sg_number_check"] = "passed"

    # ---- C. Certificate presence, contract, and cross-checks ----
    _validate_cprime_bundle_identity(
        bundle,
        blockers,
        report,
        cprime_validation_context=cprime_validation_context,
    )
    cert_id = bundle.get("certificate_identity", {})
    if not isinstance(cert_id, dict) or not cert_id:
        blockers.append(_blocker("certificate_missing",
            "bundle has no certificate_identity"))
        for k in ("certificate_check", "certificate_consistency_check",
                  "cert_sg_consistency_check", "affine_setting_check",
                  "hall_setting_check"):
            report[k] = "blocked"
        cert_id = {}
    else:
        _validate_certificate_status(cert_id, blockers, report)
        _validate_certificate_identity_contract(cert_id, blockers, report)
        _validate_sg_identity_crosscheck(
            cert_id, bundle_sg, bundle_sg_num, table_sg, table_sg_num,
            table_setting, blockers, report)
        _validate_setting(cert_id, table_setting, blockers, report)
        _validate_hall_consistency(
            cert_id, table_hall, table_hall_symbol, table_centering,
            table_setting is not None, blockers, report)

    # ---- D. Spin convention: all evidence must agree and equal table ----
    _validate_spin_convention(bundle, cert_id, table_spinful, blockers, report)

    # ---- E. HSP basis ----
    table_hsps = _normalized_hsp_set(table.get("expected_hsps"))
    irreps_by_kp = bundle.get("irreps_by_kpoint", {})
    actual_hsps = set(irreps_by_kp) if isinstance(irreps_by_kp, dict) else set()
    bundle_expected_raw = bundle.get("expected_hsps")
    if bundle_expected_raw is None:
        bundle_hsps: set[str] | None = actual_hsps
    else:
        bundle_hsps = _normalized_hsp_set(bundle_expected_raw)
    if bundle_hsps is None:
        blockers.append(_blocker("hsp_basis_malformed",
            "bundle expected_hsps is not a unique list of non-empty labels"))
        report["hsp_basis_check"] = "failed"
    elif bundle_hsps != table_hsps or actual_hsps != table_hsps:
        blockers.append(_blocker("hsp_basis_mismatch",
            f"expected_hsps mismatch: table {sorted(table_hsps)}, bundle "
            f"expected {sorted(bundle_hsps)}, actual {sorted(actual_hsps)}"))
        report["hsp_basis_check"] = "failed"
    else:
        report["hsp_basis_check"] = "passed"

    # ---- F. Irrep keys resolve exactly and unambiguously ----
    table_irreps = table.get("irreps", [])
    irrep_vector: list[int] | None = None
    if isinstance(irreps_by_kp, dict) and isinstance(table_irreps, list):
        irrep_vector = _count_irreps(irreps_by_kp, table_irreps)
        if irrep_vector is None:
            blockers.append(_blocker("irrep_key_unresolved",
                "could not resolve irrep keys: bundle irrep keys do not "
                "resolve exactly/unambiguously to table irreps"))
            report["irrep_basis_check"] = "failed"
        else:
            report["irrep_basis_check"] = "passed"
    else:
        blockers.append(_blocker("irrep_basis_malformed",
            "bundle irreps_by_kpoint or table irreps malformed"))
        report["irrep_basis_check"] = "failed"

    table_provenance = _table_provenance_for_output(
        table,
        independent_setting_identity=table_setting,
    )

    promoted = not blockers
    state = "validated_basis" if promoted else "sampled_basis"
    promoted_bundle: dict | None = None
    if promoted:
        promoted_bundle = dict(bundle)
        promoted_bundle["promotion_provenance"] = {
            "source": "promote_bundle_for_solve",
            "validation_report": dict(report),
            "table_provenance": dict(table_provenance),
            "certificate_identity": dict(cert_id),
            "promotion_input_identity": build_promotion_input_identity(bundle),
            "irrep_vector": list(irrep_vector or []),
        }

    return {
        "promoted": promoted,
        "promoted_bundle": promoted_bundle,
        "blocker_reasons": blockers,
        "validation_report": report,
        "canonical_state": state,
        "irrep_vector": irrep_vector if promoted else None,
        "table_provenance": table_provenance,
        "certificate_identity": dict(cert_id),
    }


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _table_provenance_for_output(
    table: dict,
    *,
    independent_setting_identity: dict | None = None,
    source: str | None = None,
) -> dict:
    """Return the lossless reduced-table provenance public contract."""
    provenance = table.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    result = dict(provenance)
    table_irreps = table.get("irreps", [])
    basis_count = len(table_irreps) if isinstance(table_irreps, list) else 0
    filtered_zero_ebrs = provenance.get("filtered_zero_vector_ebrs", [])
    if not isinstance(filtered_zero_ebrs, list):
        filtered_zero_ebrs = []
    dropped_source_rows = provenance.get("dropped_source_rows", [])
    if not isinstance(dropped_source_rows, list):
        dropped_source_rows = []
    result.update({
        "subspace_group_candidate": table.get(
            "subspace_group_candidate", ""
        ),
        "expected_hsps": list(table.get("expected_hsps", [])),
        "source_basis_count": provenance.get(
            "source_basis_count", basis_count
        ),
        "reduction_basis_count": provenance.get(
            "reduction_basis_count", basis_count
        ),
        "filtered_zero_vector_ebr_count": provenance.get(
            "filtered_zero_vector_ebr_count",
            len(filtered_zero_ebrs),
        ),
        "filtered_zero_vector_ebrs": list(filtered_zero_ebrs),
        "dropped_source_row_count": provenance.get(
            "dropped_source_row_count",
            len(dropped_source_rows),
        ),
        "dropped_source_rows": list(dropped_source_rows),
        "independent_setting_identity": (
            dict(independent_setting_identity)
            if isinstance(independent_setting_identity, dict)
            else None
        ),
        "setting_source": (
            independent_setting_identity.get("canonical_setting_source")
            or independent_setting_identity.get("source")
            if isinstance(independent_setting_identity, dict)
            else None
        ),
    })
    if source is not None:
        result["source"] = source
    return result


def _merge_table_input_provenance(
    table_provenance: dict,
    reduced_ebr_input: dict[str, object] | None,
) -> dict:
    return merge_table_input_provenance(
        table_provenance,
        reduced_ebr_input,
    )


def _normalized_hsp_set(value: object) -> set[str] | None:
    if not isinstance(value, list):
        return set() if value is None else None
    if not all(isinstance(h, str) and h for h in value):
        return None
    labels = list(value)
    if len(set(labels)) != len(labels):
        return None
    return set(labels)


def _validate_problem_kind_compatibility(
    *,
    bundle: dict,
    provenance: dict,
    table_space_group_number: object,
    table_spinful: object,
    blockers: list[dict[str, str]],
    report: dict[str, object],
) -> None:
    """Keep unitary and type-II grey reduced problems physically distinct."""
    unitary_kind = "unitary_valley_reduced_ebr"
    orbit_kind = "valley_orbit_reduced_ebr"
    problem_kind = bundle.get("problem_kind")
    physical_object_kind = bundle.get("physical_object_kind")
    grey_source = provenance.get("time_reversal_source")
    grey_bns_number = provenance.get("time_reversal_grey_bns_number")
    grey_unitary_sg = provenance.get("unitary_space_group_number")

    expected_physical_object_kind = {
        unitary_kind: "unitary_valley_projected_subspace",
        orbit_kind: "joint_time_reversal_valley_orbit",
    }.get(problem_kind)
    if (
        expected_physical_object_kind is None
        or physical_object_kind != expected_physical_object_kind
    ):
        blockers.append(_blocker(
            "problem_physical_object_kind_mismatch",
            f"problem_kind {problem_kind!r} is inconsistent with "
            f"physical_object_kind {physical_object_kind!r}",
        ))
        report["problem_kind_check"] = "failed"
        return

    if problem_kind == unitary_kind:
        if grey_source is not None or grey_bns_number is not None:
            blockers.append(_blocker(
                "unitary_problem_rejects_grey_table",
                "unitary valley problems cannot be promoted with type-II "
                "grey-group table provenance",
            ))
            report["problem_kind_check"] = "failed"
        elif unitary_bundle_claims_time_reversal_completion(
            bundle
        ) and not validate_tr_completed_unitary_bundle(bundle):
            blockers.append(_blocker(
                "unitary_completion_provenance_invalid",
                "TR-completed unitary valley problem lacks complete, trusted "
                "row-level observed/inferred provenance",
            ))
            report["problem_kind_check"] = "failed"
            report["completion_provenance_check"] = "failed"
        elif (
            not unitary_bundle_claims_time_reversal_completion(bundle)
            and not validate_direct_unitary_bundle(bundle)
        ):
            blockers.append(_blocker(
                "unitary_construction_provenance_invalid",
                "direct unitary valley problem lacks complete observed-row "
                "construction provenance",
            ))
            report["problem_kind_check"] = "failed"
            report["completion_provenance_check"] = "failed"
        else:
            report["problem_kind_check"] = "passed"
            report["completion_provenance_check"] = "passed"
        return

    if problem_kind != orbit_kind:
        blockers.append(_blocker(
            "problem_kind_invalid",
            f"unsupported reduced EBR problem_kind {problem_kind!r}",
        ))
        report["problem_kind_check"] = "failed"
        return

    table_provenance_complete = (
        grey_source == "irreptables_type_ii_grey_group"
        and isinstance(grey_bns_number, str)
        and bool(grey_bns_number)
        and _is_positive_int(grey_unitary_sg)
    )
    if not table_provenance_complete:
        blockers.append(_blocker(
            "time_reversal_table_provenance_missing",
            "valley-orbit problems require irreptables type-II grey-group "
            "source, BNS number, and unitary space-group provenance",
        ))

    expected_bns_number: str | None = None
    if _is_positive_int(table_space_group_number):
        try:
            expected_bns_number = derive_type_ii_bns_number(
                int(table_space_group_number)
            )
        except Exception as exc:
            blockers.append(_blocker(
                "time_reversal_grey_bns_unresolved",
                "could not independently derive the type-II BNS number: "
                f"{type(exc).__name__}: {exc}",
            ))
    if (
        expected_bns_number is not None
        and grey_bns_number != expected_bns_number
    ):
        blockers.append(_blocker(
            "time_reversal_grey_bns_mismatch",
            f"table grey BNS {grey_bns_number!r} does not match the "
            f"independently derived type-II group {expected_bns_number!r}",
        ))
    if (
        _is_positive_int(grey_unitary_sg)
        and _is_positive_int(table_space_group_number)
        and int(grey_unitary_sg) != int(table_space_group_number)
    ):
        blockers.append(_blocker(
            "time_reversal_unitary_sg_mismatch",
            f"grey table unitary SG {grey_unitary_sg!r} does not match "
            f"table SG {table_space_group_number!r}",
        ))

    if not validate_joint_grey_bundle_provenance(
        bundle=bundle,
        table_provenance=provenance,
    ):
        blockers.append(_blocker(
            "time_reversal_bundle_evidence_invalid",
            "valley-orbit bundle lacks a complete valley "
            "involution, source-irrep pairing, HSP orbits, or matching grey "
            "BNS evidence",
        ))
        report["completion_provenance_check"] = "failed"
    else:
        report["completion_provenance_check"] = "passed"

    report["problem_kind_check"] = (
        "passed"
        if not any(
            blocker["code"].startswith("time_reversal_")
            for blocker in blockers
        )
        else "failed"
    )


def validate_joint_grey_bundle_provenance(
    bundle: dict,
    table_provenance: dict,
) -> bool:
    """Validate a serialized joint type-II-grey bundle/table pair."""
    if not isinstance(bundle, dict) or not isinstance(table_provenance, dict):
        return False
    table_spinful = table_provenance.get("spinful")
    table_space_group_number = table_provenance.get("space_group_number")
    grey_unitary_sg = table_provenance.get("unitary_space_group_number")
    grey_bns_number = table_provenance.get(
        "time_reversal_grey_bns_number"
    )
    if (
        table_provenance.get("time_reversal_source")
        != "irreptables_type_ii_grey_group"
        or not isinstance(table_spinful, bool)
        or not _is_positive_int(table_space_group_number)
        or not _is_positive_int(grey_unitary_sg)
        or int(grey_unitary_sg) != int(table_space_group_number)
        or not isinstance(grey_bns_number, str)
        or not grey_bns_number
    ):
        return False
    try:
        expected_bns_number = derive_type_ii_bns_number(
            int(table_space_group_number)
        )
    except Exception:
        return False
    if grey_bns_number != expected_bns_number:
        return False
    return _joint_bundle_time_reversal_evidence_valid(
        bundle=bundle,
        table_spinful=table_spinful,
        expected_bns_number=expected_bns_number,
        table_space_group_number=table_space_group_number,
    )


def _joint_bundle_time_reversal_evidence_valid(
    *,
    bundle: dict,
    table_spinful: object,
    expected_bns_number: str | None,
    table_space_group_number: object = None,
    reviewed_source_model: dict[str, object] | None = None,
) -> bool:
    if bundle.get("valley") not in (None, ""):
        return False
    valley_orbit = bundle.get("valley_orbit")
    if (
        not isinstance(valley_orbit, list)
        or len(valley_orbit) not in (1, 2)
        or any(not isinstance(item, str) or not item for item in valley_orbit)
        or len(set(valley_orbit)) != len(valley_orbit)
    ):
        return False
    orbit_members = set(valley_orbit)

    unitary_irreps = bundle.get("unitary_valley_irreps")
    if not isinstance(unitary_irreps, dict) or set(unitary_irreps) != orbit_members:
        return False
    unitary_irrep_labels: set[str] = set()
    component_hsp_sets: list[set[str]] = []
    for component in unitary_irreps.values():
        if not isinstance(component, dict) or not component:
            return False
        component_hsp_sets.append(set(component))
        for hsp, counts in component.items():
            if (
                not isinstance(hsp, str)
                or not hsp
                or not isinstance(counts, dict)
                or not counts
                or any(
                    not isinstance(label, str)
                    or not label
                    or not isinstance(multiplicity, int)
                    or isinstance(multiplicity, bool)
                    or multiplicity <= 0
                    for label, multiplicity in counts.items()
                )
            ):
                return False
            unitary_irrep_labels.update(counts)

    evidence = bundle.get("time_reversal")
    if not isinstance(evidence, dict):
        return False
    if bundle.get("physical_object_kind") == (
        "joint_time_reversal_valley_orbit"
    ):
        completion_by_valley = evidence.get(
            "unitary_valley_irrep_completion_records"
        )
        sampled_by_valley = evidence.get(
            "source_hsp_to_sampled_kpoint_by_valley"
        )
        observed_sampled_by_valley = evidence.get(
            "observed_source_hsp_to_sampled_kpoint_by_valley"
        )
        if (
            not isinstance(completion_by_valley, dict)
            or not isinstance(sampled_by_valley, dict)
            or not isinstance(observed_sampled_by_valley, dict)
            or set(completion_by_valley) != orbit_members
            or set(sampled_by_valley) != orbit_members
            or set(observed_sampled_by_valley) != orbit_members
        ):
            return False
    expected_theta_square = -1 if table_spinful is True else 1
    if evidence.get("theta_square") != expected_theta_square:
        return False

    valley_mapping = evidence.get("time_reversal_valley_mapping")
    if not isinstance(valley_mapping, dict):
        return False
    if (
        set(valley_mapping) != orbit_members
        or set(valley_mapping.values()) != orbit_members
        or any(
            valley_mapping.get(valley_mapping.get(valley, "")) != valley
            for valley in valley_orbit
        )
        or (
            len(valley_orbit) == 1
            and valley_mapping.get(valley_orbit[0]) != valley_orbit[0]
        )
        or (
            len(valley_orbit) == 2
            and any(valley_mapping.get(valley) == valley for valley in valley_orbit)
        )
    ):
        return False

    hsp_orbits = evidence.get("time_reversal_hsp_orbits")
    if (
        not isinstance(hsp_orbits, list)
        or not hsp_orbits
        or any(not isinstance(row, dict) or not row for row in hsp_orbits)
    ):
        return False
    declared_hsps: set[str] = set()
    representative_hsps: set[str] = set()
    hsp_mapping: dict[str, str] = {}
    for row in hsp_orbits:
        members = row.get("members")
        representative = row.get("representative")
        self_mapped = row.get("self_mapped")
        if (
            not isinstance(members, list)
            or len(members) not in (1, 2)
            or any(not isinstance(hsp, str) or not hsp for hsp in members)
            or len(set(members)) != len(members)
            or representative not in members
            or not isinstance(self_mapped, bool)
            or self_mapped != (len(members) == 1)
            or declared_hsps.intersection(members)
        ):
            return False
        declared_hsps.update(members)
        representative_hsps.add(representative)
        if len(members) == 1:
            hsp_mapping[members[0]] = members[0]
        else:
            hsp_mapping[members[0]] = members[1]
            hsp_mapping[members[1]] = members[0]
    full_hsp_labels = evidence.get("full_unitary_source_hsp_labels")
    if (
        not isinstance(full_hsp_labels, list)
        or any(not isinstance(hsp, str) or not hsp for hsp in full_hsp_labels)
        or len(set(full_hsp_labels)) != len(full_hsp_labels)
        or declared_hsps != set(full_hsp_labels)
    ):
        return False
    expected_hsps = bundle.get("expected_hsps")
    irreps_by_kpoint = bundle.get("irreps_by_kpoint")
    if (
        not isinstance(expected_hsps, list)
        or not expected_hsps
        or any(not isinstance(hsp, str) or not hsp for hsp in expected_hsps)
        or len(set(expected_hsps)) != len(expected_hsps)
        or set(expected_hsps) != representative_hsps
        or not isinstance(irreps_by_kpoint, dict)
        or set(irreps_by_kpoint) != set(expected_hsps)
        or any(
            not set(expected_hsps).issubset(component_hsps)
            or not component_hsps.issubset(declared_hsps)
            for component_hsps in component_hsp_sets
        )
    ):
        return False

    irrep_pairing = evidence.get("time_reversal_irrep_pairing")
    if not isinstance(irrep_pairing, dict) or not irrep_pairing:
        return False
    if (
        set(irrep_pairing) != set(irrep_pairing.values())
        or any(
        not isinstance(label, str)
        or not label
        or not isinstance(partner, str)
        or not partner
        or irrep_pairing.get(partner) != label
        for label, partner in irrep_pairing.items()
        )
    ):
        return False
    if not unitary_irrep_labels.issubset(irrep_pairing):
        return False
    if not _unitary_components_match_time_reversal(
        unitary_irreps=unitary_irreps,
        valley_mapping=valley_mapping,
        hsp_mapping=hsp_mapping,
        irrep_pairing=irrep_pairing,
    ):
        return False
    if bundle.get("physical_object_kind") == (
        "joint_time_reversal_valley_orbit"
    ) and any(
        not _unitary_completion_records_valid(
            valley=valley,
            counts_by_hsp=unitary_irreps[valley],
            records_by_hsp=completion_by_valley[valley],
            observed_source_to_sampled=(
                observed_sampled_by_valley[valley]
            ),
            valley_mapping=valley_mapping,
            hsp_mapping=hsp_mapping,
            irrep_mapping=irrep_pairing,
            independent_hsps=representative_hsps,
            expected_spinor=table_spinful,
        )
        for valley in valley_orbit
    ):
        return False

    source_to_sampled = bundle.get("source_hsp_to_sampled_kpoint")
    representative_valley = evidence.get("representative_valley")
    source_to_sampled_by_valley = evidence.get(
        "source_hsp_to_sampled_kpoint_by_valley"
    )
    if (
        not isinstance(source_to_sampled, dict)
        or not isinstance(representative_valley, str)
        or representative_valley not in orbit_members
        or not _source_hsp_sampled_mappings_valid(
            mappings=source_to_sampled_by_valley,
            valley_orbit=valley_orbit,
            expected_hsps=expected_hsps,
        )
        or source_to_sampled_by_valley.get(representative_valley)
        != source_to_sampled
    ):
        return False

    if len(valley_orbit) == 1:
        source_model = reviewed_source_model
        if source_model is None:
            source_model = _derive_reviewed_source_validation_model(
                space_group_number=table_space_group_number,
                spinful=table_spinful,
                source_irrep_labels=unitary_irrep_labels,
            )
        if (
            not isinstance(source_to_sampled, dict)
            or set(source_to_sampled) != set(expected_hsps)
            or any(
                not isinstance(sampled, str) or not sampled
                for sampled in source_to_sampled.values()
            )
            or len(set(source_to_sampled.values())) != len(source_to_sampled)
            or not _source_hsp_bindings_valid(
                bindings=evidence.get(
                    "source_hsp_binding_by_sampled_kpoint"
                ),
                source_to_sampled=source_to_sampled,
                valley_orbit=valley_orbit,
                full_hsp_labels=set(full_hsp_labels),
                hsp_mapping=hsp_mapping,
                valley_mapping=valley_mapping,
                unitary_irreps=unitary_irreps,
                sewing_evidence=evidence.get(
                    "antiunitary_sewing_evidence"
                ),
                reviewed_source_model=source_model,
            )
            or not validate_time_reversal_sewing_report(
                evidence.get("antiunitary_sewing_evidence"),
                valley_members=valley_orbit,
                theta_square=expected_theta_square,
                required_kpoints=[
                    source_to_sampled[hsp] for hsp in expected_hsps
                ],
                required_projector_workflows=evidence.get(
                    "projector_workflow_by_sampled_kpoint"
                ),
                required_projector_provenance=evidence.get(
                    "projector_provenance_by_sampled_kpoint"
                ),
            )
        ):
            return False

    grey_bns_number = evidence.get("grey_bns_number")
    return (
        expected_bns_number is not None
        and grey_bns_number == expected_bns_number
    )


def _unitary_completion_records_valid(
    *,
    valley: str,
    counts_by_hsp: object,
    records_by_hsp: object,
    observed_source_to_sampled: object,
    valley_mapping: object = None,
    hsp_mapping: object = None,
    irrep_mapping: object = None,
    independent_hsps: set[str] | None = None,
    expected_spinor: object = None,
) -> bool:
    if (
        not isinstance(counts_by_hsp, dict)
        or not isinstance(records_by_hsp, dict)
        or not isinstance(observed_source_to_sampled, dict)
        or set(counts_by_hsp) != set(records_by_hsp)
    ):
        return False
    observed_hsps: set[str] = set()
    rebuilt: dict[str, dict[str, int]] = {}
    for hsp, records in records_by_hsp.items():
        if not isinstance(hsp, str) or not hsp or not isinstance(records, list):
            return False
        target: dict[str, int] = {}
        for record in records:
            if not isinstance(record, dict):
                return False
            irrep = record.get("irrep")
            multiplicity = record.get("multiplicity")
            kind = record.get("completion_kind")
            identity = record.get("source_candidate_identity")
            candidate_provenance = record.get(
                "source_candidate_provenance"
            )
            if (
                record.get("target_valley") != valley
                or record.get("target_source_hsp_label") != hsp
                or not isinstance(irrep, str)
                or not irrep
                or not isinstance(multiplicity, int)
                or isinstance(multiplicity, bool)
                or multiplicity <= 0
                or record.get("structural_status") != "validated"
                or record.get("readiness_status") != "trusted"
                or record.get("blockers") not in ([], None)
                or not isinstance(identity, dict)
                or not identity
                or not isinstance(candidate_provenance, dict)
                or not candidate_provenance
                or not _source_candidate_provenance_valid(
                    identity=identity,
                    provenance=candidate_provenance,
                    expected_valley=(
                        valley if kind == "observed_at_sampled_kpoint"
                        else record.get("evidence_valley")
                    ),
                    expected_hsp=(
                        hsp if kind == "observed_at_sampled_kpoint"
                        else record.get("evidence_source_hsp_label")
                    ),
                    expected_sampled=(
                        record.get("sampled_kpoint")
                        if kind == "observed_at_sampled_kpoint"
                        else record.get("evidence_sampled_kpoint")
                    ),
                    expected_irrep=(
                        irrep if kind == "observed_at_sampled_kpoint"
                        else (
                            record.get("reviewed_time_reversal_relation", {})
                            .get("evidence_irrep")
                            if isinstance(
                                record.get("reviewed_time_reversal_relation"),
                                dict,
                            )
                            else None
                        )
                    ),
                    expected_multiplicity=multiplicity,
                    expected_spinor=expected_spinor,
                )
            ):
                return False
            if kind == "observed_at_sampled_kpoint":
                sampled = record.get("sampled_kpoint")
                if (
                    not isinstance(sampled, str)
                    or not sampled
                    or observed_source_to_sampled.get(hsp) != sampled
                    or identity.get("valley") != valley
                    or identity.get("source_hsp_label") != hsp
                    or identity.get("sampled_kpoint") != sampled
                    or identity.get("irrep") != irrep
                    or identity.get("multiplicity") != multiplicity
                ):
                    return False
                observed_hsps.add(hsp)
                consistency = record.get("time_reversal_consistency")
                if (
                    independent_hsps is not None
                    and hsp not in independent_hsps
                    and not _observed_time_reversal_consistency_valid(
                        consistency=consistency,
                        valley=valley,
                        hsp=hsp,
                        irrep=irrep,
                        multiplicity=multiplicity,
                        valley_mapping=valley_mapping,
                        hsp_mapping=hsp_mapping,
                        irrep_mapping=irrep_mapping,
                        expected_spinor=expected_spinor,
                    )
                ):
                    return False
            elif kind == "inferred_by_time_reversal":
                relation = record.get("reviewed_time_reversal_relation")
                if (
                    "sampled_kpoint" in record
                    or not isinstance(record.get("evidence_valley"), str)
                    or not record.get("evidence_valley")
                    or not isinstance(
                        record.get("evidence_source_hsp_label"), str
                    )
                    or not record.get("evidence_source_hsp_label")
                    or not isinstance(
                        record.get("evidence_sampled_kpoint"), str
                    )
                    or not record.get("evidence_sampled_kpoint")
                    or not isinstance(relation, dict)
                    or relation.get("target_valley") != valley
                    or relation.get("target_source_hsp_label") != hsp
                    or relation.get("target_irrep") != irrep
                    or relation.get("evidence_valley") != record.get(
                        "evidence_valley"
                    )
                    or relation.get("evidence_source_hsp_label") != (
                        record.get("evidence_source_hsp_label")
                    )
                    or identity.get("valley") != record.get(
                        "evidence_valley"
                    )
                    or identity.get("source_hsp_label") != record.get(
                        "evidence_source_hsp_label"
                    )
                    or identity.get("sampled_kpoint") != record.get(
                        "evidence_sampled_kpoint"
                    )
                    or identity.get("irrep") != relation.get(
                        "evidence_irrep"
                    )
                    or identity.get("multiplicity") != multiplicity
                    or not _reviewed_time_reversal_relation_valid(
                        relation=relation,
                        evidence_valley=record.get("evidence_valley"),
                        target_valley=valley,
                        evidence_hsp=record.get(
                            "evidence_source_hsp_label"
                        ),
                        target_hsp=hsp,
                        evidence_irrep=relation.get("evidence_irrep"),
                        target_irrep=irrep,
                        valley_mapping=valley_mapping,
                        hsp_mapping=hsp_mapping,
                        irrep_mapping=irrep_mapping,
                    )
                ):
                    return False
            else:
                return False
            target[irrep] = target.get(irrep, 0) + multiplicity
        rebuilt[hsp] = target
    return (
        rebuilt == counts_by_hsp
        and set(observed_source_to_sampled) == observed_hsps
    )


def _nonempty_involutive_string_mapping(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(key, str)
            and bool(key)
            and isinstance(partner, str)
            and bool(partner)
            and value.get(partner) == key
            for key, partner in value.items()
        )
        and set(value) == set(value.values())
    )


def _reviewed_hsp_involution(
    rows: object,
) -> tuple[dict[str, str] | None, set[str]]:
    if not isinstance(rows, list) or not rows:
        return None, set()
    mapping: dict[str, str] = {}
    representatives: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            return None, set()
        members = row.get("members")
        representative = row.get("representative")
        self_mapped = row.get("self_mapped")
        if (
            not isinstance(members, list)
            or len(members) not in (1, 2)
            or any(not isinstance(item, str) or not item for item in members)
            or len(set(members)) != len(members)
            or representative not in members
            or self_mapped != (len(members) == 1)
            or set(members).intersection(mapping)
        ):
            return None, set()
        representatives.add(str(representative))
        if len(members) == 1:
            mapping[members[0]] = members[0]
        else:
            mapping[members[0]] = members[1]
            mapping[members[1]] = members[0]
    return mapping, representatives


def _source_candidate_provenance_valid(
    *,
    identity: dict,
    provenance: dict,
    expected_valley: object,
    expected_hsp: object,
    expected_sampled: object,
    expected_irrep: object,
    expected_multiplicity: object,
    expected_spinor: object,
) -> bool:
    source = identity.get("source")
    source_irrep = provenance.get("irrep_source_provenance")
    return (
        isinstance(source, str)
        and bool(source)
        and provenance.get("source") == source
        and isinstance(provenance.get("workflow_path"), str)
        and bool(provenance.get("workflow_path"))
        and identity.get("valley") == expected_valley
        and identity.get("source_hsp_label") == expected_hsp
        and identity.get("sampled_kpoint") == expected_sampled
        and identity.get("irrep") == expected_irrep
        and identity.get("multiplicity") == expected_multiplicity
        and isinstance(source_irrep, dict)
        and source_irrep.get("source_hsp_label") == expected_hsp
        and isinstance(source_irrep.get("source_table_spinor"), bool)
        and isinstance(expected_spinor, bool)
        and source_irrep.get("source_table_spinor") == expected_spinor
    )


def _reviewed_time_reversal_relation_valid(
    *,
    relation: object,
    evidence_valley: object,
    target_valley: object,
    evidence_hsp: object,
    target_hsp: object,
    evidence_irrep: object,
    target_irrep: object,
    valley_mapping: object,
    hsp_mapping: object,
    irrep_mapping: object,
) -> bool:
    return (
        isinstance(relation, dict)
        and isinstance(valley_mapping, dict)
        and isinstance(hsp_mapping, dict)
        and isinstance(irrep_mapping, dict)
        and relation.get("evidence_valley") == evidence_valley
        and relation.get("target_valley") == target_valley
        and relation.get("evidence_source_hsp_label") == evidence_hsp
        and relation.get("target_source_hsp_label") == target_hsp
        and relation.get("evidence_irrep") == evidence_irrep
        and relation.get("target_irrep") == target_irrep
        and valley_mapping.get(evidence_valley) == target_valley
        and hsp_mapping.get(evidence_hsp) == target_hsp
        and irrep_mapping.get(evidence_irrep) == target_irrep
    )


def _observed_time_reversal_consistency_valid(
    *,
    consistency: object,
    valley: str,
    hsp: str,
    irrep: str,
    multiplicity: int,
    valley_mapping: object,
    hsp_mapping: object,
    irrep_mapping: object,
    expected_spinor: object,
) -> bool:
    if not isinstance(consistency, dict) or consistency.get("status") != (
        "validated"
    ):
        return False
    relation = consistency.get("reviewed_time_reversal_relation")
    evidence_irrep = (
        relation.get("evidence_irrep") if isinstance(relation, dict) else None
    )
    return (
        isinstance(consistency.get("evidence_sampled_kpoint"), str)
        and bool(consistency.get("evidence_sampled_kpoint"))
        and _reviewed_time_reversal_relation_valid(
            relation=relation,
            evidence_valley=consistency.get("evidence_valley"),
            target_valley=valley,
            evidence_hsp=consistency.get("evidence_source_hsp_label"),
            target_hsp=hsp,
            evidence_irrep=evidence_irrep,
            target_irrep=irrep,
            valley_mapping=valley_mapping,
            hsp_mapping=hsp_mapping,
            irrep_mapping=irrep_mapping,
        )
        and _source_candidate_provenance_valid(
            identity=consistency.get("source_candidate_identity", {}),
            provenance=consistency.get("source_candidate_provenance", {}),
            expected_valley=consistency.get("evidence_valley"),
            expected_hsp=consistency.get("evidence_source_hsp_label"),
            expected_sampled=consistency.get("evidence_sampled_kpoint"),
            expected_irrep=evidence_irrep,
            expected_multiplicity=multiplicity,
            expected_spinor=expected_spinor,
        )
    )


def _source_hsp_sampled_mappings_valid(
    *,
    mappings: object,
    valley_orbit: list[str],
    expected_hsps: list[str],
) -> bool:
    if not isinstance(mappings, dict) or set(mappings) != set(valley_orbit):
        return False
    expected = set(expected_hsps)
    for valley in valley_orbit:
        by_source = mappings.get(valley)
        if (
            not isinstance(by_source, dict)
            or set(by_source) != expected
            or any(
                not isinstance(sampled, str) or not sampled
                for sampled in by_source.values()
            )
            or len(set(by_source.values())) != len(by_source)
        ):
            return False
    return True


def _unitary_components_match_time_reversal(
    *,
    unitary_irreps: dict[str, object],
    valley_mapping: dict[str, object],
    hsp_mapping: dict[str, str],
    irrep_pairing: dict[str, object],
) -> bool:
    for valley, raw_component in unitary_irreps.items():
        if not isinstance(raw_component, dict):
            return False
        target_valley = valley_mapping.get(valley)
        target_component = unitary_irreps.get(target_valley)
        if not isinstance(target_component, dict):
            return False
        for hsp, raw_counts in raw_component.items():
            if not isinstance(raw_counts, dict) or hsp not in hsp_mapping:
                return False
            inferred: dict[str, int] = {}
            for irrep, multiplicity in raw_counts.items():
                partner_irrep = irrep_pairing.get(irrep)
                if not isinstance(partner_irrep, str):
                    return False
                inferred[partner_irrep] = (
                    inferred.get(partner_irrep, 0) + multiplicity
                )
            actual_target = target_component.get(hsp_mapping[hsp])
            if actual_target is not None and actual_target != inferred:
                return False
    return True


def _source_hsp_bindings_valid(
    *,
    bindings: object,
    source_to_sampled: dict[str, object],
    valley_orbit: list[str],
    full_hsp_labels: set[str],
    hsp_mapping: dict[str, str],
    valley_mapping: dict[str, object],
    unitary_irreps: dict[str, object],
    sewing_evidence: object,
    reviewed_source_model: object,
) -> bool:
    if not isinstance(bindings, dict) or not isinstance(
        sewing_evidence, dict
    ):
        return False
    kpoint_mapping = sewing_evidence.get("time_reversal_kpoint_mapping")
    sampled_kpoints = sewing_evidence.get("sampled_kpoint_frac_by_name")
    if not isinstance(kpoint_mapping, dict) or not isinstance(
        sampled_kpoints, dict
    ):
        return False
    if not isinstance(reviewed_source_model, dict):
        return False
    trusted_coordinates = reviewed_source_model.get(
        "source_hsp_representative_k_frac_by_label"
    )
    trusted_rotations = reviewed_source_model.get(
        "standard_operation_rotation_frac_by_index"
    )
    centering_vectors = normalize_centering_vectors(
        reviewed_source_model.get("normalized_centering_vectors", [])
    )
    if (
        not isinstance(trusted_coordinates, dict)
        or not isinstance(trusted_rotations, dict)
        or centering_vectors is None
    ):
        return False
    expected_source_by_sampled = {
        sampled: source_hsp for source_hsp, sampled in source_to_sampled.items()
    }
    required_samples = set(expected_source_by_sampled)
    scope = set(required_samples)
    for source_hsp, sampled in source_to_sampled.items():
        partner = kpoint_mapping.get(sampled)
        if not isinstance(partner, str) or not partner:
            return False
        scope.add(partner)
        partner_hsp = hsp_mapping.get(source_hsp)
        if not isinstance(partner_hsp, str) or not partner_hsp:
            return False
        for valley in valley_orbit:
            partner_valley = valley_mapping.get(valley)
            partner_component = unitary_irreps.get(partner_valley)
            if not isinstance(partner_component, dict):
                return False
            if partner_hsp in partner_component:
                previous = expected_source_by_sampled.setdefault(
                    partner, partner_hsp
                )
                if previous != partner_hsp:
                    return False
    required_samples = set(expected_source_by_sampled)
    if set(bindings) != required_samples or not required_samples.issubset(scope):
        return False
    binding_keys = {
        "source_hsp_label",
        "classification",
        "validation_status",
        "parent_k_frac",
        "standard_k_frac",
        "source_hsp_representative_k_frac",
        "standard_operation_index",
    }
    for sampled, raw_by_valley in bindings.items():
        if not isinstance(raw_by_valley, dict) or set(raw_by_valley) != set(
            valley_orbit
        ):
            return False
        sampled_vector = _finite_vector3(sampled_kpoints.get(sampled))
        if sampled_vector is None:
            return False
        for raw_binding in raw_by_valley.values():
            if (
                not isinstance(raw_binding, dict)
                or set(raw_binding) != binding_keys
            ):
                return False
            source_hsp = raw_binding.get("source_hsp_label")
            parent_k = _finite_vector3(raw_binding.get("parent_k_frac"))
            standard_k = _finite_vector3(raw_binding.get("standard_k_frac"))
            representative_k = _finite_vector3(
                raw_binding.get("source_hsp_representative_k_frac")
            )
            classification = raw_binding.get("classification")
            operation_index = raw_binding.get("standard_operation_index")
            trusted_representative = _finite_vector3(
                trusted_coordinates.get(source_hsp)
                if isinstance(source_hsp, str) else None
            )
            if (
                not isinstance(source_hsp, str)
                or source_hsp not in full_hsp_labels
                or classification not in ("representative", "star_equivalent")
                or raw_binding.get("validation_status") != "validated"
                or parent_k is None
                or standard_k is None
                or representative_k is None
                or trusted_representative is None
                or np.linalg.norm(
                    parent_k - sampled_vector
                    - np.rint(parent_k - sampled_vector)
                ) > 5e-6
                or not centered_k_equivalent(
                    representative_k,
                    trusted_representative,
                    centering_vectors,
                    tolerance=5e-6,
                )
                or source_hsp != expected_source_by_sampled[sampled]
            ):
                return False
            if classification == "representative":
                if operation_index is not None or not centered_k_equivalent(
                    standard_k,
                    trusted_representative,
                    centering_vectors,
                    tolerance=5e-6,
                ):
                    return False
                continue
            if (
                not isinstance(operation_index, int)
                or isinstance(operation_index, bool)
                or operation_index <= 0
            ):
                return False
            rotation = _integer_matrix3(trusted_rotations.get(operation_index))
            if rotation is None:
                return False
            try:
                arm = np.linalg.inv(rotation).T @ trusted_representative
            except np.linalg.LinAlgError:
                return False
            if centered_k_equivalent(
                standard_k,
                trusted_representative,
                centering_vectors,
                tolerance=5e-6,
            ) or not centered_k_equivalent(
                standard_k,
                arm,
                centering_vectors,
                tolerance=5e-6,
            ):
                return False
    return True


@functools.lru_cache(maxsize=None)
def _derive_reviewed_source_validation_model_cached(
    space_group_number: int,
    spinful: bool,
    source_irrep_labels: tuple[str, ...],
) -> dict[str, object] | None:
    """Load independent reviewed HSP coordinates and standard operations."""
    try:
        from valleyscope.irreps.tables import (
            load_standard_irrep_table,
            resolve_ebr_source_irrep_label_evidence,
        )

        table = load_standard_irrep_table(space_group_number, spinor=spinful)
        evidence = resolve_ebr_source_irrep_label_evidence(
            table=table,
            source_basis_labels=list(source_irrep_labels),
        )
        setting = _derive_table_standard_setting(space_group_number, spinful)
    except Exception:
        return None
    reviewed_rows = evidence.get("reviewed_rows")
    if (
        evidence.get("status") != "validated"
        or not isinstance(reviewed_rows, list)
        or len(reviewed_rows) != len(source_irrep_labels)
        or setting is None
    ):
        return None
    coordinates: dict[str, list[float]] = {}
    for row in reviewed_rows:
        label = getattr(row, "label", None)
        hsp = getattr(row, "kpoint_label", None)
        coordinate = _finite_vector3(getattr(row, "k_frac", None))
        if (
            not isinstance(label, str)
            or label not in source_irrep_labels
            or not isinstance(hsp, str)
            or not hsp
            or coordinate is None
        ):
            return None
        previous = coordinates.setdefault(hsp, coordinate.tolist())
        if np.linalg.norm(np.asarray(previous) - coordinate) > 5e-6:
            return None
    rotations = {
        int(operation.table_index): operation.rotation_frac.astype(int).tolist()
        for operation in table.operations
    }
    centering = setting.get("centering_cosets")
    if normalize_centering_vectors(centering) is None:
        return None
    return {
        "source_hsp_representative_k_frac_by_label": coordinates,
        "standard_operation_rotation_frac_by_index": rotations,
        "normalized_centering_vectors": [list(vector) for vector in centering],
    }


def _derive_reviewed_source_validation_model(
    *,
    space_group_number: object,
    spinful: object,
    source_irrep_labels: set[str],
) -> dict[str, object] | None:
    if (
        not _is_positive_int(space_group_number)
        or not isinstance(spinful, bool)
        or not source_irrep_labels
        or any(not isinstance(label, str) or not label for label in source_irrep_labels)
    ):
        return None
    return _derive_reviewed_source_validation_model_cached(
        int(space_group_number),
        spinful,
        tuple(sorted(source_irrep_labels)),
    )


def _finite_vector3(value: object) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        return None
    return vector


def _integer_matrix3(value: object) -> np.ndarray | None:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if (
        matrix.shape != (3, 3)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, np.rint(matrix), atol=1e-8, rtol=0.0)
        or abs(round(float(np.linalg.det(matrix)))) != 1
    ):
        return None
    return np.rint(matrix).astype(int)


_RECOGNIZED_CENTERINGS = frozenset({"P", "A", "B", "C", "I", "F", "R"})


def _norm_symbol(value: object) -> str:
    """Whitespace-insensitive normalized international symbol."""
    return "".join(str(value or "").split())


def _validate_cprime_bundle_identity(
    bundle,
    blockers,
    report,
    *,
    cprime_validation_context,
):
    """Require producer-linked C-prime identities for every sampled HSP."""
    expected_keys = {
        "spinor_source_basis_certificate_identity",
        "double_space_group_lift_certificate_identity",
        "scoped_representation_evidence_identity",
    }
    by_kpoint = bundle.get("cprime_identity_by_kpoint")
    irreps_by_kpoint = bundle.get("irreps_by_kpoint")
    construction = bundle.get("unitary_vector_construction")
    tr_completed = (
        isinstance(construction, dict)
        and construction.get("kind")
        == "time_reversal_completed_unitary_rows"
    )
    joint_problem = (
        bundle.get("problem_kind") == "valley_orbit_reduced_ebr"
    )
    records_by_kpoint = bundle.get(
        "unitary_irrep_completion_records_by_hsp"
        if tr_completed
        else "irrep_records_by_kpoint"
    )
    ok = True
    if (
        not isinstance(by_kpoint, dict)
        or not isinstance(irreps_by_kpoint, dict)
        or set(by_kpoint) != set(irreps_by_kpoint)
        or (
            not joint_problem
            and (
                not isinstance(records_by_kpoint, dict)
                or set(records_by_kpoint) != set(irreps_by_kpoint)
            )
        )
    ):
        blockers.append(
            _blocker(
                "cprime_identity_scope_mismatch",
                "C-prime identity inventory must exactly match sampled HSPs",
            )
        )
        report["cprime_identity_check"] = "failed"
        return
    for kpoint, identity in by_kpoint.items():
        if (
            not isinstance(identity, dict)
            or set(identity) != expected_keys
            or not all(
                valid_sha256_identity(identity.get(key))
                for key in expected_keys
            )
        ):
            blockers.append(
                _blocker(
                    "cprime_identity_malformed",
                    f"malformed C-prime identities for {kpoint}",
                )
            )
            ok = False
            continue
        context_entry = (
            cprime_validation_context.get(
                identity["scoped_representation_evidence_identity"]
            )
            if isinstance(cprime_validation_context, dict)
            else None
        )
        if not isinstance(context_entry, dict):
            blockers.append(
                _blocker(
                    "cprime_producer_context_missing",
                    f"producer validation context missing for {kpoint}",
                )
            )
            ok = False
            continue
        scoped_record = context_entry.get("record")
        raw_inputs = context_entry.get("raw_inputs")
        if not isinstance(scoped_record, dict) or not isinstance(
            raw_inputs, dict
        ):
            blockers.append(
                _blocker(
                    "cprime_producer_context_malformed",
                    f"producer validation context malformed for {kpoint}",
                )
            )
            ok = False
            continue
        validation = validate_scoped_representation_evidence_record(
            scoped_record,
            **raw_inputs,
        )
        expected_scope_kind = (
            "tr_completed"
            if tr_completed or joint_problem
            else "local_irrep"
        )
        scope = scoped_record.get("scope")
        source_basis = raw_inputs.get("source_basis_record")
        if (
            validation.status != "passed"
            or scoped_record.get("status") != "passed"
            or scoped_record.get("evidence_identity")
            != identity["scoped_representation_evidence_identity"]
            or scoped_record.get("source_basis_certificate_identity")
            != identity["spinor_source_basis_certificate_identity"]
            or scoped_record.get(
                "double_space_group_lift_certificate_identity"
            )
            != identity["double_space_group_lift_certificate_identity"]
            or not isinstance(source_basis, dict)
            or source_basis.get("certificate_identity")
            != identity["spinor_source_basis_certificate_identity"]
            or not isinstance(scope, dict)
            or scope.get("scope_kind") != expected_scope_kind
        ):
            blockers.append(
                _blocker(
                    "cprime_producer_context_invalid",
                    f"producer validation context failed for {kpoint}",
                )
            )
            ok = False
            continue
        if joint_problem:
            continue
        records = records_by_kpoint.get(kpoint)
        if not isinstance(records, list) or not records:
            blockers.append(
                _blocker(
                    "cprime_record_link_missing",
                    f"missing irrep record links for {kpoint}",
                )
            )
            ok = False
            continue
        for record in records:
            if tr_completed and isinstance(record, dict):
                candidate_provenance = record.get(
                    "source_candidate_provenance"
                )
                provenance = (
                    candidate_provenance.get("irrep_source_provenance")
                    if isinstance(candidate_provenance, dict)
                    else None
                )
            else:
                provenance = (
                    record.get("irrep_source_provenance")
                    if isinstance(record, dict)
                    else None
                )
            cprime = (
                provenance.get("cprime")
                if isinstance(provenance, dict)
                else None
            )
            if cprime != identity:
                blockers.append(
                    _blocker(
                        "cprime_record_link_mismatch",
                        f"irrep record C-prime link mismatch for {kpoint}",
                    )
                )
                ok = False
                break
    report["cprime_identity_check"] = "passed" if ok else "failed"


def _validate_certificate_status(cert_id, blockers, report):
    """Certificate must be validated, not unresolved/rejected."""
    val_statuses = cert_id.get("certificate_validation_statuses", [])
    if not isinstance(val_statuses, list):
        val_statuses = []
    validation_status = str(cert_id.get("validation_status", ""))
    ok = True
    if "rejected" in val_statuses or validation_status == "rejected":
        blockers.append(_blocker("certificate_rejected",
            "certificate_identity contains a rejected validation status"))
        ok = False
    if cert_id.get("any_unresolved", True) is not False:
        blockers.append(_blocker("certificate_unresolved",
            "certificate_identity is unresolved/not_evaluated"))
        ok = False
    if validation_status != "validated":
        blockers.append(_blocker("certificate_not_validated",
            f"certificate validation_status must be 'validated', got "
            f"{validation_status!r}"))
        ok = False
    report["certificate_check"] = "passed" if ok else "failed"


def _validate_certificate_identity_contract(cert_id, blockers, report):
    """Enforce the complete serialized certificate identity contract.

    Every producer field must be present with the exact type, and each
    singular field must equal its normalized plural collection.  Missing,
    empty, extra, duplicate, malformed, zero, or Boolean values block; nothing
    is inferred from another field.
    """
    ok = True
    if not _is_positive_int(cert_id.get("sg_number")):
        blockers.append(_blocker("certificate_sg_number_missing",
            "certificate missing positive integer sg_number"))
        ok = False
    sg_symbol = cert_id.get("sg_symbol")
    if not isinstance(sg_symbol, str) or not sg_symbol.strip():
        blockers.append(_blocker("certificate_sg_symbol_missing",
            "certificate missing non-empty sg_symbol"))
        ok = False

    hall_number = cert_id.get("hall_number")
    hall_symbol = cert_id.get("hall_symbol")
    if not _is_positive_int(hall_number):
        blockers.append(_blocker("certificate_hall_number_missing",
            "certificate missing positive integer hall_number"))
        ok = False
    elif cert_id.get("hall_numbers") != [hall_number]:
        blockers.append(_blocker("certificate_field_inconsistent",
            f"hall_numbers {cert_id.get('hall_numbers')!r} != [{hall_number}]"))
        ok = False
    if not isinstance(hall_symbol, str) or not hall_symbol.strip():
        blockers.append(_blocker("certificate_hall_symbol_missing",
            "certificate missing non-empty hall_symbol"))
        ok = False
    elif cert_id.get("hall_symbols") != [hall_symbol]:
        blockers.append(_blocker("certificate_field_inconsistent",
            f"hall_symbols {cert_id.get('hall_symbols')!r} != [{hall_symbol!r}]"))
        ok = False

    centering = cert_id.get("centering_type")
    if centering not in _RECOGNIZED_CENTERINGS:
        blockers.append(_blocker("certificate_centering_invalid",
            f"unrecognized centering_type {centering!r}"))
        ok = False
    elif cert_id.get("centering_types") != [centering]:
        blockers.append(_blocker("certificate_field_inconsistent",
            f"centering_types {cert_id.get('centering_types')!r} != "
            f"[{centering!r}]"))
        ok = False

    vs = cert_id.get("validation_status")
    if not isinstance(vs, str) or not vs:
        blockers.append(_blocker("certificate_validation_status_missing",
            "certificate missing non-empty validation_status"))
        ok = False
    elif cert_id.get("certificate_validation_statuses") != [vs]:
        blockers.append(_blocker("certificate_field_inconsistent",
            "certificate_validation_statuses "
            f"{cert_id.get('certificate_validation_statuses')!r} != [{vs!r}]"))
        ok = False

    dsi = cert_id.get("distinct_setting_identities")
    if not isinstance(dsi, int) or isinstance(dsi, bool) or dsi != 1:
        blockers.append(_blocker("certificate_ambiguous_setting",
            f"distinct_setting_identities must be integer 1, got {dsi!r}"))
        ok = False

    if cert_id.get("any_unresolved") is not False:
        blockers.append(_blocker("certificate_field_inconsistent",
            f"any_unresolved must be exactly False, got "
            f"{cert_id.get('any_unresolved')!r}"))
        ok = False

    report["certificate_consistency_check"] = "passed" if ok else "failed"


def _validate_sg_identity_crosscheck(cert_id, bundle_sg, bundle_sg_num,
                                     table_sg, table_sg_num, table_setting,
                                     blockers, report):
    """Cross-check certificate/bundle/table SG number AND symbol against the
    independently derived spglib group.  Bundle/table agreement is insufficient
    when both disagree with the spglib/Hall evidence."""
    if table_setting is None:
        report["cert_sg_consistency_check"] = "failed"
        return
    canonical_number = int(table_setting["space_group_number"])
    canonical_symbol = _norm_symbol(table_setting["space_group_symbol"])
    ok = True
    for name, num, code in (
            ("certificate", cert_id.get("sg_number"), "certificate_sg_conflict"),
            ("bundle", bundle_sg_num, "bundle_sg_conflict"),
            ("table", table_sg_num, "table_sg_conflict")):
        if not _is_positive_int(num) or int(num) != canonical_number:
            blockers.append(_blocker(code,
                f"{name} SG number {num!r} != spglib SG {canonical_number}"))
            ok = False
    for name, sym, code in (
            ("certificate", cert_id.get("sg_symbol"), "certificate_symbol_conflict"),
            ("bundle", bundle_sg, "bundle_symbol_conflict"),
            ("table", table_sg, "table_symbol_conflict")):
        if _norm_symbol(sym) != canonical_symbol:
            blockers.append(_blocker(code,
                f"{name} SG symbol {sym!r} != spglib symbol "
                f"{canonical_symbol!r} for SG {canonical_number}"))
            ok = False
    report["cert_sg_consistency_check"] = "passed" if ok else "failed"



def _validate_setting(cert_id, table_setting, blockers, report):
    """Validate centering + primitive/centered affine evidence (real vocab)."""
    raw_centering = cert_id.get("centering_type")
    centering = "" if raw_centering is None else str(raw_centering)
    centering_types = cert_id.get("centering_types", [])
    if not isinstance(centering_types, list):
        centering_types = []
    if not centering:
        blockers.append(_blocker("centering_missing",
            "certificate has no centering_type; missing centering is unknown, "
            "not primitive"))
        report["affine_setting_check"] = "blocked"
        return
    if len(centering_types) > 1:
        blockers.append(_blocker("centering_ambiguous",
            f"multiple centering types in one instance: {centering_types}"))
        report["affine_setting_check"] = "failed"
        return

    validation_status = str(cert_id.get("validation_status", ""))
    relation = str(cert_id.get("primitive_conventional_relation", ""))
    if centering == "P":
        _validate_primitive_affine_setting(cert_id, validation_status,
                                           relation, blockers, report)
        return

    if centering not in _CENTERED_TYPES:
        blockers.append(_blocker("centering_unrecognized",
            f"unrecognized centering_type {centering!r}"))
        report["affine_setting_check"] = "failed"
        return
    _validate_centered_affine_setting(
        cert_id, table_setting, validation_status, relation, blockers, report,
    )


def _exact_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _valid_fractional_vector(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and float(item) == float(item)
            and float(item) not in (float("inf"), float("-inf"))
            for item in value
        )
    )


def _fractional_vector_sequence(
    value: object,
) -> tuple[tuple[float, float, float], ...] | None:
    if not isinstance(value, list) or not all(
        _valid_fractional_vector(vector) for vector in value
    ):
        return None
    normalized: list[tuple[float, float, float]] = []
    for vector in value:
        entries = []
        for item in vector:
            reduced = float(item) % 1.0
            if abs(reduced) <= 1e-8 or abs(reduced - 1.0) <= 1e-8:
                reduced = 0.0
            entries.append(round(reduced, 8))
        normalized.append(tuple(entries))
    return tuple(normalized)


def _validate_centered_affine_setting(
    cert_id, table_setting, validation_status, relation, blockers, report,
):
    """Validate a complete primitive-to-centered Hall operation bijection."""
    reasons: list[str] = []
    op_status = cert_id.get("operation_mapping_status")
    affine_status = cert_id.get("affine_validation_status")
    if validation_status != "validated":
        reasons.append(f"validation_status={validation_status!r}")
    if relation not in _CENTERED_RELATIONS:
        reasons.append(f"relation={relation!r}")
    if op_status != _OP_MAPPING_PASSED:
        reasons.append(f"operation_mapping_status={op_status!r}")
    if affine_status != _AFFINE_PASSED:
        reasons.append(f"affine_validation_status={affine_status!r}")

    canonical_status = cert_id.get("canonical_setting_status")
    canonical_source = cert_id.get("canonical_setting_source")
    hall_number = cert_id.get("hall_number")
    canonical_halls = cert_id.get("canonical_hall_numbers")
    candidate_halls = cert_id.get("canonical_candidate_hall_numbers")
    if canonical_status != "unique_match":
        reasons.append(f"canonical_setting_status={canonical_status!r}")
    if not isinstance(canonical_source, str) or not canonical_source:
        reasons.append("canonical_setting_source_missing")
    if canonical_halls != [hall_number]:
        reasons.append(
            f"canonical_hall_numbers={canonical_halls!r} expected [{hall_number!r}]"
        )
    if type(candidate_halls) is not list or any(
        not isinstance(item, int) or isinstance(item, bool)
        for item in candidate_halls
    ) or len(candidate_halls) != len(set(candidate_halls)) \
            or hall_number not in candidate_halls:
        reasons.append("canonical_candidate_hall_numbers_malformed")

    transform = cert_id.get("normalized_direct_transform")
    index = _exact_int(cert_id.get("primitive_conventional_index"))
    coset_count = _exact_int(cert_id.get("centering_coset_count"))
    if not _finite_nonsingular_3x3(transform):
        reasons.append("direct_transform_missing_or_singular")
    elif index is None or index <= 1:
        reasons.append(f"primitive_conventional_index={index!r}")
    else:
        rows = [[float(item) for item in row] for row in transform]
        (a, b, c), (d, e, f), (g, h, i) = rows
        determinant = (
            a * (e * i - f * h)
            - b * (d * i - f * g)
            + c * (d * h - e * g)
        )
        if abs(abs(determinant) - 1.0 / index) > 1e-8:
            reasons.append(
                f"transform_determinant_index_mismatch(det={determinant},index={index})"
            )
    if index is None or coset_count != index:
        reasons.append(f"centering_coset_count={coset_count!r},index={index!r}")

    origin = cert_id.get("normalized_origin_shift")
    if not _valid_fractional_vector(origin):
        reasons.append("normalized_origin_shift_missing_or_malformed")
    vectors = cert_id.get("normalized_centering_vectors")
    if type(vectors) is not list or index is None or len(vectors) != index \
            or not all(_valid_fractional_vector(vector) for vector in vectors):
        reasons.append("normalized_centering_vectors_malformed")
    elif len({tuple(float(item) for item in vector) for vector in vectors}) \
            != len(vectors):
        reasons.append("normalized_centering_vectors_duplicate")
    elif [0.0, 0.0, 0.0] not in [
        [float(item) for item in vector] for vector in vectors
    ]:
        reasons.append("identity_centering_vector_missing")
    if not isinstance(table_setting, dict):
        reasons.append("independent_table_setting_missing")
    else:
        if index != _exact_int(table_setting.get("primitive_conventional_index")):
            reasons.append("primitive_conventional_index_table_mismatch")
        expected_vectors = _fractional_vector_sequence(
            table_setting.get("centering_cosets")
        )
        actual_vectors = _fractional_vector_sequence(vectors)
        if expected_vectors is None or actual_vectors != expected_vectors:
            reasons.append("centering_vectors_table_mismatch")

    required_ids = cert_id.get("affine_required_operation_ids")
    req_count = _exact_int(cert_id.get("affine_required_op_count"))
    std_count = _exact_int(cert_id.get("affine_standard_setting_op_count"))
    expanded_count = _exact_int(cert_id.get("expanded_parent_operation_count"))
    matched_expanded = _exact_int(cert_id.get("matched_expanded_operations"))
    matched_parent = _exact_int(cert_id.get("affine_matched_operations"))
    parent_total = _exact_int(cert_id.get("affine_total_operations"))
    table_std_count = (
        _exact_int(table_setting.get("standard_setting_operation_count"))
        if isinstance(table_setting, dict) else None
    )
    if type(required_ids) is not list or any(
        not isinstance(item, int) or isinstance(item, bool)
        for item in required_ids
    ) or len(required_ids) != len(set(required_ids)):
        reasons.append("affine_required_operation_ids_malformed")
        required_ids = []
    if not (
        req_count is not None and req_count > 0
        and len(required_ids) == req_count
        and parent_total == req_count
        and matched_parent == req_count
        and index is not None
        and expanded_count == req_count * index
        and matched_expanded == expanded_count
        and std_count == expanded_count
        and table_std_count == std_count
    ):
        reasons.append(
            "centered_bijection_counts("
            f"required={req_count},parent={parent_total},matched_parent={matched_parent},"
            f"index={index},expanded={expanded_count},"
            f"matched_expanded={matched_expanded},standard={std_count})"
        )

    centered_map = cert_id.get("centered_affine_operation_map")
    if type(centered_map) is not list or not centered_map:
        reasons.append("centered_affine_operation_map_missing_or_empty")
    else:
        pair_keys: list[tuple[int, int]] = []
        standard_indices: list[int] = []
        for row in centered_map:
            if not isinstance(row, dict) or set(row) != {
                "parent_operation_id", "centering_coset_index",
                "standard_operation_index",
            }:
                reasons.append("centered_affine_operation_map_row_malformed")
                break
            parent_id = row["parent_operation_id"]
            coset_index = row["centering_coset_index"]
            standard_index = row["standard_operation_index"]
            if any(_exact_int(value) is None for value in (
                parent_id, coset_index, standard_index,
            )):
                reasons.append("centered_affine_operation_map_value_non_integer")
                break
            if parent_id not in required_ids:
                reasons.append("centered_affine_operation_map_parent_id_unknown")
                break
            if index is None or coset_index < 0 or coset_index >= index:
                reasons.append("centered_affine_operation_map_coset_out_of_range")
                break
            if std_count is None or standard_index < 0 or standard_index >= std_count:
                reasons.append("centered_affine_operation_map_target_out_of_range")
                break
            pair_keys.append((parent_id, coset_index))
            standard_indices.append(standard_index)
        if len(pair_keys) != len(set(pair_keys)):
            reasons.append("centered_affine_operation_map_duplicate_pairs")
        if len(standard_indices) != len(set(standard_indices)):
            reasons.append("centered_affine_operation_map_reused_standard_operation")
        if index is not None:
            expected_pairs = {
                (parent_id, coset_index)
                for parent_id in required_ids
                for coset_index in range(index)
            }
            if set(pair_keys) != expected_pairs:
                reasons.append("centered_affine_operation_map_incomplete_pairs")
        if std_count is not None and set(standard_indices) != set(range(std_count)):
            reasons.append("centered_affine_operation_map_incomplete_standard_coverage")

    if cert_id.get("affine_operation_map") is not None:
        reasons.append("primitive_affine_operation_map_must_be_absent_for_centered")
    for field in (
        "affine_unmatched_parent_operations",
        "affine_unmatched_centered_operation_pairs",
        "affine_unused_standard_operation_indices",
        "affine_missing_ingredients",
    ):
        if type(cert_id.get(field)) is not list or cert_id.get(field) != []:
            reasons.append(f"{field}={cert_id.get(field)!r}")
    if cert_id.get("affine_mismatch_count") != 0:
        reasons.append(f"affine_mismatch_count={cert_id.get('affine_mismatch_count')!r}")
    if cert_id.get("operation_closure_validated") is not True:
        reasons.append("operation_closure_validated_not_true")
    if cert_id.get("standard_operation_closure_validated") is not True:
        reasons.append("standard_operation_closure_validated_not_true")

    if reasons:
        blockers.append(_blocker(
            "centered_affine_evidence_invalid",
            "centered setting requires a complete expanded affine bijection; "
            + "; ".join(reasons),
        ))
        report["affine_setting_check"] = "failed"
    else:
        report["affine_setting_check"] = "passed"


def _finite_nonsingular_3x3(m: object) -> bool:
    """True when ``m`` is a finite, non-singular 3x3 numeric matrix."""
    if not isinstance(m, list) or len(m) != 3:
        return False
    rows: list[list[float]] = []
    for row in m:
        if not isinstance(row, list) or len(row) != 3:
            return False
        vals: list[float] = []
        for v in row:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return False
            f = float(v)
            if f != f or f in (float("inf"), float("-inf")):
                return False
            vals.append(f)
        rows.append(vals)
    (a, b, c), (d, e, f), (g, h, i) = rows
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return abs(det) > 1e-9


def _normalize_operation_map_key(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = int(value)
    except ValueError:
        return None
    if str(normalized) != value:
        return None
    return normalized


def _validate_operation_map_structure(op_map: dict, required_ids,
                                      req_op_count, std_op_count, cert_id,
                                      reasons: list[str]):
    """Independently verify the claimed bijection structure."""
    if not isinstance(required_ids, list) or any(
        not isinstance(item, int) or isinstance(item, bool)
        for item in required_ids
    ) or len(required_ids) != len(set(required_ids)):
        reasons.append(
            f"affine_required_operation_ids_malformed({required_ids!r})"
        )
        return
    if req_op_count is None or req_op_count <= 0:
        reasons.append(f"affine_operation_map_untestable(req_op={req_op_count!r})")
        return
    if std_op_count is None or std_op_count <= 0:
        reasons.append(f"affine_operation_map_untestable(std_op={std_op_count!r})")
        return
    r = int(req_op_count); s = int(std_op_count)
    if len(required_ids) != r:
        reasons.append(
            f"affine_required_operation_id_count({len(required_ids)}!={r})"
        )
        return
    # Keys are opaque integer IDs.  Reject aliases such as 0 and "0".
    keys: list[int] = []
    for raw_key in op_map:
        key = _normalize_operation_map_key(raw_key)
        if key is None:
            reasons.append("affine_operation_map_keys_non_integer")
            return
        if key in keys:
            reasons.append("affine_operation_map_key_alias_or_duplicate")
            return
        keys.append(key)
    if len(keys) != r:
        reasons.append(f"affine_operation_map_cardinality({len(keys)}!={r})")
        return
    if set(keys) != set(required_ids):
        reasons.append("affine_operation_map_keys_do_not_match_required_ids")
        return
    # Values: unique, within range, non-bool integers.
    values = list(op_map.values())
    for v in values:
        if not isinstance(v, int) or isinstance(v, bool):
            reasons.append(f"affine_operation_map_value_non_integer({v!r})")
            return
        if v < 0 or v >= s:
            reasons.append(f"affine_operation_map_target_out_of_range({v},0..{s-1})")
            return
    if len(set(values)) != len(values):
        reasons.append("affine_operation_map_duplicate_targets")
        return
    if set(values) != set(range(s)):
        reasons.append("affine_operation_map_targets_do_not_cover_standard_set")
        return
    # Audit collections: absent is unknown (block), explicit empty is required.
    unmatched = cert_id.get("affine_unmatched_parent_operations")
    if type(unmatched) is not list or unmatched != []:
        reasons.append(f"affine_unmatched_parent_operations={unmatched!r}")
    unused = cert_id.get("affine_unused_standard_operation_indices")
    if type(unused) is not list or unused != []:
        reasons.append(f"affine_unused_std_indices={unused!r}")


def _validate_primitive_affine_setting(cert_id, validation_status, relation,
                                       blockers, report):
    """A primitive direct-coordinate match is trusted only with a complete
    one-to-one affine operation-group bijection under the strict 3x3 identity
    transform, explicit zero-mismatch and empty-missing-ingredient evidence,
    and proven affine group closure."""
    op_status = str(cert_id.get("operation_mapping_status", ""))
    affine_status = str(cert_id.get("affine_validation_status", ""))
    transform = cert_id.get("normalized_direct_transform")
    transform_prov = str(cert_id.get("transform_provenance", ""))
    matched = cert_id.get("affine_matched_operations")
    total = cert_id.get("affine_total_operations")
    std_op = cert_id.get("affine_standard_setting_op_count")
    req_op = cert_id.get("affine_required_op_count")
    required_ids = cert_id.get("affine_required_operation_ids")
    mismatch = cert_id.get("affine_mismatch_count")
    missing = cert_id.get("affine_missing_ingredients")
    op_map = cert_id.get("affine_operation_map")
    closure = cert_id.get("operation_closure_validated")

    def _opt_int(val: object) -> int | None:
        return int(val) if isinstance(val, int) and not isinstance(val, bool) \
            else None

    reasons: list[str] = []
    if validation_status != "validated":
        reasons.append(f"validation_status={validation_status!r}")
    if relation != _PRIMITIVE_DIRECT_RELATION:
        reasons.append(f"relation={relation!r}")
    if op_status != _OP_MAPPING_PASSED:
        reasons.append(f"operation_mapping_status={op_status!r}")
    if affine_status != _AFFINE_PASSED:
        reasons.append(f"affine_validation_status={affine_status!r}")
    # Exact identity transform required for primitive direct-coordinate.
    if not _is_exact_identity_3x3(transform):
        reasons.append("direct_transform_not_exact_3x3_identity")
    if transform_prov != "primitive_direct_identity":
        reasons.append(f"transform_provenance={transform_prov!r}")

    m = _opt_int(matched); t = _opt_int(total)
    s = _opt_int(std_op); r = _opt_int(req_op)
    if not (t is not None and t > 0
            and m is not None and m == t
            and s is not None and s == t
            and r is not None and r == t):
        reasons.append(
            f"bijection_counts(matched={m},parent_total={t},"
            f"std_op={s},required={r})")
    if not (isinstance(mismatch, int) and not isinstance(mismatch, bool)
            and mismatch == 0):
        reasons.append(f"mismatch_count={mismatch!r}")
    if not (isinstance(missing, list) and len(missing) == 0):
        reasons.append(f"missing_ingredients={missing!r}")
    if not isinstance(op_map, dict) or not op_map:
        reasons.append("affine_operation_map_missing_or_empty")
    else:
        _validate_operation_map_structure(
            op_map, required_ids, r, s, cert_id, reasons)

    if closure is not True:
        reasons.append(f"operation_closure_validated={closure!r}")

    if reasons:
        blockers.append(_blocker("primitive_affine_evidence_invalid",
            "primitive direct-coordinate setting requires a complete one-to-one "
            "affine operation-group bijection; " + "; ".join(reasons)))
        report["affine_setting_check"] = "failed"
    else:
        report["affine_setting_check"] = "passed"


def _is_exact_identity_3x3(m: object) -> bool:
    """True when ``m`` is exactly [[1,0,0],[0,1,0],[0,0,1]]."""
    if isinstance(m, list) and len(m) == 3:
        for i, row in enumerate(m):
            if not isinstance(row, list) or len(row) != 3:
                return False
            for j, v in enumerate(row):
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    return False
                expected = 1.0 if i == j else 0.0
                if abs(float(v) - expected) > 1e-9:
                    return False
        return True
    return False


def _validate_hall_consistency(cert_id, table_hall, table_hall_symbol,
                               table_centering, table_resolved,
                               blockers, report):
    """Cert Hall must belong to its SG, match its centering, and equal the
    independently derived table Hall/centering."""
    bundle_hall = cert_id.get("hall_number")
    bundle_hall_symbol = str(cert_id.get("hall_symbol", ""))
    bundle_centering = "" if cert_id.get("centering_type") is None \
        else str(cert_id.get("centering_type"))
    cert_sg = cert_id.get("sg_number")
    ok = True
    if not _is_positive_int(bundle_hall) or not bundle_hall_symbol:
        blockers.append(_blocker("setting_identity_missing",
            "bundle certificate missing Hall setting identity"))
        report["hall_setting_check"] = "blocked"
        return
    # Hall must belong to the certificate's own SG number (spglib).
    if _is_positive_int(cert_sg) and not _hall_belongs_to_sg(bundle_hall, cert_sg):
        blockers.append(_blocker("hall_sg_inconsistent",
            f"Hall number {bundle_hall} does not belong to SG {cert_sg} "
            "(spglib)"))
        ok = False
    # Centering must match the Hall symbol.
    hall_centering = _centering_from_hall_symbol(bundle_hall_symbol)
    if bundle_centering and hall_centering and bundle_centering != hall_centering:
        blockers.append(_blocker("centering_hall_inconsistent",
            f"centering_type {bundle_centering!r} inconsistent with Hall "
            f"symbol {bundle_hall_symbol!r} (spglib centering {hall_centering!r})"))
        ok = False
    # Must equal the independently derived table standard setting.
    if not table_resolved:
        report["hall_setting_check"] = "failed"
        return
    if int(bundle_hall) != int(table_hall) \
            or bundle_hall_symbol != table_hall_symbol \
            or bundle_centering != table_centering:
        blockers.append(_blocker("setting_identity_mismatch",
            "certificate setting does not match independently derived table "
            f"setting: cert (hall={bundle_hall}, symbol={bundle_hall_symbol!r}, "
            f"centering={bundle_centering!r}) != table (hall={table_hall}, "
            f"symbol={table_hall_symbol!r}, centering={table_centering!r})"))
        ok = False
    report["hall_setting_check"] = "passed" if ok else "failed"


def _collect_spin_evidence(bundle, cert_id):
    """Gather every spin-convention datum available on the bundle/cert."""
    evidence: list[tuple[str, bool]] = []
    for holder, name in ((bundle, "bundle"), (cert_id, "certificate")):
        if isinstance(holder, dict):
            for key in ("spinor", "spinful", "source_table_spinor"):
                v = holder.get(key)
                if isinstance(v, bool):
                    evidence.append((f"{name}.{key}", v))
    records = bundle.get("irrep_records_by_kpoint", {})
    if isinstance(records, dict):
        for kp, kp_records in records.items():
            if not isinstance(kp_records, list):
                continue
            for rec in kp_records:
                if not isinstance(rec, dict):
                    continue
                p = rec.get("irrep_source_provenance")
                if isinstance(p, dict) and isinstance(
                        p.get("source_table_spinor"), bool):
                    evidence.append(
                        (f"record.{kp}.source_table_spinor",
                         p["source_table_spinor"]))
    return evidence


def _validate_spin_convention(bundle, cert_id, table_spinful, blockers, report):
    """All spin evidence must agree and equal the table spinful flag."""
    evidence = _collect_spin_evidence(bundle, cert_id)
    if not isinstance(table_spinful, bool):
        report["spin_convention_check"] = "failed"
        return
    if not evidence:
        blockers.append(_blocker("spin_convention_missing",
            "bundle spin convention (spinor) not recorded anywhere"))
        report["spin_convention_check"] = "blocked"
        return
    values = {v for _, v in evidence}
    if len(values) > 1:
        blockers.append(_blocker("spin_evidence_conflict",
            f"mutually inconsistent spin evidence: {evidence}"))
        report["spin_convention_check"] = "failed"
        return
    bundle_spin = next(iter(values))
    if bundle_spin != table_spinful:
        blockers.append(_blocker("spin_convention_mismatch",
            f"bundle spinor={bundle_spin} != table spinful={table_spinful}"))
        report["spin_convention_check"] = "failed"
    else:
        report["spin_convention_check"] = "passed"


def load_reduced_ebr_table(path: str | Path) -> dict:
    """Load and validate a reduced EBR table from JSON.

    Raises ValueError for missing keys, empty/malformed fields,
    non-unique labels, mismatched vector lengths, non-integer/nonnegative
    vector entries, or undocumented irrep key formats.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = _REQUIRED_TABLE_KEYS - set(raw)
    if missing:
        raise ValueError(f"reduced EBR table missing keys: {sorted(missing)}")

    # schema_version
    schema_version = raw.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("table schema_version must be a non-empty string")

    # subspace_group_candidate
    if not isinstance(raw.get("subspace_group_candidate"), str) or not raw["subspace_group_candidate"]:
        raise ValueError("table subspace_group_candidate must be a non-empty string")

    # expected_hsps
    expected_hsps = raw["expected_hsps"]
    if not isinstance(expected_hsps, list):
        raise ValueError("table expected_hsps must be a list")
    if not expected_hsps:
        raise ValueError("table expected_hsps must be a non-empty list")
    if not all(isinstance(h, str) and h for h in expected_hsps):
        raise ValueError("table expected_hsps entries must be non-empty strings")
    if len(set(expected_hsps)) != len(expected_hsps):
        raise ValueError("table expected_hsps must contain unique entries")

    # irreps
    irreps = raw["irreps"]
    if not isinstance(irreps, list) or not irreps:
        raise ValueError("table irreps must be a non-empty list")
    if not all(isinstance(label, str) and label for label in irreps):
        raise ValueError("table irreps must be non-empty strings")
    if len(set(irreps)) != len(irreps):
        raise ValueError("table irreps must be unique")
    _validate_irrep_key_format(irreps)

    # ebrs
    ebrs = raw["ebrs"]
    if not isinstance(ebrs, list) or not ebrs:
        raise ValueError("table ebrs must be a non-empty list")

    n_irreps = len(irreps)
    ebr_labels = []
    for ebr in ebrs:
        if not isinstance(ebr, dict):
            raise ValueError("each EBR entry must be a mapping")
        label = ebr.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError("each EBR must define a non-empty label")
        ebr_labels.append(label)
        vector = ebr.get("vector")
        if not isinstance(vector, list):
            raise ValueError(f"EBR '{label}' missing vector")
        if not vector:
            raise ValueError(f"EBR '{label}' vector must be non-empty")
        if len(vector) != n_irreps:
            raise ValueError(
                f"EBR '{label}' vector length {len(vector)} "
                f"!= irrep count {n_irreps}"
            )
        if not all(isinstance(v, int) and v >= 0 for v in vector):
            raise ValueError(
                f"EBR '{label}' vector must be nonnegative integers"
            )
        if not any(v > 0 for v in vector):
            raise ValueError(
                f"EBR '{label}' vector must have at least one positive entry"
            )
    if len(set(ebr_labels)) != len(ebr_labels):
        raise ValueError("table EBR labels must be unique")

    return raw


_IRREP_KEY_RE = (
    r"\A"                         # start
    r"[A-Za-z][A-Za-z0-9_]*"      # kpoint label
    r":"                          # separator
    r"-?"                         # optional leading minus (spinor convention)
    r"[A-Za-z]"                   # irrep label must start with a letter
    r"[A-Za-z0-9_+/\-]*"         # rest of irrep label
    r"(?::op\d+)?"                # optional operation suffix
    r"\Z"                         # end
)


def _validate_irrep_key_format(irreps: list[str]) -> None:
    """Validate irrep keys match the documented format.

    Format: ``<kpoint>:<irrep_label>`` with optional ``:op<N>`` suffix.
    Example valid keys:
      ``GammaM:C3_spinor_phase_+1/2``
      ``KM:C3_spinor_phase_-1/6``
      ``GammaM:C3_spinor_phase_+1/2:op1``
    """
    import re
    pattern = re.compile(_IRREP_KEY_RE)
    for label in irreps:
        if not pattern.match(label):
            raise ValueError(
                f"invalid irrep key format: {label!r}. "
                f"Expected format: <kpoint>:<irrep_label> "
                f"with optional :op<N> suffix"
            )


def build_reduced_ebr_mapping(
    *,
    ebr_export_bundle: dict | None,
    table: dict | None = None,
    max_coefficient: int = 6,
    reduced_ebr_input: dict | None = None,
    cprime_validation_context: dict[str, object] | None = None,
) -> dict:
    """Exact reduced EBR decomposition of export bundle irrep vectors.

    Parameters
    ----------
    ebr_export_bundle : output of build_ebr_export_bundle
    table : validated reduced EBR table dict (from load_reduced_ebr_table)
    max_coefficient : int, max coefficient per EBR in brute-force search
    reduced_ebr_input : optional compact non-path provenance documenting
        which reduced-EBR input source was used (table_file or spec_file).
    """
    max_coefficient = int(max_coefficient)
    if max_coefficient < 0:
        raise ValueError("max_coefficient must be nonnegative")

    if ebr_export_bundle is None:
        return _status("not_evaluated", "no export bundle available",
                       reduced_ebr_input=reduced_ebr_input)

    raw_bundles = ebr_export_bundle.get("bundles", [])
    bundles = raw_bundles if isinstance(raw_bundles, list) else []

    if table is None:
        result: dict = {
            **_status("missing_table", "no reduced EBR table provided",
                      reduced_ebr_input=reduced_ebr_input),
            "solutions": [],
            "excluded_bundles": [
                {
                    "bundle_id": b.get("bundle_id", "?"),
                    "subspace_group_candidate": b.get("subspace_group_candidate", ""),
                    "subspace_space_group": b.get("subspace_space_group", {}),
                    "irrep_source_provenance_by_kpoint": _per_kpoint_prov(b),
                    "reason": "missing_table",
                }
                for b in bundles if isinstance(b, dict)
            ],
            "table_status": "not_provided",
        }
        return result

    table_ebrs = table["ebrs"]

    solutions: list[dict] = []
    excluded: list[dict] = []

    for bundle in bundles:
        if not isinstance(bundle, dict):
            excluded.append({
                "bundle_id": "?",
                "reason": "malformed export bundle entry",
            })
            continue
        # Every bundle must pass the same validation against the actual
        # table used for this solve.  Pre-existing readiness flags are
        # not trusted evidence and never bypass validation.
        is_validation_candidate = (
            bundle.get("ready_for_reduced_table_validation") is True
        )
        if is_validation_candidate:
            promo = promote_bundle_for_solve(
                bundle=bundle,
                table=table,
                cprime_validation_context=cprime_validation_context,
            )
            if promo["promoted"] and promo["promoted_bundle"] is not None:
                bundle = promo["promoted_bundle"]
            else:
                excluded.append({
                    "bundle_id": bundle.get("bundle_id", "?"),
                    "subspace_group_candidate": bundle.get(
                        "subspace_group_candidate", ""),
                    "subspace_space_group": bundle.get(
                        "subspace_space_group", {}),
                    "irrep_source_provenance_by_kpoint": _per_kpoint_prov(
                        bundle),
                    "reason": (
                        "validation blocked: "
                        + "; ".join(
                            f"{b['code']}: {b['detail']}"
                            for b in promo["blocker_reasons"]
                        )
                    ),
                    "blocker_reasons": promo["blocker_reasons"],
                    "validation_report": promo["validation_report"],
                    "certificate_identity": promo["certificate_identity"],
                    "table_provenance": _merge_table_input_provenance(
                        promo["table_provenance"], reduced_ebr_input
                    ),
                })
                continue
        else:
            excluded.append({
                "bundle_id": bundle.get("bundle_id", "?"),
                "subspace_group_candidate": bundle.get(
                    "subspace_group_candidate", ""),
                "subspace_space_group": bundle.get(
                    "subspace_space_group", {}),
                "irrep_source_provenance_by_kpoint": _per_kpoint_prov(bundle),
                "reason": "not ready for reviewed reduced-table validation",
            })
            continue

        bundle_group = str(bundle.get("subspace_group_candidate", ""))
        irrep_counts = promo.get("irrep_vector")
        if not isinstance(irrep_counts, list):
            excluded.append({
                "bundle_id": bundle.get("bundle_id", "?"),
                "subspace_group_candidate": bundle_group,
                "subspace_space_group": bundle.get("subspace_space_group", {}),
                "irrep_source_provenance_by_kpoint": _per_kpoint_prov(bundle),
                "reason": "promotion returned no canonical irrep vector",
            })
            continue

        # --- Integer-span classification (delegated to solver) ---
        from valleyscope.analysis.reduced_ebr_solver import classify_bundle
        ebr_vectors = [list(ebr["vector"]) for ebr in table_ebrs]
        ebr_labels_list = [str(ebr.get("label", "?")) for ebr in table_ebrs]
        result = classify_bundle(
            irrep_counts, ebr_vectors, ebr_labels_list, max_coefficient,
        )
        # Extract per-kpoint irrep source provenance from bundle records.
        per_kp_prov = _build_per_kpoint_provenance(bundle)
        solution = {
            "bundle_id": bundle.get("bundle_id", ""),
            "problem_kind": bundle.get(
                "problem_kind", "unitary_valley_reduced_ebr"
            ),
            "physical_object_kind": bundle.get(
                "physical_object_kind",
                "unitary_valley_projected_subspace",
            ),
            "valley": bundle.get("valley", ""),
            "valley_orbit": bundle.get("valley_orbit", []),
            "expected_hsps": bundle.get("expected_hsps", []),
            "required_source_hsp_labels": bundle.get(
                "required_source_hsp_labels", []
            ),
            "covered_source_hsp_labels": bundle.get(
                "covered_source_hsp_labels", []
            ),
            "source_hsp_to_sampled_kpoint": bundle.get(
                "source_hsp_to_sampled_kpoint", {}
            ),
            "independent_source_hsp_to_sampled_kpoint": bundle.get(
                "independent_source_hsp_to_sampled_kpoint", {}
            ),
            "observed_source_hsp_to_sampled_kpoint": bundle.get(
                "observed_source_hsp_to_sampled_kpoint", {}
            ),
            "unitary_vector_construction": bundle.get(
                "unitary_vector_construction", {}
            ),
            "unitary_irrep_completion_records_by_hsp": bundle.get(
                "unitary_irrep_completion_records_by_hsp", {}
            ),
            "unitary_valley_irreps": bundle.get(
                "unitary_valley_irreps", {}
            ),
            "time_reversal": bundle.get("time_reversal", {}),
            "cprime_identity_by_kpoint": bundle.get(
                "cprime_identity_by_kpoint", {}
            ),
            "subspace_group_candidate": bundle_group,
            "subspace_space_group": bundle.get("subspace_space_group", {}),
            "irrep_vector": irrep_counts,
            **(per_kp_prov if per_kp_prov else {}),
            **result,
        }
        # Preserve the promotion validation evidence in the solution record.
        promo_prov = bundle.get("promotion_provenance")
        if isinstance(promo_prov, dict):
            solution["promotion_provenance"] = promo_prov
            if isinstance(promo_prov.get("validation_report"), dict):
                solution["validation_report"] = promo_prov["validation_report"]
            if isinstance(promo_prov.get("table_provenance"), dict):
                solution["table_provenance"] = _merge_table_input_provenance(
                    promo_prov["table_provenance"], reduced_ebr_input
                )
            if isinstance(promo_prov.get("certificate_identity"), dict):
                solution["certificate_identity"] = \
                    promo_prov["certificate_identity"]
        solutions.append(solution)

    mapping_status = _aggregate_mapping_status(
        solutions=solutions,
        excluded=excluded,
        input_count=len(bundles),
    )
    result = {
        "schema_version": _OUTPUT_SCHEMA_VERSION,
        "status": mapping_status,
        "table_status": "loaded",
        "solutions": solutions,
        "excluded_bundles": excluded,
        "solver": _SOLVER_NAME,
        "max_coefficient": max_coefficient,
        "interpretation": (
            "Exact integer linear combination of EBR vectors matching the "
            "bundle irrep count vector.  No heuristic fit; only exact matches "
            "are reported.  A missing_table status means no user-supplied "
            "reduced EBR table was provided."
        ),
    }
    if reduced_ebr_input is not None:
        result["reduced_ebr_input"] = dict(reduced_ebr_input)
    return result


def build_auto_reduced_ebr_mapping(
    *,
    ebr_export_bundle: dict[str, object] | None,
    spinor: bool,
    max_coefficient: int = 6,
    cprime_validation_context: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one reviewed irreptables table per canonical bundle and solve."""
    bundles = (
        ebr_export_bundle.get("bundles", [])
        if isinstance(ebr_export_bundle, dict) else []
    )
    if not isinstance(bundles, list) or not bundles:
        return _status(
            "not_evaluated",
            "no canonical HSP-vector bundles available for auto evaluation",
            reduced_ebr_input={
                "source": "auto_canonical",
                "input_bundle_count": 0,
                "reduced_table_validation_candidate_bundle_count": 0,
                "final_reduced_ebr_result_count": 0,
                "final_mapping_excluded_bundle_count": 0,
            },
        )

    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_auto_canonical_reduced_ebr_table,
        build_auto_time_reversal_reduced_ebr_table,
    )

    solutions: list[dict] = []
    excluded: list[dict] = []
    per_bundle: list[dict[str, object]] = []
    table_input_by_bundle: dict[str, dict[str, object]] = {}
    ready_count = 0
    loaded_count = 0
    tr_bundle_count = sum(
        isinstance(bundle, dict)
        and bundle.get("problem_kind") == "valley_orbit_reduced_ebr"
        for bundle in bundles
    )

    for raw_bundle in bundles:
        if not isinstance(raw_bundle, dict):
            excluded.append({
                "bundle_id": "?",
                "reason": "malformed export bundle entry",
            })
            per_bundle.append({
                "bundle_id": "?",
                "status": "blocked",
                "table_status": "not_applicable",
            })
            continue
        bundle_id = str(raw_bundle.get("bundle_id", "?"))
        if raw_bundle.get("ready_for_reduced_table_validation") is not True:
            excluded.append({
                "bundle_id": bundle_id,
                "subspace_group_candidate": raw_bundle.get(
                    "subspace_group_candidate", ""
                ),
                "subspace_space_group": raw_bundle.get(
                    "subspace_space_group", {}
                ),
                "reason": "not ready for reviewed reduced-table validation",
            })
            per_bundle.append({
                "bundle_id": bundle_id,
                "status": "blocked",
                "table_status": "not_applicable",
            })
            continue

        ready_count += 1
        try:
            table, is_time_reversal = _build_auto_table_for_bundle(
                bundle=raw_bundle,
                spinor=spinor,
                unitary_builder=build_auto_canonical_reduced_ebr_table,
                time_reversal_builder=(
                    build_auto_time_reversal_reduced_ebr_table
                ),
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            excluded.append({
                "bundle_id": bundle_id,
                "subspace_group_candidate": raw_bundle.get(
                    "subspace_group_candidate", ""
                ),
                "subspace_space_group": raw_bundle.get(
                    "subspace_space_group", {}
                ),
                "reason": f"auto reduced-table build failed: {reason}",
            })
            per_bundle.append({
                "bundle_id": bundle_id,
                "status": "blocked",
                "table_status": "blocked",
                "reason": reason,
            })
            continue

        loaded_count += 1
        table_input = _auto_table_input(table, is_time_reversal)
        table_input_by_bundle[bundle_id] = dict(table_input)
        bundle_result = build_reduced_ebr_mapping(
            ebr_export_bundle={"bundles": [raw_bundle]},
            table=table,
            max_coefficient=max_coefficient,
            reduced_ebr_input=table_input,
            cprime_validation_context=cprime_validation_context,
        )
        before = len(solutions) + len(excluded)
        for solution in bundle_result.get("solutions", []):
            if not isinstance(solution, dict):
                continue
            solution["table_status"] = "loaded"
            solutions.append(solution)
        excluded.extend(
            row for row in bundle_result.get("excluded_bundles", [])
            if isinstance(row, dict)
        )
        if len(solutions) + len(excluded) == before:
            excluded.append({
                "bundle_id": bundle_id,
                "reason": "auto mapping produced no result record",
            })
        per_bundle.append({
            "bundle_id": bundle_id,
            "sg_number": table_input.get("space_group_number"),
            "expected_hsps": table_input.get("expected_hsps", []),
            "status": bundle_result.get("status", "blocked"),
            "table_status": "loaded",
        })

    status = _aggregate_mapping_status(
        solutions=solutions,
        excluded=excluded,
        input_count=len(bundles),
    )
    if loaded_count == 0:
        table_status = "blocked"
    elif loaded_count < len(bundles):
        table_status = "partial"
    else:
        table_status = "loaded"
    if tr_bundle_count == len(bundles):
        source = "auto_time_reversal_grey"
    elif tr_bundle_count:
        source = "auto_unitary_and_time_reversal"
    else:
        source = "auto_canonical"
    return {
        "schema_version": _OUTPUT_SCHEMA_VERSION,
        "status": status,
        "table_status": table_status,
        "solutions": solutions,
        "excluded_bundles": excluded,
        "solver": _SOLVER_NAME,
        "max_coefficient": int(max_coefficient),
        "interpretation": (
            "Exact reduced EBR classification using one independently "
            "validated irreptables table per canonical HSP-vector bundle."
        ),
        "reduced_ebr_input": {
            "source": source,
            "spinful": bool(spinor),
            "input_bundle_count": len(bundles),
            "reduced_table_validation_candidate_bundle_count": ready_count,
            "final_reduced_ebr_result_count": len(solutions),
            "final_mapping_excluded_bundle_count": len(excluded),
            "table_input_provenance_by_bundle": table_input_by_bundle,
        },
        "auto_canonical_bundles": per_bundle,
    }


def _build_auto_table_for_bundle(
    *,
    bundle: dict,
    spinor: bool,
    unitary_builder,
    time_reversal_builder,
) -> tuple[dict, bool]:
    ssg = bundle.get("subspace_space_group", {})
    if not isinstance(ssg, dict) or not ssg:
        raise ValueError("missing subspace_space_group")
    sg_number = ssg.get("candidate_space_group_number")
    if not _is_positive_int(sg_number):
        raise ValueError("missing or invalid candidate_space_group_number")
    irreps_by_kpoint = bundle.get("irreps_by_kpoint", {})
    expected_hsps = bundle.get("expected_hsps", [])
    if not isinstance(irreps_by_kpoint, dict) or not irreps_by_kpoint:
        raise ValueError("missing irreps_by_kpoint")
    if not isinstance(expected_hsps, list) or not expected_hsps:
        raise ValueError("missing expected_hsps")
    group = str(bundle.get("subspace_group_candidate", ""))
    is_time_reversal = (
        bundle.get("problem_kind") == "valley_orbit_reduced_ebr"
    )
    kwargs = {
        "spinor": bool(spinor),
        "bundle_irreps_by_kpoint": irreps_by_kpoint,
        "expected_hsps": expected_hsps,
        "subspace_group_candidate": group,
        "subspace_space_group": ssg,
    }
    if not is_time_reversal:
        return unitary_builder(
            subspace_sg_number=int(sg_number), **kwargs
        ), False
    time_reversal = bundle.get("time_reversal", {})
    grey_bns = (
        time_reversal.get("grey_bns_number")
        if isinstance(time_reversal, dict) else None
    )
    if not isinstance(grey_bns, str) or not grey_bns:
        raise ValueError("valley-orbit bundle has no reviewed grey BNS number")
    return time_reversal_builder(
        unitary_space_group_number=int(sg_number),
        grey_bns_number=grey_bns,
        **kwargs,
    ), True


def _auto_table_input(table: dict, is_time_reversal: bool) -> dict:
    return _table_provenance_for_output(
        table,
        source=(
            "auto_time_reversal_grey"
            if is_time_reversal else "auto_canonical"
        ),
    )


# ---------------------------------------------------------------------------
def _count_irreps(
    bundle_irreps: dict[str, list[str]],
    table_irreps: list[str],
) -> list[int] | None:
    """Count bundle labels against table irreps without HSP-only fallback.

    Exact table labels are preferred. If one side contains an operation suffix
    such as ``:op3`` and the other side does not, a unique suffix-stripped match
    is allowed. Ambiguous operation-suffix matches are rejected.
    """
    table_index = {label: idx for idx, label in enumerate(table_irreps)}
    base_to_indices: dict[str, list[int]] = {}
    for idx, label in enumerate(table_irreps):
        base = _strip_operation_suffix(label)
        base_to_indices.setdefault(base, []).append(idx)

    counts = [0 for _ in table_irreps]
    saw_label = False
    for kp, labels in bundle_irreps.items():
        if not isinstance(labels, list):
            return None
        for label in labels:
            saw_label = True
            key = _bundle_irrep_key(str(kp), str(label))
            idx = _resolve_table_irrep_index(key, table_index, base_to_indices)
            if idx is None:
                return None
            counts[idx] += 1

    if sum(counts) == 0 and saw_label:
        return None  # couldn't resolve
    return counts


def _bundle_irrep_key(kpoint: str, label: str) -> str:
    return label if ":" in label else f"{kpoint}:{label}"


def _strip_operation_suffix(key: str) -> str:
    base, sep, suffix = key.rpartition(":")
    if sep and suffix.startswith("op") and len(suffix) > 2:
        return base
    return key


def _resolve_table_irrep_index(
    key: str,
    table_index: dict[str, int],
    base_to_indices: dict[str, list[int]],
) -> int | None:
    if key in table_index:
        return table_index[key]

    base = _strip_operation_suffix(key)
    if base != key and base in table_index:
        return table_index[base]

    candidates = base_to_indices.get(key, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def _build_per_kpoint_provenance(bundle: dict) -> dict[str, object] | None:
    """Extract per-HSP/per-irrep compact provenance from bundle records.

    Returns ``irrep_source_provenance_by_kpoint`` keyed by HSP label,
    preserving all contributing irrep records so the reduced EBR audit
    trail covers the full sampled HSP basis.
    """
    records = bundle.get("irrep_records_by_kpoint", {})
    if not isinstance(records, dict) or not records:
        return None
    by_kpoint: dict[str, list[dict[str, object]]] = {}
    for kp, kp_records in sorted(records.items()):
        if not isinstance(kp_records, list):
            continue
        kp_entries: list[dict[str, object]] = []
        for rec in kp_records:
            if not isinstance(rec, dict):
                continue
            prov = rec.get("irrep_source_provenance")
            if not isinstance(prov, dict) or not prov:
                continue
            entry: dict[str, object] = {
                "matched_irrep": rec.get("matched_irrep", ""),
                "irrep_multiplicity": rec.get("irrep_multiplicity", 1),
            }
            for key in (
                "source_hsp_label", "source_table_sg_number",
                "source_table_spinor",
                "valley_preserving_operation_ids",
                "source_table_operation_indices",
                "operation_mapping_provenance",
                "standard_setting_hsp_mapping",
            ):
                if key in prov:
                    entry[key] = prov[key]
            kp_entries.append(entry)
        if kp_entries:
            by_kpoint[str(kp)] = kp_entries
    if not by_kpoint:
        return None
    return {"irrep_source_provenance_by_kpoint": by_kpoint}


def _per_kpoint_prov(bundle: dict) -> dict[str, object] | None:
    """Shorthand: return per-kpoint provenance dict if available."""
    result = _build_per_kpoint_provenance(bundle)
    return result.get("irrep_source_provenance_by_kpoint", None) if result else None


def _status(status: str, reason: str,
            reduced_ebr_input: dict | None = None) -> dict:
    result: dict = {
        "schema_version": _OUTPUT_SCHEMA_VERSION,
        "status": status,
        "table_status": "not_applicable",
        "solutions": [],
        "excluded_bundles": [],
        "solver": _SOLVER_NAME,
        "interpretation": reason,
    }
    if reduced_ebr_input is not None:
        result["reduced_ebr_input"] = dict(reduced_ebr_input)
    return result


def _aggregate_mapping_status(
    *,
    solutions: list[dict],
    excluded: list[dict],
    input_count: int,
) -> str:
    """Aggregate one final status without losing blocked or indeterminate rows."""
    if input_count == 0:
        return "not_evaluated"
    if excluded:
        return "partial" if solutions else "blocked"
    statuses = {str(solution.get("status", "")) for solution in solutions}
    if not statuses:
        return "not_evaluated"
    if statuses == {"solved_exact"}:
        return "solved_exact"
    if statuses == {"no_exact_solution"}:
        return "no_exact_solution"
    if statuses == {"indeterminate_truncated"}:
        return "indeterminate_truncated"
    return "partial"
