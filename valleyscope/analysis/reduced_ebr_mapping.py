"""Reduced EBR mapping: exact integer decomposition from export bundles.

Loads a user-supplied reduced-EBR table, validates it, and performs exact
integer matching via Smith normal form plus bounded nonnegative search.  No
built-in tables are provided; real-material EBR claims require an explicit
table file.
"""

from __future__ import annotations

import json
from pathlib import Path

_REQUIRED_TABLE_KEYS = {"schema_version", "subspace_group_candidate",
                         "expected_hsps", "irreps", "ebrs"}
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
def _derive_table_standard_setting(sg_number: int) -> dict | None:
    """Canonical standard setting for an international SG number via spglib.

    Independent of any bundle: reads only spglib's crystallographic database.
    Returns hall_number/hall_symbol/centering_type/space_group_symbol when the
    international number maps to a UNIQUE Hall setting.  Returns None when
    several settings (origin/axis/cell choices) exist and irreptables does not
    expose which one its data uses — the caller must then block with
    ``table_standard_setting_unresolved`` rather than synthesize one.
    """
    try:
        import spglib
    except Exception:
        return None
    matches = []
    for h in range(1, 531):
        t = spglib.get_spacegroup_type(h)
        if t is not None and int(t.number) == int(sg_number):
            matches.append((h, t))
    if len(matches) != 1:
        return None
    h, t = matches[0]
    return {
        "hall_number": int(h),
        "hall_symbol": str(t.hall_symbol),
        "centering_type": _centering_from_hall_symbol(t.hall_symbol),
        "space_group_number": int(t.number),
        "space_group_symbol": str(t.international_short),
    }


# ---------------------------------------------------------------------------
# Production bundle promotion: sampled_basis → validated_basis
# ---------------------------------------------------------------------------

def promote_bundle_for_solve(*, bundle: dict, table: dict) -> dict:
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
        "sg_symbol_check": "not_attempted",
        "sg_number_check": "not_attempted",
        "certificate_check": "not_attempted",
        "certificate_consistency_check": "not_attempted",
        "cert_sg_consistency_check": "not_attempted",
        "affine_setting_check": "not_attempted",
        "hall_setting_check": "not_attempted",
        "spin_convention_check": "not_attempted",
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

    # ---- A2. Independent table standard-setting evidence (spglib) ----
    table_setting = (
        _derive_table_standard_setting(int(table_sg_num))
        if _is_positive_int(table_sg_num) else None
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
        _validate_setting(cert_id, blockers, report)
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
    if isinstance(irreps_by_kp, dict) and isinstance(table_irreps, list):
        counts = _count_irreps(irreps_by_kp, table_irreps)
        if counts is None:
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

    table_provenance = {
        "data_source": data_source,
        "package": package,
        "package_version": package_version,
        "space_group_number": table_sg_num,
        "spinful": table_spinful,
        "valleyscope_reduction": valleyscope_reduction,
        "independent_setting_identity": dict(table_setting) if table_setting else None,
        "setting_source": "spglib.get_spacegroup_type",
    }

    promoted = not blockers
    state = "validated_basis" if promoted else "sampled_basis"
    promoted_bundle: dict | None = None
    if promoted:
        promoted_bundle = dict(bundle)
        promoted_bundle["ready_for_external_solver"] = True
        promoted_bundle["hsp_basis_status"] = state
        promoted_bundle["promotion_provenance"] = {
            "source": "promote_bundle_for_solve",
            "validation_report": dict(report),
            "table_provenance": dict(table_provenance),
            "certificate_identity": dict(cert_id),
        }

    return {
        "promoted": promoted,
        "promoted_bundle": promoted_bundle,
        "blocker_reasons": blockers,
        "validation_report": report,
        "canonical_state": state,
        "table_provenance": table_provenance,
        "certificate_identity": dict(cert_id),
    }


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _normalized_hsp_set(value: object) -> set[str] | None:
    if not isinstance(value, list):
        return set() if value is None else None
    if not all(isinstance(h, str) and h for h in value):
        return None
    labels = list(value)
    if len(set(labels)) != len(labels):
        return None
    return set(labels)


_RECOGNIZED_CENTERINGS = frozenset({"P", "A", "B", "C", "I", "F", "R"})


def _norm_symbol(value: object) -> str:
    """Whitespace-insensitive normalized international symbol."""
    return "".join(str(value or "").split())


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



def _validate_setting(cert_id, blockers, report):
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
    transform = cert_id.get("normalized_direct_transform")
    centering_vectors = cert_id.get("normalized_centering_vectors")
    op_status = str(cert_id.get("operation_mapping_status", ""))
    affine_status = str(cert_id.get("affine_validation_status", ""))
    transform_ok = isinstance(transform, list) and len(transform) == 3 and all(
        isinstance(row, list) and len(row) == 3 for row in transform)
    vectors_ok = isinstance(centering_vectors, list) and bool(centering_vectors)
    if transform_ok and vectors_ok and validation_status == "validated" \
            and relation in _CENTERED_RELATIONS \
            and op_status == _OP_MAPPING_PASSED \
            and affine_status == _AFFINE_PASSED:
        report["affine_setting_check"] = "passed"
    else:
        blockers.append(_blocker("centered_affine_evidence_invalid",
            "centered setting requires a validated transform + centering "
            "vectors + passed operation/affine validation with producer "
            f"vocabulary; got transform_ok={transform_ok} "
            f"vectors_ok={vectors_ok} validation_status={validation_status!r} "
            f"relation={relation!r} operation_mapping_status={op_status!r} "
            f"affine_validation_status={affine_status!r}"))
        report["affine_setting_check"] = "failed"


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

    bundles = ebr_export_bundle.get("bundles", [])

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

    table_group = str(table.get("subspace_group_candidate", ""))
    table_irreps = table["irreps"]
    table_ebrs = table["ebrs"]
    n_irreps = len(table_irreps)
    table_expected = set(table.get("expected_hsps", []))

    solutions: list[dict] = []
    excluded: list[dict] = []

    for bundle in bundles:
        if not isinstance(bundle, dict):
            continue
        # Every bundle must pass the same validation against the actual
        # table used for this solve.  Pre-existing readiness flags are
        # not trusted evidence and never bypass validation.
        is_validation_candidate = (
            bundle.get("ready_for_reduced_table_validation") is True
        )
        is_premarked_ready = (
            bundle.get("ready_for_external_solver") is True
        )
        if is_validation_candidate or is_premarked_ready:
            promo = promote_bundle_for_solve(bundle=bundle, table=table)
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
                    "table_provenance": promo["table_provenance"],
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
                "reason": "not ready for external solver",
            })
            continue

        bundle_group = str(bundle.get("subspace_group_candidate", ""))
        if bundle_group != table_group:
            excluded.append({
                "bundle_id": bundle.get("bundle_id", "?"),
                "subspace_group_candidate": bundle_group,
                "subspace_space_group": bundle.get("subspace_space_group", {}),
                "irrep_source_provenance_by_kpoint": _per_kpoint_prov(bundle),
                "reason": (
                    f"table group {table_group} != "
                    f"bundle group {bundle_group}"
                ),
            })
            continue

        # --- Reduced-dimensional basis compatibility gate ---
        # The table and bundle must agree on the sampled HSP set.
        bundle_irreps = bundle.get("irreps_by_kpoint", {})
        actual_hsps = set(bundle_irreps) if isinstance(bundle_irreps, dict) else set()

        bundle_expected = bundle.get("expected_hsps")
        if bundle_expected is None:
            # Legacy bundle without declared expected_hsps: derive from
            # irreps_by_kpoint keys.
            bundle_expected_set = actual_hsps
        elif (
            isinstance(bundle_expected, list)
            and all(isinstance(h, str) and h for h in bundle_expected)
            and len(set(bundle_expected)) == len(bundle_expected)
        ):
            bundle_expected_set = set(bundle_expected)
        else:
            excluded.append({
                "bundle_id": bundle.get("bundle_id", "?"),
                "subspace_group_candidate": bundle.get("subspace_group_candidate", ""),
                "subspace_space_group": bundle.get("subspace_space_group", {}),
                "irrep_source_provenance_by_kpoint": _per_kpoint_prov(bundle),
                "reason": (
                    "malformed expected_hsps: expected a unique list of "
                    "non-empty HSP labels"
                ),
            })
            continue

        if bundle_expected_set != table_expected:
            excluded.append({
                "bundle_id": bundle.get("bundle_id", "?"),
                "subspace_group_candidate": bundle.get("subspace_group_candidate", ""),
                "subspace_space_group": bundle.get("subspace_space_group", {}),
                "irrep_source_provenance_by_kpoint": _per_kpoint_prov(bundle),
                "reason": (
                    f"expected_hsps mismatch: "
                    f"table has {sorted(table_expected)}, "
                    f"bundle has {sorted(bundle_expected_set)}"
                ),
            })
            continue

        if actual_hsps != table_expected:
            excluded.append({
                "bundle_id": bundle.get("bundle_id", "?"),
                "subspace_group_candidate": bundle.get("subspace_group_candidate", ""),
                "subspace_space_group": bundle.get("subspace_space_group", {}),
                "irrep_source_provenance_by_kpoint": _per_kpoint_prov(bundle),
                "reason": (
                    f"irrep HSP basis mismatch: "
                    f"table expects {sorted(table_expected)}, "
                    f"bundle irreps_by_kpoint has {sorted(actual_hsps)}"
                ),
            })
            continue
        irrep_counts = _count_irreps(bundle_irreps, table_irreps)
        if irrep_counts is None:
            excluded.append({
                "bundle_id": bundle.get("bundle_id", "?"),
                "subspace_group_candidate": bundle.get("subspace_group_candidate", ""),
                "subspace_space_group": bundle.get("subspace_space_group", {}),
                "irrep_source_provenance_by_kpoint": _per_kpoint_prov(bundle),
                "reason": "could not resolve irrep keys to table irreps",
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
            "valley": bundle.get("valley", ""),
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
                solution["table_provenance"] = promo_prov["table_provenance"]
            if isinstance(promo_prov.get("certificate_identity"), dict):
                solution["certificate_identity"] = \
                    promo_prov["certificate_identity"]
        solutions.append(solution)

    if not solutions:
        return {
            **_status("not_evaluated", "no bundles to decompose",
                      reduced_ebr_input=reduced_ebr_input),
            "solutions": [],
            "excluded_bundles": excluded,
            "table_status": "loaded" if table else "not_provided",
        }

    all_solved = all(s.get("status") == "solved_exact" for s in solutions)
    mapping_status = "solved_exact" if all_solved else "no_exact_solution"
    result = {
        "status": mapping_status,
        "mapping_status": mapping_status,
        "reduced_ebr_decomposition_status": mapping_status,
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


def _extract_bundle_irrep_provenance(bundle: dict) -> dict[str, object] | None:
    """Legacy single-record extractor — use _build_per_kpoint_provenance."""
    return _build_per_kpoint_provenance(bundle)


def _per_kpoint_prov(bundle: dict) -> dict[str, object] | None:
    """Shorthand: return per-kpoint provenance dict if available."""
    result = _build_per_kpoint_provenance(bundle)
    return result.get("irrep_source_provenance_by_kpoint", None) if result else None


def _status(status: str, reason: str,
            reduced_ebr_input: dict | None = None) -> dict:
    result: dict = {
        "status": status,
        "mapping_status": status,
        "reduced_ebr_decomposition_status": status,
        "table_status": "not_applicable",
        "solutions": [],
        "excluded_bundles": [],
        "solver": _SOLVER_NAME,
        "interpretation": reason,
    }
    if reduced_ebr_input is not None:
        result["reduced_ebr_input"] = dict(reduced_ebr_input)
    return result
