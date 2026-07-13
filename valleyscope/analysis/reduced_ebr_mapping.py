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

# Standard-setting certificate convention constants.
_TRUSTED_DATA_SOURCE = "irreptables"
_TRUSTED_PACKAGE = "irreptables"
_TRUSTED_REDUCTION = "sampled_hsp_valley_preserving"
# A primitive setting passes only when the validated certificate explicitly
# declares the primitive direct-coordinate relation.  Missing/unknown is not
# primitive.
_PRIMITIVE_DIRECT_RELATIONS = frozenset({
    "identity",
    "primitive",
    "primitive_direct",
    "primitive_direct_match",
    "primitive_conventional_identity",
})
_CENTERED_TYPES = frozenset({"A", "B", "C", "I", "F", "R"})


def _blocker(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail}


# ---------------------------------------------------------------------------
# Production bundle promotion: sampled_basis → validated_basis
# ---------------------------------------------------------------------------

def promote_bundle_for_solve(*, bundle: dict, table: dict) -> dict:
    """Validate a bundle against a reduced EBR table and promote if compatible.

    Fail-closed.  Every bundle — regardless of any pre-existing readiness
    Boolean — must pass every physical validation gate.  A pre-marked
    ``ready_for_external_solver`` flag is never trusted evidence.  Missing,
    unresolved, rejected, malformed, or mutually inconsistent evidence is a
    blocker.  There is no bypass parameter.

    Validated gates (all must pass to promote to ``validated_basis``):

    - table provenance: ``data_source == 'irreptables'``, ``package ==
      'irreptables'`` with non-empty version, positive ``space_group_number``,
      boolean ``spinful``, ``valleyscope_reduction`` exact, and a recorded
      setting identity (Hall number/symbol + centering);
    - space group number and symbol both match;
    - certificate present, ``validation_status == 'validated'``, not
      unresolved/rejected/ambiguous;
    - centering present (missing centering is *unknown*, never primitive);
    - primitive setting: certificate explicitly declares the primitive
      direct-coordinate relation; centered/nontrivial setting: validated
      normalized transform + centering vectors + passed operation/affine
      validation;
    - Hall/setting identity matches the table;
    - bundle spinor equals table ``spinful``;
    - sampled HSP basis matches the table;
    - every bundle irrep key resolves exactly and unambiguously.

    Returns a dict with ``promoted``, ``promoted_bundle``,
    ``blocker_reasons`` (structured ``{code, detail}`` dicts),
    ``validation_report``, ``canonical_state``, ``table_provenance`` and
    ``certificate_identity``.
    """
    blockers: list[dict[str, str]] = []
    report: dict[str, object] = {
        "table_provenance_check": "not_attempted",
        "sg_symbol_check": "not_attempted",
        "sg_number_check": "not_attempted",
        "certificate_check": "not_attempted",
        "centering_check": "not_attempted",
        "affine_setting_check": "not_attempted",
        "setting_identity_check": "not_attempted",
        "spin_convention_check": "not_attempted",
        "hsp_basis_check": "not_attempted",
        "irrep_basis_check": "not_attempted",
    }

    # ---- A. Table provenance (fail-closed) ----
    prov = table.get("provenance", {})
    if not isinstance(prov, dict):
        prov = {}
    data_source = str(prov.get("data_source", ""))
    package = str(prov.get("package", ""))
    package_version = str(prov.get("package_version", ""))
    table_spinful = prov.get("spinful")
    table_sg_num = prov.get("space_group_number")
    valleyscope_reduction = str(prov.get("valleyscope_reduction", ""))
    table_setting = prov.get("setting_identity")
    if not isinstance(table_setting, dict):
        table_setting = {}

    prov_ok = True
    if data_source != _TRUSTED_DATA_SOURCE:
        blockers.append(_blocker(
            "table_data_source_invalid",
            f"table data_source must be '{_TRUSTED_DATA_SOURCE}', "
            f"got {data_source!r}"))
        prov_ok = False
    if package != _TRUSTED_PACKAGE or not package_version:
        blockers.append(_blocker(
            "table_package_invalid",
            f"table package must be '{_TRUSTED_PACKAGE}' with a non-empty "
            f"package_version, got package={package!r} "
            f"version={package_version!r}"))
        prov_ok = False
    if not _is_positive_int(table_sg_num):
        blockers.append(_blocker(
            "table_sg_number_missing",
            "table provenance missing positive space_group_number"))
        prov_ok = False
    if not isinstance(table_spinful, bool):
        blockers.append(_blocker(
            "table_spinful_missing",
            "table provenance missing boolean spinful"))
        prov_ok = False
    if valleyscope_reduction != _TRUSTED_REDUCTION:
        blockers.append(_blocker(
            "table_reduction_provenance_invalid",
            f"table valleyscope_reduction must be '{_TRUSTED_REDUCTION}', "
            f"got {valleyscope_reduction!r}"))
        prov_ok = False
    table_hall = table_setting.get("hall_number")
    table_hall_symbol = str(table_setting.get("hall_symbol", ""))
    table_centering = str(table_setting.get("centering_type", ""))
    if not _is_positive_int(table_hall) or not table_hall_symbol \
            or not table_centering:
        blockers.append(_blocker(
            "table_setting_identity_missing",
            "table provenance missing setting_identity "
            "(hall_number, hall_symbol, centering_type)"))
        prov_ok = False
    report["table_provenance_check"] = "passed" if prov_ok else "failed"

    # ---- B. SG number + symbol ----
    bundle_sg = str(bundle.get("subspace_group_candidate", ""))
    table_sg = str(table.get("subspace_group_candidate", ""))
    if bundle_sg and bundle_sg == table_sg:
        report["sg_symbol_check"] = "passed"
    else:
        blockers.append(_blocker(
            "sg_symbol_mismatch",
            f"SG symbol mismatch: bundle {bundle_sg!r} != table {table_sg!r}"))
        report["sg_symbol_check"] = "failed"

    bundle_sg_num = bundle.get("subspace_sg_number")
    if not _is_positive_int(bundle_sg_num):
        blockers.append(_blocker(
            "sg_number_missing",
            "bundle missing positive subspace_sg_number"))
        report["sg_number_check"] = "failed"
    elif not _is_positive_int(table_sg_num):
        report["sg_number_check"] = "failed"  # already blocked in provenance
    elif int(bundle_sg_num) != int(table_sg_num):
        blockers.append(_blocker(
            "sg_number_mismatch",
            f"SG number mismatch: bundle {bundle_sg_num} != table "
            f"{table_sg_num}"))
        report["sg_number_check"] = "failed"
    else:
        report["sg_number_check"] = "passed"

    # ---- C. Certificate presence + validation status ----
    cert_id = bundle.get("certificate_identity", {})
    if not isinstance(cert_id, dict) or not cert_id:
        blockers.append(_blocker(
            "certificate_missing",
            "bundle has no certificate_identity"))
        report["certificate_check"] = "blocked"
        report["centering_check"] = "blocked"
        report["affine_setting_check"] = "blocked"
        report["setting_identity_check"] = "blocked"
        cert_id = {}
    else:
        val_statuses = cert_id.get("certificate_validation_statuses", [])
        if not isinstance(val_statuses, list):
            val_statuses = []
        validation_status = str(cert_id.get("validation_status", ""))
        distinct = cert_id.get("distinct_setting_identities")
        cert_ok = True
        if "rejected" in val_statuses:
            blockers.append(_blocker(
                "certificate_rejected",
                "certificate_identity contains a rejected validation status"))
            cert_ok = False
        # Fail-closed: only an explicit ``any_unresolved is False`` (fully
        # resolved) passes.  Missing, null, or truthy blocks.
        if cert_id.get("any_unresolved", True) is not False:
            blockers.append(_blocker(
                "certificate_unresolved",
                "certificate_identity is unresolved/not_evaluated"))
            cert_ok = False
        if validation_status != "validated":
            blockers.append(_blocker(
                "certificate_not_validated",
                f"certificate validation_status must be 'validated', got "
                f"{validation_status!r}"))
            cert_ok = False
        if isinstance(distinct, int) and not isinstance(distinct, bool) \
                and distinct > 1:
            blockers.append(_blocker(
                "certificate_ambiguous_setting",
                f"{distinct} distinct setting identities in one instance"))
            cert_ok = False
        report["certificate_check"] = "passed" if cert_ok else "failed"

        # ---- D. Centering + affine setting ----
        _validate_setting(cert_id, blockers, report)

        # ---- E. Setting identity match with table ----
        _validate_setting_identity_match(
            cert_id, table_hall, table_hall_symbol, table_centering,
            blockers, report)

    # ---- F. Spin convention (bundle spinor vs table spinful) ----
    bundle_spin = _bundle_spin_convention(bundle, cert_id)
    if not isinstance(table_spinful, bool):
        report["spin_convention_check"] = "failed"  # already blocked above
    elif bundle_spin is None:
        blockers.append(_blocker(
            "spin_convention_missing",
            "bundle spin convention (spinor) not recorded"))
        report["spin_convention_check"] = "blocked"
    elif bundle_spin != table_spinful:
        blockers.append(_blocker(
            "spin_convention_mismatch",
            f"bundle spinor={bundle_spin} != table spinful={table_spinful}"))
        report["spin_convention_check"] = "failed"
    else:
        report["spin_convention_check"] = "passed"

    # ---- G. HSP basis ----
    table_hsps = _normalized_hsp_set(table.get("expected_hsps"))
    irreps_by_kp = bundle.get("irreps_by_kpoint", {})
    actual_hsps = set(irreps_by_kp) if isinstance(irreps_by_kp, dict) else set()
    bundle_expected_raw = bundle.get("expected_hsps")
    if bundle_expected_raw is None:
        # Legacy bundle without a declared basis: derive from irrep keys.
        bundle_hsps: set[str] | None = actual_hsps
    else:
        bundle_hsps = _normalized_hsp_set(bundle_expected_raw)
    if bundle_hsps is None:
        blockers.append(_blocker(
            "hsp_basis_malformed",
            "bundle expected_hsps is not a unique list of non-empty labels"))
        report["hsp_basis_check"] = "failed"
    elif bundle_hsps != table_hsps or actual_hsps != table_hsps:
        blockers.append(_blocker(
            "hsp_basis_mismatch",
            f"expected_hsps mismatch: table {sorted(table_hsps)}, bundle "
            f"expected {sorted(bundle_hsps)}, actual {sorted(actual_hsps)}"))
        report["hsp_basis_check"] = "failed"
    else:
        report["hsp_basis_check"] = "passed"

    # ---- H. Irrep keys resolve exactly and unambiguously ----
    table_irreps = table.get("irreps", [])
    if isinstance(irreps_by_kp, dict) and isinstance(table_irreps, list):
        counts = _count_irreps(irreps_by_kp, table_irreps)
        if counts is None:
            blockers.append(_blocker(
                "irrep_key_unresolved",
                "could not resolve irrep keys: bundle irrep keys do not "
                "resolve exactly/unambiguously to table irreps"))
            report["irrep_basis_check"] = "failed"
        else:
            report["irrep_basis_check"] = "passed"
    else:
        blockers.append(_blocker(
            "irrep_basis_malformed",
            "bundle irreps_by_kpoint or table irreps malformed"))
        report["irrep_basis_check"] = "failed"

    table_provenance = {
        "data_source": data_source,
        "package": package,
        "package_version": package_version,
        "space_group_number": table_sg_num,
        "spinful": table_spinful,
        "valleyscope_reduction": valleyscope_reduction,
        "setting_identity": dict(table_setting),
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
    """Return the HSP label set, or None if malformed."""
    if not isinstance(value, list):
        return set() if value is None else None
    if not all(isinstance(h, str) and h for h in value):
        return None
    labels = list(value)
    if len(set(labels)) != len(labels):
        return None
    return set(labels)


def _validate_setting(
    cert_id: dict,
    blockers: list[dict[str, str]],
    report: dict[str, object],
) -> None:
    """Validate centering presence and primitive/centered affine evidence."""
    raw_centering = cert_id.get("centering_type")
    # Missing centering evidence is unknown, not primitive.  ``None`` and an
    # empty string are both treated as missing, never as "P".
    centering = "" if raw_centering is None else str(raw_centering)
    centering_types = cert_id.get("centering_types", [])
    if not isinstance(centering_types, list):
        centering_types = []
    # Missing centering evidence is unknown, not primitive.
    if not centering:
        blockers.append(_blocker(
            "centering_missing",
            "certificate has no centering_type; missing centering is unknown, "
            "not primitive"))
        report["centering_check"] = "blocked"
        report["affine_setting_check"] = "blocked"
        return
    if len(centering_types) > 1:
        blockers.append(_blocker(
            "centering_ambiguous",
            f"multiple centering types in one instance: {centering_types}"))
        report["centering_check"] = "failed"
        report["affine_setting_check"] = "failed"
        return
    report["centering_check"] = "passed"

    validation_status = str(cert_id.get("validation_status", ""))
    if centering == "P":
        relation = str(cert_id.get("primitive_conventional_relation", ""))
        if validation_status == "validated" \
                and relation in _PRIMITIVE_DIRECT_RELATIONS:
            report["affine_setting_check"] = "passed"
        else:
            blockers.append(_blocker(
                "primitive_relation_not_declared",
                "primitive setting requires a validated certificate that "
                "explicitly declares the primitive direct-coordinate relation; "
                f"got validation_status={validation_status!r} "
                f"relation={relation!r}"))
            report["affine_setting_check"] = "failed"
        return

    # Centered / nontrivial setting.
    if centering not in _CENTERED_TYPES:
        blockers.append(_blocker(
            "centering_unrecognized",
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
            and op_status == "validated" and affine_status == "validated":
        report["affine_setting_check"] = "passed"
    else:
        blockers.append(_blocker(
            "centered_affine_evidence_invalid",
            "centered setting requires a validated normalized transform, "
            "centering vectors, and passed operation/affine validation; got "
            f"transform_ok={transform_ok} vectors_ok={vectors_ok} "
            f"validation_status={validation_status!r} "
            f"operation_mapping_status={op_status!r} "
            f"affine_validation_status={affine_status!r}"))
        report["affine_setting_check"] = "failed"


def _validate_setting_identity_match(
    cert_id: dict,
    table_hall: object,
    table_hall_symbol: str,
    table_centering: str,
    blockers: list[dict[str, str]],
    report: dict[str, object],
) -> None:
    """Compare the bundle certificate setting identity with the table."""
    bundle_hall = cert_id.get("hall_number")
    bundle_hall_symbol = str(cert_id.get("hall_symbol", ""))
    bundle_centering = str(cert_id.get("centering_type", ""))
    if not _is_positive_int(bundle_hall) or not bundle_hall_symbol:
        blockers.append(_blocker(
            "setting_identity_missing",
            "bundle certificate missing Hall setting identity"))
        report["setting_identity_check"] = "blocked"
        return
    if not _is_positive_int(table_hall) or not table_hall_symbol \
            or not table_centering:
        report["setting_identity_check"] = "failed"  # table already blocked
        return
    if int(bundle_hall) != int(table_hall) \
            or bundle_hall_symbol != table_hall_symbol \
            or bundle_centering != table_centering:
        blockers.append(_blocker(
            "setting_identity_mismatch",
            "Hall/setting identity mismatch: bundle "
            f"(hall={bundle_hall}, symbol={bundle_hall_symbol!r}, "
            f"centering={bundle_centering!r}) != table "
            f"(hall={table_hall}, symbol={table_hall_symbol!r}, "
            f"centering={table_centering!r})"))
        report["setting_identity_check"] = "failed"
    else:
        report["setting_identity_check"] = "passed"


def _bundle_spin_convention(bundle: dict, cert_id: dict) -> bool | None:
    """Resolve the bundle spin convention from certificate/records.

    Priority: explicit ``spinor``/``spinful`` on the bundle or the certificate
    identity, then a consistent ``source_table_spinor`` across all irrep source
    provenance records.  Returns None when absent or inconsistent.
    """
    for holder in (bundle, cert_id):
        if isinstance(holder, dict):
            for key in ("spinor", "spinful", "source_table_spinor"):
                v = holder.get(key)
                if isinstance(v, bool):
                    return v
    seen: set[bool] = set()
    records = bundle.get("irrep_records_by_kpoint", {})
    if isinstance(records, dict):
        for kp_records in records.values():
            if not isinstance(kp_records, list):
                continue
            for rec in kp_records:
                if not isinstance(rec, dict):
                    continue
                p = rec.get("irrep_source_provenance")
                if isinstance(p, dict) and isinstance(
                        p.get("source_table_spinor"), bool):
                    seen.add(p["source_table_spinor"])
    if len(seen) == 1:
        return next(iter(seen))
    return None


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
