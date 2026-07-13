"""Real-producer promotion helpers for reduced-EBR tests.

These helpers build a certificate identity through the ACTUAL producer
(``StandardSettingCertificate`` + ``ebr_problem_instances._certificate_identity``)
and let the promotion validator derive the table standard setting
independently from spglib.  Nothing here copies a setting from the bundle
into the table, and no hand-written certificate vocabulary is used — the
serializer emits the production field names/enum values verbatim.

For a group whose international number does not map to a unique spglib Hall
setting, ``real_certificate_identity`` returns None; the caller must then
expect a ``table_standard_setting_unresolved`` (table) or convention blocker.
"""

from __future__ import annotations

from valleyscope.analysis.standard_setting_kmap import (
    build_standard_setting_certificate,
)
from valleyscope.analysis.ebr_problem_instances import _certificate_identity
from valleyscope.analysis.reduced_ebr_mapping import (
    _derive_table_standard_setting,
)


def real_certificate_identity(sg_number: int, sg_symbol: str) -> dict | None:
    """Build a real validated primitive certificate identity via the producer.

    Uses the spglib-canonical Hall setting for ``sg_number``.  Returns None
    when the SG number has no unique standard setting (the promotion path is
    then legitimately unresolved and cannot be forced).
    """
    setting = _derive_table_standard_setting(int(sg_number))
    if setting is None or setting["centering_type"] != "P":
        return None
    cert = build_standard_setting_certificate(
        standard_match={
            "number": int(sg_number),
            "international_short": sg_symbol,
            "hall_number": setting["hall_number"],
            "hall_symbol": setting["hall_symbol"],
            "operation_ids": [0, 1, 2],
        },
        validation_status="validated",
        parent_basis_operation_ids=[0, 1, 2],
        parent_k_frac=[0.0, 0.0, 0.0],
        resolved_hsp_label="GM1",
    )
    # Field assignments mirror the producer's validated primitive direct-match
    # branch in standard_setting_kmap.resolve_standard_setting_hsp_label.
    cert.operation_mapping_status = "not_attempted"
    cert.centering_status = "primitive_direct_match"
    cert.primitive_conventional_relation = "direct_coordinate_match"
    cert.translation_validation_status = "passed"
    candidate = {
        "subspace_space_group": {
            "candidate_space_group_number": int(sg_number),
            "candidate_space_group_symbol": sg_symbol,
        },
        "irrep_source_provenance": {
            "standard_setting_hsp_mapping": {
                "standard_setting_certificate": cert.to_dict(),
            },
        },
    }
    return _certificate_identity([candidate])


def real_promotion(export_bundle: dict, table: dict) -> dict | None:
    """Attach a real validated certificate + spin evidence to each bundle.

    The table standard setting is NOT copied here — the validator derives it
    independently from spglib.  This only ensures the table provenance carries
    the six trusted fields and that each bundle carries a producer-built
    certificate and consistent per-record spin evidence.

    Returns the (mutated) export bundle, or None when the table SG number has
    no unique standard setting so no valid promotion is possible.
    """
    prov = table.get("provenance")
    if not isinstance(prov, dict):
        prov = {}
        table["provenance"] = prov
    prov.setdefault("data_source", "irreptables")
    prov.setdefault("package", "irreptables")
    prov.setdefault("package_version", "0.0.test")
    prov.setdefault("valleyscope_reduction", "sampled_hsp_valley_preserving")
    spinful = prov.get("spinful")
    if not isinstance(spinful, bool):
        spinful = False
        prov["spinful"] = spinful
    sg = prov.get("space_group_number")
    symbol = str(table.get("subspace_group_candidate", ""))
    if not (isinstance(sg, int) and not isinstance(sg, bool) and sg > 0):
        return None
    cert_identity = real_certificate_identity(int(sg), symbol)
    if cert_identity is None:
        return None
    for b in export_bundle.get("bundles", []):
        if not isinstance(b, dict):
            continue
        b["certificate_identity"] = dict(cert_identity)
        b["subspace_sg_number"] = int(sg)
        _inject_spin_records(b, spinful)
    return export_bundle


def _inject_spin_records(bundle: dict, spinful: bool) -> None:
    """Ensure every irrep row carries a consistent source_table_spinor."""
    irreps = bundle.get("irreps_by_kpoint", {})
    records = bundle.get("irrep_records_by_kpoint")
    if not isinstance(records, dict):
        records = {}
        bundle["irrep_records_by_kpoint"] = records
    if isinstance(irreps, dict):
        for kp, labels in irreps.items():
            row = records.get(kp)
            if not isinstance(row, list) or not row:
                row = [{"matched_irrep": (labels[0] if labels else ""),
                        "irrep_multiplicity": 1,
                        "irrep_source_provenance": {}}]
                records[kp] = row
            for rec in row:
                if isinstance(rec, dict):
                    p = rec.get("irrep_source_provenance")
                    if not isinstance(p, dict):
                        p = {}
                        rec["irrep_source_provenance"] = p
                    p["source_table_spinor"] = spinful


# Test-only symbol -> international SG number for groups with a UNIQUE spglib
# standard setting.  Used only to backfill an absent table sg_number so the
# real trust chain can run; it is not production logic.
_REAL_UNIQUE_SG = {"P3": 143, "P4": 75, "P1": 1, "P6": 168, "I4": 79}


def attach_promotion(export_bundle: dict, table: dict):
    """Route an export bundle + table through the REAL trust chain.

    Backfills the table SG number from the group symbol when a unique spglib
    setting exists, then attaches a producer-built certificate via
    ``real_promotion``.  The table standard setting is never copied from the
    bundle.  Returns (export_bundle, table).  When the group has no unique
    setting, the bundle is left without a valid certificate so the validator
    blocks (the caller must expect that).
    """
    prov = table.get("provenance")
    if not isinstance(prov, dict):
        prov = {}
        table["provenance"] = prov
    sg = prov.get("space_group_number")
    if not (isinstance(sg, int) and not isinstance(sg, bool) and sg > 0):
        symbol = str(table.get("subspace_group_candidate", ""))
        if symbol in _REAL_UNIQUE_SG:
            prov["space_group_number"] = _REAL_UNIQUE_SG[symbol]
    real_promotion(export_bundle, table)
    return export_bundle, table
