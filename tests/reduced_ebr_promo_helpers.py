"""Shared helpers to make existing reduced-EBR tests pass the fail-closed
promotion validator without the removed ``require_reviewed_table`` bypass.

``attach_promotion`` injects a self-consistent validated primitive
standard-setting certificate into every bundle and a matching setting
identity into the table provenance, so the production mapping path runs its
real physical validation instead of a disabled gate.  Hall values are
internally consistent placeholders; tests that assert crystallographic Hall
correctness live in ``test_reduced_ebr_promotion.py``.
"""

from __future__ import annotations


def validated_cert(
    *,
    sg_number: int,
    sg_symbol: str,
    hall_number: int = 1,
    hall_symbol: str = "H 1",
    centering: str = "P",
) -> dict:
    return {
        "hall_numbers": [hall_number],
        "hall_symbols": [hall_symbol],
        "centering_types": [centering],
        "certificate_validation_statuses": ["validated"],
        "any_unresolved": False,
        "distinct_setting_identities": 1,
        "sg_number": sg_number,
        "sg_symbol": sg_symbol,
        "hall_number": hall_number,
        "hall_symbol": hall_symbol,
        "centering_type": centering,
        "primitive_conventional_relation": "identity",
        "transform_provenance": "derived_affine_equivalence",
        "validation_status": "validated",
        "operation_mapping_status": "validated",
        "affine_validation_status": "validated",
    }


def setting_identity(*, sg_symbol: str, hall_number: int = 1,
                     hall_symbol: str = "H 1", centering: str = "P") -> dict:
    return {
        "hall_number": hall_number,
        "hall_symbol": hall_symbol,
        "centering_type": centering,
        "space_group_symbol": sg_symbol,
    }


def attach_promotion(
    export_bundle: dict,
    table: dict,
    *,
    sg_number: int | None = None,
    hall_number: int = 1,
    hall_symbol: str = "H 1",
    centering: str = "P",
) -> tuple[dict, dict]:
    """Make ``table`` and every bundle in ``export_bundle`` promotion-valid.

    Mutates and returns both.  Reuses the table's existing provenance where
    present so builder-produced tables keep their real ``space_group_number``,
    ``spinful``, and package version.
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
    if not (isinstance(sg, int) and not isinstance(sg, bool) and sg > 0):
        sg = sg_number if sg_number is not None else 143
        prov["space_group_number"] = sg

    sg_symbol = str(table.get("subspace_group_candidate", ""))
    prov["setting_identity"] = setting_identity(
        sg_symbol=sg_symbol, hall_number=hall_number,
        hall_symbol=hall_symbol, centering=centering)

    cert = validated_cert(
        sg_number=sg, sg_symbol=sg_symbol, hall_number=hall_number,
        hall_symbol=hall_symbol, centering=centering)
    for b in export_bundle.get("bundles", []):
        if isinstance(b, dict):
            b["certificate_identity"] = dict(cert)
            b["subspace_sg_number"] = sg
            b["spinor"] = spinful
    return export_bundle, table
