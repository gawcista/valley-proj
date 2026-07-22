"""Real-affine primitive certificate helpers for reduced-EBR tests.

Primitive standard-setting certificates are produced by the ACTUAL resolver
(``resolve_standard_setting_hsp_label``) fed a REAL ``StandardIrrepTable`` and a
complete detected-operation set built from spglib standard operations, so the
generic affine ``{R | tau}`` operation-equivalence gate actually runs.  Nothing
is mutated after the resolver returns; there is no Gamma-only coordinate stub.

Only primitive space groups with a unique spglib Hall setting yield a validated
certificate here.  Centered/multiple-Hall settings remain Phase E and produce
None (the caller must then expect a fail-closed blocker).
"""

from __future__ import annotations

import numpy as np

from valleyscope.analysis.standard_setting_kmap import (
    resolve_standard_setting_hsp_label,
)
from valleyscope.analysis.ebr_problem_instances import _certificate_identity
from valleyscope.analysis.reduced_ebr_mapping import (
    _derive_table_standard_setting,
)


def _detected_standard_operations(hall_number: int):
    """Complete detected-operation set from spglib standard operations."""
    import spglib
    sym = spglib.get_symmetry_from_database(int(hall_number))
    if sym is None:
        return None
    rots = sym["rotations"]
    trans = sym["translations"]
    return [{
        "operation_id": i,
        "rotation_frac": np.asarray(rots[i], dtype=float).tolist(),
        "translation_frac": np.asarray(trans[i], dtype=float).tolist(),
    } for i in range(len(rots))]


def real_primitive_certificate_dict(sg_number: int, sg_symbol: str,
                                    *, spinor: bool = False) -> dict | None:
    """Raw resolver certificate for a primitive SG via a real table + real
    spglib affine operations, with no post-mutation."""
    setting = _derive_table_standard_setting(int(sg_number))
    if setting is None or setting["centering_type"] != "P":
        return None
    from valleyscope.irreps.tables import load_standard_irrep_table
    try:
        table = load_standard_irrep_table(int(sg_number), spinor=bool(spinor))
    except Exception:
        return None
    detected = _detected_standard_operations(setting["hall_number"])
    if not detected:
        return None
    standard_match = {
        "number": int(sg_number),
        "international_short": sg_symbol,
        "hall_number": setting["hall_number"],
        "hall_symbol": setting["hall_symbol"],
        "operation_ids": [op["operation_id"] for op in detected],
    }
    _, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=table,
        standard_match=standard_match,
        detected_operations=detected,
    )
    if blocker is not None:
        return None
    cert = prov.get("standard_setting_certificate")
    return dict(cert) if isinstance(cert, dict) else None


def real_primitive_certificate_identity(sg_number: int, sg_symbol: str,
                                        *, spinor: bool = False) -> dict | None:
    """Serialized certificate identity from a real affine primitive cert."""
    cert = real_primitive_certificate_dict(sg_number, sg_symbol, spinor=spinor)
    if cert is None:
        return None
    candidate = {
        "subspace_space_group": {
            "candidate_space_group_number": int(sg_number),
            "candidate_space_group_symbol": sg_symbol,
        },
        "irrep_source_provenance": {
            "standard_setting_hsp_mapping": {
                "standard_setting_certificate": cert,
            },
        },
    }
    return _certificate_identity([candidate])


def add_real_certificate_to_candidates(ebr_input_candidates: dict,
                                       sg_number: int, sg_symbol: str,
                                       *, spinor: bool = False) -> dict:
    """Inject a REAL affine primitive certificate + spin into each input
    candidate so the workflow serializes a validated bundle identity naturally
    (no bundle injection)."""
    cert = real_primitive_certificate_dict(sg_number, sg_symbol, spinor=spinor)
    if cert is None:
        return ebr_input_candidates
    for c in ebr_input_candidates.get("candidates", []):
        if not isinstance(c, dict):
            continue
        prov = c.get("irrep_source_provenance")
        if not isinstance(prov, dict):
            prov = {}
            c["irrep_source_provenance"] = prov
        classification = c.get("projected_hsp_classification")
        source_hsp = (
            classification.get("source_hsp_label")
            if isinstance(classification, dict) else None
        )
        if not isinstance(source_hsp, str) or not source_hsp:
            source_hsp = c.get("kpoint")
        if isinstance(source_hsp, str) and source_hsp:
            if not isinstance(
                prov.get("source_hsp_label"), str
            ) or not prov.get("source_hsp_label"):
                prov["source_hsp_label"] = source_hsp
        c.setdefault(
            "source",
            "fixture/"
            f"{c.get('valley', 'unknown')}/{source_hsp or 'unknown'}",
        )
        kmap = prov.get("standard_setting_hsp_mapping")
        if not isinstance(kmap, dict):
            kmap = {}
            prov["standard_setting_hsp_mapping"] = kmap
        kmap["standard_setting_certificate"] = dict(cert)
        prov["source_table_spinor"] = bool(spinor)
    return ebr_input_candidates


def attach_real_certificate(export_bundle: dict, table: dict) -> dict | None:
    """Attach a REAL affine primitive certificate + spin to each bundle for an
    explicitly-labelled validated-affine mapping-contract fixture.

    The table must already declare its ``space_group_number``; the standard
    setting is NOT copied from the bundle (the validator re-derives it via
    spglib).  Returns None when the SG is not a unique primitive setting."""
    prov = table.get("provenance")
    if not isinstance(prov, dict):
        prov = {}
        table["provenance"] = prov
    sg = prov.get("space_group_number")
    if not (isinstance(sg, int) and not isinstance(sg, bool) and sg > 0):
        return None
    prov.setdefault("data_source", "irreptables")
    prov.setdefault("package", "irreptables")
    prov.setdefault("package_version", "0.0.test")
    prov.setdefault("valleyscope_reduction", "sampled_hsp_valley_preserving")
    spinful = prov.get("spinful")
    if not isinstance(spinful, bool):
        spinful = False
        prov["spinful"] = spinful
    symbol = str(table.get("subspace_group_candidate", ""))
    cert_identity = real_primitive_certificate_identity(
        int(sg), symbol, spinor=spinful)
    if cert_identity is None:
        return None
    for b in export_bundle.get("bundles", []):
        if isinstance(b, dict):
            b["certificate_identity"] = dict(cert_identity)
            b["subspace_sg_number"] = int(sg)
            _inject_spin_records(b, spinful)
    return export_bundle


def complete_table_provenance(table: dict, sg_number: int | None = None,
                              spinful: bool | None = None) -> dict:
    """Fill the conventional irreptables provenance constants on a table.

    ``sg_number`` / ``spinful`` are declared explicitly by the caller (no
    fabricated symbol->SG lookup).  Certificates and crystallographic setting
    evidence are untouched (the validator derives setting from spglib)."""
    prov = table.get("provenance")
    if not isinstance(prov, dict):
        prov = {}
        table["provenance"] = prov
    if sg_number is not None:
        prov["space_group_number"] = int(sg_number)
    prov.setdefault("data_source", "irreptables")
    prov.setdefault("package", "irreptables")
    prov.setdefault("package_version", "0.0.test")
    prov.setdefault("valleyscope_reduction", "sampled_hsp_valley_preserving")
    if spinful is not None:
        prov["spinful"] = bool(spinful)
    elif not isinstance(prov.get("spinful"), bool):
        prov["spinful"] = False
    return table


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
