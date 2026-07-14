"""Real-resolver promotion helpers for reduced-EBR tests.

Certificates are produced by the ACTUAL resolver
(``resolve_standard_setting_hsp_label``) and serialized by
``ebr_problem_instances._certificate_identity``.  No field is assigned or
mutated after construction, no setting is copied from the bundle into the
table (the validator re-derives it from spglib), and no symbol->SG lookup is
fabricated: the table must already declare its ``space_group_number``.

Only primitive space groups with a unique spglib Hall setting yield a
validated certificate here (the resolver's direct-coordinate-match path).
Centered/nontrivial affine certificates require Phase E and are not produced.
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


class _GammaCoordinateTable:
    """Minimal coordinate table: only Gamma (origin) matches a label."""

    def __init__(self, sg_number: int, sg_symbol: str):
        self.number = int(sg_number)
        self.name = sg_symbol
        self.spinor = True

    def match_kpoint_label(self, k_frac, *, tolerance: float = 1e-6):
        if float(np.linalg.norm(np.asarray(k_frac, dtype=float))) <= tolerance:
            return "GM"
        return None


def resolver_certificate_identity(sg_number: int, sg_symbol: str) -> dict | None:
    """Real resolver -> serialized certificate identity, with no post-mutation.

    Returns None when the SG number has no unique primitive spglib setting, so
    the promotion path is legitimately unresolved rather than forced.
    """
    setting = _derive_table_standard_setting(int(sg_number))
    if setting is None or setting["centering_type"] != "P":
        return None
    _, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=_GammaCoordinateTable(int(sg_number), sg_symbol),
        standard_match={
            "number": int(sg_number),
            "international_short": sg_symbol,
            "hall_number": setting["hall_number"],
            "hall_symbol": setting["hall_symbol"],
            "operation_ids": [0, 1, 2],
        },
    )
    if blocker is not None:
        return None
    cert = prov.get("standard_setting_certificate")
    if not isinstance(cert, dict):
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


def apply_resolver_certificate(export_bundle: dict, table: dict) -> dict | None:
    """Attach a REAL resolver certificate + spin evidence to each bundle.

    The table must already declare ``space_group_number`` in its provenance;
    the standard setting is NOT copied from the bundle.  Returns the export
    bundle, or None when the SG number has no unique primitive setting (the
    caller must then expect a fail-closed blocker).
    """
    prov = table.get("provenance")
    if not isinstance(prov, dict):
        prov = {}
        table["provenance"] = prov
    sg = prov.get("space_group_number")
    if not (isinstance(sg, int) and not isinstance(sg, bool) and sg > 0):
        return None
    # The SG number must be declared explicitly (no fabricated symbol->SG
    # lookup); the remaining trusted provenance fields are conventional.
    prov.setdefault("data_source", "irreptables")
    prov.setdefault("package", "irreptables")
    prov.setdefault("package_version", "0.0.test")
    prov.setdefault("valleyscope_reduction", "sampled_hsp_valley_preserving")
    spinful = prov.get("spinful")
    if not isinstance(spinful, bool):
        spinful = False
        prov["spinful"] = spinful
    symbol = str(table.get("subspace_group_candidate", ""))
    cert_identity = resolver_certificate_identity(int(sg), symbol)
    if cert_identity is None:
        return None
    for b in export_bundle.get("bundles", []):
        if isinstance(b, dict):
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


def _resolver_certificate_dict(sg_number: int, sg_symbol: str) -> dict | None:
    """Raw resolver certificate dict (pre-serialization); no post-mutation."""
    setting = _derive_table_standard_setting(int(sg_number))
    if setting is None or setting["centering_type"] != "P":
        return None
    _, blocker, prov = resolve_standard_setting_hsp_label(
        k_frac=np.array([0.0, 0.0, 0.0]),
        table=_GammaCoordinateTable(int(sg_number), sg_symbol),
        standard_match={
            "number": int(sg_number), "international_short": sg_symbol,
            "hall_number": setting["hall_number"],
            "hall_symbol": setting["hall_symbol"], "operation_ids": [0, 1, 2],
        },
    )
    if blocker is not None:
        return None
    cert = prov.get("standard_setting_certificate")
    return dict(cert) if isinstance(cert, dict) else None


def add_resolver_certificate_to_candidates(
    ebr_input_candidates: dict, sg_number: int, sg_symbol: str,
    *, spinful: bool = False) -> dict:
    """Inject a REAL resolver certificate + spin into each input candidate so
    the workflow serializes a validated bundle identity naturally (no bundle
    injection).  Returns the mutated candidates dict."""
    cert = _resolver_certificate_dict(int(sg_number), sg_symbol)
    if cert is None:
        return ebr_input_candidates
    for c in ebr_input_candidates.get("candidates", []):
        if not isinstance(c, dict):
            continue
        prov = c.get("irrep_source_provenance")
        if not isinstance(prov, dict):
            prov = {}
            c["irrep_source_provenance"] = prov
        kmap = prov.get("standard_setting_hsp_mapping")
        if not isinstance(kmap, dict):
            kmap = {}
            prov["standard_setting_hsp_mapping"] = kmap
        kmap["standard_setting_certificate"] = dict(cert)
        prov["source_table_spinor"] = spinful
    return ebr_input_candidates


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
