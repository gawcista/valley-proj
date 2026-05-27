"""Derived HSP-star valley-preserving characters from unitary space-group conjugation.

For a source HSP k0 with explicit valley-preserving character chi_{k0,a0}(g),
and a space-group operation r mapping k0 -> k1 with a1 = pi_r(a0), the derived
character at k1 for h = r g r^{-1} is:

    chi_{k1,a1}(h) = chi_{k0,a0}(g)

This is a similarity-transform relation at the character level and does not
require additional DFT.  Only unitary space-group derivations are implemented;
TRS/antiunitary derivations are marked not_implemented.
"""

from __future__ import annotations

import numpy as np

from valleyscope.analysis.hsp_star_conjugation import build_hsp_star_conjugation_report


def build_hsp_star_derived_characters(
    *,
    conjugation_report: dict[str, object],
    source_character_diagnostics: dict[str, dict[str, object]],
    target_subspace_closure_blockers: list[str] | None = None,
) -> dict[str, object]:
    """Build derived HSP-star character entries from the conjugation graph.

    Parameters
    ----------
    conjugation_report : output of build_hsp_star_conjugation_report
    source_character_diagnostics : {kpoint_name: character_diagnostic_dict}
        Per-kpoint valley-preserving character diagnostics (the per_valley
        portion of symmetry_adapted_valley_report.valley_preserving_character_diagnostics).
        Each entry should be the full character diagnostic dict with a "per_valley" key.
    target_subspace_closure_blockers : list[str] or None
        If the source kpoint has target_subspace_closure_failed, derived characters
        are blocked.

    Returns
    -------
    dict with keys: status, derivation_type, by_target_kpoint, blocked_sources
    """
    closure_blockers = list(target_subspace_closure_blockers or [])
    source_closure_failed = "target_subspace_closure_failed" in closure_blockers

    derived_entries: list[dict[str, object]] = []
    blocked_sources: list[dict[str, object]] = []

    by_source = conjugation_report.get("by_source_kpoint", {})
    if not isinstance(by_source, dict):
        return _empty_derived_report("no conjugation data available")

    for source_kpoint, entries in by_source.items():
        if not isinstance(entries, list):
            continue
        source_char_data = source_character_diagnostics.get(source_kpoint, {})
        if not isinstance(source_char_data, dict):
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            conjugation_status = str(entry.get("conjugation_status", ""))

            if conjugation_status == "antiunitary_not_implemented":
                derived_entries.append(_derived_entry(
                    entry=entry,
                    derivation_type="unitary_space_group",
                    status="not_implemented",
                    reason="TRS/antiunitary derivation not implemented",
                    trusted_for_ebr_input=False,
                ))
                continue

            if conjugation_status != "matched":
                derived_entries.append(_derived_entry(
                    entry=entry,
                    derivation_type="unitary_space_group",
                    status=conjugation_status,
                    reason=str(entry.get("reason", "")),
                    trusted_for_ebr_input=False,
                ))
                continue

            source_valley = str(entry.get("source_valley", ""))
            source_op_id = entry.get("source_preserving_operation_id")
            source_chars = _find_source_character(
                source_char_data, source_valley, source_op_id,
            )

            if source_chars is None:
                derived_entries.append(_derived_entry(
                    entry=entry,
                    derivation_type="unitary_space_group",
                    status="source_character_unavailable",
                    reason=(
                        f"no character data for source valley={source_valley}, "
                        f"op={source_op_id} at kpoint={source_kpoint}"
                    ),
                    trusted_for_ebr_input=False,
                ))
                continue

            source_status = str(source_chars.get("source_diagnostic_status", ""))
            source_trusted = bool(source_chars.get("source_trusted", False))

            if source_closure_failed:
                blocked_sources.append({
                    "source_kpoint": source_kpoint,
                    "source_valley": source_valley,
                    "source_operation_id": source_op_id,
                    "reason": "target_subspace_closure_failed at source",
                })
                derived_entries.append(_derived_entry(
                    entry=entry,
                    derivation_type="unitary_space_group",
                    status="blocked_by_source_closure",
                    reason="source target_subspace_closure_failed",
                    trusted_for_ebr_input=False,
                ))
                continue

            if source_status == "diagnostic_only" or not source_trusted:
                derived_entries.append(_derived_entry(
                    entry=entry,
                    derivation_type="unitary_space_group",
                    status="diagnostic_only",
                    reason="source character is diagnostic_only",
                    character=source_chars.get("character"),
                    eigenphases=source_chars.get("eigenphases"),
                    trusted_for_ebr_input=False,
                ))
                continue

            derived_entries.append(_derived_entry(
                entry=entry,
                derivation_type="unitary_space_group",
                status="derived",
                reason="",
                character=source_chars.get("character"),
                eigenphases=source_chars.get("eigenphases"),
                trusted_for_ebr_input=source_trusted,
            ))

    status = "ok" if derived_entries else "not_evaluated"
    return {
        "status": status,
        "derivation_type": "unitary_space_group",
        "derivation_formula": "chi_{k1,a1}(h) = chi_{k0,a0}(g) with h = r g r^{-1}, k1 = r k0, a1 = pi_r(a0)",
        "antiunitary_status": "not_implemented",
        "entries": derived_entries,
        "blocked_sources": blocked_sources,
    }


def collect_derived_characters_by_target(
    derived_report: dict[str, object],
) -> dict[str, dict[str, dict[object, dict[str, object]]]]:
    """Collect derived characters indexed by (target_kpoint, target_valley, operation_id).

    Returns {target_kpoint: {valley: {op_id: {character, eigenphases, trusted, status}}}}.
    """
    result: dict[str, dict[str, dict[object, dict[str, object]]]] = {}
    for entry in derived_report.get("entries", []):
        if not isinstance(entry, dict):
            continue
        if not entry.get("trusted_for_ebr_input", False):
            continue
        target_kp = str(entry.get("target_kpoint_label", ""))
        target_valley = str(entry.get("target_valley", ""))
        target_op = entry.get("derived_target_operation_id")
        if not target_kp or not target_valley or target_op is None:
            continue
        char_val = entry.get("character")
        phases = entry.get("eigenphases")
        result.setdefault(target_kp, {}).setdefault(target_valley, {})[target_op] = {
            "character": char_val,
            "eigenphases": list(phases) if phases else [],
            "trusted_for_ebr_input": True,
            "derivation_status": str(entry.get("status", "")),
        }
    return result


def has_derived_character_for(
    derived_report: dict[str, object],
    kpoint: str,
    valley: str,
    operation_id: object,
) -> bool:
    """Check if a trusted derived character exists for a given (kpoint, valley, op)."""
    by_target = collect_derived_characters_by_target(derived_report)
    kp_data = by_target.get(kpoint, {})
    valley_data = kp_data.get(valley, {})
    entry = valley_data.get(operation_id)
    return entry is not None and bool(entry.get("trusted_for_ebr_input", False))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_source_character(
    char_data: dict[str, object],
    valley: str,
    op_id: object,
) -> dict[str, object] | None:
    """Extract source character from per_valley character diagnostics."""
    per_valley = char_data.get("per_valley", {})
    if not isinstance(per_valley, dict):
        return None
    items = per_valley.get(valley, [])
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("operation_id") == op_id:
            char = item.get("character")
            if char is None:
                return None
            return {
                "character": char,
                "eigenphases": item.get("eigenphases"),
                "source_diagnostic_status": str(char_data.get("status", "")),
                "source_trusted": (
                    not bool(char_data.get("diagnostic_only", True))
                    and bool(char_data.get("local_irrep_ready", False))
                ),
                "source_unitarity_error": item.get("representation_unitarity_error", 0.0),
            }
    return None


def _derived_entry(
    *,
    entry: dict[str, object],
    derivation_type: str,
    status: str,
    reason: str,
    trusted_for_ebr_input: bool,
    character: object = None,
    eigenphases: list[float] | None = None,
) -> dict[str, object]:
    e: dict[str, object] = {
        "source_kpoint": entry.get("source_kpoint"),
        "target_kpoint_label": entry.get("target_kpoint_label"),
        "target_frac": entry.get("target_frac"),
        "derivation_operation_id": entry.get("mapping_operation_id"),
        "source_valley": entry.get("source_valley"),
        "target_valley": entry.get("target_valley"),
        "source_operation_id": entry.get("source_preserving_operation_id"),
        "derived_target_operation_id": entry.get("derived_target_operation_id"),
        "derivation_type": derivation_type,
        "status": status,
        "reason": reason,
        "trusted_for_ebr_input": trusted_for_ebr_input,
    }
    if character is not None:
        if isinstance(character, complex):
            e["character"] = {"real": character.real, "imag": character.imag}
        else:
            e["character"] = character
    if eigenphases is not None:
        e["eigenphases"] = list(eigenphases)
    return e


def _empty_derived_report(reason: str) -> dict[str, object]:
    return {
        "status": "not_evaluated",
        "derivation_type": "unitary_space_group",
        "derivation_formula": "chi_{k1,a1}(h) = chi_{k0,a0}(g) with h = r g r^{-1}",
        "antiunitary_status": "not_implemented",
        "entries": [],
        "blocked_sources": [],
        "reason": reason,
    }
