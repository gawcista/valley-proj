"""Derived HSP-star valley-preserving characters from unitary space-group conjugation.

For a source HSP k0 with explicit valley-preserving character chi_{k0,a0}(g),
and a space-group operation r mapping k0 -> k1 with a1 = pi_r(a0), the derived
character at k1 for h = r g r^{-1} is:

    chi_{k1,a1}(h) = chi_{k0,a0}(g)

This is a similarity-transform relation at the character level and does not
require additional DFT.  Only unitary (det=1) space-group derivations are
implemented.  Improper unitary (det=-1) operations are schema-recognised but
marked not supported; they are distinct from antiunitary (TRS) operations
which are not represented in the current spglib unitary operation list.
"""

from __future__ import annotations

from valleyscope.analysis.target_subspace_closure import (
    check_target_subspace_closure_blocked_for_operation,
)


def build_hsp_star_derived_characters(
    *,
    conjugation_report: dict[str, object],
    source_character_diagnostics: dict[str, dict[str, object]],
    target_subspace_closure_report: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build derived HSP-star character entries from the conjugation graph.

    Parameters
    ----------
    conjugation_report : output of build_hsp_star_conjugation_report
    source_character_diagnostics : {kpoint_name: character_diagnostic_dict}
        Per-kpoint valley-preserving character diagnostics.  Each entry is the
        full character diagnostic dict with a "per_valley" key.  All singleton
        subspaces for a given kpoint should be merged into one dict so that
        any valley's character can be found.
    target_subspace_closure_report : dict or None
        Used to check per-(source_kpoint, source_operation_id) whether the
        D_raw at the source is closed.

    Returns
    -------
    dict with keys: status, derivation_type, entries, blocked_sources
    """
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

            if conjugation_status in ("antiunitary_not_implemented",
                                       "improper_unitary_not_supported",
                                       "source_not_in_hsp_little_group"):
                derived_entries.append(_derived_entry(
                    entry=entry,
                    derivation_type="unitary_space_group",
                    status="not_implemented",
                    reason=str(entry.get("reason", "conjugation not supported")),
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

            # Check per-(source_kpoint, source_op) closure, not global.
            if target_subspace_closure_report is not None and source_op_id is not None:
                if check_target_subspace_closure_blocked_for_operation(
                    target_subspace_closure_report, source_kpoint, source_op_id,
                ):
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
                reason = "source character is diagnostic_only"
                if source_status == "diagnostic_only":
                    reason += f" (source_status={source_status})"
                if not source_trusted:
                    source_local_ready = source_char_data.get("local_irrep_ready", False)
                    reason += f" (source_local_irrep_ready={source_local_ready})"
                derived_entries.append(_derived_entry(
                    entry=entry,
                    derivation_type="unitary_space_group",
                    status="diagnostic_only",
                    reason=reason,
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
        "derivation_formula": (
            "chi_{k1,a1}(h) = chi_{k0,a0}(g) with "
            "h = r g r^{-1}, k1 = r k0, a1 = pi_r(a0)"
        ),
        "antiunitary_status": (
            "not_represented_in_current_spglib_unitary_operation_list"
        ),
        "entries": derived_entries,
        "blocked_sources": blocked_sources,
    }


def _find_source_character(
    char_data: dict[str, object],
    valley: str,
    op_id: object,
) -> dict[str, object] | None:
    """Extract source character from per_valley character diagnostics.

    Trust is determined per-valley from per_valley_diagnostic_only and
    per_valley_ready dicts.  Falls back to global flags if per-valley
    data is not available.
    """
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
            # Per-valley trust: use per_valley_diagnostic_only and
            # per_valley_ready if available, otherwise fall back to
            # global flags.
            pv_diag = char_data.get("per_valley_diagnostic_only", {})
            pv_ready = char_data.get("per_valley_ready", {})
            if isinstance(pv_diag, dict) and valley in pv_diag:
                valley_diag = bool(pv_diag[valley])
            else:
                valley_diag = bool(char_data.get("diagnostic_only", True))
            if isinstance(pv_ready, dict) and valley in pv_ready:
                valley_ready = bool(pv_ready[valley])
            else:
                valley_ready = bool(char_data.get("local_irrep_ready", False))
            return {
                "character": char,
                "eigenphases": item.get("eigenphases"),
                "source_diagnostic_status": (
                    "diagnostic_only" if valley_diag else "ok"
                ),
                "source_trusted": (not valley_diag and valley_ready),
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
        "source_frac": entry.get("source_frac"),
        "target_kpoint_label": entry.get("target_kpoint_label"),
        "target_kpoint_key": entry.get("target_kpoint_key"),
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
        "antiunitary_status": (
            "not_represented_in_current_spglib_unitary_operation_list"
        ),
        "entries": [],
        "blocked_sources": [],
        "reason": reason,
    }
