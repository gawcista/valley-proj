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
    r"[A-Za-z][A-Za-z0-9_+/\-]*"  # irrep label
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
) -> dict:
    """Exact reduced EBR decomposition of export bundle irrep vectors.

    Parameters
    ----------
    ebr_export_bundle : output of build_ebr_export_bundle
    table : validated reduced EBR table dict (from load_reduced_ebr_table)
    max_coefficient : int, max coefficient per EBR in brute-force search
    """
    max_coefficient = int(max_coefficient)
    if max_coefficient < 0:
        raise ValueError("max_coefficient must be nonnegative")

    if ebr_export_bundle is None:
        return _status("not_evaluated", "no export bundle available")

    bundles = ebr_export_bundle.get("bundles", [])

    if table is None:
        return {
            **_status("missing_table", "no reduced EBR table provided"),
            "solutions": [],
            "excluded_bundles": [
                {"bundle_id": b.get("bundle_id", "?"), "reason": "missing_table"}
                for b in bundles if isinstance(b, dict)
            ],
            "table_status": "not_provided",
        }

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
        if not bundle.get("ready_for_external_solver"):
            excluded.append({
                "bundle_id": bundle.get("bundle_id", "?"),
                "reason": "not ready for external solver",
            })
            continue

        bundle_group = str(bundle.get("subspace_group_candidate", ""))
        if bundle_group != table_group:
            excluded.append({
                "bundle_id": bundle.get("bundle_id", "?"),
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
                "reason": (
                    "malformed expected_hsps: expected a unique list of "
                    "non-empty HSP labels"
                ),
            })
            continue

        if bundle_expected_set != table_expected:
            excluded.append({
                "bundle_id": bundle.get("bundle_id", "?"),
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
        solutions.append({
            "bundle_id": bundle.get("bundle_id", ""),
            "valley": bundle.get("valley", ""),
            "subspace_group_candidate": bundle_group,
            "irrep_vector": irrep_counts,
            **result,
        })

    if not solutions:
        return {
            **_status("not_evaluated", "no bundles to decompose"),
            "solutions": [],
            "excluded_bundles": excluded,
            "table_status": "loaded" if table else "not_provided",
        }

    all_solved = all(s.get("status") == "solved_exact" for s in solutions)
    mapping_status = "solved_exact" if all_solved else "no_exact_solution"
    return {
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


def _status(status: str, reason: str) -> dict:
    return {
        "status": status,
        "mapping_status": status,
        "reduced_ebr_decomposition_status": status,
        "table_status": "not_applicable",
        "solutions": [],
        "excluded_bundles": [],
        "solver": _SOLVER_NAME,
        "interpretation": reason,
    }
