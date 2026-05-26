from __future__ import annotations

from typing import Any

import numpy as np

from valleyscope.symmetry.little_group import reciprocal_transform


def build_hsp_star_report(
    *,
    kpoint_frac_by_name: dict[str, Any],
    operations: list[dict[str, Any]],
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Report whether configured HSP labels cover their symmetry star.

    This is a data-availability diagnostic.  It does not synthesize missing
    wavefunctions; it only tells the caller which star representatives are
    already explicit in the HDF5/config input and which representatives must be
    generated from symmetry.  The default tolerance is intentionally looser than
    machine precision because HSP fractions in VASP-derived fixtures are often
    stored with about six decimal digits, e.g. 0.333333 for 1/3.
    """
    kpoints = {
        str(name): np.asarray(frac, dtype=float)
        for name, frac in kpoint_frac_by_name.items()
    }
    by_kpoint: dict[str, Any] = {}
    missing_any = False

    for source_label, source_frac in kpoints.items():
        representatives: list[dict[str, Any]] = []
        operation_targets: list[dict[str, Any]] = []
        for operation in operations:
            rotation = operation.get("rotation_frac")
            op_id = operation.get("operation_id")
            if rotation is None or op_id is None:
                continue
            target_frac = reciprocal_transform(np.asarray(rotation), source_frac)
            canonical = _canonical_frac(target_frac)
            matched = _match_kpoint(canonical, kpoints, tolerance=tolerance)
            operation_targets.append(
                {
                    "operation_id": op_id,
                    "kind": operation.get("kind", ""),
                    "order": operation.get("order"),
                    "target_kpoint": matched,
                    "target_frac": _json_frac(canonical),
                    "available": matched is not None,
                }
            )
            _append_unique_representative(
                representatives,
                canonical_frac=canonical,
                matched_label=matched,
                operation_id=op_id,
                tolerance=tolerance,
            )

        explicit = [
            rep for rep in representatives
            if rep.get("matched_kpoint") is not None
        ]
        symmetry_derivable = [
            rep for rep in representatives
            if rep.get("matched_kpoint") is None
        ]
        complete = len(symmetry_derivable) == 0
        missing_any = missing_any or not complete
        by_kpoint[source_label] = {
            "status": "complete" if complete else "symmetry_derivable",
            "complete": complete,
            "requires_additional_dft": False,
            "requires_symmetry_derivation": not complete,
            "source_frac": _json_frac(_canonical_frac(source_frac)),
            "star_size": len(representatives),
            "explicit_count": len(explicit),
            "symmetry_derivable_count": len(symmetry_derivable),
            "explicit_representatives": explicit,
            "symmetry_derivable_representatives": symmetry_derivable,
            "operation_targets": operation_targets,
            "interpretation": (
                "Representatives absent from the input HDF5 are symmetry-"
                "derivable from the source HSP when the DFT calculation "
                "respects the reported space-group operation. They do not "
                "require additional DFT, but the workflow must explicitly "
                "construct the symmetry-derived HSP-star basis before assigning "
                "local valley-preserving characters at those star members. This "
                "does not by itself explain failures at HSPs whose little group "
                "is already present, such as Gamma."
            ),
        }

    return {
        "status": (
            "symmetry_derivable" if missing_any else "complete"
        ) if by_kpoint else "not_evaluated",
        "by_kpoint": by_kpoint,
    }


def _canonical_frac(frac: np.ndarray) -> np.ndarray:
    arr = np.asarray(frac, dtype=float)
    canonical = arr - np.floor(arr)
    canonical[np.isclose(canonical, 1.0, atol=1e-10)] = 0.0
    canonical[np.isclose(canonical, 0.0, atol=1e-10)] = 0.0
    return canonical


def _periodic_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return delta - np.rint(delta)


def _same_frac(a: np.ndarray, b: np.ndarray, *, tolerance: float) -> bool:
    return bool(np.allclose(_periodic_delta(a, b), 0.0, atol=tolerance))


def _match_kpoint(
    frac: np.ndarray,
    kpoints: dict[str, np.ndarray],
    *,
    tolerance: float,
) -> str | None:
    for label, candidate in kpoints.items():
        if _same_frac(frac, candidate, tolerance=tolerance):
            return label
    return None


def _append_unique_representative(
    representatives: list[dict[str, Any]],
    *,
    canonical_frac: np.ndarray,
    matched_label: str | None,
    operation_id: object,
    tolerance: float,
) -> None:
    for item in representatives:
        if _same_frac(
            np.asarray(item["canonical_frac"], dtype=float),
            canonical_frac,
            tolerance=tolerance,
        ):
            item.setdefault("generated_by_operation_ids", []).append(operation_id)
            if item.get("matched_kpoint") is None and matched_label is not None:
                item["matched_kpoint"] = matched_label
            return
    representatives.append(
        {
            "canonical_frac": _json_frac(canonical_frac),
            "matched_kpoint": matched_label,
            "generated_by_operation_ids": [operation_id],
        }
    )


def _json_frac(frac: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(frac, dtype=float).tolist()]
