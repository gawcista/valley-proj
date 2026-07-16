from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np
try:
    from irreptables.irreps import IrrepTable  # irreptables >= 3.0
except ImportError:
    from irreptables import IrrepTable         # irreptables < 3.0


@dataclass(frozen=True, slots=True)
class StandardTableOperation:
    table_index: int
    rotation_frac: np.ndarray
    translation_frac: np.ndarray
    spin_rotation: np.ndarray
    time_reversal: bool


@dataclass(frozen=True, slots=True)
class StandardIrrep:
    label: str
    kpoint_label: str
    k_frac: np.ndarray
    dimension: int
    characters: dict[int, complex]


@dataclass(frozen=True, slots=True)
class StandardIrrepTable:
    number: int
    name: str
    spinor: bool
    operations: tuple[StandardTableOperation, ...]
    irreps: tuple[StandardIrrep, ...]

    def operation_by_index(self, table_index: int) -> StandardTableOperation:
        for operation in self.operations:
            if operation.table_index == table_index:
                return operation
        raise KeyError(f"Unknown table operation index: {table_index}")

    def irreps_by_kpoint(self, kpoint_label: str) -> list[StandardIrrep]:
        return [irrep for irrep in self.irreps if irrep.kpoint_label == kpoint_label]

    def operation_indices_for_kpoint(self, kpoint_label: str) -> list[int]:
        indices: set[int] = set()
        for irrep in self.irreps_by_kpoint(kpoint_label):
            indices.update(irrep.characters)
        return sorted(indices)

    def match_kpoint_label(self, k_frac: np.ndarray, *, tolerance: float = 1e-6) -> str | None:
        k_frac = np.asarray(k_frac, dtype=float)
        labels_by_coordinate: dict[str, np.ndarray] = {}
        for irrep in self.irreps:
            labels_by_coordinate.setdefault(irrep.kpoint_label, irrep.k_frac)
        # Centering-aware k-point equivalence derived from the table's
        # space-group name (e.g. "P3", "C2/c"), not from HSP coordinates.
        centering = _table_centering_from_name(self)
        for label, table_k_frac in labels_by_coordinate.items():
            if _kpoint_matches_centered(k_frac, table_k_frac, centering, tolerance):
                return label
        return None


@dataclass(frozen=True, slots=True)
class OperationMappingReport:
    status: str
    mapping_by_operation_id: dict[Any, int]
    unmatched_operation_ids: list[Any]
    unused_table_operation_indices: list[int]
    provenance: str = "exact_spatial"


def load_standard_irrep_table(spacegroup_number: int, *, spinor: bool) -> StandardIrrepTable:
    raw_table = IrrepTable(str(spacegroup_number), spinor=spinor)
    return _standard_table_from_raw(raw_table, number=spacegroup_number)


def resolve_ebr_source_irrep_label_evidence(
    *,
    table: StandardIrrepTable,
    source_basis_labels: list[str],
) -> dict[str, object]:
    """Resolve EBR labels against canonical and compatible package tables.

    The primary irreptables irrep table may intentionally contain only
    canonical k-vector representatives while the EBR basis contains further
    source rows.  Such rows are accepted as noncanonical only when a unique
    irreptables correptable has the same affine/spin operation inventory and
    contains every otherwise-missing label.  No label-shape heuristic is used.
    """
    canonical = {irrep.label: irrep for irrep in table.irreps}
    missing = [label for label in source_basis_labels if label not in canonical]
    by_label: dict[str, dict[str, object]] = {
        label: _source_irrep_evidence(
            canonical[label],
            status="canonical_standard_irrep",
            source_table="standard_irrep_table",
        )
        for label in source_basis_labels
        if label in canonical
    }
    if not missing:
        return {
            "status": "validated",
            "by_label": by_label,
            "auxiliary_source_table": None,
            "blocker": "",
        }

    suffix = "spin.dat" if table.spinor else "scal.dat"
    prefix = f"irreps-SG={table.number}."
    candidates: list[tuple[str, StandardIrrepTable]] = []
    try:
        directory = resources.files("irreptables").joinpath(
            "data", "correptables"
        )
        resource_rows = sorted(
            (
                row for row in directory.iterdir()
                if row.name.startswith(prefix)
                and row.name.endswith(f"-{suffix}")
            ),
            key=lambda row: row.name,
        )
        for resource in resource_rows:
            table_token = resource.name[len("irreps-SG="):-len(f"-{suffix}")]
            try:
                with resources.as_file(resource) as resource_path:
                    raw = IrrepTable(
                        table_token,
                        spinor=table.spinor,
                        name=str(resource_path),
                    )
                candidate = _standard_table_from_raw(
                    raw, number=table.number
                )
            except (AssertionError, OSError, TypeError, ValueError):
                continue
            if not _same_operation_setting(table, candidate):
                continue
            candidate_labels = {irrep.label for irrep in candidate.irreps}
            if all(label in candidate_labels for label in missing):
                candidates.append((resource.name, candidate))
    except (ModuleNotFoundError, OSError):
        candidates = []

    if len(candidates) != 1:
        return {
            "status": "blocked",
            "by_label": by_label,
            "auxiliary_source_table": None,
            "blocker": (
                "ebr_source_irrep_labels_missing_from_standard_table: "
                f"{missing}; compatible auxiliary source tables="
                f"{[name for name, _ in candidates]}"
            ),
        }

    source_name, source_table = candidates[0]
    auxiliary = {irrep.label: irrep for irrep in source_table.irreps}
    for label in missing:
        by_label[label] = _source_irrep_evidence(
            auxiliary[label],
            status="validated_noncanonical_ebr_source_row",
            source_table=source_name,
        )
    return {
        "status": "validated",
        "by_label": by_label,
        "auxiliary_source_table": source_name,
        "blocker": "",
    }


def _standard_table_from_raw(
    raw_table: object,
    *,
    number: int,
) -> StandardIrrepTable:
    operations = tuple(
        StandardTableOperation(
            table_index=index,
            rotation_frac=np.asarray(symop.R, dtype=int),
            translation_frac=np.asarray(symop.t, dtype=float),
            spin_rotation=np.asarray(symop.S, dtype=complex),
            time_reversal=bool(symop.time_reversal),
        )
        for index, symop in enumerate(raw_table.symmetries, start=1)
    )
    irreps = tuple(
        StandardIrrep(
            label=str(irrep.name),
            kpoint_label=str(irrep.kpname),
            k_frac=np.asarray(irrep.k, dtype=float),
            dimension=int(irrep.dim),
            characters={
                int(index): complex(value)
                for index, value in irrep.characters.items()
            },
        )
        for irrep in raw_table.irreps
    )
    return StandardIrrepTable(
        number=int(number),
        name=str(raw_table.name).strip(),
        spinor=bool(raw_table.spinor),
        operations=operations,
        irreps=irreps,
    )


def _source_irrep_evidence(
    irrep: StandardIrrep,
    *,
    status: str,
    source_table: str,
) -> dict[str, object]:
    return {
        "status": status,
        "label": irrep.label,
        "source_hsp_label": irrep.kpoint_label,
        "standard_k_frac": [float(value) for value in irrep.k_frac],
        "dimension": irrep.dimension,
        "source_table": source_table,
    }


def _same_operation_setting(
    primary: StandardIrrepTable,
    auxiliary: StandardIrrepTable,
) -> bool:
    if (
        primary.number != auxiliary.number
        or primary.spinor != auxiliary.spinor
        or len(primary.operations) != len(auxiliary.operations)
    ):
        return False
    for left, right in zip(primary.operations, auxiliary.operations):
        if (
            left.table_index != right.table_index
            or not np.array_equal(left.rotation_frac, right.rotation_frac)
            or left.time_reversal != right.time_reversal
        ):
            return False
        translation_delta = left.translation_frac - right.translation_frac
        if np.linalg.norm(
            translation_delta - np.rint(translation_delta)
        ) > 5e-6:
            return False
        if np.linalg.norm(left.spin_rotation - right.spin_rotation) > 5e-5:
            return False
    return True


def match_table_operations(
    detected_operations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    table: StandardIrrepTable,
    *,
    tolerance: float = 1e-8,
    source_hsp_label: str | None = None,
) -> OperationMappingReport:
    """Match detected operations to table operations.

    1. Exact spatial matching (rotation + translation mod lattice).
    2. If unmatched remain, attempt unique group-isomorphism fallback
       restricted to the HSP little group when ``source_hsp_label`` is
       provided.
    """
    mapping: dict[Any, int] = {}
    unmatched: list[Any] = []
    used_table_indices: set[int] = set()

    for operation in detected_operations:
        operation_id = operation.get("operation_id")
        table_index = _match_one_operation(operation, table, tolerance=tolerance)
        if table_index is None:
            unmatched.append(operation_id)
            continue
        mapping[operation_id] = table_index
        used_table_indices.add(table_index)

    # --- Group-isomorphism fallback ---
    provenance = "exact_spatial"
    if unmatched:
        iso_result = _try_group_isomorphism_fallback(
            detected_operations=detected_operations,
            table=table,
            partial_mapping=mapping,
            unmatched_ids=unmatched,
            used_table_indices=used_table_indices,
            tolerance=tolerance,
            source_hsp_label=source_hsp_label,
        )
        if iso_result is not None:
            mapping.update(iso_result["mapping"])
            used_table_indices.update(iso_result["mapping"].values())
            unmatched = iso_result.get("still_unmatched", [])
            provenance = iso_result.get("provenance", "unique_group_isomorphism")

    unused_table_indices = [
        operation.table_index
        for operation in table.operations
        if operation.table_index not in used_table_indices
    ]
    status = "complete" if not unmatched and not unused_table_indices else "incomplete"
    return OperationMappingReport(
        status=status,
        mapping_by_operation_id=mapping,
        unmatched_operation_ids=unmatched,
        unused_table_operation_indices=unused_table_indices,
        provenance=provenance,
    )


def _try_group_isomorphism_fallback(
    *,
    detected_operations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    table: StandardIrrepTable,
    partial_mapping: dict[Any, int],
    unmatched_ids: list[Any],
    used_table_indices: set[int],
    tolerance: float,
    source_hsp_label: str | None,
) -> dict[str, Any] | None:
    """Attempt unique group-isomorphism operation mapping.

    When exact spatial matching leaves unmatched operations, try a
    conservative group-isomorphism fallback: build multiplication tables,
    anchor the identity element from the partial mapping or from
    operation content, enumerate product-preserving bijections, and
    accept only when exactly one bijection exists.

    Returns None if zero or multiple isomorphisms are found.
    """
    unmatched_ops = [
        op for op in detected_operations
        if op.get("operation_id") not in partial_mapping
    ]
    if not unmatched_ops:
        return None

    n_det = len(detected_operations)
    if n_det == 0:
        return None

    # Build detected multiplication table for ALL detected ops.
    det_rot = [np.asarray(op.get("rotation_frac", np.eye(3)), dtype=int)
               for op in detected_operations]
    det_trans = [np.asarray(op.get("translation_frac", np.zeros(3)), dtype=float)
                 for op in detected_operations]
    det_mult = _build_multiplication_table(det_rot, det_trans, tolerance)

    # Candidate table operations: HSP little group indices.
    if source_hsp_label:
        all_candidate = table.operation_indices_for_kpoint(source_hsp_label)
    else:
        all_candidate = [op.table_index for op in table.operations]

    # Ensure we have the identity (index 1) included.
    if 1 not in all_candidate:
        all_candidate = [1] + all_candidate

    # Include already-matched table indices so we build the full group.
    full_candidate = sorted(set(all_candidate) | used_table_indices)
    if len(full_candidate) != n_det:
        return None  # size mismatch

    # Build table multiplication table.
    tbl_rot = []
    tbl_trans = []
    tbl_idx_map = {}  # position -> table_index
    for pos, tidx in enumerate(full_candidate):
        op = table.operation_by_index(tidx)
        tbl_rot.append(op.rotation_frac)
        tbl_trans.append(op.translation_frac)
        tbl_idx_map[pos] = tidx
    tbl_mult = _build_multiplication_table(tbl_rot, tbl_trans, tolerance)

    if det_mult is None or tbl_mult is None:
        return None

    # Find identity positions from content.
    det_id = _find_identity_position(det_rot, det_trans, tolerance)
    tbl_id = _find_identity_position(tbl_rot, tbl_trans, tolerance)
    if det_id is None or tbl_id is None:
        return None

    # Build pre-seeded mapping from partial_mapping.
    pre_map: dict[int, int] = {}  # det_pos -> tbl_pos
    for det_pos, op in enumerate(detected_operations):
        op_id = op.get("operation_id")
        tidx = partial_mapping.get(op_id)
        if tidx is not None:
            # Find tbl_pos for this tidx
            for tpos, tid in tbl_idx_map.items():
                if tid == tidx:
                    pre_map[det_pos] = tpos
                    break

    # Verify identity consistency if both are pre-mapped.
    if det_id in pre_map and pre_map[det_id] != tbl_id:
        return None

    isomorphisms = _find_group_isomorphisms(
        det_mult, tbl_mult, det_id, tbl_id, pre_map=pre_map,
    )

    if len(isomorphisms) != 1:
        return None

    # Build mapping from the unique isomorphism.
    iso = isomorphisms[0]
    new_mapping: dict[Any, int] = {}
    for det_pos, tbl_pos in enumerate(iso):
        det_op_id = detected_operations[det_pos].get("operation_id")
        if det_op_id is not None and det_op_id not in partial_mapping:
            new_mapping[det_op_id] = tbl_idx_map[tbl_pos]

    return {
        "mapping": new_mapping,
        "still_unmatched": [],
        "provenance": "unique_group_isomorphism",
    }


def _build_multiplication_table(
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    tolerance: float,
) -> list[list[int]] | None:
    """Build multiplication table: mult[i][j] = index of product i*j.

    Product: rotation_i @ rotation_j, rotation_i @ trans_j + trans_i (mod lattice).
    Returns None if any product cannot be uniquely identified.
    """
    n = len(rotations)
    mult = [[-1] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            prod_rot = np.rint(rotations[i] @ rotations[j]).astype(int)
            prod_trans = rotations[i] @ translations[j] + translations[i]
            matches = []
            for k in range(n):
                if not np.array_equal(prod_rot, rotations[k]):
                    continue
                delta = prod_trans - translations[k]
                delta_mod = delta - np.rint(delta)
                if np.linalg.norm(delta_mod) <= tolerance:
                    matches.append(k)
            if len(matches) != 1:
                return None  # product not uniquely identified
            mult[i][j] = matches[0]
    return mult


def _find_identity_position(
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    tolerance: float,
) -> int | None:
    """Find the identity element in an operation list."""
    for i, (rot, trans) in enumerate(zip(rotations, translations)):
        if np.array_equal(rot, np.eye(3, dtype=int)):
            if np.linalg.norm(trans - np.rint(trans)) <= tolerance:
                trans_mod = trans - np.rint(trans)
                if np.linalg.norm(trans_mod) <= tolerance:
                    return i
    return None


def _find_group_isomorphisms(
    det_mult: list[list[int]],
    tbl_mult: list[list[int]],
    det_id: int,
    tbl_id: int,
    pre_map: dict[int, int] | None = None,
) -> list[list[int]]:
    """Enumerate all product-preserving bijections det -> table.

    Returns at most 2 isomorphisms; stops at 2 because any count >= 2
    means ambiguous (non-unique).
    """
    n = len(det_mult)
    used = [False] * n
    mapping = [-1] * n  # detected_pos -> table_pos
    results: list[list[int]] = []

    mapping[det_id] = tbl_id
    used[tbl_id] = True

    # Apply pre-seeded mapping.
    if pre_map:
        for dpos, tpos in pre_map.items():
            if mapping[dpos] >= 0 and mapping[dpos] != tpos:
                return []  # conflict
            mapping[dpos] = tpos
            used[tpos] = True

    def _search(pos: int) -> None:
        if len(results) >= 2:
            return
        if pos == n:
            if not _is_product_preserving_bijection(mapping, det_mult, tbl_mult):
                return
            results.append(list(mapping))
            return
        if mapping[pos] >= 0:
            _search(pos + 1)
            return
        for tpos in range(n):
            if used[tpos]:
                continue
            # Verify product preservation with all already-mapped elements.
            ok = True
            for i in range(n):
                if mapping[i] < 0:
                    continue
                if mapping[det_mult[i][pos]] >= 0:
                    if mapping[det_mult[i][pos]] != tbl_mult[mapping[i]][tpos]:
                        ok = False
                        break
                if mapping[det_mult[pos][i]] >= 0:
                    if mapping[det_mult[pos][i]] != tbl_mult[tpos][mapping[i]]:
                        ok = False
                        break
            if not ok:
                continue
            mapping[pos] = tpos
            used[tpos] = True
            _search(pos + 1)
            if len(results) >= 2:
                return
            mapping[pos] = -1
            used[tpos] = False

    _search(0)
    return results


def _is_product_preserving_bijection(
    mapping: list[int],
    det_mult: list[list[int]],
    tbl_mult: list[list[int]],
) -> bool:
    """Verify the full multiplication-table homomorphism condition."""
    for i, row in enumerate(det_mult):
        for j, det_product in enumerate(row):
            if mapping[det_product] != tbl_mult[mapping[i]][mapping[j]]:
                return False
    return True


def _match_one_operation(
    operation: dict[str, Any],
    table: StandardIrrepTable,
    *,
    tolerance: float,
) -> int | None:
    rotation = np.rint(
        np.asarray(operation.get("rotation_frac", np.eye(3)), dtype=float)
    ).astype(int)
    translation = np.asarray(operation.get("translation_frac", np.zeros(3)), dtype=float)
    for table_operation in table.operations:
        if not np.array_equal(rotation, table_operation.rotation_frac):
            continue
        if _translation_matches(translation, table_operation.translation_frac, tolerance):
            return table_operation.table_index
    return None


def _rotation_order(rotation: np.ndarray, max_order: int = 12) -> int | None:
    matrix = np.asarray(rotation, dtype=int)
    product = np.eye(3, dtype=int)
    for order in range(1, max_order + 1):
        product = product @ matrix
        if np.array_equal(product, np.eye(3, dtype=int)):
            return order
    return None


def _translation_matches(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    delta_mod_lattice = delta - np.rint(delta)
    return bool(np.linalg.norm(delta_mod_lattice) <= tolerance)


def _table_centering_from_name(table) -> str | None:
    """Derive centering type from the irrep table's space-group name.

    The first character of the irreptables space-group name
    (e.g. ``"P3"``, ``"C2/c"``, ``"Fm-3m"``, ``"R-3c"``) is the
    Bravais-lattice centering symbol.  This is the convention used
    by irreptables / Bilbao Crystallographic Server tables.

    Returns ``None`` when the name is missing or unrecognised.
    Unknown centering must block trusted HSP matching.
    """
    name = str(getattr(table, "name", "")).strip()
    if name and name[0] in "ABCFIPR":
        return name[0]
    return None


def _centering_translations(centering: str) -> list[np.ndarray] | None:
    """Direct-lattice centering translations for reciprocal-lattice membership.

    Returns the fractional translation vectors (including identity) that
    define the centering.  For R-centering the obverse/reverse convention
    is not available from the centering letter alone, so it returns None
    (blocked).  Unknown centering also returns None.
    """
    identity = np.array([0.0, 0.0, 0.0])
    if centering == "P":
        return [identity]
    if centering == "A":
        return [identity, np.array([0.0, 0.5, 0.5])]
    if centering == "B":
        return [identity, np.array([0.5, 0.0, 0.5])]
    if centering == "C":
        return [identity, np.array([0.5, 0.5, 0.0])]
    if centering == "I":
        return [identity, np.array([0.5, 0.5, 0.5])]
    if centering == "F":
        return [
            identity,
            np.array([0.0, 0.5, 0.5]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.5, 0.5, 0.0]),
        ]
    # R-centering and unrecognized centering: blocked.
    return None


def _kpoint_matches_centered(
    left: np.ndarray,
    right: np.ndarray,
    centering: str,
    tolerance: float,
) -> bool:
    """Centering-aware k-point equivalence modulo conventional reciprocal lattice.

    Uses the generic reciprocal-lattice membership condition: for two
    k-points in the conventional reciprocal basis, they are equivalent
    iff their integer difference n satisfies n·t_c ∈ Z for every
    direct-lattice centering translation t_c.

    For primitive settings (t_c = {(0,0,0)} only) this reduces to
    component-wise modulo-1 comparison.

    Returns False when the centering convention is unavailable
    (R-centering without obverse/reverse choice, or unrecognized
    centering type).
    """
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    delta_mod = delta - np.rint(delta)
    if np.linalg.norm(delta_mod) > tolerance:
        return False

    translations = _centering_translations(centering)
    if translations is None:
        return False  # blocked: centering convention unavailable

    delta_float = np.rint(delta)
    for t_c in translations:
        dot = np.dot(delta_float, t_c)
        if abs(dot - np.rint(dot)) > tolerance:
            return False
    return True
