"""Structural table loading stays separate from numerical promotion trust."""

import importlib
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import valleyscope


@pytest.fixture
def table_payload():
    return {
        "schema_version": "1.0.0",
        "subspace_group_candidate": "P1",
        "expected_hsps": ["GM"],
        "irreps": ["GM:GM1"],
        "ebrs": [{"label": "unit-fixture", "vector": [1]}],
        "provenance": {"review_status": "fixture-only", "nested": {"notes": ["retain"]}},
        "user_notes": ["structural validation is not promotion"],
    }


@pytest.mark.parametrize("module_name", [
    "valleyscope.analysis.reduced_ebr_table",
    "valleyscope.analysis.reduced_ebr_mapping",
])
def test_loader_preserves_full_payload_and_validation_errors(tmp_path, table_payload, module_name):
    loader = importlib.import_module(module_name).load_reduced_ebr_table
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table_payload), encoding="utf-8")
    assert loader(path) == table_payload

    table_payload["ebrs"][0]["vector"] = [0]
    path.write_text(json.dumps(table_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="vector must have at least one positive entry"):
        loader(path)


@pytest.mark.parametrize("entrypoint", ["loader", "legacy_loader", "mapping", "builder"])
@pytest.mark.parametrize("invalid_entry", [True, False, -1, 1.0, "1", None])
def test_table_entrypoints_reject_noninteger_vector_entries(
    tmp_path, table_payload, entrypoint, invalid_entry,
):
    """Structural unit test: a positive entry must not hide an invalid one."""
    table_payload["irreps"] = ["GM:GM1", "GM:GM2"]
    table_payload["ebrs"][0]["vector"] = [1, invalid_entry]
    before = deepcopy(table_payload)
    with pytest.raises(ValueError, match="vector must be nonnegative integers"):
        if entrypoint in {"loader", "legacy_loader"}:
            module = "reduced_ebr_table" if entrypoint == "loader" else "reduced_ebr_mapping"
            loader = importlib.import_module(f"valleyscope.analysis.{module}").load_reduced_ebr_table
            path = tmp_path / "table.json"
            path.write_text(json.dumps(table_payload), encoding="utf-8")
            loader(path)
        elif entrypoint == "mapping":
            from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping
            build_reduced_ebr_mapping(ebr_export_bundle={"bundles": []}, table=table_payload)
        else:
            from valleyscope.analysis.irreptables_runtime_table_builder import _validate_reduced_table_dict
            _validate_reduced_table_dict(table_payload)
    assert table_payload == before


@pytest.mark.parametrize("invalid_table", [None, True, 1, "table", [], [["ebrs"]]])
def test_loader_rejects_nonobject_json(tmp_path, invalid_table):
    from valleyscope.analysis.reduced_ebr_table import load_reduced_ebr_table
    path = tmp_path / "table.json"
    path.write_text(json.dumps(invalid_table), encoding="utf-8")
    with pytest.raises(ValueError, match="table must be a mapping"):
        load_reduced_ebr_table(path)


def test_dict_validation_preserves_payload_without_promoting_it(table_payload):
    from valleyscope.analysis.reduced_ebr_table import validate_reduced_ebr_table
    before = deepcopy(table_payload)
    assert validate_reduced_ebr_table(table_payload) is table_payload
    assert table_payload == before


def test_mapping_validates_supplied_table_even_without_export(table_payload):
    from valleyscope.analysis.reduced_ebr_mapping import build_reduced_ebr_mapping
    table_payload["ebrs"][0]["vector"] = [True]
    with pytest.raises(ValueError, match="vector must be nonnegative integers"):
        build_reduced_ebr_mapping(ebr_export_bundle=None, table=table_payload)


@pytest.mark.parametrize("invalid_table", [[], {}, {"expected_hsps": None}])
def test_direct_table_consumers_reject_malformed_structure(invalid_table):
    from valleyscope.analysis.reduced_ebr_mapping import (
        build_reduced_ebr_mapping, promote_bundle_for_solve,
    )
    with pytest.raises(ValueError, match="table"):
        build_reduced_ebr_mapping(ebr_export_bundle={"bundles": []}, table=invalid_table)
    result = promote_bundle_for_solve(bundle={}, table=invalid_table)
    assert result["promoted"] is False
    assert result["promoted_bundle"] is None
    assert result["blocker_reasons"][0]["code"] == "table_structure_invalid"


@pytest.mark.parametrize("invalid_entry", [True, False])
def test_promotion_blocks_boolean_table_before_physical_validation(table_payload, invalid_entry):
    """No synthetic readiness: malformed tables are rejected even with no bundle evidence."""
    from valleyscope.analysis.reduced_ebr_mapping import promote_bundle_for_solve
    table_payload["irreps"] = ["GM:GM1", "GM:GM2"]
    table_payload["ebrs"][0]["vector"] = [1, invalid_entry]
    result = promote_bundle_for_solve(bundle={}, table=table_payload)
    assert result["promoted"] is False
    assert result["promoted_bundle"] is None
    assert result["irrep_vector"] is None
    assert result["canonical_state"] == "sampled_basis"
    assert [row["code"] for row in result["blocker_reasons"]] == ["table_structure_invalid"]
    assert "vector must be nonnegative integers" in result["blocker_reasons"][0]["detail"]
    assert set(result["validation_report"].values()) == {"not_attempted"}


@pytest.mark.parametrize("invalid_entry", [True, False])
def test_cli_rejects_boolean_table_without_writing_mapping(tmp_path, table_payload, invalid_entry, capsys):
    from valleyscope.cli import main
    table_payload["irreps"] = ["GM:GM1", "GM:GM2"]
    table_payload["ebrs"][0]["vector"] = [1, invalid_entry]
    table_path, bundle_path, output_path = (
        tmp_path / name for name in ("table.json", "bundle.json", "mapping.json")
    )
    table_path.write_text(json.dumps(table_payload), encoding="utf-8")
    bundle_path.write_text(json.dumps({"bundles": []}), encoding="utf-8")
    assert main(["map-reduced-ebr", str(bundle_path), str(table_path), "-o", str(output_path)]) == 1
    assert "vector must be nonnegative integers" in capsys.readouterr().err
    assert not output_path.exists()


def _run_isolated(script, *args, stdlib_only=False):
    # The package root also works when these tests run against an installed wheel.
    root = str(Path(valleyscope.__file__).resolve().parent.parent)
    bootstrap = "import sys\nsys.path.insert(0, " + repr(root) + ")\n"
    return subprocess.run(
        [sys.executable, "-B", *(["-S"] if stdlib_only else []), "-c",
         bootstrap + textwrap.dedent(script), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_table_loader_works_without_numerical_dependencies(tmp_path, table_payload):
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table_payload), encoding="utf-8")
    result = _run_isolated("""
        import json
        from valleyscope.analysis.reduced_ebr_table import load_reduced_ebr_table
        print(json.dumps(load_reduced_ebr_table(sys.argv[1]), sort_keys=True))
        assert "valleyscope.analysis.reduced_ebr_mapping" not in sys.modules
        assert "numpy" not in sys.modules
        assert "spglib" not in sys.modules
    """, path, stdlib_only=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == table_payload


@pytest.mark.parametrize("entrypoint", ["builder", "cli"])
def test_table_building_does_not_import_promotion(tmp_path, entrypoint):
    """A synthetic source isolates table construction, not physical acceptance."""
    spec = {
        "schema_version": "1.0.0",
        "data_source": "irreptables",
        "space_group_number": 1,
        "spinful": False,
        "source_hsp_by_irrep": {"GM1": "GM", "A1": "A"},
        "valleyscope_key_by_source_irrep": {"GM1": "GM:GM1", "A1": "A:A1"},
        "expected_hsps": ["GM"],
        "allowed_irrep_keys": ["GM:GM1"],
        "subspace_group_candidate": "P1",
        "provenance": {"review_status": "fixture-only", "nested": {"notes": ["retain"]}},
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    result = _run_isolated("""
        import importlib.abc
        import json
        from pathlib import Path

        class NoPromotion(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "valleyscope.analysis.reduced_ebr_mapping":
                    raise AssertionError("table construction must not import promotion")
                return None

        sys.meta_path.insert(0, NoPromotion())
        import valleyscope.analysis.irreptables_runtime_table_builder as builder

        def source_loader(space_group_number, spinful):
            assert (space_group_number, spinful) == (1, False)
            return {
                "basis": {"irrep_labels": ["GM1", "A1"], "degeneracies": [1, 1]},
                "ebrs": [
                    {"ebr_name": "kept", "vector": [1, 0]},
                    {"ebr_name": "filtered", "vector": [0, 1]},
                ],
            }

        spec_path = Path(sys.argv[1])
        output_path = spec_path.with_name("table.json")
        if sys.argv[2] == "builder":
            table = builder.build_reduced_table_from_spec_file(
                spec_path, source_loader=source_loader,
            )
        else:
            builder._load_ebr_data_from_irreptables = source_loader
            from valleyscope.cli import main
            assert main(["build-reduced-ebr-table", str(spec_path),
                         "--output", str(output_path)]) == 0
            table = json.loads(output_path.read_text(encoding="utf-8"))

        assert table["irreps"] == ["GM:GM1"]
        assert [(row["label"], row["vector"]) for row in table["ebrs"]] == [("kept", [1])]
        assert table["provenance"]["review_status"] == "fixture-only"
        assert table["provenance"]["nested"] == {"notes": ["retain"]}
        assert table["provenance"]["filtered_zero_vector_ebrs"] == ["filtered"]
        assert "valleyscope.analysis.reduced_ebr_mapping" not in sys.modules
        print("table construction isolated from promotion")
    """, spec_path, entrypoint)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "table construction isolated from promotion" in result.stdout
