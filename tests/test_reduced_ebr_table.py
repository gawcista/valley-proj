"""Structural table loading stays separate from numerical promotion trust."""

import importlib
import json
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
