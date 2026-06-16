from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from valleyscope.workflows.extract_wavecar import extract_wavecar_to_h5
from valleyscope.workflows.analyze_hsp import analyze_hsp
from valleyscope.analysis.reduced_ebr_mapping import (
    load_reduced_ebr_table,
    build_reduced_ebr_mapping,
)
from valleyscope.reports.json_report import write_json
from valleyscope.analysis.database_ingestion_record import (
    load_database_ingestion_record_from_directory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="valleyscope")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze-hsp", help="Run HSP valley projection diagnostics")
    analyze.add_argument("config", type=Path)
    extract = subparsers.add_parser("extract-wavecar", help="Extract selected WAVECAR coefficients to V1 HDF5")
    extract.add_argument("config", type=Path)
    _add_map_reduced_ebr_parser(subparsers)
    _add_collect_database_record_parser(subparsers)
    _add_collect_database_index_parser(subparsers)
    _add_build_reduced_ebr_table_parser(subparsers)
    _add_inspect_source_basis_parser(subparsers)
    _add_scaffold_spec_parser(subparsers)
    _add_validate_spec_parser(subparsers)
    args = parser.parse_args(argv)
    if args.command == "analyze-hsp":
        outputs = analyze_hsp(args.config)
        if outputs.get("summary_stdout", True):
            print(str(outputs["summary_text"]), end="")
        return 0
    if args.command == "extract-wavecar":
        output = extract_wavecar_to_h5(args.config)
        print(f"Wrote selected wavefunctions to {output}")
        return 0
    if args.command == "map-reduced-ebr":
        return _map_reduced_ebr(args)
    if args.command == "collect-database-record":
        return _collect_database_record(args)
    if args.command == "collect-database-index":
        return _collect_database_index(args)
    if args.command == "build-reduced-ebr-table":
        return _build_reduced_ebr_table(args)
    if args.command == "inspect-ebr-source":
        return _inspect_ebr_source(args)
    if args.command == "scaffold-spec":
        return _scaffold_spec(args)
    if args.command == "validate-spec":
        return _validate_spec(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


# ---------------------------------------------------------------------------
# map-reduced-ebr
# ---------------------------------------------------------------------------

def _add_map_reduced_ebr_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "map-reduced-ebr",
        help="Offline exact-integer reduced EBR mapping from export bundle + external/reviewed table",
    )
    p.add_argument(
        "bundle",
        type=Path,
        help="Path to valley_ebr_export_bundle.json",
    )
    p.add_argument(
        "table",
        nargs="?",
        type=Path,
        default=None,
        help="Path to external reduced EBR table JSON (use --table-name for reviewed package-data tables)",
    )
    p.add_argument(
        "--table-name",
        type=str,
        default=None,
        help="Name of a reviewed package-data table (loads via catalog; mutually exclusive with positional table)",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("valley_reduced_ebr_mapping.json"),
        help="Output path for valley_reduced_ebr_mapping.json (default: %(default)s)",
    )
    p.add_argument(
        "--max-coefficient",
        type=int,
        default=6,
        help="Max coefficient per EBR in brute-force search (default: %(default)s)",
    )


def _map_reduced_ebr(args) -> int:
    bundle_path = Path(args.bundle)
    output_path = Path(args.output)
    max_coeff = int(args.max_coefficient)

    # --- Resolve table source ---
    table_path = args.table
    table_name = args.table_name
    if table_path is not None and table_name is not None:
        print(
            "error: positional table and --table-name are mutually exclusive",
            file=sys.stderr,
        )
        return 1
    if table_path is None and table_name is None:
        print(
            "error: either a positional external table path or "
            "--table-name is required",
            file=sys.stderr,
        )
        return 1

    # Load export bundle.
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: cannot read export bundle '{bundle_path}': {exc}", file=sys.stderr)
        return 1

    # Load table.
    if table_path is not None:
        try:
            table = load_reduced_ebr_table(table_path)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            print(f"error: cannot load external reduced EBR table "
                  f"'{table_path}': {exc}", file=sys.stderr)
            return 1
    else:
        try:
            from valleyscope.data.reduced_ebr.catalog import (
                load_reviewed_reduced_ebr_table,
            )
            table = load_reviewed_reduced_ebr_table(table_name)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: cannot load reviewed package-data table "
                  f"{table_name!r}: {exc}", file=sys.stderr)
            return 1

    # Compute mapping.
    mapping = build_reduced_ebr_mapping(
        ebr_export_bundle=bundle,
        table=table,
        max_coefficient=max_coeff,
    )

    # Write output.
    write_json(output_path, mapping)

    # Print compact summary.
    status = mapping.get("status", "?")
    solutions = mapping.get("solutions", [])
    excluded = mapping.get("excluded_bundles", [])
    solved = sum(1 for s in solutions if s.get("status") == "solved_exact")
    unsolved = len(solutions) - solved

    atomic = sum(1 for s in solutions
                 if s.get("classification") == "atomic-compatible-candidate")
    fragile = sum(1 for s in solutions
                  if s.get("classification") == "fragile-topology-candidate")
    stable = sum(1 for s in solutions
                 if s.get("classification") == "stable-topology-candidate")

    print(f"status:              {status}")
    print(f"total bundles:       {len(solutions)}")
    print(f"solved (exact):      {solved}")
    print(f"no exact solution:   {unsolved}")
    if atomic or fragile or stable:
        print(f"  atomic-compatible: {atomic}")
        print(f"  fragile-topology:  {fragile}")
        print(f"  stable-topology:   {stable}")
    print(f"excluded:            {len(excluded)}")
    print(f"reduced EBR mapping: {output_path}")
    return 0


# ---------------------------------------------------------------------------
# collect-database-record
# ---------------------------------------------------------------------------

def _add_collect_database_record_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "collect-database-record",
        help="Build a database ingestion record from a run output directory",
    )
    p.add_argument(
        "run_dir",
        type=Path,
        help="Path to the analyze-hsp output directory",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("database_ingestion_record.json"),
        help="Output path (default: %(default)s)",
    )


def _collect_database_record(args) -> int:
    run_dir = Path(args.run_dir)
    output_path = Path(args.output)

    if not run_dir.is_dir():
        print(f"error: not a directory: {run_dir}", file=sys.stderr)
        return 1

    record = load_database_ingestion_record_from_directory(str(run_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, record)

    status = record.get("record_status", "?")
    bundle_count = record.get("ready_bundle_count", 0)
    irrep_count = len(record.get("valley_irrep_records", []))
    errors = record.get("validation_errors", [])

    print(f"record status:          {status}")
    print(f"ready bundles:          {bundle_count}")
    print(f"trusted irrep records:  {irrep_count}")
    if errors:
        print(f"validation errors:      {len(errors)}")
        for e in errors:
            print(f"  - {e}")
    print(f"ingestion record:       {output_path}")
    if errors:
        return 1
    return 0


# ---------------------------------------------------------------------------
# collect-database-index
# ---------------------------------------------------------------------------

def _add_collect_database_index_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "collect-database-index",
        help="Build a compact multi-run database index from ingestion records",
    )
    p.add_argument(
        "records",
        nargs="+",
        type=Path,
        help="One or more database_ingestion_record.json file paths",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("database_index.json"),
        help="Output path for the index (default: %(default)s)",
    )


def _collect_database_index(args) -> int:
    from valleyscope.analysis.database_index import load_database_index_from_files
    from valleyscope.reports.json_report import write_json

    record_paths = [str(p) for p in args.records]
    index = load_database_index_from_files(record_paths)
    write_json(args.output, index)
    print(f"database_index:        {args.output}")
    print(f"record count:          {index['record_count']}")
    print(f"ready bundle total:    {index['ready_bundle_count_total']}")
    print(f"reduced EBR total:    {index['reduced_ebr_record_count_total']}")
    status = index["status_counts"]
    print(f"status counts:         has_ready={status.get('has_ready_ebr_bundles')} "
          f"no_ready={status.get('no_ready_ebr_bundles')} "
          f"invalid={status.get('invalid_missing_summary')}")
    if index["validation_errors"]:
        return 1
    return 0


# ---------------------------------------------------------------------------
# build-reduced-ebr-table
# ---------------------------------------------------------------------------

def _add_build_reduced_ebr_table_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "build-reduced-ebr-table",
        help="Build a ValleyScope reduced EBR table from an irreptables mapping spec",
    )
    p.add_argument(
        "spec",
        type=Path,
        help=(
            "Path to JSON mapping spec "
            "(space_group_number, spinful, source_hsp_by_irrep, etc.)"
        ),
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("valley_reduced_ebr_table.json"),
        help="Output path for the reduced EBR table (default: %(default)s)",
    )
    p.add_argument(
        "--source-basis",
        type=Path,
        default=None,
        help="Optional path to inspect-ebr-source JSON for preflight validation",
    )


def _build_reduced_ebr_table(args) -> int:
    from valleyscope.analysis.irreptables_runtime_table_builder import (
        build_reduced_table_from_spec_file,
    )

    spec_path = Path(args.spec)
    output_path = Path(args.output)

    if not spec_path.is_file():
        print(f"error: spec file not found: {spec_path}", file=sys.stderr)
        return 1

    # Preflight validation (optional).
    if args.source_basis is not None:
        source_basis_path = Path(args.source_basis)
        if not source_basis_path.is_file():
            print(f"error: source basis file not found: {source_basis_path}", file=sys.stderr)
            return 1
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            source_basis = json.loads(source_basis_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            print(f"error: cannot read preflight inputs: {exc}", file=sys.stderr)
            return 1
        from valleyscope.analysis.reduced_ebr_spec_template_validator import (
            validate_mapping_spec_against_source_basis,
        )
        try:
            result = validate_mapping_spec_against_source_basis(spec, source_basis)
        except ValueError as exc:
            print(f"error: preflight validation failed: {exc}", file=sys.stderr)
            return 1
        if not result["valid"]:
            print("error: preflight validation failed:", file=sys.stderr)
            for e in result["errors"]:
                print(f"  - {e}", file=sys.stderr)
            return 1
        print("preflight validation passed")

    try:
        table = build_reduced_table_from_spec_file(str(spec_path))
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_json(output_path, table)
    try:
        load_reduced_ebr_table(output_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: wrote invalid reduced EBR table '{output_path}': {exc}", file=sys.stderr)
        return 1

    provenance = table.get("provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    print(f"space group number: {provenance.get('space_group_number', '?')}")
    print(f"spinful:            {provenance.get('spinful', '?')}")
    print(f"subspace group:     {table['subspace_group_candidate']}")
    print(f"expected HSPs:      {table['expected_hsps']}")
    print(f"irreps:             {len(table['irreps'])} keys")
    print(f"EBRs:               {len(table['ebrs'])} vectors")
    print(f"filtered zero EBRs: {provenance.get('filtered_zero_vector_ebr_count', 0)}")
    print(f"reduced EBR table:  {output_path}")
    return 0


# ---------------------------------------------------------------------------
# inspect-ebr-source
# ---------------------------------------------------------------------------

def _add_inspect_source_basis_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "inspect-ebr-source",
        help="Inspect irreptables source basis for authoring a reduced EBR spec",
    )
    p.add_argument(
        "--space-group-number",
        type=int,
        required=True,
        help="Projected-subspace / moire space group number (e.g. 150 for P321)",
    )
    p.add_argument(
        "--spinful",
        action="store_true",
        default=True,
        help="Load double-valued (spinor) data (default: true)",
    )
    p.add_argument(
        "--spinless",
        action="store_true",
        default=False,
        help="Load single-valued (spinless) data instead",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        help="Write canonical inspection JSON to this path",
    )


def _inspect_ebr_source(args) -> int:
    from valleyscope.analysis.reduced_ebr_source_basis_inspector import (
        inspect_irreptables_source_basis,
    )
    from valleyscope.reports.json_report import write_json

    spinful = not args.spinless
    try:
        info = inspect_irreptables_source_basis(
            int(args.space_group_number), spinful=spinful,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        write_json(args.output, info)
        print(f"inspection payload: {args.output}")

    print(f"space group number:  {info['space_group_number']}")
    print(f"spinful:             {info['spinful']}")
    print(f"source basis count:  {info['source_basis_count']}")
    print(f"source EBR count:    {info['source_ebr_count']}")
    return 0


# ---------------------------------------------------------------------------
# scaffold-spec
# ---------------------------------------------------------------------------

def _add_scaffold_spec_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "scaffold-spec",
        help="Scaffold a mapping spec template from an inspect-ebr-source JSON",
    )
    p.add_argument(
        "source_basis",
        type=Path,
        help="Path to inspect-ebr-source output JSON",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("spec_template.json"),
        help="Output path for the template (default: %(default)s)",
    )
    p.add_argument(
        "--schema-version",
        choices=["1.0.0", "1.1.0"],
        default="1.0.0",
        help="Schema version for the template (default: 1.0.0)",
    )


def _scaffold_spec(args) -> int:
    import json
    from valleyscope.analysis.reduced_ebr_spec_template_validator import (
        build_mapping_spec_template,
    )
    from valleyscope.reports.json_report import write_json

    source_path = Path(args.source_basis)
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: cannot read source basis '{source_path}': {exc}", file=sys.stderr)
        return 1

    try:
        template = build_mapping_spec_template(source, schema_version=args.schema_version)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_json(args.output, template)
    print(f"template spec:       {args.output}")
    print(f"source labels:       {len(template['source_hsp_by_irrep'])}")
    print(f"schema version:      {template['schema_version']}")
    print(f"Fill all 'REQUIRED_FILL_BY_HUMAN' placeholders before building.")
    return 0


# ---------------------------------------------------------------------------
# validate-spec
# ---------------------------------------------------------------------------

def _add_validate_spec_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "validate-spec",
        help="Preflight-validate a reduced EBR mapping spec against a source basis",
    )
    p.add_argument(
        "spec",
        type=Path,
        help="Path to the mapping spec JSON to validate",
    )
    p.add_argument(
        "source_basis",
        type=Path,
        help="Path to inspect-ebr-source output JSON",
    )


def _validate_spec(args) -> int:
    import json
    from valleyscope.analysis.reduced_ebr_spec_template_validator import (
        validate_mapping_spec_against_source_basis,
    )

    try:
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: cannot read spec '{args.spec}': {exc}", file=sys.stderr)
        return 1
    try:
        source = json.loads(Path(args.source_basis).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: cannot read source basis '{args.source_basis}': {exc}", file=sys.stderr)
        return 1

    try:
        result = validate_mapping_spec_against_source_basis(spec, source)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"valid:  {result['valid']}")
    print(f"summary: {result['summary']}")
    if result["errors"]:
        for e in result["errors"]:
            print(f"  - {e}")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
