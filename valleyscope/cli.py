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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="valleyscope")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze-hsp", help="Run HSP valley projection diagnostics")
    analyze.add_argument("config", type=Path)
    extract = subparsers.add_parser("extract-wavecar", help="Extract selected WAVECAR coefficients to V1 HDF5")
    extract.add_argument("config", type=Path)
    _add_map_reduced_ebr_parser(subparsers)
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
    parser.error(f"Unknown command: {args.command}")
    return 2


# ---------------------------------------------------------------------------
# map-reduced-ebr
# ---------------------------------------------------------------------------

def _add_map_reduced_ebr_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "map-reduced-ebr",
        help="Offline exact-integer reduced EBR mapping from export bundle + external table",
    )
    p.add_argument(
        "bundle",
        type=Path,
        help="Path to valley_ebr_export_bundle.json",
    )
    p.add_argument(
        "table",
        type=Path,
        help="Path to external reduced EBR table JSON (required; no built-in tables)",
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
    table_path = Path(args.table)
    output_path = Path(args.output)
    max_coeff = int(args.max_coefficient)

    # Load export bundle.
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: cannot read export bundle '{bundle_path}': {exc}", file=sys.stderr)
        return 1

    # Load and validate external table.
    try:
        table = load_reduced_ebr_table(table_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: cannot load reduced EBR table '{table_path}': {exc}", file=sys.stderr)
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


if __name__ == "__main__":
    raise SystemExit(main())
