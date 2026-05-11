from __future__ import annotations

import argparse
from pathlib import Path

from valley_proj.workflows.extract_wavecar import extract_wavecar_to_h5
from valley_proj.workflows.analyze_hsp import analyze_hsp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="valley-proj")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze-hsp", help="Run HSP valley projection diagnostics")
    analyze.add_argument("config", type=Path)
    extract = subparsers.add_parser("extract-wavecar", help="Extract selected WAVECAR coefficients to V1 HDF5")
    extract.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    if args.command == "analyze-hsp":
        outputs = analyze_hsp(args.config)
        print(f"Wrote valley analysis outputs to {next(iter(outputs.values())).parent}")
        return 0
    if args.command == "extract-wavecar":
        output = extract_wavecar_to_h5(args.config)
        print(f"Wrote selected wavefunctions to {output}")
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
