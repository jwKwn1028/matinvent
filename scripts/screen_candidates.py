#!/usr/bin/env python3
"""Rank CIF/POSCAR/extxyz candidates with the ionic-conductor reward features."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rewards.calculators.ionic import IonicConductivity  # noqa: E402
from rewards.calculators.ionic.descriptors import (  # noqa: E402
    DEFAULT_CARRIER_FRACTION_TARGET,
    DEFAULT_HOP_CUTOFF,
)
from rewards.calculators.ionic.io import (  # noqa: E402
    discover_structure_paths,
    load_structure_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute transparent ion-transport descriptors and rank crystal "
            "structures before expensive validation."
        )
    )
    parser.add_argument("inputs", nargs="+", help="Structure files or directories")
    parser.add_argument(
        "--mobile-species",
        default="Ca",
        help="Comma-separated mobile ions (default: Ca, interpreted as Ca2+)",
    )
    parser.add_argument("--temperature-k", type=float, default=298.15)
    parser.add_argument("--hop-cutoff", type=float, default=DEFAULT_HOP_CUTOFF)
    parser.add_argument(
        "--charge-number", type=float, help="Positive ionic valence (Ca: 2)"
    )
    parser.add_argument(
        "--carrier-fraction-target", type=float, default=DEFAULT_CARRIER_FRACTION_TARGET
    )
    parser.add_argument(
        "--model",
        help="Optional calibrated surrogate JSON from fit_ionic_surrogate.py",
    )
    parser.add_argument(
        "--rank-by",
        help="Output field to maximize (default: calibrated target or transport_score)",
    )
    parser.add_argument("--top", type=int, help="Keep only the top N rows")
    parser.add_argument("--output", default="ionic_candidates.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = discover_structure_paths(args.inputs)
    if not paths:
        raise SystemExit("No supported structure files were found")

    output = Path(args.output).expanduser().resolve()
    calculator = IonicConductivity(
        root_dir=str(output.parent),
        mobile_species=args.mobile_species,
        temperature_k=args.temperature_k,
        hop_cutoff=args.hop_cutoff,
        charge_number=args.charge_number,
        carrier_fraction_target=args.carrier_fraction_target,
        model_path=args.model,
    )
    structures = []
    sources: list[str] = []
    load_failures: list[dict[str, object]] = []
    for path in paths:
        try:
            loaded = load_structure_file(path)
            if not loaded:
                raise ValueError("structure file contains no frames")
        except Exception as exc:
            load_failures.append(
                {
                    "formula": "unknown",
                    "mobile_species": ",".join(calculator.mobile_species),
                    "charge_number": calculator.charge_number,
                    **{task: math.nan for task in calculator.available_tasks},
                    "error": f"{type(exc).__name__}: {exc}",
                    "source": str(path),
                }
            )
            continue
        structures.extend(loaded)
        sources.extend(
            [
                str(path) if len(loaded) == 1 else f"{path}#{index}"
                for index in range(len(loaded))
            ]
        )

    records = calculator.evaluate(structures)
    for source, record in zip(sources, records):
        record["source"] = source
    records.extend(load_failures)

    rank_by = args.rank_by or (
        calculator.model.target if calculator.model is not None else "transport_score"
    )
    if rank_by not in records[0]:
        available = ", ".join(records[0])
        raise SystemExit(f"Unknown --rank-by field {rank_by!r}. Available: {available}")

    def rank_value(record: dict[str, object]) -> float:
        try:
            value = float(record[rank_by])
            return value if math.isfinite(value) else -math.inf
        except (TypeError, ValueError):
            return -math.inf

    records.sort(key=rank_value, reverse=True)
    if args.top is not None:
        if args.top <= 0:
            raise SystemExit("--top must be positive")
        records = records[: args.top]

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "source"] + [name for name in records[0] if name != "source"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, record in enumerate(records, start=1):
            writer.writerow({"rank": rank, **record})

    failures = sum(bool(record["error"]) for record in records)
    print(
        f"Ranked {len(records)} structures by {rank_by}; "
        f"{failures} failed; wrote {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
