#!/usr/bin/env python3
"""Fit an auditable ridge surrogate from structures and conductivity labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rewards.calculators.ionic.descriptors import (  # noqa: E402
    SURROGATE_FEATURE_NAMES,
    DEFAULT_CARRIER_FRACTION_TARGET,
    DEFAULT_HOP_CUTOFF,
    featurize_structure,
    normalize_mobile_species,
    resolve_charge_number,
)
from rewards.calculators.ionic.io import load_structure_file  # noqa: E402
from rewards.calculators.ionic.surrogate import (  # noqa: E402
    DEFAULT_TARGET,
    cross_validated_rmse,
    fit_linear_surrogate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a standardized ridge model to measured or simulated ionic "
            "conductivity. Input paths are resolved relative to the CSV file."
        )
    )
    parser.add_argument("dataset", help="CSV containing structure_path and target")
    parser.add_argument("--output", default="models/ionic_surrogate.json")
    parser.add_argument("--path-column", default="structure_path")
    parser.add_argument("--target-column", default=DEFAULT_TARGET)
    parser.add_argument(
        "--target-scale",
        choices=("log10", "linear"),
        default="log10",
        help="Convert linear S/cm values to log10 before fitting",
    )
    parser.add_argument("--mobile-species", default="Ca")
    parser.add_argument(
        "--charge-number", type=float, help="Positive ionic valence (Ca: 2)"
    )
    parser.add_argument(
        "--carrier-fraction-target", type=float, default=DEFAULT_CARRIER_FRACTION_TARGET
    )
    parser.add_argument("--temperature-k", type=float, default=298.15)
    parser.add_argument("--hop-cutoff", type=float, default=DEFAULT_HOP_CUTOFF)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--descriptors-output",
        help="Optional CSV containing the exact training descriptors",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    mobile_species = normalize_mobile_species(args.mobile_species)
    charge_number = resolve_charge_number(mobile_species, args.charge_number)

    with dataset.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("The training CSV is empty")
    required = {args.path_column, args.target_column}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(
            f"Training CSV is missing columns: {', '.join(sorted(missing))}"
        )

    records: list[dict[str, float]] = []
    targets: list[float] = []
    audit_rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        source = Path(row[args.path_column]).expanduser()
        if not source.is_absolute():
            source = dataset.parent / source
        try:
            structures = load_structure_file(source)
            if len(structures) != 1:
                raise ValueError(
                    "each training row must point to exactly one structure"
                )
            record = featurize_structure(
                structures[0],
                mobile_species=mobile_species,
                temperature_k=args.temperature_k,
                hop_cutoff=args.hop_cutoff,
                charge_number=charge_number,
                carrier_fraction_target=args.carrier_fraction_target,
            ).as_dict()
            if record["mobile_fraction"] <= 0:
                raise ValueError(
                    "training structure contains none of the selected mobile ion"
                )
            target = float(row[args.target_column])
            if args.target_scale == "linear":
                if target <= 0:
                    raise ValueError("linear conductivity targets must be positive")
                target = math.log10(target)
            if not np.isfinite(target) or not all(
                np.isfinite(record[name]) for name in SURROGATE_FEATURE_NAMES
            ):
                raise ValueError("target and surrogate features must be finite")
        except Exception as exc:
            skipped.append(f"row {row_number}: {type(exc).__name__}: {exc}")
            continue
        records.append(record)
        targets.append(target)
        audit_rows.append(
            {
                "structure_path": str(source.resolve()),
                DEFAULT_TARGET: target,
                **record,
            }
        )

    if len(records) < 3:
        details = "\n".join(skipped[:5])
        raise SystemExit(
            f"At least three valid rows are required; found {len(records)}.\n{details}"
        )

    rmse = cross_validated_rmse(
        records,
        targets,
        ridge_alpha=args.ridge_alpha,
        folds=args.folds,
        seed=args.seed,
    )
    model = fit_linear_surrogate(
        records,
        targets,
        ridge_alpha=args.ridge_alpha,
        target_name=DEFAULT_TARGET,
        validation_rmse=rmse,
        metadata={
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "training_samples": len(records),
            "source_dataset": dataset.name,
            "source_dataset_sha256": sha256_file(dataset),
            "source_target_column": args.target_column,
            "source_target_scale": args.target_scale,
            "mobile_species": list(mobile_species),
            "charge_number": charge_number,
            "carrier_fraction_target": args.carrier_fraction_target,
            "temperature_k": args.temperature_k,
            "hop_cutoff": args.hop_cutoff,
            "ridge_alpha": args.ridge_alpha,
            "cv_folds": min(max(2, args.folds), len(records)),
            "cv_seed": args.seed,
            "skipped_rows": len(skipped),
        },
    )
    output = model.save(args.output)

    if args.descriptors_output:
        descriptor_output = Path(args.descriptors_output).expanduser().resolve()
        descriptor_output.parent.mkdir(parents=True, exist_ok=True)
        with descriptor_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
            writer.writeheader()
            writer.writerows(audit_rows)

    print(
        json.dumps(
            {
                "model": str(output.resolve()),
                "samples": len(records),
                "skipped": len(skipped),
                "cross_validated_rmse_log10_s_cm": rmse,
            },
            indent=2,
        )
    )
    if skipped:
        print("Skipped rows:", file=sys.stderr)
        print("\n".join(skipped), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
