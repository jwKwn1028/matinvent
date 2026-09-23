"""MatInvent reward calculator for solid ionic conductor discovery."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple, List

import numpy as np
from pymatgen.core import Structure

from rewards.calculators.base import Calculator
from rewards.calculators.ionic.descriptors import (
    DESCRIPTOR_NAMES,
    DEFAULT_CARRIER_FRACTION_TARGET,
    DEFAULT_HOP_CUTOFF,
    DEFAULT_MOBILE_SPECIES,
    featurize_structure,
    normalize_mobile_species,
    resolve_charge_number,
)
from rewards.calculators.ionic.surrogate import LinearSurrogate


class IonicConductivity(Calculator):
    """Evaluate structure-only ion-transport descriptors in one batched pass.

    Parameters
    ----------
    task
        Field returned by :meth:`calc` for compatibility with the original
        one-property calculator interface.  :class:`rewards.reward.Reward`
        uses :meth:`calc_many` when this calculator is shared by objectives.
    model_path
        Optional JSON model produced by ``scripts/fit_ionic_surrogate.py``.
        Without it, only explicitly named screening proxies are emitted.
    """

    def __init__(
        self,
        root_dir: str,
        task: str = "transport_score",
        mobile_species: str | Iterable[str] = DEFAULT_MOBILE_SPECIES,
        temperature_k: float = 298.15,
        hop_cutoff: float = DEFAULT_HOP_CUTOFF,
        charge_number: float | None = None,
        carrier_fraction_target: float = DEFAULT_CARRIER_FRACTION_TARGET,
        model_path: str | None = None,
        allow_model_mismatch: bool = False,
        strict: bool = False,
    ) -> None:
        super().__init__(root_dir, task)
        self.mobile_species = normalize_mobile_species(mobile_species)
        self.charge_number = resolve_charge_number(self.mobile_species, charge_number)
        self.carrier_fraction_target = float(carrier_fraction_target)
        if (
            not np.isfinite(self.carrier_fraction_target)
            or not 0 < self.carrier_fraction_target < 1
        ):
            raise ValueError("carrier_fraction_target must be between 0 and 1")
        self.temperature_k = float(temperature_k)
        self.hop_cutoff = float(hop_cutoff)
        self.strict = strict
        self.model = LinearSurrogate.load(model_path) if model_path else None
        if self.model is not None:
            unknown_features = set(self.model.feature_names) - set(DESCRIPTOR_NAMES)
            if unknown_features:
                raise ValueError(
                    "Surrogate requires unsupported descriptors: "
                    + ", ".join(sorted(unknown_features))
                )
            if self.model.target in DESCRIPTOR_NAMES:
                raise ValueError(
                    f"Surrogate target {self.model.target!r} collides with a built-in "
                    "descriptor name"
                )
            if not allow_model_mismatch:
                self._validate_model_context()

    @property
    def available_tasks(self) -> tuple[str, ...]:
        tasks = list(DESCRIPTOR_NAMES)
        if self.model is not None:
            tasks.extend(
                [self.model.target, "surrogate_applicability", "surrogate_uncertainty"]
            )
        return tuple(tasks)

    def evaluate(self, structures: Sequence[Structure]) -> list[dict[str, object]]:
        """Return one descriptor record per structure without writing files."""

        records: list[dict[str, object]] = []
        for structure in structures:
            try:
                record: dict[str, object] = {
                    "formula": structure.composition.reduced_formula,
                    "mobile_species": ",".join(self.mobile_species),
                    "charge_number": self.charge_number,
                    **featurize_structure(
                        structure,
                        mobile_species=self.mobile_species,
                        temperature_k=self.temperature_k,
                        hop_cutoff=self.hop_cutoff,
                        charge_number=self.charge_number,
                        carrier_fraction_target=self.carrier_fraction_target,
                    ).as_dict(),
                    "error": "",
                }
            except Exception as exc:
                if self.strict:
                    raise
                record = {
                    "formula": getattr(
                        getattr(structure, "composition", None),
                        "reduced_formula",
                        "unknown",
                    ),
                    "mobile_species": ",".join(self.mobile_species),
                    "charge_number": self.charge_number,
                    **{name: np.nan for name in DESCRIPTOR_NAMES},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            records.append(record)

        if self.model is not None and records:
            feature_names = self.model.feature_names
            valid = np.asarray(
                [
                    all(np.isfinite(float(record[name])) for name in feature_names)
                    for record in records
                ],
                dtype=bool,
            )
            prediction = np.full(len(records), np.nan, dtype=float)
            applicability = np.full(len(records), np.nan, dtype=float)
            uncertainty = np.full(len(records), np.nan, dtype=float)
            if valid.any():
                valid_records = [record for record, keep in zip(records, valid) if keep]
                prediction[valid] = self.model.predict(valid_records)
                applicability[valid] = self.model.applicability(valid_records)
                uncertainty[valid] = self.model.uncertainty(valid_records)
            for index, record in enumerate(records):
                record[self.model.target] = float(prediction[index])
                record["surrogate_applicability"] = float(applicability[index])
                record["surrogate_uncertainty"] = float(uncertainty[index])

        return records

    def calc_many(
        self,
        samples: Tuple[List[Structure], str],
        label: str = "tmp",
    ) -> dict[str, np.ndarray]:
        """Evaluate all available tasks once and save an auditable CSV table."""

        structures = samples[0]
        records = self.evaluate(structures)
        self._write_records(records, label)
        return {
            task: np.asarray([record[task] for record in records], dtype=float)
            for task in self.available_tasks
        }

    def calc(
        self,
        samples: Tuple[List[Structure], str],
        label: str = "tmp",
    ) -> np.ndarray:
        results = self.calc_many(samples, label)
        if self.task not in results:
            available = ", ".join(sorted(results))
            raise ValueError(
                f"Unknown ionic-conductor task {self.task!r}. Available: {available}"
            )
        return results[self.task]

    def _write_records(
        self, records: Sequence[Mapping[str, object]], label: str
    ) -> Path | None:
        if not records:
            return None
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("._")
        safe_label = safe_label or "batch"
        output = Path(self.root_dir).resolve() / f"{safe_label}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(records[0])
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        return output

    def _validate_model_context(self) -> None:
        """Reject accidental inference under settings unlike model training."""

        metadata = self.model.metadata or {}
        expected_species = metadata.get("mobile_species")
        if expected_species is not None:
            expected = normalize_mobile_species(expected_species)
            if set(expected) != set(self.mobile_species):
                raise ValueError(
                    f"Surrogate was trained for {expected}, but calculator uses "
                    f"{self.mobile_species}. Set matching mobile_species."
                )

        for key, actual in (
            ("charge_number", self.charge_number),
            ("temperature_k", self.temperature_k),
            ("hop_cutoff", self.hop_cutoff),
            ("carrier_fraction_target", self.carrier_fraction_target),
        ):
            expected_value = metadata.get(key)
            if expected_value is not None and not np.isclose(
                float(expected_value), actual, rtol=0.0, atol=1e-8
            ):
                raise ValueError(
                    f"Surrogate was trained with {key}={expected_value}, but "
                    f"calculator uses {actual}. Set matching descriptor settings."
                )
        required = {
            "mobile_species",
            "charge_number",
            "temperature_k",
            "hop_cutoff",
            "carrier_fraction_target",
        }
        missing = {key for key in required if metadata.get(key) is None}
        if missing:
            raise ValueError(
                f"Surrogate metadata missing {sorted(missing)}. Refit with the "
                "current fitting CLI using conductivity data for the selected ion."
            )
