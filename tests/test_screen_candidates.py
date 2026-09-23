from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def write_cif(structure: Structure, path: Path) -> None:
    CifWriter(structure).write_file(path, mode="wt")


class ScreenCandidatesTests(unittest.TestCase):
    def test_default_ca_fit_rejects_structures_without_calcium(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            write_cif(
                Structure(Lattice.cubic(4), ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
                directory / "li.cif",
            )
            dataset = directory / "training.csv"
            with dataset.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["structure_path", "log10_ionic_conductivity_s_cm"])
                writer.writerows([["li.cif", value] for value in (-3, -4, -5)])
            output = directory / "model.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts/fit_ionic_surrogate.py"),
                    str(dataset),
                    "--output",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("none of the selected mobile ion", completed.stderr)
            self.assertFalse(output.exists())

    def test_cli_ranks_and_writes_candidate_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            write_cif(
                Structure(
                    Lattice.cubic(3.0),
                    ["Ca", "O"],
                    [[0, 0, 0], [0.5, 0.5, 0.5]],
                ),
                directory / "connected.cif",
            )
            write_cif(
                Structure(
                    Lattice.cubic(6.0),
                    ["Ca", "O"],
                    [[0, 0, 0], [0.5, 0.5, 0.5]],
                ),
                directory / "disconnected.cif",
            )
            output = directory / "ranking.csv"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts" / "screen_candidates.py"),
                    str(directory),
                    "--hop-cutoff",
                    "3.2",
                    "--output",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertIn("Ranked 2 structures", completed.stdout)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["source"].endswith("connected.cif"))
        self.assertGreater(
            float(rows[0]["transport_score"]), float(rows[1]["transport_score"])
        )

    def test_fit_cli_produces_model_consumable_by_screen_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            dataset = directory / "training.csv"
            with dataset.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["structure_path", "log10_ionic_conductivity_s_cm"])
                for index, lattice_constant in enumerate((3.0, 3.2, 3.4, 4.2, 5.2)):
                    path = directory / f"candidate_{index}.cif"
                    write_cif(
                        Structure(
                            Lattice.cubic(lattice_constant),
                            ["Ca", "O"],
                            [[0, 0, 0], [0.5, 0.5, 0.5]],
                        ),
                        path,
                    )
                    writer.writerow([path.name, -2.5 - lattice_constant / 2.0])

            model = directory / "surrogate.json"
            subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts" / "fit_ionic_surrogate.py"),
                    str(dataset),
                    "--output",
                    str(model),
                    "--folds",
                    "3",
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            output = directory / "calibrated.csv"
            subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts" / "screen_candidates.py"),
                    str(directory / "candidate_0.cif"),
                    "--model",
                    str(model),
                    "--output",
                    str(output),
                ],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            with output.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertTrue(model.exists())
            payload = json.loads(model.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["metadata"]["source_dataset_sha256"]), 64)
            self.assertEqual(payload["metadata"]["mobile_species"], ["Ca"])
            self.assertEqual(payload["metadata"]["charge_number"], 2.0)

        self.assertIn("log10_ionic_conductivity_s_cm", row)
        self.assertIn("surrogate_applicability", row)
        self.assertEqual(row["mobile_species"], "Ca")
        self.assertEqual(float(row["charge_number"]), 2.0)


if __name__ == "__main__":
    unittest.main()
