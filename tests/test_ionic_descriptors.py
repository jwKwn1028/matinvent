from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

from rewards.calculators import IonicConductivity
from rewards.calculators.ionic.descriptors import featurize_structure


def toy_structure(lattice_constant: float, mobile: str = "Li") -> Structure:
    return Structure(
        Lattice.cubic(lattice_constant),
        [mobile, "O"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


class IonicDescriptorTests(unittest.TestCase):
    def test_periodic_mobile_network_distinguishes_connected_cells(self) -> None:
        connected = featurize_structure(toy_structure(3.0), hop_cutoff=3.2)
        disconnected = featurize_structure(toy_structure(6.0), hop_cutoff=3.2)

        self.assertEqual(connected.mobile_sublattice_dimensionality, 3.0)
        self.assertEqual(disconnected.mobile_sublattice_dimensionality, 0.0)
        self.assertGreater(connected.transport_score, disconnected.transport_score)

    def test_missing_requested_mobile_ion_receives_zero_transport_score(self) -> None:
        descriptor = featurize_structure(toy_structure(4.0, mobile="Na"))

        self.assertEqual(descriptor.mobile_fraction, 0.0)
        self.assertEqual(descriptor.transport_score, 0.0)

    def test_invalid_physical_options_are_rejected(self) -> None:
        structure = toy_structure(3.0)
        with self.assertRaisesRegex(ValueError, "temperature_k"):
            featurize_structure(structure, temperature_k=0)
        with self.assertRaisesRegex(ValueError, "hop_cutoff"):
            featurize_structure(structure, hop_cutoff=-1)

    def test_batch_calculator_writes_auditable_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            calculator = IonicConductivity(root_dir=tmpdir, hop_cutoff=3.2)
            results = calculator.calc_many(
                ([toy_structure(3.0), toy_structure(6.0)], "unused.xyz"),
                label="step/one",
            )

            self.assertIn("transport_score", results)
            self.assertEqual(results["transport_score"].shape, (2,))
            self.assertTrue(np.isfinite(results["transport_score"]).all())
            output = Path(tmpdir) / "step_one.csv"
            self.assertTrue(output.exists())
            self.assertIn("mobile_sublattice_dimensionality", output.read_text())


if __name__ == "__main__":
    unittest.main()
