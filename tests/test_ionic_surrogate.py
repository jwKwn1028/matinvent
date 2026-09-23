from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rewards.calculators.ionic.calc import IonicConductivity
from rewards.calculators.ionic.surrogate import (
    LinearSurrogate,
    cross_validated_rmse,
    fit_linear_surrogate,
)


class IonicSurrogateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = ("mobile_fraction", "free_volume_fraction")
        self.records = [
            {"mobile_fraction": 0.10, "free_volume_fraction": 0.30},
            {"mobile_fraction": 0.20, "free_volume_fraction": 0.40},
            {"mobile_fraction": 0.30, "free_volume_fraction": 0.35},
            {"mobile_fraction": 0.40, "free_volume_fraction": 0.50},
            {"mobile_fraction": 0.50, "free_volume_fraction": 0.45},
        ]
        self.targets = [
            4.0 * row["mobile_fraction"] + row["free_volume_fraction"] - 5.0
            for row in self.records
        ]

    def test_model_round_trip_preserves_predictions(self) -> None:
        model = fit_linear_surrogate(
            self.records,
            self.targets,
            feature_names=self.features,
            ridge_alpha=1e-8,
            validation_rmse=0.2,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = model.save(Path(tmpdir) / "model.json")
            loaded = LinearSurrogate.load(path)

        np.testing.assert_allclose(
            loaded.predict(self.records), model.predict(self.records), atol=1e-12
        )
        np.testing.assert_allclose(loaded.applicability(self.records), 1.0)
        np.testing.assert_allclose(loaded.uncertainty(self.records), 0.2)

    def test_out_of_domain_point_has_lower_applicability(self) -> None:
        model = fit_linear_surrogate(
            self.records,
            self.targets,
            feature_names=self.features,
        )
        inside = model.applicability([self.records[2]])[0]
        outside = model.applicability(
            [{"mobile_fraction": 1.2, "free_volume_fraction": 0.9}]
        )[0]

        self.assertEqual(inside, 1.0)
        self.assertLess(outside, inside)

    def test_cross_validation_returns_finite_error(self) -> None:
        rmse = cross_validated_rmse(
            self.records,
            self.targets,
            feature_names=self.features,
            ridge_alpha=0.1,
            folds=3,
        )
        self.assertTrue(np.isfinite(rmse))
        self.assertGreaterEqual(rmse, 0.0)

    def test_calculator_rejects_mobile_ion_mismatch(self) -> None:
        model = fit_linear_surrogate(
            self.records,
            self.targets,
            feature_names=self.features,
            metadata={"mobile_species": ["Li"]},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = model.save(Path(tmpdir) / "model.json")
            with self.assertRaisesRegex(ValueError, "trained for"):
                IonicConductivity(
                    root_dir=tmpdir,
                    mobile_species="Ca",
                    model_path=str(path),
                )

    def test_calcium_surrogate_rejects_monovalent_or_missing_metadata(self) -> None:
        for metadata, message in [
            ({}, "metadata missing"),
            ({"mobile_species": ["Ca"], "charge_number": 1.0}, "charge_number"),
        ]:
            with (
                self.subTest(metadata=metadata),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                model = fit_linear_surrogate(
                    self.records,
                    self.targets,
                    feature_names=self.features,
                    metadata=metadata,
                )
                path = model.save(Path(tmpdir) / "model.json")
                with self.assertRaisesRegex(ValueError, message):
                    IonicConductivity(root_dir=tmpdir, model_path=str(path))


if __name__ == "__main__":
    unittest.main()
