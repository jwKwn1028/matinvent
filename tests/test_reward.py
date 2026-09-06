from __future__ import annotations

import tempfile
import unittest

import numpy as np

try:
    from omegaconf import OmegaConf

    from rewards.reward import Reward
except ImportError:
    OmegaConf = None
    Reward = None


class SharedCalculator:
    def __init__(self) -> None:
        self.calls = 0

    def calc_many(self, samples, label="tmp") -> dict[str, np.ndarray]:
        self.calls += 1
        return {"transport_score": np.asarray([0.8, 0.2])}


class IndividualCalculator:
    def calc(self, samples, label="tmp") -> np.ndarray:
        return np.asarray([1.0, 1.0])


@unittest.skipUnless(OmegaConf is not None, "omegaconf is a development dependency")
class RewardTests(unittest.TestCase):
    def test_shared_and_individual_calculators_mix_in_one_reward(self) -> None:
        shared = SharedCalculator()
        objectives = OmegaConf.create(
            [
                {
                    "name": "transport_score",
                    "target": "ascending",
                    "minv": 0.0,
                    "maxv": 1.0,
                    "weight": 0.75,
                },
                {
                    "name": "external",
                    "calculator": IndividualCalculator(),
                    "target": "ascending",
                    "minv": 0.0,
                    "maxv": 2.0,
                    "weight": 0.25,
                },
            ],
            flags={"allow_objects": True},
        )
        constraints = OmegaConf.create(
            [{"name": "transport_score", "min": 0.3, "penalty": 0.0}]
        )
        with tempfile.TemporaryDirectory() as root_dir:
            reward = Reward(
                root_dir=root_dir,
                prop_cfg=objectives,
                reward_threshold=0.7,
                reduce="weight",
                calculator=shared,
                constraint_cfg=constraints,
            )
            scores, properties, failed = reward.scoring(
                ([object(), object()], "unused"), "test"
            )

        np.testing.assert_allclose(scores, [0.725, 0.0])
        self.assertEqual(set(properties), {"transport_score", "external"})
        self.assertFalse(failed.any())
        self.assertEqual(shared.calls, 1)

    def test_weighted_reward_rejects_non_normalized_weights(self) -> None:
        objectives = OmegaConf.create(
            [
                {
                    "name": "transport_score",
                    "target": "ascending",
                    "minv": 0.0,
                    "maxv": 1.0,
                    "weight": 0.5,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as root_dir:
            with self.assertRaisesRegex(ValueError, "sum to 1.0"):
                Reward(
                    root_dir=root_dir,
                    prop_cfg=objectives,
                    reward_threshold=0.7,
                    reduce="weight",
                )


if __name__ == "__main__":
    unittest.main()
