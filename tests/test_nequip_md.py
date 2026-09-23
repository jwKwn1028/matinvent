from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice, Structure

from rewards.calculators import NequIPMD
from rewards.calculators.nequip_md.analysis import (
    diffusion_statistics,
    nernst_einstein_conductivity,
    read_dump,
)


def brownian_trajectory(frames=401, mobile=256, host=64, seed=7, interval=0.1):
    random = np.random.default_rng(seed)
    positions = np.zeros((frames, mobile + host, 3))
    # D = 1 A^2/ps => per-axis increment variance = 2 D dt.
    positions[1:, :mobile] = np.cumsum(
        random.normal(scale=np.sqrt(2 * interval), size=(frames - 1, mobile, 3)),
        axis=0,
    )
    mask = np.arange(mobile + host) < mobile
    return positions, mask, np.ones(mobile + host)


class DiffusionAnalysisTests(unittest.TestCase):
    def test_brownian_diffusion_and_unit_conversion(self):
        positions, mask, masses = brownian_trajectory()
        result, msd = diffusion_statistics(positions, mask, masses, 0.1)
        self.assertAlmostEqual(result["tracer_diffusivity_cm2_s"], 1e-4, delta=1.5e-5)
        self.assertGreater(result["msd_r_squared"], 0.98)
        self.assertAlmostEqual(result["msd_exponent"], 1.0, delta=0.15)
        self.assertEqual(result["framework_msd_a2"], 0.0)
        self.assertEqual(msd.shape[1], 3)

    def test_framework_drift_removed_without_removing_mobile_collective_motion(self):
        positions, mask, masses = brownian_trajectory()
        baseline, _ = diffusion_statistics(positions, mask, masses, 0.1)
        drift = np.arange(len(positions))[:, None, None] * np.array([1.5, -0.2, 0.4])
        shifted, _ = diffusion_statistics(positions + drift, mask, masses, 0.1)
        self.assertAlmostEqual(
            baseline["tracer_diffusivity_cm2_s"], shifted["tracer_diffusivity_cm2_s"]
        )
        # All mobile ions follow one trajectory. Subtracting mobile COM would
        # erase this motion; framework referencing must retain it.
        positions[:, mask] = positions[:, :1]
        collective, _ = diffusion_statistics(positions, mask, masses, 0.1)
        self.assertGreater(collective["msd_growth_a2"], 1.0)

    def test_ballistic_motion_is_identified_by_exponent(self):
        positions, mask, masses = brownian_trajectory()
        positions[:, mask] = np.arange(len(positions))[:, None, None] * 0.02
        result, _ = diffusion_statistics(positions, mask, masses, 0.1)
        self.assertAlmostEqual(result["msd_exponent"], 2.0)

    def test_nernst_einstein_units(self):
        # n = 1e28 m^-3, D = 1e-10 m^2/s, T = 300 K.
        value = nernst_einstein_conductivity(1e-6, 10, 1000, 300, charge_number=1)
        self.assertAlmostEqual(value, 0.06197496, places=7)
        calcium = nernst_einstein_conductivity(1e-6, 10, 1000, 300, charge_number=2)
        self.assertAlmostEqual(calcium, 4.0 * value)


class NequIPMDTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "test.nequip.pth"
        self.model.write_bytes(b"test fixture, not a trained potential")
        self.structure = Structure(
            Lattice.cubic(5),
            ["Ca", "Ca", "P", "S"],
            [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5]],
        )

    def calculator(self, **kwargs):
        options = dict(
            root_dir=str(self.root / "results"),
            model_path=str(self.model),
            lammps_command=shutil.which("true"),
            type_names={"Ca": "calcium", "P": "P", "S": "S"},
            temperature_k=800,
            min_cell_length_a=15,
            seeds=[17, 29],
            production_steps=20000,
            sample_interval=100,
            min_r_squared=0.8,
            exponent_bounds=[0.5, 1.5],
            max_replica_cv=1.0,
        )
        options.update(kwargs)
        return NequIPMD(**options)

    def fake_lammps(self, args, *, cwd, **kwargs):
        # Only replaces execution; real structure export, input generation,
        # dump parsing, analysis and reward conversion run normally.
        from ase.io import read

        atoms = read(
            str(cwd / "structure.data"), format="lammps-data", atom_style="atomic"
        )
        elements = sorted(set(atoms.get_chemical_symbols()))
        symbols = atoms.get_chemical_symbols()
        mask = np.asarray([symbol == "Ca" for symbol in symbols])
        frames = 201
        random = np.random.default_rng(int(cwd.name.split("_")[-1]))
        positions = np.tile(atoms.positions, (frames, 1, 1))
        positions[1:, mask] += np.cumsum(
            random.normal(scale=np.sqrt(0.2), size=(frames - 1, mask.sum(), 3)),
            axis=0,
        )
        mode = getattr(self, "fixture_mode", "diffusive")
        if mode == "immobile":
            positions[:] = atoms.positions
        elif mode == "melting":
            positions[1:, ~mask] += np.cumsum(
                random.normal(scale=1.0, size=(frames - 1, (~mask).sum(), 3)),
                axis=0,
            )
        with (cwd / "trajectory.lammpstrj").open("w") as handle:
            for frame, coordinates in enumerate(positions):
                handle.write(
                    f"ITEM: TIMESTEP\n{frame * 100}\nITEM: NUMBER OF ATOMS\n{len(atoms)}\n"
                )
                handle.write("ITEM: BOX BOUNDS pp pp pp\n0 15\n0 15\n0 15\n")
                handle.write("ITEM: ATOMS id type xu yu zu\n")
                # Reverse row order to exercise stable ID sorting.
                for index in reversed(range(len(atoms))):
                    x, y, z = coordinates[index]
                    handle.write(
                        f"{index + 1} {elements.index(symbols[index]) + 1} {x} {y} {z}\n"
                    )
        np.savetxt(
            cwd / "temperature.dat",
            np.column_stack(
                (
                    np.arange(0, 20001, 100),
                    np.full(201, 2000 if mode == "hot" else 800),
                )
            ),
        )
        return subprocess.CompletedProcess(args, 0)

    def test_end_to_end_adapter_with_synthetic_trajectories(self):
        calculator = self.calculator()
        with patch(
            "rewards.calculators.nequip_md.calc.subprocess.run",
            side_effect=self.fake_lammps,
        ) as run:
            result = calculator.calc_many(([self.structure], "unused"), "step/one")
        self.assertEqual(run.call_count, 2)
        self.assertAlmostEqual(result["tracer_diffusivity_cm2_s"][0], 1e-4, delta=3e-5)
        self.assertTrue(np.isfinite(result["log10_tracer_diffusivity_cm2_s"]).all())
        batch = next((self.root / "results").glob("step_one_*"))
        protocol = json.loads((batch / "protocol.json").read_text())
        self.assertEqual(len(protocol["model_sha256"]), 64)
        script = (batch / "candidate_0000/seed_17/in.lammps").read_text()
        self.assertIn(f'pair_coeff * * "{self.model}" calcium P S', script)
        self.assertEqual(protocol["mobile_species"], "Ca")
        self.assertEqual(protocol["charge_number"], 2.0)
        expected_sigma = nernst_einstein_conductivity(
            result["tracer_diffusivity_cm2_s"][0], 54, 15**3, 800, 2
        )
        self.assertAlmostEqual(result["conductivity_ne_s_cm"][0], expected_sigma)
        self.assertIn("xu yu zu", script)
        self.assertNotIn("com yes", script)
        self.assertTrue((batch / "candidate_0000/seed_17/msd.csv").is_file())
        with (batch / "results.csv").open() as handle:
            self.assertEqual(next(csv.DictReader(handle))["error"], "")

    def test_failure_is_nan_and_audited_with_no_proxy_fallback(self):
        calculator = self.calculator()
        with patch(
            "rewards.calculators.nequip_md.calc.subprocess.run",
            side_effect=subprocess.TimeoutExpired("lmp", 1),
        ):
            result = calculator.calc_many(([self.structure], "unused"))
        self.assertTrue(np.isnan(result["log10_tracer_diffusivity_cm2_s"]).all())
        error = next(
            (self.root / "results").glob("*/candidate_0000/error.txt")
        ).read_text()
        self.assertIn("TimeoutExpired", error)

    def test_unresolved_or_non_solid_trajectories_are_rejected(self):
        for mode, reason in [
            ("immobile", "insufficient resolved ion motion"),
            ("melting", "framework motion"),
            ("hot", "temperature differs"),
        ]:
            with self.subTest(mode=mode):
                self.fixture_mode = mode
                calculator = self.calculator()
                with patch(
                    "rewards.calculators.nequip_md.calc.subprocess.run",
                    side_effect=self.fake_lammps,
                ):
                    result = calculator.calc_many(([self.structure], "unused"), mode)
                self.assertTrue(np.isnan(result["tracer_diffusivity_cm2_s"]).all())
                error = next(
                    (self.root / "results").glob(f"{mode}_*/candidate_0000/error.txt")
                ).read_text()
                self.assertIn(reason, error)

    def test_md_gpu_assignment_does_not_change_generator_environment(self):
        calculator = self.calculator(cuda_visible_devices="GPU-md")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-generator"}):
            with patch(
                "rewards.calculators.nequip_md.calc.subprocess.run",
                side_effect=self.fake_lammps,
            ) as run:
                calculator.calc_many(([self.structure], "unused"))
            self.assertEqual(
                run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-md"
            )
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-generator")

    def test_default_hydra_reward_uses_md_and_propagates_failure(self):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        path = (
            Path(__file__).resolve().parents[1] / "configs/reward/ionic_conductor.yaml"
        )
        config = OmegaConf.load(path)
        config.root_dir = str(self.root / "reward")
        config.calculator.root_dir = str(self.root / "descriptors")
        config.prop_cfg[0].calculator.root_dir = str(self.root / "md")
        environment = {
            "NEQUIP_MODEL_PATH": str(self.model),
            "LAMMPS_COMMAND": shutil.which("true"),
            "IONIC_MD_TEMPERATURE_K": "800",
        }
        with patch.dict(os.environ, environment):
            reward = instantiate(config)
        with patch(
            "rewards.calculators.nequip_md.calc.subprocess.run",
            side_effect=subprocess.TimeoutExpired("lmp", 1),
        ):
            scores, properties, failed = reward.scoring(([self.structure], "unused"))
        self.assertEqual(scores.tolist(), [0.0])
        self.assertEqual(failed.tolist(), [True])
        self.assertIn("log10_tracer_diffusivity_cm2_s", properties)
        self.assertNotIn("transport_score", properties)

    def test_unsupported_chemistry_rejected_before_execution(self):
        calculator = self.calculator(type_names={"Ca": "Ca", "S": "S"})
        with patch("rewards.calculators.nequip_md.calc.subprocess.run") as run:
            result = calculator.calc_many(([self.structure], "unused"))
        run.assert_not_called()
        self.assertTrue(np.isnan(result["tracer_diffusivity_cm2_s"]).all())

    def test_skewed_cell_supercell_uses_face_distances(self):
        calculator = self.calculator(min_cell_length_a=12)
        structure = self.structure.copy()
        structure.lattice = Lattice([[5, 0, 0], [4, 3, 0], [0, 0, 5]])
        atoms, _, repeats = calculator._prepare_structure(structure)
        self.assertEqual(repeats, [4, 4, 3])
        self.assertEqual(len(atoms), 4 * 4 * 4 * 3)

    def test_missing_runtime_fails_at_setup(self):
        with self.assertRaises(FileNotFoundError):
            self.calculator(lammps_command="missing-lammps-for-test")
        with self.assertRaises(FileNotFoundError):
            self.calculator(model_path=str(self.root / "missing.nequip.pth"))

    def test_dump_requires_unwrapped_coordinates(self):
        calculator = self.calculator()
        with patch(
            "rewards.calculators.nequip_md.calc.subprocess.run",
            side_effect=self.fake_lammps,
        ):
            calculator.calc_many(([self.structure], "unused"))
        dump = next(
            (self.root / "results").glob(
                "*/candidate_0000/seed_17/trajectory.lammpstrj"
            )
        )
        dump.write_text(dump.read_text().replace("id type xu yu zu", "id type x y z"))
        with self.assertRaisesRegex(ValueError, "unwrapped"):
            read_dump(dump)

    def test_replica_inconsistency_is_unresolved(self):
        calculator = self.calculator(max_replica_cv=0.1)
        with patch.object(
            calculator,
            "_run_replica",
            side_effect=[
                {"tracer_diffusivity_cm2_s": 1e-6},
                {"tracer_diffusivity_cm2_s": 1e-4},
            ],
        ):
            result = calculator.calc_many(([self.structure], "unused"))
        self.assertTrue(np.isnan(result["tracer_diffusivity_cm2_s"]).all())


if __name__ == "__main__":
    unittest.main()
