"""Run a compiled NequIP potential in single-rank LAMMPS for each candidate."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from ase.io import write
from pymatgen.core import Element, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from rewards.calculators.base import Calculator
from rewards.calculators.ionic.descriptors import resolve_charge_number
from rewards.calculators.nequip_md.analysis import (
    diffusion_statistics,
    nernst_einstein_conductivity,
    read_dump,
)


class NequIPMD(Calculator):
    """Measured MD tracer diffusion; no descriptor fallback or 298 K extrapolation.

    ``model_path`` must be an already compiled pair_nequip model in eV/angstrom
    units, validated for the requested chemistry and temperature. ``type_names``
    explicitly maps chemical symbols to the model's training-time type names.
    Global setup errors fail immediately. Individual failed or unresolved
    trajectories yield NaN, which Reward treats as failed evaluations.
    """

    def __init__(
        self,
        root_dir: str,
        model_path: str,
        type_names: Mapping[str, str],
        temperature_k: float,
        lammps_command: str = "lmp",
        cuda_visible_devices: str | None = None,
        mobile_species: str = "Ca",
        task: str = "log10_tracer_diffusivity_cm2_s",
        timestep_ps: float = 0.001,
        equilibration_steps: int = 20000,
        production_steps: int = 100000,
        sample_interval: int = 100,
        thermostat_damping_ps: float = 0.1,
        min_cell_length_a: float = 20.0,
        max_atoms: int = 4096,
        seeds: Sequence[int] = (17, 29, 43),
        timeout_seconds: float = 7200,
        fit_start_fraction: float = 0.1,
        fit_end_fraction: float = 0.5,
        min_r_squared: float = 0.9,
        exponent_bounds: Sequence[float] = (0.75, 1.25),
        min_msd_growth_a2: float = 1.0,
        max_framework_msd_a2: float = 2.0,
        max_replica_cv: float = 0.5,
        max_temperature_deviation: float = 0.2,
        charge_number: float | None = None,
        strict: bool = False,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve(strict=True)
        if not self.model_path.is_file():
            raise ValueError("model_path must be a compiled NequIP model file")
        if any(c in str(self.model_path) for c in '\n\r"$'):
            raise ValueError("NequIP model path contains unsupported LAMMPS characters")
        executable = shutil.which(lammps_command)
        if executable is None:
            raise FileNotFoundError(f"LAMMPS executable not found: {lammps_command}")
        self.lammps_command = str(Path(executable).resolve())
        self.cuda_visible_devices = cuda_visible_devices
        self.type_names = dict(type_names)
        if not self.type_names:
            raise ValueError(
                "type_names must explicitly describe the potential's elements"
            )
        for symbol, name in self.type_names.items():
            Element(symbol)
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.+-]+", name):
                raise ValueError(f"Invalid NequIP model type name: {name!r}")
        Element(mobile_species)
        charge_number = resolve_charge_number(mobile_species, charge_number)
        if mobile_species not in self.type_names:
            raise ValueError("The mobile species is absent from type_names")
        positive = {
            "temperature_k": temperature_k,
            "timestep_ps": timestep_ps,
            "thermostat_damping_ps": thermostat_damping_ps,
            "min_cell_length_a": min_cell_length_a,
            "timeout_seconds": timeout_seconds,
            "min_msd_growth_a2": min_msd_growth_a2,
            "max_framework_msd_a2": max_framework_msd_a2,
            "max_replica_cv": max_replica_cv,
            "max_temperature_deviation": max_temperature_deviation,
            "charge_number": charge_number,
        }
        for name, value in positive.items():
            if not np.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, float(value))
        for name, value in {
            "equilibration_steps": equilibration_steps,
            "production_steps": production_steps,
            "sample_interval": sample_interval,
            "max_atoms": max_atoms,
        }.items():
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            setattr(self, name, int(value))
        if (
            production_steps % sample_interval
            or production_steps // sample_interval < 40
        ):
            raise ValueError(
                "Production must contain at least 40 complete sampling intervals"
            )
        self.seeds = [int(seed) for seed in seeds]
        if (
            len(self.seeds) < 2
            or len(set(self.seeds)) != len(self.seeds)
            or any(seed <= 0 or seed >= 900000000 for seed in self.seeds)
            or any(int(seed) != seed for seed in seeds)
        ):
            raise ValueError(
                "Supply at least two distinct integer seeds in (0, 900000000)"
            )
        if not 0 < fit_start_fraction < fit_end_fraction <= 0.5:
            raise ValueError("Fit fractions must satisfy 0 < start < end <= 0.5")
        if not 0 <= min_r_squared <= 1:
            raise ValueError("min_r_squared must be in [0, 1]")
        if (
            len(exponent_bounds) != 2
            or not np.isfinite(exponent_bounds).all()
            or not 0 < exponent_bounds[0] < exponent_bounds[1]
        ):
            raise ValueError("exponent_bounds must be a positive increasing pair")
        self.fit_start_fraction, self.fit_end_fraction = (
            fit_start_fraction,
            fit_end_fraction,
        )
        self.min_r_squared = min_r_squared
        self.exponent_bounds = tuple(exponent_bounds)
        self.mobile_species, self.strict = mobile_species, strict
        digest = hashlib.sha256()
        with self.model_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.model_sha256 = digest.hexdigest()
        super().__init__(str(Path(root_dir).resolve()), task)
        if task not in self.available_tasks:
            raise ValueError(f"Unknown NequIPMD task: {task}")

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return (
            "log10_tracer_diffusivity_cm2_s",
            "tracer_diffusivity_cm2_s",
            "tracer_diffusivity_std_cm2_s",
            "conductivity_ne_s_cm",
        )

    def _prepare_structure(self, structure: Structure):
        if not structure.is_ordered:
            raise ValueError("MD requires explicit ordered atom occupancies")
        elements = sorted(element.symbol for element in structure.composition.elements)
        unsupported = set(elements) - self.type_names.keys()
        if unsupported:
            raise ValueError(
                f"Elements absent from the NequIP mapping: {sorted(unsupported)}"
            )
        if self.mobile_species not in elements or len(elements) < 2:
            raise ValueError("Both mobile ions and framework atoms are required")
        lattice = np.asarray(structure.lattice.matrix)
        volume = abs(float(np.linalg.det(lattice)))
        if not np.isfinite(lattice).all() or not np.isfinite(volume) or volume <= 0:
            raise ValueError("Invalid simulation cell")
        # Face-to-face distances, not vector lengths, handle skewed cells.
        heights = volume / np.asarray(
            [
                np.linalg.norm(np.cross(lattice[1], lattice[2])),
                np.linalg.norm(np.cross(lattice[2], lattice[0])),
                np.linalg.norm(np.cross(lattice[0], lattice[1])),
            ]
        )
        repeats = np.maximum(
            1, np.ceil(self.min_cell_length_a / heights - 1e-10)
        ).astype(int)
        if len(structure) * np.prod(repeats) > self.max_atoms:
            raise ValueError("Required MD supercell exceeds max_atoms")
        supercell = structure.copy()
        supercell.make_supercell(repeats.tolist())
        atoms = AseAtomsAdaptor.get_atoms(supercell)
        atoms.wrap()
        return atoms, elements, repeats.tolist()

    def _input_script(self, elements: list[str], seed: int) -> str:
        model_types = " ".join(self.type_names[element] for element in elements)
        return f'''units metal
atom_style atomic
boundary p p p
atom_modify map array
read_data structure.data
pair_style nequip
pair_coeff * * "{self.model_path}" {model_types}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
timestep {self.timestep_ps}
thermo {self.sample_interval}
thermo_style custom step temp pe ke etotal press vol
thermo_modify lost error flush yes
min_style fire
minimize 0.0 0.01 2000 20000
velocity all create {self.temperature_k} {seed} mom yes rot no dist gaussian
fix equil all nvt temp {self.temperature_k} {self.temperature_k} {self.thermostat_damping_ps}
run {self.equilibration_steps}
unfix equil
reset_timestep 0
fix production all nvt temp {self.temperature_k} {self.temperature_k} {self.thermostat_damping_ps}
dump trajectory all custom {self.sample_interval} trajectory.lammpstrj id type xu yu zu
dump_modify trajectory sort id first yes format float %.16g
variable measured_temperature equal temp
fix temperatures all ave/time {self.sample_interval} 1 {self.sample_interval} v_measured_temperature file temperature.dat
run {self.production_steps}
write_data final.data
'''

    def _run_replica(self, atoms, elements: list[str], seed: int, directory: Path):
        directory.mkdir()
        write(
            str(directory / "structure.data"),
            atoms,
            format="lammps-data",
            specorder=elements,
            atom_style="atomic",
            units="metal",
            masses=True,
        )
        (directory / "in.lammps").write_text(
            self._input_script(elements, seed), encoding="utf-8"
        )
        with (directory / "stdout.log").open("w", encoding="utf-8") as output:
            environment = os.environ.copy()
            if self.cuda_visible_devices is not None:
                environment["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
            subprocess.run(
                [self.lammps_command, "-in", "in.lammps", "-log", "log.lammps"],
                cwd=directory,
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_seconds,
                check=True,
            )
        steps, positions, atom_types = read_dump(directory / "trajectory.lammpstrj")
        if (
            steps[0] != 0
            or steps[-1] != self.production_steps
            or not np.all(np.diff(steps) == self.sample_interval)
        ):
            raise ValueError("Trajectory does not cover the complete production run")
        symbols = atoms.get_chemical_symbols()
        expected_types = np.asarray([elements.index(symbol) + 1 for symbol in symbols])
        if not np.array_equal(atom_types, expected_types):
            raise ValueError("Trajectory atom types do not match the input structure")
        temperatures = np.loadtxt(directory / "temperature.dat", ndmin=2)
        # fix ave/time with Nrepeat=1 can also emit the production step-zero
        # sample during setup. Require every subsequent sampling step below.
        if temperatures.shape[1] == 2 and temperatures[0, 0] == 0:
            temperatures = temperatures[1:]
        expected_steps = np.arange(
            self.sample_interval, self.production_steps + 1, self.sample_interval
        )
        if (
            temperatures.shape != (len(expected_steps), 2)
            or not np.isfinite(temperatures).all()
            or not np.array_equal(temperatures[:, 0], expected_steps)
        ):
            raise ValueError("Incomplete or invalid production temperature record")
        mean_temperature = float(temperatures[:, 1].mean())
        mobile_mask = np.asarray([symbol == self.mobile_species for symbol in symbols])
        statistics, msd = diffusion_statistics(
            positions,
            mobile_mask,
            atoms.get_masses(),
            self.sample_interval * self.timestep_ps,
            self.fit_start_fraction,
            self.fit_end_fraction,
        )
        statistics["mean_temperature_k"] = mean_temperature
        np.savetxt(
            directory / "msd.csv",
            msd,
            delimiter=",",
            comments="",
            header="lag_ps,mobile_msd_a2,framework_msd_a2",
        )
        (directory / "analysis.json").write_text(
            json.dumps(statistics, indent=2), encoding="utf-8"
        )
        reasons = []
        if statistics["tracer_diffusivity_cm2_s"] <= 0:
            reasons.append("non-positive diffusion slope")
        if statistics["msd_r_squared"] < self.min_r_squared:
            reasons.append("MSD fit is not sufficiently linear")
        if (
            not self.exponent_bounds[0]
            <= statistics["msd_exponent"]
            <= self.exponent_bounds[1]
        ):
            reasons.append("MSD exponent is outside the diffusive regime")
        if statistics["msd_growth_a2"] < self.min_msd_growth_a2:
            reasons.append("insufficient resolved ion motion")
        if statistics["framework_msd_a2"] > self.max_framework_msd_a2:
            reasons.append("framework motion exceeds the solid-state screening limit")
        if (
            abs(mean_temperature / self.temperature_k - 1)
            > self.max_temperature_deviation
        ):
            reasons.append("production temperature differs from the target")
        if not np.isfinite(list(statistics.values())).all():
            reasons.append("non-finite diffusion diagnostics")
        if reasons:
            raise ValueError("Unresolved MD: " + "; ".join(reasons))
        return statistics

    def calc_many(self, samples: tuple[list[Structure], str], label: str = "tmp"):
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("._") or "batch"
        batch_dir = Path(tempfile.mkdtemp(prefix=f"{safe_label}_", dir=self.root_dir))
        protocol = {
            key: value
            for key, value in vars(self).items()
            if key != "model_path" and not key.startswith("_")
        }
        protocol["model_path"] = str(self.model_path)
        protocol["units"] = "metal (eV, angstrom, ps)"
        protocol["ensemble"] = "fixed-cell NVT after fixed-cell atomic minimization"
        (batch_dir / "protocol.json").write_text(
            json.dumps(protocol, indent=2), encoding="utf-8"
        )
        records = []
        for index, structure in enumerate(samples[0]):
            candidate_dir = batch_dir / f"candidate_{index:04d}"
            candidate_dir.mkdir()
            record = {
                "formula": structure.composition.reduced_formula,
                "temperature_k": self.temperature_k,
                "mobile_species": self.mobile_species,
                "charge_number": self.charge_number,
                "directory": str(candidate_dir),
                **dict.fromkeys(self.available_tasks, np.nan),
                "error": "",
            }
            try:
                atoms, elements, repeats = self._prepare_structure(structure)
                (candidate_dir / "structure.json").write_text(
                    json.dumps(
                        {
                            "supercell_repeats": repeats,
                            "atom_count": len(atoms),
                            "volume_a3": atoms.get_volume(),
                            "element_order": elements,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                replicas = [
                    self._run_replica(
                        atoms, elements, seed, candidate_dir / f"seed_{seed}"
                    )
                    for seed in self.seeds
                ]
                diffusion = np.asarray(
                    [x["tracer_diffusivity_cm2_s"] for x in replicas]
                )
                mean, std = float(diffusion.mean()), float(diffusion.std(ddof=1))
                if std / mean > self.max_replica_cv:
                    raise ValueError(
                        "Unresolved MD: diffusivity varies excessively between replicas"
                    )
                record.update(
                    {
                        "tracer_diffusivity_cm2_s": mean,
                        "log10_tracer_diffusivity_cm2_s": float(np.log10(mean)),
                        "tracer_diffusivity_std_cm2_s": std,
                        "conductivity_ne_s_cm": nernst_einstein_conductivity(
                            mean,
                            atoms.get_chemical_symbols().count(self.mobile_species),
                            atoms.get_volume(),
                            self.temperature_k,
                            self.charge_number,
                        ),
                    }
                )
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                (candidate_dir / "error.txt").write_text(
                    record["error"], encoding="utf-8"
                )
                if self.strict:
                    raise
            records.append(record)
        if records:
            with (batch_dir / "results.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
        return {
            task: np.asarray([record[task] for record in records], dtype=float)
            for task in self.available_tasks
        }

    def calc(
        self, samples: tuple[list[Structure], str], label: str = "tmp"
    ) -> np.ndarray:
        return self.calc_many(samples, label)[self.task]
