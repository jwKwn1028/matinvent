"""Tracer diffusion from unwrapped, fixed-cell MD trajectories.

Diffusion is measured relative to the host framework. Mobile-ion center-of-mass
motion must not be removed: doing so would suppress collective ion motion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_dump(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the exact LAMMPS custom dump used here, checking atom/cell identity."""
    steps, frames = [], []
    reference_ids = reference_types = reference_box = None
    with path.open(encoding="utf-8") as handle:
        while line := handle.readline():
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError("Invalid or truncated trajectory: expected TIMESTEP")
            steps.append(int(handle.readline()))
            if handle.readline().strip() != "ITEM: NUMBER OF ATOMS":
                raise ValueError("Missing atom count")
            count = int(handle.readline())
            box_header = handle.readline().strip()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError("Missing box bounds")
            box = np.asarray(
                [[float(x) for x in handle.readline().split()] for _ in range(3)]
            )
            if not np.isfinite(box).all():
                raise ValueError("Non-finite box bounds")
            if handle.readline().strip() != "ITEM: ATOMS id type xu yu zu":
                raise ValueError("Trajectory requires unwrapped xu yu zu coordinates")
            rows = np.asarray(
                [[float(x) for x in handle.readline().split()] for _ in range(count)]
            )
            if rows.shape != (count, 5) or not np.isfinite(rows).all():
                raise ValueError("Invalid or truncated atom coordinates")
            rows = rows[np.argsort(rows[:, 0])]
            ids, types = rows[:, 0], rows[:, 1]
            if not np.array_equal(ids, np.arange(1, count + 1)):
                raise ValueError("Atom IDs must be unique and contiguous")
            if reference_ids is None:
                reference_ids, reference_types, reference_box = ids, types, box
                reference_header = box_header
            elif (
                not np.array_equal(ids, reference_ids)
                or not np.array_equal(types, reference_types)
                or box_header != reference_header
                or box.shape != reference_box.shape
                or not np.allclose(box, reference_box, rtol=0, atol=1e-8)
            ):
                raise ValueError(
                    "Atom identities or simulation cell changed during NVT"
                )
            frames.append(rows[:, 2:])
    if len(frames) < 20:
        raise ValueError("At least 20 trajectory frames are required")
    steps = np.asarray(steps, dtype=int)
    if not np.all(np.diff(steps) == steps[1] - steps[0]) or steps[1] <= steps[0]:
        raise ValueError("Trajectory timesteps must be uniformly increasing")
    if not np.array_equal(reference_types, reference_types.astype(int)):
        raise ValueError("Non-integral atom types")
    return steps, np.asarray(frames), reference_types.astype(int)


def diffusion_statistics(
    positions: np.ndarray,
    mobile_mask: np.ndarray,
    masses: np.ndarray,
    frame_interval_ps: float,
    fit_start_fraction: float = 0.1,
    fit_end_fraction: float = 0.5,
) -> tuple[dict[str, float], np.ndarray]:
    """Fit multi-origin MSD; return diagnostics and (lag, MSD) audit data.

    The result is the orientationally averaged tracer D = Tr(D_tensor)/3,
    including for anisotropic crystals. A^2/ps is converted to cm^2/s by 1e-4.
    Overlapping time origins are not independent uncertainty estimates.
    """
    positions = np.asarray(positions, dtype=float)
    mobile_mask = np.asarray(mobile_mask, dtype=bool)
    masses = np.asarray(masses, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 3 or positions.shape[0] < 20:
        raise ValueError("Expected at least 20 frames of (atoms, 3) positions")
    if mobile_mask.shape != (positions.shape[1],) or masses.shape != mobile_mask.shape:
        raise ValueError("Atom masks and masses must match the trajectory")
    if not mobile_mask.any() or mobile_mask.all():
        raise ValueError("Both mobile ions and framework atoms are required")
    if (
        not np.isfinite(positions).all()
        or not np.isfinite(masses).all()
        or (masses <= 0).any()
    ):
        raise ValueError("Positions and positive masses must be finite")
    if not np.isfinite(frame_interval_ps) or frame_interval_ps <= 0:
        raise ValueError("frame_interval_ps must be positive")
    if not 0 < fit_start_fraction < fit_end_fraction <= 0.5:
        raise ValueError("Fit fractions must satisfy 0 < start < end <= 0.5")

    host_com = np.average(
        positions[:, ~mobile_mask], axis=1, weights=masses[~mobile_mask]
    )
    relative = positions - host_com[:, None, :]
    frame_count = len(positions)
    lags = np.unique(
        np.linspace(
            max(1, int((frame_count - 1) * fit_start_fraction)),
            int((frame_count - 1) * fit_end_fraction),
            64,
            dtype=int,
        )
    )
    if len(lags) < 4:
        raise ValueError("Too few lag points in the diffusion fit window")
    msd, host_msd = [], []
    for lag in lags:
        # Bound memory and cost while retaining many uniformly spaced origins.
        origins = np.unique(np.linspace(0, frame_count - lag - 1, 200, dtype=int))
        displacement = relative[origins + lag] - relative[origins]
        squared = np.sum(displacement**2, axis=2)
        msd.append(float(squared[:, mobile_mask].mean()))
        host_msd.append(float(squared[:, ~mobile_mask].mean()))
    times = lags * frame_interval_ps
    msd = np.asarray(msd)
    slope, intercept = np.polyfit(times, msd, 1)
    residual = np.sum((msd - (slope * times + intercept)) ** 2)
    total = np.sum((msd - msd.mean()) ** 2)
    r_squared = 1.0 - residual / total if total > 0 else 0.0
    exponent = (
        float(np.polyfit(np.log(times), np.log(msd), 1)[0]) if (msd > 0).all() else 0.0
    )
    statistics = {
        "tracer_diffusivity_cm2_s": float(slope / 6.0 * 1e-4),
        "msd_r_squared": float(r_squared),
        "msd_exponent": exponent,
        "msd_growth_a2": float(msd[-1] - msd[0]),
        "framework_msd_a2": float(host_msd[-1]),
        "fit_start_ps": float(times[0]),
        "fit_end_ps": float(times[-1]),
    }
    return statistics, np.column_stack((times, msd, host_msd))


def nernst_einstein_conductivity(
    diffusivity_cm2_s: float,
    mobile_count: int,
    volume_a3: float,
    temperature_k: float,
    charge_number: float,
) -> float:
    """Uncorrelated-ion Nernst–Einstein estimate, in S/cm, at the MD temperature."""
    values = (diffusivity_cm2_s, mobile_count, volume_a3, temperature_k, charge_number)
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Nernst–Einstein inputs must be finite and positive")
    number_density_m3 = mobile_count / (volume_a3 * 1e-30)
    diffusion_m2_s = diffusivity_cm2_s * 1e-4
    sigma_s_m = (
        number_density_m3
        * (charge_number * 1.602176634e-19) ** 2
        * diffusion_m2_s
        / (1.380649e-23 * temperature_k)
    )
    return float(sigma_s_m / 100.0)
