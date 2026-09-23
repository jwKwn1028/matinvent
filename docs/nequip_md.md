# LAMMPS–NequIP transport reward

The default `reward=ionic_conductor` now replaces the 55% geometric transport
term with **log10 Ca tracer diffusivity in cm²/s measured by NequIP MD**, at an
explicitly supplied temperature. The remaining network-dimensionality (20%),
short-contact (15%), and Ca-fraction (10%) terms supply structural screening.
The default chemical system is exploratory **Ca–P–S**, and the mobile ion is
**Ca²⁺**. All four `ionic_conductor*` presets now use Ca²⁺. The optional sodium
preset remains a separate Na⁺ workflow. The geometric proxy, balanced and
calibrated presets retain their respective evaluation methods.

The Ca-fraction prior is **3/13 ≈ 0.231**, from charge-balanced Ca₃(PS₄)₂, with
a broad 0.20 tolerance. This is an explicit exploration prior, not an established
optimal Ca concentration or a restriction to one formula. The descriptor's
carrier-fraction window uses the same configurable center. Its Ca–Ca graph uses
a **4.5 Å** cutoff as an initial geometric setting. Neither parameter is a
measured Ca migration property; inspect sensitivity across the search space.

## Prerequisites

- A **trained and compiled NequIP potential**, validated for the elements,
  structures, defects and temperatures being searched. NequIP is an architecture,
  not a universal Ca–P–S potential; no trained potential is bundled here.
  A Li–P–S checkpoint cannot be converted by renaming its Li type to Ca: obtain
  or train a potential on Ca-containing reference energies and forces.
- A LAMMPS executable with `pair_nequip`, compatible with that model's NequIP
  version and compiler. This adapter uses the **single MPI rank** `pair_nequip`
  interface, not Allegro or ML-IAP. It invokes a local executable directly.
- The existing ASE and pymatgen dependencies. The compiled potential/LAMMPS
  installation can live separately from the generator's Python environment.
- Model energies in **eV** and lengths in **Å**, matching LAMMPS `units metal`.

Consult the official [pair-style installation and compatibility instructions](https://github.com/mir-group/pair_nequip_allegro)
and [NequIP compilation instructions](https://nequip.readthedocs.io/en/latest/integrations/lammps/pair_styles.html).
For a compatible modern NequIP installation, a TorchScript compilation example is:

```bash
nequip-compile /absolute/path/to/trained.ckpt \
  /absolute/path/to/ca_p_s.nequip.pth --device cuda --mode torchscript
```

Model type names are explicitly mapped in
`configs/reward/ionic_conductor.yaml`: `Ca: Ca`, `P: P`, `S: S`. Change the values
if training used different names. Unsupported candidate elements are rejected.
Changing the generator's chemical system also requires an appropriate potential
and matching type map; editing the map alone does not extend the potential.

## Run

Supply absolute paths, since Hydra changes the working directory:

```bash
export NEQUIP_MODEL_PATH=/absolute/path/to/ca_p_s.nequip.pth
export LAMMPS_COMMAND=/absolute/path/to/lmp
# Example only: choose within the potential's validated solid-state range.
export IONIC_MD_TEMPERATURE_K=800
python -u main.py reward=ionic_conductor logger=csv
```

Temperature and potential are required; neither is silently selected. Missing
files/executables fail during calculator setup. `bash scripts/run_rl.sh` uses
the same reward and environment settings.

### Remote cluster execution

Run the whole RL process inside an allocated compute-node job. After loading the
cluster's compatible LAMMPS/NequIP libraries and activating the MatInvent Python
environment, set the variables above and run:

```bash
bash scripts/run_nequip_md_rl.sh eval_size=4
```

This entry point stays in the foreground so the scheduler tracks the job and its
exit status. Use your cluster's Slurm/PBS/resource directives around it. The
adapter launches one LAMMPS process at a time on the allocated node; it does not
SSH, submit nested jobs, or use multiple MPI ranks. LAMMPS and NequIP need not be
installed on the local workstation.

The generator remains loaded while the LAMMPS subprocess runs, so account for
both models' GPU memory. If the allocation provides separate GPUs, set
`NEQUIP_MD_CUDA_VISIBLE_DEVICES` to the scheduler-authorized device identifier(s)
for MD; the generator's environment is unchanged. Otherwise both processes inherit
the allocation's device visibility. Match the compiled model to its execution
device and the cluster's installed runtime.

`IONIC_MD_WORK_DIR` can point to an absolute directory on job scratch for large
trajectory files. Copy that directory to persistent storage before the job's
scratch is removed. The generator checkpoint, reference database and compiled
NequIP potential should be staged before running on nodes without internet access.
The potential must be supplied separately; installing NequIP does not train one.

## Protocol and interpretation

For each already filtered candidate, the evaluator:

1. Builds an ordered periodic supercell with at least 20 Å between opposing cell
   faces; rejects cells requiring more than 4096 atoms.
2. Minimizes atomic positions with NequIP at fixed cell, then equilibrates for
   20 ps using NVT. The cell is inherited from the upstream relaxation: there
   is no NequIP cell optimization or thermal expansion in this protocol.
3. Runs 100 ps NVT production with a 1 fs timestep, retaining unwrapped
   coordinates every 0.1 ps. Repeats with three velocity seeds (17, 29, 43).
4. Subtracts framework center-of-mass drift, then averages squared Ca
   displacements over multiple time origins. It fits lag times from 10% to 50%
   of the trajectory duration. Ca center-of-mass motion is **not** subtracted.
5. Calculates `D = slope(MSD)/6`, the orientationally averaged tracer diffusivity.
   The Å²/ps → cm²/s conversion is `1e-4`. It averages D across replicas and
   records their sample standard deviation.

See [LAMMPS diffusion guidance](https://docs.lammps.org/Howto_diffusion.html) and
[unwrapped dump coordinates](https://docs.lammps.org/dump.html).

The initial reward scaling maps `log10(D)` from -8 to -4 onto [0, 1]. These are
configurable screening bounds, not a validated target. Calibrate them and the
simulation settings on known materials at the same temperature before RL.
At the defaults there are up to 3 × 120,000 MD steps per candidate (plus
minimization); budget MD cost separately from the generator's batch size.

Each replica must show positive diffusivity, MSD fit R² ≥ 0.9, log–log MSD
exponent between 0.75 and 1.25, and at least 1 Å² MSD growth over the fit window.
Framework MSD at the longest lag must remain ≤ 2 Å², mean temperature must be
within 20% of its target, and replica standard deviation/mean diffusivity must
be ≤ 0.5. These checks are configurable screening diagnostics, not proof of
convergence or crystalline phase retention. Inspect trajectories, extend runs,
and check cell-size and thermostat dependence for finalists. Fast motion alone
can reward melting, decomposition or extrapolation by the potential.

Insufficient motion, anomalous dynamics, inconsistent replicas, timeout, or
LAMMPS failure yields **NaN plus a recorded reason**, with no heuristic fallback.
The existing reward engine assigns zero to failed evaluations and excludes them
from successful training samples. This labels transport as unresolved, not as a
measurement of zero conductivity. Short runs can therefore bias selection toward
fast diffusers; extend the protocol to resolve slower candidates.

`conductivity_ne_s_cm` is an additional diagnostic from
`sigma_NE = N (2e)² D / (V k_B T)`, using Ca²⁺ and uncorrelated ions. At equal
number density, D and T, this is four times the monovalent estimate. The factor
does not multiply D or change NequIP forces. The potential must describe Ca
chemistry; `atom_style atomic` does not add explicit Coulomb interactions.
It is **not the optimized default property**, does not include ion–ion cross
correlations, and is not extrapolated to room temperature. Quantitative charge
conductivity requires collective charge-displacement/current analysis. A 298 K
prediction requires adequately sampled temperature-dependent data, confirmation
of a common phase/mechanism, and a justified extrapolation model.

### Optional Ca conductivity surrogate

`ionic_conductor_calibrated` requires a newly fitted model with **Ca²⁺ conductivity
labels** at a consistent temperature. The fitting and screening CLIs default to
Ca²⁺ and store/check ion, charge, temperature, hop cutoff and composition prior.
Li-trained models and models lacking this context are rejected by default.
Editing metadata is not calibration; supply appropriate Ca training data.

```bash
python scripts/fit_ionic_surrogate.py data/ca_conductivity.csv \
  --mobile-species Ca --charge-number 2 --output models/ca_conductivity.json
```

The default surrogate temperature is 298.15 K, separate from elevated-temperature
MD. For labels at another temperature, set `--temperature-k` and the corresponding
reward calculator's `temperature_k` consistently. The band-gap, formation-energy
and synthesizability models in the balanced preset remain generic pretrained
estimators; switching the mobile species does not validate them on Ca–P–S.

## Audit outputs and verification

Every evaluation creates a unique batch directory under
`rewards/ionic_conductor/nequip_md`, preserving repeated evaluations. It contains:

- `protocol.json`: model SHA-256, executable path, temperature and all settings;
- `results.csv`: per-candidate diffusion, replica scatter, NE estimate and errors;
- candidate supercell metadata and per-seed LAMMPS input/data/logs;
- `trajectory.lammpstrj`, `temperature.dat`, `msd.csv`, and `analysis.json`;
- `error.txt` for failed or unresolved candidates.

The unit tests use synthetic Brownian trajectories and a simulated process
boundary to check analysis, type ordering, units and failure propagation. They
do not validate a trained potential or prove that a particular LAMMPS binary
can execute it. Before RL, run real smoke tests and benchmark transport against
reference structures/calculations for the supplied checkpoint.
