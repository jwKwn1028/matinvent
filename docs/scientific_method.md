# Scientific method and validation

## Scope

The default Ca2+ reward now uses LAMMPS–NequIP MD tracer diffusivity for its
transport term. See [the MD protocol and setup](nequip_md.md). It requires a
validated compiled potential and an explicit simulation temperature. The
descriptor-only reward remains available as `ionic_conductor_proxy`; the
descriptor discussion below also applies to the structural terms and screening
CLI. MD diffusivity is reported at the simulation temperature. Its optional
Nernst–Einstein conductivity estimate assumes uncorrelated ions and does not
establish room-temperature charge conductivity. The default search is exploratory
Ca–P–S. The NE conversion uses charge +2e; the diffusivity itself is not rescaled
by valence. A Ca-containing trained potential is required.

MatInvent-Ion's fast descriptors answer a narrow question: *which members of a
large generated batch have structural motifs worth evaluating next?* They do not
establish room-temperature conductivity, thermodynamic stability, electronic
insulation, electrochemical stability, or interface compatibility.

## Built-in descriptor

For a selected mobile-ion set (Ca by default, charge +2), the calculator measures:

- `mobile_fraction`: mobile-ion stoichiometric fraction;
- `mobile_number_density`: mobile ions per cubic angstrom;
- `free_volume_fraction`: one minus a clipped sum of tabulated-radius sphere
  volumes divided by cell volume;
- `mean_mobile_framework_clearance`: nearest framework distance minus tabulated
  mobile/framework radii, averaged over mobile sites;
- `mobile_sublattice_dimensionality`: rank (0--3) of independent lattice-winding
  vectors in a periodic mobile-site graph under `hop_cutoff`;
- `mobile_sublattice_connectivity`: dimensionality times the fraction of mobile
  sites in the corresponding periodic component;
- `anion_softness`: a weak, hand-set [0, 1] ordering of common anions;
- `short_contact_score`: penalty for severe radius-normalized atom overlap; and
- `bottleneck_score`: geometric combination of clearance and periodic
  connectivity.

The bounded ranking proxy is

```text
core = 0.35 connectivity + 0.30 carrier + 0.20 free-volume + 0.15 softness
transport_score = contact * (0.55 core + 0.45 bottleneck) * carrier-fraction window
```

The carrier and free-volume terms use broad smooth windows. The carrier-fraction
window is centered at a configurable 3/13 for the Ca–P–S presets, the Ca fraction
of charge-balanced Ca3(PS4)2. This is a stoichiometric prior, not a validated
conductivity optimum. The initial Ca-site hop cutoff is 4.5 Å and must be checked
for sensitivity. The convenience
`activation_energy_proxy_ev = 0.85 - 0.65 * transport_score`, and the
`log10_conductivity_proxy` applies an Arrhenius-shaped mapping with an unfitted
prefactor and a z² charge factor. Neither mapping is fitted to Ca data. Their
purpose is relative interpretation only; the MD reward does not use them.

### Known blind spots

- The periodic graph connects occupied mobile sites, not vacant/interstitial
  transition states. Disorder and defect concentration can dominate transport.
- Radius-sphere free volume is not a Voronoi, bond-valence, or potential-energy
  landscape.
- Static structures omit lattice dynamics, correlated hops, grain boundaries,
  processing history, and interfaces.
- The anion-softness ordering is not a computed electrochemical stability metric.
- Generated structures can change ranking after relaxation or symmetry breaking.

For those reasons, optimize multiple gated objectives and keep chemical diversity;
do not optimize `log10_conductivity_proxy` alone.

## Calibrated surrogate

`scripts/fit_ionic_surrogate.py` computes the exact same descriptors, standardizes
them, and fits ridge regression. The resulting JSON is deliberately simple enough
to audit without a pickle or framework-specific checkpoint. It stores the training
range for every feature. At inference:

- `surrogate_applicability = exp(-distance_beyond_training_range)`; and
- `surrogate_uncertainty` inflates cross-validated RMSE outside that range.

The default shuffled K-fold RMSE is a diagnostic, not a publication-grade estimate.
Surrogates must record their mobile species and ionic charge; Ca2+ inference
rejects monovalent-ion or missing-context models by default. The fitting CLI
rejects structures with none of the selected mobile ion. Use Ca2+ transport
labels rather than reusing a Li conductivity dataset.
For scientific evaluation, split by composition system or structure prototype,
reserve a never-touched test set, report data provenance, and quantify repeated
measurements or calculation fidelity. Avoid mixing bulk and total conductivity or
temperatures without an explicit normalization model.

## Candidate-promotion funnel

1. **Generation and sanity checks**
   - condition MatterGen on a mobile-ion-containing chemical system and a
     physically relevant energy-above-hull range;
   - composition/charge plausibility and minimum-distance checks;
   - deduplication by composition and structure matching;
   - novelty against the reference database.
2. **Relaxation and thermodynamic screening**
   - relax with a domain-validated ML potential or DFT;
   - recompute symmetry and descriptors;
   - calculate formation energy and energy above hull with compatible references.
3. **Electronic and electrochemical checks**
   - require electronic insulation at the selected theory level;
   - calculate reduction/oxidation stability and reaction energies against both
     electrodes; consider protective interphases separately.
4. **Transport barriers**
   - identify defects/carrier concentrations explicitly;
   - use bond-valence/site-energy methods as a prescreen;
   - calculate representative migration paths with NEB.
5. **Finite-temperature transport**
   - run converged multi-temperature AIMD or validated ML-potential MD;
   - inspect diffusion dimensionality and correlated motion;
   - fit Arrhenius behavior with uncertainty and finite-size/time checks.
6. **Practical down-selection**
   - assess metastability, synthesizability, abundance/cost, air/moisture response,
     mechanical properties, grain boundaries, and electrode interfaces;
   - export a provenance-rich shortlist for experimental synthesis and EIS.

Do not promote a candidate solely because it ranks above known materials on the
fast proxy. Promotion requires agreement across independent evidence levels.

## Planned experimentation

Nothing in this repository has yet been validated for Ca²⁺. The items below are
ordered by dependency: results from a later group are uninterpretable until the
earlier ones pass. Record each as a dated entry with its commit, configuration
and raw outputs.

### 1. Potential validation (blocks everything downstream)

- Obtain or train a Ca–P–S NequIP potential and report energy/force parity
  against DFT on a held-out set that spans the generated composition range,
  defected and off-stoichiometric cells, and the MD temperature — not only
  relaxed ground states. Report per-element force RMSE.
- Check extrapolation on RL-generated structures, which are not drawn from the
  training distribution. Track a distance-to-training-manifold or ensemble
  disagreement measure per candidate and report the fraction flagged.
- Benchmark the full MD protocol on reference Ca conductors and on a known poor
  conductor at the same temperature. Confirm that the reward window in
  `configs/reward/ionic_conductor.yaml` (`minv: -8.0`, `maxv: -4.0`) actually
  brackets them; it is currently an assumed screening scale.

### 2. MD protocol convergence

- Extend production from 100 ps to 500 ps and 1 ns on a fixed finalist set and
  report how `log10_tracer_diffusivity_cm2_s` and replica scatter move. Ca²⁺ is
  expected to be slow; the current window may not resolve it, and unresolved
  candidates become NaN rather than low scores.
- Raise replicas from 3 seeds to 5–8 and report the standard error of the mean,
  not only the sample standard deviation currently recorded.
- Sweep `min_cell_length_a` at 20, 25 and 30 Å to measure finite-size dependence
  of D at fixed temperature.
- Sweep thermostat damping around the 0.1 ps default, and compare NVT production
  against NVE continued from the equilibrated state.
- Halve the timestep to 0.5 fs and confirm D is unchanged.
- The cell is inherited from upstream relaxation with no thermal expansion. For
  finalists, run NPT at the MD temperature, then re-measure D in the expanded
  cell and report the difference. This is a known, unquantified bias.

### 3. Temperature dependence

- The reward uses a single temperature, so no room-temperature statement is
  currently supported. Run four to five temperatures per finalist, fit Arrhenius
  with uncertainty, and report the activation energy.
- Test the Arrhenius assumption rather than assuming it: check for curvature,
  and confirm a common mechanism across temperature via diffusion
  dimensionality, hop statistics and radial distribution functions.
- Measure the Haven ratio by comparing tracer D to a charge diffusivity from the
  collective current, to quantify how far `conductivity_ne_s_cm` is from a real
  charge conductivity. Collective-current analysis is not implemented yet.

### 4. Descriptor and prior validation

- Now that both paths exist, evaluate the geometric proxy against MD on a common
  candidate set: report Spearman rank correlation between `transport_score` and
  MD `log10_tracer_diffusivity_cm2_s`, plus recall of the MD top decile at
  several proxy cutoffs. That determines whether `ionic_conductor_proxy` is
  usable as a prefilter and at what cost in missed candidates.
- Sweep `hop_cutoff` from 3.5 to 6.0 Å and report both the change in
  `mobile_sublattice_dimensionality`/`connectivity` and the correlation with MD
  D at each value. The 4.5 Å default is a geometric guess.
- Ablate the composition prior by setting the `mobile_fraction` term weight to
  zero, and compare the discovered composition distribution against the run that
  centers on 3/13. The prior may exclude better Ca ratios.
- Replace the hand-set `anion_softness` ordering with a computed quantity and
  compare rankings.

### 5. Reward and RL behavior

- Audit for reward hacking. Fast motion can indicate melting, amorphization or
  potential extrapolation rather than solid-state conduction; check phase
  retention for every finalist via post-run symmetry, framework MSD and RDF.
- Track the NaN rate per RL iteration. Because unresolved candidates score zero,
  the policy can be pushed toward whatever the protocol happens to resolve
  rather than toward better conductors.
- Run a matched-budget ablation of the MD reward against the proxy reward, then
  judge both candidate sets with independent evidence (NEB migration barriers or
  longer MD) that neither reward optimized.
- Track composition and structure-prototype diversity across iterations to
  confirm the 55%-weighted transport term is not collapsing the search.

### 6. Calibrated surrogate

- No Ca²⁺ conductivity dataset has been assembled. Build one with recorded
  provenance, measurement type and temperature, then evaluate with
  composition-system splits and a reserved test set before enabling
  `ionic_conductor_calibrated`.

## Reproducibility checklist

- Pin the generator checkpoint and repository revision.
- Save the fully resolved Hydra `hparams.yaml`.
- Preserve all per-loop descriptor tables and long-term memory.
- Record the surrogate JSON and its source dataset checksum.
- Report random seeds, mobile species, temperature, hop cutoff, and filtering
  thresholds.
- Retain failed calculations and failure reasons rather than silently dropping them.
- Re-rank relaxed structures; do not report only pre-relaxation scores.
