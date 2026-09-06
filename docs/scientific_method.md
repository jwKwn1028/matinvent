# Scientific method and validation

## Scope

MatInvent-Ion's fast descriptors answer a narrow question: *which members of a
large generated batch have structural motifs worth evaluating next?* They do not
establish room-temperature conductivity, thermodynamic stability, electronic
insulation, electrochemical stability, or interface compatibility.

## Built-in descriptor

For a selected mobile-ion set (Li by default), the calculator measures:

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

The carrier and free-volume terms use broad smooth windows. The convenience
`activation_energy_proxy_ev = 0.85 - 0.65 * transport_score`, and the
`log10_conductivity_proxy` applies an Arrhenius-shaped mapping with an unfitted
prefactor. Their purpose is relative interpretation only.

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

## Reproducibility checklist

- Pin the generator checkpoint and repository revision.
- Save the fully resolved Hydra `hparams.yaml`.
- Preserve all per-loop descriptor tables and long-term memory.
- Record the surrogate JSON and its source dataset checksum.
- Report random seeds, mobile species, temperature, hop cutoff, and filtering
  thresholds.
- Retain failed calculations and failure reasons rather than silently dropping them.
- Re-rank relaxed structures; do not report only pre-relaxation scores.
