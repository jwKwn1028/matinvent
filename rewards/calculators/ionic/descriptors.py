"""Structure-only descriptors for fast-ion conductor screening.

The functions in this module are intentionally lightweight.  They are useful for
ranking large batches during reinforcement learning, but they do not replace
defect calculations, migration-barrier calculations, or molecular dynamics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, pi
from typing import Iterable

import numpy as np
from pymatgen.core import Element, Structure


BOLTZMANN_EV_PER_K = 8.617333262e-5
DEFAULT_MOBILE_SPECIES = ("Ca",)
DEFAULT_HOP_CUTOFF = 4.5
# Ca3(PS4)2 is charge balanced for Ca2+/P5+/S2-. This is a configurable
# composition prior for exploratory Ca-P-S screening, not a transport optimum.
DEFAULT_CARRIER_FRACTION_TARGET = 3.0 / 13.0

# A deliberately coarse, monotonic proxy for anion polarizability.  Values are
# normalized to [0, 1] and are used only as one weak contribution to the ranking
# score.  They must not be interpreted as measured polarizabilities.
ANION_SOFTNESS = {
    "F": 0.10,
    "O": 0.20,
    "N": 0.25,
    "Cl": 0.50,
    "S": 0.65,
    "Br": 0.75,
    "Se": 0.82,
    "I": 0.95,
    "Te": 1.00,
}


@dataclass(frozen=True)
class IonicDescriptors:
    """Numerical features and physics-inspired transport proxies for one crystal."""

    mobile_fraction: float
    mobile_number_density: float
    free_volume_fraction: float
    mean_mobile_framework_clearance: float
    mobile_sublattice_dimensionality: float
    mobile_sublattice_connectivity: float
    bottleneck_score: float
    anion_softness: float
    short_contact_score: float
    transport_score: float
    activation_energy_proxy_ev: float
    log10_conductivity_proxy: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


DESCRIPTOR_NAMES = tuple(IonicDescriptors.__dataclass_fields__)
SURROGATE_FEATURE_NAMES = (
    "mobile_fraction",
    "mobile_number_density",
    "free_volume_fraction",
    "mean_mobile_framework_clearance",
    "mobile_sublattice_dimensionality",
    "mobile_sublattice_connectivity",
    "bottleneck_score",
    "anion_softness",
    "short_contact_score",
)


def normalize_mobile_species(species: str | Iterable[str]) -> tuple[str, ...]:
    """Return validated, de-duplicated element symbols."""

    if isinstance(species, str):
        values = species.split(",")
    else:
        values = species

    normalized: list[str] = []
    for value in values:
        symbol = str(value).strip()
        if not symbol:
            continue
        # Construction validates the symbol and canonicalizes capitalization.
        symbol = Element(symbol).symbol
        if symbol not in normalized:
            normalized.append(symbol)

    if not normalized:
        raise ValueError("At least one mobile-ion species is required")
    return tuple(normalized)


def resolve_charge_number(
    mobile_species: str | Iterable[str],
    charge_number: float | None = None,
) -> float:
    """Use Ca2+ by default; retain explicit monovalent-ion compatibility.

    Charge is used in conductivity bookkeeping, never to rescale diffusion or
    replace a potential trained for the selected chemical species.
    """
    symbols = normalize_mobile_species(mobile_species)
    known = {"Ca": 2.0, "Li": 1.0, "Na": 1.0}
    charges = {known[symbol] for symbol in symbols if symbol in known}
    if charge_number is None:
        if len(charges) != 1 or any(symbol not in known for symbol in symbols):
            raise ValueError(
                "Specify a single charge_number for the selected mobile species"
            )
        charge_number = charges.pop()
    charge_number = float(charge_number)
    if not np.isfinite(charge_number) or charge_number <= 0:
        raise ValueError("charge_number must be finite and positive")
    if any(
        symbol in known and not np.isclose(charge_number, known[symbol])
        for symbol in symbols
    ):
        raise ValueError(
            f"charge_number={charge_number} conflicts with the ionic charge of {symbols}"
        )
    return charge_number


def _symbol(specie: object) -> str:
    element = getattr(specie, "element", specie)
    symbol = getattr(element, "symbol", None)
    if symbol is None:
        raise ValueError(f"Cannot resolve an element from species {specie!r}")
    return str(symbol)


def _element_radius(symbol: str) -> float:
    """Return a robust radius in angstrom for geometric normalization."""

    element = Element(symbol)
    for value in (element.average_ionic_radius, element.atomic_radius):
        if value is not None and float(value) > 0:
            return float(value)
    # A neutral fallback keeps uncommon elements screenable while the
    # short-contact feature remains conservative.
    return 1.0


def _site_fraction(site: object, selected: set[str]) -> float:
    return float(
        sum(
            occupancy
            for specie, occupancy in site.species.items()
            if _symbol(specie) in selected
        )
    )


def _site_radius(site: object) -> float:
    total = float(sum(site.species.values()))
    if total <= 0:
        return 1.0
    return float(
        sum(
            float(occupancy) * _element_radius(_symbol(specie))
            for specie, occupancy in site.species.items()
        )
        / total
    )


def _packing_fraction(structure: Structure) -> float:
    occupied_volume = 0.0
    for site in structure:
        for specie, occupancy in site.species.items():
            radius = _element_radius(_symbol(specie))
            occupied_volume += float(occupancy) * 4.0 * pi * radius**3 / 3.0
    return float(np.clip(occupied_volume / structure.volume, 0.0, 1.0))


def _periodic_mobile_network(
    structure: Structure,
    mobile_mask: np.ndarray,
    hop_cutoff: float,
) -> tuple[int, float]:
    """Measure dimensionality of a periodically connected mobile-ion graph.

    Sites are graph nodes and candidate hops shorter than ``hop_cutoff`` are
    edges.  A cycle carrying a non-zero lattice translation demonstrates a
    percolating direction.  The rank of independent translation vectors is the
    network dimensionality (0D--3D).
    """

    mobile_indices = np.flatnonzero(mobile_mask)
    if mobile_indices.size == 0:
        return 0, 0.0

    centers, neighbors, offsets, distances = structure.get_neighbor_list(
        r=hop_cutoff, exclude_self=True
    )
    adjacency: dict[int, list[tuple[int, np.ndarray]]] = {
        int(index): [] for index in mobile_indices
    }
    for center, neighbor, offset, distance in zip(
        centers, neighbors, offsets, distances
    ):
        center = int(center)
        neighbor = int(neighbor)
        if distance <= 1e-8 or not mobile_mask[center] or not mobile_mask[neighbor]:
            continue
        adjacency[center].append(
            (neighbor, np.rint(np.asarray(offset, dtype=float)).astype(int))
        )

    visited: set[int] = set()
    best_dimension = 0
    best_connectivity = 0.0
    total_mobile = len(mobile_indices)

    for start in mobile_indices:
        start = int(start)
        if start in visited:
            continue

        translations = {start: np.zeros(3, dtype=int)}
        component: set[int] = {start}
        winding_vectors: list[np.ndarray] = []
        stack = [start]
        visited.add(start)

        while stack:
            center = stack.pop()
            for neighbor, offset in adjacency[center]:
                expected = translations[center] + offset
                if neighbor not in translations:
                    translations[neighbor] = expected
                    component.add(neighbor)
                    visited.add(neighbor)
                    stack.append(neighbor)
                else:
                    winding = expected - translations[neighbor]
                    if np.any(winding):
                        winding_vectors.append(winding)

        dimension = (
            int(np.linalg.matrix_rank(np.asarray(winding_vectors, dtype=float)))
            if winding_vectors
            else 0
        )
        dimension = min(dimension, 3)
        connectivity = dimension / 3.0 * len(component) / total_mobile
        if (dimension, connectivity) > (best_dimension, best_connectivity):
            best_dimension = dimension
            best_connectivity = connectivity

    return best_dimension, float(np.clip(best_connectivity, 0.0, 1.0))


def _geometry_features(
    structure: Structure,
    mobile_mask: np.ndarray,
    neighbor_cutoff: float = 8.0,
) -> tuple[float, float]:
    """Return mean nearest framework clearance and a short-contact score."""

    radii = np.asarray([_site_radius(site) for site in structure], dtype=float)
    centers, neighbors, _, distances = structure.get_neighbor_list(
        r=neighbor_cutoff, exclude_self=True
    )

    nearest_clearance: dict[int, float] = {}
    contact_ratios: list[float] = []
    for center, neighbor, distance in zip(centers, neighbors, distances):
        center = int(center)
        neighbor = int(neighbor)
        distance = float(distance)
        if distance <= 1e-8:
            continue

        radius_sum = radii[center] + radii[neighbor]
        if radius_sum > 0:
            contact_ratios.append(distance / radius_sum)

        if mobile_mask[center] and not mobile_mask[neighbor]:
            clearance = distance - radius_sum
            nearest_clearance[center] = min(
                nearest_clearance.get(center, float("inf")), clearance
            )

    clearance = (
        float(np.mean(list(nearest_clearance.values())))
        if nearest_clearance
        else float("nan")
    )
    if contact_ratios:
        minimum_ratio = min(contact_ratios)
        # Full credit above 80% of the tabulated-radius sum and a smooth hard
        # penalty for obviously overlapping generated structures.
        short_contact_score = float(np.clip((minimum_ratio - 0.55) / 0.25, 0.0, 1.0))
    else:
        short_contact_score = 0.0
    return clearance, short_contact_score


def _anion_softness(structure: Structure, mobile_species: set[str]) -> float:
    weighted_sum = 0.0
    weight = 0.0
    for element, amount in structure.composition.get_el_amt_dict().items():
        if element in mobile_species or element not in ANION_SOFTNESS:
            continue
        weighted_sum += float(amount) * ANION_SOFTNESS[element]
        weight += float(amount)
    return weighted_sum / weight if weight else 0.0


def featurize_structure(
    structure: Structure,
    mobile_species: str | Iterable[str] = DEFAULT_MOBILE_SPECIES,
    temperature_k: float = 298.15,
    hop_cutoff: float = DEFAULT_HOP_CUTOFF,
    charge_number: float | None = None,
    carrier_fraction_target: float = DEFAULT_CARRIER_FRACTION_TARGET,
) -> IonicDescriptors:
    """Compute fast, interpretable descriptors for a periodic structure.

    ``transport_score`` is a bounded screening heuristic.  The activation-energy
    and conductivity values derived from it are explicitly named ``*_proxy``;
    use a fitted surrogate or atomistic simulation for quantitative claims.
    """

    mobile_symbols = normalize_mobile_species(mobile_species)
    charge_number = resolve_charge_number(mobile_symbols, charge_number)
    mobile_set = set(mobile_symbols)
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive")
    if hop_cutoff <= 0:
        raise ValueError("hop_cutoff must be positive")
    if not np.isfinite(carrier_fraction_target) or not 0 < carrier_fraction_target < 1:
        raise ValueError("carrier_fraction_target must be between 0 and 1")
    if (
        len(structure) == 0
        or not np.isfinite(structure.volume)
        or structure.volume <= 0
    ):
        raise ValueError(
            "structure must contain sites in a finite, positive-volume cell"
        )

    mobile_site_fraction = np.asarray(
        [_site_fraction(site, mobile_set) for site in structure], dtype=float
    )
    mobile_mask = mobile_site_fraction > 0.0
    composition = structure.composition.get_el_amt_dict()
    mobile_count = sum(float(composition.get(symbol, 0.0)) for symbol in mobile_symbols)
    total_count = float(structure.composition.num_atoms)
    mobile_fraction = mobile_count / total_count if total_count else 0.0
    mobile_number_density = mobile_count / float(structure.volume)

    packing_fraction = _packing_fraction(structure)
    free_volume_fraction = float(np.clip(1.0 - packing_fraction, 0.0, 1.0))
    dimensionality, connectivity = _periodic_mobile_network(
        structure, mobile_mask, hop_cutoff
    )
    clearance, short_contact_score = _geometry_features(structure, mobile_mask)
    softness = _anion_softness(structure, mobile_set)

    if np.isfinite(clearance):
        clearance_score = exp(-0.5 * ((clearance - 0.05) / 0.45) ** 2)
    else:
        clearance_score = 0.0
    free_volume_score = exp(-0.5 * ((free_volume_fraction - 0.45) / 0.22) ** 2)
    carrier_fraction_score = exp(
        -0.5 * ((mobile_fraction - carrier_fraction_target) / 0.16) ** 2
    )
    density_score = float(np.clip(mobile_number_density / 0.04, 0.0, 1.0))
    carrier_score = (carrier_fraction_score * density_score) ** 0.5
    bottleneck_score = (clearance_score * connectivity) ** 0.5

    framework_count = total_count - mobile_count
    if mobile_count <= 0 or framework_count <= 0:
        transport_score = 0.0
    else:
        transport_core = (
            0.35 * connectivity
            + 0.30 * carrier_score
            + 0.20 * free_volume_score
            + 0.15 * softness
        )
        transport_score = short_contact_score * (
            0.55 * transport_core + 0.45 * bottleneck_score
        )
        # A fully occupied mobile sublattice can look geometrically connected
        # while having too few vacancies for facile transport.  Retain a small
        # floor for aliovalent-defect chemistry, but strongly down-rank extreme
        # carrier fractions in the structure-only proxy.
        transport_score *= 0.15 + 0.85 * carrier_fraction_score
        transport_score = float(np.clip(transport_score, 0.0, 1.0))

    # Arrhenius-inspired mapping used only to make relative score differences
    # easier to interpret.  The prefactor and barrier mapping are not fitted.
    activation_energy_proxy = 0.85 - 0.65 * transport_score
    # Include z^2 in this explicitly UNCALIBRATED conductivity-shaped proxy.
    # Neither its prefactor nor its activation-energy mapping predicts Ca transport.
    log10_conductivity_proxy = (
        2.0
        + 2.0 * np.log10(charge_number)
        - activation_energy_proxy / (BOLTZMANN_EV_PER_K * temperature_k * np.log(10.0))
    )

    return IonicDescriptors(
        mobile_fraction=float(mobile_fraction),
        mobile_number_density=float(mobile_number_density),
        free_volume_fraction=free_volume_fraction,
        mean_mobile_framework_clearance=float(clearance),
        mobile_sublattice_dimensionality=float(dimensionality),
        mobile_sublattice_connectivity=connectivity,
        bottleneck_score=float(bottleneck_score),
        anion_softness=float(softness),
        short_contact_score=short_contact_score,
        transport_score=transport_score,
        activation_energy_proxy_ev=float(activation_energy_proxy),
        log10_conductivity_proxy=float(log10_conductivity_proxy),
    )
