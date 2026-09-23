"""Reward calculators exposed through lazy imports.

Keeping optional scientific stacks lazy lets lightweight tools such as the ionic
screening CLI run without importing ALIGNN, FairChem, or remote DFT machinery.
Hydra can continue to use the original ``rewards.calculators.ClassName`` targets.
"""

from __future__ import annotations

from importlib import import_module


_CALCULATORS = {
    "ALIGNN": ("rewards.calculators.alignn.calc", "ALIGNN"),
    "DFTCalc": ("rewards.calculators.dft.calc", "DFTCalc"),
    "FairChem": ("rewards.calculators.fairchem.calc", "FairChem"),
    "IonicConductivity": (
        "rewards.calculators.ionic.calc",
        "IonicConductivity",
    ),
    "NequIPMD": ("rewards.calculators.nequip_md.calc", "NequIPMD"),
    "PyMatGen": ("rewards.calculators.pymatgen.calc", "PyMatGen"),
    "SynScore": ("rewards.calculators.syn_score.calc", "SynScore"),
}

__all__ = sorted(_CALCULATORS)


def __getattr__(name: str):
    if name not in _CALCULATORS:
        raise AttributeError(name)
    module_name, attribute = _CALCULATORS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
