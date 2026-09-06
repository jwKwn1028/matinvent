import os
from numbers import Real
from typing import List, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf
from pymatgen.core.structure import Structure


def linear_scaling(values, minv=0.0, maxv=6.0):
    if maxv <= minv:
        raise ValueError("maxv must be greater than minv")
    return np.clip((np.asarray(values, dtype=float) - minv) / (maxv - minv), 0.0, 1.0)


def average_props(prop_dict):
    if not prop_dict:
        raise ValueError("At least one reward property is required")
    return np.asarray(list(prop_dict.values()), dtype=float).mean(axis=0)


def min_props(prop_dict):
    if not prop_dict:
        raise ValueError("At least one reward property is required")
    return np.asarray(list(prop_dict.values()), dtype=float).min(axis=0)


class Reward:
    def __init__(
        self,
        root_dir: str,
        prop_cfg: DictConfig,
        reward_threshold: float,
        reduce: str = "mean",
        calculator=None,
        constraint_cfg=None,
        **kwargs,
    ) -> None:
        if reduce not in ["mean", "min", "weight"]:
            raise ValueError("reduce must be one of: mean, min, weight")
        if not prop_cfg:
            raise ValueError("prop_cfg must contain at least one objective")
        if not 0.0 <= float(reward_threshold) <= 1.0:
            raise ValueError("reward_threshold must be between 0 and 1")
        if reduce == "weight":
            weights = [_cfg.get("weight") for _cfg in prop_cfg]
            if any(weight is None for weight in weights):
                raise ValueError("Every weighted reward property requires a weight")
            weights = [float(weight) for weight in weights]
            if any(weight < 0.0 for weight in weights):
                raise ValueError("Reward weights must be non-negative")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Reward weights must sum to 1.0")
        self.root_dir = root_dir
        self.prop_cfg = prop_cfg
        self.threshold = float(reward_threshold)
        self.cfg = OmegaConf.create(kwargs)
        self.reduce = reduce
        self.calculator = calculator
        self.constraint_cfg = constraint_cfg or []
        objective_names = {_cfg.name for _cfg in prop_cfg}
        for constraint in self.constraint_cfg:
            if constraint.name not in objective_names:
                raise ValueError(
                    f"Constraint property {constraint.name!r} must also be a "
                    "reward objective"
                )
            lower = constraint.get("min")
            upper = constraint.get("max")
            if lower is None and upper is None:
                raise ValueError(
                    f"Constraint {constraint.name!r} requires min and/or max"
                )
            if lower is not None and upper is not None and float(lower) > float(upper):
                raise ValueError(
                    f"Constraint {constraint.name!r} has min greater than max"
                )
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def calc_props(self, samples: Tuple[List[Structure], str], label: str = "tmp"):
        prop_dict, prop_list = {}, []
        batch_results = None
        if self.calculator is not None:
            if not hasattr(self.calculator, "calc_many"):
                raise TypeError("A shared reward calculator must implement calc_many")
            batch_results = self.calculator.calc_many(samples, label)

        sample_count = len(samples[0])
        for _cfg in self.prop_cfg:
            property_calculator = _cfg.get("calculator")
            if property_calculator is not None:
                _prop1 = property_calculator.calc(samples, label)
            elif batch_results is None:
                raise ValueError(
                    f"Reward property {_cfg.name!r} has no calculator and no "
                    "shared calculator is configured"
                )
            else:
                source = _cfg.get("source", _cfg.name)
                if source not in batch_results:
                    available = ", ".join(sorted(batch_results))
                    raise KeyError(
                        f"Reward property {source!r} was not returned by the shared "
                        f"calculator. Available: {available}"
                    )
                _prop1 = batch_results[source]
            _prop1 = np.asarray(_prop1, dtype=float)
            if _prop1.ndim != 1 or len(_prop1) != sample_count:
                raise ValueError(
                    f"Calculator output for {_cfg.name!r} must have shape "
                    f"({sample_count},), got {_prop1.shape}"
                )
            prop_list.append(_prop1)
            _prop2 = np.nan_to_num(_prop1, nan=0.0, posinf=0.0, neginf=0.0)
            prop_dict[_cfg.name] = _prop2.astype(float)

        prop_list = np.array(prop_list)
        none_ids = (~np.isfinite(prop_list)).any(axis=0)

        return prop_dict, none_ids

    def scoring(self, samples: Tuple[List[Structure], str], label: str = "tmp"):

        prop_dict, failed_mask = self.calc_props(samples, label)

        scaled_prop_dict = {}
        for _cfg in self.prop_cfg:
            target = _cfg.target
            if target == "ascending":
                _sprop = linear_scaling(
                    values=prop_dict[_cfg.name],
                    minv=_cfg.minv,
                    maxv=_cfg.maxv,
                )
            elif target == "descending":
                _sprop = linear_scaling(
                    values=-prop_dict[_cfg.name],
                    minv=-_cfg.maxv,
                    maxv=-_cfg.minv,
                )
            elif isinstance(target, Real) and not isinstance(target, bool):
                diff = np.abs(prop_dict[_cfg.name] - float(target))
                _sprop = linear_scaling(
                    values=-diff,
                    minv=-_cfg.maxv,
                    maxv=-_cfg.minv,
                )
            else:
                raise TypeError(
                    "prop cfg.target must be a float or descending or ascending"
                )

            scaled_prop_dict[_cfg.name] = _sprop

        if self.reduce == "mean":
            rewards = average_props(scaled_prop_dict)
        elif self.reduce == "min":
            rewards = min_props(scaled_prop_dict)
        elif self.reduce == "weight":
            for _cfg in self.prop_cfg:
                if "weight" not in _cfg:
                    raise ValueError(f"Reward property {_cfg.name!r} requires a weight")
                w = float(_cfg.weight)
                scaled_prop_dict[_cfg.name] = scaled_prop_dict[_cfg.name] * w
            sprop_list = list(scaled_prop_dict.values())
            sprop_arr = np.array(sprop_list)
            rewards = sprop_arr.sum(axis=0)

        for constraint in self.constraint_cfg:
            name = constraint.name
            if name not in prop_dict:
                raise KeyError(
                    f"Constraint property {name!r} must also be a reward objective"
                )
            violates = np.zeros(len(rewards), dtype=bool)
            if constraint.get("min") is not None:
                violates |= prop_dict[name] < float(constraint.get("min"))
            if constraint.get("max") is not None:
                violates |= prop_dict[name] > float(constraint.get("max"))
            rewards[violates] = float(constraint.get("penalty", 0.0))

        rewards[failed_mask] = 0.0
        return rewards, prop_dict, failed_mask
