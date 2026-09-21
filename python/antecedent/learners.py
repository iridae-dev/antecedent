"""Typed prediction models shared by causal estimators and transport providers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias


def _positive(name: str, value: float, *, zero: bool = False) -> None:
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def _count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 2**32 - 1:
        raise ValueError(f"{name} must be a positive 32-bit integer")


@dataclass(frozen=True, slots=True)
class Auto:
    """Select a learner inside each training fold."""

    def _wire(self) -> dict[str, object]:
        return {"kind": "auto"}


@dataclass(frozen=True, slots=True)
class Linear:
    """Ordinary least squares."""

    def _wire(self) -> dict[str, object]:
        return {"kind": "linear"}


@dataclass(frozen=True, slots=True)
class Logistic:
    """Binary logistic regression."""

    def _wire(self) -> dict[str, object]:
        return {"kind": "logistic"}


@dataclass(frozen=True, slots=True)
class Ridge:
    """Least squares with a nonnegative ridge penalty."""

    penalty: float = 1.0

    def __post_init__(self) -> None:
        _positive("penalty", self.penalty, zero=True)

    def _wire(self) -> dict[str, object]:
        return {"kind": "ridge", "lambda": self.penalty}


@dataclass(frozen=True, slots=True)
class ElasticNet:
    """Elastic-net regression; l1_ratio is between zero and one."""

    penalty: float = 1.0
    l1_ratio: float = 0.5

    def __post_init__(self) -> None:
        _positive("penalty", self.penalty, zero=True)
        if not math.isfinite(self.l1_ratio) or not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be finite and between zero and one")

    def _wire(self) -> dict[str, object]:
        return {"kind": "elastic_net", "lambda": self.penalty, "l1_ratio": self.l1_ratio}


@dataclass(frozen=True, slots=True)
class GradientBoostedTrees:
    """Gradient boosting through the optional CPU provider."""

    trees: int = 300
    depth: int = 6
    learning_rate: float = 0.05

    def __post_init__(self) -> None:
        _count("trees", self.trees)
        _count("depth", self.depth)
        _positive("learning_rate", self.learning_rate)

    def _wire(self) -> dict[str, object]:
        return {
            "kind": "gradient_boosted_trees",
            "trees": self.trees,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
        }


@dataclass(frozen=True, slots=True)
class RandomForest:
    """Random forest or extra-trees through the optional CPU provider."""

    extra_trees: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.extra_trees, bool):
            raise ValueError("extra_trees must be a boolean")

    def _wire(self) -> dict[str, object]:
        return {"kind": "random_forest", "extra_trees": self.extra_trees}


@dataclass(frozen=True, slots=True)
class NeuralNet:
    """Optional neural provider; excluded from the CPU feature bundle."""

    hidden: int = 32
    epochs: int = 40
    learning_rate: float = 0.05

    def __post_init__(self) -> None:
        _count("hidden", self.hidden)
        _count("epochs", self.epochs)
        _positive("learning_rate", self.learning_rate)

    def _wire(self) -> dict[str, object]:
        return {
            "kind": "neural_net",
            "hidden": self.hidden,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
        }


LearnerSpec: TypeAlias = (
    Auto | Linear | Logistic | Ridge | ElasticNet | GradientBoostedTrees | RandomForest | NeuralNet
)


def _learner_wire(value: LearnerSpec | str) -> dict[str, object] | str:
    return value if isinstance(value, str) else value._wire()


__all__ = [
    "Auto",
    "Linear",
    "Logistic",
    "Ridge",
    "ElasticNet",
    "GradientBoostedTrees",
    "RandomForest",
    "NeuralNet",
    "LearnerSpec",
]
