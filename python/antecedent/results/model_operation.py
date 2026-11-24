"""The answer of a fitted-model operation posed through ``antecedent.analyze``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import CausalTypeError


@dataclass(frozen=True, slots=True)
class ModelOperationResult:
    """A fitted-model operation answered by the compiled ``Attribute`` or ``Simulate`` route.

    ``value`` is the engine's answer, unchanged: a list of ``ArrowStrength`` for
    ``ArrowStrengths``, a ``PopulationDoContrast`` for ``PopulationContrast``, a ``DoSample``
    for ``DoSampling``, a ``ModelEvaluation`` for ``FittedModelEvaluation`` and a float for
    ``InterventionalMean`` and ``SimultaneousCounterfactual``. ``route`` is the compiled route
    the answer ran under: its contract, the claim sealed over it and the inspection record.

    The answer is a statement about the mechanisms fitted on the supplied graph, not an
    identified effect of the population; the route runs with calibration status
    ``not_measured``.
    """

    query: Any
    kind: str
    value: Any
    route: Any
    _prepared: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def _from_native(cls, query: Any, native: Any) -> ModelOperationResult:
        return cls(query=query, kind=str(native.kind), value=native.value, route=native.route)

    def _require(self, *kinds: str) -> Any:
        if self.kind not in kinds:
            raise CausalTypeError(f"a {self.kind} answer carries no {' or '.join(kinds)} value")
        return self.value

    @property
    def arrow_strengths(self) -> list[Any]:
        """Every edge's strength (``ArrowStrengths``)."""
        return list(self._require("arrow_strengths"))

    @property
    def contrast(self) -> Any:
        """The population do-contrast with its paired standard error (``PopulationContrast``)."""
        return self._require("population_do_contrast")

    @property
    def do_sample(self) -> Any:
        """The do-sampler's draws, weights and diagnostics (``DoSampling``)."""
        return self._require("do_sample")

    @property
    def evaluation(self) -> Any:
        """The falsification evaluation report (``FittedModelEvaluation``)."""
        return self._require("model_evaluation")

    @property
    def mean(self) -> float:
        """The outcome mean (``InterventionalMean``, ``SimultaneousCounterfactual``)."""
        return float(self._require("interventional_mean", "simultaneous_counterfactual"))


__all__ = ["ModelOperationResult"]
