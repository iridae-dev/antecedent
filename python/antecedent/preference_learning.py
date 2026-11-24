"""Optional: learn the trade-off ranges of a value from recorded choices.

Every name here is the object of the same name in :mod:`antecedent.decision_learning`: the
recorded choices (:class:`Records`, :class:`Record`, :class:`Option` and the belief sources),
the two learners (:class:`CompatibleSet`, :class:`ChoiceModel`), their results, and the
reviewed :class:`AcceptedPreferences` with :func:`load`. Accepted preferences plug into
``preferences.Contract(preferences=accepted, ...)`` or ``decision.Contract(value=accepted, ...)``.
"""

from __future__ import annotations

from .decision_learning import (
    AcceptedPreferences,
    AnalysisAsOf,
    BeliefSource,
    ChoiceEstimate,
    ChoiceModel,
    ChoiceModelResult,
    CompatibleSet,
    CompatibleSetResult,
    Conflict,
    Forecast,
    LearnedSet,
    Option,
    ParameterEstimate,
    ParameterRange,
    PerDecider,
    Pooled,
    Realized,
    Record,
    Records,
    Region,
    RegionCell,
    ReviewItem,
    ShapeSearch,
    Stated,
    Undetermined,
    load,
)
from .decision_learning import (
    _mean_wire as _mean_wire,
)
from .decision_learning import (
    _raise as _raise,
)

__all__ = [
    "AcceptedPreferences",
    "AnalysisAsOf",
    "BeliefSource",
    "ChoiceEstimate",
    "ChoiceModel",
    "ChoiceModelResult",
    "CompatibleSet",
    "CompatibleSetResult",
    "Conflict",
    "Forecast",
    "LearnedSet",
    "Option",
    "ParameterEstimate",
    "ParameterRange",
    "PerDecider",
    "Pooled",
    "Realized",
    "Record",
    "Records",
    "Region",
    "RegionCell",
    "ReviewItem",
    "ShapeSearch",
    "Stated",
    "Undetermined",
    "load",
]
