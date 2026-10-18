"""Declared handling of missing values.

Every route that reads rows says what it does with a missing value, and no route deletes,
weights or restricts rows on its own where that would change the design:

* average-effect routes keep their complete-case selection, now reported with row
  accounting (``missing="complete_case"`` names the default);
* randomized, panel (difference-in-differences), temporal and NOTEARS routes refuse any
  missing value until a policy is declared, naming the first missing cell and the policies
  the route can execute.

Declare a policy with a name or a :class:`MissingPolicy`::

    analyze(data, query=RandomizedEffect(...), missing="complete_case")
    analyze(
        data,
        query=RandomizedEffect(...),
        missing=MissingPolicy.inverse_probability(covariates=("baseline",)),
    )

Each executed policy states the missingness assumption it needs (see the missing-data
guide and :attr:`MissingDataView.assumption`) and every result carries
:class:`RowAccountingView`: rows total, kept and dropped, missing values per variable, the
reason for each exclusion and a digest of exactly which rows were dropped.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from .errors import CausalTypeError, CausalValueError

#: The policy names a route can be asked for.
MissingPolicyName = Literal[
    "error",
    "complete_case",
    "inverse_probability",
    "balanced_panel",
    "available_case",
    "pairwise_available",
]

_NAMES: tuple[str, ...] = (
    "error",
    "complete_case",
    "inverse_probability",
    "balanced_panel",
    "available_case",
    "pairwise_available",
)


@dataclass(frozen=True, slots=True)
class MissingPolicy:
    """How a route handles missing values.

    * ``"error"``: refuse at the first missing cell (the default of randomized, panel,
      temporal and NOTEARS routes; explicit on average-effect routes).
    * ``"complete_case"``: drop rows with a missing value in an analysed variable. The default
      of average-effect routes. On a randomized trial it averages the responders of each arm
      and needs outcomes missing completely at random within arm; on a series it drops the
      lag windows that read a missing time point and leaves time on its original grid.
    * ``"inverse_probability"``: randomized outcomes missing at random given the arm and the
      declared baseline ``covariates``. Each responder is weighted by the inverse of its
      estimated probability of responding within its arm, and every assigned participant
      stays in its arm's denominator.
    * ``"balanced_panel"``: panel designs; drop every unit with a missing period, never a
      single unit-period.
    * ``"available_case"``: panel designs; each contrast uses the units observed in the
      periods it needs. Unit and period identities are preserved.
    """

    policy: MissingPolicyName
    covariates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.policy not in _NAMES:
            raise CausalValueError(
                f"unknown missing-data policy {self.policy!r}; use one of {list(_NAMES)}",
                reason_code="invalid_argument",
            )
        covariates = self.covariates
        if isinstance(covariates, str) or not isinstance(covariates, Iterable):
            raise CausalTypeError(
                "covariates must be a sequence of column names", reason_code="invalid_argument"
            )
        covariates = tuple(covariates)
        if any(not isinstance(name, str) or not name.strip() for name in covariates):
            raise CausalValueError(
                "covariates must be non-empty column names", reason_code="invalid_argument"
            )
        if len(set(covariates)) != len(covariates):
            raise CausalValueError(
                "covariates must be distinct column names", reason_code="invalid_argument"
            )
        if covariates and self.policy != "inverse_probability":
            raise CausalValueError(
                "covariates name the observation model of policy 'inverse_probability'; "
                f"policy {self.policy!r} reads none",
                reason_code="invalid_argument",
            )
        object.__setattr__(self, "covariates", covariates)

    @classmethod
    def error(cls) -> MissingPolicy:
        """Refuse a table with a missing value."""
        return cls("error")

    @classmethod
    def complete_case(cls) -> MissingPolicy:
        """Drop rows with a missing analysed value, and report them."""
        return cls("complete_case")

    @classmethod
    def inverse_probability(cls, covariates: Iterable[str] = ()) -> MissingPolicy:
        """Reweight responders of each arm by their probability of responding."""
        return cls("inverse_probability", tuple(covariates))

    @classmethod
    def balanced_panel(cls) -> MissingPolicy:
        """Drop whole units with a missing period."""
        return cls("balanced_panel")

    @classmethod
    def available_case(cls) -> MissingPolicy:
        """Use each contrast's observed units."""
        return cls("available_case")


def coerce_missing(value: Any) -> MissingPolicy | None:
    """The :class:`MissingPolicy` named by ``value``; ``None`` stays ``None``."""
    if value is None or isinstance(value, MissingPolicy):
        return value
    if isinstance(value, str):
        return MissingPolicy(value)  # type: ignore[arg-type]
    raise CausalTypeError(
        "missing must be a policy name or antecedent.missing.MissingPolicy, got "
        f"{type(value).__name__}",
        reason_code="invalid_argument",
    )


def native_declaration(policy: MissingPolicy | None, names: Iterable[str]) -> dict[str, Any] | None:
    """The ``missing`` prepare option: policy name and covariate column positions."""
    if policy is None:
        return None
    position = {name: index for index, name in enumerate(names)}
    unknown = [name for name in policy.covariates if name not in position]
    if unknown:
        raise CausalValueError(
            f"missing-data covariates {unknown} are not columns of the analysed table; "
            f"available: {list(position)}",
            reason_code="invalid_argument",
        )
    return {"policy": policy.policy, "covariates": [position[n] for n in policy.covariates]}


__all__ = ["MissingPolicy", "MissingPolicyName", "coerce_missing"]
