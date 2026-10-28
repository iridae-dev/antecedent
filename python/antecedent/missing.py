"""Declared handling of missing values.

Every route that reads rows says what it does with a missing value, and no route deletes,
weights or restricts rows on its own where that would change the design:

* average-effect routes keep their complete-case selection, now reported with row
  accounting (``missing="complete_case"`` names the default);
* randomized, panel (difference-in-differences), temporal and NOTEARS routes refuse any
  missing value until a policy is declared, naming the first missing cell and the policies
  the route can execute.

Average-effect routes also execute multiple imputation with valid pooling::

    analyze(
        data,
        query=AverageEffect(...),
        missing=MissingPolicy.multiple_imputation(imputations=40, seed=7),
    )

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
    "multiple_imputation",
]

_NAMES: tuple[str, ...] = (
    "error",
    "complete_case",
    "inverse_probability",
    "balanced_panel",
    "available_case",
    "pairwise_available",
    "multiple_imputation",
)

_MODELS: tuple[str, ...] = ("chained_equations", "joint_normal")
_ASSUMPTIONS: tuple[str, ...] = ("missing_at_random", "missing_not_at_random")


@dataclass(frozen=True, slots=True)
class ImputationSpec:
    """The declared inputs of a multiple-imputation analysis.

    Every field enters the claim identity.

    * ``model``: ``"chained_equations"`` (each incomplete variable drawn from a
      Bayesian-bootstrap fit of its conditional model: normal linear for a continuous
      variable, logistic for a 0/1 one) or ``"joint_normal"`` (multivariate normal data
      augmentation with a Jeffreys posterior draw; refuses an incomplete binary variable).
    * ``imputations``: number of completed tables ``m`` (2 to 1000; default 20).
    * ``iterations``: sweeps before each completed table is taken (1 to 1000; default 10).
    * ``seed``: seed of the keyed random streams; table ``j`` opens its own stream, so the
      result does not depend on threads.
    * ``assumption``: ``"missing_at_random"``. ``"missing_not_at_random"`` is refused with a
      sensitivity analysis named as the next attempt.
    * ``bootstrap_replicates``: bootstrap replicates run inside every completed table by an
      estimator that publishes no analytic variance (a static mediation decomposition), as that
      table's within-imputation variance (default 0: such an estimator is then refused).
      Estimators with an analytic variance ignore it.
    """

    model: str = "chained_equations"
    imputations: int = 20
    iterations: int = 10
    seed: int = 0
    assumption: str = "missing_at_random"
    bootstrap_replicates: int = 0

    def __post_init__(self) -> None:
        if self.model not in _MODELS:
            raise CausalValueError(
                f"unknown imputation model {self.model!r}; use one of {list(_MODELS)}",
                reason_code="invalid_argument",
            )
        if self.assumption not in _ASSUMPTIONS:
            raise CausalValueError(
                f"unknown missingness assumption {self.assumption!r}; use one of "
                f"{list(_ASSUMPTIONS)}",
                reason_code="invalid_argument",
            )
        for name in ("imputations", "iterations", "seed", "bootstrap_replicates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CausalTypeError(
                    f"{name} must be a non-negative integer", reason_code="invalid_argument"
                )

    def to_native(self) -> dict[str, Any]:
        """The ``imputation`` member of the native ``missing`` option."""
        native: dict[str, Any] = {
            "model": self.model,
            "imputations": self.imputations,
            "iterations": self.iterations,
            "seed": self.seed,
            "assumption": self.assumption,
        }
        if self.bootstrap_replicates:
            native["bootstrap_replicates"] = self.bootstrap_replicates
        return native


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
    * ``"multiple_imputation"``: average-effect, conditional-effect, response, distribution,
      mediation, randomized and single-series temporal routes; every row is kept, the missing
      cells are drawn ``imputation.imputations`` times from a proper imputation model, the
      route's estimator runs on every completed table and the fits are pooled (Rubin's rules
      with Barnard-Rubin degrees of freedom, or one stacked posterior under Bayesian
      inference). A route whose answer has several coordinates (the cells of a conditional
      effect, a response curve, the atoms of a distribution, a mediation decomposition, the
      arms of a randomized design, a series' horizons) pools them jointly: the result's
      ``missing_data.imputation.vector`` holds the within-, between- and total covariance, each
      coordinate's degrees of freedom and fraction of missing information, and the pooled Wald
      test. A randomized design adds the arm each row was assigned to the imputation model; a
      series is imputed by a lagged model whose parameter draws resample contiguous blocks.
      ``covariates`` name the auxiliary variables of the imputation model; without them the
      model reads every float column. ``missing_data.imputation.assumptions_digest`` identifies
      the model and design assumptions the pooled answer rests on.
    """

    policy: MissingPolicyName
    covariates: tuple[str, ...] = ()
    imputation: ImputationSpec | None = None

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
        if covariates and self.policy not in ("inverse_probability", "multiple_imputation"):
            raise CausalValueError(
                "covariates name the observation model of policy 'inverse_probability' or the "
                f"auxiliary variables of 'multiple_imputation'; policy {self.policy!r} reads none",
                reason_code="invalid_argument",
            )
        object.__setattr__(self, "covariates", covariates)
        if self.policy == "multiple_imputation":
            if self.imputation is None:
                object.__setattr__(self, "imputation", ImputationSpec())
            elif not isinstance(self.imputation, ImputationSpec):
                raise CausalTypeError(
                    "imputation must be an antecedent.missing.ImputationSpec",
                    reason_code="invalid_argument",
                )
        elif self.imputation is not None:
            raise CausalValueError(
                f"an imputation specification is read only by 'multiple_imputation', not "
                f"{self.policy!r}",
                reason_code="invalid_argument",
            )

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

    @classmethod
    def multiple_imputation(
        cls,
        *,
        model: str = "chained_equations",
        imputations: int = 20,
        iterations: int = 10,
        seed: int = 0,
        assumption: str = "missing_at_random",
        auxiliary: Iterable[str] = (),
        bootstrap_replicates: int = 0,
    ) -> MissingPolicy:
        """Impute the missing cells ``imputations`` times, fit every completed table, pool."""
        return cls(
            "multiple_imputation",
            tuple(auxiliary),
            ImputationSpec(
                model, imputations, iterations, seed, assumption, bootstrap_replicates
            ),
        )


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
    declaration: dict[str, Any] = {
        "policy": policy.policy,
        "covariates": [position[n] for n in policy.covariates],
    }
    if policy.imputation is not None:
        declaration["imputation"] = policy.imputation.to_native()
    return declaration


__all__ = ["ImputationSpec", "MissingPolicy", "MissingPolicyName", "coerce_missing"]
