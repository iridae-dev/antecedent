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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from .errors import CausalTypeError, CausalValueError, StructuredRefusal

if TYPE_CHECKING:
    from .execution import Execution

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
        for name, minimum in (("imputations", 2), ("iterations", 1)):
            if not minimum <= getattr(self, name) <= 1000:
                raise CausalValueError(
                    f"{name} must be between {minimum} and 1000", reason_code="invalid_argument"
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


class GraphSelectionImputationRefusal(StructuredRefusal):
    """A refusal of the imputation with graph selection re-run on every completed table.

    A :class:`~antecedent.errors.StructuredRefusal` carrying the registered ``code`` and the
    namespaced ``detail`` (``graph_selection_imputation.<slot>`` or the imputation's own
    ``multiple_imputation.<slot>``).
    """


@dataclass(frozen=True, slots=True)
class GraphSelectionImputationResult:
    """The pooled answer of :func:`multiple_imputation_with_graph_selection`.

    ``effect`` and ``posterior_sd`` summarise the equal-weight stack of the completed tables'
    effect posteriors, each a mixture over that table's own graph posterior; ``credible_interval``
    is the central posterior interval of the stack at ``level``, a posterior statement
    conditional on the imputation model and the exact-DAG search, not a coverage claim.
    ``points`` are the tables' posterior means. Three things stay apart: ``within`` is the mean
    posterior variance inside a table, ``between`` is the spread of the table means (the
    uncertainty the missing cells and the changing selected structure add), and ``assumption``
    is the declared missingness mechanism, which the data cannot test. ``structure`` names how
    the graph posterior was obtained (``"graph_selection_rerun_per_imputation"``).
    """

    effect: float
    posterior_sd: float
    credible_interval: Mapping[str, float] | None
    posterior_draws: int | None
    imputations: int
    points: tuple[float, ...]
    within: float
    between: float
    total: float
    pooling: str
    estimator: str
    imputed_cells: Mapping[str, int]
    assumptions_digest: str
    assumption: str
    structure: str


def multiple_imputation_with_graph_selection(
    data: Mapping[str, Any],
    *,
    treatment: str,
    outcome: str,
    imputations: int = 20,
    iterations: int = 10,
    model: str = "chained_equations",
    n_draws: int = 1000,
    prior_scale: float = 10.0,
    level: float = 0.95,
    execution: Execution | None = None,
) -> GraphSelectionImputationResult:
    """Multiple imputation of a Bayesian average effect with graph selection re-run per table.

    ``data`` maps column names to float columns (missing cells ``NaN`` or ``None``); every
    column must be numeric. Each of the ``imputations`` completed tables is searched for its
    own exact posterior over directed acyclic graphs of the columns, in column order (at most
    six columns, the enumeration cap; larger tables are searched by order MCMC), the
    Bayesian g-computation effect of ``treatment`` on ``outcome`` is estimated under that
    posterior (the adjustment is whatever each sampled graph identifies), and the tables'
    effect posteriors are stacked with equal weight, so the pooled spread carries the change
    in the selected structure as well as the change in the filled cells. The missingness
    mechanism is missing at random, declared and untestable. The seed of the imputation and
    of the effect draws is ``execution.seed``. Refusals raise
    :class:`GraphSelectionImputationRefusal`.
    """
    import json

    from ._native import graph_selection_imputation_run as _run
    from ._data import _is_missing_scalar
    from ._refusal import raise_refusal
    from .execution import native_controls

    if not isinstance(data, Mapping) or not data:
        raise CausalTypeError("data must be a non-empty mapping of column names to values")
    ImputationSpec(imputations=imputations, iterations=iterations)
    names = [str(name) for name in data]
    columns = [
        [float("nan") if _is_missing_scalar(v) else float(v) for v in data[name]]
        for name in data
    ]
    text, refusal = _run(
        names,
        columns,
        treatment,
        outcome,
        model=model,
        imputations=imputations,
        iterations=iterations,
        n_draws=n_draws,
        prior_scale=float(prior_scale),
        level=float(level),
        **native_controls(execution, seed="seed"),
    )
    raise_refusal(refusal, GraphSelectionImputationRefusal)
    if text is None:  # pragma: no cover - the native contract
        raise CausalValueError("the native routine returned neither a result nor a refusal")
    raw = json.loads(text)
    return GraphSelectionImputationResult(
        effect=raw["effect"],
        posterior_sd=raw["posterior_sd"],
        credible_interval=raw["credible_interval"],
        posterior_draws=raw["posterior_draws"],
        imputations=raw["imputations"],
        points=tuple(raw["points"]),
        within=raw["within"],
        between=raw["between"],
        total=raw["total"],
        pooling=raw["pooling"],
        estimator=raw["estimator"],
        imputed_cells={entry["variable"]: entry["cells"] for entry in raw["imputed_cells"]},
        assumptions_digest=raw["assumptions_digest"],
        assumption=raw["assumption"],
        structure=raw["structure"],
    )


__all__ = [
    "GraphSelectionImputationRefusal",
    "GraphSelectionImputationResult",
    "ImputationSpec",
    "MissingPolicy",
    "MissingPolicyName",
    "coerce_missing",
    "multiple_imputation_with_graph_selection",
]
