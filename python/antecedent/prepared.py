"""Prepared studies and the prepare / estimate / rebind / reconfigure lifecycle.

:func:`prepare` validates a question against data and a causal structure, takes an owned copy
of the observations, runs identification and compiles the estimation program, without fitting
anything. The result is an immutable :class:`PreparedAnalysis`. :func:`estimate` executes it and
returns an :class:`~antecedent.outcomes.Outcome`; :func:`rebind` and :func:`reconfigure` return
new prepared branches on other data or other declarations and leave the original unchanged.
:func:`analyze` is ``estimate(prepare(...))`` in one call.

Scientific refusals (the question is not identified, the route is not supported, positivity
fails) are returned on the outcome as :attr:`Outcome.refusal <antecedent.outcomes.Outcome.refusal>`.
Malformed arguments and operational failures (cancellation, resource limits, serialization)
raise :class:`~antecedent.errors.CausalError` subclasses.

Example::

    import antecedent as ant

    study = ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect(treatment="t", outcome="y"),
        execution=ant.Execution(seed=19),
    )
    study.inspect()                 # identification and contract; nothing is fitted
    outcome = ant.estimate(study)
    if outcome.refusal is None:
        print(outcome.answer.value, outcome.uncertainty)
    else:
        print(outcome.refusal.code, outcome.refusal.remedy)
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, overload

from ._freeze import freeze
from ._prepared_inspection import InspectionPage
from ._study_lifecycle import guard_design, replay_declaration, retain
from .errors import CausalTypeError, CausalValueError
from .execution import Execution
from .external_learners import Learners, OriginScopedLearners
from .outcomes import Outcome, Refusal, as_outcome, capture_outcome, require_success

T = TypeVar("T")


class _AutomaticDataDiagnostics:
    """Private omitted-argument marker retained across reconfiguration."""

    def __repr__(self) -> str:
        return "<automatic>"


_AUTO_DATA_DIAGNOSTICS = _AutomaticDataDiagnostics()
if TYPE_CHECKING:
    from ._batch import BatchResult, PreparedBatch
    from ._prepared_identification import PreparedIdentification
    from .compiled_study import WrittenStudy
    from .cyclic import CyclicResult, PreparedCyclicFamily, PreparedCyclicSystem
    from .exposure_measurement import MeasuredAnalysis, PreparedMeasuredAnalysis
    from .validation import TargetRefutation, TargetRefutationSpec

CyclicStudyT = TypeVar("CyclicStudyT", "PreparedCyclicFamily", "PreparedCyclicSystem")


def _preparation_declarations(**values: Any) -> Mapping[str, Any]:
    """Bind scientific declarations to the canonical signature without preparing work."""
    from inspect import signature

    bound = signature(prepare).bind(None, **values)
    bound.apply_defaults()
    return retain(
        {
            name: value
            for name, value in bound.arguments.items()
            # Measurement and learner declarations have their own preparation routes.
            if name not in {"data", "execution", "measurement", "measurement_draws", "imputations"}
            and not (name == "learners" and value is None)
        }
    )


@dataclass(frozen=True, slots=True)
class Reuse:
    """What a prepared branch kept from the study it was derived from.

    Read it as :attr:`PreparedAnalysis.reuse` after :func:`rebind` or :func:`reconfigure` to see
    which earlier work the new branch still uses and which it must redo.

    * ``retained`` -- labels of the work carried over unchanged, for example ``"declarations"``,
      ``"data"``, ``"identification"`` or ``"program"``. Empty for a study built directly by
      :func:`prepare`.
    * ``invalidated`` -- labels of the work the branch discards and recomputes when it is next
      estimated, for example ``"fits"``, ``"scores"``, ``"statistical_plan"``, ``"physical_plan"``,
      ``"draws"`` or ``"family_inference"`` (the joint inference of a batch the branch left).
    * ``preparation`` -- ``"compiled"`` for a study prepared from scratch or branched without any
      change, ``"reprepared"`` when the branch was prepared again on new data or declarations.
    """

    retained: tuple[str, ...] = ()
    invalidated: tuple[str, ...] = ()
    preparation: str = "compiled"


@dataclass(frozen=True, slots=True, repr=False)
class PreparedAnalysis(Generic[T]):
    """An immutable prepared study: a validated question, its owned data and its identification.

    Returned by :func:`prepare`. Preparation has already checked the declarations, copied the
    observations into library-owned storage, identified the estimand and compiled the program that
    estimates it; no model has been fitted. The object never changes: :meth:`estimate` returns a new
    :class:`~antecedent.outcomes.Outcome`, and :meth:`rebind` / :meth:`reconfigure` return new
    studies.

    Read first:

    * :attr:`refusal` -- ``None`` when preparation succeeded; otherwise the scientific refusal that
      every later :meth:`estimate` returns unchanged.
    * :meth:`inspect` -- the identification, support, uncertainty and assumption slots known before
      execution.
    * :attr:`identification` -- the retained identification proofs.
    * :attr:`reuse` -- what this branch kept from the study it was derived from (:class:`Reuse`).

    Public fields:

    * ``execution`` -- the :class:`~antecedent.execution.Execution` (seed, threads, limits,
      cancellation, observers) every operation on this study runs under.
    * ``reuse`` -- a :class:`Reuse` record.

    Most methods delegate to the module-level function of the same name, so
    ``study.estimate()`` and ``antecedent.estimate(study)`` are the same call.
    """

    _engine: Any = field(repr=False, compare=False)
    execution: Execution = field(repr=False, compare=False)
    _declarations: Mapping[str, Any] = field(repr=False, compare=False)
    _data: Any = field(default=None, repr=False, compare=False)
    reuse: Reuse = field(default_factory=Reuse)
    _refused: Outcome[Any] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_declarations", freeze(dict(self._declarations)))

    def estimate(self) -> Outcome[T]:
        """Execute this study and return its outcome; see :func:`antecedent.estimate`."""
        return estimate(self)

    def save_compiled_study(self) -> WrittenStudy:
        """Save an executed average-effect branch with its full source table.

        Call this on ``outcome.study`` after a successful estimate. Retain the
        expected identity separately for ``antecedent.compiled_study.open_study``.
        """
        save = getattr(self._engine, "save_compiled_study", None)
        if self._refused is not None or not callable(save):
            raise CausalValueError("a completed portable average-effect study is required")
        return cast("WrittenStudy", self.execution._call(save))

    def rebind(self, *, data: Any) -> PreparedAnalysis[T]:
        """Return a new study on other observations; see :func:`rebind`."""
        return rebind(self, data=data)

    def reconfigure(self, **declarations: Any) -> PreparedAnalysis[T]:
        """Return a new study with changed declarations; see :func:`reconfigure`."""
        return reconfigure(self, **declarations)

    def retarget(self, weights: Any, *, depends_on: Sequence[str]) -> Outcome[Any]:
        """Re-estimate for a reweighted population without refitting; see :func:`retarget`."""
        return retarget(self, weights, depends_on=depends_on)

    def reexecute_retarget(
        self, artifact: bytes, *, retained_identity: str | None = None
    ) -> Outcome[Any]:
        """Re-run an exported row-weight retarget; see :func:`reexecute_retarget`."""
        return reexecute_retarget(self, artifact, retained_identity=retained_identity)

    def mechanism_sensitivity(self, **declarations: Any) -> Outcome[Any]:
        """Evaluate exact-transport mechanism sensitivity; see :func:`mechanism_sensitivity`."""
        return mechanism_sensitivity(self, **declarations)

    def plan_view(self) -> Any:
        """Return the compiled program's layers without executing it; see :func:`plan_view`."""
        return plan_view(self)

    def refute(self, *, suite: Any = "placebo") -> Outcome[Any]:
        """Run a refutation suite against this study's completed run; see :func:`refute`."""
        return refute(self, suite=suite)

    def refute_target(
        self,
        *,
        checks: TargetRefutationSpec | None = None,
        execution: Execution | None = None,
    ) -> Outcome[TargetRefutation]:
        """Run target-specific refutation checks on this study.

        Delegates to :func:`antecedent.validation.refute_target`. ``checks`` is a
        :class:`~antecedent.validation.TargetRefutationSpec` (``None`` selects that function's
        default checks) and ``execution`` overrides this study's execution for the checks only
        (``None`` keeps it). Returns an :class:`~antecedent.outcomes.Outcome` holding a
        :class:`~antecedent.validation.TargetRefutation`.
        """
        from .validation import refute_target

        return refute_target(self, checks=checks, execution=execution)

    def preflight(self) -> Any:
        """Return the structural inspection from before identification; see :func:`preflight`."""
        return preflight(self)

    def diagnose(self, *, execution: Execution | None = None) -> Any:
        """Report fit-free design diagnostics of the retained table and adjustment set.

        Delegates to :func:`antecedent.preflight.diagnose`. ``execution`` overrides this study's
        execution for the diagnostic only; ``None`` keeps it. Nothing is fitted.
        """
        from .preflight import diagnose

        return diagnose(self, execution=execution)

    @property
    def data_diagnostics(self) -> Any:
        """The data-quality report retained from preparation.

        The report of the automatic (or declared) data diagnostics that ran when the study was
        prepared, as returned by :func:`antecedent.data_diagnostics.retained_report`. Reading it
        does not rescan the data. Use :meth:`diagnose_data` to run a suite explicitly.
        """
        from .data_diagnostics import retained_report

        return retained_report(self)

    def diagnose_data(self, *, suite: Any = None, execution: Execution | None = None) -> Any:
        """Run a data-diagnostics suite on this study's owned observations.

        Delegates to :func:`antecedent.diagnose_data`. ``suite`` is a
        :class:`~antecedent.data_diagnostics.DataDiagnostics` declaration (``None`` selects the
        default suite) and ``execution`` overrides this study's execution for the scan only.
        """
        from .data_diagnostics import diagnose_data

        return diagnose_data(self, suite=suite, execution=execution)

    def diagnose_fit(self, *, execution: Execution | None = None) -> Outcome[Any]:
        """Fit diagnostic propensity models on the retained design and report their diagnostics.

        Delegates to :func:`antecedent.preflight.diagnose_fit` and returns its
        :class:`~antecedent.outcomes.Outcome`. No effect is estimated and the study is unchanged.
        ``execution`` overrides this study's execution for the diagnostic only.
        """
        from .preflight import diagnose_fit

        return diagnose_fit(self, execution=execution)

    def plan_rank_drop(
        self, priority: Sequence[str] | None = None, *, execution: Execution | None = None
    ) -> Any:
        """Plan which linearly dependent design columns to drop, without fitting.

        Delegates to :func:`antecedent.preflight.plan_rank_drop`, which documents ``priority``;
        ``execution`` overrides this study's execution for the plan only.
        """
        from .preflight import plan_rank_drop

        return plan_rank_drop(self, priority=priority, execution=execution)

    def estimate_cost(self, *, execution: Execution | None = None) -> Any:
        """Read the prepared plan's scientific budgets and work counts, without executing it.

        Delegates to :func:`antecedent.preflight.estimate_cost`. The counts describe the prepared
        plan; they are a planning estimate, not a measured execution. ``execution`` overrides this
        study's execution for the read only.
        """
        from .preflight import estimate_cost

        return estimate_cost(self, execution=execution)

    def staleness(
        self, data: Any, *, policy: Any = None, execution: Execution | None = None
    ) -> Outcome[Any]:
        """Compare new observations with this study's data and report staleness.

        Delegates to :func:`antecedent.staleness` with this study as the reference. ``data`` is the
        newer table, ``policy`` an optional drift policy (``None`` selects the default) and
        ``execution`` overrides this study's execution for the comparison only. Returns an
        :class:`~antecedent.outcomes.Outcome`.
        """
        from .drift import staleness

        return staleness(self, data, policy=policy, execution=execution)

    def staleness_baseline(
        self, *, policy: Any = None, execution: Execution | None = None
    ) -> Outcome[Any]:
        """Record the drift baseline of this study's own data for later staleness checks.

        Delegates to :func:`antecedent.drift.baseline`. ``policy`` and ``execution`` have the
        meaning they have on :meth:`staleness`. Returns an :class:`~antecedent.outcomes.Outcome`.
        """
        from .drift import baseline

        return baseline(self, policy=policy, execution=execution)

    def treatment_score_posterior(
        self,
        *,
        prior_sd: float = 2.0,
        draws: int = 4000,
        clip: float = 0.05,
        summary_rows: Sequence[int] = (),
        execution: Execution | None = None,
    ) -> Outcome[Any]:
        """Fit a posterior for the treatment (propensity) score of this study.

        Delegates to :func:`antecedent.nuisance_scores.treatment_score_posterior`, which documents
        the model and the returned record. The result describes treatment probabilities; it is not a
        posterior of a causal effect.

        * ``prior_sd`` -- standard deviation of the ``N(0, prior_sd**2)`` prior on every logit
          coefficient, intercept included (default ``2.0``; must be positive).
        * ``draws`` -- number of posterior draws (default ``4000``).
        * ``clip`` -- score clipping bound, positive and strictly below ``0.5`` (default ``0.05``).
        * ``summary_rows`` -- original row positions whose scores are summarised (default none).
        * ``execution`` -- overrides this study's execution for this operation only.

        Returns an :class:`~antecedent.outcomes.Outcome`.
        """
        from .nuisance_scores import treatment_score_posterior

        return treatment_score_posterior(
            self,
            prior_sd=prior_sd,
            draws=draws,
            clip=clip,
            summary_rows=summary_rows,
            execution=execution,
        )

    def temporal_treatment_score(
        self,
        histories: Any,
        *,
        penalty: Any = None,
        intercept_sd: float = 1.0,
        slope_sd: float = 1.0,
        draws: int = 4000,
        execution: Execution | None = None,
    ) -> Outcome[Any]:
        """Fit treatment scores on one row per subject of a retained panel of trajectories.

        Delegates to :func:`antecedent.nuisance_scores.temporal_treatment_score`, which documents
        the model and the returned record. The result is not a posterior of a causal effect, and a
        single dependent series is refused.

        * ``histories`` -- sequences of numeric history values, passed to that function.
        * ``penalty`` -- a positive ridge penalty. When given, the fit is a ridge fit with an
          unpenalized intercept and no posterior uncertainty; ``None`` (default) fits a Laplace
          posterior under independent Gaussian coefficient priors.
        * ``intercept_sd`` / ``slope_sd`` -- prior standard deviations of the intercept and of the
          slopes (default ``1.0`` each; must be positive).
        * ``draws`` -- number of posterior draws (default ``4000``).
        * ``execution`` -- overrides this study's execution for this operation only.

        Returns an :class:`~antecedent.outcomes.Outcome`.
        """
        from .nuisance_scores import temporal_treatment_score

        return temporal_treatment_score(
            self,
            histories,
            penalty=penalty,
            intercept_sd=intercept_sd,
            slope_sd=slope_sd,
            draws=draws,
            execution=execution,
        )

    def preview_transform(self, intent: str) -> Mapping[str, Any]:
        """Preview a proposed change without executing it; see :func:`preview_transform`."""
        return preview_transform(self, intent)

    @property
    def participants(self) -> tuple[str, ...] | None:
        """The original participant label of every prepared row, or ``None``.

        ``None`` unless :func:`prepare` was given ``participant=``. The labels are the named
        column's values as text, in the table's original row order. Reading them fits and scans
        nothing.
        """
        return self._engine.participants

    @property
    def validator_names(self) -> tuple[str, ...]:
        """Names of the custom validators retained at preparation.

        Empty when preparation was refused. Reading the names does not invoke the validator
        callbacks.
        """
        if self._refused is not None:
            return ()
        return self._engine.validator_names

    @property
    def identification(self) -> PreparedIdentification:
        """The retained identification proofs, as a navigable view.

        Returns a ``PreparedIdentification``. Composite questions keep one proof per horizon, graph
        sample or completion case; select a child to inspect one proof. Nothing is executed or
        converted until the view's ``to_dict()`` is called.

        A refused preparation returns a view marked unavailable with reason ``preparation_refused``
        that carries the refusal. A provider-backed study returns one marked
        ``provider_identification_not_independently_verified``.

        Raises:
            CausalUnsupportedError: for a policy-program question, which has no identified estimand
                (``reason_code="option_not_applicable"``).
            CausalStateError: when the preparation exposes no identification view.
        """
        from ._prepared_identification import PreparedIdentification
        from ._prepared_provider import PreparedProvider

        if self._refused is not None:
            return PreparedIdentification.unavailable(
                "preparation_refused", refusal=self._refused.refusal, execution=self.execution
            )
        if isinstance(self._engine, PreparedProvider):
            return PreparedIdentification.unavailable(
                "provider_identification_not_independently_verified",
                kind="provider",
                execution=self.execution,
            )
        from ._prepared_policy import PreparedPolicyAnalysis

        if isinstance(self._engine, PreparedPolicyAnalysis):
            from .errors import CausalUnsupportedError

            raise CausalUnsupportedError(
                "a policy-program question has no identified estimand to navigate",
                reason_code="option_not_applicable",
            )
        if getattr(self._engine, "_kind", None) == "declared_learners":
            return PreparedIdentification.unavailable(
                "declared_learners",
                kind="declared_learners",
                execution=self.execution,
            )
        native = getattr(self._engine, "_native", None)
        if native is None or not hasattr(native, "identification_view"):
            from .errors import CausalStateError

            raise CausalStateError(
                "this scientific preparation does not expose a native identification view",
            )
        return PreparedIdentification.from_native(native.identification_view(), self.execution)

    def inspect(self) -> Any:
        """Return what this study knows now, without estimating.

        Returns an ``InspectionReport`` whose ``identification``, ``support``, ``uncertainty`` and
        ``assumptions`` slots, contract identities and ``operations`` describe the study at its
        current stage. ``stage`` is ``"prepared"`` before execution and ``"executed"`` on a study
        retained by a completed outcome (``outcome.study``), where the report is that of the latest
        result. A refused preparation reports its refusal with ``stage="prepared"``.
        """
        if self._refused is not None:
            return self._refused.inspect().model_copy(
                update={
                    "stage": "prepared",
                    "operations": ("inspect", "estimate", "rebind", "reconfigure"),
                }
            )
        from ._prepared_inspection import inspect_prepared
        from ._prepared_provider import PreparedProvider

        if isinstance(self._engine, PreparedProvider):
            return self._engine.inspect()
        from ._prepared_policy import PreparedPolicyAnalysis

        if isinstance(self._engine, PreparedPolicyAnalysis):
            return self._engine.inspect()
        if getattr(self._engine, "_kind", None) == "learned_trial":
            # This native family exposes its own bounded four-slot metadata.
            return self._engine.inspect().model_copy(update={"stage": "prepared"})
        if getattr(self._engine, "_kind", None) == "declared_learners":
            return self._engine.inspect()
        completed = getattr(self._engine, "_last_result", None)
        if completed is not None:
            return completed.inspect().model_copy(update={"stage": "executed"})
        if getattr(self._engine, "_transport", None) is not None:
            # Transport stages retain a plan-sized native record of the checked
            # derivation and four reasoning slots; reading it solves nothing.
            return self._engine.inspect_plan().model_copy(
                update={
                    "stage": "prepared",
                    "operations": ("inspect", "estimate", "rebind", "reconfigure"),
                }
            )
        return inspect_prepared(self)

    def _plan_summary(self) -> Mapping[str, Any]:
        native = getattr(self._engine, "_native", None)
        if self._refused is not None or native is None or not hasattr(native, "plan_summary"):
            return {}
        return cast("Mapping[str, Any]", native.plan_summary())

    @property
    def structure_source(self) -> str | None:
        """Where the causal structure came from: ``"explicit"`` or ``"accepted"``.

        ``"explicit"`` for a graph passed directly, ``"accepted"`` for an
        :class:`~antecedent.AcceptedGraph`; this is the structure axis of the support matrix, fixed
        at preparation. ``None`` when the study has no plan summary (a refused preparation or a
        family without one).
        """
        summary = self._plan_summary()
        return str(summary.get("structure_source", "explicit")) if summary else None

    @property
    def evidence_status(self) -> str | None:
        """Whether the prepared analysis cell is ``"supported"`` or ``"not_supported"``.

        ``None`` when the plan summary carries no evidence status for this question.
        """
        raw = self._plan_summary().get("evidence_status")
        return None if raw is None else str(raw)

    def assumption_obligations(self, *, offset: int = 0, limit: int = 32) -> InspectionPage:
        """Page through the assumption obligations retained at preparation.

        ``offset`` is the index of the first item (default ``0``) and ``limit`` the page size
        (default ``32``). Returns an ``InspectionPage``. Nothing is recomputed.
        """
        from ._prepared_inspection import inspection_page

        return inspection_page(self, "assumption_obligations", offset, limit)

    def uncertainty_components(self, *, offset: int = 0, limit: int = 32) -> InspectionPage:
        """Page through the uncertainty components the prepared program will report.

        ``offset`` is the index of the first item (default ``0``) and ``limit`` the page size
        (default ``32``). Returns an ``InspectionPage``. Nothing is recomputed.
        """
        from ._prepared_inspection import inspection_page

        return inspection_page(self, "uncertainty_components", offset, limit)

    def evidence_obligations(self, *, offset: int = 0, limit: int = 32) -> InspectionPage:
        """Page through the evidence obligations derived from the identification proof.

        ``offset`` is the index of the first item (default ``0``) and ``limit`` the page size
        (default ``32``). Returns an ``InspectionPage``. Nothing is recomputed.
        """
        from ._prepared_inspection import inspection_page

        return inspection_page(self, "evidence_obligations", offset, limit)

    def evidence_actions(self, *, offset: int = 0, limit: int = 32) -> InspectionPage:
        """Page through the evidence-gathering actions the study names, without running them.

        ``offset`` is the index of the first item (default ``0``) and ``limit`` the page size
        (default ``32``). Returns an ``InspectionPage``.
        """
        from ._prepared_inspection import inspection_page

        return inspection_page(self, "evidence_actions", offset, limit)

    @property
    def refusal(self) -> Refusal | None:
        """The scientific refusal preparation ended with, or ``None`` when it succeeded.

        When set, :meth:`estimate` returns an outcome carrying this same
        :class:`~antecedent.outcomes.Refusal` and runs nothing.
        """
        return None if self._refused is None else self._refused.refusal

    def report(self, format: str = "markdown") -> str:
        """Render this study as a readable report.

        ``format`` is ``"markdown"`` (default) or ``"html"`` (one self-contained page). The report
        restates the records the study holds, the prepared contract before execution and the
        executed claim after it, and computes nothing. A refused preparation renders its refusal.

        Raises:
            CausalValueError: when a refused preparation is asked for a format other than
                ``"markdown"`` or ``"html"``.
        """
        if self._refused is not None:
            if format not in ("markdown", "html"):
                raise CausalValueError("report format must be markdown or html")
            return self._refused.inspect()._repr_html_() if format == "html" else str(self.refusal)
        return self._engine.report(format)

    @property
    def query(self) -> Any:
        """The query this study was prepared for, as declared."""
        return self._declarations["query"]

    def __repr__(self) -> str:
        from .data_diagnostics import _review_count

        count = _review_count(self)
        return f"PreparedAnalysis(query={type(self.query).__name__}, stage='prepared', data_reviews={count})"

    def _repr_html_(self) -> str:
        import html

        return f"<pre>{html.escape(repr(self))}</pre>"


def prepare(
    data: Any,
    *,
    query: Any,
    graph: Any = None,
    discovery: Any = None,
    inference: Any = None,
    execution: Execution | None = None,
    identifier: Any = None,
    estimator: Any = None,
    learners: Learners | OriginScopedLearners | None = None,
    estimator_config: Mapping[str, Any] | None = None,
    refute: Any = None,
    class_prior: Any = None,
    max_completions: int | None = None,
    participant: str | None = None,
    missing: Any = None,
    population_registry: Any = None,
    validators: Sequence[Any] | Mapping[str, Any] | None = None,
    regimes: Sequence[int] | None = None,
    provider: Any = None,
    sampling_design: str | None = None,
    unit_key: Any = None,
    accept_discovered: bool = False,
    data_diagnostics: Any = _AUTO_DATA_DIAGNOSTICS,
    retain_joint_horizons: bool = False,
    on_refusal: Literal["refuse", "describe"] = "refuse",
    evidence_candidates: Sequence[Any] | None = None,
    outcome_units: str | None = None,
    quantity_population: str = "target",
    quantity_transform: str = "identity",
    dose_units: str | None = None,
    measurement: Any = None,
    measurement_draws: int = 30,
    imputations: Sequence[Any] | None = None,
) -> PreparedAnalysis[Any]:
    """Validate a causal question against data and a structure, and identify it, without fitting.

    Preparation checks every declaration, copies ``data`` into library-owned immutable storage,
    identifies the estimand under the declared structure, and compiles and quotes the estimation
    program. No estimator runs; call :func:`estimate` for that. :func:`analyze` takes the same
    arguments and does both.

    Every argument either reaches the prepared study or is refused; none is silently dropped.
    Omitted scientific budgets (bootstrap replicates, posterior draws, refutation depth) resolve
    to the engine's defaults.

    Data and question:

    * ``data`` -- the observations: a mapping of column name to array, a pandas ``DataFrame`` or
      Arrow table, a path to a ``.parquet`` / ``.arrow`` / ``.feather`` / ``.arrows`` file or a
      directory of Parquet files, a snapshot from :func:`antecedent.data.table`, or an
      :mod:`antecedent.data` panel, multi-environment or event frame. Transport queries take
      their own evidence objects.
    * ``query`` -- the question, for example :class:`~antecedent.AverageEffect`; one of the query
      classes in :mod:`antecedent.query` or a stage-module query such as
      :class:`antecedent.experiment.RandomizedEffect`. Required.

    Causal structure (give at most one of ``graph`` and ``discovery``; giving both raises
    :class:`~antecedent.errors.CausalValueError`):

    * ``graph`` -- a :class:`~antecedent.Dag`, :class:`~antecedent.Cpdag`, :class:`~antecedent.Pag`,
      :class:`~antecedent.Admg` or :class:`~antecedent.TemporalDag`, a list of ``(cause, effect)``
      edges, or an :class:`~antecedent.AcceptedGraph`. Default ``None``.
    * ``discovery`` -- a graph posterior (the result of a posterior search, or a constructed
      ``GraphPosterior``) to average over, or a finished discovery: a
      :class:`~antecedent.discovery.DiscoveryResult` or the compiled route of a discovery
      study. A finished result is admitted through ``accepted_graph`` / ``accept_discovered``
      and is not searched again. A discovery configuration that has not been run is rejected
      with :class:`~antecedent.errors.CausalTypeError`: run it first. Default ``None``.
    * ``accept_discovered`` -- retained with the declarations (default ``False``). For a
      finished discovery it is the review gate: pending marks are refused until it is true.
      A study with no discovery has nothing to review, so the flag is not a second decision.
    * ``regimes`` -- regime labels for regime-aware discovery (``RPCMCI``); refused for any other
      declaration. Default ``None``.
    * ``class_prior`` -- a :class:`~antecedent.ClassPrior` over the completions of an incomplete
      temporal graph; requires :class:`~antecedent.Bayesian` inference on a temporal CPDAG or PAG.
    * ``max_completions`` -- cap on the completions enumerated for such a graph; refused for a
      structure without class completions. Default ``None`` (the engine's bound).

    Method:

    * ``inference`` -- :class:`~antecedent.Frequentist` or :class:`~antecedent.Bayesian`. Default
      ``None``, which means ``Frequentist()``. Transport queries take a ``TransportInference``.
    * ``identifier`` -- an :class:`~antecedent.Identifier` or its string id. Default ``None``
      leaves the choice of identification strategy to the engine.
    * ``estimator`` -- an :class:`~antecedent.Estimator`, its string id, or a typed configuration
      from :mod:`antecedent.estimators` such as ``Aipw(...)``. Default ``None`` selects the
      engine's default estimator for the identified estimand.
    * ``learners`` -- nuisance learners for the study (:class:`~antecedent.external_learners.Learners`,
      or an origin-scoped series declaration). ``estimator`` is then the only knob: ``aipw``,
      ``dml`` (the partially linear score), ``dr_learner`` or ``causal_forest``. There is no
      ``score`` argument. With learners, omitting ``estimator`` selects ``dml``.
      Omitting ``learners`` fits the engine's own nuisances.
    * ``estimator_config`` -- a mapping of estimator options; an unknown key is refused.
    * ``refute`` -- a :class:`~antecedent.Refute` suite or its name (``"full"``, ``"placebo"``,
      ``"cheap"``, ``"none"``), or ``False`` for none. ``True`` is rejected because it names no
      suite. Default ``None`` leaves the choice to the engine's defaults and the inference
      ``profile``.
    * ``validators`` -- custom validator callbacks, as a sequence or a mapping of name to
      callable. Their names are retained (:attr:`PreparedAnalysis.validator_names`).
    * ``provider`` -- the evidence provider of a transport query; any other query refuses it.

    Rows and populations:

    * ``participant`` -- name of a column of row labels. The column is removed from the analysis
      variables and its values are kept as :attr:`PreparedAnalysis.participants`. Refused on
      transport studies and on frame or design routes.
    * ``missing`` -- how missing values are handled: a policy name or an
      :class:`antecedent.missing.MissingPolicy`. With the default ``None``, average-effect routes
      keep complete cases and report the rows dropped; randomized, panel and temporal routes refuse
      a missing value until a policy is declared.
    * ``sampling_design`` -- how rows were sampled: ``"independent_subject_trajectories"``,
      ``"clustered_panel"`` or ``"dependent_time_series"``. Every resampling or sample-splitting
      step is held to it, and a request the design does not allow is refused. Any other name
      raises :class:`~antecedent.errors.CausalValueError`. Default ``None`` (no declaration).
    * ``unit_key`` -- an :class:`antecedent.UnitKey` naming the column that identifies the unit
      each row observes and the identifier space it belongs to. Read when a decision joins the
      outcomes of separately prepared studies: studies of different tables keyed in the same
      space are judged for shared evidence by the units they share. Applies to a static
      :class:`~antecedent.ResponseCurve` on one table; refused elsewhere. Default ``None``.
    * ``population_registry`` -- definitions of the named and custom target populations a query's
      ``target_population`` refers to (:mod:`antecedent.population`).

    Diagnostics and refusals:

    * ``data_diagnostics`` -- ``True``, ``False`` or a :class:`~antecedent.DataDiagnostics`
      declaration. When omitted, the diagnostics run automatically for native studies; read them
      from :attr:`PreparedAnalysis.data_diagnostics`. Provider studies refuse an explicit value.
    * ``on_refusal`` -- ``"refuse"`` (default) or ``"describe"``. With ``"describe"``, a question
      refused for lack of identification keeps its refusal, and the outcome additionally carries a
      noncausal description of the observed data (``outcome.associational``) and the evidence that
      would resolve the refusal (``outcome.evidence_plan``); see :mod:`antecedent.fallback`. Not
      available for transport and policy-program queries.
    * ``evidence_candidates`` -- candidate evidence sources for that description; requires
      ``on_refusal="describe"``.

    Reporting units (labels on the reported quantity; they do not change the estimate):

    * ``outcome_units`` -- unit label of the outcome. Requires a response result. Default ``None``.
    * ``quantity_population`` -- population label of the reported quantity, default ``"target"``.
    * ``quantity_transform`` -- transform label, default ``"identity"``. A non-default population
      or transform requires ``outcome_units``.
    * ``dose_units`` -- unit label of the dose of a static :class:`~antecedent.ResponseCurve`;
      nonempty, at most 256 bytes, and requires ``outcome_units``.

    Measured exposure:

    * ``measurement`` -- a declaration from :mod:`antecedent.exposure_measurement`
      (:class:`~antecedent.exposure_measurement.MisclassifiedExposure`,
      :class:`~antecedent.exposure_measurement.SoftExposure`,
      :class:`~antecedent.exposure_measurement.DetectedExposure` or
      :class:`~antecedent.exposure_measurement.RepeatCodedExposure`) on the query's treatment.
      The exposure is imputed ``measurement_draws`` times from its declared measurement, the
      outcome and the covariates, one study is prepared per completed table, and
      :func:`estimate` pools the effects by Rubin's rules. Returns a
      :class:`~antecedent.exposure_measurement.PreparedMeasuredAnalysis`; its outcome's value is a
      :class:`~antecedent.exposure_measurement.MeasuredAnalysis`.
    * ``measurement_draws`` -- the number of imputations, at least 2. Default 30.
    * ``imputations`` -- a sequence of completed tables (series for ``DbnPosterior``) beside a
      posterior-search ``discovery=`` (``ExactDagPosterior``, ``OrderMcmc``, ``StructureMcmc``,
      ``CiScreenedPosterior``, ``DbnPosterior``): the engine searches each table, pools the
      posteriors once and the study mixes the effect over the pooled posterior. Without it, a
      posterior search in ``discovery=`` is run once on ``data``.
    * The imputations draw from the ``execution``'s seed.

    Other:

    * ``execution`` -- an :class:`~antecedent.Execution` holding the seed, threads, resource
      limits, cancellation token and observers. Default ``None`` means ``Execution()`` (seed 1).
    * ``retain_joint_horizons`` -- keep the replicates that refit all horizons of a temporal
      :class:`~antecedent.ResponseCurve` or :class:`~antecedent.InterventionResponse` together, so
      quantities built from several horizons keep their dependence. Any other query refuses it.
      Default ``False``.

    Returns a :class:`PreparedAnalysis`. A scientific refusal met during preparation (for example
    ``effect_not_identified``) does not raise: it is retained as :attr:`PreparedAnalysis.refusal`
    and returned by every later :func:`estimate`. Read ``study.inspect()`` next.

    Raises:
        CausalTypeError: when ``execution`` is not an ``Execution``, a unit label is not a string,
            or ``discovery`` is an unexecuted discovery configuration.
        CausalValueError: when ``on_refusal`` is not ``"refuse"`` or ``"describe"``, a unit label
            is empty, ``quantity_population`` or ``quantity_transform`` is set without
            ``outcome_units``, or ``dose_units`` is invalid or used with another query.
        CausalUnsupportedError: with ``reason_code="option_not_applicable"`` when an option does
            not apply to the query or structure, for example ``evidence_candidates`` without
            ``on_refusal="describe"``, ``class_prior`` on a static graph, or ``participant`` on a
            transport study.

    Example::

        study = antecedent.prepare(
            data,
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=antecedent.AverageEffect(treatment="t", outcome="y"),
            inference=antecedent.Frequentist(bootstrap=500),
            execution=antecedent.Execution(seed=7),
        )
        if study.refusal is None:
            outcome = antecedent.estimate(study)
    """
    from .estimation import _PreparedEngine as Engine

    policy = Execution() if execution is None else execution
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    if learners is not None:
        from .errors import CausalUnsupportedError

        if measurement is not None or imputations is not None:
            raise CausalUnsupportedError(
                "a learner study does not take measurement or imputations",
                reason_code="option_not_applicable",
            )
        if policy.on_stage is not None:
            raise CausalUnsupportedError(
                "a learner study does not stream estimate stages",
                reason_code="stage_stream_unavailable",
            )
        declarations = _preparation_declarations(
            **{
                "query": query,
                "graph": graph,
                "discovery": discovery,
                "inference": inference,
                "identifier": identifier,
                "estimator": estimator,
                "learners": learners,
                "estimator_config": estimator_config,
                "refute": refute,
                "class_prior": class_prior,
                "max_completions": max_completions,
                "participant": participant,
                "missing": missing,
                "population_registry": population_registry,
                "validators": validators,
                "regimes": regimes,
                "provider": provider,
                "sampling_design": sampling_design,
                "unit_key": unit_key,
                "accept_discovered": accept_discovered,
                "data_diagnostics": data_diagnostics,
                "retain_joint_horizons": retain_joint_horizons,
                "on_refusal": on_refusal,
                "evidence_candidates": evidence_candidates,
                "outcome_units": outcome_units,
                "quantity_population": quantity_population,
                "quantity_transform": quantity_transform,
                "dose_units": dose_units,
            }
        )
        from .external_learners import prepare_learner_study

        return prepare_learner_study(data, execution=policy, declarations=declarations)
    if measurement is not None:
        from .exposure_measurement import prepare_measured

        if unit_key is not None:
            from .errors import CausalUnsupportedError

            raise CausalUnsupportedError(
                "unit_key labels the rows of one prepared table; a measured exposure prepares "
                "one study per completed table",
                reason_code="option_not_applicable",
            )

        return cast(
            "PreparedAnalysis[Any]",
            prepare_measured(
                data,
                measurement=measurement,
                draws=measurement_draws,
                execution=policy,
                query=query,
                graph=graph,
                discovery=discovery,
                inference=inference,
                identifier=identifier,
                estimator=estimator,
                estimator_config=estimator_config,
                refute=refute,
                class_prior=class_prior,
                max_completions=max_completions,
                participant=participant,
                missing=missing,
                population_registry=population_registry,
                validators=validators,
                regimes=regimes,
                provider=provider,
                sampling_design=sampling_design,
                accept_discovered=accept_discovered,
                data_diagnostics=data_diagnostics,
                retain_joint_horizons=retain_joint_horizons,
                on_refusal=on_refusal,
                evidence_candidates=evidence_candidates,
                outcome_units=outcome_units,
                quantity_population=quantity_population,
                quantity_transform=quantity_transform,
                dose_units=dose_units,
            ),
        )
    from ._table_file import coerce_table_source

    data = coerce_table_source(data)
    if on_refusal not in ("refuse", "describe"):
        raise CausalValueError("on_refusal must be refuse or describe")
    if evidence_candidates is not None and on_refusal != "describe":
        from .errors import CausalUnsupportedError

        raise CausalUnsupportedError(
            "evidence_candidates are consumed only when an unidentified question is described; "
            "declare on_refusal='describe'",
            reason_code="option_not_applicable",
        )
    if outcome_units is None and (
        quantity_population != "target" or quantity_transform != "identity"
    ):
        raise CausalValueError("quantity population and transform require outcome_units")
    if outcome_units is not None:
        for name, value in (
            ("outcome_units", outcome_units),
            ("quantity_population", quantity_population),
            ("quantity_transform", quantity_transform),
        ):
            if not isinstance(value, str):
                raise CausalTypeError(f"{name} must be a string")
            if not value.strip():
                raise CausalValueError(f"{name} must be non-empty")
    if dose_units is not None:
        from .query import ResponseCurve

        if not isinstance(dose_units, str):
            raise CausalTypeError("dose_units must be a string")
        if not dose_units.strip() or len(dose_units.encode()) > 256 or outcome_units is None:
            raise CausalValueError(
                "dose_units must be nonempty, at most 256 bytes and requires outcome_units"
            )
        if not isinstance(query, ResponseCurve):
            raise CausalValueError("dose_units requires a static ResponseCurve query")
    if discovery is not None and graph is None:
        from .discovery import _POOLING_SEARCHES, pool_search_over_imputations

        if isinstance(discovery, _POOLING_SEARCHES):
            # The posterior search runs in the engine (on every completed table when imputations
            # are given, pooled once); the study takes its executed posterior.
            if imputations is not None:
                discovery = pool_search_over_imputations(discovery, imputations, execution=policy)
            else:
                discovery = discovery.run(data, execution=policy)
        elif imputations is not None:
            raise CausalValueError(
                "imputations pool a posterior search: pass discovery=<search> and no graph="
            )
    elif imputations is not None:
        raise CausalValueError(
            "imputations pool a posterior search: pass discovery=<search> and no graph="
        )
    if discovery is not None and callable(getattr(discovery, "run", None)):
        raise CausalTypeError(
            "preparation requires an executed discovery result; run discovery.discover first"
        )
    declarations = _preparation_declarations(
        **{
            "query": query,
            "graph": graph,
            "discovery": discovery,
            "inference": inference,
            "identifier": identifier,
            "estimator": estimator,
            "estimator_config": estimator_config,
            "refute": refute,
            "class_prior": class_prior,
            "max_completions": max_completions,
            "participant": participant,
            "missing": missing,
            "population_registry": population_registry,
            "validators": validators,
            "regimes": regimes,
            "provider": provider,
            "sampling_design": sampling_design,
            "unit_key": unit_key,
            "accept_discovered": accept_discovered,
            "data_diagnostics": data_diagnostics,
            "retain_joint_horizons": retain_joint_horizons,
            "on_refusal": on_refusal,
            "evidence_candidates": evidence_candidates,
            "outcome_units": outcome_units,
            "quantity_population": quantity_population,
            "quantity_transform": quantity_transform,
            "dose_units": dose_units,
        }
    )
    from .model_operations import MODEL_OPERATION_QUERIES

    if isinstance(query, MODEL_OPERATION_QUERIES):
        # A fitted-model operation keeps every column in the caller's order: its mechanisms are
        # fitted on the graph over the whole table, as fit_gcm fits them, so the variable order
        # (and with it every draw) matches the FittedGcm method it replaces.
        from ._prepared_model_operation import PreparedModelOperation

        operation = capture_outcome(
            policy._call,
            lambda: PreparedModelOperation.prepare(
                data,
                execution=policy,
                **{
                    key: value
                    for key, value in declarations.items()
                    if not (key == "data_diagnostics" and value is _AUTO_DATA_DIAGNOSTICS)
                },
            ),
        )
        refused = operation.refusal is not None
        return PreparedAnalysis(
            operation.value,
            policy,
            declarations,
            data if refused else require_success(operation)._data,
            _refused=operation if refused else None,
        )
    from ._table_snapshot import Table, table
    from .data import EventFrame, MultiEnvFrame, PanelFrame
    from .transport import Transport
    from .transport.advanced import (
        ExactTransportQuery,
        StatisticalTransportQuery,
        TransportQuery,
        TransportResponseGridQuery,
        TrialAipwQuery,
    )

    transport = isinstance(
        query,
        (
            TransportQuery,
            Transport,
            ExactTransportQuery,
            StatisticalTransportQuery,
            TransportResponseGridQuery,
            TrialAipwQuery,
        ),
    )
    from .query import ResponseCurve

    row_source = None
    static_response = isinstance(query, ResponseCurve) and not query.is_temporal
    if unit_key is not None and (transport or not static_response):
        from .errors import CausalUnsupportedError

        raise CausalUnsupportedError(
            "unit_key labels the rows of one table whose static ResponseCurve a decision may "
            f"join; {type(query).__name__} does not read it",
            reason_code="option_not_applicable",
        )
    if transport or isinstance(data, (Table, PanelFrame, MultiEnvFrame, EventFrame)):
        owned = data
    else:
        from ._owned_input import project_prepared_input

        projected = policy._call(
            project_prepared_input,
            data,
            query=query,
            graph=graph,
            estimator=estimator,
            estimator_config=estimator_config,
            participant=participant,
            missing=missing,
        )
        from ._row_evidence import RowSource

        # The rows of the projected table keep the identity of the table they were taken from,
        # so studies of other columns of the same rows are judged as sharing those rows.
        row_source = RowSource.capture(
            data,
            unit_key=unit_key,
            provenance=static_response,
        )
        owned = policy._call(table, projected)
    options = replay_declaration(declarations)
    if options["data_diagnostics"] is _AUTO_DATA_DIAGNOSTICS:
        options["data_diagnostics"] = True
    if options["validators"] is not None and not isinstance(options["validators"], Mapping):
        # Request collections are frozen on the public handle. The native
        # callback boundary accepts a concrete list and retains each original
        # callable, so only the small declaration container is materialized.
        options["validators"] = list(options["validators"])
    for name in (
        "on_refusal",
        "evidence_candidates",
        "outcome_units",
        "quantity_population",
        "quantity_transform",
        "dose_units",
    ):
        options.pop(name)
    # This backend flag controls implicit discovery execution, which this
    # lifecycle prohibits. Explicit graphs and retained posterior declarations
    # have no live discovery review step to waive. A finished discovery result
    # still has that gate, so the caller's flag is the one ``accept_discovered`` reads.
    from .estimation import _is_finished_discovery

    if not _is_finished_discovery(discovery) and (
        discovery is None or not callable(getattr(discovery, "run", None))
    ):
        options["accept_discovered"] = True
    # Resolved estimator defaults stay native. Explicit statistical budgets belong
    # to inference and are passed through the one existing native lowering path.
    # The ordinary design cell reads its inference like any analysis (and refuses a
    # bootstrap it does not support); the family transport queries carry theirs in
    # TransportInference, which the transport lowering applies itself.
    bootstrap = (
        None
        if transport and not isinstance(query, TransportQuery)
        else getattr(inference, "bootstrap", None)
    )

    # Every transport query but the ordinary trial-IPW design cell lowers through its own
    # family engine, which streams no stages and runs no worker pool.
    family_transport = transport and not isinstance(query, (TransportQuery, TrialAipwQuery))

    def build() -> Any:
        from .extensibility import ProviderQuery

        if family_transport:
            from ._transport_lifecycle import refuse_inapplicable_controls

            refuse_inapplicable_controls(policy, threads=False)
        if isinstance(query, ProviderQuery):
            from ._prepared_provider import PreparedProvider

            return PreparedProvider.prepare(
                owned,
                query=query,
                execution=policy,
                **{
                    key: value
                    for key, value in declarations.items()
                    if key != "query"
                    and not (key == "data_diagnostics" and value is _AUTO_DATA_DIAGNOSTICS)
                },
            )
        if isinstance(query, TrialAipwQuery):
            from ._prepared_trial import prepare_trial

            return prepare_trial(owned, declarations=declarations, execution=policy)
        return Engine.prepare(
            owned,
            **options,
            row_source=row_source,
            # Rust describes a refusal on the study that was refused (the transport families
            # keep their own refusal records).
            on_refusal="refuse" if transport else on_refusal,
            evidence_candidates=None if transport else evidence_candidates,
            bootstrap=bootstrap,
            latency=getattr(inference, "profile", None),
            # The family transport lowering rejects ordinary loose controls. The
            # active Execution owns its native context; execution forks bind
            # the requested RNG policy before any statistical run. Controls the family
            # cannot honour were refused above; progress is reported around each estimate.
            seed=policy.seed,
            threads=policy.threads,
            cancel=policy.cancel,
            on_progress=None if family_transport else policy.on_progress,
            on_stage=None if family_transport else policy.on_stage,
        )

    prepared = capture_outcome(policy._call, build)
    return PreparedAnalysis(
        prepared.value,
        policy,
        declarations,
        owned,
        _refused=prepared if prepared.refusal is not None else None,
    )


def _fork(study: PreparedAnalysis[Any]) -> Any:
    branch = getattr(study._engine, "_branch", None)
    if callable(branch):
        return branch(study.execution)
    engine = copy.copy(study._engine)
    native = engine._native
    if native is not None:
        engine._native = (
            study.execution._call(native.fork, seed=study.execution.seed)
            if engine._transport is not None
            else native.fork()
        )
    from .estimation import _Controls

    engine._last_result = None
    engine._seed = study.execution.seed
    engine._threads = study.execution.threads
    if engine._transport is not None:
        from ._transport_lifecycle import refuse_inapplicable_controls

        # A family transport engine honours cancellation natively; its progress is reported
        # around the estimate, and the controls it cannot honour are refused, never dropped.
        refuse_inapplicable_controls(study.execution, threads=False)
        engine._controls = _Controls(cancel=study.execution.cancel)
        return engine
    engine._controls = _Controls(
        cancel=study.execution.cancel,
        on_progress=study.execution.on_progress,
        on_stage=study.execution.on_stage,
    )
    return engine


@overload
def estimate(study: PreparedAnalysis[T]) -> Outcome[T]: ...


@overload
def estimate(study: PreparedBatch) -> Outcome[BatchResult]: ...


@overload
def estimate(study: PreparedMeasuredAnalysis) -> Outcome[MeasuredAnalysis]: ...


@overload
def estimate(study: PreparedCyclicFamily | PreparedCyclicSystem) -> Outcome[CyclicResult]: ...


def estimate(study: Any) -> Outcome[Any]:
    """Execute a prepared study and return its outcome.

    Runs the estimator, inference and any declared refutation of ``study`` under the study's
    :class:`~antecedent.Execution`. ``study`` is a :class:`PreparedAnalysis`, a prepared batch
    from :func:`antecedent.estimation.prepare_many`, or a prepared cyclic system. The study itself
    is not modified.

    Returns an :class:`~antecedent.outcomes.Outcome`. Check ``outcome.refusal`` first:

    * ``None`` -- ``outcome.value`` is the typed result. Read ``outcome.answer`` for the estimate
      and ``outcome.uncertainty``, ``outcome.support`` and ``outcome.assumptions`` for what
      qualifies it. ``outcome.study`` is the executed branch, usable with :func:`rebind`,
      :func:`reconfigure`, :func:`refute` and :func:`retarget`.
    * set -- a scientific refusal, with ``code``, ``message`` and ``remedy``. A study whose
      preparation was refused returns that refusal without running anything. If the study was
      prepared with ``on_refusal="describe"`` and the refusal is an identification refusal,
      ``outcome.associational`` and ``outcome.evidence_plan`` are also set.

    Raises:
        CausalTypeError: when ``study`` is not a prepared study.
        CausalError: for operational failures such as cancellation, an exceeded resource limit or
            a failing callback. The raised error carries the study as ``error.study``.
        CausalUnsupportedError: when the study declares ``outcome_units`` and the result is not a
            response result (``reason_code="option_not_applicable"``).
    """
    from ._batch import PreparedBatch, estimate_many
    from .exposure_measurement import PreparedMeasuredAnalysis

    if isinstance(study, PreparedBatch):
        return estimate_many(study)
    if isinstance(study, PreparedMeasuredAnalysis):
        return study.estimate()
    if not isinstance(study, PreparedAnalysis):
        if _is_cyclic_preparation(study):
            return study.execute()
        raise CausalTypeError("estimate requires a PreparedAnalysis")
    if study._refused is not None:
        return _finish_outcome(study, study._refused)
    engine = _fork(study)
    from .errors import CausalError

    estimate_engine = engine.estimate
    if getattr(engine, "_transport", None) is not None:
        observer = study.execution

        def estimate_engine() -> Any:
            observer._report_progress(0.0, "estimate")
            result = engine.estimate()
            observer._report_progress(1.0, "estimate")
            return result

    try:
        outcome = capture_outcome(study.execution._call, estimate_engine)
    except CausalError as error:
        # Keep the same operational exception and traceback, while exposing
        # the public immutable branch rather than its private engine header.
        setattr(error, "study", study)  # noqa: B010 — public execution context on the original error
        raise
    if outcome.refusal is not None and isinstance(outcome.refusal._error, CausalError):
        # A captured refusal re-raises through require_success: its context is the
        # public branch, not the private engine header the inner operation attached.
        setattr(outcome.refusal._error, "study", study)  # noqa: B010
    completed = getattr(outcome.value, "_prepared", None)
    execution = getattr(outcome.value, "_execution", None)
    if completed is None and getattr(engine, "_kind", None) == "learned_trial":
        completed = engine
    elif completed is None and getattr(engine, "_last_result", None) is not None:
        # A family result that does not point back at its study (an exact or
        # statistical transport distribution) still completed this branch; its
        # native handle stays the live one, so the branch reports as executed.
        completed, execution = engine, None
    from .transport._wrap import _ExportAdapter

    if (
        completed is not None
        and execution is not None
        and not isinstance(execution, _ExportAdapter)
    ):
        # Keep the actual materialized native snapshot, including retained
        # score tables and fits, instead of the declaration used to start it.
        # A transport family result carries an export adapter, not a native
        # handle: the engine's own handle stays the live one for rebinding.
        completed = copy.copy(completed)
        completed._native = execution
    retained = study if completed is None else replace(study, _engine=completed)
    return capture_outcome(_finish_outcome, retained, outcome)


def _finish_outcome(study: PreparedAnalysis[T], outcome: Outcome[T]) -> Outcome[T]:
    declarations = study._declarations
    if outcome.refusal is not None:
        if declarations.get("on_refusal") != "describe":
            return replace(outcome, _study=study)
        from . import fallback
        from ._names import table_names
        from .errors import CausalError, StructuredRefusal
        from .estimation import _Controls
        from .outcomes import outcome_from_error

        error = outcome.refusal._error
        if not isinstance(error, CausalError) or not fallback.is_identification_refusal(error):
            return replace(outcome, _study=study)
        stage = getattr(error, "fallback", None)
        if stage is None:
            return replace(outcome, _study=study)
        description = fallback.from_native(
            error,
            stage,
            query=study.query,
            graph=declarations.get("graph"),
            columns=table_names(study._data),
            execution=study.execution,
            controls=_Controls(
                cancel=study.execution.cancel,
                on_progress=study.execution.on_progress,
                on_stage=study.execution.on_stage,
            ),
        )
        if isinstance(description.associational, fallback.FallbackUnavailable):
            unavailable = description.associational
            associational = outcome_from_error(
                StructuredRefusal(
                    {
                        "code": unavailable.code,
                        "detail": unavailable.detail,
                        "message": unavailable.message,
                    }
                )
            )
        else:
            associational = as_outcome(description.associational)
        return replace(
            outcome,
            _study=study,
            associational=associational,
            evidence_plan=description.evidence_plan,
        )
    units = declarations.get("outcome_units")
    if units is not None:
        from .errors import CausalUnsupportedError
        from .results._report import copy_model
        from .results.response import CausalResponseView

        result = outcome.value
        if not isinstance(result, CausalResponseView):
            raise CausalUnsupportedError(
                "outcome_units requires a native response result",
                reason_code="option_not_applicable",
            )
        population, transform = (
            declarations["quantity_population"],
            declarations["quantity_transform"],
        )
        quantities = result.response_coordinates(
            outcome_units=units,
            population=population,
            transform=transform,
        )
        import json

        record = json.loads(
            study.execution._call(
                study._engine._native.execution_record_json,
                outcome_units=units,
                population=population,
                transform=transform,
            )
        )
        # Labels change the canonical claim, but never trigger artifact encoding.
        from .results._slots import ReasoningSlots

        reasoning = ReasoningSlots.from_result_section(record["contract"], record["evidence"])
        result = copy_model(
            result, quantities=quantities, reasoning=reasoning, claim_id=reasoning.claim_id
        )
        if declarations.get("dose_units") is not None:
            from .program_claims import ProgramBinding

            result = copy_model(
                result,
                program_binding=ProgramBinding.from_response(
                    result,
                    outcome_units=units,
                    dose_units=declarations["dose_units"],
                    population=population,
                    transform=transform,
                ),
            )
        outcome = replace(outcome, value=result)
    from .results._report import ResultModel

    if isinstance(outcome.value, ResultModel):
        result = outcome.value.model_copy()
        result._live_study = study
        outcome = replace(outcome, value=result)
    return replace(outcome, _study=study)


def plan_view(study: PreparedAnalysis[Any]) -> Any:
    """Return every layer of the compiled program of a prepared study, without executing it.

    The view states the scientific problem, the identification product (status, assumptions,
    estimand, adjustment set, unresolved obligations), the statistical program, the physical plan
    with its quoted work, and the uncertainty and claim that execution will publish. Reading it
    identifies, fits and scans nothing.

    Raises:
        CausalTypeError: when ``study`` is not a :class:`PreparedAnalysis`.
        CausalUnsupportedError: with ``reason_code="plan_view_unavailable"`` when the preparation
            was refused or the study executes no lowered program (transport studies).
    """
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("plan_view requires a PreparedAnalysis")
    if study._refused is not None:
        from .errors import CausalUnsupportedError

        raise CausalUnsupportedError(
            "a refused preparation has no lowered executable program",
            reason_code="plan_view_unavailable",
        )
    return study._engine.plan_view()


def preflight(study: PreparedAnalysis[Any]) -> Any:
    """Return the structural inspection of a prepared study from before identification.

    A cheaper view than :meth:`PreparedAnalysis.inspect`: it reports the structural obligations of
    the declared study and has no ``program_id``, because that identity depends on identification.
    Nothing is fitted or compiled. A refused preparation returns its inspection report, and a
    transport study returns the same report as ``inspect()``.

    Raises:
        CausalTypeError: when ``study`` is not a :class:`PreparedAnalysis`.
    """
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("preflight requires a PreparedAnalysis")
    return study.inspect() if study._refused is not None else study._engine.preflight()


def preview_transform(study: PreparedAnalysis[Any], intent: str) -> Mapping[str, Any]:
    """Preview a proposed change to a prepared study without executing or modifying it.

    ``intent`` names the change: ``display_precision``, ``filter_display``,
    ``compatible_data_replace``, ``retarget``, ``filter_population``, ``new_conditional_query``,
    ``change_graph``, ``change_prior``, ``change_physical_policy`` or
    ``average_unweighted_class``.

    Returns a read-only mapping that names the intent, whether it would be refused (with the same
    refusal code the operation itself gives), the obligations it carries, and the identities of
    the frozen inputs it would reuse under ``input_<domain>`` keys.

    Raises:
        CausalTypeError: when ``study`` is not a :class:`PreparedAnalysis`.
        CausalUnsupportedError: when the preparation was refused
            (``reason_code="option_not_applicable"``).
    """
    from .errors import CausalUnsupportedError

    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("preview_transform requires a PreparedAnalysis")
    if study._refused is not None:
        raise CausalUnsupportedError(
            "a refused preparation has no executable program to transform",
            reason_code="option_not_applicable",
        )
    return freeze(study._engine.preview_transform(intent))


def refute(study: PreparedAnalysis[Any], *, suite: Any = "placebo") -> Outcome[Any]:
    """Run a refutation suite against a study's completed run.

    Use it after :func:`estimate`, on ``outcome.study``, to run a deeper suite than the estimate
    did. ``suite`` is a :class:`~antecedent.Refute` member or its name; the default ``"placebo"``
    runs the placebo checks, ``"full"`` the full suite and ``"cheap"`` the inexpensive checks. The
    suite runs on its own snapshot of the completed run under the study's execution, so the
    original outcome is unchanged.

    Returns an :class:`~antecedent.outcomes.Outcome` whose value is the result with the suite's
    validation verdict. A refused preparation returns its refusal.

    Raises:
        CausalTypeError: when ``study`` is not a :class:`PreparedAnalysis`.
        CausalUnsupportedError: when the study retains no run to validate, or the query has no
            refuter suite (response curves, transport, interference and attribution queries).
    """
    from .errors import CausalUnsupportedError

    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("refute requires a PreparedAnalysis")
    if study._refused is not None:
        return _finish_outcome(study, study._refused)
    native = getattr(study._engine, "_native", None)
    if native is None or not callable(getattr(native, "snapshot", None)):
        raise CausalUnsupportedError(
            "this preparation retains no completed run for a validation suite",
            reason_code="option_not_applicable",
        )
    engine = copy.copy(study._engine)
    engine._native = study.execution._call(native.snapshot)
    engine._last_result = None
    outcome = capture_outcome(
        study.execution._call,
        engine.refute,
        suite=suite,
        seed=study.execution.seed,
        threads=study.execution.threads,
        cancel=study.execution.cancel,
    )
    execution = getattr(outcome.value, "_execution", None)
    if execution is not None:
        engine._native = execution
    return _finish_outcome(replace(study, _engine=engine), outcome)


def retarget(
    study: PreparedAnalysis[Any], weights: Any, *, depends_on: Sequence[str]
) -> Outcome[Any]:
    """Re-estimate a completed study for a reweighted population, without refitting.

    ``weights`` gives one weight per prepared row and defines the target population as a
    reweighting of the observed rows. ``depends_on`` names the variables the weights were
    computed from. The engine binds the weights to the study's data snapshot and checks that none
    of those variables is a descendant of the treatment; nonconstant weights require a nonempty
    ``depends_on``, and a nonempty ``depends_on`` requires a directed graph (DAG or ADMG).

    The estimate is recomputed from the study's retained score table, or, for a retained OLS or
    unpenalized GLM mean model, by standardization with that model's own coefficient covariance,
    conditional on the declared target law. The latter is not doubly robust. No model is refitted.

    Returns an :class:`~antecedent.outcomes.Outcome`. A study with no retained scores or compatible
    fit refuses the request (``score_table_unavailable``); a refused preparation returns its
    refusal.

    Raises:
        CausalTypeError: when ``study`` is not a :class:`PreparedAnalysis`.

    Example::

        outcome = antecedent.estimate(study)
        older = antecedent.prepared.retarget(outcome.study, weights, depends_on=["age"])
    """
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("retarget requires a PreparedAnalysis")
    if study._refused is not None:
        return _finish_outcome(study, study._refused)
    engine = _fork(study)
    outcome = capture_outcome(
        study.execution._call,
        engine.retarget,
        weights,
        depends_on=depends_on,
        seed=study.execution.seed,
        threads=study.execution.threads,
    )
    return _finish_outcome(replace(study, _engine=engine), outcome)


def reexecute_retarget(
    study: PreparedAnalysis[Any], artifact: bytes, *, retained_identity: str | None = None
) -> Outcome[Any]:
    """Re-run the row-weight retarget carried by an exported artifact, without refitting.

    ``artifact`` is the exported bytes of a retargeted result. Its weights are bound to one data
    snapshot and to a score table or adjusted-fit identity. Returns an
    :class:`~antecedent.outcomes.Outcome`; a study holding a different snapshot is refused with
    ``row_weights_bound_to_snapshot`` and no rows are reweighted. A refused preparation returns
    its refusal.

    Raises:
        CausalTypeError: when ``study`` is not a :class:`PreparedAnalysis`.
    """
    from . import _retained

    _retained.admit(
        artifact,
        retained_identity,
        "antecedent.prepared.reexecute_retarget",
        "antecedent.prepared.reexecute_retarget(..., retained_identity=antecedent.artifacts.artifact_identity(retained_bytes))",
    )
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("reexecute_retarget requires a PreparedAnalysis")
    if study._refused is not None:
        return _finish_outcome(study, study._refused)
    engine = _fork(study)
    outcome = capture_outcome(
        study.execution._call,
        engine.reexecute_retarget,
        artifact,
        seed=study.execution.seed,
        threads=study.execution.threads,
        retained_identity=_retained.ADMITTED if retained_identity is None else retained_identity,
    )
    return _finish_outcome(replace(study, _engine=engine), outcome)


def mechanism_sensitivity(study: PreparedAnalysis[Any], **declarations: Any) -> Outcome[Any]:
    """Evaluate fixed-graph mechanism sensitivity of a prepared exact transport.

    Applies only to a study prepared from an exact transport query. The keyword ``declarations``
    state the outcome values, parent cardinalities, the two treatment levels, the source kernel
    and parent laws with their regimes and snapshots, the largest perturbation fraction
    (``max_fraction``) and, optionally, a decision threshold and the mechanism to perturb. The
    study's cancellation token applies.

    Returns an :class:`~antecedent.outcomes.Outcome` holding the sensitivity record as a mapping.
    A refused preparation returns its refusal.

    Raises:
        CausalTypeError: when ``study`` is not a :class:`PreparedAnalysis`.
        CausalUnsupportedError: when the study is not a prepared exact transport
            (``reason_code="option_not_applicable"``).
    """
    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("mechanism_sensitivity requires a PreparedAnalysis")
    if study._refused is not None:
        return _finish_outcome(study, study._refused)
    return capture_outcome(
        study.execution._call,
        study._engine.mechanism_sensitivity,
        cancel=study.execution.cancel,
        **declarations,
    )


@overload
def rebind(study: PreparedAnalysis[T], *, data: Any) -> PreparedAnalysis[T]: ...


@overload
def rebind(study: PreparedBatch, *, data: Any) -> PreparedBatch: ...


@overload
def rebind(study: CyclicStudyT, *, data: Any) -> CyclicStudyT: ...


def rebind(study: Any, *, data: Any) -> Any:
    """Return a new prepared study on other observations, keeping every declaration.

    ``data`` must be compatible with the prepared schema (it must provide the prepared analysis
    columns). It is copied into library-owned storage like the data given to :func:`prepare`.
    The query, structure, identification and estimator stay as prepared; fits and scores are
    recomputed when the new study is estimated. ``study`` is unchanged, and earlier outcomes keep
    their own data.

    ``study`` may also be a prepared batch or a prepared cyclic system; the same type is returned.
    Read ``reuse`` on the returned :class:`PreparedAnalysis` for what was retained and
    invalidated. Rebinding a study whose preparation was refused prepares it again on ``data``.

    Raises:
        CausalTypeError: when ``study`` is not a prepared study.
        CausalUnsupportedError: for a held-out policy-value study, whose recommendations and
            nuisance predictions are bound to the prepared rows
            (``reason_code="option_not_applicable"``); prepare a new study instead.

    Example::

        this_week = antecedent.rebind(study, data=new_rows)
        outcome = antecedent.estimate(this_week)
    """
    from ._batch import PreparedBatch, rebind_many
    from .exposure_measurement import PreparedMeasuredAnalysis

    if isinstance(study, PreparedBatch):
        return rebind_many(study, data=data)
    if isinstance(study, PreparedMeasuredAnalysis):
        return study.rebind(data=data)
    if not isinstance(study, PreparedAnalysis):
        if _is_cyclic_preparation(study):
            return study.rebind(data)
        raise CausalTypeError("rebind requires a PreparedAnalysis")
    if getattr(study._engine, "_batch_family", False):
        engine = copy.copy(study._engine)
        engine._batch_family = False
        member_branch = rebind(replace(study, _engine=engine), data=data)
        return replace(
            member_branch,
            reuse=replace(
                member_branch.reuse,
                invalidated=(*member_branch.reuse.invalidated, "family_inference"),
            ),
        )
    from ._table_file import coerce_table_source

    data = coerce_table_source(data)
    if study._refused is not None:
        return prepare(data, execution=study.execution, **replay_declaration(study._declarations))
    from ._table_snapshot import Table, table
    from .data import EventFrame, MultiEnvFrame, PanelFrame

    engine = copy.copy(study._engine)
    if getattr(engine, "_transport", None) is None and not isinstance(
        data, (Table, EventFrame, MultiEnvFrame, PanelFrame)
    ):
        from ._owned_input import project_prepared_input

        declaration = study._declarations
        data = study.execution._call(
            table,
            project_prepared_input(
                data,
                query=declaration["query"],
                graph=declaration["graph"],
                estimator=declaration["estimator"],
                estimator_config=declaration["estimator_config"],
                participant=declaration["participant"],
                missing=declaration["missing"],
            ),
        )
    bind = getattr(engine, "_bind", None)
    if callable(bind):
        return replace(
            study,
            _engine=bind(data, study.execution),
            _data=data,
            reuse=Reuse(("declarations",), ("fits", "scores"), "reprepared"),
        )
    guard_design(engine, data)
    from .errors import CausalUnsupportedError
    from .policy import MultiActionPolicyValue, PolicyValue

    if isinstance(getattr(engine, "_query", None), (PolicyValue, MultiActionPolicyValue)):
        raise CausalUnsupportedError(
            "PolicyValue recommendations, nuisance predictions, and subject ownership are bound to the prepared evaluation rows; prepare a new study for new data",
            reason_code="option_not_applicable",
        )
    if engine._transport is not None:
        from .transport._day1 import Transport, catalog_from_evidence

        if isinstance(engine._query, Transport):
            # A table bound to a Transport declaration is lowered through its evidence
            # catalog exactly as at preparation; the new stage belongs to the new branch.
            stage = engine._transport_stage or {}
            catalog, bound = catalog_from_evidence(engine._query, data, graph=stage.get("graph"))
            if bound is not None:
                data = bound
            engine._transport_stage = retain({**stage, "catalog": catalog, "bound": bound})
        native = study.execution._call(
            engine._native.bind_snapshot,
            engine._transport.payload(data),
            cancel=study.execution.cancel,
        )
    else:
        _names, columns, frame = engine._click_payload(data)
        native = study.execution._call(
            engine._native.bind_snapshot,
            columns,
            frame=frame,
            seed=study.execution.seed,
            threads=study.execution.threads,
        )
    engine._native = native
    engine._snapshot_data = data
    engine._last_result = None
    return replace(
        study,
        _engine=engine,
        _data=data,
        reuse=Reuse(("declarations",), ("fits", "scores", "statistical_plan"), "reprepared"),
    )


@overload
def reconfigure(study: PreparedAnalysis[T], **declarations: Any) -> PreparedAnalysis[T]: ...


@overload
def reconfigure(study: PreparedBatch, **declarations: Any) -> PreparedBatch: ...


@overload
def reconfigure(study: CyclicStudyT, **declarations: Any) -> CyclicStudyT: ...


def reconfigure(study: Any, **declarations: Any) -> Any:
    """Return a new prepared study with changed declarations or execution, without executing it.

    ``declarations`` are keyword arguments of :func:`prepare` other than ``data`` (use
    :func:`rebind` to change data), for example ``query=``, ``graph=``, ``estimator=``,
    ``inference=`` or ``execution=``. ``study`` is unchanged.

    What is reused depends on what changes (read ``reuse`` on the result):

    * nothing -- the same study, with data, identification and program retained;
    * only ``execution`` -- data and identification are retained; the physical plan, fits and
      draws are recomputed under the new seed, threads and limits;
    * any scientific declaration -- the study is prepared again on the same owned data, so
      identification and the program are recomputed.

    ``study`` may also be a prepared batch or a prepared cyclic system; the same type is returned.

    Raises:
        CausalTypeError: when ``study`` is not a prepared study, a keyword is not a preparation
            declaration, or ``execution`` is not an :class:`~antecedent.Execution`.
        CausalValueError: when ``data`` is passed, or the study retains no data to prepare again.

    Example::

        reseeded = antecedent.reconfigure(study, execution=antecedent.Execution(seed=2))
        bayesian = antecedent.reconfigure(study, inference=antecedent.Bayesian())
    """
    from ._batch import PreparedBatch, reconfigure_many
    from .exposure_measurement import PreparedMeasuredAnalysis

    if isinstance(study, PreparedBatch):
        return reconfigure_many(study, **declarations)
    if isinstance(study, PreparedMeasuredAnalysis):
        return study.reconfigure(**declarations)
    if not isinstance(study, PreparedAnalysis):
        if _is_cyclic_preparation(study):
            if "data" in declarations:
                raise CausalValueError("data changes use rebind(study, data=...)")
            return study.reconfigure(**declarations)
        raise CausalTypeError("reconfigure requires a PreparedAnalysis")
    if getattr(study._engine, "_batch_family", False):
        engine = copy.copy(study._engine)
        engine._batch_family = False
        member_branch = reconfigure(replace(study, _engine=engine), **declarations)
        return replace(
            member_branch,
            reuse=replace(
                member_branch.reuse,
                invalidated=(*member_branch.reuse.invalidated, "family_inference"),
            ),
        )
    if "data" in declarations:
        raise CausalValueError("data changes use rebind(study, data=...)")
    unknown = declarations.keys() - study._declarations.keys() - {"execution"}
    if unknown:
        raise CausalTypeError(f"unknown preparation declarations: {', '.join(sorted(unknown))}")
    policy = declarations.pop("execution", study.execution)
    if not isinstance(policy, Execution):
        raise CausalTypeError("execution must be an Execution declaration")
    if study._refused is not None and getattr(study._engine, "_kind", None) == "learned_trial":
        from ._prepared_trial import reconfigure_trial

        request = {**replay_declaration(study._declarations), **declarations}
        return reconfigure_trial(study, request, policy)
    if not declarations:
        if study._refused is not None:
            return prepare(study._data, execution=policy, **replay_declaration(study._declarations))
        if policy is study.execution:
            return replace(study, reuse=Reuse(("data", "identification", "program")))
        branch = getattr(study._engine, "_branch", None)
        if callable(branch):
            return replace(study, _engine=branch(policy), execution=policy)
        engine = copy.copy(study._engine)
        if engine._native is not None:
            if engine._transport is not None:
                from ._transport_lifecycle import refuse_inapplicable_controls

                refuse_inapplicable_controls(policy, threads=False)
                engine._native = policy._call(engine._native.fork, seed=policy.seed)
            else:
                engine._native = policy._call(
                    engine._native.reconfigure_execution, seed=policy.seed, threads=policy.threads
                )
        engine._last_result = None
        engine._seed, engine._threads = policy.seed, policy.threads
        return replace(
            study,
            execution=policy,
            _engine=engine,
            reuse=Reuse(
                ("data", "identification"), ("physical_plan", "fits", "draws"), "reprepared"
            ),
        )
    request = {**replay_declaration(study._declarations), **declarations}
    if getattr(study._engine, "_kind", None) == "learned_trial":
        from ._prepared_trial import reconfigure_trial

        return reconfigure_trial(study, request, policy)
    data = study._data
    if data is None:
        raise CausalValueError(
            "this retained study has no Python data view for scientific reconfiguration"
        )
    branch = prepare(data, execution=policy, **request)
    return replace(
        branch,
        reuse=Reuse(("data",), ("identification", "program", "fits", "scores"), "reprepared"),
    )


def analyze(
    data: Any,
    *,
    query: Any,
    graph: Any = None,
    discovery: Any = None,
    inference: Any = None,
    execution: Execution | None = None,
    identifier: Any = None,
    estimator: Any = None,
    learners: Learners | OriginScopedLearners | None = None,
    estimator_config: Mapping[str, Any] | None = None,
    refute: Any = None,
    class_prior: Any = None,
    max_completions: int | None = None,
    participant: str | None = None,
    missing: Any = None,
    population_registry: Any = None,
    validators: Sequence[Any] | Mapping[str, Any] | None = None,
    regimes: Sequence[int] | None = None,
    provider: Any = None,
    sampling_design: str | None = None,
    unit_key: Any = None,
    accept_discovered: bool = False,
    data_diagnostics: Any = _AUTO_DATA_DIAGNOSTICS,
    retain_joint_horizons: bool = False,
    on_refusal: Literal["refuse", "describe"] = "refuse",
    evidence_candidates: Sequence[Any] | None = None,
    outcome_units: str | None = None,
    quantity_population: str = "target",
    quantity_transform: str = "identity",
    dose_units: str | None = None,
    measurement: Any = None,
    measurement_draws: int = 30,
    imputations: Sequence[Any] | None = None,
) -> Outcome[Any]:
    """Prepare and estimate a causal question in one call.

    Pass the same declarations as :func:`prepare`, including the required ``query``.
    With declared ``learners``, ``estimator="dml"`` selects the partially linear score
    (the default) and ``estimator="aipw"`` selects the augmented inverse-probability score.

    Returns an :class:`~antecedent.outcomes.Outcome`. Check ``outcome.refusal`` first. When it is
    ``None``, read ``outcome.answer`` (the estimate), ``outcome.uncertainty``, ``outcome.support``
    and ``outcome.assumptions``, or call ``outcome.inspect()`` for the full report;
    ``outcome.value`` is the typed result and ``outcome.study`` the executed study for
    :func:`rebind`, :func:`reconfigure` and :func:`refute`. When it is set, the question was
    refused on scientific grounds: read ``outcome.refusal.code`` and ``outcome.refusal.remedy``.

    A scientific refusal met while preparing or while estimating is returned on the outcome,
    never raised. Malformed arguments (:class:`~antecedent.errors.CausalTypeError`,
    :class:`~antecedent.errors.CausalValueError`) and operational failures still raise.

    Example::

        import antecedent as ant

        outcome = ant.analyze(
            data,
            graph=[("z", "t"), ("z", "y"), ("t", "y")],
            query=ant.AverageEffect(treatment="t", outcome="y"),
        )
        if outcome.refusal is None:
            print(outcome.answer.value)
        else:
            print(outcome.refusal.code, outcome.refusal.remedy)
    """
    def run() -> Outcome[Any]:
        study = prepare(
            data,
            query=query,
            graph=graph,
            discovery=discovery,
            inference=inference,
            execution=execution,
            identifier=identifier,
            estimator=estimator,
            learners=learners,
            estimator_config=estimator_config,
            refute=refute,
            class_prior=class_prior,
            max_completions=max_completions,
            participant=participant,
            missing=missing,
            population_registry=population_registry,
            validators=validators,
            regimes=regimes,
            provider=provider,
            sampling_design=sampling_design,
            unit_key=unit_key,
            accept_discovered=accept_discovered,
            data_diagnostics=data_diagnostics,
            retain_joint_horizons=retain_joint_horizons,
            on_refusal=on_refusal,
            evidence_candidates=evidence_candidates,
            outcome_units=outcome_units,
            quantity_population=quantity_population,
            quantity_transform=quantity_transform,
            dose_units=dose_units,
            measurement=measurement,
            measurement_draws=measurement_draws,
            imputations=imputations,
        )
        return estimate(study)

    return capture_outcome(run)


def _is_cyclic_preparation(study: Any) -> bool:
    # Stateful streaming/adaptive sessions deliberately do not participate in
    # immutable branching through structural duck typing.
    from .cyclic import PreparedCyclicFamily, PreparedCyclicSystem

    return isinstance(study, (PreparedCyclicFamily, PreparedCyclicSystem))
