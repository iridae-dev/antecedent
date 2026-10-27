"""Physical controls live on ``antecedent.Execution`` and nowhere else.

A public Python-defined callable never takes a loose seed, thread count, cancellation token,
memory limit or progress/stage observer: it takes ``execution: Execution | None = None`` and
translates it for the engine (``antecedent.execution.native_controls``). A seed that is a
declared scientific field of a spec or record stays, and each such exception is named in
``ALLOWLIST`` with its reason. The list holds the scientific seeds and the ``Execution``
constructors only.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

import antecedent

PHYSICAL = re.compile(
    r"^(?:.*_)?(?:seed|threads|n_threads|num_threads|cancel|cancel_token|cancellation|"
    r"on_progress|on_stage|max_bytes|memory_bytes|memory_limit|memory_limit_bytes|"
    r"max_memory_bytes)$"
)

# "<defining module>.<qualified name>:<parameter>" -> reason.
ALLOWLIST: dict[str, str] = {
    # --- constructors and declarations of Execution itself ---------------------------------
    "antecedent.execution.Execution:seed": "Execution owns the seed",
    "antecedent.execution.Execution:threads": "Execution owns the thread count",
    "antecedent.execution.Execution:cancel": "Execution owns the cancellation token",
    "antecedent.execution.Execution:on_progress": "Execution owns the progress observer",
    "antecedent.execution.Execution:on_stage": "Execution owns the stage observer",
    "antecedent.execution.ResourceLimits:memory_bytes": "the memory limit Execution carries",
    "antecedent.execution.CachePolicy:max_bytes": "the cache bound Execution carries",
    "antecedent.execution.CachePolicy.bounded:max_bytes": "the cache bound Execution carries",
    "antecedent.execution.native_controls:seed": "names the native keyword that receives the seed",
    "antecedent.execution.native_controls:threads": "names the native keyword for threads",
    "antecedent.execution.native_controls:cancel": "names the native keyword for cancellation",
    "antecedent.execution.native_controls:on_progress": "names the native keyword for progress",
    # --- scientific seeds: declared fields of a spec, request or record ---------------------
    "antecedent.design.planning.Simulation:seed": "declared simulation stream of the planned trial",
    "antecedent.design.planning.SimulatedCharacteristics:seed": "stream recorded on the result",
    "antecedent.design._declarations.SignalSpec:rng_seed": "declared Monte Carlo stream of a value",
    "antecedent.design._declarations.ConsumedRanking:rng_seed": "stream recorded on the ranking",
    "antecedent.design.evidence.EvidenceValue:rng_seed": "declared Monte Carlo stream of a value",
    "antecedent.design.evidence.rank_studies:rng_seed": "declared Monte Carlo stream of the ranking",
    "antecedent.design.ranking.rank_designs:rng_seed": "declared Monte Carlo stream, in the request",
    "antecedent.design.study_ranking.StudyProgram:rng_seed": "stream recorded on the program",
    "antecedent.design.study_ranking.compile_for_study:rng_seed": "declared stream, in the program",
    "antecedent.design.study_ranking.evaluate_for_study:rng_seed": "declared stream, in the request",
    "antecedent._batch.SimultaneousInterval:seed": "draw seed recorded on the interval",
    "antecedent.adaptive_inference.WeightingProcedureReport:seed": "stream recorded on the report",
    "antecedent.decision_reliability.StreamId:run_seed": "identity of a declared draw stream",
    "antecedent.decision_reliability.CohortUtilityDraws:run_seed": "identity of a declared stream",
    "antecedent.decision_reliability.OffPolicyReturnDraws:run_seed": "identity of a declared stream",
    "antecedent.decision_reliability.RolloutUtilityDraws:seed": "identity of a declared stream",
    "antecedent.dose_grid.dose_functional:band_seed": "declared simultaneous-band seed",
    "antecedent.drift.StalenessPolicy:seed": "declared replay seed of the policy",
    "antecedent.graph_falsification.GraphFalsification:execution_seed": "seed recorded on the result",
    "antecedent.graph_falsification.GraphFalsification:permutation_seed": "declared permutation seed",
    "antecedent.graph_falsification.evaluate_graph:permutation_seed": "declared permutation seed",
    "antecedent.latent_class.LatentClassEffects:seed": "seed recorded on the result",
    "antecedent.measurement_models.ContinuousErrorData:seed": "declared seed of the error data",
    "antecedent.measurement_models.ContinuousMeasurementEstimate:seed": "seed recorded on the result",
    "antecedent.missing.ImputationSpec:seed": "declared imputation seed",
    "antecedent.missing.MissingPolicy.multiple_imputation:seed": "declared imputation seed",
    "antecedent.nonlinear_mediation.MediationBootstrap:seed": "declared bootstrap seed",
    "antecedent.provider_callbacks.CallbackContext:stream_seed": "stream handed to a callback",
    "antecedent.provider_callbacks.Request:seed": "declared seed of a callback request",
    "antecedent.recalc_bayesian.JointBayesianRequest:seed": "declared seed of the request",
    "antecedent.recalc_external.ExternalCallbackRequest:seed": "declared seed of the request",
    "antecedent.recalc_external.CallbackInputs:seed": "stream handed to an external callback",
    "antecedent.recalc_external.CallbackInputs:cancellation": "token handed to an external callback",
    "antecedent.recalc_temporal.CheckedTemporalIntervalConfig:bootstrap_seed": "declared bootstrap seed",
    "antecedent.recalc_temporal.CheckedTemporalIntervalIdentity:producing_seed": "producer seed in identity",
    "antecedent.repair.SearchReceipt:memory_limit_bytes": "limit recorded on a search receipt",
    "antecedent.results._views.EstimateView:crossfit_seed": "cross-fit seed recorded on the view",
    "antecedent.results._views.ImputationView:seed": "imputation seed recorded on the view",
    "antecedent.results._views.PerformanceView:peak_memory_bytes": "measured peak, not a control",
    "antecedent.results._views.PhysicalPlanView:estimated_peak_memory_bytes": "plan telemetry",
    "antecedent.results._views.PhysicalPlanView:worker_threads": "plan telemetry",
    "antecedent.selective_discovery.StaticContract:seed": "declared discovery seed",
    "antecedent.selective_discovery.TemporalContract:seed": "declared discovery seed",
    "antecedent.selective_discovery.SplitEffect:_seed": "split seed recorded on the effect",
    "antecedent.selective_discovery.ParentSetEffect:_seed": "split seed recorded on the effect",
    "antecedent.selective_discovery.discover_on_split:split_seed": "declared split seed",
    "antecedent.selective_discovery.sample_split_effect:split_seed": "declared split seed",
    "antecedent.selective_discovery.sample_split_parent_set_effect:split_seed": "declared split seed",
    "antecedent.sequential_evaluation.RolloutEstimate:seed": "stream recorded on the estimate",
    "antecedent.sequential_evaluation.stream_uniform:seed": "declared stream of the rollout",
    "antecedent.transport._cpdag_scenarios.ScenarioCovarianceResult:seed": "seed recorded on the result",
    "antecedent.transport._learned_continuous.LearnedContinuousEstimate:seed": "seed recorded on the result",
    "antecedent.transport._smoothed_dose.SmoothedDoseEstimate:seed": "seed recorded on the result",
    "antecedent.transport.trial_compatible.TrialConfidenceSet:execution_seed": "seed recorded on the set",
    # --- scientific seeds: the stream or fold plan is recorded in the result or artifact ---
    "antecedent._batch.BatchResult.retarget:simultaneous_seed": "draw seed of the band, recorded on the interval",
    "antecedent._batch.CandidateScreen.from_units:seed": "split seed, recorded in the screen id and receipt",
    "antecedent.adaptive_inference.ParticipantProcedure.simulate:seed": "declared simulation stream, recorded in the receipt",
    "antecedent.adaptive_inference.ParticipantProcedure.simulate_history:seed": "declared simulation stream of the history",
    "antecedent.adaptive_inference.RepeatedMeasuresProcedure.derive:seed": "declared simulation stream of the derivation",
    "antecedent.adaptive_inference.RepeatedMeasuresProcedure.derive_paradigm:seed": "declared simulation stream of the derivation",
    "antecedent.dose_grid.dose_grid_posterior:seed": "draw seed stored in the artifact and replayed from it",
    "antecedent.latent_class.latent_class_effects:seed": "seed recorded on the result and replayed from the artifact",
    "antecedent.nonlinear_mediation.mediation_effects:seed": "bootstrap seed recorded on the result and artifact",
    "antecedent.scientific_bridge.PreparedDistributionBridge.derive:seed": "mapping seed recorded with the derived table",
    "antecedent.sequential_evaluation.rollout:seed": "stream recorded on the rollout estimate",
    "antecedent.transport._joint_sensitivity.consume_joint_mechanism_sensitivity_artifact:max_memory_bytes": "consumer ceiling on the stored search envelope, not Execution memory",
}


def _public_modules() -> list[object]:
    modules = [antecedent]
    for info in pkgutil.walk_packages(antecedent.__path__, "antecedent."):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        modules.append(importlib.import_module(info.name))
    return modules


def _callables(module: object):
    exported = getattr(module, "__all__", None)
    names = exported if exported is not None else [n for n in dir(module) if not n.startswith("_")]
    for name in names:
        try:
            obj = getattr(module, name)
        except AttributeError:
            continue
        if inspect.isclass(obj):
            if getattr(obj, "__module__", "").split(".")[0] != "antecedent":
                continue
            yield obj
            for member_name, member in vars(obj).items():
                if member_name.startswith("_"):
                    continue
                target = (
                    member.__func__ if isinstance(member, classmethod | staticmethod) else member
                )
                if isinstance(target, property):
                    continue
                if inspect.isfunction(target):
                    yield target
        elif inspect.isfunction(obj):
            yield obj


def _violations() -> dict[str, str]:
    found: dict[str, str] = {}
    seen: set[int] = set()
    for module in _public_modules():
        for obj in _callables(module):
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            try:
                parameters = inspect.signature(obj).parameters
            except (TypeError, ValueError):
                continue
            owner = f"{obj.__module__}.{obj.__qualname__}"
            for name, parameter in parameters.items():
                if parameter.kind is parameter.VAR_KEYWORD and name == "execution":
                    found[f"{owner}:**{name}"] = "a hidden loose-control forwarder"
                elif PHYSICAL.match(name):
                    found[f"{owner}:{name}"] = name
    return found


def test_no_public_callable_takes_a_loose_physical_control() -> None:
    found = _violations()
    unexplained = sorted(set(found) - set(ALLOWLIST))
    assert not unexplained, (
        "public callables take a loose physical control; take `execution: Execution | None` "
        "instead (or name a scientific seed in ALLOWLIST with its reason):\n  "
        + "\n  ".join(unexplained)
    )


def test_the_allowlist_names_only_live_entries() -> None:
    found = _violations()
    stale = sorted(set(ALLOWLIST) - set(found))
    assert not stale, "migrated or removed entries still allow-listed:\n  " + "\n  ".join(stale)


def test_every_allowlist_entry_states_its_reason() -> None:
    assert all(reason.strip() for reason in ALLOWLIST.values())
