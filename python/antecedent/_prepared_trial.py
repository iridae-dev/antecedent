"""Canonical trial transport lowering over one owned native snapshot."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ._study_lifecycle import retain
from .errors import CausalTypeError, CausalUnsupportedError
from .execution import Execution
from .outcomes import capture_outcome
from .prepared import PreparedAnalysis, Reuse
from .transport._impl.trial import (
    TransportControls,
    TransportInference,
    TrialAipw,
    TrialAipwData,
    TrialAipwQuery,
)


def _settings(
    declarations: Mapping[str, Any],
) -> tuple[TrialAipwQuery, TrialAipw, TransportInference, TransportControls]:
    query = declarations["query"]
    if not isinstance(query, TrialAipwQuery):
        raise CausalTypeError("trial reconfiguration requires TrialAipwQuery")
    for name in (
        "graph",
        "discovery",
        "identifier",
        "estimator",
        "estimator_config",
        "refute",
        "class_prior",
        "max_completions",
        "participant",
        "missing",
        "population_registry",
        "validators",
        "regimes",
        "sampling_design",
        "outcome_units",
        "dose_units",
    ):
        if declarations.get(name) is not None:
            raise CausalUnsupportedError(
                f"{name}= does not apply to the certified trial transport request",
                reason_code="option_not_applicable",
            )
    if declarations.get("retain_joint_horizons"):
        raise CausalUnsupportedError(
            "trial contrasts have no joint horizons", reason_code="option_not_applicable"
        )
    provider = declarations.get("provider")
    inference = declarations.get("inference")
    controls = declarations.get("controls")
    if provider is not None and not isinstance(provider, TrialAipw):
        raise CausalTypeError("TrialAipwQuery requires provider=TrialAipw(...)")
    if inference is not None and not isinstance(inference, TransportInference):
        raise CausalTypeError("TrialAipwQuery requires inference=TransportInference(...)")
    if controls is not None and not isinstance(controls, TransportControls):
        raise CausalTypeError("trial controls must use TransportControls")
    provider, inference, controls = (
        provider or TrialAipw(),
        inference or TransportInference(),
        controls or TransportControls(),
    )
    if (controls.max_operations, controls.max_depth, controls.max_support_rows) != (
        10_000_000,
        256,
        1_000_000,
    ):
        raise CausalUnsupportedError(
            "Discrete evaluator limits do not apply to trial AIPW; use memory_bytes and cancel",
            reason_code="option_not_applicable",
        )
    return query, provider, inference, controls


def _options(provider: TrialAipw, inference: TransportInference) -> str:
    from .learners import _learner_wire

    return json.dumps(
        dict(
            outcome=_learner_wire(provider.outcome),
            membership=_learner_wire(provider.membership),
            folds=provider.folds,
            bootstrap=inference.bootstrap,
            coverage_level=inference.coverage_level,
        )
    )


def prepare_trial(data: Any, *, declarations: Mapping[str, Any], execution: Execution) -> Any:
    from .transport._impl.trial import prepare_trial as compile_trial

    query, provider, inference, controls = _settings(declarations)
    if not isinstance(data, TrialAipwData):
        raise CausalTypeError("TrialAipwQuery requires TrialAipwData")
    controls = replace(
        controls, cancel=execution.cancel if execution.cancel is not None else controls.cancel
    )
    engine = compile_trial(
        query, data, provider=provider, inference=inference, controls=controls, seed=execution.seed
    )
    engine._query = engine._display_query = query
    return engine


def reconfigure_trial(
    study: PreparedAnalysis[Any], declarations: Mapping[str, Any], execution: Execution
) -> PreparedAnalysis[Any]:
    declarations = retain(declarations)
    query, provider, inference, controls = _settings(declarations)
    engine = copy.copy(study._engine)
    candidate = capture_outcome(
        execution._call,
        engine._native.reconfigure,
        query.graph,
        list(query.diagram.selections),
        query.diagram.source,
        query.diagram.target,
        query.treatment,
        query.outcome,
        _options(provider, inference),
        seed=execution.seed,
        memory_bytes=controls.memory_bytes,
        cancel=execution.cancel if execution.cancel is not None else controls.cancel,
    )
    from .estimation import _Controls

    if candidate.refusal is None:
        engine._native = candidate.value
    engine._query = engine._display_query = query
    engine._last_result = None
    engine._controls = _Controls(
        cancel=execution.cancel if execution.cancel is not None else controls.cancel
    )
    return replace(
        study,
        _engine=engine,
        execution=execution,
        _declarations=declarations,
        _refused=candidate if candidate.refusal is not None else None,
        reuse=Reuse(
            ("data",),
            ("identification", "program", "statistical_plan", "fits", "scores"),
            "reprepared",
        ),
    )
