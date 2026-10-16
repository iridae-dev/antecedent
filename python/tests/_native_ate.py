"""One-shot static ATE through the native prepared handle.

The facade refuses a few spellings that the native constructor still takes (the loose
``running_variable=`` / ``cutoff=`` / ``bandwidth=`` triple with ``estimator="rd.sharp"``) and
does not return the native result record; tests that pin those native contracts drive
``PreparedAnalysis.prepare(...).estimate(...)`` directly through this helper.
"""

from __future__ import annotations

from typing import Any

from antecedent._native import AteAnalysisResult, PreparedAnalysis


def native_ate(
    names: list[str],
    columns: list[Any],
    edges: list[tuple[str, str]],
    treatment: str,
    outcome: str,
    *,
    refute: Any = None,
    validators: Any = None,
    bootstrap: int | None = None,
    seed: int = 1,
    threads: int | None = None,
    **prepare_kwargs: Any,
) -> AteAnalysisResult:
    options = {
        key: value
        for key, value in (("refute", refute), ("validators", validators), ("bootstrap", bootstrap))
        if value is not None
    }
    prepared = PreparedAnalysis.prepare(
        names,
        columns,
        edges,
        treatment,
        outcome,
        seed=seed,
        threads=threads,
        options=options or None,
        **prepare_kwargs,
    )
    return prepared.estimate(names, columns, seed=seed, threads=threads)
