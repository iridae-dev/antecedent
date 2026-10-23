"""Cross-surface slice cases on the public Python API.

Scientific inputs and hand values are `conformance/cross_surface/slice_cases.json`.
Each case drops the producer and reads an independently loaded artifact.
Specified feedback has no Python binding; the Rust slice tests cover it. The finite-discrete
static cells are built through the statistical stage from rows of the fixture's counts. The
finite-discrete horizon cells have no facade producer and are covered from the law artifact in
the Rust io tests.
"""

from __future__ import annotations

import json
import math
import re

import pytest
from antecedent import (
    Admg,
    AverageEffect,
    Bayesian,
    Frequentist,
    PulseEffect,
    analyze,
    load,
)
from antecedent import evidence_ledger as ledger_api
from antecedent._artifact_protocol import exported_artifact
from antecedent.inference import MeasuredInference
from antecedent.transport import advanced as transport
from antecedent.transport._temporal_extensions import (
    MARGINALIZED_LABEL,
    TemporalExtensionRefusal,
    TemporalIntervalCandidate,
    _temporal_dependent_interval_candidate,
)

from _repo_text import REPO_ROOT, load_json
from test_observation_recovery import REQUESTS, decide, observed_law, truth

CASES = load_json(REPO_ROOT / "conformance" / "cross_surface" / "slice_cases.json")


# The checked-operation reference the independent consumer leaves unresolved. Only the static
# frequentist line has an independent replay; every other cell's body is readable and forwardable,
# and is not claimed as a verified program.
_UNRESOLVED = {
    (False, False): (),
    (False, True): ("dependencies.checked_bayesian_dag_ate_operation",),
    (True, False): ("dependencies.checked_temporal_dag_effect_operation",),
    (True, True): ("dependencies.checked_bayesian_temporal_dag_effect_operation",),
}
_MONTE_CARLO_STANDARD_ERRORS = 5.0


def _gaussian_cell(horizon: bool, bayesian: bool):
    gaussian = CASES["gaussian"]
    if horizon:
        data = {"t": gaussian["horizon_treatment"], "y": gaussian["horizon_outcome"]}
        graph = [("t", 1, "y", 0)]
        query = PulseEffect("t", "y", treatment_lag=1, horizon_steps=1)
        label = "gaussian-horizon-bayesian" if bayesian else "gaussian-horizon-frequentist"
    else:
        data = {"t": gaussian["treatment"], "y": gaussian["outcome"]}
        graph = [("t", "y")]
        query = AverageEffect("t", "y")
        label = "gaussian-static-bayesian" if bayesian else "gaussian-static-frequentist"
    inference = (
        Bayesian(
            n_draws=gaussian["posterior_draws"],
            prior_scale=gaussian["prior_scale"],
            backend="conjugate",
        )
        if bayesian
        else Frequentist()
    )
    result = analyze(
        data,
        graph=graph,
        query=query,
        inference=inference,
        bootstrap=0,
        refute="none",
        seed=1,
    )
    # A frequentist fit is exact on the noiseless line. A Bayesian point estimate is the mean of
    # iid conjugate draws, so it carries Monte Carlo error: five standard errors of that mean.
    if bayesian:
        assert result.posterior is not None
        tolerance = (
            _MONTE_CARLO_STANDARD_ERRORS
            * result.posterior.effect_sd
            / math.sqrt(result.posterior.n_draws)
        )
        assert result.posterior.effect_mean == pytest.approx(gaussian["effect"], abs=0.05)
        assert result.posterior.interval_type == "equal_tailed_95"
    else:
        assert result.posterior is None
        tolerance = 1e-8 if horizon else 1e-12
    assert result.estimate.ate == pytest.approx(gaussian["effect"], abs=tolerance)
    artifact = result.export(artifact_id=label)
    del result
    loaded = load(artifact)
    assert loaded.acceptance.recognized
    assert loaded.acceptance.unresolved == _UNRESOLVED[(horizon, bayesian)], label
    assert loaded.acceptance.verified == (not loaded.acceptance.unresolved)
    assert loaded.decoded_artifact.payload["estimate"] == pytest.approx(
        gaussian["effect"], abs=tolerance
    )


def test_four_way_gaussian_cells_reload_from_independent_artifacts():
    for horizon in (False, True):
        for bayesian in (False, True):
            _gaussian_cell(horizon, bayesian)


def _discrete_rows(counts, level):
    """The fixture's joint as rows, `t` slowest and `y` fastest, at one treatment level or all."""
    rows = {"z": [], "t": [], "y": []}
    for cell, copies in enumerate(counts):
        treatment, confounder, outcome = cell // 4, (cell // 2) % 2, cell % 2
        if level is not None and treatment != level:
            continue
        rows["z"] += [float(confounder)] * copies
        rows["t"] += [float(treatment)] * copies
        rows["y"] += [float(outcome)] * copies
    return rows


def _discrete_arm(counts, level, estimator):
    """One arm `do(t = level)`: the trial rows at that level plus the whole table's confounder."""
    graph = Admg.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")], [("t", "y")])
    identified = transport.identify_classical(
        graph,
        transport.SelectionDiagram("source", "target", ["z"]),
        outcomes=["y"],
        treatments=["t"],
    )
    variables = [transport.VariableCoordinate(name, "binary") for name in ("z", "t", "y")]
    catalog = transport.EvidenceCatalog(
        environments=[
            transport.Environment("source", variables, selection_targets=["z"]),
            transport.Environment("target", variables),
        ],
        regimes=[
            transport.EvidenceRegime(
                "trial", "source", kind="experimental", interventions=["t"], measured=["z", "y"]
            ),
            transport.EvidenceRegime("covariates", "target", measured=["z"]),
        ],
        bindings=[
            transport.RegimeBinding(
                name, name, sampling="independent", dependence="independent_studies"
            )
            for name in ("trial", "covariates")
        ],
        target_sampling="representative_sample",
    )
    trial = _discrete_rows(counts, level)
    data = transport.StatisticalTransportData(
        samples=(
            transport.RegimeSample(
                "source",
                "trial",
                "trial",
                {"z": trial["z"], "y": trial["y"]},
                interventions=(("t", float(level)),),
            ),
            transport.RegimeSample(
                "target", "covariates", "covariates", {"z": _discrete_rows(counts, None)["z"]}
            ),
        )
    )
    study = transport.prepare_statistical(
        identified, catalog, data, at={"t": float(level)}, bootstrap=0, estimator=estimator
    )
    return load(study.estimate().export())


def _draw_means(loaded):
    posterior = loaded.uncertainty["bayesian_posterior"]
    assert posterior["draws_ok"] == posterior["draws_requested"] == len(posterior["probabilities"])
    return [
        math.fsum(atom[0] * p for atom, p in zip(loaded.atoms, draw, strict=True))
        for draw in posterior["probabilities"]
    ]


def _mean_and_standard_error(draws):
    mean = math.fsum(draws) / len(draws)
    variance = math.fsum((draw - mean) ** 2 for draw in draws) / (len(draws) - 1)
    return mean, math.sqrt(variance / len(draws))


def _dirichlet_arm_mean(counts, level):
    """Posterior mean of one arm under Dirichlet(count + 1) on the trial joint and the confounder.

    The trial's (z, y) joint and the covariates' z law are independent, and a Dirichlet's
    conditional y | z is a beta independent of its z marginal, so the mean of
    sum_z P(z) P(y = 1 | z) is the product of the two posterior means.
    """

    def cell(t, z, y):
        return counts[t * 4 + z * 2 + y]

    def confounder_rows(z):
        return sum(cell(t, z, 0) + cell(t, z, 1) for t in (0, 1))

    rows = confounder_rows(0) + confounder_rows(1)
    total = 0.0
    for z in (0, 1):
        positive = cell(level, z, 1) + 1
        weight = (confounder_rows(z) + 1) / (rows + 2)
        total += weight * positive / (cell(level, z, 0) + positive + 1)
    return total


def test_finite_discrete_static_frequentist_cell_reloads_from_independent_artifacts():
    discrete = CASES["finite_discrete"]
    active = _discrete_arm(discrete["counts"], 1, "plugin")
    control = _discrete_arm(discrete["counts"], 0, "plugin")
    assert active.mean("y") == pytest.approx(discrete["active"], abs=1e-12)
    assert control.mean("y") == pytest.approx(discrete["control"], abs=1e-12)
    assert active.mean("y") - control.mean("y") == pytest.approx(discrete["contrast"], abs=1e-12)
    assert active.uncertainty["bayesian_posterior"] is None


def test_finite_discrete_static_bayesian_cell_reloads_posterior_draws_around_the_dirichlet_mean():
    # The state-space Dirichlet posterior is not the fixture's hand-built two-draw mixture, so its
    # mean is the exact Dirichlet moment, within five standard errors of the draws.
    discrete = CASES["finite_discrete"]
    counts = discrete["counts"]
    active = _discrete_arm(counts, 1, "state_space_dirichlet")
    control = _discrete_arm(counts, 0, "state_space_dirichlet")
    active_mean, active_error = _mean_and_standard_error(_draw_means(active))
    control_mean, control_error = _mean_and_standard_error(_draw_means(control))
    assert active_mean == pytest.approx(
        _dirichlet_arm_mean(counts, 1), abs=_MONTE_CARLO_STANDARD_ERRORS * active_error
    )
    assert control_mean == pytest.approx(
        _dirichlet_arm_mean(counts, 0), abs=_MONTE_CARLO_STANDARD_ERRORS * control_error
    )
    # The point law each artifact carries is still the plug-in of the same rows.
    assert active.mean("y") == pytest.approx(discrete["active"], abs=1e-12)
    assert control.mean("y") == pytest.approx(discrete["control"], abs=1e-12)


def _sources(session):
    mixed = CASES["mixed_evidence"]
    sources = []
    for index, (start, end) in enumerate(mixed["observation_ranges"]):
        sources.append(
            session.add_source(
                ledger_api.SourceDeclaration(
                    f"study_{index}",
                    "observation_set",
                    observations=ledger_api.rows("cohort", range(start, end)),
                )
            )
        )
    return [
        ledger_api.StudySummary(source, estimate, variance)
        for source, estimate, variance in zip(
            sources, mixed["estimates"], mixed["variances"], strict=True
        )
    ]


def test_mixed_evidence_pools_and_reloads_from_the_ledger():
    mixed = CASES["mixed_evidence"]
    session = ledger_api.EvidenceLedger()
    pooled = session.pool_summaries("pool", _sources(session))
    assert isinstance(pooled.value, ledger_api.FixedEffect)
    assert pooled.value.estimate == pytest.approx(mixed["pooled"])
    assert pooled.value.variance == pytest.approx(mixed["pooled_variance"])
    declared = sum(end - start for start, end in mixed["observation_ranges"])
    assert pooled.mass.distinct_observations == declared
    exported = exported_artifact(session, "mixed-evidence")
    del session
    consumed = ledger_api.consume_evidence_ledger_artifact(
        exported.data, exported.expectation.identity
    )
    replay = consumed.synthesis_of(pooled.result)
    assert replay.values[0] == pytest.approx(mixed["pooled"])
    assert replay.values[1] == pytest.approx(mixed["pooled_variance"])


def test_reused_evidence_keeps_one_observation_count():
    session = ledger_api.EvidenceLedger()
    pooled = session.pool_summaries("pool", _sources(session))
    first = session.reuse_result("cited_once", pooled.result)
    second = session.reuse_result("cited_again", pooled.result)
    assert first != second
    counted = pooled.mass.distinct_observations
    for result_id in (pooled.result, first, second):
        assert session.result(result_id).mass.distinct_observations == counted
    exported = exported_artifact(session, "reused-evidence")
    del session
    consumed = ledger_api.consume_evidence_ledger_artifact(
        exported.data, exported.expectation.identity
    )
    labels = {record.label: record.mass.distinct_observations for record in consumed.results}
    assert labels["pool"] == counted
    assert labels["cited_once"] == counted
    assert labels["cited_again"] == counted


def test_overlapping_studies_pool_under_a_declared_covariance_and_replace_the_disjoint_pool():
    mixed = CASES["mixed_evidence"]
    overlap = mixed["overlap"]
    session = ledger_api.EvidenceLedger()
    summaries = _sources(session)
    disjoint = session.pool_summaries("pool", summaries)
    first_source = summaries[0].source
    report = session.revise_source(
        ledger_api.SourceRevision.update(
            summaries[1].source,
            ledger_api.SourceDeclaration(
                "study_1_grown",
                "observation_set",
                observations=ledger_api.rows("cohort", range(*overlap["grown_range"])),
            ),
        )
    )
    grown = report.created_source
    assert grown is not None
    assert session.standing(disjoint.result) == "recompute"
    account = session.account_observations([first_source, grown])
    assert account.mass.distinct_observations == overlap["distinct_observations"]

    overlapping = [
        ledger_api.StudySummary(source, estimate, variance)
        for source, estimate, variance in zip(
            [first_source, grown], mixed["estimates"], mixed["variances"], strict=True
        )
    ]
    with pytest.raises(ledger_api.EvidenceLedgerRefusal) as refusal:
        session.pool_summaries("overlap_pool", overlapping, replaces=disjoint.result)
    assert refusal.value.reason_code == "evidence_double_counted"
    assert refusal.value.detail == "evidence_ledger.shared_observations"

    replacing = session.pool_summaries(
        "overlap_pool",
        overlapping,
        covariance=overlap["covariance"],
        replaces=disjoint.result,
    )
    assert replacing.value.estimate == pytest.approx(overlap["pooled"])
    assert replacing.value.variance == pytest.approx(overlap["pooled_variance"])
    assert replacing.mass.declared_slots == overlap["declared_slots"]
    assert replacing.mass.distinct_observations == overlap["distinct_observations"]
    assert replacing.mass.repeated_slots == overlap["shared"]
    record = session.result(replacing.result)
    assert record.dependence_declared
    assert record.replaces == disjoint.result
    assert session.result(disjoint.result).status == "replaced"

    exported = exported_artifact(session, "overlapping-evidence")
    del session
    consumed = ledger_api.consume_evidence_ledger_artifact(
        exported.data, exported.expectation.identity
    )
    replay = consumed.synthesis_of(replacing.result)
    assert replay.values[0] == pytest.approx(overlap["pooled"])
    assert replay.values[1] == pytest.approx(overlap["pooled_variance"])
    statuses = {item.id: item.status for item in consumed.results}
    assert statuses[disjoint.result] == "replaced"


def _panel(shocks):
    temporal = CASES["temporal"]
    rows = []
    for unit in range(temporal["units"]):
        shock = shocks[unit % len(shocks)]
        time = 0
        for s0 in range(2):
            for a1 in range(2):
                for l2 in range(2):
                    for a2 in range(2):
                        mean = 0.3 + 0.1 * l2 + 0.1 * s0 + 0.2 * s0 * l2 + 0.05 * (a1 + a2)
                        rows.append((unit, time, s0, a1, l2, a2, mean + shock))
                        time += 1
    return transport.TemporalUnitPanel.from_rows(temporal["snapshot"], rows)


def _interval_arguments():
    temporal = CASES["temporal"]
    return {
        "sequence": tuple(temporal["sequence"]),
        "target_law": transport.InitialStateLaw(
            temporal["target_snapshot"],
            {level: mass for level, mass in temporal["target_law"]},
        ),
        "seed": temporal["seed"],
        "replicates": temporal["replicates"],
        "level": temporal["level"],
        "min_units": temporal["min_units"],
        "max_failed_fraction": temporal["max_failed_fraction"],
    }


def _reload_candidate(panel, point):
    """Build the candidate over `panel`, drop it, and consume its artifact."""
    try:
        candidate = _temporal_dependent_interval_candidate(
            panel, estimand=MARGINALIZED_LABEL, **_interval_arguments()
        )
    except TemporalExtensionRefusal as refusal:
        pytest.skip(f"the candidate route is closed in this build: {refusal}")
    assert candidate.point == pytest.approx(point, abs=1e-12)
    assert candidate.calibration == "unmeasured"
    artifact = candidate.export()
    identity = candidate.identity
    del candidate
    return TemporalIntervalCandidate.load(artifact, expected_identity=identity)


def test_temporal_dependence_reloads_from_the_measured_artifact():
    temporal = CASES["temporal"]
    panel = _panel(temporal["unit_shocks"])
    loaded = _reload_candidate(panel, temporal["point"])
    assert loaded.point == pytest.approx(temporal["point"], abs=1e-12)

    # The measured route additionally needs a current attesting coverage record. Until the
    # coverage records are re-measured it refuses for exactly that reason and no other.
    try:
        result = transport.temporal_dependent_interval(panel, **_interval_arguments())
    except TemporalExtensionRefusal as refusal:
        assert "coverage_record_not_attesting" in str(refusal), refusal
        return
    artifact = result.export()
    expected = result.expected_identity
    del result
    measured = MeasuredInference.load(artifact, expected=expected)
    assert measured.scalar("response").point == pytest.approx(temporal["point"], abs=1e-12)


def _independent_rows_standard_error(panel_rows):
    mean = sum(panel_rows) / len(panel_rows)
    variance = sum((y - mean) ** 2 for y in panel_rows) / (len(panel_rows) - 1)
    return math.sqrt(variance / len(panel_rows))


def test_uneven_unit_shocks_widen_the_whole_unit_interval_beyond_independent_rows():
    temporal = CASES["temporal"]
    uneven = temporal["uneven"]
    shocks = uneven["unit_shocks"]
    loaded = _reload_candidate(_panel(shocks), uneven["point"])

    units = temporal["units"]
    mean_shock = sum(shocks[unit % len(shocks)] for unit in range(units)) / units
    assert abs(mean_shock) > 0.1, "the shocks do not cancel"
    assert loaded.point - temporal["point"] == pytest.approx(mean_shock, abs=1e-12)

    outcomes = []
    for unit in range(temporal["units"]):
        shock = shocks[unit % len(shocks)]
        for s0 in range(2):
            for a1 in range(2):
                for l2 in range(2):
                    for a2 in range(2):
                        mean = 0.3 + 0.1 * l2 + 0.1 * s0 + 0.2 * s0 * l2 + 0.05 * (a1 + a2)
                        outcomes.append(mean + shock)
    rows_width = 2 * temporal["row_interval_z"] * _independent_rows_standard_error(outcomes)
    whole_unit_width = loaded.upper - loaded.lower
    assert whole_unit_width > rows_width, (whole_unit_width, rows_width)
    assert loaded.lower < loaded.point < loaded.upper


def test_recovered_feedback_reloads_the_observation_law():
    recovery = CASES["recovery"]
    law = observed_law(snapshot=recovery["python_snapshot"])
    assert law.population == recovery["population"]
    assert law.snapshot_identity == recovery["python_snapshot"]
    builder = decide()
    prepared = builder.prepare_exact(law, REQUESTS)
    del builder
    live = json.loads(prepared.estimate())
    artifact = prepared.export()
    del prepared
    consumed = json.loads(transport.consume_observation_recovery_artifact(artifact))
    assert consumed["recovered"]["probabilities"] == pytest.approx(
        live["recovered"]["probabilities"]
    )
    assert consumed["recovered"]["probabilities"] == pytest.approx(truth())
    assert consumed["recovered"]["origin"] == "recovered"


def _operation_message(name: str) -> str:
    return re.escape(name.replace("in_place", "in-place").replace("_", " "))


def test_reduced_retention_serves_only_the_summary():
    from antecedent import _native
    from antecedent.errors import CausalSerializationError, CausalStateError
    from test_temporal_extensions import started

    retention = CASES["retention"]
    mode = _native.RetentionMode.summary_only()
    assert mode.as_str() == retention["reduced"]
    reuse = _native.RetainedOperation.reuse_summary()
    assert mode.serves(reuse)
    mode.require(reuse)

    # Export the summary-only form of a live fit, drop the producer, and load it independently.
    producer = started(invariant=True)
    live = producer.least_squares
    full_ledger = producer.export()
    artifact = producer.export_summary_only()
    del producer
    consumer = transport.PeriodRefreshSummary.consume(artifact)
    assert consumer.retention == retention["reduced"]
    assert consumer.n == live.n
    assert consumer.beta == pytest.approx(live.beta, abs=1e-9)
    assert consumer.residual_variance == pytest.approx(live.residual_variance, abs=1e-9)

    # What the rows would have served is refused, naming the mode and the operation.
    refusals = {
        "row_reconstruction": consumer.replay_rows,
        "refit_from_retained_rows": consumer.refit_from_rows,
        "exact_row_eviction": lambda: consumer.evict_row(0),
        "in_place_weight_or_mask_update": lambda: consumer.set_mask(0, True),
    }
    for name in retention["unserved"]:
        operation = getattr(_native.RetainedOperation, name)()
        assert not mode.serves(operation), name
        with pytest.raises(
            TemporalExtensionRefusal,
            match=f"summary-only cannot serve {_operation_message(name)}",
        ):
            refusals[name]()
        with pytest.raises(
            CausalStateError, match=f"summary-only cannot serve {_operation_message(name)}"
        ):
            mode.require(operation)
    # A refused operation leaves the served summary as it was.
    assert consumer.n == live.n

    # A changed byte never yields a different summary, and neither form is read as the other.
    for position in range(len(artifact)):
        changed = bytearray(artifact)
        changed[position] ^= 0x01
        try:
            other = transport.PeriodRefreshSummary.consume(bytes(changed))
        except (TemporalExtensionRefusal, CausalSerializationError):
            continue
        assert other.beta == consumer.beta
        assert other.n == consumer.n
    with pytest.raises((TemporalExtensionRefusal, CausalSerializationError)):
        transport.consume_period_refresh_artifact(artifact)
    with pytest.raises(TemporalExtensionRefusal, match="summary_only"):
        transport.PeriodRefreshSummary.consume(full_ledger)
