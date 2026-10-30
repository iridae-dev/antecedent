# SPDX-License-Identifier: MIT OR Apache-2.0
"""One marketing investigation: response, allocation, evidence, decision and next study.

Run offline with the 3.0 Python extension. All business units and prices are synthetic.
Generating parameters stay outside the estimator declaration. The small Monte Carlo run
illustrates the workflow; read its diagnostics before using its posterior scientifically.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import replace

from antecedent import (
    CachePolicy,
    Execution,
    decision,
    joint_panel as jp,
    scientific_bridge as sb,
)
from antecedent.evidence_ledger import EvidenceLedger, SourceDeclaration, SourceRevision
from antecedent.intervention_programs import var
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

SEED = 20261030
HORIZON = 24
CHANNELS = ("search", "social", "video")
EXECUTION = Execution(seed=SEED, cache=CachePolicy.bounded(128_000_000))
# Independent generating truth, never supplied as fitted parameters or initialization.
TRUTH = {
    "lag": (0.6, 0.3, 0.1),
    "half": 2.0,
    "coefficients": (3.0, 2.0, 1.0),
    "interaction": 1.5,
}
PROGRAMS = {
    "balanced": (2.0, 2.0, 2.0),
    "search_video": (3.0, 0.0, 3.0),
    "search_social": (3.0, 3.0, 0.0),
}


def increment(levels):
    """Terminal response under a constant program: lags have washed out by week 23."""
    saturated = [x / (x + TRUTH["half"]) for x in levels]
    return math.fsum(
        c * s for c, s in zip(TRUTH["coefficients"], saturated, strict=True)
    ) + (TRUTH["interaction"] * saturated[0] * saturated[2])


def simulate():
    rng = random.Random(SEED)
    groups, scenarios = {}, []
    for group in range(4):
        z = [rng.gauss(0, 1) for _ in range(74)]
        # Budgets respond to observed anticipated demand. No randomized program is assigned.
        data = {
            name: [
                max(0.0, rng.choice((0.0, 2.0, 4.0)) + 0.25 * demand) for demand in z
            ]
            for name in CHANNELS
        }
        data.update(z=z, season=[float(t % 4) for t in range(74)], y=[None, None])
        for t in range(2, 74):
            s = []
            for name in CHANNELS:
                exposure = sum(
                    w * data[name][t - k] for k, w in enumerate(TRUTH["lag"])
                )
                s.append(exposure / (exposure + TRUTH["half"]))
            data["y"].append(
                10
                + 0.3 * group
                + 0.7 * z[t]
                + (0, 0.2, -0.1, 0.1)[t % 4]
                + sum(c * x for c, x in zip(TRUTH["coefficients"], s, strict=True))
                + TRUTH["interaction"] * s[0] * s[2]
                + rng.gauss(0, 0.2)
            )
        groups[group] = data
        future_rng = random.Random(SEED + 100 + group)
        scenarios.append(
            {
                "group": group,
                "history": {n: data[n][-2:] for n in CHANNELS},
                "confounder": [future_rng.gauss(0, 1) for _ in range(HORIZON)],
                "season": [t % 4 for t in range(74, 74 + HORIZON)],
            }
        )
    return groups, scenarios


def worlds(scenarios, name, levels):
    return [
        {
            "id": label,
            "reference": i == 0,
            "channels": {
                n: [v] * HORIZON for n, v in zip(CHANNELS, doses, strict=True)
            },
            "scenarios": scenarios,
        }
        for i, (label, doses) in enumerate((("hold", (0.0, 0.0, 0.0)), (name, levels)))
    ]


def bridge_for(target):
    setting = sb.ExposureSetting(
        target.population_id,
        "program_period:23",
        "program_period:23",
        1.0,
        "declared_joint_program",
        1.0,
        "panel_period",
        target.regime_id,
    )
    return sb.BridgeContract(
        "parameter_intervention_response",
        replace(target, functional_id="experimental_sample_mean"),
        "model_parameter",
        target,
        "intervention_response",
        sb.Scope(target.population_id, target.regime_id),
        sb.AffineMapping(0.0, 1.0),
        assumptions=(
            sb.Assumption(
                "geo_likelihood",
                "parameter_interpretation",
                "Independent Gaussian geo contrast measures this exact finite-population terminal program contrast",
                "geo_experiment",
            ),
        ),
        mapping_uncertainty=sb.MappingUncertainty("fixed_identity", 0.0, 0.0, 0.0),
        source_dependence=sb.SourceDependence(
            "slope_covariance", "retained_slots", 0.0
        ),
        estimand_bridge=sb.EstimandBridge(
            "same_program",
            "geo_experiment",
            setting,
            setting,
            0.0,
            "panel_period",
            0.0,
            "declared_joint_program",
        ),
    )


def study_value(contrasts, weights, margin, cost, sigma, cells=512):
    """Independent EVSI quadrature for hold versus implement, with an error bound."""
    gains = [w * (margin * c - cost) for c, w in zip(contrasts, weights, strict=True)]
    lower, upper = min(contrasts) - 8 * sigma, max(contrasts) + 8 * sigma
    step = (upper - lower) / cells
    scale = 1 / (sigma * math.sqrt(2 * math.pi))
    future = 0.0
    for cell in range(cells):
        y = lower + (cell + 0.5) * step
        advantage = math.fsum(
            g * scale * math.exp(-0.5 * ((y - c) / sigma) ** 2)
            for c, g in zip(contrasts, gains, strict=True)
        )
        future += step * max(advantage, 0.0)
    absolute = math.fsum(abs(g) for g in gains)
    # Midpoint quadrature bound from the Gaussian density's total variation,
    # plus an eight-standard-deviation tail bound for the mixture.
    error = step / 2 * absolute * math.sqrt(2 / math.pi) / sigma
    error += absolute * 2 * math.exp(-32) / (8 * math.sqrt(2 * math.pi))
    return future - max(math.fsum(gains), 0.0), error


def run():
    started = time.perf_counter()
    groups, scenarios = simulate()
    prepared = jp.prepare(
        groups,
        channels=CHANNELS,
        outcome="y",
        confounder="z",
        season="season",
        seasons=4,
        interval_ns=604_800_000_000_000,
        worlds=worlds(scenarios, "balanced", PROGRAMS["balanced"]),
        interactions={"search_video": var("search") * var("video")},
        functional={"terms": [(0, -1.0), (1, 1.0)], "reference": 0},
        options={
            "optimizer": {"max_iterations": 2000, "gradient_tolerance": 1e-5},
            "initial": None,
            "inference": {
                "kind": "bayesian",
                "prior": {
                    "coefficient_sd": 10.0,
                    "log_half_saturation_mean": 0.0,
                    "log_half_saturation_sd": 1.0,
                    "lag": [],
                    "innovation_scale": 2.0,
                    "group_scale": 2.0,
                },
                "sampler": {
                    "chains": 4,
                    "warmup": 8000,
                    "draws": 12000,
                    "proposal_scale": 0.35,
                    "max_rhat": 1.05,
                    "min_ess": 100.0,
                },
            },
        },
        execution=EXECUTION,
    )
    inspection = prepared.inspect()
    assert inspection["fits_performed"] == 0
    print(
        "Compiled question and evidence requirements:",
        json.dumps(inspection, indent=2),
        flush=True,
    )
    fitted = require_success(prepared.estimate(execution=EXECUTION))
    report = fitted.report()
    print(
        "Fit diagnostics:",
        {
            key: report["diagnostics"][key]
            for key in (
                "optimizer_converged",
                "posterior_converged",
                "max_rhat",
                "min_bulk_ess",
                "min_tail_ess",
            )
        },
        flush=True,
    )
    assert report["diagnostics"]["optimizer_converged"]
    assert report["diagnostics"]["posterior_converged"]

    # Read fitted curves in their declared meaning: saturated lag-weighted exposure.
    parameters = report["point_parameters"]["parameters"]
    half = parameters["half_saturation"]
    curves = {
        name: [
            {
                "exposure": x,
                "isolated_response": channel["coefficient"] * x / (x + half),
            }
            for x in (0.0, 1.0, 2.0, 3.0, 4.0)
        ]
        for name, channel in zip(CHANNELS, parameters["channels"], strict=True)
    }
    print("Fitted isolated response curves:", curves, flush=True)
    print("Fitted lag weights:", [c["lag"] for c in parameters["channels"]], flush=True)
    print(
        "Geographic intercepts:",
        report["point_parameters"]["group_intercepts"],
        flush=True,
    )

    search = parameters["channels"][0]
    interaction = parameters["interactions"][0]["coefficient"]
    print(
        "Search at exposure 2, alone versus with video at exposure 2:",
        search["coefficient"] * 2 / (2 + half),
        search["coefficient"] * 2 / (2 + half) + interaction * (2 / (2 + half)) ** 2,
        flush=True,
    )
    print(
        "Search-only pulse of 4, subsequent weekly increments:",
        [
            search["coefficient"] * (4 * w) / (4 * w + half)
            for w in [*search["lag"], 0.0]
        ],
        flush=True,
    )

    # Every candidate uses the original fit; each contrast keeps a baseline and exact program.
    candidates = []
    controls = [
        {"channel": n, "offset": t, "lower": 0.0, "upper": 3.0, "unit_cost": 1.0}
        for t in range(HORIZON)
        for n in CHANNELS
    ]
    for name, levels in PROGRAMS.items():
        branch = prepared.retarget(worlds(scenarios, name, levels), execution=EXECUTION)
        result = require_success(branch.estimate(execution=EXECUTION))
        ranked = result.search_declared_programs(
            controls=controls, budget=144.0, execution=EXECUTION
        )
        assert ranked["exhaustive"] and not ranked["global_certificate"]
        row = next(row for row in ranked["rows"] if row["id"] == name)
        candidates.append((row["value"], name, result))
    _, selected, result = max(candidates, key=lambda row: row[0])
    report = result.report()
    assert prepared.inspect()["fits_performed"] == 1
    assert report["identified"] is False
    # Meaningful refusal: a missing whole-program assignment proof cannot be supplied by fit.
    rejected = [
        a for a in inspection["attempts"] if a["outcome"] in ("Rejected", "Unsupported")
    ]
    assert rejected
    print("Identification refusal:", rejected, flush=True)
    draws = report["draws"]
    assert all(row is not None for row in draws), (
        "Incomplete posterior slots cannot be conditioned"
    )
    contrasts = [row[1]["means"][-1] - row[0]["means"][-1] for row in draws]
    contrast = result.scheduled_contrast("terminal_mean")
    assert all(
        math.isclose(a, b, abs_tol=1e-9)
        for a, b in zip(contrast["draws"], contrasts, strict=True)
    )
    assert math.isclose(
        statistics.variance(contrasts),
        statistics.variance([r[1]["means"][-1] for r in draws])
        + statistics.variance([r[0]["means"][-1] for r in draws])
        - 2
        * statistics.covariance(
            [r[0]["means"][-1] for r in draws], [r[1]["means"][-1] for r in draws]
        ),
        abs_tol=1e-9,
    )
    # Independent known truth: balanced response is 3 + 1.5/4 = 3.375.
    assert math.isclose(increment(PROGRAMS["balanced"]), 3.375, abs_tol=1e-12)
    assert abs(contrast["point"] - increment(PROGRAMS[selected])) < 0.6

    # A new replicated geo experiment estimates the selected exact terminal contrast.
    # These are independent Gaussian contrast estimates, not raw geographic revenue levels.
    rng = random.Random(SEED + 1)
    geo = [rng.gauss(increment(PROGRAMS[selected]), 0.8) for _ in range(16)]
    observed = statistics.mean(geo)
    target = result.experiment_quantity("terminal_mean")
    ledger = EvidenceLedger()
    dependency = report["dependency"]
    model = ledger.add_source(
        SourceDeclaration(
            dependency, "fitted_model", [("panel", str(i)) for i in range(288)]
        )
    )
    experiment = ledger.add_source(
        SourceDeclaration(
            "geo_experiment", "experiment", [("geo", str(i)) for i in range(16)]
        )
    )
    mapping = ledger.add_source(
        SourceDeclaration(
            "exact_program_mapping", "fitted_model", [("mapping", "declaration")]
        )
    )
    updated = result.condition_experiment(
        bridge=bridge_for(target),
        ledger=ledger,
        observed_mean=observed,
        observation_variance=0.8**2,
        sample_size=len(geo),
        mapping_dependency=dependency,
        intercepts=[0.0] * len(draws),
        slopes=[1.0] * len(draws),
        model=model,
        experiment=experiment,
        mapping=mapping,
        execution=EXECUTION,
    )
    # An identity bridge for the terminal week cannot authorize an earlier window.
    try:
        result.condition_experiment(
            bridge=bridge_for(target),
            ledger=ledger,
            observed_mean=observed,
            observation_variance=0.8**2,
            sample_size=len(geo),
            mapping_dependency=dependency,
            intercepts=[0.0] * len(draws),
            slopes=[1.0] * len(draws),
            model=model,
            experiment=experiment,
            mapping=mapping,
            functional={"window_average": {"from": 0, "until": 3}},
            execution=EXECUTION,
        )
    except ValueError as error:
        bridge_refusal = str(error)
    else:
        raise AssertionError("A terminal-week bridge must refuse a changed estimand")
    posterior = updated.report(ledger)
    weights = posterior["posterior_weights"]
    variance = 0.8**2 / len(geo)
    anchor = min(contrasts, key=lambda x: abs(x - observed))
    raw = [
        math.exp((x - anchor) * (2 * observed - x - anchor) / (2 * variance))
        for x in contrasts
    ]
    normalizer = math.fsum(raw)
    assert all(
        abs(w - r / normalizer) < 1e-12 for w, r in zip(weights, raw, strict=True)
    )
    before = statistics.mean(contrasts)
    after = math.fsum(w * x for w, x in zip(weights, contrasts, strict=True))
    assert abs(after - observed) < abs(before - observed)
    assert posterior["scope"].split(";")[0] == "model_conditional_experimental_update"

    # Outcome = terminal revenue index per market; utility = contribution index per market.
    # All candidates cost 144 spend units over the horizon. Price converts them to this
    # terminal-outcome decision's units; it is not a claim about cumulative revenue.
    margin, price = 0.4, 0.009
    cost = 144 * price
    coordinates = tuple(
        ScientificQuantity.from_wire(q) for q in posterior["quantities"]
    )
    contract = decision.Contract(
        actions=(
            decision.Action(
                "hold", inputs=coordinates, utility=0 * decision.x(0), kind="policy"
            ),
            decision.Action(
                selected,
                inputs=coordinates,
                utility=margin * (decision.x(1) - decision.x(0)) - cost,
                kind="policy",
            ),
        ),
        utility_units="contribution_index",
        criterion=decision.Criterion.expected_utility(),
        target_population=target.population_id,
        horizon=coordinates[0].horizon,
    )
    chosen = updated.decide(contract, ledger)
    # A higher implementation price makes the revenue winner unattractive to the business.
    expensive = replace(
        contract,
        actions=(
            contract.actions[0],
            replace(
                contract.actions[1],
                utility=margin * (decision.x(1) - decision.x(0)) - 144 * 0.02,
            ),
        ),
    )
    assert updated.decide(expensive, ledger)["verdict"] == {"uniquely_optimal": "hold"}
    by_id = {a["id"]: a for a in chosen["actions"]}
    assert math.isclose(
        by_id[selected]["expected_utility"], margin * after - cost, abs_tol=1e-9
    )
    actions = [
        {"id": "hold", "intercept": 0.0, "coefficients": [0.0, 0.0]},
        {"id": selected, "intercept": -cost, "coefficients": [-margin, margin]},
    ]
    ranking = updated.rank_next_samples(
        actions=actions,
        candidates=[
            {
                "id": "repeat_geo",
                "sample_size": 16,
                "observation_variance": 0.64,
                "cost_utility": 0.001,
            },
            {
                "id": "costly_repeat",
                "sample_size": 16,
                "observation_variance": 0.64,
                "cost_utility": 0.05,
            },
        ],
        ledger=ledger,
        replicates=128,
        execution=EXECUTION,
    )["candidates"]
    assert ranking[0]["id"] == "repeat_geo"
    assert ranking[0]["gross_evsi"] == ranking[1]["gross_evsi"]
    for row, study_cost in zip(ranking, (0.001, 0.05), strict=True):
        assert math.isclose(
            row["net_value"], row["gross_evsi"] - study_cost, abs_tol=1e-12
        )
    assert (
        ranking[0]["admissible_action_identity"]
        == ranking[1]["admissible_action_identity"]
    )

    reference, quadrature_error = study_value(
        contrasts, weights, margin, cost, math.sqrt(0.64 / 16)
    )
    for row in ranking:
        assert (
            abs(row["gross_evsi"] - reference)
            <= quadrature_error + 6 * row["standard_error"] + 1e-9
        )

    # Utility revisions and evidence updates have reused the original fit.
    assert prepared.inspect()["fits_performed"] == 1
    fitted_artifact = bytes(result.export(execution=EXECUTION))
    updated_artifact = bytes(updated.export(ledger, execution=EXECUTION))
    updated.verify_current(updated_artifact, ledger, execution=EXECUTION)
    require_success(
        ledger.revise_source(
            require_success(
                SourceRevision.withdraw(experiment, "geo audit withdrew contrast")
            )
        )
    )
    try:
        updated.decide(contract, ledger)
    except ValueError as error:
        assert any(
            word in str(error) for word in ("withdrawn", "missing", "superseded")
        )
        withdrawal = str(error)
    else:
        raise AssertionError(
            "Current decision must depend on the experimental evidence"
        )
    replay = require_success(jp.consume(fitted_artifact, execution=EXECUTION))
    assert replay["point"] == report["point"] and not replay["grants_current_authority"]
    return {
        "selected_program": selected,
        "candidate_terminal_means": {name: value for value, name, _ in candidates},
        "fit_dependency": dependency,
        "population": target.population_id,
        "horizon": target.horizon,
        "functional": target.functional_id,
        "uncertainty": contrast["uncertainty"],
        "prior_posterior_sd": statistics.stdev(contrasts),
        "updated_posterior_sd": math.sqrt(
            math.fsum(
                w * (c - after) ** 2 for w, c in zip(weights, contrasts, strict=True)
            )
        ),
        "evsi_quadrature": reference,
        "evsi_quadrature_error_bound": quadrature_error,
        "known_terminal_contrast": increment(PROGRAMS[selected]),
        "fitted_terminal_contrast": contrast["point"],
        "prior_mean": before,
        "experimental_mean": observed,
        "posterior_mean": after,
        "prior_business_value": margin * before - cost,
        "posterior_business_value": margin * after - cost,
        "decision": chosen["verdict"],
        "next_studies": ranking,
        "bridge_refusal": bridge_refusal,
        "reuse_receipt": prepared.inspect()["fits_performed"],
        "withdrawal_refusal": withdrawal,
        "elapsed_seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
