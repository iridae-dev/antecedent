# SPDX-License-Identifier: MIT OR Apache-2.0
"""Use agents to plan human evidence, then branch on agreement or conflict.

The finite hypothesis model, binomial updates and checkpoint controller are application
code. Antecedent composes declared priors, checks mechanism transport, evaluates the launch
contract, checks identification repair and ranks externally attested future signals.
Nothing here establishes agent/human equivalence.
"""

from __future__ import annotations

import itertools
import json
import math
import time

import numpy as np
from antecedent import Admg, decision, design, priors, repair
from antecedent import typed_mechanism_transport as tmt
from antecedent.evidence_ledger import EvidenceLedger, Lineage, SourceDeclaration
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success

STAGES = ("navigation", "discovery", "checkout")
POPULATION = "recent_humans_at_launch"
CHECKPOINTS = (8, 16, 32, 64)
MINIMUM_PILOT = 16
EXPOSURE_PRICE = 0.009  # decision units per human shown any confidential content
AUTH_EXPOSURE_PRICE = 0.001  # restricted authentication-only content
FAILURE_PENALTY = 19.0  # launch utility +1 if requirements hold, -19 otherwise
FLOOR = (
    0.82  # business floor for human checkout-task completion, fixed before observations
)
RESOLUTION = (
    4096  # deterministic rounding of weighted beliefs for the equal-draw ranker
)
# Only the generator and independent truth checks read this object.
TRUTH = {
    "human_old": (0.78, 0.76, 0.80, 0.65),
    "agent_old": (0.78, 0.76, 0.90, 0.95),
    "agent_new": (0.88, 0.86, 0.98, 0.97),
    "agreement": (0.88, 0.86, 0.88, 0.75),
    "conflict": (0.88, 0.86, 0.55, 0.35),
    "auth_bypass_checkout": 0.88,
}


def traces(probabilities, n, seed):
    """One independent user per row; all three microtasks are attempted, not a funnel."""
    rng = np.random.default_rng(seed)
    return {
        stage: rng.binomial(1, p, n).astype(float)
        for stage, p in zip((*STAGES, "auth_recovery"), probabilities, strict=True)
    }


def hidden_checkout(world, bypass, uniforms):
    """Generator-only potential outcome with a shared uniform for each human."""
    base = TRUTH[world][2]
    probability = (
        TRUTH["auth_bypass_checkout"] if world == "conflict" and bypass else base
    )
    return np.asarray(uniforms) < probability


def normalize(values):
    total = math.fsum(values)
    assert total > 0
    return np.array(values, dtype=float) / total


def binomial(n, k, p):
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def quantity(name):
    return ScientificQuantity(
        variable_id=f"schema:{name}",
        variable_name=name,
        role="outcome",
        units="dimensionless",
        population_id=POPULATION,
        regime_id="redesign_v1",
        horizon=0,
        functional_id="state",
    )


def inspect_transport(data):
    """Old-site agreement is checked only on the old site; a new-site gap refuses."""
    names = (*STAGES, "auth_recovery")
    source = [
        tmt.fit_typed_mechanism(
            tmt.MechanismSpecification(name, family="bernoulli"),
            data["agent_old"],
            population="agents_old",
        )
        for name in names
    ]
    target = [
        fit.refit_in_target(data["human_old"], population="humans_old")
        for fit in source
    ]
    checks = {
        name: tmt.check_invariance(s, t)
        for name, s, t in zip(names, source, target, strict=True)
    }
    assert not checks["navigation"].rejected and not checks["discovery"].rejected
    assert checks["checkout"].rejected and checks["auth_recovery"].rejected
    old = tmt.transport_typed_mechanisms(
        tmt.TypedTransport(
            source_population="agents_old",
            target_population="humans_old",
            selection_nodes=(
                tmt.SelectionNode("population_shift", ("checkout", "auth_recovery")),
            ),
        ),
        source,
        target,
    )
    assert old.receipt.native_variables == ("auth_recovery", "checkout")
    assert (
        old.receipt.support == "extrapolative" and old.receipt.inference == "point_only"
    )
    new_source = [
        tmt.fit_typed_mechanism(
            tmt.MechanismSpecification(name, family="bernoulli"),
            data["agent_new"],
            population="agents_new",
        )
        for name in STAGES
    ]
    try:
        tmt.transport_typed_mechanisms(
            tmt.TypedTransport(
                source_population="agents_new",
                target_population=POPULATION,
                selection_nodes=(
                    tmt.SelectionNode("unresolved_human_checkout", ("checkout",)),
                ),
            ),
            new_source,
            [],
        )
    except tmt.TypedMechanismRefusal as refusal:
        assert refusal.reason_code == "transport_missing_evidence"
        missing = refusal.details
    else:
        raise AssertionError(
            "Shifted human checkout cannot be manufactured from agents"
        )
    return {
        "old_site_checks": {
            name: {"p_value": c.p_value, "rejected": c.rejected}
            for name, c in checks.items()
        },
        "old_site_support": old.receipt.support,
        "inference": old.receipt.inference,
        "uncertainty_scope": old.receipt.uncertainty_scope,
        "new_site_refusal": missing,
    }


def make_prior(data, ledger):
    means = {
        name: np.array([np.mean(rows[s]) for s in STAGES])
        for name, rows in data.items()
    }
    predicted = means["human_old"] + means["agent_new"] - means["agent_old"]
    sampling = sum(
        means[name] * (1 - means[name]) / len(data[name]["checkout"]) for name in data
    )
    # A declared transfer-discrepancy floor, independent of the number of agent runs.
    transfer = np.array([0.04, 0.04, 0.025]) ** 2
    variance = sampling + transfer
    baseline_mean, baseline_variance = means["human_old"], np.full(3, 0.15**2)
    source = priors.ExternalPriorSourceSpec(
        "old_humans_plus_agent_version_difference",
        mean=tuple(predicted),
        variance=tuple(variance),
    )
    arguments = dict(
        sources=[source],
        weights=(0.8,),
        baseline=(baseline_mean, baseline_variance),
        source_populations=("agents_redesign_effect",),
        target_population=POPULATION,
    )
    try:
        priors.compose_external_priors(**arguments)
    except ValueError as error:
        missing_policy = str(error)
    else:
        raise AssertionError("A cross-population prior must require an explicit policy")
    composed = priors.compose_external_priors(
        **arguments, transport=priors.TransportPolicy.invariant_effect_modifiers()
    )
    # Independent mixture-moment oracle. These moments are not a binomial posterior.
    expected_mean = 0.8 * predicted + 0.2 * baseline_mean
    expected_variance = 0.8 * (variance + (predicted - expected_mean) ** 2) + 0.2 * (
        baseline_variance + (baseline_mean - expected_mean) ** 2
    )
    assert np.allclose(composed.mean, expected_mean, atol=1e-12, rtol=0)
    assert np.allclose(composed.variance, expected_variance, atol=1e-12, rtol=0)
    assert np.all(variance >= transfer)
    # A finite, deliberately limited menu of scientific explanations. Three failing states
    # have the SAME observable stage rates and differ in their intervention predictions.
    hypotheses = (
        "transfer_holds",
        "authentication",
        "population_mix",
        "navigation_link",
    )
    grid = (np.arange(1000) + 0.5) / 1000
    groups = np.repeat(np.arange(4), len(grid))
    rates = np.tile(predicted, (4 * len(grid), 1))
    rates[:, 2] = np.tile(grid, 4)

    # The full grid retains mildly harmful and borderline outcomes. Only checkout
    # is modeled for the launch gate; the other stages remain descriptive diagnostics.
    def density(mean, var):
        log = -0.5 * (grid - mean) ** 2 / var
        return normalize(np.exp(log - max(log)))

    component = np.zeros((2, len(rates)))
    component[0, : len(grid)] = density(predicted[2], variance[2])
    for group in (1, 2, 3):
        component[1, groups == group] = (
            density(baseline_mean[2], baseline_variance[2]) / 3
        )
    joint = np.array([0.8, 0.2])[:, None] * component
    source_ids = [
        ledger.add_source(
            SourceDeclaration(
                name,
                "observation_set",
                [(name, str(i)) for i in range(len(rows["checkout"]))],
            )
        )
        for name, rows in data.items()
    ]
    prior_id = ledger.add_source(
        SourceDeclaration(
            "agent_derived_launch_prior",
            "prior",
            [],
            lineage=Lineage.derived_from_used_data(source_ids),
        )
    )
    assert len(ledger.sources) == 4
    return {
        "hypotheses": hypotheses,
        "groups": groups,
        "rates": rates,
        "joint": joint,
        "prior_id": prior_id,
        "predicted": predicted,
        "sampling_variance": sampling,
        "transfer_variance": transfer,
        "composition": composed.to_native_dict(),
        "policy_refusal": missing_policy,
        "source_ids": source_ids,
    }


def update(prior, data):
    """Bayes on the ORIGINAL mixture and each observed row once, at a fixed checkpoint."""
    n = len(data["checkout"])
    successes = np.array([sum(data[s]) for s in STAGES])
    log = np.sum(
        successes * np.log(prior["rates"])
        + (n - successes) * np.log1p(-prior["rates"]),
        axis=1,
    )
    likelihood = np.exp(log - max(log))
    joint = prior["joint"] * likelihood
    joint /= np.sum(joint)
    # Independent ordered Bernoulli-product computation, rather than binomial log counts.
    direct = prior["joint"].copy()
    for i in range(n):
        for j, stage in enumerate(STAGES):
            p = prior["rates"][:, j]
            direct *= np.where(data[stage][i] == 1, p, 1 - p)
    direct /= np.sum(direct)
    assert np.allclose(joint, direct, atol=1e-12, rtol=0)
    return joint


def launch_contract():
    q = quantity("requirements_hold")
    contract = decision.Contract(
        actions=(
            decision.Action("launch", inputs=(q,), utility=20 * decision.x(0) - 19),
            decision.Action("withhold", inputs=(q,), utility=0 * decision.x(0)),
        ),
        utility_units="launch_utility",
        criterion=decision.Criterion.expected_utility(),
        target_population=POPULATION,
    )
    return q, contract


def launch_decision(joint, prior, snapshot):
    q, contract = launch_contract()
    weights = tuple(np.sum(joint, axis=0))
    safe = (prior["rates"][:, 2] >= FLOOR).astype(float)
    belief = JointDistributionArtifact(
        DistributionIdentity(
            semantic="parameter_posterior",
            quantities=(q,),
            alignment="joint",
            source_id=prior.get("active_source", prior["prior_id"]),
            provider_id="declared_finite_launch_model",
            rng_id="deterministic_exact",
            snapshot_id=snapshot,
            causal_contract_id="historical_baseline_stability",
        ),
        safe[:, None].copy(),
        weights=weights,
    )
    result = require_success(contract.evaluate(belief))
    mass = float(np.dot(weights, safe))
    values = {a.id: a.expected_utility for a in result.actions}
    assert abs(values["launch"] - (20 * mass - 19)) < 1e-10
    assert result.selected == (("launch",) if mass > 0.95 else ("withhold",))
    return mass, result.selected, belief.trust


def equal_prior(values, weights):
    """Quantile rounding for the native equal-draw ranker; each atom mass error <=1/K."""
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    indices = np.searchsorted(cdf, (np.arange(RESOLUTION) + 0.5) / RESOLUTION)
    atoms = np.asarray(values)[indices]
    counts = np.bincount(indices, minlength=len(values)) / RESOLUTION
    assert max(abs(counts - np.array(weights))) <= 1 / RESOLUTION
    return atoms, counts


def rank_signals(values, weights, actions, candidates, *, prior_id, context):
    """Use native Bayes/integration on caller-attested finite experimental likelihoods."""
    atoms, rounded = equal_prior(values, weights)
    state, contract = (
        launch_contract() if context == "continue_pilot" else (quantity(context), None)
    )
    if contract is None:
        contract = decision.Contract(
            actions=tuple(
                decision.Action(
                    name, inputs=(state,), utility=intercept + slope * decision.x(0)
                )
                for name, intercept, slope in actions
            ),
            utility_units="launch_utility",
            criterion=decision.Criterion.expected_utility(),
            target_population=POPULATION,
        )
    declared = design.DesignDecision.from_contract(
        contract, prior=design.StatePrior.draws(atoms)
    )
    assert [(a.id, a.intercept, a.slope) for a in declared.actions] == actions
    request = design.SignalSpec(
        prior_id=prior_id,
        state=state,
        observation=quantity(context + "_observation"),
        evidence_lineage=(prior_id,),
        rng_seed=41,
        conditional_independence="count_likelihood_marginalizes_shared_completion_parameter",
        max_support=10_000,
    )
    studies, oracle, original_oracle = [], {}, {}
    for name, n, price, law in candidates:
        # law[branch,state] is computed outside Antecedent. Repeated equal-prior atoms
        # receive that state's same likelihood. Statistics encode successes or the difference of arm success counts.
        indices = [int(np.argmin(abs(np.array(values) - atom))) for atom in atoms]
        likelihood = law[:, indices]
        provider = design.ExternalSignal(
            provider_id="tutorial_finite_binomial_model",
            object_id=name,
            version_id="1",
            snapshot_id=context,
            attestor="tutorial_application",
            law=design.ExternalLaw.likelihood(atoms, list(range(len(law))), likelihood),
        )
        studies.append(
            design.Candidate(name, n, provider, cost=price, cost_unit="launch_utility")
        )
        utilities = np.array(
            [[intercept + slope * x for x in values] for _, intercept, slope in actions]
        )
        now = max(utilities @ rounded)
        # Independent enumeration of joint P(state,observation), no library posterior calls.
        future = sum(max(utilities @ (rounded * row)) for row in law)
        oracle[name] = future - now
        original_oracle[name] = sum(
            max(utilities @ (np.array(weights) * row)) for row in law
        ) - max(utilities @ weights)
    ranking = design.rank_designs(
        studies,
        decision=declared,
        signal=request,
        cost_map=design.CostMap("launch_utility", "launch_utility", 1.0),
    )
    for entry in ranking.candidates:
        assert entry.provider_trust == "externally_attested"
        assert abs(entry.evsi - oracle[entry.id]) < 1e-9
        cost = next(row[2] for row in candidates if row[0] == entry.id)
        assert abs(entry.net_value - (entry.evsi - cost)) < 1e-9
    exact_order = sorted(
        candidates, key=lambda c: original_oracle[c[0]] - c[2], reverse=True
    )
    assert [c.id for c in ranking.candidates] == [c[0] for c in exact_order]
    for entry in ranking.candidates:
        cost = next(c[2] for c in candidates if c[0] == entry.id)
        assert (entry.net_value > 0) == (original_oracle[entry.id] - cost > 0)
    replay = design.consume(ranking.export(), expected_identity=ranking.expectation())
    assert replay.identity == ranking.identity
    assert all(not entry.natively_replayed for entry in replay.entries)
    return ranking, rounded


def pilot_law(prior, weights, n):
    p = prior["rates"][:, 2]
    safe = p >= FLOOR
    mass = float(sum(weights[safe]))
    group_weights = [
        weights * (~safe) / max(1 - mass, 1e-300),
        weights * safe / max(mass, 1e-300),
    ]
    law = np.array(
        [
            [
                sum(w * math.comb(n, k) * p**k * (1 - p) ** (n - k))
                for w in group_weights
            ]
            for k in range(n + 1)
        ]
    )
    return mass, law


def pilot_information(prior, weights, n):
    mass, law = pilot_law(prior, weights, n)
    return rank_signals(
        [0.0, 1.0],
        [1 - mass, mass],
        [("launch", -19.0, 20.0), ("withhold", 0.0, 0.0)],
        [(f"another_{n}", n, n * EXPOSURE_PRICE, law)],
        prior_id=prior.get("active_source", prior["prior_id"]),
        context="continue_pilot",
    )[0]


def conflict_tail(prior, k, n):
    p = prior["rates"][:, 2]
    q = prior["joint"][0] / sum(prior["joint"][0])
    return float(
        sum(q * sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k + 1)))
    )


def run_branch(name, prior, ledger):
    # This is one realized illustrative trace, not a claim about every repeated pilot.
    seed = 16 if name == "agreement" else 91
    first = traces(TRUTH[name], 32, seed)
    later = traces(TRUTH[name], 32, seed + 100)
    data = {stage: np.concatenate([first[stage], later[stage]]) for stage in STAGES}
    history, block_sources = [], []
    previous = 0
    for n in CHECKPOINTS:
        current = {s: values[:n] for s, values in data.items()}
        block_sources.append(
            ledger.add_source(
                SourceDeclaration(
                    f"{name}_pilot_{previous}_{n}",
                    "observation_set",
                    [(name, str(i)) for i in range(previous, n)],
                )
            )
        )
        previous = n
        active_source = ledger.add_source(
            SourceDeclaration(
                f"{name}_posterior_{n}",
                "fitted_model",
                [],
                lineage=Lineage.derived_from_used_data(
                    [prior["prior_id"], *block_sources]
                ),
            )
        )
        current_prior = dict(prior, active_source=active_source)
        joint = update(prior, current)
        weights = np.sum(joint, axis=0)
        mass, selected, trust = launch_decision(joint, current_prior, f"{name}:{n}")
        next_n = (
            CHECKPOINTS[CHECKPOINTS.index(n) + 1] - n if n != CHECKPOINTS[-1] else None
        )
        information = (
            pilot_information(current_prior, weights, next_n) if next_n else None
        )
        k = int(sum(current["checkout"]))
        tail = conflict_tail(prior, k, n)
        # Conflict is diagnostic, not a sequential p-value test or a cause identifier.
        alpha_recommendation = priors.ConflictPolicy(p_min=0.01).shrink_alpha(
            1.0, p_value=tail, kl=0.0
        )
        status = "investigate"
        if (
            n >= MINIMUM_PILOT
            and mass >= 0.95
            and (information is None or information.best.net_value <= 0)
        ):
            status = "launch"
        elif mass < 0.05 and float(sum(weights[prior["groups"] == 1])) < 0.05:
            status = "do_not_launch"
        history.append(
            {
                "exposed": n,
                "successes": {s: int(sum(current[s])) for s in STAGES},
                "probability_requirements_hold": mass,
                "native_terminal_choice": selected,
                "transfer_component_mass": float(np.sum(joint[0])),
                "conflict_tail_diagnostic": tail,
                "external_alpha_recommendation": alpha_recommendation,
                "status": status,
                "next_batch_net_value": None
                if information is None
                else information.best.net_value,
            }
        )
        if status != "investigate" or alpha_recommendation == 0 or n == CHECKPOINTS[-1]:
            break
    return {
        "history": history,
        "weights": weights,
        "joint": joint,
        "pilot_source": active_source,
        "trust": trust,
        "data": current,
    }


def repair_candidates():
    # Authentication A is confounded with checkout C by unobserved error propensity U.
    # A directed edge plus A<->C blocks observational identification of do(A).
    contract = repair.HedgeContract(
        graph=Admg.from_edges(
            ["auth", "checkout"], [("auth", "checkout")], [("auth", "checkout")]
        ),
        treatment="auth",
        outcome="checkout",
        population=POPULATION,
        observed=("auth", "checkout"),
    )
    studies = []
    for label, n, intervene in (
        ("generic_32", 32, False),
        ("auth_8", 8, True),
        ("auth_32", 32, True),
    ):
        studies.append(
            repair.StudyCandidate(
                label=label,
                kind="experiment" if intervene else "observation",
                population=POPULATION,
                measured=("checkout",) if intervene else ("auth", "checkout"),
                sample_size=n,
                recruitment="new representative consenting human testers",
                timing="same launch window",
                unit="independent human",
                cost=n,
                cost_unit="confidential_exposures",
                interventions=("auth",) if intervene else (),
            )
        )
    result = repair.repair(contract, studies)
    assert result.classification("generic_32") == "insufficient"
    assert result.classification("auth_8") == "verified_sufficient"
    assert result.classification("auth_32") == "verified_sufficient"
    return result


def follow_up(prior, branch):
    weights = branch["weights"]
    p = prior["rates"][:, 2]
    bypass = np.where(prior["groups"] == 1, prior["predicted"][2], p)
    # Same latent completion model, now projected to the causal gain from a bypass.
    # Funding an auth repair is a new decision; it cannot authorize a patched launch.
    gains = bypass - p
    values, inverse = np.unique(gains, return_inverse=True)
    collapsed = np.bincount(inverse, weights=weights, minlength=len(values))

    def aggregate(likelihood):
        return np.bincount(
            inverse, weights=weights * likelihood, minlength=len(values)
        ) / np.maximum(collapsed, 1e-300)

    candidates = []
    for name, n in (("generic_32", 32), ("auth_8", 8), ("auth_32", 32)):
        if name.startswith("generic"):
            law = np.array(
                [
                    aggregate(math.comb(n, k) * p**k * (1 - p) ** (n - k))
                    for k in range(n + 1)
                ]
            )
            cost = n * EXPOSURE_PRICE
        else:
            arm = n // 2
            law = np.zeros((2 * arm + 1, len(values)))
            for control, treated in itertools.product(range(arm + 1), repeat=2):
                likelihood = (
                    math.comb(arm, control)
                    * p**control
                    * (1 - p) ** (arm - control)
                    * math.comb(arm, treated)
                    * bypass**treated
                    * (1 - bypass) ** (arm - treated)
                )
                law[treated - control + arm] += aggregate(likelihood)
            cost = n * AUTH_EXPOSURE_PRICE
        assert np.allclose(
            np.sum(law, axis=0), np.ones(len(values)), atol=1e-12, rtol=0
        )
        candidates.append((name, n, cost, law))
    ranking, rounded = rank_signals(
        values,
        collapsed,
        [("fund_auth_repair", -0.15, 1.0), ("abandon_this_redesign", 0.0, 0.0)],
        candidates,
        prior_id=branch["pilot_source"],
        context="authentication_repair",
    )
    sufficient = repair_candidates()
    eligible = [
        c
        for c in ranking.candidates
        if sufficient.classification(c.id) == "verified_sufficient"
    ]
    print(
        "Follow-up candidates:",
        [(r.id, r.evsi, r.net_value) for r in ranking.candidates],
        flush=True,
    )
    assert eligible[0].net_value > 0
    return {
        "repair_outcome": sufficient.outcome,
        "identification": {
            name: sufficient.classification(name) for name, _, _, _ in candidates
        },
        "ranking": [
            {
                "id": r.id,
                "evsi": r.evsi,
                "net_value": r.net_value,
                "trust": r.provider_trust,
            }
            for r in ranking.candidates
        ],
        "recommended": eligible[0].id,
        "belief_rounding_max_error": float(max(abs(rounded - collapsed))),
    }


def compare_human_only(prior, agreement):
    """Same contract, broad human-only belief and a predeclared 128-person comparator."""
    human_prior = dict(prior, prior_id="broad_human_only_uniform")
    human_prior["joint"] = np.zeros_like(prior["joint"])
    human_prior["joint"][0, prior["groups"] == 0] = 1 / sum(prior["groups"] == 0)
    small = agreement["data"]
    extra = traces(TRUTH["agreement"], 128 - len(small["checkout"]), 141)
    large = {stage: np.concatenate([small[stage], extra[stage]]) for stage in small}
    result = []
    for label, observations in (
        ("small_human_only", small),
        ("fixed_128_human_only", large),
    ):
        posterior = update(human_prior, observations)
        mass, selected, _ = launch_decision(posterior, human_prior, label)
        n = len(observations["checkout"])
        k = int(sum(observations["checkout"]))
        # A flat probability prior gives Beta(k+1,n-k+1). Its upper tail equals
        # this binomial CDF, an independent exact check of the numerical grid.
        exact = sum(binomial(n + 1, j, FLOOR) for j in range(k + 1))
        assert abs(mass - exact) < 1e-4
        result.append(
            {
                "analysis": label,
                "exposed": len(observations["checkout"]),
                "checkout_successes": int(sum(observations["checkout"])),
                "probability_requirements_hold": mass,
                "terminal_choice": selected,
            }
        )
    assert result[0]["terminal_choice"] == ("withhold",)
    assert result[1]["terminal_choice"] == ("launch",)
    return result


def transfer_sensitivity(prior, agreement):
    result = []
    for bias in (0.0, -0.04, -0.08):
        scenario = dict(prior)
        scenario["rates"] = prior["rates"].copy()
        selected = prior["groups"] == 0
        scenario["rates"][selected, 2] = np.clip(
            scenario["rates"][selected, 2] + bias, 0.001, 0.999
        )
        posterior = update(scenario, agreement["data"])
        mass, action, _ = launch_decision(posterior, scenario, f"transfer_bias:{bias}")
        result.append(
            {
                "unmodeled_human_shift": bias,
                "probability_requirements_hold": mass,
                "terminal_choice": action,
            }
        )
    assert result[0]["terminal_choice"] == ("launch",)
    assert any(row["terminal_choice"] == ("withhold",) for row in result[1:])
    return result


def main():
    started = time.perf_counter()
    data = {
        name: traces(TRUTH[name], 600 if name == "human_old" else 2000, seed)
        for name, seed in (("human_old", 11), ("agent_old", 12), ("agent_new", 13))
    }
    assert math.isclose(
        TRUTH["human_old"][2] + TRUTH["agent_new"][2] - TRUTH["agent_old"][2],
        TRUTH["agreement"][2],
        abs_tol=1e-12,
    )
    assert math.isclose(
        TRUTH["auth_bypass_checkout"] - TRUTH["conflict"][2], 0.33, abs_tol=1e-12
    )
    uniform_grid = (np.arange(10_000) + 0.5) / 10_000
    untreated = hidden_checkout("conflict", False, uniform_grid)
    treated = hidden_checkout("conflict", True, uniform_grid)
    assert np.mean(untreated) == 0.55 and np.mean(treated) == 0.88
    assert math.isclose(
        float(np.mean(treated.astype(float) - untreated)), 0.33, abs_tol=1e-12
    )
    ledger = EvidenceLedger()
    transport = inspect_transport(data)
    prior = make_prior(data, ledger)
    assert max(abs(prior["predicted"] - np.array(TRUTH["agreement"][:3]))) < 0.05
    print(
        "Declared launch contract:",
        f"checkout probability >= {FLOOR} with posterior probability >= .95; checkpoints {CHECKPOINTS}; full-site exposure price {EXPOSURE_PRICE}",
        flush=True,
    )
    print("Transport inspection:", json.dumps(transport, indent=2), flush=True)
    print(
        "Prior provenance and uncertainty:",
        json.dumps(
            {
                key: prior[key]
                for key in (
                    "predicted",
                    "sampling_variance",
                    "transfer_variance",
                    "composition",
                    "source_ids",
                    "prior_id",
                    "policy_refusal",
                )
            },
            default=lambda x: x.tolist(),
            indent=2,
        ),
        flush=True,
    )
    initial = [
        pilot_information(prior, np.sum(prior["joint"], axis=0), n).best
        for n in (8, 16, 32)
    ]
    initial_plan = sorted(initial, key=lambda r: r.net_value, reverse=True)
    assert initial_plan[0].sample_size == 16 and initial_plan[0].net_value > 0
    agreement = run_branch("agreement", prior, ledger)
    conflict = run_branch("conflict", prior, ledger)
    print("Agreement:", agreement["history"], flush=True)
    print("Conflict:", conflict["history"], flush=True)
    assert agreement["history"][-1]["status"] == "launch"
    assert agreement["history"][-1]["exposed"] == 32
    assert conflict["history"][-1]["status"] == "investigate"
    assert float(np.sum(conflict["joint"][0])) < float(np.sum(prior["joint"][0]))
    # Equal likelihoods do not resolve which of these causes is responsible.
    cause_masses = [sum(conflict["weights"][prior["groups"] == g]) for g in (1, 2, 3)]
    assert max(cause_masses) - min(cause_masses) < 1e-12
    comparison = compare_human_only(prior, agreement)
    sensitivity = transfer_sensitivity(prior, agreement)
    planned = follow_up(prior, conflict)
    print(
        json.dumps(
            {
                "initial_pilot_candidates": [
                    {"users": r.sample_size, "evsi": r.evsi, "net_value": r.net_value}
                    for r in initial_plan
                ],
                "human_only_comparison": comparison,
                "transfer_sensitivity": sensitivity,
                "follow_up": planned,
                "evidence_sources": len(ledger.sources),
                "model_scope": "declared finite hypothesis menu; historical-baseline stability assumed",
                "posterior_trust": agreement["trust"],
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
