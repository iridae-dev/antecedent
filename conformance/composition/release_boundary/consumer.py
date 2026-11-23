#!/usr/bin/env python3
"""Fresh-process consumer of the release-boundary bundle.

    python consumer.py <bundle-directory>

It holds no producer object: it reads ``manifest.json`` (retained identities, declared studies,
the decision declaration) and the artifact files the producer sealed, consumes every artifact
against its retained identity, re-derives the pool from the declared studies, re-evaluates the
decision from the consumed mean, and compares each number with a closed form written in this file.
The oracle never calls the library.

The result is one JSON line: ``{"ok": true, "report": {...}}`` or
``{"ok": false, "refusal": {"stage", "code", "detail"}}`` for the first typed refusal.
"""

from __future__ import annotations

import json
import pathlib
import sys

from antecedent import composition_verification as cv
from antecedent import decision
from antecedent import evidence_ledger as el
from antecedent import interface_contract as ic
from antecedent import measurement_models as mm
from antecedent import mechanism_composition as mc
from antecedent.errors import StructuredRefusal
from antecedent.joint_distribution import ScientificQuantity
from antecedent.outcomes import require_success

EXACT = 1e-12
LOOSE = 1e-9


class Refused(Exception):
    def __init__(self, stage: str, error: StructuredRefusal) -> None:
        super().__init__(str(error))
        self.stage = stage
        self.error = error


def excess_quantity(units: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id="excess_risk",
        variable_name="attributable excess risk",
        role="outcome",
        units=units,
        population_id="pop",
        regime_id="observational",
        horizon=0,
        functional_id="attributable_excess_risk",
    )


def prevalence_quantity() -> ScientificQuantity:
    return ScientificQuantity(
        variable_id="construct:exposure",
        variable_name="exposure prevalence",
        role="covariate",
        units="proportion",
        population_id="pop",
        regime_id="observational",
        horizon=0,
        functional_id="mean",
    )


def closed_forms(scenario: dict, studies: list[dict]) -> dict:
    """The oracle: Rogan-Gladen, the inverse-variance pool and the utility, by hand."""
    m = scenario["measurement"]
    observed = m["events"] / m["total"]
    # p* = Se pi + (1 - Sp)(1 - pi)  =>  pi = (p* + Sp - 1) / (Se + Sp - 1)
    prevalence = (observed + m["specificity"] - 1.0) / (m["sensitivity"] + m["specificity"] - 1.0)
    weights = [1.0 / s["variance"] for s in studies]
    pooled = sum(w * s["estimate"] for w, s in zip(weights, studies, strict=True)) / sum(weights)
    # Y | X ~ Bernoulli(q0 + d X): P(Y=1) - P(Y=1 | do(X=0)) = pi * d.
    excess = prevalence * pooled
    d = scenario["decision"]
    return {
        "prevalence": prevalence,
        "pooled": pooled,
        "pooled_variance": 1.0 / sum(weights),
        "excess": excess,
        "act": d["benefit_per_unit_risk"] * excess - d["programme_cost"],
    }


def consume_bundle(directory: pathlib.Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    scenario = manifest["scenario"]
    studies = manifest["studies"]
    truth = closed_forms(scenario, studies)
    blob = lambda name: (directory / name).read_bytes()  # noqa: E731

    # The evidence state replays under the identity the consumer retained.
    try:
        ledger = require_success(
            el.consume_evidence_ledger_artifact(blob("ledger.bin"), manifest["ledger_identity"])
        )
    except StructuredRefusal as error:
        raise Refused("ledger", error) from error
    synthesis = ledger.synthesis_of(manifest["pooled"])
    assert synthesis is not None, "the pooled result is carried"
    pooled, variance = synthesis.values[0], synthesis.values[1]

    # The pool re-derived from the declared studies: every observation counts once.
    fresh = el.EvidenceLedger()
    summaries = []
    try:
        for s in studies:
            source = fresh.add_source(
                el.SourceDeclaration(
                    s["label"],
                    "observation_set",
                    observations=el.rows("cohort", range(s["rows"][0], s["rows"][1])),
                )
            )
            summaries.append(el.StudySummary(source, s["estimate"], s["variance"]))
        rederived = require_success(fresh.pool_summaries("rederived", summaries))
    except StructuredRefusal as error:
        raise Refused("evidence", error) from error

    # The corrected prevalence, recomputed from its artifact.
    try:
        measurement = require_success(
            mm.consume_measurement_model_artifact(
                blob("measurement.bin"), manifest["measurement_identity"]
            )
        )
    except StructuredRefusal as error:
        raise Refused("measurement", error) from error
    prevalence = measurement.point

    # The interventional risks, recomputed from the composition artifacts.
    risks = {}
    for name, digest in (
        ("exposed", manifest["exposed_digest"]),
        ("unexposed", manifest["unexposed_digest"]),
    ):
        try:
            consumed = require_success(
                mc.consume_mechanism_composition_artifact(blob(f"{name}.bin"), digest)
            )
        except StructuredRefusal as error:
            raise Refused("mechanisms", error) from error
        risks[name] = dict(consumed.result.probabilities)["1"]
    excess = prevalence * (risks["exposed"] - risks["unexposed"])

    # The interfaces connect before anything is believed.
    options = manifest["interface"]
    model = manifest["measurement_identity"]
    measured_models = () if options["omit_measurement_model"] else (model,)
    law = options["decision_requires_law"]
    components = [
        ic.InterfaceContract(
            "measurement",
            produces=[
                ic.ProducedPort(
                    "prevalence",
                    prevalence_quantity(),
                    "estimand:exposure_prevalence",
                    ic.SupportDomain.real()
                    if options["producer_support_real"]
                    else ic.SupportDomain.interval(0.0, 1.0),
                    operations=("point", "mean"),
                    measurement_models=measured_models,
                )
            ],
        ),
        ic.InterfaceContract(
            "mechanisms",
            consumes=[
                ic.ConsumedPort(
                    "prevalence",
                    prevalence_quantity(),
                    "estimand:exposure_prevalence",
                    ic.SupportDomain.interval(0.0, 1.0),
                    operations=("mean",),
                    measurement_models=(model,),
                )
            ],
            produces=[
                ic.ProducedPort(
                    "excess_risk",
                    excess_quantity("risk"),
                    "estimand:attributable_excess_risk",
                    ic.SupportDomain.interval(-1.0, 1.0),
                    operations=("mean",),
                )
            ],
        ),
        ic.InterfaceContract(
            "decision",
            consumes=[
                ic.ConsumedPort(
                    "excess_risk",
                    excess_quantity("risk"),
                    "estimand:attributable_excess_risk",
                    ic.SupportDomain.interval(-1.0, 1.0),
                    operations=("law",) if law else ("mean",),
                    uncertainty="interventional_predictive" if law else None,
                )
            ],
        ),
    ]
    try:
        ic.check_composition(
            components,
            [
                ("measurement", "prevalence", "mechanisms", "prevalence"),
                ("mechanisms", "excess_risk", "decision", "excess_risk"),
            ],
        )
    except StructuredRefusal as error:
        raise Refused("interface", error) from error

    # The decision is re-evaluated from the consumed mean under the consumer's declaration.
    d = manifest["decision"]
    quantity = excess_quantity("risk")
    contract = decision.Contract(
        actions=(
            decision.Action(
                "act", inputs=(quantity,), utility=decision.x(0) * d["benefit"] - d["cost"]
            ),
            decision.Action("hold", inputs=(quantity,), utility=decision.x(0) * 0.0),
        ),
        utility_units="cases_averted_per_1000_net_of_cost",
        criterion=(
            decision.Criterion.threshold_probability(0.0)
            if d["needs_law"]
            else decision.Criterion.expected_utility()
        ),
        target_population="pop",
    )
    source = decision.MeanSource(
        coordinates=(excess_quantity(d["mean_units"]),),
        means=(excess,),
        provider_id="release_boundary:consumer",
        snapshot_id=ledger.state_digest,
        causal_contract_id="consumer",
    )
    try:
        result = require_success(contract.evaluate(source))
    except StructuredRefusal as error:
        raise Refused("decision", error) from error
    values = {a.id: a.expected_utility for a in result.actions}

    # The claim graph replays; it is valid at the evidence state consumed.
    try:
        verification = require_success(
            cv.consume_composition_verification_artifact(
                blob("verification.bin"), manifest["verification_identity"]
            )
        )
        verification.require_valid("decision")
        graph_valid = True
    except StructuredRefusal as error:
        raise Refused("claim_graph", error) from error
    snapshot_bound = (
        ledger.state_digest == manifest["snapshot_id"]
        and verification.ledger_state == ledger.state_digest
    )

    # Closed-form comparison at predeclared tolerances.
    checks = {
        "prevalence": (prevalence, truth["prevalence"], EXACT),
        "pooled": (pooled, truth["pooled"], EXACT),
        "rederived pool": (rederived.value.estimate, truth["pooled"], EXACT),
        "pooled variance": (variance, truth["pooled_variance"], EXACT),
        "excess risk": (excess, truth["excess"], LOOSE),
        "act": (values["act"], truth["act"], LOOSE),
        "reported act": (manifest["reported"]["act"], truth["act"], LOOSE),
    }
    for name, (got, want, tolerance) in checks.items():
        assert abs(got - want) <= tolerance, f"{name}: {got} vs {want}"
    return {
        "prevalence": prevalence,
        "pooled_effect": pooled,
        "pooled_variance": variance,
        "exposed_risk": risks["exposed"],
        "unexposed_risk": risks["unexposed"],
        "excess_risk": excess,
        "act": values["act"],
        "hold": values["hold"],
        "selected": result.selected[0],
        "snapshot_bound": snapshot_bound,
        "claim_graph_valid": graph_valid,
    }


def main(argv: list[str]) -> int:
    try:
        report = consume_bundle(pathlib.Path(argv[1]))
    except Refused as refused:
        stage = refused.stage
        error = refused.error
        payload = {
            "ok": False,
            "refusal": {"stage": stage, "code": error.reason_code, "detail": error.detail},
        }
    else:
        payload = {"ok": True, "report": report}
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
