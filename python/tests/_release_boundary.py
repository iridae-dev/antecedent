"""The release-boundary scenario, built only from public Python APIs.

A binary exposure with a declared confusion model is corrected by the measurement-model route and
enters a finite causal model; two study summaries are pooled through the evidence ledger; the
licensed mean of the attributable excess risk feeds an affine decision through a caller-declared
mean source; a claim graph revalidates what a revision of the evidence touches.

Shared by ``test_composition_release_fixture.py`` and the exit-gate story. The declared inputs and
the hand-derived values are ``conformance/composition/release_boundary/expected.json``; the
consumer that reads a written bundle in a fresh process is
``conformance/composition/release_boundary/consumer.py`` and shares no code with this module.
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace
from typing import Any

from antecedent import composition_verification as cv
from antecedent import decision
from antecedent import evidence_ledger as el
from antecedent import interface_contract as ic
from antecedent import measurement_models as mm
from antecedent import mechanism_composition as mc
from antecedent.joint_distribution import ScientificQuantity

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "conformance" / "composition" / "release_boundary"
FIXTURE: dict[str, Any] = json.loads((FIXTURE_DIR / "expected.json").read_text(encoding="utf-8"))
SCENARIO: dict[str, Any] = FIXTURE["scenario"]
CONSUMER = FIXTURE_DIR / "consumer.py"

INDICATOR = "schema:exposure_recorded"
CONSTRUCT = "construct:exposure"
SCALAR = mm.RepresentationVersion("scalar", 1)
EXACT = FIXTURE["tolerance"]["exact"]
DECISION = FIXTURE["tolerance"]["decision"]


# --------------------------------------------------------------------------- evidence ledger


def study(label: str, domain: str, rows: tuple[int, int] | list[int]) -> el.SourceDeclaration:
    return el.SourceDeclaration(
        label, "observation_set", observations=el.rows(domain, range(rows[0], rows[1]))
    )


def pool(
    ledger: el.EvidenceLedger,
    label: str,
    summaries: list[tuple[str, float, float]],
    *,
    covariance: list[float] | None = None,
    replaces: str | None = None,
) -> SimpleNamespace:
    outcome = ledger.pool_summaries(
        label,
        [el.StudySummary(source, estimate, variance) for source, estimate, variance in summaries],
        covariance=covariance,
        replaces=replaces,
    )
    assert isinstance(outcome.value, el.FixedEffect)
    return SimpleNamespace(
        result=outcome.result,
        estimate=outcome.value.estimate,
        variance=outcome.value.variance,
        declared_slots=outcome.mass.declared_slots,
        distinct_observations=outcome.mass.distinct_observations,
        head=ledger.head,
    )


# --------------------------------------------------------------------------- measurement


def measurement_model(sc: dict[str, Any] = SCENARIO) -> mm.MeasurementModel:
    m = sc["measurement"]
    rates = mm.BinaryRates(
        m["sensitivity"], m["specificity"], m["sensitivity_variance"], m["specificity_variance"]
    )
    return mm.MeasurementModel(
        indicator=mm.MeasurementVariable(INDICATOR, "noisy_misclassified", "indicator"),
        construct=mm.MeasurementVariable(CONSTRUCT, "latent_scientific", "indicator"),
        parameters=mm.BinaryParameters((rates,), min_youden=m["min_youden"]),
        population_id="pop",
        regime_id="observational",
        evidence=mm.ParameterEvidence.declared("release_fixture"),
        representation=SCALAR,
        model_id="exposure-misclassification",
    )


def measure(sc: dict[str, Any] = SCENARIO, construct: str = CONSTRUCT) -> SimpleNamespace:
    m = sc["measurement"]
    binding = mm.AnalysisBinding(
        construct_id=construct,
        population_id="pop",
        regime_id="observational",
        representation=SCALAR,
    )
    prepared = mm.prepare_measurement_model(
        measurement_model(sc), binding, mm.BinaryPrevalenceData(m["events"], m["total"])
    )
    estimate = prepared.estimate()
    return SimpleNamespace(
        point=estimate.point,
        naive=estimate.naive_point,
        standard_error=estimate.standard_error,
        artifact=estimate.export("release-boundary-measurement"),
        identity=prepared.model_identity,
    )


# --------------------------------------------------------------------------- mechanisms

_ROLES = {"X": "treatment", "Y": "outcome"}


def _quantity(variable: str) -> mc.Quantity:
    return mc.Quantity(
        variable,
        units="category",
        population_id="pop",
        role=_ROLES[variable],
        variable_name=f"label of {variable}",
    )


def _table(variable: str, tag: str, parents: list[str], rows: list[tuple[tuple[str, ...], tuple[float, float]]]):
    declaration = mc.MechanismDeclaration(
        output=_quantity(variable),
        parents=tuple(_quantity(p) for p in parents),
        artifact_id=f"declared:{variable}:{tag}",
        evidence_id=f"evidence:{variable}:{tag}",
    )
    return mc.Mechanism.declared_table(
        declaration,
        levels=["0", "1"],
        rows=[(dict(zip(parents, config, strict=True)), {"0": p0, "1": p1}) for config, (p0, p1) in rows],
    )


def compose(prevalence: float, baseline: float, effect: float) -> mc.PreparedMechanismComposition:
    """``X ~ Bernoulli(prevalence)``; ``Y | X ~ Bernoulli(baseline + effect * X)``."""
    exposed = baseline + effect
    return mc.compose_mechanisms(
        [
            _table("X", repr(prevalence), [], [((), (1.0 - prevalence, prevalence))]),
            _table(
                "Y", f"{baseline!r}:{effect!r}", ["X"], [(("0",), (1.0 - baseline, baseline)), (("1",), (1.0 - exposed, exposed))]
            ),
        ],
        target_population_id="pop",
        dependence=None,
        admit_declared_mechanisms=True,
    )


def outcome_risk(composition: mc.PreparedMechanismComposition, do: dict[str, str] | None = None) -> float:
    return composition.finite_law(do or {}).probability({"Y": "1"})


def excess_risk(composition: mc.PreparedMechanismComposition) -> float:
    """``P(Y = 1) - P(Y = 1 | do(X = 0))``."""
    return outcome_risk(composition) - outcome_risk(composition, {"X": "0"})


# --------------------------------------------------------------------------- decision


def excess_quantity(units: str = "risk") -> ScientificQuantity:
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


def declaration(sc: dict[str, Any] = SCENARIO, **overrides: Any) -> dict[str, Any]:
    fields = {
        "benefit": sc["decision"]["benefit_per_unit_risk"],
        "cost": sc["decision"]["programme_cost"],
        "mean_units": "risk",
        "needs_law": False,
    }
    fields.update(overrides)
    return fields


def contract(d: dict[str, Any]) -> decision.Contract:
    quantity = excess_quantity("risk")
    return decision.Contract(
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


def decide(d: dict[str, Any], mean: float, snapshot: str, plan: str) -> decision.Decision:
    """Evaluate through a caller-declared mean source built from the licensed mean."""
    source = decision.MeanSource(
        coordinates=(excess_quantity(d["mean_units"]),),
        means=(mean,),
        provider_id="release_boundary:composed_finite_model",
        snapshot_id=snapshot,
        causal_contract_id=plan,
    )
    return contract(d).evaluate(source)


def value_of(result: decision.Decision, action: str) -> float:
    return next(a.expected_utility for a in result.actions if a.id == action)


# --------------------------------------------------------------------------- interfaces


def _prevalence_quantity() -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=CONSTRUCT,
        variable_name="exposure prevalence",
        role="covariate",
        units="proportion",
        population_id="pop",
        regime_id="observational",
        horizon=0,
        functional_id="mean",
    )


def interface_components(
    model_identity: str,
    *,
    producer_support_real: bool = False,
    omit_measurement_model: bool = False,
    decision_requires_law: bool = False,
) -> tuple[list[ic.InterfaceContract], list[tuple[str, str, str, str]]]:
    measurement = ic.InterfaceContract(
        "measurement",
        produces=[
            ic.ProducedPort(
                "prevalence",
                _prevalence_quantity(),
                "estimand:exposure_prevalence",
                ic.SupportDomain.real() if producer_support_real else ic.SupportDomain.interval(0.0, 1.0),
                operations=("point", "mean"),
                measurement_models=() if omit_measurement_model else (model_identity,),
            )
        ],
    )
    mechanisms = ic.InterfaceContract(
        "mechanisms",
        consumes=[
            ic.ConsumedPort(
                "prevalence",
                _prevalence_quantity(),
                "estimand:exposure_prevalence",
                ic.SupportDomain.interval(0.0, 1.0),
                operations=("mean",),
                measurement_models=(model_identity,),
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
    )
    wanted = ic.ConsumedPort(
        "excess_risk",
        excess_quantity("risk"),
        "estimand:attributable_excess_risk",
        ic.SupportDomain.interval(-1.0, 1.0),
        operations=("law",) if decision_requires_law else ("mean",),
        uncertainty="interventional_predictive" if decision_requires_law else None,
    )
    consumer = ic.InterfaceContract("decision", consumes=[wanted])
    connections = [
        ("measurement", "prevalence", "mechanisms", "prevalence"),
        ("mechanisms", "excess_risk", "decision", "excess_risk"),
    ]
    return [measurement, mechanisms, consumer], connections


# --------------------------------------------------------------------------- claim graph

D = cv.digest


def claim_graph(
    *,
    survey: str,
    study_a: str,
    study_b: str,
    measurement_identity: str,
    prevalence: float,
    pooled_head: str,
    plan_identity: str,
    excess: float,
    contract_identity: str,
    act: float,
) -> cv.CompositionGraph:
    nodes = [
        cv.Node("survey", "cohort_source", identity=D("survey", survey), evidence=(survey,)),
        cv.Node("study_a", "cohort_source", identity=D("study", study_a), evidence=(study_a,)),
        cv.Node("study_b", "cohort_source", identity=D("study", study_b), evidence=(study_b,)),
        cv.Node(
            "measurement",
            "measurement_model",
            identity=D("measurement", measurement_identity, repr(prevalence)),
            representation="scalar/1",
            assumptions=("non-differential misclassification",),
        ),
        cv.Node("pooled_effect", "ledger_result", identity=D("pool", pooled_head)),
        cv.Node(
            "mechanisms",
            "mechanism_composition",
            identity=D("mechanisms", plan_identity),
            assumptions=("declared finite mechanisms",),
        ),
        cv.Node("claim", "claim", identity=D("claim", plan_identity, repr(excess))),
        cv.Node("decision", "decision", identity=D("decision", contract_identity, repr(act))),
    ]
    edges = [
        cv.Edge("measurement", "survey", "data"),
        cv.Edge("pooled_effect", "study_a", "evidence"),
        cv.Edge("pooled_effect", "study_b", "evidence"),
        cv.Edge("mechanisms", "measurement", "model_version"),
        cv.Edge("mechanisms", "pooled_effect", "data"),
        cv.Edge("claim", "mechanisms", "data"),
        cv.Edge("decision", "claim", "data"),
    ]
    return cv.CompositionGraph(nodes, edges)


def verify(
    previous: cv.CompositionGraph | None,
    current: cv.CompositionGraph,
    ledger: el.EvidenceLedger,
    pooled: str,
) -> cv.CompositionVerification:
    effects = cv.ledger_effects(ledger, {"pooled_effect": pooled})
    return cv.verify_composition(current, previous=previous, effects=effects, ledger=ledger)


# --------------------------------------------------------------------------- the pipeline


class World:
    """The producer's objects after one full run of the scenario."""

    def __init__(
        self,
        sc: dict[str, Any],
        ledger: el.EvidenceLedger,
        sources: tuple[str, str, str],
        pooled: SimpleNamespace,
        measured: SimpleNamespace,
    ) -> None:
        self.sc = sc
        self.ledger = ledger
        self.survey, self.study_a, self.study_b = sources
        self.pooled = pooled
        self.measured = measured
        self.composition = compose(measured.point, sc["baseline_risk"], pooled.estimate)
        self.plan_identity = self.composition.plan_identity
        self.excess_risk = excess_risk(self.composition)
        self.declaration = declaration(sc)
        self.result = decide(self.declaration, self.excess_risk, ledger.state_digest, self.plan_identity)
        self.graph = claim_graph(
            survey=self.survey,
            study_a=self.study_a,
            study_b=self.study_b,
            measurement_identity=measured.identity,
            prevalence=measured.point,
            pooled_head=pooled.head,
            plan_identity=self.plan_identity,
            excess=self.excess_risk,
            contract_identity=contract(self.declaration).identity,
            act=value_of(self.result, "act"),
        )

    @classmethod
    def build(cls, sc: dict[str, Any] = SCENARIO) -> World:
        ledger = el.EvidenceLedger()
        survey = ledger.add_source(study("survey", "survey", sc["survey_rows"]))
        a_decl, b_decl = sc["studies"]
        a = ledger.add_source(study(a_decl["label"], "cohort", a_decl["rows"]))
        b = ledger.add_source(study(b_decl["label"], "cohort", b_decl["rows"]))
        pooled = pool(
            ledger,
            "pooled_effect",
            [(a, a_decl["estimate"], a_decl["variance"]), (b, b_decl["estimate"], b_decl["variance"])],
        )
        return cls(sc, ledger, (survey, a, b), pooled, measure(sc))

    def gate(self) -> cv.CompositionVerification:
        """The verification a decision is released under: raises when the claim is not usable."""
        verification = verify(self.graph, self.graph, self.ledger, self.pooled.result)
        verification.require_valid("decision")
        return verification

    def revise(self) -> SimpleNamespace:
        """Study B grows; returns the successor source and the revised summary."""
        revision = self.sc["revision"]
        report = self.ledger.revise_source(
            el.SourceRevision.update(
                self.study_b, study("study_b_grown", "cohort", revision["rows"])
            )
        )
        assert report.created_source is not None
        return SimpleNamespace(
            grown=report.created_source,
            summary=(revision["estimate"], revision["variance"]),
            report=report,
        )

    def downstream_of(self, pooled: SimpleNamespace, study_b: str) -> World:
        """A world whose pooled result was recomputed: the same ledger, measurement and
        survey; everything downstream is rebuilt."""
        return World(
            self.sc,
            self.ledger,
            (self.survey, self.study_a, study_b),
            pooled,
            self.measured,
        )

    def studies(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self.sc["studies"]]

    def write_bundle(self, directory: pathlib.Path, studies: list[dict[str, Any]] | None = None) -> None:
        """Seal the producer's artifacts, retained identities and declarations in ``directory``.

        The consumer reads only these files.
        """
        directory.mkdir(parents=True, exist_ok=True)
        exported = self.ledger.export("release-boundary-ledger")
        exposed = self.composition.export({"X": "1"}, outcome="Y")
        unexposed = self.composition.export({"X": "0"}, outcome="Y")
        verification = verify(self.graph, self.graph, self.ledger, self.pooled.result).export(
            "release-boundary-verification"
        )
        files = {
            "ledger.bin": exported.artifact,
            "measurement.bin": self.measured.artifact,
            "exposed.bin": exposed.artifact,
            "unexposed.bin": unexposed.artifact,
            "verification.bin": verification.artifact,
        }
        for name, data in files.items():
            (directory / name).write_bytes(data)
        manifest = {
            "scenario": self.sc,
            "files": sorted(files),
            "ledger_identity": exported.identity,
            "pooled": self.pooled.result,
            "measurement_identity": self.measured.identity,
            "exposed_digest": exposed.premises_digest,
            "unexposed_digest": unexposed.premises_digest,
            "verification_identity": verification.identity,
            "snapshot_id": self.ledger.state_digest,
            "reported": {
                "act": value_of(self.result, "act"),
                "hold": value_of(self.result, "hold"),
                "selected": self.result.selected[0],
            },
            "studies": studies if studies is not None else self.studies(),
            "decision": self.declaration,
            "interface": {
                "producer_support_real": False,
                "omit_measurement_model": False,
                "decision_requires_law": False,
            },
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def revised_studies(sc: dict[str, Any] = SCENARIO) -> list[dict[str, Any]]:
    """The declared studies after the second one grew."""
    first, _ = (dict(s) for s in sc["studies"])
    revision = sc["revision"]
    return [
        first,
        {
            "label": "study_b_grown",
            "rows": revision["rows"],
            "estimate": revision["estimate"],
            "variance": revision["variance"],
        },
    ]
