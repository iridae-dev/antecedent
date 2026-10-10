"""X6: the per-proposal receipt linking identification repair and design ranking.

Oracles (derived by hand, as in ``crates/antecedent-design/tests/proposal_receipt.rs``):

* Back-door contract: ``z1`` and ``z2`` each confound ``t -> y``, so the only admissible
  adjustment set is ``{z1, z2}`` and only one observational joint law over ``(t, y, z1, z2)``
  repairs it; a study of ``(t, y, z1)`` alone stays insufficient.
* Transport contract: ``x -> y`` with a selection mechanism on ``x``; the source experiment
  ``do(x)`` measuring ``y`` repairs it, a source observation does not.
* Valuation: binary state, guess it for utility 1, prior 1/2. A signal of accuracy ``a`` has
  ``EVSI = a - 1/2``: 3/4 gives 1/4 and 5/8 gives 1/8. With cost 0.01 utility per USD the
  study of 20 USD has net value ``1/4 - 1/5 = 1/20`` and the study of 5 USD has net value
  ``1/8 - 1/20 = 3/40``, so the cheaper study ranks first.
"""

import copy

import pytest
from antecedent import Admg, Dag, repair
from antecedent import design as dr
from antecedent import proposals as px
from antecedent.errors import CausalTypeError, CausalUnsupportedError
from antecedent.joint_distribution import ScientificQuantity
from antecedent.transport import advanced as transport

from _refusal import assert_registered_refusal

NAMES = ["t", "y", "z1", "z2", "w1", "w2", "w3"]
EDGES = [("z1", "t"), ("z1", "y"), ("z2", "t"), ("z2", "y"), ("t", "y")]
SAMPLES = 500
DEFAULTS = {
    "sample_size": SAMPLES,
    "recruitment": "consecutive patients",
    "timing": "baseline",
    "unit": "patient",
    "cost_unit": "USD",
}


def backdoor():
    return repair.BackdoorContract(
        graph=Dag.from_edges(NAMES, EDGES),
        treatment="t",
        outcome="y",
        population="clinic",
        observed=["t", "y"],
    )


def observation(label, measured, cost, *, population="clinic"):
    return repair.StudyCandidate.observation(
        label, population=population, measured=measured, cost=cost, **DEFAULTS
    )


def narrow():
    return observation("narrow", ["t", "y", "z1"], 5)


def cohort():
    return observation("cohort", ["t", "y", "z1", "z2"], 20)


def transport_contract():
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in ("x", "y"))
    catalog = transport.EvidenceCatalog(
        environments=(
            transport.Environment("source", coordinates, selection_targets=("x",)),
            transport.Environment("target", coordinates),
        )
    )
    return repair.TransportContract(
        graph=Admg.from_edges(["x", "y"], [("x", "y")]),
        selections=["x"],
        source="source",
        target="target",
        outcomes=["y"],
        treatments=["x"],
        catalog=catalog,
    )


def experiment():
    return repair.StudyCandidate.experiment(
        "trial",
        population="source",
        interventions=["x"],
        measured=["y"],
        cost=3,
        **DEFAULTS,
    )


def source_observation():
    return observation("obs", ["y"], 1, population="source")


# --------------------------------------------------------------------- the ranking


def quantity(name):
    return ScientificQuantity(
        variable_id=f"schema:{name}",
        variable_name=name,
        role="outcome",
        units="dimensionless",
        population_id="target",
        regime_id="observational",
        horizon=0,
        functional_id="state",
    )


GUESS = dr.DesignDecision(
    contract="contract-1",
    actions=(dr.ActionUtility("guess0", 1.0, -1.0), dr.ActionUtility("guess1", 0.0, 1.0)),
    prior=dr.StatePrior.draws([0.0, 1.0]),
    utility_units="utility",
)
USD_MAP = dr.CostMap("USD", "utility", 0.01)


def signal():
    return dr.SignalSpec(
        prior_id="prior-1",
        state=quantity("state"),
        observation=quantity("signal"),
        evidence_lineage=("snapshot:a",),
        rng_seed=3,
    )


def semantic_ids(result):
    """Repair semantic id of every declared candidate, by label."""
    return {row.labels[0]: row.candidates[0] for row in result.table if len(row.labels) == 1}


def accuracy_signal(label, accuracy):
    law = dr.ExternalLaw.posterior(
        states=[0.0, 1.0],
        statistics=[0.0, 1.0],
        predictive=[0.5, 0.5],
        posterior=[[accuracy, 1.0 - accuracy], [1.0 - accuracy, accuracy]],
    )
    return dr.ExternalSignal("lab", f"signal-{label}", "v1", "snap", "lab-qa", law)


def rank(result, studies, accuracies, *, cost_of=None, ids=None):
    """Rank the repair's candidates by EVSI, valued under USD -> utility."""
    ids = ids or semantic_ids(result)
    candidates = [
        dr.Candidate(
            ids[study.label],
            study.sample_size,
            accuracy_signal(study.label, accuracies[study.label]),
            cost=float(cost_of(study) if cost_of else study.cost),
            cost_unit=study.cost_unit,
        )
        for study in studies
    ]
    return dr.rank_designs(
        candidates,
        decision=GUESS,
        signal=signal(),
        cost_map=USD_MAP,
        source_digests=("digest-b", "digest-a"),
    )


class Fixture:
    """A back-door repair of ``narrow`` (insufficient) and ``cohort`` (repairs it)."""

    def __init__(self):
        self.contract = backdoor()
        self.studies = [narrow(), cohort()]
        self.repaired = repair.repair(self.contract, self.studies)
        self.ids = semantic_ids(self.repaired)
        self.ranked = rank(self.repaired, self.studies, {"narrow": 0.625, "cohort": 0.75})
        self.repair_bytes = self.repaired.export()
        self.ranking_bytes = self.ranked.export()
        self.bundle = px.ProposalBundle.build(
            self.repair_bytes, self.ranking_bytes, contract=self.contract
        )

    @property
    def narrow_id(self):
        return self.ids["narrow"]

    @property
    def cohort_id(self):
        return self.ids["cohort"]


@pytest.fixture(scope="module")
def fx():
    return Fixture()


def refused(callable_, detail):
    with pytest.raises(px.ProposalRefusal) as caught:
        callable_()
    assert caught.value.detail == detail, caught.value.detail
    assert isinstance(caught.value, CausalUnsupportedError)
    assert_registered_refusal(caught.value)
    return caught.value


# --------------------------------------------------------------------------- receipts


def test_x6_receipt_binds_failure_delta_derivation_cost_lineage_value_and_decision(fx):
    assert fx.repaired.classification("cohort") == "verified_sufficient"
    assert fx.repaired.classification("narrow") == "insufficient"
    assert fx.bundle.candidate_ids == tuple(sorted(fx.ids.values()))
    wide = fx.bundle.proposal(fx.cohort_id)
    # Base failure: the one joint-law obligation of the frozen contract.
    assert wide.base_failure.family == "backdoor"
    assert wide.base_failure.contract == fx.repaired.contract_id
    assert wide.base_failure.obligation_ids == tuple(o.id for o in fx.repaired.obligations)
    assert len(wide.base_failure.obligation_ids) == 1
    assert wide.base_failure.premises_digest and wide.base_failure.data_digest
    # Hypothetical delta and its verified derivation.
    assert wide.hypothetical.classification == "verified_sufficient"
    assert wide.hypothetical.delta_regimes == 1
    assert wide.hypothetical.derivation_digest is not None
    assert wide.hypothetical.derivation_verified
    assert wide.hypothetical.addressed == wide.base_failure.obligation_ids
    # Cost, size and lineage.
    assert (wide.cost.units, wide.cost.unit_label) == (20, "USD")
    assert (wide.cost.sample_budget, wide.cost.sample_size) == (SAMPLES, SAMPLES)
    assert "snapshot:a" in wide.lineage.snapshots
    assert "provider:lab/snap" in wide.lineage.snapshots
    # Valuation: net 1/4 - 20 * 0.01 = 1/20, ranked second behind the cheaper study.
    assert wide.valuation.basis == "net_value"
    assert wide.valuation.evsi == pytest.approx(0.25, abs=1e-12)
    assert wide.valuation.net_value == pytest.approx(0.05, abs=1e-12)
    assert wide.valuation.rank == 1
    cheap = fx.bundle.proposal(fx.narrow_id)
    assert cheap.valuation.rank == 0
    assert cheap.valuation.net_value == pytest.approx(0.075, abs=1e-12)
    assert cheap.hypothetical.classification == "insufficient"
    assert cheap.hypothetical.derivation_digest is None
    assert cheap.valuation.signal_identity != wide.valuation.signal_identity
    # Decision receipt.
    assert wide.decision.contract_identity == "contract-1"
    assert wide.decision.utility_unit == "utility"
    assert wide.decision.ranking_identity == fx.ranked.ranking_identity
    assert wide.ranking_digest == fx.ranked.identity
    assert wide.identity and wide.identity != cheap.identity
    assert fx.bundle.verify(fx.repair_bytes, fx.ranking_bytes) is None


def test_x6_receipt_never_marks_a_hypothetical_derivation_as_available(fx):
    wide = fx.bundle.proposal(fx.cohort_id)
    # The derivation is verified, yet the receipt holds no available evidence.
    assert wide.hypothetical.derivation_verified
    assert wide.evidence_state == "hypothetical"
    assert not wide.is_available_evidence
    assert all(p.evidence_state == "hypothetical" for p in fx.bundle.proposals)
    # Presenting a proposed (or merely manipulable) regime as an arrival is refused.
    proposed = px.ArrivedEvidence.regimes(
        [px.ArrivedRegime("clinic", ["t", "y", "z1", "z2"], evidence_kind="proposed")],
        snapshot_id="snapshot:copy",
        sample_size=SAMPLES,
    )
    error = refused(
        lambda: fx.bundle.on_arrival(cohort(), proposed),
        "proposal_receipt.hypothetical_not_evidence",
    )
    assert error.reason_code == "transport_missing_evidence"
    manipulable = px.ArrivedEvidence.regimes(
        [px.ArrivedRegime("clinic", ["t", "y", "z1", "z2"], evidence_kind="manipulable")],
        snapshot_id="snapshot:copy",
        sample_size=SAMPLES,
    )
    refused(
        lambda: fx.bundle.on_arrival(cohort(), manipulable),
        "proposal_receipt.hypothetical_not_evidence",
    )


def test_x6_receipt_bundle_identity_does_not_depend_on_candidate_order(fx):
    assert list(fx.bundle.candidate_ids) == sorted(fx.bundle.candidate_ids)
    reversed_ranking = rank(
        fx.repaired, list(reversed(fx.studies)), {"narrow": 0.625, "cohort": 0.75}
    )
    assert reversed_ranking.identity == fx.ranked.identity
    rebuilt = px.ProposalBundle.build(fx.repair_bytes, reversed_ranking.export())
    assert rebuilt.identity == fx.bundle.identity
    assert rebuilt.to_dict() == fx.bundle.to_dict()
    # A stored reorder does not move the identity; the verifier refuses it instead.
    data = copy.deepcopy(fx.bundle.to_dict())
    data["proposals"].reverse()
    shuffled = px.ProposalBundle.from_dict(data)
    assert shuffled.identity == fx.bundle.identity
    refused(
        lambda: shuffled.verify(fx.repair_bytes, fx.ranking_bytes, consume=False),
        "proposal_receipt.non_canonical_order",
    )


def test_x6_receipt_round_trips_through_plain_data(fx):
    data = fx.bundle.to_dict()
    again = px.ProposalBundle.from_dict(data)
    assert again.identity == fx.bundle.identity and again.proposals == fx.bundle.proposals
    assert again.verify(fx.repair_bytes, fx.ranking_bytes) is None


# --------------------------------------------------------------------------- mutations

MUTATIONS = [
    (
        "obligation id",
        lambda r: r["base_failure"]["obligation_ids"].append("extra"),
        "obligation_mismatch",
    ),
    ("family", lambda r: r["base_failure"].update(family="transport"), "base_failure_mismatch"),
    ("contract", lambda r: r["base_failure"].update(contract="other"), "base_failure_mismatch"),
    (
        "premises digest",
        lambda r: r["base_failure"].update(premises_digest="x"),
        "base_failure_mismatch",
    ),
    ("delta digest", lambda r: r["hypothetical"].update(delta_digest="x"), "delta_mismatch"),
    (
        "derivation digest",
        lambda r: r["hypothetical"].update(derivation_digest="x"),
        "derivation_mismatch",
    ),
    (
        "classification",
        lambda r: r["hypothetical"].update(classification="not_certified"),
        "derivation_mismatch",
    ),
    ("cost units", lambda r: r["cost"].update(units=r["cost"]["units"] + 1), "cost_mismatch"),
    ("cost label", lambda r: r["cost"].update(unit_label="EUR"), "cost_mismatch"),
    (
        "sample size",
        lambda r: r["cost"].update(sample_size=r["cost"]["sample_size"] + 1),
        "cost_mismatch",
    ),
    ("source digest", lambda r: r["lineage"].update(source_digest="x"), "source_digest_mismatch"),
    (
        "snapshot lineage",
        lambda r: r["lineage"]["snapshots"].append("snapshot:forged"),
        "lineage_mismatch",
    ),
    (
        "request fingerprint",
        lambda r: r["valuation"].update(signal_request_fingerprint="x"),
        "signal_mismatch",
    ),
    ("signal identity", lambda r: r["valuation"].update(signal_identity="x"), "signal_mismatch"),
    ("net value", lambda r: r["valuation"].update(net_value_bits=1), "value_mismatch"),
    (
        "evsi",
        lambda r: r["valuation"].update(evsi_bits=r["valuation"]["evsi_bits"] ^ 1),
        "value_mismatch",
    ),
    ("rank", lambda r: r["valuation"].update(rank=r["valuation"]["rank"] + 1), "value_mismatch"),
    (
        "decision contract",
        lambda r: r["decision"].update(contract_identity="contract-2"),
        "contract_mismatch",
    ),
    ("action set", lambda r: r["decision"].update(action_ids_digest="x"), "contract_mismatch"),
    ("ranking identity", lambda r: r["decision"].update(ranking_identity="x"), "contract_mismatch"),
    ("repair digest", lambda r: r.update(repair_report_digest="x"), "repair_digest_mismatch"),
    ("ranking digest", lambda r: r.update(ranking_digest="x"), "ranking_digest_mismatch"),
    ("receipt identity", lambda r: r.update(identity="forged"), "identity_mismatch"),
]


@pytest.mark.parametrize(("name", "mutate", "detail"), MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_x6_receipt_every_bound_field_refuses_when_mutated(fx, name, mutate, detail):
    data = copy.deepcopy(fx.bundle.to_dict())
    receipt = next(p for p in data["proposals"] if p["candidate_id"] == fx.cohort_id)
    mutate(receipt)
    mutated = px.ProposalBundle.from_dict(data)
    refused(
        lambda: mutated.verify(fx.repair_bytes, fx.ranking_bytes, consume=False),
        f"proposal_receipt.{detail}",
    )


def test_x6_receipt_bundle_level_mutations_refuse(fx):
    for field, value, detail in (
        ("identity", "forged", "bundle_identity_mismatch"),
        ("repair_report_digest", "x", "repair_digest_mismatch"),
        ("ranking_digest", "x", "ranking_digest_mismatch"),
        ("decision_contract_identity", "contract-2", "contract_mismatch"),
    ):
        data = copy.deepcopy(fx.bundle.to_dict())
        data[field] = value
        mutated = px.ProposalBundle.from_dict(data)
        refused(
            lambda m=mutated: m.verify(fx.repair_bytes, fx.ranking_bytes, consume=False),
            f"proposal_receipt.{detail}",
        )
    # A bundle that drops a ranked candidate no longer covers the ranking.
    data = copy.deepcopy(fx.bundle.to_dict())
    data["proposals"].pop()
    refused(
        lambda: px.ProposalBundle.from_dict(data).verify(
            fx.repair_bytes, fx.ranking_bytes, consume=False
        ),
        "proposal_receipt.candidate_set_mismatch",
    )
    # A proposal with an unknown candidate id.
    data = copy.deepcopy(fx.bundle.to_dict())
    data["proposals"][0]["candidate_id"] = "sc1:unknown"
    refused(
        lambda: px.ProposalBundle.from_dict(data).verify(
            fx.repair_bytes, fx.ranking_bytes, consume=False
        ),
        "proposal_receipt.candidate_not_in_ranking",
    )


# ------------------------------------------------------------- cross-artifact checks


def test_x6_receipt_a_different_ranking_is_not_the_one_the_bundle_binds(fx):
    other = rank(fx.repaired, fx.studies, {"narrow": 0.625, "cohort": 0.875})
    assert other.identity != fx.ranked.identity
    refused(
        lambda: fx.bundle.verify(fx.repair_bytes, other.export()),
        "proposal_receipt.ranking_digest_mismatch",
    )


def test_x6_receipt_candidate_in_repair_but_not_in_ranking_is_refused(fx):
    only_cheap = rank(fx.repaired, [fx.studies[0]], {"narrow": 0.625}, ids=fx.ids)
    # The repair evaluated both candidates; a ranking of one values one, so a bundle of both
    # does not cover it.
    refused(
        lambda: fx.bundle.verify(fx.repair_bytes, only_cheap.export()),
        "proposal_receipt.candidate_not_in_ranking",
    )
    single = px.ProposalBundle.build(fx.repair_bytes, only_cheap.export())
    assert single.candidate_ids == (fx.narrow_id,)


def test_x6_receipt_candidate_in_ranking_but_not_in_repair_is_refused(fx):
    ids = {**fx.ids, "cohort": "sc1:never-repaired"}
    stranger = rank(fx.repaired, fx.studies, {"narrow": 0.625, "cohort": 0.75}, ids=ids)
    refused(
        lambda: px.ProposalBundle.build(fx.repair_bytes, stranger.export()),
        "proposal_receipt.candidate_not_in_repair",
    )


def test_x6_receipt_a_ranking_that_costs_a_candidate_differently_is_refused(fx):
    dearer = rank(
        fx.repaired,
        fx.studies,
        {"narrow": 0.625, "cohort": 0.75},
        cost_of=lambda s: s.cost + 1 if s.label == "narrow" else s.cost,
    )
    refused(
        lambda: px.ProposalBundle.build(fx.repair_bytes, dearer.export()),
        "proposal_receipt.cost_mismatch",
    )


def test_x6_receipt_a_foreign_repair_artifact_is_refused(fx):
    other = repair.repair(
        backdoor(), [narrow(), observation("wider", ["t", "y", "z1", "z2", "w1"], 30)]
    )
    # Neither ranked candidate is declared in the other repair report.
    refused(
        lambda: px.ProposalBundle.build(other.export(), fx.ranking_bytes),
        "proposal_receipt.candidate_not_in_repair",
    )


def test_x6_receipt_bundle_inputs_must_be_bytes(fx):
    with pytest.raises(CausalTypeError):
        px.ProposalBundle.build("repair", fx.ranking_bytes)
    with pytest.raises(CausalTypeError):
        fx.bundle.verify(fx.repair_bytes, "ranking")


# --------------------------------------------------------------------------- arrival


def law(measured, *, population="clinic", joint=True, sample_size=SAMPLES):
    return px.ArrivedEvidence.law(
        population,
        measured,
        snapshot_id="snapshot:delivered",
        sample_size=sample_size,
        joint=joint,
    )


def test_x6_receipt_arrival_of_the_exact_proposed_law_is_verified_by_the_family_checker(fx):
    verdict = fx.bundle.on_arrival(cohort(), law(["t", "y", "z1", "z2"]))
    assert isinstance(verdict, px.Verified)
    assert verdict.checker == "backdoor.adjustment"
    assert verdict.steps[1] == "adjustment_set:[2,3]"
    assert verdict.exact
    assert verdict.snapshot_id == "snapshot:delivered"
    # The contract may be supplied at the call instead of at build.
    bare = px.ProposalBundle.build(fx.repair_bytes, fx.ranking_bytes)
    assert isinstance(
        bare.on_arrival(cohort(), law(["t", "y", "z1", "z2"]), contract=fx.contract), px.Verified
    )


def test_x6_receipt_arrival_missing_the_joint_law_is_still_insufficient(fx):
    # The right variables in the right population, but as separate marginals.
    marginals = fx.bundle.on_arrival(cohort(), law(["t", "y", "z1", "z2"], joint=False))
    assert isinstance(marginals, px.StillInsufficient) and marginals.reasons
    # A joint law that lacks z2 leaves the second back-door path open.
    narrower = fx.bundle.on_arrival(cohort(), law(["t", "y", "z1"]))
    assert isinstance(narrower, px.StillInsufficient) and narrower.reasons


def test_x6_receipt_arrival_in_a_different_population_invalidates_the_derivation(fx):
    verdict = fx.bundle.on_arrival(cohort(), law(["t", "y", "z1", "z2"], population="elsewhere"))
    assert verdict == px.Invalidated("population", "elsewhere", proposed_populations=("clinic",))


def test_x6_receipt_arrival_that_differs_in_sample_or_shape_is_refused_with_a_typed_reason(fx):
    refused(
        lambda: fx.bundle.on_arrival(
            cohort(), law(["t", "y", "z1", "z2"], sample_size=SAMPLES - 1)
        ),
        "proposal_receipt.arrival_sample_mismatch",
    )
    refused(
        lambda: fx.bundle.on_arrival(
            cohort(),
            px.ArrivedEvidence.law(
                "clinic", ["t", "y", "z1", "z2"], snapshot_id="  ", sample_size=SAMPLES
            ),
        ),
        "proposal_receipt.arrival_invalid",
    )
    refused(
        lambda: fx.bundle.on_arrival(
            cohort(),
            px.ArrivedEvidence.regimes([], snapshot_id="snapshot:none", sample_size=SAMPLES),
        ),
        "proposal_receipt.arrival_empty",
    )
    # A receipt answers only the candidate it was issued for.
    stranger = observation("stranger", ["t", "y", "z1", "z2", "w1"], 40)
    refused(
        lambda: fx.bundle.on_arrival(stranger, law(["t", "y", "z1", "z2"])),
        "proposal_receipt.candidate_not_in_bundle",
    )
    # A bundle without the failed contract cannot re-identify anything.
    bare = px.ProposalBundle.build(fx.repair_bytes, fx.ranking_bytes)
    refused(
        lambda: bare.on_arrival(cohort(), law(["t", "y", "z1", "z2"])),
        "proposal_receipt.contract_required",
    )
    # An observed law is a back-door delivery, never a transport one.
    transport_fx = TransportFixture()
    refused(
        lambda: transport_fx.bundle.on_arrival(
            experiment(),
            px.ArrivedEvidence.law("source", ["x", "y"], snapshot_id="s", sample_size=SAMPLES),
        ),
        "proposal_receipt.arrival_family_mismatch",
    )


class TransportFixture:
    """A transport repair of ``trial`` (repairs it) and ``obs`` (does not)."""

    def __init__(self):
        self.contract = transport_contract()
        self.studies = [experiment(), source_observation()]
        self.repaired = repair.repair(self.contract, self.studies)
        self.ranked = rank(self.repaired, self.studies, {"trial": 0.75, "obs": 0.625})
        self.bundle = px.ProposalBundle.build(
            self.repaired.export(), self.ranked.export(), contract=self.contract
        )


def test_x6_receipt_transport_arrival_of_the_exact_source_experiment_is_verified():
    fixture = TransportFixture()
    assert fixture.repaired.classification("trial") == "verified_sufficient"
    assert fixture.repaired.classification("obs") == "insufficient"
    delivered = px.ArrivedEvidence.regimes(
        [px.ArrivedRegime("source", ["y"], interventions=["x"])],
        snapshot_id="snapshot:source-trial",
        sample_size=SAMPLES,
    )
    verdict = fixture.bundle.on_arrival(experiment(), delivered)
    assert isinstance(verdict, px.Verified)
    assert verdict.checker == "classical_transport.catalog"
    assert verdict.exact


def test_x6_receipt_transport_arrival_in_another_population_or_regime_is_invalidated():
    fixture = TransportFixture()

    def arrive(regime):
        return fixture.bundle.on_arrival(
            experiment(),
            px.ArrivedEvidence.regimes(
                [regime], snapshot_id="snapshot:delivered", sample_size=SAMPLES
            ),
        )

    # The experiment was run in the target environment instead of the source.
    moved = arrive(px.ArrivedRegime("target", ["y"], interventions=["x"]))
    assert moved == px.Invalidated("population", "target", proposed_populations=("source",))
    # The right population under another intervention set: an observation, not do(x).
    observed = arrive(px.ArrivedRegime("source", ["y"]))
    assert observed == px.Invalidated("regime", "source", arrived_interventions=())
    # Another intervention set in the right population names the variables it intervened on.
    other = arrive(px.ArrivedRegime("source", ["x"], interventions=["y"]))
    assert other == px.Invalidated("regime", "source", arrived_interventions=("y",))
