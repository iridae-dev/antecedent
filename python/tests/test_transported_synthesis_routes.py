"""Borrowing and a transported prior over an evidence ledger, as transported responses.

Reference and tolerance. Both routes are conjugate-normal and have closed forms the test writes
out by hand, independent of the library. A native claim ``y = 1, v = 1`` with a transported
claim carries its declared transport variance ``t`` added once to its sampling variance.

* Borrowing with a point-mass between-study standard deviation of zero and a flat mean prior is
  inverse-variance pooling of the inflated variances ``(1, 1 + 0.5, 1 + 0.5)`` over estimates
  ``(1, 2, 3)``: mean ``(1 + 2/1.5 + 3/1.5) / (1 + 1/1.5 + 1/1.5) = 13/7`` and variance
  ``1 / (1 + 2/1.5) = 3/7``.
* A transported prior ``N(2, 1 + 0.5)`` against the native likelihood ``N(1, 1)`` has posterior
  precision ``1 + 1/1.5`` and mean ``(1 + 2/1.5) / (1 + 1/1.5) = 1.4`` with variance ``0.6``.

Closed-form arithmetic in double precision; ``1e-12`` is the tolerance. Shared transports and
priors built from the likelihood's own rows are refused before any number, and no route offers
an interval.
"""

from __future__ import annotations

import pytest
from antecedent import evidence_ledger as el
from antecedent import transported_constructors as tc

from test_transported_constructors import SELECTION, TOL, is_digest, quantity, refused

OTHER_SELECTION = "02" * 32


def transport(selection: str = SELECTION) -> el.TransportIdentity:
    return el.TransportIdentity(
        source_populations=("trial_a",),
        target_population="registry",
        selection_diagram=selection,
        assumptions=("outcome mechanism invariant",),
    )


def native(ledger, label, start, stop):
    return ledger.add_source(
        el.SourceDeclaration(
            label, "observation_set", observations=el.rows("registry", range(start, stop))
        )
    )


def carried(ledger, label, kind, start, stop, selection=SELECTION):
    return ledger.add_source(
        el.SourceDeclaration(
            label,
            kind,
            observations=el.rows("registry", range(start, stop)),
            transport=transport(selection),
        )
    )


def build(ledger, contributions, method):
    return tc.from_synthesis(
        ledger,
        "synthesis",
        contributions,
        method,
        coordinate=quantity("pooled", "registry"),
        provider_id="synthesis-provider",
        native_population="registry",
    )


def test_borrowing_inflates_each_transported_variance_and_pools_to_the_closed_form():
    ledger = el.EvidenceLedger()
    cohort = native(ledger, "registry cohort", 0, 100)
    first = carried(ledger, "claim one", "external_claim", 1_000, 1_100, SELECTION)
    second = carried(ledger, "claim two", "external_claim", 2_000, 2_100, OTHER_SELECTION)
    before = ledger.state_digest
    method = tc.Borrow(el.MeanPrior.flat(), el.TauPrior((0.0,), (1.0,)))
    built = build(
        ledger,
        [
            tc.Contribution(cohort, 1.0, 1.0),
            tc.Contribution(first, 2.0, 1.0, transport_variance=0.5),
            tc.Contribution(second, 3.0, 1.0, transport_variance=0.5),
        ],
        method,
    )
    assert built.response.means == pytest.approx((13.0 / 7.0,), abs=TOL)
    assert built.evaluation["variance"] == pytest.approx(3.0 / 7.0, abs=TOL)
    assert built.evaluation["claim_class"] == "point_only"
    receipt = built.response.receipt
    assert receipt.route == "synthesis"
    assert receipt.source_populations == ("trial_a",)
    assert receipt.target_population == "registry"
    assert len(receipt.transport_identities) == 2
    assert all(is_digest(digest) for digest in receipt.transport_identities)
    assert receipt.uncalibrated_standard_error == pytest.approx((3.0 / 7.0) ** 0.5, abs=TOL)
    assert ledger.state_digest == before

    # A positive between-study spread lets the studies disagree: the shrinkage is toward the
    # common mean, so the variance of the common effect cannot be smaller than with tau = 0.
    spread = build(
        ledger,
        [
            tc.Contribution(cohort, 1.0, 1.0),
            tc.Contribution(first, 2.0, 1.0, transport_variance=0.5),
            tc.Contribution(second, 3.0, 1.0, transport_variance=0.5),
        ],
        tc.Borrow(el.MeanPrior.flat(), el.TauPrior((1.0,), (1.0,))),
    )
    assert spread.evaluation["variance"] > built.evaluation["variance"]


def test_two_claims_through_one_transport_cannot_be_borrowed_twice():
    ledger = el.EvidenceLedger()
    cohort = native(ledger, "registry cohort", 0, 100)
    one = carried(ledger, "claim one", "external_claim", 1_000, 1_100)
    two = carried(ledger, "claim two", "external_claim", 2_000, 2_100)  # same transport
    error = refused(
        lambda: build(
            ledger,
            [
                tc.Contribution(cohort, 1.0, 1.0),
                tc.Contribution(one, 2.0, 1.0, transport_variance=0.5),
                tc.Contribution(two, 3.0, 1.0, transport_variance=0.5),
            ],
            tc.Borrow(el.MeanPrior.flat(), el.TauPrior((0.0,), (1.0,))),
        )
    )
    assert error.reason_code == "joint_law_required"


def test_a_transported_claim_is_the_prior_of_the_native_likelihood():
    ledger = el.EvidenceLedger()
    cohort = native(ledger, "registry cohort", 0, 100)
    prior = carried(ledger, "transported prior", "prior", 1_000, 1_100)
    before = ledger.state_digest
    built = build(
        ledger,
        [
            tc.Contribution(cohort, 1.0, 1.0),
            tc.Contribution(prior, 2.0, 1.0, transport_variance=0.5),
        ],
        tc.TransportedPrior(prior),
    )
    assert built.response.means == pytest.approx((1.4,), abs=TOL)
    assert built.evaluation["variance"] == pytest.approx(0.6, abs=TOL)
    assert built.evaluation["claim_class"] == "point_only"
    receipt = built.response.receipt
    assert receipt.route == "synthesis"
    assert receipt.source_populations == ("trial_a",)
    assert receipt.uncalibrated_standard_error == pytest.approx(0.6**0.5, abs=TOL)
    assert ledger.state_digest == before


def test_the_prior_route_refuses_a_native_prior_and_a_prior_built_from_the_likelihood_rows():
    ledger = el.EvidenceLedger()
    cohort = native(ledger, "registry cohort", 0, 100)
    other = native(ledger, "second registry", 500, 600)
    refused(
        lambda: build(
            ledger,
            [tc.Contribution(cohort, 1.0, 1.0), tc.Contribution(other, 2.0, 1.0)],
            tc.TransportedPrior(other),
        )
    )
    reused = carried(ledger, "prior on registry rows", "prior", 40, 140)
    error = refused(
        lambda: build(
            ledger,
            [
                tc.Contribution(cohort, 1.0, 1.0),
                tc.Contribution(reused, 2.0, 1.0, transport_variance=0.5),
            ],
            tc.TransportedPrior(reused),
        )
    )
    assert error.reason_code == "evidence_double_counted"
