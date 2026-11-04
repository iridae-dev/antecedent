"""A declared independence between mechanism estimates, decided from the evidence ledger's rows.

Reference. Independence of mechanism estimates is a fact about the observations they were fit on:
two estimates are independent only if no observation underlies both. The reference is therefore
plain set arithmetic on the row ranges registered in the ledger, written in this test: ranges
``[0, 50)``, ``[50, 100)`` and ``[100, 150)`` share nothing, while ``[25, 75)`` shares the
``25`` rows ``[25, 50)`` with the first and ``25`` rows ``[50, 75)`` with the second. No
tolerance is involved: overlap is a count, and the refusal must appear exactly when the count is
positive, even though every evidence identity string is different (the string comparison alone
would accept all four compositions). A source with no computable observations has unknown overlap
and is never read as independent.
"""

from __future__ import annotations

import pytest
from antecedent import _reason_details as reason_details
from antecedent import evidence_ledger as el
from antecedent import mechanism_composition as mc
from antecedent.errors import CausalTypeError

from test_mechanism_composition import independent, quantity, refusal_of


def source(ledger, label, start, stop):
    return ledger.add_source(
        el.SourceDeclaration(
            label, "observation_set", observations=el.rows("study", range(start, stop))
        )
    )


def mechanism(variable, evidence, intercept, sigma, parents=()):
    names = [name for name, _ in parents]
    return mc.Mechanism.declared_linear(
        mc.MechanismDeclaration(
            output=quantity(variable),
            parents=tuple(quantity(name) for name in names),
            artifact_id=f"fit:{variable}",
            evidence_id=evidence,
        ),
        intercept=intercept,
        sigma=sigma,
        coefficients=dict(parents),
    )


def chain(x_evidence, m_evidence, y_evidence):
    return [
        mechanism("X", x_evidence, 1.0, 2.0),
        mechanism("M", m_evidence, 0.5, 1.0, [("X", 2.0)]),
        mechanism("Y", y_evidence, -1.0, 0.5, [("X", 0.5), ("M", 3.0)]),
    ]


def compose(mechanisms, ledger=None):
    return mc.compose_mechanisms(
        mechanisms,
        target_population_id="pop",
        dependence=independent(),
        admit_declared_mechanisms=True,
        ledger=ledger,
    )


def test_disjoint_rows_are_independent_and_the_composition_prepares():
    ledger = el.EvidenceLedger()
    x = source(ledger, "x study", 0, 50)
    m = source(ledger, "m study", 50, 100)
    y = source(ledger, "y study", 100, 150)
    prepared = compose(chain(x, m, y), ledger)
    assert prepared.variables == ("X", "M", "Y")
    # The ledger check adds no scientific number: the plan is the one an unchecked call makes.
    assert compose(chain(x, m, y)).plan_identity == prepared.plan_identity


def test_overlapping_rows_refute_the_independence_though_every_identity_differs():
    ledger = el.EvidenceLedger()
    x = source(ledger, "x study", 0, 50)
    m = source(ledger, "m study overlapping both neighbours", 25, 75)
    y = source(ledger, "y study", 50, 100)
    assert len({x, m, y}) == 3
    # The string comparison alone accepts this composition ...
    assert compose(chain(x, m, y)).variables == ("X", "M", "Y")
    # ... the ledger does not: X and M share 25 rows, M and Y share 25 rows.
    error = refusal_of(lambda: compose(chain(x, m, y), ledger))
    assert error.reason_code == "joint_law_required"
    shared = reason_details.mechanism_composition.POSTERIOR_DEPENDENCE_SHARED_EVIDENCE_DECLARED_INDEPENDENT
    assert error.details == (shared, shared)
    assert {item["offending"] for item in error.refusals} == {"X|M", "M|Y"}
    assert error.remedy


def test_a_source_whose_rows_are_not_computable_is_unknown_never_independent():
    ledger = el.EvidenceLedger()
    x = source(ledger, "x study", 0, 50)
    y = source(ledger, "y study", 100, 150)
    vendor = ledger.add_source(el.SourceDeclaration("vendor claim", "external_claim"))
    error = refusal_of(lambda: compose(chain(x, vendor, y), ledger))
    assert error.details
    assert set(error.details) == {
        reason_details.mechanism_composition.POSTERIOR_DEPENDENCE_EVIDENCE_OVERLAP_UNKNOWN
    }
    assert {item["offending"] for item in error.refusals} == {"X|M", "M|Y"}


def test_the_ledger_must_be_a_ledger():
    with pytest.raises(CausalTypeError, match="EvidenceLedger"):
        compose(chain("a" * 64, "b" * 64, "c" * 64), ledger=object())
