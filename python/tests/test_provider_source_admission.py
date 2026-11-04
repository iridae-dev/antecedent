"""A provider's output is admitted as source evidence for a transported claim, or refused.

Reference. Admission is a decision on declarations, so each expectation is the specification
read off the declarations written in this file rather than a computed number: the provider object
is the Gaussian law of ``test_provider_callbacks`` established in population ``trial`` under
``do(a=1)`` with the operation ``mean`` among its capabilities; a source naming that coordinate,
a trust the route accepts and the assumptions it relies on is admitted; each omitted or altered
declaration (an assumption, the coordinate, a trust the route does not accept, a receipt)
refuses with the specific ``provider_transport`` detail. The reference provider offers draws and
a law, so the admitted contract is ``not_measured`` (never measured), supplies the
``external_provider`` stage and is never native target data; the identity is a function of the
declarations only (equal declarations give an equal digest, any changed request, assumption or
trust gives another). No tolerance is involved.
"""

from __future__ import annotations

import dataclasses

import pytest
from antecedent import _reason_details as reason_details
from antecedent import interface_contract as ic
from antecedent import provider_callbacks as pc
from antecedent import transport_handoff as th
from antecedent.errors import CausalTypeError, CausalValueError

from _refusal import assert_registered_refusal
from _transport_fixtures import digest
from test_provider_callbacks import declaration, gaussian_probes, identity, quantity

ASSUMPTIONS = ("recovery mechanism invariant across populations",)


def source(**overrides) -> th.ProviderSource:
    base = th.ProviderSource(
        provider=declaration(),
        component="trial_provider",
        quantity=quantity(),
        semantic_identity="estimand:outcome_law",
        support=pc.Support.real(),
        assumptions=ASSUMPTIONS,
        evidence=(digest(5),),
        attestor="lab",
    )
    return dataclasses.replace(base, **overrides)


def transport(sources=("trial", "trial_b"), assumptions=ASSUMPTIONS) -> th.TransportIdentity:
    return th.TransportIdentity(tuple(sources), "target", digest(7), tuple(assumptions))


def consumer(population="trial") -> ic.InterfaceContract:
    return ic.InterfaceContract(
        "transport_route",
        consumes=[
            ic.ConsumedPort(
                "transport_source",
                quantity(population=population),
                "estimand:outcome_law",
                ic.SupportDomain.real(),
                operations=("mean",),
                dependence="declared_evidence",
                provenance=("external_provider",),
            )
        ],
    )


def refusal_of(call) -> pc.CallbackRefusal:
    with pytest.raises(pc.CallbackRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    return caught.value


def test_an_attested_provider_output_is_admitted_as_external_source_evidence():
    admitted = th.admit_provider_source(source())
    assert admitted.trust == "externally_attested"
    assert admitted.native_target_data is False
    assert (admitted.population, admitted.regime) == ("trial", "do(a=1)")
    assert "mean" in admitted.operations
    assert len(admitted.identity) == 64
    port = admitted.contract.produces[0]
    assert port.name == "provider_source"
    assert "external_provider" in port.provenance
    # The provider offers draws and a law, so its output carries uncertainty it never measured.
    assert port.calibration.status == "not_measured"
    assert port.evidence == (digest(5),)
    # Equal declarations give an equal identity; a changed request, assumption or trust another.
    assert th.admit_provider_source(source()).identity == admitted.identity
    other_request = declaration(
        provider=pc.ProviderIdentity("lab", "gaussian", "1.0", "snap-1", "req-2")
    )
    assert th.admit_provider_source(source(provider=other_request)).identity != admitted.identity
    more = source(assumptions=(*ASSUMPTIONS, "no selection on the outcome"))
    assert th.admit_provider_source(more).identity != admitted.identity
    assert th.admit_provider_source(source(attestor="other lab")).identity != admitted.identity


def test_each_missing_or_altered_declaration_refuses_with_its_own_detail():
    detail = reason_details.provider_transport
    none = refusal_of(lambda: th.admit_provider_source(source(assumptions=())))
    assert none.detail == detail.ASSUMPTIONS_MISSING
    elsewhere = refusal_of(
        lambda: th.admit_provider_source(source(quantity=quantity(population="somewhere")))
    )
    assert elsewhere.detail == detail.QUANTITY_NOT_DECLARED
    native = refusal_of(lambda: th.admit_provider_source(source(trust="native_supported")))
    assert native.detail == detail.NATIVE_TRUST_CLAIMED
    no_attestor = refusal_of(lambda: th.admit_provider_source(source(attestor=None)))
    assert no_attestor.detail == detail.TRUST_INCOMPLETE
    # An attestation is not a verification where the route requires one, and a label is not a
    # receipt.
    required = refusal_of(
        lambda: th.admit_provider_source(source(requirement="exact_request_verified"))
    )
    assert required.detail == detail.VERIFICATION_REQUIRED
    unbacked = refusal_of(lambda: th.admit_provider_source(source(trust="exact_request_verified")))
    assert unbacked.detail == detail.VERIFICATION_REQUIRED
    with pytest.raises(CausalValueError):
        th.admit_provider_source(source(trust="trusted"))
    with pytest.raises(CausalTypeError):
        th.admit_provider_source(object())  # type: ignore[arg-type]


def test_a_receipt_for_the_exact_provider_object_admits_it_as_verified():
    receipt = declaration().verify(gaussian_probes())
    verified = th.admit_provider_source(
        source(trust="exact_request_verified", requirement="exact_request_verified", receipt=receipt)
    )
    assert verified.trust == "verified_extension"
    assert verified.native_target_data is False
    attested = th.admit_provider_source(source())
    assert verified.identity != attested.identity
    # A receipt issued for another provider object does not cover this one.
    other = declaration(provider=identity("2.0")).verify(gaussian_probes())
    mismatched = refusal_of(
        lambda: th.admit_provider_source(
            source(trust="exact_request_verified", receipt=other)
        )
    )
    assert mismatched.detail == reason_details.provider_transport.RECEIPT_DOES_NOT_COVER


def test_an_admitted_source_connects_only_to_a_transport_it_belongs_to():
    admitted = th.admit_provider_source(source())
    connection = admitted.connect(consumer(), "transport_source", transport=transport())
    assert connection.producer == "trial_provider"
    assert connection.consumer == "transport_route"
    assert connection.calibration.status == "not_measured"

    # The population must be a source population of the transport.
    not_a_source = refusal_of(
        lambda: admitted.connect(
            consumer(), "transport_source", transport=transport(sources=("trial_b",))
        )
    )
    assert not_a_source.detail == reason_details.provider_transport.POPULATION_NOT_A_SOURCE
    # Every assumption the provider relies on must be carried by the transport.
    uncarried = refusal_of(
        lambda: admitted.connect(
            consumer(), "transport_source", transport=transport(assumptions=("something else",))
        )
    )
    assert uncarried.detail == reason_details.provider_transport.ASSUMPTIONS_NOT_CARRIED
    # A consumer that needs another population is a connection defect, never admitted.
    other_population = refusal_of(
        lambda: admitted.connect(
            consumer(population="target"), "transport_source", transport=transport()
        )
    )
    assert other_population.detail.startswith("interface_contract.")
    with pytest.raises(CausalTypeError):
        admitted.connect(object(), "transport_source", transport=transport())  # type: ignore[arg-type]
