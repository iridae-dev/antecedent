"""Every transport binding reports its identification status in one vocabulary."""

import pytest
from antecedent import _native
from antecedent.errors import CausalValueError
from antecedent.transport import advanced as transport
from antecedent.transport._restricted import RestrictedTransportIdentification

CANONICAL = (
    "identified",
    "proven_non_transportable",
    "missing_evidence",
    "not_certified",
    "budget_cancel",
)
LEGACY = {
    "exhausted": "budget_cancel",
    "stopped": "budget_cancel",
    "unevaluated": "budget_cancel",
    "structurally_unidentified": "proven_non_transportable",
    "combined_identified": "identified",
    "named_route": "identified",
}


def test_one_canonical_spelling_shared_with_rust():
    canonical, legacy = _native.transport_identification_statuses()
    assert tuple(canonical) == CANONICAL == transport.IDENTIFICATION_STATUSES
    assert dict(legacy) == LEGACY == dict(transport.LEGACY_IDENTIFICATION_STATUSES)
    assert not set(LEGACY) & set(CANONICAL)


def test_every_spelling_reads_as_its_canonical_status():
    for status in CANONICAL:
        assert transport.identification_status(status) == status
    for spelling, status in LEGACY.items():
        assert transport.identification_status(spelling) == status
        assert transport.identification_status({"outcome": spelling}) == status
    # Execution statuses and the IdentificationStatus spellings are other vocabularies.
    for other in ("support_failure", "unsupported_provider", "NotIdentified", "not_identified"):
        with pytest.raises(CausalValueError, match="unknown transport identification status"):
            transport.identification_status(other)


def test_python_wrappers_report_the_canonical_status_beside_their_outcome():
    combined = RestrictedTransportIdentification(
        outcome="combined_identified", reason=None, rules=(), formula="", source=None
    )
    assert combined.outcome == "combined_identified"
    assert combined.identification_status == "identified"
    stopped = RestrictedTransportIdentification(
        outcome="exhausted", reason=None, rules=(), formula="", source=None
    )
    assert stopped.identification_status == "budget_cancel"
    assert transport.identification_status(stopped) == "budget_cancel"
    certificate = transport.MissingEvidenceCertificate("transport_missing_evidence", (), "")
    missing = transport.TransportIdentification(None, certificate, outcome="missing_evidence")
    assert missing.identification_status == "missing_evidence"


def test_transport_identification_certificate_carries_the_canonical_status():
    from antecedent import Admg, AverageEffect, identify
    from antecedent import transport as day1

    query = day1.Transport(
        AverageEffect("x", "y"),
        target="target",
        evidence=day1.Evidence(
            source=day1.Source(
                "source", kind="experimental", interventions=["x"], sampling="independent"
            ),
            target_sampling="representative_sample",
        ),
    )
    ident = identify(graph=Admg.from_edges(["x", "y"], [("x", "y")]), query=query)
    status = ident.certificate["identification_status"]
    assert status in CANONICAL
    assert status == transport.identification_status(ident.certificate["outcome"])
