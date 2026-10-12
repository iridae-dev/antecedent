"""Canonical provider-trust wire against the spellings older surfaces still store.

Rust ``ProviderTrust::as_wire`` emits ``externally_attested`` and
``verified_extension``. These aliases still parse and are not written:

- ``external_attested`` (distribution artifact codec)
- ``exact_request_verified``
- ``external_exact_request_verified``
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from antecedent.errors import CausalValueError
from antecedent.extensibility import ProviderTrust
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)

CANONICAL = {
    "externally_attested": ("external_attested",),
    "verified_extension": ("exact_request_verified", "external_exact_request_verified"),
    "native_licensed": (),
    "unverified": (),
}


def _law(**kwargs: object) -> JointDistributionArtifact:
    identity = DistributionIdentity(
        "interventional_predictive",
        (ScientificQuantity("y", "y", "outcome", "units", "target", "do(t=1)", 0, "mean"),),
        "joint",
        "study",
        "lab",
        "deterministic_exact",
        "snap",
        "contract",
    )
    draws = np.array([[1.0]], dtype=np.float64)
    return JointDistributionArtifact(identity, draws, **kwargs)  # type: ignore[arg-type]


def test_aliases_parse_to_the_canonical_wire() -> None:
    assert ProviderTrust("external_attested") is ProviderTrust.EXTERNALLY_ATTESTED
    assert ProviderTrust.EXTERNALLY_ATTESTED.value == "externally_attested"
    assert ProviderTrust("exact_request_verified") is ProviderTrust.VERIFIED_EXTENSION
    assert ProviderTrust("external_exact_request_verified") is ProviderTrust.VERIFIED_EXTENSION
    assert ProviderTrust.VERIFIED_EXTENSION.value == "verified_extension"
    for canonical, aliases in CANONICAL.items():
        assert ProviderTrust(canonical).value == canonical
        for alias in aliases:
            assert ProviderTrust(alias) is ProviderTrust(canonical)
            assert ProviderTrust(alias).value == canonical
    with pytest.raises(ValueError, match="not a valid ProviderTrust"):
        ProviderTrust("native")


def test_distribution_claim_reads_canonical_and_refuses_an_unearned_receipt() -> None:
    claimed = _law(claimed_trust="externally_attested")
    aliased = _law(trust="external_attested")
    assert claimed.trust == aliased.trust == "externally_attested"
    assert json.loads(aliased._native.metadata_json)["trust"] == "external_attested"
    for stamped in (
        "verified_extension",
        "exact_request_verified",
        "external_exact_request_verified",
    ):
        with pytest.raises(CausalValueError) as refused:
            _law(trust=stamped)
        assert refused.value.reason_code == "invalid_argument"
