# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent policy declaration consumption never reconstructs a fitted answer."""

from dataclasses import FrozenInstanceError

import pytest
from antecedent.errors import (
    CausalError,
    CausalSerializationError,
    CausalTypeError,
    CausalValueError,
)
from antecedent.intervention_programs import (
    consume_policy_program,
    encode_policy_program,
    generated,
    policy_program_identity,
    transform,
)


def artifact_and_identity():
    names = ("x", "y")
    programs = (transform("x", generated() + 1),)
    return (
        names,
        programs,
        encode_policy_program(names, programs),
        policy_program_identity(names, programs),
    )


def test_policy_consumer_verifies_declaration_and_preserves_exact_bytes():
    names, programs, artifact, identity = artifact_and_identity()
    consumed = consume_policy_program(
        artifact, expected_identity=identity, supplied={"names": names}
    )
    assert consumed.identity == identity
    assert consumed.programs == programs
    assert consumed.acceptance.verified
    assert consumed.acceptance.identity_bound
    assert not consumed.acceptance.replayable
    assert not consumed.acceptance.resumable
    assert consumed.answer.kind == "unavailable"
    assert consumed.inspect().stage == "consumed"
    assert bytes(consumed.export()) == artifact
    assert not hasattr(consumed, "_prepared")
    with pytest.raises(FrozenInstanceError):
        consumed.names = ("edited",)


def test_policy_consumer_requires_explicit_unique_ordered_names():
    _, _, artifact, _ = artifact_and_identity()
    with pytest.raises(CausalTypeError):
        consume_policy_program(artifact)
    with pytest.raises(CausalTypeError):
        consume_policy_program(artifact, supplied={"x", "y"})
    with pytest.raises(CausalValueError):
        consume_policy_program(artifact, supplied=("x", "x"))


def test_policy_consumer_rejects_an_independent_identity_mismatch():
    names, _, artifact, identity = artifact_and_identity()
    other = ("0" if identity[0] != "0" else "1") + identity[1:]
    with pytest.raises(CausalSerializationError):
        consume_policy_program(artifact, expected_identity=other, supplied=names)


def test_policy_consumer_native_verification_rejects_truncation():
    names, _, artifact, _ = artifact_and_identity()
    with pytest.raises(CausalError):
        consume_policy_program(artifact[:-1], supplied=names)


def test_policy_consumer_without_expectation_makes_no_identity_binding_claim():
    names, _, artifact, _ = artifact_and_identity()
    consumed = consume_policy_program(artifact, supplied=names)
    assert consumed.acceptance.verified
    assert not consumed.acceptance.identity_bound
