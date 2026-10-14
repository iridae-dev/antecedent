"""Arbitrary law, posterior and nested callbacks through the Python facade.

The reference callback is a Gaussian law implemented with ``math.erf`` and ``statistics.NormalDist``;
its mean, distribution function and quantile function are compared with values computed in the test
(the closed form and tabulated z quantiles), and a posterior update with the conjugate-normal
posterior. Provider callables are counted so every refusal asserts what actually ran. A sentinel
provider shows that consuming, inspecting and planning never call a callable.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
import textwrap
from statistics import NormalDist

import pytest
from antecedent import provider_callbacks as pc
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.joint_distribution import ScientificQuantity

from _refusal import assert_registered_refusal

MU = 2.0
SD = 1.5


def quantity(population="trial", regime="do(a=1)"):
    return ScientificQuantity.of(
        "outcome", "systolic", units="mmHg", population=population, regime=regime
    )


def identity(version="1.0"):
    return pc.ProviderIdentity("lab", "gaussian", version, "snap-1", "req-1")


def declaration(**overrides):
    base = pc.CallbackDeclaration(
        "gaussian_law",
        "law",
        identity(),
        quantity(),
        "estimand:outcome_law",
        "interventional_predictive",
        pc.Support.real(),
        operations=("mean", "sample", "cdf", "quantile"),
        dependencies=pc.Dependencies.complete(
            [pc.Dependency("model", "parameters", "mu=2;sd=1.5")]
        ),
        replay=pc.Replay.seeded_stochastic("outcome_draws"),
        trust=pc.Trust.externally_attested("lab"),
        evidence=(),
    )
    return dataclasses.replace(base, **overrides)


def mean_only(**overrides):
    overrides.setdefault("replay", pc.Replay.deterministic_pure())
    return declaration(operations=("mean",), **overrides)


class CountedProvider(pc.FunctionProvider):
    """A function provider that records the operations it was asked."""

    calls: list


def gaussian(version="1.0", mu=MU, calls=None):
    calls = [] if calls is None else calls
    law = NormalDist(mu, SD)

    def fn(query, ctx):
        calls.append(query.operation)
        if query.operation == "mean":
            return pc.Output.mean(mu)
        if query.operation == "sample":
            rng = ctx.rng()
            return pc.Output.sample(rng.gauss(mu, SD) for _ in range(query.draws))
        if query.operation == "cdf":
            return pc.Output.cdf(
                0.5 * (1 + math.erf((t - mu) / (SD * math.sqrt(2)))) for t in query.points
            )
        if query.operation == "quantile":
            return pc.Output.quantile(law.inv_cdf(p) for p in query.points)
        raise ValueError("not implemented")

    provider = CountedProvider(identity(version), fn)
    provider.calls = calls
    return provider


def counted(provider):
    return provider.calls


def refusal_of(callable_, *args, **kwargs):
    with pytest.raises(pc.CallbackRefusal) as caught:
        callable_(*args, **kwargs)
    assert_registered_refusal(caught.value)
    return caught.value


# -- reference values ----------------------------------------------------------------------------


def test_gaussian_callback_mean_cdf_quantile_match_closed_forms():
    provider = gaussian()
    session = pc.CallbackSession()
    decl = declaration()
    mean = session.execute(provider, decl, pc.Request(pc.Query.mean(), seed=3))
    assert mean.output.values == (MU,)
    assert mean.standing == "licensed_external"
    thresholds = [MU - 1.96 * SD, MU, MU + SD]
    cdf = session.execute(provider, decl, pc.Request(pc.Query.cdf(thresholds), seed=3))
    for t, got in zip(thresholds, cdf.output.values, strict=True):
        closed = 0.5 * (1 + math.erf((t - MU) / (SD * math.sqrt(2))))
        assert got == pytest.approx(closed, abs=1e-12)
    # Tabulated: Phi(-1.96) = 0.0249978951482205, Phi(0) = 0.5, Phi(1) = 0.8413447460685429.
    assert cdf.output.values == pytest.approx(
        (0.0249978951482205, 0.5, 0.8413447460685429), abs=1e-9
    )
    quantile = session.execute(
        provider, decl, pc.Request(pc.Query.quantile([0.5, 0.9, 0.975]), seed=3)
    )
    z = (0.0, 1.2815515655446004, 1.959963984540054)
    assert quantile.output.values == pytest.approx(tuple(MU + SD * v for v in z), abs=1e-9)
    n = 20000
    sample = session.execute(provider, decl, pc.Request(pc.Query.sample(n), seed=3))
    draws = sample.output.values
    mean_hat = sum(draws) / n
    var_hat = sum((x - mean_hat) ** 2 for x in draws) / (n - 1)
    assert abs(mean_hat - MU) <= 5 * SD / math.sqrt(n)
    assert abs(var_hat - SD**2) <= 5 * SD**2 * math.sqrt(2 / (n - 1))
    assert session.invocations == 4
    assert len(counted(provider)) == 4


def test_conjugate_posterior_update_matches_closed_form():
    # Prior N(0, 4), noise variance 1, y = 2: precision 1.25, mean 1.6, variance 0.8.
    calls = []

    def fn(query, ctx):
        calls.append(query.operation)
        precision = 1 / 4 + 1 / 1
        mean = (0 / 4 + query.observation / 1) / precision
        rng = ctx.rng()
        sd = math.sqrt(1 / precision)
        return pc.Output.posterior_update(mean, (rng.gauss(mean, sd) for _ in range(query.draws)))

    decl = declaration(
        component="posterior",
        kind="posterior",
        meaning="causal_functional_posterior",
        operations=("mean", "sample", "posterior_update"),
        update=("study-a", "prior-0", quantity(regime="observational")),
    )
    session = pc.CallbackSession()
    n = 20000
    result = session.execute(
        pc.FunctionProvider(identity(), fn),
        decl,
        pc.Request(pc.Query.posterior_update(2.0, draws=n), seed=1),
    )
    mean, *draws = result.output.values
    assert mean == pytest.approx(1.6, abs=1e-12)
    m = sum(draws) / n
    v = sum((x - m) ** 2 for x in draws) / (n - 1)
    assert abs(m - 1.6) <= 5 * math.sqrt(0.8 / n)
    assert abs(v - 0.8) <= 5 * 0.8 * math.sqrt(2 / (n - 1))
    # The update is declared, so a quantile query is refused before any invocation.
    before = len(calls)
    error = refusal_of(
        session.execute,
        pc.FunctionProvider(identity(), fn),
        decl,
        pc.Request(pc.Query.quantile([0.5])),
    )
    assert error.reason_code == "external_capability_missing"
    assert error.invocations == 0
    assert len(calls) == before


def test_operations_are_negotiated_separately():
    session = pc.CallbackSession()
    sample_only = declaration(operations=("sample",))
    provider = gaussian()
    for query in (pc.Query.mean(), pc.Query.cdf([0.0]), pc.Query.quantile([0.5])):
        error = refusal_of(session.execute, provider, sample_only, pc.Request(query))
        assert error.reason_code == "external_capability_missing"
        assert error.invocations == 0
    assert counted(provider) == []
    # A distribution function does not supply a quantile function.
    cdf_only = declaration(operations=("cdf",))
    error = refusal_of(session.execute, provider, cdf_only, pc.Request(pc.Query.quantile([0.5])))
    assert error.reason_code == "external_capability_missing"
    assert error.detail == "callback_protocol.operation_not_declared"
    # An update needs its declaration and the observation law.
    error = refusal_of(
        declaration(
            kind="posterior", meaning="parameter_posterior", operations=("posterior_update",)
        ).validate
    )
    assert error.detail == "callback_protocol.update_declaration_mismatch"
    error = refusal_of(declaration(operations=("mean", "posterior_update")).validate)
    assert error.detail == "callback_protocol.operation_not_for_kind"


# -- replay, reuse, invalidation ----------------------------------------------------------------


def test_deterministic_replay_does_not_invoke_and_a_changed_dependency_recomputes():
    provider = gaussian()
    session = pc.CallbackSession()
    decl = mean_only()
    request = pc.Request(pc.Query.mean())
    first = session.execute(provider, decl, request)
    assert (first.reused, first.invocations) == (False, 1)
    assert session.plan(decl, request).plan == "reuse"
    again = session.execute(provider, decl, request)
    assert (again.reused, again.invocations) == (True, 0)
    assert again.output == first.output
    assert len(counted(provider)) == 1
    changed = dataclasses.replace(
        decl,
        dependencies=pc.Dependencies.complete([pc.Dependency("model", "parameters", "mu=3")]),
    )
    plan = session.plan(changed, request)
    assert (plan.plan, plan.reason) == ("recompute", "dependency_changed:parameters")
    recomputed = session.execute(gaussian(mu=3.0, calls=counted(provider)), changed, request)
    assert recomputed.output.values == (3.0,)
    assert recomputed.recomputed_because == "dependency_changed:parameters"
    assert len(session) == 2


def test_stochastic_stream_rules():
    decl = declaration()

    def draws(seed, declaration_=decl, warm_up=False):
        session = pc.CallbackSession()
        provider = gaussian()
        if warm_up:
            session.execute(provider, declaration_, pc.Request(pc.Query.mean(), seed=99))
            session.execute(provider, declaration_, pc.Request(pc.Query.sample(7), seed=98))
        done = session.execute(provider, declaration_, pc.Request(pc.Query.sample(6), seed=seed))
        return done.output.values

    base = draws(11)
    assert len(base) == 6
    assert base == draws(11), "the same seed replays bit for bit"
    assert base == draws(11, warm_up=True), "earlier calls do not move a stream"
    assert base != draws(12), "another seed is another stream"
    other = declaration(replay=pc.Replay.seeded_stochastic("noise_draws"))
    assert base != draws(11, other), "another semantic stream is independent"
    # A deterministic callback is given no generator.
    seen = []

    def probe(query, ctx):
        seen.append(ctx.stream_seed)
        return pc.Output.mean(MU)

    pc.CallbackSession().execute(
        pc.FunctionProvider(identity(), probe), mean_only(), pc.Request(pc.Query.mean())
    )
    assert seen == [None]


def test_hidden_state_change_is_detected_and_unexposed_state_is_never_reused():
    version = {"n": 1}
    calls = []

    def fn(query, ctx):
        calls.append(version["n"])
        return pc.Output.mean(10.0 * version["n"])

    provider = pc.FunctionProvider(identity(), fn, state=lambda: f"weights-v{version['n']}")
    decl = mean_only(replay=pc.Replay.stateful(exposes_state_identity=True))
    request = pc.Request(pc.Query.mean())
    session = pc.CallbackSession()
    first = session.execute(provider, decl, request)
    assert first.output.values == (10.0,) and first.state_identity == "weights-v1"
    assert session.execute(provider, decl, request).reused
    version["n"] = 2
    changed = session.execute(provider, decl, request)
    assert not changed.reused
    assert changed.recomputed_because == "state_changed"
    assert changed.output.values == (20.0,)
    assert changed.standing == "diagnostic"
    assert len(calls) == 2
    # Unexposed state is never shown current.
    hidden = mean_only(replay=pc.Replay.stateful(exposes_state_identity=False))
    other = pc.CallbackSession()
    other.execute(provider, hidden, request)
    second = other.execute(provider, hidden, request)
    assert second.recomputed_because == "hidden_state_unobservable"
    assert len(calls) == 4


def test_provider_replacement_is_refused_and_is_a_new_callback():
    session = pc.CallbackSession()
    v1, v2 = gaussian("1.0"), gaussian("2.0", mu=4.0)
    decl = mean_only()
    request = pc.Request(pc.Query.mean())
    session.execute(v1, decl, request)
    error = refusal_of(session.execute, v2, decl, request)
    assert error.reason_code == "external_binding_mismatch"
    assert error.detail == "callback_protocol.provider_identity_mismatch"
    assert error.invocations == 0 and counted(v2) == []
    replaced = dataclasses.replace(decl, provider=identity("2.0"))
    assert replaced.identity != decl.identity
    done = session.execute(v2, replaced, request)
    assert done.output.values == (4.0,) and not done.reused
    assert len(session) == 2
    assert session.record(0)["declaration"]["identity"]["version_id"] == "1.0"


# -- output checking, failures, limits ----------------------------------------------------------


def test_malformed_laws_are_refused_whole():
    def bad(query, ctx):
        if query.operation == "cdf":
            return pc.Output.cdf([0.9, 0.2])
        if query.operation == "sample":
            return pc.Output.sample([float("nan")] * query.draws)
        return pc.Output.mean(0.0)

    provider = pc.FunctionProvider(identity(), bad)
    session = pc.CallbackSession()
    decl = declaration()
    error = refusal_of(session.execute, provider, decl, pc.Request(pc.Query.cdf([0.0, 1.0])))
    assert error.reason_code == "callback_output_invalid"
    assert error.detail == "callback_protocol.output_not_a_law"
    assert error.invocations == 1
    error = refusal_of(session.execute, provider, decl, pc.Request(pc.Query.sample(3)))
    assert error.detail == "callback_protocol.output_not_finite"
    error = refusal_of(session.execute, provider, decl, pc.Request(pc.Query.quantile([0.5])))
    assert error.detail == "callback_protocol.output_shape"
    assert len(session) == 0


def test_failure_cancellation_and_limits_are_typed_and_counted():
    def failing(query, ctx):
        raise RuntimeError("backend down")

    session = pc.CallbackSession()
    decl = mean_only()
    request = pc.Request(pc.Query.mean())
    error = refusal_of(session.execute, pc.FunctionProvider(identity(), failing), decl, request)
    assert error.reason_code == "callback_provider_unavailable"
    assert error.invocations == 1 and len(session) == 0
    # Cancelled before the call: nothing runs.
    token = pc.CallbackToken()
    token.cancel()
    provider = gaussian()
    error = refusal_of(session.execute, provider, decl, request, token=token)
    assert error.reason_code == "cancelled_no_claim"
    assert error.invocations == 0 and counted(provider) == []

    # Cancelled during the call: it ran, the result is discarded.
    def cancelling(query, ctx):
        ctx.token.cancel()
        return pc.Output.mean(MU)

    error = refusal_of(
        session.execute,
        pc.FunctionProvider(identity(), cancelling),
        decl,
        request,
        token=pc.CallbackToken(),
    )
    assert error.detail == "callback_protocol.cancelled" and error.invocations == 1
    error = refusal_of(
        session.execute, provider, decl, request, policy=pc.ExecutionPolicy(max_invocations=0)
    )
    assert error.reason_code == "callback_limit_exceeded" and error.invocations == 0
    error = refusal_of(
        session.execute,
        gaussian(),
        declaration(),
        pc.Request(pc.Query.sample(5)),
        policy=pc.ExecutionPolicy(max_output_values=3),
    )
    assert error.detail == "callback_protocol.output_too_large"


def test_side_effect_retry_is_refused_and_never_invoked_outside_execution():
    calls = []
    fail_first = {"on": True}

    def fn(query, ctx):
        calls.append(ctx.attempt)
        if fail_first["on"]:
            fail_first["on"] = False
            raise RuntimeError("timeout after the effect may have happened")
        return pc.Output.mean(MU)

    provider = pc.FunctionProvider(identity(), fn)
    request = pc.Request(pc.Query.mean())
    decl = mean_only(replay=pc.Replay.side_effecting())
    session = pc.CallbackSession()
    for phase in ("planning", "verification"):
        error = refusal_of(
            session.execute,
            provider,
            decl,
            request,
            policy=pc.ExecutionPolicy(phase=phase, allow_side_effects=True),
        )
        assert error.reason_code == "callback_side_effect_forbidden"
        assert error.detail == "callback_protocol.invocation_outside_execution"
    assert calls == []
    error = refusal_of(session.execute, provider, decl, request)
    assert error.detail == "callback_protocol.side_effects_not_allowed" and calls == []
    # Planning reads the declaration only.
    assert session.plan(decl, request).plan == "recompute"
    allow = pc.ExecutionPolicy(allow_side_effects=True)
    first = refusal_of(session.execute, provider, decl, request, policy=allow)
    assert first.invocations == 1 and first.request_attempts == 1
    retry = refusal_of(session.execute, provider, decl, request, policy=allow)
    assert retry.detail == "callback_protocol.retry_outside_policy"
    assert retry.invocations == 0 and calls == [1]
    # Under an idempotent contract with a key the declared retry is allowed once.
    calls.clear()
    fail_first["on"] = True
    keyed = mean_only(replay=pc.Replay.side_effecting(max_attempts=2))
    keyed_request = pc.Request(pc.Query.mean(), idempotency_key="order-17")
    fresh = pc.CallbackSession()
    refusal_of(fresh.execute, provider, keyed, keyed_request, policy=allow)
    done = fresh.execute(provider, keyed, keyed_request, policy=allow)
    assert done.invocations == 1 and done.standing == "diagnostic"
    assert fresh.execute(provider, keyed, keyed_request, policy=allow).reused
    assert calls == [1, 2]


def test_opaque_dependencies_make_reuse_ineligible_and_the_result_diagnostic():
    provider = gaussian()
    decl = mean_only(dependencies=pc.Dependencies.opaque())
    session = pc.CallbackSession()
    request = pc.Request(pc.Query.mean())
    first = session.execute(provider, decl, request)
    second = session.execute(provider, decl, request)
    assert not second.reused and second.recomputed_because == "opaque_dependencies"
    assert first.standing == "diagnostic"
    assert len(counted(provider)) == 2
    undeclared = mean_only(
        dependencies=pc.Dependencies.undeclared(), replay=pc.Replay.side_effecting(max_attempts=2)
    )
    error = refusal_of(
        session.execute,
        provider,
        undeclared,
        request,
        policy=pc.ExecutionPolicy(allow_side_effects=True),
    )
    assert error.reason_code == "callback_reuse_ineligible"
    assert len(counted(provider)) == 2


# -- nested callbacks -------------------------------------------------------------------------


def test_nested_callback_binds_inputs_and_reports_dependents_and_overlap():
    source = "ab" * 32
    other_source = "cd" * 32
    inner = declaration(
        component="inner",
        replay=pc.Replay.deterministic_pure(),
        operations=("mean",),
        evidence=(source,),
    )
    session = pc.CallbackSession()
    inner_done = session.execute(gaussian(), inner, pc.Request(pc.Query.mean()))
    outer = declaration(
        component="shifted",
        kind="nested",
        provider=pc.ProviderIdentity("lab", "shifted", "1.0", "snap-1", "req-1"),
        replay=pc.Replay.deterministic_pure(),
        operations=("mean",),
        inputs=(pc.Input.from_callback("inner", inner),),
        evidence=(),
    )
    assert outer.effective_evidence() == (source,)
    request = pc.Request(pc.Query.mean(), input_digests={"inner": values_digest_of(inner_done)})

    def shifted(query, ctx):
        return pc.Output.mean(ctx.inputs["inner"].values[0] + 3.0)

    provider = pc.FunctionProvider(outer.provider, shifted)
    done = session.execute(provider, outer, request, inputs={"inner": inner_done})
    assert done.output.values == (5.0,)
    # A tampered digest or a missing input is refused before the provider runs.
    bad = pc.Request(pc.Query.mean(), input_digests={"inner": "00"})
    error = refusal_of(session.execute, provider, outer, bad, inputs={"inner": inner_done})
    assert error.reason_code == "external_binding_mismatch"
    error = refusal_of(session.execute, provider, outer, request)
    assert error.detail == "callback_protocol.request_mismatch"
    assert session.dependents_of(inner.identity) == (1,)
    assert session.dependents_of(outer.identity) == ()
    # Overlap: a sibling over disjoint evidence shares nothing; undeclared evidence is unknown.
    sibling = dataclasses.replace(
        outer,
        component="sibling",
        inputs=(
            pc.Input.from_callback("inner", dataclasses.replace(inner, evidence=(other_source,))),
        ),
    )
    assert sibling.effective_evidence() == (other_source,)
    unknown = dataclasses.replace(outer, evidence=None)
    assert unknown.effective_evidence() is None


def values_digest_of(execution):
    return pc.values_digest(execution.output.values)


# -- closed routes ------------------------------------------------------------------------------


def test_closed_routes_refuse_with_registered_codes_and_remedies():
    for route, code, detail in (
        (pc.refuse_interval, "cell_not_licensed", "callback_protocol.interval_not_licensed"),
        (
            pc.refuse_posterior_guarantee,
            "cell_not_licensed",
            "callback_protocol.posterior_guarantee_not_licensed",
        ),
        (
            pc.refuse_native_trust,
            "cell_not_licensed",
            "callback_protocol.native_trust_not_conferred",
        ),
        (
            pc.refuse_identification_licence,
            "cell_not_licensed",
            "callback_protocol.identification_not_conferred",
        ),
        (pc.refuse_joint_law, "joint_law_required", "callback_protocol.joint_law_not_conferred"),
    ):
        error = refusal_of(route)
        assert (error.reason_code, error.detail) == (code, detail)
        assert error.remedy
    # There is no native trust level to declare.
    with pytest.raises(CausalValueError, match="trust level"):
        declaration(trust=pc.Trust("native_licensed")).validate()


# -- artifacts ----------------------------------------------------------------------------------


def exported():
    session = pc.CallbackSession()
    decl = declaration()
    done = session.execute(gaussian(), decl, pc.Request(pc.Query.cdf([0.5, 2.0, 3.5]), seed=9))
    return session, decl, done, session.export(done, "callback-result-1")


def test_exported_result_round_trips_and_mutations_are_refused():
    _, decl, done, result = exported()
    assert result.artifact == exported()[3].artifact
    consumed = pc.consume_callback_result(result.artifact, result.identity)
    assert consumed.declaration_identity == decl.identity
    assert consumed.stage == "computed"
    assert consumed.values == pytest.approx(done.output.values, abs=0)
    meta = pc.inspect_callback_result(result.artifact)
    assert meta["stage"]["status"] == "computed" and meta["standing"] == "licensed_external"
    assert meta["contract"]["produces"][0]["calibration"]["status"] == "unmeasured"
    assert meta["contract"]["produces"][0]["calibration"]["record_id"] is None
    error = refusal_of(pc.consume_callback_result, result.artifact, "0" * 64)
    assert error.detail == "callback_protocol.identity_mismatch"
    for offset in (len(result.artifact) // 3, len(result.artifact) // 2, len(result.artifact) - 5):
        corrupt = bytearray(result.artifact)
        corrupt[offset] ^= 0x01
        refusal_of(pc.consume_callback_result, bytes(corrupt))
    refusal_of(pc.consume_callback_result, result.artifact[:-17])
    refusal_of(pc.consume_callback_result, b"not an artifact")
    with pytest.raises(pc.CallbackRefusal):
        pc.inspect_callback_result(b"\x00\x01\x02")


def test_untrusted_artifact_never_executes_code():
    sentinel = []

    def boom(query, ctx):
        sentinel.append("called")
        raise AssertionError("a callable must not run for consume, inspect or plan")

    provider = pc.FunctionProvider(identity(), boom)
    session, decl, _, result = exported()
    pc.consume_callback_result(result.artifact)
    pc.inspect_callback_result(result.artifact)
    session.plan(decl, pc.Request(pc.Query.mean()))
    fresh = pc.CallbackSession()
    resumed = fresh.resume(result.artifact, expected_identity=result.identity)
    assert resumed.kind == "unavailable"
    # A hostile artifact: corrupted bytes with the sentinel nearby are refused and nothing runs.
    corrupt = bytearray(result.artifact)
    corrupt[len(corrupt) // 2] ^= 0xFF
    refusal_of(fresh.resume, bytes(corrupt), provider=provider)
    assert sentinel == []


def test_fresh_process_resume_with_and_without_a_provider():
    _, _, done, result = exported()
    fresh = pc.CallbackSession()
    unavailable = fresh.resume(result.artifact, expected_identity=result.identity)
    assert unavailable.kind == "unavailable"
    assert unavailable.refusal.reason_code == "callback_provider_unavailable"
    assert fresh.invocations == 0
    assert fresh.record(unavailable.record)["stage"] == "unavailable"
    # The historical result stays inspectable in the session.
    assert fresh.record(0)["stage"] == "computed"
    calls = []
    compatible = gaussian(calls=calls)
    answered = pc.CallbackSession().resume(
        result.artifact, expected_identity=result.identity, provider=compatible
    )
    assert answered.kind == "answered" and answered.execution.reused
    assert answered.execution.output.values == done.output.values
    assert calls == []
    # A replaced provider is refused, not substituted.
    error = refusal_of(
        pc.CallbackSession().resume,
        result.artifact,
        provider=gaussian("2.0", mu=7.0),
    )
    assert error.reason_code == "external_binding_mismatch"
    # A truly fresh interpreter consumes and resumes the same bytes.
    script = textwrap.dedent(
        """
        import json, sys
        from antecedent import provider_callbacks as pc
        artifact = bytes.fromhex(sys.argv[1])
        consumed = pc.consume_callback_result(artifact, sys.argv[2])
        resumed = pc.CallbackSession().resume(artifact, expected_identity=sys.argv[2])
        print(json.dumps([consumed.stage, resumed.kind, list(consumed.values)]))
        """
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, result.artifact.hex(), result.identity],
        capture_output=True,
        text=True,
        check=True,
    )
    stage, kind, values = json.loads(out.stdout)
    assert (stage, kind) == ("computed", "unavailable")
    assert values == pytest.approx(list(done.output.values), abs=0)


def test_provider_objects_need_an_identity():
    with pytest.raises(CausalTypeError):
        pc.CallbackSession().execute(
            object(),
            mean_only(),
            pc.Request(pc.Query.mean()),  # type: ignore[arg-type]
        )
