"""Candidate-feature success lifecycle; no released-positive or calibration assertion."""

import json
import subprocess
import sys
from dataclasses import replace

import antecedent as ac
import numpy as np
import pytest
from antecedent import _native
from antecedent.errors import CausalSerializationError, CausalUnsupportedError, CausalValueError
from antecedent.transport import advanced as tr

CANDIDATE = hasattr(_native, "joint_transport_candidate")
internal_only = pytest.mark.skipif(
    not CANDIDATE, reason="internal candidate lifecycle is not a released positive"
)


def fixture(learned=False, varying="intercept", sharing="independent_varying_blocks"):
    graph = ac.Admg.from_edges(["a", "x", "y"], [("a", "y"), ("x", "y")])
    identified = tr.identify(
        graph=graph,
        query=tr.TransportQuery(
            ac.ResponseCurve("a", "y", grid=[0.0, 1.0]),
            tr.SelectionDiagram("source", "target", ["x"]),
            source_experiments=["a"],
        ),
    )
    assert identified.transportable
    source_records = []
    for j in range(2):
        k = np.arange(24, dtype=float)
        x = 1.2 * np.sin(0.37 * (k + 40 * j)) + 0.2
        a = np.tile([0.0, 1.0], 12)
        y = 1 + 0.5 * x + 0.4 * x * x + a * (2 + 1.5 * x + 1.2 * x * x) + 0.03 * np.cos(k + j)
        source_records.append(
            tr.JointTransportSource(
                f"s{j}",
                "source",
                tr.JointTransportIdentity(f"snapshot-s{j}", tuple(f"s{j}:{i}" for i in range(24))),
                {"a": a, "x": x, "y": y},
                0.09,
            )
        )
    x_target = np.array([-0.4, -0.2, 0.1, 0.3, 0.6])
    target = tr.JointTransportTarget(
        "target",
        tr.JointTransportIdentity("snapshot-target", tuple(f"target:{i}" for i in range(5))),
        {"x": x_target},
    )
    basis = 2 if learned else 1
    q = 1 + 2 * basis if varying == "intercept" else 1 + basis
    r = 1 if varying == "intercept" else 1 + basis

    def prior(n):
        covariance = np.eye(n) * 4.0 + (np.ones((n, n)) - np.eye(n)) * 0.15
        return tr.GaussianTransportPrior(
            tuple(0.1 * np.arange(n)), tuple(tuple(row) for row in covariance)
        )

    priors = tr.JointTransportPriors(prior(q), prior(r))
    options = dict(
        sources=source_records,
        target=target,
        features=["x"],
        identification=identified,
        priors=priors,
        varying=varying,
        sharing=sharing,
        draws=256,
        seed=77,
    )
    if learned:
        options["basis_degree"] = 2
    return options


def call(learned=False, **options):
    from antecedent.transport._joint_posterior import _candidate

    defaults = dict(
        treatment="a",
        outcome="y",
        varying="intercept",
        sharing="independent_varying_blocks",
        dependence="independent_samples",
        basis_degree=2 if learned else 1,
        max_unsupported_mass=0.0,
        conflict_z_threshold=3.0,
    )
    return _candidate(**(defaults | options | {"learned": learned}))


def oracle(options, learned):
    basis = 2 if learned else 1
    q = 1 + 2 * basis if options["varying"] == "intercept" else 1 + basis
    r = 1 if options["varying"] == "intercept" else 1 + basis
    blocks = 1 if options["sharing"] == "shared_varying_block" else len(options["sources"])
    n = q + r * blocks
    covariance = np.zeros((n, n))
    covariance[:q, :q] = options["priors"].invariant.covariance
    means = np.zeros(n)
    means[:q] = options["priors"].invariant.mean
    for b in range(blocks):
        covariance[q + b * r : q + (b + 1) * r, q + b * r : q + (b + 1) * r] = options[
            "priors"
        ].varying.covariance
        means[q + b * r : q + (b + 1) * r] = options["priors"].varying.mean
    precision = np.linalg.inv(covariance)
    rhs = precision @ means
    for j, source in enumerate(options["sources"]):
        x, a = source.data["x"], source.data["a"]
        phi = np.column_stack([x**power for power in range(1, basis + 1)])
        rows = np.zeros((len(x), n))
        rows[:, 0] = a
        rows[:, 1 : 1 + basis] = a[:, None] * phi
        block = 0 if blocks == 1 else j
        if options["varying"] == "intercept":
            rows[:, 1 + basis : q] = phi
        rows[:, q + block * r] = 1.0
        if r > 1:
            rows[:, q + block * r + 1 : q + (block + 1) * r] = phi
        precision += rows.T @ rows / source.noise_variance
        rhs += rows.T @ source.data["y"] / source.noise_variance
    posterior_covariance = np.linalg.inv(precision)
    posterior_mean = posterior_covariance @ rhs
    c = np.zeros(n)
    c[0] = 1.0
    target_x = options["target"].data["x"]
    c[1 : 1 + basis] = [np.mean(target_x**power) for power in range(1, basis + 1)]
    return posterior_mean, posterior_covariance, c


@internal_only
@pytest.mark.parametrize("learned", [False, True])
@pytest.mark.parametrize("varying", ["intercept", "intercept_and_covariates"])
@pytest.mark.parametrize("sharing", ["independent_varying_blocks", "shared_varying_block"])
def test_candidate_frozen_four_plus_four_coordinates_full_joint_oracle(learned, varying, sharing):
    options = fixture(learned, varying, sharing)
    result = call(learned, **options)
    mean, covariance, c = oracle(options, learned)
    assert result.posterior_mean == pytest.approx(mean, abs=2e-10)
    assert np.asarray(result.posterior_covariance) == pytest.approx(covariance, abs=2e-10)
    assert result.target_effect_mean == pytest.approx(c @ mean, abs=2e-10)
    assert result.target_effect_variance == pytest.approx(c @ covariance @ c, abs=2e-10)
    assert result.calibration == "unmeasured" and result.release_status == "candidate_only"
    with pytest.raises(TypeError, match="unexpected keyword"):
        replace(result, kind="learned_gaussian" if not learned else "gaussian")
    assert result.draws.shape == (256, len(result.draw_names))
    assert not result.draws.flags.writeable
    target_col = result.draw_names.index("effect.target")
    assert result.draws[:, target_col] == pytest.approx(result.draws[:, : len(mean)] @ c, abs=2e-10)
    artifact = result.export()
    consumed = tr.consume_joint_transport_posterior(
        artifact, kind=result.kind, expected_identity=result.expectation()
    )
    assert consumed.to_dict() == result.to_dict()
    assert consumed.posterior_covariance == result.posterior_covariance
    if learned:
        assert result.to_dict()["result"]["provider"]["provider"]
    # Reports/draw views are copies: caller mutation cannot rewrite the retained artifact.
    report = consumed.to_dict()
    report["result"]["target_effect_mean"] = 999.0
    assert consumed.target_effect_mean == result.target_effect_mean
    assert consumed.export() == artifact
    damaged = bytearray(artifact)
    damaged[len(damaged) // 2] ^= 0xFF
    with pytest.raises((CausalValueError, CausalUnsupportedError, CausalSerializationError)):
        tr.consume_joint_transport_posterior(
            bytes(damaged), kind=result.kind, expected_identity=result.expectation()
        )
    # A genuine new execution has a new data binding, even with the same row IDs.
    first = options["sources"][0]
    changed_data = dict(first.data)
    changed_data["y"] = [float(y) + 0.25 for y in first.data["y"]]
    changed = call(
        learned,
        **(options | {"sources": [replace(first, data=changed_data), options["sources"][1]]}),
    )
    assert changed.expectation()["data_digest"] != result.expectation()["data_digest"]
    with pytest.raises((CausalValueError, CausalUnsupportedError, CausalSerializationError)):
        tr.consume_joint_transport_posterior(
            changed.export(), kind=result.kind, expected_identity=result.expectation()
        )
    with pytest.raises((CausalValueError, CausalUnsupportedError, CausalSerializationError)):
        tr.consume_joint_transport_posterior(
            artifact,
            kind=result.kind,
            expected_identity={
                "premises_digest": "changed",
                "data_digest": result.expectation()["data_digest"],
            },
        )


@internal_only
@pytest.mark.parametrize("learned", [False, True])
def test_candidate_source_prior_overlap_population_schema_and_original_proof_refusals(learned):
    options = fixture(learned)
    first = options["sources"][0]
    cases = []
    cases.append({"sources": [first, replace(options["sources"][1], identity=first.identity)]})
    bad_prior = replace(options["priors"].invariant, bank_id="bank", consumed=(first.identity,))
    cases.append({"priors": replace(options["priors"], invariant=bad_prior)})
    cases.append({"target": replace(options["target"], population="other")})
    cases.append({"sources": [replace(first, population="other"), options["sources"][1]]})
    cases.append({"features": ["y"]})
    cases.append({"treatment": "x"})
    cases.append({"target": replace(options["target"], data={"x": [1000.0] * 5})})
    for changes in cases:
        with pytest.raises((CausalValueError, CausalUnsupportedError)):
            call(learned, **(options | changes))
    # Mutating the public proof view does not alter the original native derivation.
    altered = replace(options["identification"], formula=None, outcome="not_certified")
    assert call(
        learned, **(options | {"identification": altered})
    ).target_effect_mean == pytest.approx(call(learned, **options).target_effect_mean)


@internal_only
@pytest.mark.parametrize("learned", [False, True])
def test_candidate_fresh_process_consumer_retains_joint_law(learned, tmp_path):
    result = call(learned, **fixture(learned))
    path = tmp_path / "posterior.bin"
    path.write_bytes(result.export())
    code = """import json,sys; from pathlib import Path; from antecedent.transport import advanced as t
r=t.consume_joint_transport_posterior(Path(sys.argv[1]).read_bytes(),kind=sys.argv[2],expected_identity=json.loads(sys.argv[3])); print(json.dumps({'mean':r.target_effect_mean,'variance':r.target_effect_variance,'calibration':r.calibration,'shape':list(r.draws.shape)}))"""
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(path),
            result.kind,
            json.dumps(result.expectation()),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(done.stdout)
    assert value["mean"] == result.target_effect_mean
    assert value["variance"] == result.target_effect_variance
    assert value["calibration"] == "unmeasured"
    assert value["shape"] == list(result.draws.shape)


def test_default_joint_transport_still_refuses_real_valid_scope():
    if CANDIDATE:
        pytest.skip("default-wheel refusal is checked separately from candidate feature proof")
    with pytest.raises(CausalUnsupportedError) as error:
        tr.joint_bayesian_transport(sources=[{"id": "source"}], target={"x": [0.1]}, features=["x"])
    assert error.value.reason_code == "cell_not_licensed"


def test_candidate_result_constructor_and_prior_dimension_bound_do_not_accept_fabrication():
    from collections.abc import Sequence

    from antecedent.errors import CausalTypeError

    with pytest.raises(CausalTypeError):
        tr.JointTransportPosterior()

    class Oversized(Sequence):
        def __len__(self):
            return 257

        def __getitem__(self, index):
            raise AssertionError("oversized prior must not be materialized")

    with pytest.raises(CausalValueError, match="1..256"):
        tr.GaussianTransportPrior(Oversized(), Oversized())._wire()
