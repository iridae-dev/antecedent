"""Posterior bytes load independently as records, without restoring a study."""

import gc
import subprocess
import sys
from dataclasses import FrozenInstanceError
from hashlib import sha256

import antecedent as ac
import numpy as np
import pytest
from antecedent import _native
from antecedent._posterior_artifact import consume_posterior
from antecedent.errors import CausalSerializationError, CausalTypeError, CausalValueError
from antecedent.outcomes import require_success


def posterior_bytes(*, summaries_only=False, multiple=False):
    names = ["ate", "coefficient"] if multiple else ["ate"]
    means = [2.0, 5.0] if multiple else [2.0]
    lower = [1.05, 4.05] if multiple else [1.05]
    upper = [2.95, 5.95] if multiple else [2.95]
    kwargs = {
        "n_draws": 3,
        "mean": means,
        "sd": [1.0] * len(names),
        "q025": lower,
        "q975": upper,
        "backend_id": "conjugate",
        "identification": "NonparametricallyIdentified",
        "quantity_names": names,
    }
    if summaries_only:
        native = _native.PosteriorArtifact.from_moments(**kwargs)
    else:
        native = _native.PosteriorArtifact(
            **kwargs, draws=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] if multiple else [1.0, 2.0, 3.0]
        )
    return _native.encode_posterior_artifact(native)


def test_historical_summary_only_bytes_load_as_sealed_record():
    data = posterior_bytes(summaries_only=True)
    result = require_success(ac.load(data))
    assert result.acceptance.status == "sealed"
    assert result.acceptance.verified is False
    assert result.acceptance.identity_bound is False
    assert result.acceptance.replayable is False
    assert result.acceptance.resumable is False
    assert result.study is None
    assert result.quantity_names == ("ate",)
    assert result.n_draws == 3
    assert result.mean == (2.0,)
    assert result.sd == (1.0,)
    assert result.quantity("ate").draws.size == 0
    assert result.inspect().stage == "consumed"
    assert result.inspect().identification.available is False
    assert result.answer.kind == "structured"
    assert bytes(result.export()) == data


def test_explicit_encoding_digest_binds_only_bytes():
    data = posterior_bytes()
    digest = sha256(data).hexdigest()
    consumed = consume_posterior(data, expected_encoding_sha256=digest)
    assert consumed.acceptance.identity_bound
    assert consumed.acceptance.status == "sealed"
    assert not consumed.acceptance.verified
    loaded = require_success(ac.load(data, expected={"expected_encoding_sha256": digest}))
    assert loaded.acceptance.identity_bound
    assert bytes(loaded.export()) == data
    assert loaded.export().expectation.identity == {"expected_encoding_sha256": digest}
    with pytest.raises(CausalValueError, match="differs"):
        ac.load(data, expected={"expected_encoding_sha256": "0" * 64})
    with pytest.raises(CausalValueError, match="hexadecimal"):
        consume_posterior(data, expected_encoding_sha256="not a digest")
    with pytest.raises(CausalTypeError, match=r"expected_encoding_sha256 must be a hexadecimal string"):
        consume_posterior(data, expected_encoding_sha256=12)


def test_matching_digest_still_requires_native_semantic_parser(monkeypatch):
    original = _native.decode_posterior_artifact
    calls = []

    def checked_decode(data):
        calls.append(data)
        return original(data)

    monkeypatch.setattr(_native, "decode_posterior_artifact", checked_decode)
    malformed = b"invalid posterior layout"
    with pytest.raises(CausalSerializationError, match="magic"):
        consume_posterior(malformed, expected_encoding_sha256=sha256(malformed).hexdigest())
    assert calls == [malformed]


def test_quantity_selection_is_explicit_and_retains_native_owner():
    result = consume_posterior(posterior_bytes(multiple=True))
    with pytest.raises(CausalValueError, match="explicitly"):
        result.quantity()
    with pytest.raises(CausalValueError, match="missing"):
        result.quantity("absent")
    first = result.quantity("ate")
    second = result.quantity("coefficient")
    assert first.mean == 2.0
    assert second.mean == 5.0
    assert first.sd == 1.0
    assert first.answer.kind == "structured"
    assert first.answer.kind != "distribution"
    draws = second.draws
    del result, second
    gc.collect()
    np.testing.assert_array_equal(draws, [4.0, 5.0, 6.0])
    assert not draws.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        draws[0] = -10
    with pytest.raises(ValueError, match="WRITEABLE"):
        draws.setflags(write=True)
    with pytest.raises(FrozenInstanceError, match=r"cannot assign to field '_index'"):
        first._index = 1


def test_repr_and_inspection_do_not_serialize_posterior(monkeypatch):
    result = consume_posterior(posterior_bytes())

    def no_serialization(*args, **kwargs):
        raise AssertionError("full posterior serialization was not requested")

    monkeypatch.setattr(type(result), "to_dict", no_serialization)
    representation = repr(result)
    assert "sealed" in representation
    assert "array" not in representation
    assert len(representation) < 400
    assert "mean" not in result.inspect().details
    assert "draws" not in result.inspect().details
    assert "array" not in result._repr_html_()


def test_fresh_process_load_has_no_live_preparation_or_data(tmp_path):
    data = posterior_bytes()
    path = tmp_path / "posterior.bin"
    path.write_bytes(data)
    script = """
from pathlib import Path
from hashlib import sha256
import sys
import antecedent as ac
from antecedent.outcomes import require_success
data = Path(sys.argv[1]).read_bytes()
loaded = require_success(ac.load(data, expected={'expected_encoding_sha256': sha256(data).hexdigest()}))
assert loaded.study is None
assert not hasattr(loaded, '_prepared')
assert not hasattr(loaded, '_data')
assert loaded.quantity('ate').mean == 2.0
assert loaded.quantity('ate').draws.tolist() == [1.0, 2.0, 3.0]
assert loaded.acceptance.status == 'sealed'
assert not loaded.acceptance.replayable and not loaded.acceptance.resumable
assert bytes(loaded.export()) == data
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_retained_bayesian_result_exports_and_loads_without_refitting(monkeypatch):
    rng = np.random.default_rng(41)
    z = rng.normal(size=48)
    t = z + rng.normal(size=48)
    y = 2.0 * t + z + rng.normal(size=48)
    study = ac.prepare(
        {"t": t, "z": z, "y": y},
        query=ac.AverageEffect("t", "y"),
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        inference=ac.Bayesian(n_draws=32, backend="conjugate"),
        refute=False,
    )
    outcome = ac.estimate(study)
    result = require_success(outcome)

    def unexpected_refit(*args, **kwargs):
        raise AssertionError("posterior export or load attempted another fit")

    monkeypatch.setattr(type(study._engine), "estimate", unexpected_refit)
    exported = outcome.export(quantity="posterior")
    assert exported.kind == "causal_posterior"
    loaded = require_success(ac.load(exported, expected=exported.expectation))
    assert loaded.quantity("ate").mean == pytest.approx(result.posterior.effect_mean)
    assert loaded.n_draws == 32
    assert loaded.acceptance.status == "sealed"
    assert loaded.acceptance.identity_bound
    assert not loaded.acceptance.replayable
    assert not loaded.acceptance.resumable
    assert loaded.study is None
    assert bytes(loaded.export()) == bytes(exported)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"mean": []}, "summaries must match"),
        ({"sd": []}, "summaries must match"),
        ({"q025": []}, "summaries must match"),
        ({"q975": []}, "summaries must match"),
        ({"quantity_names": []}, "summaries must match"),
        ({"mean": [float("nan")]}, "summaries must be finite"),
        ({"sd": [float("inf")]}, "summaries must be finite"),
        ({"q025": [float("nan")]}, "summaries must be finite"),
        ({"q975": [float("inf")]}, "summaries must be finite"),
        ({"sd": [-0.1]}, "non-negative SD"),
        ({"q025": [3.0], "q975": [1.0]}, "ordered quantiles"),
        ({"n_draws": 0}, "draw count must be positive"),
        ({"n_draws": 2**32}, "n_draws exceeds u32"),
        ({"unidentified_mass": float("nan")}, "unidentified mass"),
        ({"unidentified_mass": 0.8, "subsampled_out_mass": 0.3}, "subsampled-out mass"),
        ({"backend_id": "  "}, "backend id"),
        ({"identification": "invented"}, "identification status"),
    ],
)
def test_moments_constructor_refuses_invalid_summaries(change, message):
    kwargs = dict(n_draws=3, mean=[2.0], sd=[1.0], q025=[1.0], q975=[3.0],
                  backend_id="conjugate", identification="NonparametricallyIdentified",
                  quantity_names=["ate"])
    kwargs.update(change)
    with pytest.raises(CausalValueError, match=message):
        _native.PosteriorArtifact.from_moments(**kwargs)


def test_moments_constructor_accepts_deterministic_and_skewed_summaries():
    # Zero SD is a legitimate point mass; a nonzero-SD mean need not lie
    # inside its central quantile interval for a skewed distribution.
    artifact = _native.PosteriorArtifact.from_moments(
        n_draws=3, mean=[2.0, 2.0], sd=[0.0, 10.0],
        q025=[2.0, 0.0], q975=[2.0, 1.0], backend_id="conjugate",
        identification="NonparametricallyIdentified", quantity_names=["ate", "skewed"],
    )
    assert np.isnan(artifact.hessian_condition)
    decoded = _native.decode_posterior_artifact(_native.encode_posterior_artifact(artifact))
    assert decoded.mean == [2.0, 2.0]
    assert decoded.sd == [0.0, 10.0]
    assert decoded.q025 == [2.0, 0.0]
    assert decoded.q975 == [2.0, 1.0]
    assert decoded.quantity_names == ["ate", "skewed"]
    assert len(decoded.draws) == 0


def test_moments_accept_zero_sd_with_distinct_tiny_quantiles():
    # A variance calculation can underflow to zero while values remain distinct.
    artifact = _native.PosteriorArtifact.from_moments(
        n_draws=2, mean=[0.0], sd=[0.0], q025=[-1e-200], q975=[1e-200],
        backend_id="conjugate", identification="NonparametricallyIdentified",
        quantity_names=["ate"],
    )
    decoded = _native.decode_posterior_artifact(_native.encode_posterior_artifact(artifact))
    assert decoded.sd == [0.0]
    assert decoded.q025 == [-1e-200]
    assert decoded.q975 == [1e-200]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"draws": [1.0, 2.0]}, "draws length"),
        ({"n_draws": 2}, "draws length"),
        ({"draws": [1.0, float("nan"), 3.0]}, "draws must be finite"),
        ({"draws": [1.0, 2.0, float("inf")]}, "draws must be finite"),
        ({"mean": [3.0]}, "disagree with the embedded draws"),
        ({"sd": [-1.0]}, "non-negative SD"),
        ({"q975": []}, "summaries must match"),
        ({"n_draws": 0}, "draw count must be positive"),
        ({"n_draws": 2**32}, "n_draws exceeds u32"),
    ],
)
def test_draw_constructor_refuses_invalid_metadata_and_draws(change, message):
    kwargs = dict(n_draws=3, mean=[2.0], sd=[1.0], q025=[1.05], q975=[2.95],
                  draws=[1.0, 2.0, 3.0], backend_id="conjugate",
                  identification="NonparametricallyIdentified", quantity_names=["ate"])
    kwargs.update(change)
    with pytest.raises(CausalValueError, match=message):
        _native.PosteriorArtifact(**kwargs)


def test_draw_constructor_accepts_valid_draws_and_explicit_empty_summary():
    kwargs = dict(n_draws=3, mean=[2.0], sd=[1.0], q025=[1.05], q975=[2.95],
                  backend_id="conjugate", identification="NonparametricallyIdentified",
                  quantity_names=["ate"])
    full = _native.PosteriorArtifact(**kwargs, draws=[1.0, 2.0, 3.0])
    roundtrip = _native.decode_posterior_artifact(_native.encode_posterior_artifact(full))
    np.testing.assert_array_equal(roundtrip.draws, [1.0, 2.0, 3.0])
    assert np.isnan(roundtrip.hessian_condition)
    summary = _native.PosteriorArtifact(**kwargs, draws=[])
    decoded = _native.decode_posterior_artifact(_native.encode_posterior_artifact(summary))
    assert len(decoded.draws) == 0
    assert decoded.mean == [2.0]


@pytest.mark.parametrize("name", ["coef_", "coef_x", "coef_-1", "coef_01", "coef_+1", "coef_4294967296"])
def test_posterior_encoding_refuses_invalid_reserved_coefficient_indices(name):
    artifact = _native.PosteriorArtifact.from_moments(
        n_draws=3, mean=[2.0], sd=[1.0], q025=[1.0], q975=[3.0],
        backend_id="conjugate", identification="NonparametricallyIdentified",
        quantity_names=[name],
    )
    with pytest.raises(CausalValueError, match="invalid indexed coefficient quantity"):
        _native.encode_posterior_artifact(artifact)


def test_posterior_encoding_preserves_indexed_coefficient_and_named_effect():
    artifact = _native.PosteriorArtifact.from_moments(
        n_draws=3, mean=[2.0, 3.0], sd=[1.0, 0.5], q025=[1.0, 2.0], q975=[3.0, 4.0],
        backend_id="conjugate", identification="NonparametricallyIdentified",
        quantity_names=["coef_7", "my_named_effect"],
    )
    decoded = _native.decode_posterior_artifact(_native.encode_posterior_artifact(artifact))
    assert decoded.quantity_names == ["coef_7", "my_named_effect"]
    assert decoded.mean == [2.0, 3.0]
