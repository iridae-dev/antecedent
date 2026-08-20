"""PosteriorArtifact zero-copy / copy contract for np.asarray."""

from __future__ import annotations

import gc

import numpy as np
import pytest

pytest.importorskip("antecedent")
from antecedent._native import (
    PosteriorArtifact,
    decode_posterior_artifact,
    encode_posterior_artifact,
)
from antecedent.results._views import PosteriorView


def _artifact_with_draws() -> PosteriorArtifact:
    draws = [1.0, 2.0, 3.0, 4.0]
    return PosteriorArtifact(
        n_draws=4,
        mean=[2.5],
        sd=[1.0],
        q025=[1.0],
        q975=[4.0],
        draws=draws,
        backend_id="conjugate",
        identification="identified",
        quantity_names=["ate"],
    )


def test_asarray_default_is_readonly_view():
    art = _artifact_with_draws()
    arr = np.asarray(art)
    assert arr.dtype == np.float64
    assert arr.shape == (4,)
    assert np.allclose(arr, [1.0, 2.0, 3.0, 4.0])
    assert not arr.flags.writeable
    with pytest.raises((ValueError, BufferError)):
        arr[0] = 99.0


def test_asarray_copy_true_is_independent():
    art = _artifact_with_draws()
    view = np.asarray(art, copy=False)
    owned = np.asarray(art, copy=True)
    assert np.allclose(view, owned)
    assert owned.flags.writeable
    owned[0] = -1.0
    assert np.asarray(art)[0] == 1.0


def test_asarray_keeps_artifact_alive_after_name_drop():
    art = _artifact_with_draws()
    arr = np.asarray(art)
    del art
    gc.collect()
    # Buffer must remain valid while `arr` holds the PyBuffer object ref.
    assert float(arr.sum()) == 10.0


def test_asarray_copy_false_dtype_cast_errors():
    art = _artifact_with_draws()
    with pytest.raises((ValueError, TypeError)):
        np.asarray(art, dtype=np.float32, copy=False)


def test_asarray_default_dtype_cast_copies_if_needed():
    art = _artifact_with_draws()
    cast = np.asarray(art, dtype=np.float32)
    assert cast.dtype == np.float32
    assert np.allclose(cast, [1.0, 2.0, 3.0, 4.0])


def test_asarray_copy_true_dtype_cast_ok():
    art = _artifact_with_draws()
    cast = np.asarray(art, dtype=np.float32, copy=True)
    assert cast.dtype == np.float32
    assert np.allclose(cast, [1.0, 2.0, 3.0, 4.0])
    assert cast.flags.writeable


def test_posterior_view_forwards_copy():
    art = _artifact_with_draws()
    encoded = encode_posterior_artifact(art)
    view = PosteriorView(
        effect_mean=2.5,
        effect_sd=1.0,
        q025=1.0,
        q975=4.0,
        n_draws=4,
        p_below_zero=0.0,
        backend="conjugate",
        artifact=encoded,
    )
    owned = np.asarray(view, copy=True)
    assert owned.flags.writeable
    assert np.allclose(owned, [1.0, 2.0, 3.0, 4.0])
    # Round-trip decode still matches (copy did not mutate artifact bytes).
    decoded = decode_posterior_artifact(encoded)
    assert np.allclose(np.asarray(decoded, copy=True), owned)
    with pytest.raises((ValueError, TypeError)):
        np.asarray(view, dtype=np.float32, copy=False)
