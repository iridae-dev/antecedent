"""Bulk snapshots preserve legacy scientific inputs and independent artifacts."""

import dataclasses
import gc
import json

import numpy as np
import pytest
from antecedent import transport as tr
from antecedent.cyclic import PreparedCyclicSystem, consume_artifact
from antecedent.transport import advanced


def legacy_data(data):
    """Exercise the retained native JSON compatibility entry point."""

    class Legacy(type(data)):
        def __getattribute__(self, name):
            if name == "_buffers":
                raise AttributeError(name)
            return super().__getattribute__(name)

    return Legacy(**{f.name: getattr(data, f.name) for f in dataclasses.fields(data)})


def strided(values):
    array = np.asarray(values)
    backing = np.empty(len(array) * 2, dtype=array.dtype)
    backing[::2] = array
    return backing[::2]


@pytest.mark.parametrize("route", ["trial", "continuous", "smoothed"])
def test_typed_transport_matches_json_and_owns_input(route):
    if route == "trial":
        from test_learned_transport import trial_fixture

        query, data = trial_fixture()

        def prepare(value):
            return advanced.prepare(query, value, inference=tr.TransportInference(0))
    elif route == "continuous":
        from test_learned_continuous_transport import fixture, options

        query, data = fixture()

        def prepare(value):
            return advanced.prepare_learned_continuous(query, value, options=options(), seed=5)
    else:
        from test_smoothed_dose_transport import fixture, options

        query, data = fixture(n_trial=600, n_target=400)

        def prepare(value):
            return advanced.prepare_smoothed_dose(query, value, options=options(), seed=5)

    reference = prepare(legacy_data(data))
    expected = reference.estimate()
    columns = {
        field.name: strided(getattr(data, field.name))
        for field in dataclasses.fields(data)
        if field.name not in ("sampling", "covariates")
    }
    columns["covariates"] = {name: strided(values) for name, values in data.covariates.items()}
    owned = dataclasses.replace(data, **columns)
    prepared = prepare(owned)
    for column in columns.values():
        if isinstance(column, np.ndarray):
            column[:] = 0
        elif isinstance(column, dict):
            for values in column.values():
                values[:] = 0
    del owned, columns
    gc.collect()
    actual = prepared.estimate()
    if route == "trial":
        assert actual.export() == expected.export()
    else:
        assert prepared.export() == reference.export()


def test_cyclic_buffers_match_legacy_report_artifact_and_request():
    from test_cyclic import model

    noise = np.tile([0.1, 99.0, 0.2, 99.0, -0.1, 99.0, -0.2, 99.0], 16).reshape(32, 4)[:, ::2]
    # A genuinely strided two-dimensional disturbance matrix.
    assert not noise.flags.c_contiguous
    prepared = model().prepare([[], [{"variable": 0, "value": 2.0}]], noise, factual=[0.3, 0.7])
    request = prepared.request_json
    assert isinstance(prepared.request_json, str)
    legacy = PreparedCyclicSystem(request)
    noise[:] = 100.0
    del noise
    gc.collect()
    actual, expected = prepared.execute(), legacy.execute()
    assert actual.report == expected.report
    assert actual.export() == expected.export()
    assert (
        consume_artifact(actual.export(), expected_identity=actual.identity).report == actual.report
    )
    assert json.loads(request) == json.loads(prepared.request_json)
    retargeted = prepared.retarget([[]], [[0.2, 0.1]])
    assert (
        retargeted.execute().report
        == PreparedCyclicSystem(retargeted.request_json).execute().report
    )


@pytest.mark.parametrize("noise", [[[1.0]], [[]], [[float("nan"), 0.0]], [[1.0, 2.0, 3.0]]])
def test_cyclic_buffers_reject_invalid_rows(noise):
    from test_cyclic import model

    with pytest.raises(ValueError, match="shape|finite|dimension|columns"):
        model().prepare([[]], noise)


@pytest.mark.parametrize("functional", ["level", "derivative", "contrast"])
def test_dose_parallel_and_memory_limited_replay(functional):
    from antecedent import _native
    from antecedent.dose_grid import consume_dose_grid_artifact, dose_functional
    from antecedent.errors import CausalCancelledError, CausalResourceError

    dose = np.linspace(0.0, 4.0, 2049)
    outcome = 1.0 + dose / 2.0 + dose**2 / 4.0 + 0.01 * np.sin(np.arange(len(dose)))
    kwargs = dict(bandwidth=0.4, bandwidth_range=(0.1, 1.0), functional=functional)
    if functional == "contrast":
        kwargs["contrast"] = (1.0, 3.0)
    else:
        kwargs["grid"] = np.linspace(0.5, 3.5, 33).tolist()
    serial = dose_functional(dose, outcome, threads=1, **kwargs)
    parallel = dose_functional(dose, outcome, threads=4, **kwargs)
    assert parallel.to_dict() == serial.to_dict()
    assert parallel.export() == serial.export()
    limited = dose_functional(dose, outcome, threads=4, max_bytes=len(dose) * 128 + 8192, **kwargs)
    assert limited.to_dict() == serial.to_dict()
    assert consume_dose_grid_artifact(parallel.export(), threads=4).to_dict() == serial.to_dict()
    with pytest.raises(CausalResourceError, match="memory|resource|budget"):
        dose_functional(dose, outcome, threads=4, max_bytes=1, **kwargs)
    token = _native.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError, match="cancel"):
        dose_functional(dose, outcome, threads=4, cancel=token, **kwargs)


def test_cyclic_refresh_buffers_match_json_refit_and_own_rows():
    from antecedent.cyclic import Equation, EquilibriumModel

    model = EquilibriumModel(
        (Equation("a", 0.0, (0.0, 0.4), "identity"), Equation("b", 0.0, (0.5, 0.0), "identity")),
        (1.0, 0.0, 0.0, 1.0),
    )
    rows = [
        {"values": [1.0, 0.7], "clamps": [{"variable": 0, "value": 1.0}]},
        {"values": [-1.0, -0.7], "clamps": [{"variable": 0, "value": -1.0}]},
    ]
    prepared = model.prepare(
        [[{"variable": 0, "value": 2.0}]],
        [[0.0, 0.0]],
        parameters=[{"kind": "weight", "equation": 1, "parent": 0}],
        rows=rows,
    )
    replacement = [
        {**row, "values": strided([row["values"][0], row["values"][1] / 2])} for row in rows
    ]
    refreshed = prepared.refresh(replacement)
    assert refreshed.inspect()["fits_performed"] == 0
    legacy_request = json.loads(prepared.request_json)
    legacy_request["fit"]["rows"] = [
        {**row, "values": row["values"].tolist()} for row in replacement
    ]
    expected = PreparedCyclicSystem(json.dumps(legacy_request)).execute()
    for row in replacement:
        row["values"][:] = 0.0
    actual = refreshed.execute()
    assert refreshed.inspect()["fits_performed"] == 1
    assert actual.report == expected.report
    assert actual.export() == expected.export()
    assert prepared.execute().report["worlds"][0][0]["values"][1] == pytest.approx(1.4, abs=1e-4)
    assert actual.report["worlds"][0][0]["values"][1] == pytest.approx(0.7, abs=1e-4)


def test_trial_compact_output_preserves_full_native_compatibility_and_artifact():
    from antecedent.transport._impl.trial import _learned_trial

    from test_learned_transport import trial_fixture

    query, data = trial_fixture()
    study = advanced.prepare(query, data, inference=tr.TransportInference(5, seed=8))
    native = study._native
    compact = native.estimate_summary()
    compact_result = _learned_trial(native, compact)
    full = native.estimate()
    full_result = _learned_trial(native, full)
    assert compact_result.to_dict() == full_result.to_dict()
    assert compact_result.export() == full_result.export()
    assert len(full_result.replicates) == 5
    assert native.last_result_summary() == compact
    full_data = json.loads(full)
    assert {"membership", "mu0", "mu1", "scores"} <= full_data.keys()
    assert not ({"membership", "mu0", "mu1", "scores"} & json.loads(compact).keys())
    refreshed = native.refresh_summary(data)
    assert _learned_trial(native, refreshed).export() == full_result.export()
