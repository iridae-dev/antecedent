"""External scientific results bound to an identified causal contract."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap

import antecedent as ac
import numpy as np
import pytest
from antecedent import external
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.extensibility import ProviderTrust

GRID = [0.0, 1.0, 2.0]
EDGES = [("x", "a"), ("x", "y"), ("a", "y")]
NAMES = ["x", "a", "y"]


def _spec(**kwargs):
    ident = ac.identify(graph=EDGES, names=NAMES, query=ac.ResponseCurve("a", "y", grid=GRID))
    return external.response(ident, outcome_units="mmHg", population="target", **kwargs)


def _provider(**kwargs):
    fields = {
        "provider_id": "lab",
        "object_id": "curve",
        "version": "v3",
        "snapshot": "snap-9",
        "request": "req-1",
        "meaning": "interventional_predictive",
        "capabilities": ("mean",),
    }
    return external.ProviderObject(**{**fields, **kwargs})


def _response(**kwargs):
    # Closed form: E[Y | do(a)] = 1 + 2a.
    fields = {
        "provider": _provider(),
        "values": [1.0, 3.0, 5.0],
        "evidence": ("factor:z",),
        "assumptions": ("ignorability",),
        "attested_by": "lab",
    }
    return external.Response(**{**fields, **kwargs})


def _spec_with_premises():
    return _spec(require_evidence=("factor:z",), require_assumptions=("ignorability",))


def test_coordinates_come_from_the_identified_query_and_units_are_never_inferred():
    spec = _spec()
    assert [q.regime_id for q in spec.quantities] == ["do(a=0)", "do(a=1)", "do(a=2)"]
    assert {q.units for q in spec.quantities} == {"mmHg"}
    assert spec.identification == "nonparametrically_identified"
    ident = ac.identify(graph=EDGES, names=NAMES, query=ac.ResponseCurve("a", "y", grid=GRID))
    with pytest.raises(CausalValueError, match="outcome_units"):
        external.response(ident)


def test_bound_claim_is_inspectable_exportable_and_never_native():
    claim = _spec_with_premises().bind(_response())
    assert np.allclose(claim.values, [1.0, 3.0, 5.0])
    assert claim.native is False
    assert claim.trust is ProviderTrust.EXTERNALLY_ATTESTED
    assert claim.trust is not ProviderTrust.NATIVE_LICENSED
    # Provider declared no support, so it is missing evidence, not supported.
    assert claim.support == ("missing_evidence",) * 3
    assert claim.support_status == "missing_evidence"
    assert claim.uncertainty_method is None
    assert claim.provenance_label == "external:lab/curve@v3#snap-9"
    assert "not estimated natively" in claim.claim()
    assert {"causal_contract", "evidence", "external_provider"} <= claim.stages_behind()
    inspection = claim.inspect()
    assert inspection.native is False
    assert [link.id for link in inspection.lineage][-1] == "claim"
    assert "trust: externally_attested" in str(inspection)


def test_verified_trust_is_recomputed_from_independent_probes():
    probes = tuple(
        external.VerificationProbe(kind, 1.0, 1.0, 0.0)
        for kind in ("shape", "support", "moments", "known_truth")
    )
    provider = _provider(capabilities=("mean",))
    claim = _spec().bind(_response(provider=provider, attested_by=None, probes=probes))
    assert claim.trust is ProviderTrust.VERIFIED_EXTENSION
    assert claim.trust is not ProviderTrust.NATIVE_LICENSED

    with pytest.raises(CausalUnsupportedError) as missing:
        _spec().bind(_response(attested_by=None, probes=probes[:2]))
    assert missing.value.reason_code == "external_verification_failed"
    assert missing.value.detail == "external_object_verification.missing_probe"
    assert missing.value.offending == "known_truth"

    failing = (*probes[:3], external.VerificationProbe("known_truth", 1.5, 1.0, 1e-9))
    with pytest.raises(CausalUnsupportedError) as failed:
        _spec().bind(_response(attested_by=None, probes=failing))
    assert failed.value.detail == "external_object_verification.failed_probe"


def test_each_mismatch_refuses_with_a_registered_code_and_structured_fields():
    spec = _spec_with_premises()
    wrong_units = tuple(
        dataclasses.replace(q, units="kPa") if i == 1 else q for i, q in enumerate(spec.quantities)
    )
    with pytest.raises(CausalUnsupportedError) as units:
        spec.bind(_response(quantities=wrong_units))
    assert units.value.reason_code == "quantity_semantics_mismatch"
    assert units.value.detail == "external_response_binding.coordinate_units"
    assert (units.value.offending, units.value.expected, units.value.supplied) == (
        "coordinate[1]",
        "mmHg",
        "kPa",
    )
    assert units.value.stage == "bind"

    with pytest.raises(CausalUnsupportedError) as graph:
        spec.bind(_response(graph_id="graph:other"))
    assert graph.value.reason_code == "external_binding_mismatch"
    assert graph.value.detail == "external_response_binding.graph"

    with pytest.raises(CausalUnsupportedError) as evidence:
        spec.bind(_response(evidence=()))
    assert evidence.value.offending == "factor:z"
    assert evidence.value.remedy

    with pytest.raises(CausalUnsupportedError) as short:
        spec.bind(_response(values=[1.0, 3.0]))
    assert short.value.detail == "external_response_binding.dimension"

    with pytest.raises(CausalUnsupportedError) as unattested:
        spec.bind(_response(attested_by=None))
    assert unattested.value.reason_code == "invalid_argument"

    unidentified = dataclasses.replace(spec, identification="not_identified")
    with pytest.raises(CausalUnsupportedError) as notid:
        unidentified.bind(_response())
    assert notid.value.reason_code == "effect_not_identified"

    with pytest.raises(CausalUnsupportedError) as meaning:
        spec.bind(_response(provider=_provider(meaning="posterior_predictive")))
    assert meaning.value.reason_code == "distribution_meaning_mismatch"
    assert meaning.value.expected == "InterventionalPredictive"
    assert meaning.value.supplied == "PosteriorPredictive"


def test_observational_law_is_refused_without_a_checked_equivalence():
    # One requested coordinate: do(a=1). The provider offers the observational law.
    full = _spec_with_premises()
    spec = dataclasses.replace(full, quantities=(full.quantities[1],))
    observational = (dataclasses.replace(spec.quantities[0], regime_id=external.OBSERVATIONAL),)
    offered = {
        "provider": _provider(),
        "quantities": observational,
        "values": [3.0],
    }
    with pytest.raises(CausalUnsupportedError) as refused:
        spec.bind(_response(**offered))
    assert refused.value.detail == "external_response_binding.unchecked_observational_law"
    assert refused.value.remedy

    other_graph = external.Equivalence("graph:other", "do(a=1)", "backdoor:x")
    with pytest.raises(CausalUnsupportedError):
        dataclasses.replace(spec, equivalences=(other_graph,)).bind(_response(**offered))

    checked = dataclasses.replace(
        spec, equivalences=(external.Equivalence(spec.graph_id, "do(a=1)", "backdoor:x"),)
    )
    claim = checked.bind(_response(**offered))
    assert "transformation" in claim.stages_behind()
    assert "equivalence:backdoor:x" in [link.id for link in claim.lineage]


def test_observational_conditional_law_over_a_dose_grid_binds_under_its_equivalence():
    base = _spec_with_premises()
    offered = base.observational_quantities()
    assert [q.conditioning[0].value_id for q in offered] == ["0", "1", "2"]
    assert {q.regime_id for q in offered} == {external.OBSERVATIONAL}

    with pytest.raises(CausalUnsupportedError) as unlicensed:
        base.bind(_response(quantities=offered))
    # Without any equivalence the first observed difference is the conditioning.
    assert unlicensed.value.detail == "external_response_binding.coordinate_conditioning"
    assert (unlicensed.value.expected, unlicensed.value.supplied) == ("", "a=0")
    assert unlicensed.value.offending == "coordinate[0]"
    assert "checked equivalence" in unlicensed.value.remedy

    spec = dataclasses.replace(base, equivalences=(base.observational_equivalence("backdoor:x"),))
    claim = spec.bind(_response(quantities=offered))
    assert np.allclose(claim.values, [1.0, 3.0, 5.0])
    # The bound claim answers the interventional request, not the conditional one.
    assert [q.regime_id for q in claim.quantities] == ["do(a=0)", "do(a=1)", "do(a=2)"]
    assert all(not q.conditioning for q in claim.quantities)
    assert "equivalence:backdoor:x" in [link.id for link in claim.lineage]
    assert "transformation" in claim.stages_behind()

    # A dose mapped to the wrong intervention is not licensed.
    crossed = external.Equivalence.conditioned(
        spec.graph_id, "a", {"0": "do(a=0)", "1": "do(a=2)", "2": "do(a=1)"}, "backdoor:x"
    )
    with pytest.raises(CausalUnsupportedError) as wrong:
        dataclasses.replace(spec, equivalences=(crossed,)).bind(_response(quantities=offered))
    assert wrong.value.detail == "external_response_binding.unchecked_observational_law"
    assert wrong.value.offending == "coordinate[1]"

    # Nothing else about the coordinate is rescued by the licence.
    wrong_population = (
        offered[0],
        dataclasses.replace(offered[1], population_id="source"),
        offered[2],
    )
    with pytest.raises(CausalUnsupportedError) as population:
        spec.bind(_response(quantities=wrong_population))
    assert population.value.reason_code == "quantity_semantics_mismatch"
    assert population.value.detail == "external_response_binding.coordinate_population"


def test_equivalence_names_exactly_one_scope():
    with pytest.raises(CausalValueError, match="exactly one scope"):
        external.Equivalence("g", None, "check")
    with pytest.raises(CausalValueError, match="exactly one scope"):
        external.Equivalence("g", "do(a=1)", "check", "a", {"1": "do(a=1)"})
    with pytest.raises(CausalValueError, match="go together"):
        external.Equivalence("g", "do(a=1)", "check", None, {"1": "do(a=1)"})
    custom = external.response(
        ac.identify(graph=EDGES, names=NAMES, query=ac.ResponseCurve("a", "y", grid=GRID)),
        quantities=_spec().quantities,
    )
    with pytest.raises(CausalValueError, match="dose grid"):
        custom.observational_equivalence("check")


def test_unknown_capability_names_are_refused_not_ignored():
    with pytest.raises(CausalUnsupportedError) as unknown:
        _spec().bind(_response(provider=_provider(capabilities=("mean", "teleport"))))
    assert unknown.value.reason_code == "invalid_argument"
    assert unknown.value.detail == "external_response_binding.malformed_capability"


def test_fresh_process_consumer_loads_only_under_its_own_identity():
    spec = _spec_with_premises()
    claim = spec.bind(_response())
    data = claim.export()
    assert spec.load(data, expected=claim.identity).values.tolist() == [1.0, 3.0, 5.0]
    changed = {**claim.identity, "snapshot_id": "other-snapshot"}
    with pytest.raises(Exception, match="differs"):
        spec.load(data, expected=changed)

    script = textwrap.dedent(
        """
        import json, sys
        import numpy as np
        import antecedent as ac
        from antecedent import external

        ident = ac.identify(
            graph=[("x", "a"), ("x", "y"), ("a", "y")],
            names=["x", "a", "y"],
            query=ac.ResponseCurve("a", "y", grid=[0.0, 1.0, 2.0]),
        )
        spec = external.response(ident, outcome_units="mmHg", population="target")
        expected = json.loads(sys.argv[2])
        claim = spec.load(open(sys.argv[1], "rb").read(), expected=expected)
        v = claim.values
        assert abs((v[2] - v[0]) / 2.0 - 2.0) < 1e-12 and abs(v[0] - 1.0) < 1e-12
        assert claim.native is False
        assert claim.provenance_label == "external:lab/curve@v3#snap-9"
        """
    )
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".bin") as handle:
        handle.write(data)
        handle.flush()
        done = subprocess.run(
            [sys.executable, "-c", script, handle.name, json.dumps(claim.identity)],
            capture_output=True,
            text=True,
            check=False,
        )
    assert done.returncode == 0, done.stderr
