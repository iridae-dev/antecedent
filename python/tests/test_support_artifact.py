"""Finite coordinate support: ordered per-coordinate labels survive a fresh reader."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap

import antecedent
import numpy as np
from antecedent import external

_ORDER = (
    "supported",
    "weak_overlap",
    "extrapolative",
    "outside_empirical_support",
    "missing_evidence",
)
_GRID = [0.0, 1.0, 2.0]
_EDGES = [("x", "a"), ("x", "y"), ("a", "y")]
_NAMES = ["x", "a", "y"]


def _worst(labels):
    return max(labels, key=_ORDER.index)


def _spec(**kwargs):
    ident = antecedent.identify(
        graph=_EDGES, names=_NAMES, query=antecedent.ResponseCurve("a", "y", grid=_GRID)
    )
    return external.response(ident, outcome_units="mmHg", population="target", **kwargs)


def _provider():
    return external.ProviderObject(
        provider_id="lab",
        object_id="curve",
        version="v3",
        snapshot="snap-9",
        request="req-1",
        meaning="interventional_predictive",
        capabilities=("mean",),
    )


def _bind(spec, support):
    return spec.bind(
        external.Response(
            provider=_provider(),
            values=[1.0, 3.0, 5.0],
            evidence=("factor:z",),
            assumptions=("ignorability",),
            attested_by="lab",
            support=support,
        )
    )


def _premised_spec():
    return _spec(require_evidence=("factor:z",), require_assumptions=("ignorability",))


def test_three_ordered_doses_exclude_only_the_middle_and_keep_order_through_export():
    declared = ("supported", "weak_overlap", "supported")
    spec = _premised_spec()
    claim = _bind(spec, declared)
    assert claim.support == declared
    assert [q.regime_id for q in spec.quantities] == ["do(a=0)", "do(a=1)", "do(a=2)"]
    # Only the middle dose is non-supported; the summary is the worst label.
    assert [i for i, label in enumerate(claim.support) if label != "supported"] == [1]
    assert claim.support_status == "weak_overlap"
    assert claim.support_status == _worst(claim.support)

    reloaded = spec.load(claim.export(), expected_identity=claim.identity)
    assert reloaded.support == declared
    assert reloaded.support_status == "weak_overlap"
    assert np.allclose(reloaded.values, [1.0, 3.0, 5.0])

    # Relabeling: moving the weak dose moves its label, never the summary.
    moved = _bind(_premised_spec(), ("weak_overlap", "supported", "supported"))
    assert moved.support == ("weak_overlap", "supported", "supported")
    assert moved.support_status == "weak_overlap"


def test_missing_evidence_is_not_zero_overlap_and_ranks_weakest():
    undeclared = _bind(_premised_spec(), None)
    assert undeclared.support == ("missing_evidence",) * 3
    assert undeclared.support_status == "missing_evidence"

    zero_overlap = _bind(_premised_spec(), ("weak_overlap",) * 3)
    outside = _bind(_premised_spec(), ("outside_empirical_support",) * 3)
    assert zero_overlap.support == ("weak_overlap",) * 3
    assert outside.support == ("outside_empirical_support",) * 3
    assert undeclared.support != zero_overlap.support
    assert undeclared.support != outside.support
    assert zero_overlap.support_status != undeclared.support_status
    assert outside.support_status != undeclared.support_status

    # Declared labels never collapse into missing evidence, and a mixed summary
    # is the worst label under SupportStatus severity (missing_evidence last).
    mixed = _bind(_premised_spec(), ("supported", "missing_evidence", "outside_empirical_support"))
    assert mixed.support == ("supported", "missing_evidence", "outside_empirical_support")
    assert mixed.support_status == "missing_evidence"
    assert _ORDER.index("missing_evidence") > _ORDER.index("outside_empirical_support")
    declared_only = _bind(
        _premised_spec(), ("supported", "weak_overlap", "outside_empirical_support")
    )
    assert declared_only.support_status == "outside_empirical_support"
    assert "missing_evidence" not in declared_only.support


_READER = textwrap.dedent(
    """
    import json, sys
    import antecedent

    loaded = antecedent.artifacts.loads(open(sys.argv[1], "rb").read())

    def find(node, key, out):
        if isinstance(node, dict):
            for name, value in node.items():
                if name == key:
                    out.append(value)
                find(value, key, out)
        elif isinstance(node, (list, tuple)):
            for item in node:
                find(item, key, out)
        return out

    found = {}
    for key in ("point_status", "grid", "scope", "id"):
        found[key] = find(loaded.payload, key, []) + find(loaded.contract, key, [])
    print(json.dumps({
        "kind": loaded.payload_kind,
        "names": list(loaded.variable_names),
        "found": found,
    }))
    """
)


def test_static_response_artifact_is_readable_by_a_fresh_process_with_its_coordinates():
    rng = np.random.default_rng(17)
    treatment = rng.normal(size=400)
    outcome = 2.0 * treatment + rng.normal(scale=0.2, size=400)
    grid = [-0.5, 0.0, 0.5]
    result = antecedent.analyze(
        {"a": treatment, "y": outcome},
        query=antecedent.ResponseCurve("a", "y", grid=grid),
        graph=[("a", "y")],
    )
    labels = list(result.support.point_status)
    assert labels == ["supported"] * 3
    by_id = {d.id: d for d in result.support.diagnostics}
    assert by_id["response.local_ess"].scope == "per_coordinate"
    assert len(by_id["response.local_ess"].values) == len(grid)

    data = result.export()
    with tempfile.NamedTemporaryFile(suffix=".bin") as handle:
        handle.write(data)
        handle.flush()
        done = subprocess.run(
            [sys.executable, "-c", _READER, handle.name],
            capture_output=True,
            text=True,
            check=False,
        )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report["kind"] == "analysis_result"
    assert report["names"] == ["a", "y"]
    found = report["found"]
    # The reader exposes the canonical artifact, not a typed support report; the
    # per-point labels, the ordered grid and the diagnostic scopes must be carried
    # and must equal the producer's, not merely be absent.
    assert found["point_status"] == [labels]
    assert {"values": grid} in found["grid"]
    assert [float(v) for v in next(g for g in found["grid"] if isinstance(g, list))] == grid
    assert "response.local_ess" in found["id"]
    assert {"per_coordinate", "global"} <= {s for s in found["scope"] if isinstance(s, str)}
    produced = {d.id: d.scope for d in result.support.diagnostics}
    assert produced["response.local_ess"] == "per_coordinate"
    assert produced["response.outcome_tail_ratio"] == "global"
