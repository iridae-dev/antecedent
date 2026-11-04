"""Result-level method provenance through the Python surface.

A backdoor AIPW outcome names the records of the route it executed (``identify.backdoor``,
``estimate.aipw``) with their documented deviations, and a study's bibliography lists each
source once. The record table and the de-duplication are the Rust implementation's; the Rust
route test is ``crates/antecedent/tests/aipw_ate_route_evidence.rs``.
"""

from __future__ import annotations

import antecedent as ant
import numpy as np
from antecedent.citations import bibliography, capability_provenance, method_provenance
from antecedent.estimators import Aipw

QUERY = ant.AverageEffect("t", "y")
GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]


def confounded(seed: int = 5, n: int = 300) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    t = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(float)
    return {"z": z, "t": t, "y": 2.0 * t + z + rng.normal(size=n)}


def test_aipw_outcome_cites_the_executed_route() -> None:
    outcome = ant.analyze(confounded(), graph=GRAPH, query=QUERY, estimator=Aipw(bootstrap=0))
    provenance = outcome.method_provenance
    assert provenance.unlinked == ()
    assert [record.id for record in provenance.records] == ["identify.backdoor", "estimate.aipw"]
    aipw = provenance.record("estimate.aipw")
    assert aipw is not None
    assert aipw.cited_for == (("estimator", "aipw"),)
    assert len(aipw.digest) == 64
    assert any(text.startswith("no cross-fitting on the ATT/ATC") for text in aipw.deviations)
    assert outcome.value.method_provenance == provenance

    citations = outcome.citations()
    dml = next(c for c in citations if c.source.doi == "10.1111/ectj.12097")
    assert dml.source.doi_verified
    assert dml.source.year == 2018
    assert ("estimate.aipw", aipw.digest) in dml.records
    assert any(c.kind == "deviation" and "not inherited" in c.text for c in dml.caveats)
    assert dml.reference.endswith("https://doi.org/10.1111/ectj.12097")


def test_bibliography_lists_a_shared_source_once() -> None:
    aipw = method_provenance("backdoor.adjustment", "aipw")
    linear = method_provenance("backdoor.adjustment", "linear.adjustment.ate")
    merged = bibliography([aipw, linear])
    keys = [c.source.doi or c.source.title for c in merged]
    assert len(keys) == len(set(keys))
    assert len(merged) < len(aipw.citations()) + len(linear.citations())


def test_unknown_ids_are_unlinked_and_capabilities_resolve() -> None:
    unknown = method_provenance(estimator="no.such.estimator")
    assert unknown.records == ()
    assert unknown.unlinked == (("estimator", "no.such.estimator"),)
    capability = capability_provenance("adaptive_inference.group_sequential")
    assert capability.unlinked == ()
    assert capability.records[0].id == "adaptive_inference.group_sequential"
    assert capability.records[0].cited_for[0] == (
        "capability",
        "adaptive_inference.group_sequential",
    )
