# From a notebook to a scientific system

The same specialized scientific objects support exploration and durable consumption. Preparation establishes a scientific question and retains native identification; estimation produces a frozen execution; export preserves bytes and independently retained expectations.

```python
import antecedent as ant

study = ant.prepare(data, graph=edges,
                    query=ant.AverageEffect("treatment", "outcome"),
                    bootstrap=0, refute=False)
study.identification       # cached identification, assumptions and derivation
study.inspect_plan()       # preparation; no fit has run
result = study.estimate()
result.inspect()           # answer, standing, assumptions and limitations
checked = result.refute(suite="cheap")
updated = study.refresh(new_observations)
revised = study.reprepare(new_design_data)
```

Named design columns may be omitted during observation updates. Supplying changed assignments or design IDs requires repreparation. A result always refers to its original execution; `result.refute()` remains attached to that execution after later study refreshes.

Persist a result as bytes and retain its expected identities in a separate trusted record:

```python
artifact = checked.artifact()
artifact_path.write_bytes(artifact.data)
expected = artifact.expectation

# In an independent consumer, obtain expected from the retained trusted record.
loaded = ant.load(artifact_path.read_bytes(), expected=expected)
loaded.answer
loaded.acceptance          # identity binding, replayability, resumability
```

Specialized producers use the same persistence protocol and return their own typed consumed results. Metadata inspection uses `ant.artifacts.inspect(...)`; loading never runs a callback or recreates a live study. Bundles accept scientific bytes, envelopes and supported exporting objects.

For composition, pass a complete compatible mean producer directly to a decision or inverse query. If its scientific coordinates are incomplete, use `producer.as_mean_source(quantity=...)` with explicit units, population, regime and functional meaning. This adapter preserves a mean's standing; additional laws or joint uncertainty still require their existing declarations and licenses.

Run `examples/python/notebook_to_system_24.py` for a complete reproducible journey with synthetic data, deferred refutation, guarded observation refresh, explicit repreparation, composition and fresh-process consumption. Its companion is `examples/notebooks/notebook_to_system_24.ipynb`. See the [2.4 migration guide](../migrations/2.4-python-ux.md) for changed interfaces.
