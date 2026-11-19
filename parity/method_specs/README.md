# Method specifications

`specs.toml` holds one record per child in `parity/method_children/children.toml`. Each carries the identifying result, assumptions, sampling units, estimator or posterior construction, diagnostics and failure policy of its method, and the links that tie it to the catalog and the clean-room review. It records no support status; that comes only from the support model.

`python3 scripts/reconcile_method_specs.py` completes and checks the records. `--check` fails on a record whose structured fields (parent, route, provenance links, API review) differ from the derived ones, on a provenance record that does not exist or is not clean-room, on an unreviewed public surface, and on a scientific field left `unlisted`. It does not read the text of the scientific fields.

| Field | Meaning |
| --- | --- |
| `parent`, `route`, `target` | Follow the child. A child with no route keeps the route its record already names, or `none`. |
| `provenance`, `api_review` | The clean-room provenance records that cover the child's implementing module (or the reason the module implements no algorithm, or that it has no module) and whether the module's public surface is in the reviewed API snapshot. `scripts/gate_provenance_schema.py` fails a method with an implementing module that has neither. |
| `identifying_result`, `assumptions`, `sampling_units`, `estimator`, `diagnostics`, `failure_policy` | Hand-written scientific text, never replaced. A field left `unlisted` is filled from the module's own header when the records are rewritten. |
