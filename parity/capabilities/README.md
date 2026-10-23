# Public route manifest (B02)

`route_manifest.toml` is the 2.4.1 public-route inventory. Geometric cells are grouped by query and graph class. One licensed coordinate is not one record. The coordinate vocabulary stays in `parity/support_axes.toml`; cell licenses stay in `support_licensed.toml`, `support_closed.toml`, and `support_n_a.toml`.

Graphless design licenses are one record each, from `support_graphless.toml`. They do not inherit a geometric cell.

Programmable intervention and mechanism selection are not geometric licenses:

- `gcm.model.intervention_programs` executes policy simulation and counterfactual replay on a fitted model. A hard intervention can fold into an identified `AverageEffect` or `ResponseCurve`. A transform or a non-constant assignment stays simulable and is refused as an identified policy route. No mechanism is fit on that route.
- `gcm.model.mechanism_selection` fits and scores candidate mechanism families on held-out splits. That fit is not a causal identification claim and does not publish a calibrated interval for a response or effect. The manifest marks it `fitting_gap` for that missing inferential guarantee.
- `discovery.static`, `discovery.temporal`, `gcm.attribute` and `gcm.simulate` are the compiled `Discover`, `Attribute` and `Simulate` routes. They execute with a lowered program identity, contract and claim, and are `execution_only`: explicitly unlicensed, with no coverage record, no support-matrix cell and no `licensed_routes.toml` row.

`decision_ports` and `artifact_consumer` are `unlisted` where the support files do not name them. That is not a `not_applicable` license.
