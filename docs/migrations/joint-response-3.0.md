# Joint-response contracts in 3.0

`ContrastTerm::coefficient` is now `f64`. Existing differences and two-factor
interactions still produce exactly `+1.0` and `-1.0`. Consumers should use the
coefficient directly when assembling a shared covariance quadratic form or an
aligned functional draw. Do not cast it to an integer or subtract marginal
interval endpoints.

`InterventionContrast::linear` accepts an explicitly listed finite combination
of worlds, one finite nonzero coefficient per world, and an explicit reference.
It supports higher-order factorial contrasts and weighted time-window targets.
Every world keeps its declared history and time frame. Population, outcome,
units and conditioning must remain compatible. Adding a term changes the
scientific question; it does not identify the additional world.

Scientific target wire version 2 adds this general linear form. Readers accept
version 1 fixed forms and revalidate their domain identity. A version 1 payload
cannot carry the new linear form. Existing fixed difference/interaction target
identities retain their earlier encoding. Unknown wire versions refuse.

Panel interaction features use the existing Rust `ValueExpr` and Python `Expr`
syntax. `PanelInteractionSpec::from_saturated_features` explicitly binds channel
reads to lag-weighted, saturated inputs. Source IDs, feature expressions and input
meaning are retained dependencies. Current/generated values from another model,
lagged feature reads and callback execution do not acquire a panel feature
binding merely because they are valid expressions elsewhere.

Compilation, model fitting and scientific authorization remain separate.
Model-conditional results require explicit conditional decision/search use;
complete aligned uncertainty and current evidence authority remain prerequisites
where the requested consumer needs them. Calibration and release acceptance do
not follow from transporting a fitted law.
