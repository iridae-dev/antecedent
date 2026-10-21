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

Prepared panel callers use `Study::prepare_joint_panel` or the existing prepared
study's `compile_joint_panel`. Retarget worlds on the prepared program to reuse a
compatible fit. Reuse follows the caller's `CachePolicy`; test contexts that
explicitly disable caching do not promise reuse. Inspection does not fit.

Use `mean_source`, `law_source`, or `project_*_source` for decision/search input.
They retain scientific permission and the exact projected question. Choose
model-conditional consumption explicitly when the design has not identified the
question. `numerical_*` accessors expose component values without granting that
permission. Nonterminal windows and window changes carry actual prefix-world
targets and their own question identity.

Frozen-fit artifact replay preserves aligned draw slots, failures and uncertainty
meaning. Historical replay grants no current decision authorization. Current
experiment-update consumers supply the expected compiled program/frozen law and
current ledger authority; withdrawal, supersession and trust changes can refuse
current use while historical numerical replay remains available.

Frequentist local Gaussian/cluster-sandwich realizations are approximations, not
posterior draws. Bayesian laws must pass their actual convergence checks. Search
certificates are scoped to an explicitly enumerated finite domain; continuous
search reports a best-found feasible point. Neither artifact interchange nor a
successful fit promotes unmeasured calibration or release permission.

The new Gaussian `compile_joint_response` and `retarget` methods borrow
`&ScientificTarget`. The prepared program owns the retained target/proof; callers
need not allocate a second public builder to execute it.

`PanelExperimentResult::law_source` returns a retained experimental source whose
`authorize`, `evaluate` and `search` require the caller's current evidence ledger.
Keeping that source does not keep withdrawn or superseded evidence authorized.
Explicit historical numerical extraction remains available.
