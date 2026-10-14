# What the transport and uncertainty routes do not do

This page lists, for each delivered package, what it does not do, the request
most likely to be refused, and the remedy. It states limits only; the guides
state what each route does. Every route here is point-only, a structural
envelope, an assumption range, or derived-unmeasured. None licenses a
confidence interval, a posterior coverage statement or a calibration claim.

| Package | Does not do | Most likely refused request | Remedy |
| --- | --- | --- | --- |
| [Transported cohort effect](guides/transported-decisions-and-measurement.md) | Test the selection premise; estimate overlap or a density ratio; inherit the source model's standard error | An interval or standard error for a cohort contrast | Use the point; an interval needs a measured record at that coordinate |
| [Measurement transport](guides/transported-decisions-and-measurement.md) | Propagate the uncertainty of the invariance premise; pool differential error | Sensitivity-and-specificity correction after declaring specificity only | Declare every parameter the route uses, or validate the model in the target |
| [Mechanism transport](guides/mechanism-and-synthesis-transport.md) | Carry a mechanism behind a selection arrow; carry tables, non-Gaussian or externally bound mechanisms; average with a native refit | A mechanism the diagram marks as shifted | Refit it in the target; use finite transport for tables |
| [Transported synthesis](guides/mechanism-and-synthesis-transport.md) | Decide dependence between source and native estimates; give an interval or posterior coverage | A transport with no declared uncertainty, or several claims through one transport without covariance | Declare the transport variance once and the shared covariance |
| [Surrogate bridge transport](guides/mechanism-and-synthesis-transport.md) | Discharge the transport premise; substitute for an identified effect | A bridge whose terminal outcome has a selection arrow into it | Validate the surrogate relation in the target |
| [Transported policy, adaptive source, period shift](guides/transported-policy-and-adaptive-source.md) | Shift a transition, observation or utility mechanism; learn a drift model; give an interval | A shifted dynamics mechanism or a target law outside the source support | Evaluate with a model fitted in the target |
| [Discovered-structure transport](guides/discovered-structure-transport.md) | Pick a class member; weight members; certify non-transportability; handle PAGs or temporal structure | A request over a class too large to enumerate or with unresolved pairs | Add background knowledge or declare a graph |
| [Per-population discovery](guides/discovered-structure-transport.md) | Pool rows; test invariance; turn an edge difference into selection targets | Stacked populations or relabelled duplicates | Use the pooled algorithm directly with declared assumptions; supply disjoint rows |
| [Sample split](guides/discovered-structure-transport.md) | Control the selection event's error; split dependent, weighted or masked rows | An interval on the same rows that selected the graph | Split exchangeable rows |
| [Provider-sourced transport](guides/provider-sourced-transport.md) | Execute a provider; prove a declared population; raise calibration | An attested-only provider where verification is required | Obtain a verification receipt covering the object |
| [Composed spread and exceedance](guides/composed-uncertainty.md) | Give coverage without a record at the composed coordinate | An interval for a composition | Use the spread, or measure the coordinate |
| [Whole-path uncertainty](guides/composed-uncertainty.md) | Propagate declared structure; handle fitted or sampled mechanisms; add the two spreads | A cross-world row without a unit-sampling declaration | Declare fixed units or a population sample |
| [Decision-reliability derived uncertainty](guides/decision-reliability-uncertainty.md) | Cover a wrong law or unretained candidates; publish without a record | A published interval or a ranking probability from a point-only producer | Keep per-draw utilities from one joint law; await measurement |
| [Adaptive procedure uncertainty](guides/adaptive-procedure-uncertainty.md) | Handle more than the bounded design; claim coverage; cross paradigms | A posterior under value-dependent loss, or fixed-sample inference on an adaptive design | Model the loss; use the adaptive-weighting or anytime-valid paradigm |
| [Finite-SCM discretization](guides/finite-scm-discretization.md) | Bound variances, probabilities or counterfactuals; take latent loadings, non-Gaussian or bounded support | Levels left undeclared, or an interval on the discretized value | Declare levels; use the point and its bound |
| [Assumption range and revalidation](guides/assumption-range-and-revalidation.md) | Choose the bound; attach a probability; carry a record to another target | A range for a discovered structure, or calibration at a new target | Use the structural envelope; re-measure |
| [Inverse and decision reach](guides/inverse-and-decision-reach.md) | Supply a law from a point; supply an interval region; rank actions from scenario inversion | A quantile or probability on a point source | Supply a law (draws or a finite table) |
| [Interface-contract adapters and overlap](guides/interface-contract-adapters.md) | Infer support, estimand or calibration; treat unknown overlap as disjoint | Independence declared where overlap is unknown | Register the sources in the evidence ledger |

## Reading a refusal

A refusal carries a registered reason code, a typed detail and a remedy. The
codes are listed in the refusal registry. A refusal is not a failure of the
data; it states which premise or declaration was not met.

## Labels that are never upgraded

- `point_only`: the value is one number under a declared premise.
- Structural envelope: a statement about which members of a class the rules
  certify, with no probability.
- Assumption range: the value at declared shifts of a premise, with no
  probability on the shifts.
- Derived-unmeasured: the method is derived and its law is exact under a stated
  model; no coverage has been measured.
- Transported: a claim about a target population carried with its transport
  identity; never presented as native target data.
