# Composed uncertainty: spread, exceedance and the whole path

## Run a small example

Compute the product of two columns from one aligned synthetic law; check the hand-derived mean and variance before requesting an interval.

<!-- example: examples/python/composed_uncertainty.py -->

Continue below for the exact contracts and refused adjacent requests.


A quantity computed from several uncertain inputs has the spread of the
composition, not of its components. Two routes read that spread at the composed
coordinate. Neither produces an interval without a measured record.

| Route | Rust surface | Label |
| --- | --- | --- |
| Composed spread, interval candidate and exceedance decision | `antecedent_stats::joint_law` (`draw_spread`, `normal_interval_candidate`, `composed_exceedance_decision`), `antecedent_contract::check_composed_interval` | `uncalibrated_spread`; candidates `unmeasured` |
| Whole path of the finite Markovian cross-world row | `antecedent_counterfactual::selective::whole_path_uncertainty::derive_whole_path_uncertainty` | `uncalibrated_spread` |

The whole-path account is a sealed sibling of the cross-world execution:
`CrossWorldExecution::whole_path_uncertainty(UnitSampling, ctx)` in the facade,
`CrossWorldExecution.whole_path_uncertainty("fixed_units" |
"independent_sample")` in Python, each returning a `WholePathAccount` with
`inspect()` and `verify_against(execution)`. Provenance records:
`joint_law.composed_spread_and_exceedance`,
`counterfactual.whole_path_uncertainty`.

## What each kind of dependence information licenses

- A joint moment (a covariance matrix) gives the exact variance of a linear
  combination, and a mixture with known weights gives its mean and variance.
- An aligned joint law (rows that are realizations of one joint law, proven by a
  shared alignment identity) licenses nonlinear functionals by row pairing.
- Only the marginals give a structural envelope (Frechet/Hoeffding, Makarov): an
  envelope is neither a point nor an interval and does not substitute for a
  joint law. A declared correlation range is an assumption range, labelled so.

Pairing the rows of unrelated marginal draws by index silently assumes
independence and is refused. A transported claim's uncertainty is three
pieces: source sampling, target sampling and the transport functional; naming
some and omitting others is refused with the missing pieces listed.

## Composed interval and exceedance

An interval construction at the composed coordinate (normal, draw percentile or
marginal) is a candidate with calibration status `unmeasured`. The interval
reading stays closed until a `CoverageRecord` measured at exactly that
`ComposedCoordinate` (operation, route, transported or not, input roles) and
construction exists, and then only at its recorded nominal level. A
component's record is never a composed coordinate's. An exceedance decision
carries a Monte Carlo band, which is numerical error and not coverage.

## Whole path

The path is abduction, action, per-world prediction and summary. The route
accounts for four sources: structural (declared, not propagated), posterior (the
exact predictive variance of the unit's counterfactual quantity under its
exogenous posterior, all worlds reading one state), sampling (fixed units or an
independent sample, which you declare) and numerical (exact enumeration, no
draws). The two spreads, `realized_mean_predictive_variance` and
`unit_mean_sampling_variance`, are for different estimands and are never added.
The derivation re-abduces every unit and refuses a result computed from other
mechanisms, evidence, worlds or quantity. The account is not a field of the
execution's wire: a consumer derives it again from the consumed execution and
the declared unit sampling, and `verify_against` refuses an account that does
not re-derive from that execution (`cross_world_uncertainty.account_mismatch`).
Units times joint exogenous states are bounded; sampled-path and transported
cross-world executions have no whole-path derivation and are refused.

## What is refused, and the remedy

| Refused request | Reason code or detail | Remedy |
| --- | --- | --- |
| Interval of a composition with no record at its coordinate | closed reading (`cell_not_licensed`) | Use the spread, or measure the coordinate |
| Row pairing of unaligned or marginal inputs | `joint_law_required` (`alignment_mismatch`, `index_pairing_refused`) | Supply draws from one joint law |
| Independence declared for inputs of unknown evidence overlap | `sampling_dependence_unknown` (`joint_law_propagation.evidence_overlap_unknown`) | Register the sources in the evidence ledger |
| Independence declared for evidence the ledger shows is shared | `evidence_double_counted` (`joint_law_propagation.shared_evidence_declared_independent`) | Treat the inputs as dependent and supply a covariance or an aligned joint law |
| Some transport pieces named, some omitted | missing-pieces refusal | Name source sampling, target sampling and the transport functional |
| Unit sampling not declared | `sampling_dependence_unknown` (`cross_world_uncertainty.unit_sampling_undeclared`) | Declare fixed units or a sample from a population |
| A sampled-path or transported cross-world path | `cell_not_licensed` (`cross_world_uncertainty.sampled_path_not_licensed`, `transported_path_closed`) | Use exact enumeration in the population the mechanisms were declared for |
| Fitted or sampled mechanisms in a cross-world row | closed | Use the sampled counterfactual route with its posterior input |
| Interval or posterior from the whole path | closed | Use the labelled spread |
