# Narrow trade-offs from past decisions and stated comparisons

Use recorded past decisions, and comparisons people stated when asked, to narrow the trade-off
ranges of a value model. This step is optional. A [decision across trade-offs](preference-decisions.md)
needs only the ranges you declare; use this page when you also hold records of what was chosen,
from which options, under which expectations, and you want those choices to inform the ranges.
The result is a narrower admissible set, or an estimate with a measure over the set, that you
review and accept before any decision reads it.

Each recorded choice says that the option taken was valued at least as highly as the options
rejected. Under the declared value model and the expectations recorded at the time, that is a
linear inequality on the trade-off parameters, or, when the value has
[shape parameters](#shape-parameters), an inequality in them that the methods handle by
subdivision or by profiling. The compatible-set method intersects those inequalities with your
declared set. The choice-model method reads the choices as noisy and
estimates the parameters.

## A first narrowing

```python
from antecedent import preference_learning as pl
from antecedent import preferences as pf
from antecedent.outcomes import require_success

records = pl.Records(
    [
        pl.Record(
            "r1",
            decider="clinic_a",
            options=[
                pl.Option("treat", means={"benefit": 2.0, "harm": 1.0}),
                pl.Option("wait", means={"benefit": 1.0, "harm": 0.0}),
            ],
            chosen="treat",
            source=pl.Forecast("forecast-r1"),
            observation_rows=("ledger:r1",),
        ),
        # ... one Record per past decision
    ],
    population="target",
    pooling=pl.Pooled("one_shared_set_of_preferences"),
)

accepted = pl.CompatibleSet().accept(model, records)
accepted.ranges        # the accepted range of each trade-off
accepted.pending       # what to review before relying on it
accepted = accepted.acknowledge(*accepted.pending)

contract = pf.Contract(preferences=accepted, actions=actions, target_population="target")
```

`model` is the same value model you would declare for a decision, here
`benefit − harm_weight × harm` with `harm_weight` declared from 0 to 4. Each option gives what
was expected of every attribute, by attribute id, in the attribute's units. Choosing
`(benefit 2, harm 1)` over `(benefit 1, harm 0)` implies `harm_weight ≤ 1`, so the accepted
range is 0 to 1. The accepted preferences take the place of `model=` in a
[decision contract](preference-decisions.md#bind-actions).

What this is and is not:

- It recovers the trade-offs people acted on, under the expectations they held and the value
  model you declared. Their biases come with it.
- It does not say what the trade-offs should be. A learned range is evidence about past
  behavior, not a norm.
- A learned set is still a range over a declared assumption: that each choice was the best
  option within a tolerance. It is not a probability.
- An estimate from the choice model does carry a measure over the set. That measure is a
  normal approximation to the sampling distribution of the estimate, not a posterior. The
  calibration status of its intervals is resolved from the shipped coverage table; it reads
  `not_measured` until a coverage record of that construction exists.

## What a record holds

`pl.Record(id, decider=..., options=..., chosen=..., source=..., observation_rows=...)`:

| Field | Meaning |
| --- | --- |
| `options` | Every alternative listed at the choice, at least two, each a `pl.Option` with an id |
| Expectation per option | What was expected of each attribute: `means={attribute: value}`, or aligned `draws={attribute: [...]}` with optional `weights` |
| `source` | Where the expectations came from: `pl.Forecast`, `pl.AnalysisAsOf`, `pl.Realized` or `pl.Stated` |
| `chosen` | The option taken; it must be available |
| `decider` | Who chose, or who made the statement |
| `observation_rows` | The evidence-ledger rows the record rests on |

`pl.Records(records, population=..., pooling=...)` holds the records of one population and says
whether they are read per decider (`pl.PerDecider()`, the default) or pooled
(`pl.Pooled(assumption)`).

### Where expectations came from

What matters is what the decider expected, not what happened. A record names its source:

| Source | Use it when |
| --- | --- |
| `pl.Forecast(source_id)` | A forecast was written down before the choice |
| `pl.AnalysisAsOf(result_identity, as_of)` | The decider read an analysis result; the record names the result and the date |
| `pl.Realized(assumption)` | You read what later happened as what was expected; the record names that assumption (`choice_record.realized_assumption_blank` when it is blank) |
| `pl.Stated(elicitation, assumption)` | The person said which hypothetical bundle they would take; see [stated comparisons](#stated-comparisons) |

`pl.Option.from_result(contract, source)` reads what an analysis source expected of each action
of a preference contract, located exactly as the decision locates it, for a record whose source
is `pl.AnalysisAsOf`.

An expectation from means is exact only for a part of the value that is affine in the
attributes. Any other part needs the expectation as draws, and a record that gives means is
refused for it (`choice_record.belief_law_required`).

### Options that existed but were not allowed

An option can be listed as unavailable, with the constraint that excluded it
(`pl.Option("twenty", unavailable="budget_cap")`): a budget cap, a regulation, a policy. Choosing another option over it says nothing about preferences, so it
adds no inequality. It is kept in the record and its digest, and reported, but teaches
nothing. A record whose only available option was chosen is valid and teaches nothing either.

### Evidence rows

Each record names the evidence-ledger rows it rests on. The same row under two records would
count one piece of evidence twice and is refused (`choice_record.observation_row_double_counted`,
code `evidence_double_counted`). Rows are compared by their exact spelling after surrounding
whitespace is trimmed.

### Stated comparisons

A stated comparison records what someone said they would choose among hypothetical bundles in
an elicitation exercise. Each bundle is given by exact attribute values. It names the
elicitation and the assumption that the statement reflects the preferences the person would act
on. The compatible-set method reads it like a recorded choice; the learned set then also stands
on that assumption, and lists it. Accepted preferences learned from records that include stated
comparisons carry a review item saying how many there were.

```python
stated = pl.Record(
    "s1",
    decider="panel",
    options=[
        pl.Option("bundle_a", means={"benefit": 0.0, "harm": 0.0}),
        pl.Option("bundle_b", means={"benefit": 3.0, "harm": 1.0}),
    ],
    chosen="bundle_a",
    source=pl.Stated("elicitation-1", "statements_reflect_choices"),
    observation_rows=("survey:s1",),
)
```

## Two methods

| | Compatible set | Choice model |
| --- | --- | --- |
| Reads a choice as | the best option within a declared tolerance | more likely the higher its value, under a logit choice model |
| Assumption named | `assumption.recorded_choices_optimal_within_tolerance` | `assumption.choice_probability_logit_in_declared_value` |
| Returns | each parameter's declared and learned range, the choices that bind, or a conflict | a point estimate per parameter, its standard error and interval, a choice-consistency estimate and a measure over the set |
| Kind of answer | a range over an assumption; no probability | an estimate; the measure is a normal approximation to its sampling distribution |
| Contradictory choices | reported, with the tolerance needed | absorbed as noise |
| Stated and recorded together | allowed | refused in one fit |
| [Shape parameters](#shape-parameters) | learned by subdividing them; outer ranges and the kept boxes | profiled; profile likelihood-ratio intervals |
| Python | `pl.CompatibleSet(tolerance=...)` | `pl.ChoiceModel(level=..., draws=..., seed=...)` |

### Compatible set

A choice of option `c` over a rejected option `j` adds the relation

```text
sum_k theta_k (T_ck - T_jk) >= -tolerance - (F_c - F_j)
```

where `F` is the expected part of the value that moves with no parameter and `T_k` the expected
part multiplied by parameter `k`. Each parameter's learned range comes from two linear programs
over the declared set cut by these relations. A parameter is reported as narrowed when its
learned range lies strictly inside the declared range (as restricted by the declared
relations). The binding choices are the relations tight at the extremes found.

```python
learned = require_success(pl.CompatibleSet().learn(model, records))
learned.pooled            # a LearnedSet, or a Conflict (pooled records)
learned.per_decider       # one LearnedSet or Conflict per decider
learned.conflicts         # every contradictory outcome
learned.pooled.ranges     # per parameter: declared, learned, varied, narrowed
learned.pooled.binding    # (record, rejected option) pairs that bound the set
```

`learn` returns the full outcome; `accept` learns and accepts in one step, and refuses
contradictory records.

Reading a choice as a linear inequality comes from revealed preference (Afriat, 1967;
Varian, 1982). This is not a nonparametric revealed-preference test: the value model is
declared, and the question is which of its parameters are compatible with the choices. Keeping
the whole compatible set, not one representative, follows Greco, Mousseau and Słowiński
(2008).

### Choice model

The probability that option `i` was chosen is proportional to `exp(lambda * V_i)`, where `V_i`
is the declared value of option `i` under its recorded expectations and `lambda > 0` is the
choice consistency: larger means choices follow the value more closely. This is the
conditional logit of McFadden (1974), fitted by maximum likelihood. Each trade-off is the ratio
of its coefficient to the consistency, so its standard error is a delta-method value
(van der Vaart, 1998). With one decision-maker the covariance is the inverse observed
information. With `C` decision-makers and `q` coefficients, the covariance is clustered by
decider (Liang and Zeger, 1986) and multiplied by `C / (C - 1)` (Cameron and Miller, 2015) when
`C` is at least `2q + 1`; between two and `2q` decision-makers the fit is refused
(`preference_choice_model.too_few_deciders_to_cluster`), because so few clusters cannot carry
the covariance and the model covariance is not substituted.

The measure is drawn from the normal approximation to the estimate (Krinsky and Robb, 1986),
mapped to trade-off values and kept only inside the declared set. A draw that leaves some
parameter unresolved is kept only when some value of the unresolved parameters completes it to
a point of the set, the same check the evaluation applies to a measure. The share dropped is
reported.

A shape fit is polished by Fisher scoring, with Newton steps on the observed information where
scoring stalls. A polish that does not reach a stationary point is refused
(`preference_choice_model.not_converged`); a profile above the polished maximum starts the
polish once more from the profile's best point, so the estimate and its intervals are read at
the maximum.

```python
fit = require_success(pl.ChoiceModel(draws=500, seed=7).learn(model, records))
estimate = fit.pooled                 # or fit.per_decider[name]: an estimate or its refusal
estimate.consistency                  # how sharply choices follow value
estimate.parameters                   # point estimate, standard error, interval, declared range
estimate.interval_calibration         # "not_measured" until a coverage record exists
estimate.not_learnable                # parameters no record varied

accepted = pl.ChoiceModel(draws=500, seed=7).accept(model, records)
weighted = pf.Contract(preferences=accepted, actions=actions, target_population="target",
                       policy=pf.Policy.weighted())
```

Accepted preferences from the choice model carry the estimate's measure into the contract. Use
it under `Policy.weighted()`, or read its shares beside any verdict. A contract built on them
refuses a second measure passed to `evaluate`.

## What cannot be learned

A parameter can be learned only when the records vary it: two available options of some record
must differ in the part of the value that parameter multiplies.

| Situation | Compatible set | Choice model |
| --- | --- | --- |
| No record varies the parameter | keeps its declared range; review item `parameter_not_varied` | left out of the fit and listed as not learnable; review item `parameter_not_learnable` |
| Records vary it but the range does not narrow | review item `parameter_not_narrowed` | not applicable |
| The fixed part never differs within a record | not applicable | refused: the scale of the trade-offs is not determined (`preference_choice_model.unit_of_account_missing`) |
| Two parameters move together in every record | ranges reflect the joint constraint | refused (`preference_choice_model.rank_deficient`) |
| Some trade-off values explain every choice with no error (separation) | ranges as computed | refused: the likelihood has no finite maximum (`preference_choice_model.separated`) |
| A fixed parameter | folded into the fixed part and not learned; listed in the ranges at its value, with no review item | folded into the fixed part; not estimated, not among the estimates, and no review item |
| A shape parameter (inside a kink, `exp`, `ln` or a product of parameters) | learned by subdivision; see [shape parameters](#shape-parameters) | profiled; see [shape parameters](#shape-parameters) |
| A shape parameter the value differences within a record never read (it adds the same amount to every option) | keeps its declared range | left out of the fit and listed as not learnable; review item `parameter_not_learnable` |

## Shape parameters

A shape parameter is one the value is not linear in: a risk attitude inside `exp`, a breakpoint
inside a kink, a parameter multiplied by another. A recorded choice under such a value is a
*value relation*: the chosen option's value is at least the rejected option's, as a function of
every parameter. Both methods learn from them.

```python
risk = pf.strength("risk", units="1/usd", lower=0.1, upper=3.0, assumption="panel")
x = pf.attribute("x", units="usd")
model = pf.ValueModel(-pf.exp(-(risk * x)), utility_units="u")

# A sure 1 usd chosen over an even lottery on 0 and 3 usd.
choice = pl.Record(
    "r1", decider="panel",
    options=(pl.Option("sure", draws={"x": [1.0]}),
             pl.Option("lottery", draws={"x": [0.0, 3.0]}, weights=[0.5, 0.5])),
    chosen="sure", source=pl.Forecast("forecast-r1"), observation_rows=("ledger:r1",),
)
records = pl.Records((choice,), "target", pl.Pooled("one_panel"))

learned = require_success(pl.CompatibleSet().learn(model, records)).pooled
learned.ranges[0].basis       # "enclosure": an outer range
learned.ranges[0].learned     # about (0.481, 3.0): risk >= -ln((sqrt(5) - 1) / 2)
learned.region.cells          # the kept boxes of risk, each "inside" or "boundary"
```

**Compatible set.** Only the shape parameters are subdivided. On each box of shape values the
differences of expected value between the chosen and the rejected option are enclosed with
outward-rounded interval arithmetic, which leaves a linear system in the remaining trade-offs
with interval coefficients. A box is dropped when no shape value in it admits trade-offs that
satisfy every record (the united solution set of Oettli and Prager, 1964, is empty), kept as
`inside` when some trade-offs satisfy every record at every shape value of the box, and split
otherwise. Splitting stops when every box is decided, when the box budget is spent
(`CompatibleSet(max_boxes=...)`, 4,096 by default) or when every width is at most a fraction of
its declared range (`min_width_fraction=`, `1e-4` by default).

- A learned range is the hull of the kept boxes (for a linear trade-off, of each kept box's
  outer linear-program range), so it holds every compatible value and possibly more:
  `basis="enclosure"`, not `"linear_program"`. It is exact only up to the width of the boxes.
- `LearnedSet.region` lists the kept boxes (`RegionCell`: the box, its status `inside` or
  `boundary`, and the outer range of each linear trade-off over it), the boxes used and whether
  the budget ran out.
- The records conflict exactly when every box is dropped, a proof of emptiness. When no box is
  proven `inside` before splitting stops, neither compatibility nor contradiction is proven and
  the outcome is `pl.Undetermined`, with the undecided boxes in its `region`; a larger box budget,
  a smaller minimum width or narrower shape ranges may settle it. Accepting it is refused
  (`accepted_preferences.set_undetermined`).
- The learned set keeps the records as value relations, and accepted preferences carry them
  into a contract, which [decides on the certified route](preference-decisions.md#two-routes-and-what-each-guarantees).

**Choice model.** The likelihood is profiled over the shape parameters: at each shape value the
conditional logit is fitted in the consistency and the linear trade-offs, and the shape values
are searched from the centre of the declared box and the first points of a Sobol sequence
(Sobol, 1967, with the direction numbers of Joe and Kuo, 2008), then refined locally. Each shape
parameter gets a profile likelihood-ratio interval (Venzon and Moolgavkar, 1988) at the
requested level, with its ends found by bisection on the profile and truncated at the declared
bounds, and
`interval_method="profile_likelihood"`; the other trade-offs keep normal-approximation
intervals (`"wald"`). `ChoiceEstimate.shape_parameters` names the profiled parameters, and
`ChoiceModel(shape_search=pl.ShapeSearch(...))` sets the grid size, the number of shape
parameters searched (3 by default), the profile grid and the refinement limit.

```python
# noisy_records: several hundred recorded choices between pairs of sure amounts
fit = require_success(pl.ChoiceModel(draws=300, seed=3).learn(model, noisy_records))
fit.pooled.shape_parameters                 # ("risk",)
fit.pooled.parameters[0].interval_method    # "profile_likelihood"
fit.pooled.interval_calibration             # "not_measured"
```

A shape estimate is refused rather than reported when it would mislead:

| Situation | Refusal |
| --- | --- |
| The profile log-likelihood varies by less than half the chi-square(1) quantile over the declared range: the choices barely inform the shape | `preference_choice_model.shape_not_identified` |
| The estimate lies on a declared bound, where the normal approximation fails | `preference_choice_model.shape_estimate_at_bound` |
| The profile has two separated maxima near the overall maximum, so the likelihood-ratio set is not an interval | `preference_choice_model.profile_multimodal` |
| More shape parameters than the search covers | `preference_choice_model.shape_search_budget` |
| The choices are separated at some shape value | `preference_choice_model.separated` |

Fix a shape parameter to one value, record choices that tell its values apart, narrow its range
to one mode, or use the compatible-set method, which needs no noise model.

## Contradictions and the tolerance they need

When no admissible trade-off satisfies every choice, the compatible-set method returns a
conflict, not an error. The conflict names a smallest set of records that cannot hold together
with the declared set, found by exhaustive search up to a configured size, or an irreducible
set found by the deletion filter of Chinneck and Dravnieks (1991). An irreducible set need not
be smallest. The conflict also reports `tolerance_needed`: the smallest tolerance, in utility
units, at which every record becomes compatible.

Under shape parameters a conflict is a proof by subdivision that no shape value admits
trade-offs compatible with the records. Its kind is `not_minimised` when it could not be proven
that a smaller group of those records is compatible, and its `tolerance_needed` is a tolerance
at which the records hold together at the best shape value examined: enough, but possibly more
than the smallest.

```python
conflict = learned.pooled                  # a pl.Conflict
conflict.records, conflict.kind            # the records, "smallest", "irreducible" or "not_minimised"
conflict.tolerance_needed                  # in utility units
pl.CompatibleSet(tolerance=conflict.tolerance_needed).learn(model, records)
```

You then choose. Raise the tolerance to the value stated, accepting that choices may fall that
far short of the best option. Remove a record you can show was misrecorded. Or keep the
declared ranges. Accepting a contradictory outcome is refused
(`accepted_preferences.choices_contradictory`), with the tolerance needed in `expected` and the
records in `supplied`. A positive tolerance always becomes a review item.

## One decision-maker or several

Choices are read per decider unless you name a pooling assumption.

- **Per decider** (`pl.PerDecider()`, the default). Each decider gets an outcome of their own
  in `per_decider`. With several deciders the compatible-set method also reports `common`: the
  trade-offs compatible with every decider at once. When that is a conflict, `disagreeing`
  names the deciders whose records conflict.
- **Pooled** (`pl.Pooled(assumption)`). Every record is read as coming from one set of
  trade-offs, and the pooling assumption is listed. Use it when you can defend that the
  deciders share their trade-offs.

`accept(model, records, decider=name)` accepts one decider's preferences; a name the records do
not hold is refused (`accepted_preferences.decider_unknown`). `accept(model, records)` with no
decider takes the pooled outcome, which needs pooled records
(`accepted_preferences.pooled_not_available` otherwise). A pooled choice-model fit clusters
its covariance by decider under the rule above.

Accepting a learned set or estimate directly (`AcceptedPreferences::from_compatible_set` or
`from_estimate` in Rust) takes the records it was learned from: records whose digest differs
are refused (`accepted_preferences.records_mismatch`), and the count of stated comparisons is
read from them. A learned set that drops or changes a declared parameter, range or relation is
refused (`accepted_preferences.declaration_not_kept`), and so is one that adds a relation no
recorded choice implies (`accepted_preferences.added_relation_not_from_choices`). The set or
estimate itself is then learned again from the model, the records and the options supplied
(the seed included, so an estimate's measure draws are the same) and must be reproduced, within
the replay tolerance: a range, a relation or an estimate edited after learning, or options other
than the ones it was learned with, are refused (`accepted_preferences.learned_not_reproduced`,
naming the first field that differs). Recomputing, not checking a stored digest, is what
detects an edit, since a digest over the content can be recomputed by whoever edits it; the
cost is one run of the learner. `accept`, which learns and accepts in one step, does not run it
twice.

## Why a fit never mixes stated and recorded choices

A stated comparison and a real choice need not share a consistency scale: people can be more
or less consistent when they answer a question than when they act. The choice model estimates
one consistency per fit, so a fit whose records hold both is refused
(`preference_choice_model.stated_and_recorded_mixed`). Fit them separately and compare, or use
the compatible-set method, which has no noise scale.

## Accept, review and reuse

Learned preferences reach a decision only through `AcceptedPreferences`: a reviewed, versioned
handle. It follows the [accepted graph](selective-discovery.md#hold-an-accepted-graph-and-estimate-from-it-repeatedly)
of discovery. You learn once, review, and then reuse; nothing relearns implicitly.

| Operation | Effect |
| --- | --- |
| `pl.AcceptedPreferences.declared(model)` | the declared ranges alone, version 1, nothing pending |
| `CompatibleSet(...).accept(...)`, `ChoiceModel(...).accept(...)` | the learned set or the estimate, with the learning options and the decider, version 1 |
| `pending` | review items derived from the basis, less those already acknowledged |
| `acknowledge(*items)` | a new handle, version increased by one; each item must be pending (`accepted_preferences.not_pending` otherwise) |
| `relearn(records, decider=..., using=...)` | relearns on request, with the stored options unless `using=` gives others; a new handle, version increased by one, acknowledgements cleared |
| `ranges`, `learned`, `estimate`, `measure`, `basis`, `version`, `identity` | what the handle holds |
| `pf.Contract(preferences=accepted, ...)` | a decision contract over the accepted model, with the accepted measure when there is one |

```python
accepted.pending        # (ReviewItem("parameter_not_varied", "cost_weight"),)
reviewed = accepted.acknowledge(*accepted.pending)
reviewed.version        # 2
```

Review items (`pl.ReviewItem`, with `kind`, `parameter` and `amount`):

| `kind` | Rust | Raised when |
| --- | --- | --- |
| `parameter_not_varied` | `ParameterNotVaried` | No record varied the parameter; it keeps its declared range. A parameter declared at a single value is stated outright and raises no item |
| `parameter_not_narrowed` | `ParameterNotNarrowed` | Records varied the parameter but its range did not narrow |
| `tolerance_applied` | `ToleranceApplied` | The compatible set was learned with a positive tolerance |
| `parameter_not_learnable` | `ParameterNotLearnable` | The estimate leaves the parameter unresolved |
| `estimate_outside_declared` | `EstimateOutsideDeclared` | The point estimate lies outside the declared range |
| `draws_outside_declared` | `DrawsOutsideDeclared` | A share of the measure's draws fell outside the declared set and was dropped |
| `stated_comparisons_used` | `StatedComparisonsUsed { count }` | The records the preferences were learned from include stated comparisons; `amount` is their number, also read as `accepted.stated_comparisons` |

The handle is stored as `accepted_preferences_v1` in the checksummed container, with the
declared model, the basis, the learning options, the decider selection, the version and the
acknowledged items.

```python
exported = reviewed.export()
loaded = pl.load(exported, expected_identity=reviewed.identity, records=records)
```

It loads only under an identity you retained apart from the bytes. With `records=`, a learned
basis is relearned from them and must be reproduced; different records are refused
(`preference_artifact.replay_records_differ`). A declared basis replays without records.
Reals are compared within a stated relative tolerance, because linear-program and fitting
outputs need not agree to the last bit across platforms.

## Limits

- The method recovers the trade-offs people acted on under the recorded expectations, biases
  included. If the forecasts were wrong or the deciders were mistaken, the learned set says so
  only through contradictions.
- It does not say what the trade-offs should be.
- Under shape parameters the compatible set is an outer enclosure, exact only up to the width
  of its boxes, and within the box budget neither compatibility nor contradiction may be
  provable (`Undetermined`). The choice model's shape intervals are profile likelihood-ratio
  intervals, `not_measured` until a coverage record of their construction exists.
- Motives the value model does not contain are not detected unless they push a choice beyond
  the declared tolerance. Within it, an unrecorded motive is indistinguishable from the
  declared trade-offs.
- The choice-model intervals read as calibrated only from a shipped coverage record of their
  construction (`not_measured` until one exists), and its measure is not a posterior.

## Refusals you are likely to meet

The full tables are in
[preference sets and recorded choices](reference/refusal-codes.md#learning-preferences-from-recorded-choices).

| Detail | Code | Remedy |
| --- | --- | --- |
| `choice_record.realized_assumption_blank` | `invalid_argument` | Name the assumption that realized outcomes stand for expectations |
| `choice_record.observation_row_double_counted` | `evidence_double_counted` | Keep each evidence row in one record |
| `choice_record.belief_law_required` | `decision_contract_unsatisfied` | Record the expectation as draws |
| `preference_choice_model.unit_of_account_missing` | `design_rank_deficient` | Record choices whose options differ in the fixed part |
| `preference_choice_model.rank_deficient` | `design_rank_deficient` | Record choices that vary the named parameters separately, or fix one |
| `preference_choice_model.separated` | `glm_fit_refused` | Use the compatible-set method |
| `preference_choice_model.shape_not_identified` | `design_rank_deficient` | Fix the shape parameter, record choices that tell its values apart, or use the compatible-set method |
| `preference_choice_model.shape_estimate_at_bound` | `uncertainty_representation_insufficient` | Widen the range if the bound is not a limit of the preference, or use the compatible-set method |
| `preference_choice_model.profile_multimodal` | `glm_fit_refused` | Narrow the declared range to one mode, or use the compatible-set method |
| `preference_choice_model.shape_search_budget` | `invalid_argument` | Fix some shape parameters to single values, or raise `ShapeSearch(max_dimensions=...)` |
| `preference_set_learning.region_budget_invalid` | `invalid_argument` | Give `max_boxes` of at least one and `min_width_fraction` in [0, 1] |
| `accepted_preferences.set_undetermined` | `decision_contract_unsatisfied` | Raise the box budget, lower the minimum width, or narrow the shape ranges |
| `preference_choice_model.stated_and_recorded_mixed` | `route_not_supported` | Fit them separately, or use the compatible-set method |
| `preference_choice_model.no_draw_inside_set` | `uncertainty_representation_insufficient` | Draw more, record more choices or use the compatible-set method |
| `accepted_preferences.choices_contradictory` | `decision_contract_unsatisfied` | Raise the tolerance to the stated value or remove a named record |
| `preference_artifact.replay_records_differ` | `decision_contract_unsatisfied` | Replay with the records the preferences were learned from |

## Entry points

| Language | Entry |
| --- | --- |
| Python | `antecedent.preference_learning`: `Option`, `Forecast`, `AnalysisAsOf`, `Realized`, `Stated`, `Record`, `Records`, `PerDecider`, `Pooled`, `CompatibleSet(...).learn` and `.accept`, `ChoiceModel(...).learn` and `.accept`, `ShapeSearch`, `Region`, `RegionCell`, `Undetermined`, `AcceptedPreferences`, `ReviewItem`, `load` |
| Rust, analysis surface | `antecedent::analysis::preference_decision`: `analysis_options`, `analysis_belief_source`, `decide_accepted`, `export_accepted`, `consume_accepted` |
| Rust, methods | `antecedent_design::choice_record` (`ChoiceRecordSet`, `choice_features`, `choice_features_at`), `preference_set_learning::learn_compatible_set` (`LearnedRegion`, `UndeterminedSet`), `preference_choice_model::estimate_choice_model`, `accepted_preferences::AcceptedPreferences`, `preference_model::ValueRelation`; `antecedent_stats::fit_nonlinear_conditional_logit` |

## Sources

Afriat, S. N. (1967), the construction of utility functions from expenditure data,
doi:10.2307/2525382. Varian, H. R. (1982), the nonparametric approach to demand analysis,
doi:10.2307/1912771. Jacquet-Lagrèze, E. and Siskos, J. (1982), the UTA method,
doi:10.1016/0377-2217(82)90155-2. Greco, S., Mousseau, V. and Słowiński, R. (2008), ordinal
regression revisited, doi:10.1016/j.ejor.2007.08.013. Chinneck, J. W. and Dravnieks, E. W.
(1991), locating minimal infeasible constraint sets in linear programs,
doi:10.1287/ijoc.3.2.157. McFadden, D. (1974), conditional logit analysis of qualitative choice
behavior, in *Frontiers in Econometrics*. Liang, K.-Y. and Zeger, S. L. (1986), longitudinal
data analysis using generalized linear models, doi:10.1093/biomet/73.1.13. Cameron, A. C.
and Miller, D. L. (2015), a practitioner's guide to cluster-robust inference,
doi:10.3368/jhr.50.2.317. van der Vaart, A. W.
(1998), *Asymptotic Statistics*, doi:10.1017/CBO9780511802256. Krinsky, I. and Robb, A. L.
(1986), on approximating the statistical properties of elasticities, doi:10.2307/1924536.
Venzon, D. J. and Moolgavkar, S. H. (1988), a method for computing profile-likelihood-based
confidence intervals, doi:10.2307/2347496. Sobol, I. M. (1967), on the distribution of points
in a cube and the approximate evaluation of integrals, doi:10.1016/0041-5553(67)90144-9. Joe,
S. and Kuo, F. Y. (2008), constructing Sobol sequences with better two-dimensional projections,
doi:10.1137/070709359. Moore, R. E., Kearfott, R. B. and Cloud, M. J. (2009), *Introduction to
Interval Analysis*, doi:10.1137/1.9780898717716. Oettli, W. and Prager, W. (1964), compatibility
of approximate solution of linear equations with given error bounds for coefficients and
right-hand sides, doi:10.1007/BF01386090. The deviations from each are recorded in the method provenance a result carries; see
[citing a study](guides/citing-a-study.md).

## Where next

- [Decide when trade-offs are a range](preference-decisions.md)
- [Choose a discount when the cost of a return is a range](tutorials/decide-across-trade-offs.md),
  whose last part narrows a range from past decisions
