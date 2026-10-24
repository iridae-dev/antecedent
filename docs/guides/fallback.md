# What can be said when the effect is not identified

A causal query the declared structure does not identify ends in a typed refusal. The refusal is
the answer to the causal question, and it stays so. Two things are still worth having beside it:
a plain description of what the observed data show, and the evidence that would make the causal
question answerable. Both are opt-in, and neither replaces the refusal.

```python
import numpy as np
import antecedent as ant

rng = np.random.default_rng(11)
z = rng.normal(size=240)
t = (z + rng.normal(size=240) > 0).astype(float)
data = {"z": z, "t": t, "y": 1.0 + 2.0 * t + z + rng.normal(scale=0.3, size=240)}

# z -> t, z -> y, t -> y, and an unmeasured common cause of t and y (t <-> y).
graph = ant.Admg.from_edges(
    ["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")], [("t", "y")]
)
query = ant.AverageEffect("t", "y")

# Default: the refusal is raised, exactly as before.
#   ant.analyze(data, graph=graph, query=query)   -> EffectNotIdentified

outcome = ant.analyze(data, graph=graph, query=query, on_refusal="describe")
outcome.refusal           # the unchanged EffectNotIdentified (code, remedy, report)
outcome.associational     # a labelled-noncausal result: outcome.associational.answer.kind
outcome.evidence_plan     # obligations, next actions, templates (and a ranked plan)
print(outcome.explain())
```

## What comes back

`analyze(..., on_refusal="describe")` returns an `AnalysisWithFallback` only when the causal
query is refused for identification (`effect_not_identified`, `cross_world_not_identified`). A
query that succeeds returns its ordinary result. A search a budget stopped, a cancellation, a
resource limit, an invalid argument and every data or estimation error are raised exactly as
without the option.

* `refusal` is the original refusal object, unchanged. It is the answer to the causal question.
* `associational` is the compiled associational route
  ([noncausal associations](associational-route.md)) over the same outcome, treatment and
  adjustment variables. The adjustment variables are chosen by a declared rule: the measured
  parents of the treatment in the declared directed or bidirected graph, plus the query's
  declared effect modifier. The study is built through the same compiler as any other (`Study`
  compile, retained execution, plan view), is labelled noncausal
  (`answer.kind == "associational"`), carries the refusal's reason code in its identification
  layer (`refused_after`), and keeps the inference you chose (Bayesian posterior or analytic and
  resampling standard errors, all unmeasured for coverage). It ranks no action.
* `evidence_plan` is the evidence the failure determines. `obligations` are the verified evidence
  obligations, each with its proof step; `next_actions` are the evidence-gathering actions they
  imply (`measure_variable:<name>`, `run_experiment:<name>`, `find_instrument`,
  `collect_panel`); `templates` are the unfinished studies they determine. An assumption (an
  instrument's validity, the time-invariance a panel relies on) is listed as an obligation and is
  satisfied by no study.

Each of `associational` and `evidence_plan` is either present or a `FallbackUnavailable` with a
registered code, a stable `fallback.*` detail and the next attempt: no association form exists for
the query kind (a counterfactual, a distribution), a variable is not measured, the data are a
panel, the declared graph is not a directed or bidirected one, or the declared graph identifies
the effect (so the refusal came from the estimator and no structural gap is owed).

## Ranking the options that would close the gap

Declare the studies you could run and the plan ranks them. Cost, timing, recruitment and sample
size are yours; nothing is defaulted.

```python
from antecedent import repair

trial = repair.StudyCandidate.experiment(
    "randomized trial", population="study", interventions=["t"], measured=["y"],
    cost=9, sample_size=100, recruitment="consecutive recruitment", timing="baseline",
    unit="patient", cost_unit="usd", sample_budget=100, feasible=True,
)
outcome = ant.analyze(
    data, graph=graph, query=query, on_refusal="describe", evidence_candidates=[trial]
)
plan = outcome.evidence_plan.plan
plan.best.labels, plan.best.gate     # each option is checked by the theorem of its route
```

The plan is hypothetical identification, not an estimate: the ordering is `point_only` and its
calibration `unmeasured`. Plan again over other candidates, or with a declared value and cost
map, with `antecedent.design.evidence_plan(outcome.evidence_plan.contract, ...)`.

## What the description is not

The associational result is **not an answer to the causal question.** It describes the observed
law of the columns, with whatever confounding, selection and reverse direction that carries. It is
not the effect of an intervention, it is not identified, and it ranks no action; a scientific
bridge refuses it wherever a causal quantity is required. `repr`, `explain()` and `report()` of the
returned object say so, and the host record (`antecedent.host.fallback_response`) shows
`causal_status: unsupported` beside the status of the noncausal claim.

## From Rust

```rust
use antecedent::analysis::fallback::{FallbackOutcome, FallbackRequest};

match study.analyze_with_fallback(&FallbackRequest::default(), &ctx)? {
    FallbackOutcome::Causal(result) => { /* the ordinary result */ }
    FallbackOutcome::Described(fallback) => {
        // fallback.refusal, fallback.associational, fallback.evidence
        println!("{}", antecedent::report::render_fallback_report(
            &fallback, antecedent::report::ReportFormat::Markdown));
    }
    _ => {}
}
```

Pass `FallbackRequest { evidence: Some(request) }` to rank declared studies. Any other error
(`CausalError::Cancelled`, a resource limit, a capped search) is returned as the error it is.
