# Inspect a plan and read a report

Before anything is estimated you can inspect the program a study compiles to, and afterwards render the records as a readable report. A plan view shows every layer of the program a study compiles to before
anything is estimated. It covers one study under any route: an average effect
on a DAG, a temporal response, a randomized trial, a structure search, a
mixture over a graph posterior, a cyclic family or a standalone experiment
plan. The view names the route and the family that lowered it. It shows the
lowered program's identity, which every result of that program carries. It
shows the support cell and its licensing status, and then one section per layer.

Reading a view does not identify, fit, scan rows, enumerate graphs, simulate or
lease from an admission pool. It reads what compile retained, or what
preparation installed.

## Three stages

| Call | What runs | What the view shows |
| --- | --- | --- |
| `Study::inspect()` (Python `prepared.preflight()`) | Nothing beyond the structural record. It does not identify (`compiler.inspect_does_not_identify`). | No plan view. The identification and uncertainty slots stay `Unavailable`. |
| `Study::plan_view(ctx)` | `Study::compile`. A route whose program retains identification identifies once, as compile always does. A linear adjustment quotes its design width from the schema and the categorical levels. No fit, sample, simulation or estimate runs. | Stage `compiled`: the lowered program, the physical record compile sized and the compile context's kernel policy. A graph posterior's atoms come from the supplied posterior. |
| `PhysicalExecutionPlan::plan_view()` | Nothing. It reads the plan it is called on. | Stage `compiled`, without the study: no support status, and no atom kind for a graph posterior. |
| `PreparedStudy::plan_view()` (Python `prepared.plan_view()`) | Nothing. It reads the prepared handle. | Stage `prepared`. Also shows the identification preparation installed (`installed_at_prepare`) and the sealed operation it retained (`operation`). |

A cyclic family handle (`prepare_cyclic`, `prepare_cyclic_family`) and a
standalone experiment plan (`compile_planning`) have `plan_view()` too. Their
programs carry no physical record.

## Layers

| Layer | Rust | Python | What it holds |
| --- | --- | --- | --- |
| Route | `route()`, `family()`, `identity()`, `depth()` | `route`, `family`, `identity`, `depth` | The route cell (the `lowering` column of `parity/licensed_routes.toml`), its family, and the lowered program identity. |
| 1. Scientific problem | `scientific()` | `scientific` | The target: a world-indexed target, one per horizon, or a typed functional keyed by the query digest. Also the plan id, the data modality, the graph class and the query variables. |
| 2. Identification | `identification()` | `identification` | See [Identification](#identification). |
| 3. Statistical program | `statistical()` | `statistical` | Identifier, estimator, realization, model slots (role, family, inputs, output), cross-fitting folds, resampling unit and replicates, posterior algorithm, draws and prior, and the design. |
| 4. Evidence and decision | `evidence()` | `evidence` | Sources and their roles, the composition (single, transported, pooled, atom mixture), and, when compile started from one, the decision (carrier, actions, constraints, utility form and unit) with its EVSI Monte Carlo store, or the planning request with its simulation quote. |
| 5. Physical plan | `physical()` | `physical` | Operator contracts with their scratch, retained and output byte estimates, summed (`None` when an operator declares none, since an absent estimate is not zero). Also the quoted design width, or the reason the route quotes none, the physical record (materializations, kernels, batch, workspace, peak and copy bytes, task schedule, workers) and the kernel policy. |
| 6. Uncertainty and claim | `uncertainty()`, `claim()` | `uncertainty` | How the published uncertainty must be read (sampling, posterior, envelope, none), the claim kind, the realization, resampling and posterior plans, and the evidence-port request a decision puts to the program. |
| Support | `support()` | `support` | The support-matrix coordinate, whether it is licensed, and the cell status. |
| Mixture | `mixture()` | `mixture` | Graph posteriors only. See [Graph posteriors](#graph-posteriors). |
| Operation | `operation()` and `PreparedStudy::retained_operation()` | `operation` | Prepared stage only. The sealed operation preparation retained. In Rust, `retained_operation()` returns it as a typed `RetainedOperation`. In Python, `operation` is a dict with `kind` and that operation's inspection record. |

## Identification

`identification().kind()` takes one of four values:

- `retained`: compile identified the study and retained the products.
- `deferred`: compile refused. Preparation identifies and reports the same refusal. `reason` says why.
- `per_execution`: the route identifies on each execution. Sharp regression discontinuity and cross-world probes do this. `reason` says why.
- `non_graphical`: a feedback model carries a specified-model, experimental-recovery or cyclic input/output identification. It is never a graphical product.

`product()` is the checked identification product bound to the scientific
target. It holds the outcome, the claim class, the estimands, the assumptions,
the required factors and the unresolved obligations.

`retained()` lists each product preparation installs:

- `static`: the status, the selected estimand with its adjustment set, and its assumptions.
- `pag` and `cpdag`: the envelope status, the shared estimand, the completions enumerated at compile, the truncated completions, and the identified and unidentified mass.
- `temporal` and `temporal_class`: the same, by horizon.
- `transport`: the certificate outcome.
- The posterior products: the atom count and the masses.

`status()` is the weakest status the products state.

## Graph posteriors

A graph-posterior study has no single structure, so `compile_logical` still
refuses it. Its plan view is its single logical plan. `mixture()` lists each
distinct structure in the order it first appears. For each structure it gives:

- the graph key;
- the summed posterior weight and the number of samples;
- whether the structure identifies the target, with its status and estimand (or, for a class atom, the estimand its completions share, the number of completions and the identified weight inside the envelope);
- the total weight, and the identified and unidentified mass, which are reported and never dropped;
- the composition: an atom mixture.

A cross-world cell identifies on its probe at each execution. Its atoms come
from the supplied posterior, and `identified` is `None`.

## Example

```rust
let plan_view = study.plan_view(&ctx)?;              // compile, then view
assert_eq!(plan_view.identity(), study.compile(&ctx)?.lowered_program_identity().unwrap());
let prepared = study.prepare(&ctx)?;
let view = prepared.plan_view()?;                    // no identification, fit or scan
println!("{}", serde_json::to_string_pretty(&view.to_json())?);
match prepared.retained_operation() {
    antecedent::analysis::RetainedOperation::CheckedLinear(lowering) => { /* design, backend */ }
    other => println!("retained {}", other.name()),
}
```

```python
prepared = ant.prepare(data, graph=edges, query=ant.AverageEffect("t", "y"))
view = prepared.plan_view()           # ProgramPlanView; nothing executes
view.identification.status            # "nonparametrically_identified"
view.identification.retained[0]["estimands"][0]["adjustment_set"]   # ["z"]
view.physical.quoted_width            # encoded design columns, or None
view.operation["kind"]                # the sealed operation preparation retained
view.to_dict()                        # the JSON-safe whole
```

## Limits

- The physical layer has one dense design per route. Byte estimates are planner
  estimates, not measurements. A measured execution's facts stay on its result.
- A view taken from a plan without its study shows no support status. For a
  graph posterior, it also shows no atom kind and lists no atoms for a cross-world cell.
- `inspect()` stays non-identifying. Use `plan_view` to see what identification produced.

## Read a report

A report turns a contract into a document a person can read: Markdown for a
repository, a review or a notebook, or one HTML page for a browser. It is
rendered from the records a study already holds, so it adds no analysis. Every
number in it is a value one of those records carries, printed to at most twelve
significant digits. The wording is fixed text chosen by the records' own status
tags, and the same records render to the same bytes: there is no clock, random
source, unordered iteration or environment read.

### Calls

| Where | Call | Reads |
| --- | --- | --- |
| Python | `prepared.report("markdown")` or `prepared.report("html")` | After an estimate, the executed claim and the contract it ran under; before one, the prepared contract. Always the plan view. |
| Rust, handle | `PreparedStudy::report(result, ctx, format)` | The same, with `result: Option<&StudyResult>`. |
| Rust, records | `antecedent::report::render_report(&ReportInput::new(&contract).with_claim(&claim).with_plan(&plan), format)` | Whatever records the caller holds. A contract from `Study::inspect()` alone reports a cell the support matrix refuses, with its blockers and next actions. |

Rendering identifies, fits and scans nothing, runs no estimate and changes no
handle. A Python handle that executes no lowered study program, such as a
transport stage, has no plan view and raises `plan_view_unavailable`. A cell
the support matrix refuses raises at `prepare`, so Python has no handle to
report on; the Rust contract-only report covers it.

### Sections

| Section | What it states |
| --- | --- |
| Summary | The query, inference, identifier, estimator, route, claim kind and value, the support status, and fixed sentences keyed to those statuses. |
| Claim | Only with a claim: claim id, kind, value and units, the identification, support and evaluated domains, and the execution digest. |
| Identification | Graph class, structure source, identifier and estimator, and the identification slot: status and the identified, unidentified, unevaluable and incomplete-search masses, or why the slot is unavailable. |
| Assumptions | The obligations with scope, kind, status, source, required check and description, and the assumptions the plan's identification relies on. |
| Diagnostics and evidence | The evidence the claim references, attested evidence (marked as not re-verifiable), checks that failed or have not run, and the obligations identification left unresolved. |
| Uncertainty | The uncertainty slot's components and whether each is omitted, the uncertainty the program must publish, and the calibration slot with its basis. |
| Support and licence | The support status, the support slot and the support cell of the program. |
| Refusals and limits | The operation report: applicability, readiness, blockers, unmet obligations, next actions and licensed neighbours, with the limits the other records state (a capped search, unidentified mass, omitted uncertainty, calibration that is not measured). |
| Program plan | The route, family, stage and identity, the scientific problem, the identification product and its estimands, the statistical, evidence and physical layers, and a graph posterior's mixture. The sized physical record and kernel policy describe the machine and are left out. |
| Identities | The domain-separated digests of the contract. |

A status is reported as the records carry it. A route that executes without a
support-matrix licence says so, and a calibration slot that is not `calibrated`
says that the reported uncertainty has no measured-coverage record. The report
grants no licence and upgrades no status.

### Escaping

The HTML is one self-contained document with inline style. It has no script
element, loads no resource and sets a content security policy that forbids
both. Every value taken from an input, such as a variable name, an obligation
description or an evidence reference, is HTML-escaped. The Markdown
backslash-escapes every markup character in the same values, and both formats
replace control characters with spaces so a value cannot start a new line or
row.
