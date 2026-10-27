# Your first analysis

Estimate an effect, read the claim it makes, and carry the analysis onto new data. This page follows one analysis through the lifecycle Antecedent uses everywhere: **prepare**, **inspect** the plan, **estimate**, read the claim, **rebind** to new observations, **reconfigure** a declaration, then **export** and **load**. Python comes first; a Rust section follows.

Install Antecedent first with [the installation guide](installation.md).

## The question

In this simulated experiment, treatment increases the outcome by 2 units. A baseline measurement, `z`, also affects the outcome. Treatment is randomly assigned, so `z` does not determine who receives it. Because the data are simulated, the true effect is known and the result can be checked against it. Real data never come with that check; the graph and design are your declarations.

What you declare: the data table, the graph (`treatment` and `z` both affect `outcome`), and the question, an average effect of `treatment` on `outcome`. What Antecedent learns: the effect, from the data, under those declarations.

## Prepare and inspect

<!-- exec: quickstart -->
```python
import numpy as np
import antecedent as ant

rng = np.random.default_rng(17)
z = rng.normal(size=160)
treatment = rng.binomial(1, 0.5, size=160).astype(float)
data = {
    "treatment": treatment,
    "outcome": 2 * treatment + z + rng.normal(size=160),
    "z": z,
}
graph = [("treatment", "outcome"), ("z", "outcome")]
query = ant.AverageEffect("treatment", "outcome")

study = ant.prepare(
    ant.data.table(data), graph=graph, query=query,
    inference=ant.Frequentist(bootstrap=25),
    execution=ant.Execution(seed=19),
)
print(study.inspect())
```

`ant.prepare` checks that the declared structure identifies the effect and compiles a plan with compatible methods and the evidence they require. It fits nothing. `study.inspect()` shows the plan before you spend time executing it. Scientific inference (`ant.Frequentist`, `ant.Bayesian`) goes in `inference=`; seed, threads, cancellation and resource limits go in `execution=`.

## Estimate and read the claim

<!-- exec: quickstart -->
```python
result = ant.estimate(study)
print("Answer type:", result.answer.kind)
print("Estimated effect:", result.answer.value)
print("Uncertainty:", result.uncertainty.summary)
print("Calibration:", result.inspect().calibration.status)
assert abs(result.answer.value - 2) < 0.5  # the simulated truth is 2
```

Expect a `point` answer close to **2**. Sampling noise means it will not be exactly 2. The small bootstrap budget keeps the page quick; it is not a recommended budget for a final analysis.

`result` is an `Outcome`. Read `result.answer.kind` before extracting a number, then the identification, support, uncertainty and calibration. [Reading a result](reading-a-result.md) explains each. Every successful analysis retains a reusable study: `result.study` carries the graph, question and settings forward.

The full report is available as data or as text:

<!-- exec: quickstart -->
```python
report = result.inspect().to_dict()
print("Identification:", report["identification"]["summary"])
print(study.report("markdown"))  # the prepared contract and plan
```

## Rebind to new observations

Keep the study to reuse the graph, question and settings. Here the outcome is simulated with a treatment effect one unit larger:

<!-- exec: quickstart -->
```python
new_data = {**data, "outcome": data["outcome"] + treatment}
updated_study = ant.rebind(study, data=new_data)
updated = ant.estimate(updated_study)
print("Original:", result.answer.value)
print("Updated:", updated.answer.value)
```

The updated estimate is one unit larger. `ant.rebind` returns a new branch and leaves the source study and the earlier result unchanged. New data must have the same schema and design; a changed design needs a new `ant.prepare`. `updated_study.reuse` records which work was retained and which was invalidated.

## Reconfigure a declaration

`ant.rebind` changes observations. `ant.reconfigure` changes a declaration, such as the execution settings or the query, again as a new branch:

<!-- exec: quickstart -->
```python
reseeded = ant.reconfigure(study, execution=ant.Execution(seed=23))
reseeded_result = ant.estimate(reseeded)
print("Reseeded:", reseeded_result.answer.value)
```

## Export and load

<!-- exec: quickstart -->
```python
artifact = result.export()
loaded = ant.load(bytes(artifact), expected=artifact.expectation)
print("Acceptance:", loaded.acceptance.status)
print(loaded.answer)
```

The loaded result keeps the same answer. Keep `artifact.expectation` on the producing side and pass it as `expected=` when loading, so the consumer checks the identity it expects. `acceptance.status` is `verified` when the consumer replayed a verified program from the bytes alone, or `sealed` when the route ran through a sealed checked operation: loading then recognizes the artifact, verifies its contract and identities, keeps the recorded answer, and names the operation it cannot replay in `acceptance.unresolved`. Acceptance checks the saved result's contract, not whether its causal assumptions are true. Loading restores the report, not the dataset or a live study for new estimates.

## When an analysis is refused

A valid question that the declared structure cannot answer returns an `Outcome` carrying a typed refusal, not an exception. Here an unmeasured common cause of treatment and outcome is declared as a bidirected edge, so the effect is not identified:

<!-- exec: quickstart -->
```python
confounded = ant.Admg.from_edges(
    ["treatment", "outcome", "z"],
    [("treatment", "outcome"), ("z", "outcome")],
    bidirected=[("treatment", "outcome")],
)
refused = ant.analyze(data, graph=confounded, query=query)
print(refused.accepted, refused.answer.kind, refused.refusal.code)

# Opt in to a labelled noncausal description beside the unchanged refusal.
described = ant.analyze(data, graph=confounded, query=query, on_refusal="describe")
print(described.associational.answer.kind)  # "associational": never a causal answer
print(len(described.evidence_plan.next_actions), "ways to resolve it")
```

The refusal is the primary outcome. It names what was asked, what is missing and what would resolve it. `described.associational` is a separate outcome and `described.evidence_plan` lists the experiments, measurements or assumptions that would resolve the question. More rows alone cannot repair a missing causal assumption, and adding graph edges only to make a refusal disappear changes the question.

Invalid arguments, cancellation, resource limits and callback failures are different: they raise `ant.CausalError` subclasses. A refusal is scientific information; a raised error is an operational failure. Handle them separately. `ant.analyze` is the one-shot form of prepare then estimate.

## The same analysis in Rust

Run the smallest effect-estimation example from a source checkout:

```bash
cargo run -p antecedent --example ate_quickstart
```

The example creates a table, declares the schema and the causal graph, and asks for the average treatment effect. Its simulated outcome is `y = 1 + 2t + 3z` with `z` confounding `t`, so adjusting for `z` recovers exactly 2. The crude contrast of `y` on `t` does not, because the `z` gap between arms adds 1.5.

```rust
let schema = CausalSchemaBuilder::new()
    .continuous("t").treatment()
    .continuous("y").outcome()
    .continuous("z").context()
    .build()?;
let data = TabularData::try_from_schema_f64(
    schema.clone(),
    [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
)?;
let dag = Dag::from_named_edges(&schema, &[("z", "t"), ("z", "y"), ("t", "y")])?;
let query = AverageEffectQuery::binary_ate(schema.id_of("t")?, schema.id_of("y")?);

let result = Study::tabular(data)
    .graph(dag)
    .query(query)
    .refute(RefuteSuite::None)
    .bootstrap_replicates(0)
    .build()?
    .run(&ExecutionContext::for_tests(1))?;

println!("effect = {:.4}", result.effect());
println!("status = {:?}", result.identification.status);
```

This is an excerpt; read the [complete source](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/rust/ate_quickstart.rs), which includes the data generation and asserts the result. It disables refutation and bootstrap intervals and uses a parametric model, so it does not establish overlap for a real dataset. Expect `effect = 2.0000` and a nonparametrically identified status. Rust returns scientific refusals as typed values on the same terms as Python; `cargo doc -p antecedent --open` builds the API reference for your checkout.

## Next steps

- [Reading a result](reading-a-result.md) explains each field of the claim.
- [Choose an example](../examples.md) for weighting, discovery or temporal effects.
- [Inspect before estimating](../python-options.md#prepare-explicitly-when-useful) and [configure populations, likelihoods and overlap](../python-options.md).
- [Browse the Python API](../python-api.md), or begin a [tutorial](../tutorials/observational-study.md).
