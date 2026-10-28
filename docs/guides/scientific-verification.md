# How Antecedent checks its scientific methods

You can follow a numerical claim in Antecedent from its source method to the calculation that checks it, the test that consumes that calculation, and the evidence status reported with a result. This guide follows that path through augmented inverse-probability weighting (AIPW), then uses arrow strength to show a second kind of mathematical check.

Antecedent's development applies high-assurance engineering principles. The practical question behind that approach is: **what evidence supports this particular answer?** A published method, a passing test and a measured interval answer different parts of that question. The records keep those distinctions available for review.

For the shorter introduction, read [Assurance](../concepts/assurance.md). For interpreting an analysis you have already run, start with [Results are claims](../concepts/results-are-claims.md).

## Follow the claim, then its evidence

An estimand is the quantity a scientific question asks for: an average effect in a population, for example. An estimator is the procedure used to estimate it. A check needs to name both precisely enough to distinguish the requested calculation from a nearby one.

| Evidence | The question it answers | Where to inspect it |
| --- | --- | --- |
| Method provenance | Which method is implemented, under which assumptions and deviations? | [Algorithm provenance](https://github.com/iridae-dev/antecedent/blob/main/provenance/README.md) |
| Independent numerical checks | Does a named calculation agree with a reference derived separately? | [Evidence and oracle rules](https://github.com/iridae-dev/antecedent/blob/main/parity/README.md) |
| Executing support evidence | Which combination of question, structure, inference and estimator has been exercised? | [Support matrix](../support-matrix.md) |
| Calibration | How did the uncertainty procedure behave across repeated simulated datasets in a measured setting? | [Coverage records](../development.md#coverage-records) |

These checks have different scopes. Agreement on a coefficient does not establish an entire workflow's interval coverage. A supported average effect does not automatically support a derivative or a decision built from it. The evidence needs to follow the operation that produces the claim.

## AIPW: start with the method record

AIPW combines an outcome model with a model of treatment assignment. For a binary treatment, its average-effect score has the form:

```text
score = μ₁(X) − μ₀(X)
        + T·(Y − μ₁(X))/e(X)
        − (1 − T)·(Y − μ₀(X))/(1 − e(X))
```

Here, `e(X)` is the fitted treatment probability, and `μ₁(X)` and `μ₀(X)` are fitted outcome means under treatment and control. Identification still requires assumptions such as unconfoundedness given the adjustment variables and adequate treatment overlap. The score itself cannot establish those assumptions from observations.

Open [estimate.aipw.toml](https://github.com/iridae-dev/antecedent/blob/main/provenance/estimate.aipw.toml). Its paper entries name the source literature; its implementation deviations explain which parts of that literature apply to the actual routes.

The distinction matters here. The untrimmed, all-observed ATE route uses five-fold cross-fitting: nuisance models are fitted on other folds before scoring a held-out row. The treated-population, untreated-population, trimmed and predicate-target routes fit their nuisances on the full sample. The record explicitly withholds the cross-fitting guarantees for those full-sample routes. It also names the fixed logistic-propensity and linear-outcome families rather than implying support for arbitrary machine-learning nuisances.

That is what provenance contributes beyond a citation: it connects a method's premises to specific implementation choices. The [provenance rules](https://github.com/iridae-dev/antecedent/blob/main/provenance/README.md) also require disclosure of implementation exposure, source attribution and material deviations. A record's presence alone does not show that someone has checked every theorem against every code path.

The AIPW record makes its own review boundary explicit. Its `method_review.status` is `reviewed_with_limits`; the review did not retrieve the cited publications. It compared the scoped implementation with standard definitions evaluated by separately written numerical references. This is a bounded implementation review, with that limitation recorded.

## Reconstruct the numerical answer independently

The AIPW review links a [reference generator](https://github.com/iridae-dev/antecedent/blob/main/crates/antecedent-estimate/tests/oracle_data/aipw_reference.py), its [frozen numerical output](https://github.com/iridae-dev/antecedent/blob/main/crates/antecedent-estimate/tests/oracle_data/aipw_reference.json), and the [Rust assertions that consume it](https://github.com/iridae-dev/antecedent/blob/main/crates/antecedent-estimate/tests/aipw_reference.rs).

The reference uses 60-digit arithmetic through mpmath. It fits its own logistic propensity model and arm-specific least-squares outcome models without importing Antecedent. Higher precision helps distinguish rounding from a formula discrepancy; the separate calculation provides the comparison.

For the trimmed ATE, the reference checks uncertainty by a different route from the implementation's influence formula. It perturbs each observation's case mass, refits the estimator and calculates how the final estimate changes. A case mass is the weight an observation receives in that refit. The derivative measures how much that observation contributes to local variation in the fitted answer.

This exposes an important distinction between two sets of observations. Trimming removes rows from the outcome fits and the final average, but the propensity model is still fitted on every row. A trimmed-out observation can therefore affect the estimate through that model. The derivative must account for it.

The reference holds membership of the retained set fixed while taking this derivative. It checks that local calculation; it does not establish behavior at a discontinuous trimming boundary.

### The assertions compare effects and standard errors

The Rust test `trimmed_ate_matches_the_refit_and_its_case_mass_derivative` reads the frozen tables, executes Antecedent's estimator and compares both its effect and analytic standard error. It uses relative tolerances of `1e-8` for the effect and `1e-6` for the standard error.

Separate assertions check treated- and untreated-population estimates against their stated efficient influence functions. Another uses recorded folds to compare the cross-fitted ATE. Those tests answer distinct questions: the treated-population variance check is not presented as the case-mass derivative of the estimator under arbitrary misspecification.

The record reports a defect found through this review. The trimmed estimator's propensity correction originally used only retained rows, even though all rows shaped the propensity fit. The corrected influence accounts for every row. The recorded standard errors changed by about 0.3–0.9% on the reference tables. The [resolved finding and its scope](https://github.com/iridae-dev/antecedent/blob/main/provenance/estimate.aipw.toml) remain beside the evidence.

The example shows why a close point estimate can coexist with an incorrect standard error, and why checking the refitted procedure can reveal more than comparing two implementations of the same influence formula.

### Read the exclusions with the successful comparisons

This AIPW review excludes bootstrap inference, penalized propensities, sampling weights, custom targets, cluster-DML, and cluster, multiway and HAC standard errors. Its multiprecision generator has retained output, but the record says the generator is not run in a locked isolated environment or wired into a gate. The Rust tests compare against its frozen output.

There is also a separate [AIPW conformance fixture](https://github.com/iridae-dev/antecedent/blob/main/conformance/estimate/aipw/expected.json). It checks an estimate against the known effect of a simulated structural model. Its historical DoWhy block ran propensity-score weighting, not AIPW, and the executing assertion does not compare that block. The fixture therefore labels it as context rather than external AIPW evidence.

These are useful checks with different propositions. Combining their names into a general “externally validated AIPW” claim would lose the distinction the records preserve.

## Arrow strength: derive the same quantity two ways

The [arrow-strength reference](https://github.com/iridae-dev/antecedent/blob/main/conformance/attribution/arrow_strength/reference.py) checks linear-Gaussian structural models. An edge's coefficient alone does not describe the variation it transmits: the parent's variance matters too.

In the [fixture](https://github.com/iridae-dev/antecedent/blob/main/conformance/attribution/arrow_strength/expected.json), one parent has variance 100 and coefficient 0.75; another has variance 0.001 and coefficient 2.5. Their variance contributions are respectively:

```text
0.75² × 100   = 56.25
2.5² × 0.001 = 0.00625
```

Ranking by absolute coefficient would reverse that ordering.

The reference also derives the edge's causal strength as a divergence between the original joint Gaussian law and the law after cutting that edge. The cut feeds the child an independent copy of the parent drawn from the parent's marginal distribution. It computes that divergence from covariance matrices, exact rational determinants and inverses, and a 60-digit logarithm. It separately evaluates the closed form:

```text
strength = ½ ln(1 + β² Var(parent) / σ²_child)
```

Here `σ²_child` is the child's disturbance variance. The two calculations agree to `1e-40` on the reference cases, including a model with correlated parents. The [executing attribution tests](https://github.com/iridae-dev/antecedent/blob/main/crates/antecedent-attribution/src/anomaly.rs) compare the product with those expected values.

This reference uses only the Python standard library and retains a locked script environment. Its review still names a shared premise: both derivations use the same definition of the edge cut. Agreement cannot independently rule out a shared misunderstanding of that definition. The fixture also distinguishes asserted outputs from contextual fields and makes no uncertainty or calibration claim.

## Keep a review attached to what was reviewed

Evidence can become stale when an implementation, a reference or an assertion changes. Method-review records bind the provenance content and named artifacts with content digests. Fixture reference reviews similarly record their generators, comparison helpers, environment locks, assertions and numerical payloads.

For example, [the provenance checker](https://github.com/iridae-dev/antecedent/blob/main/scripts/provenance_links.py) compares a method review's stored artifact digests with the files currently present. Editing a bound implementation without updating its review produces a stale-artifact problem. Changing citation or deviation content can also invalidate the recorded review. The [evidence rules](https://github.com/iridae-dev/antecedent/blob/main/parity/README.md) describe the corresponding fixture checks.

A digest establishes which bytes were reviewed. A reviewer must still determine whether the reference is independent, the assertion exercises the claimed behavior, and the scientific correspondence is justified. Updating a hash alone does not perform that review.

## Measure uncertainty as a complete procedure

A numerical reference checks selected calculations. Calibration asks how an uncertainty procedure behaves over repeated datasets from specified generating processes: how often an interval covers its target, how often the procedure reports an answer, and how useful the reported intervals are.

The measured procedure needs to include the operations that affect its answer, such as fitting, sample splitting, selection and resampling. Evidence for individual components does not establish coverage for their composition. Monte Carlo error also matters: an observed coverage fraction is itself an estimate.

Antecedent retains measurement records with their settings and source revision. [Development's coverage documentation](../development.md#coverage-records) explains how code dependencies are assigned to facets and how attestation checks whether a measurement still applies. Relevant changes require affected measurements or the reviewed replay evidence permitted by that process. CI checks applicability; it does not rerun every simulation on every commit.

[Calibration runtime and evidence reuse](../calibration-runtime.md) explains the operational side: frozen measurement sources, seed domains, retained tallies, producer executables and partial-run recovery. Historical measurements and successful component checks do not automatically confer current calibration. Read the matching status for the procedure and build in use.

## Read the evidence in the result

A result reports identification, support, uncertainty, assumptions and calibration separately. [Results are claims](../concepts/results-are-claims.md) explains those fields, and [reading a result](../start/reading-a-result.md) shows how to inspect them.

“Licensed” names a combination with executing support evidence; it is unrelated to the software license. An interval can be implemented without a matching calibration measurement. A model-conditional response can be useful without a design that identifies the real-world effect. Where unlicensed execution is available, successful execution preserves that status.

Export and replay checks preserve specified identities, calculations and interpretation for supported consumers. Artifact acceptance establishes the checks reported by that consumer. Scientific assumptions about the real system remain the researcher's responsibility.

To assess a capability, follow the target through its method record, the independent calculation, the executing assertion and the applicable measurement. Each link should state what it establishes and leave its limits visible to the next person or program using the answer.
