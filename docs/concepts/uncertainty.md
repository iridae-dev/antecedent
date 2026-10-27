# Uncertainty

A causal answer can be uncertain in different ways, and Antecedent keeps the ways apart because they call for different responses. Reporting one interval for all of them would hide which one it covers.

## Kinds of uncertainty

| Kind | What it is about | How it appears |
| --- | --- | --- |
| Sampling | The finite data behind an estimate | A standard error, bootstrap or posterior interval, with a recorded construction |
| Parameter | Fitted parameters of a model | Posterior draws or a covariance |
| Structural (graph) | Which causal structure holds, when it is discovered or only partly oriented | A posterior over graphs, or an envelope across completions of a CPDAG or PAG, with `unidentified_mass` kept visible |
| Assumption | A premise that data cannot test, such as no unmeasured confounding or invariance across populations | A declared range or sensitivity grid, and where the conclusion changes |
| Numerical | Finite computation or simulation | A Monte Carlo band or numerical error; not a sampling interval |

Priors do not upgrade an unidentified atom to an identified one, and a failed estimation on an identified graph is *unevaluable* mass, a different axis from unidentified mass.

## Structural uncertainty

Discovered structure is evidence. Antecedent keeps compatible graphs in play, reports the belief on which the effect is not identified, and refuses to invent orientations the data do not support. An exact DAG posterior assigns probability to each graph; a CPDAG or PAG stands for many completions, and enumeration weight over completions is not posterior probability unless you supply a class prior. Because the identified-conditional mean excludes the unidentified share, a lone effect estimate is incomplete when unidentified mass is large enough to change a decision. Report the split. See [graph uncertainty](../graph_uncertainty.md).

## Assumption ranges are not intervals

A sensitivity range says what a quantity could be if the assumption holds anywhere inside the declared set. It is not a probability or a confidence interval, so the routes decline to put weights on it or add it to a sampling interval. Report the sampling interval beside the range, and state where the best action switches. See [sensitivity and robustness](../sensitivity-and-robustness.md).

## Joint uncertainty for contrasts and compositions

When an answer combines several uncertain quantities, the dependence between them matters. Rows from different fits are never paired by index; a row pairing is the statement that the rows are realizations of one joint law. Linear combinations and contrasts need a covariance or an aligned joint law; nonlinear functionals need an aligned joint law, and a covariance alone does not suffice. Subtracting two marginal interval endpoints does not give the interval of the contrast. If the dependence is unknown, the request is refused with a code that names the missing declaration. See [composition-aware uncertainty](../joint-law-propagation.md).

## Reading calibration status

An interval has a construction (how it was formed) and, separately, a calibration status (whether a measured record supports it). Read it from the result:

```python
result.inspect().calibration.status
```

| Status | Meaning |
| --- | --- |
| `calibrated` | A coverage record matches this execution, the execution is inside that record's measured scope, and the record still attests the current code |
| `scope_not_assessed` | A record matches, but the execution is outside its scope, the record marks a boundary, or the record is stale or no longer attesting |
| `unavailable` | No record exists; a reason code says why |

"Calibrated" is a property of a matching measurement that still applies, not of the method in general. A passing diagnostic, an identified effect and a successful run do not establish it. An interval can be a useful, well-defined construction and still be uncalibrated for the setting you ran.

A route without a matching record reports `unavailable` with its reason, even when its interval is well defined. Read each result's status rather than assuming one; the [support matrix](../support-matrix.md) lists which constructions have measurements.

## Point-only and enclosures

Some routes return a value with no interval, because no interval claim is licensed. A finite-law decision value can be exact under the supplied law and still lack a sampling interval. Others return a structural envelope or an assumption range. Each label says what the range ranges over; none is a confidence statement unless its calibration status says so.
