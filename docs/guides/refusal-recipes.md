# Refusal, repair, and a usable answer

A refusal identifies an unmet scientific or execution requirement. Repair the declaration or supply the missing evidence; do not catch the error and substitute a more convenient number.

This executable recipe shows a retained randomized design that refuses changed assignments, preserves the preceding result, and then accepts explicit repreparation. It also demonstrates a declared mixture whose spread cannot be read as a licensed interval.

<!-- example: examples/python/refusal_recipes.py -->

## Choose the repair for the failure

| Failure | Inspect | Scientifically valid next step |
| --- | --- | --- |
| Uncertain graph directions | Answer kind, envelope, unidentified mass | Keep the class-preserving answer or review directions with substantive knowledge; see [graph uncertainty](../graph_uncertainty.md). |
| Insufficient overlap | Support and per-coordinate diagnostics | Restrict the scientific target only when justified, or collect evidence; see [tier diagnostics](tier-diagnostics.md). |
| Changed retained design | Structured error and named offending column | Reprepare under the changed design; never pretend it is an observation-only refresh. |
| Missing transport evidence | Identification certificate and support slots | Supply the required population/regime law; see [transport failures](transport-failure.md). |
| Mean offered where a law is required | Consumer contract and remedy | Supply the required joint law, or use a criterion that only needs the declared mean. |
| Interval outside measured scope | Calibration status and reason | Report the actual standing; do not inherit a component's coverage. |

The [full refusal reference](../reference/refusal-codes.md) lists registered details and remedies. The [transport examples](../examples.md) provide neighboring successes and failures for exact-law and complementary-source identification.

## Read a refusal in code

Every stage module raises a subclass of `antecedent.errors.StructuredRefusal`, which
exposes the same machine-readable fields on every class. Branch on `code` and `detail`,
not on the message text:

```python
from antecedent import msm_sensitivity
from antecedent.errors import CausalUnsupportedError, StructuredRefusal

try:
    # a perturbation bound below one is refused
    msm_sensitivity.msm_ate_sensitivity(strata, 0.5)
except StructuredRefusal as refusal:
    refusal.code        # "invalid_argument" here; a scientific refusal has e.g. "cell_not_licensed"
    refusal.detail      # "msm_sensitivity.lambda_below_one"
    refusal.offending   # the field, quantity, node or identity at fault, or None
    refusal.remedy      # what to change in order to proceed, or None
except CausalUnsupportedError as refusal:
    # ExternalRefusal and the design-ranking refusals are CausalUnsupportedError with the
    # same fields: reason_code, remedy, and (where present) detail / offending / stage.
    refusal.reason_code, refusal.remedy
```

A refusal whose code is `invalid_argument` rejects a malformed argument rather than a
scientific route and is also a `ValueError`; an operational failure such as cancellation is
not a scientific verdict. `strata` is the stratified table of the
[sensitivity guide](../sensitivity-and-robustness.md). See
[refusal codes and remedies](../reference/refusal-codes.md#structured-refusals-and-their-remedies)
for the base class, the families and the common remedies.

## Run neighboring structural and evidence outcomes

For uncertain graph directions, run [class-preserving CPDAG](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/class_preserving_cpdag.py). It asserts that the uncertain answer has no scalar value, then demonstrates an explicitly reviewed orientation. The orientation is supplied scientific knowledge, not a numerical trick to force a point.

For missing transport evidence, run [complementary sources](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/transport_meta_grid.py). The same query returns supported response values with the complete laws and retains unsupported coordinates when a required law is removed. Restore the missing evidence rather than dropping that coordinate from the report.

```bash
python examples/python/class_preserving_cpdag.py
python examples/python/transport_meta_grid.py
```

For positivity, [propensity weighting](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/propensity_weighting.py) prints effective sample size and extreme propensities. These are diagnostics to interpret, not a blanket permission to trim: restricting the population or clipping weights changes the analysis and requires its own declared settings and standing.
