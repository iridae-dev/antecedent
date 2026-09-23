# A result is a claim, not a number

`0.23` is almost no scientific meaning by itself. A useful result says what
was intervened on, for whom, relative to what, under which structure, how the
claim was identified and estimated, what data supported it, what uncertainty it
represents, and which assumptions remain active.

Antecedent therefore returns answer shapes rather than treating every analysis
as a scalar:

| Shape | Meaning |
| --- | --- |
| `point` | A scalar claim under its recorded contract. |
| `bounds` | A set/range determined by the declared structural knowledge. |
| `partial` | A claim with unresolved identification or structural components. |
| `response` | A curve, surface, derivative, or intervention response. |
| `structured` | Several linked causal results. |
| `unavailable` | No usable claim, with a reason rather than fabricated precision. |

Bounds are not automatically confidence intervals. A confidence interval does
not automatically represent graph uncertainty. An exported result preserves a
contract; loading it does not prove the real-world causal assumptions true.

Use `inspect()` before projecting a result to a number, report, dashboard, or
API payload. The [Python workflow](python-workflow.md#read-the-answer) shows
the corresponding programmatic interface.
