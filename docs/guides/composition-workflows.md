# Choose a composition workflow

Start with the decision or scientific question you have. The examples below connect the data, the model and the final result in one inspectable program. Each walkthrough states what it assumes. It states what the result can tell you.

| If you want to… | Start here | Example API |
| --- | --- | --- |
| Compare actions in a fitted feedback loop | [Compare actions after fitting the feedback](../tutorials/feedback-three-questions.md#compare-actions-after-fitting-the-feedback) | Rust |
| Apply a trial result to a new population | [Move a trial effect into a target decision](transported-decisions-and-measurement.md#move-a-trial-effect-into-a-target-decision) | Rust |
| Check whether an early measure predicts a later outcome in that population | [Check an early measure in both populations](transported-decisions-and-measurement.md#check-an-early-measure-in-both-populations) | Rust |
| Trace a treatment effect through selected links over time | [Follow only the treatment links you chose](temporal-designs.md#follow-only-the-treatment-links-you-chose) | Rust |
| Keep repeated observations together when estimating uncertainty | [Resample whole groups through the analysis](composed-uncertainty.md#resample-whole-groups-through-the-analysis) | Rust |
| Use two measurements that can be wrong together | [Fit a joint error table from labelled readings](transported-decisions-and-measurement.md#fit-a-joint-error-table-from-labelled-readings) | Rust |
| Evaluate rules on action logs kept apart from training | [Evaluate a rule on a separate log](../tutorials/programmable-policies.md#evaluate-a-rule-on-a-separate-log) | Rust |
| Save a fitted decision and reproduce it later | [Replay a fitted decision](../tutorials/replay-and-current-decisions.md#replay-a-fitted-decision) | Python core tutorial and Rust fitted example |
| Look for different ways a recorded action was delivered | [Discover latent soft interventions](latent-soft-interventions.md#two-rollouts-with-the-same-headline-effect) | Python and Rust route, Python example |
| Forecast clustered arrivals and use them in a decision | [When outside events interrupt a launch](../tutorials/event-driven-launch.md) | Python and Rust route, Python example |

## Read the evidence status

Before you execute a plan, read its `support` field ([plan view](plan-view.md)). It reports whether your installed build supports that call. The [support matrix](../support-matrix.md) lists the supported units and how support is decided. The fitted feedback, target decision, and other Rust examples describe the scope of the implemented methods. Where a result says `not_measured`, its uncertainty has no matching calibration measurement.

The examples use declared models or synthetic data. Their checks cover the computation shown, such as replaying a decision or matching a known path effect. Use [scientific verification](scientific-verification.md) to find how Antecedent establishes support and calibration for a particular call.
