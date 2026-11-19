# When outside events interrupt a launch

You can open a launch as planned, or hold load below a limit at a cost. Outside announcements can arrive in bursts. A quiet hour and an hour with several announcements leave you facing different conditions, even if the rest of the launch plan is identical.

- **You will learn** how to fit a model to event timestamps, compare two actions on the same simulated arrivals, and value one more hour of observations.
- **Before you start** read [how to build from source](../start/installation.md#build-from-source). Use a development build while the routes await release evidence.
- **Run it** with `python examples/python/event_driven_launch.py`. The [complete example](#the-complete-example) generates its own history.

If the installed package can inspect the plan but cannot execute it, build the checkout with `maturin develop --release --manifest-path python/Cargo.toml --features development` in your active virtual environment.

## Fit to the time you actually observed

The example generates 600 hours of announcements. Some arrive independently. An announcement also makes another one more likely for a short time. The script tells the model when recording began, when it stopped, and which event type the timestamps describe. The four-hour forecast starts at the cutoff. It has no access to events beyond that time.

The Hawkes model fits a background rate and the expected number of direct follow-up arrivals per announcement. The example supplies the rate at which that extra activity fades. Read the fit report before using the forecast. A weak fit or a parameter at the fitted boundary is worth investigating even when the plan can produce paths.

## Compare actions under the same announcements

Each simulated arrival adds a pulse to the input that drives load. Both actions see that same pulse at the same time and use the same background fluctuations. The open launch lets load evolve. The hold action fixes it at 0.25 and pays a declared utility cost. The decision compares the average utility of the two actions across paired forecasts.

The outcome here is peak load over every solver step. A second decision contract reads the probability of staying at or below 0.8. It uses the same forecast and noise seed and leaves the business cost out of that physical risk calculation. The report also compares the answer with a grid that uses half the step size. A large difference means the numerical resolution matters. A threshold measured on that grid answers whether a recorded grid state exceeded the threshold.

## Decide what another hour of observation would buy

Suppose the decision will be made an hour later either way. Before deciding, you might learn whether an announcement arrived during that hour. The example compares the best action with that signal against the best action without it, with both choices made at the same time.

The study carries the arrivals in the observation hour into the continuation. Even with one fitted parameter setting, a recent announcement can make another arrival more likely. Read `baseline_action`, `branch_actions`, and `net_value` together. The branch actions show what the observation could change. Net value subtracts the stated cost of observing in the same units as the decision utility. The result depends on the modeled signal and the candidate actions.

## What this established

The same compiled forecast can supply both actions. The study compares choices with and without the observation at a common decision time. The script checks that the observation probabilities sum to one and that the gross value is nonnegative up to numerical tolerance.

## Where to go next

The example uses synthetic data and a declared causal model for load. The fit describes clusters of announcements in that history. The effect of holding load comes from the declared equations. For rolling forecasts, provider paths, grid diagnostics, and the separately gated count interval, see [event forecasts and stochastic decisions](../guides/event-processes.md).

## The complete example

<!-- example: examples/python/event_driven_launch.py -->
