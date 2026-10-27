"""Evaluate a longitudinal treatment regime by inverse-probability g-formula.

`antecedent.regimes.evaluate_regime_value` estimates the value of a prespecified
static or history-adaptive treatment rule under caller-declared sequential
randomization probabilities. It reweights the observed trajectories that follow
the regime and reports the value, effective sample size, matched fraction, and
maximum weight. It is a point utility (`uncertainty == "not_estimated"`) and
does not verify that the supplied probabilities are correct.

The fixture is simulated with a declared mechanism, so its answers are known.
Two randomization times at p=0.5 give four equally likely treatment paths, one
observed per path, and the outcome is 2 + 1.0 a1 + 0.5 a2 for treatments a1 and
a2. Each regime's value is therefore the outcome of its own path: 2.0 for never
treat, 3.0 for treat first only, 3.5 for always treat. These are fixture truths
of that declared mechanism, not claims about a real trial. With one observed
trajectory per path, each estimate rests on a single matched row.
Install with `python -m pip install antecedent`; see examples/README.md."""

from __future__ import annotations

import numpy as np
from antecedent.regimes import evaluate_regime_value


def main() -> None:
    histories = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outcomes = [2.0 + 1.0 * a1 + 0.5 * a2 for a1, a2 in histories]
    probabilities = np.full((4, 2), 0.5)
    truth = {(False, False): 2.0, (True, False): 3.0, (True, True): 3.5}

    for regime, expected in truth.items():
        result = evaluate_regime_value(
            outcomes=outcomes,
            treatment_history=histories,
            regime=list(regime),
            treatment_probabilities=probabilities,
        )
        print(
            f"regime {regime}: value={result.value} (truth {expected}), "
            f"effective sample size={result.effective_sample_size}, "
            f"matched fraction={result.matched_observed_fraction}, "
            f"maximum weight={result.maximum_weight}"
        )
        # The value recovers the declared mechanism's outcome for that path.
        assert abs(result.value - expected) < 1e-9, (regime, result.value)
        # One path matches one observed row at probability 0.25: weight 4, ESS 1.
        assert abs(result.effective_sample_size - 1.0) < 1e-9
        assert abs(result.matched_observed_fraction - 0.25) < 1e-9
        assert abs(result.maximum_weight - 4.0) < 1e-9
        # The value is a point utility; the probabilities are the caller's declaration.
        assert result.uncertainty == "not_estimated"
        assert result.support_status == "caller_supplied_sequential_probabilities"
    print("Read this as a value under declared probabilities, with no interval.")


if __name__ == "__main__":
    main()
