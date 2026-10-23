#!/usr/bin/env python3
"""First answer for nonlinear Bayesian inference.

One `analyze` call fits a Bernoulli-logit outcome model by Hamiltonian Monte Carlo and
returns the average treatment effect as a risk difference with its posterior interval. The
requested diagnostics are inside the call: the sampler's effective-sample-size, split-R-hat
and chain-movement publication gates, and the measured draw control that refuses an interval
whose Monte Carlo error is too large. Data synthesis sits outside the timed section.

    python parity/performance_lanes/entry_points/nonlinear_bayes.py --role control
    python parity/performance_lanes/entry_points/nonlinear_bayes.py --role candidate
"""

from __future__ import annotations

from typing import Any

import antecedent
import numpy as np
from _entry import arguments, digest_arrays, run_entry

DATA_SEED = 3
EXECUTION_SEED = 2
ROWS = 3000
DRAWS = 4000
GRAPH = [("z", "t"), ("z", "y"), ("t", "y")]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def synthesize(rows: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(DATA_SEED)
    z = rng.normal(size=rows)
    t = (rng.uniform(size=rows) < sigmoid(z)).astype(np.float64)
    y = (rng.uniform(size=rows) < sigmoid(-0.5 + 1.2 * t + 0.8 * z)).astype(np.float64)
    return {"t": t, "y": y, "z": z}


def main() -> None:
    args = arguments(__doc__ or "")
    data = synthesize(args.rows or ROWS)
    query = antecedent.AverageEffect(treatment="t", outcome="y")
    inference = antecedent.Bayesian(
        n_draws=DRAWS, backend="hmc", likelihood="logit", draw_controls="measured"
    )

    def section() -> Any:
        return antecedent.analyze(
            data,
            graph=GRAPH,
            query=query,
            inference=inference,
            refute="none",
            seed=EXECUTION_SEED,
            threads=args.threads,
        )

    def fields(result: Any) -> dict[str, Any]:
        posterior = result.posterior
        return {
            "effect": result.answer.value,
            "posterior_mean": None if posterior is None else posterior.effect_mean,
            "posterior_sd": None if posterior is None else posterior.effect_sd,
            "interval": None if posterior is None else (posterior.q025, posterior.q975),
            "draws": None if posterior is None else posterior.n_draws,
            "backend": None if posterior is None else posterior.backend,
            "diagnostic_codes": sorted(line.split(":", 1)[0] for line in result.diagnostics),
        }

    run_entry(
        "nonlinear_bayes",
        args,
        digest_arrays(data),
        section,
        fields,
        required=("effect", "interval", "draws", "backend"),
    )


if __name__ == "__main__":
    main()
