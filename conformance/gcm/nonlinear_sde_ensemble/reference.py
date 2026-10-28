#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent reference for the nonlinear SDE ensemble and quasi-likelihood families.

Three documents are produced from NumPy alone, with no code shared with the production
implementation (which evaluates its coefficients through a portable expression tape):

* ``rng``: values of the counter-based Gaussian generator, defined by 64-bit SplitMix64
  mixing of ``(seed, path, step, component)`` and the cosine branch of Box-Muller.
* ``epidemic``: the documented SIR system with a seasonal input, sqrt-type state-dependent
  noise and reflecting bounds. Euler-Maruyama ensembles at the declared step and at half the
  step share Brownian paths, for a baseline, an input-replacement world ("lockdown") and a
  component-hold world ("suppress", do(I = 20) from t = 5). The production ensemble must replay these moments.
* ``cir``: a mean-reverting square-root diffusion with a piecewise-constant input. Its mean
  and variance obey exact moment ODEs, whatever the noise realization, so the Monte Carlo
  ensemble is checked against a closed form (and the mean against an RK4 integration of the
  moment ODEs).
* ``quasi_likelihood``: trajectories of the epidemic drawn with ``numpy.random.default_rng``
  (a different generator and code path) and the Euler Gaussian transition log likelihood
  evaluated densely at two parameter vectors.

    python3 conformance/gcm/nonlinear_sde_ensemble/reference.py          # write
    python3 conformance/gcm/nonlinear_sde_ensemble/reference.py --check  # compare
"""

from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "expected.json"

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
TWO53 = 9007199254740992.0
N_POP = 1000.0
INV_N = 0.001
SEASON_TIMES = [0.0, 5.0, 10.0, 15.0, 20.0]
SEASON_VALUES = [-1.0, 0.0, 1.0, 0.0, -1.0]


def mix(x):
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = x + GOLDEN
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def counter_normal(seed, path, step, component):
    """Standard normals addressed by (seed, path, step, component); `path` may be an array."""
    key = mix(
        mix(
            mix(mix(np.uint64(seed)) ^ np.asarray(path, dtype=np.uint64))
            ^ np.uint64(step)
        )
        ^ np.uint64(component)
    )
    u1 = ((mix(key) >> np.uint64(11)).astype(np.float64) + 0.5) / TWO53
    u2 = ((mix(key ^ np.uint64(1)) >> np.uint64(11)).astype(np.float64) + 0.5) / TWO53
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def schedule_value(times, values, t, linear):
    n = len(times)
    if t <= times[0]:
        return values[0]
    if t >= times[n - 1]:
        return values[n - 1]
    index = bisect.bisect_right(times, t) - 1
    if not linear:
        return values[index]
    w = (t - times[index]) / (times[index + 1] - times[index])
    return values[index] * (1.0 - w) + values[index + 1] * w


def action_value(action, t, tol):
    times, values = action
    if t + tol < times[0]:
        return None
    return values[bisect.bisect_right(times, t + tol) - 1]


def reflect(x, lower, upper):
    width = upper - lower
    r = np.mod(x - lower, 2.0 * width)
    r = np.where(r > width, 2.0 * width - r, r)
    return lower + r


def epidemic_coefficients(x, u, theta):
    beta, amplitude, gamma = theta
    s, i = x[:, 0], x[:, 1]
    flux = beta * (1.0 + amplitude * u) * s * i * INV_N
    recovery = gamma * i
    drift = np.stack([-flux, flux - recovery, recovery], axis=1)
    variance = np.stack([flux, flux + recovery, recovery], axis=1)
    return drift, np.sqrt(np.maximum(variance, 0.0))


def simulate_epidemic(
    theta, season, actions, initial, start, step, steps, every, paths, seed, fine
):
    """Euler-Maruyama of the epidemic over `paths` paths; returns recorded states."""
    m = 2 if fine else 1
    dt = step / m
    tol = 1e-9 * step
    sqrt_half = np.sqrt(step / 2.0)
    index = np.arange(paths, dtype=np.uint64)
    x = np.tile(np.asarray(initial, dtype=np.float64), (paths, 1))

    def hold(t):
        for variable, action in actions:
            value = action_value(action, t, tol)
            if value is not None:
                x[:, variable] = value

    hold(start)
    recorded = [x.copy()]
    for n in range(steps * m):
        t = start + n * dt
        u = schedule_value(season[0], season[1], t, season[2])
        drift, diffusion = epidemic_coefficients(x, u, theta)
        held = {v for v, a in actions if action_value(a, t, tol) is not None}
        nxt = x.copy()
        for i in range(3):
            if i in held:
                continue
            if fine:
                dw = sqrt_half * counter_normal(seed, index, n, i)
            else:
                dw = sqrt_half * (
                    counter_normal(seed, index, 2 * n, i)
                    + counter_normal(seed, index, 2 * n + 1, i)
                )
            nxt[:, i] = x[:, i] + drift[:, i] * dt + diffusion[:, i] * dw
        x = reflect(nxt, 0.0, N_POP)
        hold(t + dt)
        if (n + 1) % (every * m) == 0:
            recorded.append(x.copy())
    return np.stack(recorded, axis=1)  # paths x times x 3


EPIDEMIC = {
    "theta": [0.6, 0.3, 0.25],
    "initial": [950.0, 50.0, 0.0],
    "start": 0.0,
    "step": 0.05,
    "steps": 400,
    "output_every": 80,
    "paths": 200,
    "seed": 20260101,
}


def epidemic_document() -> dict:
    spec = EPIDEMIC
    season = (SEASON_TIMES, SEASON_VALUES, True)
    lockdown = ([0.0], [-1.0], False)
    worlds = {
        "baseline": (season, []),
        "lockdown": (lockdown, []),
        "suppress": (season, [(1, ([5.0], [20.0]))]),
    }
    runs = {}
    for name, (input_schedule, actions) in worlds.items():
        args = (
            spec["theta"],
            input_schedule,
            actions,
            spec["initial"],
            spec["start"],
            spec["step"],
            spec["steps"],
            spec["output_every"],
            spec["paths"],
            spec["seed"],
        )
        runs[name] = (
            simulate_epidemic(*args, fine=False),
            simulate_epidemic(*args, fine=True),
        )
    out = {"spec": spec, "worlds": {}, "contrasts": {}}
    for name, (coarse, fine) in runs.items():
        out["worlds"][name] = {
            "mean": coarse.mean(axis=0).tolist(),
            "variance": coarse.var(axis=0, ddof=1).tolist(),
            "weak_difference": (coarse.mean(axis=0) - fine.mean(axis=0)).tolist(),
        }
    for name, control in (("lockdown", "baseline"), ("suppress", "baseline")):
        diff = runs[name][0] - runs[control][0]
        out["contrasts"][f"{name}_vs_{control}"] = {
            "mean_difference": diff.mean(axis=0).tolist(),
            "standard_error": (
                diff.std(axis=0, ddof=1) / np.sqrt(spec["paths"])
            ).tolist(),
        }
    return out


CIR = {
    "a": 1.5,
    "b": 0.5,
    "sigma": 0.5,
    "x0": 10.0,
    "knots": [0.0, 5.0],
    "forcing": [1.0, 2.0],
    "times": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
}


def cir_document() -> dict:
    a, b, sigma, x0 = CIR["a"], CIR["b"], CIR["sigma"], CIR["x0"]

    def mean_at(t):
        m5 = a * 1.0 / b + (x0 - a * 1.0 / b) * np.exp(-b * 5.0)
        if t <= 5.0:
            return a * 1.0 / b + (x0 - a * 1.0 / b) * np.exp(-b * t)
        return a * 2.0 / b + (m5 - a * 2.0 / b) * np.exp(-b * (t - 5.0))

    # RK4 on the exact moment ODEs m' = a u - b m, v' = -2 b v + sigma^2 m, per constant segment.
    def integrate(state, u, length, h=1e-3):
        def f(s):
            return np.array([a * u - b * s[0], -2.0 * b * s[1] + sigma * sigma * s[0]])

        for _ in range(int(round(length / h))):
            k1 = f(state)
            k2 = f(state + 0.5 * h * k1)
            k3 = f(state + 0.5 * h * k2)
            k4 = f(state + h * k3)
            state = state + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return state

    mean, variance = [], []
    for t in CIR["times"]:
        first = integrate(np.array([x0, 0.0]), 1.0, min(t, 5.0))
        state = first if t <= 5.0 else integrate(first, 2.0, t - 5.0)
        assert abs(state[0] - mean_at(t)) < 1e-9, (
            "moment ODE and closed-form mean disagree"
        )
        mean.append(float(state[0]))
        variance.append(float(state[1]))
    return {**CIR, "exact_mean": mean, "exact_variance": variance}


def rng_document() -> dict:
    cases = [
        (7, 0, 0, 0),
        (7, 3, 11, 2),
        (20260101, 199, 799, 1),
        (0, 0, 0, 0),
        (123456789, 77, 5, 15),
    ]
    return {
        "cases": [
            {
                "seed": s,
                "path": p,
                "step": n,
                "component": c,
                "value": float(counter_normal(s, p, n, c)),
            }
            for s, p, n, c in cases
        ]
    }


QUASI = {
    "seed": 99,
    "step": 0.05,
    "steps": 400,
    "trajectories": 2,
    "initials": [[950.0, 50.0, 0.0], [900.0, 100.0, 0.0]],
    "truth": [0.6, 0.3, 0.25],
    "alternative": [0.5, 0.2, 0.3],
}


def quasi_log_likelihood(theta, times, states):
    total = 0.0
    for k in range(len(times) - 1):
        dt = times[k + 1] - times[k]
        u = schedule_value(SEASON_TIMES, SEASON_VALUES, times[k], True)
        drift, diffusion = epidemic_coefficients(states[k : k + 1], u, theta)
        variance = diffusion[0] ** 2 * dt
        residual = states[k + 1] - states[k] - drift[0] * dt
        total += float(
            np.sum(-0.5 * (np.log(2.0 * np.pi * variance) + residual**2 / variance))
        )
    return total


def quasi_document() -> dict:
    spec = QUASI
    rng = np.random.default_rng(spec["seed"])
    units = []
    for initial in spec["initials"]:
        x = np.array(initial, dtype=np.float64)
        times = [k * spec["step"] for k in range(spec["steps"] + 1)]
        rows = [x.copy()]
        for k in range(spec["steps"]):
            u = schedule_value(SEASON_TIMES, SEASON_VALUES, times[k], True)
            drift, diffusion = epidemic_coefficients(x[None, :], u, spec["truth"])
            x = (
                x
                + drift[0] * spec["step"]
                + diffusion[0] * np.sqrt(spec["step"]) * rng.standard_normal(3)
            )
            x = reflect(x, 0.0, N_POP)
            rows.append(x.copy())
        units.append({"times": times, "values": np.stack(rows).tolist()})
    for unit in units:
        states = np.array(unit["values"])
        unit["log_likelihood_truth"] = quasi_log_likelihood(
            spec["truth"], unit["times"], states
        )
        unit["log_likelihood_alternative"] = quasi_log_likelihood(
            spec["alternative"], unit["times"], states
        )
    return {
        **spec,
        "units": units,
        "log_likelihood_truth": sum(u["log_likelihood_truth"] for u in units),
        "log_likelihood_alternative": sum(
            u["log_likelihood_alternative"] for u in units
        ),
    }


def build() -> dict:
    return {
        "oracle": {
            "kind": "independent_numpy_implementation",
            "method": (
                "NumPy re-implementation of the counter-based generator, Euler-Maruyama with "
                "shared half-step Brownian paths, reflecting bounds, held components and replaced "
                "input schedules; exact moment ODEs of a square-root diffusion with a "
                "piecewise-constant input; dense Euler transition log likelihood."
            ),
            "independence_basis": (
                "Coefficients are written out in NumPy rather than evaluated through the "
                "expression tape; the quasi-likelihood data come from numpy.random.default_rng. "
                "The author of the reference is the repository author and independence from the "
                "production implementation has not been reviewed."
            ),
            "generator": "conformance/gcm/nonlinear_sde_ensemble/reference.py",
            "historical_exposure": "Historical author exposure is unrecorded and remains unknown.",
        },
        "schema_version": 1,
        "rng": rng_document(),
        "epidemic": epidemic_document(),
        "cir": cir_document(),
        "quasi_likelihood": quasi_document(),
    }


def main() -> int:
    if "--check" in sys.argv[1:]:
        committed = json.loads(EXPECTED.read_text(encoding="utf-8"))
        fresh = build()
        problems = []
        if fresh["oracle"] != committed["oracle"]:
            problems.append("oracle record differs")

        def compare(name, a, b):
            if not np.allclose(
                np.asarray(a, dtype=float),
                np.asarray(b, dtype=float),
                rtol=1e-9,
                atol=1e-9,
            ):
                problems.append(name)

        for name, world in fresh["epidemic"]["worlds"].items():
            for key, value in world.items():
                compare(
                    f"epidemic.{name}.{key}",
                    value,
                    committed["epidemic"]["worlds"][name][key],
                )
        for name, contrast in fresh["epidemic"]["contrasts"].items():
            for key, value in contrast.items():
                compare(
                    f"epidemic.{name}.{key}",
                    value,
                    committed["epidemic"]["contrasts"][name][key],
                )
        compare("cir.mean", fresh["cir"]["exact_mean"], committed["cir"]["exact_mean"])
        compare(
            "cir.variance",
            fresh["cir"]["exact_variance"],
            committed["cir"]["exact_variance"],
        )
        compare(
            "rng",
            [c["value"] for c in fresh["rng"]["cases"]],
            [c["value"] for c in committed["rng"]["cases"]],
        )
        # Stored data are evaluated, not regenerated.
        stored = committed["quasi_likelihood"]
        total = sum(
            quasi_log_likelihood(stored["truth"], u["times"], np.array(u["values"]))
            for u in stored["units"]
        )
        compare("quasi_likelihood.truth", total, stored["log_likelihood_truth"])
        for problem in problems:
            print(f"nonlinear sde reference mismatch: {problem}", file=sys.stderr)
        return 1 if problems else 0
    EXPECTED.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
