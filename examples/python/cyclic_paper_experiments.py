# SPDX-License-Identifier: MIT OR Apache-2.0
"""Independent cyclic paper protocols using Antecedent's native estimators.

Paper-derived data protocols, not translated reference implementations. The
Gaussian Bicycle lane is its applicable no-technical-noise synthetic setting;
this program makes no claim to reproduce its count-likelihood variational model.
NODAGS uses its cyclic linear/ReLU data families and joint-intervention holdouts,
with exact native likelihoods and explicit proximal fitting instead of a neural
mask/log-determinant estimator. Every deviation and failed outcome is retained.

Run from an installed release wheel. `--full` uses published feature/sample
sizes; reduced defaults are an engineering smoke test, never a paper reproduction.
Each fitted request is exported and independently refitted in a fresh CPU process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from antecedent import Execution
from antecedent.cyclic import AdamPolicy, Equation, EquilibriumModel, OuMechanisms
from antecedent.outcomes import require_success
from scipy.linalg import solve_continuous_lyapunov

PAPERS = {
    "nodags": "https://proceedings.mlr.press/v206/sethuraman23a/sethuraman23a.pdf",
    "bicycle": "https://proceedings.mlr.press/v236/rohbeck24a/rohbeck24a.pdf",
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def average_precision(truth, scores):
    truth = np.asarray(truth, dtype=bool)
    order = np.argsort(-np.asarray(scores), kind="stable")
    truth, scores = truth[order], np.asarray(scores)[order]
    total = int(truth.sum())
    if not total:
        return None
    hits, old_recall, result = 0, 0.0, 0.0
    for i in range(len(truth)):
        hits += int(truth[i])
        if i == len(truth) - 1 or scores[i] != scores[i + 1]:
            recall = hits / total
            result += (recall - old_recall) * hits / (i + 1)
            old_recall = recall
    return float(result)


def replay(result, directory):
    artifact = directory / "fit.ant"
    encoded = bytes(result.export())
    artifact.write_bytes(encoded)
    (directory / "expected_identity.txt").write_text(result.identity + "\n")
    code = (
        "import json,pathlib;from antecedent.cyclic import consume_family_artifact;"
        "from antecedent.outcomes import require_success;"
        f"r=require_success(consume_family_artifact(pathlib.Path({str(artifact)!r}).read_bytes(),expected_identity={result.identity!r}));"
        "print(json.dumps(r.to_dict(),sort_keys=True,separators=(',',':')))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=directory,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    reconstructed = json.loads(process.stdout)
    if reconstructed != result.to_dict():
        raise AssertionError("fresh-process scientific replay differs")
    return {
        "identity": result.identity,
        "bytes": len(encoded),
        "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
        "fresh_process_cpu_replay": True,
    }


def erdos(rng, n, bicycle=False):
    mask = rng.random((n, n)) < 2 / (n - 1)
    np.fill_diagonal(mask, False)
    if bicycle:
        weights = rng.uniform(0.25, 0.95, (n, n)) * rng.choice([-1, 1], (n, n))
    else:
        # Paper specifies uniform weights and post-scaling, not the interval.
        weights = rng.uniform(-0.5, 0.5, (n, n))
    return mask, weights * mask


def joint_targets(rng, n, count, avoid=()):
    seen = set(avoid)
    output = []
    for _ in range(10000):
        k = min(n, 2 + int(rng.random() < 0.5))
        target = tuple(sorted(rng.choice(n, k, replace=False).tolist()))
        if target not in seen:
            seen.add(target)
            output.append(target)
            if len(output) == count:
                return output
    raise ValueError("not enough distinct held-out joint intervention regimes")


def equilibrium_data(rng, weights, targets, count, activation, sd=0.5):
    n = len(weights)
    noise = rng.normal(0, sd, (count, n))
    assigned = rng.normal(size=(count, len(targets)))
    state = np.zeros((count, n))
    for _ in range(500):
        next_state = state @ weights.T
        if activation == "relu":
            next_state = np.maximum(next_state, 0)
        next_state += noise
        next_state[:, targets] = assigned
        if np.max(np.abs(next_state - state)) < 1e-11:
            state = next_state
            break
        state = next_state
    else:
        raise AssertionError("independent contraction generator did not converge")
    return state


def rows(values, targets):
    return [
        {
            "values": list(map(float, row)),
            "clamps": [{"variable": int(i), "value": float(row[i])} for i in targets],
        }
        for row in values
    ]


def nonlinear_nll(weights, intercepts, regimes, activation, sd=0.5):
    total, free_count = 0.0, 0
    n = len(weights)
    for values, targets in regimes:
        free = [i for i in range(n) if i not in targets]
        for start in range(0, len(values), 512):
            x = values[start : start + 512]
            predictor = x @ weights.T
            active = (predictor > 0).astype(float) if activation == "relu" else np.ones_like(x)
            mechanism = np.maximum(predictor, 0) if activation == "relu" else predictor
            residual = (x - intercepts - mechanism)[:, free]
            jac = (
                np.eye(len(free))[None, :, :]
                - active[:, free, None] * weights[np.ix_(free, free)][None, :, :]
            )
            sign, logdet = np.linalg.slogdet(jac)
            if np.any(sign == 0):
                return None
            total += float(
                np.sum(
                    0.5
                    * (
                        np.sum((residual / sd) ** 2, axis=1)
                        + len(free) * np.log(2 * np.pi * sd * sd)
                    )
                    - logdet
                )
            )
            free_count += len(x) * len(free)
    return total / free_count


def nodags(args, seed, output):
    rng = np.random.default_rng(seed)
    n, per_regime = (20, 5000) if args.full else (4, 50)
    activation = "identity" if args.lane == "nodags-linear" else "relu"
    mask, weights = erdos(rng, n)
    # A documented sufficient contractivity normalization, stronger than spectral
    # normalization. It avoids treating sampled derivative norms as certificates.
    weights *= 0.8 / max(float(np.abs(weights).sum(axis=1).max()), 0.8)
    training = []
    observational_count = 20000 if args.full else 0
    if observational_count:
        training.extend(
            rows(equilibrium_data(rng, weights, (), observational_count, activation), ())
        )
    for i in range(n):
        data = equilibrium_data(rng, weights, (i,), per_regime, activation)
        training.extend(rows(data, (i,)))
    count = 20 if args.full else 3
    validation_targets = joint_targets(rng, n, min(10, count))
    test_targets = joint_targets(rng, n, count, validation_targets)
    validation = [
        (equilibrium_data(rng, weights, target, per_regime, activation), target)
        for target in validation_targets
    ]
    heldout = [
        (equilibrium_data(rng, weights, target, per_regime, activation), target)
        for target in test_targets
    ]
    initial = rng.uniform(-0.001, 0.001, (n, n)) if activation == "relu" else np.zeros((n, n))
    np.fill_diagonal(initial, 0)
    template = EquilibriumModel(
        tuple(Equation(f"mechanism:{i}", 0.0, tuple(initial[i]), activation) for i in range(n)),
        tuple((np.eye(n) * 0.25).ravel()),
    )
    parameters = [{"kind": "intercept", "equation": i} for i in range(n)] + [
        {"kind": "weight", "equation": i, "parent": j} for i in range(n) for j in range(n) if i != j
    ]
    attempts = []
    best = None
    for penalty in (0.001, 0.01):
        prepared = (
            template.fit_adam(
                parameters,
                training,
                [row for values, targets in validation for row in rows(values, targets)],
                policy=AdamPolicy(
                    max_epochs=min(args.max_iterations, 200),
                    minimum_epochs=20,
                    patience=10,
                    l1_penalty=penalty,
                ),
                execution=Execution(seed=seed),
            )
            if activation == "relu"
            else template.fit_gradient(
                parameters,
                training,
                learning_rate=0.3,
                score_tolerance=0.001 * len(training),
                max_iterations=args.max_iterations,
                l1_penalty=penalty * len(training),
            )
        )
        result = require_success(prepared.execute())
        fit = result.to_dict()["answer"]["result"]
        if activation == "relu":
            fit = dict(
                fit,
                converged=fit["validation_converged"] or fit["training_score_converged"],
                score_norm=fit["training_score_per_row"] * len(training),
                iterations=fit["epochs"],
                line_search_exhausted=False,
            )
        estimated = np.array([eq["weights"] for eq in fit["model"]["equations"]])
        intercepts = np.array([eq["intercept"] for eq in fit["model"]["equations"]])
        val = nonlinear_nll(estimated, intercepts, validation, activation)
        attempts.append(
            {
                "penalty_per_row": penalty,
                "validation_nll": val,
                "converged": fit["converged"],
                "score_per_row": fit["score_norm"] / len(training),
                "iterations": fit["iterations"],
                "line_search_exhausted": fit["line_search_exhausted"],
            }
        )
        print(
            json.dumps({"seed": seed, "lane": args.lane, "attempt": attempts[-1]}),
            flush=True,
        )
        if val is not None and (best is None or val < best[0]):
            best = (val, result, fit, estimated, intercepts, prepared)
    if best is None:
        raise AssertionError("all likelihood candidates failed")
    _, result, fit, estimated, intercepts, prepared = best
    write(output / "request.json", json.loads(prepared.request_json))
    write(output / "native_report.json", result.to_dict())
    off = ~np.eye(n, dtype=bool)
    baseline = nonlinear_nll(np.zeros_like(weights), np.zeros(n), heldout, activation)
    fitted = nonlinear_nll(estimated, intercepts, heldout, activation)
    oracle = nonlinear_nll(weights, np.zeros(n), heldout, activation)
    return {
        "lane": args.lane,
        "seed": seed,
        "mode": "paper_size" if args.full else "smoke",
        "paper": PAPERS["nodags"],
        "paper_sections": ["5.1", "B.1", "B.3"],
        "nodes": n,
        "samples_per_single_intervention": per_regime,
        "training_rows": len(training),
        "observational_samples": observational_count,
        "validation_targets": validation_targets,
        "test_targets": test_targets,
        "heldout_regimes_disjoint_from_training": True,
        "attempts": attempts,
        "fit_converged": fit["converged"],
        "auprc": average_precision(mask[off], np.abs(estimated[off])),
        "weight_rmse": float(np.sqrt(np.mean((weights[off] - estimated[off]) ** 2))),
        "heldout_nll_per_free_coordinate": fitted,
        "oracle_nll_per_free_coordinate": oracle,
        "null_nll_per_free_coordinate": baseline,
        "deviations": [
            "Conservative infinity-operator-norm contraction normalization, not spectral rescaling.",
            "Known independent noise sd=0.5; uniform raw coefficients in [-0.5,0.5].",
            "Exact full Jacobians and proximal likelihood fitting in the data-generating single-layer family; no copied neural-mask or stochastic logdet code.",
            "Two predeclared penalties selected by disjoint joint-intervention validation; final joint test regimes never select a model.",
        ],
        "artifact": replay(result, output),
    }


def ou_law(drift, forcing, factor, targets):
    matrix, bias, loading = drift.copy(), forcing.copy(), factor.copy()
    replacements = []
    for i in targets:
        matrix[i] = 0
        matrix[i, i] = -1
        bias[i] = 0.1
        loading[i] = 0
        loading[i, i] = 0.001
        replacements.append(
            {
                "equation": f"equation:{i}",
                "drift": matrix[i].tolist(),
                "forcing": float(bias[i]),
                "diffusion_factor": loading[i].tolist(),
            }
        )
    if np.linalg.eigvals(matrix).real.max() >= 0:
        return None, replacements
    mean = np.linalg.solve(-matrix, bias)
    covariance = solve_continuous_lyapunov(matrix, -(loading @ loading.T))
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise AssertionError("independent OU covariance is not positive definite")
    return (mean, covariance), replacements


def gaussian_nll(values, mean, covariance):
    _, logdet = np.linalg.slogdet(covariance)
    residual = values - mean
    return float(
        0.5
        * (
            np.sum(residual * np.linalg.solve(covariance, residual.T).T)
            + len(values) * (logdet + len(mean) * np.log(2 * np.pi))
        )
    )


def bicycle(args, seed, output):
    rng = np.random.default_rng(seed)
    n, count, controls = (10, 250, 500) if args.full else (4, 30, 60)
    all_pairs = [(i, j) for i in range(n) for j in range(i)]
    chosen = rng.choice(len(all_pairs), min(20, len(all_pairs)), replace=False)
    pairs = [all_pairs[int(i)] for i in chosen]
    rejected = 0
    while True:
        mask, beta = erdos(rng, n, bicycle=True)
        drift = beta - np.eye(n)
        targets = [()] + [(i,) for i in range(n)] + pairs
        if all(ou_law(drift, np.ones(n), np.eye(n) * 0.1, t)[0] is not None for t in targets):
            break
        rejected += 1
        if rejected > 1000:
            raise AssertionError("stable OU graph sampling exhausted")
    observed = []
    for target in [()] + [(i,) for i in range(n)]:
        law, replacements = ou_law(drift, np.ones(n), np.eye(n) * 0.1, target)
        data = rng.multivariate_normal(*law, controls if not target else count)
        observed.append(
            {
                "id": "observational" if not target else f"single:{target[0]}",
                "replacements": replacements,
                "values": data.tolist(),
            }
        )
    heldout = []
    worlds = []
    for target in pairs:
        law, replacements = ou_law(drift, np.ones(n), np.eye(n) * 0.1, target)
        heldout.append((rng.multivariate_normal(*law, count), law))
        worlds.append(replacements)
    levels = (0, 1, 2, 4, 6, 8, 10) if args.full else (0, n)
    summaries = []
    for k in levels:
        case = output / f"single-contexts-{k}"
        case.mkdir(parents=True, exist_ok=True)
        regimes = observed[: k + 1]
        model = OuMechanisms(
            tuple(f"equation:{i}" for i in range(n)),
            tuple((-np.eye(n)).ravel()),
            tuple(np.full(n, 0.8)),
            tuple((np.eye(n) * 0.15).ravel()),
        )
        parameters = (
            [
                {"kind": "drift", "equation": i, "parent": j}
                for i in range(n)
                for j in range(n)
                if i != j
            ]
            + [{"kind": "forcing", "equation": i} for i in range(n)]
            + [{"kind": "diffusion", "equation": i, "source": i} for i in range(n)]
        )
        prepared = model.prepare(
            worlds,
            regimes=regimes,
            parameters=parameters,
            learning_rate=0.1,
            score_tolerance=0.001 * sum(len(r["values"]) for r in regimes),
            max_iterations=args.max_iterations,
        )
        result = require_success(prepared.execute())
        answer = result.to_dict()["answer"]["result"]
        fit = answer["fit"]
        estimated = np.array(fit["model"]["drift"]).reshape(n, n)
        nll, oracle, mae, failed = 0.0, 0.0, [], []
        reference_error = 0.0
        for i, (prediction, (values, truth)) in enumerate(
            zip(answer["laws"], heldout, strict=True)
        ):
            if "failure" in prediction:
                failed.append({"world": i, **prediction})
                continue
            mean, covariance = (
                np.array(prediction["mean"]),
                np.array(prediction["covariance"]).reshape(n, n),
            )
            independent, _ = ou_law(
                estimated,
                np.array(fit["model"]["forcing"]),
                np.array(fit["model"]["diffusion_factor"]).reshape(n, n),
                pairs[i],
            )
            if independent is None:
                raise AssertionError(
                    "native stationary law contradicts independent drift eigenvalues"
                )
            reference_error = max(
                reference_error,
                float(np.max(np.abs(covariance - independent[1]))),
                float(np.max(np.abs(mean - independent[0]))),
            )
            nll += gaussian_nll(values, mean, covariance)
            oracle += gaussian_nll(values, *truth)
            mae.extend(np.abs(mean - truth[0]).tolist())
        off = ~np.eye(n, dtype=bool)
        denominator = (len(pairs) - len(failed)) * count * n
        row = {
            "single_contexts": k,
            "training_rows": sum(len(r["values"]) for r in regimes),
            "converged": fit["converged"],
            "score_norm": fit["score_norm"],
            "iterations": fit["iterations"],
            "line_search_exhausted": fit["line_search_exhausted"],
            "auprc": average_precision(mask[off], np.abs(estimated[off])),
            "heldout_mean_mae": float(np.mean(mae)) if mae else None,
            "heldout_nll_per_coordinate_on_evaluated_worlds": nll / denominator
            if denominator
            else None,
            "oracle_nll_on_same_worlds": oracle / denominator if denominator else None,
            "failed_worlds": failed,
            "all_requested_worlds_evaluated": not failed,
            "independent_law_max_absolute_error": reference_error,
            "artifact": replay(result, case),
        }
        if reference_error > 1e-7:
            raise AssertionError("native OU law differs from independent SciPy reference")
        write(case / "request.json", json.loads(prepared.request_json))
        write(case / "native_report.json", result.to_dict())
        write(case / "metrics.json", row)
        summaries.append(row)
        print(json.dumps({"seed": seed, "lane": "bicycle", "case": row}), flush=True)
    return {
        "lane": "bicycle",
        "seed": seed,
        "mode": "paper_size" if args.full else "smoke",
        "paper": PAPERS["bicycle"],
        "paper_sections": ["3.1-3.3", "4.1", "A.1"],
        "nodes": n,
        "samples_per_single_context": count,
        "observational_samples": controls,
        "test_pairs": pairs,
        "rejected_nonstationary_graph_draws": rejected,
        "deviations": [
            "Applicable Gaussian SEM1 synthetic experiment; no technical count-noise or latent-count ELBO claim.",
            "Exact Lyapunov/mean likelihood gradients and bounded local L-BFGS/Armijo optimization, not a reference-code port.",
            "Paper coefficient intervals and equation-level forcing/diffusion perturbations; reject graphs lacking any requested stationary law.",
            "No hyperparameter or graph selection uses held-out pair outcomes; low-intervention fits are not claimed structurally identified.",
        ],
        "cases": summaries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane", choices=("nodags-linear", "nodags-relu", "bicycle"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.trials < 1 or args.trials > 20:
        parser.error("trials must be in 1..20")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "lane": args.lane,
        "full": args.full,
        "trials": args.trials,
        "max_iterations": args.max_iterations,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol": "independent_paper_derived_data_and_native_public_workflows",
        "calibration": "separate_not_assessed",
        "attestation": "separate_not_performed",
    }
    write(args.output_dir / "manifest.json", manifest)
    reports = []
    for seed in range(args.trials):
        output = args.output_dir / f"seed-{seed}"
        output.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        report = (bicycle if args.lane == "bicycle" else nodags)(args, seed, output)
        report["wall_seconds"] = time.monotonic() - start
        write(output / "metrics.json", report)
        reports.append(report)
        write(args.output_dir / "results.json", {"manifest": manifest, "trials": reports})
        print(
            json.dumps({"completed_seed": seed, "wall_seconds": report["wall_seconds"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
