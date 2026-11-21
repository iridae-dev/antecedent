"""Independent development study of the 400-row wide two-decision design.

Compare the same outcome panels, five cross-fitting folds, [0.05, 0.95] floor,
and interval constructions. Only the source of assignment probabilities varies.
The extra treatment histories never contribute outcomes or observations to the
reported contrast. Every refusal counts in the all-attempt denominator.
"""

import numpy as np


def expit(x):
    return 1 / (1 + np.exp(-x))


def panel(rng, n, slope):
    baseline = rng.uniform(-np.sqrt(3), np.sqrt(3), n)
    first_probability = expit(slope * baseline)
    first = rng.random(n) < first_probability
    intermediate = (
        0.5 * baseline + 0.8 * first + rng.uniform(-np.sqrt(3), np.sqrt(3), n) * 0.5
    )
    second_probability = expit(1.2 * (intermediate - 0.4))
    second = rng.random(n) < second_probability
    outcome = (
        1
        + 0.5 * baseline
        + first
        + 1.5 * second
        + 0.4 * intermediate
        + rng.normal(size=n)
    )
    history = np.column_stack([np.ones(n), baseline, first, intermediate])
    return history, first, second, outcome, first_probability, second_probability


def logistic_fit(features, actions):
    beta = np.zeros(features.shape[1])
    for _ in range(20):
        probability = expit(features @ beta)
        information = features.T @ (
            (probability * (1 - probability))[:, None] * features
        )
        step = np.linalg.solve(information, features.T @ (actions - probability))
        beta += step
        if max(abs(step)) < 1e-10:
            return beta
    raise ArithmeticError("logistic fit did not converge")


def outcome_terms(history, first, second, outcome, folds):
    terms = np.zeros((2, 3, len(first)))
    for fold in range(5):
        train, test = folds != fold, folds == fold
        for action in (0, 1):
            rows = train & (second == action)
            terminal = (
                history @ np.linalg.lstsq(history[rows], outcome[rows], rcond=None)[0]
            )
            rows = train & (first == action)
            nested = (
                history[:, :2]
                @ np.linalg.lstsq(history[rows, :2], terminal[rows], rcond=None)[0]
            )
            terms[action, 0, test] = nested[test]
            terms[action, 1, test] = ((first == action) * (terminal - nested))[test]
            terms[action, 2, test] = (
                (first == action) * (second == action) * (outcome - terminal)
            )[test]
    return terms


def run(rng, slope, attempts):
    counts = np.zeros((4, 3), dtype=int)
    for _ in range(attempts):
        history, first, second, outcome, true_first, true_second = panel(
            rng, 400, slope
        )
        external, ext_first, ext_second, *_ = panel(rng, 8000, slope)
        folds = np.arange(400) % 5
        terms = outcome_terms(history, first, second, outcome, folds)
        p, q = np.zeros(400), np.zeros(400)
        for fold in range(5):
            train, test = folds != fold, folds == fold
            p[test] = expit(
                history[test, :2] @ logistic_fit(history[train, :2], first[train])
            )
            q[test] = expit(history[test] @ logistic_fit(history[train], second[train]))
        probabilities = [(p, q)]
        for m in (2000, 8000):
            p = expit(history[:, :2] @ logistic_fit(external[:m, :2], ext_first[:m]))
            q = expit(history @ logistic_fit(external[:m], ext_second[:m]))
            probabilities.append((p, q))
        probabilities.append((true_first, true_second))
        bootstrap = None
        for i, (p, q) in enumerate(probabilities):
            if min(p.min(), q.min(), 1 - p.max(), 1 - q.max()) < 0.05:
                counts[i, 0] += 1
                continue
            active = terms[1, 0] + terms[1, 1] / p + terms[1, 2] / (p * q)
            control = (
                terms[0, 0] + terms[0, 1] / (1 - p) + terms[0, 2] / ((1 - p) * (1 - q))
            )
            scores = active - control
            counts[i, 1] += (
                abs(scores.mean() - 2.82) <= 1.95996398454 * scores.std(ddof=1) / 20
            )
            if bootstrap is None:
                bootstrap = rng.exponential(size=(2000, 400))
                bootstrap /= bootstrap.sum(axis=1, keepdims=True)
            lo, hi = np.quantile(bootstrap @ scores, [0.025, 0.975])
            counts[i, 2] += lo <= 2.82 <= hi
    return counts


if __name__ == "__main__":
    rng = np.random.default_rng(0x375ECE)
    for slope, attempts in [(1.5, 400), (2.8, 100)]:
        print("baseline slope", slope, "panels", attempts)
        for name, row in zip(
            ("cross-fit", "external 2000", "external 8000", "known mechanism"),
            run(rng, slope, attempts),
        ):
            print(name, "refused, Wald covered, Bayesian bootstrap covered", *row)
