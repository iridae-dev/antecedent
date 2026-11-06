#!/usr/bin/env python3
"""Compare treatment regimes over three periods under time-varying confounding.

The question: what total reward would subjects collect if everyone followed "always treat",
"never treat" or the rule "treat in a period when the current health score is below zero"?
The data is simulated from a declared process. A health score `L` drives both the chance of
treatment and the reward, and treatment lifts the next score, so a past treatment changes the
very covariate that decides the next one. That is time-varying confounding affected by past
treatment, which a single adjusted regression cannot handle.

The truth for each regime is computed by Monte Carlo: the same process is simulated 400,000
times with the regime imposed. The script then fits its own nuisance models from one observed
sample of 4,000 subjects, hands the predictions and treatment probabilities to
`regimes.evaluate_sequential_gformula` and `regimes.evaluate_sequential_doubly_robust`, and
asserts each value against that truth within a stated tolerance, plus the ordering of the
regimes. Each value is also checked against a short numpy recomputation of what the function
documents. Two inputs are refused with their reasons.

Both values are point-only: no standard error or interval, and no check that models were
trained without the subjects they predict. Their validity rests on sequential exchangeability
(the score captures all confounding), positivity and the models you supply."""

from __future__ import annotations

# %%
import numpy as np
from antecedent.errors import CausalValueError
from antecedent.regimes import evaluate_sequential_doubly_robust, evaluate_sequential_gformula

SEED = 20260613
N = 4_000
N_TRUTH = 400_000
T = 3
CONFOUNDING = 0.6  # log-odds of treatment per unit of health score
CARRY, LIFT, SD_L = 0.6, 0.8, 0.6  # next score = CARRY * score + LIFT * treated + noise
R_SCORE, R_TREAT, SD_R = 0.8, 0.3, 0.5  # reward = R_SCORE * score + R_TREAT * treated + noise
TOL_GFORMULA, TOL_DR = 0.3, 0.4


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(n, rng, policy=None):
    """Scores, treatments and rewards, shape (n, T); `policy(t, score)` imposes a regime."""
    scores, treated, rewards = (np.empty((n, T)) for _ in range(3))
    score = rng.normal(size=n)
    for t in range(T):
        scores[:, t] = score
        if policy is None:
            treated[:, t] = rng.random(n) < expit(CONFOUNDING * score)
        else:
            treated[:, t] = policy(t, score)
        rewards[:, t] = R_SCORE * score + R_TREAT * treated[:, t] + SD_R * rng.normal(size=n)
        score = CARRY * score + LIFT * treated[:, t] + SD_L * rng.normal(size=n)
    return scores, treated.astype(bool), rewards


def treat_when_low(t, past_treatments, covariates):
    """The API callback form: the rule reads the current score (last covariate row)."""
    return bool(covariates[-1][0] < 0.0)


# Each regime: the API spelling, and the same rule vectorised for the simulation.
REGIMES = {
    "always treat": ([True] * T, lambda t, score: np.ones(score.shape, bool)),
    "never treat": ([False] * T, lambda t, score: np.zeros(score.shape, bool)),
    "treat if score < 0": (treat_when_low, lambda t, score: score < 0.0),
}

# %% [markdown]
# ## Truth by Monte Carlo
#
# Simulate the declared process with each regime imposed and average the total reward. This is
# the value each estimator below is trying to recover from confounded data alone.

# %%
truth = {}
for name, (_, policy) in REGIMES.items():
    _, _, rewards = simulate(N_TRUTH, np.random.default_rng(SEED + 1), policy)
    truth[name] = float(rewards.sum(axis=1).mean())
    print(f"{name:20s} true value {truth[name]:.3f}")
assert truth["never treat"] < truth["treat if score < 0"] < truth["always treat"]

# %% [markdown]
# ## One observed sample and its nuisance models
#
# Subjects were treated by the confounded process, not by any regime. The script fits the
# treatment model (logistic in the current score), then for each regime a backward chain of
# outcome regressions (the Q predictions), cross-fitted over two folds. The treatment-model
# fold structure is recorded for the API, which does not check it.

# %%
rng = np.random.default_rng(SEED)
scores, treated, rewards = simulate(N, rng)
total = rewards.sum(axis=1)
earlier = np.cumsum(rewards, axis=1) - rewards  # reward collected before each period
folds = rng.permutation(N) % 2
subject_ids = [f"s{i}" for i in range(N)]


def fit_logistic(x, y):
    beta = np.zeros(x.shape[1])
    for _ in range(25):
        p = expit(x @ beta)
        gram = x.T @ (x * (p * (1 - p))[:, None]) + 1e-8 * np.eye(x.shape[1])
        beta += np.linalg.solve(gram, x.T @ (y - p))
    return beta


propensity = np.empty((N, T))  # P(treated | current score), cross-fitted
for k in (0, 1):
    train = folds != k
    design = np.column_stack([np.ones(train.sum() * T), scores[train].ravel()])
    beta = fit_logistic(design, treated[train].ravel().astype(float))
    propensity[~train] = expit(beta[0] + beta[1] * scores[~train])


def q_predictions(policy):
    """Backward chain: Q[:, t] predicts the next pseudo-outcome when period t follows the regime."""
    q = np.zeros((N, T))
    target = total
    for t in reversed(range(T)):

        def features(action):
            return np.column_stack([np.ones(N), earlier[:, t], scores[:, t], action])

        regime_action = policy(t, scores[:, t]).astype(float)
        for k in (0, 1):
            train = folds != k
            coef = np.linalg.lstsq(
                features(treated[:, t].astype(float))[train], target[train], rcond=None
            )[0]
            q[~train, t] = (features(regime_action)[~train]) @ coef
        target = q[:, t]
    return q


print("treatment probability range:", propensity.min().round(3), propensity.max().round(3))

# %% [markdown]
# ## Sequential g-formula
#
# The g-formula sums period-by-period reward predictions under the regime. Here each prediction
# is the expected reward in that period given the subject's starting score, from a linear model
# of the score and reward fitted to the sample and simulated forward under the regime. The
# probabilities are only checked for positivity.

# %%
step = np.column_stack(
    [
        np.ones((T - 1) * N),
        scores[:, :-1].T.reshape(-1),
        treated[:, :-1].T.reshape(-1).astype(float),
    ]
)
score_coef = np.linalg.lstsq(step, scores[:, 1:].T.reshape(-1), rcond=None)[0]
score_sd = float(np.std(scores[:, 1:].T.reshape(-1) - step @ score_coef))
reward_design = np.column_stack(
    [np.ones(N * T), scores.T.reshape(-1), treated.T.reshape(-1).astype(float)]
)
reward_coef = np.linalg.lstsq(reward_design, rewards.T.reshape(-1), rcond=None)[0]


def gformula_predictions(policy, draws=40):
    sim = np.random.default_rng(SEED + 2)
    score = np.repeat(scores[:, 0], draws)
    out = np.empty((N, T))
    for t in range(T):
        action = policy(t, score).astype(float)
        mean_reward = reward_coef[0] + reward_coef[1] * score + reward_coef[2] * action
        out[:, t] = mean_reward.reshape(N, draws).mean(axis=1)
        score = (
            score_coef[0]
            + score_coef[1] * score
            + score_coef[2] * action
            + score_sd * sim.normal(size=score.shape)
        )
    return out


gformula = {}
for name, (regime, policy) in REGIMES.items():
    predictions = gformula_predictions(policy)
    result = evaluate_sequential_gformula(
        period_outcome_predictions=predictions,
        treatment_history=treated,
        regime=regime,
        treatment_probabilities=propensity,
        subject_ids=subject_ids,
        fold_ids=folds,
        covariate_history=scores[:, :, None],
    )
    gformula[name] = result.value
    print(f"{name:20s} g-formula {result.value:.3f}  (true {truth[name]:.3f})")
    assert abs(result.value - truth[name]) < TOL_GFORMULA, (name, result.value)
    # The documented estimate is the mean over subjects of the summed period predictions.
    assert abs(result.value - predictions.sum(axis=1).mean()) < 1e-9
    assert result.subject_count == N
print("uncertainty:", result.uncertainty, "| support:", result.support_status)

# %% [markdown]
# ## Sequential doubly robust value
#
# This estimator starts from the Q predictions and adds a backward correction that reweights
# subjects whose observed treatments followed the regime by the inverse of their treatment
# probabilities. Always-treat subjects are rare after three periods, so look at the smallest
# regime-action probability the call reports.

# %%
doubly_robust = {}
for name, (regime, policy) in REGIMES.items():
    q = q_predictions(policy)
    result = evaluate_sequential_doubly_robust(
        outcomes=total,
        outcome_observed=np.ones(N, bool),
        observation_history=np.ones((N, T), bool),
        treatment_history=treated,
        regime=regime,
        q_predictions=q,
        treatment_probabilities=propensity,
        subject_ids=subject_ids,
        fold_ids=folds,
        prediction_fold_ids=folds,
        covariate_history=scores[:, :, None],
    )
    doubly_robust[name] = result.value
    print(
        f"{name:20s} doubly robust {result.value:.3f}  (true {truth[name]:.3f}), "
        f"min regime-action probability {result.minimum_regime_action_probability:.3f}"
    )
    assert abs(result.value - truth[name]) < TOL_DR, (name, result.value)

    # Independent recomputation of the documented backward recursion.
    actions = np.column_stack([policy(t, scores[:, t]) for t in range(T)])
    value = total
    for t in reversed(range(T)):
        p_action = np.where(actions[:, t], propensity[:, t], 1.0 - propensity[:, t])
        value = q[:, t] + (treated[:, t] == actions[:, t]) * (value - q[:, t]) / p_action
    assert abs(result.value - value.mean()) < 1e-8, (name, result.value, value.mean())
print("uncertainty:", result.uncertainty, "| crossfit:", result.crossfit_status)

# %% [markdown]
# ## Rank the regimes
#
# Both estimators should order the regimes as the Monte Carlo truth does. The naive comparison
# below, the mean total reward among subjects who happened to follow each static regime, ignores
# the confounding by score and the effect of treatment on later scores.

# %%
order = sorted(truth, key=truth.get)
assert sorted(gformula, key=gformula.get) == order
assert sorted(doubly_robust, key=doubly_robust.get) == order
static_followers = {"always treat": treated.all(axis=1), "never treat": ~treated.any(axis=1)}
for name, followed in static_followers.items():
    naive = float(total[followed].mean())
    print(
        f"{name:20s} naive {naive:.3f} from {int(followed.sum())} subjects (true {truth[name]:.3f})"
    )
print("regimes from worst to best:", " < ".join(order))

# %% [markdown]
# ## Where the estimators refuse
#
# A treatment probability under the positivity floor makes the corrections meaningless, and
# Q predictions must carry the same fold ownership as the subjects. Both are refused.


# %%
def refusal_of(call):
    try:
        call()
    except CausalValueError as error:
        return error
    raise AssertionError("the call should have been refused")


regime, policy = REGIMES["always treat"]
q = q_predictions(policy)
shared = dict(
    outcomes=total,
    outcome_observed=np.ones(N, bool),
    observation_history=np.ones((N, T), bool),
    treatment_history=treated,
    regime=regime,
    q_predictions=q,
    subject_ids=subject_ids,
    fold_ids=folds,
)
small = propensity.copy()
small[0, 0] = 0.001
positivity = refusal_of(
    lambda: evaluate_sequential_doubly_robust(
        **shared, treatment_probabilities=small, prediction_fold_ids=folds
    )
)
misaligned = refusal_of(
    lambda: evaluate_sequential_doubly_robust(
        **shared, treatment_probabilities=propensity, prediction_fold_ids=1 - folds
    )
)
for label, error in (("probability below the floor", positivity), ("misaligned folds", misaligned)):
    code = getattr(error, "reason_code", None)
    first = (str(error).strip().splitlines() or [""])[0][:160]
    print(f"{label}: {type(error).__name__} " + (f"reason_code={code}" if code else f"- {first}"))
assert "positivity" in str(positivity) and "fold ownership" in str(misaligned)
