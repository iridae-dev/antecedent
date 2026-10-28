# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Observation-adjusted temporal response: lag-aligned pseudo-outcome, per-horizon g-computation
and the outer observation bootstrap, solved exactly.

Regenerate with:
    uv run --script temporal_response_observation_adjusted_reference.py > temporal_response_observation_adjusted_reference.json
(reads temporal_response_observation_adjusted_streams.json, the circular-block anchors the library
drew, which the Rust test regenerates and compares before using them).

Series: z_t = 0.5 z_{t-1} + noise, x_t = 0.6 z_t + 0.4 z_{t-1} + 0.8 noise,
y_t = 1.5 x_{t-1} + 0.7 x_{t-2} + 0.5 z_{t-1} + 0.8 z_{t-2} + noise (rounded to 9 digits).
The policy is a pulse at offset -1, so the observation conditioning set (x, z) is read at lag 1:
table row i pairs the outcome, indicator and recorded time of series row i + 1 with x_i, z_i.

Selected outcome (R = 1 observed).  The selection probability is the logistic maximum-likelihood fit
on (1, x_{t-1}, z_{t-1}) (Newton at 40 digits) clamped into [floor, 1 - floor].  Plain IPW fits every
row and uses R y / p.  Augmented: the 47 table rows are cut into `folds` contiguous time blocks
(row i goes to block min(i * folds // 47, folds - 1)); row i's p comes from the logistic fit on the
other blocks, and its outcome prediction m from the least-squares fit of y on the conditioning
columns and the downstream design columns x_{t-2}, z_{t-2} over the other blocks' selected rows
(row 0, whose t-2 columns fall before the series, uses the conditioning-only fit on the same
training blocks); the pseudo-value is m + R (y - m) / p.  Right-censored: Breslow Cox fit of the
censoring indicator on the lag-1 (x, z) and weight 1 / S(t-) for an uncensored row, 0 otherwise;
the pseudo-value is y times the weight.  The series row 0 keeps its recorded outcome.
The pseudo-outcome series, taken to the nearest double as the estimator stores it, is regressed
per horizon exactly as in temporal_response_gcomp_reference.py (h = 1: intercept, x_{t-1}, z_{t-1},
z_{t-2}; h = 2: intercept, x_{t-2}, z_{t-2}; windows t >= 2) and the level at dose a is
beta' cbar(a) (surface layout dose-major, horizon-minor).

Bootstrap replicate: the anchors a (series row indices >= 2) of one replicate each contribute the
lag-aligned observation row (x_{a-1}, z_{a-1}; extras x_{a-2}, z_{a-2}; y_a, R_a or the censored
triple) and the design rows of both horizons.  The observation nuisance is refit on exactly those
rows with folds keyed by the source position a - 1 (span 47) so repeated anchors share a fold, the
pseudo-outcomes align with the anchors, each horizon is refit on the anchors' rows and the
replicate surface is beta*' cbar*(a) with column means over the replicate.  The product scales each
replicate's deviation from the full-sample surface by a dispersion factor and a per-cell kernel
factor it discloses (relied on, not recomputed here), so the reference reports the raw replicate
surfaces and their sample standard deviations; the test forms z * factor * kernel * sd.
Everything except the logistic and Cox fits (40 digits) is Fraction arithmetic.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import math
import os
import sys
from fractions import Fraction as F

from mpmath import lu_solve, matrix, mp, mpf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "antecedent-stats", "tests", "oracle_data"))

mp.dps = 40
N = 100
LAG = 1
DOSES = [-1.0, 0.0, 1.0]
HORIZONS = [1, 2]
SURVIVAL_FLOOR = mpf("0.01")

_state = 0x2545F4914F6CDD1D


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def normal():
    return sum(u01() for _ in range(12)) - 6.0


def generate():
    z, x, y = [0.0] * N, [0.0] * N, [0.0] * N
    for t in range(N):
        zp = z[t - 1] if t >= 1 else 0.0
        z[t] = round(0.5 * zp + normal(), 9)
        x[t] = round(0.6 * z[t] + 0.4 * zp + 0.8 * normal(), 9)
        x1 = x[t - 1] if t >= 1 else 0.0
        x2 = x[t - 2] if t >= 2 else 0.0
        z2 = z[t - 2] if t >= 2 else 0.0
        y[t] = round(1.5 * x1 + 0.7 * x2 + 0.5 * zp + 0.8 * z2 + 0.6 * normal(), 9)
    return x, y, z


def sigmoid(v):
    return 1.0 / (1.0 + math.exp(-v))


# ---------------------------------------------------------------- exact fits


def logistic_fit(features, labels):
    """Logistic maximum likelihood on (1, features) by Newton's method at 40 digits."""
    p = len(features[0]) + 1
    rows = [[mpf(1)] + [mpf(v) for v in f] for f in features]
    y = [mpf(int(v)) for v in labels]
    beta = [mpf(0)] * p
    for _ in range(100):
        grad = [mpf(0)] * p
        hess = matrix(p, p)
        for row, yi in zip(rows, y):
            eta = sum(b * v for b, v in zip(beta, row))
            mu = 1 / (1 + mp.exp(-eta))
            for j in range(p):
                grad[j] += (yi - mu) * row[j]
                for k in range(p):
                    hess[j, k] += mu * (1 - mu) * row[j] * row[k]
        step = lu_solve(hess, matrix(grad))
        beta = [beta[j] + step[j] for j in range(p)]
        if max(abs(step[j]) for j in range(p)) < mpf(10) ** -32:
            break
    else:
        raise AssertionError("logistic fit did not converge (separation)")
    return beta


def clamp_probability(beta, feature, floor):
    eta = beta[0] + sum(b * mpf(v) for b, v in zip(beta[1:], feature))
    return min(max(1 / (1 + mp.exp(-eta)), floor), 1 - floor)


def ols_fit(features, outcomes):
    p = len(features[0]) + 1
    rows = [[mpf(1)] + [mpf(v) for v in f] for f in features]
    gram, rhs = matrix(p, p), matrix(p, 1)
    for row, yi in zip(rows, outcomes):
        for j in range(p):
            rhs[j] += row[j] * mpf(yi)
            for k in range(p):
                gram[j, k] += row[j] * row[k]
    return lu_solve(gram, rhs)


def predict(coef, feature):
    return coef[0] + sum(c * mpf(v) for c, v in zip(coef[1:], feature))


def selected_pseudo(rows, mode, folds, floor, span, lead):
    """rows: dicts with cov, ext (or None), y, r, pos.  Returns (pseudo, weights, probabilities)."""
    n = len(rows)
    prob = [None] * n
    pseudo = [None] * n
    if mode == "ipw":
        beta = logistic_fit([r["cov"] for r in rows], [r["r"] for r in rows])
        for i, r in enumerate(rows):
            prob[i] = clamp_probability(beta, r["cov"], floor)
            pseudo[i] = mpf(r["y"]) / prob[i] if r["r"] == 1 else mpf(0)
    else:
        fold_of = [min(r["pos"] * folds // span, folds - 1) for r in rows]
        for f in range(folds):
            valid = [i for i in range(n) if fold_of[i] == f]
            train = [i for i in range(n) if fold_of[i] != f]
            if not valid:
                continue
            beta = logistic_fit([rows[i]["cov"] for i in train], [rows[i]["r"] for i in train])
            sel = [i for i in train if rows[i]["r"] == 1]
            need_base = rows[0]["ext"] is None or any(i < lead for i in valid)
            base = ols_fit([rows[i]["cov"] for i in sel], [rows[i]["y"] for i in sel]) if need_base else None
            aug = None
            if rows[0]["ext"] is not None:
                sel_a = [i for i in sel if i >= lead]
                aug = ols_fit([rows[i]["cov"] + rows[i]["ext"] for i in sel_a], [rows[i]["y"] for i in sel_a])
            for i in valid:
                r = rows[i]
                prob[i] = clamp_probability(beta, r["cov"], floor)
                if aug is not None and i >= lead:
                    m = predict(aug, r["cov"] + r["ext"])
                else:
                    m = predict(base, r["cov"])
                pseudo[i] = m + (mpf(r["y"]) - m) / prob[i] if r["r"] == 1 else m
    weights = [1 / prob[i] if rows[i]["r"] == 1 else mpf(0) for i in range(n)]
    return pseudo, weights, prob


def cox_fit(time, failure, Z):
    """Breslow partial likelihood fit; returns coefficients, centred linear predictors and jumps."""
    n, p = len(time), len(Z[0])
    means = [sum(mpf(Z[i][j]) for i in range(n)) / n for j in range(p)]
    X = [[mpf(Z[i][j]) - means[j] for j in range(p)] for i in range(n)]
    T = [mpf(t) for t in time]
    b = mp.zeros(p, 1)

    def ingredients(b):
        eta = [sum(X[i][j] * b[j] for j in range(p)) for i in range(n)]
        score = mp.zeros(p, 1)
        info = mp.zeros(p, p)
        for i in range(n):
            if not failure[i]:
                continue
            risk = [j for j in range(n) if T[j] >= T[i]]
            w = [mp.e ** eta[j] for j in risk]
            S0 = sum(w)
            S1 = [sum(wj * X[j][k] for wj, j in zip(w, risk)) for k in range(p)]
            for k in range(p):
                score[k] += X[i][k] - S1[k] / S0
                for m in range(p):
                    S2 = sum(wj * X[j][k] * X[j][m] for wj, j in zip(w, risk))
                    info[k, m] += S2 / S0 - S1[k] * S1[m] / S0**2
        return score, info

    for _ in range(100):
        score, info = ingredients(b)
        step = info ** -1 * score
        b += step
        if mp.norm(step) < mpf("1e-35"):
            break
    eta = [sum(X[i][j] * b[j] for j in range(p)) for i in range(n)]
    jumps, cum = [], mpf(0)
    for tk in sorted({T[i] for i in range(n) if failure[i]}):
        d = sum(1 for i in range(n) if failure[i] and T[i] == tk)
        S0 = sum(mp.e ** eta[j] for j in range(n) if T[j] >= tk)
        cum += d / S0
        jumps.append((tk, cum))
    return eta, jumps, T


def censored_pseudo(rows):
    """rows: dicts with cov, y (recorded time), e (event).  Cox IPCW pseudo-values."""
    time = [r["y"] for r in rows]
    cens = [r["e"] == 0.0 for r in rows]
    eta, jumps, T = cox_fit(time, cens, [r["cov"] for r in rows])
    pseudo, weights = [], []
    for i, r in enumerate(rows):
        cum = mpf(0)
        for tk, c in jumps:
            if tk < T[i]:
                cum = c
        surv = mp.e ** (-cum * mp.e ** eta[i])
        assert surv > SURVIVAL_FLOOR, "censoring survival under the floor"
        w = 1 / surv if r["e"] == 1.0 else mpf(0)
        weights.append(w)
        pseudo.append(mpf(r["y"]) * w)
    return pseudo, weights


# ---------------------------------------------------------------- g-computation (exact)


def solve(A, b):
    n = len(A)
    M = [row[:] + [bb] for row, bb in zip(A, b)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [v / piv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [row[n] for row in M]


def design_row(h, x, z, a):
    """Regressors of the outcome at row a for horizon h (treatment first)."""
    if h == 1:
        return [x[a - 1], z[a - 1], z[a - 2]]
    return [x[a - 2], z[a - 2]]


def surface(x, z, y, anchors):
    """Dose-major, horizon-minor levels of the per-horizon OLS over the given outcome rows."""
    fits = []
    for h in HORIZONS:
        rows = [[F(1)] + [F(v) for v in design_row(h, x, z, a)] for a in anchors]
        outcomes = [F(y[a]) for a in anchors]
        p = len(rows[0])
        gram = [[sum(r[j] * r[k] for r in rows) for k in range(p)] for j in range(p)]
        rhs = [sum(r[j] * o for r, o in zip(rows, outcomes)) for j in range(p)]
        beta = solve(gram, rhs)
        means = [sum(r[j] for r in rows) / len(rows) for j in range(p)]
        fits.append((beta, means))
    out = []
    for dose in DOSES:
        for beta, means in fits:
            level = beta[0] * means[0] + beta[1] * F(dose) + sum(b * m for b, m in zip(beta[2:], means[2:]))
            out.append(level)
    return out


def double(v):
    if isinstance(v, (float, F)):
        return float(v)
    return float(mp.nstr(v, 25))


# ---------------------------------------------------------------- cases


def selected_rows(x, y_obs, z, r, anchors):
    """Lag-aligned observation rows at the given outcome-time anchors."""
    return [
        {
            "cov": [x[a - LAG], z[a - LAG]],
            "ext": [x[a - 2], z[a - 2]] if a >= 2 else [float("nan")] * 2,
            "y": y_obs[a], "r": r[a], "pos": a - LAG,
        }
        for a in anchors
    ]


def case_selected(name, data, mode, folds, floor, streams):
    x, y, z, r = data["x"], data["y"], data["z"], data["r"]
    anchors = list(range(LAG, N))
    effective = 0.01 if floor is None else floor
    rows = selected_rows(x, y, z, r, anchors)
    lead = 1 if mode == "aipw" else 0
    if mode == "ipw":
        for row in rows:
            row["ext"] = None
    pseudo, weights, prob = selected_pseudo(rows, mode, folds, mpf(effective), N - LAG, lead)
    pseudo_d = [y[0]] + [double(v) for v in pseudo]
    weights_d = [0.0] + [double(w) for w in weights]
    clamped = sum(1 for p in prob if p in (mpf(effective), 1 - mpf(effective)))
    return {
        "name": name, "series": "selected", "mode": mode, "folds": folds, "floor": floor,
        "clamped_probabilities": clamped,
        "pseudo": pseudo_d, "weights": weights_d,
        "weight_diagnostic": weight_diag(weights_d),
        "surface": [double(v) for v in surface(x, z, pseudo_d, range(2, N))],
        "bootstrap": bootstrap(
            streams[name], x, z,
            lambda anch: replicate_pseudo_selected(data, mode, folds, effective, anch),
        ),
    }


def replicate_pseudo_selected(data, mode, folds, floor, anchors):
    x, y, z, r = data["x"], data["y"], data["z"], data["r"]
    rows = selected_rows(x, y, z, r, anchors)
    if mode == "ipw":
        for row in rows:
            row["ext"] = None
    pseudo, _, _ = selected_pseudo(rows, mode, folds, mpf(floor), N - LAG, 0)
    return [double(v) for v in pseudo]


def case_censored(name, data, streams):
    x, y, z, e = data["x"], data["y"], data["z"], data["e"]

    def pseudo_at(anchors):
        rows = [{"cov": [x[a - LAG], z[a - LAG]], "y": y[a], "e": e[a]} for a in anchors]
        return censored_pseudo(rows)

    pseudo, weights = pseudo_at(range(LAG, N))
    pseudo_d = [y[0]] + [double(v) for v in pseudo]
    weights_d = [0.0] + [double(w) for w in weights]
    return {
        "name": name, "series": "censored", "mode": "cox", "folds": 0, "floor": None,
        "clamped_probabilities": 0,
        "pseudo": pseudo_d, "weights": weights_d,
        "weight_diagnostic": weight_diag(weights_d),
        "surface": [double(v) for v in surface(x, z, pseudo_d, range(2, N))],
        "bootstrap": bootstrap(
            streams[name], x, z, lambda anch: [double(v) for v in pseudo_at(anch)[0]]
        ),
    }


def weight_diag(weights):
    positive = [w for w in weights if w > 0]
    total = sum(weights)
    squares = sum(w * w for w in weights)
    return [min(positive), max(weights), total * total / squares]


def bootstrap(entry, x, z, replicate_fn):
    draws, pseudos = [], []
    for anchors in entry["anchors"]:
        pseudo = replicate_fn(anchors)
        pseudos.append(pseudo)
        draws.append([double(v) for v in surface_levels(x, z, anchors, pseudo)])
    sds = []
    for c in range(len(draws[0])):
        vals = [F(d[c]) for d in draws]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        sds.append(float(mp.sqrt(mpf(var.numerator) / mpf(var.denominator))))
    return {
        "first_anchor": entry["first_anchor"], "block": entry["block"],
        "replicate_pseudo": pseudos, "raw_draws": draws, "raw_sd": sds,
    }


def surface_levels(x, z, anchors, outcomes):
    fits = []
    for h in HORIZONS:
        rows = [[F(1)] + [F(v) for v in design_row(h, x, z, a)] for a in anchors]
        p = len(rows[0])
        gram = [[sum(r[j] * r[k] for r in rows) for k in range(p)] for j in range(p)]
        rhs = [sum(r[j] * F(o) for r, o in zip(rows, outcomes)) for j in range(p)]
        beta = solve(gram, rhs)
        means = [sum(r[j] for r in rows) / len(rows) for j in range(p)]
        fits.append((beta, means))
    out = []
    for dose in DOSES:
        for beta, means in fits:
            out.append(beta[0] * means[0] + beta[1] * F(dose) + sum(b * m for b, m in zip(beta[2:], means[2:])))
    return out


def main():
    path = os.path.join(HERE, "temporal_response_observation_adjusted_streams.json")
    streams = json.load(open(path))["cases"] if os.path.exists(path) else None
    x, y_lat, z = generate()
    # selection depends on the lag-1 conditioning set (missing at random given it)
    r = [1.0] + [1.0 if u01() < sigmoid(0.2 + 0.6 * x[t - 1] + 0.4 * z[t - 1]) else 0.0 for t in range(1, N)]
    y_sel = [v if rr == 1.0 else 0.0 for v, rr in zip(y_lat, r)]
    selected = {"x": x, "z": z, "y": y_sel, "r": r}
    # right censoring of a positive recorded outcome (censoring time rises with x_{t-1})
    latent = [10.0 + v for v in y_lat]
    c = [round((9.0 + 7.0 * u01()) * math.exp(0.15 * (x[t - 1] if t else 0.0)), 9) for t in range(N)]
    y_cens = [min(a, b) for a, b in zip(latent, c)]
    e = [1.0 if a <= b else 0.0 for a, b in zip(latent, c)]
    censored = {"x": x, "z": z, "y": y_cens, "e": e}
    if streams is None:
        # first pass: series and case options only, enough for the Rust test to draw the streams
        cases = [
            {"name": n, "series": s_, "mode": m, "folds": f, "floor": fl}
            for n, s_, m, f, fl in [
                ("aipw_default_five_folds", "selected", "aipw", 5, None),
                ("aipw_three_folds", "selected", "aipw", 3, None),
                ("ipw_in_sample", "selected", "ipw", 5, None),
                ("aipw_declared_floor", "selected", "aipw", 5, 0.3),
                ("right_censored_cox", "censored", "cox", 0, None),
            ]
        ]
        streams = {}
        emit(x, y_sel, z, r, y_cens, c, e, cases)
        return
    cases = [
        case_selected("aipw_default_five_folds", selected, "aipw", 5, None, streams),
        case_selected("aipw_three_folds", selected, "aipw", 3, None, streams),
        case_selected("ipw_in_sample", selected, "ipw", 5, None, streams),
        case_selected("aipw_declared_floor", selected, "aipw", 5, 0.3, streams),
        case_censored("right_censored_cox", censored, streams),
    ]
    emit(x, y_sel, z, r, y_cens, c, e, cases)


def emit(x, y_sel, z, r, y_cens, c, e, cases):
    print(json.dumps({
        "n": N, "lag": LAG, "doses": DOSES, "horizons": HORIZONS,
        "series": {
            "selected": {"x": x, "y": y_sel, "z": z, "r": r},
            "censored": {"x": x, "y": y_cens, "z": z, "c": c, "e": e},
        },
        "cases": cases,
    }, indent=1))


if __name__ == "__main__":
    main()
