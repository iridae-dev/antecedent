# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""40-digit reference for the Cox/Breslow fit and the Cox inverse-censoring weights.

Regenerate with:  uv run --script cox_ipcw_reference.py > cox_ipcw_reference.json

For rows i with time t_i, covariates z_i and failure indicator f_i, the Breslow partial
log-likelihood is  sum_{i: f_i} [ z_i'b - log sum_{j: t_j >= t_i} exp(z_j'b) ]  (tied failures
share one risk set), maximised by Newton's method with the exact score and information.  The
Breslow baseline jumps at each distinct failure time t_k by d_k / sum_{j: t_j >= t_k} exp(z_j'b),
the cumulative baseline Lambda0 is their running sum, the survival of a row just before its time is
exp(-Lambda0(t_i-) exp(z_i'b)) with Lambda0(t-) the jumps strictly before t, and the baseline is
reported at the covariate means (centre = mean(z)'b), i.e. Lambda0 for the centred covariates.
The IPCW fit uses the censoring indicator 1 - event as the failure, and weighs an observed row
1 / survival_before and a censored row 0.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random

import mpmath as mp

mp.mp.dps = 40


def s(x):
    return mp.nstr(x, 30)


def fit(time, failure, Z):
    n, p = len(time), len(Z[0])
    means = [sum(mp.mpf(Z[i][j]) for i in range(n)) / n for j in range(p)]
    X = [[mp.mpf(Z[i][j]) - means[j] for j in range(p)] for i in range(n)]
    T = [mp.mpf(t) for t in time]
    b = mp.zeros(p, 1)

    def ingredients(b):
        eta = [sum(X[i][j] * b[j] for j in range(p)) for i in range(n)]
        ll = mp.mpf(0)
        score = mp.zeros(p, 1)
        info = mp.zeros(p, p)
        for i in range(n):
            if not failure[i]:
                continue
            risk = [j for j in range(n) if T[j] >= T[i]]
            w = [mp.e ** eta[j] for j in risk]
            S0 = sum(w)
            S1 = [sum(wj * X[j][k] for wj, j in zip(w, risk)) for k in range(p)]
            ll += eta[i] - mp.log(S0)
            for k in range(p):
                score[k] += X[i][k] - S1[k] / S0
                for m in range(p):
                    S2 = sum(wj * X[j][k] * X[j][m] for wj, j in zip(w, risk))
                    info[k, m] += S2 / S0 - S1[k] * S1[m] / S0**2
        return ll, score, info

    for _ in range(100):
        ll, score, info = ingredients(b)
        step = info ** -1 * score
        b += step
        if mp.norm(step) < mp.mpf("1e-35"):
            break
    eta = [sum(X[i][j] * b[j] for j in range(p)) for i in range(n)]
    jumps = []
    cum = mp.mpf(0)
    for tk in sorted({T[i] for i in range(n) if failure[i]}):
        d = sum(1 for i in range(n) if failure[i] and T[i] == tk)
        S0 = sum(mp.e ** eta[j] for j in range(n) if T[j] >= tk)
        cum += d / S0
        jumps.append((tk, cum))
    return b, means, eta, jumps, T


def before(jumps, t):
    v = mp.mpf(0)
    for tk, c in jumps:
        if tk < t:
            v = c
    return v


def at(jumps, t):
    v = mp.mpf(0)
    for tk, c in jumps:
        if tk <= t:
            v = c
    return v


def main():
    rng = random.Random(77)
    n = 64
    Z = [[rng.gauss(0, 1), float(rng.random() < 0.45)] for _ in range(n)]
    time, fail, event = [], [], []
    for i in range(n):
        h = 0.5 * (2.718281828 ** (0.6 * Z[i][0] - 0.4 * Z[i][1]))
        t_fail = rng.expovariate(h)
        t_cens = rng.expovariate(0.35 * (2.718281828 ** (0.5 * Z[i][1] - 0.3 * Z[i][0])))
        t = round(min(t_fail, t_cens, 5.0) * 4) / 4 + 0.25  # quarter-unit grid creates ties
        time.append(t)
        event.append(1.0 if t_fail <= t_cens and t_fail <= 5.0 else 0.0)
        fail.append(rng.random() < 0.7)
    assert any(fail) and 0 < sum(event) < n

    # Breslow fit with the failure indicator `fail`.
    b, means, eta, jumps, T = fit(time, fail, Z)
    center = sum(b[j] * means[j] for j in range(2))
    # The reported coefficient is on the raw covariate scale; the centre is mean(z)'b.
    breslow = {
        "coefficients": [s(b[0]), s(b[1])], "center": s(center),
        "jump_times": [float(t) for t, _ in jumps],
        "cumulative_baseline": [s(c) for _, c in jumps],
        "baseline_before": [{"t": float(t), "value": s(before(jumps, mp.mpf(t)))} for t in (0.25, 0.75, 1.25, 2.0, 6.0)],
        "baseline_at": [{"t": float(t), "value": s(at(jumps, mp.mpf(t)))} for t in (0.25, 0.75, 1.25, 2.0, 6.0)],
    }

    # IPCW: failures are the censorings.
    cens = [e == 0.0 for e in event]
    bc, _, etac, jc, T = fit(time, cens, Z)
    surv, weights = [], []
    for i in range(n):
        sv = mp.e ** (-before(jc, T[i]) * mp.e ** etac[i])
        surv.append(sv)
        weights.append(1 / sv if event[i] == 1.0 else mp.mpf(0))
    ipcw = {
        "coefficients": [s(bc[0]), s(bc[1])],
        "survival_before": [s(v) for v in surv], "weights": [s(w) for w in weights],
    }
    print(json.dumps({"time": time, "event": event, "failure": fail,
                      "covariates": Z, "breslow": breslow, "ipcw": ipcw,
                      "ties": len(time) - len(set(time))}))


main()
