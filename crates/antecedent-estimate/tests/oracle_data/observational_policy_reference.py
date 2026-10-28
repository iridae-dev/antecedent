# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Doubly robust policy value with recorded splits, evaluated at 60 digits.

Regenerate with:  uv run --script observational_policy_reference.py > observational_policy_reference.json

The data, the declared policy and the split of every row are part of the case (the seeded fold
plan is a separate kernel and is not under review; the test asserts that the product's scored
order reproduces the recorded splits).  For each split the propensity is the unpenalised
baseline-category multinomial logit maximum likelihood on the training rows (Newton's method),
unless a known propensity is declared; each action's outcome regression is ordinary least squares
of the outcome on [1, Z] over the training rows at that action, predicted on the scored rows.
Score of action c at row i:  s_c = mu_c + 1{T_i = c} (y_i - mu_c) / e_c.
Policy: the declared action (prespecified) or the action with the best predicted outcome (first
of equals).  Value = mean of the policy scores over all scored rows, SE = sd / sqrt(m) with the
n-1 divisor; each declared action's value and the paired difference with the reference action
use the same scored rows.  Values are written as nearest binary64.
Repository-authored (MIT OR Apache-2.0).
"""
import json
from mpmath import mp, mpf, matrix, lu_solve, exp, sqrt

mp.dps = 60
_state = 0x3C6EF372FE94F82B


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def softmax(eta):
    m = max(eta)
    ex = [exp(e - m) for e in eta]
    s = sum(ex)
    return [e / s for e in ex]


def multinomial_fit(X, cat, k):
    p = len(X[0])
    dim = (k - 1) * p
    beta = [mpf(0)] * dim
    for _ in range(80):
        g = matrix(dim, 1)
        H = matrix(dim, dim)
        for x, c in zip(X, cat):
            eta = [mpf(0)] + [sum(beta[(a - 1) * p + j] * x[j] for j in range(p)) for a in range(1, k)]
            pr = softmax(eta)
            for a in range(1, k):
                for j in range(p):
                    g[(a - 1) * p + j] += x[j] * ((1 if c == a else 0) - pr[a])
                    for b in range(1, k):
                        w = pr[a] * ((1 if a == b else 0) - pr[b])
                        for l in range(p):
                            H[(a - 1) * p + j, (b - 1) * p + l] += x[j] * x[l] * w
        step = lu_solve(H, g)
        beta = [beta[i] + step[i] for i in range(dim)]
        if max(abs(step[i]) for i in range(dim)) < mpf(10) ** -45:
            break
    return beta


def multinomial_predict(beta, x, k):
    p = len(x)
    eta = [mpf(0)] + [sum(beta[(a - 1) * p + j] * x[j] for j in range(p)) for a in range(1, k)]
    return softmax(eta)


def ols(rows, y):
    p = len(rows[0])
    A, b = matrix(p, p), matrix(p, 1)
    for x, t in zip(rows, y):
        for i in range(p):
            b[i] += x[i] * t
            for j in range(p):
                A[i, j] += x[i] * x[j]
    sol = lu_solve(A, b)
    return [sol[i] for i in range(p)]


def mean_se(vals):
    m = len(vals)
    mu = sum(vals) / m
    var = sum((v - mu) ** 2 for v in vals) / (m - 1)
    return mu, sqrt(var / m)


def reference(case):
    k = len(case["support"])
    n = len(case["y"])
    X = [[mpf(1)] + [mpf(z[i]) for z in case["covariates"]] for i in range(n)]
    y = [mpf(v) for v in case["y"]]
    cat = [case["support"].index(t) for t in case["treatment"]]
    obj = case["objective"]
    scored = []  # (row, policy_choice, action scores list, propensity list)
    for split in case["splits"]:
        train, valid = split["train"], split["valid"]
        if case.get("known_propensity") is None:
            beta = multinomial_fit([X[i] for i in train], [cat[i] for i in train], k)
        coef = []
        for c in range(k):
            tr = [i for i in train if cat[i] == c]
            coef.append(ols([X[i] for i in tr], [y[i] for i in tr]))
        for i in valid:
            if case.get("known_propensity") is None:
                e = multinomial_predict(beta, X[i], k)
            else:
                e = [mpf(v) for v in case["known_propensity"]]
            mu = [sum(coef[c][j] * X[i][j] for j in range(len(X[i]))) for c in range(k)]
            s = [mu[c] + (1 if cat[i] == c else 0) * (y[i] - mu[c]) / e[c] for c in range(k)]
            if case["evaluation"] == "prespecified":
                choice = case["support"].index(case["policy"][i])
            else:
                choice = 0
                for c in range(1, k):
                    if (mu[c] > mu[choice]) if obj == "maximize" else (mu[c] < mu[choice]):
                        choice = c
            scored.append((i, choice, s, e))
    m = len(scored)
    value, se = mean_se([s[c] for _, c, s, _ in scored])
    out = {
        "row_order": [i for i, _, _, _ in scored],
        "scores": [float(s[c]) for _, c, s, _ in scored],
        "policy_actions": [case["support"][c] for _, c, _, _ in scored],
        "value": float(value),
        "se": float(se),
        "action_values": [], "action_se": [],
        "policy_share": [sum(1 for _, c, _, _ in scored if c == a) / m for a in range(k)],
        "propensity_min": [float(min(e[a] for _, _, _, e in scored)) for a in range(k)],
    }
    for a in range(k):
        v, s_ = mean_se([s[a] for _, _, s, _ in scored])
        out["action_values"].append(float(v))
        out["action_se"].append(float(s_))
    if case.get("reference") is not None:
        a = case["support"].index(case["reference"])
        v, s_ = mean_se([s[c] - s[a] for _, c, s, _ in scored])
        out["incremental_value"], out["incremental_se"] = float(v), float(s_)
    return out


def make_case(name, n, support, covs, offset_y, evaluation, objective, folds=None, heldout=None,
              reference_level=None, known=None, y_offset=0.0, cov_offset=0.0):
    k = len(support)
    z = [[cov_offset + (u01() * 2 - 1) * (1 if j == 0 else 0.5) for _ in range(n)] for j in range(covs)]
    # confounded assignment: propensity logits depend on the first covariate
    treat, y = [], []
    for i in range(n):
        zc = z[0][i] - cov_offset
        logits = [0.0] + [0.8 * zc * (a if a % 2 else -a) * 0.5 + 0.2 * a for a in range(1, k)]
        pr = [float(v) for v in softmax([mpf(l) for l in logits])]
        if known is not None:
            pr = known
        r, acc, a_i = u01(), 0.0, k - 1
        for a in range(k):
            acc += pr[a]
            if r < acc:
                a_i = a
                break
        treat.append(float(support[a_i]))
        mean = y_offset + 1 + (0.5 * a_i) * zc + (0.3 * zc if a_i == 0 else -0.4 * zc * a_i) + ((z[1][i] - cov_offset) * 0.2 if covs > 1 else 0.0)
        y.append(mean + (u01() - 0.5))
    case = {"name": name, "support": support, "objective": objective, "evaluation": evaluation,
            "y": y, "treatment": treat, "covariates": z, "reference": reference_level,
            "known_propensity": known, "folds": folds, "heldout": heldout}
    if evaluation == "prespecified":
        case["policy"] = [float(support[1] if z[0][i] - cov_offset > 0 else support[0]) for i in range(n)]
    return case


if __name__ == "__main__":
    import sys
    recorded = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    cases = []
    specs = [
        ("prespecified three actions, three folds", 90, [0.0, 1.0, 2.0], 1, "prespecified", "maximize", 3, None, 1.0, 0.0, 0.0, None),
        ("held out, two covariates, minimise", 150, [0.0, 1.0, 2.0], 2, "held_out", "minimize", None, 0.4, 0.0, 0.0, 0.0, None),
        ("cross fitted learning, four folds", 140, [0.0, 1.0, 2.0], 2, "cross_fitted", "maximize", 4, None, 2.0, 0.0, 0.0, None),
        ("known constant propensity, two folds", 100, [0.0, 1.0], 1, "prespecified", "maximize", 2, None, 0.0, 0.0, 0.0, [0.5, 0.5]),
        ("outcome offset 1e6 and covariate offset 1e3", 100, [0.0, 1.0, 2.0], 2, "prespecified", "maximize", 3, None, 0.0, 1e6, 1e3, None),
    ]
    for name, n, support, covs, ev, obj, folds, heldout, ref, yoff, coff, known in specs:
        case = make_case(name, n, support, covs, 0, ev, obj, folds, heldout, ref, known, yoff, coff)
        n_rows = n
        rec = recorded.get(name)
        if rec is None:
            case["splits"] = None
        else:
            case["splits"] = rec
        if case["splits"] is not None:
            case["expected"] = reference(case)
        cases.append(case)
    print(json.dumps({"cases": cases}, indent=1))
