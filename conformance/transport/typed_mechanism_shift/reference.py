# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.1.3", "scipy==1.14.1"]
# ///
"""Independent reference for typed mechanism transport under a mechanism shift.

Generates the source and target samples, fits every outcome mechanism on the
*target* sample with numpy/scipy (least squares, or scipy BFGS polished by a
root solve of the score equations), and evaluates the target interventional mean
E[outcome | do(price = p)] by Gauss-Hermite quadrature over the invariant
(source-fitted) income law. The parameter standard error of each response is the
first-order delta method with a finite-difference gradient.

The Rust route never sees these numbers while fitting: it fits the same data
itself, and the test compares its transported response with this reference and
with the closed-form truth of the data-generating process.

Run: uv run --script conformance/transport/typed_mechanism_shift/reference.py          (rewrites data.json and expected.json)
     uv run --frozen --script conformance/transport/typed_mechanism_shift/reference.py --check  (compare)
"""
import json
import pathlib
import sys

import numpy as np
from scipy import optimize, special, stats

HERE = pathlib.Path(__file__).resolve().parent
N = 3000
GRID = [0.0, 1.0, 2.0]
SEED = 20261027
QUAD_X, QUAD_W = np.polynomial.hermite_e.hermegauss(120)
QUAD_W = QUAD_W / np.sqrt(2.0 * np.pi)

# True coefficients on the basis [1, price, price^2, income, income^2].
SOURCE_TRUTH = {
    "demand": [2.0, -1.0, 0.0, 0.3, 0.0],
    "orders": [0.3, 0.2, 0.0, 0.1, 0.0],
    "buys": [-0.5, 0.5, 0.0, 0.3, 0.0],
    "spend": [1.0, 0.2, 0.0, -0.1, 0.0],
    "basket": [0.5, 0.3, 0.0, 0.2, 0.0],
    "noise_h": [1.0, 0.8, 0.0, 0.2, 0.0],
}
TARGET_TRUTH = {
    "demand": [2.5, -0.8, -0.5, 0.3, 0.0],
    "orders": [0.4, 0.5, -0.2, 0.1, 0.0],
    "buys": [-0.3, -0.8, 0.3, 0.3, 0.0],
    "spend": [1.2, -0.4, 0.1, -0.1, 0.0],
    "basket": [1.0, -0.4, 0.12, 0.2, 0.0],
    "noise_h": [0.5, 1.5, -0.4, 0.2, 0.0],
}
FAMILY = {
    "demand": "gaussian",
    "orders": "poisson",
    "buys": "bernoulli",
    "spend": "gamma",
    "basket": "log_normal",
    "noise_h": "heteroscedastic_gaussian",
}
GAMMA_SHAPE = 4.0
BASKET_SIGMA = 0.3
DEMAND_SIGMA = 0.4


def basis(price, income):
    price = np.asarray(price, dtype=float)
    income = np.asarray(income, dtype=float)
    return np.column_stack([np.ones_like(price), price, price**2, income, income**2])


def logistic(x):
    return special.expit(x)


def hetero_logvar(price, income, source):
    return (-1.0 + 0.8 * (price - 1.0)) if source else (-1.0 + 1.0 * (price - 1.0))


def draw_population(rng, source):
    truth = SOURCE_TRUTH if source else TARGET_TRUTH
    income = rng.standard_normal(N)
    price = 1.0 + 0.5 * income + 0.5 * rng.standard_normal(N)
    x = basis(price, income)
    out = {"income": income, "price": price}
    out["demand"] = x @ truth["demand"] + DEMAND_SIGMA * rng.standard_normal(N)
    out["orders"] = rng.poisson(np.exp(x @ truth["orders"])).astype(float)
    out["buys"] = (rng.random(N) < logistic(x @ truth["buys"])).astype(float)
    mu = np.exp(x @ truth["spend"])
    out["spend"] = rng.gamma(GAMMA_SHAPE, mu / GAMMA_SHAPE)
    out["basket"] = np.exp(x @ truth["basket"] + BASKET_SIGMA * rng.standard_normal(N))
    sd = np.exp(0.5 * hetero_logvar(price, income, source))
    out["noise_h"] = x @ truth["noise_h"] + sd * rng.standard_normal(N)
    return out


def round_columns(columns):
    return {k: [float(f"{v:.9f}") for v in vals] for k, vals in columns.items()}


# ----------------------------------------------------------------- fitting

def polish(score, x0):
    sol = optimize.root(score, x0, method="hybr", tol=1e-13)
    return sol.x if sol.success else x0


def fit_gaussian(x, y):
    n, k = x.shape
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    rss = float(np.sum((y - x @ beta) ** 2))
    s2 = rss / (n - k)
    cov = s2 * np.linalg.inv(x.T @ x)
    return {"beta": beta, "cov": cov, "sigma": np.sqrt(s2), "lnsigma_var": 1.0 / (2.0 * (n - k))}


def fit_glm(family, x, y):
    n, k = x.shape
    if family == "poisson":
        nll = lambda b: float(np.sum(np.exp(x @ b) - y * (x @ b)))
        grad = lambda b: x.T @ (np.exp(x @ b) - y)
        start = np.zeros(k)
        start[0] = np.log(y.mean())
    elif family == "bernoulli":
        nll = lambda b: float(np.sum(np.logaddexp(0.0, x @ b) - y * (x @ b)))
        grad = lambda b: x.T @ (logistic(x @ b) - y)
        start = np.zeros(k)
    else:
        nll = lambda b: float(np.sum(y * np.exp(-(x @ b)) + x @ b))
        grad = lambda b: x.T @ (1.0 - y * np.exp(-(x @ b)))
        start = np.zeros(k)
        start[0] = np.log(y.mean())
    res = optimize.minimize(nll, start, jac=grad, method="BFGS", options={"gtol": 1e-10, "maxiter": 2000})
    beta = polish(grad, res.x)
    eta = x @ beta
    if family == "poisson":
        w = np.exp(eta)
        cov = np.linalg.inv((x * w[:, None]).T @ x)
    elif family == "bernoulli":
        p = logistic(eta)
        cov = np.linalg.inv((x * (p * (1 - p))[:, None]).T @ x)
    else:
        mu = np.exp(eta)
        phi = float(np.sum(((y - mu) / mu) ** 2)) / (n - k)
        cov = phi * np.linalg.inv(x.T @ x)
    return {"beta": beta, "cov": cov}


def fit_hetero(x, z, y):
    n, k = x.shape
    q = z.shape[1]
    beta0 = np.linalg.lstsq(x, y, rcond=None)[0]
    r0 = y - x @ beta0
    g0 = np.zeros(q)
    g0[0] = np.log(np.mean(r0**2))

    def split(t):
        return t[:k], t[k:]

    def nll(t):
        b, g = split(t)
        lv = z @ g
        r = y - x @ b
        return float(0.5 * np.sum(lv + r**2 * np.exp(-lv)))

    def grad(t):
        b, g = split(t)
        lv = z @ g
        r = y - x @ b
        e = np.exp(-lv)
        gb = -(x.T @ (r * e))
        gg = 0.5 * (z.T @ (1.0 - r**2 * e))
        return np.concatenate([gb, gg])

    start = np.concatenate([beta0, g0])
    res = optimize.minimize(nll, start, jac=grad, method="BFGS", options={"gtol": 1e-10, "maxiter": 5000})
    t = polish(grad, res.x)
    b, g = split(t)
    w = np.exp(-(z @ g))
    cov = np.linalg.inv((x * w[:, None]).T @ x)
    return {"beta": b, "cov": cov}


# --------------------------------------------------------------- responses

def response(family, coeffs, scale, income_m, income_s, p):
    z = income_m + income_s * QUAD_X
    eta = coeffs[0] + coeffs[1] * p + coeffs[2] * p * p + coeffs[3] * z + coeffs[4] * z * z
    if family in ("gaussian", "heteroscedastic_gaussian"):
        g = eta
    elif family == "log_normal":
        g = np.exp(eta + 0.5 * scale * scale)
    elif family == "bernoulli":
        g = logistic(eta)
    else:
        g = np.exp(eta)
    return float(np.sum(QUAD_W * g))


def response_se(family, fit, income, p):
    coeffs = fit["beta"]
    extra = 0.0
    theta = list(coeffs)
    cov_blocks = [fit["cov"]]
    if family == "log_normal":
        theta.append(np.log(fit["sigma"]))
        cov_blocks.append(np.array([[fit["lnsigma_var"]]]))
    theta += [income["m"], np.log(income["s"])]
    cov_blocks.append(np.array([[income["s"] ** 2 / income["n"], 0.0], [0.0, 1.0 / (2.0 * (income["n"] - 1))]]))
    dim = len(theta)
    sigma = np.zeros((dim, dim))
    at = 0
    for block in cov_blocks:
        sigma[at : at + block.shape[0], at : at + block.shape[0]] = block
        at += block.shape[0]

    def value(t):
        c = np.array(t[:5])
        pos = 5
        scale = 0.0
        if family == "log_normal":
            scale = float(np.exp(t[pos]))
            pos += 1
        return response(family, c, scale, t[pos], float(np.exp(t[pos + 1])), p)

    theta = np.array(theta, dtype=float)
    g = np.zeros(dim)
    for j in range(dim):
        h = 1e-5 * (1.0 + abs(theta[j]))
        up, dn = theta.copy(), theta.copy()
        up[j] += h
        dn[j] -= h
        g[j] = (value(up) - value(dn)) / (2.0 * h)
    return float(np.sqrt(g @ sigma @ g)), value(theta)


def wald_p(theta_s, cov_s, theta_t, cov_t):
    d = theta_s - theta_t
    stat = float(d @ np.linalg.solve(cov_s + cov_t, d))
    return stat, float(stats.chi2.sf(stat, len(d)))


def standardized_price_fit(income, price, centre, scale):
    z = (income - centre) / scale
    x = np.column_stack([np.ones_like(z), z])
    f = fit_gaussian(x, price)
    theta = np.concatenate([f["beta"], [np.log(f["sigma"])]])
    cov = np.zeros((3, 3))
    cov[:2, :2] = f["cov"]
    cov[2, 2] = f["lnsigma_var"]
    return theta, cov


def build():
    rng = np.random.default_rng(SEED)
    source = draw_population(rng, True)
    target = draw_population(rng, False)
    shifted_price = 1.0 + 0.9 * target["income"] + 0.5 * rng.standard_normal(N)
    data = {
        "source": round_columns(source),
        "target": {**round_columns(target), **round_columns({"price_shifted": shifted_price})},
    }
    # Everything below is computed from the rounded samples that are saved.
    src = {k: np.array(v) for k, v in data["source"].items()}
    tgt = {k: np.array(v) for k, v in data["target"].items()}

    inc_m = float(src["income"].mean())
    inc_s = float(np.sqrt(np.sum((src["income"] - inc_m) ** 2) / (N - 1)))
    income = {"m": inc_m, "s": inc_s, "n": N}

    x = basis(tgt["price"], tgt["income"])
    z = np.column_stack([np.ones(N), tgt["price"], tgt["income"]])
    outcomes = {}
    for name, family in FAMILY.items():
        y = tgt[name]
        if family in ("gaussian", "log_normal"):
            fit = fit_gaussian(x, np.log(y) if family == "log_normal" else y)
        elif family == "heteroscedastic_gaussian":
            fit = fit_hetero(x, z, y)
        else:
            fit = fit_glm(family, x, y)
        sigma_true = {"log_normal": BASKET_SIGMA}.get(family, 0.0)
        reference, reference_se, truth = [], [], []
        for p in GRID:
            se, value = response_se(family, fit, income, p)
            reference.append(value)
            reference_se.append(se)
            truth.append(response(family, np.array(TARGET_TRUTH[name]), sigma_true, 0.0, 1.0, p))
        outcomes[name] = {
            "family": family,
            "reference": reference,
            "reference_se": reference_se,
            "truth": truth,
            "coefficients": [float(c) for c in fit["beta"]],
        }
        for r, s, t in zip(reference, reference_se, truth):
            assert abs(r - t) < 4.5 * s, (name, r, t, s)

    centre = float(src["income"].mean())
    scale = float(np.sqrt(np.mean((src["income"] - centre) ** 2)))
    ts, cs = standardized_price_fit(src["income"], src["price"], centre, scale)
    tt, ct = standardized_price_fit(tgt["income"], tgt["price"], centre, scale)
    tt2, ct2 = standardized_price_fit(tgt["income"], tgt["price_shifted"], centre, scale)
    income_theta = lambda col: (np.array([col.mean(), np.log(np.sqrt(np.sum((col - col.mean()) ** 2) / (N - 1)))]))
    income_cov = lambda col: np.diag([np.sum((col - col.mean()) ** 2) / (N - 1) / N, 1.0 / (2.0 * (N - 1))])
    invariance = {
        "price": wald_p(ts, cs, tt, ct),
        "price_shifted": wald_p(ts, cs, tt2, ct2),
        "income": wald_p(income_theta(src["income"]), income_cov(src["income"]), income_theta(tgt["income"]), income_cov(tgt["income"])),
    }
    assert invariance["price"][1] > 0.05 and invariance["income"][1] > 0.05
    assert invariance["price_shifted"][1] < 1e-12

    expected = {
        "oracle": {
            "kind": "independent_reimplementation",
            "method": "target-sample fit by numpy least squares or scipy BFGS plus root-polished score equations; Gauss-Hermite quadrature over the source-fitted income law; delta-method standard error from a finite-difference gradient",
            "independence_basis": "the fits are computed here with numpy and scipy from the saved samples; no Rust code or Rust-produced number enters this file. The retained generator and the executing assertions were read for the review recorded in reference_review.",
            "generator": "conformance/transport/typed_mechanism_shift/reference.py",
            "generator_retained": True,
            "environment_lock": "conformance/transport/typed_mechanism_shift/reference.py.lock",
            "command": "uv run --frozen --script conformance/transport/typed_mechanism_shift/reference.py --check",
            "asserted_outputs": [
                "outcomes[*].reference and reference_se (Monte Carlo and relative-SE tolerances)",
                "outcomes[*].truth with reference_se (4.5 sigma band)",
                "invariance[*].statistic and p_value (1e-6)",
            ],
            "context_only_outputs": ["outcomes[*].coefficients", "sample_size", "income"],
            "executing_assertion": "crates/antecedent-model/tests/typed_mechanism_transport.rs::transported_response_matches_the_independent_target_fit_for_every_family",
            "historical_exposure": "Historical author exposure is unrecorded and remains unknown.",
        },
        "grid": GRID,
        "sample_size": N,
        "income": {"mean": inc_m, "sd": inc_s},
        "outcomes": outcomes,
        "invariance": {k: {"statistic": v[0], "p_value": v[1]} for k, v in invariance.items()},
    }
    return data, expected, outcomes


def close(left, right, rtol=1e-6, atol=1e-9):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(close(left[k], right[k], rtol, atol) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, rtol, atol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
        return abs(left - right) <= atol + rtol * abs(right)
    return left == right


def main():
    data, expected, outcomes = build()
    if "--check" in sys.argv[1:]:
        committed = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
        review = committed.get("oracle", {}).get("reference_review")
        stored = {**committed, "oracle": {k: v for k, v in committed["oracle"].items() if k != "reference_review"}}
        if not close(expected, stored) or not close(data, json.loads((HERE / "data.json").read_text(encoding="utf-8"))):
            print("typed mechanism reference mismatch", file=sys.stderr)
            return 1
        return 0
    review = None
    path = HERE / "expected.json"
    if path.is_file():
        review = json.loads(path.read_text(encoding="utf-8")).get("oracle", {}).get("reference_review")
    if review is not None:
        expected["oracle"]["reference_review"] = review
    (HERE / "data.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    for name, o in outcomes.items():
        print(name, o["reference"], o["truth"], o["reference_se"])
    print({k: v for k, v in expected["invariance"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
