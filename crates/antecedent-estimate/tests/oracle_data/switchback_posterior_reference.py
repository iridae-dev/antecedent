# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact Normal-Inverse-Gamma reference for the declared-dependence switchback posterior.

Regenerate with:  uv run --script switchback_posterior_reference.py > switchback_posterior_reference.json

Model: for sequence s, y_s = X_s b + e_s with e_s ~ N(0, sigma2 R_s), independent sequences,
b | sigma2 ~ N(0, sigma2 diag(1/precision)), sigma2 ~ InvGamma(a0, b0).  The script builds the
dense block-diagonal Omega = blockdiag(R_s) from the declared correlation (independent,
compound symmetric or AR(1)), inverts it, and computes
    Lambda_n = P + X' Omega^-1 X,   m_n = Lambda_n^-1 X' Omega^-1 y,
    a_n = a0 + n/2,                 b_n = b0 + (e' Omega^-1 e + m_n' P m_n) / 2,   e = y - X m_n,
which is algebraically the product's y' Omega^-1 y - m_n' X' Omega^-1 y but is evaluated from the
residual so it does not cancel.  Every linear functional w'b has a Student-t marginal with
2 a_n degrees of freedom, location w'm_n and squared scale (b_n / a_n) w' Lambda_n^-1 w;
the interval quantile and the sign probability are evaluated from the incomplete beta function
at 60 digits.  Everything is computed from the exact binary64 inputs.  Repository-authored
(MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import math

import mpmath as mp

mp.mp.dps = 60


class Lcg:
    def __init__(self, s): self.s = s
    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)


def corr(kind, rho, m):
    R = mp.zeros(m, m)
    for i in range(m):
        for j in range(m):
            if kind == "independent":
                R[i, j] = 1 if i == j else 0
            elif kind == "compound":
                R[i, j] = 1 if i == j else mp.mpf(rho)
            else:
                R[i, j] = mp.mpf(rho) ** abs(i - j)
    return R


def t_cdf(t, nu):
    t, nu = mp.mpf(t), mp.mpf(nu)
    x = nu / (nu + t * t)
    tail = mp.betainc(nu / 2, mp.mpf(1) / 2, 0, x, regularized=True) / 2
    return 1 - tail if t > 0 else tail


def t_ppf(p, nu):
    return mp.findroot(lambda t: t_cdf(t, nu) - p, (mp.mpf(0), mp.mpf(50)), solver="anderson", tol=1e-40, maxsteps=300)


def design(assign, carry, periods_fe):
    m = len(assign)
    rows = []
    for t in range(m):
        row = [1, int(assign[t])]
        if carry:
            row.append(0 if t == 0 else int(assign[t - 1]))
        if periods_fe:
            row += [1 if j == t else 0 for j in range(1, m)]
        rows.append(row)
    return rows


def run(name, seqs, kind, rho, carry, periods_fe, effect_prec, nuis_prec, a0, b0, level=0.9):
    width = 2 + (1 if carry else 0) + ((len(seqs[0]["y"]) - 1) if periods_fe else 0)
    G = mp.zeros(width, width); xty = mp.zeros(width, 1)
    blocks = []
    n = 0
    for s in seqs:
        m = len(s["y"]); n += m
        X = mp.matrix(design(s["assign"], carry, periods_fe))
        Ri = corr(kind, rho, m) ** -1
        y = mp.matrix([mp.mpf(v) for v in s["y"]])
        G += X.T * Ri * X
        xty += X.T * Ri * y
        blocks.append((X, Ri, y))
    prec = [mp.mpf(effect_prec) if (j == 1 or (carry and j == 2)) else mp.mpf(nuis_prec) for j in range(width)]
    P = mp.diag(prec)
    Lam = P + G
    Linv = Lam ** -1
    mn = Linv * xty
    ss = mp.mpf(0)
    for X, Ri, y in blocks:
        e = y - X * mn
        ss += (e.T * Ri * e)[0]
    ss += (mn.T * P * mn)[0]
    an = mp.mpf(a0) + mp.mpf(n) / 2
    bn = mp.mpf(b0) + ss / 2

    def marg(w):
        w = mp.matrix(w)
        loc = (w.T * mn)[0]
        sc = mp.sqrt(bn / an * (w.T * Linv * w)[0])
        nu = 2 * an
        out = {"location": float(loc), "scale": float(sc), "df": float(nu),
               "interval_radius": float(sc * t_ppf(mp.mpf(1) / 2 + mp.mpf(level) / 2, nu)),
               "probability_positive": float(1 - t_cdf(-loc / sc, nu))}
        out["sd"] = float(sc * mp.sqrt(nu / (nu - 2))) if nu > 2 else None
        return out
    unit = lambda ps: [1 if j in ps else 0 for j in range(width)]
    res = {"direct": marg(unit([1]))}
    if carry:
        res["carryover"] = marg(unit([2])); res["sustained"] = marg(unit([1, 2]))
    first = 2 + (1 if carry else 0)
    res["periods"] = [marg(unit([first + j])) for j in range(width - first)]
    return {"name": name, "kind": kind, "rho": rho, "carryover": carry, "period_effects": periods_fe,
            "effect_precision": effect_prec, "nuisance_precision": nuis_prec, "variance_shape": a0,
            "variance_scale": b0, "level": level,
            "sequences": [{"id": s["id"], "y": s["y"], "assign": s["assign"]} for s in seqs],
            "shape": float(an), "scale": float(bn), "result": res}


def make(seed, lengths, base=0.0, effect=1.0, noise=1.0):
    r = Lcg(seed)
    seqs = []
    for k, m in enumerate(lengths):
        assign = [r.u() < 0.5 for _ in range(m)]
        if all(assign) or not any(assign):
            assign[0] = not assign[0]
        y = [round(base + effect * assign[t] + 0.5 * (assign[t - 1] if t else 0) + noise * (r.u() - 0.5) * 2 + 0.2 * t, 6) for t in range(m)]
        seqs.append({"id": f"s{k:02d}", "y": y, "assign": assign})
    return seqs


def main():
    cases = []
    eq = lambda n, m: [m] * n
    cases.append(run("independent_no_carryover", make(1, eq(6, 4)), "independent", 0.0, False, False, 0.5, 0.1, 2.0, 1.5))
    cases.append(run("compound_with_carryover", make(2, eq(8, 5)), "compound", 0.4, True, False, 1.0, 0.01, 3.0, 2.0))
    cases.append(run("autoregressive_negative_with_period_effects", make(3, eq(6, 4)), "ar", -0.6, False, True, 0.2, 0.05, 1.5, 1.0))
    cases.append(run("compound_near_admissible_lower_bound", make(4, eq(6, 4)), "compound", -1 / 3 + 1e-3, True, False, 0.5, 0.1, 2.0, 1.0))
    cases.append(run("autoregressive_near_unit_correlation", make(5, eq(6, 5)), "ar", 0.999, False, False, 0.5, 0.1, 2.0, 1.0))
    cases.append(run("unequal_sequence_lengths", make(6, [3, 5, 4, 6, 4]), "ar", 0.3, True, False, 0.5, 0.1, 2.0, 1.0))
    cases.append(run("weak_prior", make(7, eq(6, 4)), "compound", 0.2, True, False, 1e-8, 1e-8, 0.5, 0.5))
    cases.append(run("strong_prior", make(8, eq(6, 4)), "independent", 0.0, False, False, 1e6, 1e6, 2.0, 1.0))
    for base in (1e3, 1e5, 1e7):
        cases.append(run(f"outcome_offset_{base:g}", make(9, eq(8, 4), base=base), "independent", 0.0, False, False, 0.5, 1e-16, 2.0, 1.0))
    cases.append(run("outcome_offset_1e6_correlated", make(10, eq(8, 4), base=1e6), "compound", 0.5, True, False, 0.5, 1e-16, 2.0, 1.0))
    print(json.dumps(cases, indent=1))


if __name__ == "__main__":
    main()
