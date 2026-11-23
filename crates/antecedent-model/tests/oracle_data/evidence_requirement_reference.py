# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the evidence-requirement solves and the finite Bayes posterior.

Regenerate with:  uv run --script evidence_requirement_reference.py > evidence_requirement_reference.json

Written from the definitions, in exact rational arithmetic (the doubles' exact values) wherever the
quantity is rational and at 60 digits (mpmath) for large counts and for logarithms:
  Bayes factor to move prior p to level q:     [q/(1-q)] / [p/(1-p)]
  identical binary-signal observations:        smallest n with  odds(p) * BF^n  >=  odds(q)   (raise)
                                               or               <=  odds(q)                    (lower)
                                               BF = r_h/r_c (positive readings) or (1-r_h)/(1-r_c)
  Beta-Bernoulli predictive after n successes: (a + n)/(a + b + n);  after n failures: a/(a + b + n);
                                               the smallest n is a ceiling of an exact rational
  finite Bayes model:                          posterior_h  proportional to  prior_h prod_o L[h][o]^count_o,
                                               predictive_o = sum_h posterior_h L[h][o]

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import math
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60
mpf = mp.mpf


def txt(x, digits=30):
    if isinstance(x, F):
        return mp.nstr(mpf(x.numerator) / x.denominator, digits)
    return mp.nstr(x, digits)


def odds(p):
    p = F(p)
    return p / (1 - p)


def bayes_factor_cases():
    out = []
    for prior in (1e-12, 0.001, 0.2, 0.5, 0.9, 1 - 2**-30):
        for level in (1e-9, 0.01, 0.3, 0.5, 0.95, 1 - 2**-20):
            for direction in ("raise", "lower"):
                needed = (direction == "raise" and level > prior) or (direction == "lower" and level < prior)
                out.append({"prior": prior, "level": level, "direction": direction, "already_satisfied": not needed,
                            "factor": txt(odds(level) / odds(prior)) if needed else "1"})
    return out


def exact_n(prior, level, direction, bf, cap):
    """Smallest n >= 1 with the odds condition, by exact rational powers up to cap, else None."""
    p, t = odds(prior), odds(level)
    value = p
    for n in range(1, cap + 1):
        value *= bf
        if (direction == "raise" and value >= t) or (direction == "lower" and value <= t):
            return n, value / (1 + value)
    return None, None


def signal_cases():
    out = []
    specs = [
        (0.1, 0.9, "raise", 0.9, 0.2), (0.5, 0.99, "raise", 0.8, 0.3), (0.5, 0.01, "lower", 0.8, 0.3), (0.3, 0.999999, "raise", 0.6, 0.55),
        (0.5, 0.75, "raise", 0.95, 0.05), (0.01, 0.5, "raise", 0.9, 0.1), (0.7, 0.2, "lower", 0.4, 0.6), (0.2, 0.9, "raise", 1.0, 0.5),
        (0.2, 0.9, "raise", 0.5, 0.0), (0.2, 0.05, "lower", 0.0, 0.5), (0.2, 0.05, "lower", 0.5, 1.0), (0.2, 0.9, "raise", 0.5, 0.5),
        (0.2, 0.9, "raise", 0.3, 0.6), (0.5, 0.9, "raise", 0.5 + 2**-30, 0.5), (0.5, 0.9, "raise", 0.5 + 2**-19, 0.5), (0.5, 0.2, "lower", 0.5 + 2**-19, 0.5), (0.4, 0.4000000001, "raise", 0.75, 0.25), (0.5, 0.5 + 2**-40, "raise", 0.5 + 2**-10, 0.5),
    ]
    for prior, level, direction, rh, rc, in specs:
        num, den = (F(rh), F(rc)) if direction == "raise" else (1 - F(rh), 1 - F(rc))
        case = {"prior": prior, "level": level, "direction": direction, "rate_h": rh, "rate_c": rc, "max": 1 << 20}
        if (direction == "raise" and level <= prior) or (direction == "lower" and level >= prior):
            case["status"] = "satisfied"
        elif num == 0 and den == 0:
            case["status"] = "no_move"
        elif den == 0:
            # infinite Bayes factor: one positive reading settles it, but it cannot lower anything
            if direction == "raise":
                case["status"] = "ok"; case["n"] = 1; case["value"] = "1"
            else:
                case["status"] = "no_move"
        elif num == 0:
            if direction == "lower":
                case["status"] = "ok"; case["n"] = 1; case["value"] = "0"
            else:
                case["status"] = "no_move"
        else:
            bf = num / den
            if (direction == "raise" and bf <= 1) or (direction == "lower" and bf >= 1):
                case["status"] = "no_move"
            else:
                # n by logarithms at 60 digits, confirmed exactly when small
                ratio = mp.log(mpf(odds(level).numerator) / odds(level).denominator / (mpf(odds(prior).numerator) / odds(prior).denominator)) \
                    / mp.log(mpf(bf.numerator) / bf.denominator)
                n_guess = int(mp.ceil(ratio))
                if n_guess > case["max"]:
                    case["status"] = "budget"
                else:
                    if n_guess <= 3000:
                        n, value = exact_n(prior, level, direction, bf, 3000)
                        assert n == n_guess, (n, n_guess, case)
                    case["status"] = "ok"
                    case["n"] = n_guess
                    lo = mp.log(mpf(odds(prior).numerator) / odds(prior).denominator) + n_guess * mp.log(mpf(bf.numerator) / bf.denominator)
                    case["value"] = txt(1 / (1 + mp.exp(-lo)))
                    case["bf"] = txt(bf)
        out.append(case)
    # a budget case with the extremum
    return out


def conjugate_cases():
    out = []
    for a, b, level, direction in [(1, 1, 0.9, "raise"), (2, 8, 0.5, "raise"), (0.5, 0.5, 0.99, "raise"), (5, 5, 0.1, "lower"), (3, 1, 0.01, "lower"),
                                   (1e-3, 1e3, 0.5, "raise"), (1e6, 1e6, 0.5000005, "raise"), (1, 1, 0.5, "raise"), (1, 1, 0.2, "raise")]:
        # the level is the decimal the caller wrote; the solve counts a level matched within 1e-12 as reached
        a_, b_, q = F(a), F(b), F(str(level))
        mean = a_ / (a_ + b_)
        tol = F("1e-12")  # a level matched within 1e-12 in probability counts as reached
        if (direction == "raise" and mean >= q - tol) or (direction == "lower" and mean <= q + tol):
            out.append({"a": a, "b": b, "level": level, "direction": direction, "status": "satisfied", "mean": txt(mean)})
            continue
        if direction == "raise":
            n = math.ceil(((q - tol) * (a_ + b_) - a_) / (1 - (q - tol)))
            value = (a_ + n) / (a_ + b_ + n)
        else:
            n = math.ceil(a_ / (q + tol) - a_ - b_)
            value = a_ / (a_ + b_ + n)
        out.append({"a": a, "b": b, "level": level, "direction": direction, "status": "ok", "n": int(n), "value": txt(value)})
    return out


def dyadic(rnd, k, bits=6):
    total = 1 << bits
    cuts = sorted(rnd.sample(range(1, total), k - 1))
    return [(b - a) / total for a, b in zip([0] + cuts, cuts + [total])]


def bayes_models(rnd):
    out = []
    for h, o, big in [(2, 2, False), (3, 3, False), (5, 4, False), (4, 2, False), (3, 3, True), (2, 4, True)]:
        prior = dyadic(rnd, h)
        like = [dyadic(rnd, o) for _ in range(h)]
        counts = [rnd.randint(0, 6 if big else 40) * (10**rnd.randint(4, 8) if big else 1) for _ in range(o)]
        if sum(counts) == 0:
            counts[0] = 1
        weights = []
        for i in range(h):
            w = mpf(prior[i].hex() and F(prior[i]).numerator) / F(prior[i]).denominator
            for j in range(o):
                if counts[j]:
                    lf = F(like[i][j])
                    w *= (mpf(lf.numerator) / lf.denominator) ** counts[j]
            weights.append(w)
        z = sum(weights)
        post = [w / z for w in weights]
        predictive = [sum(post[i] * mpf(F(like[i][j]).numerator) / F(like[i][j]).denominator for i in range(h)) for j in range(o)]
        out.append({"prior": prior, "likelihood": [v for row in like for v in row], "levels": o, "counts": counts,
                    "posterior": [txt(p) for p in post], "log_evidence": txt(mp.log(z)), "predictive": [txt(p) for p in predictive]})
    return out


def beta_cases():
    out = []
    for a, b, s, f in [(1, 1, 3, 7), (0.5, 0.5, 0, 0), (2, 3, 10**9, 5), (1e-3, 1e-3, 1, 0), (1e300, 1e300, 5, 5), (1.5, 2.5, 2**52, 2**52 - 1)]:
        a_, b_ = F(a) + s, F(b) + f
        out.append({"a": a, "b": b, "s": s, "f": f, "mean_prior": txt(F(a) / (F(a) + F(b))), "mean_post": txt(a_ / (a_ + b_)),
                    "alpha": txt(a_), "beta": txt(b_)})
    return out


def main():
    rnd = random.Random(20261028)
    print(json.dumps({"bayes_factor": bayes_factor_cases(), "signal": signal_cases(), "conjugate": conjugate_cases(),
                      "bayes_models": bayes_models(rnd), "beta": beta_cases()}, separators=(",", ":")))


main()
