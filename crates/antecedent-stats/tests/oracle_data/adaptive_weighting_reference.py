# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for adaptively weighted augmented inverse-probability estimation.

Regenerate with:  uv run --script adaptive_weighting_reference.py > adaptive_weighting_reference.json

Written from the estimator's definition (Hadad, Hirshberg, Zhan, Wager and Athey 2021), at 50
digits.  For unit i, history H_{i-1} and arm a:
  m_i(a)      = (strength * prior_mean + sum of earlier outcomes of a) / (strength + earlier count of a)
  Gamma_i(a)  = m_i(a) + 1[A_i = a] (Y_i - m_i(a)) / e_i(a)
  h_i         = 1, or (sum over the estimand's arms of 1/e_i(arm))^(-1/2), times an optional multiplier
  estimate    = sum h_i v_i / sum h_i,   variance = sum h_i^2 (v_i - estimate)^2 / (sum h_i)^2,
  effective units = (sum h)^2 / sum h^2,  largest share = max h / sum h,
with v_i = Gamma_i(a), or Gamma_i(plus) - Gamma_i(minus) for a contrast.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random

import mpmath as mp

mp.mp.dps = 50
mpf = mp.mpf


def txt(x):
    return mp.nstr(x, 30)


def scores(arms, assigned, probs, outcomes, mean, strength):
    sums, counts = [mpf(0)] * arms, [0] * arms
    out = []
    for i, (a, y) in enumerate(zip(assigned, outcomes)):
        row = []
        for k in range(arms):
            m = (mpf(strength) * mpf(mean) + sums[k]) / (mpf(strength) + counts[k])
            row.append(m + ((mpf(y) - m) / mpf(probs[i][k]) if k == a else 0))
        out.append(row)
        sums[a] += mpf(y)
        counts[a] += 1
    return out


def estimate(arms, probs, sc, members, contrast, scheme, multiplier):
    w, v = [], []
    for i in range(len(sc)):
        inv = sum(1 / mpf(probs[i][a]) for a in members)
        h = mpf(1) if scheme == "uniform" else 1 / mp.sqrt(inv)
        if multiplier:
            h *= mpf(multiplier[i])
        w.append(h)
        v.append(sc[i][members[0]] - sc[i][members[1]] if contrast else sc[i][members[0]])
    total = sum(w)
    est = sum(h * x for h, x in zip(w, v)) / total
    var = sum((h * (x - est)) ** 2 for h, x in zip(w, v)) / total**2
    sq = sum(h * h for h in w)
    return {"estimate": txt(est), "variance": txt(var), "weight_sum": txt(total), "effective_units": txt(total**2 / sq),
            "max_weight_share": txt(max(w) / total)}


def dyadic_row(rnd, arms, floor_bits=None):
    bits = 10
    cuts = sorted(rnd.sample(range(1, 1 << bits), arms - 1))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [1 << bits])]
    return [p / (1 << bits) for p in parts]


def main():
    rnd = random.Random(20261028)
    cases = []
    for arms, n, mean, strength, outcome_scale in [
        (2, 8, 0.5, 1.0, 1.0), (3, 40, 0.0, 2.0, 1.0), (2, 200, 0.3, 1e-9, 1.0), (2, 60, 0.3, 1e9, 1.0), (4, 120, 10.0, 0.5, 1e3),
        (2, 50, 1e6, 1.0, 1.0), (5, 80, 0.5, 3.0, 1.0), (2, 30, 0.5, 1.0, 1e-6),
    ]:
        probs = [dyadic_row(rnd, arms) for _ in range(n)]
        if arms == 2 and n == 50:
            probs = [[1 / 1024, 1023 / 1024] if rnd.random() < 0.5 else [1023 / 1024, 1 / 1024] for _ in range(n)]
        assigned = [rnd.choices(range(arms), weights=p)[0] for p in probs]
        outcomes = [round(rnd.gauss(0, 1) * outcome_scale * 1024) / 1024 for _ in range(n)]
        sc = scores(arms, assigned, probs, outcomes, mean, strength)
        case = {"arms": arms, "assigned": assigned, "probabilities": probs, "outcomes": outcomes, "prior_mean": mean, "strength": strength,
                "scores": [txt(x) for row in sc for x in row], "estimates": []}
        multipliers = [None, [round(0.5 + 3 * rnd.random(), 3) for _ in range(n)]]
        for members, contrast in [((0,), False), ((arms - 1,), False), ((0, arms - 1), True)]:
            for scheme in ("uniform", "variance_stabilizing"):
                for multiplier in multipliers:
                    m = list(members)
                    case["estimates"].append({"estimand": {"contrast": list(members) if contrast else None, "arm": None if contrast else members[0]},
                                              "scheme": scheme, "multiplier": multiplier, "floor": min(min(probs[i][a] for a in members) for i in range(n)),
                                              **estimate(arms, probs, sc, m, contrast, scheme, multiplier)})
        cases.append(case)
    print(json.dumps({"cases": cases}, separators=(",", ":")))


main()
