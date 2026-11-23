# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact finite-SCM recovery and the BCa interval of the sampled observation recovery.

Regenerate in two stages (the bootstrap replicate effects come from the product's seeded
resampling, so they are inputs of the BCa part of this reference, not outputs of it):

  python3 recovery_sampled_observation_reference.py inputs > inputs.json
  (the test's ignored dump test reads inputs.json, runs the product and writes stream.json with
   the bootstrap replicate effects of the sampled table)
  uv run --script recovery_sampled_observation_reference.py final stream.json > recovery_sampled_observation_reference.json

The structural model is O -> X0, O -> X1, X0 -> X1 (treatment X0, outcome X1), a response R0
that depends on O and a response R1 that depends on X0; a variable is seen only when its response
is 1, O is always seen.  Every probability is a multiple of 1/4, so each of the 128 observation
patterns has a probability that is a multiple of 1/512 and a table of 4096 rows holds every
pattern an exact integer number of times.  The exact backdoor effect is
sum_o P(o) [P(X1 = 1 | X0 = 1, o) - P(X1 = 1 | X0 = 0, o)].

Recovery from observed pattern counts N (derived from the factorisation, not from the product):
p(o, x0, x1) = P(R0=1,R1=1,o,x0,x1) / (P(R0=1|o) q(x0)), P(R0=1|o) = P(R0=1,o)/P(o),
q(x0) = P(R0=1,R1=1,X0=x0)/P(R0=1,X0=x0); the effect is sum_o p(o) [p(o,1,1)/p(o,1,.) -
p(o,0,1)/p(o,0,.)] with p normalised by its total.  BCa: z0 = Phi^-1((#{b < theta} + #{b = theta}/2)/B),
a = sum m (mean - e)^3 / (6 (sum m (mean - e)^2)^1.5) over the delete-one-row effects e with row
multiplicities m, adjusted tails Phi(z0 + (z0 + z_alpha)/(1 - a (z0 + z_alpha))), type-7 quantiles
of the sorted bootstrap effects.  Evaluated at 50 digits.  Repository-authored (MIT OR Apache-2.0);
mpmath is not a product dependency.
"""
import json
import sys
from fractions import Fraction as F

P_O = F(1, 2)
P_X0 = [F(1, 4), F(3, 4)]            # by O
P_X1 = [F(1, 4), F(1, 2), F(1, 2), F(3, 4)]  # by 2 * X0 + O
P_R0 = [F(1, 2), F(3, 4)]            # by O
P_R1 = [F(1, 2), F(3, 4)]            # by X0


def bern(p1, level):
    return p1 if level == 1 else 1 - p1


def pattern_probabilities():
    out = {}
    for o in range(2):
        for x0 in range(2):
            for x1 in range(2):
                for r0 in range(2):
                    for r1 in range(2):
                        p = (bern(P_O, o) * bern(P_X0[o], x0) * bern(P_X1[2 * x0 + o], x1)
                             * bern(P_R0[o], r0) * bern(P_R1[x0], r1))
                        pat = (r0 | (r1 << 1), (r0 & x0) | ((r1 & x1) << 1), o)
                        out[pat] = out.get(pat, 0) + p
    return out


def exact_truth():
    effect = F(0)
    for o in range(2):
        effect += bern(P_O, o) * (P_X1[2 + o] - P_X1[o])
    return effect


class Lcg:
    def __init__(self, s):
        self.s = s

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)


def effect_of(counts, mp):
    """Recovered effect from pattern counts {(responses, proxies, fully): count}."""
    n = sum(counts.values())
    def cnt(pred):
        return sum(c for (r, p, o), c in counts.items() if pred(r, p, o))
    law = {}
    for o in range(2):
        p_o = mp.mpf(cnt(lambda r, p, oo: oo == o)) / n
        p_r0_o = mp.mpf(cnt(lambda r, p, oo: oo == o and r & 1)) / n / p_o
        for x0 in range(2):
            num = cnt(lambda r, p, oo: r == 3 and (p & 1) == x0)
            den = cnt(lambda r, p, oo: (r & 1) and (p & 1) == x0)
            q = mp.mpf(num) / den
            for x1 in range(2):
                cell = mp.mpf(cnt(lambda r, p, oo: r == 3 and oo == o and (p & 1) == x0 and ((p >> 1) & 1) == x1)) / n
                law[(o, x0, x1)] = cell / (p_r0_o * q)
    total = sum(law.values())
    eff = mp.mpf(0)
    for o in range(2):
        mass_o = sum(law[(o, a, b)] for a in range(2) for b in range(2)) / total
        eff += mass_o * (law[(o, 1, 1)] / (law[(o, 1, 0)] + law[(o, 1, 1)]) - law[(o, 0, 1)] / (law[(o, 0, 0)] + law[(o, 0, 1)]))
    return eff, {k: v / total for k, v in law.items()}


def sample_table(n, seed):
    probs = sorted(pattern_probabilities().items())
    g = Lcg(seed)
    counts = {}
    for _ in range(n):
        u = F(g.u()).limit_denominator(10**12)
        acc = F(0)
        chosen = probs[-1][0]
        for pat, p in probs:
            acc += p
            if u < acc:
                chosen = pat
                break
        counts[chosen] = counts.get(chosen, 0) + 1
    return counts


def table_json(counts):
    return [[r, p, o, c] for (r, p, o), c in sorted(counts.items())]


def stage_inputs():
    exact = {pat: int(p * 4096) for pat, p in pattern_probabilities().items()}
    assert all(F(v, 4096) == pattern_probabilities()[k] for k, v in exact.items())
    return {"exact_table": table_json(exact), "sampled_table": table_json(sample_table(1500, 7))}


def stage_final(stream):
    import mpmath as mp
    mp.mp.dps = 50
    inputs = stage_inputs()
    exact_counts = {(r, p, o): c for r, p, o, c in inputs["exact_table"]}
    eff_exact, law_exact = effect_of(exact_counts, mp)
    truth = exact_truth()
    sampled = {(r, p, o): c for r, p, o, c in inputs["sampled_table"]}
    point, _ = effect_of(sampled, mp)
    jack = []
    for pat, c in sorted(sampled.items()):
        reduced = dict(sampled)
        reduced[pat] -= 1
        e, _ = effect_of(reduced, mp)
        jack.append({"pattern": list(pat), "multiplicity": c, "effect": float(e)})
    n = sum(sampled.values())
    mean = sum(mp.mpf(j["effect"]) * j["multiplicity"] for j in jack) / n
    sq = sum((mean - mp.mpf(j["effect"])) ** 2 * j["multiplicity"] for j in jack)
    cu = sum((mean - mp.mpf(j["effect"])) ** 3 * j["multiplicity"] for j in jack)
    accel = cu / (6 * sq ** mp.mpf(1.5))
    samples = sorted(mp.mpf(v) for v in stream["replicate_effects"])
    B = len(samples)
    below = sum(1 for v in samples if v < point)
    ties = sum(1 for v in samples if v == point)
    rank = (below + mp.mpf(ties) / 2) / B
    ninv = lambda p: mp.sqrt(2) * mp.erfinv(2 * p - 1)
    ncdf = lambda z: mp.erfc(-z / mp.sqrt(2)) / 2
    z0 = ninv(rank)
    adj = []
    for tail in (mp.mpf("0.025"), mp.mpf("0.975")):
        z = z0 + ninv(tail)
        adj.append(ncdf(z0 + z / (1 - accel * z)))
    def q7(p):
        h = (B - 1) * p
        lo = int(mp.floor(h))
        hi = min(lo + 1, B - 1)
        return samples[lo] + (h - lo) * (samples[hi] - samples[lo])
    return {
        "exact_table": inputs["exact_table"], "sampled_table": inputs["sampled_table"],
        "truth_effect": float(truth), "truth_effect_exact": str(truth),
        "exact_table_effect": float(eff_exact),
        "exact_table_law": {f"{o}{a}{b}": float(v) for (o, a, b), v in sorted(law_exact.items())},
        "point_effect": float(point), "jackknife": jack,
        "replicate_effects": stream["replicate_effects"],
        "bias_correction": float(z0), "acceleration": float(accel),
        "adjusted_probabilities": [float(a) for a in adj],
        "lower": float(q7(adj[0])), "upper": float(q7(adj[1])),
    }


if sys.argv[1] == "inputs":
    print(json.dumps(stage_inputs()))
else:
    print(json.dumps(stage_final(json.load(open(sys.argv[2])))))
