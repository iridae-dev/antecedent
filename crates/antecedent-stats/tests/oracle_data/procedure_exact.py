"""Exact definition-level law of a declared adaptive procedure, by path enumeration.

Library for the procedure references (procedure_law_reference.py and its siblings).  Repository-
authored (MIT OR Apache-2.0); standard library only (fractions, decimal).

The law is built from the generative description, not from the product's merged-state recursion:
every path of (assigned arm, delivered arm, outcome, retained) for every unit is expanded in order
with exact rational probabilities.  A unit enrolled at look k is observed once look k + lag has
passed; the policy at look k reads only the retained outcomes of batches 0 .. k - 1 - lag; the
stopping rule is read after the batch of look k is added to the observed counts of batches
0 .. k - lag; the final selection applies when the rule names no decision.  Posterior quantities
use Beta posteriors with integer shapes, evaluated exactly:

  P(Y > X), X ~ Beta(a, b), Y ~ Beta(c, d), n = c + d - 1:
      sum_{j=0}^{c-1} C(n, j) B(a + j, b + n - j) / B(a, b)
  (Beta(c, d) tail as a binomial cdf, then the Beta integral of each monomial), cross-checked
  against the Miller finite sum in `check_exceedance`.
"""
from decimal import Decimal, getcontext
from fractions import Fraction as F
from math import comb, factorial

getcontext().prec = 60


def frac(text):
    return F(text)


def beta_fn(x, y):
    return F(factorial(x - 1) * factorial(y - 1), factorial(x + y - 1))


_exceed_cache = {}


def exceeds(a, b, c, d):
    """P(Y > X) for X ~ Beta(a, b), Y ~ Beta(c, d), integer shapes."""
    key = (a, b, c, d)
    if key not in _exceed_cache:
        n = c + d - 1
        _exceed_cache[key] = sum(
            (comb(n, j) * beta_fn(a + j, b + n - j) / beta_fn(a, b) for j in range(c)), F(0)
        )
    return _exceed_cache[key]


def check_exceedance():
    """The Miller finite sum agrees with `exceeds` (an independent closed form)."""
    for (a, b, c, d) in [(1, 1, 1, 1), (2, 3, 4, 1), (5, 2, 3, 6), (1, 4, 7, 2)]:
        miller = sum(
            (beta_fn(a + i, b + d) / (F(d + i) * beta_fn(1 + i, d) * beta_fn(a, b)) for i in range(c)),
            F(0),
        )
        assert miller == exceeds(a, b, c, d), (a, b, c, d)


def mean_beta(a, b):
    return F(a, a + b)


def positive_part_expectation(a, b, c, d):
    """E[(Y - X)+] for X ~ Beta(a, b), Y ~ Beta(c, d)."""
    # E[Y; Y > X] = mean(Y) P(Y' > X), Y' ~ Beta(c + 1, d); E[X; Y > X] = mean(X) P(Y > X'),
    # X' ~ Beta(a + 1, b).
    return mean_beta(c, d) * exceeds(a, b, c + 1, d) - mean_beta(a, b) * exceeds(a + 1, b, c, d)


class Margin:
    """Smallest distance of a threshold comparison, to certify no float/exact flip."""

    smallest = F(1)

    @classmethod
    def note(cls, value, threshold):
        gap = abs(F(value) - F(threshold))
        if gap < cls.smallest:
            cls.smallest = gap


def posterior(prior, counts, arm):
    return prior[0] + counts[2 * arm], prior[1] + counts[2 * arm + 1]


def best_probabilities(prior, counts):
    # two arms only
    a, b = posterior(prior, counts, 0)
    c, d = posterior(prior, counts, 1)
    p1 = exceeds(a, b, c, d)
    return [1 - p1, p1]


def losses(prior, counts):
    a, b = posterior(prior, counts, 0)
    c, d = posterior(prior, counts, 1)
    return [positive_part_expectation(a, b, c, d), positive_part_expectation(c, d, a, b)]


def first_max(values):
    best = 0
    for i, v in enumerate(values):
        if i and v > values[best]:
            best = i
    return best


def policy_row(decl, counts, arms):
    kind = decl["assignment"]["kind"]
    spec = decl["assignment"]
    if kind == "fixed":
        return [frac(p) for p in spec["probabilities"]]
    if kind == "epsilon_greedy":
        eps = frac(spec["epsilon"])
        pa, pb = spec["prior"]
        num = lambda a: pa + counts[2 * a]
        den = lambda a: pa + pb + counts[2 * a] + counts[2 * a + 1]
        greedy = [0]
        for a in range(1, arms):
            b = greedy[0]
            lhs, rhs = num(a) * den(b), num(b) * den(a)
            if lhs > rhs:
                greedy = [a]
            elif lhs == rhs:
                greedy.append(a)
        row = [eps / arms] * arms
        for a in greedy:
            row[a] += (1 - eps) / len(greedy)
        return row
    if kind in ("thompson", "thompson_floored"):
        assert arms == 2
        row = best_probabilities(spec["prior"], counts)
        if kind == "thompson_floored":
            floor = frac(spec["floor"])
            for p in row:
                Margin.note(p, floor)
            if row[0] < floor:
                row = [floor, 1 - floor]
            elif row[1] < floor:
                row = [1 - floor, floor]
        return row
    if kind == "ucb":
        units = [counts[2 * a] + counts[2 * a + 1] for a in range(arms)]
        row = [F(0)] * arms
        for a in range(arms):
            if units[a] == 0:
                row[a] = F(1)
                return row
        total = Decimal(sum(units))
        c = Decimal(spec["exploration"])
        scores = [
            Decimal(counts[2 * a]) / Decimal(units[a]) + (c * total.ln() / Decimal(units[a])).sqrt()
            for a in range(arms)
        ]
        best = 0
        for a in range(1, arms):
            if scores[a] > scores[best]:
                best = a
        row[best] = F(1)
        return row
    raise ValueError(kind)


def assess(decl, look, enrolled, observed, arms):
    stop = decl["stopping"]
    kind = stop["kind"]
    if kind == "none":
        return False, None
    if kind == "cap":
        return enrolled >= stop["max_units"], None
    if kind == "best":
        best = best_probabilities(stop["prior"], observed)
        arm = first_max(best)
        Margin.note(best[arm], frac(stop["threshold"]))
        return (best[arm] >= frac(stop["threshold"]), arm) if best[arm] >= frac(stop["threshold"]) else (False, None)
    if kind == "loss":
        ls = losses(stop["prior"], observed)
        arm = first_max([-x for x in ls])
        Margin.note(ls[arm], frac(stop["threshold"]))
        return (True, arm) if ls[arm] < frac(stop["threshold"]) else (False, None)
    if kind == "futility":
        assert stop["margin"] == "0"
        a, b = posterior(stop["prior"], observed, 0)
        c, d = posterior(stop["prior"], observed, 1)
        p = exceeds(a, b, c, d)
        Margin.note(p, frac(stop["threshold"]))
        return (True, 0) if p < frac(stop["threshold"]) else (False, None)
    if kind == "wald":
        s0, f0, s1, f1 = observed
        n0, n1 = s0 + f0, s1 + f1
        if n0 == 0 or n1 == 0:
            return False, None
        p0, p1 = F(s0, n0), F(s1, n1)
        var = p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1
        if var <= 0:
            return False, None
        z2 = (p1 - p0) ** 2 / var  # z squared exactly; sign from p1 - p0
        crit = Decimal(stop["critical"][look])
        z = Decimal(z2.numerator) / Decimal(z2.denominator)
        z = z.sqrt() if p1 >= p0 else -z.sqrt()
        side = stop["sidedness"]
        if side == "upper":
            Margin.note(F(z), F(crit))
            return (True, 1) if z >= crit else (False, None)
        Margin.note(F(abs(z)), F(crit))
        return (True, 1 if z > 0 else 0) if abs(z) >= crit else (False, None)
    raise ValueError(kind)


def select(decl, observed, arms):
    sel = decl["selection"]
    if sel["kind"] == "none":
        return None
    if sel["kind"] == "mean":
        pa, pb, need = 0, 0, True
    else:
        pa, pb = sel["prior"]
        need = False
    num = lambda a: pa + observed[2 * a]
    den = lambda a: pa + pb + observed[2 * a] + observed[2 * a + 1]
    best = None
    for a in range(arms):
        if need and den(a) == 0:
            continue
        if best is None:
            best = a
        elif num(a) * den(best) > num(best) * den(a):
            best = a
    return best


def counts_of(units, upto_batch, arms):
    c = [0] * (2 * arms)
    for (batch, attributed, success, retained) in units:
        if batch <= upto_batch and retained:
            c[2 * attributed + (0 if success else 1)] += 1
    return c


def enumerate_paths(decl, kernel, visit):
    """Call visit(look, stopped, decision, units, probability) for every terminal path.

    A unit is (batch, attributed arm, success, retained, delivered arm, effective row), the
    effective row being the probability of each attributed arm at the unit's enrolment look.
    """
    arms = decl["arms"]
    lag = decl.get("lag", 0)
    retain_s, retain_f = [frac(x) for x in decl.get("retain", ["1", "1"])]
    exposure = decl.get("exposure", {})
    failure = [frac(x) for x in exposure.get("failure", ["0"] * arms)]
    fallback = exposure.get("fallback", 0)
    attribution = exposure.get("attribution", "delivered")
    enrol = decl["enrolment"]
    cumulative = [sum(enrol[: k + 1]) for k in range(len(enrol))]

    def deliveries(a):
        if failure[a] > 0 and fallback != a:
            return [(a, 1 - failure[a]), (fallback, failure[a])]
        return [(a, F(1))]

    def success_probability(delivered, history):
        if kernel[0] == "fixed":
            return frac(kernel[1][delivered])
        a, b = kernel[1][delivered]
        s = sum(1 for h in history if h[4] == delivered and h[2])
        f = sum(1 for h in history if h[4] == delivered and not h[2])
        return F(a + s, a + b + s + f)

    def run_batch(look, remaining, row, effective, units, prob, then):
        if remaining == 0:
            then(units, prob)
            return
        for a in range(arms):
            if row[a] == 0:
                continue
            for d, pd in deliveries(a):
                if pd == 0:
                    continue
                ps = success_probability(d, units)
                for success, p_out, keep in ((True, ps, retain_s), (False, 1 - ps, retain_f)):
                    if p_out == 0:
                        continue
                    for retained, p_keep in ((True, keep), (False, 1 - keep)):
                        if p_keep == 0:
                            continue
                        attributed = a if attribution == "assigned" else d
                        unit = (look, attributed, success, retained, d, effective)
                        run_batch(
                            look, remaining - 1, row, effective, units + [unit],
                            prob * row[a] * pd * p_out * p_keep, then,
                        )

    def go(look, units, prob):
        before = counts_of([u[:4] for u in units], look - 1 - lag, arms)
        row = policy_row(decl, before, arms)
        if attribution == "assigned":
            effective = tuple(row)
        else:
            eff = [F(0)] * arms
            for a in range(arms):
                for d, pd in deliveries(a):
                    eff[d] += row[a] * pd
            effective = tuple(eff)

        def after(units2, prob2):
            observed = counts_of([u[:4] for u in units2], look - lag, arms)
            stop, decision = assess(decl, look, cumulative[look], observed, arms)
            last = look + 1 == len(enrol)
            if stop or last:
                dec = decision if stop and decision is not None else select(decl, observed, arms)
                visit(look, stop, dec, [u for u in units2 if u[0] <= look - lag and u[3]], prob2)
            else:
                go(look + 1, units2, prob2)

        run_batch(look, enrol[look], row, effective, units, prob, after)

    go(0, [], F(1))


def enumerate_law(decl, kernel):
    """Atoms {(look, stopped, observed tuple, decision): Fraction}."""
    arms = decl["arms"]
    atoms = {}

    def visit(look, stop, dec, revealed, prob):
        observed = counts_of([u[:4] for u in revealed], look, arms)
        key = (look, stop, tuple(observed), dec)
        atoms[key] = atoms.get(key, F(0)) + prob

    enumerate_paths(decl, kernel, visit)
    assert sum(atoms.values()) == 1
    return atoms


def atoms_json(atoms):
    out = []
    for (look, stopped, observed, decision), p in sorted(
        atoms.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], -1 if kv[0][3] is None else kv[0][3])
    ):
        out.append(
            {
                "look": look,
                "stopped": stopped,
                "observed": list(observed),
                "decision": decision,
                "p": f"{p.numerator}/{p.denominator}",
                "pf": float(p),
            }
        )
    return out
