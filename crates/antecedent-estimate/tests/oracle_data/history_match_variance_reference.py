# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Variance of the period-matched history contrast, by exhaustive noise enumeration, exactly.

Regenerate with:  uv run --script history_match_variance_reference.py > history_match_variance_reference.json

Each design fixes scores, arms, periods and sampling weights.  Every treated unit is matched to the
nearest control of its own period (lowest input index on a tie, donors reused, no caliper).  Given
the matches the contrast sum_q s_q (y_q - y_donor(q)), s_q = weight_q / total, has conditional
variance  sum_q s_q^2 sigma_t^2 + sum_j c_j^2 sigma_c^2  when the outcome noise of every treated
(control) unit has the same variance sigma_t^2 (sigma_c^2) and the means are constant within an arm
and period, c_j being the share the queries of donor j carry.  The Abadie-Imbens estimator replaces
each unit's variance by (1/2)(y_i - y_l(i))^2, l(i) the nearest other unit of the same arm and
period (lowest input index on a tie, brute-force search here), so its expectation over the noise
equals that variance exactly; the script asserts this over all 2^n sign patterns of the noise
(each unit is mean +- scale) and writes the standard error of every pattern.  Everything is
Fraction arithmetic on dyadic inputs; the standard error is the nearest binary64 to the square
root.  Repository-authored (MIT OR Apache-2.0).
"""
import itertools
import json
import math
from fractions import Fraction as F


def design(name, units, sigma_t, sigma_c):
    """units: list of dict(score, treated, period, weight, mean)."""
    n = len(units)
    # matches
    pairs = []
    for q, u in enumerate(units):
        if not u["treated"]:
            continue
        cands = [(abs(F(u["score"]) - F(v["score"])), j) for j, v in enumerate(units)
                 if not v["treated"] and v["period"] == u["period"]]
        if cands:
            pairs.append((q, min(cands)[1]))
    total = sum(F(units[q]["weight"]) for q, _ in pairs)
    share = {q: F(units[q]["weight"]) / total for q, _ in pairs}
    donated = {}
    for q, j in pairs:
        donated[j] = donated.get(j, F(0)) + share[q]
    true_var = sum(s * s for s in share.values()) * F(sigma_t) ** 2 + \
        sum(c * c for c in donated.values()) * F(sigma_c) ** 2

    def neighbour(i):
        u = units[i]
        c = [(abs(F(u["score"]) - F(v["score"])), j) for j, v in enumerate(units)
             if j != i and v["treated"] == u["treated"] and v["period"] == u["period"]]
        return min(c)[1] if c else None

    nb = {i: neighbour(i) for i in range(n)}
    needed = [q for q, _ in pairs] + list(donated)
    reals = []
    expected = F(0)
    for signs in itertools.product([-1, 1], repeat=n):
        y = [F(units[i]["mean"]) + signs[i] * F(sigma_t if units[i]["treated"] else sigma_c)
             for i in range(n)]
        if any(nb[i] is None for i in needed):
            var = None
        else:
            var = sum(share[q] ** 2 * (y[q] - y[nb[q]]) ** 2 / 2 for q, _ in pairs) + \
                sum(c * c * (y[j] - y[nb[j]]) ** 2 / 2 for j, c in donated.items())
            expected += var
        contrast = sum(share[q] * (y[q] - y[j]) for q, j in pairs)
        reals.append(dict(outcomes=[float(v) for v in y], contrast=float(contrast),
                          standard_error=None if var is None else math.sqrt(float(var))))
    if all(nb[i] is not None for i in needed):
        assert expected / 2 ** n == true_var, (name, expected / 2 ** n, true_var)
    return dict(name=name, units=units, true_variance=float(true_var), realizations=reals)


def u(score, treated, period, weight, mean):
    return dict(score=score, treated=treated, period=period, weight=weight, mean=mean)


def main():
    d = []
    # one period, three treated sharing one nearby control, unequal weights, a second control far away
    d.append(design("shared_donor_one_period", [
        u(1.0, True, 0, 1.0, 5.0), u(1.25, True, 0, 2.0, 5.0), u(0.75, True, 0, 0.5, 5.0),
        u(1.0, False, 0, 1.0, 2.0), u(8.0, False, 0, 1.0, 2.0),
        u(1.5, True, 0, 1.5, 5.0), u(7.5, False, 0, 1.0, 2.0), u(2.0, True, 0, 1.0, 5.0)],
        0.5, 0.75))
    # two periods, equal scores across arms, a tie in distance, a lone-control period
    d.append(design("two_periods_with_ties", [
        u(0.0, True, 1, 1.0, 3.0), u(1.0, False, 1, 1.0, 1.0), u(-1.0, False, 1, 1.0, 1.0),
        u(0.5, True, 1, 2.0, 3.0), u(2.0, False, 1, 1.0, 1.0),
        u(4.0, True, 2, 1.0, 3.0), u(4.0, True, 2, 3.0, 3.0), u(5.0, False, 2, 1.0, 1.0),
        u(3.0, False, 2, 1.0, 1.0)],
        0.25, 0.5))
    # a period whose only control has no same-arm neighbour: no standard error
    d.append(design("lone_control_has_no_neighbour", [
        u(0.0, True, 1, 1.0, 3.0), u(1.0, True, 1, 1.0, 3.0), u(0.5, False, 1, 1.0, 1.0)],
        0.5, 0.5))
    json.dump(d, fp=__import__("sys").stdout, separators=(",", ":"))


main()
