# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact moments of the identified-atom mixture envelope, in exact rational arithmetic.

Regenerate with:  uv run --script envelope_reference.py > envelope_reference.json

Atoms i carry a weight w_i, an identification flag and a graph key; atoms sharing a key pool their
weight (so repeated samples of one graph count as one component) and atoms with zero weight drop
out.  With W_id the total identified weight, the component of key k has weight w_k / W_id and the
supplied draws x_k1..x_kn (n draws in every component).  The published moments are those of the
mixture of the empirical component laws:
    mean = sum_k (w_k / W_id) mean_k,
    var  = sum_k (w_k / W_id) [ (1/n) sum_j (x_kj - mean)^2 ]      (population divisor n),
which contains the between-component disagreement through (x - mean).  The retained unidentified
mass is  W_unid / W_total,  never redistributed.  Every quantity is computed with Fractions from
the exact binary values of the draws; sqrt is taken only for the final standard deviation, in
decimal with 50 digits.  Repository-authored (MIT OR Apache-2.0).
"""
import json
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 50


def draws(key, n, centre, spread):
    return [centre + spread * ((((j * (3 + key) + key) % 17) - 8) / 8.0) for j in range(n)]


def case(name, atoms, n):
    # atoms: (key, weight, identified)
    pooled = {}
    unidentified = Fraction(0)
    total = Fraction(0)
    for key, weight, ident in atoms:
        w = Fraction(weight)
        total += w
        if not ident:
            unidentified += w
        elif w != 0:
            pooled[key] = pooled.get(key, Fraction(0)) + w
    w_id = sum(pooled.values())
    comp = {k: draws(k, n, 0.5 * k, 0.2 + 0.1 * k) for k in pooled}
    mean = sum(pooled[k] / w_id * sum(Fraction(x) for x in comp[k]) / n for k in pooled)
    var = sum(pooled[k] / w_id * sum((Fraction(x) - mean) ** 2 for x in comp[k]) / n for k in pooled)
    sd = (Decimal(var.numerator) / Decimal(var.denominator)).sqrt()
    keys = sorted({a[0] for a in atoms})
    return dict(name=name, atoms=[dict(key=k, weight=w, identified=i) for k, w, i in atoms],
                draws={str(k): draws(k, n, 0.5 * k, 0.2 + 0.1 * k) for k in keys},
                mean=float(mean), sd=float(sd), unidentified_mass=float(unidentified / total))


print(json.dumps([
    case("three_components_with_unidentified_mass", [(1, 0.25, True), (2, 0.5, True), (3, 0.125, True), (4, 0.125, False)], 64),
    case("repeated_keys_and_zero_weight", [(1, 0.5, True), (2, 0.25, True), (1, 0.125, True), (3, 0.0, True), (2, 0.125, True)], 40),
], separators=(",", ":")))
