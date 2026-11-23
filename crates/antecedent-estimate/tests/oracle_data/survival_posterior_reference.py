# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact moments of the piecewise-exponential survival posterior with gamma hazard priors.

Regenerate with:  uv run --script survival_posterior_reference.py > survival_posterior_reference.json

An arm's hazard on interval k = (c_{k-1}, c_k] is Gamma(a0 + d_k, r0 + e_k), where d_k counts events with
c_{k-1} < t <= c_k and e_k = sum_i max(0, min(t_i, c_k) - max(entry_i, c_{k-1})) is the exact time at risk.
With independent intervals and l_k = c_k - c_{k-1}:
  S = prod_k exp(-h_k l_k),  E[exp(-q h l)] = (r / (r + q l))^a,
  RMST = sum_k A_k prod_{j<k} exp(-h_j l_j), A_k = (1 - exp(-h_k l_k)) / h_k,
  E[A_k] = integral_0^l (r / (r + s))^a ds,  E[A_k exp(-h_k l_k)] = integral_0^l (r / (r + s + l))^a ds,
so the survival and RMST means and variances follow from products of these moments; the two arms
are independent, hence the variances of the treated-minus-control contrasts add.  The integrals
and products are evaluated in mpmath at 40 digits from the exact binary64 inputs.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import mpmath as mp

mp.mp.dps = 40


def posterior(arm_rows, cuts, prior, entries):
    out = []
    lower = mp.mpf(0)
    for upper in cuts:
        upper = mp.mpf(upper)
        d, e = 0, mp.mpf(0)
        for (t, ev), en in zip(arm_rows, entries):
            t = mp.mpf(t)
            e += max(mp.mpf(0), min(t, upper) - max(mp.mpf(en), lower))
            if ev == 1 and lower < t <= upper:
                d += 1
        out.append(dict(events=d, exposure=e, a=mp.mpf(prior[0]) + d, r=mp.mpf(prior[1]) + e, l=upper - lower))
        lower = upper
    return out


def stay(p, q):
    return (p["r"] / (p["r"] + q * p["l"])) ** p["a"]


def area(p, shift):
    return mp.quad(lambda s: (p["r"] / (p["r"] + s + shift)) ** p["a"], [0, p["l"]])


def moments(arm):
    S1 = mp.mpf(1)
    S2 = mp.mpf(1)
    for p in arm:
        S1 *= stay(p, 1)
        S2 *= stay(p, 2)
    # RMST = sum_k A_k P_k;  E and E[square] by the product structure.
    K = len(arm)
    EA = [area(p, 0) for p in arm]
    EAe = [area(p, p["l"]) for p in arm]
    EAA = []
    for p in arm:
        # E[A^2] = 2 integral_0^l integral_0^s E[exp(-h (u... )] : A = integral_0^l exp(-h s) ds so
        # A^2 = 2 integral_{0<s<u<l} exp(-h (s + u)) and E = 2 integral_0^l integral_0^u (r/(r+s+u))^a ds du.
        EAA.append(2 * mp.quad(lambda u: mp.quad(lambda s: (p["r"] / (p["r"] + s + u)) ** p["a"], [0, u]), [0, p["l"]]))
    m1 = mp.mpf(0)
    carried = mp.mpf(1)
    for k in range(K):
        m1 += carried * EA[k]
        carried *= stay(arm[k], 1)
    m2 = mp.mpf(0)
    for k in range(K):
        pre2 = mp.mpf(1)
        for j in range(k):
            pre2 *= stay(arm[j], 2)
        m2 += pre2 * EAA[k]
        for m in range(k + 1, K):
            mid = mp.mpf(1)
            for j in range(k + 1, m):
                mid *= stay(arm[j], 1)
            # prod_{j<k} E exp(-2 h l) * E[A_k exp(-h_k l_k)] * prod_{k<j<m} E exp(-h l) * E[A_m]
            m2 += 2 * pre2 * EAe[k] * mid * EA[m]
    return S1, S2 - S1 * S1, m1, m2 - m1 * m1


def case(name, duration, event, treated, cuts, prior, entry=None):
    entries = entry if entry is not None else [0.0] * len(duration)
    arms = {}
    for flag in (False, True):
        idx = [i for i in range(len(duration)) if treated[i] == flag]
        arms[flag] = posterior([(duration[i], event[i]) for i in idx], cuts, prior, [entries[i] for i in idx])
    sc, vc, rc, wc = moments(arms[False])
    st, vt, rt, wt = moments(arms[True])
    return dict(
        name=name, duration=duration, event=event, treated=treated, entry=entry, cuts=cuts, prior=list(prior),
        control=[dict(events=p["events"], exposure=float(p["exposure"]), shape=float(p["a"]), rate=float(p["r"])) for p in arms[False]],
        treated_arm=[dict(events=p["events"], exposure=float(p["exposure"]), shape=float(p["a"]), rate=float(p["r"])) for p in arms[True]],
        control_survival=float(sc), treated_survival=float(st), survival_difference=float(st - sc),
        survival_difference_sd=float(mp.sqrt(vc + vt)),
        control_rmst=float(rc), treated_rmst=float(rt), rmst_difference=float(rt - rc),
        rmst_difference_sd=float(mp.sqrt(wc + wt)),
    )


d0 = [0.5, 1.2, 2.5, 3.5, 1.8, 0.9, 0.7, 1.5, 2.2, 3.2, 3.8, 1.1]
e0 = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
t0 = [False] * 6 + [True] * 6
d1 = [0.8, 1.6, 2.9, 3.4, 1.1, 2.1, 0.6, 1.9, 2.7, 3.1, 1.4, 2.3, 3.6, 0.9]
e1 = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1]
t1 = [False] * 7 + [True] * 7
en1 = [0.0, 0.3, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.7, 0.0, 0.2, 0.0, 1.25, 0.0]
cases = [
    case("flat_prior", d0, e0, t0, [1.0, 2.0, 3.0], (1.0, 1.0)),
    case("informative_prior", d0, e0, t0, [1.0, 2.0, 3.0], (4.5, 6.0)),
    case("delayed_entry_four_intervals", d1, e1, t1, [0.75, 1.5, 2.25, 3.5], (2.0, 3.0), en1),
]
print(json.dumps(cases, separators=(",", ":")))
