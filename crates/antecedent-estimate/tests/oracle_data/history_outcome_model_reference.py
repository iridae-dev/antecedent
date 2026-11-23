# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact posterior of the period-adjusted history outcome-model effect on the treated.

Regenerate with:  uv run --script history_outcome_model_reference.py > history_outcome_model_reference.json

Units carry a history, an outcome, an arm and a calendar period.  Only periods holding both arms
are kept (a treated unit in any other period is reported as unused).  With score s_i the mean of
the unit's history and s_T the mean score of the kept treated units, the design columns are
(1, t, s, d_2, ..., d_P, t (s - s_T)) for the period indicators d_p of every kept period after the
first.  Under b | s2 ~ N(0, s2 c0^2 I), s2 ~ Inv-Gamma(1e-3, 1e-3) the marginal of the treatment
coefficient (the model-based effect on the treated, since the interaction is centred at the
treated mean score) is Student-t with mean mn[1] and variance bn / (an - 1) Vn[1,1], where
Vn = (I / c0^2 + X'X)^-1, mn = Vn X'y, an = 1e-3 + n / 2 and
bn = 1e-3 + (y'y - mn' Vn^-1 mn) / 2.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 40


def units():
    out = []
    for i in range(72):
        period = (10, 20, 30)[i % 3]
        treated = ((i * 5 + 1) % 7) < 3
        history = [(((i * (3 + j) + 2 * j) % 11) - 5) / 5.0 + 0.4 * (i % 3) for j in range(3)]
        score = sum(history) / len(history)
        outcome = (1.0 + 0.3 * (period // 10) + 1.1 * score + (1.6 + 0.5 * score if treated else 0.0)
                   + 0.3 * ((((i * 13 + 4) % 17) - 8) / 8.0))
        out.append(dict(history=history, outcome=outcome, treated=treated, period=period))
    # period 40 holds treated units only: reported, never fitted.
    for i in range(3):
        history = [0.1 * i, 0.2, 0.3 + 0.1 * i]
        out.append(dict(history=history, outcome=2.0 + i, treated=True, period=40))
    return out


def reference(us, c0):
    arms = {}
    for u in us:
        arms.setdefault(u["period"], set()).add(u["treated"])
    kept = sorted(p for p, a in arms.items() if len(a) == 2)
    rows = [u for u in us if u["period"] in kept]
    unused = [i for i, u in enumerate(us) if u["period"] not in kept and u["treated"]]
    score = lambda u: sum(u["history"]) / len(u["history"])
    treated = [u for u in rows if u["treated"]]
    s_t = sum(mp.mpf(score(u)) for u in treated) / len(treated)
    X = mp.matrix([[1, 1 if u["treated"] else 0, mp.mpf(score(u)),
                    *[1 if u["period"] == p else 0 for p in kept[1:]],
                    (mp.mpf(score(u)) - s_t) if u["treated"] else 0] for u in rows])
    Y = mp.matrix([mp.mpf(u["outcome"]) for u in rows])
    p = X.cols
    prec = X.T * X + mp.eye(p) / mp.mpf(c0) ** 2
    vn = prec ** -1
    mn = vn * (X.T * Y)
    an = mp.mpf("0.001") + mp.mpf(len(rows)) / 2
    bn = mp.mpf("0.001") + ((Y.T * Y)[0] - (mn.T * prec * mn)[0]) / 2
    return dict(rows=len(rows), treated_rows=len(treated), periods=len(kept), unused_treated=unused,
                mean=float(mn[1]), sd=float(mp.sqrt(bn / (an - 1) * vn[1, 1])))


us = units()
print(json.dumps(dict(prior_scale=10.0, units=us, **reference(us, 10.0)), separators=(",", ":")))
