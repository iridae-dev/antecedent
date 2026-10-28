# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Independent references for the numerics the umbrella cyclic-systems record owns on the estimation
side: projected Adam on the feedback likelihood and the Gaussian (Euler-type) quasi-likelihood of a
diffusion, direct and through an extended Kalman filter.

    uv run --offline --script cyclic_general_systems_reference.py > cyclic_general_systems_reference.json

Nothing here reads product code and no paper was retrieved.

adam   The feedback likelihood of a clamped-regressor equation a = c + act(s a + w b) + u, u ~ N(0, v),
       whose per-row log density is  -ln(2 pi v)/2 - u^2/(2 v) + ln|1 - s| (the Jacobian term exists
       only when the self coefficient s is learned), is written out with its analytic scores.  The
       Adam recurrence (beta 0.9 / 0.999, bias corrections 1 - beta^t, step lr m_hat / (sqrt(v_hat) +
       1e-8), ascent direction), the proximal L1 shrinkage of incoming weights, the row-sum
       contraction projection, the validation-patience rule, the best-validation model and the KKT
       stationarity score are replayed step by step at 60 digits for full-batch epochs (one gradient
       step per epoch, so the library's shuffled batch order cannot matter).
ql     Per transition the Euler Gaussian density N(x1; x0 + f(x0) dt, g(x0)^2 dt) summed over a path;
       for an Ornstein-Uhlenbeck process it is compared with the exact transition density
       N(mu + (x0 - mu) e^{-theta dt}, sigma^2 (1 - e^{-2 theta dt}) / (2 theta)) on nested
       subsamplings of one exact path.  The extended Kalman likelihood of a linear drift is the
       ordinary Kalman innovation likelihood of the Euler-discretised model, replayed with substeps.
"""
import json
import math
import random
import sys

import mpmath as mp

mp.mp.dps = 60
EPS = mp.mpf(2) ** -52


def num(x):
    return float(mp.nstr(x, 30))


def sign(x):
    return mp.mpf(1) if x >= 0 else mp.mpf(-1)


# ---------------------------------------------------------------- Adam

def row_terms(model, params, row):
    """Per-row log density and score of the free equation 0 for clamp on variable 1."""
    eq = model["equations"][0]
    v = mp.mpf(model["noise_covariance"][0])
    a, b = mp.mpf(row[0]), mp.mpf(row[1])
    c, s, w = (mp.mpf(params["intercept"]), mp.mpf(params["self"]), mp.mpf(params["weight"]))
    if eq["activation"] == "identity":
        u = a - c - s * a - w * b
        du_dc, du_dw, du_ds = -1, -b, -a
    else:
        z = s * a + w * b
        u = a - c - mp.tanh(z)
        sech2 = 1 - mp.tanh(z) ** 2
        du_dc, du_dw, du_ds = -1, -b * sech2, -a * sech2
    jac = 1 - s if eq["activation"] == "identity" else mp.mpf(1) - s * (1 - mp.tanh(s * a + w * b) ** 2)
    ll = -mp.log(2 * mp.pi * v) / 2 - u * u / (2 * v) + mp.log(abs(jac))
    score = {
        "intercept": -u * du_dc / v,
        "weight": -u * du_dw / v,
        "self": -u * du_ds / v + (-1 / (1 - s) if eq["activation"] == "identity" else 0),
    }
    return ll, score


def adam_case(spec):
    model = spec["model"]
    eq = model["equations"][0]
    assert eq["activation"] == "identity" or spec["learn"]["self"] is False
    p = {"intercept": mp.mpf(eq["intercept"]), "self": mp.mpf(eq["weights"][0]), "weight": mp.mpf(eq["weights"][1])}
    learn = spec["learn"]
    names = [k for k in ("self", "weight", "intercept") if learn[k]]
    pol = spec["policy"]
    lr, l1, margin = mp.mpf(pol["learning_rate"]), mp.mpf(pol["l1_penalty"]), mp.mpf(pol["contraction_margin"])
    train, valid = spec["train"], spec["validation"]

    def total(rows):
        ll, sc = mp.mpf(0), {k: mp.mpf(0) for k in p}
        for r in rows:
            a, b = row_terms(model, p, r)
            ll += a
            for k in sc:
                sc[k] += b[k]
        return ll, sc

    def lipschitz_sum(q):
        return abs(q["self"]) + abs(q["weight"])

    fixed_names = [k for k in ("self", "weight") if not learn[k]]
    m = {k: mp.mpf(0) for k in names}
    v2 = {k: mp.mpf(0) for k in names}
    best_ll, _ = total(valid)
    best_params, best_epoch = dict(p), 0
    reference = best_ll / len(valid)
    stale, steps, stopped = 0, 0, False
    trace = []
    for epoch in range(1, pol["max_epochs"] + 1):
        _, sc = total(train)
        steps += 1
        c1 = 1 - mp.mpf("0.9") ** steps
        c2 = 1 - mp.mpf("0.999") ** steps
        proposed = dict(p)
        for k in names:
            g = sc[k] / len(train)
            m[k] = mp.mpf("0.9") * m[k] + mp.mpf("0.1") * g
            v2[k] = mp.mpf("0.999") * v2[k] + mp.mpf("0.001") * g * g
            scale = mp.sqrt(v2[k] / c2) + mp.mpf("1e-8")
            nxt = p[k] + lr * (m[k] / c1) / scale
            if k in ("self", "weight"):
                nxt = sign(nxt) * max(abs(nxt) - lr * l1 / scale, 0)
            proposed[k] = nxt
        # contraction projection on the learned incoming weights of the free equation
        fixed = sum((abs(proposed[k]) for k in fixed_names), mp.mpf(0))
        learned = sum(abs(proposed[k]) for k in ("self", "weight") if learn[k])
        assert fixed <= margin
        if fixed + learned > margin and learned > 0:
            factor = max((margin - fixed) * (1 - 8 * EPS) / learned, 0)
            for k in ("self", "weight"):
                if learn[k]:
                    proposed[k] *= factor
        assert lipschitz_sum(proposed) <= margin, "projection must license the proposal"
        p = proposed
        ll, _ = total(valid)
        per_row = ll / len(valid)
        if ll > best_ll:
            best_ll, best_params, best_epoch = ll, dict(p), epoch
        if per_row > reference + mp.mpf(pol["relative_tolerance"]) * max(abs(reference), 1):
            reference, stale = per_row, 0
        else:
            stale += 1
        trace.append(num(per_row))
        if epoch >= pol["minimum_epochs"] and stale >= pol["patience"]:
            stopped = True
            break
    # the training score is taken at the best model, not the last
    p_last, p = p, best_params
    ll_train, sc = total(train)
    maximum = mp.mpf(0)
    for k in names:
        raw = sc[k] / len(train)
        if k in ("self", "weight"):
            value = best_params[k]
            score = raw - l1 if value > 0 else (raw + l1 if value < 0 else sign(raw) * max(abs(raw) - l1, 0))
        else:
            score = raw
        maximum = max(maximum, abs(score))
    converged = maximum <= mp.mpf(pol["score_tolerance"])
    reason = "validation_patience" if stopped else ("training_score" if converged else "epoch_budget")
    return {
        "name": spec["name"], "note": spec["note"], "model": model,
        "learned": [{"kind": "intercept", "equation": 0}] * learn["intercept"]
        + [{"kind": "weight", "equation": 0, "parent": 0}] * learn["self"]
        + [{"kind": "weight", "equation": 0, "parent": 1}] * learn["weight"],
        "policy": pol, "train": train, "validation": valid,
        "expected": {
            "trace": trace, "best_epoch": best_epoch, "epochs": len(trace), "gradient_steps": steps,
            "intercept": num(best_params["intercept"]), "self": num(best_params["self"]), "weight": num(best_params["weight"]),
            "training_log_likelihood": num(ll_train), "validation_log_likelihood": num(best_ll),
            "training_score_per_row": num(maximum), "validation_converged": stopped,
            "training_score_converged": bool(converged), "stopping_reason": reason,
            "final_weight_row_sum": num(abs(p_last["self"]) + abs(p_last["weight"])),
        },
    }


def make_rows(rng, n, c, s, w, sd, bscale=1.5, decimals=6):
    rows = []
    for _ in range(n):
        b = round(rng.uniform(-bscale, bscale), decimals)
        u = rng.gauss(0, sd)
        a = (c + w * b + u) / (1 - s)
        rows.append([round(a, decimals), b])
    return rows


def adam_cases():
    rng = random.Random(20261028)

    def model(c, s, w, act, v):
        return {"equations": [
            {"id": "a", "intercept": c, "weights": [s, w], "activation": act},
            {"id": "b", "intercept": 0.0, "weights": [0.0, 0.0], "activation": "identity"}],
            "noise_covariance": [v, 0.0, 0.0, 1.0]}

    def policy(**kw):
        base = {"learning_rate": 0.05, "batch_size": 64, "max_epochs": 60, "minimum_epochs": 5, "patience": 6,
                "relative_tolerance": 1e-4, "score_tolerance": 1e-3, "l1_penalty": 0.0, "contraction_margin": 0.95, "seed": 0}
        base.update(kw)
        return base

    out = []
    specs = [
        dict(name="intercept_and_slope_with_patience_stop", note="learn c and w under L1 0.02; stops on validation patience",
             model=model(0.0, 0.0, 0.1, "identity", 0.25), learn={"self": False, "weight": True, "intercept": True},
             policy=policy(l1_penalty=0.02, max_epochs=200, patience=5, minimum_epochs=10),
             train=make_rows(rng, 14, 0.4, 0.0, 0.7, 0.5), validation=make_rows(rng, 9, 0.4, 0.0, 0.7, 0.5)),
        dict(name="learned_self_feedback_hits_contraction_margin",
             note="true row sum 1.6 exceeds the margin 0.9: the projection binds and the Jacobian term ln|1-s| is in the score",
             model=model(0.0, 0.1, 0.2, "identity", 0.16), learn={"self": True, "weight": True, "intercept": True},
             policy=policy(max_epochs=80, minimum_epochs=80, patience=80, contraction_margin=0.9),
             train=make_rows(rng, 16, 0.1, 0.5, 1.1, 0.4), validation=make_rows(rng, 8, 0.1, 0.5, 1.1, 0.4)),
        dict(name="saturating_activation_reports_training_score_stop",
             note="tanh mechanism, huge score tolerance: not stopped by validation, reported as converged on the training score",
             model=model(0.0, 0.0, 0.2, "tanh", 0.09), learn={"self": False, "weight": True, "intercept": True},
             policy=policy(max_epochs=30, minimum_epochs=30, patience=30, l1_penalty=0.01, score_tolerance=10.0),
             train=[[round(math.tanh(0.9 * b) + 0.1 + rng.gauss(0, 0.3), 6), b] for b in
                    [round(rng.uniform(-2, 2), 6) for _ in range(15)]],
             validation=[[round(math.tanh(0.9 * b) + 0.1 + rng.gauss(0, 0.3), 6), b] for b in
                         [round(rng.uniform(-2, 2), 6) for _ in range(8)]]),
        dict(name="l1_shrinks_a_weak_slope_to_exactly_zero",
             note="L1 penalty above every score keeps the slope at exactly zero for every epoch; the KKT score uses the subgradient branch",
             model=model(0.4, 0.0, 0.0, "identity", 0.25), learn={"self": False, "weight": True, "intercept": False},
             policy=policy(l1_penalty=2.0, max_epochs=40, minimum_epochs=40, patience=40, score_tolerance=1e-9),
             train=make_rows(rng, 12, 0.4, 0.0, 0.0, 0.5), validation=make_rows(rng, 8, 0.4, 0.0, 0.0, 0.5)),
    ]
    for spec in specs:
        out.append(adam_case(spec))
    zero = out[-1]["expected"]
    assert zero["weight"] == 0.0, zero["weight"]
    return out


def minibatch_case():
    """Batch size 2 over 6 rows: the library shuffles with its own stream, so the reference tabulates the
    result of every possible batch partition (90 ordered ones) after epoch one and, for each, of every
    partition in epoch two.  The library trace must equal exactly one entry of each table, which pins that
    an epoch visits every row once, steps once per minibatch (the bias-correction counter keeps running
    across epochs) and averages the score over the rows of its batch."""
    import itertools
    rng = random.Random(4242)
    rows = [[round(0.5 + 0.8 * b + rng.gauss(0, 0.4), 6), b] for b in [round(rng.uniform(-1.5, 1.5), 6) for _ in range(6)]]
    valid = [[round(0.5 + 0.8 * b + rng.gauss(0, 0.4), 6), b] for b in [round(rng.uniform(-1.5, 1.5), 6) for _ in range(5)]]
    var = mp.mpf("0.25")

    def ll_rows(c, w, rs):
        total = mp.mpf(0)
        for a, b in rs:
            u = mp.mpf(a) - c - w * mp.mpf(b)
            total += -mp.log(2 * mp.pi * var) / 2 - u * u / (2 * var)
        return total

    def score(c, w, rs):
        gc = gw = mp.mpf(0)
        for a, b in rs:
            u = mp.mpf(a) - c - w * mp.mpf(b)
            gc += u / var
            gw += u * mp.mpf(b) / var
        return gc / len(rs), gw / len(rs)

    lr = mp.mpf("0.05")

    def step(state, batch):
        c, w, m, v, t = state
        t += 1
        g = score(c, w, [rows[i] for i in batch])
        c1, c2 = 1 - mp.mpf("0.9") ** t, 1 - mp.mpf("0.999") ** t
        new_p, new_m, new_v = [], [], []
        for k, p in enumerate((c, w)):
            mk = mp.mpf("0.9") * m[k] + mp.mpf("0.1") * g[k]
            vk = mp.mpf("0.999") * v[k] + mp.mpf("0.001") * g[k] ** 2
            new_p.append(p + lr * (mk / c1) / (mp.sqrt(vk / c2) + mp.mpf("1e-8")))
            new_m.append(mk)
            new_v.append(vk)
        return (new_p[0], new_p[1], new_m, new_v, t)

    partitions = []
    for first in itertools.combinations(range(6), 2):
        rest = [i for i in range(6) if i not in first]
        for second in itertools.combinations(rest, 2):
            third = tuple(i for i in rest if i not in second)
            partitions.append((first, second, third))
    assert len(partitions) == 90
    start = (mp.mpf("0.0"), mp.mpf("0.1"), [mp.mpf(0)] * 2, [mp.mpf(0)] * 2, 0)
    epoch1, epoch2 = [], []
    for part in partitions:
        st = start
        for batch in part:
            st = step(st, batch)
        epoch1.append(num(ll_rows(st[0], st[1], valid) / len(valid)))
        row = []
        for part2 in partitions:
            s2 = st
            for batch in part2:
                s2 = step(s2, batch)
            row.append(num(ll_rows(s2[0], s2[1], valid) / len(valid)))
        epoch2.append(row)
    return {
        "model": {"equations": [
            {"id": "a", "intercept": 0.0, "weights": [0.0, 0.1], "activation": "identity"},
            {"id": "b", "intercept": 0.0, "weights": [0.0, 0.0], "activation": "identity"}],
            "noise_covariance": [0.25, 0.0, 0.0, 1.0]},
        "train": rows, "validation": valid,
        "policy": {"learning_rate": 0.05, "batch_size": 2, "max_epochs": 2, "minimum_epochs": 2, "patience": 2,
                   "relative_tolerance": 1e-4, "score_tolerance": 1e-3, "l1_penalty": 0.0, "contraction_margin": 0.95, "seed": 3},
        "epoch_one": epoch1, "epoch_two": epoch2,
    }


# ---------------------------------------------------------------- quasi likelihood

def ou_euler(path, theta, mu, sigma):
    ll = mp.mpf(0)
    for (t0, x0), (t1, x1) in zip(path, path[1:]):
        dt = mp.mpf(t1) - mp.mpf(t0)
        mean = mp.mpf(x0) + mp.mpf(theta) * (mp.mpf(mu) - mp.mpf(x0)) * dt
        var = mp.mpf(sigma) ** 2 * dt
        ll += -(mp.log(2 * mp.pi * var) + (mp.mpf(x1) - mean) ** 2 / var) / 2
    return ll


def ou_exact(path, theta, mu, sigma):
    ll = mp.mpf(0)
    th, mu, sg = mp.mpf(theta), mp.mpf(mu), mp.mpf(sigma)
    for (t0, x0), (t1, x1) in zip(path, path[1:]):
        dt = mp.mpf(t1) - mp.mpf(t0)
        mean = mu + (mp.mpf(x0) - mu) * mp.exp(-th * dt)
        var = sg ** 2 * (1 - mp.exp(-2 * th * dt)) / (2 * th)
        ll += -(mp.log(2 * mp.pi * var) + (mp.mpf(x1) - mean) ** 2 / var) / 2
    return ll


def K(v):
    return {"op": "constant", "value": v}


def V(i):
    return {"op": "variable", "variable": i}


def P(i):
    return {"op": "parameter", "parameter": i}


def ADD(*t):
    return {"op": "add", "terms": list(t)}


def MUL(*t):
    return {"op": "multiply", "factors": list(t)}


def kalman(system, theta, obs_model, prior, times, ys, filter_step):
    """Extended Kalman innovation log likelihood of a linear drift with constant diffusion."""
    d = len(system["variables"])
    A = mp.matrix(d, d)
    b0 = [mp.mpf(0)] * d
    for i in range(d):
        zero = [mp.mpf(0)] * d
        val0, _ = ev(system["drift"][i], zero, theta)
        b0[i] = val0
        for j in range(d):
            e = [mp.mpf(0)] * d
            e[j] = mp.mpf(1)
            v1, _ = ev(system["drift"][i], e, theta)
            A[i, j] = v1 - val0
    g2 = [ev(system["diffusion"][i]["expression"], [mp.mpf(0)] * d, theta)[0] ** 2 for i in range(d)]
    H = mp.matrix(obs_model["matrix_rows"])
    off = mp.matrix(obs_model["offset"])
    R = mp.diag([mp.mpf(s) ** 2 for s in obs_model["noise_standard_deviation"]])
    m = H.rows
    mean = mp.matrix(prior["mean"])
    cov = mp.matrix(prior["covariance_rows"])

    def update(mean, cov, y):
        z = mp.matrix(y) - (H * mean + off)
        S = H * cov * H.T + R
        Sinv = S ** -1
        ll = -(m * mp.log(2 * mp.pi) + mp.log(mp.det(S)) + (z.T * Sinv * z)[0]) / 2
        K_ = cov * H.T * Sinv
        return mean + K_ * z, cov - K_ * H * cov, ll

    mean, cov, total = update(mean, cov, ys[0])
    for k in range(1, len(times)):
        dt = times[k] - times[k - 1]
        sub = math.ceil(dt / filter_step) if filter_step > 0 else 1
        sub = max(sub, 1)
        h = dt / sub
        for _ in range(sub):
            F = mp.eye(d) + A * mp.mpf(h)
            cov = F * cov * F.T + mp.diag([x * mp.mpf(h) for x in g2])
            mean = mean + (A * mean + mp.matrix(b0)) * mp.mpf(h)
        mean, cov, ll = update(mean, cov, ys[k])
        total += ll
    return total


def ev(expr, vs, theta):
    op = expr["op"]
    n = len(vs)
    if op == "constant":
        return mp.mpf(expr["value"]), [mp.mpf(0)] * n
    if op == "variable":
        g = [mp.mpf(0)] * n
        g[expr["variable"]] = mp.mpf(1)
        return vs[expr["variable"]], g
    if op == "parameter":
        return mp.mpf(theta[expr["parameter"]]), [mp.mpf(0)] * n
    if op == "add":
        parts = [ev(t, vs, theta) for t in expr["terms"]]
        return sum(p[0] for p in parts), [sum(p[1][i] for p in parts) for i in range(n)]
    if op == "multiply":
        parts = [ev(t, vs, theta) for t in expr["factors"]]
        value = mp.mpf(1)
        for p in parts:
            value *= p[0]
        grad = []
        for i in range(n):
            tot = mp.mpf(0)
            for k, p in enumerate(parts):
                term = p[1][i]
                for m_, q in enumerate(parts):
                    if m_ != k:
                        term *= q[0]
                tot += term
            grad.append(tot)
        return value, grad
    raise ValueError(op)


def direct_euler(system, theta, trajectories):
    """Euler Gaussian transition log density for a fully observed diagonal-noise system."""
    d = len(system["variables"])
    total = mp.mpf(0)
    for traj in trajectories:
        for a, b in zip(traj, traj[1:]):
            dt = mp.mpf(b["time"]) - mp.mpf(a["time"])
            x = [mp.mpf(v) for v in a["values"]]
            for i in range(d):
                f, _ = ev(system["drift"][i], x, theta)
                h, _ = ev(system["diffusion"][i]["expression"], x, theta)
                var = (h if system["diffusion"][i].get("square_root") else h * h) * dt
                resid = mp.mpf(b["values"][i]) - x[i] - f * dt
                total -= (mp.log(2 * mp.pi) + mp.log(var) + resid * resid / var) / 2
    return total


def obs_list(path):
    return [{"time": t, "values": [x]} for t, x in path]


def ql_cases():
    rng = random.Random(910)
    theta0, mu0, sigma0 = 1.5, 0.8, 0.6
    # one exact OU path on a dyadic grid, nested subsamplings of it
    n = 1024
    horizon = 4.0
    dt = horizon / n
    x = 0.3
    full = [(0.0, x)]
    mean_f = math.exp(-theta0 * dt)
    sd_f = sigma0 * math.sqrt((1 - math.exp(-2 * theta0 * dt)) / (2 * theta0))
    for k in range(1, n + 1):
        x = mu0 + (x - mu0) * mean_f + sd_f * rng.gauss(0, 1)
        full.append((k * dt, round(x, 9)))
    system = {
        "variables": ["x"], "inputs": [],
        "parameters": [{"name": "theta", "value": theta0}, {"name": "mu", "value": mu0}, {"name": "sigma", "value": sigma0}],
        "drift": [MUL(P(0), ADD(P(1), MUL(K(-1.0), V(0))))],
        "diffusion": [{"expression": P(2), "square_root": False}],
        "boundaries": [],
    }
    points = [[1.5, 0.8, 0.6], [1.2, 0.5, 0.7], [2.0, 1.0, 0.5]]
    levels = []
    for stride in [128, 64, 32, 16, 8, 4]:
        path = full[::stride]
        dt_level = horizon * stride / n
        def profile(f):
            return mp.findroot(lambda th: mp.diff(lambda t: f(path, t, mu0, sigma0), th), mp.mpf(theta0))

        levels.append({
            "interval": dt_level, "observations": obs_list(path),
            "theta_hat_euler": num(profile(ou_euler)), "theta_hat_exact": num(profile(ou_exact)),
            "values": [{"theta": pt, "euler": num(ou_euler(path, *pt)), "exact": num(ou_exact(path, *pt))} for pt in points],
        })
    out = {"ou": {"system": system, "levels": levels, "points": points}}

    # irregular times, two trajectories, still OU: the Euler formula only
    irregular = []
    for _ in range(2):
        t, xx, path = 0.0, rng.uniform(0.2, 1.2), []
        for _ in range(9):
            path.append((round(t, 6), round(xx, 6)))
            t += rng.choice([0.03, 0.05, 0.08, 0.11])
            xx += rng.gauss(0, 0.2)
        irregular.append(path)
    out["ou_irregular"] = {
        "trajectories": [obs_list(p) for p in irregular],
        "values": [{"theta": pt, "euler": num(sum(ou_euler(p, *pt) for p in irregular))} for pt in points],
    }

    # state-dependent square-root diffusion and a two-state system with a state-dependent drift
    cir = {
        "variables": ["x"], "inputs": [],
        "parameters": [{"name": "kappa", "value": 1.2}, {"name": "theta", "value": 1.0}, {"name": "s2", "value": 0.3}],
        "drift": [MUL(P(0), ADD(P(1), MUL(K(-1.0), V(0))))],
        "diffusion": [{"expression": MUL(P(2), V(0)), "square_root": True}],
        "boundaries": [{"lower": 0.0, "upper": None, "rule": "reflect"}],
    }
    cir_path = [(0.0, 0.9)]
    for k in range(1, 12):
        cir_path.append((round(k * 0.07, 6), round(abs(cir_path[-1][1] + rng.gauss(0, 0.15)) + 0.05, 6)))
    cir_points = [[1.2, 1.0, 0.3], [0.9, 1.2, 0.5]]
    out["square_root"] = {
        "system": cir, "trajectories": [obs_list(cir_path)], "points": cir_points,
        "values": [{"theta": pt, "euler": num(direct_euler(cir, pt, [obs_list(cir_path)]))} for pt in cir_points],
    }
    coupled = {
        "variables": ["x", "y"], "inputs": [],
        "parameters": [{"name": "a", "value": 0.5}, {"name": "b", "value": -0.4}],
        "drift": [ADD(MUL(K(-1.2), V(0)), MUL(P(0), V(1))), ADD(MUL(P(1), V(0)), MUL(K(-0.7), V(1)), K(0.2))],
        "diffusion": [{"expression": K(0.3), "square_root": False}, {"expression": K(0.2), "square_root": False}],
        "boundaries": [],
    }
    cpath = [{"time": 0.0, "values": [0.2, -0.1]}]
    for k in range(1, 8):
        cpath.append({"time": round(k * 0.06, 6), "values": [round(cpath[-1]["values"][0] + rng.gauss(0, 0.12), 6),
                                                              round(cpath[-1]["values"][1] + rng.gauss(0, 0.1), 6)]})
    cpts = [[0.5, -0.4], [0.8, -0.2]]
    out["coupled_direct"] = {
        "system": coupled, "trajectories": [cpath], "points": cpts,
        "values": [{"theta": pt, "euler": num(direct_euler(coupled, pt, [cpath]))} for pt in cpts],
    }
    # extended Kalman likelihood: partial and noisy observation of the coupled linear system, with substeps
    times = [0.0, 0.1, 0.25, 0.3, 0.52, 0.6]
    for name, rows, offset, sds in [
        ("kalman_partial", [[1.0, 0.5]], [0.1], [0.15]),
        ("kalman_full_rank", [[1.0, 0.5], [0.0, 1.0]], [0.1, -0.05], [0.15, 0.2]),
    ]:
        ys = []
        state = [0.2, -0.1]
        for _ in times:
            state = [state[0] + rng.gauss(0, 0.1), state[1] + rng.gauss(0, 0.1)]
            ys.append([round(sum(r[j] * state[j] for j in range(2)) + offset[i] + rng.gauss(0, sds[i]), 6)
                       for i, r in enumerate(rows)])
        observation = {"matrix": [v for r in rows for v in r], "offset": offset, "noise_standard_deviation": sds}
        prior = {"mean": [0.2, -0.1], "covariance": [0.3, 0.05, 0.05, 0.2]}
        record = {"system": coupled, "observation": observation, "prior": prior, "times": times, "values_y": ys,
                  "points": cpts, "results": []}
        for filter_step in [0.0, 0.04]:
            for pt in cpts:
                ll = kalman(coupled, pt, {"matrix_rows": rows, "offset": offset, "noise_standard_deviation": sds},
                            {"mean": prior["mean"], "covariance_rows": [[0.3, 0.05], [0.05, 0.2]]}, times, ys, filter_step)
                record["results"].append({"filter_step": filter_step, "theta": pt, "log_likelihood": num(ll)})
        out[name] = record
    return out


def main():
    json.dump({"note": "Generated by cyclic_general_systems_reference.py (mpmath, 60 digits); no product code, no retrieved paper.",
               "adam": adam_cases(), "minibatch": minibatch_case(), "ql": ql_cases()}, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
