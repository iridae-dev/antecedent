# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Independent references for the numerics owned by the umbrella cyclic-systems record, model side:
the Dormand-Prince 5(4) integrator, the SplitMix64 / Box-Muller counter noise, the Euler-Maruyama
and diagonal Milstein ensembles, and the Krawczyk polynomial-system isolation.

    uv run --offline --script cyclic_general_systems_reference.py > cyclic_general_systems_reference.json

Nothing here reads product code.  Each part is written from the published definition:

* tableau   Dormand and Prince RK5(4)7M coefficients, written as Fractions and then *checked* against
            every rooted-tree order condition up to order five for the propagated solution and up to
            order four for the embedded one (so a transcription error cannot pass).
* ode       the standard accept / reject loop (inf-norm error ratio over atol + rtol * max(|y|,|y1|),
            factor 0.9 ratio^(-1/5) clamped to [0.1, 5], 5 on a zero error, exact landing on observation
            and action times) with 60-digit stage arithmetic; exact solutions by matrix exponentials,
            by sinh(x) = sinh(x0) e^t for dx/dt = tanh(x), or by direct closed forms.
* rng       SplitMix64 finalizer and the documented counter key in Python big integers; the
            two uniforms are formed in IEEE doubles exactly as documented and the Box-Muller cosine
            branch is evaluated at 60 digits from those doubles.  The published SplitMix64 constants
            are named below; the seed-zero output sequence quoted from memory is only a cross-check
            (nothing was retrieved).
* sde       Euler-Maruyama and diagonal Milstein, with the Brownian increment of the coarse level built
            from two fine increments, reflecting / clamping bounds, held components, linear and
            piecewise-constant inputs, the half-step refinement, ensemble moments and the step
            halving functionals, all at 60 digits with the frozen counter noise.
* krawczyk  exact Fraction Sturm sequences isolate every real root of small integer polynomials;
            multi-variable systems reduce to a univariate eliminant verified exactly at rational points.
"""
import json
import math
import sys
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60
MASK = (1 << 64) - 1


def num(x, digits=30):
    """A 60-digit mpf rendered as a JSON float that round-trips an f64 at most."""
    return float(mp.nstr(x, digits))


# ---------------------------------------------------------------- Dormand-Prince tableau

C = [F(0), F(1, 5), F(3, 10), F(4, 5), F(8, 9), F(1), F(1)]
A = [
    [],
    [F(1, 5)],
    [F(3, 40), F(9, 40)],
    [F(44, 45), F(-56, 15), F(32, 9)],
    [F(19372, 6561), F(-25360, 2187), F(64448, 6561), F(-212, 729)],
    [F(9017, 3168), F(-355, 33), F(46732, 5247), F(49, 176), F(-5103, 18656)],
    [F(35, 384), F(0), F(500, 1113), F(125, 192), F(-2187, 6784), F(11, 84)],
]
B5 = [F(35, 384), F(0), F(500, 1113), F(125, 192), F(-2187, 6784), F(11, 84), F(0)]
B4 = [F(5179, 57600), F(0), F(7571, 16695), F(393, 640), F(-92097, 339200), F(187, 2100), F(1, 40)]


def check_tableau():
    n = 7
    full = [[A[i][j] if j < len(A[i]) else F(0) for j in range(n)] for i in range(n)]
    for i in range(n):
        assert sum(full[i]) == C[i], f"row sum {i}"
    mul = lambda M, v: [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]
    had = lambda u, v: [a * b for a, b in zip(u, v)]
    dot = lambda u, v: sum(a * b for a, b in zip(u, v))
    one = [F(1)] * n
    c = C
    c2, c3, c4 = had(c, c), had(c, had(c, c)), had(had(c, c), had(c, c))
    Ac = mul(full, c)
    Ac2 = mul(full, c2)
    Ac3 = mul(full, c3)
    AAc = mul(full, Ac)
    A_cAc = mul(full, had(c, Ac))
    AAc2 = mul(full, Ac2)
    AAAc = mul(full, AAc)
    order4 = [
        (one, F(1)), (c, F(1, 2)), (c2, F(1, 3)), (Ac, F(1, 6)),
        (c3, F(1, 4)), (had(c, Ac), F(1, 8)), (Ac2, F(1, 12)), (AAc, F(1, 24)),
    ]
    order5 = [
        (c4, F(1, 5)), (had(c2, Ac), F(1, 10)), (had(c, Ac2), F(1, 15)), (had(Ac, Ac), F(1, 20)),
        (had(c, AAc), F(1, 30)), (Ac3, F(1, 20)), (A_cAc, F(1, 40)), (AAc2, F(1, 60)), (AAAc, F(1, 120)),
    ]
    for vec, value in order4:
        assert dot(B5, vec) == value and dot(B4, vec) == value, (vec, value)
    for vec, value in order5:
        assert dot(B5, vec) == value, (vec, value)
    # the embedded weights are NOT fifth order (that is what makes the difference an error estimate)
    assert any(dot(B4, vec) != value for vec, value in order5)
    # first-same-as-last: the seventh stage row equals the propagated weights
    assert full[6] == B5[:6] + [F(0)]


check_tableau()
MA = [[mp.mpf(x.numerator) / x.denominator for x in row] for row in A]
MB5 = [mp.mpf(x.numerator) / x.denominator for x in B5]
MB4 = [mp.mpf(x.numerator) / x.denominator for x in B4]

# ---------------------------------------------------------------- ODE

ACT = {
    "identity": lambda x: x,
    "tanh": mp.tanh,
    "sin": mp.sin,
    "relu": lambda x: max(x, mp.mpf(0)),
}


def ode_drift(system, y):
    n = len(system["equations"])
    out = []
    for i, e in enumerate(system["equations"]):
        s = sum(mp.mpf(e["weights"][j]) * y[j] for j in range(n))
        out.append(mp.mpf(e["intercept"]) + ACT[e["activation"]](s) - mp.mpf(system["decay"][i]) * y[i])
    return out


def dp_step(f, y, h):
    h = mp.mpf(h)
    n = len(y)
    k = [f(y)]
    for row in MA[1:]:
        k.append(f([y[i] + h * sum(row[j] * k[j][i] for j in range(len(row))) for i in range(n)]))
    new = [y[i] + h * sum(MB5[j] * k[j][i] for j in range(7)) for i in range(n)]
    err = [h * sum((MB5[j] - MB4[j]) * k[j][i] for j in range(7)) for i in range(n)]
    return new, err


def integrate(system, x0, times, schedule, policy):
    n = len(x0)
    atol, rtol = policy["absolute_tolerance"], policy["relative_tolerance"]
    y = [mp.mpf(v) for v in x0]
    clamped = [None] * n
    t, h = float(times[0]), float(policy["initial_step"])
    event = 0
    accepted = rejected = 0
    states = []
    for target in times:
        while True:
            while event < len(schedule) and schedule[event]["time"] <= t:
                for c in schedule[event]["clamps"]:
                    clamped[c["variable"]] = mp.mpf(c["value"])
                    y[c["variable"]] = mp.mpf(c["value"])
                event += 1
            if t >= target:
                break
            boundary = min(target, schedule[event]["time"]) if event < len(schedule) else target
            h = min(h, boundary - t)

            def f(v):
                v = [clamped[i] if clamped[i] is not None else v[i] for i in range(n)]
                d = ode_drift(system, v)
                return [mp.mpf(0) if clamped[i] is not None else d[i] for i in range(n)]

            new, err = dp_step(f, y, h)
            ratio = max(
                abs(err[i]) / (mp.mpf(atol) + mp.mpf(rtol) * max(abs(y[i]), abs(new[i]))) for i in range(n)
            )
            ratio_f = float(ratio)
            if ratio_f <= 1.0:
                y = new
                t = boundary if h >= boundary - t else t + h
                accepted += 1
            else:
                rejected += 1
            h *= 5.0 if ratio_f == 0.0 else min(max(0.9 * ratio_f ** -0.2, 0.1), 5.0)
        states.append(list(y))
    return states, accepted, rejected


def exact_linear(system_matrix, offset, x0, times, schedule):
    """Piecewise exact solution of x' = M x + b with sustained clamps (clamped rows zeroed)."""
    n = len(x0)
    clamped = {}
    state = [mp.mpf(v) for v in x0]
    t = mp.mpf(times[0])
    event = 0
    out = []

    def advance(t0, t1):
        nonlocal state
        aug = mp.zeros(n + 1, n + 1)
        for i in range(n):
            if i in clamped:
                continue
            for j in range(n):
                aug[i, j] = system_matrix[i][j]
            aug[i, n] = offset[i]
        e = mp.expm(aug * (mp.mpf(t1) - mp.mpf(t0)))
        vec = state + [mp.mpf(1)]
        state = [sum(e[i, j] * vec[j] for j in range(n + 1)) for i in range(n)]

    for target in times:
        while True:
            while event < len(schedule) and schedule[event]["time"] <= float(t):
                for c in schedule[event]["clamps"]:
                    clamped[c["variable"]] = mp.mpf(c["value"])
                    state[c["variable"]] = mp.mpf(c["value"])
                event += 1
            if float(t) >= target:
                break
            boundary = min(target, schedule[event]["time"]) if event < len(schedule) else target
            advance(t, boundary)
            t = mp.mpf(boundary)
        out.append(list(state))
    return out


def linear_parts(system):
    n = len(system["equations"])
    M = [[mp.mpf(system["equations"][i]["weights"][j]) - (mp.mpf(system["decay"][i]) if i == j else 0)
          for j in range(n)] for i in range(n)]
    b = [mp.mpf(e["intercept"]) for e in system["equations"]]
    return M, b


def eq(weights, intercept=0.0, activation="identity"):
    return {"id": "e", "intercept": intercept, "weights": weights, "activation": activation}


def named(system):
    for i, e in enumerate(system["equations"]):
        e["id"] = f"v{i}"
    return system


def ode_cases():
    default = {"absolute_tolerance": 1e-9, "relative_tolerance": 1e-7, "initial_step": 0.01, "max_steps": 100000}
    cases = []

    def add(name, system, x0, times, schedule, policy, exact=None, note=""):
        system = named(system)
        states, acc, rej = integrate(system, x0, times, schedule, policy)
        record = {
            "name": name, "note": note, "system": system, "initial": x0, "times": times,
            "schedule": schedule, "policy": policy,
            "states": [[num(v) for v in s] for s in states], "accepted": acc, "rejected": rej,
        }
        if exact is not None:
            record["exact"] = [[num(v) for v in s] for s in exact]
            err = max(abs(a - b) for s, e in zip(states, exact) for a, b in zip(s, e))
            record["exact_error_of_reference"] = num(err)
        cases.append(record)

    rotation = {"equations": [eq([0.0, -1.0]), eq([1.0, 0.0])], "decay": [0.0, 0.0]}
    M, b = linear_parts(rotation)
    t4 = [0.0, 0.5, 1.0, 3.0]
    add("rotation", rotation, [1.0, 0.0], t4, [], default,
        [[mp.cos(mp.mpf(t)), mp.sin(mp.mpf(t))] for t in t4], "x' = -y, y' = x")
    sched = [
        {"time": 0.5, "clamps": [{"variable": 0, "value": 2.0}]},
        {"time": 1.0, "clamps": [{"variable": 1, "value": -0.5}]},
    ]
    t5 = [0.0, 0.5, 1.0, 2.0]
    add("rotation_with_timed_clamps", rotation, [1.0, 0.0], t5, sched, default,
        exact_linear(M, b, [1.0, 0.0], t5, sched), "sustained clamps replace a mechanism and zero its derivative")
    decay = {"equations": [eq([0.0], 1.5)], "decay": [0.75]}
    t3 = [0.0, 1.0, 4.0, 9.0]
    add("decay_toward_equilibrium", decay, [-2.0], t3, [], default,
        [[mp.mpf(2) + (mp.mpf(-2) - 2) * mp.exp(-mp.mpf("0.75") * mp.mpf(t))] for t in t3],
        "x' = 1.5 - 0.75 x, x = 2 + (x0 - 2) e^(-0.75 t)")
    tanh_sys = {"equations": [eq([1.0], 0.0, "tanh")], "decay": [0.0]}
    tight = {"absolute_tolerance": 1e-13, "relative_tolerance": 1e-11, "initial_step": 0.1, "max_steps": 100000}
    t2 = [0.0, 0.7, 1.5, 2.5]
    add("tanh_has_sinh_solution", tanh_sys, [0.6], t2, [], tight,
        [[mp.asinh(mp.sinh(mp.mpf("0.6")) * mp.exp(mp.mpf(t)))] for t in t2],
        "x' = tanh(x) has sinh(x) = sinh(x0) e^t")
    stiff = {"equations": [eq([0.0, 0.5], 2.0), eq([0.8, 0.0], -1.0)], "decay": [40.0, 25.0]}
    Ms, bs = linear_parts(stiff)
    t6 = [0.0, 0.2, 1.0, 2.0]
    stiff_policy = {"absolute_tolerance": 1e-10, "relative_tolerance": 1e-9, "initial_step": 0.5, "max_steps": 100000}
    add("stiff_decay_with_rejections", stiff, [3.0, -4.0], t6, [], stiff_policy,
        exact_linear(Ms, bs, [3.0, -4.0], t6, []), "large first step: the controller must reject and shrink")
    # one accepted step: the library result is exactly one Dormand-Prince step
    pair = {"equations": [eq([0.0, 0.9], 0.1, "tanh"), eq([-0.7, 0.0], 0.0, "sin")], "decay": [0.2, 0.0]}
    loose = {"absolute_tolerance": 1e6, "relative_tolerance": 1e6, "initial_step": 0.8, "max_steps": 10}
    add("single_accepted_step", pair, [0.4, -0.3], [0.0, 0.8], [], loose, None,
        "tolerances so loose that exactly one step of 0.8 is accepted: the state is one DP5 step")
    hard = {"absolute_tolerance": 1e-10, "relative_tolerance": 1e-10, "initial_step": 1.0, "max_steps": 100000}
    add("nonlinear_pair_with_rejections", pair, [0.4, -0.3], [0.0, 1.0, 2.5, 6.0], [], hard, None,
        "tanh/sin feedback with a large first step: rejection counts are compared exactly")
    zero = {"equations": [eq([0.0], 0.0)], "decay": [0.0]}
    grow = {"absolute_tolerance": 1e-9, "relative_tolerance": 1e-7, "initial_step": 0.01, "max_steps": 100000}
    add("zero_error_grows_by_five", zero, [1.0], [0.0, 10.0], [], grow,
        [[mp.mpf(1)], [mp.mpf(1)]], "zero drift: error 0, so the step grows by exactly 5 until it is cut at the end")
    relu = {"equations": [eq([0.0, 1.0], 0.0, "relu"), eq([-1.0, 0.0], 0.0)], "decay": [0.5, 0.5]}
    add("relu_kink_decay", relu, [1.0, 0.5], [0.0, 1.0, 3.0], [], default, None,
        "a piecewise-linear mechanism: the controller handles the kink by step rejection")
    return cases


# ---------------------------------------------------------------- counter noise

GOLDEN = 0x9E3779B97F4A7C15
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB


def mix(x):
    z = (x + GOLDEN) & MASK
    z = ((z ^ (z >> 30)) * M1) & MASK
    z = ((z ^ (z >> 27)) * M2) & MASK
    return z ^ (z >> 31)


def inv_xorshift(z, s):
    x = z
    for _ in range(64 // s + 1):
        x = z ^ (x >> s)
    return x & MASK


def unmix(y):
    z = inv_xorshift(y, 31)
    z = (z * pow(M2, -1, 1 << 64)) & MASK
    z = inv_xorshift(z, 27)
    z = (z * pow(M1, -1, 1 << 64)) & MASK
    z = inv_xorshift(z, 30)
    return (z - GOLDEN) & MASK


def key_of(seed, path, step, comp):
    return mix(mix(mix(mix(seed) ^ path) ^ step) ^ comp)


def uniforms(key):
    scale = 9007199254740992.0
    u1 = (float(mix(key) >> 11) + 0.5) / scale
    u2 = (float(mix(key ^ 1) >> 11) + 0.5) / scale
    return u1, u2


def counter_normal_mp(seed, path, step, comp):
    u1, u2 = uniforms(key_of(seed, path, step, comp))
    angle = 2.0 * math.pi * u2
    return mp.sqrt(-2 * mp.log(mp.mpf(u1))) * mp.cos(mp.mpf(angle))


def ideal_normal_mp(key):
    """Box-Muller of the exact rationals (k + 1/2) / 2^53, without double rounding."""
    k1, k2 = mix(key) >> 11, mix(key ^ 1) >> 11
    u1 = (mp.mpf(k1) + mp.mpf(1) / 2) / mp.mpf(2) ** 53
    u2 = (mp.mpf(k2) + mp.mpf(1) / 2) / mp.mpf(2) ** 53
    return mp.sqrt(-2 * mp.log(u1)) * mp.cos(2 * mp.pi * u2)


def rng_cases():
    # constants of the published finalizer; seed-zero outputs recalled from the SplitMix64 literature
    recalled_seed_zero = [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    stream = [mix(((k * GOLDEN) & MASK)) for k in range(3)]
    assert stream == recalled_seed_zero, [hex(v) for v in stream]
    mixes = []
    probe = [0, 1, GOLDEN, MASK, MASK - GOLDEN + 1, 0x0123456789ABCDEF, 1 << 63, 12345678901234567890]
    for x in probe:
        mixes.append({"x": str(x), "mix": str(mix(x))})
    # bijectivity: the explicit inverse reproduces the input
    for x in probe:
        assert unmix(mix(x)) == x
    cases = []
    tuples = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (7, 3, 1000, 2),
              (12345, 999, 2**20, 15), (MASK, MASK, MASK, MASK), (2**63, 2**32, 2**40 + 5, 17),
              (987654321, 5, 5, 5), (42, 1023, 77, 3)]
    # cases that drive each uniform to the ends of its range: pick the component so that the final key
    # has the wanted mixed output (the mixer is a bijection)
    base = mix(mix(mix(mix(0) ^ 0) ^ 0) ^ 0)
    wanted = []

    def key_for_top(target_hi):
        # need mix(key) >> 11 == target_hi: choose mix(key) = target_hi << 11 | low bits, key = unmix(...)
        return unmix((target_hi << 11) | 0x155)

    extremes = {}
    for label, hi in [("u1_min", 0), ("u1_max", (1 << 53) - 1), ("u1_half", 1 << 52), ("u1_just_below_half", (1 << 52) - 1)]:
        key = key_for_top(hi)
        component = (mix(mix(mix(0) ^ 0) ^ 0)) ^ unmix(key)
        assert key_of(0, 0, 0, component) == key
        tuples.append((0, 0, 0, component))
        extremes[(0, 0, 0, component)] = label
    for tup in tuples:
        key = key_of(*tup)
        u1, u2 = uniforms(key)
        record = {
            "seed": str(tup[0]), "path": str(tup[1]), "step": str(tup[2]), "component": str(tup[3]),
            "key": str(key), "u1": u1, "u2": u2,
            "value": num(counter_normal_mp(*tup)),
            "ideal_value": num(ideal_normal_mp(key)),
        }
        if tup in extremes:
            record["label"] = extremes[tup]
        cases.append(record)
    return {"mixes": mixes, "cases": cases, "recalled_seed_zero_outputs": [str(v) for v in recalled_seed_zero]}


# ---------------------------------------------------------------- SDE (60-digit scheme replay)

def ev(expr, vs, theta):
    """Value and gradient (over the coordinates vs) of a mechanism expression."""
    op = expr["op"]
    nvar = len(vs)
    if op == "constant":
        return mp.mpf(expr["value"]), [mp.mpf(0)] * nvar
    if op == "variable":
        g = [mp.mpf(0)] * nvar
        g[expr["variable"]] = mp.mpf(1)
        return vs[expr["variable"]], g
    if op == "parameter":
        return mp.mpf(theta[expr["parameter"]]), [mp.mpf(0)] * nvar
    if op == "add":
        parts = [ev(t, vs, theta) for t in expr["terms"]]
        return sum(p[0] for p in parts), [sum(p[1][i] for p in parts) for i in range(nvar)]
    if op == "multiply":
        parts = [ev(t, vs, theta) for t in expr["factors"]]
        value = mp.mpf(1)
        for p in parts:
            value *= p[0]
        grad = []
        for i in range(nvar):
            total = mp.mpf(0)
            for k, p in enumerate(parts):
                term = p[1][i]
                for m, q in enumerate(parts):
                    if m != k:
                        term *= q[0]
                total += term
            grad.append(total)
        return value, grad
    inner, g = ev(expr["argument"], vs, theta)
    if op == "tanh":
        s = 1 - mp.tanh(inner) ** 2
        return mp.tanh(inner), [s * x for x in g]
    if op == "sin":
        return mp.sin(inner), [mp.cos(inner) * x for x in g]
    if op == "softplus":
        return mp.log1p(mp.exp(inner)), [x / (1 + mp.exp(-inner)) for x in g]
    raise ValueError(op)


def input_value(inp, t):
    ts, vals = inp["times"], inp["values"]
    n = len(ts)
    if t <= ts[0]:
        return mp.mpf(vals[0])
    if t >= ts[-1]:
        return mp.mpf(vals[-1])
    idx = max(i for i in range(n) if ts[i] <= t)
    if inp.get("interpolation", "piecewise_constant") == "piecewise_constant":
        return mp.mpf(vals[idx])
    w = mp.mpf(t - ts[idx]) / mp.mpf(ts[idx + 1] - ts[idx])
    return mp.mpf(vals[idx]) * (1 - w) + mp.mpf(vals[idx + 1]) * w


def apply_bound(b, x):
    lo, hi, rule = b.get("lower"), b.get("upper"), b.get("rule", "reflect")
    if lo is None and hi is None:
        return x
    if rule == "clamp":
        if lo is not None:
            x = max(x, mp.mpf(lo))
        if hi is not None:
            x = min(x, mp.mpf(hi))
        return x
    if lo is not None and hi is None:
        return mp.mpf(lo) + abs(x - lo)
    if lo is None:
        return mp.mpf(hi) - abs(mp.mpf(hi) - x)
    w = mp.mpf(hi) - mp.mpf(lo)
    r = (x - lo) - 2 * w * mp.floor((x - lo) / (2 * w))
    if r > w:
        r = 2 * w - r
    return mp.mpf(lo) + r


def action_value(action, t, tol):
    ts = action["times"]
    if t + tol < ts[0]:
        return None
    idx = max(i for i in range(len(ts)) if ts[i] <= t + tol)
    return mp.mpf(action["values"][idx])


def run_path(system, theta, world_inputs, actions, policy, x0, path, fine):
    d = len(system["variables"])
    m = 2 if fine else 1
    step = policy["step"]
    dt = step / m
    tol = 1e-9 * step
    record_every = policy["output_every"] * m
    total = policy["steps"] * m
    bounds = system.get("boundaries") or [{}] * d
    x = [mp.mpf(v) for v in x0]

    def apply_actions(x, t):
        for a in actions:
            v = action_value(a, t, tol)
            if v is not None:
                x[a["variable"]] = v

    apply_actions(x, policy["start"])
    out = [list(x)]
    sqrt_half = mp.sqrt(mp.mpf(step) / 2)
    for n in range(total):
        t = policy["start"] + n * dt
        vs = list(x) + [input_value(inp, t) for inp in world_inputs]
        nxt = [None] * d
        for i in range(d):
            if any(a["variable"] == i and action_value(a, t, tol) is not None for a in actions):
                nxt[i] = x[i]
                continue
            f, _ = ev(system["drift"][i], vs, theta)
            h, gh = ev(system["diffusion"][i]["expression"], vs, theta)
            sq = system["diffusion"][i].get("square_root", False)
            g = mp.sqrt(max(h, mp.mpf(0))) if sq else h
            if fine:
                dw = sqrt_half * counter_normal_mp(policy["seed"], path, n, i)
            else:
                dw = sqrt_half * (counter_normal_mp(policy["seed"], path, 2 * n, i)
                                  + counter_normal_mp(policy["seed"], path, 2 * n + 1, i))
            value = x[i] + f * mp.mpf(dt) + g * dw
            if policy["scheme"] == "milstein":
                dh = gh[i]
                if sq:
                    corr = dh / 4 if h > 0 else mp.mpf(0)
                else:
                    corr = g * dh / 2
                value += corr * (dw * dw - mp.mpf(dt))
            nxt[i] = value
        x = [apply_bound(bounds[i], nxt[i]) for i in range(d)]
        apply_actions(x, t + dt)
        if (n + 1) % record_every == 0:
            out.append(list(x))
    return out


def moments(levels, key):
    """Ensemble report quantities from per-path states: levels[path][time][dim]."""
    paths = len(levels)
    n_out = len(levels[0])
    d = len(levels[0][0])
    mean = [[sum(levels[p][t][i] for p in range(paths)) / paths for i in range(d)] for t in range(n_out)]
    var = [[sum((levels[p][t][i] - mean[t][i]) ** 2 for p in range(paths)) / (paths - 1) for i in range(d)]
           for t in range(n_out)]
    cov = [[sum((levels[p][t][i] - mean[t][i]) * (levels[p][t][j] - mean[t][j]) for p in range(paths)) / (paths - 1)
            for j in range(d) for i in range(d)] for t in range(n_out)]
    return mean, var, cov


def numerical(coarse, fine):
    paths, n_out, d = len(coarse), len(coarse[0]), len(coarse[0][0])
    weak, strong, rich, se = [], [], [], []
    for t in range(n_out):
        w, s, r, e = [], [], [], []
        for i in range(d):
            diff = [coarse[p][t][i] - fine[p][t][i] for p in range(paths)]
            md = sum(diff) / paths
            vd = sum((v - md) ** 2 for v in diff) / (paths - 1)
            mf = sum(fine[p][t][i] for p in range(paths)) / paths
            w.append(md)
            e.append(mp.sqrt(vd / paths))
            s.append(mp.sqrt(sum(v * v for v in diff) / paths))
            r.append(2 * mf - sum(coarse[p][t][i] for p in range(paths)) / paths)
        weak.append(w), strong.append(s), rich.append(r), se.append(e)
    return weak, se, strong, rich


def grid(a):
    return [[num(v) for v in row] for row in a]


def sde_case(name, system, theta, x0, policy, worlds, note):
    base_inputs = system.get("inputs", [])
    plans = [("baseline", [], base_inputs)]
    for w in worlds:
        ins = [dict(i) for i in base_inputs]
        for r in w.get("inputs", []):
            slot = [k for k, i in enumerate(base_inputs) if i["name"] == r["name"]][0]
            ins[slot] = r
        plans.append((w["id"], w.get("actions", []), ins))
    result = []
    for pid, actions, inputs in plans:
        coarse = [run_path(system, theta, inputs, actions, policy, x0, p, False) for p in range(policy["paths"])]
        fine = [run_path(system, theta, inputs, actions, policy, x0, p, True) for p in range(policy["paths"])]
        mean, var, cov = moments(coarse, None)
        weak, se, strong, rich = numerical(coarse, fine)
        result.append({
            "id": pid, "coarse": [[[num(v) for v in row] for row in path] for path in coarse],
            "mean": grid(mean), "variance": grid(var), "covariance": grid(cov),
            "weak_difference": grid(weak), "weak_difference_standard_error": grid(se),
            "strong_rms_difference": grid(strong), "richardson_mean": grid(rich),
            "_coarse_mp": coarse, "_fine_mp": fine,
        })
    contrasts = []
    if len(plans) == 2:
        paths = policy["paths"]
        t, c = result[1], result[0]
        diff = [[[t["_coarse_mp"][p][s][i] - c["_coarse_mp"][p][s][i] for i in range(len(x0))]
                 for s in range(len(t["_coarse_mp"][0]))] for p in range(paths)]
        dfine = [[[t["_fine_mp"][p][s][i] - c["_fine_mp"][p][s][i] for i in range(len(x0))]
                  for s in range(len(t["_coarse_mp"][0]))] for p in range(paths)]
        mean, _, _ = moments(diff, None)
        weak, se, strong, rich = numerical(diff, dfine)
        contrasts.append({"treated": plans[1][0], "control": "baseline", "mean_difference": grid(mean),
                          "weak_difference": grid(weak), "weak_difference_standard_error": grid(se)})
    for r in result:
        del r["_coarse_mp"], r["_fine_mp"]
    return {"name": name, "note": note, "system": system, "theta": theta, "initial": x0, "policy": policy,
            "worlds": worlds, "reports": result, "contrasts": contrasts}


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


def TANH(a):
    return {"op": "tanh", "argument": a}


def SIN(a):
    return {"op": "sin", "argument": a}


def sde_cases():
    out = []
    pol = lambda **kw: dict({"start": 0.0, "step": 0.05, "steps": 20, "output_every": 5, "scheme": "euler_maruyama",
                             "paths": 6, "seed": 12345, "weak_tolerance": 0.02, "check_step": True,
                             "retain_final_samples": False}, **kw)
    gbm = {
        "variables": ["x"], "inputs": [], "parameters": [{"name": "mu", "value": 0.3}, {"name": "sigma", "value": 0.4}],
        "drift": [MUL(P(0), V(0))], "diffusion": [{"expression": MUL(P(1), V(0)), "square_root": False}],
        "boundaries": [],
    }
    for scheme in ["euler_maruyama", "milstein"]:
        out.append(sde_case(f"geometric_{scheme}", gbm, [0.3, 0.4], [1.0], pol(scheme=scheme), [],
                            "dX = mu X dt + sigma X dW; Milstein adds sigma^2 X (dW^2 - dt)/2"))
    cir = {
        "variables": ["x"], "inputs": [],
        "parameters": [{"name": "kappa", "value": 1.2}, {"name": "theta", "value": 1.0}, {"name": "s2", "value": 0.09}],
        "drift": [MUL(P(0), ADD(P(1), MUL(K(-1.0), V(0))))],
        "diffusion": [{"expression": MUL(P(2), V(0)), "square_root": True}],
        "boundaries": [{"lower": 0.0, "upper": 3.0, "rule": "reflect"}],
    }
    for scheme in ["euler_maruyama", "milstein"]:
        out.append(sde_case(f"square_root_reflecting_{scheme}", cir, [1.2, 1.0, 0.09], [0.8],
                            pol(scheme=scheme, steps=40, output_every=10, step=0.025, paths=5, seed=2024), [],
                            "dX = kappa (theta - X) dt + sqrt(s2 X) dW on [0, 3], reflecting; Milstein h'/4 (dW^2 - dt)"))
    # a square-root diffusion started at exactly zero under a clamp: h = 0 so the Milstein term is skipped
    cir_clamp = json.loads(json.dumps(cir))
    cir_clamp["boundaries"] = [{"lower": 0.0, "upper": None, "rule": "clamp"}]
    out.append(sde_case("square_root_zero_start_clamped_milstein", cir_clamp, [1.2, 1.0, 0.5], [0.0],
                        pol(scheme="milstein", steps=30, output_every=10, step=0.02, paths=6, seed=77), [],
                        "x0 = 0: the first diffusion is exactly zero, a negative step is clamped back to exactly zero"))
    coupled = {
        "variables": ["x", "y"],
        "inputs": [{"name": "u", "interpolation": "linear", "times": [0.0, 0.5, 1.0], "values": [0.0, 1.0, -0.5],
                    "envelope": [-1.0, 1.0]}],
        "parameters": [{"name": "a", "value": 0.8}, {"name": "b", "value": 0.5}],
        "drift": [
            ADD(MUL(K(-1.0), P(0), V(0)), MUL(P(1), TANH(V(1))), MUL(K(0.3), V(2))),
            ADD(MUL(K(0.2), V(0)), MUL(K(-0.5), V(1)), SIN(V(2))),
        ],
        "diffusion": [
            {"expression": ADD(K(0.3), MUL(K(0.2), TANH(V(0)))), "square_root": False},
            {"expression": ADD(K(0.04), MUL(K(0.05), V(1))), "square_root": True},
        ],
        "boundaries": [{"lower": -2.0, "upper": 2.0, "rule": "reflect"}, {"lower": 0.0, "upper": None, "rule": "clamp"}],
    }
    worlds = [{
        "id": "treated",
        "actions": [{"variable": 1, "times": [0.25, 0.5], "values": [1.5, 0.2]}],
        "inputs": [{"name": "u", "interpolation": "piecewise_constant", "times": [0.0, 0.4], "values": [0.5, -0.5]}],
    }]
    for scheme in ["euler_maruyama", "milstein"]:
        out.append(sde_case(f"coupled_world_{scheme}", coupled, [0.8, 0.5], [0.5, 0.3],
                            pol(scheme=scheme, steps=20, output_every=4, step=0.05, paths=7, seed=31337), worlds,
                            "two states, a linear input, a held component with a schedule, a replaced input, "
                            "reflecting and clamping bounds and a paired contrast"))
    return out


# ---------------------------------------------------------------- Krawczyk reference: Sturm sequences

def pmul(a, b):
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def padd(a, b):
    n = max(len(a), len(b))
    return trim([(a[i] if i < len(a) else F(0)) + (b[i] if i < len(b) else F(0)) for i in range(n)])


def trim(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def pdiv(a, b):
    a = list(a)
    q = [F(0)] * max(len(a) - len(b) + 1, 1)
    while len(a) >= len(b) and a != [F(0)]:
        k = len(a) - len(b)
        c = a[-1] / b[-1]
        q[k] = c
        for i, y in enumerate(b):
            a[i + k] -= c * y
        a.pop()
        while len(a) > 1 and a[-1] == 0:
            a.pop()
        if not a:
            a = [F(0)]
    return trim(q), trim(a)


def pdiff(p):
    return trim([i * c for i, c in enumerate(p)][1:] or [F(0)])


def peval(p, x):
    r = F(0)
    for c in reversed(p):
        r = r * x + c
    return r


def pgcd(a, b):
    a, b = trim(a), trim(b)
    while b != [F(0)]:
        _, r = pdiv(a, b)
        a, b = b, r
    return [c / a[-1] for c in a]


def sturm_chain(p):
    chain = [trim(p), pdiff(p)]
    while chain[-1] != [F(0)]:
        _, r = pdiv(chain[-2], chain[-1])
        if r == [F(0)]:
            break
        chain.append([-c for c in r])
    return chain


def sign_changes(chain, x):
    signs = [s for s in (peval(c, x) for c in chain) if s != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if (a > 0) != (b > 0))


def count_roots(chain, a, b):
    """Distinct real roots in the half-open interval (a, b]."""
    return sign_changes(chain, a) - sign_changes(chain, b)


def squarefree(p):
    g = pgcd(p, pdiff(p))
    q, r = pdiv(p, g)
    assert r == [F(0)]
    return q


def isolate(p, lo, hi, width=F(1, 1 << 110)):
    """Brackets (a, b] of width <= `width` isolating each distinct real root of p inside [lo, hi]."""
    q = squarefree(p)
    chain = sturm_chain(q)
    out = []
    # the closed lower end: a root exactly at lo is handled by testing p(lo)
    if peval(q, lo) == 0:
        out.append((lo, lo))
    stack = [(lo, hi)]
    while stack:
        a, b = stack.pop()
        n = count_roots(chain, a, b)
        if n == 0:
            continue
        if n == 1 and b - a <= width:
            out.append((a, b))
            continue
        if b - a <= width:
            raise RuntimeError("cluster below the bracket width")
        mid = (a + b) / 2
        stack.append((mid, b))
        stack.append((a, mid))
    return sorted(out)


def outward(a, b):
    lo = float(a)
    if F(lo) > a:
        lo = math.nextafter(lo, -math.inf)
    hi = float(b)
    if F(hi) < b:
        hi = math.nextafter(hi, math.inf)
    return [math.nextafter(lo, -math.inf), math.nextafter(hi, math.inf)] if a != b else [lo, hi]


def poly_from_terms(terms):
    deg = max(t[0] for t in terms)
    p = [F(0)] * (deg + 1)
    for d, c in terms:
        p[d] += F(c)
    return trim(p)


def multiplicity_roots(p, lo, hi):
    """Each distinct root with its multiplicity, via the chain of square-free parts of gcds."""
    levels = []
    cur = trim(p)
    while len(cur) > 1:
        sf = squarefree(cur)
        levels.append(isolate(sf, lo, hi))
        cur = pgcd(cur, pdiff(cur))
        if len(cur) <= 1:
            break
    base = levels[0]
    result = []
    for br in base:
        mult = 1
        for deeper in levels[1:]:
            if any(not (d[1] < br[0] or d[0] > br[1]) for d in deeper):
                mult += 1
            else:
                break
        result.append((br, mult))
    return result


def univariate_case(name, coeffs, domain, note, policy=None):
    """coeffs: ascending integer / dyadic coefficients."""
    p = [F(c) for c in coeffs]
    lo, hi = F(domain[0]), F(domain[1])
    roots = multiplicity_roots(p, lo, hi)
    terms = [{"coefficient": float(c), "powers": [i]} for i, c in enumerate(coeffs) if c != 0]
    return {
        "name": name, "note": note,
        "system": {"variables": ["x"], "equations": [{"id": "p", "terms": terms}],
                   "domain": [{"lower": float(domain[0]), "upper": float(domain[1])}]},
        "policy": policy or {"max_boxes": 10000, "minimum_width": 1e-8},
        "roots": [{"multiplicity": m, "brackets": [outward(*br)]} for br, m in roots],
        "distinct_real_roots": len(roots),
    }


def krawczyk_cases():
    cases = []
    cases.append(univariate_case("cubic_three_simple_roots", [0, -1, 0, 1], (-2, 2), "x^3 - x"))
    wil = [F(1)]
    for r in range(1, 7):
        wil = pmul(wil, [F(-r), F(1)])
    cases.append(univariate_case("wilkinson_six", wil, (0, 7), "(x-1)(x-2)...(x-6), all simple, wide domain"))
    d = F(1, 1024)
    close = pmul([F(-1), F(1)], [-(1 + d), F(1)])
    cases.append(univariate_case("two_close_roots", close, (0, 3), "roots 1 and 1 + 2^-10"))
    cases.append(univariate_case("no_real_roots", [1, 0, 0, 0, 1], (-3, 3), "x^4 + 1"))
    dbl = pmul(pmul([F(0), F(1)], [F(0), F(1)]), pmul([F(-1), F(1)], [F(2), F(1)]))
    cases.append(univariate_case("double_root_stays_unresolved", dbl, (-3, 3), "x^2 (x-1)(x+2): the double root at 0 has "
                                 "a singular Jacobian and cannot be certified unique"))
    trip = pmul(pmul([F(-1), F(1)], pmul([F(-1), F(1)], [F(-1), F(1)])), [F(3), F(1)])
    cases.append(univariate_case("triple_root_stays_unresolved", trip, (-5, 5), "(x-1)^3 (x+3)"))
    bnd = pmul([F(0), F(1)], pmul([F(-2), F(1)], [F(2), F(1)]))
    cases.append(univariate_case("roots_on_the_domain_boundary", bnd, (-2, 2), "x (x-2)(x+2) on [-2, 2]: closed domain"))
    cases.append(univariate_case("degree_eight_oscillating", pmul(pmul([F(-1), F(1)], [F(1), F(1)]),
                 pmul(pmul([F(-2), F(1)], [F(2), F(1)]), pmul(pmul([F(-3), F(1)], [F(3), F(1)]),
                 pmul([F(-1, 2), F(1)], [F(1, 2), F(1)])))), (-4, 4), "eight simple roots +-1/2, +-1, +-2, +-3"))
    # two-variable systems
    def sys2(e1, e2, domain):
        def terms(poly):
            return [{"coefficient": float(c), "powers": list(pw)} for pw, c in poly]
        return {"variables": ["x", "y"], "equations": [{"id": "e1", "terms": terms(e1)}, {"id": "e2", "terms": terms(e2)}],
                "domain": [{"lower": float(domain[0]), "upper": float(domain[1])}] * 2}
    # x^2 + y^2 = 5, x y = 2: eliminate y = 2/x, then x^4 - 5 x^2 + 4 = (x^2-1)(x^2-4)
    e1 = [((2, 0), 1), ((0, 2), 1), ((0, 0), -5)]
    e2 = [((1, 1), 1), ((0, 0), -2)]
    for x in [F(1, 3), F(3, 2), F(-5, 7), F(11, 5)]:
        y = 2 / x
        assert x * x + y * y - 5 == (peval([F(4), F(0), F(-5), F(0), F(1)], x)) / (x * x)
    eliminant = [F(4), F(0), F(-5), F(0), F(1)]
    xs = multiplicity_roots(eliminant, F(-4), F(4))
    assert len(xs) == 4 and all(m == 1 for _, m in xs)
    roots = []
    for (a, b), _ in xs:
        ylo, yhi = sorted([2 / a, 2 / b])
        roots.append({"multiplicity": 1, "brackets": [outward(a, b), outward(ylo, yhi)]})
    cases.append({"name": "circle_hyperbola", "note": "x^2 + y^2 = 5, x y = 2: (+-1, +-2) and (+-2, +-1)",
                  "system": sys2(e1, e2, (-4, 4)), "policy": {"max_boxes": 20000, "minimum_width": 1e-8},
                  "roots": roots, "distinct_real_roots": 4})
    # a non-symmetric Jacobian: y = 3 - x^2 and 2 x + y^2 = 6; eliminant (x-1)(x^3 + x^2 - 5x - 3)
    e1 = [((2, 0), 1), ((0, 1), 1), ((0, 0), -3)]
    e2 = [((1, 0), 2), ((0, 2), 1), ((0, 0), -6)]
    quartic = [F(3), F(2), F(-6), F(0), F(1)]
    for x in [F(1, 3), F(-3, 2), F(5, 7), F(2)]:
        y = 3 - x * x
        assert 2 * x + y * y - 6 == peval(quartic, x)
    xs = multiplicity_roots(quartic, F(-4), F(4))
    roots = []
    for (a, b), m in xs:
        assert m == 1
        # y = 3 - x^2 on the bracket: interval image
        cands = [3 - a * a, 3 - b * b] + ([F(3)] if a <= 0 <= b else [])
        ylo, yhi = min(cands), max(cands)
        if not (-4 <= ylo and yhi <= 4):
            continue
        roots.append({"multiplicity": 1, "brackets": [outward(a, b), outward(ylo, yhi)]})
    cases.append({"name": "parabola_pair_non_symmetric", "note": "y = 3 - x^2, 2x + y^2 = 6 on [-4,4]^2: eliminant "
                  "(x-1)(x^3+x^2-5x-3); roots whose y lies outside the domain are excluded",
                  "system": sys2(e1, e2, (-4, 4)), "policy": {"max_boxes": 50000, "minimum_width": 1e-8},
                  "roots": roots, "distinct_real_roots": len(roots)})
    # no solutions: x^2 + y^2 = 1 and x + y = 3 (the line misses the circle)
    e1 = [((2, 0), 1), ((0, 2), 1), ((0, 0), -1)]
    e2 = [((1, 0), 1), ((0, 1), 1), ((0, 0), -3)]
    # y = 3 - x: x^2 + (3-x)^2 - 1 = 2x^2 - 6x + 8, discriminant 36 - 64 < 0
    assert 36 - 4 * 2 * 8 < 0
    cases.append({"name": "line_misses_circle", "note": "no real solution (discriminant 36 - 64 < 0): the whole box must be excluded",
                  "system": sys2(e1, e2, (-4, 4)), "policy": {"max_boxes": 50000, "minimum_width": 1e-8},
                  "roots": [], "distinct_real_roots": 0})
    return cases


def main():
    document = {
        "note": "Generated by cyclic_general_systems_reference.py with mpmath at 60 digits and exact Fractions; "
                "no product code and no retrieved paper was consulted.",
        "ode": ode_cases(),
        "rng": rng_cases(),
        "sde": sde_cases(),
        "krawczyk": krawczyk_cases(),
    }
    json.dump(document, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
