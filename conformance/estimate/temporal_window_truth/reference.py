# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the temporal window fixture.

Two stationary binary structural processes, every noise independent and uniform on equally likely
bins. Enumerating the noise cells of the unrolled structural model with exact rational arithmetic
gives, from the structural equations alone (no identification formula, no Antecedent code), the
effects of an intervention at a period of the window on an outcome at a later period, and the
exact law of the observed window.

contrast (columns x, y, w; a window of three periods, the history period, the treated period and
the outcome period). x and w are fair coins at every period; the outcome follows both of its own
period and its own past:

    y(t) = 1[Uy(t) < r(x(t), w(t), y(t-1))]      Uy uniform on four bins

    r(x, w, y(t-1) = 0):  (0,0) -> 0   (1,0) -> 2   (0,1) -> 1   (1,1) -> 1      (sum 4)
    r(x, w, y(t-1) = 1):  (0,0) -> 2   (1,0) -> 4   (0,1) -> 3   (1,1) -> 3      (sum 12)

The mean of r over (x, w) is 1 and 3, so the outcome chain has transition probabilities 1/4 and 3/4
and the stationary law P(y = 1) = 1/2: the cell that opens the window draws y(-1) as a fair coin.
The treatment x(1) reaches y(2) through y(1) (x(1) -> y(1) -> y(2)); a sustained policy on
x(1), x(2) also sets the parent x(2) of y(2). The conditional contrasts condition on w(1), read at
the first treated period, which no treated period can reach.

mediation (columns c, x, w, m, y; a window of two periods, the history period and the pulse
period, the outcome read at the pulse period). c and w are fair coins, the treatment follows the
previous period's c, the mediator follows the treatment and w, and the outcome follows the
treatment and the mediator:

    x(t) = 1[Ux(t) < 1 + 2 c(t-1)]     Ux uniform on four bins
    m(t) = 1[Um(t) < q(x(t), w(t))]    Um uniform on four bins, q = {(0,0): 1, (0,1): 2, (1,0): 2, (1,1): 3}
    y(t) = 1[Uy(t) < s(x(t), m(t))]    Uy uniform on two bins,  s = {(0,0): 0, (1,0): 1, (0,1): 1, (1,1): 1}

The total effect E[Y(1, M(1))] - E[Y(0, M(0))], the natural direct effect E[Y(1, M(0))] -
E[Y(0, M(0))] and the pure natural indirect effect E[Y(0, M(1))] - E[Y(0, M(0))] are evaluated by
setting the treatment argument of the outcome equation and the treatment argument of the
mediator equation separately on every noise cell of the pulse period.

Each noise cell of a window is one equally likely draw of the whole window, so the exact law of the
window is the count of the cells that give each observed window. A series whose consecutive
windows have exactly that law is an Eulerian circuit of the multigraph whose edges are the observed
windows, drawn in a fixed pseudo-random order so that neither half of the circuit carries a
different law by construction. The series stored in the fixture is that circuit; the check
re-counts its windows and requires the exact counts.

The contrast series repeats the exact window law sixteen times (the gcd-reduced counts times
sixteen), so a random contiguous stretch or a column rotation of a refit still holds every
conditioning cell of the full table.

A planted regime changes the effect of the treatment in the outcome equation and keeps both
deterministic cells, the stationary law and so the support of the windows; the fixture stores its
exact-law series too, for the stationarity check, and its contrasts.

bow: the two structural models in which a latent cause u of the treatment and the outcome gives
the same observed law with and without an effect of the treatment, so the effect is not a function
of the observed law (the hedge a window with a bow arc holds).

    uv run --frozen --script conformance/estimate/temporal_window_truth/reference.py --check
    uv run --frozen --script conformance/estimate/temporal_window_truth/reference.py --self-test

Standard library only; imports no Antecedent code and no statistics package.

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import sys
from fractions import Fraction as F
from functools import reduce
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "expected.json"
ALPHABET = "0123456789abcdefghijklmnopqrstuv"
CONTRAST_SCALE = 16

# ---------------------------------------------------------------------------------------------
# The contrast process.
# ---------------------------------------------------------------------------------------------

R_PREV0 = {(0, 0): 0, (1, 0): 2, (0, 1): 1, (1, 1): 1}
R_PREV1 = {(0, 0): 2, (1, 0): 4, (0, 1): 3, (1, 1): 3}
# The planted regime keeps both deterministic cells and the stationary law (the sums are 4 and 12)
# and changes the effect of the treatment, so both regimes have the same support.
R_PREV0_PLANTED = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 1}
R_PREV1_PLANTED = {(0, 0): 3, (1, 0): 4, (0, 1): 3, (1, 1): 2}
CONTRAST_PERIODS = 3


def contrast_threshold(x: int, w: int, y_prev: int, flipped: bool) -> int:
    if flipped:
        table = R_PREV1_PLANTED if y_prev else R_PREV0_PLANTED
    else:
        table = R_PREV1 if y_prev else R_PREV0
    return table[(x, w)]


def contrast_run(cell, do: dict[int, int], flipped: bool):
    """The window of one noise cell under an intervention: period -> value set on x."""
    y_prev, periods = cell
    xs, ws, ys = [], [], []
    for t, (x, w, uy) in enumerate(periods):
        x = do.get(t, x)
        y = int(uy < contrast_threshold(x, w, y_prev, flipped))
        xs.append(x)
        ws.append(w)
        ys.append(y)
        y_prev = y
    return xs, ys, ws


def contrast_cells():
    per_period = list(itertools.product((0, 1), (0, 1), range(4)))
    for y_prev in (0, 1):
        for periods in itertools.product(per_period, repeat=CONTRAST_PERIODS):
            yield (y_prev, periods)


def contrast_truths(flipped: bool) -> dict[str, F]:
    cells = list(contrast_cells())
    n = len(cells)

    def mean_outcome(do: dict[int, int], w1: int | None = None) -> F:
        rows = [
            contrast_run(cell, do, flipped)
            for cell in cells
            if w1 is None or cell[1][1][1] == w1
        ]
        return F(sum(ys[2] for _, ys, _ in rows), len(rows))

    def conditional(treated: tuple[int, ...]) -> F:
        total = F(0)
        for w1 in (0, 1):
            weight = F(sum(1 for cell in cells if cell[1][1][1] == w1), n)
            total += weight * (
                mean_outcome({t: 1 for t in treated}, w1)
                - mean_outcome({t: 0 for t in treated}, w1)
            )
        return total

    return {
        "pulse": mean_outcome({1: 1}) - mean_outcome({1: 0}),
        "sustained": mean_outcome({1: 1, 2: 1}) - mean_outcome({1: 0, 2: 0}),
        "conditional_pulse": conditional((1,)),
        "conditional_sustained": conditional((1, 2)),
    }


def contrast_window_counts(flipped: bool) -> dict[tuple, int]:
    counts: dict[tuple, int] = {}
    for cell in contrast_cells():
        xs, ys, ws = contrast_run(cell, {}, flipped)
        key = tuple(x + 2 * y + 4 * w for x, y, w in zip(xs, ys, ws, strict=True))
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------------------------
# The mediation process.
# ---------------------------------------------------------------------------------------------

Q_MEDIATOR = {(0, 0): 1, (0, 1): 2, (1, 0): 2, (1, 1): 3}
S_OUTCOME = {(0, 0): 0, (1, 0): 1, (0, 1): 1, (1, 1): 1}
MEDIATION_PERIODS = 2


def mediation_cells():
    per_period = list(itertools.product((0, 1), range(4), (0, 1), range(4), range(2)))
    for c_prev in (0, 1):
        for periods in itertools.product(per_period, repeat=MEDIATION_PERIODS):
            yield (c_prev, periods)


def mediation_run(cell):
    """Observed window and, at the pulse period, the nested potential outcomes."""
    c_prev, periods = cell
    states = []
    nested = None
    for t, (c, ux, w, um, uy) in enumerate(periods):
        x = int(ux < 1 + 2 * c_prev)
        m = int(um < Q_MEDIATOR[(x, w)])
        y = int(uy < S_OUTCOME[(x, m)])
        states.append(c + 2 * x + 4 * w + 8 * m + 16 * y)
        if t == MEDIATION_PERIODS - 1:
            mediator_of = {a: int(um < Q_MEDIATOR[(a, w)]) for a in (0, 1)}
            nested = {
                (a, b): int(uy < S_OUTCOME[(a, mediator_of[b])]) for a in (0, 1) for b in (0, 1)
            }
        c_prev = c
    return tuple(states), nested


def mediation_truths() -> dict[str, F]:
    runs = [mediation_run(cell)[1] for cell in mediation_cells()]
    n = len(runs)

    def mean(a: int, b: int) -> F:
        return F(sum(nested[(a, b)] for nested in runs), n)

    return {
        "total_effect": mean(1, 1) - mean(0, 0),
        "natural_direct_effect": mean(1, 0) - mean(0, 0),
        "natural_indirect_effect": mean(0, 1) - mean(0, 0),
    }


def mediation_window_counts() -> dict[tuple, int]:
    counts: dict[tuple, int] = {}
    for cell in mediation_cells():
        key = mediation_run(cell)[0]
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------------------------
# Eulerian circuit with the exact window law.
# ---------------------------------------------------------------------------------------------


class XorShift:
    """A fixed pseudo-random stream (xorshift64*), independent of the interpreter's generator."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def below(self, bound: int) -> int:
        s = self.state
        s ^= s >> 12
        s ^= (s << 25) & 0xFFFFFFFFFFFFFFFF
        s ^= s >> 27
        self.state = s
        return ((s * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF) % bound


def reduce_counts(counts: dict[tuple, int], scale: int = 1) -> dict[tuple, int]:
    """The counts divided by their gcd and multiplied by `scale`. A refit on a stretch of the
    series or on rotated columns needs every conditioning cell the full table has, so the rarest
    window must occur `scale` times at least."""
    common = reduce(math.gcd, counts.values())
    return {key: n // common * scale for key, n in counts.items()}


def euler_series(counts: dict[tuple, int], seed: int) -> list[int]:
    """States of a series whose consecutive windows are the keys of `counts`, each exactly
    `counts[key]` times, as the edges of an Eulerian circuit through (window - 1)-tuples."""
    out: dict[tuple, list[list]] = {}
    for key, n in sorted(counts.items()):
        out.setdefault(key[:-1], []).append([key[1:], n])
    rng = XorShift(seed)
    start = min(out)
    stack = [start]
    circuit: list[tuple] = []
    while stack:
        node = stack[-1]
        edges = [edge for edge in out.get(node, []) if edge[1] > 0]
        if edges:
            pick = rng.below(sum(edge[1] for edge in edges))
            for edge in edges:
                if pick < edge[1]:
                    edge[1] -= 1
                    stack.append(edge[0])
                    break
                pick -= edge[1]
        else:
            circuit.append(stack.pop())
    circuit.reverse()
    nodes = circuit
    states = list(nodes[0]) + [node[-1] for node in nodes[1:]]
    return states


def window_counts_of(states: list[int], length: int) -> dict[tuple, int]:
    counts: dict[tuple, int] = {}
    for i in range(len(states) - length + 1):
        key = tuple(states[i : i + length])
        counts[key] = counts.get(key, 0) + 1
    return counts


def encode(states: list[int]) -> str:
    return "".join(ALPHABET[s] for s in states)


def decode(text: str) -> list[int]:
    return [ALPHABET.index(ch) for ch in text]


# ---------------------------------------------------------------------------------------------
# Table-only checks and the bow witness.
# ---------------------------------------------------------------------------------------------


def mediation_functionals_from_windows(counts: dict[tuple, int]) -> dict[str, F]:
    """Total, natural direct and pure natural indirect effect of the pulse period from the
    exact window law alone: the textbook mediation functionals of the last period's table (the
    treatment has the previous period's c as its only parent, so no adjustment is needed)."""
    rows = []
    for (s0, s1), n in counts.items():
        rows.append(((s1 >> 1) & 1, (s1 >> 3) & 1, (s1 >> 4) & 1, n))

    def prob(m: int, x: int) -> F:
        num = sum(n for xx, mm, _, n in rows if xx == x and mm == m)
        den = sum(n for xx, _, _, n in rows if xx == x)
        return F(num, den)

    def mean_y(x: int, m: int) -> F:
        sel = [(yy, n) for xx, mm, yy, n in rows if xx == x and mm == m]
        return F(sum(y * n for y, n in sel), sum(n for _, n in sel))

    def mean_total(x: int) -> F:
        sel = [(yy, n) for xx, _, yy, n in rows if xx == x]
        return F(sum(y * n for y, n in sel), sum(n for _, n in sel))

    nde = sum(prob(m, 0) * (mean_y(1, m) - mean_y(0, m)) for m in (0, 1))
    nie = sum(mean_y(0, m) * (prob(m, 1) - prob(m, 0)) for m in (0, 1))
    return {
        "total_effect": mean_total(1) - mean_total(0),
        "natural_direct_effect": nde,
        "natural_indirect_effect": nie,
    }


def bow_witness() -> dict:
    """Two structural models over (x, y1, y2) with one latent fair coin u and the same observed
    law whose effects of x(1) on y2 differ: not identified."""

    def law(effect: bool) -> dict[tuple[int, int, int], F]:
        out: dict[tuple[int, int, int], F] = {}
        for u in (0, 1):
            x = u
            y1 = x if effect else u
            y2 = y1
            out[(x, y1, y2)] = out.get((x, y1, y2), F(0)) + F(1, 2)
        return out

    def do_mean(effect: bool, a: int) -> F:
        total = F(0)
        for u in (0, 1):
            y1 = a if effect else u
            total += F(1, 2) * y1
        return total

    return {
        "observed_law_without_effect": {str(k): str(v) for k, v in sorted(law(False).items())},
        "observed_law_with_effect": {str(k): str(v) for k, v in sorted(law(True).items())},
        "effect_without": str(do_mean(False, 1) - do_mean(False, 0)),
        "effect_with": str(do_mean(True, 1) - do_mean(True, 0)),
    }


def numbers(values: dict[str, F]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items()}


def exact(values: dict[str, F]) -> dict[str, str]:
    return {key: str(value) for key, value in values.items()}


# ---------------------------------------------------------------------------------------------
# The fixture.
# ---------------------------------------------------------------------------------------------


def build() -> dict:
    straight = contrast_truths(False)
    planted = contrast_truths(True)
    straight_counts = reduce_counts(contrast_window_counts(False), CONTRAST_SCALE)
    planted_counts = reduce_counts(contrast_window_counts(True), CONTRAST_SCALE)
    mediation_counts = reduce_counts(mediation_window_counts())
    truths = mediation_truths()
    contrast_series = euler_series(straight_counts, 0x5EED_0001)
    planted_series = euler_series(planted_counts, 0x5EED_0002)
    mediation_series = euler_series(mediation_counts, 0x5EED_0003)
    return {
        "schema_version": 1,
        "case": "temporal_window_truth",
        "contrast": {
            "columns": ["x", "y", "w"],
            "state": "x + 2 y + 4 w per period",
            "periods": CONTRAST_PERIODS,
            "windows": sum(straight_counts.values()),
            "series": encode(contrast_series),
            "planted_series": encode(planted_series),
            "truth": numbers(straight),
            "truth_exact": exact(straight),
            "planted_truth": numbers(planted),
            "query": {
                "treatment": "x",
                "outcome": "y",
                "modifier": "w",
                "horizon_steps": 2,
                "sustained_until": 1,
            },
        },
        "mediation": {
            "columns": ["c", "x", "w", "m", "y"],
            "state": "c + 2 x + 4 w + 8 m + 16 y per period",
            "periods": MEDIATION_PERIODS,
            "windows": sum(mediation_counts.values()),
            "series": encode(mediation_series),
            "truth": numbers(truths),
            "truth_exact": exact(truths),
            "query": {"treatment": "x", "outcome": "y", "mediators": ["m"], "horizon": 1},
        },
        "bow_witness": bow_witness(),
        "frequentist": {"estimator": "functional.effect", "absolute_tolerance": 1e-9},
        "bayesian": {
            "estimator": "functional.effect",
            "posterior_mean_tolerance": 0.03,
            "note": "Bayesian-bootstrap posterior mean within tolerance of the truth; declared acceptance band",
        },
    }


def close(got, want, label, problems, tol=1e-12):
    if not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {float(got)!r} != fixture {want!r}")


def check(document: dict) -> list[str]:
    problems: list[str] = []
    contrast = document["contrast"]
    straight = contrast_truths(False)
    planted = contrast_truths(True)
    for key, want in contrast["truth"].items():
        close(straight[key], want, f"contrast {key}", problems)
    for key, want in contrast["planted_truth"].items():
        close(planted[key], want, f"planted {key}", problems)
        if abs(planted[key] - straight[key]) < F(1, 20):
            problems.append(f"planted {key} must differ materially from the contrast")
    # The conditional contrast is the mean over the modifier of the conditional contrasts, which
    # equals the unconditional one because the modifier precedes the treated period.
    close(straight["conditional_pulse"], straight["pulse"], "conditional pulse", problems)
    close(straight["conditional_sustained"], straight["sustained"], "conditional sustained", problems)
    if min(abs(straight["pulse"]), abs(straight["sustained"])) < F(1, 100):
        problems.append("the pulse and sustained contrasts must be material")
    if abs(straight["pulse"] - straight["sustained"]) < F(1, 100):
        problems.append("the sustained contrast must differ from the pulse contrast")
    if set(contrast_window_counts(False)) != set(contrast_window_counts(True)):
        problems.append("the planted regime must have the support of the process")
    for name, flipped in (("series", False), ("planted_series", True)):
        counts = reduce_counts(contrast_window_counts(flipped), CONTRAST_SCALE)
        if window_counts_of(decode(contrast[name]), CONTRAST_PERIODS) != counts:
            problems.append(f"contrast {name}: the windows are not the exact law of the process")
    if contrast["windows"] != sum(reduce_counts(contrast_window_counts(False), CONTRAST_SCALE).values()):
        problems.append("contrast windows count")
    mediation = document["mediation"]
    truths = mediation_truths()
    for key, want in mediation["truth"].items():
        close(truths[key], want, f"mediation {key}", problems)
    counts = reduce_counts(mediation_window_counts())
    if window_counts_of(decode(mediation["series"]), MEDIATION_PERIODS) != counts:
        problems.append("mediation series: the windows are not the exact law of the process")
    plug_in = mediation_functionals_from_windows(counts)
    for key, want in mediation["truth"].items():
        close(plug_in[key], want, f"mediation {key} from the window law", problems)
    t = mediation["truth"]
    if min(abs(t["natural_direct_effect"]), abs(t["natural_indirect_effect"])) < 0.02:
        problems.append("the direct and indirect effects must be material")
    if abs(t["total_effect"] - t["natural_direct_effect"] - t["natural_indirect_effect"]) < 1e-3:
        problems.append("an interaction should separate the pure indirect effect")
    bow = document["bow_witness"]
    if bow["observed_law_without_effect"] != bow["observed_law_with_effect"]:
        problems.append("the bow witness models must share one observed law")
    if bow["effect_without"] == bow["effect_with"]:
        problems.append("the bow witness models must differ in the effect")
    if bow != bow_witness():
        problems.append("the bow witness is not the derived one")
    return problems


MUTATIONS = [
    (
        "a window of the series changed",
        lambda d: d["contrast"].__setitem__(
            "series", "0" + d["contrast"]["series"][1:] if d["contrast"]["series"][0] != "0" else "1" + d["contrast"]["series"][1:]
        ),
    ),
    ("pulse truth changed", lambda d: d["contrast"]["truth"].__setitem__("pulse", 0.3)),
    ("sustained truth changed", lambda d: d["contrast"]["truth"].__setitem__("sustained", 0.3)),
    ("direct truth changed", lambda d: d["mediation"]["truth"].__setitem__("natural_direct_effect", 0.3)),
    ("indirect truth changed", lambda d: d["mediation"]["truth"].__setitem__("natural_indirect_effect", 0.3)),
    ("planted truth changed", lambda d: d["contrast"]["planted_truth"].__setitem__("pulse", 0.3)),
]


def self_test(document: dict) -> list[str]:
    failures = []
    if check(document):
        failures.append("the unmodified fixture is rejected")
    for label, mutate in MUTATIONS:
        wrong = copy.deepcopy(document)
        mutate(wrong)
        if not check(wrong):
            failures.append(f"accepted: {label}")
    return failures


def main() -> int:
    if "--write" in sys.argv:
        retained = json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.is_file() else {}
        document = {"oracle": retained.get("oracle", {}), **build()}
        FIXTURE.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
        return 0
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problems = self_test(document) if "--self-test" in sys.argv else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
