"""Exact optima and vertex sets of small linear programs by brute-force basis enumeration.

Regenerate with:  python3 linear_program_reference.py > linear_program_reference.json

Every program is `minimize c . x` subject to linear constraints (at most, at least, equal) and
variable bounds.  Each candidate vertex is the solution of n of the hyperplanes {a_i . x = b_i}
and {x_j = l_j}, {x_j = u_j}; the n x n system is solved in exact rational arithmetic
(fractions.Fraction, Gauss-Jordan), singular choices are skipped, and a candidate is kept when it
satisfies every constraint and bound exactly.  The optimum is the smallest objective over the kept
vertices; a program with no feasible vertex is reported infeasible.  Every program here has a
region that contains no line, so a non-empty region has a vertex and the minimum over the
vertices is the minimum over the region (every program is bounded in its objective by
construction: finite bounds, or constraints that close the region).
Vertex sets are the distinct kept vertices in lexicographic order.

The pseudo-random programs come from a fixed 64-bit linear congruential generator with small
integer coefficients in dimensions 2 to 5; most are built around an integer point of their box
so that they are feasible.  Standard library only.
Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction
from itertools import combinations

_state = 0x9E3779B97F4A7C15


def next_int(low, high):
    """Uniform-ish integer in [low, high] from a fixed LCG."""
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return low + (_state >> 33) % (high - low + 1)


def hyperplanes(constraints, lower, upper):
    n = len(lower)
    planes = [([Fraction(v) for v in c["coefficients"]], Fraction(c["rhs"])) for c in constraints]
    for j in range(n):
        unit = [Fraction(1) if k == j else Fraction(0) for k in range(n)]
        if lower[j] is not None:
            planes.append((unit, Fraction(lower[j])))
        if upper[j] is not None:
            planes.append((unit, Fraction(upper[j])))
    return planes


def exact_solve(rows, rhs):
    """Gauss-Jordan on an n x n rational system; None when singular."""
    n = len(rows)
    a = [list(rows[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col] != 0), None)
        if pivot is None:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r != col and a[r][col] != 0:
                f = a[r][col] / a[col][col]
                a[r] = [a[r][k] - f * a[col][k] for k in range(n + 1)]
    return [a[i][n] / a[i][i] for i in range(n)]


def feasible(x, constraints, lower, upper):
    for c in constraints:
        lhs = sum(Fraction(v) * xi for v, xi in zip(c["coefficients"], x))
        rhs = Fraction(c["rhs"])
        if c["sense"] == "at_most" and lhs > rhs:
            return False
        if c["sense"] == "at_least" and lhs < rhs:
            return False
        if c["sense"] == "equal" and lhs != rhs:
            return False
    for j, xj in enumerate(x):
        if lower[j] is not None and xj < lower[j]:
            return False
        if upper[j] is not None and xj > upper[j]:
            return False
    return True


def exact_vertices(constraints, lower, upper):
    n = len(lower)
    planes = hyperplanes(constraints, lower, upper)
    found = set()
    for basis in combinations(range(len(planes)), n):
        x = exact_solve([planes[k][0] for k in basis], [planes[k][1] for k in basis])
        if x is not None and feasible(x, constraints, lower, upper):
            found.add(tuple(x))
    return sorted(found)


def text(f):
    return f"{f.numerator}/{f.denominator}"


def program_record(name, objective, constraints, lower, upper):
    verts = exact_vertices(constraints, lower, upper)
    record = {
        "name": name,
        "objective": objective,
        "constraints": constraints,
        "lower": lower,
        "upper": upper,
    }
    if not verts:
        record["status"] = "infeasible"
    else:
        best = min(sum(Fraction(c) * xi for c, xi in zip(objective, v)) for v in verts)
        record["status"] = "optimal"
        record["objective_value"] = float(best)
        record["objective_exact"] = text(best)
    return record, verts


def polytope_record(name, constraints, lower, upper, verts):
    return {
        "name": name,
        "constraints": constraints,
        "lower": lower,
        "upper": upper,
        "vertices": [[float(v) for v in vertex] for vertex in verts],
        "vertices_exact": [[text(v) for v in vertex] for vertex in verts],
    }


def con(coefficients, sense, rhs):
    return {"coefficients": coefficients, "sense": sense, "rhs": rhs}


HAND = [
    # Two-variable production program: max 3x + 5y, x <= 4, 2y <= 12, 3x + 2y <= 18.
    ("production_two_goods", [-3, -5],
     [con([1, 0], "at_most", 4), con([0, 2], "at_most", 12), con([3, 2], "at_most", 18)],
     [0, 0], [None, None]),
    # Beale's cycling example (scaled by 100 to integers); minimum -1/20.
    ("beale_cycling", [-75, 15000, -2, 600],
     [con([25, -6000, -4, 900], "at_most", 0), con([50, -9000, -2, 300], "at_most", 0),
      con([0, 0, 1, 0], "at_most", 1)],
     [0, 0, 0, 0], [None, None, None, None]),
    # Diet-style covering program with at-least constraints.
    ("covering_at_least", [2, 3, 4],
     [con([1, 1, 0], "at_least", 3), con([0, 1, 1], "at_least", 4), con([1, 0, 1], "at_least", 5)],
     [0, 0, 0], [None, None, None]),
    # Equality constraints with a free variable closed by the constraints.
    ("equality_with_free_variable", [1, 2, -1],
     [con([1, 1, 1], "equal", 6), con([1, -1, 0], "equal", 1), con([0, 0, 1], "at_most", 4)],
     [None, 0, 0], [None, 10, 10]),
    # Free variables in a simplex-shaped region: x + y <= 2, x >= -1, y >= -1.
    ("free_variables_triangle", [1, 1],
     [con([1, 1], "at_most", 2), con([1, 0], "at_least", -1), con([0, 1], "at_least", -1)],
     [None, None], [None, None]),
    # Degenerate vertex: three constraints through the optimum in two dimensions.
    ("degenerate_vertex", [-1, -1],
     [con([1, 0], "at_most", 1), con([0, 1], "at_most", 1), con([1, 1], "at_most", 2),
      con([2, 1], "at_most", 3)],
     [0, 0], [None, None]),
    # Duplicate and redundant constraints.
    ("redundant_constraints", [-1, -2],
     [con([1, 2], "at_most", 4), con([2, 4], "at_most", 8), con([1, 2], "at_most", 4),
      con([1, 0], "at_most", 10)],
     [0, 0], [3, 3]),
    # Every variable fixed by its bounds.
    ("all_fixed", [3, -2, 1], [con([1, 1, 1], "at_most", 10)], [1, 2, 3], [1, 2, 3]),
    # Fixed bounds that violate a constraint.
    ("all_fixed_infeasible", [1, 1], [con([1, 1], "at_least", 5)], [1, 1], [1, 1]),
    # x >= 3 and x <= 2.
    ("contradictory_pair", [1, 0],
     [con([1, 0], "at_least", 3), con([1, 0], "at_most", 2)], [None, 0], [None, 1]),
    # Equalities that cannot hold together.
    ("inconsistent_equalities", [0, 0, 0],
     [con([1, 1, 0], "equal", 1), con([0, 1, 1], "equal", 1), con([1, 2, 1], "equal", 3)],
     [-5, -5, -5], [5, 5, 5]),
    # No constraints at all.
    ("bounds_only", [1, -1, 2], [], [-1, -2, 0], [1, 2, 4]),
    # Negative lower bounds and a mixed-sign objective.
    ("negative_box_cut", [1, 1, 1],
     [con([1, 1, 1], "at_least", -2), con([1, -1, 0], "at_most", 1)],
     [-3, -3, -3], [3, 3, 3]),
]

POLYTOPES = [
    ("unit_cube", [], [0, 0, 0], [1, 1, 1]),
    ("standard_simplex", [con([1, 1, 1], "at_most", 1)], [0, 0, 0], [None, None, None]),
    ("cube_cut_by_plane", [con([1, 1, 1], "at_most", 2)], [0, 0, 0], [1, 1, 1]),
    ("degenerate_pyramid",
     [con([1, 0, 1], "at_most", 1), con([-1, 0, 1], "at_most", 1), con([0, 1, 1], "at_most", 1),
      con([0, -1, 1], "at_most", 1), con([1, 1, 2], "at_most", 2), con([1, 1, 1], "at_most", 3)],
     [-1, -1, 0], [1, 1, None]),
    ("square_with_duplicates",
     [con([1, 0], "at_most", 1), con([2, 0], "at_most", 2), con([0, 1], "at_most", 1)],
     [0, 0], [None, None]),
    ("hexagon",
     [con([1, 1], "at_most", 3), con([-1, -1], "at_most", 3), con([1, -1], "at_most", 2),
      con([-1, 1], "at_most", 2)],
     [-2, -2], [2, 2]),
    ("segment_from_equality", [con([1, 1], "equal", 1)], [0, 0], [None, None]),
    ("empty_region", [con([1, 1], "at_least", 3)], [0, 0], [1, 1]),
]


def random_program(index):
    """Dimension cycles through 2..5.  Most programs place their constraints around an integer
    point of the box so they are feasible; every fourth draws its right-hand sides freely."""
    n = 2 + index % 4
    m = next_int(1, 6)
    objective = [next_int(-4, 4) for _ in range(n)]
    lower = [next_int(-3, 0) for _ in range(n)]
    upper = [lo + next_int(0, 5) for lo in lower]
    anchor = [next_int(lo, hi) for lo, hi in zip(lower, upper)]
    constraints = []
    for _ in range(m):
        coefficients = [next_int(-3, 3) for _ in range(n)]
        pick = next_int(0, 9)
        sense = "equal" if pick == 0 else ("at_least" if pick <= 3 else "at_most")
        if index % 4 == 3:
            rhs = next_int(-4, 8)
        else:
            at_anchor = sum(a * p for a, p in zip(coefficients, anchor))
            slack = next_int(0, 3)
            rhs = at_anchor + (slack if sense == "at_most" else -slack if sense == "at_least" else 0)
        constraints.append(con(coefficients, sense, rhs))
    return f"random_{index:02d}", objective, constraints, lower, upper


def main():
    programs = []
    polytopes = []
    for name, objective, constraints, lower, upper in HAND:
        record, _ = program_record(name, objective, constraints, lower, upper)
        programs.append(record)
    for index in range(36):
        name, objective, constraints, lower, upper = random_program(index)
        record, verts = program_record(name, objective, constraints, lower, upper)
        programs.append(record)
        polytopes.append(polytope_record(name, constraints, lower, upper, verts))
    for name, constraints, lower, upper in POLYTOPES:
        polytopes.append(
            polytope_record(name, constraints, lower, upper,
                            exact_vertices(constraints, lower, upper)))
    print(json.dumps({"programs": programs, "polytopes": polytopes}, indent=1))


if __name__ == "__main__":
    main()
