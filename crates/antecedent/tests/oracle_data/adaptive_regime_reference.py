"""Exact rational two-period adaptive g-formula and delete-subject covariance."""

import json
from fractions import Fraction as F
from pathlib import Path

rows = []
for a in range(2):
    for state in range(2):
        for b in range(2):
            for _ in range(6 if a == state else 2):
                y = 2 * a + 3 * b + 4 * state + a * b
                rows.append((a, state, b, F(y)))


def world(sample, first, adaptive):
    first_arm = [row for row in sample if row[0] == first]
    result = F(0)
    for state in range(2):
        state_rows = [row for row in first_arm if row[1] == state]
        action = state if adaptive else 0
        outcome_rows = [row for row in state_rows if row[2] == action]
        result += F(len(state_rows), len(first_arm)) * sum(row[3] for row in outcome_rows) / len(outcome_rows)
    return result


def values(sample):
    a, b = world(sample, 1, True), world(sample, 0, False)
    return [a, 2 * a, b, 2 * b]


n = len(rows)
point = values(rows)
leave = [values(rows[:i] + rows[i + 1 :]) for i in range(n)]
center = [sum(row[j] for row in leave) / n for j in range(4)]
cov = [
    F(n - 1, n) * sum((row[i] - center[i]) * (row[j] - center[j]) for row in leave)
    for i in range(4)
    for j in range(4)
]
Path(__file__).with_suffix('.json').write_text(
    json.dumps({'means': [float(x) for x in point], 'covariance': [float(x) for x in cov]}, indent=2) + '\n'
)
