"""Exact rational matched-pair arm means and covariance, independent of Rust."""

import json
from fractions import Fraction as F
from pathlib import Path

pairs = []
for pair in range(30):
    values = []
    for side in (0, 1):
        i = 2 * pair + side
        values.append(
            (F(2) + F(i % 7, 5) + F(pair, 10), F(4) + F(i % 5, 3) - F(pair, 20))
        )
    treated = pair % 2
    control = 1 - treated
    pairs.append([*values[treated], *values[control]])
means = [sum(row[j] for row in pairs) / len(pairs) for j in range(4)]
covariance = [
    sum((row[i] - means[i]) * (row[j] - means[j]) for row in pairs)
    / (len(pairs) * (len(pairs) - 1))
    for i in range(4)
    for j in range(4)
]
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "means": [float(x) for x in means],
            "covariance": [float(x) for x in covariance],
        },
        indent=2,
    )
    + "\n"
)
