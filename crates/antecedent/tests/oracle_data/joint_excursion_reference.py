"""Exact rational reference for two lag-one excursion outcomes.

Run directly to regenerate the JSON; this is deterministic arithmetic, not calibration.
Each unit has three observed decisions and a terminal follow-up. The four
coordinates are treated outcome 0/1 followed by control outcome 0/1.
"""
import json
from fractions import Fraction as F
from pathlib import Path

N = 32
PERIODS = 4
sums = []
for unit in range(N):
    row = [F(0) for _ in range(4)]
    for period in range(PERIODS - 1):
        treated = ((unit * 7 + period * 5) % 11) < 5
        y = F(2) + F(unit % 7, 5) + F(period, 4) + (F(3, 2) if treated else 0)
        z = F(4) + F(unit % 5, 3) - F(period, 5) + (F(2, 3) if treated else 0)
        offset = 0 if treated else 2
        row[offset] += 2 * y
        row[offset + 1] += 2 * z
    sums.append(row)
denominator = N * (PERIODS - 1)
means = [sum(row[j] for row in sums) / denominator for j in range(4)]
covariance = [
    sum((row[i] - 3 * means[i]) * (row[j] - 3 * means[j]) for row in sums)
    * F(N, (N - 1) * denominator**2)
    for i in range(4) for j in range(4)
]
result = {"means": [float(v) for v in means], "covariance": [float(v) for v in covariance]}
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
