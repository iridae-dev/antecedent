# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Fisher-z interval and Student-t p-value of a partial correlation, in 50-digit arithmetic.

Regenerate with:  uv run --script fisher_z_reference.py > fisher_z_reference.json

For a partial correlation r with df = n - |Z| - 2 residual degrees of freedom, Fisher's variance
stabilising transform gives  atanh(r) ~ N(atanh(rho), 1 / (df - 1))  (n - |Z| - 3 = df - 1), so the
two-sided interval is tanh(atanh(r) -+ z_{(1+level)/2} / sqrt(df - 1)) with z the exact normal
quantile (mpmath's inverse error function), and the test of rho = 0 refers
t = r sqrt(df / (1 - r^2)) to Student's t with df degrees of freedom: p = I_{df/(df+t^2)}(df/2, 1/2).
Rows are [r, df, level, lower, upper, p]; the interval is omitted for df <= 1.
Repository-authored (MIT OR Apache-2.0).
"""
import json

import mpmath as mp

mp.mp.dps = 50

rs = ["0", "1e-8", "-0.05", "0.1", "-0.5", "0.9", "-0.99", "0.999999", "0.99999999999"]
dfs = [1, 2, 3, 5, 10, 100, 10000, 1000000]
levels = ["0.5", "0.9", "0.95", "0.99", "0.999999"]

rows = []
for rs_ in rs:
    r = mp.mpf(float(mp.mpf(rs_)))  # the binary64 value the product receives
    for df in dfs:
        # p-value (the product clamps |r| away from 1 by 1e-15 only; none of these rows reach it)
        t2 = r * r * df / (1 - r * r)
        a, x = mp.mpf(df) / 2, mp.mpf(df) / (df + t2)
        try:
            p = mp.betainc(a, mp.mpf(1) / 2, 0, x, regularized=True)
        except ValueError:
            # tiny x: the defining series x^a / (a B(a, 1/2)) sum_k (a)_k (1/2)_k / ((a+1)_k k!) x^k
            term, total, k = mp.mpf(1), mp.mpf(1), 0
            while abs(term) > mp.mpf(10) ** -60 * abs(total):
                term *= (a + k) * (mp.mpf(1) / 2 + k) / ((a + 1 + k) * (k + 1)) * x
                total += term
                k += 1
            p = mp.exp(a * mp.log(x) - mp.log(a) - mp.log(mp.beta(a, mp.mpf(1) / 2))) * total
        for level in levels:
            lv = mp.mpf(float(mp.mpf(level)))  # the binary64 value the product receives
            if df > 1:
                zc = mp.sqrt(2) * mp.erfinv(lv)  # two-sided quantile z_{(1+level)/2}
                z = mp.atanh(r)
                se = 1 / mp.sqrt(df - 1)
                lo, hi = mp.tanh(z - zc * se), mp.tanh(z + zc * se)
                rows.append([float(r), df, float(lv), float(lo), float(hi), float(p)])
            else:
                rows.append([float(r), df, float(lv), None, None, float(p)])
print(json.dumps({"rows": rows}))
