# ADMG front-door InterventionalDistribution known truth

**Suite path:** `conformance/estimate/admg_frontdoor_distribution`

A binary structural causal model with an unobserved confounder `U`:

- `U ~ Bernoulli(0.5)`
- `P(T=1 | U=0) = 0.2`, `P(T=1 | U=1) = 0.8`
- `P(M=1 | T=0) = 0.25`, `P(M=1 | T=1) = 0.75`
- `P(Y=1 | M, U) = 0.1 + 0.5·M + 0.3·U`

Marginalizing `U` gives the front-door ADMG `T -> M -> Y`, `T <-> Y`. The
interventional truth comes from the SCM directly (not from the formula under
test): `P(Y=1 | do(T=t)) = Σ_m P(m | t) · (0.1 + 0.5·m + 0.3·E[U]) =
0.25 + 0.5·P(M=1 | t)`, so `P(Y=1 | do(T=0)) = 0.375` and
`P(Y=1 | do(T=1)) = 0.625`.

The table is the exact observed law at 800 rows,
`count(t, m, y) = 800 · Σ_u P(u) P(t | u) P(m | t) P(y | m, u)`, and every count
is an integer. There is no random data generation. On this table the
front-door functional `Σ_m P(m | t) Σ_t' P(y | m, t') P(t')` equals the SCM
truth exactly, so the Frequentist plug-in must match to `1e-12`. The Bayesian
functional (Bayesian bootstrap over the observed cells) is checked against the
same truth with `bayesian.absolute_tolerance`.

Reproduce the table and truth with exact rationals:

```python
from fractions import Fraction as F
pt = {0: F(1, 5), 1: F(4, 5)}          # P(T=1 | U=u)
pm = {0: F(1, 4), 1: F(3, 4)}          # P(M=1 | T=t)
py = lambda m, u: F(1, 10) + F(1, 2) * m + F(3, 10) * u
b = lambda p, x: p if x else 1 - p
count = {(t, m, y): 800 * sum(F(1, 2) * b(pt[u], t) * b(pm[t], m) * b(py(m, u), y)
                              for u in (0, 1))
         for t in (0, 1) for m in (0, 1) for y in (0, 1)}
truth = {t: sum(b(pm[t], m) * sum(F(1, 2) * py(m, u) for u in (0, 1)) for m in (0, 1))
         for t in (0, 1)}                # {0: 3/8, 1: 5/8}
```

`crates/antecedent/tests/prepared_analysis.rs::admg_frontdoor_distribution_known_truth`
runs both interventions on explicit and accepted ADMGs, Frequentist and
Bayesian, fresh and prepared.

## Expected summary

Top-level keys: `bayesian, case, columns, contingency_table, frequentist, graph, identification, rows, schema_version, scm, truth` (11 fields).
