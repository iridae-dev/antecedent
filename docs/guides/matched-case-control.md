# Matched case-control odds ratio (2.2 E7)

`antecedent.matched.matched_case_control_odds_ratio` (Rust:
`antecedent_estimate::conditional_odds_ratio`) estimates one quantity from matched
sets sampled on the outcome: the **conditional odds ratio** of a binary exposure,
by exact conditional likelihood (conditional logistic regression). It is a new
design row, not a conversion of the AIPW result, and its claim is `point_only`.
The record is `2.2E.E7.matched_case_control_odds_ratio` in
`parity/promotion_2_2.toml`.

```python
from antecedent.matched import matched_case_control_odds_ratio

result = matched_case_control_odds_ratio(
    stratum=[1, 1, 2, 2, 3, 3],   # matching strata
    case=[1, 0, 1, 0, 1, 0],      # 1 = case, 0 = control
    exposed=[1, 0, 1, 0, 0, 1],   # binary exposure
    sampling="matched_case_control",  # required declaration
)
result.odds_ratio, result.counts
```

## Estimand and population

Within matched set `i`, `logit P(case | x) = alpha_i + beta x` with an unrestricted
`alpha_i`. The conditional odds ratio is `theta = exp(beta)`. Sampling on the outcome
changes only the `alpha_i`, so `theta` is the same in the sampled sets and in the
population they were drawn from, **provided** the sets are matched on the declared
stratum and the model's odds-ratio-is-common assumption holds. The cell checks neither
(it takes no graph, so it makes no identification claim about confounders beyond the
matching the caller declares).

## Derivation

Condition set `i` of `n` members, `k` cases and `t` exposed on `(n, k, t)`. The number
`A` of exposed cases is noncentral hypergeometric:

    P(A = a) = C(k, a) C(n-k, t-a) e^{beta a} / sum_j C(k, j) C(n-k, t-j) e^{beta j},
    max(0, t-(n-k)) <= a <= min(k, t).

`alpha_i` is gone. With `E_i`, `V_i` the mean and variance of `A` at `beta`:

    l(beta) = sum_i log P(A_i = a_i)
    U(beta) = l'(beta)  = sum_i (a_i - E_i(beta))        (score)
    I(beta) = -l''(beta) = sum_i V_i(beta)               (information)

`l` is strictly concave (an exponential family in `a`), so the maximizer is unique
when it exists. A single case per set (1:M matching) is the usual
`e^{beta x_case} / sum_m e^{beta x_m}` form; the code enumerates the equivalent
hypergeometric weights, and the tests check both against each other.

With 1:1 matching the informative sets are the discordant pairs and
`U(beta) = b - (b + c) e^beta / (1 + e^beta)`, hence the closed form `theta = b / c`
(`b` pairs with the case exposed, `c` with the control exposed). For one type of 1:2
set (`n = 3`, `k = 1`, `t = 1`) with `a` of `m` sets having the case exposed,
`theta = 2a / (m - a)`.

The estimate is found by bracketing the monotone score and a safeguarded Newton
iteration (the information gives the step, bisection the guarantee); the solve observes
a cancellation token on entry (before the set tally) and each iteration. It is finite iff the observed exposed-case total
lies strictly between the smallest and largest totals the informative sets allow.

## Strata that carry no information

Never dropped silently; `result.counts` reports each class, and they sum to `total`:

| class | meaning |
| --- | --- |
| `outcome_degenerate` | no case or no control (includes `singleton`, sets of one member) |
| `exposure_concordant` | a case and a control, but every member has the same exposure |
| `informative` | contributes to the likelihood |

Adding any number of the first two classes leaves the estimate bit-for-bit unchanged.

## What is refused

| request | reason code | detail |
| --- | --- | --- |
| population risk, absolute risk, risk difference, risk ratio | `effect_not_identified` | `matched_case_control.absolute_risk_not_identified` |
| any other estimand | `route_not_supported` | `matched_case_control.estimand_not_supported` |
| a sampling design other than `matched_case_control` | `invalid_argument` | `matched_case_control.sampling_design` |
| an interval (`level=`) | `cell_not_licensed` | `matched_case_control.interval_withheld` |
| every set concordant or outcome-degenerate | `effect_not_identified` | `matched_case_control.no_informative_sets` |
| the conditional likelihood has no finite maximizer | `route_not_supported` | `matched_case_control.estimate_not_finite` |
| non-0/1, missing or unequal-length rows | `invalid_argument` | `matched_case_control.invalid_data` |
| cancelled solve (a stop, never a verdict) | `transport_budget_cancel` | `matched_case_control.budget` |

Risk-scale quantities are refused because the case fraction is fixed by the sampling
design: an absolute risk needs the outcome prevalence or the selection fractions, and
this cell takes neither.

No interval is reported. A Wald interval `beta_hat +- z / sqrt(I(beta_hat))` rests on
asymptotics (many informative sets, bounded set size, a fixed `beta`, independent sets)
that no coverage record measures, so the route stays closed until one does.
