# prior_bank_weighted_target

Known-truth values for transporting a prior toward a target population declared as a
weighting over the source rows (`crates/antecedent-prob/src/transport_target.rs`), separate
from `prior_bank_transport`, which covers tag-defined populations.

- `kernel_weights`: Gaussian, Epanechnikov and tricube kernel values at fixed distances and
  bandwidth, from the closed forms.
- `graded_rows`: rows with effect `i` and weight `i + 1`; the weighted mean, Kish effective
  sample size and sampling variance are summed term by term.
- `neighbourhood_rows`: a planted neighbourhood whose rows share one effect, so the
  transported mean is that effect and no variance is added to the source prior.
- `refusals`: an equal-weight target of 15 rows is below the Kish floor of 20 and is refused
  with `transport_target_overlap_failed`.

`reference.py` re-derives every expected value with the standard library only.
