# Method children (B03)

208 leaves in `children.toml`. Each is one family method × static/temporal × frequentist/Bayesian counterpart, or an explicit boundary on an axis that does not exist. Named methods that must not collapse into a family row are separate children: categorical and multi-arm trials, factorial trials, switchback (P09); continuous dose, smoothed dose, dose grids, quantiles (P07); regression discontinuity, fuzzy/kink local ratios, augmented panel DiD, staggered event studies, synthetic control (P11).

This is not the support-matrix product in `parity/support_axes.toml` (queries × graph classes × structures × inferences × validations). Route ids are from `parity/capabilities/route_manifest.toml`. A field that the manifest does not state is `unlisted`. `module = "none"` means no implementation module. `gap = "none"` means that counterpart already exists. No child is `not_applicable`.

X01 and X06 are core. The X02 finite-lag slice is a proposal for B04. It is not admitted. X02.broader, X03, X04.1, X04.2, and X05 are promotion candidates, not release scope.

## Counts by parent

| Parent | Children |
| --- | --- |
| P01 | 8 |
| P02 | 8 |
| P03 | 8 |
| P04 | 4 |
| P05 | 4 |
| P06 | 4 |
| P07 | 24 |
| P08 | 4 |
| P09 | 28 |
| P10 | 4 |
| P11 | 24 |
| P12 | 8 |
| P13 | 8 |
| P14 | 4 |
| P15 | 16 |
| P16 | 4 |
| P17 | 4 |
| P18 | 4 |
| P19 | 4 |
| P20 | 4 |
| X01 | 4 |
| X02 | 4 |
| X02.broader | 4 |
| X03 | 4 |
| X04.1 | 4 |
| X04.2 | 4 |
| X05 | 4 |
| X06 | 4 |

## Counts by gap

| Gap | Children |
| --- | --- |
| none | 57 |
| target | 36 |
| implementation | 33 |
| statistical | 31 |
| validation | 21 |
| temporal-identification | 18 |
| theory | 12 |

## Why this X02 slice

The roadmap's grouped response asks for a finite carryover, a saturation, one selected interaction, a group intercept, seasonality, and a group/time disturbance. X06's reference panel is two channels, finite carryover, saturation, one known interaction, geographic partial pooling, one observable confounder, and serial/group dependence. The slice is the smallest parameterization that can be the thing that panel would fit, if B05 later admits it.

1. Design: one balanced geographic group-time panel, finite seasonal dummies, one additive observable confounder.
2. Carryover: simplex distributed lag of order 2 (nonnegative weights, sum to 1). Infinite geometric adstock is not in the slice.
3. Saturation: Michaelis-Menten `a/(a+k)` with `k>0`, shared across groups, with a checked derivative. Free Hill shape is broader.
4. Interaction: one bilinear product of the two saturated channel exposures.
5. Pooling: partial pooling of group intercepts only. Lag, saturation, interaction, and confounder coefficients are completely pooled.
6. Dependence: Gaussian AR(1) within group, independent across groups. No spatial residual.
7. Inference: frequentist Gaussian estimating equations with a group-cluster sandwich; Bayesian counterpart is that likelihood with a hierarchical intercept prior, a Dirichlet on lag weights, and a prior on `k`. Tempered generalized Bayes is not the proposal.
8. Target: the joint contrast of two channel schedules. The static child is the lag-0 reduction (no carryover, independent rows).

No module implements this. That absence is implementation. Whether the panel identifies the joint contrast is unreviewed research (`temporal-identification` on the temporal children). B04 has to accept or reject the slice. These four children do not admit it.

X02.broader (geometric adstock, free Hill shape, group-specific saturation, spatial residuals, extra channels and interactions), X03, finite-candidate X04.1, continuous-search X04.2, and comprehensive X05 stay promotion candidates.
