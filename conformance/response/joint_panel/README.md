# Independent two-channel panel reference

This is the X06.1 reference design, shared by the planned frequentist and Bayesian
consumers. It is a fixture, not a fitted workflow or a new causal method. No
production estimator, simulator, identification routine or external baseline is
used to generate its truth. `reference.py --check` reproduces all five artifacts.

The retained seed is 20261021. Four independent geographic groups each contain 24
outcome periods and two observed pre-period channel assignments. Group effects
are draws from a shared Normal(0, 0.35²) law: a future partial-pooling fit must
estimate this hierarchy rather than treating the retained effects as known.
Within-group outcome errors are stationary Gaussian AR(1), rho=0.45, innovation
SD=0.2, initialized from variance 0.2²/(1−0.45²). Seasons have four declared dummy
levels. The observed normal covariate z contributes to the outcome in both designs.

For each channel, exposures are weighted lag sums before saturation:

- A: `0.6*A[t] + 0.3*A[t-1] + 0.1*A[t-2]`.
- B: `0.5*B[t] + 0.35*B[t-1] + 0.15*B[t-2]`.
- Saturation is `s(x)=x/(x+2)` for both channels.
- Conditional mean is `2 + group_effect + season[t%4] + 0.7*z + 3*s(A) + 2*s(B) + interaction*s(A)*s(B)`.

The additive variant sets interaction=0; the interaction variant sets it to 1.5.
Both share exactly the same seeded assignments, covariates and noise. Missing
outcomes at times −2 and −1 are history, not extra outcome observations.
`expected.json` retains the complete truth separately from the observed CSVs,
including the group effects, innovations, AR errors and hidden common cause.

## Target and held-out intervention

The four worlds set the two channels to (0,0), (4,0), (0,4), and (4,4) for the full
24-period horizon. They use each group's same retained pre-period history,
period-specific z, season and intercept. The reference functional averages the
conditional outcome means uniformly over these four groups and then takes
`joint − a − b + reference`. It integrates out mean-zero future outcome noise;
it is conditional on this finite population and its histories/covariates. It is
neither a predictive outcome distribution nor a superpopulation contrast. No
parameter uncertainty or calibration is generated here. Assignment changes with
z, so conditioning on the declared observed confounder is part of the design.

The 24-step constant joint program is held out as an entire observed trajectory.
Individual joint assignments are present in the randomized data. Conditional per-time channel randomization makes z an observed assignment/outcome
common cause: P(A=4|z) is 0.35 when z<0 and 0.65 otherwise; P(B=4|z) is 0.65 when
z<0 and 0.35 otherwise. The two assignments are independent given z, so every
joint cell has probability at least 0.1225 conditional on history and z. This supplies a design premise for a subsequent
sequential identification attempt, not an automatic license for an arbitrary
fitted model. The long program has small but positive assignment probability.

In the confounded design, both channel assignments equal
`4*I(U + 0.4*z > 0)`, and `0.8*U` also enters the outcome. U is retained only in
truth, not observed data. The off-diagonal worlds have no assignment support,
and unobserved confounding remains after adjusting for z. A consumer must withhold
an identified joint causal contrast. A fitted smooth response is not a repair of
these gaps. The fixture metadata records this expected refusal; it does not claim
that an unimplemented production consumer has already enforced it.

## Checks and later consumers

`python3 -m unittest discover -s scripts -p test_joint_panel_reference.py` checks
retained regeneration, independent steady-state and transient contrast formulas,
actual finite-lag/AR equations, held-out trajectories and the negative design.
The reference is registered in the existing fixture/provenance infrastructure and
runs in `scripts/check_conformance_references.py` alongside other independent
references. Q08/Q09 and X06.2 must exercise actual fitting, whole-procedure
uncertainty, refusal and artifact consumers. The additive variant is an input for
a future matched Meridian comparison; no external parity is claimed.
