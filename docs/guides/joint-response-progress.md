# Joint-response work and acceptance

The X-series uses the existing scientific target, identification, estimation,
composition and decision owners. It introduces no crate or parallel execution
pipeline. Item completion follows the full TODO criterion; an implemented
component does not close a whole practitioner workflow.

## Independent reference design (X06.1)

The reference is in `conformance/response/joint_panel`. Four seeded geographic
groups carry two observed pre-period assignments and 24 outcome periods, with
L=2 channel-specific carryover, shared Michaelis–Menten saturation, a bilinear
interaction, hierarchical intercept truth, seasons, an observed assignment/outcome
confounder and stationary Gaussian AR(1) errors within groups. Additive and
interaction variants share random draws. A separate latent-confounded design has
no off-diagonal assignment support and requires withholding an identified claim.

The target averages conditional mean outcomes over the retained groups,
covariates and initial channel histories. It integrates future outcome noise;
it is neither predictive outcome uncertainty nor a population parameter posterior.
Four shared-history world trajectories, their assembled interaction and a held-out
sustained joint program are retained independently of production estimator code.
Independent steady-state and transient formulas test the reference values.

`parity/fixtures/joint_panel_reference.toml`, the existing conformance reference
checker, `parity/v3_test_registry.toml` and
`provenance/response.joint_panel_reference.toml` register Q01/Q06 evidence.
Seven focused reference tests and all eight registered reference generators pass.
The generator is deliberately independent: it supplies scientific truth to the
future compiled workflow, not a public estimator that bypasses the compiler.

X06.1 is complete. X06.2, X06, and their frequentist/Bayesian acceptance children
remain open. They require the actual compiled X02 fit, its identification attempts,
uncertainty, production refusals, independent artifacts, calibration and a pinned
additive comparison. Known mechanism simulation does not satisfy those criteria.

## Reproduction

```sh
python3 -m unittest discover -s scripts -p test_joint_panel_reference.py
python3 conformance/response/joint_panel/reference.py --check
python3 scripts/check_conformance_references.py
python3 scripts/check_v3_test_registry.py
python3 scripts/gate_provenance_schema.py
```

The reference generator uses only the Python standard library. Other existing
reference generators require the repository's Python development environment.
