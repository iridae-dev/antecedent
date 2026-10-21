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

## Scientific contracts in production (X01.1 and X01.2)

Retained identification can now bind the complete inference-independent target,
not only a list of world labels. The binding checks exact world coverage and
requires an assembled obligation for a contrast or dose-by-time surface. An
unsupported or model-conditional world cannot become an identified assembled
claim because its numerical result is finite. A bound product refuses changed
population, outcome, intervention, units, conditioning, history or assumptions.
Frequentist/Bayesian realization selection does not change target identity.
Legacy unbound product digests remain unchanged; target-bound products use a
separate versioned identity payload and refuse unbound reuse.

The existing Gaussian compiler constructs and retains this binding from its
actual query and identification. It preserves the outcome units from the source
schema. Compilation remains free of fitting and repeats no identification.

The existing batch joint-cell consumer now requires a checked target and actual
frozen-score support for every required world before exposing a family contrast.
It retains the source derivation, assumptions, explicit reference, outcome units
and whole-functional obligation beside the existing shared-row covariance
calculation. Failed binding withholds family contrast outputs while preserving
supported individual cells. A three-factor table cannot silently stand for the
first four cells of a two-factor interaction. No parallel contrast fitter or
public estimator was added.

Existing static frequentist/Bayesian mediation and fixed temporal DAG mediation
retain their theorem-scoped targets. These retain the requested component,
interventions, mediators, population, units and cross-world premises. Fixed
temporal targets use the actual per-horizon history and the existing observed
lag-window law; no independent initial-state distribution is invented. Refresh
checks this binding before fitting. Generic joint-world validation refuses to
replace the mediation theorem.

These additions do not close X01.1 or X01.2 as a whole. Temporal class/posterior
mediation metadata and broader compiled joint-world consumers still need their
full scope; the common-world functional execution, aligned parameter draws,
failed/unidentified mass, public evidence/decision ports, independent artifacts
and whole-contrast calibration remain separate acceptance obligations. In
particular, the new native batch binding is not persisted by historical numerical
artifact formats and grants no new artifact or interval authority. No scientific
license or calibration record was promoted.

X02's constrained finite-lag nonlinear fitted family and supported-world design
attempts remain open. X03 mechanism changes, X04 program search and X05
experiment-to-decision closure must consume that fitted compiler interface;
standalone evaluators over caller-provided response arrays would not complete
those items. Their existing owners and method specifications remain in place.

## Scoped verification

The final combined selection passed 86 tests covering the changed Gaussian,
batch contrast and mediation paths and their estimation engines. The 15 focused
identification-product tests passed separately. These checks include real public
prepared execution after builder disposal, supported/unsupported world refusals,
source-query mismatch protections, declared units and target changes. Clippy
passed with warnings denied for the affected libraries and test targets.
Formatting was applied only to changed Rust sources.

The reference's seven tests, eight existing reference generators, provenance
schema/path audit, parity schema check and test-registry checks passed. Generated public
API snapshots and their comparison gate passed for the affected identification,
estimation and facade crates. This is implementation evidence; it is not whole-procedure
calibration, external baseline agreement, full Python interchange or release
attestation.

## Reproduction

```sh
python3 -m unittest discover -s scripts -p test_joint_panel_reference.py
python3 conformance/response/joint_panel/reference.py --check
python3 scripts/check_conformance_references.py
python3 scripts/check_v3_test_registry.py
python3 scripts/gate_provenance_schema.py
cargo clippy -p antecedent-identify -p antecedent-estimate -p antecedent --lib --tests -- -D warnings
bash scripts/test.sh dev -p antecedent-identify --lib -E 'test(compiled_product)'
bash scripts/test.sh dev -p antecedent -p antecedent-estimate --lib --test suite_02 --test suite_03 --test suite_05 -E 'test(prepared_gaussian) | test(joint_contrast_binding) | test(mediation) | test(numeric_pins::batch) | test(cell_aipw) | test(three_factor_interaction_batch) | test(zero_effect_pair_family)'
python3 scripts/gate_public_api.py --crate antecedent-identify --crate antecedent-estimate --crate antecedent
```

The reference generator uses only the Python standard library. Other existing
reference generators require the repository's Python development environment.

## Retained scheduled and unresolved questions

Scheduled interaction validation now projects each channel's own sequence before
checking the factorial worlds. Changing one channel no longer appears to change
every channel. Unsupported compiler outcomes can bind their real scientific target
without inventing an identified estimand; rebinding to another question is refused.

The shared artifact owner now encodes the complete scientific target, including
population, observation unit, outcome units and functional, conditioning, every
world action, history/initial law, graph/model assumptions, inference selection,
and theorem-scoped mediation premises. Decoding rebuilds domain constructors and
checks the scientific identity. Transporting a target grants no identification.
Unknown versions and fields, changed units, and missing mediation premises refuse.

Checkpoint validation: 49 scientific-target, compiled-product and target-wire tests
passed. Strict Clippy passed for `antecedent-core`, `antecedent-identify` and
`antecedent-io-core` with `--no-deps --lib --tests -- -D warnings`. The broader
component selection passed 100 tests before the additional unsupported-binding
regression. This checkpoint does not close X01.1 or X01.2; production consumers,
arbitrary-channel expansion, public interchange and acceptance remain in progress.
