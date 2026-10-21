# Joint-response implementation and acceptance

The X-series extends the existing causal compiler, model, estimation, composition,
decision and artifact owners. No new crate or standalone fitting pipeline is
introduced. This report distinguishes implemented workflows from the statistical
acceptance owned by Q09 and E14.

## Scope and compiler path

Channels have stable identities and a common declared time axis. Sparse interaction
features use the existing Rust `ValueExpr` / Python `intervention_programs.Expr`
syntax and the shared `NodeProgram` compiler/executor. There is no fixed channel
count or interaction order and no automatic powerset expansion. Feature reads
explicitly bind to lag-weighted, saturated channel values. Foreign/generated,
lagged feature and callback reads are refused for this binding.

General linear scientific contrasts retain each world's finite nonzero coefficient,
explicit reference, population, units, conditioning and history. Weighted windows
and higher-order contrasts retain their actual questions. Existing fixed contrast
identities remain compatible. Target wire version 2 carries the general form;
version 1 remains readable for its earlier fixed forms.

| Workflow | Retained compiler and numerical execution | Scientific boundary |
| --- | --- | --- |
| Static Gaussian joint response | Existing checked Gaussian product, one fitted design, shared covariance, retarget reuse, native and independent artifact consumers | Actual graph identification and source support required; conditional covariate support is not invented from marginal counts |
| Nonlinear group/time panel | `Study::prepare_joint_panel` / `PreparedStudy::compile_joint_panel`, retained target, ordered design attempts, physical choice, dependency identities and synchronized fit cache | Unsupported observational worlds trigger the assignment attempt; finite fitted responses alone grant no causal license |
| Dose-by-time surfaces | Existing scalar Set/policy contract translated into explicit panel paths and mixed-horizon prepared execution | Policy must assign every modeled period; undeclared fixed channels and implicit post-pulse continuation are refused |
| Frequentist response law | ML, group scores and cluster sandwich, aligned local parameter/group-effect realizations | Approximate fitted conditional response uncertainty; not a Bayesian posterior or measured calibration |
| Bayesian response law | Proper-prior posterior, shared parameter/group-effect slots, real mixing diagnostics | Nonconverged or failed draws cannot become a usable posterior merely because their values are finite |
| Time-varying response | Piecewise candidates, smooth fallback, full-selection bootstrap or declared Bayesian path/model union | Unique exposure/saturation/composition/mechanism attribution is refused; explicit reference-window change is evaluated instead |
| Finite program search | Existing decision engine, actual schedules, bounds, budgets, risk demands and exhaustive supplied-domain evaluation | Certificate applies only to the supplied finite set under the retained law; selected-decision reliability is unmeasured |
| Continuous program search | Frozen-fit compiler adapter and checked coordinate pattern search | Best feasible point and stopping evidence; no global gap or certificate is invented |
| Experiment update and next study | Licensed bridge, current evidence ledger, paired uncertain mappings, frozen-joint importance update, conditional decision and declared Gaussian future-study ranking | Actual converged Bayesian law required; conditional modular update and smaller same-estimand ranking, not an identified longitudinal effect |

Compilation/inspection do not fit. Public prepared execution survives builder
removal. Compatible world and utility changes reuse a frozen fit when caller cache
policy permits. Changed source, priors, initial law, mechanism or relevant numerical
policy change actual dependencies. Explicitly disabled caches remain disabled.
Output and scratch admission uses actual retained widths; failed slots remain in
the shared draw axis. Cancellation and resource refusal remain typed.

Primary decision/search ports retain the scientific target, proof, question
projection, dependency identity and output admission. Raw numerical extraction is
explicitly named `numerical_*`. Conditional consumption does not acquire causal
authority. Window and change projections retain the actual prefix-world question,
rather than attaching a new label to an unrelated terminal target.

## Identification attempts and adjusted claims

The earlier Gaussian-panel proposal does not identify a nonlinear interaction.
The production panel records its observational attempt and the concrete reason a
required world is unsupported. A retained compatible static source proof can bind
only its exact source/design/context law, without transferring a Gaussian numerical
restriction. A whole-program randomized assignment can establish supported world
means under its declared history and exogenous-context premises. Per-period
assignment metadata cannot certify an unsupported entire trajectory. The
confounded independent fixture withholds the causal claim.

Existing static, fixed-temporal and temporal-class mediation consumers retain their
own theorem, interventions, component and actual horizon. The class posterior
uses the retained mediation cache; an ATE witness is not substituted for it.
Unresolved graph/completion slots remain unresolved.

The experiment chain validates population, quantity, window, duration, dose,
exposure, recency/spend, transport and dependence through the existing bridge
contract. Current consumption checks relevant ledger identities/revisions, trust,
withdrawal, supersession and duplicate/prior origins. Independent historical replay
uses the stored frozen law and grants no current authorization. Artifacts do not
attest that stored fit numbers were generated honestly; current consumers must
supply the expected compiled program and frozen-law authority.

## TODO status

X01.1, X01.2, X02 and X06.2 implementation criteria are complete. The local TODO
records their concrete evidence. X03, X04.1, X04.2 and X05 implementations are
present, while their boxes retain the required Q09/E14 acceptance. X01/X06 final
acceptance awaits whole-procedure calibration and decision-reliability evidence.

## Independent evidence and remaining acceptance

`conformance/response/joint_panel` retains independent seeded panel truth,
additive/interaction variants, held-out programs, a confounded unsupported design
and whole-program randomized interactions. The standard-library reference imports
no production estimator. Cross-interpreter checks preserve exact CSV and metadata;
finite JSON truth uses an absolute tolerance of 1e-13 for arithmetic roundoff.

`parity/fixtures/joint_panel_procedures.toml` registers the complete frequentist,
Bayesian, temporal-selection, search and experiment-to-decision procedures. These
are **unmeasured** registrations, not coverage records or release attestation.

X01/X06 final acceptance and X03/X04/X05 statistical acceptance remain open where
Q09/E14 require whole-procedure measurements. Numerical implementation tests
do not close those items. The method specifications preserve their scientific
review status and failure boundaries.

## Verification record

| Check | Result |
| --- | --- |
| Shared compiler/probability/bridge/search selection | 48 passed |
| Public panel/Gaussian, numerical engines, artifact and mediation selection | 100 distinct tests passed: 97 on the combined run, then the three corrected test declarations passed on focused reruns |
| Actual Bayesian experiment chain | Four chains, 8,000 warmup and 12,000 retained draws per chain; convergence, changed decision, next-study ranking, independent replay and current-authority refusals passed |
| Python panel/distribution/causal artifacts | 68 passed against the rebuilt worktree extension |
| Gaussian Rust-to-Python posterior interchange | 256 × 2 draws, means and full covariance exactly equal by IEEE bits; changed expected identity refused; no current authority minted |
| Strict Clippy | All ten affected published crates plus the Python extension, libraries and test targets, warnings denied |
| Public APIs | All ten affected published crate snapshots generated and comparison gate passed |
| Independent references | Nine independent reference tests and the complete conformance reference-generator gate passed; reference check also passed on Python 3.12, 3.13 and 3.14 |
| Registry/provenance/schema | 42 test registrations; 459 provenance records/1,247 path references; parity manifest and close-out schema passed |
| Python declarations | Ruff and mypy passed |
| Formatting | Cargo formatting applied to changed Rust sources through a temporary snapshot; diff whitespace check passed |

The three final workflow failures were invalid test declarations: two shared a
constant-only action utility that the existing decision contract correctly
refused; another changed inference options without changing its explicit target
label. Their declarations were corrected and the exact affected tests rerun.
No production permission check or convergence threshold was weakened.

Earlier broad selection also reached a measured-inference consumer and refused
its nonattesting inference/coverage records. No measurement record or scientific
gate was changed to make this repair pass. This is an explicit limitation, not a
claim that the unrelated record problem was proven pre-existing. The focused
final selections exercise the changed workflows. The full release and support
matrix execution campaigns were not run; citation auditing is reported separately.

## Reproduction commands

Use a dedicated `CARGO_TARGET_DIR` and the worktree's built Python virtual
environment. `scripts/test.sh dev` selects unoptimized focused verification with
line-table debug information; it is not a performance measurement.

```sh
bash scripts/test.sh dev -p antecedent-contract -p antecedent-prob -p antecedent-design -p antecedent-identify -p antecedent-model -p antecedent-io-core --lib -E 'test(bridge) | test(finite_belief) | test(preposterior) | test(finite_program_search) | test(compiled_product) | test(compiled_identification_wire) | test(column_program) | test(explicit_column_bindings)'
ANTECEDENT_GAUSSIAN_INTERCHANGE_DIR=/private/tmp/antecedent-x-gaussian-interchange bash scripts/test.sh dev -p antecedent-estimate -p antecedent-io -p antecedent --lib --test suite_01 --test suite_02 -E 'test(joint_panel) | test(joint_gaussian) | test(joint_linear) | test(mediation)'
python/.venv/bin/python scripts/check_joint_gaussian_interchange.py /private/tmp/antecedent-x-gaussian-interchange
(cd python && .venv/bin/python -m pytest -q tests/test_joint_panel.py tests/test_causal_artifacts.py tests/test_joint_distribution_artifact.py)
cargo clippy --offline --locked --keep-going --no-deps -p antecedent-core -p antecedent-contract -p antecedent-identify -p antecedent-model -p antecedent-estimate -p antecedent-prob -p antecedent-io-core -p antecedent-design -p antecedent-io -p antecedent -p antecedent-py --lib --tests -- -D warnings
python3 scripts/check_v3_test_registry.py
python3 scripts/gate_provenance_schema.py
python3 scripts/gate_checked_execution.py /private/tmp/antecedent-x-checked-evidence.toml all
```

The checked-execution citation audit covers 723 licensed cells and 791 estimator
citations; emitting those citations does not mean all their tests ran. The focused
acceptance selection above executes the changed families. No full release,
performance or statistical calibration campaign is claimed. The native Python
extension used here was built in the development profile.

The final focused corrections can be reproduced with:

```sh
bash scripts/test.sh dev -p antecedent --test suite_01 -E 'test(actual_fit_cache_invalidates_changed_history_initialization_seed_and_kernel_policy) | test(static_panel_coarser_prefix_keeps_the_original_temporal_clock)'
bash scripts/test.sh dev -p antecedent --test suite_01 -E 'test(actual_bayesian_panel_fit_preserves_mixing_diagnostics_and_shared_draw_axes)'
```

Implementation commits are on `codex/x-series-unblocked` in the separate managed
worktree. The original `3.0.0` checkout was preserved. `TODO.md` is an ignored local
planning file: its completed implementation boxes and explicit acceptance notes
are updated locally; this tracked report and the method/release-scope records
preserve the committed evidence.
