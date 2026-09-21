# Antecedent 2.0

**Release preparation:** these are the active 2.0 docs. The published package
is still 1.11.0 until the release metadata and tag are advanced.

Antecedent makes causal analysis a reviewable program: define a question and
causal structure, identify the available claim, estimate through a supported
path, inspect its assumptions and diagnostics, then reuse or export the same
contract.

## Start here

| Goal | Start with |
| --- | --- |
| Run an analysis in Python | [Python workflow](python-workflow.md) |
| Use the Rust facade | [Rust quickstart](rust-quickstart.md) |
| Choose a supported route | [Supported analyses](supported-analyses.md) |
| Understand what can be claimed | [Capabilities](capabilities.md) and the [support matrix](support-matrix.md) |
| Learn the system shape | [Architecture](architecture.md) |

## The 2.0 whole

Existing graph, identification, estimation, validation, response, temporal,
Bayesian, attribution, design, state, and artifact workflows remain first-class.
2.0 adds a Rust-native learner substrate for honest nuisance prediction and a
population/evidence-aware transport foundation; neither bypasses the existing
identification and claim lifecycle.

Use DML, DR-Learner, or causal forest only where their assumptions, support,
and reported diagnostics fit the question. Use structural transport only when
the evidence catalog represents the actual populations and regimes. A refusal
or an unavailable answer is useful information, not an estimator fallback.

## Evidence status

The repository retains executable conformance and compatibility evidence. The
long-running 2.0 calibration measurement remains planned, not completed; see
the [draft release notes](release-notes/v2.0.0.md) and `result.calibration` for
the scope of any reported interval.
