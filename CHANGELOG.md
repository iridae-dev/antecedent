# Changelog

## 2.0.0 — draft

This is a release-preparation record. The published package version remains
1.11.0 until the 2.0 release cut; historical per-release prose has been
removed from the working tree and remains available in Git history.

### Cohesive causal workflow

- Keeps the established identify → estimate → inspect → refresh → export →
  consume lifecycle across discovery, graph uncertainty, identification,
  estimation, validation, temporal and response analysis, Bayesian inference,
  attribution, design, state, and artifacts.
- Preserves fail-closed support boundaries, typed assumptions, diagnostics,
  provenance, and durable claims instead of strengthening a result at an API
  or serialization seam.

### Learners and heterogeneous effects

- Adds the Rust-native learner substrate and learner specifications for
  fold-local prediction.
- Adds DML, DR-Learner, and honest causal-forest paths with held-out nuisance
  diagnostics, overlap/trimming disclosure, learner provenance, and explicit
  limits on CATE uncertainty.

### Structural transport

- Adds population and evidence-regime catalogs, typed transport outcomes,
  population-aware expression evaluation, and durable derivations.
- Distinguishes an unavailable joint/provider from an invalid request,
  theorem-stage non-certification, support failure, or numerical failure.

### Calibration status

- Does not claim a new calibration attestation. The long-running measurement
  program remains a release-roadmap item; existing records retain only their
  recorded scope.
