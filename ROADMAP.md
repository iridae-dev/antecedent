# Antecedent roadmap

## 2.5 — Acceleration workstream (conditional)

Moved from the 2.4 plan: this work has not started, and no 2.4 claim depends on
it. It remains conditional and blocks nothing.

Start state: `parity/accelerator_decision.md` records a CPU baseline and no
accelerator backend; `benches/baselines/neural_crossfit.md` is the 2.1 timing
note.

- [ ] Reconfirm the CPU whole-workload baseline on the 2.5 branch: cross-fitted neural propensity and outcome models including fold planning, fit, prediction, and return to physical row order. Record host, commit, backend, and stage times. The 2.1 number is not a same-host control.
- [ ] Build a candidate opt-in accelerator backend sufficient to measure host-device transfer, fold orchestration, prediction ordering, and repeatability against that same-host CPU run. Until it exists, the missing backend is not a negative result.
- [ ] Decide from the same-host measurement. Open a public backend only if the end-to-end gain is material; otherwise record the deferral, keep the CPU provider, and leave the route closed.
- [ ] If material: publish one opt-in backend/platform with a `cpu`/`gpu`/`auto` policy, an explicit refusal when the device or library is missing, device/backend/dtype/library provenance, and seeded replay.
- [ ] Add `ExecutionContext` compute and memory leases, fold/learner oversubscription controls, OOM and cancellation behavior, a CPU/accelerator numerical tolerance, and a small-workload fixture where CPU wins.
- [ ] CPU wheel works without accelerator drivers; a separate accelerator distribution installs; artifacts stay consumable on CPU. If any gate fails, defer without changing other claims.

Evidence required: same-host timing of the whole cross-fitted neural workload,
including transfer and fold orchestration. This is a performance gate, not a
coverage test.
