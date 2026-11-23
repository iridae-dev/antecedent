# Independent cyclic workflow fixtures

`general-equilibrium/` is the complete first integration milestone: independently
generated overlapping nonlinear loops with correlated disturbances, fitting from
single-intervention data, joint-intervention prediction, shared-disturbance
counterfactuals, and a fresh-process CPU consumer. `expected-identity.json` is the
separately retained expectation; the consumer does not trust an identity extracted
from the container.

Regenerate with an installed release wheel:

```sh
python examples/python/general_cyclic_systems.py --output-dir parity/cyclic/general-equilibrium
```

The family fixtures (`specified-feedback-pair`, `overlapping-loops`,
`recovered-asymptotic-normal`, `recovered-bootstrap`, `recovered-unresolved-direction`,
`dynamic-inference-ou`, `previous-record-format`) each hold `request.json`,
`report.json`, the family record `family.ant`, `expected-identity.json` and
`manifest.json`. Regenerate them with

```sh
cargo test -p antecedent --test cyclic_family_fixtures -- --ignored write_
```

`crates/antecedent/tests/cyclic_independent_consumer.rs` consumes them without
calling the producer.

Paper protocols and their explicit deviations are in
`examples/python/cyclic_paper_experiments.py`. Experiment metrics retain original
convergence failures and model-conditional interpretation. Calibration and
attestation are separate and are not asserted by these fixtures.


Run the full paper-derived lanes with an installed wheel, NumPy and SciPy:

```sh
python examples/python/cyclic_paper_experiments.py --lane nodags-linear --full --trials 5 --max-iterations 1000 --output-dir /tmp/cyclic-papers/nodags-linear
python examples/python/cyclic_paper_experiments.py --lane nodags-relu --full --trials 5 --max-iterations 100 --output-dir /tmp/cyclic-papers/nodags-relu
python examples/python/cyclic_paper_experiments.py --lane bicycle --full --trials 5 --max-iterations 2000 --output-dir /tmp/cyclic-papers/bicycle
python parity/cyclic/paper-experiments/ou_continuation_protocol.py --base-dir /tmp/cyclic-papers/bicycle --output-dir /tmp/cyclic-papers/continuations --workers 4
python parity/cyclic/paper-experiments/ou_standardization_protocol.py --base-dir /tmp/cyclic-papers/continuations --output-dir /tmp/cyclic-papers/standardized
```

The tracked metrics record the executed results; output directories retain the
request, full report, independent expected identity and replay artifact for each
selected fit. `protocol_intervention_only.py` preserves the earlier experiment
allocation; the accepted protocol includes the paper's observational training rows.
See `docs/development/cyclic-systems.md` for results, explicit deviations and
failed/unresolved outcomes.
