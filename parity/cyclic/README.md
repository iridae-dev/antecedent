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

Paper protocols and their explicit deviations are in
`examples/python/cyclic_paper_experiments.py`. Experiment metrics retain original
convergence failures and model-conditional interpretation. Calibration and
attestation are separate and are not asserted by these fixtures.
