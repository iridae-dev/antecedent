# Prepared cyclic execution in 3.0

Cyclic preparation now compiles and retains a typed program without fitting,
sampling, simulating or encoding an artifact. `inspect()` remains cheap after
execution: `preparation_evaluations` counts preparation work and stays zero.
For fitted equilibrium preparations, `fits_performed` starts at zero and counts
fits actually performed by that handle during execution.

`Study::prepare_cyclic_family` retains the original family request, source
identity and applicable equilibrium, dynamic or equation compiler products.
`estimate` executes that retained program; `export` then encodes its result.
The prepared owner no longer retains an encoded artifact or a complete report.
Independent artifact consumers execute the same program rather than an IO-owned
numerical implementation. Existing version-one artifacts retain their replay
semantics and request identity checks.

Rust `rebind` and Python family `refresh` share a frozen statistical product only
when its fit dependencies match. Changing response worlds can reuse bootstrap parameters, posterior draws
or an OU mechanism fit. Changing observations, mechanism parameters, priors,
factual initial laws or fitting policies invalidates the relevant fit. Changing
the execution kernel also refits. Numerical initialization used only to propagate
an already fitted response is distinct from a factual initial-law dependency.
The prepared owner uses synchronized caches and can outlive the original builder.

These changes preserve each family's model-conditional interpretation, failed
draw slots, unresolved solution regions and unmeasured calibration. Compilation
does not add identification or scientific authorization. Resource accounting and
cross-family release acceptance remain open until their separate checks pass.
