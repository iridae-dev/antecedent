# Antecedent

Antecedent helps you estimate causal effects in Python and Rust. Supply data,
a causal question, and a graph or discovery method. It checks whether the
question can be answered under the stated assumptions, then estimates the
effect or explains why it cannot.

Results keep the assumptions, uncertainty, and diagnostics alongside the answer.
When several causal structures remain plausible, that uncertainty stays visible.

## Start here

These docs describe **Antecedent 1.11**. The Python quickstart begins with
installation from PyPI.

| What you want to do | Start with |
|---|---|
| Run your first Python analysis | [Python quickstart](python-workflow.md) |
| Use Rust | [Rust quickstart](rust-quickstart.md) |
| Learn from a worked example | [Examples](examples.md) |
| Check whether your analysis is supported | [Supported analyses](supported-analyses.md) |
| Look up a Python method | [Python API](python-api.md) |

## Explore a question

- **How does an effect change with dose?** Read [causal responses](causal-responses.md).
- **Does it differ across populations or outcomes?** Read [local, distributional, and joint effects](local-distributional-joint.md).
- **Was the outcome censored or selected?** Read the [observation contract](observation-contract.md).
- **Can evidence transfer, or do units affect each other?** Read [transport and interference](transport-interference.md).
- **Can an earlier study inform a new one?** Read about the [prior bank](priors.md).

A supported analysis still depends on its assumptions and on adequate data.
The [capabilities](capabilities.md) page describes the available methods;
the [support matrix](support-matrix.md) records which combinations can run.

## Go deeper

Read the [architecture](architecture.md), [evidence and conformance](conformance/README.md),
or [development guide](development.md). For changes in this version, see the
[1.11.0 release notes](release-notes/v1.11.0.md) and the
[1.11 finding closeout](reviews/v1.11-finding-closeout.md).
