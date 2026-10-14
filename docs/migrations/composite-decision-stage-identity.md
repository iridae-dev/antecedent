# Composite decision stage identity

## What changed

The terminal decision stage of a composite recalculation is identified by the contract, the support policy and input order, the native program binding and, for every callback branch, its provider, snapshot, program, seed, trust requirement and calibration. Typed standing inputs (calibration, representation, coupling, provider standing) enter an effective identity only when declared.

## Class

Migrated. A workflow that declares no standing input keeps every effective identity it had. A receipt stores the declared own digest of each stage as an opaque value, so receipts written earlier are consumed unchanged. The composite decision's own digest differs, so the first recalculation after the change recomputes the decision stage once and reuses everything upstream.

## Who is affected

Callers that retain stage identities of a composite workflow across the change. No stored artifact is refused.

## How to regenerate

Nothing to regenerate; run the workflow and retain the new plan.
