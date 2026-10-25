# Current decision authority in 3.0

`FlagshipWorkflow::decision()` now returns a result. Withdrawal and supersession
invalidate dependent decision access, recalculation and export through the
existing evidence ledger. Independent cost and temporal summaries remain usable.
Trust loss is represented by `DecisionAuthority::withdraw_source` with the reason
for retiring the source.

`export_decision()` writes a version 2 envelope containing the numerical replay
payload, its identity, the ledger result identity and exact source revisions.
`load_decision` requires caller-held current `DecisionAuthority`. An embedded old
ledger cannot authorize an artifact. `into_authority()` retains that authority
when discarding the workflow. Source or numerical-payload changes invalidate the
older decision's authority.

`replay_decision()` explicitly returns historical numerical moments, without
evaluating a currently authorized decision. It accepts both version 2 envelopes
and the older joint-law artifacts. `export_historical_replay()` exports the latter
payload for independent numerical consumers. Historical replay never renews
withdrawn scientific permission.

Contract-bound affine uncertainty propagation and support mapping now require
`BridgeContract::check_licensing()`. Experimental-prior import validates the
candidate contract before changing it. Context-free numerical helpers retain
their numerical role and grant no bridge authority.

Validation: the 14 flagship unit tests cover replay, invalidation and independent
summaries. Bridge propagation tests include refusal without the principal
assumption. This closes only the propagation-licensing portion; mean
provenance and decision-classification consolidation remain open.
