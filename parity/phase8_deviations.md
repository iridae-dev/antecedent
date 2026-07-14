# Phase 8 deviations

Intentional deferrals from DESIGN.md §32 (Phase 8) as reconciled against
`parity/pag.toml`.

## 1. FCI / RFCI → later parity

Static PAG discovery (FCI/RFCI) is listed in broader discovery parity but is
**not** a Phase 8 deliverable. LPCMCI is the Phase 8 discovery surface.

## 2. Full recursive ID / IDC → later

Phase 8 ships **generalized adjustment** over CPDAG/PAG completions with
`IdentificationEnvelope` and preserved unidentified mass. Full semi-Markovian
ID/IDC remains deferred (see DESIGN §10.2).

## 3. J-PCMCI+ / RPCMCI → Phase 9

Unchanged.

## Verification

Every `phase = 8` row in `parity/pag.toml` with `status = "done"` is backed by
conformance under `conformance/phase8/`, unit tests, and/or criterion benches
mapped in `scripts/gate_phase8.sh`.
