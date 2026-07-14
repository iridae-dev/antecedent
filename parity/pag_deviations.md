# PAG / latent-confounder deviations

Intentional waivers relative to `parity/pag.toml`. LPCMCI and generalized
adjustment envelopes are `done`. J-PCMCI+ / RPCMCI are `done` under tigramite /
context inventories.

## 1. FCI / RFCI

Static PAG discovery (FCI/RFCI) is not delivered. LPCMCI is the PAG discovery
surface. Tracked as `intentional_deviation` on `pag.discovery.fci_rfci`.

## 2. Full recursive ID / IDC

The library ships **generalized adjustment** over CPDAG/PAG completions with
`IdentificationEnvelope` and preserved unidentified mass. Full semi-Markovian
ID/IDC remains waived (DESIGN §10.2). Tracked as `intentional_deviation` on
`pag.identify.full_id_idc` (and `dowhy.identify.general_id`).

## Verification

PAG `done` rows are backed by conformance under `conformance/pag/`, unit tests,
and/or criterion benches mapped in `scripts/gate_pag.sh`.
