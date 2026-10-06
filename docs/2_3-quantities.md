# Scientific quantities (2.3)

`ScientificQuantity` (core: `antecedent-core/src/quantity.rs`; wire:
`ScientificQuantityWire`) names one scalar coordinate by stable variable ID,
display name, role, units, population, intervention/regime, horizon,
functional, conditioning and scale/transform.

## Where it is used

- Joint distribution artifacts: one descriptor per draw coordinate.
- External requests and results: each requested/returned value is bound to a
  descriptor.
- Decision contracts: utilities and outcomes reference descriptors.
- Response artifacts (optional): `CausalResponseWire.coordinates` carries one
  descriptor per response value, in value order. Absent on older artifacts,
  which decode and re-encode unchanged. When present the decoder refuses a
  wrong length, an invalid descriptor, or two values sharing a coordinate.
  In Python, `antecedent.results.coordinates.response_coordinates` derives the
  descriptors of a response curve exactly as `antecedent.external.response`
  does, and `CausalResponseView.quantities` can hold them.

## Identity rule

Identity never relies on position or display label. Two coordinates are the
same only when variable ID, population, regime, horizon, functional,
conditioning and transform all match. Units are declared, never converted.

## Evidence obligations

Evidence obligations are not implemented yet. When they land they will
reuse `ScientificQuantity` to name the coordinates they cover.
