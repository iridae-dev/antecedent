# Catalog distribution identities in 3.0

Catalog descriptors now use `catalog_distribution.v2`. Intervention values are
length-framed so label delimiters cannot merge assignment boundaries. Optional
study, snapshot and dataset text distinguishes absence from an explicitly empty
value. Variable sets remain sorted and projection trails retain their order.
`Value::Float64` keeps its bitwise equality and identity, including signed zero.

Every catalog descriptor identity changes. Re-derive mixed-source and observation
recovery proofs from complete trusted catalog and source premises. Earlier
records remain decodable for inspection, but current replay recomputes descriptor
identities and refuses v1 proof leaves or recovered result identities, including
resealed artifacts. Replacing the prefix cannot resolve a potentially ambiguous
v1 value list. Artifact wire versions remain unchanged; no v1-to-v2 identity
equivalence is asserted.
