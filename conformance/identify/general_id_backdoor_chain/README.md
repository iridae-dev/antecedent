# Identify oracle: general_id_backdoor_chain

Recorded pinned baseline 0.14 `identify_effect` output for the confounded
backdoor graph `Z -> T`, `Z -> Y`, `T -> Y`.

`antecedent-identify::backdoor::tests::confounding_requires_z` and
`antecedent-identify::id::tests::backdoor_chain_identified` parse this fixture,
check the frozen baseline status, graph, and backdoor estimand family, then
require Antecedent's backdoor and general-ID implementations to identify the
same graph.
