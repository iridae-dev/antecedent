# Identify oracle: general_id_hedge

Recorded pinned baseline 0.14 `identify_effect` output for the bow-arc graph
`T -> Y`, `T <-> Y`.

`antecedent-identify::id::tests::hedge_not_identified` parses this fixture,
checks the frozen baseline's unidentified status and `graph_dot`, then requires
Antecedent's general-ID implementation to return `NotIdentified` with a hedge
diagnostic for the same graph.
