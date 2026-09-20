# antecedent-learn-burn

Optional Burn provider for Antecedent nuisance models (roadmap L).

This crate is **not** on the default `ml-full` graph. Enable
`antecedent-learn`'s `ml-gpu` feature. Training uses the NdArray CPU
backend; the public learner still emits `Vec<f64>` OOF predictions into
Antecedent scores. `antecedent-estimate` never names Burn.
