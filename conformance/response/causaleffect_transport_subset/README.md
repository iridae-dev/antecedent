# causaleffect transport on the supported sID subset

Black-box parity for Antecedent's sound single-source transport rules against
R `causaleffect` 1.3.15 `transport()`.

Only cases inside Antecedent's implemented subset are compared:

- direct transport when selection does not reach the outcome;
- exogenous pre-treatment standardization (`transport.sid.standardize`).

General multi-node c-component cases remain `NotCertified` in Antecedent and
are not part of this claim. The frozen oracle stores the rendered formula
string; the consuming test checks Antecedent's certificate rule and formula
kind, not string identity with R's TeX-like rendering.
