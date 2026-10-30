# Fit-free descriptive diagnostics

The uniform five-point distribution has mean zero, second central moment 2 and
fourth central moment 34/5. Its population skewness is zero and its population
excess kurtosis is (34/5)/4 - 3 = -1.3. These are independently derived descriptive
statistics; no model test, fit, confidence interval or causal claim is involved.

`fixture.json` is consumed by the Rust engine and Python conformance tests. The
same tests separately exercise policy identities, bounds, missingness and typed
equality. No external reference implementation was observed.
