# Python API

New to Antecedent? Start with the [Python quickstart](python-workflow.md). For configuration details, use the [workflow reference](python-options.md).

<!-- pdoc adds this sibling directory after MkDocs builds the narrative. -->
<strong><a href="../python/antecedent.html">Browse the Python API</a></strong>

Read the Docs builds this reference from the matching Antecedent release (and
falls back to the checked-out source only before that release reaches PyPI).
The link stays within the documentation version you are reading.

To generate a local reference for the installed package, run:

```bash
python -m pip install antecedent pdoc
python -m pdoc antecedent -o site/python
```

For Rust, use `cargo doc -p antecedent --open` for this checkout, or visit the [published crate reference](https://docs.rs/antecedent).
