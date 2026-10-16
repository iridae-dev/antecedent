#!/usr/bin/env python3
"""Build installed-wheel API documentation, including lazy public refusal classes."""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import pkgutil
import re
from pathlib import Path


def complete_doc_namespace(public, owner, cls=None):
    """Resolve signature-only imports in this documentation process.

    Facades re-export objects whose defining globals differ from __module__.
    Read exact imports from the defining source; never replace library files.
    """
    globals_ = getattr(owner, "__globals__", {})
    target = inspect.getmodule(owner)
    targets = [public] + ([target] if target is not None and target is not public else [])
    annotations = {
        **getattr(cls, "__annotations__", {}),
        **getattr(owner, "__annotations__", {}),
    }
    if cls is not None:
        class_module = inspect.getmodule(cls)
        if class_module is not None and class_module not in targets:
            targets.append(class_module)
    tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", " ".join(str(v) for v in annotations.values())))
    values = {token: globals_[token] for token in tokens if token in globals_}
    if cls is not None:
        for base in cls.__mro__:
            base_module = inspect.getmodule(base)
            if base_module is not None:
                for token in tokens:
                    if token not in values and token in vars(base_module):
                        values[token] = vars(base_module)[token]
    filename = getattr(getattr(owner, "__code__", None), "co_filename", None)
    package = globals_.get("__package__", public.__package__)
    if filename and Path(filename).is_file():
        tree = ast.parse(Path(filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module != "__future__":
                wanted = [
                    alias
                    for alias in node.names
                    if (alias.asname or alias.name) in tokens
                    and (alias.asname or alias.name) not in values
                ]
                if not wanted:
                    continue
                try:
                    name = (
                        importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                        if node.level
                        else node.module
                    )
                    imported = importlib.import_module(name)
                    for alias in wanted:
                        values[alias.asname or alias.name] = getattr(imported, alias.name)
                except (ImportError, AttributeError):
                    # pdoc still reports any unresolved signature; optional
                    # integrations need not be installed to document types.
                    continue
    for module in targets:
        for token, value in values.items():
            if token not in vars(module):
                setattr(module, token, value)


def build(output: Path) -> None:
    import antecedent
    import antecedent.errors as errors
    import pdoc

    # These runtime attributes are supplied by the native exception and Enum
    # bases. Declare their actual types in the docs process for field anchors.
    errors.CausalUnsupportedError.__annotations__["remedy"] = str | None
    from antecedent.extensibility import ProviderTrust

    ProviderTrust.__annotations__["value"] = str
    pdoc.render.configure(template_directory=Path(__file__).with_name("pdoc_templates"))

    def vendor_links(text):
        return re.sub(
            r"\]\(\.\./(concepts|api|errors|integrations|examples|usage)/([^)]*?)\.md(#[^)]*)?\)",
            lambda match: (
                "](https://pydantic.dev/docs/validation/latest/"
                + match[1]
                + "/"
                + match[2]
                + "/"
                + (match[3] or "")
                + ")"
            ),
            text,
        )

    pdoc.render.env.filters["vendor_links"] = vendor_links

    # pdoc inspects module dictionaries. Resolve the declared lazy exports in
    # this docs process so each refusal has its class documentation and fields.
    lazy_names = [name for name in errors.__all__ if name not in vars(errors)]
    for name in errors.__all__:
        getattr(errors, name)
    module = pdoc.doc.Module.from_name("antecedent.errors")
    missing = [
        name for name in lazy_names if not isinstance(module.members.get(name), pdoc.doc.Class)
    ]
    if missing:
        raise RuntimeError(f"Public error classes missing from API docs: {missing}")

    # Explicitly include lazy and narrowly imported public modules: root __all__
    # is a convenience surface, not the complete reference inventory.
    names = ["antecedent"] + [
        item.name
        for item in pkgutil.walk_packages(antecedent.__path__, prefix="antecedent.")
        if not any(part.startswith("_") for part in item.name.split(".")[1:])
    ]
    omissions = []
    for name in names:
        public = importlib.import_module(name)
        exports = {symbol: getattr(public, symbol) for symbol in getattr(public, "__all__", ())}
        for value in exports.values():
            owner = (
                inspect.unwrap(value)
                if inspect.isfunction(value)
                else getattr(value, "__init__", None)
            )
            cls = value if inspect.isclass(value) else None
            if cls is not None:
                # A flattened facade can make inspect slice another file at
                # the original class line number. Repair only malformed
                # source slices; leave normal public ownership unchanged.
                try:
                    ast.parse(inspect.getsource(cls))
                except SyntaxError:
                    constructor_globals = getattr(owner, "__globals__", {})
                    defining_name = constructor_globals.get("__name__")
                    if defining_name and constructor_globals.get(cls.__name__) is cls:
                        cls.__module__ = defining_name
                    else:
                        # TypedDict has no Python constructor globals. Locate
                        # its actual declaration and verify object identity.
                        package_root = Path(antecedent.__file__).parent
                        for source in package_root.rglob("*.py"):
                            tree = ast.parse(source.read_text(encoding="utf-8"))
                            if not any(
                                isinstance(node, ast.ClassDef) and node.name == cls.__name__
                                for node in tree.body
                            ):
                                continue
                            relative = source.relative_to(package_root).with_suffix("")
                            parts = list(relative.parts)
                            if parts[-1] == "__init__":
                                parts.pop()
                            candidate = "antecedent." + ".".join(parts)
                            module = importlib.import_module(candidate)
                            if getattr(module, cls.__name__, None) is cls:
                                cls.__module__ = candidate
                                break
                except (OSError, TypeError):
                    pass
            if cls is not None and hasattr(cls, "_fields") and inspect.isfunction(cls.__new__):
                # NamedTuple's synthetic constructor otherwise refers to a
                # non-importable namedtuple_* module, losing all field types.
                cls.__new__.__module__ = cls.__module__
                owner = cls.__new__
            complete_doc_namespace(public, owner, cls)
        documented = pdoc.doc.Module.from_name(name)
        for symbol, value in exports.items():
            if (inspect.isclass(value) or inspect.isfunction(value)) and not isinstance(
                documented.members.get(symbol),
                (pdoc.doc.Class, pdoc.doc.Function, pdoc.doc.Variable),
            ):
                omissions.append(f"{name}.{symbol}")
    if omissions:
        raise RuntimeError("Public API symbols missing from reference: " + ", ".join(omissions))
    print("antecedent", antecedent.__version__, antecedent.__file__)
    specs = []
    included = set()
    for name in names:
        if name not in included:
            specs.append(name)
            included.update(pdoc.extract.walk_specs([name]))
    pdoc.pdoc(*specs, output_directory=output)
    if not (output / "antecedent.html").is_file():
        raise RuntimeError("pdoc did not produce the package API page")
    absent = [name for name in names if not (output / (name.replace(".", "/") + ".html")).is_file()]
    if absent:
        raise RuntimeError("Public module pages missing: " + ", ".join(absent))
    root_html = (output / "antecedent.html").read_text(encoding="utf-8")
    for symbol in ("analyze", "prepare", "identify", "load"):
        if f'id="{symbol}"' not in root_html:
            raise RuntimeError(f"Root reference has no rendered anchor for {symbol}")
    print(f"Verified {len(names)} public module pages and exported constructors/results")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="directory for generated API HTML")
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
