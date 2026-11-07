# Prototype target build. Its caller must supply an independently verified,
# immutable snapshot of the intended commit; this expression does not attest
# that source identity or that the Nix daemon actually used a sandbox.
{ src, crate, targetKind, targetName ? "", featuresJson ? "[]" }:
let
  lock = builtins.fromJSON (builtins.readFile ../flake.lock);
  nixpkgsSource = builtins.fetchTree lock.nodes.nixpkgs.locked;
  pkgs = import nixpkgsSource { system = "aarch64-darwin"; };
  rust = import ./evidence-toolchain.nix { inherit pkgs; };
  features = builtins.fromJSON featuresJson;
  vendor = pkgs.rustPlatform.importCargoLock { lockFile = src + "/Cargo.lock"; };
  targetArgs = if targetKind == "lib" then [ "--lib" ] else [ "--test" targetName ];
  featureArgs = if features == [] then [] else [ "--features" (builtins.concatStringsSep "," features) ];
  cargoArgs = [ "test" "--locked" "--offline" "--no-run" "--message-format=json" "-p" crate ] ++ targetArgs ++ featureArgs;
  graphArgs = [ "test" "--unit-graph" "--locked" "--offline" "--no-run" "-p" crate ] ++ targetArgs ++ featureArgs;
  # Metadata resolves from the workspace root rather than from `-p crate`.
  # Qualify features so an unrelated default member is not selected instead.
  metadataFeatureArgs = if features == [] then [] else
    [ "--features" (builtins.concatStringsSep "," (map (feature: "${crate}/${feature}") features)) ];
  metadataArgs = [ "metadata" "--format-version=1" "--locked" "--offline" ] ++ metadataFeatureArgs;
in
assert targetKind == "lib" || targetKind == "test";
assert targetKind != "test" || targetName != "";
assert builtins.isList features && builtins.all builtins.isString features;
pkgs.stdenv.mkDerivation {
  pname = "antecedent-evidence-target";
  version = "0";
  inherit src;
  # Cargo from nixpkgs uses store-provided curl/OpenSSL rather than the host
  # LibreSSL used by the upstream Cargo binary. The compiler remains pinned to
  # the separately verified official 1.85.1 toolchain.
  nativeBuildInputs = [ rust pkgs.cargo pkgs.python3 ];
  dontConfigure = true;
  dontStrip = true;
  buildPhase = ''
    # Build scripts may reuse probe files in OUT_DIR. Never use an ambient or
    # shared scratch tree: Nix's per-build TMPDIR must belong to this builder.
    umask 077
    python3 - "$TMPDIR" <<'PY'
    import os, stat, sys
    path = sys.argv[1]
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise SystemExit("Nix build scratch directory must be private to the builder")
    PY
    export HOME="$TMPDIR/home" CARGO_HOME="$TMPDIR/cargo" CARGO_TARGET_DIR="$TMPDIR/target"
    export RUSTC=${rust}/bin/rustc
    # A stable prefix keeps file!() independent of the Nix store hash, which
    # changes even for unrelated commits. Coverage is mapped back only after
    # verifying the store source against the committed checkout.
    export RUSTFLAGS="-C instrument-coverage --remap-path-prefix=$PWD=/antecedent-source" CARGO_PROFILE_TEST_OPT_LEVEL=2 CARGO_PROFILE_TEST_DEBUG=line-tables-only
    mkdir -m 700 "$HOME" "$CARGO_HOME" "$CARGO_TARGET_DIR"
    # The private parents protect scratch; outputs must remain readable after
    # Nix canonicalizes ownership in the store.
    umask 022
    cat > "$CARGO_HOME/config.toml" <<'EOF'
    [source.crates-io]
    replace-with = "vendored-sources"
    [source.vendored-sources]
    directory = "${vendor}"
    [net]
    offline = true
    EOF
    # Cargo's unit graph is unstable metadata: enable its flag only in this
    # subprocess. The test build itself never sees RUSTC_BOOTSTRAP.
    ${pkgs.cargo}/bin/cargo ${pkgs.lib.escapeShellArgs metadataArgs} > "$TMPDIR/cargo-metadata.json"
    RUSTC_BOOTSTRAP=1 ${pkgs.cargo}/bin/cargo -Z unstable-options ${pkgs.lib.escapeShellArgs graphArgs} > "$TMPDIR/cargo-unit-graph.json"
    # Inventory only this fresh workspace compilation; the analyzer is pinned
    # to the same rustc and never wraps external dependency compilations.
    mkdir -m 700 "$TMPDIR/compiler-inventory"
    RUSTC_BOOTSTRAP=1 ${rust}/bin/rustc "$PWD/tools/evidence-analyzer/driver.rs" --edition=2021 -C rpath=yes -o "$TMPDIR/evidence-driver"
    export RUSTC_WORKSPACE_WRAPPER="$TMPDIR/evidence-driver" ANTECEDENT_ANALYZER_OUTPUT="$TMPDIR/compiler-inventory"
    ${pkgs.cargo}/bin/cargo ${pkgs.lib.escapeShellArgs cargoArgs} > "$TMPDIR/cargo-messages.jsonl" || {
      cat "$TMPDIR/cargo-messages.jsonl" >&2
      exit 1
    }
  '';
  installPhase = ''
    mkdir -p "$out/bin"
    python3 - "$TMPDIR/cargo-messages.jsonl" "$out/bin/test" <<'PY'
    import json, shutil, sys
    executables = []
    for line in open(sys.argv[1], encoding="utf-8"):
        message = json.loads(line)
        if (message.get("reason") == "compiler-artifact"
                and message.get("profile", {}).get("test") is True
                and message.get("executable")):
            executables.append(message["executable"])
    if len(executables) != 1:
        raise SystemExit(f"expected one test executable, found {len(executables)}")
    shutil.copy2(executables[0], sys.argv[2])
    PY
    ln -s ${rust}/lib/rustlib/aarch64-apple-darwin/bin/llvm-profdata "$out/bin/llvm-profdata"
    ln -s ${rust}/lib/rustlib/aarch64-apple-darwin/bin/llvm-cov "$out/bin/llvm-cov"
    printf '%s\n' ${pkgs.lib.escapeShellArg (toString src)} > "$out/source-root"
    # Cargo emits absolute unpack/target paths in metadata. Preserve the raw
    # graph structure while mapping temporary paths to stable identities so a
    # local rebuild has the same output and downstream validation can compare
    # package IDs without depending on Nix's build-directory suffix.
    python3 - "$PWD" ${pkgs.lib.escapeShellArg (toString src)} "$TMPDIR" "$out" <<'PY'
    import json, pathlib, sys
    workdir, source, temporary, out = sys.argv[1:]
    def remap(value):
        if isinstance(value, str):
            return value.replace(workdir, source).replace(temporary, "/__nix_build_tmp__")
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, dict):
            return {key: remap(item) for key, item in value.items()}
        return value
    for name in ("cargo-metadata.json", "cargo-unit-graph.json"):
        data = json.loads((pathlib.Path(temporary) / name).read_text(encoding="utf-8"))
        (pathlib.Path(out) / name).write_text(json.dumps(remap(data), sort_keys=True) + "\n", encoding="utf-8")
    records = set()
    for inventory in (pathlib.Path(temporary) / "compiler-inventory").glob("*.tsv"):
        for line in inventory.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) != 6 or fields[0] != "body":
                raise SystemExit("unexpected compiler inventory record")
            path = bytes.fromhex(fields[2]).decode("utf-8")
            if path == "/antecedent-source" or path.startswith("/antecedent-source/"):
                path = workdir + path[len("/antecedent-source"):]
            if path != workdir and not path.startswith(workdir + "/"):
                raise SystemExit("compiler inventory references a non-workspace source")
            fields[2] = (source + path[len(workdir):]).encode("utf-8").hex()
            records.add("\t".join(fields))
    if not records:
        raise SystemExit("compiler inventory is empty")
    inventory_dir = pathlib.Path(out) / "compiler-inventory"
    inventory_dir.mkdir()
    (inventory_dir / "bodies.tsv").write_text("\n".join(sorted(records)) + "\n", encoding="utf-8")
    PY
    ${rust}/bin/rustc -vV > "$out/rustc-version"
    ${pkgs.cargo}/bin/cargo --version > "$out/cargo-version"
  '';
}
