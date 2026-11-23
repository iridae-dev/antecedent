# Disposable metadata-only diagnostic. It does not compile or execute tests,
# attest its sandbox or authorize evidence. The caller must verify `src`.
{ src, crate, targetKind, targetName ? "", featuresJson ? "[]", vendorOverride ? null }:
let
  lock = builtins.fromJSON (builtins.readFile ../flake.lock);
  nixpkgsSource = builtins.fetchTree lock.nodes.nixpkgs.locked;
  pkgs = import nixpkgsSource { system = "aarch64-darwin"; };
  rust = import ./evidence-toolchain.nix { inherit pkgs; };
  features = builtins.fromJSON featuresJson;
  # A supplied vendorOverride is a diagnostic input only. In particular,
  # cargo-vendor can omit archive entries, and this is not proof of equality
  # with the production importCargoLock output.
  vendor = if vendorOverride == null then
    pkgs.rustPlatform.importCargoLock { lockFile = src + "/Cargo.lock"; }
    else vendorOverride;
  targetArgs = if targetKind == "lib" then [ "--lib" ] else [ "--test" targetName ];
  featureArgs = if features == [] then [] else [ "--features" (builtins.concatStringsSep "," features) ];
  graphArgs = [ "test" "--unit-graph" "--locked" "--offline" "--no-run" "-p" crate ] ++ targetArgs ++ featureArgs;
  metadataArgs = [ "metadata" "--format-version=1" "--locked" "--offline" ] ++
    (if features == [] then [] else [ "--features" (builtins.concatStringsSep "," (map (f: "${crate}/${f}") features)) ]);
in
assert targetKind == "lib" || targetKind == "test";
assert targetKind != "test" || targetName != "";
assert builtins.isList features && builtins.all builtins.isString features;
pkgs.stdenv.mkDerivation {
  pname = "antecedent-evidence-unit-graph";
  version = "0";
  inherit src;
  nativeBuildInputs = [ rust pkgs.cargo pkgs.python3 ];
  dontConfigure = true;
  dontStrip = true;
  buildPhase = ''
    export HOME="$TMPDIR/home" CARGO_HOME="$TMPDIR/cargo" CARGO_TARGET_DIR="$TMPDIR/target"
    export RUSTC=${rust}/bin/rustc
    export RUSTFLAGS="-C instrument-coverage --remap-path-prefix=$PWD=${src}"
    export CARGO_PROFILE_TEST_OPT_LEVEL=2 CARGO_PROFILE_TEST_DEBUG=line-tables-only
    mkdir -p "$HOME" "$CARGO_HOME"
    cat > "$CARGO_HOME/config.toml" <<'EOF'
    [source.crates-io]
    replace-with = "vendored-sources"
    [source.vendored-sources]
    directory = "${vendor}"
    [net]
    offline = true
    EOF
    ${pkgs.cargo}/bin/cargo ${pkgs.lib.escapeShellArgs metadataArgs} > "$TMPDIR/cargo-metadata.json"
    RUSTC_BOOTSTRAP=1 ${pkgs.cargo}/bin/cargo -Z unstable-options ${pkgs.lib.escapeShellArgs graphArgs} > "$TMPDIR/cargo-unit-graph.json"
  '';
  installPhase = ''
    mkdir -p "$out"
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
    PY
    printf '%s\n' ${pkgs.lib.escapeShellArg (toString src)} > "$out/source-root"
    ${rust}/bin/rustc -vV > "$out/rustc-version"
    ${pkgs.cargo}/bin/cargo --version > "$out/cargo-version"
  '';
}
