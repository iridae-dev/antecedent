# Disposable PyO3 fixture only. This expression is not used to earn evidence.
{ src }:
let
  lock = builtins.fromJSON (builtins.readFile ../flake.lock);
  nixpkgsSource = builtins.fetchTree lock.nodes.nixpkgs.locked;
  pkgs = import nixpkgsSource { system = "aarch64-darwin"; };
  rust = import ./evidence-toolchain.nix { inherit pkgs; };
  python = pkgs.python313;
  vendor = pkgs.rustPlatform.importCargoLock { lockFile = src + "/Cargo.lock"; };
in
pkgs.stdenv.mkDerivation {
  pname = "antecedent-nix-python-fixture";
  version = "0";
  inherit src;
  nativeBuildInputs = [ rust pkgs.cargo python ];
  dontConfigure = true;
  dontStrip = true;
  buildPhase = ''
    export HOME="$TMPDIR/home" CARGO_HOME="$TMPDIR/cargo" CARGO_TARGET_DIR="$TMPDIR/target"
    export RUSTC=${rust}/bin/rustc PYO3_PYTHON=${python}/bin/python3.13
    export RUSTFLAGS="--remap-path-prefix=$PWD=${src}" PYTHONNOUSERSITE=1
    mkdir -p "$HOME" "$CARGO_HOME"
    cat > "$CARGO_HOME/config.toml" <<'EOF'
    [source.crates-io]
    replace-with = "vendored-sources"
    [source.vendored-sources]
    directory = "${vendor}"
    [net]
    offline = true
    EOF
    ${pkgs.cargo}/bin/cargo metadata --format-version=1 --locked --offline > "$TMPDIR/cargo-metadata.json"
    # Bootstrap enables Cargo's unstable unit-graph flag for this process only;
    # it is not present in the subsequent extension compilation.
    RUSTC_BOOTSTRAP=1 ${pkgs.cargo}/bin/cargo -Z unstable-options build --unit-graph --locked --offline --release --lib -p nix_python_boundary > "$TMPDIR/cargo-unit-graph.json"
    ${pkgs.cargo}/bin/cargo build --locked --offline --release --lib --message-format=json -p nix_python_boundary > "$TMPDIR/cargo-messages.jsonl" || {
      cat "$TMPDIR/cargo-messages.jsonl" >&2
      exit 1
    }
  '';
  installPhase = ''
    mkdir -p "$out/lib" "$out/bin"
    ${python}/bin/python3.13 - "$TMPDIR/cargo-messages.jsonl" "$out/lib/nix_python_boundary.so" <<'PY'
    import json, shutil, sys
    candidates = []
    for line in open(sys.argv[1], encoding="utf-8"):
        message = json.loads(line)
        if (message.get("reason") == "compiler-artifact"
                and message.get("target", {}).get("name") == "nix_python_boundary"
                and "cdylib" in message.get("target", {}).get("kind", [])):
            candidates.extend(path for path in message.get("filenames", []) if path.endswith(".dylib"))
    if len(candidates) != 1:
        raise SystemExit(f"expected one fixture extension, found {len(candidates)}")
    shutil.copy2(candidates[0], sys.argv[2])
    PY
    ln -s ${python}/bin/python3.13 "$out/bin/python3.13"
    ${python}/bin/python3.13 -I - "$out/python-config.json" <<'PY'
    import json, platform, sys, sysconfig
    output = {"executable": sys.executable, "version": platform.python_version(),
              "implementation": sys.implementation.name,
              "soabi": sysconfig.get_config_var("SOABI"),
              "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
              "config_args": sysconfig.get_config_var("CONFIG_ARGS")}
    with open(sys.argv[1], "w", encoding="utf-8") as stream:
        json.dump(output, stream, sort_keys=True)
        stream.write("\n")
    PY
    ${python}/bin/python3.13 - "$PWD" ${pkgs.lib.escapeShellArg (toString src)} "$TMPDIR" "$out" <<'PY'
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
