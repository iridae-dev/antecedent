# Evidence build of a list of Cargo test targets from one committed snapshot. Its
# caller must supply an independently verified, immutable snapshot of the intended
# commit; this expression does not attest that source identity or that the Nix
# daemon actually used a sandbox.
#
# Each target runs its own `cargo test --no-run` invocation, exactly as it would
# alone, in sequence over one shared target directory: Cargo reuses every unit whose
# fingerprint matches, so dependencies compile once per distinct feature set. A
# single invocation over several targets would unify their features and is never used.
{ src, targetsJson, depsSrc, depsTargetsJson }:
let
  lock = builtins.fromJSON (builtins.readFile ../flake.lock);
  nixpkgsSource = builtins.fetchTree lock.nodes.nixpkgs.locked;
  pkgs = import nixpkgsSource { system = "aarch64-darwin"; };
  lib = pkgs.lib;
  common = import ./evidence-common.nix { inherit pkgs; };
  rust = common.rust;
  dir = common.dir;
  targets = builtins.fromJSON targetsJson;
  vendor = common.vendor (builtins.unsafeDiscardStringContext (builtins.readFile (src + "/Cargo.lock")));
  deps = import ./evidence-deps.nix { inherit depsSrc depsTargetsJson; };
  # Content-addressed, so the wrapper's store path changes only with the driver.
  driver = import ./evidence-driver.nix {
    inherit pkgs;
    driver = builtins.path { path = src + "/tools/evidence-analyzer/driver.rs"; name = "evidence-driver.rs"; };
  };
  valid = t: (t.kind == "lib" && t.name == "") || (t.kind == "test" && t.name != "");
  # Metadata resolves from the workspace root rather than from `-p crate`.
  # Qualify features so an unrelated default member is not selected instead.
  metadataFeatureArgs = t: if t.features == [] then [] else
    [ "--features" (builtins.concatStringsSep "," (map (feature: "${t.crate}/${feature}") t.features)) ];
  step = t: ''
    mkdir -p "$TMPDIR/${dir t}"
    ${pkgs.cargo}/bin/cargo ${lib.escapeShellArgs ([ "metadata" "--format-version=1" "--locked" "--offline" ] ++ metadataFeatureArgs t)} > "$TMPDIR/${dir t}/cargo-metadata.json"
    # Cargo's unit graph is unstable metadata: enable its flag only in this
    # subprocess. The test build itself never sees RUSTC_BOOTSTRAP.
    RUSTC_BOOTSTRAP=1 ${pkgs.cargo}/bin/cargo -Z unstable-options ${lib.escapeShellArgs ([ "test" "--unit-graph" "--locked" "--offline" "--no-run" "-p" t.crate ] ++ common.targetArgs t ++ common.featureArgs t)} > "$TMPDIR/${dir t}/cargo-unit-graph.json"
    ${common.build t "$TMPDIR"}
  '';
in
assert builtins.isList targets && targets != [];
assert builtins.all (t: valid t && builtins.isList t.features && builtins.all builtins.isString t.features) targets;
pkgs.stdenv.mkDerivation {
  pname = "antecedent-evidence-targets";
  version = "0";
  inherit src;
  nativeBuildInputs = common.nativeBuildInputs;
  buildInputs = common.buildInputs;
  dontConfigure = true;
  dontStrip = true;
  passAsFile = [ "targetsJson" ];
  inherit targetsJson;
  buildPhase = ''
    ${common.environment vendor}
    # Start from the compiled third-party units; Cargo recompiles workspace crates
    # and anything whose fingerprint differs.
    cp -a ${deps}/target/. "$CARGO_TARGET_DIR/"
    chmod -R u+w "$CARGO_TARGET_DIR"
    # Inventory only this fresh workspace compilation; the analyzer is pinned
    # to the same rustc and never wraps external dependency compilations. A
    # stable prefix keeps file!() independent of the Nix store hash, which
    # changes even for unrelated commits. Coverage is mapped back only after
    # verifying the store source against the committed checkout.
    mkdir -m 700 "$TMPDIR/compiler-inventory"
    export RUSTC_WORKSPACE_WRAPPER="${driver}/bin/workspace-rustc" ANTECEDENT_SOURCE_ROOT="$PWD"
    export ANTECEDENT_ANALYZER_OUTPUT="$TMPDIR/compiler-inventory"
    ${lib.concatMapStrings step targets}
  '';
  installPhase = ''
    mkdir -p "$out/bin"
    # Cargo emits absolute unpack/target paths in metadata. Preserve the raw
    # graph structure while mapping temporary paths to stable identities so a
    # local rebuild has the same output and downstream validation can compare
    # package IDs without depending on Nix's build-directory suffix.
    python3 - "$PWD" ${lib.escapeShellArg (toString src)} "$TMPDIR" "$out" "$targetsJsonPath" <<'PY'
    import json, pathlib, shutil, sys
    workdir, source, temporary, out, targets_path = sys.argv[1:]
    def remap(value):
        if isinstance(value, str):
            return value.replace(workdir, source).replace(temporary, "/__nix_build_tmp__")
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, dict):
            return {key: remap(item) for key, item in value.items()}
        return value
    for target in json.loads(pathlib.Path(targets_path).read_text(encoding="utf-8")):
        rel = f"targets/{target['crate']}/{target['kind']}/{target['name'] or 'lib'}"
        built, dest = pathlib.Path(temporary) / rel, pathlib.Path(out) / rel
        (dest / "bin").mkdir(parents=True)
        executables = []
        for line in open(built / "cargo-messages.jsonl", encoding="utf-8"):
            message = json.loads(line)
            if (message.get("reason") == "compiler-artifact"
                    and message.get("profile", {}).get("test") is True
                    and message.get("executable")):
                executables.append(message["executable"])
        if len(executables) != 1:
            raise SystemExit(f"{rel}: expected one test executable, found {len(executables)}")
        shutil.copy2(executables[0], dest / "bin/test")
        for name in ("cargo-metadata.json", "cargo-unit-graph.json"):
            data = json.loads((built / name).read_text(encoding="utf-8"))
            (dest / name).write_text(json.dumps(remap(data), sort_keys=True) + "\n", encoding="utf-8")
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
    ln -s ${rust}/lib/rustlib/aarch64-apple-darwin/bin/llvm-profdata "$out/bin/llvm-profdata"
    ln -s ${rust}/lib/rustlib/aarch64-apple-darwin/bin/llvm-cov "$out/bin/llvm-cov"
    printf '%s\n' ${lib.escapeShellArg (toString src)} > "$out/source-root"
    ${rust}/bin/rustc -vV > "$out/rustc-version"
    ${pkgs.cargo}/bin/cargo --version > "$out/cargo-version"
  '';
}
