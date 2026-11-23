# Uninstrumented, optimized builds of calibration test targets from one committed snapshot.
#
# A calibration measurement runs its replicates on these binaries: coverage counters in
# hot loops cost several times the work, and threads sharing them far more. Which code a
# measurement depends on is recorded separately, by the instrumented build of the same
# source (`evidence-target.nix`) running a slice of the same replicates. This recipe takes
# the same arguments as that one and compiles the same targets and features, with the
# flags below in place of instrumentation; its third-party units are built once per
# dependency source, as there.
{ src, targetsJson, depsSrc, depsTargetsJson }:
let
  lock = builtins.fromJSON (builtins.readFile ../flake.lock);
  pkgs = import (builtins.fetchTree lock.nodes.nixpkgs.locked) { system = "aarch64-darwin"; };
  lib = pkgs.lib;
  common = import ./evidence-common.nix { inherit pkgs; };
  targets = builtins.fromJSON targetsJson;
  depsTargets = builtins.fromJSON depsTargetsJson;
  vendor = common.vendor (builtins.unsafeDiscardStringContext (builtins.readFile (src + "/Cargo.lock")));
  # Optimized as a release build is, for every unit Cargo compiles for a test target,
  # with no instrumentation and no debug assertions or overflow checks.
  release = ''
    export RUSTFLAGS=""
    for profile in DEV TEST; do
      export CARGO_PROFILE_''${profile}_OPT_LEVEL=3 CARGO_PROFILE_''${profile}_DEBUG=false
      export CARGO_PROFILE_''${profile}_DEBUG_ASSERTIONS=false CARGO_PROFILE_''${profile}_OVERFLOW_CHECKS=false
    done
  '';
  deps = pkgs.stdenv.mkDerivation {
    pname = "antecedent-calibration-release-deps";
    version = "0";
    src = depsSrc;
    nativeBuildInputs = common.nativeBuildInputs;
    buildInputs = common.buildInputs;
    dontConfigure = true;
    dontStrip = true;
    dontFixup = true;
    buildPhase = ''
      ${common.environment vendor}
      ${release}
      ${lib.concatMapStrings (g: common.build g "$TMPDIR/messages") (common.groups depsTargets)}
    '';
    installPhase = ''
      # Workspace units are stubs here; drop them so the target build compiles the real source.
      python3 - "$CARGO_TARGET_DIR" <<'PY'
      import json, pathlib, shutil, subprocess, sys
      target = pathlib.Path(sys.argv[1])
      metadata = json.loads(subprocess.check_output(
          ["${pkgs.cargo}/bin/cargo", "metadata", "--format-version=1", "--no-deps", "--offline", "--locked"], text=True))
      members = {p["name"] for p in metadata["packages"]}
      for profile in (p for p in target.iterdir() if p.is_dir()):
          for kind in (".fingerprint", "build"):
              for entry in (profile / kind).glob("*") if (profile / kind).is_dir() else ():
                  if entry.name.rsplit("-", 1)[0] in members:
                      shutil.rmtree(entry)
      PY
      mkdir -p "$out"
      cp -a "$CARGO_TARGET_DIR" "$out/target"
    '';
  };
in
assert builtins.isList targets && targets != [];
pkgs.stdenv.mkDerivation {
  pname = "antecedent-calibration-release";
  version = "0";
  inherit src;
  # In place, as the instrumented build: tests resolve fixtures from CARGO_MANIFEST_DIR.
  dontUnpack = true;
  nativeBuildInputs = common.nativeBuildInputs;
  buildInputs = common.buildInputs;
  dontConfigure = true;
  dontStrip = true;
  passAsFile = [ "targetsJson" ];
  inherit targetsJson;
  buildPhase = ''
    cd ${src}
    ${common.environment vendor}
    ${release}
    cp -a ${deps}/target/. "$CARGO_TARGET_DIR/"
    chmod -R u+w "$CARGO_TARGET_DIR"
    ${lib.concatMapStrings (g: common.build g "$TMPDIR") (common.groups targets)}
  '';
  installPhase = ''
    python3 - "$TMPDIR" "$out" "$targetsJsonPath" <<'PY'
    import json, pathlib, shutil, sys
    temporary, out, targets_path = sys.argv[1:]
    for target in json.loads(pathlib.Path(targets_path).read_text(encoding="utf-8")):
        rel = f"targets/{target['crate']}/{target['kind']}/{target['name'] or 'lib'}"
        dest = pathlib.Path(out) / rel
        (dest / "bin").mkdir(parents=True)
        group = pathlib.Path(temporary) / "groups" / target["crate"] / (",".join(target["features"]) or "-")
        executables = []
        for line in open(group / "cargo-messages.jsonl", encoding="utf-8"):
            message = json.loads(line)
            kinds = message.get("target", {}).get("kind", [])
            if target["kind"] == "test":
                ours = kinds == ["test"] and message["target"].get("name") == target["name"]
            else:
                ours = not {"test", "bench", "example", "bin"} & set(kinds)
            if (message.get("reason") == "compiler-artifact" and ours
                    and message.get("profile", {}).get("test") is True and message.get("executable")):
                executables.append(message["executable"])
        if len(executables) != 1:
            raise SystemExit(f"{rel}: expected one test executable, found {len(executables)}")
        shutil.copy2(executables[0], dest / "bin/test")
    PY
  '';
}
