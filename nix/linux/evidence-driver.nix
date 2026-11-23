# The Linux form of ../evidence-driver.nix: the compiler-inventory driver and the
# workspace wrapper that runs it.
#
# Cargo folds the workspace wrapper's path into every workspace crate's metadata
# hash, so the wrapper must not live in a per-build scratch directory: a store
# path keeps symbol names, and so the evidence binaries, identical across builds.
# The source prefix to remap reaches the wrapper through the environment, which
# Cargo does not hash. `driver` is the content-addressed driver source.
{ pkgs, driver }:
let
  rust = import ./evidence-toolchain.nix { inherit pkgs; };
in
pkgs.stdenv.mkDerivation {
  pname = "antecedent-evidence-driver";
  version = "0";
  dontUnpack = true;
  nativeBuildInputs = [ rust ];
  # librustc_driver links zlib.
  buildInputs = [ pkgs.zlib ];
  dontStrip = true;
  buildPhase = ''
    RUSTC_BOOTSTRAP=1 ${rust}/bin/rustc ${driver} --edition=2021 -C link-arg=-Wl,-rpath,${rust}/lib/rustlib/${rust.triple}/lib -C link-arg=-Wl,-rpath,${rust}/lib -o evidence-driver
  '';
  installPhase = ''
    mkdir -p "$out/bin"
    cp evidence-driver "$out/bin/evidence-driver"
    cat > "$out/bin/workspace-rustc" <<EOF
    #!${pkgs.bash}/bin/bash
    # A stable prefix keeps file!() and panic locations independent of the
    # source's store path. ANTECEDENT_SOURCE_ROOT is the committed source. The
    # sandbox mounts every build's scratch directory at one fixed path and an
    # ELF link records no object-file paths, so no linker flag is needed.
    exec "$out/bin/evidence-driver" "\$@" "--remap-path-prefix=\$ANTECEDENT_SOURCE_ROOT=/antecedent-source"
    EOF
    chmod 555 "$out/bin/workspace-rustc"
  '';
}
