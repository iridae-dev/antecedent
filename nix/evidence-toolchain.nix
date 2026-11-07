{ pkgs }:

# Official Rust 1.85.1 release archives, pinned to their published SHA-256s.
# The analyzer separately verifies rustc's release and commit before earning.
let
  rust = pkgs.fetchurl {
    url = "https://static.rust-lang.org/dist/2025-03-18/rust-1.85.1-aarch64-apple-darwin.tar.xz";
    sha256 = "7a9b1db6419f47be9179b92225e02cd083483d751b700633ca248989cfcd5501";
  };
  llvmTools = pkgs.fetchurl {
    url = "https://static.rust-lang.org/dist/2025-03-18/llvm-tools-1.85.1-aarch64-apple-darwin.tar.xz";
    sha256 = "2776d83a6eebefcc6826a2fa5b08b55cee4fba09c9a8cec64a87360f38b3204a";
  };
  rustcDev = pkgs.fetchurl {
    url = "https://static.rust-lang.org/dist/2025-03-18/rustc-dev-1.85.1-aarch64-apple-darwin.tar.xz";
    sha256 = "7b0eda5134aa7592d1f950d630279067738e65729767f28fc36e134741a1e90f";
  };
in
assert pkgs.stdenv.hostPlatform.system == "aarch64-darwin";
pkgs.stdenv.mkDerivation {
  pname = "antecedent-evidence-rust";
  version = "1.85.1";
  src = rust;
  nativeBuildInputs = [ pkgs.xz ];
  dontConfigure = true;
  dontBuild = true;
  # Nix's generic strip phase damages the .rmeta members of Rust's rlibs.
  dontStrip = true;
  installPhase = ''
    runHook preInstall
    bash ./install.sh --prefix="$out" --disable-ldconfig
    mkdir -p "$TMPDIR/llvm-tools"
    tar -xJf ${llvmTools} -C "$TMPDIR/llvm-tools" --strip-components=1
    (cd "$TMPDIR/llvm-tools" && bash ./install.sh --prefix="$out" --disable-ldconfig)
    mkdir -p "$TMPDIR/rustc-dev"
    tar -xJf ${rustcDev} -C "$TMPDIR/rustc-dev" --strip-components=1
    (cd "$TMPDIR/rustc-dev" && bash ./install.sh --prefix="$out" --disable-ldconfig)
    runHook postInstall
  '';
}
