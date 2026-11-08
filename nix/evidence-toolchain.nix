{ pkgs }:

# Official Rust 1.97.1 release archives, pinned to their published SHA-256s.
# The analyzer separately verifies rustc's release and commit before earning.
let
  rust = pkgs.fetchurl {
    url = "https://static.rust-lang.org/dist/2026-07-16/rust-1.97.1-aarch64-apple-darwin.tar.xz";
    sha256 = "c9748cc86107734a2a024069908a895de7caa2d37062fb641eef9f756938ace2";
  };
  llvmTools = pkgs.fetchurl {
    url = "https://static.rust-lang.org/dist/2026-07-16/llvm-tools-1.97.1-aarch64-apple-darwin.tar.xz";
    sha256 = "2160fccee889a092a281bd6d90c09519ca75ab9ed34d9d86301f4fb9602382f8";
  };
  rustcDev = pkgs.fetchurl {
    url = "https://static.rust-lang.org/dist/2026-07-16/rustc-dev-1.97.1-aarch64-apple-darwin.tar.xz";
    sha256 = "b0b2943399dbbd0e9bae3ea55fd9319292c15aebfe0623fdf01a077cb5567e6e";
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
