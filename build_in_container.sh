#!/bin/bash
set -e

echo "[*] Starting compilation in podman..."

# Run podman container, mounting current workspace to /workspace
podman run --rm -v "$(pwd):/workspace" -w /workspace docker.io/library/rust:latest bash -c "
  set -e
  echo '[*] Updating apt and installing dependencies...'
  apt-get update -qq && apt-get install -y -qq mingw-w64 python3-dev
  
  echo '[*] Adding target x86_64-pc-windows-gnu...'
  rustup target add x86_64-pc-windows-gnu
  
  echo '[*] Building for Linux x86_64...'
  cargo build --release --target x86_64-unknown-linux-gnu
  
  echo '[*] Building for Windows x86_64...'
  export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc
  export PYO3_CROSS_PYTHON_VERSION=3.11
  cargo build --release --target x86_64-pc-windows-gnu
"

echo "[+] Compilation successful!"
echo "[*] Binaries are available at:"
echo "    - Linux: target/x86_64-unknown-linux-gnu/release/voicecli"
echo "    - Windows: target/x86_64-pc-windows-gnu/release/voicecli.exe"
