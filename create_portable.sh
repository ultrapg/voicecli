#!/usr/bin/env bash
# ==============================================================================
# voicecli - Linux Standalone Single-Binary Portable Package Generator
# Compiles a single, unified binary with statically embedded Multi-ISA GGML C++
# engine (Baseline x86-64 SSE2 + SSE4.2 + AVX2/FMA/F16C + AVX-512 runtime CPU
# dispatch) and packages it into a clean .tar.gz bundle for Linux (x86_64).
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "      voicecli - Single-Binary Portable Bundle Builder      "
echo "============================================================"

# 1. Check prerequisites
for cmd in cargo cmake curl tar git nm objcopy ranlib readelf; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "[-] Error: Required tool '$cmd' is not installed or not in PATH."
        exit 1
    fi
done

# 2. Ensure external/qwen3-tts is present
if [ ! -f "external/qwen3-tts/CMakeLists.txt" ]; then
    echo "[*] Cloning native qwen3-tts.cpp engine (with GGML submodules)..."
    mkdir -p external
    git clone --depth 1 --recursive https://github.com/Danmoreng/qwen3-tts.cpp.git external/qwen3-tts
fi

# 3. Enforce portable baseline x86-64 codegen for Rust unless VOICECLI_NATIVE=1
if [ "${VOICECLI_NATIVE:-0}" != "1" ] && [ "$(uname -m)" = "x86_64" ]; then
    export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=x86-64"
fi

# 4. Build voicecli (statically linking Multi-ISA GGML C++ engine into single binary)
echo "[*] Step 1/3: Compiling single unified portable voicecli release binary..."
cargo build --release --offline || cargo build --release

# 5. Assemble the portable distribution folder (single binary + settings)
echo "[*] Step 2/3: Assembling portable distribution..."
DIST_DIR="voicecli"
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

# Place the single binary and settings directly into the folder
cp "target/release/voicecli" "${DIST_DIR}/voicecli"
chmod +x "${DIST_DIR}/voicecli"

if [ -f "settings.json" ]; then
    cp "settings.json" "${DIST_DIR}/settings.json"
fi

# 6. Compress into .tar.gz
echo "[*] Step 3/3: Compressing into deploy/voicecli-linux-x86_64.tar.gz..."
mkdir -p deploy
ARCHIVE_NAME="deploy/voicecli-linux-x86_64.tar.gz"
rm -f "${ARCHIVE_NAME}"
tar -czf "${ARCHIVE_NAME}" "${DIST_DIR}"

echo "[*] Cleaning up temporary staging directory..."
rm -rf "${DIST_DIR}"

echo ""
echo "============================================================"
echo "[+] SUCCESS! Portable single-binary package created:"
echo "    -> ${ARCHIVE_NAME}"
echo "    Size: $(du -h "${ARCHIVE_NAME}" | cut -f1)"
echo "    (Multi-ISA runtime CPU dispatch: x86-64 / SSE4.2 / AVX2 / AVX-512)"
echo "============================================================"
