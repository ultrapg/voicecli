#!/usr/bin/env bash
# ==============================================================================
# voicecli - Linux Standalone Single-Binary Portable Package Generator
# Compiles a single, unified binary with statically embedded GGML C++ engine
# and packages it into a clean .tar.gz bundle for Linux (x86_64).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "============================================================"
echo "      voicecli - Single-Binary Portable Bundle Builder      "
echo "============================================================"

# 1. Check prerequisites
for cmd in cargo cmake curl tar; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "[-] Error: Required tool '$cmd' is not installed or not in PATH."
        exit 1
    fi
done

# 2. Build voicecli (statically linking GGML C++ engine into single binary)
echo "[*] Step 1/3: Compiling single unified voicecli release binary..."
cargo build --release --offline || cargo build --release

# 3. Assemble the portable distribution folder (single binary + settings)
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

# 4. Compress into .tar.gz
echo "[*] Step 3/3: Compressing into deploy/voicecli-linux-x86_64.tar.gz..."
mkdir -p deploy
ARCHIVE_NAME="deploy/voicecli-linux-x86_64.tar.gz"
rm -f "${ARCHIVE_NAME}"
tar -czf "${ARCHIVE_NAME}" "${DIST_DIR}"

echo "[*] Cleaning up temporary staging directory..."
rm -rf "${DIST_DIR}"

echo ""
echo "============================================================"
echo "[+] SUCCESS! Single-binary portable package created:"
echo "    -> ${ARCHIVE_NAME}"
echo "    Size: $(du -h "${ARCHIVE_NAME}" | cut -f1)"
echo "    (Single binary does everything; models auto-download on first run)"
echo "============================================================"
