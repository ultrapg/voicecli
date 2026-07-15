#!/bin/bash
set -e

echo "[*] Cleaning Rust compilation targets..."
cargo clean

echo "[*] Deleting local portable Python environment (python-embed)..."
rm -rf python-embed

echo "[*] Deleting local model weights cache (.cache)..."
rm -rf .cache

echo "[*] Deleting generated ZIP files..."
rm -f voicecli-portable.zip voicecli.zip deploy/voicecli.zip

echo "[+] Cleanup complete! The workspace is now in a pure source-code state."
