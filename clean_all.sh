#!/bin/bash

echo "[*] Cleaning Rust compilation targets..."
cargo clean

echo "[*] Deleting local portable Python environment (python-embed)..."
if [ -d "python-embed" ]; then
    rm -rf python-embed
fi

echo "[*] Deleting local model weights cache (.cache)..."
if [ -d ".cache" ]; then
    rm -rf .cache
fi

echo "[*] Deleting generated ZIP files..."
if [ -f "voicecli-portable.zip" ]; then
    rm -f voicecli-portable.zip
fi

echo "[*] Deleting portable deployment directory (deploy)..."
if [ -d "deploy" ]; then
    rm -rf deploy
fi

echo "[+] Cleanup complete! The workspace is now in a pure source-code state and ready for GitHub."
