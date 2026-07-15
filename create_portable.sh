#!/bin/bash
set -e

# Check for GPU flag
USE_GPU=false
if [ "$1" == "gpu" ]; then
    USE_GPU=true
    echo "[*] GPU mode enabled. Will install CUDA-enabled PyTorch and flash-attn."
fi

# Check if the full environment needs to be bootstrapped
if [ ! -d "python-embed" ]; then
    echo "[*] python-embed not found. Bootstrapping portable Python environment..."
    mkdir -p python-embed
    
    URL="https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.11.9+20240415-x86_64-unknown-linux-gnu-install_only.tar.gz"
    TARBALL="python-embed.tar.gz"
    
    echo "[*] Downloading Python 3.11.9 standalone package..."
    curl -L "$URL" -o "$TARBALL"
    
    echo "[*] Extracting Python to python-embed..."
    # python-build-standalone tar.gz extracts to a "python" folder
    rm -rf python
    tar -xzf "$TARBALL"
    # Move the contents of "python" into "python-embed"
    mv python/* python-embed/
    rm -rf python "$TARBALL"
    
    echo "[*] Upgrading pip..."
    ./python-embed/bin/python3 -m pip install --upgrade pip
    
    echo "[*] Installing deep learning packages locally (this may take a few minutes)..."
    # Create a temporary directory on the SSD to avoid running out of space in /tmp (tmpfs)
    mkdir -p .pip_tmp
    export TMPDIR="$(pwd)/.pip_tmp"
    export PIP_CACHE_DIR="$(pwd)/.pip_cache"
    
    if [ "$USE_GPU" = true ]; then
        echo "[*] Installing CUDA-enabled PyTorch..."
        ./python-embed/bin/python3 -m pip install torch torchaudio
        echo "[*] Installing flash-attn (this may take a few minutes)..."
        ./python-embed/bin/python3 -m pip install flash-attn --no-build-isolation || echo "[-] Warning: flash-attn installation failed, proceeding without it."
    else
        echo "[*] Installing CPU-only PyTorch..."
        ./python-embed/bin/python3 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    ./python-embed/bin/python3 -m pip install transformers numpy soundfile accelerate qwen-tts
    
    echo "[*] Cleaning up python-embed to minimize size..."
    find python-embed -type d -name "__pycache__" -exec rm -rf {} + || true
    rm -rf .pip_tmp .pip_cache
    echo "[+] Isolated Python environment setup complete!"
fi

echo "[*] Starting voicecli release build..."
PYO3_PYTHON="$(pwd)/python-embed/bin/python3" cargo build --release

echo "[*] Setting up portable distribution folder..."
DIST_DIR="voicecli"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

echo "[*] Copying binary and assets..."
cp target/release/voicecli "$DIST_DIR"/
cp settings.json "$DIST_DIR"/

echo "[*] Copying portable Python environment..."
cp -r python-embed "$DIST_DIR"/

echo "[*] Compressing into deploy/voicecli.tar.gz..."
mkdir -p deploy
rm -f deploy/voicecli.tar.gz
tar -czf deploy/voicecli.tar.gz "$DIST_DIR"

echo "[*] Cleaning up temporary folder..."
rm -rf "$DIST_DIR"

echo "[+] Success! Standalone bundle created: deploy/voicecli.tar.gz"
