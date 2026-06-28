#!/bin/bash
set -e

# Parse args
ARCH="x86_64"
if [ "$1" = "--arch" ] && [ -n "$2" ]; then
    if [ "$2" = "arm64" ] || [ "$2" = "aarch64" ]; then
        ARCH="aarch64"
    else
        ARCH="x86_64"
    fi
else
    # Auto detect
    UNAME_M=$(uname -m)
    if [ "$UNAME_M" = "aarch64" ] || [ "$UNAME_M" = "arm64" ]; then
        ARCH="aarch64"
    else
        ARCH="x86_64"
    fi
fi

HOST_ARCH=$(uname -m)
if [ "$HOST_ARCH" = "aarch64" ] || [ "$HOST_ARCH" = "arm64" ]; then
    HOST_ARCH="aarch64"
else
    HOST_ARCH="x86_64"
fi

if [ "$ARCH" = "$HOST_ARCH" ]; then
    IS_CROSS=0
else
    IS_CROSS=1
fi

echo "[*] Target Architecture: $ARCH (Cross-compiling: $IS_CROSS)"

if [ "$ARCH" = "aarch64" ]; then
    PYTHON_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.11.9+20240415-aarch64-unknown-linux-gnu-install_only.tar.gz"
    CARGO_TARGET="aarch64-unknown-linux-gnu"
else
    PYTHON_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.11.9+20240415-x86_64-unknown-linux-gnu-install_only.tar.gz"
    CARGO_TARGET="x86_64-unknown-linux-gnu"
fi

if [ ! -d "python-embed" ]; then
    echo "[*] python-embed not found. Bootstrapping portable Python environment..."
    mkdir -p python-embed
    echo "[*] Downloading Python 3.11.9 standalone package for $ARCH..."
    curl -sL "$PYTHON_URL" | tar -xz -C python-embed --strip-components=1

    echo "[*] Downloading get-pip.py..."
    curl -sL "https://bootstrap.pypa.io/get-pip.py" -o get-pip.py

    if [ "$IS_CROSS" = "1" ]; then
        echo "[*] Cross-compiling: Downloading host Python to bootstrap packages..."
        mkdir -p host-python-embed
        if [ "$HOST_ARCH" = "aarch64" ]; then
            HOST_PY_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.11.9+20240415-aarch64-unknown-linux-gnu-install_only.tar.gz"
        else
            HOST_PY_URL="https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.11.9+20240415-x86_64-unknown-linux-gnu-install_only.tar.gz"
        fi
        curl -sL "$HOST_PY_URL" | tar -xz -C host-python-embed --strip-components=1

        HOST_PYTHON="$(pwd)/host-python-embed/bin/python3"
        $HOST_PYTHON get-pip.py

        if [ "$ARCH" = "aarch64" ]; then
            PLATFORM="manylinux2014_aarch64"
        else
            PLATFORM="manylinux2014_x86_64"
        fi

        echo "[*] Installing deep learning packages for $ARCH using host Python..."
        $HOST_PYTHON -m pip install torch torchaudio==2.4.1 transformers numpy soundfile accelerate qwen-tts \
            --target "$(pwd)/python-embed/lib/python3.11/site-packages" \
            --platform "$PLATFORM" \
            --only-binary=:all: \
            --no-warn-script-location

        rm -rf host-python-embed get-pip.py
    else
        echo "[*] Installing pip locally in python-embed..."
        ./python-embed/bin/python3 get-pip.py --no-warn-script-location
        rm get-pip.py

        echo "[*] Installing deep learning packages locally (this may take a few minutes)..."
        ./python-embed/bin/python3 -m pip install torch torchaudio==2.4.1 transformers numpy soundfile accelerate qwen-tts --no-warn-script-location
    fi

    echo "[+] Isolated Python environment setup complete!"
fi

if [ "$IS_CROSS" = "1" ]; then
    export PYO3_CROSS_PYTHON_VERSION="3.11"
    export PYO3_CROSS_LIB_DIR="$(pwd)/python-embed/lib"
else
    export PYO3_PYTHON="$(pwd)/python-embed/bin/python3"
fi

echo "[*] Starting voicecli release build for $CARGO_TARGET..."
cargo build --release --target "$CARGO_TARGET"

echo "[*] Setting up portable distribution folder..."
DIST_DIR="voicecli"
if [ -d "$DIST_DIR" ]; then
    rm -rf "$DIST_DIR"
fi
mkdir -p "$DIST_DIR"

echo "[*] Copying binary and runtime assets..."
cp "target/$CARGO_TARGET/release/voicecli" "$DIST_DIR/"
cp settings.json "$DIST_DIR/"

echo "[*] Copying portable Python environment (this may take a moment)..."
cp -r python-embed "$DIST_DIR/"

DEPLOY_FILE="deploy/voicecli-linux-${ARCH}.tar.gz"
echo "[*] Compressing into $DEPLOY_FILE..."
mkdir -p deploy
if [ -f "$DEPLOY_FILE" ]; then
    rm "$DEPLOY_FILE"
fi

tar -czf "$DEPLOY_FILE" "$DIST_DIR"

echo "[*] Cleaning up temporary folder..."
rm -rf "$DIST_DIR"

echo "[+] Success! Standalone bundle created: $DEPLOY_FILE"
