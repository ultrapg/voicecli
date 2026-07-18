#!/bin/bash
set -e

echo "[*] Preparing deploy directory..."
mkdir -p deploy

# -----------------
# 1. LINUX PORTABLE PACKAGE
# -----------------
echo "[*] Building portable package for Linux x86_64..."

if [ ! -d "python-embed" ]; then
    echo "[*] Downloading standalone Linux Python 3.11..."
    PYTHON_TARBALL="cpython_standalone.tar.gz"
    wget -q --show-progress "https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.11.7+20240107-x86_64-unknown-linux-gnu-install_only.tar.gz" -O "$PYTHON_TARBALL"
    
    echo "[*] Extracting Linux Python..."
    mkdir -p python_tmp
    tar -xf "$PYTHON_TARBALL" -C python_tmp
    mv python_tmp/python python-embed
    rm -rf python_tmp "$PYTHON_TARBALL"
    
    echo "[*] Installing required Linux Python packages (CPU only to save space)..."
    ./python-embed/bin/python3 -m pip install --upgrade pip
    ./python-embed/bin/python3 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    ./python-embed/bin/python3 -m pip install qwen-tts soundfile
fi

# Rebuild voicecli binary pointing to this Linux python-embed
echo "[*] Rebuilding voicecli binary for Linux..."
export PYO3_PYTHON="$(pwd)/python-embed/bin/python3"
cargo build --release --target x86_64-unknown-linux-gnu

# Package Linux
rm -rf deploy/voicecli-linux
mkdir -p deploy/voicecli-linux
cp target/x86_64-unknown-linux-gnu/release/voicecli deploy/voicecli-linux/
cp settings.json deploy/voicecli-linux/
cp model.py deploy/voicecli-linux/
cp -rL python-embed deploy/voicecli-linux/python-embed

echo "[*] Creating Linux tarball..."
tar -czf deploy/voicecli-linux-x86_64.tar.gz -C deploy voicecli-linux
echo "[+] Linux tarball created at deploy/voicecli-linux-x86_64.tar.gz"

# -----------------
# 2. WINDOWS PORTABLE PACKAGE
# -----------------
echo "[*] Building portable package for Windows x86_64..."

# Compile Windows binary inside container
echo "[*] Compiling Windows binary inside Podman..."
podman run --rm -v "$(pwd):/workspace" -w /workspace docker.io/library/rust:latest bash -c "
  apt-get update -qq && apt-get install -y -qq mingw-w64
  rustup target add x86_64-pc-windows-gnu
  export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc
  export PYO3_CROSS_PYTHON_VERSION=3.11
  cargo build --release --target x86_64-pc-windows-gnu
"

# Package Windows
rm -rf deploy/voicecli-windows
mkdir -p deploy/voicecli-windows
cp target/x86_64-pc-windows-gnu/release/voicecli.exe deploy/voicecli-windows/
cp settings.json deploy/voicecli-windows/
cp model.py deploy/voicecli-windows/

WIN_PY_DIR="deploy/voicecli-windows/python-embed"
echo "[*] Downloading Windows Python 3.11 embeddable package..."
mkdir -p win_py_tmp
wget -q --show-progress "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip" -O win_py_tmp/python_embed.zip

echo "[*] Extracting Windows Python..."
mkdir -p "$WIN_PY_DIR"
7z x -y win_py_tmp/python_embed.zip -o"$WIN_PY_DIR" > /dev/null
rm -rf win_py_tmp

echo "[*] Setting up python311._pth import paths..."
PTH_FILE="$WIN_PY_DIR/python311._pth"
if [ -f "$PTH_FILE" ]; then
    echo "import site" >> "$PTH_FILE"
fi

echo "[*] Installing Windows Python packages..."
mkdir -p "$WIN_PY_DIR/Lib/site-packages"

# Use host python/pip to cross-install Windows wheels
python3 -m pip install \
    --platform win_amd64 \
    --target "$WIN_PY_DIR/Lib/site-packages" \
    --only-binary=:all: \
    --implementation cp \
    --python-version 311 \
    --upgrade \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu > /dev/null
    
python3 -m pip install \
    --platform win_amd64 \
    --target "$WIN_PY_DIR/Lib/site-packages" \
    --only-binary=:all: \
    --implementation cp \
    --python-version 311 \
    --upgrade \
    qwen-tts soundfile > /dev/null

echo "[*] Compressing Windows deployment package into zip..."
rm -f deploy/voicecli-windows.zip
# Use 7z relative path to avoid full paths in the zip
cd deploy
7z a -tzip voicecli-windows.zip voicecli-windows/ > /dev/null
cd ..

echo "[+] Windows deployment zip created at deploy/voicecli-windows.zip"

echo "[+] Done creating all portable packages!"
