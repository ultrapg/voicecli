# voicecli - Standalone Qwen3-TTS Rust CLI

`voicecli` is a lightweight, standalone Command Line Interface (CLI) application written in Rust that interfaces with the **Qwen3-TTS** model series using an embedded Python interpreter. It allows you to generate speech from text using natural language description prompts (Voice Design) or clone a voice from a short reference audio clip (Voice Cloning).

Because Qwen3-TTS depends on PyTorch and Hugging Face's `transformers` library, `voicecli` uses the `PyO3` library to orchestrate Python's deep learning stack directly in-memory, compiling down into a single binary.

---

## Architecture Overview

```
                        +----------------------------+
                        |       voicecli (Rust)      |
                        |                            |
                        |   - Parses CLI (clap)      |
                        |   - Performs Path Checks   |
                        |   - Boots Python GIL       |
                        +--------------+-------------+
                                       |
                                       v (PyO3 GIL interface)
                        +--------------+-------------+
                        |      Embedded Python       |
                        |   - model.py (In-Memory)   |
                        |   - torch / torchaudio     |
                        |   - transformers API       |
                        +--------------+-------------+
                                       |
                                       v
                        +--------------+-------------+
                        |   Qwen3-TTS-12Hz-1.7B-Base |
                        |   - VRAM / RAM Inference   |
                        |   - Generates Speech       |
                        +----------------------------+
```

1. **Rust Binary**: Handles CLI argument parsing via `clap` and path validation. It starts the Python interpreter and loads the embedded Python script.
2. **Embedded Python (`model.py`)**: Runs inside the same process space. It loads `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (for style descriptions) or `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (for cloning) and runs the PyTorch GPU/CPU inference.
3. **Data Exchange**: Inputs (strings, audio arrays) are passed between Rust and Python boundary layers without starting network servers or sub-processes.

---

## System Requirements & Prerequisites

### 1. Portable Python Environment
A local portable Python 3.11.9 environment must be configured in the `python-embed/` directory, containing the required deep learning dependencies (`torch`, `torchaudio`, `transformers`, `numpy`, `soundfile`, `accelerate`, and `qwen-tts`).

### 2. Rust Toolchain
Rust 1.75+ is required to build the project. If not installed, you can get it from [rustup.rs](https://rustup.rs/).

---

## How to Build

### Windows
To compile a release binary and package it automatically into a redistributable standalone ZIP file, run the following script in your Command Prompt:
```cmd
create_portable.cmd
```
This batch script will:
1. Compile `voicecli` in release mode (`cargo build --release`).
2. Gather the compiled executable (`voicecli.exe`), required DLLs (`python311.dll`, `vcruntime140.dll`, etc.), standard library ZIPs (`python311.zip`), and the `python-embed/` runtime folder into a staging directory.
3. Compress the folder into a single standalone archive: **`deploy/voicecli.zip`**.
4. Clean up the temporary staging directory.

### Linux
To compile a release binary and package it automatically into a redistributable standalone tarball, run the following script:
```bash
./create_portable.sh
```
This script will:
1. Setup a portable standalone Python environment in `python-embed/` (using Python Build Standalone) and install all python dependencies.
2. Compile `voicecli` in release mode (`cargo build --release`).
3. Package the executable `voicecli` and the isolated `python-embed/` environment into a single standalone archive: **`deploy/voicecli.tar.gz`**.


## Usage

The CLI has two subcommands: `style` and `clone`.

### 1. `style` (Voice Design / Style Prompts)
Generates speech based on a natural language description of the acoustic voice.

```bash
# Basic usage
voicecli style --text "Welcome to the future of speech synthesis." --prompt "gender: Female. pitch: High. speed: Normal." --output style_out.wav

# Short option usage
voicecli style -t "This is a test of voice design." -p "gender: Male. pitch: Low. speed: Fast-paced." -o low_male.wav
```

**Arguments:**
* `--text` / `-t` (Required): The text string to synthesize.
* `--prompt` / `-p` (Required): The acoustic style description (e.g. emotion, age, gender, speed, tone).
* `--output` / `-o` (Optional): Path to save the `.wav` output file. Default: `output.wav`.

### 2. `clone` (Voice Cloning)
Clones a target voice using a short reference audio file (3-15 seconds recommended).

```bash
# Basic usage
voicecli clone --text "Hello, I am speaking in your voice now." --audio-in reference.wav --output cloned_out.wav

# Short option usage
voicecli clone -t "This is zero-shot voice cloning in Rust." -a reference.wav -o cloned_out.wav
```

**Arguments:**
* `--text` / `-t` (Required): The text string to synthesize.
* `--audio-in` / `-a` (Required): Path to the reference audio file.
* `--output` / `-o` (Optional): Path to save the `.wav` output file. Default: `clone_output.wav`.

---

## Advanced Configurations

### Overriding the Model Names
By default, `voicecli` uses the production-grade **`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`** and **`Qwen/Qwen3-TTS-12Hz-1.7B-Base`** models. If you want to use smaller, faster models (e.g., for testing on machines with lower VRAM), set the `VOICECLI_MODEL_NAME_STYLE` or `VOICECLI_MODEL_NAME_CLONE` environment variables:

```bash
# On Linux/macOS
export VOICECLI_MODEL_NAME_STYLE="Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign"
voicecli style -t "Hello" -p "gender: Female" -o output.wav

# On Windows (PowerShell)
$env:VOICECLI_MODEL_NAME_STYLE="Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign"
.\voicecli.exe style -t "Hello" -p "gender: Female" -o output.wav
```

---

## Performance & Execution Notes

* **Model Loading Latency**: Because embedding Python in a short-lived CLI context means the model weights must be loaded from disk into VRAM/RAM on every execution, you will experience an initialization delay of **5 to 15 seconds** before speech generation starts. This is normal for single-run deep learning CLI applications.
* **Model Cache**: Hugging Face cache directory behaves normally. On the first run, the weights (~3.4 GB for the 1.7B model) will be downloaded to your local cache path under the project directory. Subsequent runs will use the cached local files.
* **VRAM Requirements**:
  * **1.7B Model**: Requires ~8GB VRAM (recommended).
  * **0.6B Model**: Requires ~4GB VRAM.

---

## Troubleshooting

### Windows: `STATUS_DLL_NOT_FOUND (0xc0000135)`

To prevent this runtime DLL loading error on Windows, the project includes a custom `build.rs` script that automatically detects and copies all required Python and runtime DLLs (such as `python311.dll`, `python3.dll`, `vcruntime140.dll`, and `vcruntime140_1.dll`) into the compilation output folder (`target/debug` or `target/release`).

* **Running via Cargo**: `cargo run` and `cargo build` will execute successfully out-of-the-box.
* **Distributing the Binary**: If you copy the `voicecli.exe` binary to another location on your system, you must copy the generated `.dll` and `.zip` files from your `target/release` folder to the same folder as the executable, or ensure the Python installation path is added to your Windows system environment `PATH`:
  * `d:\AGProjects\voicestudio\python-embed`

---

## License

GNU General Public License v3.0
