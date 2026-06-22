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

### How to Build

To compile a release binary and package it automatically into a redistributable standalone ZIP file, run the following script in your Command Prompt:
```cmd
.\create_portable.cmd
```
This batch script will:
1. Compile `voicecli` in release mode (`cargo build --release`).
2. Bootstrap a portable Python environment in `python-embed/` (installing PyTorch, torchaudio, transformers, qwen-tts, and `torch-directml` for GPU support).
3. Gather the compiled executable (`voicecli.exe`), required DLLs, standard library ZIPs, `settings.json`, and the `python-embed/` runtime folder into a staging directory.
4. Compress the folder into a single standalone archive: **`deploy\voicecli.zip`**.
5. Clean up the temporary staging directory.

---

## Usage

The CLI has two subcommands: `style` and `clone`.

### Global Flags
* `-s` / `--settings` (Optional): Path to a `settings.json` configuration file. If not specified, `voicecli` will search for `settings.json` in the current working directory, then in the directory of the `voicecli.exe` binary. If none is found, hardcoded defaults are used.

### 1. `style` (Voice Design / Style Prompts)
Generates speech based on a natural language description of the acoustic voice.

```bash
# Basic usage with default settings
voicecli style --text "Welcome to the future of speech synthesis." --prompt "gender: Female. pitch: High. speed: Normal." --output style_out.wav

# Using custom settings
voicecli -s my_settings.json style -t "This is a test of voice design." -p "gender: Male. pitch: Low." -o low_male.wav
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

# Short option usage with custom settings
voicecli -s my_settings.json clone -t "This is zero-shot voice cloning in Rust." -a reference.wav -o cloned_out.wav
```

**Arguments:**
* `--text` / `-t` (Required): The text string to synthesize.
* `--audio-in` / `-a` (Required): Path to the reference audio file.
* `--output` / `-o` (Optional): Path to save the `.wav` output file. Default: `clone_output.wav`.

---

## Configuration (`settings.json`)

`voicecli` uses a `settings.json` file for configuring models, hardware acceleration, and generation parameters. A default template is bundled in the distribution ZIP:

```json
{
  "model_name_style": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
  "model_name_clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "device": "cpu",
  "dtype": "float32",
  "huggingface_cache_dir": null,
  "do_sample": true,
  "temperature": 1.0,
  "top_p": 0.9,
  "top_k": 50,
  "max_new_tokens": 2048,
  "repetition_penalty": 1.1
}
```

### Settings Schema
* **`model_name_style`**: The style generation model. Defaults to `"Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"`.
* **`model_name_clone`**: The voice cloning model. Defaults to `"Qwen/Qwen3-TTS-12Hz-1.7B-Base"`.
* **`device`**: Target backend for inference. Options are:
  * `"cpu"`: Standard CPU inference (default).
  * `"dml"`: DirectML GPU acceleration (Windows).
  * `"vulkan"`: Vulkan GPU acceleration.
  * `"cuda"`: NVIDIA CUDA acceleration.
* **`dtype`**: PyTorch tensor type. Options are `"float32"`, `"float16"`, or `"bfloat16"`.
  * *Tip:* For DirectML and Vulkan, using `"float16"` is highly recommended to stay within GPU VRAM limits.
* **`huggingface_cache_dir`**: Custom directory path to download and store Hugging Face model weights. If `null`, defaults to a `.cache/huggingface` folder next to the `voicecli.exe` binary.
* **`do_sample`**: Enable token sampling during text generation (boolean).
* **`temperature`**: Sampling temperature (lower = more deterministic, higher = more random/creative).
* **`top_p`**: Cumulative probability threshold for nucleus sampling.
* **`top_k`**: The number of highest probability vocabulary tokens to keep for filtering.
* **`max_new_tokens`**: Maximum length of generated speech tokens.
* **`repetition_penalty`**: Penalty factor to avoid repetitive phrasing.

---

## Hardware Acceleration & Fail-safe Behavior

* **DirectML Support (`"device": "dml"`)**: DirectML is used for hardware acceleration on Windows (across AMD, Intel, and NVIDIA GPUs). If `torch-directml` is not installed or initialization/transfer fails (e.g. out of memory), the execution automatically and gracefully falls back to CPU.
* **Vulkan Support (`"device": "vulkan"`)**: Supports Vulkan hardware acceleration. Falls back to CPU if Vulkan is unavailable on the system or transfer fails.
* **CUDA Support (`"device": "cuda"`)**: Native CUDA support. Falls back to CPU if NVIDIA drivers or CUDA libraries are missing or transfer fails.
* **Memory Optimization**: Loading the 1.7B parameter model in `"float32"` format on a GPU requires roughly 7-8 GB of VRAM. If your GPU has less VRAM or raises out-of-memory errors, configure `"dtype": "float16"` in your settings to reduce the VRAM requirement to ~3.5 GB.

---

## Troubleshooting

### Windows: `STATUS_DLL_NOT_FOUND (0xc0000135)`

To prevent this runtime DLL loading error on Windows, the project includes a custom `build.rs` script that automatically detects and copies all required Python and runtime DLLs (such as `python311.dll`, `python3.dll`, `vcruntime140.dll`, and `vcruntime140_1.dll`) into the compilation output folder (`target/debug` or `target/release`).

* **Running via Cargo**: `cargo run` and `cargo build` will execute successfully out-of-the-box.
* **Distributing the Binary**: The portable ZIP contains all necessary DLLs in the root folder alongside the executable. If you move `voicecli.exe` manually, ensure all DLL files and `python311.zip` are copied alongside it.

---

## License

GNU General Public License v3.0
