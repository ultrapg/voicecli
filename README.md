# voicecli

> **Ultra-Fast, Native Single-Binary Qwen3-TTS CLI (Rust + C/C++ GGML Engine)**  
> High-performance speech synthesis and zero-shot voice cloning with **zero Python dependencies**, full FP16/BF16 neural audio fidelity, in-process SIMD acceleration, and an interactive real-time progress terminal UI.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20x86__64-orange.svg)](#system-requirements)
[![Engine](https://img.shields.io/badge/Engine-GGML%20Native%20C%2B%2B-green.svg)](#architecture-overview)
[![Python Free](https://img.shields.io/badge/Python-0%25%20(Completely%20Removed)-brightgreen.svg)](#the-re-engineering-python--pytorch-vs-native-ggml)

---

## Table of Contents

- [Overview](#overview)
- [The Re-Engineering: Python / PyTorch vs. Native GGML](#the-re-engineering-python--pytorch-vs-native-ggml)
- [Architecture Overview](#architecture-overview)
- [Key Features](#key-features)
- [Quick Start: Standalone Portable Bundle](#quick-start-standalone-portable-bundle)
- [CLI Usage & Command Reference](#cli-usage--command-reference)
  - [1. Voice Design / Acoustic Style (`voicecli style`)](#1-voice-design--acoustic-style-voicecli-style)
  - [2. Zero-Shot Voice Cloning (`voicecli clone`)](#2-zero-shot-voice-cloning-voicecli-clone)
  - [3. Auto-Chunk Mode for Long Texts (`--chunk`)](#3-auto-chunk-mode---chunk)
- [Configuration & Settings](#configuration--settings)
  - [Configuration File (`settings.json`)](#configuration-file-settingsjson)
  - [Environment Variables](#environment-variables)
- [Supported Models (GGUF BF16)](#supported-models-gguf-bf16)
- [Building from Source](#building-from-source)
  - [Prerequisites](#prerequisites)
  - [Compiling the Unified Binary](#compiling-the-unified-binary)
  - [Generating the Portable `.tar.gz` Package](#generating-the-portable-targz-package)
- [Technical Specifications & Audio Pipeline](#technical-specifications--audio-pipeline)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [License & Acknowledgments](#license--acknowledgments)

---

## Overview

`voicecli` is a high-performance command-line interface for **Qwen3-TTS**, Alibaba Cloud's state-of-the-art text-to-speech and voice-cloning model family.

Earlier implementations of `voicecli` relied on an embedded Python runtime, PyTorch, Hugging Face `transformers`, and CUDA/C++ shared libraries wrapped through PyO3. While functional, that architecture required multi-gigabyte environments, suffered from high cold-start latency (importing PyTorch and CUDA runtimes took 10–20 seconds before generating a single audio sample), consumed massive RAM/VRAM, and was fragile to distribute.

**This release completely replaces `https://github.com/ultrapg/voicecli` with a 100% native single-binary engine.**  
The entire PyTorch and Python stack has been eliminated. The inference core is built directly on native C/C++ utilizing **GGML** tensor primitives with hardware-accelerated SIMD instructions (AVX2, AVX512, FMA, NEON). The result is a single, self-contained executable under 4 MB (1.6 MB compressed) that can run anywhere on Linux x86_64 without installing Python or deep learning frameworks.

---

## The Re-Engineering: Python / PyTorch vs. Native GGML

| Metric / Feature | Original Python / PyTorch Implementation | New Native Single-Binary Architecture |
| :--- | :--- | :--- |
| **Runtime Dependencies** | Python 3.11, PyTorch, Torchaudio, Transformers, PyO3, LibCUDA | **None** (Statically linked single binary) |
| **Distribution Package Size** | Multi-GB folder or complex virtualenv bundle | **~1.6 MB** (`voicecli-linux-x86_64.tar.gz`) |
| **Binary Executable Size** | Fragile wrapper DLLs + hundreds of shared `.so` files | **~3.9 MB single unified binary** |
| **Cold-Start Time** | 10 – 20+ seconds (Python interpreter + PyTorch init) | **Sub-second startup** + direct model memory mapping |
| **Hardware Acceleration** | Requires CUDA GPU or heavy PyTorch CPU backend | **Native CPU SIMD (AVX2, AVX-512, FMA) via GGML** |
| **Audio Quality & Weights** | BF16 / FP16 PyTorch tensors | **Full-precision BF16 GGUF weights** (exact parity) |
| **Audio Writing Pipeline** | `scipy.io.wavfile` / `soundfile` / `torchaudio` | **In-process native 24 kHz 16-bit PCM WAV writer** |
| **Interactive Terminal UX** | Plain stdout logs or PyTorch debug dumps | **Braille spinner, animated progress bar & RTF speed stats** |
| **Installation Friction** | Complex pip / conda / embed setup | **Extract `.tar.gz` and run** (zero configuration) |

---

## Architecture Overview

`voicecli` combines high-level CLI ergonomics with low-level SIMD inference in a unified process space:

```
+--------------------------------------------------------------------------+
|                            voicecli Executable                           |
|                                                                          |
|  +--------------------------------------------------------------------+  |
|  |                 Rust Frontend Layer (src/main.rs)                  |  |
|  |   - CLI argument parsing via clap                                  |  |
|  |   - Subcommands: `style` and `clone`                               |  |
|  |   - Portable path resolution & environment configuration           |  |
|  +-----------------------------------+--------------------------------+  |
|                                      |                                   |
|                                      v (Zero-Overhead Native C FFI)      |
|  +-----------------------------------+--------------------------------+  |
|  |                  Native C Engine Bridge (src/model.c)              |  |
|  |   - Configuration parser (`settings.json` + ENV overrides)         |  |
|  |   - Auto-setup: Resumable Hugging Face curl downloader             |  |
|  |   - Interactive terminal UI: Braille spinner & dynamic progress bar|  |
|  |   - Real-time factor (RTF) timing calculations                     |  |
|  |   - In-process 16-bit 24kHz RIFF/WAVE file serializer              |  |
|  +-----------------------------------+--------------------------------+  |
|                                      |                                   |
|                                      v (Direct C++ In-Process Call)      |
|  +-----------------------------------+--------------------------------+  |
|  |             qwen3-tts.cpp Inference Engine & GGML Core            |  |
|  |   - 151,676-token BPE Text Tokenizer                               |  |
|  |   - 12Hz Speech Tokenizer & Vocoder (Decoder)                      |  |
|  |   - 1.7B / 0.6B Autoregressive Transformer with Code Prediction    |  |
|  |   - Speaker Encoder & Reference Voice Tokenizer (ICL)              |  |
|  |   - SIMD Vectorized Tensor Math (AVX2 / AVX512 / FMA / CPU Backend)|  |
|  +--------------------------------------------------------------------+  |
+--------------------------------------------------------------------------+
                                       |
                                       v
         +-----------------------------------------------------------+
         |               Direct GGUF Weight Evaluation               |
         |  - qwen-talker-1.7b-voicedesign-BF16.gguf (Voice Design)  |
         |  - qwen-talker-1.7b-base-BF16.gguf        (Voice Clone)   |
         |  - qwen-tokenizer-12hz-BF16.gguf          (12Hz Vocoder)  |
         +-----------------------------------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |  High-Fidelity 24 kHz WAV Output  |
                     +-----------------------------------+
```

---

## Key Features

- **Single Unified Binary**: The complete GGML tensor runtime and Qwen3-TTS C++ engine are statically compiled directly into `voicecli`. No secondary helper executables, no external runtimes, no DLL hell.
- **Natural Language Voice Design (`style`)**: Control speaker identity, age, pitch, speaking pace, mood, and emotion using plain text descriptions (e.g. `"gender: Female. pitch: High. speed: Fast-paced. emotion: Cheerful."`).
- **Zero-Shot Voice Cloning (`clone`)**: Clone any target speaker's voice from a short reference audio clip (3 to 15 seconds) without fine-tuning.
- **Interactive Real-Time Progress Bar**:
  - Live animated braille spinner (`⠋ ⠙ ⠹ ⠸ ⠼ ⠴ ⠦ ⠧ ⠇ ⠏`).
  - Smooth terminal progress bar with frame counters and estimated audio duration.
  - Automatically detects whether standard error is a TTY (`isatty`), keeping CI/CD and piped logs clean while providing rich interactive output in terminals.
- **Auto-Setup on First Run**: If model weights are not present locally, `voicecli` automatically fetches the official BF16 GGUF weights directly from Hugging Face with a resumable download progress bar.
- **Auto-Chunk Mode for Long Texts (`--chunk`)**:
  - Automatically segment long text files (`--file <PATH.txt>`) or inline text into natural sentence-level chunks (~250–350 characters).
  - Preserves titles, abbreviations (`Dr.`, `Mr.`, `e.g.`, `etc.`), and paragraph structure.
  - Keeps the 3.8 GB model resident in memory to synthesize all chunks consecutively without model reload delays.
  - Inserts 250ms natural breathing pauses between chunks and stitches everything into a single seamless output WAV file.
- **Zero Audio Processing Dependencies**: Implements a dedicated in-process 16-bit PCM 24 kHz WAV serializer, outputting broadcast-quality audio files directly.
- **Flexible Configuration**: Fine-tune model selections and paths via `settings.json` or override them on the fly using environment variables.

---

## Quick Start: Standalone Portable Bundle

Pre-built standalone releases require no installation, no compilation, and no package managers.

### 1. Extract the Archive
```bash
# Extract the portable bundle (approx. 1.6 MB)
tar -xzf deploy/voicecli-linux-x86_64.tar.gz
cd voicecli
```

The extracted directory contains only:
```
voicecli/
├── voicecli        # Statically linked single executable (~3.9 MB)
└── settings.json   # Configuration file
```

### 2. Run Voice Design (Style Synthesis)
```bash
./voicecli style \
  --text "Hello! This speech is synthesized natively without Python or PyTorch." \
  --prompt "gender: Female. pitch: Medium. speed: Normal." \
  --output speech.wav
```
*(On your first execution, `voicecli` will automatically download the necessary GGUF model files into the local directory and then synthesize the audio).*

### 3. Run Zero-Shot Voice Cloning
```bash
./voicecli clone \
  --text "Now I am speaking with the acoustic characteristics of your reference audio." \
  --audio-in path/to/reference.wav \
  --output cloned_voice.wav
```

### 4. Synthesize Long Text Files with Auto-Chunking
```bash
./voicecli style \
  --file article.txt \
  --prompt "gender: Male. pitch: Deep. speed: Normal. tone: Professional." \
  --chunk \
  --output full_recording.wav
```

---

## CLI Usage & Command Reference

```
Usage: voicecli <COMMAND>

Commands:
  style  Generate speech using Voice Design style descriptions
  clone  Clone a voice from a short reference audio file
  help   Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

---

### 1. Voice Design / Acoustic Style (`voicecli style`)

Generates speech from text guided by natural language prompts describing vocal characteristics, pitch, gender, pacing, or emotional coloring.

```bash
voicecli style [OPTIONS] --prompt <PROMPT> (--text <TEXT> | --file <PATH.txt>)
```

#### Arguments & Flags

| Flag | Short | Default | Description |
| :--- | :---: | :---: | :--- |
| `--text` | `-t` | *None* | The textual content to convert into speech (use either `--text` or `--file`). |
| `--file` | `-f` | *None* | Path to a `.txt` file containing the text to convert into speech. |
| `--prompt` | `-p` | *(Required)* | Acoustic style instructions (e.g. gender, pitch, speed, mood). |
| `--output` | `-o` | `output.wav` | Path where the output 24 kHz `.wav` file will be saved. |
| `--chunk` | | `false` | **Auto-Chunk Mode**: Automatically split long texts into natural sentence chunks, synthesize in-memory, and stitch into one seamless audio file. |

#### Examples

```bash
# Basic voice synthesis
./voicecli style \
  -t "Good morning! You have three meetings scheduled for today." \
  -p "gender: Female. pitch: High. speed: Normal. tone: Cheerful." \
  -o morning.wav

# Synthesize long text / article from a .txt file with Auto-Chunking
./voicecli style \
  -f article.txt \
  -p "gender: Male. pitch: Deep. speed: Slow. tone: Documentary narration." \
  --chunk \
  -o audiobook_chapter.wav
```

---

### 2. Zero-Shot Voice Cloning (`voicecli clone`)

Clones an individual speaker's voice using a short reference audio file (3 to 15 seconds recommended).

```bash
voicecli clone [OPTIONS] --audio-in <AUDIO_IN> (--text <TEXT> | --file <PATH.txt>)
```

#### Arguments & Flags

| Flag | Short | Default | Description |
| :--- | :---: | :---: | :--- |
| `--text` | `-t` | *None* | The textual content to synthesize (use either `--text` or `--file`). |
| `--file` | `-f` | *None* | Path to a `.txt` file containing the text to synthesize. |
| `--audio-in` | `-a` | *(Required)* | Path to the reference `.wav` audio clip (3–15 seconds). |
| `--output` | `-o` | `clone_output.wav` | Path where the cloned output `.wav` file will be saved. |
| `--chunk` | | `false` | **Auto-Chunk Mode**: Automatically split long texts into natural sentence chunks, synthesize in-memory, and stitch into one seamless audio file. |

#### Examples

```bash
# Clone a voice from reference sample
./voicecli clone \
  -t "This sentence is spoken entirely in the vocal timbre of the reference sample." \
  -a my_voice_sample.wav \
  -o cloned_result.wav

# Clone a voice for a full document from a .txt file with Auto-Chunking
./voicecli clone \
  -f my_notes.txt \
  -a my_voice_sample.wav \
  --chunk \
  -o cloned_full_speech.wav
```

---

### 3. Auto-Chunk Mode (`--chunk`)

When synthesizing long text inputs (such as articles, essays, or audiobook chapters):
- Pass the `--chunk` flag along with `--file <PATH.txt>` or `--text <TEXT>`.
- `voicecli` intelligently segments the text at sentence and paragraph boundaries (~250–350 characters per chunk) while preserving abbreviations (`Dr.`, `Mr.`, `e.g.`).
- The 3.8 GB model weights are loaded **once** into memory, synthesizing all chunks consecutively at maximum SIMD speed with zero reload latency.
- Subtle 250ms silence padding is automatically inserted between chunks for a natural human speaking cadence, stitching everything into a single, unified output `.wav` file.

---

## Configuration & Settings

`voicecli` uses a hierarchical configuration system. Values are resolved in the following priority order:
1. **Environment Variables** (Highest precedence)
2. **`settings.json`** (Located in the working directory or beside the executable)
3. **Internal Built-in Defaults** (Lowest precedence)

### Configuration File (`settings.json`)

```json
{
  "model_name_style": "qwen-talker-1.7b-voicedesign-BF16.gguf",
  "model_name_clone": "qwen-talker-1.7b-base-BF16.gguf",
  "tokenizer_model": "qwen-tokenizer-12hz-BF16.gguf"
}
```

- **`model_name_style`**: The GGUF model file used for style-based synthesis (`voicecli style`).
- **`model_name_clone`**: The GGUF model file used for zero-shot voice cloning (`voicecli clone`).
- **`tokenizer_model`**: The 12Hz neural audio codec vocoder model.

### Environment Variables

You can override any setting without editing files by exporting environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `VOICECLI_MODEL_DIR` | Custom directory path where `.gguf` model files are stored. | Searches `models/`, executable directory, or current folder |
| `VOICECLI_MODEL_NAME_STYLE` | Filename of the style synthesis model to load. | `qwen-talker-1.7b-voicedesign-BF16.gguf` |
| `VOICECLI_MODEL_NAME_CLONE` | Filename of the voice cloning model to load. | `qwen-talker-1.7b-base-BF16.gguf` |
| `VOICECLI_TOKENIZER_MODEL` | Filename of the audio tokenizer / vocoder model. | `qwen-tokenizer-12hz-BF16.gguf` |
| `VOICECLI_BASE_DIR` | Manually overrides the application base directory. | Directory containing the `voicecli` binary |

#### Example: Using a Shared Central Model Directory
```bash
export VOICECLI_MODEL_DIR="/opt/models/qwen3-tts"
./voicecli style -t "Using shared model cache." -p "clear voice" -o shared.wav
```

---

## Supported Models (GGUF BF16)

`voicecli` operates with full bfloat16 (`BF16`) precision weights to preserve the original audio fidelity of the PyTorch reference models:

| Model Filename | Architecture | Role / Purpose | File Size |
| :--- | :---: | :--- | :---: |
| `qwen-tokenizer-12hz-BF16.gguf` | 12Hz Codec | Neural vocoder & reference audio encoder | ~358 MB |
| `qwen-talker-1.7b-voicedesign-BF16.gguf` | 1.7B Transformer | Style-conditioned speech generation (`style`) | ~3.8 GB |
| `qwen-talker-1.7b-base-BF16.gguf` | 1.7B Transformer | Zero-shot voice cloning (`clone`) | ~3.8 GB |
| `qwen-talker-0.6b-base-BF16.gguf` | 0.6B Transformer | Ultra-fast cloning for low-memory environments | ~1.8 GB |

Weights are mirrored and maintained at [`Serveurperso/Qwen3-TTS-GGUF`](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF) on Hugging Face.

---

## Building from Source

If you want to build `voicecli` directly from source code:

### Prerequisites

Ensure the following native build tools are installed on your Linux system:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake curl git

# Install Rust toolchain (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Compiling the Unified Binary

1. Clone the repository recursively to fetch all submodules and engine sources:
   ```bash
   git clone https://github.com/ultrapg/voicecli.git
   cd voicecli
   ```

2. Compile in release mode with Cargo:
   ```bash
   cargo build --release
   ```

The custom `build.rs` script will automatically compile `src/model.c`, link against the static GGML engine libraries, and produce the single unified executable at `target/release/voicecli`.

### Generating the Portable `.tar.gz` Package

Run the bundle creation script:
```bash
./create_portable.sh
```

This script will:
1. Compile the release binary (`cargo build --release`).
2. Stage the `voicecli` binary alongside `settings.json`.
3. Package everything into a clean archive at `deploy/voicecli-linux-x86_64.tar.gz`.

---

## Technical Specifications & Audio Pipeline

- **Audio Sampling Rate**: 24,000 Hz (24 kHz)
- **Audio Encoding**: 16-bit Linear PCM (Single Channel / Mono)
- **Audio Format**: RIFF / WAVE (.wav)
- **Speech Tokenizer Frequency**: 12 Hz (12 frames of discrete acoustic codes per second of generated audio)
- **Codebooks**: 16 codebooks per frame
- **Text Tokenizer**: 151,676-token BPE tokenizer
- **Inference Precision**: Full BF16 / FP16 SIMD execution
- **Thread Scheduling**: Automatic multi-core thread scaling matching host CPU topology

---

## Troubleshooting & FAQ

### Q: Does `voicecli` require a GPU or NVIDIA drivers?
**No.** `voicecli` runs natively on CPU using optimized SIMD instructions (AVX2 / AVX512 / FMA). You do not need CUDA, ROCm, or dedicated GPU hardware to generate speech.

### Q: Where are downloaded models stored?
By default, `voicecli` places downloaded models in the `models/` directory or directly alongside the executable. You can store models in any central location by setting `export VOICECLI_MODEL_DIR=/path/to/my/models`.

### Q: Why is my terminal progress bar showing multiple lines in CI/CD?
In non-interactive environments where `stderr` is not an interactive terminal (e.g. piped to `grep` or running in a CI runner), `voicecli` detects that `isatty(fileno(stderr))` is false and automatically switches from ANSI line-clearing updates to periodic milestone logging every 20 frames.

### Q: Is there a maximum text length limit?
In single-pass mode (without `--chunk`), generation is budgeted up to 4,096 audio frames (~5.7 minutes of speech, or ~850 words). If your input exceeds this, generation gracefully finalizes at the 5.7-minute mark.  
**To synthesize arbitrarily long texts (such as full articles, essays, or audiobook chapters), pass the `--chunk` flag.** Auto-Chunk Mode will split the text into natural sentence-level chunks, process them consecutively with the model resident in memory, and stitch the entire recording into a single, seamless WAV file.

### Q: How do I synthesize an audiobook or article from a `.txt` file?
Place your text in a `.txt` file and run:
```bash
./voicecli style --file chapter1.txt --prompt "gender: Male. pitch: Deep. speed: Normal. tone: Expressive narration." --chunk --output chapter1.wav
```

### Q: Can I interrupt generation safely?
Yes. Sending `Ctrl+C` cleanly terminates the process immediately without leaving dangling background threads or child processes.

---

## License & Acknowledgments

- **License**: Licensed under the [GNU General Public License v3.0](LICENSE).
- **Qwen3-TTS**: Developed and trained by Alibaba Cloud / Qwen Team.
- **GGML**: Tensor library developed by Georgi Gerganov and the GGML community.
- **Qwen3-TTS C++ Core**: Native C++ port based on `qwen3-tts.cpp`.
