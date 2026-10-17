# voicecli

> **Ultra-Fast, Native Single-Binary Qwen3-TTS CLI (Rust + C/C++ GGML Engine)**  
> High-performance speech synthesis, Voice Design (`style`), Zero-Shot & Full-ICL Voice Cloning (`clone`), Built-In CustomVoice Speakers (`speak`), and Cached Voice Profiles (`profile`) with **zero Python dependencies**, full BF16/Quantized GGUF support, in-process SIMD acceleration, and an interactive real-time progress terminal UI.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20x86__64-orange.svg)](#building-from-source)
[![Engine](https://img.shields.io/badge/Engine-GGML%20Native%20C%2B%2B-green.svg)](#architecture-overview)
[![Python Free](https://img.shields.io/badge/Python-0%25%20(Completely%20Removed)-brightgreen.svg)](#the-re-engineering-python--pytorch-vs-native-ggml)

---

## What's New in v0.3.0

- **Full In-Context Learning (ICL) Voice Cloning (`--ref-text` / `--ref-file`)**: Supply the transcript of your reference audio during `voicecli clone` or `voicecli profile` to extract both the 2048D speaker embedding AND the 12Hz reference acoustic speech codes for dramatically higher voice similarity and prosodic fidelity.
- **Reusable Voice Profiles (`voicecli profile`, `--voice-profile`, `--save-profile`)**: Pre-compute and cache speaker embeddings or Full-ICL voice profiles to `.json` / `.bin` files once, then clone instantaneously without re-encoding reference audio on subsequent runs.
- **Built-In CustomVoice Speakers (`voicecli speak`)**: Use pre-trained named speakers (e.g. `Vivian`, `Ryan`, `Serena`, `Ethan`, etc.) with optional natural-language emotion/style steering (`-p "Speak cheerfully"`) and `--list-speakers`.
- **Quantized & Lightweight Model Presets (`--model` / `-m`, `voicecli models`)**: Switch effortlessly between full `bf16` (default), `q8` (~50% less RAM, faster CPU inference), `q4` (~70% less RAM), and `0.6b` / `fast` / `tiny` lightweight models.
- **Multi-Language Control (`--lang` / `-l`)**: Explicit language selection (`auto`, `en`, `de`, `fr`, `es`, `it`, `pt`, `zh`, `ja`, `ko`, `ru`).
- **Unix Stdin/Stdout Piping & Streaming**:
  - Pipe text directly from `stdin` (`echo "Hello world" | voicecli style -p "clear voice" -o out.wav` or `--stdin`).
  - Stream generated WAV audio directly to `stdout` with `-o -` (`voicecli style -t "Hello" -p "warm" -o - | mpv -`), with all progress/diagnostic output cleanly routed to `stderr`.
  - Low-latency streaming vocoder mode (`--stream`, `--stream-chunk-sec`).
- **Automatic Format Conversion & Audio Post-Processing**:
  - Reference audio in `.mp3`, `.flac`, `.ogg`, or `.m4a` format is automatically converted to 24 kHz mono WAV when `ffmpeg` is available.
  - Peak normalization (`--normalize`) scales output audio cleanly to `-0.5 dBFS`.
  - Auto-playback (`--play`) plays generated audio immediately via `pw-play`, `paplay`, `aplay`, or `ffplay`.
- **Major Core Optimizations**:
  - **O(1) Prefix-Sum SOLA Time-Stretching + Hann Crossfade**: Precomputes prefix-sums of squares and a raised-cosine (Hann) crossfade table with coarse-to-fine lag search, accelerating SOLA time-stretching by ~15x while eliminating linear-crossfade phase artifacts.
  - **64 KB Buffered WAV Serializer with `lrintf()`**: Replaces per-sample `fwrite` calls (millions of syscalls) with 64 KB batch writes and unbiased float-to-PCM16 rounding.
  - **Smart Auto-Chunking & Markdown Cleanup**: Automatically chunks long inputs (>800 chars) unless `--no-chunk` is specified, protects decimal numbers (`3.14`) and ellipses (`...`), and supports `--clean-markdown`.
  - **Atomic Model Downloads**: Downloads to `.part` temporary files with automatic retry and size verification before atomic rename, preventing corrupted partial downloads.
  - **Full Sampling & Thread Controls**: CLI flags (`--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`, `--max-tokens`, `--threads`) and `settings.json` parameters are now fully wired into the native engine.

---

## Table of Contents

- [Overview](#overview)
- [The Re-Engineering: Python / PyTorch vs. Native GGML](#the-re-engineering-python--pytorch-vs-native-ggml)
- [Architecture Overview](#architecture-overview)
- [Quick Start: Standalone Portable Bundle](#quick-start-standalone-portable-bundle)
- [CLI Usage & Command Reference](#cli-usage--command-reference)
  - [1. Voice Design / Acoustic Style (`voicecli style`)](#1-voice-design--acoustic-style-voicecli-style)
  - [2. Zero-Shot & Full-ICL Voice Cloning (`voicecli clone`)](#2-zero-shot--full-icl-voice-cloning-voicecli-clone)
  - [3. Built-In CustomVoice Speakers (`voicecli speak`)](#3-built-in-customvoice-speakers-voicecli-speak)
  - [4. Reusable Voice Profiles (`voicecli profile`)](#4-reusable-voice-profiles-voicecli-profile)
  - [5. Model Presets & System Info (`voicecli models`)](#5-model-presets--system-info-voicecli-models)
  - [6. Auto-Chunk Mode for Long Texts (`--chunk`)](#6-auto-chunk-mode---chunk)
  - [7. Pitch-Preserving Speed Adjustment (`-s, --speed`)](#7-pitch-preserving-speed-adjustment--s---speed)
- [Configuration & Settings](#configuration--settings)
- [Supported Models & Quantization Presets](#supported-models--quantization-presets)
- [Building from Source](#building-from-source)
- [License & Acknowledgments](#license--acknowledgments)

---

## Overview

`voicecli` is a high-performance command-line interface for **Qwen3-TTS**, Alibaba Cloud's state-of-the-art text-to-speech and voice-cloning model family.

The entire PyTorch and Python stack has been eliminated. The inference core is built directly on native C/C++ utilizing **GGML** tensor primitives with hardware-accelerated SIMD instructions (AVX2, AVX512, FMA, NEON). The result is a single, self-contained executable (~3.8 MB) that runs anywhere on Linux x86_64 without installing Python or deep learning frameworks.

---

## The Re-Engineering: Python / PyTorch vs. Native GGML

| Metric / Feature | Original Python / PyTorch Implementation | Native Single-Binary Architecture (`v0.3.0`) |
| :--- | :--- | :--- |
| **Runtime Dependencies** | Python 3.11, PyTorch, Torchaudio, Transformers, PyO3, LibCUDA | **None** (Statically linked single binary) |
| **Distribution Package Size** | Multi-GB folder or complex virtualenv bundle | **~1.6 MB** (`voicecli-linux-x86_64.tar.gz`) |
| **Binary Executable Size** | Fragile wrapper DLLs + hundreds of shared `.so` files | **~3.8 MB single unified binary** |
| **Cold-Start Time** | 10 – 20+ seconds (Python interpreter + PyTorch init) | **Sub-second startup** + direct model memory mapping |
| **Hardware Acceleration** | Requires CUDA GPU or heavy PyTorch CPU backend | **Native CPU SIMD (AVX2, AVX-512, FMA) via GGML** |
| **Model Support** | Single precision only | **BF16, Q8_0, Q4_K_M (1.7B & 0.6B presets)** |
| **Voice Cloning Modes** | Basic speaker embedding | **Speaker Embedding + Full In-Context Learning (ICL) + Cached Profiles** |
| **Audio Writing Pipeline** | `scipy.io.wavfile` / `soundfile` / `torchaudio` | **64 KB buffered native 24 kHz 16-bit PCM WAV writer + stdout piping** |

---

## Architecture Overview

```
+--------------------------------------------------------------------------+
|                            voicecli Executable                           |
|                                                                          |
|  +--------------------------------------------------------------------+  |
|  |                 Rust Frontend Layer (src/main.rs)                  |  |
|  |   - CLI argument parsing via clap (`style`, `clone`, `speak`,      |  |
|  |     `profile`, `models`)                                           |  |
|  |   - Linguistic sentence & clause chunker + Markdown cleaner        |  |
|  |   - Stdin piping & portable path resolution                        |  |
|  +-----------------------------------+--------------------------------+  |
|                                      |                                   |
|                                      v (Zero-Overhead Native C FFI)      |
|  +-----------------------------------+--------------------------------+  |
|  |                  Native C Engine Bridge (src/model.c)              |  |
|  |   - Configuration parser (`settings.json` + ENV + CLI overrides)   |  |
|  |   - Atomic auto-downloader (.part -> .gguf) from Hugging Face      |  |
|  |   - O(1) prefix-sum SOLA time-stretcher with Hann crossfading      |  |
|  |   - Peak audio normalizer (-0.5 dBFS) & 64 KB buffered WAV writer  |  |
|  +-----------------------------------+--------------------------------+  |
|                                      |                                   |
|                                      v (Direct C++ In-Process Call)      |
|  +-----------------------------------+--------------------------------+  |
|  |             qwen3-tts.cpp Inference Engine & GGML Core            |  |
|  |   - 151,676-token BPE Text Tokenizer                               |  |
|  |   - 12Hz Speech Tokenizer & Vocoder (Batch & Streaming)            |  |
|  |   - 1.7B / 0.6B Autoregressive Transformer with Code Prediction    |  |
|  |   - ECAPA-TDNN Speaker Encoder & Full-ICL Speech Code Extractor    |  |
|  +--------------------------------------------------------------------+  |
+--------------------------------------------------------------------------+
```

---

## Quick Start: Standalone Portable Bundle

### 1. Extract the Archive
```bash
tar -xzf deploy/voicecli-linux-x86_64.tar.gz
cd voicecli
```

### 2. Run Voice Design (`style`)
```bash
./voicecli style \
  --text "Hello! This speech is synthesized natively without Python or PyTorch." \
  --prompt "gender: Female. pitch: Medium. speed: Normal." \
  --output speech.wav
```

### 3. Run Zero-Shot or Full-ICL Voice Cloning (`clone`)
```bash
# Fast Zero-Shot Voice Cloning (Speaker Embedding)
./voicecli clone \
  --text "Now I am speaking with the acoustic characteristics of your reference audio." \
  --audio-in reference.wav \
  --output cloned_voice.wav

# High-Fidelity Full-ICL Voice Cloning (with reference transcript + saving reusable profile)
./voicecli clone \
  --text "Full in-context learning captures both my vocal timbre and speaking prosody." \
  --audio-in reference.wav \
  --ref-text "This is the exact sentence spoken in the reference audio clip." \
  --save-profile my_voice.json \
  --output cloned_icl.wav

# Instant Cloning from Saved Voice Profile (skips audio encoding!)
./voicecli clone \
  --text "Instant synthesis using the cached voice profile." \
  --voice-profile my_voice.json \
  --output instant_clone.wav
```

### 4. Pipe from `stdin` or to `stdout`
```bash
# Read text from stdin and play immediately
echo "Unix philosophy in action." | ./voicecli style -p "gender: Male. tone: Calm." --play

# Stream generated WAV audio directly to stdout into an audio player or ffmpeg
./voicecli style -t "Streaming WAV bytes to stdout" -p "clear voice" -o - | ffplay -autoexit -nodisp -
```

---

## CLI Usage & Command Reference

```
Usage: voicecli <COMMAND>

Commands:
  style    Generate speech using natural-language Voice Design descriptions
  clone    Clone a voice from a 3-10s reference audio file or cached voice profile
  speak    Generate speech using built-in CustomVoice named speakers
  profile  Pre-extract and save a reusable voice profile (.json or .bin) from reference audio
  models   Show system info, model presets, or pre-download GGUF models
  help     Print this message or the help of the given subcommand(s)
```

### Shared Synthesis Options (`style`, `clone`, `speak`)

| Flag | Short | Default | Description |
| :--- | :---: | :---: | :--- |
| `--text <TEXT>` | `-t` | *None* | Input text to synthesize (use `-` for stdin). |
| `--file <PATH>` | `-f` | *None* | Path to a `.txt` or `.md` file to synthesize (use `-` for stdin). |
| `--stdin` | | `false` | Read input text explicitly from standard input (`stdin`). |
| `--output <PATH>` | `-o` | `*.wav` | Output `.wav` file path, or `-` to write raw WAV bytes to `stdout`. |
| `--speed <FLOAT>` | `-s` | `1.0` | Pitch-preserving SOLA speech speed multiplier (`0.25` to `3.0`). |
| `--lang <CODE>` | `-l` | `auto` | Language code: `auto`, `en`, `de`, `fr`, `es`, `it`, `pt`, `zh`, `ja`, `ko`, `ru`. |
| `--model <PRESET>` | `-m` | `bf16` | Model preset or `.gguf` filename: `bf16`, `q8`, `q4`, `0.6b`, `fast`, `tiny`. |
| `--chunk` | | *Auto >800c* | Split text into natural sentence chunks and stitch seamlessly. |
| `--no-chunk` | | `false` | Disable automatic chunking even for long texts (>800 chars). |
| `--chunk-size <N>` | | `600` | Target maximum characters per chunk when chunking is active. |
| `--threads <N>` | `-j` | `0` (Auto) | Number of CPU threads for GGML tensor inference. |
| `--temperature <F>` | | `0.9` | Sampling temperature (`0.0` = deterministic greedy decoding). |
| `--top-p <F>` | | `1.0` | Nucleus sampling probability threshold (`0.0` – `1.0`). |
| `--top-k <N>` | | `50` | Top-k sampling candidate count (`0` = disabled). |
| `--repetition-penalty <F>`| | `1.05` | Penalty to prevent repetitive audio token loops. |
| `--max-tokens <N>` | | `4096` | Maximum audio frames per generation pass. |
| `--stream` | | `false` | Use low-latency streaming vocoder synthesis. |
| `--normalize` | | `false` | Peak-normalize output audio to `-0.5 dBFS`. |
| `--play` | | `false` | Automatically play the generated audio file upon completion. |
| `--clean-markdown` | | `false` | Strip Markdown headers, emphasis, and code blocks before speaking. |

---

### 1. Voice Design / Acoustic Style (`voicecli style`)

```bash
voicecli style [OPTIONS] --prompt <PROMPT> (--text <TEXT> | --file <PATH> | --stdin)
```

### 2. Zero-Shot & Full-ICL Voice Cloning (`voicecli clone`)

```bash
voicecli clone [OPTIONS] (--audio-in <AUDIO_FILE> | --voice-profile <PROFILE>) (--text <TEXT> | --file <PATH>)
```

Additional `clone` flags:
- `--audio-in` (`-a`): Reference audio file (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a` — non-WAV files are auto-converted via `ffmpeg`).
- `--ref-text <TEXT>` / `--ref-file <PATH>`: Transcript of the reference audio. Enables **Full In-Context Learning (ICL)** mode (`qwen3_tts_extract_icl_prompt`), conditioning the transformer on both the speaker embedding and the reference speech codec tokens.
- `--save-profile <PATH>`: Save the extracted voice profile (`.json` for ICL, `.bin` for speaker embedding) for instant reuse.
- `--voice-profile <PATH>`: Load a previously saved `.json` or `.bin` voice profile directly without needing `--audio-in`.

### 3. Built-In CustomVoice Speakers (`voicecli speak`)

```bash
# List built-in speakers in the CustomVoice model
voicecli speak --list-speakers

# Synthesize speech with a named speaker and optional style prompt
voicecli speak -S Vivian -p "Speak warmly and enthusiastically" -t "Welcome back!" -o vivian.wav
```

### 4. Reusable Voice Profiles (`voicecli profile`)

```bash
# Extract a Full-ICL voice profile (.json) once
voicecli profile -a sample.wav --ref-text "Transcript of the sample audio." -o speaker_icl.json

# Extract a lightweight speaker embedding profile (.bin)
voicecli profile -a sample.wav -o speaker.bin
```

### 5. Model Presets & System Info (`voicecli models`)

```bash
# Show detected CPU threads, configured models, and available presets
voicecli models

# Pre-download quantized Q8_0 models for faster CPU inference
voicecli models --download q8
```

---

## Configuration & Settings

`voicecli` resolves configuration in the following priority order:
1. **CLI Flags** (Highest precedence)
2. **Environment Variables** (`VOICECLI_MODEL_DIR`, `VOICECLI_MODEL_NAME_STYLE`, `VOICECLI_MODEL_NAME_CLONE`, `VOICECLI_MODEL_NAME_SPEAK`, `VOICECLI_TOKENIZER_MODEL`, `VOICECLI_LANGUAGE`, `VOICECLI_TEMPERATURE`, `VOICECLI_TOP_P`, `VOICECLI_TOP_K`, `VOICECLI_MAX_TOKENS`, `VOICECLI_REPETITION_PENALTY`)
3. **`settings.json`** (Located in the working directory or beside the executable)
4. **Built-in Defaults**

```json
{
  "model_name_style": "qwen-talker-1.7b-voicedesign-BF16.gguf",
  "model_name_clone": "qwen-talker-1.7b-base-BF16.gguf",
  "model_name_speak": "qwen-talker-1.7b-customvoice-BF16.gguf",
  "tokenizer_model": "qwen-tokenizer-12hz-BF16.gguf",
  "language": "auto",
  "temperature": 0.9,
  "top_p": 1.0,
  "top_k": 50,
  "max_tokens": 4096,
  "repetition_penalty": 1.05
}
```

---

## Supported Models & Quantization Presets

Pass `--model <PRESET>` (`-m`) to select any preset on the fly:

| Preset (`-m`) | Voice Design (`style`) | Voice Clone (`clone`) | CustomVoice (`speak`) | Approx. Size |
| :--- | :--- | :--- | :--- | :---: |
| **`bf16`** *(default)* | `qwen-talker-1.7b-voicedesign-BF16.gguf` | `qwen-talker-1.7b-base-BF16.gguf` | `qwen-talker-1.7b-customvoice-BF16.gguf` | ~3.8 GB |
| **`q8`** | `qwen-talker-1.7b-voicedesign-Q8_0.gguf` | `qwen-talker-1.7b-base-Q8_0.gguf` | `qwen-talker-1.7b-customvoice-Q8_0.gguf` | ~2.1 GB |
| **`q4`** | `qwen-talker-1.7b-voicedesign-Q4_K_M.gguf` | `qwen-talker-1.7b-base-Q4_K_M.gguf` | `qwen-talker-1.7b-customvoice-Q4_K_M.gguf` | ~1.3 GB |
| **`0.6b` / `fast`** | `qwen-talker-1.7b-voicedesign-Q8_0.gguf` | `qwen-talker-0.6b-base-Q8_0.gguf` | `qwen-talker-0.6b-customvoice-Q8_0.gguf` | ~0.9 GB |
| **`0.6b-q4` / `tiny`**| `qwen-talker-1.7b-voicedesign-Q4_K_M.gguf` | `qwen-talker-0.6b-base-Q4_K_M.gguf` | `qwen-talker-0.6b-customvoice-Q4_K_M.gguf` | ~0.6 GB |

All models are automatically downloaded from [`Serveurperso/Qwen3-TTS-GGUF`](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF) on first use.

---

## Building from Source

```bash
# Clone repository
git clone https://github.com/ultrapg/voicecli.git
cd voicecli

# Build portable bundle (automatically clones external/qwen3-tts if not present)
./create_portable.sh
```

Or build directly with Cargo:
```bash
git clone --recursive https://github.com/Danmoreng/qwen3-tts.cpp.git external/qwen3-tts
cargo test
cargo build --release
```

---

## License & Acknowledgments

- **License**: Licensed under the [GNU General Public License v3.0](LICENSE).
- **Qwen3-TTS**: Developed and trained by Alibaba Cloud / Qwen Team.
- **GGML**: Tensor library developed by Georgi Gerganov and the GGML community.
- **Qwen3-TTS C++ Core**: Native C++ port based on `qwen3-tts.cpp`.
