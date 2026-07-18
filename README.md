# voicecli - Standalone Qwen3-TTS Rust CLI

`voicecli` is a lightweight, standalone Command Line Interface (CLI) application written in Rust that interfaces with the **Qwen3-TTS** model series using an embedded Python interpreter. It allows you to generate speech from text using natural language description prompts (Voice Design) or clone a voice from a short reference audio clip (Voice Cloning).

This version introduces the **`custom`** subcommand, enabling structured multi-segment speech generation with fine-grained style, speed, and pause controls via a unified JSON interface. The existing `style` and `clone` commands serve as aliases/wrappers that generate the JSON internally and route through the same unified pipeline.

---

## Architecture Overview

```
                        +----------------------------+
                        |       voicecli (Rust)      |
                        |                            |
                        |   - Parses CLI (clap)      |
                        |   - JSON Schema Validation |
                        |   - Performs Path Checks   |
                        |   - Audio resample/silence |
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
                        |   Qwen3-TTS-12Hz-1.7B      |
                        |   - VRAM / RAM Inference   |
                        |   - Generates Speech       |
                        +----------------------------+
```

1. **Rust Binary**: Handles CLI parsing via `clap`, parses and validates the input JSON configuration using `serde_json`, manages audio resampling for speed changes, appends pauses between segments, and writes the final audio using the `hound` crate.
2. **Embedded Python (`model.py`)**: Runs inside the same process space using `PyO3`. It manages the lifecycle of the models, caches voice clone prompts (x-vectors) to minimize latency, runs the PyTorch model inference, and returns raw float samples to Rust.
3. **Data Exchange**: Audio samples are returned from Python to Rust as lists of floats, avoiding external subprocesses or IPC overhead.

---

## Detailed JSON Configuration Schema (`custom` command)

The `custom` command uses a unified JSON configuration interface. It accepts the path to a JSON file via `--input` (or `-i`) and saves the final concatenated WAV file to `--output` (or `-o`).

### Schema Definition
```json
{
  "mode": "synthetic" | "clone",
  "voice": {
    "prompt": "string",     // Required for "synthetic" mode: voice style description
    "audio": "string"       // Required for "clone" mode: path to reference audio file
  },
  "segments": [
    {
      "text": "string",          // Required: text to synthesize
      "style": "string | null",  // Optional: style tags prepended to text (e.g. "whispered, tense")
      "speed": "number | null",  // Optional: speed multiplier (0.5 = half speed, 2.0 = double)
      "pause_after": "number"    // Optional: seconds of silence after this segment (defaults to 0.0)
    }
  ]
}
```

### Schema Rules & Validation
- **`mode`**: Must be either `"synthetic"` or `"clone"`.
- **`voice.prompt`**: Must be present and non-empty if `mode` is `"synthetic"`.
- **`voice.audio`**: Must be present and point to an existing reference audio file if `mode` is `"clone"`.
- **`segments`**: Must contain at least one segment.
- **`speed`**: If specified, must be greater than `0.0`.
- **`pause_after`**: If specified, must be non-negative.

---

## Usage

### 1. `custom` Subcommand (Unified JSON-driven interface)
Runs the unified multi-segment speech generation.

```bash
# Basic usage
voicecli custom --input examples/synthetic_config.json --output custom_output.wav

# Short option usage
voicecli custom -i examples/clone_config.json -o custom_cloned_output.wav
```

### 2. `style` Subcommand (Voice Design Wrapper)
Generates speech using a natural language prompt. Behind the scenes, it builds a single-segment synthetic JSON config and executes the shared backend.

```bash
voicecli style --text "Welcome to the future of speech synthesis." --prompt "gender: Female. pitch: High. speed: Normal." --output style_out.wav
```

### 3. `clone` Subcommand (Voice Cloning Wrapper)
Clones a target voice using a short reference audio file. Behaves as an alias that builds a single-segment cloning JSON config internally.

```bash
voicecli clone --text "Hello, I am speaking in your voice now." --audio-in reference.wav --output cloned_out.wav
```

---

## Portable Deployments & Cross-Compilation

To simplify deployment and keep the environment self-contained, `voicecli` can be packaged with portable Python runtimes (including CPU versions of `torch`, `torchaudio`, `soundfile`, and `qwen-tts`).

### Packaging Linux & Windows Portables on Linux
The repository contains a helper script `create_portable_packages.sh` that automates cross-compilation and packaging. It uses a temporary **Podman** container to cross-compile for Windows using `mingw-w64`.

To generate both packages, simply run:
```bash
./create_portable_packages.sh
```

This script performs the following tasks:
1. **Linux Build**:
   - Downloads a standalone Linux Python runtime (`cpython-3.11`).
   - Uses `pip` to install local Python dependencies (`torch`, `torchaudio` from PyTorch CPU index, `qwen-tts`, `soundfile`).
   - Compiles `voicecli` targeting `x86_64-unknown-linux-gnu`, setting the `rpath` so it links dynamically to the portable `python-embed/lib` folder.
   - Compresses the build into `deploy/voicecli-linux-x86_64.tar.gz`.
2. **Windows Build (Cross-Compiled)**:
   - Spawns a temporary Podman container running the standard Rust image.
   - Installs the MinGW cross-compiler and target `x86_64-pc-windows-gnu`.
   - Cross-compiles `voicecli.exe` using `CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=x86_64-w64-mingw32-gcc`. Pyo3's `generate-import-lib` feature creates the required `.lib` files automatically without needing Windows development SDKs on host.
   - Downloads the Windows Python 3.11 embeddable package and configures `python311._pth` import rules.
   - Uses the host pip with cross-platform flags to download and unpack Windows CPU wheels for `torch`, `torchaudio`, `qwen-tts`, and `soundfile` directly into the staging folder.
   - Compresses the build into `deploy/voicecli-windows.zip`.

---

## Example Configurations

You can find reference configurations in the repository under:
- **Synthetic (Voice Design)**: [examples/synthetic_config.json](examples/synthetic_config.json)
- **Voice Cloning**: [examples/clone_config.json](examples/clone_config.json)

---

## Performance & Dtype Optimizations
- **bfloat16 CPU fallback**: When no NVIDIA GPU/CUDA is detected, `voicecli` uses `torch.bfloat16` for CPU inference. This reduces memory footprint by 50% and improves inference speed significantly on CPU architectures.

## Troubleshooting

### Linux: Shared Library Missing (`libpython3.11.so.1.0`)
If you run the binary outside the portable folder and receive dynamic link errors:
- Ensure `LD_LIBRARY_PATH` includes the `python-embed/lib` folder.
- When executing the binary, it checks `python-embed` next to the executable or in the current working directory, falling back to the system environment if needed.

### Windows: `STATUS_DLL_NOT_FOUND (0xc0000135)`
- Ensure the portable `python311.dll` and other runtime dependencies are kept in the same directory as the `voicecli.exe` binary.

---

## License

GNU General Public License v3.0
