use clap::{Parser, Subcommand};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::path::Path;

#[derive(Parser)]
#[command(name = "voicecli")]
#[command(version = "0.1.0")]
#[command(author = "Antigravity")]
#[command(about = "Rust CLI for Qwen3-TTS using embedded Python via PyO3", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate speech using Voice Design style descriptions
    Style {
        /// The target text to speak
        #[arg(long, short)]
        text: String,

        /// Acoustic style description (e.g. "gender: Male. pitch: Low. speed: Fast-paced.")
        #[arg(long, short)]
        prompt: String,

        /// Path to save the generated .wav file
        #[arg(long, short, default_value = "output.wav")]
        output: String,
    },
    /// Clone a voice from a short reference audio file
    Clone {
        /// The target text to speak
        #[arg(long, short)]
        text: String,

        /// Path to the 3-10s reference audio file
        #[arg(long, short = 'a')]
        audio_in: String,

        /// Path to save the generated .wav file
        #[arg(long, short, default_value = "clone_output.wav")]
        output: String,
    },
}

fn main() {
    let cli = Cli::parse();

    // Dynamically resolve PYTHONHOME path to ensure portable operation:
    let mut python_home = None;

    // 1. Check if "python-embed" folder exists next to the running executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let local_embed = exe_dir.join("python-embed");
            if local_embed.exists() && local_embed.is_dir() {
                python_home = Some(local_embed.to_string_lossy().to_string());
            }
        }
    }

    // 2. If not next to the exe, check if "python-embed" exists in the current working directory
    if python_home.is_none() {
        let cwd_embed = Path::new("python-embed");
        if cwd_embed.exists() && cwd_embed.is_dir() {
            if let Ok(abs_cwd_embed) = cwd_embed.canonicalize() {
                python_home = Some(abs_cwd_embed.to_string_lossy().to_string());
            }
        }
    }

    // 3. Fallback to compile-time detected python home (if any)
    if python_home.is_none() {
        let compile_time_home = env!("PYTHON_SYS_HOME");
        if !compile_time_home.is_empty() {
            python_home = Some(compile_time_home.to_string());
        }
    }

    // Set PYTHONHOME dynamically if resolved
    if let Some(home) = python_home {
        std::env::set_var("PYTHONHOME", home);
    }

    // Dynamically configure HF_HOME to cache model weights on the same drive as the binary (e.g. drive D:)
    if std::env::var("HF_HOME").is_err() {
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let cache_dir = exe_dir.join(".cache").join("huggingface");
                std::env::set_var("HF_HOME", cache_dir);
            }
        }
    }

    // Initialize Python interpreter
    println!("[*] Initializing embedded Python interpreter...");
    pyo3::prepare_freethreaded_python();

    let res = Python::with_gil(|py| -> PyResult<()> {
        // Embed the Python code as a string literal
        let code = include_str!("../model.py");
        let module = PyModule::from_code_bound(py, code, "model.py", "model")?;

        match cli.command {
            Commands::Style { text, prompt, output } => {
                let func = module.getattr("generate_style")?;
                func.call1((text, prompt, output))?;
            }
            Commands::Clone { text, audio_in, output } => {
                let path = Path::new(&audio_in);
                if !path.exists() {
                    eprintln!("[-] Error: Reference audio file '{}' does not exist.", audio_in);
                    std::process::exit(1);
                }
                let func = module.getattr("generate_clone")?;
                func.call1((text, audio_in, output))?;
            }
        }
        Ok(())
    });

    if let Err(e) = res {
        eprintln!("\n[-] VoiceCLI execution encountered a Python error:");
        Python::with_gil(|py| {
            e.print(py);
        });
        std::process::exit(1);
    }
}
