use clap::{Parser, Subcommand};
use std::ffi::CString;
use std::path::Path;

extern "C" {
    fn c_generate_style(text: *const i8, prompt: *const i8, output_path: *const i8);
    fn c_generate_clone(text: *const i8, audio_path: *const i8, output_path: *const i8);
}

#[derive(Parser)]
#[command(name = "voicecli")]
#[command(version = "0.1.0")]
#[command(author = "Antigravity")]
#[command(about = "High-Performance Native Qwen3-TTS CLI (Rust + C/C++)", long_about = None)]
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

    // Dynamically resolve base directory (next to executable)
    let mut base_dir = String::new();
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_parent) = exe_path.parent() {
            base_dir = exe_parent.to_string_lossy().to_string();
        }
    }
    if !base_dir.is_empty() && std::env::var("VOICECLI_BASE_DIR").is_err() {
        std::env::set_var("VOICECLI_BASE_DIR", &base_dir);
    }

    match cli.command {
        Commands::Style { text, prompt, output } => {
            let c_text = CString::new(text).unwrap();
            let c_prompt = CString::new(prompt).unwrap();
            let c_output = CString::new(output).unwrap();
            unsafe {
                c_generate_style(c_text.as_ptr(), c_prompt.as_ptr(), c_output.as_ptr());
            }
        }
        Commands::Clone { text, audio_in, output } => {
            let path = Path::new(&audio_in);
            if !path.exists() {
                eprintln!("[-] Error: Reference audio file '{}' does not exist.", audio_in);
                std::process::exit(1);
            }
            let c_text = CString::new(text).unwrap();
            let c_audio = CString::new(audio_in).unwrap();
            let c_output = CString::new(output).unwrap();
            unsafe {
                c_generate_clone(c_text.as_ptr(), c_audio.as_ptr(), c_output.as_ptr());
            }
        }
    }
}
