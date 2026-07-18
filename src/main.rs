use clap::{Parser, Subcommand};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::path::Path;
use serde::{Deserialize, Serialize};

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
    /// Custom JSON-driven synthesis for multi-segment generation
    Custom {
        /// Path to the JSON configuration file
        #[arg(long, short)]
        input: String,

        /// Path to save the generated final .wav file
        #[arg(long, short, default_value = "custom_output.wav")]
        output: String,
    },
}

#[derive(Serialize, Deserialize, Debug)]
struct VoiceConfig {
    prompt: Option<String>,
    audio: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct SegmentConfig {
    text: String,
    style: Option<String>,
    speed: Option<f32>,
    #[serde(default)]
    pause_after: Option<f32>,
}

#[derive(Serialize, Deserialize, Debug)]
struct CustomConfig {
    mode: String, // "synthetic" | "clone"
    voice: VoiceConfig,
    segments: Vec<SegmentConfig>,
}

fn validate_config(config: &CustomConfig) -> Result<(), String> {
    if config.mode != "synthetic" && config.mode != "clone" {
        return Err(format!("Invalid mode: '{}'. Mode must be 'synthetic' or 'clone'.", config.mode));
    }
    if config.mode == "synthetic" {
        if config.voice.prompt.as_deref().unwrap_or("").trim().is_empty() {
            return Err("For 'synthetic' mode, 'voice.prompt' must be specified and non-empty.".to_string());
        }
    } else {
        let audio_path = config.voice.audio.as_deref().unwrap_or("").trim();
        if audio_path.is_empty() {
            return Err("For 'clone' mode, 'voice.audio' (path to reference audio) must be specified and non-empty.".to_string());
        }
        if !Path::new(audio_path).exists() {
            return Err(format!("Reference audio file '{}' does not exist.", audio_path));
        }
    }
    if config.segments.is_empty() {
        return Err("Configuration must contain at least one segment in 'segments'.".to_string());
    }
    for (idx, seg) in config.segments.iter().enumerate() {
        if seg.text.trim().is_empty() {
            return Err(format!("Segment {} text is empty.", idx));
        }
        if let Some(speed) = seg.speed {
            if speed <= 0.0 {
                return Err(format!("Segment {} speed must be greater than 0. Got {}.", idx, speed));
            }
        }
        if let Some(pause) = seg.pause_after {
            if pause < 0.0 {
                return Err(format!("Segment {} pause_after must be non-negative. Got {}.", idx, pause));
            }
        }
    }
    Ok(())
}

fn change_speed(samples: &[f32], speed: f32) -> Vec<f32> {
    if (speed - 1.0).abs() < 1e-4 {
        return samples.to_vec();
    }
    let new_len = (samples.len() as f32 / speed) as usize;
    let mut output = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let pos = i as f32 * speed;
        let idx = pos as usize;
        let frac = pos - idx as f32;
        if idx + 1 < samples.len() {
            let s1 = samples[idx];
            let s2 = samples[idx + 1];
            output.push(s1 + frac * (s2 - s1));
        } else if idx < samples.len() {
            output.push(samples[idx]);
        }
    }
    output
}

fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| format!("Failed to create WAV writer: {}", e))?;
    for &sample in samples {
        writer.write_sample(sample)
            .map_err(|e| format!("Failed to write WAV sample: {}", e))?;
    }
    writer.finalize().map_err(|e| format!("Failed to finalize WAV file: {}", e))?;
    Ok(())
}

fn run_generation(config: CustomConfig, output_path: &str) -> Result<(), String> {
    validate_config(&config)?;

    let res = Python::with_gil(|py| -> PyResult<Result<(), String>> {
        let code = include_str!("../model.py");
        let module = PyModule::from_code_bound(py, code, "model.py", "model")?;

        let voice_input = match config.mode.as_str() {
            "synthetic" => config.voice.prompt.as_deref().unwrap_or(""),
            "clone" => config.voice.audio.as_deref().unwrap_or(""),
            _ => unreachable!(),
        };

        // Prepare model once
        let prep_func = module.getattr("prepare_model")?;
        prep_func.call1((&config.mode, voice_input))?;

        let gen_func = module.getattr("generate_segment")?;
        let mut final_samples = Vec::new();
        let mut resolved_sample_rate = None;

        for (idx, seg) in config.segments.iter().enumerate() {
            println!("[*] Processing segment {}/{}...", idx + 1, config.segments.len());
            
            let text_to_send = if let Some(style) = &seg.style {
                if !style.trim().is_empty() {
                    format!("[{}] {}", style.trim(), seg.text)
                } else {
                    seg.text.clone()
                }
            } else {
                seg.text.clone()
            };

            // Call Python generate_segment
            let py_res = gen_func.call1((&config.mode, text_to_send, voice_input))?;
            let (raw_samples, sr): (Vec<f32>, u32) = py_res.extract()?;
            
            if resolved_sample_rate.is_none() {
                resolved_sample_rate = Some(sr);
            }

            // Apply speed if specified
            let speed = seg.speed.unwrap_or(1.0);
            let processed_samples = change_speed(&raw_samples, speed);
            final_samples.extend(processed_samples);

            // Append pause_after silence if specified
            if let Some(pause_sec) = seg.pause_after {
                if pause_sec > 0.0 {
                    let silence_len = (pause_sec * sr as f32) as usize;
                    final_samples.resize(final_samples.len() + silence_len, 0.0);
                }
            }
        }

        let sr = resolved_sample_rate.unwrap_or(24000);
        println!("[*] Concatenation finished. Writing final WAV to {} (Sample Rate: {} Hz)...", output_path, sr);
        if let Err(e) = write_wav(output_path, &final_samples, sr) {
            return Ok(Err(e));
        }
        println!("[+] Audio generation complete!");
        Ok(Ok(()))
    });

    match res {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(e),
        Err(e) => {
            let error_msg = Python::with_gil(|_py| {
                format!("Python error: {:?}", e)
            });
            Err(error_msg)
        }
    }
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

    match cli.command {
        Commands::Style { text, prompt, output } => {
            let config = CustomConfig {
                mode: "synthetic".to_string(),
                voice: VoiceConfig {
                    prompt: Some(prompt),
                    audio: None,
                },
                segments: vec![SegmentConfig {
                    text,
                    style: None,
                    speed: None,
                    pause_after: None,
                }],
            };
            if let Err(e) = run_generation(config, &output) {
                eprintln!("[-] Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Clone { text, audio_in, output } => {
            let config = CustomConfig {
                mode: "clone".to_string(),
                voice: VoiceConfig {
                    prompt: None,
                    audio: Some(audio_in),
                },
                segments: vec![SegmentConfig {
                    text,
                    style: None,
                    speed: None,
                    pause_after: None,
                }],
            };
            if let Err(e) = run_generation(config, &output) {
                eprintln!("[-] Error: {}", e);
                std::process::exit(1);
            }
        }
        Commands::Custom { input, output } => {
            // Read and parse JSON input
            let content = std::fs::read_to_string(&input);
            match content {
                Ok(json_str) => {
                    let config: Result<CustomConfig, _> = serde_json::from_str(&json_str);
                    match config {
                        Ok(config) => {
                            if let Err(e) = run_generation(config, &output) {
                                eprintln!("[-] Error: {}", e);
                                std::process::exit(1);
                            }
                        }
                        Err(e) => {
                            eprintln!("[-] JSON configuration parsing error: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[-] Failed to read JSON input file '{}': {}", input, e);
                    std::process::exit(1);
                }
            }
        }
    }
}
