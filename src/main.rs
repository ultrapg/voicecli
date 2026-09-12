use clap::{Parser, Subcommand};
use std::ffi::CString;
use std::path::Path;

extern "C" {
    fn c_generate_style(text: *const i8, prompt: *const i8, output_path: *const i8);
    fn c_generate_clone(text: *const i8, audio_path: *const i8, output_path: *const i8);
    fn c_generate_style_chunked(
        chunks: *const *const i8,
        num_chunks: i32,
        prompt: *const i8,
        output_path: *const i8,
    );
    fn c_generate_clone_chunked(
        chunks: *const *const i8,
        num_chunks: i32,
        audio_path: *const i8,
        output_path: *const i8,
    );
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
        /// The target text to speak (use either --text or --file)
        #[arg(long, short)]
        text: Option<String>,

        /// Path to a .txt file containing text to speak
        #[arg(long, short = 'f')]
        file: Option<String>,

        /// Acoustic style description (e.g. "gender: Male. pitch: Low. speed: Fast-paced.")
        #[arg(long, short)]
        prompt: String,

        /// Path to save the generated .wav file
        #[arg(long, short, default_value = "output.wav")]
        output: String,

        /// Automatically split long text into natural chunks and stitch audio together
        #[arg(long)]
        chunk: bool,
    },
    /// Clone a voice from a short reference audio file
    Clone {
        /// The target text to speak (use either --text or --file)
        #[arg(long, short)]
        text: Option<String>,

        /// Path to a .txt file containing text to speak
        #[arg(long, short = 'f')]
        file: Option<String>,

        /// Path to the 3-10s reference audio file
        #[arg(long, short = 'a')]
        audio_in: String,

        /// Path to save the generated .wav file
        #[arg(long, short, default_value = "clone_output.wav")]
        output: String,

        /// Automatically split long text into natural chunks and stitch audio together
        #[arg(long)]
        chunk: bool,
    },
}

fn resolve_input_text(text: Option<String>, file: Option<String>) -> Result<String, String> {
    match (text, file) {
        (Some(t), None) => {
            let trimmed = t.trim().to_string();
            if trimmed.is_empty() {
                Err("Provided --text is empty.".to_string())
            } else {
                Ok(trimmed)
            }
        }
        (None, Some(f)) => {
            let path = Path::new(&f);
            if !path.exists() {
                return Err(format!("Input file '{}' does not exist.", f));
            }
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read file '{}': {}", f, e))?;
            let trimmed = content.trim().to_string();
            if trimmed.is_empty() {
                Err(format!("Input file '{}' is empty.", f))
            } else {
                Ok(trimmed)
            }
        }
        (Some(_), Some(_)) => {
            Err("Please specify either --text or --file, but not both.".to_string())
        }
        (None, None) => {
            Err("Please provide input text using either --text \"...\" or --file <PATH.txt>.".to_string())
        }
    }
}

fn split_into_chunks(text: &str, target_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let text = text.trim();
    if text.is_empty() {
        return chunks;
    }

    let paragraphs: Vec<&str> = text
        .split('\n')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();

    let mut current_chunk = String::new();

    for para in paragraphs {
        let mut sentences = Vec::new();
        let mut last = 0;
        let chars: Vec<char> = para.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            if c == '.' || c == '!' || c == '?' || c == '。' || c == '！' || c == '？' {
                if i + 1 == chars.len()
                    || chars[i + 1].is_whitespace()
                    || chars[i + 1] == '"'
                    || chars[i + 1] == '\''
                {
                    let sentence_str: String = chars[last..=i].iter().collect();
                    let trimmed = sentence_str.trim();
                    let is_abbrev = trimmed.ends_with("Mr.")
                        || trimmed.ends_with("Mrs.")
                        || trimmed.ends_with("Dr.")
                        || trimmed.ends_with("Prof.")
                        || trimmed.ends_with("vs.")
                        || trimmed.ends_with("e.g.")
                        || trimmed.ends_with("i.e.")
                        || trimmed.ends_with("etc.");
                    if !is_abbrev {
                        sentences.push(sentence_str);
                        last = i + 1;
                    }
                }
            }
            i += 1;
        }
        if last < chars.len() {
            let remaining: String = chars[last..].iter().collect();
            if !remaining.trim().is_empty() {
                sentences.push(remaining);
            }
        }

        for sentence in sentences {
            let s = sentence.trim();
            if s.is_empty() {
                continue;
            }

            if s.chars().count() > target_chars + (target_chars / 2) {
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                }

                let mut clause_acc = String::new();
                for word in s.split_whitespace() {
                    if clause_acc.chars().count() + word.chars().count() + 1 > target_chars
                        && !clause_acc.is_empty()
                    {
                        chunks.push(clause_acc.clone());
                        clause_acc.clear();
                    }
                    if !clause_acc.is_empty() {
                        clause_acc.push(' ');
                    }
                    clause_acc.push_str(word);
                }
                if !clause_acc.is_empty() {
                    chunks.push(clause_acc);
                }
            } else {
                if current_chunk.chars().count() + s.chars().count() + 1 > target_chars
                    && !current_chunk.is_empty()
                {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                }
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(s);
            }
        }

        if current_chunk.chars().count() >= target_chars / 2 {
            chunks.push(current_chunk.clone());
            current_chunk.clear();
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

fn main() {
    let cli = Cli::parse();

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
        Commands::Style {
            text,
            file,
            prompt,
            output,
            chunk,
        } => {
            let input_text = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(err) => {
                    eprintln!("[-] Error: {}", err);
                    std::process::exit(1);
                }
            };

            if chunk {
                let chunks = split_into_chunks(&input_text, 280);
                if chunks.is_empty() {
                    eprintln!("[-] Error: No valid text to synthesize.");
                    std::process::exit(1);
                }

                if chunks.len() == 1 {
                    let c_text = CString::new(chunks[0].clone()).unwrap();
                    let c_prompt = CString::new(prompt).unwrap();
                    let c_output = CString::new(output).unwrap();
                    unsafe {
                        c_generate_style(c_text.as_ptr(), c_prompt.as_ptr(), c_output.as_ptr());
                    }
                } else {
                    println!(
                        "[*] Auto-Chunk Mode enabled: Split text into {} natural chunks.",
                        chunks.len()
                    );
                    let c_chunks: Vec<CString> = chunks
                        .iter()
                        .map(|s| CString::new(s.as_str()).unwrap())
                        .collect();
                    let c_ptrs: Vec<*const i8> = c_chunks.iter().map(|c| c.as_ptr()).collect();
                    let c_prompt = CString::new(prompt).unwrap();
                    let c_output = CString::new(output).unwrap();
                    unsafe {
                        c_generate_style_chunked(
                            c_ptrs.as_ptr(),
                            c_ptrs.len() as i32,
                            c_prompt.as_ptr(),
                            c_output.as_ptr(),
                        );
                    }
                }
            } else {
                let c_text = CString::new(input_text).unwrap();
                let c_prompt = CString::new(prompt).unwrap();
                let c_output = CString::new(output).unwrap();
                unsafe {
                    c_generate_style(c_text.as_ptr(), c_prompt.as_ptr(), c_output.as_ptr());
                }
            }
        }
        Commands::Clone {
            text,
            file,
            audio_in,
            output,
            chunk,
        } => {
            let path = Path::new(&audio_in);
            if !path.exists() {
                eprintln!("[-] Error: Reference audio file '{}' does not exist.", audio_in);
                std::process::exit(1);
            }

            let input_text = match resolve_input_text(text, file) {
                Ok(t) => t,
                Err(err) => {
                    eprintln!("[-] Error: {}", err);
                    std::process::exit(1);
                }
            };

            if chunk {
                let chunks = split_into_chunks(&input_text, 280);
                if chunks.is_empty() {
                    eprintln!("[-] Error: No valid text to synthesize.");
                    std::process::exit(1);
                }

                if chunks.len() == 1 {
                    let c_text = CString::new(chunks[0].clone()).unwrap();
                    let c_audio = CString::new(audio_in).unwrap();
                    let c_output = CString::new(output).unwrap();
                    unsafe {
                        c_generate_clone(c_text.as_ptr(), c_audio.as_ptr(), c_output.as_ptr());
                    }
                } else {
                    println!(
                        "[*] Auto-Chunk Mode enabled: Split text into {} natural chunks.",
                        chunks.len()
                    );
                    let c_chunks: Vec<CString> = chunks
                        .iter()
                        .map(|s| CString::new(s.as_str()).unwrap())
                        .collect();
                    let c_ptrs: Vec<*const i8> = c_chunks.iter().map(|c| c.as_ptr()).collect();
                    let c_audio = CString::new(audio_in).unwrap();
                    let c_output = CString::new(output).unwrap();
                    unsafe {
                        c_generate_clone_chunked(
                            c_ptrs.as_ptr(),
                            c_ptrs.len() as i32,
                            c_audio.as_ptr(),
                            c_output.as_ptr(),
                        );
                    }
                }
            } else {
                let c_text = CString::new(input_text).unwrap();
                let c_audio = CString::new(audio_in).unwrap();
                let c_output = CString::new(output).unwrap();
                unsafe {
                    c_generate_clone(c_text.as_ptr(), c_audio.as_ptr(), c_output.as_ptr());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_into_chunks_short() {
        let text = "Hello world! This is a simple test.";
        let chunks = split_into_chunks(text, 280);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world! This is a simple test.");
    }

    #[test]
    fn test_split_into_chunks_multiple() {
        let text = "First sentence is here. Second sentence follows right after. Third sentence is also present.\n\nNew paragraph starts here. It has another sentence. And yet another one to make it longer.";
        let chunks = split_into_chunks(text, 60);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(!chunk.trim().is_empty());
        }
    }

    #[test]
    fn test_split_into_chunks_abbreviations() {
        let text = "Dr. Smith met with Mr. Brown at 5 p.m. to discuss the results.";
        let chunks = split_into_chunks(text, 280);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("Dr. Smith"));
    }

    #[test]
    fn test_resolve_input_text_valid() {
        let res = resolve_input_text(Some("Hello there".to_string()), None);
        assert_eq!(res.unwrap(), "Hello there");
    }

    #[test]
    fn test_resolve_input_text_empty() {
        let res = resolve_input_text(Some("   ".to_string()), None);
        assert!(res.is_err());
    }

    #[test]
    fn test_resolve_input_text_both() {
        let res = resolve_input_text(Some("a".to_string()), Some("b.txt".to_string()));
        assert!(res.is_err());
    }

    #[test]
    fn test_resolve_input_text_none() {
        let res = resolve_input_text(None, None);
        assert!(res.is_err());
    }
}
