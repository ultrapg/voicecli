use clap::{Parser, Subcommand};
use std::ffi::CString;
use std::path::Path;

extern "C" {
    fn c_generate_style(text: *const i8, prompt: *const i8, output_path: *const i8, speed: f32);
    fn c_generate_clone(text: *const i8, audio_path: *const i8, output_path: *const i8, speed: f32);
    fn c_generate_style_chunked(
        chunks: *const *const i8,
        num_chunks: i32,
        prompt: *const i8,
        output_path: *const i8,
        speed: f32,
    );
    fn c_generate_clone_chunked(
        chunks: *const *const i8,
        num_chunks: i32,
        audio_path: *const i8,
        output_path: *const i8,
        speed: f32,
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

        /// Speech speed multiplier (e.g. 0.8 for slower, 1.25 for faster, default: 1.0)
        #[arg(long, short = 's', default_value_t = 1.0)]
        speed: f32,

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

        /// Speech speed multiplier (e.g. 0.8 for slower, 1.25 for faster, default: 1.0)
        #[arg(long, short = 's', default_value_t = 1.0)]
        speed: f32,

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

fn validate_speed(speed: f32) -> Result<(), String> {
    if speed < 0.25 || speed > 3.0 || speed.is_nan() {
        Err(format!(
            "Speech speed must be between 0.25 and 3.0 (got {}).",
            speed
        ))
    } else {
        Ok(())
    }
}

fn is_abbreviation(word: &str) -> bool {
    let clean = word.trim_matches(|c: char| {
        c == '"'
            || c == '\''
            || c == '('
            || c == ')'
            || c == '['
            || c == ']'
            || c == '{'
            || c == '}'
            || c == '“'
            || c == '”'
            || c == '‘'
            || c == '’'
    });
    let lower = clean.to_lowercase();

    const KNOWN_ABBREVS: &[&str] = &[
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "rev.", "gen.", "gov.",
        "sgt.", "capt.", "lt.", "col.", "vs.", "e.g.", "i.e.", "etc.", "al.",
        "approx.", "appx.", "dept.", "fig.", "no.", "vol.", "est.", "inc.",
        "corp.", "co.", "ltd.", "st.", "ave.", "rd.", "blvd.", "u.s.", "u.k.",
        "e.u.", "a.m.", "p.m.", "am.", "pm.", "a.d.", "b.c.",
    ];

    if KNOWN_ABBREVS.contains(&lower.as_str()) {
        return true;
    }

    // Single-letter initials like "J." or "W."
    let chars: Vec<char> = clean.chars().collect();
    if chars.len() == 2 && chars[0].is_alphabetic() && chars[1] == '.' {
        return true;
    }

    false
}

fn extract_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut last = 0;
    let mut i = 0;

    while i < len {
        let c = chars[i];
        if c == '.' || c == '!' || c == '?' || c == '。' || c == '！' || c == '？' {
            // Ignore ellipsis (e.g. '..')
            if i + 1 < len && chars[i + 1] == '.' {
                i += 1;
                continue;
            }

            // Consume any trailing closing quotes or brackets
            let mut end_punct = i;
            while end_punct + 1 < len {
                let next_c = chars[end_punct + 1];
                if next_c == '"'
                    || next_c == '\''
                    || next_c == '”'
                    || next_c == '’'
                    || next_c == '»'
                    || next_c == ')'
                    || next_c == ']'
                    || next_c == '}'
                {
                    end_punct += 1;
                } else {
                    break;
                }
            }

            // Check if boundary is followed by whitespace or end of string
            if end_punct + 1 == len || chars[end_punct + 1].is_whitespace() {
                // If followed by lowercase letters (e.g. dialogue tag: `"Hello!" she said.`), do not split
                let mut is_dialogue_continuation = false;
                let mut next_idx = end_punct + 1;
                while next_idx < len && chars[next_idx].is_whitespace() {
                    next_idx += 1;
                }
                if next_idx < len && chars[next_idx].is_lowercase() {
                    is_dialogue_continuation = true;
                }

                let preceding: String = chars[last..=i].iter().collect();
                let last_word = preceding.split_whitespace().last().unwrap_or("");
                if !is_dialogue_continuation && !is_abbreviation(last_word) {
                    let sentence_str: String = chars[last..=end_punct].iter().collect();
                    let trimmed = sentence_str.trim().to_string();
                    if !trimmed.is_empty() {
                        sentences.push(trimmed);
                    }
                    last = end_punct + 1;
                    i = end_punct;
                }
            }
        }
        i += 1;
    }

    if last < len {
        let remaining: String = chars[last..].iter().collect();
        let trimmed = remaining.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }
    }

    sentences
}

fn split_long_sentence(sentence: &str, target_chars: usize) -> Vec<String> {
    if sentence.chars().count() <= target_chars {
        return vec![sentence.to_string()];
    }

    let mut parts = Vec::new();
    let mut remaining = sentence.trim().to_string();

    const CLAUSE_DELIMS: &[&str] = &[
        "; ", ": ", " — ", " – ", " -- ",
        ", and ", ", but ", ", however, ", ", although ",
        ", because ", ", while ", ", which ", ", whereas ",
        ", so ", ", or ", ", yet ", ", "
    ];

    while remaining.chars().count() > target_chars {
        let count = remaining.chars().count();
        let search_limit = std::cmp::min(count, target_chars + 50);
        let window: String = remaining.chars().take(search_limit).collect();

        let mut best_break = None;

        for &delim in CLAUSE_DELIMS {
            if let Some(pos) = window.rfind(delim) {
                let char_pos = window[..pos].chars().count();
                if char_pos >= target_chars / 3 {
                    let delim_char_len = delim.trim_end().chars().count();
                    best_break = Some(char_pos + delim_char_len);
                    break;
                }
            }
        }

        if best_break.is_none() {
            if let Some(pos) = window.rfind(' ') {
                let char_pos = window[..pos].chars().count();
                if char_pos >= target_chars / 3 {
                    best_break = Some(char_pos);
                }
            }
        }

        let break_point = best_break.unwrap_or(target_chars);
        let chunk: String = remaining.chars().take(break_point).collect();
        let chunk_trimmed = chunk.trim().to_string();
        if !chunk_trimmed.is_empty() {
            parts.push(chunk_trimmed);
        }
        remaining = remaining.chars().skip(break_point).collect::<String>().trim().to_string();
    }

    if !remaining.is_empty() {
        parts.push(remaining);
    }

    parts
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
        let sentences = extract_sentences(para);

        for s in sentences {
            let sub_sentences = split_long_sentence(&s, target_chars);

            for sub in sub_sentences {
                let sub = sub.trim();
                if sub.is_empty() {
                    continue;
                }

                let current_len = current_chunk.chars().count();
                let sub_len = sub.chars().count();

                if current_len == 0 {
                    current_chunk.push_str(sub);
                } else if current_len + sub_len + 1 <= target_chars {
                    current_chunk.push(' ');
                    current_chunk.push_str(sub);
                } else {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                    current_chunk.push_str(sub);
                }
            }
        }

        // At natural paragraph boundaries, flush chunk if substantial to respect paragraph rhythm
        if current_chunk.chars().count() >= (target_chars * 2) / 3 {
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
            speed,
            chunk,
        } => {
            if let Err(err) = validate_speed(speed) {
                eprintln!("[-] Error: {}", err);
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
                let chunks = split_into_chunks(&input_text, 600);
                if chunks.is_empty() {
                    eprintln!("[-] Error: No valid text to synthesize.");
                    std::process::exit(1);
                }

                if chunks.len() == 1 {
                    let c_text = CString::new(chunks[0].clone()).unwrap();
                    let c_prompt = CString::new(prompt).unwrap();
                    let c_output = CString::new(output).unwrap();
                    unsafe {
                        c_generate_style(c_text.as_ptr(), c_prompt.as_ptr(), c_output.as_ptr(), speed);
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
                            speed,
                        );
                    }
                }
            } else {
                let c_text = CString::new(input_text).unwrap();
                let c_prompt = CString::new(prompt).unwrap();
                let c_output = CString::new(output).unwrap();
                unsafe {
                    c_generate_style(c_text.as_ptr(), c_prompt.as_ptr(), c_output.as_ptr(), speed);
                }
            }
        }
        Commands::Clone {
            text,
            file,
            audio_in,
            output,
            speed,
            chunk,
        } => {
            if let Err(err) = validate_speed(speed) {
                eprintln!("[-] Error: {}", err);
                std::process::exit(1);
            }

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
                let chunks = split_into_chunks(&input_text, 600);
                if chunks.is_empty() {
                    eprintln!("[-] Error: No valid text to synthesize.");
                    std::process::exit(1);
                }

                if chunks.len() == 1 {
                    let c_text = CString::new(chunks[0].clone()).unwrap();
                    let c_audio = CString::new(audio_in).unwrap();
                    let c_output = CString::new(output).unwrap();
                    unsafe {
                        c_generate_clone(c_text.as_ptr(), c_audio.as_ptr(), c_output.as_ptr(), speed);
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
                            speed,
                        );
                    }
                }
            } else {
                let c_text = CString::new(input_text).unwrap();
                let c_audio = CString::new(audio_in).unwrap();
                let c_output = CString::new(output).unwrap();
                unsafe {
                    c_generate_clone(c_text.as_ptr(), c_audio.as_ptr(), c_output.as_ptr(), speed);
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

    #[test]
    fn test_validate_speed_valid() {
        assert!(validate_speed(1.0).is_ok());
        assert!(validate_speed(0.25).is_ok());
        assert!(validate_speed(3.0).is_ok());
        assert!(validate_speed(1.5).is_ok());
        assert!(validate_speed(0.8).is_ok());
    }

    #[test]
    fn test_validate_speed_invalid() {
        assert!(validate_speed(0.2).is_err());
        assert!(validate_speed(3.1).is_err());
        assert!(validate_speed(-1.0).is_err());
        assert!(validate_speed(f32::NAN).is_err());
    }

    #[test]
    fn test_extract_sentences_quotes() {
        let text = "He said, \"The results are in!\" Then he smiled. \"Are you sure?\" she asked.";
        let sents = extract_sentences(text);
        assert_eq!(sents.len(), 3);
        assert_eq!(sents[0], "He said, \"The results are in!\"");
        assert_eq!(sents[1], "Then he smiled.");
        assert_eq!(sents[2], "\"Are you sure?\" she asked.");
    }

    #[test]
    fn test_extract_sentences_abbreviations_and_initials() {
        let text = "Dr. Smith met with Mr. Brown at 5 p.m. to discuss J. K. Rowling books. The U.S. team agreed.";
        let sents = extract_sentences(text);
        assert_eq!(sents.len(), 2);
        assert_eq!(sents[0], "Dr. Smith met with Mr. Brown at 5 p.m. to discuss J. K. Rowling books.");
        assert_eq!(sents[1], "The U.S. team agreed.");
    }

    #[test]
    fn test_split_long_sentence_clauses() {
        let text = "Artificial intelligence has undergone remarkable transformations over the past decade, and researchers from across the globe have contributed to breakthrough architectures, while industry leaders invested billions of dollars to build distributed computing clusters.";
        let parts = split_long_sentence(text, 120);
        assert!(parts.len() >= 2);
        for part in &parts {
            assert!(!part.trim().is_empty());
        }
    }
}
