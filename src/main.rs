use clap::{Args, Parser, Subcommand};
use std::ffi::CString;
use std::io::{IsTerminal, Read};
use std::os::raw::c_char;
use std::path::Path;

#[repr(C)]
struct VoiceCliOptions {
    output_path: *const c_char,
    model_override: *const c_char,
    language: *const c_char,
    speed: f32,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    max_tokens: i32,
    repetition_penalty: f32,
    threads: i32,
    stream: i32,
    stream_chunk_sec: f32,
    normalize: i32,
    play: i32,
    ref_text: *const c_char,
    voice_profile: *const c_char,
    save_profile: *const c_char,
}

extern "C" {
    fn c_generate_style_ex(
        text: *const c_char,
        prompt: *const c_char,
        opts: *const VoiceCliOptions,
    ) -> i32;
    fn c_generate_style_chunked_ex(
        chunks: *const *const c_char,
        num_chunks: i32,
        prompt: *const c_char,
        opts: *const VoiceCliOptions,
    ) -> i32;
    fn c_generate_clone_ex(
        text: *const c_char,
        audio_path: *const c_char,
        opts: *const VoiceCliOptions,
    ) -> i32;
    fn c_generate_clone_chunked_ex(
        chunks: *const *const c_char,
        num_chunks: i32,
        audio_path: *const c_char,
        opts: *const VoiceCliOptions,
    ) -> i32;
    fn c_generate_speak_ex(
        text: *const c_char,
        speaker: *const c_char,
        prompt: *const c_char,
        opts: *const VoiceCliOptions,
    ) -> i32;
    fn c_generate_speak_chunked_ex(
        chunks: *const *const c_char,
        num_chunks: i32,
        speaker: *const c_char,
        prompt: *const c_char,
        opts: *const VoiceCliOptions,
    ) -> i32;
    fn c_extract_profile(
        audio_path: *const c_char,
        ref_text: *const c_char,
        output_profile_path: *const c_char,
        model_override: *const c_char,
    ) -> i32;
    fn c_download_models(preset: *const c_char) -> i32;
    fn c_print_system_info();
}

#[derive(Args, Clone, Debug)]
struct CommonSynthArgs {
    /// The target text to speak (use --text, --file, or pipe via stdin)
    #[arg(long, short = 't')]
    text: Option<String>,

    /// Path to a .txt/.md file containing text to speak (use "-" for stdin)
    #[arg(long, short = 'f')]
    file: Option<String>,

    /// Read input text from standard input (stdin)
    #[arg(long)]
    stdin: bool,

    /// Speech speed multiplier (0.25 to 3.0, default: 1.0)
    #[arg(long, short = 's', default_value_t = 1.0)]
    speed: f32,

    /// Split text into natural sentence chunks and stitch audio seamlessly
    #[arg(long)]
    chunk: bool,

    /// Disable automatic chunking even for long texts (>800 chars)
    #[arg(long)]
    no_chunk: bool,

    /// Target maximum characters per chunk when chunking is active
    #[arg(long, default_value_t = 600)]
    chunk_size: usize,

    /// Language code: auto, en, de, fr, es, it, pt, zh, ja, ko, ru
    #[arg(long, short = 'l')]
    lang: Option<String>,

    /// Model preset or GGUF filename (bf16, q8, q4, 0.6b, fast, tiny)
    #[arg(long, short = 'm')]
    model: Option<String>,

    /// Number of CPU threads to use (default: auto-detect physical cores)
    #[arg(long, short = 'j', default_value_t = 0)]
    threads: i32,

    /// Sampling temperature (0.0 = greedy, default from settings.json: 0.9)
    #[arg(long, default_value_t = -1.0)]
    temperature: f32,

    /// Nucleus sampling top-p probability (0.0 to 1.0, default: 1.0)
    #[arg(long, default_value_t = 0.0)]
    top_p: f32,

    /// Top-k sampling count (0 = disabled, default: 50)
    #[arg(long, default_value_t = -1)]
    top_k: i32,

    /// Repetition penalty (default: 1.05)
    #[arg(long, default_value_t = 0.0)]
    repetition_penalty: f32,

    /// Maximum audio tokens per generation pass (default: 4096)
    #[arg(long, default_value_t = 0)]
    max_tokens: i32,

    /// Use low-latency streaming vocoder synthesis
    #[arg(long)]
    stream: bool,

    /// Streaming vocoder chunk size in seconds (default: 1.0)
    #[arg(long, default_value_t = 1.0)]
    stream_chunk_sec: f32,

    /// Peak-normalize output audio to -0.5 dBFS
    #[arg(long)]
    normalize: bool,

    /// Automatically play the generated audio file upon completion
    #[arg(long)]
    play: bool,

    /// Strip Markdown syntax (headers, links, emphasis, code fences) from input text
    #[arg(long)]
    clean_markdown: bool,
}

#[derive(Parser)]
#[command(name = "voicecli")]
#[command(version = "0.3.0")]
#[command(author = "Antigravity")]
#[command(
    about = "High-Performance Native Qwen3-TTS CLI (Rust + C/C++ GGML)",
    long_about = "Standalone native Qwen3-TTS engine for Voice Design (style), zero-shot & full-ICL Voice Cloning (clone), built-in CustomVoice speakers (speak), and cached voice profiles."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate speech using natural-language Voice Design descriptions
    Style {
        #[command(flatten)]
        common: CommonSynthArgs,

        /// Acoustic style description (e.g. "gender: Male. pitch: Low. speed: Fast-paced.")
        #[arg(long, short = 'p')]
        prompt: String,

        /// Path to save the generated .wav file (use "-" for stdout)
        #[arg(long, short = 'o', default_value = "output.wav")]
        output: String,
    },
    /// Clone a voice from a 3-10s reference audio file or cached voice profile
    Clone {
        #[command(flatten)]
        common: CommonSynthArgs,

        /// Path to the reference audio file (.wav, .mp3, .flac, .ogg, .m4a)
        #[arg(long, short = 'a')]
        audio_in: Option<String>,

        /// Transcript of the reference audio (enables high-fidelity Full-ICL voice cloning)
        #[arg(long)]
        ref_text: Option<String>,

        /// Path to a text file containing the transcript of the reference audio
        #[arg(long)]
        ref_file: Option<String>,

        /// Use a pre-extracted voice profile (.json or .bin) for instant cloning
        #[arg(long)]
        voice_profile: Option<String>,

        /// Save the extracted voice profile (.json or .bin) for fast future reuse
        #[arg(long)]
        save_profile: Option<String>,

        /// Path to save the generated .wav file (use "-" for stdout)
        #[arg(long, short = 'o', default_value = "clone_output.wav")]
        output: String,
    },
    /// Generate speech using built-in CustomVoice named speakers
    Speak {
        #[command(flatten)]
        common: CommonSynthArgs,

        /// Speaker name (or use --list-speakers to inspect available speakers)
        #[arg(long, short = 'S', default_value = "Vivian")]
        speaker: String,

        /// List all available built-in speakers in the CustomVoice model
        #[arg(long)]
        list_speakers: bool,

        /// Optional emotion/style steering prompt (e.g. "Speak happily and energetically")
        #[arg(long, short = 'p', default_value = "")]
        prompt: String,

        /// Path to save the generated .wav file (use "-" for stdout)
        #[arg(long, short = 'o', default_value = "speak_output.wav")]
        output: String,
    },
    /// Pre-extract and save a reusable voice profile (.json or .bin) from reference audio
    Profile {
        /// Path to the reference audio file (.wav, .mp3, .flac, .ogg, .m4a)
        #[arg(long, short = 'a')]
        audio_in: String,

        /// Optional transcript of the reference audio (creates a Full-ICL profile)
        #[arg(long)]
        ref_text: Option<String>,

        /// Optional file containing the transcript of the reference audio
        #[arg(long)]
        ref_file: Option<String>,

        /// Output path for the saved voice profile (e.g. my_voice.json)
        #[arg(long, short = 'o', default_value = "voice_profile.json")]
        output: String,

        /// Optional model preset or GGUF filename
        #[arg(long, short = 'm')]
        model: Option<String>,
    },
    /// Show system info, model presets, or pre-download GGUF models
    Models {
        /// Download models for a specific preset (bf16, q8, q4, 0.6b, fast, tiny)
        #[arg(long, short = 'd')]
        download: Option<String>,
    },
}

struct OwnedOptions {
    _output: CString,
    _model: Option<CString>,
    _lang: Option<CString>,
    _ref_text: Option<CString>,
    _voice_profile: Option<CString>,
    _save_profile: Option<CString>,
    raw: VoiceCliOptions,
}

impl OwnedOptions {
    fn new(
        common: &CommonSynthArgs,
        output: &str,
        ref_text: Option<&str>,
        voice_profile: Option<&str>,
        save_profile: Option<&str>,
    ) -> Self {
        let c_out = CString::new(output).expect("Invalid output path");
        let c_model = common
            .model
            .as_deref()
            .and_then(|s| CString::new(s).ok());
        let c_lang = common
            .lang
            .as_deref()
            .and_then(|s| CString::new(s).ok());
        let c_ref = ref_text.and_then(|s| CString::new(s).ok());
        let c_prof = voice_profile.and_then(|s| CString::new(s).ok());
        let c_save = save_profile.and_then(|s| CString::new(s).ok());

        let raw = VoiceCliOptions {
            output_path: c_out.as_ptr(),
            model_override: c_model.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            language: c_lang.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            speed: common.speed,
            temperature: common.temperature,
            top_p: common.top_p,
            top_k: common.top_k,
            max_tokens: common.max_tokens,
            repetition_penalty: common.repetition_penalty,
            threads: common.threads,
            stream: if common.stream { 1 } else { 0 },
            stream_chunk_sec: common.stream_chunk_sec,
            normalize: if common.normalize { 1 } else { 0 },
            play: if common.play { 1 } else { 0 },
            ref_text: c_ref.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            voice_profile: c_prof.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            save_profile: c_save.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
        };

        Self {
            _output: c_out,
            _model: c_model,
            _lang: c_lang,
            _ref_text: c_ref,
            _voice_profile: c_prof,
            _save_profile: c_save,
            raw,
        }
    }
}

fn read_stdin_string() -> Result<String, String> {
    let mut buffer = String::new();
    std::io::stdin()
        .read_to_string(&mut buffer)
        .map_err(|e| format!("Failed to read from stdin: {}", e))?;
    let trimmed = buffer.trim().to_string();
    if trimmed.is_empty() {
        Err("Standard input (stdin) is empty.".to_string())
    } else {
        Ok(trimmed)
    }
}

#[cfg(test)]
fn resolve_input_text(
    text: Option<String>,
    file: Option<String>,
) -> Result<String, String> {
    resolve_input_text_with_stdin(text, file, false)
}

fn resolve_input_text_with_stdin(
    text: Option<String>,
    file: Option<String>,
    force_stdin: bool,
) -> Result<String, String> {
    if force_stdin {
        if text.is_some() || file.is_some() {
            return Err("Cannot combine --stdin with --text or --file.".to_string());
        }
        return read_stdin_string();
    }

    match (text, file) {
        (Some(t), None) => {
            if t == "-" {
                return read_stdin_string();
            }
            let trimmed = t.trim().to_string();
            if trimmed.is_empty() {
                Err("Provided --text is empty.".to_string())
            } else {
                Ok(trimmed)
            }
        }
        (None, Some(f)) => {
            if f == "-" {
                return read_stdin_string();
            }
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
            if !std::io::stdin().is_terminal() {
                return read_stdin_string();
            }
            Err(
                "Please provide input text using --text \"...\", --file <PATH.txt>, or pipe via stdin."
                    .to_string(),
            )
        }
    }
}

fn resolve_ref_text(ref_text: Option<String>, ref_file: Option<String>) -> Result<Option<String>, String> {
    match (ref_text, ref_file) {
        (Some(t), None) => {
            let trimmed = t.trim().to_string();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed))
            }
        }
        (None, Some(f)) => {
            let path = Path::new(&f);
            if !path.exists() {
                return Err(format!("Reference transcript file '{}' does not exist.", f));
            }
            let content = std::fs::read_to_string(path)
                .map_err(|e| format!("Failed to read reference transcript '{}': {}", f, e))?;
            let trimmed = content.trim().to_string();
            if trimmed.is_empty() {
                Err(format!("Reference transcript file '{}' is empty.", f))
            } else {
                Ok(Some(trimmed))
            }
        }
        (Some(_), Some(_)) => Err("Please specify either --ref-text or --ref-file, not both.".to_string()),
        (None, None) => Ok(None),
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

/// Normalize Unicode typography and optionally strip Markdown formatting for clean TTS reading
fn preprocess_text(input: &str, clean_markdown: bool) -> String {
    let mut s = input
        .replace('\u{00A0}', " ")
        .replace('\u{200B}', "")
        .replace('…', "...");

    if clean_markdown {
        let mut cleaned_lines = Vec::new();
        let mut in_code_block = false;
        for line in s.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("```") {
                in_code_block = !in_code_block;
                continue;
            }
            if in_code_block {
                continue;
            }
            // Strip leading markdown heading markers (#, ##, etc.) and blockquote (>)
            let mut l = trimmed.trim_start_matches('#').trim_start_matches('>').trim();
            if let Some(rest) = l.strip_prefix("- ").or_else(|| l.strip_prefix("* ")) {
                l = rest;
            }
            cleaned_lines.push(l.to_string());
        }
        s = cleaned_lines.join("\n");
        // Strip inline code backticks and bold/italic markers
        s = s.replace("**", "").replace("__", "").replace('`', "");
    }

    // Collapse runs of horizontal whitespace while preserving paragraph breaks
    let mut normalized = String::with_capacity(s.len());
    for (idx, line) in s.lines().enumerate() {
        if idx > 0 {
            normalized.push('\n');
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        normalized.push_str(&words.join(" "));
    }

    normalized.trim().to_string()
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
        "sgt.", "capt.", "lt.", "col.", "vs.", "v.", "e.g.", "i.e.", "etc.", "al.",
        "approx.", "appx.", "dept.", "fig.", "no.", "vol.", "est.", "inc.",
        "corp.", "co.", "ltd.", "st.", "ave.", "rd.", "blvd.", "u.s.", "u.k.",
        "e.u.", "a.m.", "p.m.", "am.", "pm.", "a.d.", "b.c.", "ph.d.", "m.d.",
        "b.a.", "m.a.", "d.c.", "jan.", "feb.", "mar.", "apr.", "jun.", "jul.",
        "aug.", "sep.", "sept.", "oct.", "nov.", "dec.",
    ];

    if KNOWN_ABBREVS.contains(&lower.as_str()) {
        return true;
    }

    // Single-letter initials like "J." or "W."
    let chars: Vec<char> = clean.chars().collect();
    if chars.len() == 2 && chars[0].is_alphabetic() && chars[1] == '.' {
        return true;
    }

    // Multi-period acronyms like "U.S.A."
    if chars.len() >= 4 && chars.ends_with(&['.']) && chars.iter().filter(|&&c| c == '.').count() >= 2 {
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
            // Ignore ellipsis (e.g. '...')
            if c == '.' && i + 1 < len && chars[i + 1] == '.' {
                while i + 1 < len && chars[i + 1] == '.' {
                    i += 1;
                }
                i += 1;
                continue;
            }

            // Ignore decimal numbers (e.g. "3.14")
            if c == '.' && i > 0 && i + 1 < len && chars[i - 1].is_ascii_digit() && chars[i + 1].is_ascii_digit() {
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

            // Check if boundary is followed by whitespace or end of string (or CJK punctuation)
            let is_cjk_punct = c == '。' || c == '！' || c == '？';
            if is_cjk_punct || end_punct + 1 == len || chars[end_punct + 1].is_whitespace() {
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
        "; ",
        ": ",
        " — ",
        " – ",
        " -- ",
        ", and ",
        ", but ",
        ", however, ",
        ", although ",
        ", because ",
        ", while ",
        ", which ",
        ", whereas ",
        ", so ",
        ", or ",
        ", yet ",
        ", ",
        "、",
        "，",
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
        remaining = remaining
            .chars()
            .skip(break_point)
            .collect::<String>()
            .trim()
            .to_string();
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

    let target_chars = target_chars.max(80);

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

        // Flush at paragraph boundaries if substantial
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

fn should_use_chunking(common: &CommonSynthArgs, text: &str) -> bool {
    if common.no_chunk {
        return false;
    }
    if common.chunk {
        return true;
    }
    // Smart auto-chunking: automatically chunk texts > 800 characters to prevent token truncation
    text.chars().count() > 800
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

    let exit_code = match cli.command {
        Commands::Style {
            common,
            prompt,
            output,
        } => {
            if let Err(err) = validate_speed(common.speed) {
                eprintln!("[-] Error: {}", err);
                std::process::exit(1);
            }

            let raw_text = match resolve_input_text_with_stdin(
                common.text.clone(),
                common.file.clone(),
                common.stdin,
            ) {
                Ok(t) => t,
                Err(err) => {
                    eprintln!("[-] Error: {}", err);
                    std::process::exit(1);
                }
            };
            let input_text = preprocess_text(&raw_text, common.clean_markdown);
            let opts = OwnedOptions::new(&common, &output, None, None, None);
            let c_prompt = CString::new(prompt).expect("Invalid prompt string");

            if should_use_chunking(&common, &input_text) {
                let chunks = split_into_chunks(&input_text, common.chunk_size);
                if chunks.is_empty() {
                    eprintln!("[-] Error: No valid text to synthesize.");
                    std::process::exit(1);
                }
                if chunks.len() == 1 {
                    let c_text = CString::new(chunks[0].as_str()).expect("Invalid text");
                    unsafe { c_generate_style_ex(c_text.as_ptr(), c_prompt.as_ptr(), &opts.raw) }
                } else {
                    eprintln!(
                        "[*] Auto-Chunk Mode enabled: Split text into {} natural chunks (max {} chars/chunk).",
                        chunks.len(),
                        common.chunk_size
                    );
                    let c_chunks: Vec<CString> = chunks
                        .iter()
                        .map(|s| CString::new(s.as_str()).expect("Invalid chunk"))
                        .collect();
                    let c_ptrs: Vec<*const c_char> = c_chunks.iter().map(|c| c.as_ptr()).collect();
                    unsafe {
                        c_generate_style_chunked_ex(
                            c_ptrs.as_ptr(),
                            c_ptrs.len() as i32,
                            c_prompt.as_ptr(),
                            &opts.raw,
                        )
                    }
                }
            } else {
                let c_text = CString::new(input_text).expect("Invalid text");
                unsafe { c_generate_style_ex(c_text.as_ptr(), c_prompt.as_ptr(), &opts.raw) }
            }
        }
        Commands::Clone {
            common,
            audio_in,
            ref_text,
            ref_file,
            voice_profile,
            save_profile,
            output,
        } => {
            if let Err(err) = validate_speed(common.speed) {
                eprintln!("[-] Error: {}", err);
                std::process::exit(1);
            }

            if audio_in.is_none() && voice_profile.is_none() {
                eprintln!(
                    "[-] Error: Please provide either --audio-in <AUDIO_FILE> or --voice-profile <PROFILE_FILE>."
                );
                std::process::exit(1);
            }

            if let Some(ref audio) = audio_in {
                if !Path::new(audio).exists() {
                    eprintln!("[-] Error: Reference audio file '{}' does not exist.", audio);
                    std::process::exit(1);
                }
            }

            if let Some(ref prof) = voice_profile {
                if !Path::new(prof).exists() {
                    eprintln!("[-] Error: Voice profile file '{}' does not exist.", prof);
                    std::process::exit(1);
                }
            }

            let resolved_ref_text = match resolve_ref_text(ref_text, ref_file) {
                Ok(rt) => rt,
                Err(err) => {
                    eprintln!("[-] Error: {}", err);
                    std::process::exit(1);
                }
            };

            let raw_text = match resolve_input_text_with_stdin(
                common.text.clone(),
                common.file.clone(),
                common.stdin,
            ) {
                Ok(t) => t,
                Err(err) => {
                    eprintln!("[-] Error: {}", err);
                    std::process::exit(1);
                }
            };
            let input_text = preprocess_text(&raw_text, common.clean_markdown);

            let opts = OwnedOptions::new(
                &common,
                &output,
                resolved_ref_text.as_deref(),
                voice_profile.as_deref(),
                save_profile.as_deref(),
            );
            let c_audio = CString::new(audio_in.unwrap_or_default()).expect("Invalid audio path");

            if should_use_chunking(&common, &input_text) {
                let chunks = split_into_chunks(&input_text, common.chunk_size);
                if chunks.is_empty() {
                    eprintln!("[-] Error: No valid text to synthesize.");
                    std::process::exit(1);
                }
                if chunks.len() == 1 {
                    let c_text = CString::new(chunks[0].as_str()).expect("Invalid text");
                    unsafe { c_generate_clone_ex(c_text.as_ptr(), c_audio.as_ptr(), &opts.raw) }
                } else {
                    eprintln!(
                        "[*] Auto-Chunk Mode enabled: Split text into {} natural chunks (max {} chars/chunk).",
                        chunks.len(),
                        common.chunk_size
                    );
                    let c_chunks: Vec<CString> = chunks
                        .iter()
                        .map(|s| CString::new(s.as_str()).expect("Invalid chunk"))
                        .collect();
                    let c_ptrs: Vec<*const c_char> = c_chunks.iter().map(|c| c.as_ptr()).collect();
                    unsafe {
                        c_generate_clone_chunked_ex(
                            c_ptrs.as_ptr(),
                            c_ptrs.len() as i32,
                            c_audio.as_ptr(),
                            &opts.raw,
                        )
                    }
                }
            } else {
                let c_text = CString::new(input_text).expect("Invalid text");
                unsafe { c_generate_clone_ex(c_text.as_ptr(), c_audio.as_ptr(), &opts.raw) }
            }
        }
        Commands::Speak {
            common,
            speaker,
            list_speakers,
            prompt,
            output,
        } => {
            let opts = OwnedOptions::new(&common, &output, None, None, None);
            let c_prompt = CString::new(prompt).expect("Invalid prompt");

            if list_speakers {
                let c_text = CString::new("").unwrap();
                let c_spk = CString::new("list").unwrap();
                unsafe { c_generate_speak_ex(c_text.as_ptr(), c_spk.as_ptr(), c_prompt.as_ptr(), &opts.raw) }
            } else {
                if let Err(err) = validate_speed(common.speed) {
                    eprintln!("[-] Error: {}", err);
                    std::process::exit(1);
                }

                let raw_text = match resolve_input_text_with_stdin(
                    common.text.clone(),
                    common.file.clone(),
                    common.stdin,
                ) {
                    Ok(t) => t,
                    Err(err) => {
                        eprintln!("[-] Error: {}", err);
                        std::process::exit(1);
                    }
                };
                let input_text = preprocess_text(&raw_text, common.clean_markdown);
                let c_spk = CString::new(speaker).expect("Invalid speaker");

                if should_use_chunking(&common, &input_text) {
                    let chunks = split_into_chunks(&input_text, common.chunk_size);
                    if chunks.len() <= 1 {
                        let c_text = CString::new(input_text).expect("Invalid text");
                        unsafe {
                            c_generate_speak_ex(
                                c_text.as_ptr(),
                                c_spk.as_ptr(),
                                c_prompt.as_ptr(),
                                &opts.raw,
                            )
                        }
                    } else {
                        eprintln!(
                            "[*] Auto-Chunk Mode enabled: Split text into {} natural chunks.",
                            chunks.len()
                        );
                        let c_chunks: Vec<CString> = chunks
                            .iter()
                            .map(|s| CString::new(s.as_str()).expect("Invalid chunk"))
                            .collect();
                        let c_ptrs: Vec<*const c_char> =
                            c_chunks.iter().map(|c| c.as_ptr()).collect();
                        unsafe {
                            c_generate_speak_chunked_ex(
                                c_ptrs.as_ptr(),
                                c_ptrs.len() as i32,
                                c_spk.as_ptr(),
                                c_prompt.as_ptr(),
                                &opts.raw,
                            )
                        }
                    }
                } else {
                    let c_text = CString::new(input_text).expect("Invalid text");
                    unsafe {
                        c_generate_speak_ex(
                            c_text.as_ptr(),
                            c_spk.as_ptr(),
                            c_prompt.as_ptr(),
                            &opts.raw,
                        )
                    }
                }
            }
        }
        Commands::Profile {
            audio_in,
            ref_text,
            ref_file,
            output,
            model,
        } => {
            if !Path::new(&audio_in).exists() {
                eprintln!("[-] Error: Reference audio file '{}' does not exist.", audio_in);
                std::process::exit(1);
            }
            let resolved_ref = match resolve_ref_text(ref_text, ref_file) {
                Ok(rt) => rt,
                Err(err) => {
                    eprintln!("[-] Error: {}", err);
                    std::process::exit(1);
                }
            };
            let c_audio = CString::new(audio_in).expect("Invalid audio path");
            let c_out = CString::new(output).expect("Invalid output path");
            let c_ref = resolved_ref.and_then(|s| CString::new(s).ok());
            let c_model = model.and_then(|s| CString::new(s).ok());

            unsafe {
                c_extract_profile(
                    c_audio.as_ptr(),
                    c_ref.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                    c_out.as_ptr(),
                    c_model.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
                )
            }
        }
        Commands::Models { download } => {
            unsafe { c_print_system_info() };
            if let Some(preset) = download {
                let c_preset = CString::new(preset).expect("Invalid preset");
                unsafe { c_download_models(c_preset.as_ptr()) }
            } else {
                0
            }
        }
    };

    if exit_code != 0 {
        std::process::exit(exit_code);
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
        let chunks = split_into_chunks(text, 80);
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
    fn test_extract_sentences_decimals_and_ellipsis() {
        let text = "Version 3.14 was released... and it runs 2.5x faster! Really?";
        let sents = extract_sentences(text);
        assert_eq!(sents.len(), 2);
        assert_eq!(sents[0], "Version 3.14 was released... and it runs 2.5x faster!");
        assert_eq!(sents[1], "Really?");
    }

    #[test]
    fn test_preprocess_text_markdown() {
        let md = "# Welcome\nThis is **bold** and `code` with an ellipsis…\n```rust\nlet x = 1;\n```\n- Bullet item";
        let cleaned = preprocess_text(md, true);
        assert!(!cleaned.contains('#'));
        assert!(!cleaned.contains("**"));
        assert!(!cleaned.contains("let x = 1"));
        assert!(cleaned.contains("..."));
        assert!(cleaned.contains("Bullet item"));
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
        assert_eq!(
            sents[0],
            "Dr. Smith met with Mr. Brown at 5 p.m. to discuss J. K. Rowling books."
        );
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
