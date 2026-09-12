use std::path::PathBuf;
use std::process::Command;

fn main() {
    let project_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let ext_dir = PathBuf::from(&project_dir).join("external").join("qwen3-tts");
    let mut build_dir = ext_dir.join("build-static");

    // If build-static doesn't exist, check build or tmp
    if !build_dir.join("libqwen3_tts.a").exists() {
        if ext_dir.join("build").join("libqwen3_tts.a").exists() {
            build_dir = ext_dir.join("build");
        } else {
            if !ext_dir.join("CMakeLists.txt").exists() {
                println!("cargo:warning=Cloning qwen3-tts.cpp into external/qwen3-tts...");
                let status = Command::new("git")
                    .args([
                        "clone",
                        "--depth",
                        "1",
                        "--recursive",
                        "https://github.com/Danmoreng/qwen3-tts.cpp.git",
                        ext_dir.to_str().unwrap(),
                    ])
                    .status()
                    .expect("Failed to clone qwen3-tts.cpp");
                if !status.success() {
                    panic!("git clone failed");
                }
            }

            println!("cargo:warning=Building static GGML engine with cmake...");
            let b_dir = ext_dir.join("build-static");
            let _ = std::fs::create_dir_all(&b_dir);
            let status = Command::new("cmake")
                .args([
                    "-S",
                    ext_dir.to_str().unwrap(),
                    "-B",
                    b_dir.to_str().unwrap(),
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_SHARED_LIBS=OFF",
                ])
                .status()
                .expect("Failed to configure cmake");
            if !status.success() {
                panic!("cmake configure failed");
            }

            let status = Command::new("cmake")
                .args(["--build", b_dir.to_str().unwrap(), "--config", "Release", "-j4"])
                .status()
                .expect("Failed to build cmake");
            if !status.success() {
                panic!("cmake build failed");
            }
            build_dir = b_dir;
        }
    }

    // Compile src/model.c with includes for qwen3_tts_c.h
    let mut c_builder = cc::Build::new();
    c_builder
        .file("src/model.c")
        .include(ext_dir.join("src"))
        .include(ext_dir.join("ggml").join("include"));
    c_builder.compile("model");

    // Link all static GGML and Qwen3-TTS libraries directly into voicecli
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("ggml").join("src").display()
    );

    println!("cargo:rustc-link-lib=static=qwen3_tts");
    println!("cargo:rustc-link-lib=static=text_tokenizer");
    println!("cargo:rustc-link-lib=static=tts_transformer");
    println!("cargo:rustc-link-lib=static=speech_tokenizer_encoder");
    println!("cargo:rustc-link-lib=static=audio_tokenizer_decoder");
    println!("cargo:rustc-link-lib=static=audio_tokenizer_encoder");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");

    // Link C++ runtime and system dependencies
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");

    println!("cargo:rerun-if-changed=src/model.c");
    println!("cargo:rerun-if-changed=build.rs");
}
