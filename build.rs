use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn ensure_qwen3_tts_source(ext_dir: &Path, native_opt: bool) {
    let cmake_file = ext_dir.join("CMakeLists.txt");
    if !cmake_file.exists() {
        println!("cargo:warning=Cloning qwen3-tts.cpp into external/qwen3-tts...");
        if let Some(parent) = ext_dir.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
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

    // Upstream qwen3-tts.cpp hardcodes `-march=native` in CMAKE_CXX_FLAGS_RELEASE,
    // which leaks host-specific AVX-512 instructions into global C++ `.init_array`
    // constructors and causes SIGILL ("Illegal instruction") on other x86_64 CPUs.
    if !native_opt {
        if let Ok(content) = std::fs::read_to_string(&cmake_file) {
            if content.contains("-march=native") {
                let sanitized = content
                    .replace("-O3 -march=native", "-O3")
                    .replace("-march=native", "");
                std::fs::write(&cmake_file, sanitized)
                    .expect("Failed to sanitize -march=native from external/qwen3-tts/CMakeLists.txt");
            }
        }
    }
}

fn run_cmake_configure_and_build(
    src_dir: &Path,
    build_dir: &Path,
    target: &str,
    n_jobs: &str,
    extra_cmake_args: &[&str],
) {
    let _ = std::fs::create_dir_all(build_dir);
    let mut cfg = Command::new("cmake");
    cfg.args([
        "-S",
        src_dir.to_str().unwrap(),
        "-B",
        build_dir.to_str().unwrap(),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
    ]);
    cfg.args(extra_cmake_args);

    let status = cfg.status().expect("Failed to configure cmake");
    if !status.success() {
        panic!("cmake configure failed in {}", build_dir.display());
    }

    let status = Command::new("cmake")
        .args([
            "--build",
            build_dir.to_str().unwrap(),
            "--config",
            "Release",
            "--target",
            target,
            "-j",
            n_jobs,
        ])
        .status()
        .expect("Failed to build cmake target");
    if !status.success() {
        panic!("cmake build failed for target '{}' in {}", target, build_dir.display());
    }
}

fn prefix_ggml_cpu_archive(src_archive: &Path, dst_archive: &Path, prefix: &str, work_dir: &Path) {
    // Safety verification: ensure no static C++ constructors (.init_array / .ctors)
    // exist inside libggml-cpu.a that could execute before runtime CPUID dispatch.
    if let Ok(readelf_out) = Command::new("readelf")
        .env("LC_ALL", "C")
        .args(["-W", "-S", src_archive.to_str().unwrap()])
        .output()
    {
        if readelf_out.status.success() {
            let text = String::from_utf8_lossy(&readelf_out.stdout);
            for line in text.lines() {
                if line.contains(".init_array") || line.contains(".ctors") {
                    panic!(
                        "Unexpected static constructor section in {}: {}",
                        src_archive.display(),
                        line.trim()
                    );
                }
            }
        }
    }

    let nm_out = Command::new("nm")
        .env("LC_ALL", "C")
        .args(["--defined-only", "--extern-only", src_archive.to_str().unwrap()])
        .output()
        .expect("Failed to run `nm` on libggml-cpu.a (ensure binutils is installed)");
    if !nm_out.status.success() {
        panic!("`nm` failed on {}", src_archive.display());
    }

    let stdout = String::from_utf8_lossy(&nm_out.stdout);
    let mut symbols = BTreeSet::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() == 3 && parts[0].chars().all(|c| c.is_ascii_hexdigit()) {
            let sym = parts[2];
            if !sym.is_empty() && !sym.starts_with("DW.ref.") {
                symbols.insert(sym.to_string());
            }
        }
    }

    // Also rename STB_LOCAL COMDAT group signature symbols (e.g., C5Ev / D5Ev)
    // reported by `readelf -g` so the linker never discards COMDAT section groups
    // from secondary SIMD variants while keeping their renamed weak aliases undefined.
    let group_out = Command::new("readelf")
        .env("LC_ALL", "C")
        .args(["-W", "-g", src_archive.to_str().unwrap()])
        .output()
        .expect("Failed to run `readelf -g` on libggml-cpu.a");
    if !group_out.status.success() {
        panic!("`readelf -g` failed on {}", src_archive.display());
    }
    let group_text = String::from_utf8_lossy(&group_out.stdout);
    for line in group_text.lines() {
        if line.contains("COMDAT group section") {
            if let Some(after_group) = line.split("`.group'").nth(1) {
                if let (Some(lbr), Some(rbr)) = (after_group.find('['), after_group.find(']')) {
                    if rbr > lbr + 1 {
                        let sig = after_group[lbr + 1..rbr].trim();
                        if !sig.is_empty() && !sig.starts_with("DW.ref.") {
                            symbols.insert(sig.to_string());
                        }
                    }
                }
            }
        }
    }

    if !symbols.contains("ggml_backend_cpu_reg") {
        panic!(
            "Expected `ggml_backend_cpu_reg` in {}, found {} symbols",
            src_archive.display(),
            symbols.len()
        );
    }

    let map_path = work_dir.join(format!("redef_{}.txt", prefix.trim_end_matches('_')));
    let mut map_contents = String::with_capacity(symbols.len() * 64);
    for sym in &symbols {
        map_contents.push_str(sym);
        map_contents.push(' ');
        map_contents.push_str(prefix);
        map_contents.push_str(sym);
        map_contents.push('\n');
    }
    std::fs::write(&map_path, map_contents).expect("Failed to write symbol redefinition map");

    let status = Command::new("objcopy")
        .args([
            &format!("--redefine-syms={}", map_path.display()),
            src_archive.to_str().unwrap(),
            dst_archive.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run `objcopy` on libggml-cpu.a");
    if !status.success() {
        panic!("`objcopy --redefine-syms` failed on {}", src_archive.display());
    }

    let status = Command::new("ranlib")
        .arg(dst_archive.to_str().unwrap())
        .status()
        .expect("Failed to run `ranlib`");
    if !status.success() {
        panic!("`ranlib` failed on {}", dst_archive.display());
    }
}

fn main() {
    let project_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let native_opt = std::env::var("VOICECLI_NATIVE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let multi_isa_x86 = target_arch == "x86_64" && target_os == "linux" && !native_opt;

    let ext_dir = PathBuf::from(&project_dir).join("external").join("qwen3-tts");
    ensure_qwen3_tts_source(&ext_dir, native_opt);

    let n_jobs = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".to_string());

    let (base_lib_dir, ggml_lib_dir, cpu_variants_dir) = if multi_isa_x86 {
        let portable_root = ext_dir.join("build-portable");
        let base_dir = portable_root.join("base");
        let libs_dir = portable_root.join("libs");
        let _ = std::fs::create_dir_all(&libs_dir);

        let x64_lib = libs_dir.join("libggml-cpu-x64.a");
        let sse42_lib = libs_dir.join("libggml-cpu-sse42.a");
        let avx2_lib = libs_dir.join("libggml-cpu-avx2.a");
        let avx512_lib = libs_dir.join("libggml-cpu-avx512.a");

        if !base_dir.join("libqwen3_tts.a").exists() || !x64_lib.exists() {
            println!("cargo:warning=Building portable baseline x86-64 Qwen3-TTS + GGML engine...");
            let base_flags = [
                "-DGGML_NATIVE=OFF",
                "-DGGML_SSE42=OFF",
                "-DGGML_AVX=OFF",
                "-DGGML_AVX_VNNI=OFF",
                "-DGGML_AVX2=OFF",
                "-DGGML_BMI2=OFF",
                "-DGGML_FMA=OFF",
                "-DGGML_F16C=OFF",
                "-DGGML_AVX512=OFF",
                "-DGGML_AVX512_VBMI=OFF",
                "-DGGML_AVX512_VNNI=OFF",
                "-DGGML_AVX512_BF16=OFF",
                "-DGGML_AMX_TILE=OFF",
                "-DGGML_AMX_INT8=OFF",
                "-DGGML_AMX_BF16=OFF",
                "-DCMAKE_C_FLAGS=-march=x86-64 -mtune=generic",
                "-DCMAKE_CXX_FLAGS=-march=x86-64 -mtune=generic",
            ];
            run_cmake_configure_and_build(&ext_dir, &base_dir, "qwen3_tts", &n_jobs, &base_flags);
            prefix_ggml_cpu_archive(
                &base_dir.join("ggml").join("src").join("libggml-cpu.a"),
                &x64_lib,
                "voicecli_x64_",
                &libs_dir,
            );
        }

        if !sse42_lib.exists() {
            println!("cargo:warning=Building GGML CPU backend variant: SSE4.2 (x86-64-v2)...");
            let d = portable_root.join("sse42");
            let flags = [
                "-DGGML_NATIVE=OFF",
                "-DGGML_SSE42=ON",
                "-DGGML_AVX=OFF",
                "-DGGML_AVX_VNNI=OFF",
                "-DGGML_AVX2=OFF",
                "-DGGML_BMI2=OFF",
                "-DGGML_FMA=OFF",
                "-DGGML_F16C=OFF",
                "-DGGML_AVX512=OFF",
                "-DGGML_AVX512_VBMI=OFF",
                "-DGGML_AVX512_VNNI=OFF",
                "-DGGML_AVX512_BF16=OFF",
                "-DGGML_AMX_TILE=OFF",
                "-DGGML_AMX_INT8=OFF",
                "-DGGML_AMX_BF16=OFF",
                "-DCMAKE_C_FLAGS=-march=x86-64 -msse4.2 -mpopcnt -mtune=generic",
                "-DCMAKE_CXX_FLAGS=-march=x86-64 -msse4.2 -mpopcnt -mtune=generic",
            ];
            run_cmake_configure_and_build(&ext_dir, &d, "ggml-cpu", &n_jobs, &flags);
            prefix_ggml_cpu_archive(
                &d.join("ggml").join("src").join("libggml-cpu.a"),
                &sse42_lib,
                "voicecli_sse42_",
                &libs_dir,
            );
        }

        if !avx2_lib.exists() {
            println!("cargo:warning=Building GGML CPU backend variant: AVX2 + FMA + F16C (x86-64-v3)...");
            let d = portable_root.join("avx2");
            let flags = [
                "-DGGML_NATIVE=OFF",
                "-DGGML_SSE42=ON",
                "-DGGML_AVX=ON",
                "-DGGML_AVX_VNNI=OFF",
                "-DGGML_AVX2=ON",
                "-DGGML_BMI2=ON",
                "-DGGML_FMA=ON",
                "-DGGML_F16C=ON",
                "-DGGML_AVX512=OFF",
                "-DGGML_AVX512_VBMI=OFF",
                "-DGGML_AVX512_VNNI=OFF",
                "-DGGML_AVX512_BF16=OFF",
                "-DGGML_AMX_TILE=OFF",
                "-DGGML_AMX_INT8=OFF",
                "-DGGML_AMX_BF16=OFF",
                "-DCMAKE_C_FLAGS=-march=x86-64 -msse4.2 -mpopcnt -mavx -mf16c -mfma -mbmi -mbmi2 -mavx2 -mtune=generic",
                "-DCMAKE_CXX_FLAGS=-march=x86-64 -msse4.2 -mpopcnt -mavx -mf16c -mfma -mbmi -mbmi2 -mavx2 -mtune=generic",
            ];
            run_cmake_configure_and_build(&ext_dir, &d, "ggml-cpu", &n_jobs, &flags);
            prefix_ggml_cpu_archive(
                &d.join("ggml").join("src").join("libggml-cpu.a"),
                &avx2_lib,
                "voicecli_avx2_",
                &libs_dir,
            );
        }

        if !avx512_lib.exists() {
            println!("cargo:warning=Building GGML CPU backend variant: AVX-512 (x86-64-v4)...");
            let d = portable_root.join("avx512");
            let flags = [
                "-DGGML_NATIVE=OFF",
                "-DGGML_SSE42=ON",
                "-DGGML_AVX=ON",
                "-DGGML_AVX_VNNI=OFF",
                "-DGGML_AVX2=ON",
                "-DGGML_BMI2=ON",
                "-DGGML_FMA=ON",
                "-DGGML_F16C=ON",
                "-DGGML_AVX512=ON",
                "-DGGML_AVX512_VBMI=OFF",
                "-DGGML_AVX512_VNNI=OFF",
                "-DGGML_AVX512_BF16=OFF",
                "-DGGML_AMX_TILE=OFF",
                "-DGGML_AMX_INT8=OFF",
                "-DGGML_AMX_BF16=OFF",
                "-DCMAKE_C_FLAGS=-march=x86-64 -msse4.2 -mpopcnt -mavx -mf16c -mfma -mbmi -mbmi2 -mavx2 -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mtune=generic",
                "-DCMAKE_CXX_FLAGS=-march=x86-64 -msse4.2 -mpopcnt -mavx -mf16c -mfma -mbmi -mbmi2 -mavx2 -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mtune=generic",
            ];
            run_cmake_configure_and_build(&ext_dir, &d, "ggml-cpu", &n_jobs, &flags);
            prefix_ggml_cpu_archive(
                &d.join("ggml").join("src").join("libggml-cpu.a"),
                &avx512_lib,
                "voicecli_avx512_",
                &libs_dir,
            );
        }

        let ggml_src = base_dir.join("ggml").join("src");
        (base_dir, ggml_src, Some(libs_dir))
    } else {
        let dir_name = if native_opt { "build-native" } else { "build-static-portable" };
        let b_dir = ext_dir.join(dir_name);
        if !b_dir.join("libqwen3_tts.a").exists() {
            println!("cargo:warning=Building static GGML engine with cmake ({})...", dir_name);
            let mut extra = Vec::new();
            if !native_opt {
                extra.push("-DGGML_NATIVE=OFF");
            }
            run_cmake_configure_and_build(&ext_dir, &b_dir, "qwen3_tts", &n_jobs, &extra);
        }
        let ggml_src = b_dir.join("ggml").join("src");
        (b_dir, ggml_src, None)
    };

    // Compile src/model.c with portable architecture flags by default
    let mut c_builder = cc::Build::new();
    c_builder
        .file("src/model.c")
        .include(ext_dir.join("src"))
        .include(ext_dir.join("ggml").join("include"))
        .opt_level(3);

    let compiler = c_builder.get_compiler();
    if compiler.is_like_gnu() || compiler.is_like_clang() {
        c_builder.flag_if_supported("-O3");
        c_builder.flag_if_supported("-funroll-loops");
        if native_opt {
            c_builder.flag_if_supported("-march=native");
        } else if target_arch == "x86_64" {
            c_builder.flag_if_supported("-march=x86-64");
            c_builder.flag_if_supported("-mtune=generic");
        }
    }
    c_builder.compile("model");

    // Compile src/cpu_dispatch.c strictly at baseline ISA with LTO disabled
    // so runtime CPUID + XGETBV dispatch can never execute unsupported SIMD instructions.
    let mut dispatch_builder = cc::Build::new();
    dispatch_builder
        .cargo_metadata(false)
        .file("src/cpu_dispatch.c")
        .include(ext_dir.join("ggml").join("include"))
        .opt_level(2);

    if multi_isa_x86 {
        dispatch_builder.define("VOICECLI_MULTI_ISA", "1");
    }
    let disp_compiler = dispatch_builder.get_compiler();
    if disp_compiler.is_like_gnu() || disp_compiler.is_like_clang() {
        dispatch_builder.flag_if_supported("-fno-lto");
        if target_arch == "x86_64" {
            dispatch_builder.flag_if_supported("-march=x86-64");
            dispatch_builder.flag_if_supported("-mtune=generic");
        }
    }
    dispatch_builder.compile("cpu_dispatch");

    // Link search paths
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={}", base_lib_dir.display());
    println!("cargo:rustc-link-search=native={}", ggml_lib_dir.display());
    if let Some(ref vdir) = cpu_variants_dir {
        println!("cargo:rustc-link-search=native={}", vdir.display());
    }

    // Link all static Qwen3-TTS and GGML libraries in strict topological dependency order
    println!("cargo:rustc-link-lib=static=qwen3_tts");
    println!("cargo:rustc-link-lib=static=text_tokenizer");
    println!("cargo:rustc-link-lib=static=tts_transformer");
    println!("cargo:rustc-link-lib=static=speech_tokenizer_encoder");
    println!("cargo:rustc-link-lib=static=audio_tokenizer_decoder");
    println!("cargo:rustc-link-lib=static=audio_tokenizer_encoder");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=cpu_dispatch");

    if cpu_variants_dir.is_some() {
        println!("cargo:rustc-link-lib=static=ggml-cpu-avx512");
        println!("cargo:rustc-link-lib=static=ggml-cpu-avx2");
        println!("cargo:rustc-link-lib=static=ggml-cpu-sse42");
        println!("cargo:rustc-link-lib=static=ggml-cpu-x64");
    } else {
        println!("cargo:rustc-link-lib=static=ggml-cpu");
    }
    println!("cargo:rustc-link-lib=static=ggml-base");

    // Link C++ runtime and system dependencies
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");

    println!("cargo:rerun-if-changed=src/model.c");
    println!("cargo:rerun-if-changed=src/cpu_dispatch.c");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VOICECLI_NATIVE");
}
