use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

const QWEN3_TTS_PINNED_COMMIT: &str = "16bb5afcd06311031c72a8488f8d59660dc2fb46";

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
                "https://github.com/Danmoreng/qwen3-tts.cpp.git",
                ext_dir.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to clone qwen3-tts.cpp");
        if !status.success() {
            panic!("git clone failed");
        }

        let checkout_status = Command::new("git")
            .current_dir(ext_dir)
            .args(["checkout", QWEN3_TTS_PINNED_COMMIT])
            .status()
            .expect("Failed to checkout pinned qwen3-tts.cpp commit");
        if !checkout_status.success() {
            panic!("git checkout {} failed", QWEN3_TTS_PINNED_COMMIT);
        }

        let sub_status = Command::new("git")
            .current_dir(ext_dir)
            .args(["submodule", "update", "--init", "--recursive", "--depth", "1"])
            .status()
            .expect("Failed to initialize qwen3-tts.cpp submodules");
        if !sub_status.success() {
            let fallback = Command::new("git")
                .current_dir(ext_dir)
                .args(["submodule", "update", "--init", "--recursive"])
                .status()
                .expect("Failed to initialize qwen3-tts.cpp submodules");
            if !fallback.success() {
                panic!("git submodule update failed");
            }
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

fn sync_engine_overlay_dir(src_dir: &Path, dst_dir: &Path) {
    let entries = match std::fs::read_dir(src_dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    println!("cargo:rerun-if-changed={}", src_dir.display());
    for entry in entries.flatten() {
        let src_path = entry.path();
        let file_name = entry.file_name();
        let dst_path = dst_dir.join(file_name);
        if src_path.is_dir() {
            sync_engine_overlay_dir(&src_path, &dst_path);
        } else if src_path.is_file() {
            println!("cargo:rerun-if-changed={}", src_path.display());
            let new_bytes = std::fs::read(&src_path)
                .unwrap_or_else(|e| panic!("Failed to read {}: {}", src_path.display(), e));
            let needs_write = match std::fs::read(&dst_path) {
                Ok(old_bytes) => old_bytes != new_bytes,
                Err(_) => true,
            };
            if needs_write {
                if let Some(parent) = dst_path.parent() {
                    std::fs::create_dir_all(parent)
                        .unwrap_or_else(|e| panic!("Failed to mkdir {}: {}", parent.display(), e));
                }
                std::fs::write(&dst_path, &new_bytes)
                    .unwrap_or_else(|e| panic!("Failed to write {}: {}", dst_path.display(), e));
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
    let marker_path = build_dir.join("voicecli_cmake_args.txt");
    let expected_marker = format!("target={}\nargs={}\n", target, extra_cmake_args.join("\n"));
    let cache_exists = build_dir.join("CMakeCache.txt").exists();
    let marker_matches = std::fs::read_to_string(&marker_path)
        .map(|s| s == expected_marker)
        .unwrap_or(false);

    if !cache_exists || !marker_matches {
        // Remove stale CMakeCache.txt when flags change so cached option() values update cleanly.
        let _ = std::fs::remove_file(build_dir.join("CMakeCache.txt"));
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
        std::fs::write(&marker_path, expected_marker)
            .expect("Failed to write voicecli_cmake_args.txt");
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

fn archive_needs_prefix(src_archive: &Path, dst_archive: &Path) -> bool {
    let dst_meta = match std::fs::metadata(dst_archive) {
        Ok(m) => m,
        Err(_) => return true,
    };
    let src_meta = match std::fs::metadata(src_archive) {
        Ok(m) => m,
        Err(_) => return true,
    };
    match (src_meta.modified(), dst_meta.modified()) {
        (Ok(src_t), Ok(dst_t)) => src_t > dst_t,
        _ => true,
    }
}

fn prefix_ggml_cpu_archive(src_archive: &Path, dst_archive: &Path, prefix: &str, work_dir: &Path) {
    if !archive_needs_prefix(src_archive, dst_archive) {
        return;
    }

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

fn compiler_supports_flag(compiler: &cc::Tool, flag: &str) -> bool {
    Command::new(compiler.path())
        .args([flag, "-x", "c", "-", "-c", "-o", "/dev/null"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn main() {
    let project_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let native_opt = std::env::var("VOICECLI_NATIVE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let multi_isa_x86 = target_arch == "x86_64" && target_os == "linux" && !native_opt;

    let ext_dir = project_dir.join("external").join("qwen3-tts");
    ensure_qwen3_tts_source(&ext_dir, native_opt);

    let engine_overlay_dir = project_dir.join("engine");
    println!("cargo:rerun-if-changed={}", engine_overlay_dir.display());
    if engine_overlay_dir.exists() {
        sync_engine_overlay_dir(&engine_overlay_dir, &ext_dir);
    }

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
        let avx2_vnni_lib = libs_dir.join("libggml-cpu-avx2-vnni.a");
        let avx512_lib = libs_dir.join("libggml-cpu-avx512.a");
        let avx512_vnni_lib = libs_dir.join("libggml-cpu-avx512-vnni.a");
        let avx512_bf16_lib = libs_dir.join("libggml-cpu-avx512-bf16.a");

        {
            let base_flags = [
                "-DGGML_NATIVE=OFF",
                "-DGGML_OPENMP=OFF",
                "-DGGML_LLAMAFILE=ON",
                "-DGGML_CPU_REPACK=ON",
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

        {
            let d = portable_root.join("sse42");
            let flags = [
                "-DGGML_NATIVE=OFF",
                "-DGGML_OPENMP=OFF",
                "-DGGML_LLAMAFILE=ON",
                "-DGGML_CPU_REPACK=ON",
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

        {
            let d = portable_root.join("avx2");
            let flags = [
                "-DGGML_NATIVE=OFF",
                "-DGGML_OPENMP=OFF",
                "-DGGML_LLAMAFILE=ON",
                "-DGGML_CPU_REPACK=ON",
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

        {
            let d = portable_root.join("avx2_vnni");
            let flags = [
                "-DGGML_NATIVE=OFF",
                "-DGGML_OPENMP=OFF",
                "-DGGML_LLAMAFILE=ON",
                "-DGGML_CPU_REPACK=ON",
                "-DGGML_SSE42=ON",
                "-DGGML_AVX=ON",
                "-DGGML_AVX_VNNI=ON",
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
                "-DCMAKE_C_FLAGS=-march=x86-64 -msse4.2 -mpopcnt -mavx -mf16c -mfma -mbmi -mbmi2 -mavx2 -mavxvnni -mtune=generic",
                "-DCMAKE_CXX_FLAGS=-march=x86-64 -msse4.2 -mpopcnt -mavx -mf16c -mfma -mbmi -mbmi2 -mavx2 -mavxvnni -mtune=generic",
            ];
            run_cmake_configure_and_build(&ext_dir, &d, "ggml-cpu", &n_jobs, &flags);
            prefix_ggml_cpu_archive(
                &d.join("ggml").join("src").join("libggml-cpu.a"),
                &avx2_vnni_lib,
                "voicecli_avx2_vnni_",
                &libs_dir,
            );
        }

        {
            let d = portable_root.join("avx512");
            let flags = [
                "-DGGML_NATIVE=OFF",
                "-DGGML_OPENMP=OFF",
                "-DGGML_LLAMAFILE=ON",
                "-DGGML_CPU_REPACK=ON",
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

        {
            let d = portable_root.join("avx512_vnni");
            let flags = [
                "-DGGML_NATIVE=OFF",
                "-DGGML_OPENMP=OFF",
                "-DGGML_LLAMAFILE=ON",
                "-DGGML_CPU_REPACK=ON",
                "-DGGML_SSE42=ON",
                "-DGGML_AVX=ON",
                "-DGGML_AVX_VNNI=OFF",
                "-DGGML_AVX2=ON",
                "-DGGML_BMI2=ON",
                "-DGGML_FMA=ON",
                "-DGGML_F16C=ON",
                "-DGGML_AVX512=ON",
                "-DGGML_AVX512_VBMI=ON",
                "-DGGML_AVX512_VNNI=ON",
                "-DGGML_AVX512_BF16=OFF",
                "-DGGML_AMX_TILE=OFF",
                "-DGGML_AMX_INT8=OFF",
                "-DGGML_AMX_BF16=OFF",
                "-DCMAKE_C_FLAGS=-march=x86-64 -msse4.2 -mpopcnt -mavx -mf16c -mfma -mbmi -mbmi2 -mavx2 -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mavx512vbmi -mavx512vnni -mtune=generic",
                "-DCMAKE_CXX_FLAGS=-march=x86-64 -msse4.2 -mpopcnt -mavx -mf16c -mfma -mbmi -mbmi2 -mavx2 -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mavx512vbmi -mavx512vnni -mtune=generic",
            ];
            run_cmake_configure_and_build(&ext_dir, &d, "ggml-cpu", &n_jobs, &flags);
            prefix_ggml_cpu_archive(
                &d.join("ggml").join("src").join("libggml-cpu.a"),
                &avx512_vnni_lib,
                "voicecli_avx512_vnni_",
                &libs_dir,
            );
        }

        {
            let d = portable_root.join("avx512_bf16");
            let flags = [
                "-DGGML_NATIVE=OFF",
                "-DGGML_OPENMP=OFF",
                "-DGGML_LLAMAFILE=ON",
                "-DGGML_CPU_REPACK=ON",
                "-DGGML_SSE42=ON",
                "-DGGML_AVX=ON",
                "-DGGML_AVX_VNNI=OFF",
                "-DGGML_AVX2=ON",
                "-DGGML_BMI2=ON",
                "-DGGML_FMA=ON",
                "-DGGML_F16C=ON",
                "-DGGML_AVX512=ON",
                "-DGGML_AVX512_VBMI=ON",
                "-DGGML_AVX512_VNNI=ON",
                "-DGGML_AVX512_BF16=ON",
                "-DGGML_AMX_TILE=OFF",
                "-DGGML_AMX_INT8=OFF",
                "-DGGML_AMX_BF16=OFF",
                "-DCMAKE_C_FLAGS=-march=x86-64 -msse4.2 -mpopcnt -mavx -mf16c -mfma -mbmi -mbmi2 -mavx2 -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mavx512vbmi -mavx512vnni -mavx512bf16 -mtune=generic",
                "-DCMAKE_CXX_FLAGS=-march=x86-64 -msse4.2 -mpopcnt -mavx -mf16c -mfma -mbmi -mbmi2 -mavx2 -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -mavx512vbmi -mavx512vnni -mavx512bf16 -mtune=generic",
            ];
            run_cmake_configure_and_build(&ext_dir, &d, "ggml-cpu", &n_jobs, &flags);
            prefix_ggml_cpu_archive(
                &d.join("ggml").join("src").join("libggml-cpu.a"),
                &avx512_bf16_lib,
                "voicecli_avx512_bf16_",
                &libs_dir,
            );
        }

        let ggml_src = base_dir.join("ggml").join("src");
        (base_dir, ggml_src, Some(libs_dir))
    } else {
        let dir_name = if native_opt { "build-native" } else { "build-static-portable" };
        let b_dir = ext_dir.join(dir_name);
        let probe_cc = cc::Build::new().get_compiler();
        let mut extra: Vec<String> = vec![
            "-DGGML_OPENMP=OFF".to_string(),
            "-DGGML_LLAMAFILE=ON".to_string(),
            "-DGGML_CPU_REPACK=ON".to_string(),
        ];
        if !native_opt {
            extra.push("-DGGML_NATIVE=OFF".to_string());
            if target_arch == "aarch64" {
                if target_os == "macos" {
                    // All Apple Silicon (M1+) supports armv8.4-a+dotprod+fp16
                    if compiler_supports_flag(&probe_cc, "-march=armv8.2-a+dotprod+fp16") {
                        extra.push("-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16".to_string());
                    }
                } else if compiler_supports_flag(&probe_cc, "-march=armv8-a") {
                    extra.push("-DGGML_CPU_ARM_ARCH=armv8-a".to_string());
                }
            }
        } else {
            extra.push("-DGGML_NATIVE=ON".to_string());
        }
        let extra_refs: Vec<&str> = extra.iter().map(|s| s.as_str()).collect();
        run_cmake_configure_and_build(&ext_dir, &b_dir, "qwen3_tts", &n_jobs, &extra_refs);
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
        println!("cargo:rustc-link-lib=static=ggml-cpu-avx512-bf16");
        println!("cargo:rustc-link-lib=static=ggml-cpu-avx512-vnni");
        println!("cargo:rustc-link-lib=static=ggml-cpu-avx512");
        println!("cargo:rustc-link-lib=static=ggml-cpu-avx2-vnni");
        println!("cargo:rustc-link-lib=static=ggml-cpu-avx2");
        println!("cargo:rustc-link-lib=static=ggml-cpu-sse42");
        println!("cargo:rustc-link-lib=static=ggml-cpu-x64");
    } else {
        println!("cargo:rustc-link-lib=static=ggml-cpu");
    }
    println!("cargo:rustc-link-lib=static=ggml-base");

    // Link C++ runtime and system dependencies (OpenMP/libgomp eliminated via native ggml_threadpool!)
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");

    println!("cargo:rerun-if-changed=src/model.c");
    println!("cargo:rerun-if-changed=src/cpu_dispatch.c");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=VOICECLI_NATIVE");
}
