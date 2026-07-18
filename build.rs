use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let mut python_sys_home = String::new();

    // Only copy DLLs and runtime files on Windows targets
    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "windows" {
        if let Ok(out_dir) = env::var("OUT_DIR") {
            // OUT_DIR is target/<profile>/build/voicecli-<hash>/out
            let out_path = Path::new(&out_dir);
            
            // Traverse up 3 directories to reach target/<profile>
            if let Some(target_dir) = out_path
                .parent() // voicecli-<hash>
                .and_then(|p| p.parent()) // build
                .and_then(|p| p.parent()) // target/<profile>
            {
                // List of known DLL paths on Marvin's system
                // Local portable python-embed takes precedence!
                let possible_dlls = [
                    "python-embed\\python311.dll",
                    "C:\\Users\\Marvin\\AppData\\Local\\Python\\pythoncore-3.14-64\\python314.dll",
                    "C:\\Program Files\\PyManager\\runtime\\python314.dll",
                ];

                for dll_path in possible_dlls.iter() {
                    let src = Path::new(dll_path);
                    if src.exists() {
                        if let Some(src_dir) = src.parent() {
                            // Save the Python home directory path
                            python_sys_home = src_dir.to_string_lossy().to_string();
                            
                            // Copy all DLLs and ZIP files in the same folder as the target Python DLL
                            if let Ok(entries) = fs::read_dir(src_dir) {
                                for entry in entries.flatten() {
                                    let path = entry.path();
                                    if path.is_file() {
                                        let is_dll = path.extension().map_or(false, |ext| ext.eq_ignore_ascii_case("dll"));
                                        let is_zip = path.extension().map_or(false, |ext| ext.eq_ignore_ascii_case("zip"));
                                        if is_dll || is_zip {
                                            if let Some(filename) = path.file_name() {
                                                let dest = target_dir.join(filename);
                                                println!("cargo:warning=Auto-copying: {} -> {}", path.display(), dest.display());
                                                if let Err(e) = fs::copy(&path, &dest) {
                                                    println!("cargo:warning=Failed to copy file {}: {}", filename.to_string_lossy(), e);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    // Pass the discovered Python home directory to main.rs as a compile-time environment variable
    println!("cargo:rustc-env=PYTHON_SYS_HOME={}", python_sys_home);

    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "linux" {
        let local_embed_lib = Path::new("python-embed/lib");
        if local_embed_lib.exists() && local_embed_lib.is_dir() {
            if let Ok(abs_lib) = local_embed_lib.canonicalize() {
                println!("cargo:rustc-link-search=native={}", abs_lib.display());
            }
        } else {
            if let Ok(out_dir) = env::var("OUT_DIR") {
                let link_dir = Path::new(&out_dir).join("python-link");
                let _ = fs::create_dir_all(&link_dir);
                let target_so = link_dir.join("libpython3.14.so");
                if !target_so.exists() {
                    let _ = std::os::unix::fs::symlink("/usr/lib/x86_64-linux-gnu/libpython3.14.so.1.0", &target_so);
                }
                println!("cargo:rustc-link-search=native={}", link_dir.display());
            }
        }
    }

    // Ensure Cargo rebuilds if build.rs or python-embed changes
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=python-embed");
    println!("cargo:rerun-if-changed=python-embed/lib");
}
