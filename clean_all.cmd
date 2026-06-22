@echo off
setlocal

echo [*] Cleaning Rust compilation targets...
cargo clean

echo [*] Deleting local portable Python environment (python-embed)...
if exist python-embed (
    rd /s /q python-embed
)

echo [*] Deleting local model weights cache (.cache)...
if exist .cache (
    rd /s /q .cache
)

echo [*] Deleting generated ZIP files...
if exist voicecli-portable.zip (
    del /f /q voicecli-portable.zip
)

echo [*] Deleting portable deployment directory (deploy)...
if exist deploy (
    rd /s /q deploy
)

echo [+] Cleanup complete! The workspace is now in a pure source-code state and ready for GitHub.
endlocal
