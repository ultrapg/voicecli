@echo off
setlocal enabledelayedexpansion

echo [*] Starting voicecli release build...
cargo build --release
if %ERRORLEVEL% neq 0 (
    echo [-] Error: Cargo build failed!
    exit /b %ERRORLEVEL%
)

echo [*] Setting up portable distribution folder...
set DIST_DIR=voicecli-portable
if exist %DIST_DIR% (
    rd /s /q %DIST_DIR%
)
mkdir %DIST_DIR%

echo [*] Copying binary and runtime assets...
copy target\release\voicecli.exe %DIST_DIR%\
copy target\release\*.dll %DIST_DIR%\
copy target\release\*.zip %DIST_DIR%\

echo [*] Copying portable Python environment (this may take a moment)...
xcopy /e /i /q python-embed %DIST_DIR%\python-embed

echo [*] Compressing into voicecli-portable.zip...
if exist voicecli-portable.zip (
    del /f /q voicecli-portable.zip
)
powershell -Command "Compress-Archive -Path '%DIST_DIR%' -DestinationPath 'voicecli-portable.zip' -Force"

echo [*] Cleaning up temporary folder...
rd /s /q %DIST_DIR%

echo [+] Success! Standalone bundle created: voicecli-portable.zip
endlocal
