@echo off
setlocal enabledelayedexpansion

:: Check architecture
set ARCH=amd64
if /i "%~1"=="--arch" (
    set ARCH=%~2
) else (
    if "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
        set ARCH=arm64
    ) else if "%PROCESSOR_ARCHITEW6432%"=="ARM64" (
        set ARCH=arm64
    )
)

:: Host Architecture
set HOST_ARCH=amd64
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
    set HOST_ARCH=arm64
) else if "%PROCESSOR_ARCHITEW6432%"=="ARM64" (
    set HOST_ARCH=arm64
)

if "!ARCH!"=="!HOST_ARCH!" (
    set IS_CROSS=0
) else (
    set IS_CROSS=1
)

echo [*] Target Architecture: !ARCH! (Cross-compiling: !IS_CROSS!)

set PYTHON_URL_AMD64=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip
set PYTHON_URL_ARM64=https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-arm64.zip
set NUGET_URL_AMD64=https://www.nuget.org/api/v2/package/python/3.11.9
set NUGET_URL_ARM64=https://www.nuget.org/api/v2/package/pythonarm64/3.11.9

if "!ARCH!"=="arm64" (
    set PYTHON_URL=!PYTHON_URL_ARM64!
    set NUGET_URL=!NUGET_URL_ARM64!
    set CARGO_TARGET=aarch64-pc-windows-msvc
) else (
    set PYTHON_URL=!PYTHON_URL_AMD64!
    set NUGET_URL=!NUGET_URL_AMD64!
    set CARGO_TARGET=x86_64-pc-windows-msvc
)

:: Check if the full environment needs to be bootstrapped
if not exist python-embed (
    echo [*] python-embed not found. Bootstrapping portable Python environment...
    
    echo $ErrorActionPreference = "Stop"> temp_setup.ps1
    echo Write-Host "[*] Creating python-embed directory...">> temp_setup.ps1
    echo New-Item -ItemType Directory -Path "python-embed" -Force ^| Out-Null >> temp_setup.ps1
    echo $url = "!PYTHON_URL!">> temp_setup.ps1
    echo $zip = "python-embed.zip">> temp_setup.ps1
    echo Write-Host "[*] Downloading Python 3.11.9 embeddable package...">> temp_setup.ps1
    echo Invoke-WebRequest -Uri $url -OutFile $zip>> temp_setup.ps1
    echo Write-Host "[*] Extracting Python to python-embed...">> temp_setup.ps1
    echo Expand-Archive -Path $zip -DestinationPath "python-embed" -Force>> temp_setup.ps1
    echo Remove-Item $zip>> temp_setup.ps1
    
    echo Write-Host "[*] Creating python-embed\libs directory...">> temp_setup.ps1
    echo New-Item -ItemType Directory -Path "python-embed\libs" -Force ^| Out-Null >> temp_setup.ps1
    echo Write-Host "[*] Downloading Python 3.11.9 NuGet package for link libraries...">> temp_setup.ps1
    echo $nuget_url = "!NUGET_URL!">> temp_setup.ps1
    echo $nuget_zip = "python-nuget.zip">> temp_setup.ps1
    echo Invoke-WebRequest -Uri $nuget_url -OutFile $nuget_zip>> temp_setup.ps1
    echo Write-Host "[*] Extracting NuGet package...">> temp_setup.ps1
    echo Expand-Archive -Path $nuget_zip -DestinationPath "python-nuget" -Force>> temp_setup.ps1
    echo Write-Host "[*] Copying link libraries to python-embed\libs...">> temp_setup.ps1
    echo Copy-Item -Path "python-nuget\tools\libs\*" -Destination "python-embed\libs" -Force>> temp_setup.ps1
    echo Remove-Item $nuget_zip>> temp_setup.ps1
    echo Remove-Item "python-nuget" -Recurse -Force>> temp_setup.ps1
    
    echo Write-Host "[*] Downloading get-pip.py...">> temp_setup.ps1
    echo Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "python-embed\get-pip.py">> temp_setup.ps1
    
    echo Write-Host "[*] Configuring python311._pth to enable site-packages...">> temp_setup.ps1
    echo $pth = "python-embed\python311._pth">> temp_setup.ps1
    echo if ^(Test-Path $pth^) {>> temp_setup.ps1
    echo     ^(Get-Content $pth^) -replace '#import site', 'import site' ^| Set-Content $pth>> temp_setup.ps1
    echo }>> temp_setup.ps1

    powershell -NoProfile -ExecutionPolicy Bypass -File temp_setup.ps1
    set BOOTSTRAP_ERR=!ERRORLEVEL!
    if exist temp_setup.ps1 del /f /q temp_setup.ps1
    if !BOOTSTRAP_ERR! neq 0 (
        echo [-] Error: Bootstrapping Python environment failed!
        exit /b !BOOTSTRAP_ERR!
    )

    if "!IS_CROSS!"=="1" (
        echo [*] Cross-compiling: Bootstrapping packages with host Python...

        echo $ErrorActionPreference = "Stop"> temp_cross.ps1
        if "!HOST_ARCH!"=="arm64" (
            echo $host_py_url = "!PYTHON_URL_ARM64!">> temp_cross.ps1
        ) else (
            echo $host_py_url = "!PYTHON_URL_AMD64!">> temp_cross.ps1
        )
        echo New-Item -ItemType Directory -Path "host-python-embed" -Force ^| Out-Null >> temp_cross.ps1
        echo Invoke-WebRequest -Uri $host_py_url -OutFile "host-python.zip">> temp_cross.ps1
        echo Expand-Archive -Path "host-python.zip" -DestinationPath "host-python-embed" -Force>> temp_cross.ps1
        echo Remove-Item "host-python.zip">> temp_cross.ps1

        echo $host_pth = "host-python-embed\python311._pth">> temp_cross.ps1
        echo if ^(Test-Path $host_pth^) {>> temp_cross.ps1
        echo     ^(Get-Content $host_pth^) -replace '#import site', 'import site' ^| Set-Content $host_pth>> temp_cross.ps1
        echo }>> temp_cross.ps1

        powershell -NoProfile -ExecutionPolicy Bypass -File temp_cross.ps1
        if exist temp_cross.ps1 del /f /q temp_cross.ps1

        host-python-embed\python.exe python-embed\get-pip.py --no-warn-script-location

        if "!ARCH!"=="arm64" (
            set PLATFORM=win_arm64
        ) else (
            set PLATFORM=win_amd64
        )

        host-python-embed\python.exe -m pip install torch "torchaudio==2.4.1" transformers numpy soundfile accelerate qwen-tts torch-directml --target python-embed\Lib\site-packages --platform !PLATFORM! --only-binary=:all: --no-warn-script-location

        rd /s /q host-python-embed
    ) else (
        echo [*] Installing pip locally in python-embed...
        python-embed\python.exe python-embed\get-pip.py --no-warn-script-location

        echo [*] Installing deep learning packages locally (this may take a few minutes)...
        python-embed\python.exe -m pip install torch "torchaudio==2.4.1" transformers numpy soundfile accelerate qwen-tts torch-directml --no-warn-script-location
    )

    del /f /q python-embed\get-pip.py
    echo [+] Isolated Python environment setup complete!
) else (
    :: Environment exists, check if MSVC link libraries are missing
    if not exist python-embed\libs\python311.lib (
        echo [*] python-embed exists but MSVC link libraries are missing. Bootstrapping libs...
        
        echo $ErrorActionPreference = "Stop"> temp_setup_libs.ps1
        echo Write-Host "[*] Creating python-embed\libs directory...">> temp_setup_libs.ps1
        echo New-Item -ItemType Directory -Path "python-embed\libs" -Force ^| Out-Null >> temp_setup_libs.ps1
        echo Write-Host "[*] Downloading Python 3.11.9 NuGet package for link libraries...">> temp_setup_libs.ps1
        echo $nuget_url = "!NUGET_URL!">> temp_setup_libs.ps1
        echo $nuget_zip = "python-nuget.zip">> temp_setup_libs.ps1
        echo Invoke-WebRequest -Uri $nuget_url -OutFile $nuget_zip>> temp_setup_libs.ps1
        echo Write-Host "[*] Extracting NuGet package...">> temp_setup_libs.ps1
        echo Expand-Archive -Path $nuget_zip -DestinationPath "python-nuget" -Force>> temp_setup_libs.ps1
        echo Write-Host "[*] Copying link libraries to python-embed\libs...">> temp_setup_libs.ps1
        echo Copy-Item -Path "python-nuget\tools\libs\*" -Destination "python-embed\libs" -Force>> temp_setup_libs.ps1
        echo Remove-Item $nuget_zip>> temp_setup_libs.ps1
        echo Remove-Item "python-nuget" -Recurse -Force>> temp_setup_libs.ps1
        echo Write-Host "[+] Link libraries bootstrapped successfully!">> temp_setup_libs.ps1

        powershell -NoProfile -ExecutionPolicy Bypass -File temp_setup_libs.ps1
        set BOOTSTRAP_ERR=!ERRORLEVEL!
        if exist temp_setup_libs.ps1 del /f /q temp_setup_libs.ps1
        if !BOOTSTRAP_ERR! neq 0 (
            echo [-] Error: Bootstrapping link libraries failed!
            exit /b !BOOTSTRAP_ERR!
        )
    )
)

if "!IS_CROSS!"=="1" (
    set PYO3_CROSS_PYTHON_VERSION=3.11
    set "PYO3_CROSS_LIB_DIR=%CD%\python-embed\libs"
)

echo [*] Starting voicecli release build for !CARGO_TARGET!...
cargo build --release --target !CARGO_TARGET!
if !ERRORLEVEL! neq 0 (
    echo [-] Error: Cargo build failed!
    exit /b !ERRORLEVEL!
)

echo [*] Setting up portable distribution folder...
set DIST_DIR=voicecli
if exist %DIST_DIR% (
    rd /s /q %DIST_DIR%
)
mkdir %DIST_DIR%

echo [*] Copying binary and runtime assets...
copy target\!CARGO_TARGET!\release\voicecli.exe %DIST_DIR%\
copy target\!CARGO_TARGET!\release\*.dll %DIST_DIR%\
copy target\!CARGO_TARGET!\release\*.zip %DIST_DIR%\
copy settings.json %DIST_DIR%\

echo [*] Copying portable Python environment (this may take a moment)...
xcopy /e /i /q python-embed %DIST_DIR%\python-embed

echo [*] Compressing into deploy\voicecli-windows-!ARCH!.zip...
if not exist deploy mkdir deploy
if exist deploy\voicecli-windows-!ARCH!.zip (
    del /f /q deploy\voicecli-windows-!ARCH!.zip
)
tar -cf deploy\voicecli-windows-!ARCH!.zip --format=zip %DIST_DIR%

echo [*] Cleaning up temporary folder...
rd /s /q %DIST_DIR%

echo [+] Success! Standalone bundle created: deploy\voicecli-windows-!ARCH!.zip
endlocal
