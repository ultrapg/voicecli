@echo off
setlocal enabledelayedexpansion

:: Check if the full environment needs to be bootstrapped
if not exist python-embed (
    echo [*] python-embed not found. Bootstrapping portable Python environment...
    
    echo $ErrorActionPreference = "Stop"> temp_setup.ps1
    echo Write-Host "[*] Creating python-embed directory...">> temp_setup.ps1
    echo New-Item -ItemType Directory -Path "python-embed" -Force ^| Out-Null >> temp_setup.ps1
    echo $url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip">> temp_setup.ps1
    echo $zip = "python-embed.zip">> temp_setup.ps1
    echo Write-Host "[*] Downloading Python 3.11.9 embeddable package...">> temp_setup.ps1
    echo Invoke-WebRequest -Uri $url -OutFile $zip>> temp_setup.ps1
    echo Write-Host "[*] Extracting Python to python-embed...">> temp_setup.ps1
    echo Expand-Archive -Path $zip -DestinationPath "python-embed" -Force>> temp_setup.ps1
    echo Remove-Item $zip>> temp_setup.ps1
    
    echo Write-Host "[*] Creating python-embed\libs directory...">> temp_setup.ps1
    echo New-Item -ItemType Directory -Path "python-embed\libs" -Force ^| Out-Null >> temp_setup.ps1
    echo Write-Host "[*] Downloading Python 3.11.9 NuGet package for link libraries...">> temp_setup.ps1
    echo $nuget_url = "https://www.nuget.org/api/v2/package/python/3.11.9">> temp_setup.ps1
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
    echo Write-Host "[*] Installing pip locally in python-embed...">> temp_setup.ps1
    echo ^& "python-embed\python.exe" "python-embed\get-pip.py" --no-warn-script-location>> temp_setup.ps1
    echo Remove-Item "python-embed\get-pip.py">> temp_setup.ps1
    
    echo Write-Host "[*] Configuring python311._pth to enable site-packages...">> temp_setup.ps1
    echo $pth = "python-embed\python311._pth">> temp_setup.ps1
    echo if ^(Test-Path $pth^) {>> temp_setup.ps1
    echo     ^(Get-Content $pth^) -replace '#import site', 'import site' ^| Set-Content $pth>> temp_setup.ps1
    echo }>> temp_setup.ps1
    
    echo Write-Host "[*] Installing deep learning packages locally (this may take a few minutes)...">> temp_setup.ps1
    echo ^& "python-embed\python.exe" -m pip install torch "torchaudio==2.4.1" transformers numpy soundfile accelerate qwen-tts torch-directml --no-warn-script-location>> temp_setup.ps1
    echo Write-Host "[+] Isolated Python environment setup complete!">> temp_setup.ps1

    powershell -NoProfile -ExecutionPolicy Bypass -File temp_setup.ps1
    set BOOTSTRAP_ERR=!ERRORLEVEL!
    if exist temp_setup.ps1 del /f /q temp_setup.ps1
    if !BOOTSTRAP_ERR! neq 0 (
        echo [-] Error: Bootstrapping Python environment failed!
        exit /b !BOOTSTRAP_ERR!
    )
) else (
    :: Environment exists, check if MSVC link libraries are missing
    if not exist python-embed\libs\python311.lib (
        echo [*] python-embed exists but MSVC link libraries are missing. Bootstrapping libs...
        
        echo $ErrorActionPreference = "Stop"> temp_setup_libs.ps1
        echo Write-Host "[*] Creating python-embed\libs directory...">> temp_setup_libs.ps1
        echo New-Item -ItemType Directory -Path "python-embed\libs" -Force ^| Out-Null >> temp_setup_libs.ps1
        echo Write-Host "[*] Downloading Python 3.11.9 NuGet package for link libraries...">> temp_setup_libs.ps1
        echo $nuget_url = "https://www.nuget.org/api/v2/package/python/3.11.9">> temp_setup_libs.ps1
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

:: Ensure torch-directml and matching torchaudio are installed in python-embed for DirectML GPU acceleration
python-embed\python.exe -c "import torch_directml; import torchaudio" >nul 2>nul
if !ERRORLEVEL! neq 0 (
    echo [*] torch-directml not found in python-embed. Installing torch-directml and torchaudio 2.4.1...
    python-embed\python.exe -m pip install torch-directml "torchaudio==2.4.1" --no-warn-script-location
    if !ERRORLEVEL! neq 0 (
        echo [-] Warning: Failed to install torch-directml/torchaudio. DirectML GPU acceleration will be disabled.
    )
)

echo [*] Starting voicecli release build...
cargo build --release
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
copy target\release\voicecli.exe %DIST_DIR%\
copy target\release\*.dll %DIST_DIR%\
copy target\release\*.zip %DIST_DIR%\
copy settings.json %DIST_DIR%\

echo [*] Copying portable Python environment (this may take a moment)...
xcopy /e /i /q python-embed %DIST_DIR%\python-embed

echo [*] Compressing into deploy\voicecli.zip...
if not exist deploy mkdir deploy
if exist deploy\voicecli.zip (
    del /f /q deploy\voicecli.zip
)
tar -cf deploy\voicecli.zip --format=zip %DIST_DIR%

echo [*] Cleaning up temporary folder...
rd /s /q %DIST_DIR%

echo [+] Success! Standalone bundle created: deploy\voicecli.zip
endlocal
