@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
pushd "%~dp0"
if errorlevel 1 exit /b 1
set "PORTABLE=%CD%\.portable"
set "PYTHON_EXE=%PORTABLE%\python\python.exe"
set "GIT_EXE=%PORTABLE%\git\cmd\git.exe"

rem All environment changes are limited to this BAT and its child processes.
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PIP_CONFIG_FILE=NUL"
set "PIP_USER=0"
set "PIP_TARGET="
set "PIP_PREFIX="
set "PIP_NO_CACHE_DIR=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_REQUIRE_VIRTUALENV="
set "HF_HOME=%PORTABLE%\cache\huggingface"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HUGGINGFACE_HUB_CACHE=%HF_HUB_CACHE%"
set "HF_XET_CACHE=%HF_HOME%\xet"
set "HF_ASSETS_CACHE=%HF_HOME%\assets"
set "HF_HUB_DISABLE_TELEMETRY=1"

rem A full portable ZIP already contains both tools. A source checkout can
rem bootstrap them without an installer, administrator access, or WinGet.
if not exist "%PYTHON_EXE%" (
  call :get_python
  if errorlevel 1 goto failed
)
if not exist "%GIT_EXE%" (
  call :get_git
  if errorlevel 1 goto failed
)

"%PYTHON_EXE%" -c "import sys,struct,ssl,pip; assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8; print('Portable Python',sys.version.split()[0])"
if errorlevel 1 goto failed
"%GIT_EXE%" --version
if errorlevel 1 goto failed
set "PATH=%PORTABLE%\python;%PORTABLE%\python\Scripts;%PORTABLE%\git\cmd;%PATH%"
set "GIT_TERMINAL_PROMPT=0"
set "TEMP=%PORTABLE%\cache\tmp"
set "TMP=%TEMP%"
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%TEMP%" goto failed
"%PYTHON_EXE%" "%~dp0scripts\convert_windows.py" %*
set "RC=%ERRORLEVEL%"
goto done

:get_python
if exist "%PORTABLE%\python" (
  echo Error: incomplete portable Python directory: "%PORTABLE%\python"
  exit /b 1
)
if not exist "%SystemRoot%\System32\tar.exe" (
  echo Windows tar.exe is unavailable. Download the complete portable ZIP from GitHub Releases instead.
  exit /b 1
)
set "QSA_ARCHIVE=%PORTABLE%\downloads\cpython-3.12.16+20261110-x86_64-pc-windows-msvc-install_only.tar.gz"
set "QSA_ARCHIVE_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20261110/cpython-3.12.16+20261110-x86_64-pc-windows-msvc-install_only.tar.gz"
set "QSA_ARCHIVE_SHA=e23fe5b16926de1d81661b1a97352e89fb167549c55a6825796ccd0f32ec7f1b"
call :download
if errorlevel 1 exit /b 1
set "STAGE=%PORTABLE%\python-stage-%RANDOM%-%RANDOM%"
mkdir "%STAGE%"
if errorlevel 1 exit /b 1
"%SystemRoot%\System32\tar.exe" -xzf "%QSA_ARCHIVE%" -C "%STAGE%"
if errorlevel 1 exit /b 1
if not exist "%STAGE%\python\python.exe" exit /b 1
"%STAGE%\python\python.exe" -c "import sys,struct,ssl,pip; assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8"
if errorlevel 1 exit /b 1
if exist "%PORTABLE%\python" exit /b 1
move "%STAGE%\python" "%PORTABLE%\python" >nul
if errorlevel 1 exit /b 1
rd "%STAGE%" 2>nul
exit /b 0

:get_git
if exist "%PORTABLE%\git" (
  echo Error: incomplete portable Git directory: "%PORTABLE%\git"
  exit /b 1
)
set "QSA_ARCHIVE=%PORTABLE%\downloads\MinGit-2.56.1-64-bit.zip"
set "QSA_ARCHIVE_URL=https://github.com/git-for-windows/git/releases/download/v2.56.1.windows.1/MinGit-2.56.1-64-bit.zip"
set "QSA_ARCHIVE_SHA=4a92850b6e917da4fd010669b614b55dccf5b0ec333a534cc4be4700dfefdafd"
call :download
if errorlevel 1 exit /b 1
set "QSA_STAGE=%PORTABLE%\git-stage-%RANDOM%-%RANDOM%"
if not exist "%SystemRoot%\System32\tar.exe" exit /b 1
mkdir "%QSA_STAGE%"
if errorlevel 1 exit /b 1
"%SystemRoot%\System32\tar.exe" -xf "%QSA_ARCHIVE%" -C "%QSA_STAGE%"
if errorlevel 1 exit /b 1
if not exist "%QSA_STAGE%\cmd\git.exe" exit /b 1
"%QSA_STAGE%\cmd\git.exe" --version
if errorlevel 1 exit /b 1
if exist "%PORTABLE%\git" exit /b 1
move "%QSA_STAGE%" "%PORTABLE%\git" >nul
exit /b %ERRORLEVEL%

:download
if not exist "%PORTABLE%\downloads" mkdir "%PORTABLE%\downloads"
if not exist "%PORTABLE%\downloads" exit /b 1
powershell.exe -NoLogo -NoProfile -NonInteractive -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; function Get-Sha([string]$p) { $s=[IO.File]::OpenRead($p); $h=[Security.Cryptography.SHA256]::Create(); try { [BitConverter]::ToString($h.ComputeHash($s)).Replace('-','').ToLowerInvariant() } finally { $s.Dispose(); $h.Dispose() } }; $file=$env:QSA_ARCHIVE; if ([IO.File]::Exists($file) -and ((Get-Sha $file) -eq $env:QSA_ARCHIVE_SHA)) { Write-Host 'Reusing verified archive:' $file; exit 0 }; $part=$file+'.partial'; Write-Host 'Downloading portable runtime:' $env:QSA_ARCHIVE_URL; Invoke-WebRequest -UseBasicParsing -Uri $env:QSA_ARCHIVE_URL -OutFile $part; if ((Get-Sha $part) -ne $env:QSA_ARCHIVE_SHA) { throw 'Portable runtime SHA-256 mismatch' }; Move-Item -LiteralPath $part -Destination $file -Force; Write-Host 'SHA-256 OK:' $file"
exit /b %ERRORLEVEL%

:failed
set "RC=1"
echo Portable runtime preparation failed. See the message above, or download the complete Windows portable ZIP.

:done
if not "%RC%"=="0" echo Conversion failed (exit %RC%). See the message above.
if "%~1"=="" pause
popd
exit /b %RC%
