@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "INSTALL_ATTEMPTED="
pushd "%~dp0"

:find_python
set "PYTHON_EXE="
if defined QSA_PYTHON_EXE goto explicit_python

rem Prefer an existing conversion environment even if Python is not on PATH.
if exist "%~dp0.venv-win\Scripts\python.exe" (
  "%~dp0.venv-win\Scripts\python.exe" -c "import sys,struct; assert sys.version_info>=(3,11) and struct.calcsize('P')==8" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_EXE=%~dp0.venv-win\Scripts\python.exe"
    goto run_exe
  )
)

where py >nul 2>&1
if errorlevel 1 goto scan_installs
py -3.11 -c "import sys,struct; assert struct.calcsize('P')==8" >nul 2>&1
if not errorlevel 1 goto run311
py -3.12 -c "import sys,struct; assert struct.calcsize('P')==8" >nul 2>&1
if not errorlevel 1 goto run312
py -3 -c "import sys,struct; assert sys.version_info>=(3,11) and struct.calcsize('P')==8" >nul 2>&1
if not errorlevel 1 goto runpy

:scan_installs
rem Typical Python.org per-user and machine-wide installations, including a new WinGet install.
rem Check these before PATH aliases so an installed Python does not open the Microsoft Store.
for %%P in (
  "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
  "%ProgramFiles%\Python311\python.exe"
  "%ProgramFiles%\Python312\python.exe"
  "%ProgramFiles%\Python313\python.exe"
) do (
  if exist "%%~P" (
    "%%~P" -c "import sys,struct; assert sys.version_info>=(3,11) and struct.calcsize('P')==8" >nul 2>&1
    if not errorlevel 1 (
      set "PYTHON_EXE=%%~P"
      goto run_exe
    )
  )
)

:try_python
where python >nul 2>&1
if errorlevel 1 goto try_python3
python -c "import sys,struct; assert sys.version_info>=(3,11) and struct.calcsize('P')==8" >nul 2>&1
if errorlevel 1 goto try_python3
python "%~dp0scripts\convert_windows.py" %*
set "RC=%ERRORLEVEL%"
goto done

:try_python3
where python3 >nul 2>&1
if errorlevel 1 goto missing
python3 -c "import sys,struct; assert sys.version_info>=(3,11) and struct.calcsize('P')==8" >nul 2>&1
if errorlevel 1 goto missing
python3 "%~dp0scripts\convert_windows.py" %*
set "RC=%ERRORLEVEL%"
goto done

:explicit_python
set "PYTHON_EXE=%QSA_PYTHON_EXE:"=%"
if not exist "%PYTHON_EXE%" goto invalid_explicit
"%PYTHON_EXE%" -c "import sys,struct; assert sys.version_info>=(3,11) and struct.calcsize('P')==8" >nul 2>&1
if errorlevel 1 goto invalid_explicit
goto run_exe

:run_exe
echo Using Python: "%PYTHON_EXE%"
"%PYTHON_EXE%" "%~dp0scripts\convert_windows.py" %*
set "RC=%ERRORLEVEL%"
goto done

:invalid_explicit
echo Error: QSA_PYTHON_EXE does not point to a working 64-bit Python 3.11+ interpreter.
echo Set it to an existing python.exe, or clear it to use automatic discovery.
set "RC=1"
goto done

:run311
py -3.11 "%~dp0scripts\convert_windows.py" %*
set "RC=%ERRORLEVEL%"
goto done

:run312
py -3.12 "%~dp0scripts\convert_windows.py" %*
set "RC=%ERRORLEVEL%"
goto done

:runpy
py -3 "%~dp0scripts\convert_windows.py" %*
set "RC=%ERRORLEVEL%"
goto done

:missing
echo No suitable 64-bit Python 3.11 or 3.12 was found.
echo For a portable setup without installers, use convert-windows-portable.bat instead.
echo Install with: winget install -e --id Python.Python.3.11 --scope user --architecture x64
echo Or use the official installer: https://www.python.org/downloads/windows/
echo If Python is installed elsewhere, set QSA_PYTHON_EXE to its full python.exe path.
if defined INSTALL_ATTEMPTED goto missing_finish
if not "%~1"=="" goto missing_finish
where winget >nul 2>&1
if errorlevel 1 goto missing_finish
choice /C YN /N /M "Install official Python 3.11 using WinGet now? [Y/N] "
if errorlevel 2 goto missing_finish
if not errorlevel 1 goto missing_finish
set "INSTALL_ATTEMPTED=1"
winget install -e --id Python.Python.3.11 --source winget --scope user --architecture x64
if errorlevel 1 goto missing_finish
goto find_python

:missing_finish
echo After installation, close this window and launch the BAT again. Git for Windows is also required.
set "RC=1"

:done
if not "%RC%"=="0" echo Conversion failed (exit %RC%). See the message above.
if "%~1"=="" pause
popd
exit /b %RC%
