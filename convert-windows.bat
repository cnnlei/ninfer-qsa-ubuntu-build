@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
pushd "%~dp0"

where py >nul 2>&1
if errorlevel 1 goto try_python
py -3.11 -c "import sys,struct; assert struct.calcsize('P')==8" >nul 2>&1
if not errorlevel 1 goto run311
py -3.12 -c "import sys,struct; assert struct.calcsize('P')==8" >nul 2>&1
if not errorlevel 1 goto run312
py -3 -c "import sys,struct; assert sys.version_info>=(3,11) and struct.calcsize('P')==8" >nul 2>&1
if not errorlevel 1 goto runpy

:try_python
where python >nul 2>&1
if errorlevel 1 goto missing
python -c "import sys,struct; assert sys.version_info>=(3,11) and struct.calcsize('P')==8" >nul 2>&1
if errorlevel 1 goto missing
python "%~dp0scripts\convert_windows.py" %*
set "RC=%ERRORLEVEL%"
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
echo Error: install 64-bit Python 3.11 or 3.12 from https://www.python.org/downloads/windows/
echo Then launch this BAT again. Git for Windows is also required.
set "RC=1"

:done
if not "%RC%"=="0" echo Conversion failed (exit %RC%). See the message above.
if "%~1"=="" pause
popd
exit /b %RC%
