@echo off

:: Try to use python from the PATH
python --version >nul 2>&1
if errorlevel 1 (
    :: Python not found in PATH, check for embedded python
    if not exist ..\..\..\python_embeded\python.exe (
        echo Python not found in PATH and embedded python not found. Please install manually.
        pause
        exit /b 1
    ) else (
        :: Use the embedded python
        set PYTHON=..\..\..\python_embeded\python.exe
    )
) else (
    :: Use python from the PATH
    set PYTHON=python
)

:: Install the package
echo Installing...
%PYTHON% install.py
echo Done!

@pause