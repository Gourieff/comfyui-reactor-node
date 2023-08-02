@echo off
:: Exit if embedded python is not found
if not exist ..\..\..\python_embeded\python.exe (
    echo Embedded python not found. Please install manually.
    pause
    exit /b 1
)

:: Install the package
echo Installing roop requirements...
..\..\..\python_embeded\python.exe install.py
echo Finished installing roop requirements.
pause
