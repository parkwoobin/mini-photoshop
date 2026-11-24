@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment...
    python -m venv .venv
    if errorlevel 1 goto fail
)
.venv\Scripts\python.exe -c "import cv2, PIL, pillow_heif, onnxruntime, PyQt5, pptx" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies. This may take several minutes...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 goto fail
)
.venv\Scripts\python.exe app_qt.py
if errorlevel 1 goto fail
exit /b
:fail
echo Could not start. Check Python and internet connection.
pause


