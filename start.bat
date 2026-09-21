@echo off
setlocal
title Pose Viewer
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=%~dp0pose_transfer_env\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] No virtual environment was found.
    echo Expected one of:
    echo   %~dp0.venv\Scripts\python.exe
    echo   %~dp0pose_transfer_env\Scripts\python.exe
    echo.
    echo Create a Python environment and install requirements.txt first.
    pause
    exit /b 1
)

set "RTMW_CONFIG=%~dp0models\configs\rtmw-x_8xb320-270e_cocktail14-384x288.py"
set "RTMW_X_WEIGHT=%~dp0models\rtmw-x_8xb320-270e_cocktail14-384x288.pth"
set "RTMW_L_WEIGHT=%~dp0models\rtmw-l_8xb320-270e_cocktail14-384x288.pth"
set "POSE3D_WEIGHT=%~dp0models\simple3Dbaseline_h36m-f0ad73a4_20210419.pth"

if not exist "%RTMW_CONFIG%" goto DOWNLOAD_MODELS
if not exist "%RTMW_X_WEIGHT%" goto DOWNLOAD_MODELS
if not exist "%RTMW_L_WEIGHT%" goto DOWNLOAD_MODELS
if not exist "%POSE3D_WEIGHT%" goto DOWNLOAD_MODELS
goto START_VIEWER

:DOWNLOAD_MODELS
echo Required model files are missing.
echo Downloading models before starting Pose Viewer...
echo.
"%PYTHON%" "%~dp0models\download_models.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Model download failed. Pose Viewer was not started.
    pause
    exit /b 1
)

:START_VIEWER
echo Starting Pose Viewer...
echo Open http://127.0.0.1:8765/ on this computer.
echo Other devices on the same LAN can use this computer's IPv4 address on port 8765.
echo Close this window to stop the server.
echo.

"%PYTHON%" "%~dp0scripts\view_pose.py" --host 0.0.0.0
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Pose Viewer exited with code %EXIT_CODE%.
    pause
)

endlocal
exit /b %EXIT_CODE%
