@echo off
setlocal

REM Launch AudioCSV in guided mode. Adjust PYTHON_EXE if Python is not on PATH.
set "PYTHON_EXE=python"

%PYTHON_EXE% "%~dp0audio_to_csv.py" --interactive

if errorlevel 1 (
    echo.
    echo Something went wrong. Make sure Python and the required packages are installed.
    echo Install dependencies with:
    echo    %PYTHON_EXE% -m pip install librosa pyloudnorm pandas numpy soundfile pyarrow
    pause
) else (
    echo.
    echo Done! Check the folder for your freshly generated CSV file(s).
    pause
)
endlocal
