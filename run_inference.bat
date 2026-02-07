@echo off
REM PPE Detection Inference Script for Windows
REM Usage: run_inference.bat image "path/to/images" "path/to/output"

if "%1"=="" (
    echo Usage: run_inference.bat [image^|video^|webcam] [input_path] [output_path]
    echo.
    echo Examples:
    echo   run_inference.bat image "data/test/images" "results/"
    echo   run_inference.bat video "test_video.mp4" "output.mp4"
    echo   run_inference.bat webcam
    exit /b 1
)

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Run inference with provided arguments
if "%1"=="webcam" (
    python -m src.inference --source webcam --camera 0
) else if "%1"=="image" (
    if "%2"=="" (
        echo Error: Input path required for image mode
        exit /b 1
    )
    if "%3"=="" (
        python -m src.inference --source image --input "%2"
    ) else (
        python -m src.inference --source image --input "%2" --output "%3"
    )
) else if "%1"=="video" (
    if "%2"=="" (
        echo Error: Input path required for video mode
        exit /b 1
    )
    if "%3"=="" (
        python -m src.inference --source video --input "%2"
    ) else (
        python -m src.inference --source video --input "%2" --output "%3"
    )
) else (
    echo Unknown source type: %1
    echo Use: image, video, or webcam
    exit /b 1
)