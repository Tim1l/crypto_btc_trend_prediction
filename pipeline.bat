@echo off
echo Starting pipeline at %date% %time%
echo Current directory: %CD%
echo Activating virtual environment...
call C:\Users\Administrator\Desktop\crypto_btc_trend_prediction\venv\Scripts\activate.bat
echo Setting PYTHONIOENCODING to UTF-8...
set PYTHONIOENCODING=UTF-8
echo Running pipeline.py...
C:\Users\Administrator\Desktop\crypto_btc_trend_prediction\venv\Scripts\python.exe C:\Users\Administrator\Desktop\crypto_btc_trend_prediction\pipeline.py
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% neq 0 (
    echo Error occurred: %EXIT_CODE%
    echo Check the console output above for details.
) else (
    echo Successfully completed
)