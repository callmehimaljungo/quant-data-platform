@echo off
setlocal

echo ========================================================
echo RUNNING FULL QUANT PIPELINE (LOCAL MANUAL TEST)
echo ========================================================
echo.

set PYTHON_PATH=python

:: 1. BRONZE LAYER
echo [1/4] Running BRONZE Layer Ingestion...
%PYTHON_PATH% bronze/ingest.py auto --upload-r2
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Bronze ingestion failed.
    goto :error
)

echo [1/4] Running BRONZE News Loader...
%PYTHON_PATH% bronze/news_loader.py 100
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] News loading failed.
    goto :error
)

echo.
echo --------------------------------------------------------
echo.

:: 2. SILVER LAYER
echo [2/4] Running SILVER Layer Processing...
%PYTHON_PATH% silver/clean.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Silver cleaning failed.
    goto :error
)

echo [2/4] Running SILVER News Processing...
%PYTHON_PATH% silver/process_news.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Silver news processing failed.
    goto :error
)

echo.
echo --------------------------------------------------------
echo.

:: 3. GOLD LAYER
echo [3/4] Running GOLD Layer Strategy (Run All)...
%PYTHON_PATH% gold/run_all_strategies.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Gold strategies failed.
    goto :error
)

echo.
echo --------------------------------------------------------
echo.

:: 4. DASHBOARD CHECK
echo [4/4] Pipeline phases complete. Launching Dashboard...
echo (Press Ctrl+C to stop dashboard)
echo.
%PYTHON_PATH% -m streamlit run dashboard/app.py

goto :eof

:error
echo.
echo [FAIL] Pipeline execution failed. Check errors above.
exit /b 1
