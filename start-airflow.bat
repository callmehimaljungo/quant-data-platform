@echo off
echo ============================================================
echo Quant Data Platform - Airflow Setup
echo ============================================================
echo.

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not running!
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    exit /b 1
)

echo [OK] Docker found

REM Set Airflow UID (Windows)
set AIRFLOW_UID=50000

REM Create .env if not exists
if not exist .env (
    echo [WARN] .env file not found, creating from .env.example
    copy .env.example .env
)

echo.
echo Starting Airflow...
echo This may take 5-10 minutes on first run (building images)
echo.

docker-compose -f docker-compose.airflow.yml up -d

echo.
echo ============================================================
echo Airflow is starting up!
echo.
echo Web UI: http://localhost:8080
echo Username: admin
echo Password: admin
echo.
echo Wait 1-2 minutes for services to initialize.
echo ============================================================

pause
