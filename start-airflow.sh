#!/bin/bash
echo "============================================================"
echo "Quant Data Platform - Airflow Setup"
echo "============================================================"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed!"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "[OK] Docker found"

# Set Airflow UID
export AIRFLOW_UID=$(id -u)

# Create .env if not exists
if [ ! -f .env ]; then
    echo "[WARN] .env file not found, creating from .env.example"
    cp .env.example .env
fi

echo ""
echo "Starting Airflow..."
echo "This may take 5-10 minutes on first run (building images)"
echo ""

docker-compose -f docker-compose.airflow.yml up -d

echo ""
echo "============================================================"
echo "Airflow is starting up!"
echo ""
echo "Web UI: http://localhost:8080"
echo "Username: admin"
echo "Password: admin"
echo ""
echo "Wait 1-2 minutes for services to initialize."
echo "============================================================"
