#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Uber Ride Cancellation — Grafana Dashboard Setup ==="
echo ""

echo "[1/3] Exporting analysis data to SQLite..."
python3 export_data.py

echo "[2/3] Generating Grafana dashboard JSON..."
python3 generate_dashboard.py

echo "[3/3] Starting Grafana with Docker Compose..."
docker compose up -d || DOCKER_API_VERSION=1.44 docker compose up -d

echo ""
echo "============================================"
echo "  Grafana is running at: http://localhost:3000"
echo "  Login: admin / admin"
echo "  Dashboard: Uber Ride Cancellation — Analysis & Model Monitoring"
echo "============================================"
