#!/bin/bash
set -e

# Start FastAPI backend in background
echo "Starting FastAPI backend..."
uvicorn app:app --host 127.0.0.1 --port 8000 &

# Wait for FastAPI to start
sleep 2

# Start Nginx in foreground
echo "Starting Nginx..."
nginx -g "daemon off;"
