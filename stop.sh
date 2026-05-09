#!/bin/bash
pkill -f "mlflow ui"
pkill -f "uvicorn"
pkill -f "streamlit"
echo "All services stopped."
