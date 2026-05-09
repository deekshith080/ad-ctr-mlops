#!/bin/bash
cd ~/Desktop/ad-ctr-mlops
source venv/bin/activate
mlflow ui --port 5000 --host 0.0.0.0 &
uvicorn src.api.app:app --reload --port 8000 &
streamlit run streamlit_app.py &
echo "All services started!"
echo "MLflow    -> http://localhost:5000"
echo "FastAPI   -> http://localhost:8000/docs"
echo "Streamlit -> http://localhost:8501"
