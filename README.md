# Ad CTR MLOps Pipeline

End-to-end MLOps pipeline for Click-Through Rate prediction using XGBoost, MLflow, Airflow, FastAPI, Streamlit and Docker.

## Results

| Metric | Value |
|--------|-------|
| Model | XGBoost Classifier |
| Training Data | 70,000 rows Criteo dataset |
| Validation AUC | 0.7589 |
| Log Loss | 0.4517 |
| Features | 39 total |
| API Latency | under 100ms |

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Model | XGBoost | CTR prediction |
| Tracking | MLflow | Experiment versioning |
| Drift Detection | PSI | Feature monitoring |
| Orchestration | Airflow | Weekly scheduling |
| Alerts | Telegram Bot | Retraining notifications |
| API | FastAPI | Model serving |
| Dashboard | Streamlit | Visual monitoring |
| Container | Docker | Reproducible deployment |

## Architecture

```
Criteo Data 1M rows
      |
      v
Feature Engineering -- build_features.py
      |
      v
XGBoost Training -- train.py --> MLflow Tracking
      |
      v
FastAPI REST API -- /predict /health /model/info
      |
      v
Streamlit Dashboard -- live predictions and drift charts

Every Monday via Airflow DAG:
  drift_detector.py --> PSI scores for 39 features
  evaluate.py       --> AUC vs baseline 0.7589
  If severe drift or AUC drop --> Telegram alert fired
```

## Project Structure

```
ad-ctr-mlops/
├── src/
│   ├── features/build_features.py
│   ├── model/train.py
│   ├── model/evaluate.py
│   ├── monitoring/drift_detector.py
│   └── api/app.py
├── dags/drift_detection_dag.py
├── streamlit_app.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

```bash
git clone https://github.com/deekshith080/ad-ctr-mlops.git
cd ad-ctr-mlops
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/model/train.py
uvicorn src.api.app:app --reload --port 8000
streamlit run streamlit_app.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| GET | /model/info | Model metadata |
| POST | /predict | CTR prediction |

Interactive docs: http://localhost:8000/docs

## Drift Detection

| PSI Score | Status | Action |
|-----------|--------|--------|
| below 0.1 | Stable | None needed |
| 0.1 to 0.2 | Moderate | Monitor |
| above 0.2 | Severe | Retrain now |

## Monitoring Pipeline

Airflow DAG runs every Monday at 01:00 UTC.
Sends Telegram alert automatically when:
- Any feature PSI above 0.2
- AUC drops more than 2 percent vs baseline

## Author

Deekshith - Data and ML Engineer
GitHub: https://github.com/deekshith080