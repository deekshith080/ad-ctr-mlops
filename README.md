# Ad CTR MLOps Pipeline

End-to-end MLOps pipeline for Click-Through Rate prediction using XGBoost, MLflow, Airflow, FastAPI, Streamlit and Docker.

## Results

| Metric | Value |
|--------|-------|
| Model | XGBoost Classifier |
| Training Data | 70,000 rows (Criteo dataset) |
| Validation AUC | 0.7589 |
| Log Loss | 0.4517 |
| Features | 39 (13 numeric + 26 categorical) |
| API Latency | < 100ms |

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



## Project Structure



## Quick Start



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
| < 0.1 | Stable | None needed |
| 0.1 to 0.2 | Moderate | Monitor |
| > 0.2 | Severe | Retrain now |

## Monitoring Pipeline

Airflow DAG runs every Monday at 01:00 UTC.
Sends Telegram alert automatically when:
- Any feature PSI > 0.2 (severe drift)
- AUC drops more than 2% vs baseline (0.7589)

## Author

Deekshith - Data/ML Engineer
GitHub: https://github.com/deekshith080