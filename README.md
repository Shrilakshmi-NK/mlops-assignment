# mlops-assignment

### Short description
A small MLOps assignment demonstrating an end-to-end workflow: dataset tracked with DVC, reproducible experiments, MLflow experiment logging, a trained scikit-learn RandomForest model, and a simple inference app (app.py). Use this repo as a template or learning example for DVC + MLflow + reproducible model delivery.

## Table of contents
- About
- Features
- Quickstart
  - Requirements
  - Install
  - Run (train / evaluate / serve)
- Project structure
- Data & DVC
- Training & Experiment tracking (MLflow)
- Evaluation
- Inference / Serving
- Reproducing the pipeline
- Contributing
- Contact

### About
This repository contains an MLOps assignment focused on reproducibility and deployment readiness. It uses DVC to manage the dataset, MLflow to log experiments and artifacts, and a scikit-learn RandomForest model as the primary model. A lightweight inference application is provided (app.py).

### Features
- Dataset tracked with DVC (dvc.yaml, dvc.lock, .dvc files)
- Training and experiment orchestration via run_experiment.py
- MLflow logging and experiment tracking (commit notes indicate MLflow logging is implemented)
- Trained RandomForest model (commit message reports R² ≈ 0.8321)
- Lightweight inference application (app.py)
- metrics.json with run metrics

# ____________________________________________________________________________

## Quickstart

### Requirements
- Python 3.8+
- pip
- dvc (v1+)
- mlflow
- scikit-learn
- (Optional) virtualenv / venv

Install (venv + pip)
1. Clone the repo
   git clone https://github.com/Shrilakshmi-NK/mlops-assignment.git
   cd mlops-assignment

2. Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows

3. Install dependencies
   If a requirements.txt exists:
     pip install -r requirements.txt
   Otherwise install core dependencies:
     pip install scikit-learn mlflow dvc pandas numpy flask

### Data & DVC
- The dataset is tracked with DVC. Pointer file: MLOps_assignment_dataset.csv.dvc and pipeline files dvc.yaml and dvc.lock are included.
- To fetch the data (if remote storage used):
  dvc pull

- To run the DVC pipeline (if configured):
  dvc repro

### Training & Experiment tracking (MLflow)
- Training entrypoint: run_experiment.py
  Example usage (adjust flags/configs to match script):
    python run_experiment.py

- To view MLflow runs locally:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

### Evaluation
- A metrics.json file exists capturing run metrics. Use run_experiment.py to reproduce evaluation or check the MLflow run for saved metrics.

### Inference / Serving
- Inference entrypoint: app.py (a lightweight serving app).
  Example (development):
    python app.py

- If using Flask CLI:
    export FLASK_APP=app.py
    flask run --host=0.0.0.0 --port=5000

- The repository contains a .gradio directory — a Gradio interface may be included; check app.py or notebooks for usage details.

### Reproducing the pipeline
- To reproduce a full run:
  1. git checkout <commit-sha>
  2. dvc pull
  3. pip install -r requirements.txt (or install core deps)
  4. python run_experiment.py

### Project structure (detected files)
- run_experiment.py         # primary experiment & training script
- app.py                    # inference / serving app
- MLOps_assignment_dataset.csv.dvc  # DVC pointer for dataset
- dvc.yaml                  # DVC pipeline definition
- dvc.lock                  # DVC lockfile
- metrics.json              # run metrics snapshot
- .dvc/                     # DVC metadata
- .gradio/                  # (UI / Gradio project files)
- .venv/                    # virtual environment (should not be committed; present in repo)

### Configuration
- If config/ or YAML files are present, adjust the --config argument to run_experiment.py accordingly. If not present, check run_experiment.py for inline defaults or flags.

### Best practices & tips
- Lock dependencies (requirements.txt / environment.yml) and include them in MLflow run metadata for reproducibility.
- Record Git commit SHA in MLflow runs (common practice; commit messages indicate this repo logs experiment metadata).
- Use dvc repro to manage reproducible compute pipelines and dvc push to store data in remote storage.

### Contributing
- Fork the repo, create a feature branch, add tests, run the pipeline locally, and open a PR.
- Follow PEP8; use black/flake8 for formatting and linting.

### Contact
- Repo owner: Shrilakshmi-NK (GitHub: @Shrilakshmi-NK)
- For questions or issues: open an issue in the repository.
