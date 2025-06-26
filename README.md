# BeeHero Colony Strength Classifier

A modular, production-ready machine learning pipeline for classifying bee colony strength (small, medium, large) from sensor data. This refactored system features robust configuration, experiment tracking, and a FastAPI-based prediction service.

[📐 Key Architectural Decisions](architectural_decisions.md)

---

## Project Structure

```
mlops-assignment-refactored/
├── Makefile                # Automation for common tasks
├── pyproject.toml          # Project metadata and dependencies
├── requirements.txt        # (legacy) Dependency list
├── train_pipeline.py       # Main entry point for training pipeline
├── serve_api.py            # FastAPI server entry point
├── Dockerfile              # Containerization
├── src/
│   ├── config/
│   │   ├── config.yaml     # Pipeline configuration
│   │   └── schema.py       # Pydantic config schemas
│   ├── data/
│   │   ├── loader/         # Data loading logic
│   │   ├── preprocessor/   # Feature engineering steps
│   │   └── outlier_remover/ # Outlier removal logic
│   ├── models/             # Model and MLflow wrapper
│   ├── utils/              # Logging
│   ├── api/                # FastAPI app, schemas, service
│   ├── train.py            # Training logic
│   └── evaluate.py         # Evaluation logic
├── resources/
│   └── colony_size.csv     # Example dataset
└── tests/                  # Unit tests
```

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv): `pip install uv`

### Installation

```bash
git clone <repository-url>
cd mlops-assignment-refactored
make install
```

### Run Tests

```bash
make test
```

---

## Usage

### Train the Model

```bash
make train
# or with custom config:
uv run python train_pipeline.py --config src/config/config.yaml
```

- All pipeline settings are in `src/config/config.yaml`.

### Serve the API

Start the FastAPI server for predictions:

```bash
make serve

# For development (auto-reload)
make serve-dev
```

#### API Endpoints

- `GET /health` — Health check
- `POST /predict` — Predict colony strength

Example request:
```json
{
  "instances": [
    {
      "temperature": 24.5,
      "humidity": 65.0,
      "light_intensity": 800,
      "vibration": 0.015,
      "sound_frequency": 250
    }
  ]
}
```
Example response:
```json
{
  "predictions": ["Strong"],
  "record_ids": [0]
}
```

---

## Features

- **Modular pipeline**: Data loading, outlier removal, preprocessing, model, and evaluation are fully configurable.
- **MLflow integration**: Experiment tracking, model registry, and artifact logging.
- **FastAPI service**: REST API for predictions.
- **Extensible**: Add new preprocessing steps, models, or data sources with minimal code changes.
- **Testing**: Unit tests for all major components.
