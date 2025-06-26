## 🧱 Key Architectural Decisions

### 1. **Modular Pipeline Design**
The pipeline is decomposed into independent, reusable components: `DataLoader`, `OutlierRemover`, `FeaturePreprocessor`, `ColonyStrengthClassifier`, `Trainer`, and `Evaluator`. This modularity ensures high cohesion and low coupling, making the system easier to test, maintain, and extend.

---

### 2. **Configuration-Driven Architecture**
All components are fully configurable via a centralized `config.yaml` file. Each high-level module has its own configuration block, enabling reproducibility, transparency, and seamless integration with MLflow.

---

### 3. **Feature Store Pattern for Preprocessing**
Feature engineering is implemented using a declarative feature store pattern. Each transformation step is defined in the config with:
- A readable name
- A reference to the transformation class
- Step-specific parameters

This allows dynamic construction of preprocessing pipelines and supports versioning of transformations.

---

### 4. **Outlier Removal as a Separate Component**
Outlier removal is handled by a dedicated `OutlierRemover` component, separate from the sklearn-compatible feature pipeline. This is necessary because outlier removal may affect both `X` and `y`, which violates sklearn's transformer assumptions. This separation improves clarity and correctness.

---

### 5. **MLflow Integration with Feature Hashing**
MLflow is used for experiment tracking, model versioning, and artifact logging. To ensure reproducibility and traceability, a hash of the entire feature engineering pipeline is computed and logged as a parameter in MLflow. This guarantees that even small changes in preprocessing logic are reflected in the experiment metadata.

---

### 6. **FastAPI-Based Model Serving**
A FastAPI service exposes the trained model via REST endpoints:
- `GET /health` for health checks
- `POST /predict` for inference

The API uses Pydantic schemas for validation and is designed for production-readiness and easy integration.

---

### 7. **Sklearn-Compatible Custom Components**
All custom transformers and models implement the sklearn interface (`fit`, `transform`, `fit_transform`), allowing them to be composed into sklearn pipelines and used with tools like `GridSearchCV`.

---

### 8. **Clear Separation of Responsibilities**
Each script has a single responsibility:
- `train.py`: Assembles and trains the pipeline
- `evaluate.py`: Evaluates the trained model
- `train_pipeline.py`: Entry point with MLflow integration
- `serve_api.py`: Launches the prediction API

This structure improves maintainability and aligns with production best practices.

---

### 9. **Lightweight Logging and Testing**
A simple logging utility provides consistent logs across components. Unit tests cover all major modules, and the project is CI-ready via `make test`.