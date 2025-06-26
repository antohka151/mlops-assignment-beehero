import os
import re
import yaml
from pydantic import BaseModel, Field
from src.data.loader import DataLoaderConfig
from src.data.preprocessor import PreprocessorStepConfig
from src.data.outlier_remover import OutlierRemoverConfig
from src.models.colony_classifier import ColonyStrengthClassifierConfig
from typing import Any


class TrainingConfig(BaseModel):
    """High-level configuration for the training process."""
    target_column: str = Field(..., description="Name of the target variable.")
    feature_columns: list[str] = Field(
        ...,
        description="List of feature columns to use for training."
    )
    test_size: float = Field(default=0.2, ge=0.0, le=1.0)
    random_state: int = Field(default=42)
    stratify: bool = Field(default=True)

class EvaluationConfig(BaseModel):
    """Configuration for the evaluation step."""
    metrics: list[str] = Field(
        default=["accuracy", "f1_macro"],
        description="List of metrics to compute."
    )

class MLflowConfig(BaseModel):
    """Configuration for MLflow tracking."""
    tracking_uri: str = Field(..., description="The URI for the MLflow tracking server.")
    experiment_name: str = Field(..., description="The name of the MLflow experiment.")
    run_name: str = Field(
        default="run_{timestamp}",
        description="The name of the MLflow run, can include a timestamp."
    )
    registered_model_name: str = Field(
        ...,
        description="The name under which the model will be registered in MLflow."
    )


# --- Main Configuration Model ---

class PipelineConfig(BaseModel):
    """The root configuration for the entire ML pipeline."""
    data_loader: DataLoaderConfig
    outlier_remover: OutlierRemoverConfig
    data_preprocessor: list[PreprocessorStepConfig]
    model: ColonyStrengthClassifierConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    mlflow: MLflowConfig

    @classmethod
    def from_yaml(cls: 'PipelineConfig', path: str) -> 'PipelineConfig':
        """
        Loads, resolves environment variables, and validates the pipeline 
        configuration from a YAML file.

        Args:
            path (str): The path to the YAML configuration file.

        Returns:
            PipelineConfig: A validated configuration object.
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        resolved_config = cls._resolve_env_vars(config_dict)
        return cls(**resolved_config)

    @staticmethod
    def _resolve_env_vars(obj: Any) -> Any:
        """
        Recursively resolves environment variables in a nested dictionary or list.
        Converts placeholders of the form ${VAR_NAME:-default_value} to their
        corresponding environment variable values, or to the default value if the

        Args:
            obj (Any): The configuration object (dict, list, or str).

        Returns:
            Any: The object with environment variables resolved.
        """
        pattern = re.compile(r'\$\{(\w+)(?::-([^}]+))?\}')

        def resolve_string(value: str) -> str:
            """Replaces environment variable placeholders in the string with their values."""
            def replacer(match: re.Match) -> str:
                var_name, default = match.groups()
                return os.environ.get(var_name, default or '')
            return pattern.sub(replacer, value)

        if isinstance(obj, dict):
            return {k: PipelineConfig._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [PipelineConfig._resolve_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return resolve_string(obj)
        else:
            return obj