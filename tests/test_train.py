import pandas as pd
import numpy as np
import pytest
from src.train import train_model
from src.models.pyfunc_wrapper import FullPipelinePyFunc
from src.config.schema import PipelineConfig

class DummyOutlierRemoverConfig:
    group_by_column = "sensor_id"
    temperature_column = "temperature_sensor"
    max_temp_threshold = 1000
    min_temp_threshold = -1000
    def model_dump(self):
        return {
            'group_by_column': self.group_by_column,
            'temperature_column': self.temperature_column,
            'max_temp_threshold': self.max_temp_threshold,
            'min_temp_threshold': self.min_temp_threshold
        }

class DummyModelConfig:
    model_class_path = "sklearn.ensemble.RandomForestClassifier"
    hyperparameters = {'n_estimators': 1, 'max_depth': 2, 'random_state': 42}

class DummyConfig:
    # Minimal config for testing
    outlier_remover = DummyOutlierRemoverConfig()
    data_preprocessor = []
    model = DummyModelConfig()
    training = type('Training', (), {
        'feature_columns': ['f1', 'f2'],
    })()

def test_train_model_returns_fullpipeline():
    # Create dummy data
    X = pd.DataFrame({'f1': [1, 2, 3, 4], 'f2': [0.1, 0.2, 0.3, 0.4], 'sensor_id': [1, 1, 2, 2], 'temperature_sensor': [20, 21, 22, 23]})
    y = pd.Series([0, 1, 0, 1])
    config = DummyConfig()
    model = train_model(X, y, config)
    assert isinstance(model, FullPipelinePyFunc)
    # Use the public predict interface
    preds = model.predict(context=None, model_input=X)
    assert len(preds) == len(X)
