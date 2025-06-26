import pandas as pd
import numpy as np
from src.evaluate import evaluate_model
from src.models.pyfunc_wrapper import FullPipelinePyFunc
import pytest

class DummyModel(FullPipelinePyFunc):
    def predict(self, context, model_input):
        # Always predict 1 for test simplicity
        return pd.Series([1] * len(model_input), index=model_input.index)

def test_evaluate_model_accuracy():
    X_test = pd.DataFrame({'f1': [1, 2, 3], 'f2': [0.1, 0.2, 0.3]})
    y_test = pd.Series([1, 0, 1], index=X_test.index)
    model = DummyModel(None, None)
    metrics = evaluate_model(model, X_test, y_test, metrics_to_compute=["accuracy", "f1_macro"])
    assert "metrics" in metrics
    assert "accuracy" in metrics["metrics"]
    assert "f1_macro" in metrics["metrics"]
    # For this dummy, accuracy should be 2/3
    assert np.isclose(metrics["metrics"]["accuracy"], 2/3)
