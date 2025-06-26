# src/evaluator.py
# No changes needed. It correctly accepts a pyfunc_model and raw test data.
from typing import Dict, Any, List
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.models.pyfunc_wrapper import FullPipelinePyFunc

# src/evaluator.py

from typing import Dict, Any, List
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.models.pyfunc_wrapper import FullPipelinePyFunc

def evaluate_model(
    model: FullPipelinePyFunc,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics_to_compute: List[str]
) -> Dict[str, Any]:
    """Evaluates the final, wrapped PyFunc model on raw test data."""
    # Step 1: Get predictions. This now returns a pandas Series with the correct index.
    predictions: pd.Series = model.predict(context=None, model_input=X_test)

    # Step 2: Align y_test with the predictions. This now works perfectly.
    y_test_aligned = y_test.loc[predictions.index]

    scalar_metrics = {}
    artifacts = {}

    if "accuracy" in metrics_to_compute:
        scalar_metrics["accuracy"] = accuracy_score(y_test_aligned, predictions)
    if "f1_macro" in metrics_to_compute:
        scalar_metrics["f1_macro"] = f1_score(y_test_aligned, predictions, average="macro")

    return {"metrics": scalar_metrics, "artifacts": artifacts}